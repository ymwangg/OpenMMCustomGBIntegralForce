#include "CustomGBIntegral.h"
#include "openmm/OpenMMException.h"
#include <algorithm> 
#include <iostream>
#include <cstdio>
#include <chrono>
#include <cmath>
#include "CudaCharmmKernelSources.h"
#include "Units.h"
#include "openmm/internal/CharmmQuadrature.h"

using namespace::OpenMM;
using namespace::std;
#define REAL float
#define REAL3 float3
#define REAL4 float4

CustomGBIntegral::CustomGBIntegral(CudaContext& cu, const System& system, const CharmmGBMVForce& force, CudaParameterSet* &computedIntegrals, CudaParameterSet* &energyDerivs) : cu(cu), system(system),force(force),computedIntegrals(computedIntegrals),dEdI(energyDerivs){
    {
        for (int i = 0; i < force.getNumGBIntegrals(); ++i){
            std::string name;
            std::vector<int> parInt;
            std::vector<double> parReal;
            force.getGBIntegralParameters(i, name, parInt, parReal);
            if(parInt.size()==0)
                throw OpenMMException("GBSWIntegral: the order of integral must be given");
            if(parInt[0] < 2)
                throw OpenMMException("GBSWIntegral: the order of integral must be greater or equal to 2");
            _integralOrders.push_back(parInt[0]-2);
        }   
    }
    counter = 0;
        {
            radius = new CudaParameterSet(cu, 1, cu.getPaddedNumAtoms(), "CustomGBLookupTableRadius", true);
            int numParticleParams = force.getNumPerParticleParameters();
            std::map<std::string,int> paramIndex;
            for(int i = 0; i < numParticleParams; i++){
                std::string name = force.getPerParticleParameterName(i);
                paramIndex[name] = i;
            } 
            vector<vector<float> > paramVector(cu.getPaddedNumAtoms(), vector<float>(1, 0)); 
            for (int i = 0; i < cu.getNumAtoms(); ++i) {
                std::vector<double> particleParam;
                force.getParticleParameters(i, particleParam);
                paramVector[i][0] = particleParam[paramIndex["radius"]];
            }
            radius->setParameterValues(paramVector);
        }

        {
            double4 periodicBoxLength = cu.getPeriodicBoxSize();
            double d_x = periodicBoxLength.x;
            double d_y = periodicBoxLength.y;
            double d_z = periodicBoxLength.z;
            _lookupTableSize = 25;
            _lookupTableGridLength = 0.15;
            int n_x = ceil(d_x/_lookupTableGridLength)+1;
            int n_y = ceil(d_y/_lookupTableGridLength)+1;
            int n_z = ceil(d_z/_lookupTableGridLength)+1;
            _lookupTableNumberOfGridPoints[0] = n_x;
            _lookupTableNumberOfGridPoints[1] = n_y;
            _lookupTableNumberOfGridPoints[2] = n_z;
            int totalNumGridPoints = n_x*n_y*n_z;
            float step[3];
            step[0] = d_x / n_x;
            step[1] = d_y / n_y;
            step[2] = d_z / n_z;
            _lookupTableGridStep[0] = step[0];
            _lookupTableGridStep[1] = step[1];
            _lookupTableGridStep[2] = step[2];
            double lookupTableGridLength = max(max(step[0],step[1]),step[2]);
            d_lookupTable.initialize<int>(cu,totalNumGridPoints*_lookupTableSize,"CustomGBLookupTable");
            d_lookupTableNumAtoms.initialize<int>(cu,totalNumGridPoints,"CustomGBLookupTableNumAtoms");
            d_lookupTableNumGridPoints.initialize<int>(cu,3,"CustomGBLookupTableNumGridPoints");
            d_lookupTableMinCoord.initialize<float>(cu,3,"CustomGBLookupTableMinCoord");
            d_lookupTableGridStep.initialize<float>(cu,3,"CustomGBLookupTableGridStep");
            d_lookupTableNumGridPoints.upload(_lookupTableNumberOfGridPoints);
            d_lookupTableGridStep.upload(step);
            _lookupTableBufferLength  = 0.01 + 0.03;
            map<string, string> defines;
            defines["NUM_ATOMS"] = cu.intToString(cu.getNumAtoms());
            CUmodule module = cu.createModule(cu.replaceStrings(CudaCharmmKernelSources::lookupTable,defines));
            //cout<<cu.replaceStrings(CudaCharmmKernelSources::lookupTable,defines)<<endl;
            lookupTableKernel = cu.getKernel(module, "computeLookupTable");
        }
        { // quadrature
            
            //radial integral starting point
            _r0 = 0.5*OpenMM::NmPerAngstrom;

            //radial integral ending point
            _r1 = 20.0*OpenMM::NmPerAngstrom;

            //number of radial integration points using Gauss-Legendre quadrature
            int nRadialPoints = 24; 

            //rule of Lebedev quadrature for spherical integral
            int ruleLebedev = 3; //3->38points 4->50points
            vector<vector<double> > radialQuad = CharmmQuadrature::GaussLegendre(_r0, _r1, nRadialPoints);
            vector<vector<double> > sphericalQuad = CharmmQuadrature::Lebedev(ruleLebedev);
            _numQuadPoints = radialQuad.size()*sphericalQuad.size();
            vector<float4> quad(_numQuadPoints);
            vector<vector<float> > quad_w(force.getNumGBIntegrals(),vector<float>(_numQuadPoints));
            d_quad_w.resize(force.getNumGBIntegrals());

            for(int i=0; i<radialQuad.size(); ++i){
                double r = radialQuad[i][0];
                double w_r = radialQuad[i][1];
                for(int j=0; j<sphericalQuad.size(); ++j){
                    int idx = i*sphericalQuad.size()+j;
                    quad[idx].x = r*sphericalQuad[j][0];
                    quad[idx].y = r*sphericalQuad[j][1];
                    quad[idx].z = r*sphericalQuad[j][2];
                    double w_s = sphericalQuad[j][3];
                    quad[idx].w = r;
                    for(int k=0; k<force.getNumGBIntegrals(); k++){
                        quad_w[k][idx] = w_r * w_s / pow(r, _integralOrders[k]);
                    }
                }   
            } 
            d_quad.initialize<float4>(cu,radialQuad.size()*sphericalQuad.size(),
                    "CustomGBQuad");
            d_quad.upload(quad);
            for(int i=0; i<force.getNumGBIntegrals(); i++){
                d_quad_w[i].initialize<float>(cu,1*radialQuad.size()*sphericalQuad.size(),"CustomGBQuadW"+cu.intToString(i));
                d_quad_w[i].upload(quad_w[i]);
            }
        }
        {
            d_volume.initialize<float>(cu,cu.getNumAtoms()*_numQuadPoints,"CustomGBVolume");
            //compute integral kernel
            stringstream paramArgs, initParams, beforeVolume, computeVolume, afterVolume, reduction;
            paramArgs << ",const float* __restrict__ radius";
            paramArgs << ",const int* __restrict__ lookupTable";
            paramArgs << ",const int* __restrict__ lookupTableNumAtoms";
            paramArgs << ",const int* __restrict__ lookupTableNumGridPoints";
            paramArgs << ",const float* __restrict__ lookupTableMinCoor";
            paramArgs << ",const float* __restrict__ lookupTableGridStep";
            paramArgs << ",const float4* __restrict__ quad";
            _numIntegrals = force.getNumGBIntegrals();
            for(int i=0; i< _numIntegrals; i++){
                paramArgs << ",const float* __restrict__ quad_w" << cu.intToString(i);
                paramArgs << ",float* integral" << cu.intToString(i);
            }
            paramArgs << ",float* __restrict__ volume";
            for(int i=0; i< _numIntegrals; i++){
                initParams << "__shared__ float tmp_result" << cu.intToString(i) << "[32];\n";
            }
            initParams << "float3 minCoor = "
                "make_float3(lookupTableMinCoor[0],lookupTableMinCoor[1],lookupTableMinCoor[2]);\n";
            initParams << "int3 numGridPoints = "
                "make_int3(lookupTableNumGridPoints[0],lookupTableNumGridPoints[1],lookupTableNumGridPoints[2]);\n";
            initParams << "float3 gridStep = "
                "make_float3(lookupTableGridStep[0], lookupTableGridStep[1], lookupTableGridStep[2]);\n";

            beforeVolume << "float V = 1;\n";
            afterVolume << "volume[atomI*NUM_QUADRATURE_POINTS + quadIdx] = V;\n";
            for(int i=0; i<_numIntegrals; i++){
                afterVolume << "float tmp_integral" << cu.intToString(i) << " = quad_w" << cu.intToString(i) <<"[quadIdx] * (1.0 - V);\n";
            }
            reduction << "int lane = threadIdx.x % warpSize;\n";
            reduction << "int wid = threadIdx.x / warpSize;\n";
            for(int i=0; i<_numIntegrals; i++){
                string integralName = "tmp_integral" + cu.intToString(i);
                string globalIntegralName = "integral" + cu.intToString(i);
                reduction << integralName << " = warpReduceSum("<< integralName << ");\n";
                reduction << "if(lane==0) tmp_result" << cu.intToString(i) << "[wid] = " << integralName << ";\n";
                reduction << "__syncthreads();\n";
                reduction << integralName <<" = (threadIdx.x < blockDim.x / warpSize) ? tmp_result" << cu.intToString(i) << "[lane] : 0;\n";
                reduction << "if(wid==0){\n";
                reduction << integralName <<" = warpReduceSum("<< integralName << ");\n";
                reduction << "if(threadIdx.x==0) atomicAdd(&"<<globalIntegralName<<"[atomI],"<<integralName<<");\n";
                reduction << "}\n";
            }

            map<string, string> defines, macroDefines;
            defines["NUM_ATOMS"] = cu.intToString(cu.getNumAtoms());
            defines["NUM_PADDED_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
            defines["PARAM_ARGS"] = paramArgs.str();
            defines["INIT_PARAMS"] = initParams.str();
            defines["NUM_QUADRATURE_POINTS"] = cu.intToString(_numQuadPoints);
            defines["LOOKUPTABLE_SIZE"] = cu.intToString(_lookupTableSize);
            defines["BEFORE_VOLUME"] = beforeVolume.str();
            //defines["COMPUTE_VOLUME"] = computeVolume.str();
            defines["AFTER_VOLUME"] = afterVolume.str();
            defines["REDUCTION"] = reduction.str();
            macroDefines["USE_PERIODIC"] = "1";

            cout<<cu.replaceStrings(CudaCharmmKernelSources::computeGBSWIntegral, defines)<<endl;
            CUmodule module = cu.createModule(cu.replaceStrings(CudaCharmmKernelSources::computeGBSWIntegral,defines),macroDefines);
            //CUmodule module = cu.createModule(cu.replaceStrings(CudaCharmmKernelSources::computeGBSWIntegral2,defines),macroDefines);
            integralKernel = cu.getKernel(module, "computeGBSWIntegral");
        }

        {
            // reduce GB force kernel
            stringstream paramArgs, initParams, beforeVolume, computeVolume, afterVolume, reduction;
            paramArgs << ",const float* __restrict__ radius";
            paramArgs << ",const int* __restrict__ lookupTable";
            paramArgs << ",const int* __restrict__ lookupTableNumAtoms";
            paramArgs << ",const int* __restrict__ lookupTableNumGridPoints";
            paramArgs << ",const float* __restrict__ lookupTableMinCoor";
            paramArgs << ",const float* __restrict__ lookupTableGridStep";
            paramArgs << ",const float4* __restrict__ quad";
            _numIntegrals = force.getNumGBIntegrals();
            for(int i=0; i< _numIntegrals; i++){
                paramArgs << ",const float* __restrict__ quad_w" << cu.intToString(i);
                paramArgs << ",float* dEdI" << cu.intToString(i);
            }
            paramArgs << ",const float* __restrict__ volume";
            for(int i=0; i< _numIntegrals; i++){
                initParams << "__shared__ float tmp_result" << cu.intToString(i) << "[32];\n";
            }
            initParams << "float3 minCoor = "
                "make_float3(lookupTableMinCoor[0],lookupTableMinCoor[1],lookupTableMinCoor[2]);\n";
            initParams << "int3 numGridPoints = "
                "make_int3(lookupTableNumGridPoints[0],lookupTableNumGridPoints[1],lookupTableNumGridPoints[2]);\n";
            initParams << "float3 gridStep = "
                "make_float3(lookupTableGridStep[0], lookupTableGridStep[1], lookupTableGridStep[2]);\n";

            beforeVolume << "float V = volume[atomI*NUM_QUADRATURE_POINTS + quadIdx];\n";
            for(int i=0; i<_numIntegrals; i++){
                beforeVolume << "float prefactor" << cu.intToString(i) <<
                    " = quad_w" << cu.intToString(i) << "[quadIdx] * dEdI" <<
                    cu.intToString(i) << "[atomI];\n";
            }
            for(int i=0; i<_numIntegrals; i++){
                computeVolume << "forceI.x -= prefactor" << cu.intToString(i) <<
                    " * chain * delta.x;\n";
                computeVolume << "forceI.y -= prefactor" << cu.intToString(i) <<
                    " * chain * delta.y;\n";
                computeVolume << "forceI.z -= prefactor" << cu.intToString(i) <<
                    " * chain * delta.z;\n";
                computeVolume << "forceJ.x += prefactor" << cu.intToString(i) <<
                    " * chain * delta.x;\n";
                computeVolume << "forceJ.y += prefactor" << cu.intToString(i) <<
                    " * chain * delta.y;\n";
                computeVolume << "forceJ.z += prefactor" << cu.intToString(i) <<
                    " * chain * delta.z;\n";
                /*
                 * forceBuffers[index] += (long long) (force.x*0x100000000);
                 * forceBuffers[index+PADDED_NUM_ATOMS] += (long long) (force.y*0x100000000);
                 * forceBuffers[index+PADDED_NUM_ATOMS*2] += (long long) (force.z*0x100000000);
                 */
            }
            //atomicAdd(&
            computeVolume << "atomicAdd(&forceBuffers[atomI], (long long) (forceI.x*0x100000000));\n";
            computeVolume << "atomicAdd(&forceBuffers[atomI+NUM_PADDED_ATOMS], (long long) (forceI.y*0x100000000));\n";
            computeVolume << "atomicAdd(&forceBuffers[atomI+NUM_PADDED_ATOMS*2], (long long) (forceI.z*0x100000000));\n";
            computeVolume << "atomicAdd(&forceBuffers[atomJ], (long long) (forceJ.x*0x100000000));\n";
            computeVolume << "atomicAdd(&forceBuffers[atomJ+NUM_PADDED_ATOMS], (long long) (forceJ.y*0x100000000));\n";
            computeVolume << "atomicAdd(&forceBuffers[atomJ+NUM_PADDED_ATOMS*2], (long long) (forceJ.z*0x100000000));\n";

            map<string, string> defines, macroDefines;
            defines["NUM_ATOMS"] = cu.intToString(cu.getNumAtoms());
            defines["NUM_PADDED_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
            defines["PARAM_ARGS"] = paramArgs.str();
            defines["INIT_PARAMS"] = initParams.str();
            defines["NUM_QUADRATURE_POINTS"] = cu.intToString(_numQuadPoints);
            defines["LOOKUPTABLE_SIZE"] = cu.intToString(_lookupTableSize);
            defines["BEFORE_VOLUME"] = beforeVolume.str();
            defines["COMPUTE_VOLUME"] = computeVolume.str();
            defines["AFTER_VOLUME"] = afterVolume.str();
            macroDefines["USE_PERIODIC"] = "1";

            //cout<<CudaCharmmKernelSources::reduceGBSWForce<<endl;
            cout<<cu.replaceStrings(CudaCharmmKernelSources::reduceGBSWForce,defines)<<endl;
            CUmodule module = cu.createModule(cu.replaceStrings(CudaCharmmKernelSources::reduceGBSWForce,defines),macroDefines);
            //CUmodule module = cu.createModule(cu.replaceStrings(CudaCharmmKernelSources::computeGBSWIntegral2,defines),macroDefines);
            reduceForceKernel = cu.getKernel(module, "reduceGBSWForce");
        }

        {
            lookupTableArgs.push_back(&cu.getPosq().getDevicePointer());
            lookupTableArgs.push_back(cu.getPeriodicBoxSizePointer());
            lookupTableArgs.push_back(cu.getInvPeriodicBoxSizePointer());
            lookupTableArgs.push_back(&radius->getBuffers()[0].getMemory());
            lookupTableArgs.push_back(&d_lookupTable.getDevicePointer());
            lookupTableArgs.push_back(&d_lookupTableNumAtoms.getDevicePointer());
            lookupTableArgs.push_back(&d_lookupTableNumGridPoints.getDevicePointer());
            lookupTableArgs.push_back(&d_lookupTableMinCoord.getDevicePointer());
            lookupTableArgs.push_back(&d_lookupTableGridStep.getDevicePointer());
            lookupTableArgs.push_back(&_lookupTableGridLength);
            lookupTableArgs.push_back(&_lookupTableBufferLength);
            lookupTableArgs.push_back(&_lookupTableSize);
        }
        {
            integralArgs.push_back(&cu.getPosq().getDevicePointer());
            integralArgs.push_back(cu.getPeriodicBoxSizePointer());
            integralArgs.push_back(cu.getInvPeriodicBoxSizePointer());
            integralArgs.push_back(&radius->getBuffers()[0].getMemory());
            integralArgs.push_back(&d_lookupTable.getDevicePointer());
            integralArgs.push_back(&d_lookupTableNumAtoms.getDevicePointer());
            integralArgs.push_back(&d_lookupTableNumGridPoints.getDevicePointer());
            integralArgs.push_back(&d_lookupTableMinCoord.getDevicePointer());
            integralArgs.push_back(&d_lookupTableGridStep.getDevicePointer());
            integralArgs.push_back(&d_quad.getDevicePointer());
            for(int i=0; i < _numIntegrals; i++){
                integralArgs.push_back(&d_quad_w[i].getDevicePointer());
                integralArgs.push_back(&computedIntegrals->getBuffers()[i].getMemory());
            }
            integralArgs.push_back(&d_volume.getDevicePointer());
        }
        {
            reduceForceArgs.push_back(&cu.getForce().getDevicePointer());
            reduceForceArgs.push_back(&cu.getPosq().getDevicePointer());
            reduceForceArgs.push_back(cu.getPeriodicBoxSizePointer());
            reduceForceArgs.push_back(cu.getInvPeriodicBoxSizePointer());
            reduceForceArgs.push_back(&radius->getBuffers()[0].getMemory());
            reduceForceArgs.push_back(&d_lookupTable.getDevicePointer());
            reduceForceArgs.push_back(&d_lookupTableNumAtoms.getDevicePointer());
            reduceForceArgs.push_back(&d_lookupTableNumGridPoints.getDevicePointer());
            reduceForceArgs.push_back(&d_lookupTableMinCoord.getDevicePointer());
            reduceForceArgs.push_back(&d_lookupTableGridStep.getDevicePointer());
            reduceForceArgs.push_back(&d_quad.getDevicePointer());
            for(int i=0; i < _numIntegrals; i++){
                reduceForceArgs.push_back(&d_quad_w[i].getDevicePointer());
                reduceForceArgs.push_back(&dEdI->getBuffers()[i].getMemory());
            }
            reduceForceArgs.push_back(&d_volume.getDevicePointer());
        }
        printf("done with setting up integral\n");


}  

CustomGBIntegral::~CustomGBIntegral(){
}


void CustomGBIntegral::setLookupTableGridLength(double length){
    _lookupTableGridLength = length;
}

void CustomGBIntegral::setLookupTableBufferLength(double length){
    _lookupTableBufferLength = length;
}

void CustomGBIntegral::computeLookupTable(){

    vector<REAL4> oldPosq(cu.getPaddedNumAtoms());
    int numAtoms = system.getNumParticles();
    cu.getPosq().download(oldPosq);
    std::vector<OpenMM::Vec3> atomCoordinates(numAtoms,OpenMM::Vec3());
    for (int i=0; i<numAtoms; i++){
        atomCoordinates[i][0] = oldPosq[i].x;
        atomCoordinates[i][1] = oldPosq[i].y;
        atomCoordinates[i][2] = oldPosq[i].z;
    }

    double4 periodicBoxLength = cu.getPeriodicBoxSize();
    double center_of_geom[3] = {0,0,0};
    for(int i=0; i<atomCoordinates.size(); i++){
        center_of_geom[0] += atomCoordinates[i][0];
        center_of_geom[1] += atomCoordinates[i][1];
        center_of_geom[2] += atomCoordinates[i][2];
    }    
    center_of_geom[0] /= atomCoordinates.size();
    center_of_geom[1] /= atomCoordinates.size();
    center_of_geom[2] /= atomCoordinates.size();

    //get box dimensions
    double d_x = periodicBoxLength.x;
    double d_y = periodicBoxLength.y;
    double d_z = periodicBoxLength.z;
    _lookupTableMinCoordinate[0] = center_of_geom[0] - d_x/2.0;
    _lookupTableMinCoordinate[1] = center_of_geom[1] - d_y/2.0;
    _lookupTableMinCoordinate[2] = center_of_geom[2] - d_z/2.0;

    /*
    printf("%f,%f,%f\n",_lookupTableMinCoordinate[0],
            _lookupTableMinCoordinate[1],_lookupTableMinCoordinate[2]);
            */

    d_lookupTableMinCoord.upload(_lookupTableMinCoordinate);
    //initialize vdw radii
    CUresult result = cuMemsetD32(d_lookupTableNumAtoms.getDevicePointer(),0,d_lookupTableNumAtoms.getSize());
    if (result != CUDA_SUCCESS) {
        std::stringstream str;
        str<<"Error setting lookupTableNumAtoms to zero" << CudaContext::getErrorString(result)<<" ("<<result<<")";
        throw OpenMMException(str.str());
    }   

    //not periodic
    cu.executeKernel(lookupTableKernel, &lookupTableArgs[0],cu.getPaddedNumAtoms());
}

void CustomGBIntegral::evaluate(){
    CUresult result = cuMemsetD32(computedIntegrals->getBuffers()[0].getMemory(),0,cu.getNumAtoms());
    result = cuMemsetD32(computedIntegrals->getBuffers()[1].getMemory(),0,cu.getNumAtoms());
    //printf("evaluating\n");
    /*
    CUresult result = cuMemsetD32(d_integrals.getDevicePointer(),0,d_integrals.getSize());
    if (result != CUDA_SUCCESS) {
        std::stringstream str;
        str<<"Error setting integrals to zero" << CudaContext::getErrorString(result)<<" ("<<result<<")";
        throw OpenMMException(str.str());
    }   
    result = cuMemsetD32(d_gradients.getDevicePointer(),0,d_gradients.getSize());
    if (result != CUDA_SUCCESS) {
        std::stringstream str;
        str<<"Error setting gradients to zero" << CudaContext::getErrorString(result)<<" ("<<result<<")";
        throw OpenMMException(str.str());
    }
    */
    int threads = min(1024,int(ceil(float(_numQuadPoints)/32)*32));
    cuLaunchKernel(integralKernel, cu.getNumAtoms(), 1, 1, threads, 1, 1, 0, 0, &integralArgs[0], NULL);
    //cu.executeKernel(integralKernel, &integralArgs[0],cu.getPaddedNumAtoms());
}

void CustomGBIntegral::reduce(){
    int threads = min(1024,int(ceil(float(_numQuadPoints)/32)*32));
    cuLaunchKernel(reduceForceKernel, cu.getNumAtoms(), 1, 1, threads, 1, 1, 0, 0, &reduceForceArgs[0], NULL);
}
