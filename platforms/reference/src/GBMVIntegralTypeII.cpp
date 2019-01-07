#include "CharmmQuadrature.h"
#include "ReferenceForce.h"
#include "ReferencePlatform.h"
#include "Units.h"
#include "Vec3.h"
#include "GBMVIntegralTypeII.h"
#include "openmm/OpenMMException.h"
#include <cstdio>


using namespace::OpenMM;
using namespace::std;

static Vec3* extractBoxVectors(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return (Vec3*) data->periodicBoxVectors;
}

GBMVIntegralTypeII::GBMVIntegralTypeII(){
    _periodic = false;

    _gamma0 = 0.44;
    _beta = -20;
    _lambda = 0.5;
    _P1 = 0.45;
    _P2 = 1.25;
    _S0 = 0.7;

    //radial integral starting point
    _r0 = 0.5*OpenMM::NmPerAngstrom;

    //radial integral ending point
    _r1 = 20.0*OpenMM::NmPerAngstrom;

    //number of radial integration points using Gauss-Legendre quadrature
    int nRadialPoints = 24; 

    //rule of Lebedev quadrature for spherical integral
    int ruleLebedev = 4;
    vector<vector<double> > radialQuad = CharmmQuadrature::GaussLegendre(_r0, _r1, nRadialPoints);
    vector<vector<double> > sphericalQuad = CharmmQuadrature::Lebedev(ruleLebedev);
    _quad.resize(radialQuad.size()*sphericalQuad.size(), vector<double>(5,0.0));
    for(int i=0; i<radialQuad.size(); ++i){
        for(int j=0; j<sphericalQuad.size(); ++j){
            double r = radialQuad[i][0];
            double w_r = radialQuad[i][1];
            double w_s = sphericalQuad[j][3];
            int idx = i*sphericalQuad.size()+j;
            for(int k=0; k<3; ++k){
                _quad[idx][k] = r*sphericalQuad[j][k];
            }   
            _quad[idx][3] = r;
            _quad[idx][4] = w_r*w_s;
        }   
    }  

    //lookup table parameters
    //_lookupTableBufferLength = 0.20; //0.20 nm
    _lookupTableBufferLength = 0.20 + 0.21; //0.20 nm
    _lookupTableGridLength = 0.15; //0.15 nm
}

void GBMVIntegralTypeII::initialize(const OpenMM::System& system, const OpenMM::CharmmGBMVForce& force){
    if(OpenMM::CharmmGBMVForce::CutoffPeriodic==force.getNonbondedMethod()){
        _periodic = true;
    }
    _numParticles = system.getNumParticles();
    _numIntegrals = force.getNumGBIntegrals();
    _atomicRadii.resize(_numParticles);
    //query per particle names
    int numParticleParams = force.getNumPerParticleParameters();
    std::map<std::string,int> paramIndex;
    for(int i = 0; i < numParticleParams; i++){
        std::string name = force.getPerParticleParameterName(i);
        paramIndex[name] = i;
    }
    if(paramIndex.count("radius")==0)
        throw OpenMMException("GBMVIntegralTypeII: the per perticle parameter 'radius' must be defined");
    //update atomic radii
    for (int i = 0; i < _numParticles; ++i) {
        std::vector<double> particleParam;
        force.getParticleParameters(i, particleParam);
        _atomicRadii[i] = particleParam[paramIndex["radius"]];
    } 
    for (int i = 0; i < _numIntegrals; ++i){
        std::string name;
        std::vector<int> parInt;
        std::vector<double> parReal;
        force.getGBIntegralParameters(i, name, parInt, parReal);
        if(parInt.size()==0)
            throw OpenMMException("GBMVIntegralTypeII: the order of integral must be given");
        if(parInt[0] < 2)
            throw OpenMMException("GBMVIntegralTypeII: the order of integral must be greater or equal to 2");
        _orders.push_back(parInt[0]-2);
    }
    return;
}

void GBMVIntegralTypeII::setBoxVectors(OpenMM::Vec3* vectors){
    _periodicBoxVectors[0] = vectors[0];
    _periodicBoxVectors[1] = vectors[1];
    _periodicBoxVectors[2] = vectors[2];
}

void GBMVIntegralTypeII::BeforeComputation(ContextImpl& context, const std::vector<OpenMM::Vec3>& atomCoordinates){
    setBoxVectors(extractBoxVectors(context));
    computeLookupTable(atomCoordinates);
}

void GBMVIntegralTypeII::FinishComputation(ContextImpl& context, const std::vector<OpenMM::Vec3>& atomCoordinates){
    //do nothing
}

void GBMVIntegralTypeII::evaluate(const int atomI, ContextImpl& context, const std::vector<OpenMM::Vec3>& atomCoordinates, std::vector<double>& values, std::vector<std::vector<OpenMM::Vec3> >& gradients, const bool includeGradient){
    values.resize(_orders.size(), 0.0);
    if(includeGradient) gradients.resize(_orders.size());
    vector<double> prefactors(_orders.size());
    for(int q=0; q<_quad.size(); ++q){
        OpenMM::Vec3 r_q;
        for(int i=0; i<3; ++i) 
            r_q[i] = atomCoordinates[atomI][i] + _quad[q][i];
        double radius_q = _quad[q][3];
        double w_q = _quad[q][4];
        vector<int> atomList = getLookupTableAtomList(r_q);
        double sum1, sum2, sum3;
        sum1 = 0.0; sum2 = 0.0; sum3 = 0.0;
        OpenMM::Vec3 denom_vec(0.0, 0.0, 0.0);
        double pre_sum = computeVolumeFromLookupTable(atomCoordinates, r_q, atomList, sum1, sum2, sum3, denom_vec);
        double V_q = 1.0/(1.0 + pre_sum);
        for(int i=0; i<_orders.size(); ++i){
            prefactors[i] = w_q/pow(radius_q, _orders[i]);
            values[i] += prefactors[i]*(V_q);
        }
        if(includeGradient){
            for(int i=0; i<_orders.size(); ++i){
                gradients[i].resize(_numParticles, OpenMM::Vec3()); 
                computeGradientPerQuadFromLookupTable(atomI, atomCoordinates, r_q, pre_sum, gradients[i], prefactors[i], atomList,
                        sum1, sum2, sum3, denom_vec);
            }
        }
    }
    return;
}


double GBMVIntegralTypeII::computeVolumeFromLookupTable(const std::vector<OpenMM::Vec3>& atomCoordinates, const OpenMM::Vec3& r_q, const std::vector<int>& atomList, double& sum1, double& sum2, double& sum3, OpenMM::Vec3& denom_vec){
    if(atomList.size()==0) return 0.0;
    double deltaR[ReferenceForce::LastDeltaRIndex];
    sum1 = 0.0;
    sum2 = 0.0;
    for(int i=0; i<atomList.size(); ++i){
        int atomJ = atomList[i];
        if (_periodic)
            ReferenceForce::getDeltaRPeriodic(r_q, atomCoordinates[atomJ], _periodicBoxVectors, deltaR);
        else
            ReferenceForce::getDeltaR(r_q, atomCoordinates[atomJ], deltaR);
        double deltaR_qj = deltaR[ReferenceForce::RIndex];
        double deltaR2 = deltaR_qj*deltaR_qj;
        double atomicRadii_j = _atomicRadii[atomJ];
        double atomicRadii_j2 = atomicRadii_j*atomicRadii_j;
        double C_j = _P1*atomicRadii_j + _P2;
        double F_VSA = C_j / (C_j + deltaR2 - atomicRadii_j2);
        F_VSA = F_VSA*F_VSA;
        sum1 += F_VSA;
        sum2 += deltaR2*(F_VSA*F_VSA);
        OpenMM::Vec3 dr(deltaR[ReferenceForce::XIndex],deltaR[ReferenceForce::YIndex],deltaR[ReferenceForce::ZIndex]);
        denom_vec += dr*F_VSA;
    }
    sum3 = sqrt(denom_vec.dot(denom_vec));
    double sum = _S0 * sum1 * sum2 / (sum3*sum3);
    double pre_sum = exp(_beta*(sum - _lambda));
    return pre_sum;
}


void GBMVIntegralTypeII::computeGradientPerQuadFromLookupTable(const int atomI, const std::vector<OpenMM::Vec3>& atomCoordinates, const OpenMM::Vec3& r_q, const double pre_sum, std::vector<OpenMM::Vec3>& gradients, const double prefactor, const std::vector<int>& atomList, const double sum1, const double sum2, const double sum3, const OpenMM::Vec3& denom_vec){
    if(atomList.size()==0) return;
    double deltaR[ReferenceForce::LastDeltaRIndex];
    double factor = -1.0/((1.0+pre_sum)*(1.0+pre_sum)) * (_beta*pre_sum) * prefactor * _S0;
    for(int i=0; i<atomList.size(); ++i){
        int atomJ = atomList[i];
        if (_periodic)
            ReferenceForce::getDeltaRPeriodic(r_q, atomCoordinates[atomJ], _periodicBoxVectors, deltaR);
        else
            ReferenceForce::getDeltaR(r_q, atomCoordinates[atomJ], deltaR);
        OpenMM::Vec3 dr(deltaR[ReferenceForce::XIndex],deltaR[ReferenceForce::YIndex],deltaR[ReferenceForce::ZIndex]);
        double deltaR_qj = deltaR[ReferenceForce::RIndex];
        double deltaR2 = deltaR_qj*deltaR_qj;
        double atomicRadii_j = _atomicRadii[atomJ];
        double atomicRadii_j2 = atomicRadii_j*atomicRadii_j;

        double C_j = _P1*atomicRadii_j + _P2;
        double F_VSA = C_j / (C_j + deltaR2 - atomicRadii_j2);
        F_VSA = F_VSA*F_VSA;

        double tmp0 = (C_j + deltaR2 - atomicRadii_j2);
        double dF_VSA_dr_factor = -4.0*C_j*C_j/(tmp0*tmp0*tmp0);
        double tmp1 = dF_VSA_dr_factor * sum2 / (sum3*sum3) * factor;

        gradients[atomI] -= dr * tmp1;
        gradients[atomJ] += dr * tmp1;
        
        double tmp2 = 2.0*F_VSA*(F_VSA + dF_VSA_dr_factor*deltaR2) / (sum3*sum3) * sum1 * factor;
        gradients[atomI] -= dr * tmp2;
        gradients[atomJ] += dr * tmp2;

        double tmp3 = -2.0 / (sum3*sum3*sum3*sum3) * sum1 * sum2 * factor ;
        OpenMM::Vec3 denom_vec_dr1;
        denom_vec_dr1[0] = tmp3 * (denom_vec[0] * F_VSA + 
                (denom_vec[0]*dr[0] + denom_vec[1]*dr[1] + denom_vec[2]*dr[2])*dF_VSA_dr_factor*dr[0]);
        
        denom_vec_dr1[1] = tmp3 * (denom_vec[1] * F_VSA + 
                (denom_vec[0]*dr[0] + denom_vec[1]*dr[1] + denom_vec[2]*dr[2])*dF_VSA_dr_factor*dr[1]);

        denom_vec_dr1[2] = tmp3 * (denom_vec[2] * F_VSA + 
                (denom_vec[0]*dr[0] + denom_vec[1]*dr[1] + denom_vec[2]*dr[2])*dF_VSA_dr_factor*dr[2]);
        gradients[atomI] -= denom_vec_dr1;
        gradients[atomJ] += denom_vec_dr1;

    }
    return;
}

void GBMVIntegralTypeII::computeLookupTable(const std::vector<OpenMM::Vec3>& atomCoordinates){
    //_lookupTable;
    //  
    //_r1;
    OpenMM::Vec3 minCoordinate(atomCoordinates[0]);
    OpenMM::Vec3 maxCoordinate(atomCoordinates[0]);
    double maxR = 0.0;
    for(int atomI=0; atomI<_numParticles; ++atomI){
        for(int i=0; i<3; ++i){
            minCoordinate[i] = min(minCoordinate[i], atomCoordinates[atomI][i]);
            maxCoordinate[i] = max(maxCoordinate[i], atomCoordinates[atomI][i]);
        }   
        maxR = max(maxR, _atomicRadii[atomI]);
    }   
    double paddingLength = _lookupTableBufferLength + maxR  +
        sqrt(3.0)/2.0*_lookupTableGridLength + 1e-6;
    int totalNumberOfGridPoints = 1;
    for(int i=0; i<3; ++i){
        minCoordinate[i] -= paddingLength;
        maxCoordinate[i] += paddingLength;
        double length = maxCoordinate[i]-minCoordinate[i];
        _lookupTableNumberOfGridPoints[i] = static_cast<int>(
                ceil(length/_lookupTableGridLength))+1;
        if(length > 1000)
            throw OpenMM::OpenMMException("CharmmGBMVForce: GBMVIntegralTypeII lookup table dimension is too large, check atom positions!");
        _lookupTableMinCoordinate[i] = minCoordinate[i];
        _lookupTableMaxCoordinate[i] = minCoordinate[i]+(_lookupTableNumberOfGridPoints[i]-1)*_lookupTableGridLength;
        totalNumberOfGridPoints *= _lookupTableNumberOfGridPoints[i];
        //cout<<minCoordinate[i]<<" "<<maxCoordinate[i]<<" "<<_lookupTableNumberOfGridPoints[i]<<endl;
    }   
    int n_x = _lookupTableNumberOfGridPoints[0];
    int n_y = _lookupTableNumberOfGridPoints[1];
    int n_z = _lookupTableNumberOfGridPoints[2];
    _lookupTable.clear();
    _lookupTable.resize(totalNumberOfGridPoints,vector<int>());
    for(int atomI=0; atomI<_numParticles; ++atomI){
        OpenMM::Vec3 coor = atomCoordinates[atomI];
        int beginLookupTableIndex[3];
        int endLookupTableIndex[3];
        for(int i=0; i<3; ++i){
            beginLookupTableIndex[i] = floor(
                    (coor[i]-paddingLength-_lookupTableMinCoordinate[i])/_lookupTableGridLength);
            endLookupTableIndex[i] = ceil(
                    (coor[i]+paddingLength-_lookupTableMinCoordinate[i])/_lookupTableGridLength);
        }
        for(int i=beginLookupTableIndex[0]; i<=endLookupTableIndex[0]; ++i){ //x
            for(int j=beginLookupTableIndex[1]; j<=endLookupTableIndex[1]; ++j){ //y
                for(int k=beginLookupTableIndex[2]; k<=endLookupTableIndex[2]; ++k){ //z
                    int idx = i*n_y*n_z + j*n_z + k; //calculate grid idx
                    OpenMM::Vec3 gridPoint(_lookupTableMinCoordinate[0]+i*_lookupTableGridLength,
                            _lookupTableMinCoordinate[1]+j*_lookupTableGridLength,
                            _lookupTableMinCoordinate[2]+k*_lookupTableGridLength);
                    OpenMM::Vec3 diff = gridPoint - coor;
                    if(sqrt(diff.dot(diff)) < paddingLength){
                        _lookupTable[idx].push_back(atomI);
                    }
                }
            }
        }
    }
}

std::vector<int> GBMVIntegralTypeII::getLookupTableAtomList(OpenMM::Vec3 point){
    vector<int> atomList;
    int nx = _lookupTableNumberOfGridPoints[0];
    int ny = _lookupTableNumberOfGridPoints[1];
    int nz = _lookupTableNumberOfGridPoints[2];
    int idx[3];
    if(_periodic){
        for(int i=0; i<3; ++i){
            if(point[i] < _lookupTableMinCoordinate[i])
                point += _periodicBoxVectors[i];
            if(point[i] > _lookupTableMaxCoordinate[i])
                point -= _periodicBoxVectors[i];
        }
        for(int i=0; i<3; ++i){
            //if point is still outside of the lookupTable grid
            if((point[i] < _lookupTableMinCoordinate[i]) ||
                    (point[i] > _lookupTableMaxCoordinate[i])){
                return atomList;
            }
            idx[i] = static_cast<int>(floor(
                        (point[i]-_lookupTableMinCoordinate[i]) / _lookupTableGridLength));
        }
    }else{
        for(int i=0; i<3; ++i){
            //if point is outside of the lookupTable grid
            if((point[i] < _lookupTableMinCoordinate[i]) ||
                    (point[i] > _lookupTableMaxCoordinate[i])){
                return atomList;
            }
            idx[i] = static_cast<int>(floor(
                        (point[i]-_lookupTableMinCoordinate[i]) / _lookupTableGridLength));
        }
    }
    int lookupTableIdx = idx[0]*(ny*nz) + idx[1]*nz + idx[2];
    atomList = _lookupTable[lookupTableIdx];
    return atomList;
}
