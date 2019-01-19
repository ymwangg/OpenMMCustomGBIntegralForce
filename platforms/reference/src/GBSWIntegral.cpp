// timing
#include <chrono>
#define BEGIN auto start = std::chrono::system_clock::now();
#define END auto end = std::chrono::system_clock::now();std::chrono::duration<double> elapsed_seconds = end-start;cout<<"calculateIxn elapsed time: " << elapsed_seconds.count()<<endl;

#define USE_LOOKUP_TABLE true

#include "openmm/internal/CharmmQuadrature.h"
#include "ReferenceForce.h"
#include "ReferencePlatform.h"
#include "Units.h"
#include "Vec3.h"
#include "GBSWIntegral.h"
#include "openmm/OpenMMException.h"


using namespace::OpenMM;
using namespace::std;

static Vec3* extractBoxVectors(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return (Vec3*) data->periodicBoxVectors;
}

GBSWIntegral::GBSWIntegral(){
    _periodic = false;
    //half of switching distance
    _sw = 0.3*OpenMM::NmPerAngstrom;

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
    setLookupTableGridLength(0.15); //0.15nm
    setLookupTableBufferLength(0.00+_sw); //0.20nm
}

void GBSWIntegral::initialize(const OpenMM::System& system, const OpenMM::CharmmGBMVForce& force){
    if(OpenMM::CharmmGBMVForce::CutoffPeriodic==force.getNonbondedMethod()){
        _periodic = true;
    }else{
        _periodic = false;
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
        throw OpenMMException("GBSWIntegral: the per perticle parameter 'radius' must be defined");
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
            throw OpenMMException("GBSWIntegral: the order of integral must be given");
        if(parInt[0] < 2)
            throw OpenMMException("GBSWIntegral: the order of integral must be greater or equal to 2");
        _orders.push_back(parInt[0]-2);
    }
    return;
}

void GBSWIntegral::setBoxVectors(OpenMM::Vec3* vectors){
    setPeriodic(vectors); //set the periodic box vectors of the inherited class CustomGBIntegral
    _periodicBoxVectors[0] = vectors[0];
    _periodicBoxVectors[1] = vectors[1];
    _periodicBoxVectors[2] = vectors[2];
}

void GBSWIntegral::BeforeComputation(ContextImpl& context, const std::vector<OpenMM::Vec3>& atomCoordinates){
    if(_periodic) setBoxVectors(extractBoxVectors(context));
#if USE_LOOKUP_TABLE
    computeLookupTable(atomCoordinates);
#endif
}

void GBSWIntegral::FinishComputation(ContextImpl& context, const std::vector<OpenMM::Vec3>& atomCoordinates){
    //do nothing
}

void GBSWIntegral::evaluate(ContextImpl& context, const std::vector<OpenMM::Vec3>& atomCoordinates, std::vector<double>& integrals, std::vector<double>& gradients, const bool includeGradient){
    std::fill(integrals.begin(),integrals.end(),0.0);
    std::fill(gradients.begin(),gradients.end(),0.0);
    OpenMM::Vec3 r_q;
    double radius_q;
    double w_q;
    double V_q;
    vector<double> prefactors(_numIntegrals);
    vector<int>* atomList; 
    for(int q=0; q<_quad.size(); ++q){
        radius_q = _quad[q][3];
        w_q = _quad[q][4];
        for(int atomI=0; atomI<_numParticles; ++atomI){
            for(int i=0; i<3; ++i) 
                r_q[i] = atomCoordinates[atomI][i] + _quad[q][i];
#if USE_LOOKUP_TABLE
            int numListAtoms;
            getLookupTableAtomList(r_q, atomList, numListAtoms);
            if(numListAtoms==0) continue;
            //printf("getting%p\n",&atomList);
            V_q = computeVolumeFromLookupTable(atomCoordinates, r_q, *atomList, numListAtoms);
#else
            V_q = computeVolume(atomCoordinates, r_q);
#endif
            for(int i=0; i<_numIntegrals; ++i){
                int integral_globalIdx = i*_numParticles + atomI;
                prefactors[i] = w_q/pow(radius_q, _orders[i]);
                integrals[integral_globalIdx] += prefactors[i]*(1.0 - V_q);
            }
            if(includeGradient){
                for(int i=0; i<_orders.size(); ++i){
#if USE_LOOKUP_TABLE
                    computeGradientPerQuadFromLookupTable(atomI, i, atomCoordinates, r_q, V_q, gradients, prefactors[i], *atomList, numListAtoms);
#else
                    computeGradientPerQuad(atomI, i, atomCoordinates, r_q, V_q, gradients, prefactors[i]);
#endif
                }
            }
        }
    }
    return;
}

double GBSWIntegral::computeVolume(const std::vector<OpenMM::Vec3>& atomCoordinates, const OpenMM::Vec3& r_q){
    double V = 1.0;
    double deltaR[ReferenceForce::LastDeltaRIndex];
    double deltaR_qj;
    double atomicRadii_j;
    double sw = _sw;
    double sw3 = sw*sw*sw;
    double dr, dr3;
    for(int atomJ=0; atomJ<atomCoordinates.size(); ++atomJ){
        if (_periodic)
            ReferenceForce::getDeltaRPeriodic(r_q, atomCoordinates[atomJ], _periodicBoxVectors, deltaR);
        else
            ReferenceForce::getDeltaR(r_q, atomCoordinates[atomJ], deltaR);
        deltaR_qj = deltaR[ReferenceForce::RIndex];
        atomicRadii_j = (_atomicRadii[atomJ]+0.03)*0.9520;
        dr = deltaR_qj - atomicRadii_j;
        dr3 = dr*dr*dr;
        if(deltaR_qj <= atomicRadii_j - sw){
            return 0.0;
        }else if(deltaR_qj >= atomicRadii_j + sw){
            continue;
        }else{
            V *= 0.5 + 3.0/(4.0*sw) * dr - 1.0/(4.0*sw3) * dr3;
        }
    }
    return V;
}

inline double GBSWIntegral::computeVolumeFromLookupTable(const std::vector<OpenMM::Vec3>& atomCoordinates, const OpenMM::Vec3& r_q, const std::vector<int>& atomList, const int numListAtoms){
    double V = 1.0;
    double deltaR[ReferenceForce::LastDeltaRIndex];
    double deltaR_qj;
    double atomicRadii_j;
    double sw = _sw;
    double sw3 = sw*sw*sw;
    double dr, dr3;
    /*
    if(numListAtoms!=0){
        printf("(%f,%f,%f)",r_q[0],r_q[1],r_q[2]);
        for(int i=0; i<numListAtoms; ++i){
            int atomJ = atomList[i];
            printf("-%d",atomJ);
        }
        printf("\n");
    }
    */
    for(int i=0; i<numListAtoms; ++i){
        int atomJ = atomList[i];
        if (_periodic)
            ReferenceForce::getDeltaRPeriodic(r_q, atomCoordinates[atomJ], _periodicBoxVectors, deltaR);
        else
            ReferenceForce::getDeltaR(r_q, atomCoordinates[atomJ], deltaR);
        deltaR_qj = deltaR[ReferenceForce::RIndex];
        atomicRadii_j = (_atomicRadii[atomJ]+0.03)*0.9520;
        dr = deltaR_qj - atomicRadii_j;
        dr3 = dr*dr*dr;
        if(deltaR_qj <= atomicRadii_j - sw){
            return 0.0;
        }else if(deltaR_qj >= atomicRadii_j + sw){
            continue;
        }else{
            V *= 0.5 + 3.0/(4.0*sw) * dr - 1.0/(4.0*sw3) * dr3;
        }
    }
    return V;
}

inline void GBSWIntegral::computeGradientPerQuad(const int atomI, const int integralIdx, const std::vector<OpenMM::Vec3>& atomCoordinates, 
        const OpenMM::Vec3& r_q, const double V_q, std::vector<double>& gradients, const double prefactor){
    if(V_q == 0) return;
    double deltaR[ReferenceForce::LastDeltaRIndex];
    double deltaR_qj;
    double atomicRadii_j;
    double sw = _sw;
    double sw3 = sw*sw*sw;
    double dr, dr2, dr3;
    double u_j;
    double duj_drq;
    OpenMM::Vec3 dV_drq;
    for(int atomJ=0; atomJ<atomCoordinates.size(); ++atomJ){
        if(atomI==atomJ) continue;
        if (_periodic)
            ReferenceForce::getDeltaRPeriodic(r_q, atomCoordinates[atomJ], _periodicBoxVectors, deltaR);
        else
            ReferenceForce::getDeltaR(r_q, atomCoordinates[atomJ], deltaR);
        deltaR_qj = deltaR[ReferenceForce::RIndex];
        atomicRadii_j = (_atomicRadii[atomJ]+0.03)*0.9520;
        dr = deltaR_qj - atomicRadii_j;
        dr2 = dr*dr;
        dr3 = dr*dr*dr;
        if((deltaR_qj > atomicRadii_j - sw) && (deltaR_qj < atomicRadii_j + sw)){
            u_j = 0.5 + 3.0/(4.0*sw) * dr - 1.0/(4.0*sw3) * dr3;
            duj_drq = 3.0/(4.0*sw) - 3.0/(4.0*sw3) * dr2;
        }else{
            continue;
        }
        OpenMM::Vec3 r_qj_vec(deltaR[ReferenceForce::XIndex],deltaR[ReferenceForce::YIndex],
                deltaR[ReferenceForce::ZIndex]);
        double factor = prefactor*(duj_drq*V_q/u_j)/deltaR_qj;
        dV_drq[0] = deltaR[ReferenceForce::XIndex]*factor;
        dV_drq[1] = deltaR[ReferenceForce::YIndex]*factor;
        dV_drq[2] = deltaR[ReferenceForce::ZIndex]*factor;
        int grad_idx_i = integralIdx*_numParticles*_numParticles*3 +  atomI*_numParticles*3 + atomI*3;
        int grad_idx_j = integralIdx*_numParticles*_numParticles*3 +  atomI*_numParticles*3 + atomJ*3;
        for(int n = 0; n < 3; ++n){
            gradients[grad_idx_i + n] += dV_drq[n];
            gradients[grad_idx_j + n] -= dV_drq[n];
        }
    }
    return;
}

inline void GBSWIntegral::computeGradientPerQuadFromLookupTable(const int atomI, const int integralIdx, const std::vector<OpenMM::Vec3>& atomCoordinates, const OpenMM::Vec3& r_q, const double V_q, std::vector<double>& gradients, const double prefactor, const std::vector<int>& atomList, const int numListAtoms){
    if(V_q == 0) return;
    double deltaR[ReferenceForce::LastDeltaRIndex];
    double deltaR_qj;
    double atomicRadii_j;
    double sw = _sw;
    double sw3 = sw*sw*sw;
    double dr, dr2, dr3;
    double u_j;
    double duj_drq;
    OpenMM::Vec3 dV_drq;
    for(int i=0; i<numListAtoms; ++i){
        int atomJ = atomList[i];
        if(atomI==atomJ) continue;
        if (_periodic)
            ReferenceForce::getDeltaRPeriodic(r_q, atomCoordinates[atomJ], _periodicBoxVectors, deltaR);
        else
            ReferenceForce::getDeltaR(r_q, atomCoordinates[atomJ], deltaR);
        deltaR_qj = deltaR[ReferenceForce::RIndex];
        atomicRadii_j = (_atomicRadii[atomJ]+0.03)*0.9520;
        dr = deltaR_qj - atomicRadii_j;
        dr2 = dr*dr;
        dr3 = dr*dr*dr;
        if((deltaR_qj > atomicRadii_j - sw) && (deltaR_qj < atomicRadii_j + sw)){
            u_j = 0.5 + 3.0/(4.0*sw) * dr - 1.0/(4.0*sw3) * dr3;
            duj_drq = 3.0/(4.0*sw) - 3.0/(4.0*sw3) * dr2;
        }else{
            continue;
        }
        OpenMM::Vec3 r_qj_vec(deltaR[ReferenceForce::XIndex],deltaR[ReferenceForce::YIndex],
                deltaR[ReferenceForce::ZIndex]);
        double factor = prefactor*(duj_drq*V_q/u_j)/deltaR_qj;
        dV_drq[0] = deltaR[ReferenceForce::XIndex]*factor;
        dV_drq[1] = deltaR[ReferenceForce::YIndex]*factor;
        dV_drq[2] = deltaR[ReferenceForce::ZIndex]*factor;
        int grad_idx_i = integralIdx*_numParticles*_numParticles*3 +  atomI*_numParticles*3 + atomI*3;
        int grad_idx_j = integralIdx*_numParticles*_numParticles*3 +  atomI*_numParticles*3 + atomJ*3;
        for(int n = 0; n < 3; ++n){
            gradients[grad_idx_i + n] += dV_drq[n];
            gradients[grad_idx_j + n] -= dV_drq[n];
        }
    }
    return;
}
