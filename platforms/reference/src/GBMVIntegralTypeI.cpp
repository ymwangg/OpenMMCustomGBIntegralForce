#include "openmm/internal/CharmmQuadrature.h"
#include "ReferenceForce.h"
#include "ReferencePlatform.h"
#include "Units.h"
#include "Vec3.h"
#include "GBMVIntegralTypeI.h"
#include "openmm/OpenMMException.h"


using namespace::OpenMM;
using namespace::std;

static Vec3* extractBoxVectors(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return (Vec3*) data->periodicBoxVectors;
}

GBMVIntegralTypeI::GBMVIntegralTypeI(){
    _useLookupTable = true;
    _periodic = false;

    _gamma0 = 0.44;
    _beta = -100;
    _lambda = 0.1;

    //radial integral starting point
    _r0 = 0.5*OpenMM::NmPerAngstrom;

    //radial integral ending point
    _r1 = 20.0*OpenMM::NmPerAngstrom;

    //number of radial integration points using Gauss-Legendre quadrature
    int nRadialPoints = 24; 

    //rule of Lebedev quadrature for spherical integral
    int ruleLebedev = 3;
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
    setLookupTableBufferLength(0.21); //0.20nm
}

void GBMVIntegralTypeI::initialize(const OpenMM::System& system, const OpenMM::CharmmGBMVForce& force){
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
        throw OpenMMException("GBMVIntegralTypeI: the per perticle parameter 'radius' must be defined");
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
            throw OpenMMException("GBMVIntegralTypeI: the order of integral must be given");
        if(parInt[0] < 2)
            throw OpenMMException("GBMVIntegralTypeI: the order of integral must be greater or equal to 2");
        _orders.push_back(parInt[0]-2);
    }
    return;
}

void GBMVIntegralTypeI::setBoxVectors(OpenMM::Vec3* vectors){
    setPeriodic(vectors);
    _periodicBoxVectors[0] = vectors[0];
    _periodicBoxVectors[1] = vectors[1];
    _periodicBoxVectors[2] = vectors[2];
}

void GBMVIntegralTypeI::BeforeComputation(ContextImpl& context, const std::vector<OpenMM::Vec3>& atomCoordinates){
    setBoxVectors(extractBoxVectors(context));
    computeLookupTable(atomCoordinates);
}

void GBMVIntegralTypeI::FinishComputation(ContextImpl& context, const std::vector<OpenMM::Vec3>& atomCoordinates){
    //do nothing
}

void GBMVIntegralTypeI::evaluate(ContextImpl& context, const std::vector<OpenMM::Vec3>& atomCoordinates, std::vector<double>& integrals, std::vector<double>& gradients, const bool includeGradient){
    // flush integrals and gradients
    std::fill(integrals.begin(),integrals.end(),0.0);
    std::fill(gradients.begin(),gradients.end(),0.0);
    vector<double> prefactors(_numIntegrals);
    OpenMM::Vec3 r_q;
    vector<int>* atomList;
    for(int q=0; q<_quad.size(); ++q){
        double radius_q = _quad[q][3];
        double w_q = _quad[q][4];
        for(int atomI=0; atomI<_numParticles; ++atomI){
            // if the quadrature point is within the vdw radii
            // ignore it
            //if(_atomicRadii[atomI] > radius_q) continue;
            // compute the quadrature point position
            for(int i=0; i<3; ++i) 
                r_q[i] = atomCoordinates[atomI][i] + _quad[q][i];
            int numListAtoms = 0;
            double pre_sum = 0;
            if(_useLookupTable){
                getLookupTableAtomList(r_q, atomList, numListAtoms);
                pre_sum = computeVolumeFromLookupTable(atomCoordinates, r_q, *atomList, numListAtoms);
            }else{
                pre_sum = computeVolume(atomCoordinates, r_q);
            }
            double V_q = 1.0/(1.0 + pre_sum);
            for(int i=0; i<_numIntegrals; ++i){
                int integral_globalIdx = i*_numParticles + atomI;
                prefactors[i] = w_q/pow(radius_q, _orders[i]);
                integrals[integral_globalIdx] += prefactors[i]*(V_q);
            }
            for(int i=0; i<_numIntegrals; ++i){
                if(_useLookupTable){
                    computeGradientPerQuadFromLookupTable(atomI, i, atomCoordinates, r_q, pre_sum, gradients, prefactors[i], *atomList, numListAtoms);
                }else{
                    computeGradientPerQuad(atomI, i, atomCoordinates, r_q, pre_sum, gradients, prefactors[i]);
                }
            }
        }
    }
    return;
}


double GBMVIntegralTypeI::computeVolumeFromLookupTable(const std::vector<OpenMM::Vec3>& atomCoordinates, const OpenMM::Vec3& r_q, const std::vector<int>& atomList, const int numListAtoms){
    double deltaR[ReferenceForce::LastDeltaRIndex];
    double sum = 0.0;
    for(int i=0; i<numListAtoms; ++i){
        int atomJ = atomList[i];
        if (_periodic)
            ReferenceForce::getDeltaRPeriodic(r_q, atomCoordinates[atomJ], _periodicBoxVectors, deltaR);
        else
            ReferenceForce::getDeltaR(r_q, atomCoordinates[atomJ], deltaR);
        double deltaR_qj = deltaR[ReferenceForce::RIndex];
        double deltaR4 = deltaR_qj*deltaR_qj*deltaR_qj*deltaR_qj;
        double atomicRadii_j = _atomicRadii[atomJ];
        double atomicRadii_j4 = atomicRadii_j*atomicRadii_j*atomicRadii_j*atomicRadii_j;
        double gammaj = _gamma0 * log(_lambda) / (atomicRadii_j4);
        sum += exp(gammaj * deltaR4);
    }
    double pre_sum = exp(_beta*(sum - _lambda));
    return pre_sum;
}

double GBMVIntegralTypeI::computeVolume(const std::vector<OpenMM::Vec3>& atomCoordinates, const OpenMM::Vec3& r_q){
    double deltaR[ReferenceForce::LastDeltaRIndex];
    double sum = 0.0;
    for(int atomJ=0; atomJ<_numParticles; ++atomJ){
        if (_periodic)
            ReferenceForce::getDeltaRPeriodic(r_q, atomCoordinates[atomJ], _periodicBoxVectors, deltaR);
        else
            ReferenceForce::getDeltaR(r_q, atomCoordinates[atomJ], deltaR);
        double deltaR_qj = deltaR[ReferenceForce::RIndex];
        double deltaR4 = deltaR_qj*deltaR_qj*deltaR_qj*deltaR_qj;
        double atomicRadii_j = _atomicRadii[atomJ];
        double atomicRadii_j4 = atomicRadii_j*atomicRadii_j*atomicRadii_j*atomicRadii_j;
        double gammaj = _gamma0 * log(_lambda) / (atomicRadii_j4);
        sum += exp(gammaj * deltaR4);
    }
    double pre_sum = exp(_beta*(sum - _lambda));
    return pre_sum;
}


void GBMVIntegralTypeI::computeGradientPerQuadFromLookupTable(const int atomI, const int integralIdx, const std::vector<OpenMM::Vec3>& atomCoordinates, const OpenMM::Vec3& r_q, const double pre_sum, std::vector<double>& gradients, const double prefactor, const std::vector<int>& atomList, const int numListAtoms){
    double deltaR[ReferenceForce::LastDeltaRIndex];
    double factor = -1.0/((1.0+pre_sum)*(1.0+pre_sum)) * (_beta*pre_sum) * prefactor;
    double four = 4.0;
    for(int i=0; i<numListAtoms; ++i){
        int atomJ = atomList[i];
        if (_periodic)
            ReferenceForce::getDeltaRPeriodic(r_q, atomCoordinates[atomJ], _periodicBoxVectors, deltaR);
        else
            ReferenceForce::getDeltaR(r_q, atomCoordinates[atomJ], deltaR);
        double deltaR_qj = deltaR[ReferenceForce::RIndex];
        double deltaR3 = deltaR_qj*deltaR_qj*deltaR_qj;
        double deltaR4 = deltaR3*deltaR_qj;
        double atomicRadii_j = _atomicRadii[atomJ];
        double atomicRadii_j4 = atomicRadii_j*atomicRadii_j*atomicRadii_j*atomicRadii_j;
        double gammaj = _gamma0 * log(_lambda) / (atomicRadii_j4);
        double factor2 = factor*gammaj*exp(gammaj*deltaR4)*(four*deltaR3);
        OpenMM::Vec3 r_qj_vec(deltaR[ReferenceForce::XIndex],deltaR[ReferenceForce::YIndex],
                deltaR[ReferenceForce::ZIndex]);
        r_qj_vec = r_qj_vec / deltaR_qj * factor2;

        int grad_idx_i = integralIdx*_numParticles*_numParticles*3 +  atomI*_numParticles*3 + atomI*3;
        int grad_idx_j = integralIdx*_numParticles*_numParticles*3 +  atomI*_numParticles*3 + atomJ*3;
        for(int n = 0; n < 3; ++n){
            gradients[grad_idx_i + n] -= r_qj_vec[n];
            gradients[grad_idx_j + n] += r_qj_vec[n];
        }  
    }
    return;
}

void GBMVIntegralTypeI::computeGradientPerQuad(const int atomI, const int integralIdx, const std::vector<OpenMM::Vec3>& atomCoordinates, const OpenMM::Vec3& r_q, const double pre_sum, std::vector<double>& gradients, const double prefactor){
    double deltaR[ReferenceForce::LastDeltaRIndex];
    double factor = -1.0/((1.0+pre_sum)*(1.0+pre_sum)) * (_beta*pre_sum) * prefactor;
    double four = 4.0;
    for(int atomJ=0; atomJ<_numParticles; ++atomJ){
        if (_periodic)
            ReferenceForce::getDeltaRPeriodic(r_q, atomCoordinates[atomJ], _periodicBoxVectors, deltaR);
        else
            ReferenceForce::getDeltaR(r_q, atomCoordinates[atomJ], deltaR);
        double deltaR_qj = deltaR[ReferenceForce::RIndex];
        double deltaR3 = deltaR_qj*deltaR_qj*deltaR_qj;
        double deltaR4 = deltaR3*deltaR_qj;
        double atomicRadii_j = _atomicRadii[atomJ];
        double atomicRadii_j4 = atomicRadii_j*atomicRadii_j*atomicRadii_j*atomicRadii_j;
        double gammaj = _gamma0 * log(_lambda) / (atomicRadii_j4);
        double factor2 = factor*gammaj*exp(gammaj*deltaR4)*(four*deltaR3);
        OpenMM::Vec3 r_qj_vec(deltaR[ReferenceForce::XIndex],deltaR[ReferenceForce::YIndex],
                deltaR[ReferenceForce::ZIndex]);
        r_qj_vec = r_qj_vec / deltaR_qj * factor2;

        int grad_idx_i = integralIdx*_numParticles*_numParticles*3 +  atomI*_numParticles*3 + atomI*3;
        int grad_idx_j = integralIdx*_numParticles*_numParticles*3 +  atomI*_numParticles*3 + atomJ*3;
        for(int n = 0; n < 3; ++n){
            gradients[grad_idx_i + n] -= r_qj_vec[n];
            gradients[grad_idx_j + n] += r_qj_vec[n];
        }  
    }
    return;
}

