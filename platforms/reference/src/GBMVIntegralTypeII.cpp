#include "openmm/internal/CharmmQuadrature.h"
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
    _useLookupTable = true;
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
    setPeriodic(vectors);
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

void GBMVIntegralTypeII::evaluate(ContextImpl& context, const std::vector<OpenMM::Vec3>& atomCoordinates, std::vector<double>& integrals, std::vector<double>& gradients, const bool includeGradient){
    std::fill(integrals.begin(),integrals.end(),0.0);
    std::fill(gradients.begin(),gradients.end(),0.0);

    vector<double> prefactors(_numIntegrals);
    OpenMM::Vec3 r_q;
    double radius_q;
    double w_q;
    vector<int>* atomList;
    for(int q=0; q<_quad.size(); ++q){
        radius_q = _quad[q][3];
        w_q = _quad[q][4];
        for(int atomI=0; atomI < _numParticles; ++atomI){
            for(int i=0; i<3; ++i) 
                r_q[i] = atomCoordinates[atomI][i] + _quad[q][i];
            int numListAtoms;
            double sum1, sum2, sum3;
            sum1 = 0.0; sum2 = 0.0; sum3 = 0.0;
            OpenMM::Vec3 denom_vec(0.0, 0.0, 0.0);
            double pre_sum;
            double V_q;
            if(_useLookupTable){
                getLookupTableAtomList(r_q, atomList, numListAtoms);
                pre_sum = computeVolumeFromLookupTable(atomCoordinates, r_q, *atomList, numListAtoms, sum1, sum2, sum3, denom_vec);
                V_q = (numListAtoms==0 ? 0 : 1.0/(1.0 + pre_sum));
            }else{
                pre_sum = computeVolume(atomCoordinates, r_q, sum1, sum2, sum3, denom_vec);
                V_q = 1.0/(1.0 + pre_sum);
            }
            for(int i=0; i<_numIntegrals; ++i){
                int integral_globalIdx = i*_numParticles + atomI;
                prefactors[i] = w_q/pow(radius_q, _orders[i]);
                integrals[integral_globalIdx] += prefactors[i]*(V_q);
            }
            for(int i=0; i<_numIntegrals; ++i){
                if(_useLookupTable){
                    computeGradientPerQuadFromLookupTable(atomI, i, atomCoordinates, r_q, pre_sum, gradients, prefactors[i], *atomList, numListAtoms, sum1, sum2, sum3, denom_vec);
                }else{
                    computeGradientPerQuad(atomI, i, atomCoordinates, r_q, pre_sum, gradients, prefactors[i], sum1, sum2, sum3, denom_vec);
                }
            }
        }
    }
    return;
}


double GBMVIntegralTypeII::computeVolumeFromLookupTable(const std::vector<OpenMM::Vec3>& atomCoordinates, const OpenMM::Vec3& r_q, const std::vector<int>& atomList, const int numListAtoms, double& sum1, double& sum2, double& sum3, OpenMM::Vec3& denom_vec){
    if(numListAtoms==0) return 0;
    double deltaR[ReferenceForce::LastDeltaRIndex];
    sum1 = 0.0;
    sum2 = 0.0;
    for(int i=0; i<numListAtoms; ++i){
        int atomJ = atomList[i];
        if (_periodic)
            ReferenceForce::getDeltaRPeriodic(r_q, atomCoordinates[atomJ], _periodicBoxVectors, deltaR);
        else
            ReferenceForce::getDeltaR(r_q, atomCoordinates[atomJ], deltaR);
        double deltaR_qj = deltaR[ReferenceForce::RIndex];
        double deltaR2 = deltaR_qj*deltaR_qj;
        double atomicRadii_j = _atomicRadii[atomJ];
        double atomicRadii_j2 = atomicRadii_j*atomicRadii_j;
        // the factor 10 and 100 are used to convert the unit from nm to Angstrom
        double C_j = _P1*atomicRadii_j*10 + _P2;
        double F_VSA = C_j / (C_j + deltaR2*100 - atomicRadii_j2*100);
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

double GBMVIntegralTypeII::computeVolume(const std::vector<OpenMM::Vec3>& atomCoordinates, const OpenMM::Vec3& r_q, double& sum1, double& sum2, double& sum3, OpenMM::Vec3& denom_vec){
    double deltaR[ReferenceForce::LastDeltaRIndex];
    sum1 = 0.0;
    sum2 = 0.0;
    for(int atomJ=0; atomJ<_numParticles; ++atomJ){
        if (_periodic)
            ReferenceForce::getDeltaRPeriodic(r_q, atomCoordinates[atomJ], _periodicBoxVectors, deltaR);
        else
            ReferenceForce::getDeltaR(r_q, atomCoordinates[atomJ], deltaR);
        double deltaR_qj = deltaR[ReferenceForce::RIndex];
        double deltaR2 = deltaR_qj*deltaR_qj;
        double atomicRadii_j = _atomicRadii[atomJ];
        double atomicRadii_j2 = atomicRadii_j*atomicRadii_j;
        double C_j = _P1*atomicRadii_j*10 + _P2;
        double F_VSA = C_j / (C_j + deltaR2*100 - atomicRadii_j2*100);
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

void GBMVIntegralTypeII::computeGradientPerQuadFromLookupTable(const int atomI, const int integralIdx, const std::vector<OpenMM::Vec3>& atomCoordinates, const OpenMM::Vec3& r_q, const double pre_sum, std::vector<double>& gradients, const double prefactor, const std::vector<int>& atomList, const int numListAtoms, const double sum1, const double sum2, const double sum3, const OpenMM::Vec3& denom_vec){
    if(numListAtoms==0) return;
    double deltaR[ReferenceForce::LastDeltaRIndex];
    double factor = -1.0/((1.0+pre_sum)*(1.0+pre_sum)) * (_beta*pre_sum) * prefactor * _S0;
    OpenMM::Vec3 tmp_grad;
    for(int i=0; i<numListAtoms; ++i){
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

        double C_j = _P1*atomicRadii_j*10 + _P2;
        double F_VSA = C_j / (C_j + deltaR2*100 - atomicRadii_j2*100);
        F_VSA = F_VSA*F_VSA;

        double tmp0 = (C_j + deltaR2 - atomicRadii_j2);
        double dF_VSA_dr_factor = -4.0*C_j*C_j/(tmp0*tmp0*tmp0);
        double tmp1 = dF_VSA_dr_factor * sum2 / (sum3*sum3) * factor;

        int grad_idx_i = integralIdx*_numParticles*_numParticles*3 +  atomI*_numParticles*3 + atomI*3;
        int grad_idx_j = integralIdx*_numParticles*_numParticles*3 +  atomI*_numParticles*3 + atomJ*3;

        tmp_grad = dr * tmp1;
        for(int n = 0; n < 3; ++n){
            gradients[grad_idx_i + n] -= tmp_grad[n];
            gradients[grad_idx_j + n] += tmp_grad[n];
        } 
        
        double tmp2 = 2.0*F_VSA*(F_VSA + dF_VSA_dr_factor*deltaR2) / (sum3*sum3) * sum1 * factor;
        tmp_grad = dr * tmp2;
        for(int n = 0; n < 3; ++n){
            gradients[grad_idx_i + n] -= tmp_grad[n];
            gradients[grad_idx_j + n] += tmp_grad[n];
        } 

        double tmp3 = -2.0 / (sum3*sum3*sum3*sum3) * sum1 * sum2 * factor ;
        OpenMM::Vec3 denom_vec_dr1;
        denom_vec_dr1[0] = tmp3 * (denom_vec[0] * F_VSA + 
                (denom_vec[0]*dr[0] + denom_vec[1]*dr[1] + denom_vec[2]*dr[2])*dF_VSA_dr_factor*dr[0]);
        
        denom_vec_dr1[1] = tmp3 * (denom_vec[1] * F_VSA + 
                (denom_vec[0]*dr[0] + denom_vec[1]*dr[1] + denom_vec[2]*dr[2])*dF_VSA_dr_factor*dr[1]);

        denom_vec_dr1[2] = tmp3 * (denom_vec[2] * F_VSA + 
                (denom_vec[0]*dr[0] + denom_vec[1]*dr[1] + denom_vec[2]*dr[2])*dF_VSA_dr_factor*dr[2]);

        tmp_grad = denom_vec_dr1;
        for(int n = 0; n < 3; ++n){
            gradients[grad_idx_i + n] -= tmp_grad[n];
            gradients[grad_idx_j + n] += tmp_grad[n];
        } 

    }
    return;
}

void GBMVIntegralTypeII::computeGradientPerQuad(const int atomI, const int integralIdx, const std::vector<OpenMM::Vec3>& atomCoordinates, const OpenMM::Vec3& r_q, const double pre_sum, std::vector<double>& gradients, const double prefactor, const double sum1, const double sum2, const double sum3, const OpenMM::Vec3& denom_vec){
    double deltaR[ReferenceForce::LastDeltaRIndex];
    double factor = -1.0/((1.0+pre_sum)*(1.0+pre_sum)) * (_beta*pre_sum) * prefactor * _S0;
    OpenMM::Vec3 tmp_grad;
    for(int atomJ=0; atomJ<_numParticles; ++atomJ){
        if (_periodic)
            ReferenceForce::getDeltaRPeriodic(r_q, atomCoordinates[atomJ], _periodicBoxVectors, deltaR);
        else
            ReferenceForce::getDeltaR(r_q, atomCoordinates[atomJ], deltaR);
        OpenMM::Vec3 dr(deltaR[ReferenceForce::XIndex],deltaR[ReferenceForce::YIndex],deltaR[ReferenceForce::ZIndex]);
        double deltaR_qj = deltaR[ReferenceForce::RIndex];
        double deltaR2 = deltaR_qj*deltaR_qj;
        double atomicRadii_j = _atomicRadii[atomJ];
        double atomicRadii_j2 = atomicRadii_j*atomicRadii_j;

        double C_j = _P1*atomicRadii_j*10 + _P2;
        double F_VSA = C_j / (C_j + deltaR2*100 - atomicRadii_j2*100);
        F_VSA = F_VSA*F_VSA;

        double tmp0 = (C_j + deltaR2 - atomicRadii_j2);
        double dF_VSA_dr_factor = -4.0*C_j*C_j/(tmp0*tmp0*tmp0);
        double tmp1 = dF_VSA_dr_factor * sum2 / (sum3*sum3) * factor;

        int grad_idx_i = integralIdx*_numParticles*_numParticles*3 +  atomI*_numParticles*3 + atomI*3;
        int grad_idx_j = integralIdx*_numParticles*_numParticles*3 +  atomI*_numParticles*3 + atomJ*3;

        tmp_grad = dr * tmp1;
        for(int n = 0; n < 3; ++n){
            gradients[grad_idx_i + n] -= tmp_grad[n];
            gradients[grad_idx_j + n] += tmp_grad[n];
        } 
        
        double tmp2 = 2.0*F_VSA*(F_VSA + dF_VSA_dr_factor*deltaR2) / (sum3*sum3) * sum1 * factor;
        tmp_grad = dr * tmp2;
        for(int n = 0; n < 3; ++n){
            gradients[grad_idx_i + n] -= tmp_grad[n];
            gradients[grad_idx_j + n] += tmp_grad[n];
        } 

        double tmp3 = -2.0 / (sum3*sum3*sum3*sum3) * sum1 * sum2 * factor ;
        OpenMM::Vec3 denom_vec_dr1;
        denom_vec_dr1[0] = tmp3 * (denom_vec[0] * F_VSA + 
                (denom_vec[0]*dr[0] + denom_vec[1]*dr[1] + denom_vec[2]*dr[2])*dF_VSA_dr_factor*dr[0]);
        
        denom_vec_dr1[1] = tmp3 * (denom_vec[1] * F_VSA + 
                (denom_vec[0]*dr[0] + denom_vec[1]*dr[1] + denom_vec[2]*dr[2])*dF_VSA_dr_factor*dr[1]);

        denom_vec_dr1[2] = tmp3 * (denom_vec[2] * F_VSA + 
                (denom_vec[0]*dr[0] + denom_vec[1]*dr[1] + denom_vec[2]*dr[2])*dF_VSA_dr_factor*dr[2]);

        tmp_grad = denom_vec_dr1;
        for(int n = 0; n < 3; ++n){
            gradients[grad_idx_i + n] -= tmp_grad[n];
            gradients[grad_idx_j + n] += tmp_grad[n];
        } 

    }
    return;
}
