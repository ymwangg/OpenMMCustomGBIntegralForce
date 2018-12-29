#include <math.h>
#include <sstream>
#include <string.h>
#include <iostream>

#include "openmm/OpenMMException.h"
#include "CharmmReferenceKernels.h"
#include "CharmmReferenceGBSW.h"
#include "SimTKOpenMMRealType.h"
#include "ReferenceForce.h"
#include "CharmmQuadrature.h"
#include "Units.h"

using std::vector;
using namespace OpenMM;
using namespace::std;

CharmmReferenceGBSW::CharmmReferenceGBSW(int numberOfAtoms){
    _atomicRadii.resize(numberOfAtoms);
    _scaledRadiusFactors.resize(numberOfAtoms);
    _numberOfAtoms = numberOfAtoms;
    _electricConstant = -0.5*ONE_4PI_EPS0;
}

CharmmReferenceGBSW::~CharmmReferenceGBSW() {
}

void CharmmReferenceGBSW::setAtomicRadii(const vector<double>& atomicRadii) {

    if (atomicRadii.size() == _atomicRadii.size()) {
        for (unsigned int ii = 0; ii < atomicRadii.size(); ii++) {
            _atomicRadii[ii] = atomicRadii[ii];
        }
    } else {
        std::stringstream msg;
        msg << "CharmmReferenceGBSW: input size for atomic radii does not agree w/ current size: input=";
        msg << atomicRadii.size();
        msg << " current size=" << _atomicRadii.size();
        throw OpenMM::OpenMMException(msg.str());
    }

}

const vector<double>& CharmmReferenceGBSW::getAtomicRadii() const {
    return _atomicRadii;
}

void CharmmReferenceGBSW::setScaledRadiusFactors(const vector<double>& scaledRadiusFactors) {

    if (scaledRadiusFactors.size() == _scaledRadiusFactors.size()) {
        for (unsigned int ii = 0; ii < scaledRadiusFactors.size(); ii++) {
            _scaledRadiusFactors[ii] = scaledRadiusFactors[ii];
        }
    } else {
        std::stringstream msg;
        msg << "CharmmReferenceGBSW: input size for scaled radius factors does not agree w/ current size: input=";
        msg << scaledRadiusFactors.size();
        msg << " current size=" << _scaledRadiusFactors.size();
        throw OpenMM::OpenMMException(msg.str());
    }

}

const vector<double>& CharmmReferenceGBSW::getScaledRadiusFactors() const {
    return _scaledRadiusFactors;
}

void CharmmReferenceGBSW::setSolventDielectric(double solventDielectric) {
    _solventDielectric = solventDielectric;
}

double CharmmReferenceGBSW::getSolventDielectric() const {
    return _solventDielectric;
}

void CharmmReferenceGBSW::setSoluteDielectric(double soluteDielectric) {
    _soluteDielectric = soluteDielectric;
}

double CharmmReferenceGBSW::getSoluteDielectric() const {
    return _soluteDielectric;
}

void CharmmReferenceGBSW::setUseCutoff(double distance) {

     _cutoff         = true;
     _cutoffDistance = distance;
}

bool CharmmReferenceGBSW::getUseCutoff() const {
     return _cutoff;
}

double CharmmReferenceGBSW::getCutoffDistance() const {
     return _cutoffDistance;
}

void CharmmReferenceGBSW::setPeriodic(OpenMM::Vec3* vectors) {

    assert(_cutoff);

    assert(vectors[0][0] >= 2.0*_cutoffDistance);
    assert(vectors[1][1] >= 2.0*_cutoffDistance);
    assert(vectors[2][2] >= 2.0*_cutoffDistance);

    _periodic           = true;
    _periodicBoxVectors[0] = vectors[0];
    _periodicBoxVectors[1] = vectors[1];
    _periodicBoxVectors[2] = vectors[2];
}

bool CharmmReferenceGBSW::getPeriodic() {
     return _periodic;
}

const OpenMM::Vec3* CharmmReferenceGBSW::getPeriodicBox() {
     return _periodicBoxVectors;
}

double CharmmReferenceGBSW::computeEnergyForces(const vector<Vec3>& atomCoordinates,
        const vector<double>& partialCharges, vector<Vec3>& inputForces) {
    double energy = 0.0;
    const int numberOfAtoms = _numberOfAtoms;
    const double solventDielectric = _solventDielectric;
    const double soluteDielectric = _soluteDielectric;
    const double cutoffDistance = _cutoffDistance;
    const double electricConstant = _electricConstant;
    double prefactor = 2*electricConstant*(1.0/soluteDielectric - 1.0/solventDielectric); //332*(1.0/1-1.0/80)
    prefactor = 2*electricConstant*(1.0 - 1.0/80);
    cout<<solventDielectric<<endl;
    cout<<soluteDielectric<<endl;
    //cout<<electricConstant<<endl;
    /*
    cout<<_numberOfAtoms<<endl;
    cout<<cutoffDistance<<endl;
    */
    //cout<<electricConstant<<endl;
    //cout<<_cutoffDistance<<endl;
    //bornRadii.resize(_numberOfAtoms);
    std::vector<double> bornRadii = {
             2.0671048866705033  ,  
             1.6708247223794475  ,   
             1.2111449624361177  ,   
             1.2090743225808722  ,   
             2.5582359782382080  ,   
             1.7701566227630932  ,   
             2.4436070788266764  ,   
             1.6112295349242343  ,   
             1.7682060825888346  ,   
             2.3825455446487065  ,   
             1.4885826414553700  ,   
             1.7293822223394433  ,   
             2.4343525600508422  ,   
             1.8752450803679326  ,   
             2.2825091233422956  ,   
             1.1201204725992946  ,   
             1.0305836162730180  ,   
             2.2498345141236533  ,   
             1.1752458648052326  ,   
             1.1789050022567080 };
    computeBornRadii(atomCoordinates, partialCharges, bornRadii);
    for (int atomI = 0; atomI < numberOfAtoms; atomI++){
        for (int atomJ = atomI; atomJ < numberOfAtoms; atomJ++){
            double deltaR[ReferenceForce::LastDeltaRIndex];
            if (getPeriodic())
                ReferenceForce::getDeltaRPeriodic(atomCoordinates[atomI], atomCoordinates[atomJ], getPeriodicBox(), deltaR);
            else
                ReferenceForce::getDeltaR(atomCoordinates[atomI], atomCoordinates[atomJ], deltaR);
            double r = deltaR[ReferenceForce::RIndex];
            /*
            if (getUseCutoff() && deltaR[ReferenceForce::RIndex] > cutoffDistance)
                continue;
            */
            double Qi = partialCharges[atomI];
            double Qj = partialCharges[atomJ];
            double Ri = bornRadii[atomI];
            double Rj = bornRadii[atomJ];
            double fij = sqrt(r*r + Ri*Rj*exp(-r*r/(4*Ri*Rj)));
            double e = prefactor*Qi*Qj/fij;
            if(atomI!=atomJ){
                energy += e;
            }
            else{
                energy += 0.5*e;
                //cout<<atomI<<" "<<0.5*e*OpenMM::KcalPerKJ<<endl;
            }
        }
    }

    return energy;
}

void CharmmReferenceGBSW::computeBornRadii(const std::vector<OpenMM::Vec3>& atomCoordinates,const std::vector<double>& partialCharges, std::vector<double>& bornRadii){
    //vector<double> _atomicRadii;
    //vector<OpenMM::Vec3> atomCoordinates
    //vector<double> partialCharges
    //out vector<double> bornRadii
    /*
     * Ri^(-1) = alpha0*(-2/(tau*(qi^2)) * G0) + alpha1*(-2/(tau*(qi^2)) * G1)
     *
     * Ri^(-1) = alpha0*(1/eta_i  - 1/(4*PI) * Int_{eta}^{Inf} {V(r)/(r-ri)^4 * dxdydz} + 
     * alpha1*(1/(4*eta_i^4)  - 1/(4*PI) * Int_{eta}^{Inf} {V(r)/(r-ri)^7 * dxdydz})^(1/4)
     *
     * G0 = -1/2 * tau * qi^2 * (1/eta_i  - 1/(4*PI) * Int_{eta}^{Inf} {V(r)/(r-ri)^4 * dxdydz}
     *
     * G1 = -1/2 * tau * qi^2 * (1/(4*eta_i^4)  - 1/(4*PI) * Int_{eta}^{Inf} {V(r)/(r-ri)^7 * dxdydz})^(1/4)
     *
     * Int_{eta}^{Inf} {V(r)/(r-ri)^4} = Sum{ 4*PI*(r-ri)^2 * W_quad * V(r)/(r-ri)^4 }
     * Int_{eta}^{Inf} {V(r)/(r-ri)^7} = Sum{ 4*PI*(r-ri)^2 * W_quad * V(r)/(r-ri)^7 }
     *
     * V(ri + r_quad) = Multiply{ v(ri + r_quad - r_j) }
     * v(ri + r_quad - r_j) = 0 , |ri + r_quad - r_j| < Rj - w
     * v(ri + r_quad - r_j) = 1/2 + 3/(4*w)(ri + r_quad - r_j - Rj) - 1/(4w^3)*(ri + r_quad - r_j - Rj)^3 , Rj - w < |ri + r_quad - r_j| < Rj + w
     * v(ri + r_quad - r_j) = 1 , |ri + r_quad - r_j| > Rj - w
     */

    //generate quadrature points
    double r0 = 0.07;
    double r1 = 2.0;
    double alpha0 = -0.180; //-0.180
    double alpha1 = 1.817;
    double switchDistance = 0.03;
    int nRadialPoints = 24;
    int ruleLebedev = 4;
    vector<vector<double> > radialQuad = CharmmQuadrature::GaussLegendre(r0, r1, nRadialPoints);
    vector<vector<double> > sphericalQuad = CharmmQuadrature::Lebedev(ruleLebedev);
    /*
       for(auto &q : radialQuad){
       cout<<q[0]<<" "<<q[1]<<endl;
       }
       for(auto &q : sphericalQuad){
       cout<<q[0]<<" "<<q[1]<<" "<<q[2]<<" "<<q[3]<<endl;
       }*/
    vector<vector<double> > quad(radialQuad.size()*sphericalQuad.size(), vector<double>(5,0.0));
    for(int i=0; i<radialQuad.size(); ++i){
        for(int j=0; j<sphericalQuad.size(); ++j){
            double r = radialQuad[i][0];
            double w_r = radialQuad[i][1];
            double w_s = sphericalQuad[j][3];
            int idx = i*sphericalQuad.size()+j;
            for(int k=0; k<3; ++k){
                quad[idx][k] = r*sphericalQuad[j][k];
            }
            quad[idx][3] = r;
            quad[idx][4] = w_r*w_s;
        }
    }
    //Ri^(-1) = alpha0*(1/eta_i  - 1/(4*PI) * Int_{eta}^{Inf} {V(r)/(r-ri)^4 * dxdydz} + 
    //alpha1*(1/(4*eta_i^4)  - 1/(4*PI) * Int_{eta}^{Inf} {V(r)/(r-ri)^7 * dxdydz})^(1/4)
    for(int atomI=0; atomI<_numberOfAtoms; ++atomI){
        OpenMM::Vec3 atomICoordinate = atomCoordinates[atomI];
        double charge = partialCharges[atomI];
        double vdwR = _atomicRadii[atomI];
        double eta = r0;
        double integral1 = 0.0;
        double integral2 = 0.0;
        for(int i=0; i<quad.size(); ++i){
            OpenMM::Vec3 rQuad;
            rQuad[0] = atomICoordinate[0]+quad[i][0];
            rQuad[1] = atomICoordinate[1]+quad[i][1];
            rQuad[2] = atomICoordinate[2]+quad[i][2];
            double radius = quad[i][3];
            double weight = quad[i][4];
            if(abs(radius-0.0)<1e-6)
                continue;
            double molecularVolume = 1.0 - computeVolume(atomCoordinates, rQuad, switchDistance);
            integral1 += weight * molecularVolume/(radius*radius);
            integral2 += weight * molecularVolume/pow(radius,5);
        }
        double inverseBornRi = alpha0*(1.0/eta - integral1) + 
            alpha1*pow((1.0/abs(4.0*pow(eta,4)) - integral2), 1.0/4.0);
        bornRadii[atomI] = 1.0/inverseBornRi;
        //cout<<bornRadii[atomI]<<endl;
        //cout<<integral1<<" "<<integral2<<endl;
    }
    return;
}


double CharmmReferenceGBSW::computeVolume(const std::vector<OpenMM::Vec3>& atomCoordinates, const OpenMM::Vec3& quadCoordinate, const double switchDistance){
    double V = 1.0;
    for(int atomJ=0; atomJ<_numberOfAtoms; ++atomJ){
        double deltaR[ReferenceForce::LastDeltaRIndex];
        if (_periodic)
            ReferenceForce::getDeltaRPeriodic(quadCoordinate, atomCoordinates[atomJ], _periodicBoxVectors, deltaR);
        else
            ReferenceForce::getDeltaR(quadCoordinate, atomCoordinates[atomJ], deltaR);
        double r = deltaR[ReferenceForce::RIndex];
        double RatomJ = (_atomicRadii[atomJ]+0.03)*0.9520;
        if(r <= RatomJ - switchDistance){
            return 0.0;
        }else if(r >= RatomJ + switchDistance){
            continue;
        }else{
            V *= 0.5 + 3.0/(4.0*switchDistance) * (r-RatomJ) - 1.0/(4.0*pow(switchDistance,3.0))*pow(r-RatomJ,3.0);
        }
    } 
    return V;
}
