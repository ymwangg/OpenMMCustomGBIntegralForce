#include <math.h>
#include <sstream>
#include <string.h>
#include <iostream>

#include "ReferenceNeighborList.h"
#include "ReferenceForce.h"

#include "openmm/OpenMMException.h"
#include "CharmmReferenceKernels.h"
#include "CharmmReferenceGBSW.h"
#include "SimTKOpenMMRealType.h"
#include "CharmmQuadrature.h"
#include "Units.h"

using std::vector;
using namespace OpenMM;
using namespace::std;

CharmmReferenceGBSW::CharmmReferenceGBSW(int numberOfAtoms){
    _atomicRadii.resize(numberOfAtoms);
    _scaledRadiusFactors.resize(numberOfAtoms);
    _dbornR_dr_vec.resize(numberOfAtoms,std::vector<OpenMM::Vec3>(numberOfAtoms));
    _dG_dbornR.resize(numberOfAtoms);
    _numberOfAtoms = numberOfAtoms;
    _electricConstant = -0.5*ONE_4PI_EPS0;
    _r0 = 0.07;
    _r1 = 2.0;
    int nRadialPoints = 24;
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
    _lookupTableBufferLength = 0.20; //0.20 nm
    _switchingDistance = 0.03; //0.03 nm
    _lookupTableGridLength = 0.15; //0.15 nm
    _periodic = false;
    _cutoff = false;
}

CharmmReferenceGBSW::~CharmmReferenceGBSW() {
}

void CharmmReferenceGBSW::setNeighborList(OpenMM::NeighborList& neighborList){
    _neighborList = &neighborList;
}

OpenMM::NeighborList* CharmmReferenceGBSW::getNeighborList(){
    return _neighborList;
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

void CharmmReferenceGBSW::setNoCutoff(){
    _cutoff = false;
}

void CharmmReferenceGBSW::setNoPeriodic(){
    _periodic = false;
}

void CharmmReferenceGBSW::setPeriodic(){
    _periodic = true;
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
    double prefactor = 2*electricConstant*(1.0/soluteDielectric - 1.0/solventDielectric); //-332*(1.0/1-1.0/80)
    prefactor = 2*electricConstant*(1.0 - 1.0/80);
    //cout<<solventDielectric<<endl;
    //cout<<soluteDielectric<<endl;
    //cout<<electricConstant<<endl;
    /*
    cout<<_numberOfAtoms<<endl;
    cout<<cutoffDistance<<endl;
    */
    //cout<<electricConstant<<endl;
    //cout<<_cutoffDistance<<endl;
    std::vector<std::vector<int>> neighborMatrix;
    neighborMatrix.clear();
    neighborMatrix.resize(numberOfAtoms, vector<int>());
    //use cutoff
    if(_cutoff){
        for(int atomI = 0; atomI < numberOfAtoms; atomI++){
            neighborMatrix[atomI].push_back(atomI);
        }
        for(auto &c : *_neighborList){
            int atomI, atomJ;
            if(c.first>c.second){
                atomI = c.second;
                atomJ = c.first;
            }else{
                atomI = c.first;
                atomJ = c.second;
            }
            neighborMatrix[atomI].push_back(atomJ);
        }
    }
    //no cutoff N^2
    else{
        for(int atomI = 0; atomI < numberOfAtoms; ++atomI){
            for(int atomJ = atomI; atomJ < numberOfAtoms; ++atomJ){
                neighborMatrix[atomI].push_back(atomJ);
            }
        }
    }
    /*
    for(int i = 0; i < numberOfAtoms; ++i){
        for(int j = 0; j < neighborMatrix[i].size(); ++j){
            cout<<neighborMatrix[i][j]<<" ";
        }
        cout<<endl;
    }*/
    computeLookupTable(atomCoordinates);

    /*
    std::vector<double> bornRadii = {
        0.241532,0.195838,0.158818,0.158351,0.303748,0.223532,0.297326,
        0.211035,0.212215,0.259904,0.189517,0.182483,0.248971,0.181756,
        0.254480,0.132924,0.131805,0.243577,0.173051,0.178189};
        */
    std::vector<double> bornRadii;
    bornRadii.resize(numberOfAtoms);
    //computeBornRadii(atomCoordinates, partialCharges, bornRadii);
    computeBornRadiiFast(atomCoordinates, partialCharges, bornRadii);
    _dG_dbornR.resize(numberOfAtoms, 0.0);

    for (int i = 0; i < numberOfAtoms; ++i){
        int atomI = i;
        for (int j = 0; j < neighborMatrix[i].size(); ++j){
            int atomJ = neighborMatrix[i][j];
            double deltaR[ReferenceForce::LastDeltaRIndex];
            if (getPeriodic())
                ReferenceForce::getDeltaRPeriodic(atomCoordinates[atomI], atomCoordinates[atomJ], getPeriodicBox(), deltaR);
            else
                ReferenceForce::getDeltaR(atomCoordinates[atomI], atomCoordinates[atomJ], deltaR);
            if (getUseCutoff() && deltaR[ReferenceForce::RIndex] > cutoffDistance)
                continue;
            double r_ij = deltaR[ReferenceForce::RIndex];
            //compute energy
            double q_i = partialCharges[atomI];
            double q_j = partialCharges[atomJ];
            double R_i = bornRadii[atomI];
            double R_j = bornRadii[atomJ];
            double D_ij = (r_ij*r_ij) / (4.0*R_i*R_j);
            double f_ij = sqrt(r_ij*r_ij + R_i*R_j*exp(-D_ij));
            double e_ij = prefactor * q_i * q_j / f_ij;
            if(atomI!=atomJ) 
                energy += e_ij;
            else 
                energy += 0.5*e_ij;

            //compute dG_dr force
            double dG_dr_part1 = q_i * q_j * (4.0-exp(-D_ij));
            double dG_dr = -0.25 * prefactor * dG_dr_part1 / (f_ij*f_ij*f_ij);
            OpenMM::Vec3 rij_vec(deltaR[ReferenceForce::XIndex],deltaR[ReferenceForce::YIndex],
                    deltaR[ReferenceForce::ZIndex]);
            OpenMM::Vec3 force1_ij = (-rij_vec) * (-dG_dr);
            inputForces[atomI] += force1_ij;
            inputForces[atomJ] -= force1_ij;

            //compute dG_dR
            if(atomI==atomJ){
                _dG_dbornR[atomI] += -0.5 * prefactor * q_i * q_j / (R_i*R_i);
            }else{
                double dG_dR_part1 = q_i*q_j*exp(-D_ij);
                double dG_dR_part2 = f_ij*f_ij*f_ij;
                double common_part = -0.5 * prefactor * (dG_dR_part1/dG_dR_part2);
                double dG_dRi = common_part * (R_j + (r_ij*r_ij)/(4.0*R_i));
                double dG_dRj = common_part * (R_i + (r_ij*r_ij)/(4.0*R_j));
                _dG_dbornR[atomI] += dG_dRi;
                _dG_dbornR[atomJ] += dG_dRj;
            }
        }
        //compute dG_dR*dR_dr force
        for (int atomJ = 0; atomJ < numberOfAtoms; atomJ++){
            double deltaR[ReferenceForce::LastDeltaRIndex];
            if (getPeriodic())
                ReferenceForce::getDeltaRPeriodic(atomCoordinates[atomI], atomCoordinates[atomJ], getPeriodicBox(), deltaR);
            else
                ReferenceForce::getDeltaR(atomCoordinates[atomI], atomCoordinates[atomJ], deltaR);
            if (getUseCutoff() && deltaR[ReferenceForce::RIndex] > cutoffDistance)
                continue;
            double dG_dRi = _dG_dbornR[atomI];
            OpenMM::Vec3 force2_j = _dbornR_dr_vec[atomI][atomJ] * (-dG_dRi);
            inputForces[atomJ] += force2_j;
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
    double alpha0 = -0.180; //-0.180
    double alpha1 = 1.817;
    double switchDistance = 0.03;
    //Ri^(-1) = alpha0*(1/eta_i  - 1/(4*PI) * Int_{eta}^{Inf} {V(r)/(r-ri)^4 * dxdydz} + 
    //alpha1*(1/(4*eta_i^4)  - 1/(4*PI) * Int_{eta}^{Inf} {V(r)/(r-ri)^7 * dxdydz})^(1/4)
    _dbornR_dr_vec.resize(_numberOfAtoms,std::vector<OpenMM::Vec3>(_numberOfAtoms));
    vector<double> VolumeQuad(_quad.size());
    for(int atomI=0; atomI<_numberOfAtoms; ++atomI){
        //compute Born Radius
        OpenMM::Vec3 atomICoordinate = atomCoordinates[atomI];
        double charge = partialCharges[atomI];
        double vdwR = _atomicRadii[atomI];
        double eta = _r0;
        double eta4 = eta*eta*eta*eta;
        double integral1 = 0.0;
        double integral2 = 0.0;
        for(int i=0; i<_quad.size(); ++i){
            OpenMM::Vec3 rQuad;
            rQuad[0] = atomICoordinate[0] + _quad[i][0];
            rQuad[1] = atomICoordinate[1] + _quad[i][1];
            rQuad[2] = atomICoordinate[2] + _quad[i][2];
            double radius = _quad[i][3];
            double weight = _quad[i][4];
            if(radius == 0)
                continue;
            VolumeQuad[i] = computeVolume(atomCoordinates, rQuad, switchDistance);
            double molecularVolume = 1.0 - VolumeQuad[i];
            integral1 += weight * molecularVolume/(radius*radius);
            double radius5 = radius*radius*radius*radius*radius;
            integral2 += weight * molecularVolume/radius5;
        }
        double inverseBornRi = alpha0*(1.0/eta - integral1) + 
            alpha1*pow((1.0/(4.0*eta4) - integral2), 1.0/4.0);
        bornRadii[atomI] = 1.0/inverseBornRi;
        //cout<<bornRadii[atomI]<<endl;
        //cout<<integral1<<" "<<integral2<<endl;
        //compute dRBorn_dr
        for(int i=0; i<_quad.size(); ++i){
            OpenMM::Vec3 rQuad;
            rQuad[0] = atomICoordinate[0] + _quad[i][0];
            rQuad[1] = atomICoordinate[1] + _quad[i][1];
            rQuad[2] = atomICoordinate[2] + _quad[i][2];
            double radius = _quad[i][3];
            double radius5 = radius*radius*radius*radius*radius;
            double weight = _quad[i][4];
            if(radius == 0)
                continue;
            double part1 = alpha0/(radius*radius);
            double part2 = (1.0/4.0)*alpha1*pow((1.0/(4.0*eta4)-integral2),-3.0/4.0)/radius5;
            double prefactor = (bornRadii[atomI]*bornRadii[atomI])*weight*(part1+part2);
            compute_dbornR_dr_vec(atomCoordinates, atomI, prefactor, rQuad, VolumeQuad[i], switchDistance);
        }
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
        double w = switchDistance;
        double w3 = w*w*w;
        double dr = r - RatomJ;
        double dr3 = dr*dr*dr;
        if(r <= RatomJ - switchDistance){
            return 0.0;
        }else if(r >= RatomJ + switchDistance){
            continue;
        }else{
            V *= 0.5 + 3.0/(4.0*w) * dr - 1.0/(4.0*w3) * dr3;
        }
    } 
    return V;
}

void CharmmReferenceGBSW::compute_dbornR_dr_vec(const std::vector<OpenMM::Vec3>& atomCoordinates, const int atomI, const double prefactor, const OpenMM::Vec3& quadCoordinate, const double volumeI, const double switchDistance){
    //_dbornR_dr_vec;
    if(volumeI==0) return;
    for(int atomK=0; atomK<_numberOfAtoms; ++atomK){
        if(atomI==atomK) continue;
        double deltaRik[ReferenceForce::LastDeltaRIndex];
        if (_periodic)
            ReferenceForce::getDeltaRPeriodic(quadCoordinate, atomCoordinates[atomK], _periodicBoxVectors, deltaRik);
        else
            ReferenceForce::getDeltaR(quadCoordinate, atomCoordinates[atomK], deltaRik);
        double rik = deltaRik[ReferenceForce::RIndex];
        double Rk = (_atomicRadii[atomK]+0.03)*0.9520;
        double uk;
        double duk_dri;
        double w = switchDistance;
        double w3 = w*w*w;
        double dr = rik - Rk;
        double dr2 = dr*dr;
        double dr3 = dr*dr*dr;
        if(rik>Rk-switchDistance && rik<Rk+switchDistance){
            uk = 0.5 + 3.0/(4.0*w) * dr - 1.0/(4.0*w3) * dr3;
            duk_dri = 3.0/(4.0*w) - 3.0/(4.0*w3) * dr2;
        }else{
            continue;
        }
        OpenMM::Vec3 rik_vec(deltaRik[ReferenceForce::XIndex],deltaRik[ReferenceForce::YIndex],
                deltaRik[ReferenceForce::ZIndex]);
        rik_vec = rik_vec / rik;
        OpenMM::Vec3 dRi_dri = rik_vec*(prefactor*(duk_dri*volumeI/uk));
        _dbornR_dr_vec[atomI][atomI] += dRi_dri;
        _dbornR_dr_vec[atomI][atomK] -= dRi_dri;
    }
    return;
}

void CharmmReferenceGBSW::computeLookupTable(const vector<Vec3>& atomCoordinates){
    //_lookupTable;
    //
    //_r1;
    OpenMM::Vec3 minCoordinate(atomCoordinates[0]);
    OpenMM::Vec3 maxCoordinate(atomCoordinates[0]);
    double maxR = 0.0;
    for(int atomI=0; atomI<_numberOfAtoms; ++atomI){
        for(int i=0; i<3; ++i){
            minCoordinate[i] = min(minCoordinate[i], atomCoordinates[atomI][i]);
            maxCoordinate[i] = max(maxCoordinate[i], atomCoordinates[atomI][i]);
        }
        maxR = max(maxR, _atomicRadii[atomI]);
    }
    double paddingLength = _lookupTableBufferLength + maxR + _switchingDistance +
        sqrt(3.0)/2.0*_lookupTableGridLength + 1e-6;
    int totalNumberOfGridPoints = 1;
    for(int i=0; i<3; ++i){
        minCoordinate[i] -= paddingLength;
        maxCoordinate[i] += paddingLength;
        _lookupTableNumberOfGridPoints[i] = static_cast<int>(
                ceil((maxCoordinate[i]-minCoordinate[i])/_lookupTableGridLength))+1;
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
    for(int atomI=0; atomI<_numberOfAtoms; ++atomI){
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
    /*
    for(int i=0; i<totalNumberOfGridPoints; ++i){
        if(_lookupTable[i].size()!=0){
            int x_id = i/(n_x*n_y);
            int y_id = i%(n_x*n_y)/n_y;
            int z_id = i%(n_x*n_y)%n_y;
            double x = _lookupTableMinCoordinate[0]+x_id*_lookupTableGridLength;
            double y = _lookupTableMinCoordinate[1]+y_id*_lookupTableGridLength;
            double z = _lookupTableMinCoordinate[2]+z_id*_lookupTableGridLength;
            cout<<x<<" "<<y<<" "<<z<<endl;
            for(auto &atomI : _lookupTable[i]){
                cout<<atomI<<" ";
            }
            cout<<endl;
        }
    }
    */
}

std::vector<int> CharmmReferenceGBSW::getLookupTableAtomList(OpenMM::Vec3 point){
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

void CharmmReferenceGBSW::computeBornRadiiFast(const std::vector<OpenMM::Vec3>& atomCoordinates,const std::vector<double>& partialCharges, std::vector<double>& bornRadii){
    //vector<double> _atomicRadii;
    //vector<OpenMM::Vec3> atomCoordinates
    //vector<double> partialCharges
    //generate quadrature points
    double alpha0 = -0.180; //-0.180
    double alpha1 = 1.817;
    double switchDistance = 0.03;
    _dbornR_dr_vec.resize(_numberOfAtoms,std::vector<OpenMM::Vec3>(_numberOfAtoms));
    vector<double> VolumeQuad(_quad.size());
    for(int atomI=0; atomI<_numberOfAtoms; ++atomI){
        //compute Born Radius
        OpenMM::Vec3 atomICoordinate = atomCoordinates[atomI];
        double charge = partialCharges[atomI];
        double vdwR = _atomicRadii[atomI];
        double eta = _r0;
        double eta4 = eta*eta*eta*eta;
        double integral1 = 0.0;
        double integral2 = 0.0;
        for(int i=0; i<_quad.size(); ++i){
            OpenMM::Vec3 rQuad;
            rQuad[0] = atomICoordinate[0] + _quad[i][0];
            rQuad[1] = atomICoordinate[1] + _quad[i][1];
            rQuad[2] = atomICoordinate[2] + _quad[i][2];
            vector<int> atomList = getLookupTableAtomList(rQuad);
            if(atomList.size()==0){
                VolumeQuad[i] = 1.0;
                continue;
            }
            else VolumeQuad[i] = computeVolumeFast(atomCoordinates, rQuad, switchDistance, atomList);
            double radius = _quad[i][3];
            double weight = _quad[i][4];
            double molecularVolume = 1.0 - VolumeQuad[i];
            integral1 += weight * molecularVolume/(radius*radius);
            double radius5 = radius*radius*radius*radius*radius;
            integral2 += weight * molecularVolume/radius5;
        }
        double inverseBornRi = alpha0*(1.0/eta - integral1) + 
            alpha1*pow((1.0/(4.0*eta4) - integral2), 1.0/4.0);
        bornRadii[atomI] = 1.0/inverseBornRi;
        //cout<<bornRadii[atomI]<<endl;
        //cout<<integral1<<" "<<integral2<<endl;
        //compute dRBorn_dr
        for(int i=0; i<_quad.size(); ++i){
            OpenMM::Vec3 rQuad;
            rQuad[0] = atomICoordinate[0] + _quad[i][0];
            rQuad[1] = atomICoordinate[1] + _quad[i][1];
            rQuad[2] = atomICoordinate[2] + _quad[i][2];
            vector<int> atomList = getLookupTableAtomList(rQuad);
            if(atomList.size() == 0) continue;
            double radius = _quad[i][3];
            double radius5 = radius*radius*radius*radius*radius;
            double weight = _quad[i][4];
            double part1 = alpha0/(radius*radius);
            double part2 = (1.0/4.0)*alpha1*pow((1.0/(4.0*eta4)-integral2),-3.0/4.0)/radius5;
            double prefactor = (bornRadii[atomI]*bornRadii[atomI])*weight*(part1+part2);
            compute_dbornR_dr_vec_Fast(atomCoordinates, atomI, prefactor, rQuad, VolumeQuad[i], switchDistance, atomList);
        }
    }
    return;
}

double CharmmReferenceGBSW::computeVolumeFast(const std::vector<OpenMM::Vec3>& atomCoordinates, const OpenMM::Vec3& quadCoordinate, const double switchDistance, const std::vector<int>& atomList){
    double V = 1.0;
    for(const int& atomJ : atomList){
        double deltaR[ReferenceForce::LastDeltaRIndex];
        if (_periodic)
            ReferenceForce::getDeltaRPeriodic(quadCoordinate, atomCoordinates[atomJ], _periodicBoxVectors, deltaR);
        else
            ReferenceForce::getDeltaR(quadCoordinate, atomCoordinates[atomJ], deltaR);
        double r = deltaR[ReferenceForce::RIndex];
        double RatomJ = (_atomicRadii[atomJ]+0.03)*0.9520;
        double w = switchDistance;
        double w3 = w*w*w;
        double dr = r - RatomJ;
        double dr3 = dr*dr*dr;
        if(r <= RatomJ - switchDistance){
            return 0.0;
        }else if(r >= RatomJ + switchDistance){
            continue;
        }else{
            V *= 0.5 + 3.0/(4.0*w) * dr - 1.0/(4.0*w3) * dr3;
        }
    } 
    return V;
}

void CharmmReferenceGBSW::compute_dbornR_dr_vec_Fast(const std::vector<OpenMM::Vec3>& atomCoordinates, const int atomI, const double prefactor, const OpenMM::Vec3& quadCoordinate, const double volumeI, const double switchDistance, const std::vector<int>& atomList){
    //_dbornR_dr_vec;
    if(volumeI==0) return;
    for(const int& atomK : atomList){
        if(atomI==atomK) continue;
        double deltaRik[ReferenceForce::LastDeltaRIndex];
        if (_periodic)
            ReferenceForce::getDeltaRPeriodic(quadCoordinate, atomCoordinates[atomK], _periodicBoxVectors, deltaRik);
        else
            ReferenceForce::getDeltaR(quadCoordinate, atomCoordinates[atomK], deltaRik);
        double rik = deltaRik[ReferenceForce::RIndex];
        double Rk = (_atomicRadii[atomK]+0.03)*0.9520;
        double uk;
        double duk_dri;
        double w = switchDistance;
        double w3 = w*w*w;
        double dr = rik - Rk;
        double dr2 = dr*dr;
        double dr3 = dr*dr*dr;
        if(rik>Rk-switchDistance && rik<Rk+switchDistance){
            uk = 0.5 + 3.0/(4.0*w) * dr - 1.0/(4.0*w3) * dr3;
            duk_dri = 3.0/(4.0*w) - 3.0/(4.0*w3) * dr2;
        }else{
            continue;
        }
        OpenMM::Vec3 rik_vec(deltaRik[ReferenceForce::XIndex],deltaRik[ReferenceForce::YIndex],
                deltaRik[ReferenceForce::ZIndex]);
        rik_vec = rik_vec / rik;
        OpenMM::Vec3 dRi_dri = rik_vec*(prefactor*(duk_dri*volumeI/uk));
        _dbornR_dr_vec[atomI][atomI] += dRi_dri;
        _dbornR_dr_vec[atomI][atomK] -= dRi_dri;
    }
    return;
}
