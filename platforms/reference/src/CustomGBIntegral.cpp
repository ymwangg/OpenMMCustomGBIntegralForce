#include "CustomGBIntegral.h"
#include "openmm/OpenMMException.h"
#include <algorithm> 
#include <iostream>

using namespace::OpenMM;
using namespace::std;

CustomGBIntegral::CustomGBIntegral(): _periodic(false) {
    _lookupTableBufferLength = 0.20; //0.20 nm + _sw
    _lookupTableGridLength = 0.15; //0.15 nm
}

CustomGBIntegral::~CustomGBIntegral(){
}

void CustomGBIntegral::setPeriodic(OpenMM::Vec3* vectors){
    _periodic           = true;
    _periodicBoxVectors[0] = vectors[0];
    _periodicBoxVectors[1] = vectors[1];
    _periodicBoxVectors[2] = vectors[2];
}

void CustomGBIntegral::setLookupTableGridLength(double length){
    _lookupTableGridLength = length;
}

void CustomGBIntegral::setLookupTableBufferLength(double length){
    _lookupTableBufferLength = length;
}

void CustomGBIntegral::computeLookupTable(const std::vector<OpenMM::Vec3>& atomCoordinates){
    //_lookupTable;
    //  
    //_r1;
    int numParticles = atomCoordinates.size();
    for(int i=0; i<3; ++i){
        _minCoordinate[i] = atomCoordinates[0][i];
        _maxCoordinate[i] = atomCoordinates[0][i];
    }
    double maxR = 0.3; //3 Angstrom is the max VDW radius
    for(int atomI=0; atomI<numParticles; ++atomI){
        for(int i=0; i<3; ++i){
            _minCoordinate[i] = min(_minCoordinate[i], atomCoordinates[atomI][i]);
            _maxCoordinate[i] = max(_maxCoordinate[i], atomCoordinates[atomI][i]);
        }
    }
    double paddingLength = maxR + sqrt(3.0)/2.0*_lookupTableGridLength + 1e-6;
    int totalNumberOfGridPoints = 1;
    for(int i=0; i<3; ++i){
        _minCoordinate[i] -= paddingLength;
        _maxCoordinate[i] += paddingLength;
        double length = _maxCoordinate[i]-_minCoordinate[i];
        _lookupTableNumberOfGridPoints[i] = static_cast<int>(
                ceil(length/_lookupTableGridLength))+1;
        if(length > 1000)
            throw OpenMM::OpenMMException("CharmmGBMVForce: CustomGBIntegral lookup table dimension is too large, check atom positions!");
        _lookupTableMinCoordinate[i] = _minCoordinate[i];
        _lookupTableMaxCoordinate[i] = _minCoordinate[i]+(_lookupTableNumberOfGridPoints[i]-1)*_lookupTableGridLength;
        totalNumberOfGridPoints *= _lookupTableNumberOfGridPoints[i];
        //cout<<minCoordinate[i]<<" "<<maxCoordinate[i]<<" "<<_lookupTableNumberOfGridPoints[i]<<endl;
    }
    int n_x = _lookupTableNumberOfGridPoints[0];
    int n_y = _lookupTableNumberOfGridPoints[1];
    int n_z = _lookupTableNumberOfGridPoints[2];
    _lookupTable.clear();
    _lookupTable.resize(totalNumberOfGridPoints,vector<int>());
    for(int atomI=0; atomI<numParticles; ++atomI){
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


void CustomGBIntegral::getLookupTableAtomList(OpenMM::Vec3 point, std::vector<int>& atomList){
    atomList.clear();
    int nx = _lookupTableNumberOfGridPoints[0];
    int ny = _lookupTableNumberOfGridPoints[1];
    int nz = _lookupTableNumberOfGridPoints[2];
    if(_periodic){
        OpenMM::Vec3 points[3];
        points[0] = point;
        points[1] = point;
        points[2] = point;
        bool points_included[3];
        points_included[0] = true; points_included[1] = false; points_included[2] = false;
        for(int i=0; i<3; ++i){
            if(validateTransform(points[1] + _periodicBoxVectors[i])){
                points_included[1] = true;
                points[1] += _periodicBoxVectors[i];
            }
            if(validateTransform(points[1] - _periodicBoxVectors[i])){
                points_included[2] = true;
                points[2] -= _periodicBoxVectors[i];
            }
        }
        for(int n=0; n<3; ++n){
            int idx[3];
            bool included = true;
            for(int i=0; i<3; ++i){
                //if point is still outside of the lookupTable grid
                if((points[n][i] < _lookupTableMinCoordinate[i]) ||
                        (points[n][i] > _lookupTableMaxCoordinate[i])){
                    included = false;
                    continue;
                }
                idx[i] = static_cast<int>(floor(
                            (points[n][i]-_lookupTableMinCoordinate[i]) / _lookupTableGridLength));
            }
            if(included && points_included[n]){
                int lookupTableIdx = idx[0]*(ny*nz) + idx[1]*nz + idx[2];
                for(auto idx : _lookupTable[lookupTableIdx]){
                    atomList.push_back(idx);
                }
            }
        }
        return;
    }else{
        int idx[3];
        for(int i=0; i<3; ++i){
            //if point is outside of the lookupTable grid
            if((point[i] < _lookupTableMinCoordinate[i]) ||
                    (point[i] > _lookupTableMaxCoordinate[i])){
                return;
            }
            idx[i] = static_cast<int>(floor(
                        (point[i]-_lookupTableMinCoordinate[i]) / _lookupTableGridLength));
        }
        int lookupTableIdx = idx[0]*(ny*nz) + idx[1]*nz + idx[2];
        atomList = _lookupTable[lookupTableIdx];
        return;
    }
}

bool CustomGBIntegral::validateTransform(OpenMM::Vec3 newVec){
    for(int i=0; i<3; ++i){
        if( (newVec[i] > _maxCoordinate[i]) || (newVec[i] < _minCoordinate[i]) )
            return false;
    }
    return true;
}
