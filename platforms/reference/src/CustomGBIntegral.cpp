#include "CustomGBIntegral.h"
#include "openmm/OpenMMException.h"
#include <algorithm> 
#include <iostream>
#include <cstdio>
#include <chrono>

using namespace::OpenMM;
using namespace::std;

CustomGBIntegral::CustomGBIntegral(): _periodic(false) {
    _lookupTableBufferLength = 0.03; //0.20 nm + _sw
    _lookupTableGridLength = 0.15; //0.15 nm
    _lookupTableSize = 32;
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
    auto start = std::chrono::system_clock::now();
    //_lookupTable;
    //  
    //_r1;
    int numParticles = atomCoordinates.size();
    //if system is periodic
    if(_periodic){
        float d_x = _periodicBoxVectors[0][0];
        float d_y = _periodicBoxVectors[1][1];
        float d_z = _periodicBoxVectors[2][2];
        if(_periodicBoxVectors[0][1]!=0 || _periodicBoxVectors[0][2]!=0 ||
                _periodicBoxVectors[1][0]!=0 || _periodicBoxVectors[1][2]!=0 ||
                _periodicBoxVectors[2][0]!=0 || _periodicBoxVectors[2][1]!=0)
            throw OpenMM::OpenMMException("CharmmGBMVForce: Only Rectangular Periodic Box is Supported!");
        int n_x = ceil(d_x/_lookupTableGridLength) + 1;
        int n_y = ceil(d_y/_lookupTableGridLength) + 1;
        int n_z = ceil(d_z/_lookupTableGridLength) + 1;
        _lookupTableNumberOfGridPoints[0] = n_x;
        _lookupTableNumberOfGridPoints[1] = n_y;
        _lookupTableNumberOfGridPoints[2] = n_z;
        int totalNumberOfGridPoints = n_x*n_y*n_z;
        if(_lookupTable.empty()) _lookupTable.resize(n_x*n_y*n_z,vector<int>(_lookupTableSize));
        if(_lookupTableNumAtoms.empty()) _lookupTableNumAtoms.resize(n_x*n_y*n_z);

        /*
        for(int i=0; i<totalNumberOfGridPoints; i++) 
            _lookupTableNumAtoms[i] = 0;
            */
        std::fill(_lookupTableNumAtoms.begin(),_lookupTableNumAtoms.end(),0.0);

        OpenMM::Vec3 coor;
        float beginCoor[3]; 
        float endCoor[3]; 
        float step = _lookupTableGridLength;
        for(int atomI=0; atomI<numParticles; ++atomI){
            float paddingLength = _atomicRadii[atomI] + 
                sqrt(3.0)/2.0*_lookupTableGridLength + _lookupTableBufferLength;
            coor = atomCoordinates[atomI];
            for(int i=0; i<3; ++i){
                beginCoor[i] = (coor[i] - paddingLength);
                endCoor[i] = (coor[i] + paddingLength);
            }
            //printf("(%f,%f,%f) to (%f,%f,%f) for (%f,%f,%f)\n",beginCoor[0],beginCoor[1],beginCoor[2],
            //        endCoor[0],endCoor[1],endCoor[2],coor[0],coor[1],coor[2]);
            for(float x = beginCoor[0]; x < endCoor[0]; x += step){
                float x0 = x - floor(x/d_x)*d_x;
                int idx_x = x0/step;
                for(float y = beginCoor[1]; y < endCoor[1]; y += step){
                    float y0 = y - floor(y/d_y)*d_y;
                    int idx_y = y0/step;
                    for(float z = beginCoor[2]; z < endCoor[2]; z += step){
                        float z0 = z - floor(z/d_z)*d_z;
                        int idx_z = z0/step;
                        if( (x-coor[0])*(x-coor[0]) + (y-coor[1])*(y-coor[1]) 
                                + (z-coor[2])*(z-coor[2]) < paddingLength*paddingLength){
                            int idx = idx_x*n_y*n_z + idx_y*n_z + idx_z;
                            if(_lookupTableNumAtoms[idx] < _lookupTableSize){
                                _lookupTable[idx][_lookupTableNumAtoms[idx]] = atomI;
                                _lookupTableNumAtoms[idx] ++;
                            }
                        }
                    }
                }
            }
        }
    }
    //not periodic
    else{
        for(int i=0; i<3; ++i){
            _minCoordinate[i] = atomCoordinates[0][i];
            _maxCoordinate[i] = atomCoordinates[0][i];
        }
        for(int atomI=1; atomI<numParticles; ++atomI){
            for(int i=0; i<3; ++i){
                _minCoordinate[i] = min(_minCoordinate[i], atomCoordinates[atomI][i]);
                _maxCoordinate[i] = max(_maxCoordinate[i], atomCoordinates[atomI][i]);
            }
        }
        int totalNumberOfGridPoints = 1;
        float maxR = 0.03;
        float paddingLength = maxR + sqrt(3.0)/2.0*_lookupTableGridLength + _lookupTableBufferLength;
        for(int i=0; i<3; ++i){
            _minCoordinate[i] -= paddingLength;
            _maxCoordinate[i] += paddingLength;
            float length = _maxCoordinate[i]-_minCoordinate[i];
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
            float paddingLength = _atomicRadii[atomI] + 
                sqrt(3.0)/2.0*_lookupTableGridLength + _lookupTableBufferLength;
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
    } //end not periodic
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    cout<<"building lookupTable elapsed time: " << elapsed_seconds.count()<<endl;
}


void CustomGBIntegral::getLookupTableAtomList(OpenMM::Vec3 point, std::vector<int>& atomList){
    atomList.clear();
    //lookuptable size
    int nx = _lookupTableNumberOfGridPoints[0];
    int ny = _lookupTableNumberOfGridPoints[1];
    int nz = _lookupTableNumberOfGridPoints[2];
    if(_periodic){
        //box vector
        double d_x = _periodicBoxVectors[0][0];
        double d_y = _periodicBoxVectors[1][1];
        double d_z = _periodicBoxVectors[2][2];
        if(_periodicBoxVectors[0][1]!=0 || _periodicBoxVectors[0][2]!=0 ||
                _periodicBoxVectors[1][0]!=0 || _periodicBoxVectors[1][2]!=0 ||
                _periodicBoxVectors[2][0]!=0 || _periodicBoxVectors[2][1]!=0)
            throw OpenMM::OpenMMException("CharmmGBMVForce: Only Rectangular Periodic Box is Supported!");
        //transform the point into the original box
        double x0 = point[0] - floor(point[0]/d_x)*d_x;
        double y0 = point[1] - floor(point[1]/d_y)*d_y;
        double z0 = point[2] - floor(point[2]/d_z)*d_z;
        //calculate lookupTable index
        int idx_x = static_cast<int>(floor(x0/_lookupTableGridLength)); 
        int idx_y = static_cast<int>(floor(y0/_lookupTableGridLength)); 
        int idx_z = static_cast<int>(floor(z0/_lookupTableGridLength)); 
        int idx = idx_x*(ny*nz) + idx_y*nz + idx_z;
        atomList = _lookupTable[idx];
        //printf("(%f,%f,%f)->(%f,%f,%f)->(%d,%d,%d)->%d->%d\n",point[0],point[1],point[2],
        //        x0,y0,z0,idx_x,idx_y,idx_z,idx,atomList.size());
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
    }
    return;
}

bool CustomGBIntegral::validateTransform(OpenMM::Vec3 newVec){
    for(int i=0; i<3; ++i){
        if( (newVec[i] > _maxCoordinate[i]) || (newVec[i] < _minCoordinate[i]) )
            return false;
    }
    return true;
}
