#include "CustomGBIntegral.h"
#include "openmm/OpenMMException.h"
#include <algorithm> 
#include <iostream>
#include <cstdio>
#include <chrono>
#include <cmath>

using namespace::OpenMM;
using namespace::std;

CustomGBIntegral::CustomGBIntegral(): _periodic(false) {
    _lookupTableBufferLength = 0.03; //0.20 nm + _sw
    _lookupTableGridLength = 0.15; //0.15 nm
    _lookupTableSize = 64;
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
    //_r1;
    int numParticles = atomCoordinates.size();
    //if system is periodic
    if(_periodic){
        //only support rectangular box
        if(_periodicBoxVectors[0][1]!=0 || _periodicBoxVectors[0][2]!=0 ||
                _periodicBoxVectors[1][0]!=0 || _periodicBoxVectors[1][2]!=0 ||
                _periodicBoxVectors[2][0]!=0 || _periodicBoxVectors[2][1]!=0)
            throw OpenMM::OpenMMException("CharmmGBMVForce: Only Rectangular Periodic Box is Supported!");

        //compute center of geometry
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
        double d_x = _periodicBoxVectors[0][0];
        double d_y = _periodicBoxVectors[1][1];
        double d_z = _periodicBoxVectors[2][2];

        //randomly move molecule in space
        //center_of_geom[0] += (((double) rand() / (RAND_MAX)) + 1)*d_x;
        //center_of_geom[1] += (((double) rand() / (RAND_MAX)) + 1)*d_y;
        //center_of_geom[2] += (((double) rand() / (RAND_MAX)) + 1)*d_z;
        //printf("center=(%f,%f,%f)\n",center_of_geom[0],center_of_geom[1],center_of_geom[2]);

        //compute lookuptable dimensions
        _lookupTableMinCoordinate[0] = center_of_geom[0] - d_x/2.0;
        _lookupTableMinCoordinate[1] = center_of_geom[1] - d_y/2.0;
        _lookupTableMinCoordinate[2] = center_of_geom[2] - d_z/2.0;

        _lookupTableMaxCoordinate[0] = center_of_geom[0] + d_x/2.0;
        _lookupTableMaxCoordinate[1] = center_of_geom[1] + d_y/2.0;
        _lookupTableMaxCoordinate[2] = center_of_geom[2] + d_z/2.0;
        //printf("lookupTable min=(%f,%f,%f) max=(%f,%f,%f)\n",_lookupTableMinCoordinate[0],
        //        _lookupTableMinCoordinate[1],_lookupTableMinCoordinate[2],
        //        _lookupTableMaxCoordinate[0],_lookupTableMaxCoordinate[1],
        //        _lookupTableMaxCoordinate[2]);

        int n_x = ceil(d_x/_lookupTableGridLength)+1;
        int n_y = ceil(d_y/_lookupTableGridLength)+1;
        int n_z = ceil(d_z/_lookupTableGridLength)+1;

        _lookupTableNumberOfGridPoints[0] = n_x;
        _lookupTableNumberOfGridPoints[1] = n_y;
        _lookupTableNumberOfGridPoints[2] = n_z;
        int totalNumGridPoints = n_x*n_y*n_z;

        if(_lookupTable.empty()) _lookupTable.resize(totalNumGridPoints,vector<int>(_lookupTableSize));
        if(_lookupTableNumAtoms.empty()) _lookupTableNumAtoms.resize(totalNumGridPoints);
        std::fill(_lookupTableNumAtoms.begin(),_lookupTableNumAtoms.end(),0.0);

        //printf("(%f,%f,%f)->(%f,%f,%f)->(%d,%d,%d)\n",d_x/_lookupTableGridLength,
        //        d_y/_lookupTableGridLength,d_z/_lookupTableGridLength,
        //        floor(d_x/_lookupTableGridLength),floor(d_y/_lookupTableGridLength),floor(d_z/_lookupTableGridLength),
        //        n_x,n_y,n_z);

        //compute step size of each dimension
        double step[3];
        step[0] = d_x / n_x;
        step[1] = d_y / n_y;
        step[2] = d_z / n_z;
        _lookupTableGridStep[0] = step[0];
        _lookupTableGridStep[1] = step[1];
        _lookupTableGridStep[2] = step[2];
        double lookupTableGridLength = max(max(step[0],step[1]),step[2]);

        //printf("(%f,%f,%f)->(%d,%d,%d)->(%f,%f,%f)\n",d_x,d_y,d_z,n_x,n_y,n_z,step[0],step[1],step[2]);
        //printf("%f->%f\n",_lookupTableGridLength,lookupTableGridLength);
        for(int atomI=0; atomI<numParticles; ++atomI){
            double paddingLength = _atomicRadii[atomI] + 
                sqrt(3.0)/2.0*lookupTableGridLength + _lookupTableBufferLength;
            OpenMM::Vec3 coor = atomCoordinates[atomI];
            double beginCoor[3]; 
            double endCoor[3]; 
            for(int i=0; i<3; ++i){
                beginCoor[i] = (coor[i] - paddingLength - lookupTableGridLength);
                endCoor[i] = (coor[i] + paddingLength + lookupTableGridLength);
            }
            //printf("(%f,%f,%f) to (%f,%f,%f) for (%f,%f,%f)\n",beginCoor[0],beginCoor[1],beginCoor[2],
            //        endCoor[0],endCoor[1],endCoor[2],coor[0],coor[1],coor[2]);
            for(double x = beginCoor[0]; x < endCoor[0]; x += step[0]){
                for(double y = beginCoor[1]; y < endCoor[1]; y += step[1]){
                    for(double z = beginCoor[2]; z < endCoor[2]; z += step[2]){
                        double x0 = x - d_x * floor((x-_lookupTableMinCoordinate[0])/d_x);
                        double y0 = y - d_y * floor((y-_lookupTableMinCoordinate[1])/d_y);
                        double z0 = z - d_z * floor((z-_lookupTableMinCoordinate[2])/d_z);
                        int idx_x = floor((x0-_lookupTableMinCoordinate[0])/step[0]);
                        int idx_y = floor((y0-_lookupTableMinCoordinate[1])/step[1]);
                        int idx_z = floor((z0-_lookupTableMinCoordinate[2])/step[2]);
                        double r = sqrt((x-coor[0])*(x-coor[0]) + (y-coor[1])*(y-coor[1]) + 
                                (z-coor[2])*(z-coor[2]));
                        if( (x-coor[0])*(x-coor[0]) + (y-coor[1])*(y-coor[1]) 
                                + (z-coor[2])*(z-coor[2]) < paddingLength*paddingLength){
                            //printf("r=%f pad=%f\n",r,paddingLength);
                            int idx = idx_x*n_y*n_z + idx_y*n_z + idx_z;
                            /*
                               printf("r0=(%f,%f,%f) pad=%f (%f,%f,%f)->(%f,%f,%f)->
                               (%d,%d,%d)->(%d)\n",coor[0],coor[1],coor[2],
                               paddingLength,x,y,z,x0,y0,z0,idx_x,idx_y,idx_z,idx);
                               */
                            if(_lookupTableNumAtoms[idx] < _lookupTableSize){
                                _lookupTable[idx][_lookupTableNumAtoms[idx]] = atomI;
                                _lookupTableNumAtoms[idx] ++;
                            }else{
                                //printf("warning%d\n",idx);
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
        double maxR = 0.03;
        double paddingLength = maxR + sqrt(3.0)/2.0*_lookupTableGridLength + _lookupTableBufferLength;
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
        }
        int n_x = _lookupTableNumberOfGridPoints[0];
        int n_y = _lookupTableNumberOfGridPoints[1];
        int n_z = _lookupTableNumberOfGridPoints[2];
        _lookupTable.resize(totalNumberOfGridPoints,vector<int>(_lookupTableSize));
        _lookupTableNumAtoms.resize(totalNumberOfGridPoints);
        std::fill(_lookupTableNumAtoms.begin(),_lookupTableNumAtoms.end(),0.0);

        //printf("min(%f,%f,%f) max(%f,%f,%f) n(%d,%d,%d)\n",_minCoordinate[0],_minCoordinate[1],_minCoordinate[2],
        //        _maxCoordinate[0],_maxCoordinate[1],_maxCoordinate[2],n_x,n_y,n_z);

        for(int atomI=0; atomI<numParticles; ++atomI){
            double paddingLength = _atomicRadii[atomI] + 
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
                        if(diff[0]*diff[0]+diff[1]*diff[1]+diff[2]*diff[2] < paddingLength*paddingLength){
                            if(_lookupTableNumAtoms[idx] < _lookupTableSize){
                                _lookupTable[idx][_lookupTableNumAtoms[idx]] = atomI;
                                _lookupTableNumAtoms[idx] ++;
                            }
                        }
                    }
                }
            }
        }
    } //end not periodic
    //long sum = 0;
    //for(auto &c : _lookupTableNumAtoms) sum += c;
    //cout<<"total number of atoms in lookup table = "<<sum<<endl;

    /**
    for(int i=0; i<_lookupTableNumberOfGridPoints[0]; i++){
        for(int j=0; j<_lookupTableNumberOfGridPoints[1]; j++){
            for(int k=0; k<_lookupTableNumberOfGridPoints[2]; k++){
                int idx = i*_lookupTableNumberOfGridPoints[1]*_lookupTableNumberOfGridPoints[2] + 
                    j*_lookupTableNumberOfGridPoints[2] + k;
                if(_lookupTableNumAtoms[idx]!=0){
                printf("(%d,%d,%d)-",i,j,k);
                for(int m=0; m<_lookupTableNumAtoms[idx]; m++) printf("-%d",_lookupTable[idx][m]);
                printf("\n");
                }
            }    
        }    
    }
    */
    //put the closest atom to the grid point in the first place
    /*
    OpenMM::Vec3 point;
    for(int i=0; i<_lookupTableNumberOfGridPoints[0]; i++){
        for(int j=0; j<_lookupTableNumberOfGridPoints[1]; j++){
            for(int k=0; k<_lookupTableNumberOfGridPoints[2]; k++){
                int idx = i*_lookupTableNumberOfGridPoints[1]*_lookupTableNumberOfGridPoints[2] + 
                    j*_lookupTableNumberOfGridPoints[2] + k;
                if(_lookupTableNumAtoms[idx]!=0){
                    point[0] = i*_lookupTableGridStep[0] + _minCoordinate[0];
                    point[1] = j*_lookupTableGridStep[1] + _minCoordinate[1];
                    point[2] = k*_lookupTableGridStep[2] + _minCoordinate[2];
                    int idx_min = _lookupTable[idx][0];
                    double dr_min = (atomCoordinates[idx_min]-point).dot(atomCoordinates[idx_min]-point);
                    for(int n=1; n<_lookupTableNumAtoms[idx]; n++){
                        double dr = (atomCoordinates[_lookupTable[idx][n]]-point).
                            dot(atomCoordinates[_lookupTable[idx][n]]-point);
                        if(dr < dr_min){
                            swap(_lookupTable[idx][0],_lookupTable[idx][n]);
                            dr_min = dr;
                        }
                    }
                }
            }
        } 
    }    
    */
    //end find min
}


void CustomGBIntegral::getLookupTableAtomList(OpenMM::Vec3 point, std::vector<int>* &atomList, int& numAtoms){
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
        double x0 = point[0] - floor((point[0]-_lookupTableMinCoordinate[0])/d_x)*d_x;
        double y0 = point[1] - floor((point[1]-_lookupTableMinCoordinate[1])/d_y)*d_y;
        double z0 = point[2] - floor((point[2]-_lookupTableMinCoordinate[2])/d_z)*d_z;
        //printf("(%f,%f,%f)->(%d,%d,%d)\n",point[0],point[1],point[2]);
        //calculate lookupTable index
        int idx_x = (x0-_lookupTableMinCoordinate[0])/_lookupTableGridStep[0];
        int idx_y = (y0-_lookupTableMinCoordinate[1])/_lookupTableGridStep[1]; 
        int idx_z = (z0-_lookupTableMinCoordinate[2])/_lookupTableGridStep[2]; 
        int idx = idx_x*(ny*nz) + idx_y*nz + idx_z;
        atomList = &_lookupTable[idx];
        numAtoms = _lookupTableNumAtoms[idx];
        //printf("%p\n",&_lookupTable[idx]);
        //printf("(%f,%f,%f)->(%f,%f,%f)->(%d,%d,%d)->%d\n",point[0],point[1],point[2],x0,y0,z0,
        //        idx_x,idx_y,idx_z,idx);
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
        atomList = &_lookupTable[lookupTableIdx];
        numAtoms = _lookupTableNumAtoms[lookupTableIdx];
    }
    /*
    printf("(%f,%f,%f) %d ",point[0],point[1],point[2], numAtoms);
    for(auto &c : atomList) printf("-%d",c);
    printf("\n");
    */
    return;
}

bool CustomGBIntegral::validateTransform(OpenMM::Vec3 newVec){
    for(int i=0; i<3; ++i){
        if( (newVec[i] > _maxCoordinate[i]) || (newVec[i] < _minCoordinate[i]) )
            return false;
    }
    return true;
}
