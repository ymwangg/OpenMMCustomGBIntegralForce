
/**
 * Compute a force based on pair interactions.
 */
extern "C" __global__ void computeLookupTable(const real4* __restrict__ posq, const real4 periodicBoxSize, const real4 invPeriodicBoxSize, const float* __restrict__ radius, int* __restrict__ lookupTable, int* __restrict__ lookupTableNumAtoms, const int* __restrict__ lookupTableNumGridPoints, const float* __restrict__ lookupTableMinCoor, const float* __restrict__ lookupTableGridStep, const float lookupTableGridLength, const float lookupTableBufferLength, const int lookupTableSize)
{   
    int n_x = lookupTableNumGridPoints[0];
    int n_y = lookupTableNumGridPoints[1];
    int n_z = lookupTableNumGridPoints[2];
    float3 beginCoor;
    float3 endCoor;
    float3 minCoor;
    minCoor.x = lookupTableMinCoor[0];
    minCoor.y = lookupTableMinCoor[1];
    minCoor.z = lookupTableMinCoor[2];
    float3 gridStep;
    gridStep.x = lookupTableGridStep[0];
    gridStep.y = lookupTableGridStep[1];
    gridStep.z = lookupTableGridStep[2];

    for (unsigned int index = blockIdx.x*blockDim.x+threadIdx.x; index < NUM_ATOMS; index += blockDim.x*gridDim.x){
        //0.3 changed to atomicRadii
        real4 pos = posq[index];
        float vdwRadii = radius[index];
        //printf("%d-(%f,%f,%f)-(%f)\n",index,pos.x,pos.y,pos.z,vdwRadii);
        // sqrt(3)/2 = 0.8660254
        float paddingLength =  vdwRadii + sqrtf(3.0)/2*lookupTableGridLength + lookupTableBufferLength;
        /*
        beginCoor.x = (pos.x - paddingLength - gridStep.x);
        endCoor.x = (pos.x + paddingLength + gridStep.x);
        beginCoor.y = (pos.y - paddingLength - gridStep.y);
        endCoor.y = (pos.y + paddingLength + gridStep.y);
        beginCoor.z = (pos.z - paddingLength - gridStep.z);
        endCoor.z = (pos.z + paddingLength + gridStep.z);
        */
        beginCoor.x = (pos.x - paddingLength);
        endCoor.x = (pos.x + paddingLength);
        beginCoor.y = (pos.y - paddingLength);
        endCoor.y = (pos.y + paddingLength);
        beginCoor.z = (pos.z - paddingLength);
        endCoor.z = (pos.z + paddingLength);
        int s=0;
        for(float x = beginCoor.x; x < endCoor.x; x += gridStep.x){
            for(float y = beginCoor.y; y < endCoor.y; y += gridStep.y){
                for(float z = beginCoor.z; z < endCoor.z; z += gridStep.z){
                    //if(index==0) printf("%f,%f,%f\n",x,y,z);

                    //apply periodic boundary condition
                    float x0 = x - periodicBoxSize.x * 
                        floorf((x - minCoor.x)/periodicBoxSize.x);
                    float y0 = y - periodicBoxSize.y * 
                        floorf((y - minCoor.y)/periodicBoxSize.y);
                    float z0 = z - periodicBoxSize.z * 
                        floorf((z - minCoor.z)/periodicBoxSize.z);

                    //calculate lookupTable grid index
                    int idx_x = floorf((x0 - minCoor.x) / gridStep.x);
                    int idx_y = floorf((y0 - minCoor.y) / gridStep.y);
                    int idx_z = floorf((z0 - minCoor.z) / gridStep.z);

                    float r2 = (x-pos.x)*(x-pos.x) + (y-pos.y)*(y-pos.y) + (z-pos.z)*(z-pos.z);
                    if(r2 < paddingLength*paddingLength){
                        int lookupTableIdx = idx_x*n_y*n_z + idx_y*n_z + idx_z;
                        int insertionIdx = atomicAdd(&lookupTableNumAtoms[lookupTableIdx],1);
                        if(insertionIdx < lookupTableSize){
                            int location = lookupTableIdx*lookupTableSize + insertionIdx;
                            /*
                            if(location < 0) {
                                printf("abnormal %d<-(%d,%d,%d)\n",location,lookupTableIdx,lookupTableSize,insertionIdx);
                                printf("(%f,%f,%f)->(%f,%f,%f)->min(%f,%f,%f)(%d,%d,%d)\n",x,y,z,x0,y0,z0,minCoor.x,minCoor.y,minCoor.z,
                                        idx_x,idx_y,idx_z);
                            }
                            */
                            lookupTable[location] = index;
                        }else{
                            insertionIdx = atomicAdd(&lookupTableNumAtoms[lookupTableIdx],-1);
                        }
                    }
                    s ++;
                } // loop z
            } // loop y
        } // loop x
    } // loop threadIdx
}
