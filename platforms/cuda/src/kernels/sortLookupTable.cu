
extern "C" __global__ void sortLookupTable(const real4* __restrict__ posq, 
#ifdef USE_PERIODIC
        const real4 periodicBoxSize, const real4 invPeriodicBoxSize, 
#endif
        int* __restrict__ lookupTable, int* __restrict__ lookupTableNumAtoms, 
        const int* __restrict__ lookupTableNumGridPoints, const float* __restrict__ lookupTableMinCoor, 
        const float* __restrict__ lookupTableGridStep)
{   
    int n_x = lookupTableNumGridPoints[0];
    int n_y = lookupTableNumGridPoints[1];
    int n_z = lookupTableNumGridPoints[2];
    float3 minCoor;
    minCoor.x = lookupTableMinCoor[0];
    minCoor.y = lookupTableMinCoor[1];
    minCoor.z = lookupTableMinCoor[2];
    float3 gridStep;
    gridStep.x = lookupTableGridStep[0];
    gridStep.y = lookupTableGridStep[1];
    gridStep.z = lookupTableGridStep[2];

    __shared__ float dist2[LOOKUPTABLE_SIZE];
    __shared__ int offset[LOOKUPTABLE_SIZE];

    int idx_x = blockIdx.x;
    int idx_y = blockIdx.y;
    int idx_z = blockIdx.z;
    int lookupTableIdx = idx_x*n_y*n_z + idx_y*n_z + idx_z;
    int numAtoms = lookupTableNumAtoms[lookupTableIdx];
    //if(numAtoms>0) printf("%d,%d,%d,%d,%d\n",idx_x,idx_y,idx_z,lookupTableIdx,numAtoms);
    
    /*
    if(threadIdx.x < numAtoms){
        int atomIdx = lookupTableIdx*LOOKUPTABLE_SIZE + threadIdx.x;
        int atomI = lookupTable[atomIdx];
        real4 pos = posq[atomI];
        float3 gridPoint = make_float3(minCoor.x + idx_x*gridStep.x, minCoor.y + idx_y*gridStep.y, minCoor.z + idx_z*gridStep.z);
        float3 delta = make_float3(pos.x - gridPoint.x, 
                pos.y - gridPoint.y, pos.z - gridPoint.z);
#ifdef USE_PERIODIC
        APPLY_PERIODIC_TO_DELTA(delta);
#endif
        float r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
        dist2[threadIdx.x] = r2;
        offset[threadIdx.x] = -1;
        __syncthreads();
        int location = 0;
        for(int i=0; i<numAtoms; i++){
            if(i!=threadIdx.x && r2 > dist2[i]) location++;
        }
        //atomicAdd(&offset[threadIdx.x],1);
        //location += offset[threadIdx.x];
        lookupTable[lookupTableIdx*LOOKUPTABLE_SIZE + location] = atomI;
    }
    */

}
