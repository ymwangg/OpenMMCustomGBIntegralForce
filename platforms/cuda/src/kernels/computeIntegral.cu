__inline__ __device__
float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) 
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

extern "C" __global__ void computeGBSWIntegral(const real4* __restrict__ posq 
#ifdef USE_PERIODIC
        ,const real4 periodicBoxSize, const real4 invPeriodicBoxSize
#endif
PARAM_ARGS)
{   
    INIT_PARAMS
    //each blockIdx.x maps to an atom
    for(int atomI = blockIdx.x; atomI < NUM_ATOMS; atomI += gridDim.x){
        float4 posI = posq[atomI];
        int quadIdx = blockIdx.y*blockDim.x + threadIdx.x;
        if(quadIdx < NUM_QUADRATURE_POINTS){
        float4 quadPosR = make_float4(posI.x + quad[quadIdx].x, 
                posI.y + quad[quadIdx].y, posI.z + quad[quadIdx].z, quad[quadIdx].w);
#ifdef USE_PERIODIC
        float3 quadPos0 = make_float3(
                quadPosR.x - periodicBoxSize.x * floorf((quadPosR.x - minCoor.x)/periodicBoxSize.x),
                quadPosR.y - periodicBoxSize.y * floorf((quadPosR.y - minCoor.y)/periodicBoxSize.y),
                quadPosR.z - periodicBoxSize.z * floorf((quadPosR.z - minCoor.z)/periodicBoxSize.z));
        int3 lookupTableIdx3 = make_int3(
                (quadPos0.x - minCoor.x)/gridStep.x,
                (quadPos0.y - minCoor.y)/gridStep.y,
                (quadPos0.z - minCoor.z)/gridStep.z);
#else
        int3 lookupTableIdx3 = make_int3(
                (quadPosR.x - minCoor.x)/gridStep.x,
                (quadPosR.y - minCoor.y)/gridStep.y,
                (quadPosR.z - minCoor.z)/gridStep.z);
#endif
        //compute Volume
        BEFORE_VOLUME
#ifdef USE_LOOKUP_TABLE
        int lookupTableIdx = (lookupTableIdx3.x*(numGridPoints.y*numGridPoints.z) + 
                lookupTableIdx3.y*numGridPoints.z + lookupTableIdx3.z);
        int numLookupTableAtoms = lookupTableNumAtoms[lookupTableIdx];
        for(int i = 0; i<numLookupTableAtoms; i++){
            int atomJ = lookupTable[lookupTableIdx*LOOKUPTABLE_SIZE + i];
#else
        for(int atomJ = 0; atomJ < NUM_ATOMS; atomJ++){
#endif
            float4 posJ = posq[atomJ];
            float3 delta = make_float3(posJ.x - quadPosR.x, 
                    posJ.y - quadPosR.y, posJ.z - quadPosR.z);
#ifdef USE_PERIODIC
            APPLY_PERIODIC_TO_DELTA(delta);
#endif
            COMPUTE_VOLUME

        } // end atomJ

        AFTER_VOLUME

        REDUCTION

        } // end quadrature point 
    } // end atomI
}
