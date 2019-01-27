__inline__ __device__
float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) 
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

extern "C" __global__ void reduceGBSWForce(unsigned long long* __restrict__ forceBuffers,
        const real4* __restrict__ posq 
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
        //for(int quadIdx = threadIdx.x; quadIdx < NUM_QUADRATURE_POINTS; quadIdx += blockDim.x){
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
        int lookupTableIdx = (lookupTableIdx3.x*(numGridPoints.y*numGridPoints.z) + 
                lookupTableIdx3.y*numGridPoints.z + lookupTableIdx3.z);
        int numLookupTableAtoms = lookupTableNumAtoms[lookupTableIdx];
        //compute Volume
        BEFORE_VOLUME
        for(int i = 0; i<numLookupTableAtoms; i++){
            float3 forceI = make_float3(0,0,0);
            float3 forceJ = make_float3(0,0,0);
            int atomJ = lookupTable[lookupTableIdx*25 + i];
            float4 posJ = posq[atomJ];
            float3 delta = make_float3(posJ.x - quadPosR.x, 
                    posJ.y - quadPosR.y, posJ.z - quadPosR.z);
#ifdef USE_PERIODIC
            APPLY_PERIODIC_TO_DELTA(delta);
#endif
            float sw = 0.03;
            float sw3 = sw*sw*sw;
            float deltaR_qj = sqrt(delta.x*delta.x + delta.y*delta.y + delta.z*delta.z);
            float atomicRadii_j = (radius[atomJ]+0.03)*0.9520;
            float dr = deltaR_qj - atomicRadii_j;
            float dr2 = dr*dr;
            float dr3 = dr*dr*dr;
            float u_j;
            float duj_drq;
            if((deltaR_qj > atomicRadii_j - sw) && (deltaR_qj < atomicRadii_j + sw)){
                u_j = 0.5 + 3.0/(4.0*sw) * dr - 1.0/(4.0*sw3) * dr3;
                duj_drq = 3.0/(4.0*sw) - 3.0/(4.0*sw3) * dr2;
            }else{
                continue;
            } 
            float chain = (duj_drq*V/u_j)/deltaR_qj;
            COMPUTE_VOLUME
        }
        AFTER_VOLUME
        // reduction
        /*
        int lane = threadIdx.x % warpSize;
        int wid = threadIdx.x / warpSize;
        res = warpReduceSum(res);
        if(lane==0) tmp_result[wid] = res;
        __syncthreads();
        res = (threadIdx.x < blockDim.x / warpSize) ? tmp_result[lane] : 0;
        if(wid==0){
            res = warpReduceSum(res);
            if(threadIdx.x == 0) atomicAdd(&integrals[atomI],res);
        }
        */
        } // end check quadrature point boundary
    }
    /*
    __syncthreads();
    for(int atomI = blockIdx.x; atomI < NUM_ATOMS; atomI += gridDim.x){
        if(threadIdx.x == 0 && blockIdx.y==0)
            printf("%d %f %f %f -> %f %f\n",atomI,posq[atomI].x,posq[atomI].y,posq[atomI].z,integral0[atomI],integral1[atomI]);
    }
    */
}
