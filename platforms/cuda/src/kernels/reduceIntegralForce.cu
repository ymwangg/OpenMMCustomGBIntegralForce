extern "C" __global__ void reduceGBSWForce(unsigned long long* __restrict__ forceBuffers,
        const real4* __restrict__ posq 
#ifdef USE_PERIODIC
        ,const real4 periodicBoxSize, const real4 invPeriodicBoxSize
#endif
PARAM_ARGS)
{   
    float3 minCoor = make_float3(lookupTableMinCoor[0],lookupTableMinCoor[1],lookupTableMinCoor[2]);
    int3 numGridPoints = make_int3(lookupTableNumGridPoints[0],lookupTableNumGridPoints[1],lookupTableNumGridPoints[2]);
    float3 gridStep = make_float3(lookupTableGridStep[0], lookupTableGridStep[1], lookupTableGridStep[2]);

    INIT_PARAMS

    //each blockIdx.x maps to an atom
    for(int atomI = blockIdx.x; atomI < NUM_ATOMS; atomI += gridDim.x){
        float3 forceI = make_float3(0.0,0.0,0.0);
        float4 posI = posq[atomI];
        int quadIdx = blockIdx.y*blockDim.x + threadIdx.x;
        LOAD_VOLUME
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
            float3 forceJ = make_float3(0.0,0.0,0.0);
            float4 posJ = posq[atomJ];
            float3 delta = make_float3(posJ.x - quadPosR.x, 
                    posJ.y - quadPosR.y, posJ.z - quadPosR.z);
#ifdef USE_PERIODIC
            APPLY_PERIODIC_TO_DELTA(delta);
#endif
            COMPUTE_FORCE
            APPLY_CHAIN_RULE


            if(abs(forceJ.x)>1e-10) atomicAdd(&forceBuffers[atomJ], static_cast<unsigned long long>((long long) (forceJ.x*0x100000000)));
            if(abs(forceJ.y)>1e-10) atomicAdd(&forceBuffers[atomJ+NUM_PADDED_ATOMS], static_cast<unsigned long long>((long long) (forceJ.y*0x100000000)));
            if(abs(forceJ.z)>1e-10) atomicAdd(&forceBuffers[atomJ+NUM_PADDED_ATOMS*2], static_cast<unsigned long long>((long long) (forceJ.z*0x100000000)));
        } // end atomJ
        /*
        float3 origin = make_float3(1/((real) 0x100000000) * (long long) forceBuffers[atomI] ,
                1/((real) 0x100000000) * (long long)forceBuffers[atomI+NUM_PADDED_ATOMS],
                1/((real) 0x100000000) * (long long)forceBuffers[atomI+NUM_PADDED_ATOMS*2]);
                */
        if(abs(forceI.x)>1e-10) atomicAdd(&forceBuffers[atomI], static_cast<unsigned long long>((long long) (forceI.x*0x100000000)));
        if(abs(forceI.y)>1e-10) atomicAdd(&forceBuffers[atomI+NUM_PADDED_ATOMS], static_cast<unsigned long long>((long long) (forceI.y*0x100000000)));
        if(abs(forceI.z)>1e-10) atomicAdd(&forceBuffers[atomI+NUM_PADDED_ATOMS*2], static_cast<unsigned long long>((long long) (forceI.z*0x100000000)));
        /*
        float3 end = make_float3(1/((real) 0x100000000) * (long long) forceBuffers[atomI] ,
                1/((real) 0x100000000) * (long long)forceBuffers[atomI+NUM_PADDED_ATOMS],
                1/((real) 0x100000000) * (long long)forceBuffers[atomI+NUM_PADDED_ATOMS*2]);
        if(end.x*end.x+end.y*end.y+end.z*end.z > 1e9){
            printf("%e,%e,%e + %e,%e,%e -> %e,%e,%e\n",forceI.x,forceI.y,forceI.z,origin.x,origin.y,origin.z,end.x,end.y,end.z);
        }
            */
        } // end quadrature point
    } // end atomI
}
