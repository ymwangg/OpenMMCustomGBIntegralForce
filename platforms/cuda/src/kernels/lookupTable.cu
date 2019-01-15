
/**
 * Compute a force based on pair interactions.
 */
extern "C" __global__ void computeLookupTable( const real4* __restrict__ posq, float** lookupTable, int* lookupTableNumAtoms,
//#ifdef USE_CUTOFF
        real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ)
//#endif
        {   
        //PARAMETER_ARGUMENTS) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x ;
    real4 pos = posq[idx];
    //printf("%d %f %f %f %f\n",idx,pos.x,pos.y,pos.z,pos.w);
    if(idx==0){
        printf("periodicBoxSize %f %f %f %f\n",periodicBoxSize.x,periodicBoxSize.y,periodicBoxSize.z,periodicBoxSize.w);
        printf("invPeriodicBoxSize %f %f %f %f\n",invPeriodicBoxSize.x,invPeriodicBoxSize.y,invPeriodicBoxSize.z,invPeriodicBoxSize.w);
        printf("periodicBoxVecX %f %f %f %f\n",periodicBoxVecX.x,periodicBoxVecX.y,periodicBoxVecX.z,periodicBoxVecX.w);
        printf("periodicBoxVecY %f %f %f %f\n",periodicBoxVecY.x,periodicBoxVecY.y,periodicBoxVecY.z,periodicBoxVecY.w);
        printf("periodicBoxVecZ %f %f %f %f\n",periodicBoxVecZ.x,periodicBoxVecZ.y,periodicBoxVecZ.z,periodicBoxVecZ.w);
    }
}
