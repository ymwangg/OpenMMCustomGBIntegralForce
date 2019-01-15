/**
 * Compute per-particle values and total derivatives
 */

extern "C" __global__ void computePerParticleValues(real4* posq 
        PARAMETER_ARGUMENTS) {
    for (unsigned int index = blockIdx.x*blockDim.x+threadIdx.x; index < NUM_ATOMS; index += blockDim.x*gridDim.x) {
        // Calculate other values and total derivatives of dValuedIntegral and dValuedParam
        real4 pos = posq[index];
        COMPUTE_VALUES
    }
}
