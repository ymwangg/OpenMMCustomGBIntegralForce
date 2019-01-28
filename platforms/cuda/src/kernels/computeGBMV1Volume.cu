//COMPUTE_VOLUME
float deltaR_qj = sqrtf(delta.x*delta.x + delta.y*delta.y + delta.z*delta.z);
float deltaR4 = deltaR_qj*deltaR_qj*deltaR_qj*deltaR_qj;
float atomicRadii_j = radius[atomJ];
float atomicRadii_j4 = atomicRadii_j*atomicRadii_j*atomicRadii_j*atomicRadii_j;
float gammaj = GAMMA * logf(LAMBDA) / (atomicRadii_j4);
sum += expf(gammaj * deltaR4);
