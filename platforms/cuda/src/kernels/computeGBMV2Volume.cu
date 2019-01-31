//COMPUTE_VOLUME
float deltaR_qj = sqrtf(delta.x*delta.x + delta.y*delta.y + delta.z*delta.z);
float deltaR2 = deltaR_qj*deltaR_qj;
float atomicRadii_j = radius[atomJ];
float atomicRadii_j2 = atomicRadii_j*atomicRadii_j;
float C_j = P1 * atomicRadii_j + P2;
float F_VSA = C_j / (C_j + deltaR2 - atomicRadii_j2);
F_VSA = F_VSA*F_VSA;
sum1 += F_VSA;
sum2 += deltaR2*(F_VSA*F_VSA);
vector_sum.x += F_VSA * delta.x;
vector_sum.y += F_VSA * delta.y;
vector_sum.z += F_VSA * delta.z;
