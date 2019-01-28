
/*
   This kernel is used to compute the GBSW force
   float chain##idx## = dEdI
   float3 delta = r_j - r_q
*/

// for atomJ in neighbor list of quadrature point q
float deltaR_qj = sqrtf(delta.x*delta.x + delta.y*delta.y + delta.z*delta.z);
float deltaR3 = deltaR_qj*deltaR_qj*deltaR_qj;
float deltaR4 = deltaR3*deltaR_qj;
float atomicRadii_j = radius[atomJ];
float atomicRadii_j4 = atomicRadii_j*atomicRadii_j*atomicRadii_j*atomicRadii_j;
float gammaj = GAMMA * logf(LAMBDA) / (atomicRadii_j4);
float factor2 = factor*gammaj*expf(gammaj*deltaR4)*(4.0*deltaR3);
float dIdr = -1.0/(deltaR_qj)*factor2;
// end for atomJ in neighbor list of quadrature point q

