
/*
   This kernel is used to compute the GBSW force
   float chain##idx## = dEdI
   float3 delta = r_j - r_q
*/

// for atomJ in neighbor list of quadrature point q
float3 dIdr_vec = make_float3(0,0,0);
float deltaR_qj = sqrtf(delta.x*delta.x + delta.y*delta.y + delta.z*delta.z);
float deltaR2 = deltaR_qj*deltaR_qj;
float atomicRadii_j = radius[atomJ];
float atomicRadii_j2 = atomicRadii_j*atomicRadii_j;

float C_j = P1 * atomicRadii_j * 10.0 + P2;
float F_VSA = C_j / (C_j + deltaR2*100.0 - atomicRadii_j2*100.0);
F_VSA = F_VSA*F_VSA;

float tmp0 = (C_j + deltaR2*100.0 - atomicRadii_j2*100.0);
float dF_VSA_dr_factor = -400.0*C_j*C_j/(tmp0*tmp0*tmp0);
float tmp1 = 1.0 * dF_VSA_dr_factor * sum2 / (sum3*sum3);

dIdr_vec.x -= tmp1 * delta.x;
dIdr_vec.y -= tmp1 * delta.y;
dIdr_vec.z -= tmp1 * delta.z;

float tmp2 = 2.0*F_VSA*(F_VSA + dF_VSA_dr_factor*deltaR2) / (sum3*sum3) * sum1;
dIdr_vec.x -= tmp2 * delta.x;
dIdr_vec.y -= tmp2 * delta.y;
dIdr_vec.z -= tmp2 * delta.z;

float tmp3 = -2.0 / (sum3*sum3*sum3*sum3) * sum1 * sum2;
float3 denom_vec_dr1;
denom_vec_dr1.x = tmp3 * (vector_sum.x * F_VSA +
        (vector_sum.x*delta.x + vector_sum.y*delta.y + vector_sum.z*delta.z)*dF_VSA_dr_factor*delta.x);

denom_vec_dr1.y = tmp3 * (vector_sum.y * F_VSA +
        (vector_sum.x*delta.x + vector_sum.y*delta.y + vector_sum.z*delta.z)*dF_VSA_dr_factor*delta.y);

denom_vec_dr1.z = tmp3 * (vector_sum.z * F_VSA +
        (vector_sum.x*delta.x + vector_sum.y*delta.y + vector_sum.z*delta.z)*dF_VSA_dr_factor*delta.z);

dIdr_vec.x -= denom_vec_dr1.x;
dIdr_vec.y -= denom_vec_dr1.y;
dIdr_vec.z -= denom_vec_dr1.z;

