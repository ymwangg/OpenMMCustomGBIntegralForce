
/*
   This kernel is used to compute the GBSW force
   float chain##idx## = dEdI
   float3 delta = r_j - r_q
*/

// for atomJ in neighbor list of quadrature point q

float sw = SWITCH_DISTANCE;
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
float dIdr = (duj_drq*V/u_j)/deltaR_qj;

// end for atomJ in neighbor list of quadrature point q

