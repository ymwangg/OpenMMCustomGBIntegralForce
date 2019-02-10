//COMPUTE_VOLUME
float sw = 0.03;
float sw3 = sw*sw*sw;
float deltaR_qj = sqrtf(delta.x*delta.x + delta.y*delta.y + delta.z*delta.z);
float atomicRadii_j = (radius[atomJ]+0.03)*0.9520;
float dr = deltaR_qj - atomicRadii_j;
float dr3 = dr*dr*dr;
if(deltaR_qj <= atomicRadii_j - sw){
    V = 0.0;
    break;
}else if(deltaR_qj >= atomicRadii_j + sw){
    continue;
}else{
    V *= 0.5 + 3.0/(4.0*sw) * dr - 1.0/(4.0*sw3) * dr3;
}
