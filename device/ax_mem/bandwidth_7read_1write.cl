
__kernel void ax(__global double * restrict w,
                        __global const double * restrict p,
                        __global const double * restrict g1,
                        __global const double * restrict g2,
                        __global const double * restrict g3,
                        __global const double * restrict g4,
                        __global const double * restrict g5,
                        __global const double * restrict g6,
                        __global const double * restrict dxm1,
                        int N){

    #pragma unroll 32
    for(unsigned ijk=0; ijk<N; ++ijk){
       w[ijk] = p[ijk] + g1[ijk]+ g2[ijk]+ g3[ijk]+ g4[ijk]+ g5[ijk]+ g6[ijk];
    }
}

