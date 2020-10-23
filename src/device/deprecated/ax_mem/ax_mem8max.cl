#define LX1 8
#define LY1 8
#define LZ1 8

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

    #pragma max_concurrency 256
    for(unsigned ele = 0; ele < N; ele += LX1*LY1*LZ1){
       #pragma unroll 32
        for(unsigned ijk=0; ijk<LX1*LY1*LZ1; ++ijk){
            w[ijk+ele] = p[ijk+ele] + g1[ijk+ele]+ g2[ijk+ele]+ g3[ijk+ele]+ g4[ijk+ele]+ g5[ijk+ele]+ g6[ijk+ele];
        }
    }
}

