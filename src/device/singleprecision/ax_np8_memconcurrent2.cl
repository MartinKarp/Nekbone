#define LX1 8 
#define LY1 8 
#define LZ1 8 

__attribute__((scheduler_target_fmax_mhz(300)))
__kernel void ax(__global float * restrict w,
                        __global const float * restrict p,
                        __global const float * restrict g1,
                        __global const float * restrict g2,
                        __global const float * restrict g3,
                        __global const float * restrict g4,
                        __global const float * restrict g5,
                        __global const float * restrict g6,
                        __global const float * restrict dxm1,
                        __global const float * restrict dxtm1,
                        int N){
    #pragma max_concurrency(32) 
    for(unsigned ele = 0; ele < N; ele += LX1*LY1*LZ1){
        float shur[LX1*LY1*LZ1];
        float shus[LX1*LY1*LZ1];
        float shut[LX1*LY1*LZ1];
        float shw[LX1*LY1*LZ1];
        float shg1[LX1*LY1*LZ1];
        float shg2[LX1*LY1*LZ1];
        float shg3[LX1*LY1*LZ1];
        float shg4[LX1*LY1*LZ1];
        float shg5[LX1*LY1*LZ1];
        float shg6[LX1*LY1*LZ1];
        float shu[LX1*LY1*LZ1];
        float shdxm1[LX1*LY1];
        float shdxtm1[LX1*LY1];
        #pragma unroll 32
        for(unsigned ij=0; ij<LX1*LY1; ++ij){
            shdxm1[ij] = dxm1[ij];
            shdxtm1[ij] = dxtm1[ij];
        }

        #pragma unroll 32
        #pragma nofusion
        for(unsigned ijk=0; ijk<LX1*LY1*LZ1; ++ijk){
            shu[ijk] = p[ijk + ele];
            shg1[ijk] = g1[ijk + ele];
            shg2[ijk] = g2[ijk + ele];
            shg3[ijk] = g3[ijk + ele];
            shg4[ijk] = g4[ijk + ele];
            shg5[ijk] = g5[ijk + ele];
            shg6[ijk] = g6[ijk + ele];
        }
        #pragma unroll 32
        #pragma nofusion
        for(unsigned ijk=0; ijk<LX1*LY1*LZ1; ++ijk){
            shw[ijk] = shu[ijk] + shg1[ijk]+ shg2[ijk]+ shg3[ijk]+ shg4[ijk]+ shg5[ijk]+ shg6[ijk];
        }
 
        #pragma unroll 32
        #pragma nofusion
        for(unsigned ijk=0; ijk<LX1*LY1*LZ1; ++ijk)
            w[ijk + ele] = shw[ijk];
    }
}

