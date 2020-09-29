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
        double shur[LX1*LY1*LZ1];
        double shus[LX1*LY1*LZ1];
        double shut[LX1*LY1*LZ1];
        double shw[LX1*LY1*LZ1];
    	double shu[LX1*LY1*LZ1];
        double shdxm1[LX1*LY1];
        double shg1[LX1*LY1*LZ1];
        double shg2[LX1*LY1*LZ1];
        double shg3[LX1*LY1*LZ1];
        double shg4[LX1*LY1*LZ1];
        double shg5[LX1*LY1*LZ1];
        double shg6[LX1*LY1*LZ1];
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

