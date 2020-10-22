#define LX1 16 
#define LY1 16 
#define LZ1 16 

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
        #pragma unroll 64
        for(unsigned ij=0; ij<LX1*LY1; ++ij){
            shdxm1[ij] = dxm1[ij];
            shdxtm1[ij] = dxtm1[ij];
        }

        #pragma unroll 64
        for(unsigned ijk=0; ijk<LX1*LY1*LZ1; ++ijk){
            shu[ijk] = p[ijk + ele];
            shg1[ijk] = g1[ijk + ele];
            shg2[ijk] = g2[ijk + ele];
            shg3[ijk] = g3[ijk + ele];
            shg4[ijk] = g4[ijk + ele];
            shg5[ijk] = g5[ijk + ele];
            shg6[ijk] = g6[ijk + ele];
        }
        #pragma loop_coalesce
        #pragma ii 1
        for (unsigned k=0; k<LZ1; ++k){
            for(unsigned j = 0; j < LY1; j++){
                #pragma unroll 8
                for(unsigned i = 0; i < LX1; i++){
                    int ij = i + j*LX1;
                    int ijk = ij + k*LX1*LY1;
                    float G00 = shg1[ijk];
                    float G01 = shg2[ijk];
                    float G02 = shg3[ijk];
                    float G11 = shg4[ijk];
                    float G12 = shg5[ijk];
                    float G22 = shg6[ijk];
                    float rtmp = 0.0;
                    float stmp = 0.0;
                    float ttmp = 0.0;
                    #pragma unroll 
                    for (unsigned l = 0; l<LX1; l++){
                      rtmp += shdxtm1[l+i*LX1] * shu[l+j*LX1 +k*LX1*LY1];
                      stmp += shdxtm1[l+j*LX1] * shu[i+l*LX1 + k*LX1*LY1];
                      ttmp += shdxtm1[l+k*LX1] * shu[ij + l*LX1*LY1];
                    }
                    shur[ijk] = G00*rtmp
                             + G01*stmp
                             + G02*ttmp;
                    shus[ijk] = G01*rtmp
                             + G11*stmp
                             + G12*ttmp;
                    shut[ijk]  = G02*rtmp
                             + G12*stmp
                             + G22*ttmp;
                }
            }
        }
        #pragma loop_coalesce 
        #pragma ii 1
        for (unsigned k=0; k<LZ1; ++k){
            for(unsigned j = 0; j < LY1; j++){
                #pragma unroll 8
                for(unsigned i = 0; i < LX1; i++){
                    int ij = i + j*LX1;
                    int ijk = ij + k*LX1*LY1;
                    
                    float wijke = 0.0;
                    #pragma unroll 
                    for(unsigned l = 0; l<LX1; l++){
                        wijke += shdxm1[l + i*LX1] * shur[l+j*LX1+k*LX1*LY1];
                        wijke += shdxm1[l + j*LX1] * shus[i+l*LX1+k*LX1*LY1];
                        wijke += shdxm1[l + k*LX1] * shut[i+j*LX1+l*LX1*LY1];
                    }
                    shw[ijk] = wijke;
                }
            }
        }

        #pragma unroll 64
        for(unsigned ijk=0; ijk<LX1*LY1*LZ1; ++ijk)
            w[ijk + ele] = shw[ijk];
    }
}

