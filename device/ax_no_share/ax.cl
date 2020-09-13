#define LX1 10
#define LY1 10
#define LZ1 10

__attribute__((scheduler_target_fmax_mhz(300)))
__kernel void ax(__global double * restrict w,
                        __global const double * restrict p,
                        __global const double * restrict g1,
                        __global const double * restrict g2,
                        __global const double * restrict g3,
                        __global const double * restrict g4,
                        __global const double * restrict g5,
                        __global const double * restrict g6,
                        __global const double * restrict dxm1,
                        __global const double * restrict dxtm1,
                        int N){

    #pragma speculated_iterations LX1*LY1*LZ1
    for(unsigned ele = 0; ele < N; ele += LX1*LY1*LZ1){
        double shur[LX1*LY1*LZ1];
        double shus[LX1*LY1*LZ1];
        double shut[LX1*LY1*LZ1];
       
        #pragma speculated_iterations LX1*LY1*LZ1
        #pragma loop_coalesce
        for (unsigned k=0; k<LZ1; ++k){
            for(unsigned j = 0; j < LY1; j++){
                for(unsigned i = 0; i < LX1; i++){
                    int ij = i + j*LX1;
                    int ijk = ij + k*LX1*LY1;
                    double G00 = g1[ijk + ele];
                    double G01 = g2[ijk + ele];
                    double G02 = g3[ijk + ele];
                    double G11 = g4[ijk + ele];
                    double G12 = g5[ijk + ele];
                    double G22 = g6[ijk + ele];
                    double rtmp = 0.0;
                    double stmp = 0.0;
                    double ttmp = 0.0;
                    #pragma unroll 
                    for (unsigned l = 0; l<LX1; l++){
                      rtmp += dxtm1[l+i*LX1] * p[l+j*LX1 +k*LX1*LY1 + ele];
                      stmp += dxtm1[l+j*LX1] * p[i+l*LX1 + k*LX1*LY1 + ele];
                      ttmp += dxtm1[l+k*LX1] * p[ij + l*LX1*LY1 + ele];
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
        #pragma speculated_iterations LX1*LY1*LZ1
        #pragma loop_coalesce 
        for (unsigned k=0; k<LZ1; ++k){
            for(unsigned j = 0; j < LY1; j++){
                #pragma unroll 2
                for(unsigned i = 0; i < LX1; i++){
                    int ij = i + j*LX1;
                    int ijk = ij + k*LX1*LY1;
                    
                    double wijke = 0.0;
                    #pragma unroll 
                    for(unsigned l = 0; l<LX1; l++){
                        wijke += dxm1[l + i*LX1] * shur[l+j*LX1+k*LX1*LY1];
                        wijke += dxm1[l + j*LX1] * shus[i+l*LX1+k*LX1*LY1];
                        wijke += dxm1[l + k*LX1] * shut[i+j*LX1+l*LX1*LY1];
                    }
                    w[ijk + ele] = wijke;
                }
            }
        }
    }
}

