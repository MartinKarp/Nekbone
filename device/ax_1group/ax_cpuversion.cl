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

    for(unsigned ele = 0; ele < N; ele += LX1*LY1*LZ1){
        double shur[LX1*LY1*LZ1];
        double shus[LX1*LY1*LZ1];
        double shut[LX1*LY1*LZ1];
        double shw[LX1*LY1*LZ1];
        double r[LX1*LY1*LZ1];
        double s[LX1*LY1*LZ1];
        double t[LX1*LY1*LZ1];
        #pragma nofusion
        for (unsigned k=0; k<LZ1; ++k){
            for(unsigned j = 0; j < LY1; j++){
                for(unsigned i = 0; i < LX1; i++){
                    int ij = i + j*LX1;
                    int ijk = ij + k*LX1*LY1;
                    double rtmp = 0.0;
                    #pragma unroll
                    for (unsigned l = 0; l<LX1; l++){
                      rtmp += dxm1[i+l*LX1] * p[l+j*LX1 +k*LX1*LY1];
                    }
                    r[ijk] = rtmp;
                }
            }
        }
        #pragma nofusion
        for (unsigned k=0; k<LZ1; ++k){
            for(unsigned j = 0; j < LY1; j++){
                for(unsigned i = 0; i < LX1; i++){
                    int ij = i + j*LX1;
                    int ijk = ij + k*LX1*LY1;
                    double stmp = 0.0;
                    #pragma unroll
                    for (unsigned l = 0; l<LX1; l++){
                      stmp += dxtm1[l+j*LX1] * p[i+l*LX1 + k*LX1*LY1];
                    }
                    s[ijk] = stmp;
                }
            }
        }
        #pragma nofusion
        for (unsigned k=0; k<LZ1; ++k){
            for(unsigned j = 0; j < LY1; j++){
                for(unsigned i = 0; i < LX1; i++){
                    int ij = i + j*LX1;
                    int ijk = ij + k*LX1*LY1;
                    double ttmp = 0.0;
                    #pragma unroll
                    for (unsigned l = 0; l<LX1; l++){
                      ttmp += dxtm1[l+k*LX1] * p[ij + l*LX1*LY1];
                    }
                    t[ijk] = ttmp;
                }
            }
        }
        for(unsigned ijk=0; ijk<LX1*LY1*LZ1; ++ijk){
            double G00 = g1[ijk + ele];
            double G01 = g2[ijk + ele];
            double G02 = g3[ijk + ele];
            double G11 = g4[ijk + ele];
            double G12 = g5[ijk + ele];
            double G22 = g6[ijk + ele];
            shur[ijk] = G00*r[ijk]
                     + G01*s[ijk]
                     + G02*t[ijk];
            shus[ijk] = G01*r[ijk]
                     + G11*s[ijk]
                     + G12*t[ijk];
            shut[ijk]  = G02*r[ijk]
                     + G12*s[ijk]
                     + G22*t[ijk];
        }
        #pragma nofusion
        for (unsigned k=0; k<LZ1; ++k){
            for(unsigned j = 0; j < LY1; j++){
                for(unsigned i = 0; i < LX1; i++){
                    int ij = i + j*LX1;
                    int ijk = ij + k*LX1*LY1;
                    
                    double wijke = 0.0;
                    for(unsigned l = 0; l<LX1; l++){
                        wijke += dxm1[l + i*LX1] * shur[l+j*LX1+k*LX1*LY1];
                    }
                    shw[ijk] = wijke;
                }
            }
        }
        #pragma nofusion
        for (unsigned k=0; k<LZ1; ++k){
            for(unsigned j = 0; j < LY1; j++){
                for(unsigned i = 0; i < LX1; i++){
                    int ij = i + j*LX1;
                    int ijk = ij + k*LX1*LY1;
                    
                    double wijke = 0.0;
                    for(unsigned l = 0; l<LX1; l++){
                        wijke += dxm1[l + j*LX1] * shus[i+l*LX1+k*LX1*LY1];
                    }
                    shw[ijk] += wijke;
                }
            }
        }
        #pragma nofusion
        for (unsigned k=0; k<LZ1; ++k){
            for(unsigned j = 0; j < LY1; j++){
                for(unsigned i = 0; i < LX1; i++){
                    int ij = i + j*LX1;
                    int ijk = ij + k*LX1*LY1;
                    
                    double wijke = 0.0;
                    for(unsigned l = 0; l<LX1; l++){
                        wijke += dxm1[l + k*LX1] * shut[i+j*LX1+l*LX1*LY1];
                    }
                    shw[ijk] += wijke;
                }
            }
        }
    }
}

