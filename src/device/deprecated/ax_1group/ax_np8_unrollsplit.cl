#define LX1 8 
#define LY1 8 
#define LZ1 8 

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
        double r[LX1*LY1*LZ1];
        double s[LX1*LY1*LZ1];
        double t[LX1*LY1*LZ1];
        double shw[LX1*LY1*LZ1];
        double shg1[LX1*LY1*LZ1];
        double shg2[LX1*LY1*LZ1];
        double shg3[LX1*LY1*LZ1];
        double shg4[LX1*LY1*LZ1];
        double shg5[LX1*LY1*LZ1];
        double shg6[LX1*LY1*LZ1];
        double shu[LX1*LY1*LZ1];
        double shdxm1[128];
        double shdxtm1[128];
        #pragma unroll 32
        for(unsigned ij=0; ij<LX1*LY1; ++ij){
            shdxm1[ij] = dxm1[ij];
            shdxtm1[ij] = dxtm1[ij];
        }

        #pragma unroll 32
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
        for (unsigned k=0; k<LZ1; ++k){
            for(unsigned j = 0; j < LY1; j++){
                #pragma unroll
                for(unsigned i = 0; i < LX1; i++){
                    int ij = i + j*LX1;
                    int ijk = ij + k*LX1*LY1;
                    double rtmp = 0.0;
                    double stmp = 0.0;
                    double ttmp = 0.0;
                    #pragma unroll 
                    for (unsigned l = 0; l<LX1; l++){
                      rtmp += shdxtm1[l+i*LX1] * shu[l+j*LX1 +k*LX1*LY1];
                      stmp += shdxtm1[l+j*LX1] * shu[i+l*LX1 + k*LX1*LY1];
                      ttmp += shdxtm1[l+k*LX1] * shu[ij + l*LX1*LY1];
                    }
                    r[ijk] = rtmp;
                    s[ijk] = stmp;
                    t[ijk] = ttmp;                   
                }
            }
        }
        #pragma unroll 32 
        for(unsigned ijk=0; ijk<LX1*LY1*LZ1; ++ijk){
            double G00 = shg1[ijk];
            double G01 = shg2[ijk];
            double G02 = shg3[ijk];
            double G11 = shg4[ijk];
            double G12 = shg5[ijk];
            double G22 = shg6[ijk];
            
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
        #pragma loop_coalesce
        for (unsigned k=0; k<LZ1; ++k){
            for(unsigned j = 0; j < LY1; j++){
                #pragma unroll
                for(unsigned i = 0; i < LX1; i++){
                    int ij = i + j*LX1;
                    int ijk = ij + k*LX1*LY1;
                    
                    double wijke = 0.0;
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

        #pragma unroll 32 
        for(unsigned ijk=0; ijk<LX1*LY1*LZ1; ++ijk)
            w[ijk + ele] = shw[ijk];
    }
}

