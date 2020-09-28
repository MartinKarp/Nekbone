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

    //printf("%.8f\n",shdxm1[5]);
    for(unsigned e = 0; e < N/(LX1*LY1*LZ1); e++){
        double shu[LX1*LY1*LZ1];
        double shur[LX1*LY1*LZ1];
        double shus[LX1*LY1*LZ1];
        double shut[LX1*LY1*LZ1];
        double shdxm1[LX1*LY1];
        double shdxtm1[LX1*LY1];

        for(unsigned ij=0; ij<LX1*LY1; ++ij){
            shdxm1[ij] = dxm1[ij];
            shdxtm1[ij] = dxtm1[ij];
        }
        printf("hej"); 
        int ele = e*LX1*LY1*LZ1;
        for(unsigned k=0; k<LZ1; ++k){
            for(unsigned j = 0; j < LX1; j++){
                for(unsigned i = 0; i < LX1; i++){
                    int ij = i + j*LX1;
                    int ijk = ij + k*LX1*LY1;
                    shu[ijk] = p[ijk + ele];
                    //printf("lel %.10e \n",p[ijk + ele]);
                }
            }
        }   
        //printf("%f %f %f %f %f %f\n",gxyz[N*6-6],gxyz[N*6-5],gxyz[N*6-4],gxyz[N*6-3],gxyz[N*6-2],gxyz[N*6-1]);
        for (unsigned k=0; k<LZ1; ++k){
            for(unsigned j = 0; j < LY1; j++){
                for(unsigned i = 0; i < LX1; i++){
                    int ij = i + j*LX1;
                    int ijk = ij + k*LX1*LY1;
                    double G00 = g1[ijk+ele];
                    double G01 = g2[ijk+ele];
                    double G02 = g3[ijk+ele];
                    double G11 = g4[ijk+ele];
                    double G12 = g5[ijk+ele];
                    double G22 = g6[ijk+ele];
                    double rtmp = 0.0;
                    double stmp = 0.0;
                    double ttmp = 0.0;
                    for (unsigned l = 0; l<LX1; l++){
                      rtmp += shdxm1[i+l*LX1] * shu[l+j*LX1 +k*LX1*LY1];
                      stmp += shdxm1[j+l*LX1] * shu[i+l*LX1 + k*LX1*LY1];
                      ttmp += shdxm1[k+l*LX1] * shu[ij + l*LX1*LY1];
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
                    //printf("%.10e %.10e %.10e %.10e\n",shur[ijk],rtmp,G00, gxyz[6*ijk+ele*6]);
                }
            }
        }
        for (unsigned k=0; k<LZ1; ++k){
            for(unsigned j = 0; j < LY1; j++){
                for(unsigned i = 0; i < LX1; i++){
                    int ij = i + j*LX1;
                    int ijk = ij + k*LX1*LY1;
                    double wijke = 0.0;
                    for (unsigned l = 0; l<LX1; l++){
                      wijke += shdxm1[l + i*LX1] * shur[l+j*LX1+k*LX1*LY1];
                      wijke += shdxm1[l + j*LX1] * shus[i+l*LX1+k*LX1*LY1];
                      wijke += shdxm1[l + k*LX1] * shut[i+j*LX1+l*LX1*LY1];
                    }
                    //printf("%f\n",wijke-w[ijk+ele]);
                    w[ijk + ele] = wijke;
                }
            }
        }
    }
}
