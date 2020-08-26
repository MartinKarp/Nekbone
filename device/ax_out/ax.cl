#define LX1 10
#define LY1 10
#define LZ1 10
__kernel void ax(__global double * restrict w,
                        __global const double * restrict p,
                        __global const double * restrict gxyz,
                        __global const double * restrict dxm1,
                        int N){
    
    const double shdxm1[LX1*LY1];
    for(unsigned j = 0; j < LX1; j++){
        for(unsigned i = 0; i < LX1; i++){
            shdxm1[i + j*LX1] = dxm1[i + j*LX1];    
        }
    }
    double shu[LX1*LY1*LZ1];
    double shur[LX1*LY1*LZ1];
    double shus[LX1*LY1*LZ1];
    double shut[LX1*LY1*LZ1];
    double shw[LX1*LY1*LZ1];
    for(unsigned e = 0; e < N/(LX1*LY1*LZ1); e++){
        int ele = e*LX1*LY1*LZ1;
        for(unsigned k=0; k<LZ1; ++k){
            for(unsigned j = 0; j < LX1; j++){
                for(unsigned i = 0; i < LX1; i++){
                    int ij = i + j*LX1;
                    int ijk = ij + k*LX1*LY1;
                    shu[ijk] = p[ijk + ele];
                }
            }
        }   
        for (unsigned k=0; k<LZ1; ++k){
            for(unsigned j = 0; j < LY1; j++){
                for(unsigned i = 0; i < LX1; i++){
                    int ij = i + j*LX1;
                    int ijk = ij + k*LX1*LY1;
                    double G00 = gxyz[0+6*ijk+ele*6];
                    double G01 = gxyz[1+6*ijk+ele*6];
                    double G02 = gxyz[2+6*ijk+ele*6];
                    double G11 = gxyz[3+6*ijk+ele*6];
                    double G12 = gxyz[4+6*ijk+ele*6];
                    double G22 = gxyz[5+6*ijk+ele*6];
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
                   shw[ijk + ele] = wijke;
                }
            }
        }
        for(unsigned k=0; k<LZ1; ++k){
            for(unsigned j = 0; j < LX1; j++){
                for(unsigned i = 0; i < LX1; i++){
                    int ij = i + j*LX1;
                    int ijk = ij + k*LX1*LY1;
                    w[ijk] = shw[ijk + ele];
                }
            }
        }
    }
}

