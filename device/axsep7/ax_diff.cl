#define LX1 10
#define LY1 10
#define LZ1 10

__kernel void ax(__global double * restrict w,
                        __global const double * restrict p,
                        __global const double * restrict gxyz,
                        __global const double * restrict dxm1,
                        int N){


    for(unsigned ele = 0; ele < N; ele += LX1*LY1*LZ1){
        double shur[LX1*LY1*LZ1];
        double shus[LX1*LY1*LZ1];
        double shut[LX1*LY1*LZ1];
        double r[LX1*LY1*LZ1];
        double s[LX1*LY1*LZ1];
        double t[LX1*LY1*LZ1];
        double shw[LX1*LY1*LZ1];
    	double shu[LX1*LY1*LZ1];
        double shdxm1[LX1*LY1];
        #pragma unroll
        for(unsigned ij=0; ij<LX1*LY1; ++ij)
            shdxm1[ij] = dxm1[ij];
        
        #pragma unroll 32
        for(unsigned ijk=0; ijk<LX1*LY1*LZ1; ++ijk){
            shu[ijk] = p[ijk + ele];
            //shg[0+6*ijk] = gxyz[0+6*ijk+ele*6];
            //shg[1+6*ijk] = gxyz[1+6*ijk+ele*6];
            //shg[2+6*ijk] = gxyz[2+6*ijk+ele*6];
            //shg[3+6*ijk] = gxyz[3+6*ijk+ele*6];
            //shg[4+6*ijk] = gxyz[4+6*ijk+ele*6];
            //shg[5+6*ijk] = gxyz[5+6*ijk+ele*6];
            
        }
        #pragma loop_coalesce
        for (unsigned k=0; k<LZ1; ++k){
            for(unsigned j = 0; j < LY1; j++){
                for(unsigned i = 0; i < LX1; i++){
                    int ij = i + j*LX1;
                    int ijk = ij + k*LX1*LY1;
                    double rtmp = 0.0;
                    double stmp = 0.0;
                    double ttmp = 0.0;
                    #pragma unroll
                    for (unsigned l = 0; l<LX1; l++){
                      rtmp += shdxm1[i+l*LX1] * shu[l+j*LX1 +k*LX1*LY1];
                      stmp += shdxm1[j+l*LX1] * shu[i+l*LX1 + k*LX1*LY1];
                      ttmp += shdxm1[k+l*LX1] * shu[ij + l*LX1*LY1];
                    }
                    r[ijk] = rtmp;
                    s[ijk] = stmp;
                    t[ijk] = ttmp;
                }
            }
        }
        #pragma loop_coalesce
        #pragma unroll 2
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
            }
        }
        #pragma loop_coalesce
        for (unsigned k=0; k<LZ1; ++k){
            for(unsigned j = 0; j < LY1; j++){
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

