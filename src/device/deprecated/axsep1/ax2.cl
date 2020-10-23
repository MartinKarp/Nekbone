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
                    double G00 = gxyz[0+6*ijk+ele*6];
                    double G01 = gxyz[1+6*ijk+ele*6];
                    double G02 = gxyz[2+6*ijk+ele*6];
                    double G11 = gxyz[3+6*ijk+ele*6];
                    double G12 = gxyz[4+6*ijk+ele*6];
                    double G22 = gxyz[5+6*ijk+ele*6];
                    double rtmp = 0.0;
                    double stmp = 0.0;
                    double ttmp = 0.0;
                    double shft_r[31];
                    double shft_s[31];
                    double shft_t[31];
                    #pragma unroll
		    for (unsigned l = 0; l<31; l++){
                        shft_r[l] = 0.0;
                        shft_s[l] = 0.0;
                        shft_t[l] = 0.0;
                    }



		    for (unsigned l = 0; l<LX1; l++){
                      shft_r[30] = shft_r[0] + shdxm1[i+l*LX1] * shu[l+j*LX1 +k*LX1*LY1];
                      shft_s[30] = shft_s[0] + shdxm1[j+l*LX1] * shu[i+l*LX1 + k*LX1*LY1];
                      shft_t[30] = shft_t[0] + shdxm1[k+l*LX1] * shu[ij + l*LX1*LY1];
                      #pragma unroll
                      for (unsigned l2 = 0; l2 < 30; l2++){
                          shft_r[l2] = shft_r[l2 + 1];  
                          shft_s[l2] = shft_s[l2 + 1];  
                          shft_t[l2] = shft_t[l2 + 1];  
                      }
                    }
                    #pragma unroll
		    for (unsigned l = 0; l<31; l++){
                        rtmp += shft_r[l];
                        stmp += shft_s[l];
                        ttmp += shft_t[l];
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
        for (unsigned k=0; k<LZ1; ++k){
            for(unsigned j = 0; j < LY1; j++){
                for(unsigned i = 0; i < LX1; i++){
                    int ij = i + j*LX1;
                    int ijk = ij + k*LX1*LY1;
                    
                    double wijke = 0.0;
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

