#define LX1 10
#define LY1 10
#define LZ1 10
#define M 12
// Copyright (C) 2013-2018 Altera Corporation, San Jose, California, USA. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the Software
// without restriction, including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to
// whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
// 
// This agreement shall be governed in all respects by the laws of the State of California and
// by the laws of the United States of America.

 // ACL kernel for after Ax subroutine
__kernel void ax(__global double * restrict w,
                        __global double * restrict p,
                        __global double * restrict gxyz,
                        __global double * restrict dxm1,
                        int N,
                        int iter){
  
    //The Ax subroutine until we call dssum.
    const int m1 = M;
    const int m2 = m1 * m1;
    double shdxm1[LX1*LY1];
    for(unsigned j = 0; j < LX1; j++){
        for(unsigned i = 0; i < LX1; i++){
            shdxm1[i + j*LX1] = dxm1[i + j*LX1];    
        }
    }
    
    //printf("%.8f\n",shdxm1[5]);
    for(unsigned e = 0; e < N/(LX1*LY1*LZ1); e++){
        double shu[LX1*LY1*LZ1];
        double shur[LX1*LY1*LZ1];
        double shus[LX1*LY1*LZ1];
        double shut[LX1*LY1*LZ1];
        
        int ele = e*LX1*LY1*LZ1;
        for(unsigned k=0; k<LZ1; ++k){
            for(unsigned j = 0; j < LX1; j++){
                #pragma unroll
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
                    double G00 = gxyz[0+6*ijk+ele*6];
                    double G01 = gxyz[1+6*ijk+ele*6];
                    double G02 = gxyz[2+6*ijk+ele*6];
                    double G11 = gxyz[3+6*ijk+ele*6];
                    double G12 = gxyz[4+6*ijk+ele*6];
                    double G22 = gxyz[5+6*ijk+ele*6];
                    double rtmp = 0.0;
                    double stmp = 0.0;
                    double ttmp = 0.0;
                    #pragma unroll
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
                #pragma unroll 2 
                for(unsigned i = 0; i < LX1; i++){
                    int ij = i + j*LX1;
                    int ijk = ij + k*LX1*LY1;
                    double wijke = 0.0;
                    #pragma unroll
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

