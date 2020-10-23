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
__kernel void post_ax(__global double * restrict x, 
                         __global const double * restrict p,
                         __global double * restrict r,
                         __global double * restrict w,
                         __global const double * restrict c,
                         __global double * restrict rtr,
                         __global const double * restrict rtz1,
                         __global const double * restrict cmask,
                         int N)
{  
    for(unsigned i = 0; i < N; ++i){
        w[i] += 0.1 * p[i];
    }

    int lim = cmask[1];
    for(unsigned i = 2; i < lim+2; ++i){
	        int k = cmask[i];
            w[k-1] = 0.0;
    }


    //  call add2s2(x,p,alpha,n)                                        ! 2n
    //  call add2s2(r,w,alphm,n)                                        ! 2n
    double rtr_copies[M];
    double res = 0.0;

//  pap=glsc3(w,c,p,n)   
    
    for(int i = 0; i < M; ++i) 
        rtr_copies[i] = 0;
    for( int i = 0; i < N; ++i){
        double cur = rtr_copies[M-1] +  w[i]*p[i]*c[i];
        #pragma unroll
        for(unsigned j = M-1; j>0; j--){
            rtr_copies[j] = rtr_copies[j-1];
        }
        rtr_copies[0] = cur; 
    }
    #pragma unroll
    for(unsigned i = 0; i < M; i++){
        res += rtr_copies[i];
    }

    double pap = res;
    double alpha = rtz1[0]/pap;
    double alphm = -1.*alpha; 
    res = 0.0;

    for(int i = 0; i < M; ++i) 
        rtr_copies[i] = 0;
    for( int i = 0; i < N; ++i){
    	x[i] = x[i] + alpha * p[i];
    	r[i] = r[i] + alphm * w[i];
    }   
    for( int i = 0; i < N; ++i){
        double cur = rtr_copies[M-1] +  r[i]*r[i]*c[i];
        #pragma unroll
        for(unsigned j = M-1; j>0; j--){
            rtr_copies[j] = rtr_copies[j-1];
        }
        rtr_copies[0] = cur;
    }
    #pragma unroll
    for(unsigned i = 0; i < M; i++){
        res += rtr_copies[i];
    }
    * rtr = res;
  }
__kernel void pre_dssum(__global double * restrict z,
                        __global double * restrict r,
                        __global double * restrict c,
                        __global double * restrict w,
                        __global double * restrict p,
                        __global double * restrict rtz1,
                        __global double * restrict gxyz,
                        __global double * restrict dxm1,
                        __global double * restrict dxtm1,
                        __global double * restrict ur, 
                        __global double * restrict us, 
                        __global double * restrict ut, 
                        __global double * restrict wk, 
                        int N,
                        int iter){
//    call solveM(z,r,n)    ! preconditioner here

//    rtz2=rtz1                                                       ! OPS
//    rtz1=glsc3(r,c,z,n)   ! parallel weighted inner product r^T C z ! 3n

//    beta = rtz1/rtz2
//    if (iter.eq.1) beta=0.0
//    call add2s1(p,z,beta,n)                                         ! 2n
  
    double rtr_copies[M];
    double rtz2 = rtz1[0]; 
    double res = 0.0;
    // double ur[M*M*M];
    // double us[M*M*M];
    // double ut[M*M*M];
    // double wk[M*M*M];
    for( int i = 0; i < N; ++i){
    	z[i] = r[i];
    }
    for(int i = 0; i < M; ++i) 
    rtr_copies[i] = 0;
    
    for( int i = 0; i < N; ++i){
    	double cur = rtr_copies[M-1] +  r[i]*c[i]*z[i];
    	#pragma unroll
        for(unsigned j = M-1; j>0; j--){
            rtr_copies[j] = rtr_copies[j-1];
        }
    	rtr_copies[0] = cur; 
    }
    
    #pragma unroll
    for(unsigned i = 0; i < M; i++){
        res += rtr_copies[i];
    }
    * rtz1 = res;
    double beta = res/rtz2;
    
    if (iter == 1){
        beta = 0.0;
    }
    for( int i = 0; i < N; ++i){
        p[i] = z[i] + beta * p[i];
        //printf("%.10e \n",p[i]);
    }

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


//                        __global double * restrict g,
//                        __global double * restrict ur, 
//                        __global double * restrict us, 
//                        __global double * restrict ut, 
//                        __global double * restrict wk,





//    for(int e = 0;e < nelt; ++e){
//        int nel = e * m1*m1*m1;
//
//    //  call mxm(D ,m1,u,m1,ur,m2)
//    //  ur = dxm1 * p
//        for(unsigned i = 0; i < m2; ++i){
//            for(unsigned j = 0; j < m1; ++j){
//                double temp = 0.0;
//                #pragma unroll
//                for(unsigned l = 0; l < m1; ++l){
//                    temp += dxm1[l+i*m1] * p[j + l*m1 + nel]; 
//                }
//                ur[j+i*m1] = temp;
//            }
//        }
//        
//
//    //  do k=0,n
//    //     call mxm(u(0,0,k),m1,Dt,m1,us(0,0,k),m1)
//    //  enddo
//        for(unsigned k = 0; k < m1){
//            int k_indx = k * m1*m1*m1;
//            for(unsigned i = 0; i < m1; ++i){
//                for(unsigned j = 0; j < m1; ++j){
//                    double temp = 0.0;
//                    #pragma unroll
//                    for(unsigned l = 0; l < m1; ++l){
//                        temp += p[l + i*m1 + k_indx + nel] * dxtm1[j +l*m1];
//                    }
//                    us[j+i*m1 + k_indx] = temp;
//                }
//            }
//        }
//
//    //  call mxm(u,m2,Dt,m1,ut,m1)
//        for(unsigned i = 0; i < m1; ++i){
//            for(unsigned j = 0; j < m1; ++j){
//                double temp = 0.0;
//                #pragma unroll
//                for(unsigned l = 0; l < m2; ++l){
//                    temp += p[l+i*m1] * p[j + l*m1 + nel]; 
//                }
//                ut[j+i*m1] = temp;
//            }
//        }
//    }
