#define LX1 10
#define LY1 10
#define LZ1 10
#include <stdio.h>
#include "nvmlPower.hpp"
__global__ void ax_cuda2_kernel(double* __restrict__ w, const double* __restrict__ u, const double* __restrict__ gxyz, const double* __restrict__ dxm1, const double* __restrict__ dxtm1){
/*      real, intent(out) :: w(lx1,ly1,lz1,lelt)
      real u(lx1,ly1,lz1,lelt)
      real ur  (lx1,ly1,lz1,lelt)
      real us  (lx1,ly1,lz1,lelt)
      real ut  (lx1,ly1,lz1,lelt)

      real gxyz(lx1,ly1,lz1,2*ldim,lelt)

      real, intent(in) :: dxm1(lx1,lx1)
      real, intent(in) :: dxtm1(lx1,lx1)*/

      double rtmp,stmp,ttmp,wijke;
      __shared__ double shdxm1[LX1*LY1];
      __shared__ double shu[LX1*LY1];
      __shared__ double shur[LX1*LY1];
      __shared__ double shus[LX1*LY1];
      double ru[LZ1];
      double rw[LZ1];
      double rut;
      double G00,G01,G02,G11,G12,G22;
      int l,e,i,j,k,ijk,ij,ele;

      e = blockIdx.x;
      j = threadIdx.y;
      i = threadIdx.x;
      ij = i + j*LX1;
      ele = e*LX1*LY1*LZ1;

      shdxm1[ij] = dxm1[ij];
      #pragma unroll
      for( k = 0; k < LZ1; ++k){
        ru[k] = u[ij + k*LX1*LY1 + ele];
        rw[k] = 0.0;
      }

// Perform the strided accesses.  Each thread in the block proceeds in
// lockstep.
      __syncthreads();
      #pragma unroll
      for (k=0; k<LZ1; ++k){
        ijk = ij + k*LX1*LY1; 
        G00 = gxyz[ijk+0*LX1*LY1*LZ1+ele*6];
        G01 = gxyz[ijk+1*LX1*LY1*LZ1+ele*6];
        G02 = gxyz[ijk+2*LX1*LY1*LZ1+ele*6]; 
        G11 = gxyz[ijk+3*LX1*LY1*LZ1+ele*6];
        G12 = gxyz[ijk+4*LX1*LY1*LZ1+ele*6];
        G22 = gxyz[ijk+5*LX1*LY1*LZ1+ele*6];
        ttmp = 0.0;
        shu[ij] = ru[k];
        for (l = 0; l<LX1; l++){
          ttmp += shdxm1[k+l*LX1] * ru[l];
        }
        __syncthreads();
 
        rtmp = 0.0;
        stmp = 0.0;
        #pragma unroll
        for (l = 0; l<LX1; l++){
          rtmp += shdxm1[i+l*LX1] * shu[l+j*LX1];
          stmp += shdxm1[j+l*LX1] * shu[i+l*LX1];
        }
        shur[ij] = G00*rtmp
                 + G01*stmp
                 + G02*ttmp;
        rut      = G02*rtmp
                 + G12*stmp 
                 + G22*ttmp;
        shus[ij] = G01*rtmp
                 + G11*stmp
                 + G12*ttmp;

      __syncthreads();

        wijke = 0.0;
        #pragma unroll
        for (l = 0; l<LX1; l++){
          wijke += shdxm1[l + i*LX1] * shur[l+j*LX1];
          rw[l] += shdxm1[k+l*LX1] * rut; 
          wijke += shdxm1[l + j*LX1] * shus[i+l*LX1];
        }
        rw[k] += wijke;
      }
      #pragma unroll
      for (k=0; k<LZ1; ++k){
        w[ij + k*LX1*LY1 + ele] = rw[k]; 
      }
}
extern "C" {
  void ax_cuda2_(double* __restrict__ w, const double* __restrict__ u, const double* __restrict__ gxyz,
		 const double* __restrict__ dxm1, const double* __restrict__ dxtm1, const int *nel){
    ax_cuda2_kernel<<<*nel,dim3(LX1,LY1,1)>>>(w, u, gxyz, dxm1, dxtm1);
  }
}
