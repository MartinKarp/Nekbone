#define LX1 8
#define LY1 8
#define LZ1 8

__kernel void ax(__global double * restrict w,
                        __global const double * restrict p,
                        __global const double * restrict g1,
                        __global const double * restrict g2,
                        __global const double * restrict g3,
                        __global const double * restrict g4,
                        __global const double * restrict g5,
                        __global const double * restrict g6,
                        __global const double * restrict dxm1,
                        int N){
   double temp = 0.0;
   temp += p[0] * g1[0];
   temp += p[1] * g1[1];
   temp += p[2] * g1[2];
   temp += p[3] * g1[3];
   w[0] = temp;
}

