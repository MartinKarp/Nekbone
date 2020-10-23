#define LX1 10
#define LY1 10
#define LZ1 10
__kernel void ax(__global double * restrict w,
                        __global const double * restrict p,
                        __global const double * restrict gxyz,
                        __global const double * restrict u,
                        int N){
    #pragma unroll 32 
    for( unsigned i = 0; i < N; i++){
	w[i] = p[i] + gxyz[i] + u[i];
    }
}

