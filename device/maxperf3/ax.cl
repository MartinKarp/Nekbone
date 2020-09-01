#define LX1 10
#define LY1 10
#define LZ1 10
__kernel void ax(__global double * restrict w,
                        __global const double * restrict p,
                        __global const double * restrict gxyz,
                        __global const double * restrict u,
                        int N){
    
    for(unsigned ele = 0; ele < N; ele +=LX1*LY1*LZ1){
        #pragma unroll 32
	for(unsigned ijk=0; ijk<LX1*LY1*LZ1; ++ijk){
            w[ijk + ele] =  p[ijk + ele];
        }   
    }
}

