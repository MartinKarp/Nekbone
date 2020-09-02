#define LX1 10
#define LY1 10
#define LZ1 10
__kernel void ax(__global double * restrict w,
                        __global const double * restrict p,
                        __global const double * restrict gxyz,
                        __global const double * restrict dxm1,
                        int N){
    
    for(unsigned e = 0; e < N/(LX1*LY1*LZ1); e++){
        int ele = e*LX1*LY1*LZ1;
    	double shu[LX1*LY1*LZ1];
        #pragma unroll 32
	for(unsigned ijk=0; ijk<LX1*LY1*LZ1; ++ijk){
            shu[ijk] = p[ijk + ele];
        }   
        #pragma unroll 32
        for(unsigned ijk=0; ijk<LX1*LY1*LZ1; ++ijk){
            w[ijk + ele] = shu[ijk];
        }
    }
}

