
__attribute__((scheduler_target_fmax_mhz(300)))
__kernel void ax(__global float * restrict w,
                        __global const float * restrict p,
                        __global const float * restrict g1,
                        __global const float * restrict g2,
                        __global const float * restrict g3,
                        __global const float * restrict g4,
                        __global const float * restrict g5,
                        __global const float * restrict g6,
                        __global const float * restrict dxm1,
                        __global const float * restrict dxtm1,
                        int N){
    #pragma unroll 32
    for(unsigned ijk = 0; ijk < N; ijk += 1){
       w[ijk] = p[ijk] + g1[ijk]+ g2[ijk]+ g3[ijk]+ g4[ijk]+ g5[ijk]+ g6[ijk];
    }
}

