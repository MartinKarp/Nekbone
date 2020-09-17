#define LX1 10
#define LY1 10
#define LZ1 10
__attribute__((num_simd_work_items(2)))
__attribute__((num_compute_units(1)))
__attribute__((reqd_work_group_size(LX1,LY1,1)))
__kernel void ax(__global double * restrict w,
                        __global const double * restrict p,
                        __global const double * restrict g1,
                        __global const double * restrict g2,
                        __global const double * restrict g3,
                        __global const double * restrict g4,
                        __global const double * restrict g5,
                        __global const double * restrict g6,
                        __global const double * restrict dxm1,
                        __global const double * restrict dxtm1,
                        int N){

    local double  shdxm1[LX1*LY1];
    local double  shdxtm1[LX1*LY1];
    local double  shu[LX1*LY1];
    local double  shur[LX1*LY1];
    local double  shus[LX1*LY1];
    double __attribute__((register)) ru[LZ1];
    double __attribute__((register)) rw[LZ1];
    double rut;

    //int e = blockIdx.x;
    //int j = threadIdx.y;
    //int i = threadIdx.x;
    //int ij = i + j*LX1;
    //int ele = e*LX1*LY1*LZ1;
    int e = get_group_id(0);
    int i = get_local_id(0);
    int j = get_local_id(1);
    int ij = i + j*LX1;
    int ele = e*LX1*LY1*LZ1;

    shdxm1[ij] = dxm1[ij];
    shdxtm1[ij] = dxm1[ij];
    #pragma unroll
    for(unsigned  k = 0; k < LZ1; ++k){
      ru[k] = __burst_coalesced_load(&p[ij + k*LX1*LY1 + ele]);
      rw[k] = 0.0;
    }

// Pform the strided accesses.  Each thread in the block proceeds in
// lkstep.
    barrier(CLK_LOCAL_MEM_FENCE); 
    for (unsigned k=0; k<LZ1; ++k){
        int ijk = ij + k*LX1*LY1; 
        double G00 =__burst_coalesced_load(&g1[ijk+ele]);
        double G01 =__burst_coalesced_load(&g2[ijk+ele]);
        double G02 =__burst_coalesced_load(&g3[ijk+ele]); 
        double G11 =__burst_coalesced_load(&g4[ijk+ele]);
        double G12 =__burst_coalesced_load(&g5[ijk+ele]);
        double G22 =__burst_coalesced_load(&g6[ijk+ele]);
        double ttmp = 0.0;
        shu[ij] = ru[k];
        #pragma unroll
        for (unsigned l = 0; l<LX1; l++){
          ttmp += shdxtm1[l+k*LX1] * ru[l];
        }
        barrier(CLK_LOCAL_MEM_FENCE); 
        double rtmp = 0.0;
        double stmp = 0.0;
        #pragma unroll
        for (unsigned l = 0; l<LX1; l++){
          rtmp += shdxtm1[l+i*LX1] * shu[l+j*LX1];
          stmp += shdxtm1[l+j*LX1] * shu[i+l*LX1];
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

        barrier(CLK_LOCAL_MEM_FENCE); 

        double wijke = 0.0;
        #pragma unroll
        for (unsigned l = 0; l<LX1; l++){
          wijke += shdxm1[l + i*LX1] * shur[l+j*LX1];
          rw[l] += shdxtm1[l+k*LX1] * rut; 
          wijke += shdxm1[l + j*LX1] * shus[i+l*LX1];
        }
        rw[k] += wijke;
    }
    #pragma unroll
    for (unsigned k=0; k<LZ1; ++k){
      w[ij + k*LX1*LY1 + ele] = rw[k]; 
    }
}
