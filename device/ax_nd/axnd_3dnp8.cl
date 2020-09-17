#define LX1 8
#define LY1 8
#define LZ1 8    

__attribute__((num_compute_units(1)))
__attribute__((num_simd_work_items(8)))
__attribute__((reqd_work_group_size(LX1,LY1,LZ1)))
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
    local double  shu[LX1*LY1*LZ1];
    local double  shur[LX1*LY1*LZ1];
    local double  shus[LX1*LY1*LZ1];
    local double  shut[LX1*LY1*LZ1];

    int e = get_group_id(0);
    //int ijk = get_local_id(0);
    //int i = ijk % LX1;
    //int j =  ((ijk - i) % (LX1*LY1))/LX1; 
    //int k = ijk - i - j* LX1;
    //int ij = i + j *LX1;
    int i = get_local_id(0);
    int j = get_local_id(1);
    int k = get_local_id(2);
    int ij = i + j*LX1;
    int ijk = ij + k*LX1*LY1; 
    int ele = e*LX1*LY1*LZ1;

    shdxm1[ij] = __burst_coalesced_load(&dxm1[ij]);
    shdxtm1[ij] = __burst_coalesced_load(&dxtm1[ij]);
    shu[ijk] = __burst_coalesced_load(&p[ijk]);   
// Pform the strided accesses.  Each thread in the block proceeds in
// lkstep.
    double G00 =__burst_coalesced_load(&g1[ijk+ele]);
    double G01 =__burst_coalesced_load(&g2[ijk+ele]);
    double G02 =__burst_coalesced_load(&g3[ijk+ele]); 
    double G11 =__burst_coalesced_load(&g4[ijk+ele]);
    double G12 =__burst_coalesced_load(&g5[ijk+ele]);
    double G22 =__burst_coalesced_load(&g6[ijk+ele]);
    barrier(CLK_LOCAL_MEM_FENCE); 
    double ttmp = 0.0;
    double rtmp = 0.0;
    double stmp = 0.0;
    #pragma unroll
    for (unsigned l = 0; l<LX1; l++){
        rtmp += shdxtm1[l+i*LX1] * shu[l+j*LX1 +k*LX1*LY1];
        stmp += shdxtm1[l+j*LX1] * shu[i+l*LX1 + k*LX1*LY1];
        ttmp += shdxtm1[l+k*LX1] * shu[ij + l*LX1*LY1];
    }
    shur[ijk] = G00*rtmp
             + G01*stmp
             + G02*ttmp;
    shus[ijk]= G01*rtmp
             + G11*stmp
             + G12*ttmp;
    shut[ijk] = G02*rtmp
             + G12*stmp 
             + G22*ttmp;
    barrier(CLK_LOCAL_MEM_FENCE); 
    double wijke = 0.0;
    #pragma unroll 
    for(unsigned l = 0; l<LX1; l++){
        wijke += shdxm1[l + i*LX1] * shur[l+j*LX1+k*LX1*LY1];
        wijke += shdxm1[l + j*LX1] * shus[i+l*LX1+k*LX1*LY1];
        wijke += shdxm1[l + k*LX1] * shut[i+j*LX1+l*LX1*LY1];
    }
    w[ijk + ele] = wijke;
}

