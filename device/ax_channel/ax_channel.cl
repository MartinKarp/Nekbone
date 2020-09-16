#define LX1 10
#define LY1 10
#define LZ1 10

typedef struct
{
	double data[1024];
} E_SIZE;

#pragma OPENCL EXTENSION cl_intel_channels : enable
channel E_SIZE g1_ele __attribute__((depth(16)));
channel E_SIZE g2_ele __attribute__((depth(16)));
channel E_SIZE g3_ele __attribute__((depth(16)));
channel E_SIZE g4_ele __attribute__((depth(16)));
channel E_SIZE g5_ele __attribute__((depth(16)));
channel E_SIZE g6_ele __attribute__((depth(16)));
channel E_SIZE u_ele __attribute__((depth(16)));
channel E_SIZE w_ele __attribute__((depth(16)));

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

    double shdxm1[LX1*LY1];
    #pragma unroll
    for(unsigned ij=0; ij<LX1*LY1; ++ij)
        shdxm1[ij] = dxm1[ij];

    for(unsigned ele = 0; ele < N; ele += LX1*LY1*LZ1){
        double shur[LX1*LY1*LZ1]; 
        double shus[LX1*LY1*LZ1];
        double shut[LX1*LY1*LZ1];
        E_SIZE chu, chw, chg1, chg2, chg3, chg4, chg5, chg6;
        chu = read_channel_intel(u_ele);
        chg1 = read_channel_intel(g1_ele);
        chg2 = read_channel_intel(g2_ele);
        chg3 = read_channel_intel(g3_ele);
        chg4 = read_channel_intel(g4_ele);
        chg5 = read_channel_intel(g5_ele);
        chg6 = read_channel_intel(g6_ele);
        #pragma loop_coalesce 
        for(unsigned k = 0; k<LZ1; ++k){
            for(unsigned j = 0; j < LY1; j++){
                for(unsigned i = 0; i < LX1; i++){
                    int ij = i + j*LX1;
                    int ijk = ij + k*LX1*LY1;
                    double G00 = chg1.data[ijk];
                    double G01 = chg2.data[ijk];
                    double G02 = chg3.data[ijk];
                    double G11 = chg4.data[ijk];
                    double G12 = chg5.data[ijk];
                    double G22 = chg6.data[ijk];
                    double rtmp = 0.0;
                    double stmp = 0.0;
                    double ttmp = 0.0;
                    for (unsigned l = 0; l<LX1; l++){
                      rtmp += shdxm1[i+l*LX1] * chu.data[l+j*LX1 +k*LX1*LY1];
                      stmp += shdxm1[j+l*LX1] * chu.data[i+l*LX1 + k*LX1*LY1];
                      ttmp += shdxm1[k+l*LX1] * chu.data[ij + l*LX1*LY1];
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
                }
            }
        }
        #pragma loop_coalesce 
        for (unsigned k=0; k<LZ1; ++k){
            for(unsigned j = 0; j < LY1; j++){
                for(unsigned i = 0; i < LX1; i++){
                    int ij = i + j*LX1;
                    int ijk = ij + k*LX1*LY1;
                    
                    double wijke = 0.0;
                    for(unsigned l = 0; l<LX1; l++){
                        wijke += shdxm1[l + i*LX1] * shur[l+j*LX1+k*LX1*LY1];
                        wijke += shdxm1[l + j*LX1] * shus[i+l*LX1+k*LX1*LY1];
                        wijke += shdxm1[l + k*LX1] * shut[i+j*LX1+l*LX1*LY1];
                    }
                    chw.data[ijk] = wijke;
                }
            }
        }
        write_channel_intel(w_ele,chw);
    }
}

__attribute__((scheduler_target_fmax_mhz(300)))
__kernel void producer(__global double * restrict w,
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
    for(int e = 0;  e<N; e+= LX1*LY1*LZ1){
        E_SIZE chu,chg1, chg2, chg3, chg4, chg5, chg6;
        #pragma unroll 32
        for(int i = 0; i<(LX1*LY1*LZ1); i++){
            chu.data[i] = p[i+e];
            chg1.data[i] = g1[i+e];
            chg2.data[i] = g2[i+e];
            chg3.data[i] = g3[i+e];
            chg4.data[i] = g4[i+e];
            chg5.data[i] = g5[i+e];
            chg6.data[i] = g6[i+e];
        }    
        write_channel_intel(u_ele,chu);
        write_channel_intel(g1_ele,chg1);
        write_channel_intel(g2_ele,chg2);
        write_channel_intel(g3_ele,chg3);
        write_channel_intel(g4_ele,chg4);
        write_channel_intel(g5_ele,chg5);
        write_channel_intel(g6_ele,chg6);
    }
}

__attribute__((scheduler_target_fmax_mhz(300)))
__kernel void writer(__global double * restrict w,
                     int N){
    for(int e = 0;  e<N; e+= LX1*LY1*LZ1){;
        E_SIZE out;
        out = read_channel_intel(w_ele);
        #pragma unroll 32
        for(int i = 0; i<(LX1*LY1*LZ1); i++){
            w[i+e] = out.data[i];
        }   
    } 
}


