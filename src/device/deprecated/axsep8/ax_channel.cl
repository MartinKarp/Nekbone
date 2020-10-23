#define LX1 10
#define LY1 10
#define LZ1 10

typedef struct
{
	double data[6];
} G_WIDTH;

#pragma OPENCL EXTENSION cl_intel_channels : enable
channel G_WIDTH g_chan __attribute__((depth(16)));

__kernel void ax(__global double * restrict w,
                        __global const double * restrict p,
                        __global const double * restrict gxyz,
                        __global const double * restrict dxm1,
                        int N){


    for(unsigned ele = 0; ele < N; ele += LX1*LY1*LZ1){
        double shur[LX1*LY1*LZ1];
        double shus[LX1*LY1*LZ1];
        double shut[LX1*LY1*LZ1];
        double shw[LX1*LY1*LZ1];
    	double shu[LX1*LY1*LZ1];
        double shdxm1[LX1*LY1];
        #pragma unroll
        for(unsigned ij=0; ij<LX1*LY1; ++ij)
            shdxm1[ij] = dxm1[ij];
        
        #pragma unroll 32
        for(unsigned ijk=0; ijk<LX1*LY1*LZ1; ++ijk){
            shu[ijk] = p[ijk + ele];
        }
        #pragma loop_coalesce 
        for (unsigned k=0; k<LZ1; ++k){
            for(unsigned j = 0; j < LY1; j++){
                for(unsigned i = 0; i < LX1; i++){
                    int ij = i + j*LX1;
                    int ijk = ij + k*LX1*LY1;
                    G_WIDTH g6 = read_channel_intel(g_chan);
                    double G00 = g6.data[0];
                    double G01 = g6.data[1];
                    double G02 = g6.data[2];
                    double G11 = g6.data[3];
                    double G12 = g6.data[4];
                    double G22 = g6.data[5];
                    double rtmp = 0.0;
                    double stmp = 0.0;
                    double ttmp = 0.0;
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
                    shw[ijk] = wijke;
                }
            }
        }

        #pragma unroll 32
        for(unsigned ijk=0; ijk<LX1*LY1*LZ1; ++ijk)
            w[ijk + ele] = shw[ijk];
    }
}

__kernel void producer(__global const double * restrict gxyz, int N){
 
    for(int i = 0; i<N; i++){
        G_WIDTH g6;
        g6.data[0] = gxyz[0 + 6*i];
        g6.data[1] = gxyz[1 + 6*i];
        g6.data[2] = gxyz[2 + 6*i];
        g6.data[3] = gxyz[3 + 6*i];
        g6.data[4] = gxyz[4 + 6*i];
        g6.data[5] = gxyz[5 + 6*i];
        write_channel_intel(g_chan,g6);
    }    
}
