#include "aocl_mmd.h"
#include <stddef.h>
#include <stdio.h>
#include "CL/opencl.h"
#include "CL/cl_ext.h"
void power_(cl_platform_id * platform_ids, cl_device_id * device_ids)
{
	typedef void* (*aocl_mmd_card_info_fn_t)(const char*, aocl_mmd_info_t, size_t, void*, size_t* );
	typedef void* (*clGetBoardExtensionFunctionAddressIntelFPGA_fn)(const char*,cl_device_id device );
	aocl_mmd_card_info_fn_t aocl_mmd_card_info_fn = NULL;
	float power;
	size_t returnedSize;
	
	clGetBoardExtensionFunctionAddressIntelFPGA_fn clGetBoardExtensionFunctionAddressIntelFPGA = NULL;
	void *tempPointer = NULL;
	clGetBoardExtensionFunctionAddressIntelFPGA =
	(clGetBoardExtensionFunctionAddressIntelFPGA_fn)clGetExtensionFunctionAddressForPlatform(platform_ids[1],
	"clGetBoardExtensionFunctionAddressIntelFPGA");
	
	
	if (clGetBoardExtensionFunctionAddressIntelFPGA == NULL)
	{
		printf ("Failed to get clGetBoardExtensionFunctionAddressIntelFPGA\n");
	}
	tempPointer = NULL;
	tempPointer = clGetBoardExtensionFunctionAddressIntelFPGA("aocl_mmd_card_info",device_ids[1]);
	aocl_mmd_card_info_fn =
	(aocl_mmd_card_info_fn_t)tempPointer;
	if (aocl_mmd_card_info_fn == NULL )
	{
	printf ("Failed to get aocl_mmd_card_info_fn address\n");
	}
	//Note aclbitt_s10_pcie0 is the card name string
	aocl_mmd_card_info_fn("aclbitt_s10_pcie0", AOCL_MMD_CONCURRENT_READS, sizeof(float),(void*) &power,&returnedSize);
	printf("returnedSize = %u, ", returnedSize);
        printf("Power = %f W\n", power);
}
