#include <stdlib.h>
#include <malloc.h>
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "mem_trunc.h"

static int max_threads=0;
static float *temp=NULL;


void init_dim (void){
	CUdevice dev=0;
	struct CUdevprop* prop=NULL;
	int device=0, maxThreadsPerBlock=0;

	cuInit(0);
	cuDeviceGet(&dev, device);
	cuDeviceGetAttribute(&maxThreadsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev);
	max_threads=maxThreadsPerBlock;
	temp=(float *)malloc(max_threads*sizeof(float));
}

void truncate (float *GPU, double *CPU, int elements){
	int i=0, offset=0;
	

	while (elements){
		for (i=0;i<max_threads && i<elements;i++){
			temp[i]=(float)CPU[i];
		}
		cuMemcpyHtoD((CUdeviceptr)(GPU+offset), temp, sizeof(float)*imin(elements,max_threads));
		offset+=max_threads;/*offset for the GPU pointer*/
		elements=imax(0,(elements-max_threads));
	}
}

