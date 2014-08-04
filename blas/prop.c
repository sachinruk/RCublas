#include "CUDA_include/cuda.h"
#include "CUDA_include/cuda_runtime_api.h"
#include "Blas/prop.h"
#include "Blas/setdev.h"

char _prop (void){
	int i=0,devices=0, choose_device=0, compute_capability=0;
	int major=0, minor=0;
	CUdevice dev;
	//Properties compare;
	cuInit(0);
	
	/*if (devices>MAX_DEVICE_COUNT)
		;error in code*/

	if (cuDeviceGetCount(&devices)!=CUDA_SUCCESS){
		devices=0; /*safety measure probably not required*/
	}

	for (i=0; i<devices; i++){
		cuDeviceGet(&dev, i);
		cuDeviceComputeCapability(& major, & minor, dev);
		if (minor>compute_capability){/*compare compute capabities with previous device*/
			choose_device=i;
			compute_capability=minor;/*set new capabilit*/
		}
		//else if (prop[i].minor==compute_capability)
			//if prop[i].clockRate;/*need to set other condition for comparison-clockrate*/	
	}
	
	//cudaSetDevice(choose_device);
	set_dev(choose_device);
	
	if (compute_capability==DOUBLE_CAPABLE)
	return 1;
	else return 0;
	
}

