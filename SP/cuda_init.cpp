#ifdef NEED_CUDA

#include "header.h"

#include <assert.h>
#include <cuda_runtime.h>

dim3 blockDim;
dim3 gridDim;

void cuda_init()
{
    int device = 0;
    assert(cudaSuccess == cudaSetDeviceFlags(cudaDeviceMapHost));
    assert(cudaSuccess == cudaSetDevice(device));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    assert(deviceProp.canMapHostMemory);
    assert(deviceProp.unifiedAddressing);
}

#endif
