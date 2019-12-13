#ifdef NEED_CUDA

#include "header.h"

#include <assert.h>
#include <cuda_runtime.h>

dim3 blockDim_;
dim3 gridDim_;

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

void cuda_init_sizes()
{
    blockDim_ = dim3(8, 8, 8);
    gridDim_ = dim3(
        (grid_points[0] - 1) / blockDim_.x + 1,
        (grid_points[1] - 1) / blockDim_.y + 1,
        (grid_points[2] - 1) / blockDim_.z + 1
    );
}

#endif
