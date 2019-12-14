#ifdef NEED_CUDA

#include "header.h"

#include <assert.h>
#include <cuda_runtime.h>

dim3 blockDim_;
dim3 gridDim_;

dim3 blockDimZY;
dim3 gridDimZY;

dim3 blockDimYX;
dim3 gridDimYX;

dim3 blockDimXZ;
dim3 gridDimXZ;

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
    blockDim_ = blockDimZY = blockDimYX = blockDimXZ = dim3(8, 8, 8);
    gridDim_ = gridDimZY = gridDimYX = gridDimXZ = dim3(
        (grid_points[0] - 1) / blockDim_.x + 1,
        (grid_points[1] - 1) / blockDim_.y + 1,
        (grid_points[2] - 1) / blockDim_.z + 1
    );
    blockDimZY.x = gridDimZY.x = 1;
    blockDimYX.z = gridDimYX.z = 1;
    blockDimXZ.y = gridDimXZ.y = 1;
}

#endif
