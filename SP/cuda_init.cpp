#include "header.h"
#include <assert.h>
#include <cuda_runtime.h>

dim3 blockDim_;
dim3 gridDim_;

dim3 blockDimYZ;
dim3 gridDimYZ;

dim3 blockDimXY;
dim3 gridDimXY;

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
    blockDim_ = blockDimYZ = blockDimXY = blockDimXZ = dim3(8, 8, 8);
    gridDim_ = gridDimYZ = gridDimXY = gridDimXZ = dim3(
        (grid_points[0] - 1) / blockDim_.x + 1,
        (grid_points[1] - 1) / blockDim_.y + 1,
        (grid_points[2] - 1) / blockDim_.z + 1
    );
    blockDimYZ.x = gridDimYZ.x = 1;
    blockDimXY.z = gridDimXY.z = 1;
    blockDimXZ.y = gridDimXZ.y = 1;
}
