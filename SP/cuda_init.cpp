#ifdef NEED_CUDA

#include "header.h"

#include <iostream>
#include <omp.h>
#include <assert.h>
#include <cuda_runtime.h>

thread_local dim3 blockDim_;
thread_local dim3 gridDim_;

thread_local dim3 blockDimYZ;
thread_local dim3 gridDimYZ;

thread_local dim3 blockDimXY;
thread_local dim3 gridDimXY;

thread_local dim3 blockDimXZ;
thread_local dim3 gridDimXZ;

thread_local dim3 gridElems;
thread_local dim3 gridOffset;

__thread int device;
int device_count;

void cuda_preinit()
{
    CHK_CUDA_OK(cudaGetDeviceCount(&device_count));
    assert(device_count);
    omp_set_num_threads(std::min(omp_get_max_threads(), device_count));
    device_count = omp_get_max_threads();
}

void cuda_init()
{
    device = omp_get_thread_num();
#pragma omp critical
    std::cout << "Initializing device " << device + 1 << "/" << device_count << std::endl;
    CHK_CUDA_OK(cudaSetDevice(device));
}

void cuda_init_sizes()
{
    int n = omp_get_num_threads();
    int i = omp_get_thread_num();
    gridElems = dim3(
        grid_points[0],
        grid_points[1],
        (grid_points[2] - 1) / n + 1
    );
    gridOffset = dim3(
        0,
        0,
        gridElems.z * i
    );
    blockDim_ = blockDimYZ = blockDimXY = blockDimXZ = dim3(8, 8, 8);
    gridDim_ = gridDimYZ = gridDimXY = gridDimXZ = dim3(
        (gridElems.x - 1) / blockDim_.x + 1,
        (gridElems.y - 1) / blockDim_.y + 1,
        (gridElems.z - 1) / blockDim_.z + 1
    );
    blockDimYZ.x = gridDimYZ.x = 1;
    blockDimXY.z = gridDimXY.z = 1;
    blockDimXZ.y = gridDimXZ.y = 1;
}

#endif
