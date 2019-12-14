#ifdef NEED_CUDA

#include "header.h"

#include <omp.h>
#include <assert.h>
#include <cuda_runtime.h>

__thread dim3 blockDim_;
__thread dim3 gridDim_;

__thread dim3 blockDimYZ;
__thread dim3 gridDimYZ;

__thread dim3 blockDimXY;
__thread dim3 gridDimXY;

__thread dim3 blockDimXZ;
__thread dim3 gridDimXZ;

__thread dim3 gridElems;
__thread dim3 gridOffset;

void cuda_preinit()
{
    omp_set_num_threads(cudaGetDeviceCount());
}

void cuda_init()
{
    int device = omp_get_thread_num();
    assert(cudaSuccess == cudaSetDevice(device));
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
        threadsPerZ * i
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
