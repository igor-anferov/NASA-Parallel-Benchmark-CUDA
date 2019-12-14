#ifdef NEED_CUDA

#include "header.h"

#include <omp.h>
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

#endif
