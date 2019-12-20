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

void init_()
{
    assert(cudaSuccess == cudaSetDevice(0));
}

void init_grid_()
{
    blockDim_ = blockDimYZ = blockDimXY = blockDimXZ = dim3(32, 4, 4);
    gridDim_ = gridDimYZ = gridDimXY = gridDimXZ = dim3(
        (grid_points[0] - 1) / blockDim_.x + 1,
        (grid_points[1] - 1) / blockDim_.y + 1,
        (grid_points[2] - 1) / blockDim_.z + 1
    );
    blockDimYZ.x = gridDimYZ.x = 1;
    blockDimYZ.y *= 8;
    blockDimYZ.z *= 4;
    gridDimYZ.y = (gridDimYZ.y - 1) / 8 + 1;
    gridDimYZ.z = (gridDimYZ.z - 1) / 4 + 1;

    blockDimXY.z = gridDimXY.z = 1;
    blockDimXZ.y = gridDimXZ.y = 1;
}
