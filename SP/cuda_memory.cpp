#include <assert.h>
#include <cuda_runtime.h>

#include "header.h"

__thread int *dev_grid_points/*[3]*/;
__thread double (*dev_u      )/*[KMAX]*/[JMAXP+1][IMAXP+1][5];
__thread double (*dev_us     )/*[KMAX]*/[JMAXP+1][IMAXP+1];
__thread double (*dev_vs     )/*[KMAX]*/[JMAXP+1][IMAXP+1];
__thread double (*dev_ws     )/*[KMAX]*/[JMAXP+1][IMAXP+1];
__thread double (*dev_qs     )/*[KMAX]*/[JMAXP+1][IMAXP+1];
__thread double (*dev_rho_i  )/*[KMAX]*/[JMAXP+1][IMAXP+1];
__thread double (*dev_speed  )/*[KMAX]*/[JMAXP+1][IMAXP+1];
__thread double (*dev_square )/*[KMAX]*/[JMAXP+1][IMAXP+1];
__thread double (*dev_rhs    )/*[KMAX]*/[JMAXP+1][IMAXP+1][5];
__thread double (*dev_forcing)/*[KMAX]*/[JMAXP+1][IMAXP+1][5];

#define ALLOCATE(ptr, size) assert(cudaSuccess == cudaMalloc((void**)&(ptr), (size)*sizeof(*ptr)))
#define DEALLOCATE(ptr) assert(cudaSuccess == cudaFree(ptr))
#define MEMCPY(src, dst, size, kind) assert(cudaSuccess == cudaMemcpy(dst, src, (size)*sizeof(*src), kind))
#define HOST2DEV(ptr, size) MEMCPY(ptr, dev_ ## ptr, size, cudaMemcpyHostToDevice)
#define DEV2HOST(ptr, size) MEMCPY(dev_ ## ptr, ptr, size, cudaMemcpyDeviceToHost)

void allocate_device()
{
    ALLOCATE(dev_grid_points, 3);
/* common /fields/ */
    ALLOCATE(dev_u, KMAX);
    ALLOCATE(dev_us, KMAX);
    ALLOCATE(dev_vs, KMAX);
    ALLOCATE(dev_ws, KMAX);
    ALLOCATE(dev_qs, KMAX);
    ALLOCATE(dev_rho_i, KMAX);
    ALLOCATE(dev_speed, KMAX);
    ALLOCATE(dev_square, KMAX);
    ALLOCATE(dev_rhs, KMAX);
    ALLOCATE(dev_forcing, KMAX);
}

void deallocate_device()
{
    DEALLOCATE(dev_grid_points);
/* common /fields/ */
    DEALLOCATE(dev_u);
    DEALLOCATE(dev_us);
    DEALLOCATE(dev_vs);
    DEALLOCATE(dev_ws);
    DEALLOCATE(dev_qs);
    DEALLOCATE(dev_rho_i);
    DEALLOCATE(dev_speed);
    DEALLOCATE(dev_square);
    DEALLOCATE(dev_rhs);
    DEALLOCATE(dev_forcing);
}

void cuda_memcpy_host_to_device()
{
    HOST2DEV(grid_points, 3);
    HOST2DEV(u, KMAX);
    HOST2DEV(us, KMAX);
    HOST2DEV(vs, KMAX);
    HOST2DEV(ws, KMAX);
    HOST2DEV(qs, KMAX);
    HOST2DEV(rho_i, KMAX);
    HOST2DEV(speed, KMAX);
    HOST2DEV(square, KMAX);
    HOST2DEV(rhs, KMAX);
    HOST2DEV(forcing, KMAX);
}

void cuda_memcpy_device_to_host()
{
    DEV2HOST(grid_points, 3);
    DEV2HOST(u, KMAX);
    DEV2HOST(us, KMAX);
    DEV2HOST(vs, KMAX);
    DEV2HOST(ws, KMAX);
    DEV2HOST(qs, KMAX);
    DEV2HOST(rho_i, KMAX);
    DEV2HOST(speed, KMAX);
    DEV2HOST(square, KMAX);
    DEV2HOST(rhs, KMAX);
    DEV2HOST(forcing, KMAX);
}
