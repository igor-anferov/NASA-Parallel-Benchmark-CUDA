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

#define ALLOCATE(ptr, size) CHK_CUDA_OK(cudaMalloc((void**)&(ptr), (size)*sizeof(*(ptr))))
#define DEALLOCATE(ptr) CHK_CUDA_OK(cudaFree(ptr))
#define MEMCPY(src, dst, size, kind) CHK_CUDA_OK(cudaMemcpy(dst, src, (size)*sizeof(*(src)), kind))
#define HOST2DEV(ptr, size) MEMCPY(ptr, dev_ ## ptr, size, cudaMemcpyHostToDevice)
#define DEV2HOST(ptr, size) MEMCPY(dev_ ## ptr, ptr, size, cudaMemcpyDeviceToHost)
#define DEV2HOST_PART(ptr, size) DEV2HOST(ptr + gridOffset.z, gridElems.z)
#define DEV2HOST_HALO(ptr, size) { \
    if (device > 0) \
        DEV2HOST(ptr + gridOffset.z, size); \
    if (device < device_count - 1) \
        DEV2HOST(ptr + gridOffset.z + gridElems.z - size, size); \
}
#define HOST2DEV_HALO(ptr, size) { \
    if (device > 0) \
        HOST2DEV(ptr + gridOffset.z - size, size); \
    if (device < device_count - 1) \
        HOST2DEV(ptr + gridOffset.z + gridElems.z, size); \
}

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
    if (timeron) {
        timer_start(t_comm);
    }
    HOST2DEV(grid_points, 3);
/* common /fields/ */
    HOST2DEV(u, KMAX);
    HOST2DEV(forcing, KMAX);
    if (timeron) {
        timer_stop(t_comm);
    }
}

void cuda_memcpy_device_to_host()
{
    if (timeron) {
        timer_start(t_comm);
    }
/* common /fields/ */
    DEV2HOST_PART(u, KMAX);
    DEV2HOST_PART(rhs, KMAX);
    if (timeron) {
        timer_stop(t_comm);
    }
}

void cuda_sync_rhs()
{
    if (timeron) {
        timer_start(t_comm);
    }
    DEV2HOST_HALO(u, 2);
    DEV2HOST_HALO(us, 1);
    DEV2HOST_HALO(vs, 1);
    DEV2HOST_HALO(ws, 1);
    DEV2HOST_HALO(qs, 1);
    DEV2HOST_HALO(rho_i, 1);
    DEV2HOST_HALO(square, 1);
#pragma omp barrier
    HOST2DEV_HALO(u, 2);
    HOST2DEV_HALO(us, 1);
    HOST2DEV_HALO(vs, 1);
    HOST2DEV_HALO(ws, 1);
    HOST2DEV_HALO(qs, 1);
    HOST2DEV_HALO(rho_i, 1);
    HOST2DEV_HALO(square, 1);
    if (timeron) {
        timer_stop(t_comm);
    }
}

void cuda_sync_z_solve()
{
    if (timeron) {
        timer_start(t_comm);
    }
    DEV2HOST_PART(ws, KMAX);
    DEV2HOST_PART(rho_i, KMAX);
    DEV2HOST_PART(speed, KMAX);
    DEV2HOST_PART(rhs, KMAX);
#pragma omp barrier
    HOST2DEV(ws, KMAX);
    HOST2DEV(rho_i, KMAX);
    HOST2DEV(speed, KMAX);
    HOST2DEV(rhs, KMAX);
    if (timeron) {
        timer_stop(t_comm);
    }
}
