#include <assert.h>
#include <cuda_runtime.h>

#include "header.h"

int **dev_grid_points/*[3]*/;
double (**dev_u      )/*[KMAX]*/[5][JMAXP+1][IMAXP+1];
double (**dev_us     )/*[KMAX]*/[JMAXP+1][IMAXP+1];
double (**dev_vs     )/*[KMAX]*/[JMAXP+1][IMAXP+1];
double (**dev_ws     )/*[KMAX]*/[JMAXP+1][IMAXP+1];
double (**dev_qs     )/*[KMAX]*/[JMAXP+1][IMAXP+1];
double (**dev_rho_i  )/*[KMAX]*/[JMAXP+1][IMAXP+1];
double (**dev_speed  )/*[KMAX]*/[JMAXP+1][IMAXP+1];
double (**dev_square )/*[KMAX]*/[JMAXP+1][IMAXP+1];
double (**dev_rhs    )/*[KMAX]*/[5][JMAXP+1][IMAXP+1];
double (**dev_forcing)/*[KMAX]*/[5][JMAXP+1][IMAXP+1];

#define ALLOCATE(ptr, size) { \
    if (device == 0) \
        ptr = (typeof(ptr)) malloc(device_count * sizeof(*(ptr))); \
    _Pragma("omp barrier") \
    CHK_CUDA_OK(cudaMalloc((void**)&(ptr)[device], (size)*sizeof(*(ptr)[device]))); \
}
#define DEALLOCATE(ptr) { \
    CHK_CUDA_OK(cudaFree(ptr[device])); \
    _Pragma("omp barrier") \
    if (device == 0) \
        free(ptr); \
}
#define MEMCPY(src, dst, size, kind) CHK_CUDA_OK(cudaMemcpy(dst, src, (size)*sizeof(*(src)), kind))
#define MEMCPY_P2P(src, si, dst, di, size) CHK_CUDA_OK(cudaMemcpyPeer(dst, di, src, si, (size)*sizeof(*(src))))
#define HOST2DEV_FROM(ptr, offset, size) MEMCPY(ptr + offset, dev_ ## ptr[device] + offset, size, cudaMemcpyHostToDevice)
#define HOST2DEV(ptr, size) HOST2DEV_FROM(ptr, 0, size)
#define DEV2HOST(ptr, offset, size) MEMCPY(dev_ ## ptr[device] + offset, ptr + offset, size, cudaMemcpyDeviceToHost)
#define DEV2DEV(ptr, si, di, offset, size) MEMCPY_P2P(dev_ ## ptr[si] + offset, si, dev_ ## ptr[di] + offset, di, size)
#define DEV2HOST_PART(ptr, size) DEV2HOST(ptr, gridOffset.z, gridElems.z)
#define UPDATE_HALO(ptr, size) { \
    if (device > 0) \
        DEV2DEV(ptr, device, device - 1, gridOffset.z, size); \
    if (device < device_count - 1) \
        DEV2DEV(ptr, device, device + 1, gridOffset.z + gridElems.z - size, size); \
}
#define BCAST(ptr) { \
    for (int i = 0; i < device_count; ++i) { \
        if (i == device) \
            continue; \
        DEV2DEV(ptr, device, i, gridOffset.z, gridElems.z); \
    } \
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
#pragma omp barrier
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
    UPDATE_HALO(u, 2);
    UPDATE_HALO(us, 1);
    UPDATE_HALO(vs, 1);
    UPDATE_HALO(ws, 1);
    UPDATE_HALO(qs, 1);
    UPDATE_HALO(rho_i, 1);
    UPDATE_HALO(square, 1);
#pragma omp barrier
    if (timeron) {
        timer_stop(t_comm);
    }
}

void cuda_sync_z_solve()
{
    if (timeron) {
        timer_start(t_comm);
    }
#pragma omp barrier
    BCAST(ws);
    BCAST(rho_i);
    BCAST(speed);
    BCAST(rhs);
#pragma omp barrier
    if (timeron) {
        timer_stop(t_comm);
    }
}
