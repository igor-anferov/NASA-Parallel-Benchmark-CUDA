#include "header.h"
#include <assert.h>
#include <stdlib.h>
#include <cuda_runtime.h>

void mem_alloc()
{
    assert(grid_points = malloc(3*sizeof(*grid_points)));

    assert(u = malloc(5*sizeof(*u)));
    assert(us = malloc(KMAX*sizeof(*us)));
    assert(vs = malloc(KMAX*sizeof(*vs)));
    assert(ws = malloc(KMAX*sizeof(*ws)));
    assert(qs = malloc(KMAX*sizeof(*qs)));
    assert(rho_i = malloc(KMAX*sizeof(*rho_i)));
    assert(speed = malloc(KMAX*sizeof(*speed)));
    assert(square = malloc(KMAX*sizeof(*square)));
    assert(rhs = malloc(5*sizeof(*rhs)));
    assert(forcing = malloc(5*sizeof(*forcing)));

    assert(cv = malloc(PROBLEM_SIZE*sizeof(*cv)));
    assert(rhon = malloc(PROBLEM_SIZE*sizeof(*rhon)));
    assert(rhos = malloc(PROBLEM_SIZE*sizeof(*rhos)));
    assert(rhoq = malloc(PROBLEM_SIZE*sizeof(*rhoq)));
    assert(cuf = malloc(PROBLEM_SIZE*sizeof(*cuf)));
    assert(q = malloc(PROBLEM_SIZE*sizeof(*q)));
    assert(ue = malloc(PROBLEM_SIZE*sizeof(*ue)));
    assert(buf = malloc(PROBLEM_SIZE*sizeof(*buf)));

    assert(lhs = malloc((IMAXP+1)*sizeof(*lhs)));
    assert(lhsp = malloc((IMAXP+1)*sizeof(*lhsp)));
    assert(lhsm = malloc((IMAXP+1)*sizeof(*lhsm)));
}

void mem_free()
{
    free(grid_points);
    grid_points = NULL;


    free(u);
    u = NULL;
    free(us);
    us = NULL;
    free(vs);
    vs = NULL;
    free(ws);
    ws = NULL;
    free(qs);
    qs = NULL;
    free(rho_i);
    rho_i = NULL;
    free(speed);
    speed = NULL;
    free(square);
    square = NULL;
    free(rhs);
    rhs = NULL;
    free(forcing);
    forcing = NULL;

    free(cv);
    cv = NULL;
    free(rhon);
    rhon = NULL;
    free(rhos);
    rhos = NULL;
    free(rhoq);
    rhoq = NULL;
    free(cuf);
    cuf = NULL;
    free(q);
    q = NULL;
    free(ue);
    ue = NULL;
    free(buf);
    buf = NULL;

    free(lhs);
    lhs = NULL;
    free(lhsp);
    lhsp = NULL;
    free(lhsm);
    lhsm = NULL;
}

__thread int *device_grid_points;
__thread double (*device_u)[KMAX][JMAXP+1][IMAXP+1];
__thread double (*device_us)[JMAXP+1][IMAXP+1];
__thread double (*device_vs)[JMAXP+1][IMAXP+1];
__thread double (*device_ws)[JMAXP+1][IMAXP+1];
__thread double (*device_qs)[JMAXP+1][IMAXP+1];
__thread double (*device_rho_i)[JMAXP+1][IMAXP+1];
__thread double (*device_speed)[JMAXP+1][IMAXP+1];
__thread double (*device_square)[JMAXP+1][IMAXP+1];
__thread double (*device_rhs)[KMAX][JMAXP+1][IMAXP+1];
__thread double (*device_forcing)[KMAX][JMAXP+1][IMAXP+1];

void device_mem_alloc()
{
    assert(cudaSuccess == cudaMalloc((void**)&(device_grid_points), 3*sizeof(*device_grid_points)));

    assert(cudaSuccess == cudaMalloc((void**)&(device_u), 5*sizeof(*device_u)));
    assert(cudaSuccess == cudaMalloc((void**)&(device_us), KMAX*sizeof(*device_us)));
    assert(cudaSuccess == cudaMalloc((void**)&(device_vs), KMAX*sizeof(*device_vs)));
    assert(cudaSuccess == cudaMalloc((void**)&(device_ws), KMAX*sizeof(*device_ws)));
    assert(cudaSuccess == cudaMalloc((void**)&(device_qs), KMAX*sizeof(*device_qs)));
    assert(cudaSuccess == cudaMalloc((void**)&(device_rho_i), KMAX*sizeof(*device_rho_i)));
    assert(cudaSuccess == cudaMalloc((void**)&(device_speed), KMAX*sizeof(*device_speed)));
    assert(cudaSuccess == cudaMalloc((void**)&(device_square), KMAX*sizeof(*device_square)));
    assert(cudaSuccess == cudaMalloc((void**)&(device_rhs), 5*sizeof(*device_rhs)));
    assert(cudaSuccess == cudaMalloc((void**)&(device_forcing), 5*sizeof(*device_forcing)));
}

void device_mem_free()
{
    assert(cudaSuccess == cudaFree(device_grid_points));

    assert(cudaSuccess == cudaFree(device_u));
    assert(cudaSuccess == cudaFree(device_us));
    assert(cudaSuccess == cudaFree(device_vs));
    assert(cudaSuccess == cudaFree(device_ws));
    assert(cudaSuccess == cudaFree(device_qs));
    assert(cudaSuccess == cudaFree(device_rho_i));
    assert(cudaSuccess == cudaFree(device_speed));
    assert(cudaSuccess == cudaFree(device_square));
    assert(cudaSuccess == cudaFree(device_rhs));
    assert(cudaSuccess == cudaFree(device_forcing));
}

void host_to_device_memcpy()
{
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    assert(cudaSuccess == cudaMemcpyAsync(device_grid_points, grid_points, 3*sizeof(*grid_points), cudaMemcpyHostToDevice, stream1));
    assert(cudaSuccess == cudaMemcpyAsync(device_forcing, forcing, 5*sizeof(*forcing), cudaMemcpyHostToDevice, stream2));
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
}

void device_to_host_memcpy()
{
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    assert(cudaSuccess == cudaMemcpyAsync(u, device_u, 5*sizeof(*u), cudaMemcpyDeviceToHost, stream1));
    assert(cudaSuccess == cudaMemcpyAsync(rhs, device_rhs, 5*sizeof(*rhs), cudaMemcpyDeviceToHost, stream2));
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
}
