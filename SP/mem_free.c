#include "header.h"
#include <assert.h>
#include <cuda_runtime.h>

void mem_free()
{
    assert(cudaSuccess == cudaFreeHost(grid_points));

    assert(cudaSuccess == cudaFreeHost(u));
    assert(cudaSuccess == cudaFreeHost(us));
    assert(cudaSuccess == cudaFreeHost(vs));
    assert(cudaSuccess == cudaFreeHost(ws));
    assert(cudaSuccess == cudaFreeHost(qs));
    assert(cudaSuccess == cudaFreeHost(rho_i));
    assert(cudaSuccess == cudaFreeHost(speed));
    assert(cudaSuccess == cudaFreeHost(square));
    assert(cudaSuccess == cudaFreeHost(rhs));
    assert(cudaSuccess == cudaFreeHost(forcing));

    assert(cudaSuccess == cudaFreeHost(cv));
    assert(cudaSuccess == cudaFreeHost(rhon));
    assert(cudaSuccess == cudaFreeHost(rhos));
    assert(cudaSuccess == cudaFreeHost(rhoq));
    assert(cudaSuccess == cudaFreeHost(cuf));
    assert(cudaSuccess == cudaFreeHost(q));
    assert(cudaSuccess == cudaFreeHost(ue));
    assert(cudaSuccess == cudaFreeHost(buf));

    assert(cudaSuccess == cudaFreeHost(lhs));
    assert(cudaSuccess == cudaFreeHost(lhsp));
    assert(cudaSuccess == cudaFreeHost(lhsm));
}
