#include "header.h"
#include <assert.h>
#include <cuda_runtime.h>

void mem_alloc()
{
    assert(cudaSuccess == cudaHostAlloc((void**)&(grid_points), 3*sizeof(*grid_points), 0));

    assert(cudaSuccess == cudaHostAlloc((void**)&(u), KMAX*sizeof(*u), 0));
    assert(cudaSuccess == cudaHostAlloc((void**)&(us), KMAX*sizeof(*us), 0));
    assert(cudaSuccess == cudaHostAlloc((void**)&(vs), KMAX*sizeof(*vs), 0));
    assert(cudaSuccess == cudaHostAlloc((void**)&(ws), KMAX*sizeof(*ws), 0));
    assert(cudaSuccess == cudaHostAlloc((void**)&(qs), KMAX*sizeof(*qs), 0));
    assert(cudaSuccess == cudaHostAlloc((void**)&(rho_i), KMAX*sizeof(*rho_i), 0));
    assert(cudaSuccess == cudaHostAlloc((void**)&(speed), KMAX*sizeof(*speed), 0));
    assert(cudaSuccess == cudaHostAlloc((void**)&(square), KMAX*sizeof(*square), 0));
    assert(cudaSuccess == cudaHostAlloc((void**)&(rhs), KMAX*sizeof(*rhs), 0));
    assert(cudaSuccess == cudaHostAlloc((void**)&(forcing), KMAX*sizeof(*forcing), 0));

    assert(cudaSuccess == cudaHostAlloc((void**)&(cv), PROBLEM_SIZE*sizeof(*cv), 0));
    assert(cudaSuccess == cudaHostAlloc((void**)&(rhon), PROBLEM_SIZE*sizeof(*rhon), 0));
    assert(cudaSuccess == cudaHostAlloc((void**)&(rhos), PROBLEM_SIZE*sizeof(*rhos), 0));
    assert(cudaSuccess == cudaHostAlloc((void**)&(rhoq), PROBLEM_SIZE*sizeof(*rhoq), 0));
    assert(cudaSuccess == cudaHostAlloc((void**)&(cuf), PROBLEM_SIZE*sizeof(*cuf), 0));
    assert(cudaSuccess == cudaHostAlloc((void**)&(q), PROBLEM_SIZE*sizeof(*q), 0));
    assert(cudaSuccess == cudaHostAlloc((void**)&(ue), PROBLEM_SIZE*sizeof(*ue), 0));
    assert(cudaSuccess == cudaHostAlloc((void**)&(buf), PROBLEM_SIZE*sizeof(*buf), 0));

    assert(cudaSuccess == cudaHostAlloc((void**)&(lhs), (IMAXP+1)*sizeof(*lhs), 0));
    assert(cudaSuccess == cudaHostAlloc((void**)&(lhsp), (IMAXP+1)*sizeof(*lhsp), 0));
    assert(cudaSuccess == cudaHostAlloc((void**)&(lhsm), (IMAXP+1)*sizeof(*lhsm), 0));
}
