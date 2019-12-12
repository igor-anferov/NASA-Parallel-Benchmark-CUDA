#include "header.h"

#include <assert.h>

#ifdef NEED_CUDA
#include <cuda_runtime.h>
#define ALLOCATE(ptr, size) assert(cudaSuccess == cudaHostAlloc((void**)&(ptr), (size)*sizeof(*ptr), 0))
#else
#include <stdlib.h>
#define ALLOCATE(ptr, size) assert(ptr = malloc((size)*sizeof(*ptr)))
#endif

void allocate()
{
/* common /global/ */
    ALLOCATE(grid_points, 3);

/* common /fields/ */
    ALLOCATE(u, KMAX);
    ALLOCATE(us, KMAX);
    ALLOCATE(vs, KMAX);
    ALLOCATE(ws, KMAX);
    ALLOCATE(qs, KMAX);
    ALLOCATE(rho_i, KMAX);
    ALLOCATE(speed, KMAX);
    ALLOCATE(square, KMAX);
    ALLOCATE(rhs, KMAX);
    ALLOCATE(forcing, KMAX);

/* common /work_1d/ */
    ALLOCATE(cv, PROBLEM_SIZE);
    ALLOCATE(rhon, PROBLEM_SIZE);
    ALLOCATE(rhos, PROBLEM_SIZE);
    ALLOCATE(rhoq, PROBLEM_SIZE);
    ALLOCATE(cuf, PROBLEM_SIZE);
    ALLOCATE(q, PROBLEM_SIZE);
    ALLOCATE(ue, PROBLEM_SIZE);
    ALLOCATE(buf, PROBLEM_SIZE);

/* common /work_lhs/ */
    ALLOCATE(lhs, IMAXP+1);
    ALLOCATE(lhsp, IMAXP+1);
    ALLOCATE(lhsm, IMAXP+1);
}
