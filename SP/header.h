//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is a serial C version of the NPB SP code. This C        //
//  version is developed by the Center for Manycore Programming at Seoul   //
//  National University and derived from the serial Fortran versions in    //
//  "NPB3.3-SER" developed by NAS.                                         //
//                                                                         //
//  Permission to use, copy, distribute and modify this software for any   //
//  purpose with or without fee is hereby granted. This software is        //
//  provided "as is" without express or implied warranty.                  //
//                                                                         //
//  Information on NPB 3.3, including the technical report, the original   //
//  specifications, source code, results and information on how to submit  //
//  new results, is available at:                                          //
//                                                                         //
//           http://www.nas.nasa.gov/Software/NPB/                         //
//                                                                         //
//  Send comments or suggestions for this C version to cmp@aces.snu.ac.kr  //
//                                                                         //
//          Center for Manycore Programming                                //
//          School of Computer Science and Engineering                     //
//          Seoul National University                                      //
//          Seoul 151-744, Korea                                           //
//                                                                         //
//          E-mail:  cmp@aces.snu.ac.kr                                    //
//                                                                         //
//-------------------------------------------------------------------------//

//-------------------------------------------------------------------------//
// Authors: Sangmin Seo, Jungwon Kim, Jun Lee, Jeongho Nah, Gangwon Jo,    //
//          and Jaejin Lee                                                 //
//-------------------------------------------------------------------------//

//---------------------------------------------------------------------
// The following include file is generated automatically by the
// "setparams" utility. It defines 
//      problem_size:  12, 64, 102, 162 (for class T, A, B, C)
//      dt_default:    default time step for this problem size if no
//                     config file
//      niter_default: default number of iterations for this problem size
//---------------------------------------------------------------------
#pragma once

#include "stdlib.h"
#include "stdio.h"

#include "npbparams.h"
#include "type.h"

#include "timers.h"
#ifdef NEED_CUDA
#include <cuda_runtime.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef NEED_CUDA
extern __thread int device;
extern int device_count;

#define CHK_CUDA_OK(expr) { \
    cudaError_t __status__ = (expr); \
    if (__status__ != cudaSuccess) { \
        fprintf(stderr, "%s:%d\t [thread %d] %s\n%s (%s)\n", \
            __FILE__, __LINE__, device, #expr, cudaGetErrorName(__status__), cudaGetErrorString(__status__)); \
        abort(); \
    } \
}

#endif

/* common /global/ */
extern int *grid_points/*[3]*/;
extern int **dev_grid_points/*[3]*/;
extern int nx2, ny2, nz2;
extern logical timeron;

#ifdef __NVCC__
extern __thread dim3 blockDim_;
extern __thread dim3 gridDim_;

extern __thread dim3 blockDimYZ;
extern __thread dim3 gridDimYZ;

extern __thread dim3 blockDimXY;
extern __thread dim3 gridDimXY;

extern __thread dim3 blockDimXZ;
extern __thread dim3 gridDimXZ;

extern __thread dim3 gridElems;
extern __thread dim3 gridOffset;
#elif defined __cplusplus
extern thread_local dim3 blockDim_;
extern thread_local dim3 gridDim_;

extern thread_local dim3 blockDimYZ;
extern thread_local dim3 gridDimYZ;

extern thread_local dim3 blockDimXY;
extern thread_local dim3 gridDimXY;

extern thread_local dim3 blockDimXZ;
extern thread_local dim3 gridDimXZ;

extern thread_local dim3 gridElems;
extern thread_local dim3 gridOffset;
#endif

#ifdef __NVCC__
__device__
#endif
static const double ce[5][13] = {
  { 2.0, 0.0, 0.0, 4.0, 5.0, 3.0, 0.5, 0.02, 0.01, 0.03, 0.5, 0.4, 0.3 },
  { 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.01, 0.03, 0.02, 0.4, 0.3, 0.5 },
  { 2.0, 2.0, 0.0, 0.0, 0.0, 2.0, 3.0, 0.04, 0.03, 0.05, 0.3, 0.5, 0.4 },
  { 2.0, 2.0, 0.0, 0.0, 0.0, 2.0, 3.0, 0.03, 0.05, 0.04, 0.2, 0.1, 0.3 },
  { 5.0, 4.0, 3.0, 2.0, 0.1, 0.4, 0.3, 0.05, 0.04, 0.03, 0.1, 0.3, 0.2 }
};

#define c1 1.4
#define c2 0.4
#define c3 0.1
#define c4 1.0
#define c5 1.4

#define bt 0.70710678118654752440084436210484903928483593768847403658833986899536623923105351942519376716382078636750692311545

#define c1c2 (c1 * c2)
#define c1c5 (c1 * c5)
#define c3c4 (c3 * c4)
#define c1345 (c1c5 * c3c4)

#define conz1 (1.0-c1c5)

#define dx1 0.75
#define dx2 0.75
#define dx3 0.75
#define dx4 0.75
#define dx5 0.75

#define dy1 0.75
#define dy2 0.75
#define dy3 0.75
#define dy4 0.75
#define dy5 0.75

#define dz1 1.0
#define dz2 1.0
#define dz3 1.0
#define dz4 1.0
#define dz5 1.0
 
#define dxmax max(dx3, dx4)
#define dymax max(dy2, dy4)
#define dzmax max(dz2, dz3)
 
#define dssp (0.25 * max(dx1, max(dy1, dz1)))
 
#define c4dssp (4.0 * dssp)
#define c5dssp (5.0 * dssp)

#define c2iv  2.5
#define con43 (4.0/3.0)
#define con16 (1.0/6.0)

/* common /constants/ */
extern double tx1, tx2, tx3, ty1, ty2, ty3, tz1, tz2, tz3, dt, 
              xxcon1, xxcon2, dnzm1, dtdssp, dttx1,
              xxcon3, xxcon4, xxcon5, dx1tx1, dx2tx1, dx3tx1,
              dx4tx1, dx5tx1, yycon1, yycon2, yycon3, yycon4,
              yycon5, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1,
              zzcon1, zzcon2, zzcon3, zzcon4, zzcon5, dz1tz1, 
              dz2tz1, dz3tz1, dz4tz1, dz5tz1, dnxm1, dnym1, 
              dttx2, dtty1, dtty2, dttz1, dttz2, c2dttx1, 
              c2dtty1, c2dttz1, comz1, comz4, comz5, comz6, 
              c3c4tx3, c3c4ty3, c3c4tz3;

#define IMAX    PROBLEM_SIZE
#define JMAX    PROBLEM_SIZE
#define KMAX    PROBLEM_SIZE
#define IMAXP   (IMAX/2*2)
#define JMAXP   (JMAX/2*2)

//---------------------------------------------------------------------
// To improve cache performance, grid dimensions padded by 1 
// for even number sizes only
//---------------------------------------------------------------------
/* common /fields/ */
extern double (*u      )/*[KMAX]*/[5][JMAXP+1][IMAXP+1];
extern double (*us     )/*[KMAX]*/[JMAXP+1][IMAXP+1];
extern double (*vs     )/*[KMAX]*/[JMAXP+1][IMAXP+1];
extern double (*ws     )/*[KMAX]*/[JMAXP+1][IMAXP+1];
extern double (*qs     )/*[KMAX]*/[JMAXP+1][IMAXP+1];
extern double (*rho_i  )/*[KMAX]*/[JMAXP+1][IMAXP+1];
extern double (*speed  )/*[KMAX]*/[JMAXP+1][IMAXP+1];
extern double (*square )/*[KMAX]*/[JMAXP+1][IMAXP+1];
extern double (*rhs    )/*[KMAX]*/[5][JMAXP+1][IMAXP+1];
extern double (*forcing)/*[KMAX]*/[5][JMAXP+1][IMAXP+1];

#ifdef NEED_CUDA
/* common /fields/ */
extern double (**dev_u      )/*[KMAX]*/[5][JMAXP+1][IMAXP+1];
extern double (**dev_us     )/*[KMAX]*/[JMAXP+1][IMAXP+1];
extern double (**dev_vs     )/*[KMAX]*/[JMAXP+1][IMAXP+1];
extern double (**dev_ws     )/*[KMAX]*/[JMAXP+1][IMAXP+1];
extern double (**dev_qs     )/*[KMAX]*/[JMAXP+1][IMAXP+1];
extern double (**dev_rho_i  )/*[KMAX]*/[JMAXP+1][IMAXP+1];
extern double (**dev_speed  )/*[KMAX]*/[JMAXP+1][IMAXP+1];
extern double (**dev_square )/*[KMAX]*/[JMAXP+1][IMAXP+1];
extern double (**dev_rhs    )/*[KMAX]*/[5][JMAXP+1][IMAXP+1];
extern double (**dev_forcing)/*[KMAX]*/[5][JMAXP+1][IMAXP+1];
#endif

/* common /work_1d/ */
extern double (*cv  )/*[PROBLEM_SIZE]*/;
extern double (*rhon)/*[PROBLEM_SIZE]*/;
extern double (*rhos)/*[PROBLEM_SIZE]*/;
extern double (*rhoq)/*[PROBLEM_SIZE]*/;
extern double (*cuf )/*[PROBLEM_SIZE]*/;
extern double (*q   )/*[PROBLEM_SIZE]*/;
extern double (*ue  )/*[PROBLEM_SIZE]*/[5];
extern double (*buf )/*[PROBLEM_SIZE]*/[5];

/* common /work_lhs/ */
extern double (*lhs )/*[IMAXP+1]*/[IMAXP+1][5];
extern double (*lhsp)/*[IMAXP+1]*/[IMAXP+1][5];
extern double (*lhsm)/*[IMAXP+1]*/[IMAXP+1][5];

//-----------------------------------------------------------------------
// Timer constants
//-----------------------------------------------------------------------
#define t_total     1
#define t_rhsx      2
#define t_rhsy      3
#define t_rhsz      4
#define t_rhs       5
#define t_xsolve    6
#define t_ysolve    7
#define t_zsolve    8
#define t_rdis1     9
#define t_rdis2     10
#define t_txinvr    11
#define t_pinvr     12
#define t_ninvr     13
#define t_tzetar    14
#define t_add       15
#define t_init      16
#define t_comm      17
#define t_last      17


//-----------------------------------------------------------------------
#ifdef NEED_CUDA
void cuda_preinit();
void cuda_init();
void cuda_init_sizes();
void allocate_device();
void deallocate_device();
void cuda_memcpy_host_to_device();
void cuda_memcpy_device_to_host();
void cuda_sync_rhs();
void cuda_sync_z_solve();
#endif
void allocate();
void initialize();
void lhsinit(int ni, int nj);
void lhsinitj(int nj, int ni);
#ifdef __NVCC__
__device__
#endif
void exact_solution(double xi, double eta, double zeta, double dtemp[5]);
void exact_rhs();
void set_constants();
void adi();
void compute_rhs();
void x_solve();
void ninvr();
void y_solve();
void pinvr();
void z_solve();
void tzetar();
void add();
void txinvr();
void error_norm(double rms[5]);
void rhs_norm(double rms[5]);
void verify(int no_time_steps, char *Class, logical *verified);
void deallocate();

#ifdef __cplusplus
}
#endif
