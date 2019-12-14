#include "header.h"

__device__ void lhsinit_kernel(
    int ni, int nj,
    double (*lhs )/*[IMAXP+1]*/[IMAXP+1][5],
    double (*lhsp)/*[IMAXP+1]*/[IMAXP+1][5],
    double (*lhsm)/*[IMAXP+1]*/[IMAXP+1][5]
) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int k = blockDim.z * blockIdx.z + threadIdx.z;

  int m;

  //---------------------------------------------------------------------
  // zap the whole left hand side for starters
  // set all diagonal values to 1. This is overkill, but convenient
  //---------------------------------------------------------------------
  if (i == 0 && k == 0)
      if (j >= 1 && j <= nj) {
        for (m = 0; m < 5; m++) {
          lhs [j][0][m] = 0.0;
          lhsp[j][0][m] = 0.0;
          lhsm[j][0][m] = 0.0;
          lhs [j][ni][m] = 0.0;
          lhsp[j][ni][m] = 0.0;
          lhsm[j][ni][m] = 0.0;
        }
        lhs [j][0][2] = 1.0;
        lhsp[j][0][2] = 1.0;
        lhsm[j][0][2] = 1.0;
        lhs [j][ni][2] = 1.0;
        lhsp[j][ni][2] = 1.0;
        lhsm[j][ni][2] = 1.0;
      }
}

__device__ void lhsinitj_kernel(
    int nj, int ni,
    double (*lhs )/*[IMAXP+1]*/[IMAXP+1][5],
    double (*lhsp)/*[IMAXP+1]*/[IMAXP+1][5],
    double (*lhsm)/*[IMAXP+1]*/[IMAXP+1][5]
) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int k = blockDim.z * blockIdx.z + threadIdx.z;

  int m;

  //---------------------------------------------------------------------
  // zap the whole left hand side for starters
  // set all diagonal values to 1. This is overkill, but convenient
  //---------------------------------------------------------------------
  if (j == 0 && k == 0)
      if (i >= 1 && i <= ni) {
        for (m = 0; m < 5; m++) {
          lhs [0][i][m] = 0.0;
          lhsp[0][i][m] = 0.0;
          lhsm[0][i][m] = 0.0;
          lhs [nj][i][m] = 0.0;
          lhsp[nj][i][m] = 0.0;
          lhsm[nj][i][m] = 0.0;
        }
        lhs [0][i][2] = 1.0;
        lhsp[0][i][2] = 1.0;
        lhsm[0][i][2] = 1.0;
        lhs [nj][i][2] = 1.0;
        lhsp[nj][i][2] = 1.0;
        lhsm[nj][i][2] = 1.0;
      }
}
