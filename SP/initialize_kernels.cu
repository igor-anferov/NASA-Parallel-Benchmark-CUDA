#include "header.h"

__device__ __attribute__((weak)) void lhsinit_kernel(
    int ni, int nj,
    double (*lhs )/*[IMAXP+1]*/[5],
    double (*lhsp)/*[IMAXP+1]*/[5],
    double (*lhsm)/*[IMAXP+1]*/[5]
) {
  int m;

  //---------------------------------------------------------------------
  // zap the whole left hand side for starters
  // set all diagonal values to 1. This is overkill, but convenient
  //---------------------------------------------------------------------
  for (m = 0; m < 5; m++) {
    lhs [0][m] = 0.0;
    lhsp[0][m] = 0.0;
    lhsm[0][m] = 0.0;
    lhs [ni][m] = 0.0;
    lhsp[ni][m] = 0.0;
    lhsm[ni][m] = 0.0;
  }
  lhs [0][2] = 1.0;
  lhsp[0][2] = 1.0;
  lhsm[0][2] = 1.0;
  lhs [ni][2] = 1.0;
  lhsp[ni][2] = 1.0;
  lhsm[ni][2] = 1.0;
}

__device__ __attribute__((weak)) void lhsinitj_kernel(
    int nj, int ni,
    double (*lhs )/*[IMAXP+1]*/[5],
    double (*lhsp)/*[IMAXP+1]*/[5],
    double (*lhsm)/*[IMAXP+1]*/[5]
) {
  int m;

  //---------------------------------------------------------------------
  // zap the whole left hand side for starters
  // set all diagonal values to 1. This is overkill, but convenient
  //---------------------------------------------------------------------
  for (m = 0; m < 5; m++) {
    lhs [0][m] = 0.0;
    lhsp[0][m] = 0.0;
    lhsm[0][m] = 0.0;
    lhs [nj][m] = 0.0;
    lhsp[nj][m] = 0.0;
    lhsm[nj][m] = 0.0;
  }
  lhs [0][2] = 1.0;
  lhsp[0][2] = 1.0;
  lhsm[0][2] = 1.0;
  lhs [nj][2] = 1.0;
  lhsp[nj][2] = 1.0;
  lhsm[nj][2] = 1.0;
  
}
