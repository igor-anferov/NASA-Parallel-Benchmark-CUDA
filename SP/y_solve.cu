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

#include <assert.h>
#include <omp.h>
#include "header.h"
#include "initialize_kernels.cu"

//---------------------------------------------------------------------
// this function performs the solution of the approximate factorization
// step in the y-direction for all five matrix components
// simultaneously. The Thomas algorithm is employed to solve the
// systems for the y-lines. Boundary conditions are non-periodic
//---------------------------------------------------------------------
__global__ void y_solve_kernel(
    dim3 gridOffset,
    int *grid_points/*[3]*/,
    int nx2, int ny2, int nz2,
    double (*vs     )/*[KMAX]*/[JMAXP+1][IMAXP+1],
    double (*rho_i  )/*[KMAX]*/[JMAXP+1][IMAXP+1],
    double (*speed  )/*[KMAX]*/[JMAXP+1][IMAXP+1],
    double (*rhs    )/*[KMAX]*/[5][JMAXP+1][IMAXP+1],
    double dtty1, double dtty2, double comz1, double comz4, double comz5, double comz6, double c2dtty1
) {
  int i = blockDim.x * blockIdx.x + threadIdx.x + gridOffset.x;
  int k = blockDim.z * blockIdx.z + threadIdx.z + gridOffset.z;

  double cv  [PROBLEM_SIZE];
  double rhoq[PROBLEM_SIZE];
  double lhs [IMAXP+1][5];
  double lhsp[IMAXP+1][5];
  double lhsm[IMAXP+1][5];

  int j, j1, j2, m;
  double ru1, fac1, fac2;

  if (k >= 1 && k <= grid_points[2]-2) {
    lhsinitj_kernel(ny2+1, nx2, lhs, lhsp, lhsm);

    //---------------------------------------------------------------------
    // Computes the left hand side for the three y-factors   
    //---------------------------------------------------------------------

    //---------------------------------------------------------------------
    // first fill the lhs for the u-eigenvalue         
    //---------------------------------------------------------------------
    if (i >= 1 && i <= grid_points[0]-2) {
      for (j = 0; j <= grid_points[1]-1; j++) {
        ru1 = c3c4*rho_i[k][j][i];
        cv[j] = vs[k][j][i];
        rhoq[j] = max(max(dy3+con43*ru1, dy5+c1c5*ru1), max(dymax+ru1, dy1));
      }

      for (j = 1; j <= grid_points[1]-2; j++) {
        lhs[j][0] =  0.0;
        lhs[j][1] = -dtty2 * cv[j-1] - dtty1 * rhoq[j-1];
        lhs[j][2] =  1.0 + c2dtty1 * rhoq[j];
        lhs[j][3] =  dtty2 * cv[j+1] - dtty1 * rhoq[j+1];
        lhs[j][4] =  0.0;
      }
    }

    //---------------------------------------------------------------------
    // add fourth order dissipation                             
    //---------------------------------------------------------------------
    if (i >= 1 && i <= grid_points[0]-2) {
      j = 1;
      lhs[j][2] = lhs[j][2] + comz5;
      lhs[j][3] = lhs[j][3] - comz4;
      lhs[j][4] = lhs[j][4] + comz1;

      lhs[j+1][1] = lhs[j+1][1] - comz4;
      lhs[j+1][2] = lhs[j+1][2] + comz6;
      lhs[j+1][3] = lhs[j+1][3] - comz4;
      lhs[j+1][4] = lhs[j+1][4] + comz1;
    }

    for (j = 3; j <= grid_points[1]-4; j++) {
      if (i >= 1 && i <= grid_points[0]-2) {
        lhs[j][0] = lhs[j][0] + comz1;
        lhs[j][1] = lhs[j][1] - comz4;
        lhs[j][2] = lhs[j][2] + comz6;
        lhs[j][3] = lhs[j][3] - comz4;
        lhs[j][4] = lhs[j][4] + comz1;
      }
    }

    if (i >= 1 && i <= grid_points[0]-2) {
      j = grid_points[1]-3;
      lhs[j][0] = lhs[j][0] + comz1;
      lhs[j][1] = lhs[j][1] - comz4;
      lhs[j][2] = lhs[j][2] + comz6;
      lhs[j][3] = lhs[j][3] - comz4;

      lhs[j+1][0] = lhs[j+1][0] + comz1;
      lhs[j+1][1] = lhs[j+1][1] - comz4;
      lhs[j+1][2] = lhs[j+1][2] + comz5;
    }

    //---------------------------------------------------------------------
    // subsequently, for (the other two factors                    
    //---------------------------------------------------------------------
    for (j = 1; j <= grid_points[1]-2; j++) {
      if (i >= 1 && i <= grid_points[0]-2) {
        lhsp[j][0] = lhs[j][0];
        lhsp[j][1] = lhs[j][1] - dtty2 * speed[k][j-1][i];
        lhsp[j][2] = lhs[j][2];
        lhsp[j][3] = lhs[j][3] + dtty2 * speed[k][j+1][i];
        lhsp[j][4] = lhs[j][4];
        lhsm[j][0] = lhs[j][0];
        lhsm[j][1] = lhs[j][1] + dtty2 * speed[k][j-1][i];
        lhsm[j][2] = lhs[j][2];
        lhsm[j][3] = lhs[j][3] - dtty2 * speed[k][j+1][i];
        lhsm[j][4] = lhs[j][4];
      }
    }


    //---------------------------------------------------------------------
    // FORWARD ELIMINATION  
    //---------------------------------------------------------------------
    for (j = 0; j <= grid_points[1]-3; j++) {
      j1 = j + 1;
      j2 = j + 2;
      if (i >= 1 && i <= grid_points[0]-2) {
        fac1 = 1.0/lhs[j][2];
        lhs[j][3] = fac1*lhs[j][3];
        lhs[j][4] = fac1*lhs[j][4];
        for (m = 0; m < 3; m++) {
          rhs[k][m][j][i] = fac1*rhs[k][m][j][i];
        }
        lhs[j1][2] = lhs[j1][2] - lhs[j1][1]*lhs[j][3];
        lhs[j1][3] = lhs[j1][3] - lhs[j1][1]*lhs[j][4];
        for (m = 0; m < 3; m++) {
          rhs[k][m][j1][i] = rhs[k][m][j1][i] - lhs[j1][1]*rhs[k][m][j][i];
        }
        lhs[j2][1] = lhs[j2][1] - lhs[j2][0]*lhs[j][3];
        lhs[j2][2] = lhs[j2][2] - lhs[j2][0]*lhs[j][4];
        for (m = 0; m < 3; m++) {
          rhs[k][m][j2][i] = rhs[k][m][j2][i] - lhs[j2][0]*rhs[k][m][j][i];
        }
      }
    }

    //---------------------------------------------------------------------
    // The last two rows in this grid block are a bit different, 
    // since they for (not have two more rows available for the
    // elimination of off-diagonal entries
    //---------------------------------------------------------------------
    j  = grid_points[1]-2;
    j1 = grid_points[1]-1;
    if (i >= 1 && i <= grid_points[0]-2) {
      fac1 = 1.0/lhs[j][2];
      lhs[j][3] = fac1*lhs[j][3];
      lhs[j][4] = fac1*lhs[j][4];
      for (m = 0; m < 3; m++) {
        rhs[k][m][j][i] = fac1*rhs[k][m][j][i];
      }
      lhs[j1][2] = lhs[j1][2] - lhs[j1][1]*lhs[j][3];
      lhs[j1][3] = lhs[j1][3] - lhs[j1][1]*lhs[j][4];
      for (m = 0; m < 3; m++) {
        rhs[k][m][j1][i] = rhs[k][m][j1][i] - lhs[j1][1]*rhs[k][m][j][i];
      }
      //---------------------------------------------------------------------
      // scale the last row immediately 
      //---------------------------------------------------------------------
      fac2 = 1.0/lhs[j1][2];
      for (m = 0; m < 3; m++) {
        rhs[k][m][j1][i] = fac2*rhs[k][m][j1][i];
      }
    }

    //---------------------------------------------------------------------
    // for (the u+c and the u-c factors                 
    //---------------------------------------------------------------------
    for (j = 0; j <= grid_points[1]-3; j++) {
      j1 = j + 1;
      j2 = j + 2;
      if (i >= 1 && i <= grid_points[0]-2) {
        m = 3;
        fac1 = 1.0/lhsp[j][2];
        lhsp[j][3]    = fac1*lhsp[j][3];
        lhsp[j][4]    = fac1*lhsp[j][4];
        rhs[k][m][j][i]  = fac1*rhs[k][m][j][i];
        lhsp[j1][2]   = lhsp[j1][2] - lhsp[j1][1]*lhsp[j][3];
        lhsp[j1][3]   = lhsp[j1][3] - lhsp[j1][1]*lhsp[j][4];
        rhs[k][m][j1][i] = rhs[k][m][j1][i] - lhsp[j1][1]*rhs[k][m][j][i];
        lhsp[j2][1]   = lhsp[j2][1] - lhsp[j2][0]*lhsp[j][3];
        lhsp[j2][2]   = lhsp[j2][2] - lhsp[j2][0]*lhsp[j][4];
        rhs[k][m][j2][i] = rhs[k][m][j2][i] - lhsp[j2][0]*rhs[k][m][j][i];

        m = 4;
        fac1 = 1.0/lhsm[j][2];
        lhsm[j][3]    = fac1*lhsm[j][3];
        lhsm[j][4]    = fac1*lhsm[j][4];
        rhs[k][m][j][i]  = fac1*rhs[k][m][j][i];
        lhsm[j1][2]   = lhsm[j1][2] - lhsm[j1][1]*lhsm[j][3];
        lhsm[j1][3]   = lhsm[j1][3] - lhsm[j1][1]*lhsm[j][4];
        rhs[k][m][j1][i] = rhs[k][m][j1][i] - lhsm[j1][1]*rhs[k][m][j][i];
        lhsm[j2][1]   = lhsm[j2][1] - lhsm[j2][0]*lhsm[j][3];
        lhsm[j2][2]   = lhsm[j2][2] - lhsm[j2][0]*lhsm[j][4];
        rhs[k][m][j2][i] = rhs[k][m][j2][i] - lhsm[j2][0]*rhs[k][m][j][i];
      }
    }

    //---------------------------------------------------------------------
    // And again the last two rows separately
    //---------------------------------------------------------------------
    j  = grid_points[1]-2;
    j1 = grid_points[1]-1;
    if (i >= 1 && i <= grid_points[0]-2) {
      m = 3;
      fac1 = 1.0/lhsp[j][2];
      lhsp[j][3]    = fac1*lhsp[j][3];
      lhsp[j][4]    = fac1*lhsp[j][4];
      rhs[k][m][j][i]  = fac1*rhs[k][m][j][i];
      lhsp[j1][2]   = lhsp[j1][2] - lhsp[j1][1]*lhsp[j][3];
      lhsp[j1][3]   = lhsp[j1][3] - lhsp[j1][1]*lhsp[j][4];
      rhs[k][m][j1][i] = rhs[k][m][j1][i] - lhsp[j1][1]*rhs[k][m][j][i];

      m = 4;
      fac1 = 1.0/lhsm[j][2];
      lhsm[j][3]    = fac1*lhsm[j][3];
      lhsm[j][4]    = fac1*lhsm[j][4];
      rhs[k][m][j][i]  = fac1*rhs[k][m][j][i];
      lhsm[j1][2]   = lhsm[j1][2] - lhsm[j1][1]*lhsm[j][3];
      lhsm[j1][3]   = lhsm[j1][3] - lhsm[j1][1]*lhsm[j][4];
      rhs[k][m][j1][i] = rhs[k][m][j1][i] - lhsm[j1][1]*rhs[k][m][j][i];

      //---------------------------------------------------------------------
      // Scale the last row immediately 
      //---------------------------------------------------------------------
      rhs[k][3][j1][i]   = rhs[k][3][j1][i]/lhsp[j1][2];
      rhs[k][4][j1][i]   = rhs[k][4][j1][i]/lhsm[j1][2];
    }


    //---------------------------------------------------------------------
    // BACKSUBSTITUTION 
    //---------------------------------------------------------------------
    j  = grid_points[1]-2;
    j1 = grid_points[1]-1;
    if (i >= 1 && i <= grid_points[0]-2) {
      for (m = 0; m < 3; m++) {
        rhs[k][m][j][i] = rhs[k][m][j][i] - lhs[j][3]*rhs[k][m][j1][i];
      }

      rhs[k][3][j][i] = rhs[k][3][j][i] - lhsp[j][3]*rhs[k][3][j1][i];
      rhs[k][4][j][i] = rhs[k][4][j][i] - lhsm[j][3]*rhs[k][4][j1][i];
    }

    //---------------------------------------------------------------------
    // The first three factors
    //---------------------------------------------------------------------
    for (j = grid_points[1]-3; j >= 0; j--) {
      j1 = j + 1;
      j2 = j + 2;
      if (i >= 1 && i <= grid_points[0]-2) {
        for (m = 0; m < 3; m++) {
          rhs[k][m][j][i] = rhs[k][m][j][i] - 
                            lhs[j][3]*rhs[k][m][j1][i] -
                            lhs[j][4]*rhs[k][m][j2][i];
        }

        //-------------------------------------------------------------------
        // And the remaining two
        //-------------------------------------------------------------------
        rhs[k][3][j][i] = rhs[k][3][j][i] - 
                          lhsp[j][3]*rhs[k][3][j1][i] -
                          lhsp[j][4]*rhs[k][3][j2][i];
        rhs[k][4][j][i] = rhs[k][4][j][i] - 
                          lhsm[j][3]*rhs[k][4][j1][i] -
                          lhsm[j][4]*rhs[k][4][j2][i];
      }
    }
  }
}

void y_solve() {
  if (timeron) timer_start(t_ysolve);
  y_solve_kernel <<< gridDimXZ, blockDimXZ >>> (
    gridOffset,
    dev_grid_points[device],
    nx2, ny2, nz2,
    dev_vs[device], dev_rho_i[device], dev_speed[device], dev_rhs[device],
    dtty1, dtty2, comz1, comz4, comz5, comz6, c2dtty1
  );
  if (timeron) timer_stop(t_ysolve);

  pinvr();
}
