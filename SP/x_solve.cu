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
#include "header.h"
#include "initialize_kernels.cu"

//---------------------------------------------------------------------
// this function performs the solution of the approximate factorization
// step in the x-direction for all five matrix components
// simultaneously. The Thomas algorithm is employed to solve the
// systems for the x-lines. Boundary conditions are non-periodic
//---------------------------------------------------------------------
__global__ void x_solve_kernel(
    int *grid_points/*[3]*/,
    int nx2, int ny2, int nz2,
    double (*us     )/*[KMAX]*/[JMAXP+1][IMAXP+1],
    double (*rho_i  )/*[KMAX]*/[JMAXP+1][IMAXP+1],
    double (*speed  )/*[KMAX]*/[JMAXP+1][IMAXP+1],
    double (*rhs    )/*[KMAX]*/[JMAXP+1][IMAXP+1][5],
    double (*cv  )/*[PROBLEM_SIZE]*/,
    double (*rhon)/*[PROBLEM_SIZE]*/,
    double dttx1, double dttx2, double comz1, double comz4, double comz5, double comz6, double c2dttx1
) {
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int k = blockDim.z * blockIdx.z + threadIdx.z;

  double _lhs [IMAXP+1][5];
  double _lhsp[IMAXP+1][5];
  double _lhsm[IMAXP+1][5];

  int i, i1, i2, m;
  double ru1, fac1, fac2;

  if (k >= 1 && k <= nz2) {
    lhsinit_kernel(nx2+1, ny2, _lhs, _lhsp, _lhsm);

    //---------------------------------------------------------------------
    // Computes the left hand side for the three x-factors  
    //---------------------------------------------------------------------

    //---------------------------------------------------------------------
    // first fill the _lhs for the u-eigenvalue                   
    //---------------------------------------------------------------------
    if (j >= 1 && j <= ny2) {
      for (i = 0; i <= grid_points[0]-1; i++) {
        ru1 = c3c4*rho_i[k][j][i];
        cv[i] = us[k][j][i];
        rhon[i] = max(max(dx2+con43*ru1,dx5+c1c5*ru1), max(dxmax+ru1,dx1));
      }

      for (i = 1; i <= nx2; i++) {
        _lhs[i][0] =  0.0;
        _lhs[i][1] = -dttx2 * cv[i-1] - dttx1 * rhon[i-1];
        _lhs[i][2] =  1.0 + c2dttx1 * rhon[i];
        _lhs[i][3] =  dttx2 * cv[i+1] - dttx1 * rhon[i+1];
        _lhs[i][4] =  0.0;
      }
    }

    //---------------------------------------------------------------------
    // add fourth order dissipation                             
    //---------------------------------------------------------------------
    if (j >= 1 && j <= ny2) {
      i = 1;
      _lhs[i][2] = _lhs[i][2] + comz5;
      _lhs[i][3] = _lhs[i][3] - comz4;
      _lhs[i][4] = _lhs[i][4] + comz1;

      _lhs[i+1][1] = _lhs[i+1][1] - comz4;
      _lhs[i+1][2] = _lhs[i+1][2] + comz6;
      _lhs[i+1][3] = _lhs[i+1][3] - comz4;
      _lhs[i+1][4] = _lhs[i+1][4] + comz1;
    }

    if (j >= 1 && j <= ny2) {
      for (i = 3; i <= grid_points[0]-4; i++) {
        _lhs[i][0] = _lhs[i][0] + comz1;
        _lhs[i][1] = _lhs[i][1] - comz4;
        _lhs[i][2] = _lhs[i][2] + comz6;
        _lhs[i][3] = _lhs[i][3] - comz4;
        _lhs[i][4] = _lhs[i][4] + comz1;
      }
    }

    if (j >= 1 && j <= ny2) {
      i = grid_points[0]-3;
      _lhs[i][0] = _lhs[i][0] + comz1;
      _lhs[i][1] = _lhs[i][1] - comz4;
      _lhs[i][2] = _lhs[i][2] + comz6;
      _lhs[i][3] = _lhs[i][3] - comz4;

      _lhs[i+1][0] = _lhs[i+1][0] + comz1;
      _lhs[i+1][1] = _lhs[i+1][1] - comz4;
      _lhs[i+1][2] = _lhs[i+1][2] + comz5;
    }

    //---------------------------------------------------------------------
    // subsequently, fill the other factors (u+c), (u-c) by adding to 
    // the first  
    //---------------------------------------------------------------------
    if (j >= 1 && j <= ny2) {
      for (i = 1; i <= nx2; i++) {
        _lhsp[i][0] = _lhs[i][0];
        _lhsp[i][1] = _lhs[i][1] - dttx2 * speed[k][j][i-1];
        _lhsp[i][2] = _lhs[i][2];
        _lhsp[i][3] = _lhs[i][3] + dttx2 * speed[k][j][i+1];
        _lhsp[i][4] = _lhs[i][4];
        _lhsm[i][0] = _lhs[i][0];
        _lhsm[i][1] = _lhs[i][1] + dttx2 * speed[k][j][i-1];
        _lhsm[i][2] = _lhs[i][2];
        _lhsm[i][3] = _lhs[i][3] - dttx2 * speed[k][j][i+1];
        _lhsm[i][4] = _lhs[i][4];
      }
    }

    //---------------------------------------------------------------------
    // FORWARD ELIMINATION  
    //---------------------------------------------------------------------

    //---------------------------------------------------------------------
    // perform the Thomas algorithm; first, FORWARD ELIMINATION     
    //---------------------------------------------------------------------
    if (j >= 1 && j <= ny2) {
      for (i = 0; i <= grid_points[0]-3; i++) {
        i1 = i + 1;
        i2 = i + 2;
        fac1 = 1.0/_lhs[i][2];
        _lhs[i][3] = fac1*_lhs[i][3];
        _lhs[i][4] = fac1*_lhs[i][4];
        for (m = 0; m < 3; m++) {
          rhs[k][j][i][m] = fac1*rhs[k][j][i][m];
        }
        _lhs[i1][2] = _lhs[i1][2] - _lhs[i1][1]*_lhs[i][3];
        _lhs[i1][3] = _lhs[i1][3] - _lhs[i1][1]*_lhs[i][4];
        for (m = 0; m < 3; m++) {
          rhs[k][j][i1][m] = rhs[k][j][i1][m] - _lhs[i1][1]*rhs[k][j][i][m];
        }
        _lhs[i2][1] = _lhs[i2][1] - _lhs[i2][0]*_lhs[i][3];
        _lhs[i2][2] = _lhs[i2][2] - _lhs[i2][0]*_lhs[i][4];
        for (m = 0; m < 3; m++) {
          rhs[k][j][i2][m] = rhs[k][j][i2][m] - _lhs[i2][0]*rhs[k][j][i][m];
        }
      }
    }

    //---------------------------------------------------------------------
    // The last two rows in this grid block are a bit different, 
    // since they for (not have two more rows available for the
    // elimination of off-diagonal entries
    //---------------------------------------------------------------------
    if (j >= 1 && j <= ny2) {
      i  = grid_points[0]-2;
      i1 = grid_points[0]-1;
      fac1 = 1.0/_lhs[i][2];
      _lhs[i][3] = fac1*_lhs[i][3];
      _lhs[i][4] = fac1*_lhs[i][4];
      for (m = 0; m < 3; m++) {
        rhs[k][j][i][m] = fac1*rhs[k][j][i][m];
      }
      _lhs[i1][2] = _lhs[i1][2] - _lhs[i1][1]*_lhs[i][3];
      _lhs[i1][3] = _lhs[i1][3] - _lhs[i1][1]*_lhs[i][4];
      for (m = 0; m < 3; m++) {
        rhs[k][j][i1][m] = rhs[k][j][i1][m] - _lhs[i1][1]*rhs[k][j][i][m];
      }

      //---------------------------------------------------------------------
      // scale the last row immediately 
      //---------------------------------------------------------------------
      fac2 = 1.0/_lhs[i1][2];
      for (m = 0; m < 3; m++) {
        rhs[k][j][i1][m] = fac2*rhs[k][j][i1][m];
      }
    }

    //---------------------------------------------------------------------
    // for (the u+c and the u-c factors                 
    //---------------------------------------------------------------------
    if (j >= 1 && j <= ny2) {
      for (i = 0; i <= grid_points[0]-3; i++) {
        i1 = i + 1;
        i2 = i + 2;

        m = 3;
        fac1 = 1.0/_lhsp[i][2];
        _lhsp[i][3]    = fac1*_lhsp[i][3];
        _lhsp[i][4]    = fac1*_lhsp[i][4];
        rhs[k][j][i][m]  = fac1*rhs[k][j][i][m];
        _lhsp[i1][2]   = _lhsp[i1][2] - _lhsp[i1][1]*_lhsp[i][3];
        _lhsp[i1][3]   = _lhsp[i1][3] - _lhsp[i1][1]*_lhsp[i][4];
        rhs[k][j][i1][m] = rhs[k][j][i1][m] - _lhsp[i1][1]*rhs[k][j][i][m];
        _lhsp[i2][1]   = _lhsp[i2][1] - _lhsp[i2][0]*_lhsp[i][3];
        _lhsp[i2][2]   = _lhsp[i2][2] - _lhsp[i2][0]*_lhsp[i][4];
        rhs[k][j][i2][m] = rhs[k][j][i2][m] - _lhsp[i2][0]*rhs[k][j][i][m];

        m = 4;
        fac1 = 1.0/_lhsm[i][2];
        _lhsm[i][3]    = fac1*_lhsm[i][3];
        _lhsm[i][4]    = fac1*_lhsm[i][4];
        rhs[k][j][i][m]  = fac1*rhs[k][j][i][m];
        _lhsm[i1][2]   = _lhsm[i1][2] - _lhsm[i1][1]*_lhsm[i][3];
        _lhsm[i1][3]   = _lhsm[i1][3] - _lhsm[i1][1]*_lhsm[i][4];
        rhs[k][j][i1][m] = rhs[k][j][i1][m] - _lhsm[i1][1]*rhs[k][j][i][m];
        _lhsm[i2][1]   = _lhsm[i2][1] - _lhsm[i2][0]*_lhsm[i][3];
        _lhsm[i2][2]   = _lhsm[i2][2] - _lhsm[i2][0]*_lhsm[i][4];
        rhs[k][j][i2][m] = rhs[k][j][i2][m] - _lhsm[i2][0]*rhs[k][j][i][m];
      }
    }

    //---------------------------------------------------------------------
    // And again the last two rows separately
    //---------------------------------------------------------------------
    if (j >= 1 && j <= ny2) {
      i  = grid_points[0]-2;
      i1 = grid_points[0]-1;

      m = 3;
      fac1 = 1.0/_lhsp[i][2];
      _lhsp[i][3]    = fac1*_lhsp[i][3];
      _lhsp[i][4]    = fac1*_lhsp[i][4];
      rhs[k][j][i][m]  = fac1*rhs[k][j][i][m];
      _lhsp[i1][2]   = _lhsp[i1][2] - _lhsp[i1][1]*_lhsp[i][3];
      _lhsp[i1][3]   = _lhsp[i1][3] - _lhsp[i1][1]*_lhsp[i][4];
      rhs[k][j][i1][m] = rhs[k][j][i1][m] - _lhsp[i1][1]*rhs[k][j][i][m];

      m = 4;
      fac1 = 1.0/_lhsm[i][2];
      _lhsm[i][3]    = fac1*_lhsm[i][3];
      _lhsm[i][4]    = fac1*_lhsm[i][4];
      rhs[k][j][i][m]  = fac1*rhs[k][j][i][m];
      _lhsm[i1][2]   = _lhsm[i1][2] - _lhsm[i1][1]*_lhsm[i][3];
      _lhsm[i1][3]   = _lhsm[i1][3] - _lhsm[i1][1]*_lhsm[i][4];
      rhs[k][j][i1][m] = rhs[k][j][i1][m] - _lhsm[i1][1]*rhs[k][j][i][m];

      //---------------------------------------------------------------------
      // Scale the last row immediately
      //---------------------------------------------------------------------
      rhs[k][j][i1][3] = rhs[k][j][i1][3]/_lhsp[i1][2];
      rhs[k][j][i1][4] = rhs[k][j][i1][4]/_lhsm[i1][2];
    }

    //---------------------------------------------------------------------
    // BACKSUBSTITUTION 
    //---------------------------------------------------------------------
    if (j >= 1 && j <= ny2) {
      i  = grid_points[0]-2;
      i1 = grid_points[0]-1;
      for (m = 0; m < 3; m++) {
        rhs[k][j][i][m] = rhs[k][j][i][m] - _lhs[i][3]*rhs[k][j][i1][m];
      }

      rhs[k][j][i][3] = rhs[k][j][i][3] - _lhsp[i][3]*rhs[k][j][i1][3];
      rhs[k][j][i][4] = rhs[k][j][i][4] - _lhsm[i][3]*rhs[k][j][i1][4];
    }

    //---------------------------------------------------------------------
    // The first three factors
    //---------------------------------------------------------------------
    if (j >= 1 && j <= ny2) {
      for (i = grid_points[0]-3; i >= 0; i--) {
        i1 = i + 1;
        i2 = i + 2;
        for (m = 0; m < 3; m++) {
          rhs[k][j][i][m] = rhs[k][j][i][m] - 
                            _lhs[i][3]*rhs[k][j][i1][m] -
                            _lhs[i][4]*rhs[k][j][i2][m];
        }

        //-------------------------------------------------------------------
        // And the remaining two
        //-------------------------------------------------------------------
        rhs[k][j][i][3] = rhs[k][j][i][3] - 
                          _lhsp[i][3]*rhs[k][j][i1][3] -
                          _lhsp[i][4]*rhs[k][j][i2][3];
        rhs[k][j][i][4] = rhs[k][j][i][4] - 
                          _lhsm[i][3]*rhs[k][j][i1][4] -
                          _lhsm[i][4]*rhs[k][j][i2][4];
      }
    }
  }
}

void x_solve() {
  if (timeron) timer_start(t_xsolve);
  x_solve_kernel <<< gridDimZY, blockDimZY >>> (
    grid_points,
    nx2, ny2, nz2,
    us, rho_i, speed, rhs,
    cv, rhon,
    dttx1, dttx2, comz1, comz4, comz5, comz6, c2dttx1
  );
  if (timeron) timer_stop(t_xsolve);
  assert(cudaSuccess == cudaDeviceSynchronize());

  //---------------------------------------------------------------------
  // Do the block-diagonal inversion          
  //---------------------------------------------------------------------
  ninvr();
}
