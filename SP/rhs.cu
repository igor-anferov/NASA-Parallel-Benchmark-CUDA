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

#include <math.h>
#include "header.h"

__global__ void rhs_start_kernel(
    int* grid_points,
    double (*u)[KMAX][JMAXP+1][IMAXP+1],
    double (*us)[JMAXP+1][IMAXP+1],
    double (*vs)[JMAXP+1][IMAXP+1],
    double (*ws)[JMAXP+1][IMAXP+1],
    double (*qs)[JMAXP+1][IMAXP+1],
    double (*rho_i)[JMAXP+1][IMAXP+1],
    double (*speed)[JMAXP+1][IMAXP+1],
    double (*square)[JMAXP+1][IMAXP+1],
    double (*rhs)[KMAX][JMAXP+1][IMAXP+1],
    double (*forcing)[KMAX][JMAXP+1][IMAXP+1]
) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int k = blockDim.z * blockIdx.z + threadIdx.z;

  int m;
  double rho_inv, aux;
  if (k >= 0 && k <= grid_points[2]-1) {
    if (j >= 0 && j <= grid_points[1]-1) {
      if (i >= 0 && i <= grid_points[0]-1) {
        rho_inv = 1.0/u[0][k][j][i];
        rho_i[k][j][i] = rho_inv;
        us[k][j][i] = u[1][k][j][i] * rho_inv;
        vs[k][j][i] = u[2][k][j][i] * rho_inv;
        ws[k][j][i] = u[3][k][j][i] * rho_inv;
        square[k][j][i] = 0.5* (
            u[1][k][j][i]*u[1][k][j][i] + 
            u[2][k][j][i]*u[2][k][j][i] +
            u[3][k][j][i]*u[3][k][j][i] ) * rho_inv;
        qs[k][j][i] = square[k][j][i] * rho_inv;
        //-------------------------------------------------------------------
        // (don't need speed and ainx until the lhs computation)
        //-------------------------------------------------------------------
        aux = c1c2*rho_inv* (u[4][k][j][i] - square[k][j][i]);
        speed[k][j][i] = sqrt(aux);
      }
    }
  }

  //---------------------------------------------------------------------
  // copy the exact forcing term to the right hand side;  because 
  // this forcing term is known, we can store it on the whole grid
  // including the boundary                   
  //---------------------------------------------------------------------
  if (k >= 0 && k <= grid_points[2]-1) {
    if (j >= 0 && j <= grid_points[1]-1) {
      if (i >= 0 && i <= grid_points[0]-1) {
        for (m = 0; m < 5; m++) {
          rhs[m][k][j][i] = forcing[m][k][j][i];
        }
      }
    }
  }
}

__global__ void rhs_xi_kernel(
    int nx2, int ny2, int nz2,
    double (*u)[KMAX][JMAXP+1][IMAXP+1],
    double (*us)[JMAXP+1][IMAXP+1],
    double (*vs)[JMAXP+1][IMAXP+1],
    double (*ws)[JMAXP+1][IMAXP+1],
    double (*qs)[JMAXP+1][IMAXP+1],
    double (*rho_i)[JMAXP+1][IMAXP+1],
    double (*square)[JMAXP+1][IMAXP+1],
    double (*rhs)[KMAX][JMAXP+1][IMAXP+1],
    double dx1tx1, double dx2tx1, double dx3tx1, double dx4tx1, double dx5tx1, double tx2,
    double xxcon2, double xxcon3, double xxcon4, double xxcon5
) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int k = blockDim.z * blockIdx.z + threadIdx.z;

  int m;
  double uijk, up1, um1;
  if (k >= 1 && k <= nz2) {
    if (j >= 1 && j <= ny2) {
      if (i >= 1 && i <= nx2) {
        uijk = us[k][j][i];
        up1  = us[k][j][i+1];
        um1  = us[k][j][i-1];

        rhs[0][k][j][i] = rhs[0][k][j][i] + dx1tx1 * 
          (u[0][k][j][i+1] - 2.0*u[0][k][j][i] + u[0][k][j][i-1]) -
          tx2 * (u[1][k][j][i+1] - u[1][k][j][i-1]);

        rhs[1][k][j][i] = rhs[1][k][j][i] + dx2tx1 * 
          (u[1][k][j][i+1] - 2.0*u[1][k][j][i] + u[1][k][j][i-1]) +
          xxcon2*con43 * (up1 - 2.0*uijk + um1) -
          tx2 * (u[1][k][j][i+1]*up1 - u[1][k][j][i-1]*um1 +
                (u[4][k][j][i+1] - square[k][j][i+1] -
                 u[4][k][j][i-1] + square[k][j][i-1]) * c2);

        rhs[2][k][j][i] = rhs[2][k][j][i] + dx3tx1 * 
          (u[2][k][j][i+1] - 2.0*u[2][k][j][i] + u[2][k][j][i-1]) +
          xxcon2 * (vs[k][j][i+1] - 2.0*vs[k][j][i] + vs[k][j][i-1]) -
          tx2 * (u[2][k][j][i+1]*up1 - u[2][k][j][i-1]*um1);

        rhs[3][k][j][i] = rhs[3][k][j][i] + dx4tx1 * 
          (u[3][k][j][i+1] - 2.0*u[3][k][j][i] + u[3][k][j][i-1]) +
          xxcon2 * (ws[k][j][i+1] - 2.0*ws[k][j][i] + ws[k][j][i-1]) -
          tx2 * (u[3][k][j][i+1]*up1 - u[3][k][j][i-1]*um1);

        rhs[4][k][j][i] = rhs[4][k][j][i] + dx5tx1 * 
          (u[4][k][j][i+1] - 2.0*u[4][k][j][i] + u[4][k][j][i-1]) +
          xxcon3 * (qs[k][j][i+1] - 2.0*qs[k][j][i] + qs[k][j][i-1]) +
          xxcon4 * (up1*up1 -       2.0*uijk*uijk + um1*um1) +
          xxcon5 * (u[4][k][j][i+1]*rho_i[k][j][i+1] - 
                2.0*u[4][k][j][i]*rho_i[k][j][i] +
                    u[4][k][j][i-1]*rho_i[k][j][i-1]) -
          tx2 * ( (c1*u[4][k][j][i+1] - c2*square[k][j][i+1])*up1 -
                  (c1*u[4][k][j][i-1] - c2*square[k][j][i-1])*um1 );
      }
    }

    //---------------------------------------------------------------------
    // add fourth order xi-direction dissipation               
    //---------------------------------------------------------------------
    if (j >= 1 && j <= ny2) {
      if (i == 1)
          for (m = 0; m < 5; m++) {
            rhs[m][k][j][i] = rhs[m][k][j][i]- dssp * 
              (5.0*u[m][k][j][i] - 4.0*u[m][k][j][i+1] + u[m][k][j][i+2]);
          }

      if (i == 2)
          for (m = 0; m < 5; m++) {
            rhs[m][k][j][i] = rhs[m][k][j][i] - dssp * 
              (-4.0*u[m][k][j][i-1] + 6.0*u[m][k][j][i] -
                4.0*u[m][k][j][i+1] + u[m][k][j][i+2]);
          }
    }

    if (j >= 1 && j <= ny2) {
      if (i >= 3 && i <= nx2-2) {
        for (m = 0; m < 5; m++) {
          rhs[m][k][j][i] = rhs[m][k][j][i] - dssp * 
            ( u[m][k][j][i-2] - 4.0*u[m][k][j][i-1] + 
            6.0*u[m][k][j][i] - 4.0*u[m][k][j][i+1] + 
              u[m][k][j][i+2] );
        }
      }
    }

    if (j >= 1 && j <= ny2) {
      if (i == nx2-1)
          for (m = 0; m < 5; m++) {
            rhs[m][k][j][i] = rhs[m][k][j][i] - dssp *
              ( u[m][k][j][i-2] - 4.0*u[m][k][j][i-1] + 
              6.0*u[m][k][j][i] - 4.0*u[m][k][j][i+1] );
          }

      if (i == nx2)
          for (m = 0; m < 5; m++) {
            rhs[m][k][j][i] = rhs[m][k][j][i] - dssp *
              ( u[m][k][j][i-2] - 4.0*u[m][k][j][i-1] + 5.0*u[m][k][j][i] );
          }
    }
  }
}

__global__ void rhs_eta_kernel(
    int nx2, int ny2, int nz2,
    double (*u)[KMAX][JMAXP+1][IMAXP+1],
    double (*us)[JMAXP+1][IMAXP+1],
    double (*vs)[JMAXP+1][IMAXP+1],
    double (*ws)[JMAXP+1][IMAXP+1],
    double (*qs)[JMAXP+1][IMAXP+1],
    double (*rho_i)[JMAXP+1][IMAXP+1],
    double (*square)[JMAXP+1][IMAXP+1],
    double (*rhs)[KMAX][JMAXP+1][IMAXP+1],
    double dy1ty1, double dy2ty1, double dy3ty1, double dy4ty1, double dy5ty1, double ty2,
    double yycon2, double yycon3, double yycon4, double yycon5
) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int k = blockDim.z * blockIdx.z + threadIdx.z;

  int m;
  double vijk, vp1, vm1;
  if (k >= 1 && k <= nz2) {
    if (j >= 1 && j <= ny2) {
      if (i >= 1 && i <= nx2) {
        vijk = vs[k][j][i];
        vp1  = vs[k][j+1][i];
        vm1  = vs[k][j-1][i];

        rhs[0][k][j][i] = rhs[0][k][j][i] + dy1ty1 * 
          (u[0][k][j+1][i] - 2.0*u[0][k][j][i] + u[0][k][j-1][i]) -
          ty2 * (u[2][k][j+1][i] - u[2][k][j-1][i]);

        rhs[1][k][j][i] = rhs[1][k][j][i] + dy2ty1 * 
          (u[1][k][j+1][i] - 2.0*u[1][k][j][i] + u[1][k][j-1][i]) +
          yycon2 * (us[k][j+1][i] - 2.0*us[k][j][i] + us[k][j-1][i]) -
          ty2 * (u[1][k][j+1][i]*vp1 - u[1][k][j-1][i]*vm1);

        rhs[2][k][j][i] = rhs[2][k][j][i] + dy3ty1 * 
          (u[2][k][j+1][i] - 2.0*u[2][k][j][i] + u[2][k][j-1][i]) +
          yycon2*con43 * (vp1 - 2.0*vijk + vm1) -
          ty2 * (u[2][k][j+1][i]*vp1 - u[2][k][j-1][i]*vm1 +
                (u[4][k][j+1][i] - square[k][j+1][i] - 
                 u[4][k][j-1][i] + square[k][j-1][i]) * c2);

        rhs[3][k][j][i] = rhs[3][k][j][i] + dy4ty1 * 
          (u[3][k][j+1][i] - 2.0*u[3][k][j][i] + u[3][k][j-1][i]) +
          yycon2 * (ws[k][j+1][i] - 2.0*ws[k][j][i] + ws[k][j-1][i]) -
          ty2 * (u[3][k][j+1][i]*vp1 - u[3][k][j-1][i]*vm1);

        rhs[4][k][j][i] = rhs[4][k][j][i] + dy5ty1 * 
          (u[4][k][j+1][i] - 2.0*u[4][k][j][i] + u[4][k][j-1][i]) +
          yycon3 * (qs[k][j+1][i] - 2.0*qs[k][j][i] + qs[k][j-1][i]) +
          yycon4 * (vp1*vp1       - 2.0*vijk*vijk + vm1*vm1) +
          yycon5 * (u[4][k][j+1][i]*rho_i[k][j+1][i] - 
                  2.0*u[4][k][j][i]*rho_i[k][j][i] +
                    u[4][k][j-1][i]*rho_i[k][j-1][i]) -
          ty2 * ((c1*u[4][k][j+1][i] - c2*square[k][j+1][i]) * vp1 -
                 (c1*u[4][k][j-1][i] - c2*square[k][j-1][i]) * vm1);
      }
    }

    //---------------------------------------------------------------------
    // add fourth order eta-direction dissipation         
    //---------------------------------------------------------------------
    if (j == 1)
        if (i >= 1 && i <= nx2) {
          for (m = 0; m < 5; m++) {
            rhs[m][k][j][i] = rhs[m][k][j][i]- dssp * 
              ( 5.0*u[m][k][j][i] - 4.0*u[m][k][j+1][i] + u[m][k][j+2][i]);
          }
        }

    if (j == 2)
        if (i >= 1 && i <= nx2) {
          for (m = 0; m < 5; m++) {
            rhs[m][k][j][i] = rhs[m][k][j][i] - dssp * 
              (-4.0*u[m][k][j-1][i] + 6.0*u[m][k][j][i] -
                4.0*u[m][k][j+1][i] + u[m][k][j+2][i]);
          }
        }

    if (j >= 3 && j <= ny2-2) {
      if (i >= 1 && i <= nx2) {
        for (m = 0; m < 5; m++) {
          rhs[m][k][j][i] = rhs[m][k][j][i] - dssp * 
            ( u[m][k][j-2][i] - 4.0*u[m][k][j-1][i] + 
            6.0*u[m][k][j][i] - 4.0*u[m][k][j+1][i] + 
              u[m][k][j+2][i] );
        }
      }
    }

    if (j == ny2-1)
        if (i >= 1 && i <= nx2) {
          for (m = 0; m < 5; m++) {
            rhs[m][k][j][i] = rhs[m][k][j][i] - dssp *
              ( u[m][k][j-2][i] - 4.0*u[m][k][j-1][i] + 
              6.0*u[m][k][j][i] - 4.0*u[m][k][j+1][i] );
          }
        }

    if (j == ny2)
        if (i >= 1 && i <= nx2) {
          for (m = 0; m < 5; m++) {
            rhs[m][k][j][i] = rhs[m][k][j][i] - dssp *
              ( u[m][k][j-2][i] - 4.0*u[m][k][j-1][i] + 5.0*u[m][k][j][i] );
          }
        }
  }
}

__global__ void rhs_zeta_kernel(
    int nx2, int ny2, int nz2,
    double (*u)[KMAX][JMAXP+1][IMAXP+1],
    double (*us)[JMAXP+1][IMAXP+1],
    double (*vs)[JMAXP+1][IMAXP+1],
    double (*ws)[JMAXP+1][IMAXP+1],
    double (*qs)[JMAXP+1][IMAXP+1],
    double (*rho_i)[JMAXP+1][IMAXP+1],
    double (*square)[JMAXP+1][IMAXP+1],
    double (*rhs)[KMAX][JMAXP+1][IMAXP+1],
    double dz1tz1, double dz2tz1, double dz3tz1, double dz4tz1, double dz5tz1, double tz2,
    double zzcon2, double zzcon3, double zzcon4, double zzcon5
) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int k = blockDim.z * blockIdx.z + threadIdx.z;

  int m;
  double wijk, wp1, wm1;
  if (k >= 1 && k <= nz2) {
    if (j >= 1 && j <= ny2) {
      if (i >= 1 && i <= nx2) {
        wijk = ws[k][j][i];
        wp1  = ws[k+1][j][i];
        wm1  = ws[k-1][j][i];

        rhs[0][k][j][i] = rhs[0][k][j][i] + dz1tz1 * 
          (u[0][k+1][j][i] - 2.0*u[0][k][j][i] + u[0][k-1][j][i]) -
          tz2 * (u[3][k+1][j][i] - u[3][k-1][j][i]);

        rhs[1][k][j][i] = rhs[1][k][j][i] + dz2tz1 * 
          (u[1][k+1][j][i] - 2.0*u[1][k][j][i] + u[1][k-1][j][i]) +
          zzcon2 * (us[k+1][j][i] - 2.0*us[k][j][i] + us[k-1][j][i]) -
          tz2 * (u[1][k+1][j][i]*wp1 - u[1][k-1][j][i]*wm1);

        rhs[2][k][j][i] = rhs[2][k][j][i] + dz3tz1 * 
          (u[2][k+1][j][i] - 2.0*u[2][k][j][i] + u[2][k-1][j][i]) +
          zzcon2 * (vs[k+1][j][i] - 2.0*vs[k][j][i] + vs[k-1][j][i]) -
          tz2 * (u[2][k+1][j][i]*wp1 - u[2][k-1][j][i]*wm1);

        rhs[3][k][j][i] = rhs[3][k][j][i] + dz4tz1 * 
          (u[3][k+1][j][i] - 2.0*u[3][k][j][i] + u[3][k-1][j][i]) +
          zzcon2*con43 * (wp1 - 2.0*wijk + wm1) -
          tz2 * (u[3][k+1][j][i]*wp1 - u[3][k-1][j][i]*wm1 +
                (u[4][k+1][j][i] - square[k+1][j][i] - 
                 u[4][k-1][j][i] + square[k-1][j][i]) * c2);

        rhs[4][k][j][i] = rhs[4][k][j][i] + dz5tz1 * 
          (u[4][k+1][j][i] - 2.0*u[4][k][j][i] + u[4][k-1][j][i]) +
          zzcon3 * (qs[k+1][j][i] - 2.0*qs[k][j][i] + qs[k-1][j][i]) +
          zzcon4 * (wp1*wp1 - 2.0*wijk*wijk + wm1*wm1) +
          zzcon5 * (u[4][k+1][j][i]*rho_i[k+1][j][i] - 
                  2.0*u[4][k][j][i]*rho_i[k][j][i] +
                    u[4][k-1][j][i]*rho_i[k-1][j][i]) -
          tz2 * ((c1*u[4][k+1][j][i] - c2*square[k+1][j][i])*wp1 -
                 (c1*u[4][k-1][j][i] - c2*square[k-1][j][i])*wm1);
      }
    }
  }

  //---------------------------------------------------------------------
  // add fourth order zeta-direction dissipation                
  //---------------------------------------------------------------------
  if (k == 1)
      if (j >= 1 && j <= ny2) {
        if (i >= 1 && i <= nx2) {
          for (m = 0; m < 5; m++) {
            rhs[m][k][j][i] = rhs[m][k][j][i]- dssp * 
              (5.0*u[m][k][j][i] - 4.0*u[m][k+1][j][i] + u[m][k+2][j][i]);
          }
        }
      }

  if (k == 2)
      if (j >= 1 && j <= ny2) {
        if (i >= 1 && i <= nx2) {
          for (m = 0; m < 5; m++) {
            rhs[m][k][j][i] = rhs[m][k][j][i] - dssp * 
              (-4.0*u[m][k-1][j][i] + 6.0*u[m][k][j][i] -
                4.0*u[m][k+1][j][i] + u[m][k+2][j][i]);
          }
        }
      }

  if (k >= 3 && k <= nz2-2) {
    if (j >= 1 && j <= ny2) {
      if (i >= 1 && i <= nx2) {
        for (m = 0; m < 5; m++) {
          rhs[m][k][j][i] = rhs[m][k][j][i] - dssp * 
            ( u[m][k-2][j][i] - 4.0*u[m][k-1][j][i] + 
            6.0*u[m][k][j][i] - 4.0*u[m][k+1][j][i] + 
              u[m][k+2][j][i] );
        }
      }
    }
  }

  if (k == nz2-1)
      if (j >= 1 && j <= ny2) {
        if (i >= 1 && i <= nx2) {
          for (m = 0; m < 5; m++) {
            rhs[m][k][j][i] = rhs[m][k][j][i] - dssp *
              ( u[m][k-2][j][i] - 4.0*u[m][k-1][j][i] + 
              6.0*u[m][k][j][i] - 4.0*u[m][k+1][j][i] );
          }
        }
      }

  if (k == nz2)
      if (j >= 1 && j <= ny2) {
        if (i >= 1 && i <= nx2) {
          for (m = 0; m < 5; m++) {
            rhs[m][k][j][i] = rhs[m][k][j][i] - dssp *
              ( u[m][k-2][j][i] - 4.0*u[m][k-1][j][i] + 5.0*u[m][k][j][i] );
          }
        }
      }
}

__global__ void rhs_end_kernel(
    int nx2, int ny2, int nz2,
    double (*u)[KMAX][JMAXP+1][IMAXP+1],
    double (*us)[JMAXP+1][IMAXP+1],
    double (*vs)[JMAXP+1][IMAXP+1],
    double (*ws)[JMAXP+1][IMAXP+1],
    double (*qs)[JMAXP+1][IMAXP+1],
    double (*rho_i)[JMAXP+1][IMAXP+1],
    double (*square)[JMAXP+1][IMAXP+1],
    double (*rhs)[KMAX][JMAXP+1][IMAXP+1],
    double dt
) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int k = blockDim.z * blockIdx.z + threadIdx.z;

  int m;
  if (k >= 1 && k <= nz2) {
    if (j >= 1 && j <= ny2) {
      if (i >= 1 && i <= nx2) {
        for (m = 0; m < 5; m++) {
          rhs[m][k][j][i] = rhs[m][k][j][i] * dt;
        }
      }
    }
  }
}

void compute_rhs()
{
  if (timeron) timer_start(t_rhs);

  rhs_start_kernel <<< gridDim_, blockDim_ >>> (
    device_grid_points, device_u, device_us, device_vs, device_ws, device_qs, device_rho_i, device_speed, device_square, device_rhs, device_forcing 
  );

  //---------------------------------------------------------------------
  // compute xi-direction fluxes 
  //---------------------------------------------------------------------
  if (timeron) timer_start(t_rhsx);
  rhs_xi_kernel <<< gridDim_, blockDim_ >>> (
    nx2, ny2, nz2, device_u, device_us, device_vs, device_ws, device_qs, device_rho_i, device_square, device_rhs, dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1, tx2, xxcon2, xxcon3, xxcon4, xxcon5
  );
  if (timeron) timer_stop(t_rhsx);

  //---------------------------------------------------------------------
  // compute eta-direction fluxes 
  //---------------------------------------------------------------------
  if (timeron) timer_start(t_rhsy);
  rhs_eta_kernel <<< gridDim_, blockDim_ >>> (
    nx2, ny2, nz2, device_u, device_us, device_vs, device_ws, device_qs, device_rho_i, device_square, device_rhs, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1, ty2, yycon2, yycon3, yycon4, yycon5
  );
  if (timeron) timer_stop(t_rhsy);

  //---------------------------------------------------------------------
  // compute zeta-direction fluxes 
  //---------------------------------------------------------------------
  if (timeron) timer_start(t_rhsz);
  rhs_zeta_kernel <<< gridDim_, blockDim_ >>> (
    nx2, ny2, nz2, device_u, device_us, device_vs, device_ws, device_qs, device_rho_i, device_square, device_rhs, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1, tz2, zzcon2, zzcon3, zzcon4, zzcon5
  );
  if (timeron) timer_stop(t_rhsz);

  rhs_end_kernel <<< gridDim_, blockDim_ >>> (
    nx2, ny2, nz2, device_u, device_us, device_vs, device_ws, device_qs, device_rho_i, device_square, device_rhs, dt
  );

//  assert(cudaSuccess == cudaDeviceSynchronize());
  if (timeron) timer_stop(t_rhs);
}
