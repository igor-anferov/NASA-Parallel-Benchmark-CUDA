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

//---------------------------------------------------------------------
// block-diagonal matrix-vector multiplication                  
//---------------------------------------------------------------------
__global__ void txinvr_kernel(
    dim3 gridOffset,
    int nx2, int ny2, int nz2,
    double (*us     )/*[KMAX]*/[JMAXP+1][IMAXP+1],
    double (*vs     )/*[KMAX]*/[JMAXP+1][IMAXP+1],
    double (*ws     )/*[KMAX]*/[JMAXP+1][IMAXP+1],
    double (*qs     )/*[KMAX]*/[JMAXP+1][IMAXP+1],
    double (*rho_i  )/*[KMAX]*/[JMAXP+1][IMAXP+1],
    double (*speed  )/*[KMAX]*/[JMAXP+1][IMAXP+1],
    double (*rhs    )/*[KMAX]*/[JMAXP+1][IMAXP+1][5]
) {
  int i = blockDim.x * blockIdx.x + threadIdx.x + gridOffset.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y + gridOffset.y;
  int k = blockDim.z * blockIdx.z + threadIdx.z + gridOffset.z;

  double t1, t2, t3, ac, ru1, uu, vv, ww, r1, r2, r3, r4, r5, ac2inv;

  if (k >= 1 && k <= nz2) {
    if (j >= 1 && j <= ny2) {
      if (i >= 1 && i <= nx2) {
        ru1 = rho_i[k][j][i];
        uu = us[k][j][i];
        vv = vs[k][j][i];
        ww = ws[k][j][i];
        ac = speed[k][j][i];
        ac2inv = ac*ac;

        r1 = rhs[k][j][i][0];
        r2 = rhs[k][j][i][1];
        r3 = rhs[k][j][i][2];
        r4 = rhs[k][j][i][3];
        r5 = rhs[k][j][i][4];

        t1 = c2 / ac2inv * ( qs[k][j][i]*r1 - uu*r2  - vv*r3 - ww*r4 + r5 );
        t2 = bt * ru1 * ( uu * r1 - r2 );
        t3 = ( bt * ru1 * ac ) * t1;

        rhs[k][j][i][0] = r1 - t1;
        rhs[k][j][i][1] = - ru1 * ( ww*r1 - r4 );
        rhs[k][j][i][2] =   ru1 * ( vv*r1 - r3 );
        rhs[k][j][i][3] = - t2 + t3;
        rhs[k][j][i][4] =   t2 + t3;
      }
    }
  }
}

void txinvr()
{
  if (timeron) timer_start(t_txinvr);
  txinvr_kernel <<< gridDim_, blockDim_ >>> (
    gridOffset, nx2, ny2, nz2, dev_us, dev_vs, dev_ws, dev_qs, dev_rho_i, dev_speed, dev_rhs
  );
  CHK_CUDA_OK(cudaDeviceSynchronize());
  if (timeron) timer_stop(t_txinvr);
}

