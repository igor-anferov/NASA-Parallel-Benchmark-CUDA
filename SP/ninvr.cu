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
__global__ void ninvr_kernel(
    dim3 gridOffset,
    int nx2, int ny2, int nz2,
    double (*rhs)/*[KMAX]*/[5][JMAXP+1][IMAXP+1]
) {
  int i = blockDim.x * blockIdx.x + threadIdx.x + gridOffset.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y + gridOffset.y;
  int k = blockDim.z * blockIdx.z + threadIdx.z + gridOffset.z;
  double r1, r2, r3, r4, r5, t1, t2;

  if (k >= 1 && k <= nz2) {
    if (j >= 1 && j <= ny2) {
      if (i >= 1 && i <= nx2) {
        r1 = rhs[k][0][j][i];
        r2 = rhs[k][1][j][i];
        r3 = rhs[k][2][j][i];
        r4 = rhs[k][3][j][i];
        r5 = rhs[k][4][j][i];

        t1 = bt * r3;
        t2 = 0.5 * ( r4 + r5 );

        rhs[k][0][j][i] = -r2;
        rhs[k][1][j][i] =  r1;
        rhs[k][2][j][i] = bt * ( r4 - r5 );
        rhs[k][3][j][i] = -t1 + t2;
        rhs[k][4][j][i] =  t1 + t2;
      }
    }
  }
}

void ninvr() {
  if (timeron) timer_start(t_ninvr);
  ninvr_kernel <<< gridDim_, blockDim_ >>> (
    gridOffset, nx2, ny2, nz2, dev_rhs
  );
  if (timeron) timer_stop(t_ninvr);
}
