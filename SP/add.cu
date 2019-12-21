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
// addition of update to the vector u
//---------------------------------------------------------------------
__global__ void add_kernel(
    dim3 gridOffset,
    int nx2, int ny2, int nz2,
    double (*u  )/*[KMAX]*/[5][JMAXP+1][IMAXP+1],
    double (*rhs)/*[KMAX]*/[5][JMAXP+1][IMAXP+1]
) {
  int i = blockDim.x * blockIdx.x + threadIdx.x + gridOffset.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y + gridOffset.y;
  int k = blockDim.z * blockIdx.z + threadIdx.z + gridOffset.z;
  int m;

  if (k >= 1 && k <= nz2) {
    if (j >= 1 && j <= ny2) {
      if (i >= 1 && i <= nx2) {
#pragma unroll
        for (m = 0; m < 5; m++) {
          u[k][m][j][i] = u[k][m][j][i] + rhs[k][m][j][i];
        }
      }
    }
  }
}

void add()
{
  if (timeron) timer_start(t_add);
  add_kernel <<< gridDim_, blockDim_ >>> (
    gridOffset, nx2, ny2, nz2, dev_u, dev_rhs
  );
  if (timeron) timer_stop(t_add);
}
