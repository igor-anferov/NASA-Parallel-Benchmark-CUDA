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

#ifndef NEED_CUDA

#include <math.h>
#include "header.h"

void compute_rhs()
{
  int i, j, k, m;
  double aux, rho_inv, uijk, up1, um1, vijk, vp1, vm1, wijk, wp1, wm1;


  if (timeron) timer_start(t_rhs);
  //---------------------------------------------------------------------
  // compute the reciprocal of density, and the kinetic energy, 
  // and the speed of sound. 
  //---------------------------------------------------------------------
  for (k = 0; k <= grid_points[2]-1; k++) {
    for (j = 0; j <= grid_points[1]-1; j++) {
      for (i = 0; i <= grid_points[0]-1; i++) {
        rho_inv = 1.0/u[k][0][j][i];
        rho_i[k][j][i] = rho_inv;
        us[k][j][i] = u[k][1][j][i] * rho_inv;
        vs[k][j][i] = u[k][2][j][i] * rho_inv;
        ws[k][j][i] = u[k][3][j][i] * rho_inv;
        square[k][j][i] = 0.5* (
            u[k][1][j][i]*u[k][1][j][i] + 
            u[k][2][j][i]*u[k][2][j][i] +
            u[k][3][j][i]*u[k][3][j][i] ) * rho_inv;
        qs[k][j][i] = square[k][j][i] * rho_inv;
        //-------------------------------------------------------------------
        // (don't need speed and ainx until the lhs computation)
        //-------------------------------------------------------------------
        aux = c1c2*rho_inv* (u[k][4][j][i] - square[k][j][i]);
        speed[k][j][i] = sqrt(aux);
      }
    }
  }

  //---------------------------------------------------------------------
  // copy the exact forcing term to the right hand side;  because 
  // this forcing term is known, we can store it on the whole grid
  // including the boundary                   
  //---------------------------------------------------------------------
  for (k = 0; k <= grid_points[2]-1; k++) {
    for (j = 0; j <= grid_points[1]-1; j++) {
      for (i = 0; i <= grid_points[0]-1; i++) {
        for (m = 0; m < 5; m++) {
          rhs[k][m][j][i] = forcing[k][m][j][i];
        }
      }
    }
  }

  //---------------------------------------------------------------------
  // compute xi-direction fluxes 
  //---------------------------------------------------------------------
  if (timeron) timer_start(t_rhsx);
  for (k = 1; k <= nz2; k++) {
    for (j = 1; j <= ny2; j++) {
      for (i = 1; i <= nx2; i++) {
        uijk = us[k][j][i];
        up1  = us[k][j][i+1];
        um1  = us[k][j][i-1];

        rhs[k][0][j][i] = rhs[k][0][j][i] + dx1tx1 * 
          (u[k][0][j][i+1] - 2.0*u[k][0][j][i] + u[k][0][j][i-1]) -
          tx2 * (u[k][1][j][i+1] - u[k][1][j][i-1]);

        rhs[k][1][j][i] = rhs[k][1][j][i] + dx2tx1 * 
          (u[k][1][j][i+1] - 2.0*u[k][1][j][i] + u[k][1][j][i-1]) +
          xxcon2*con43 * (up1 - 2.0*uijk + um1) -
          tx2 * (u[k][1][j][i+1]*up1 - u[k][1][j][i-1]*um1 +
                (u[k][4][j][i+1] - square[k][j][i+1] -
                 u[k][4][j][i-1] + square[k][j][i-1]) * c2);

        rhs[k][2][j][i] = rhs[k][2][j][i] + dx3tx1 * 
          (u[k][2][j][i+1] - 2.0*u[k][2][j][i] + u[k][2][j][i-1]) +
          xxcon2 * (vs[k][j][i+1] - 2.0*vs[k][j][i] + vs[k][j][i-1]) -
          tx2 * (u[k][2][j][i+1]*up1 - u[k][2][j][i-1]*um1);

        rhs[k][3][j][i] = rhs[k][3][j][i] + dx4tx1 * 
          (u[k][3][j][i+1] - 2.0*u[k][3][j][i] + u[k][3][j][i-1]) +
          xxcon2 * (ws[k][j][i+1] - 2.0*ws[k][j][i] + ws[k][j][i-1]) -
          tx2 * (u[k][3][j][i+1]*up1 - u[k][3][j][i-1]*um1);

        rhs[k][4][j][i] = rhs[k][4][j][i] + dx5tx1 * 
          (u[k][4][j][i+1] - 2.0*u[k][4][j][i] + u[k][4][j][i-1]) +
          xxcon3 * (qs[k][j][i+1] - 2.0*qs[k][j][i] + qs[k][j][i-1]) +
          xxcon4 * (up1*up1 -       2.0*uijk*uijk + um1*um1) +
          xxcon5 * (u[k][4][j][i+1]*rho_i[k][j][i+1] - 
                2.0*u[k][4][j][i]*rho_i[k][j][i] +
                    u[k][4][j][i-1]*rho_i[k][j][i-1]) -
          tx2 * ( (c1*u[k][4][j][i+1] - c2*square[k][j][i+1])*up1 -
                  (c1*u[k][4][j][i-1] - c2*square[k][j][i-1])*um1 );
      }
    }

    //---------------------------------------------------------------------
    // add fourth order xi-direction dissipation               
    //---------------------------------------------------------------------
    for (j = 1; j <= ny2; j++) {
      i = 1;
      for (m = 0; m < 5; m++) {
        rhs[k][m][j][i] = rhs[k][m][j][i]- dssp * 
          (5.0*u[k][m][j][i] - 4.0*u[k][m][j][i+1] + u[k][m][j][i+2]);
      }

      i = 2;
      for (m = 0; m < 5; m++) {
        rhs[k][m][j][i] = rhs[k][m][j][i] - dssp * 
          (-4.0*u[k][m][j][i-1] + 6.0*u[k][m][j][i] -
            4.0*u[k][m][j][i+1] + u[k][m][j][i+2]);
      }
    }

    for (j = 1; j <= ny2; j++) {
      for (i = 3; i <= nx2-2; i++) {
        for (m = 0; m < 5; m++) {
          rhs[k][m][j][i] = rhs[k][m][j][i] - dssp * 
            ( u[k][m][j][i-2] - 4.0*u[k][m][j][i-1] + 
            6.0*u[k][m][j][i] - 4.0*u[k][m][j][i+1] + 
              u[k][m][j][i+2] );
        }
      }
    }

    for (j = 1; j <= ny2; j++) {
      i = nx2-1;
      for (m = 0; m < 5; m++) {
        rhs[k][m][j][i] = rhs[k][m][j][i] - dssp *
          ( u[k][m][j][i-2] - 4.0*u[k][m][j][i-1] + 
          6.0*u[k][m][j][i] - 4.0*u[k][m][j][i+1] );
      }

      i = nx2;
      for (m = 0; m < 5; m++) {
        rhs[k][m][j][i] = rhs[k][m][j][i] - dssp *
          ( u[k][m][j][i-2] - 4.0*u[k][m][j][i-1] + 5.0*u[k][m][j][i] );
      }
    }
  }
  if (timeron) timer_stop(t_rhsx);

  //---------------------------------------------------------------------
  // compute eta-direction fluxes 
  //---------------------------------------------------------------------
  if (timeron) timer_start(t_rhsy);
  for (k = 1; k <= nz2; k++) {
    for (j = 1; j <= ny2; j++) {
      for (i = 1; i <= nx2; i++) {
        vijk = vs[k][j][i];
        vp1  = vs[k][j+1][i];
        vm1  = vs[k][j-1][i];

        rhs[k][0][j][i] = rhs[k][0][j][i] + dy1ty1 * 
          (u[k][0][j+1][i] - 2.0*u[k][0][j][i] + u[k][0][j-1][i]) -
          ty2 * (u[k][2][j+1][i] - u[k][2][j-1][i]);

        rhs[k][1][j][i] = rhs[k][1][j][i] + dy2ty1 * 
          (u[k][1][j+1][i] - 2.0*u[k][1][j][i] + u[k][1][j-1][i]) +
          yycon2 * (us[k][j+1][i] - 2.0*us[k][j][i] + us[k][j-1][i]) -
          ty2 * (u[k][1][j+1][i]*vp1 - u[k][1][j-1][i]*vm1);

        rhs[k][2][j][i] = rhs[k][2][j][i] + dy3ty1 * 
          (u[k][2][j+1][i] - 2.0*u[k][2][j][i] + u[k][2][j-1][i]) +
          yycon2*con43 * (vp1 - 2.0*vijk + vm1) -
          ty2 * (u[k][2][j+1][i]*vp1 - u[k][2][j-1][i]*vm1 +
                (u[k][4][j+1][i] - square[k][j+1][i] - 
                 u[k][4][j-1][i] + square[k][j-1][i]) * c2);

        rhs[k][3][j][i] = rhs[k][3][j][i] + dy4ty1 * 
          (u[k][3][j+1][i] - 2.0*u[k][3][j][i] + u[k][3][j-1][i]) +
          yycon2 * (ws[k][j+1][i] - 2.0*ws[k][j][i] + ws[k][j-1][i]) -
          ty2 * (u[k][3][j+1][i]*vp1 - u[k][3][j-1][i]*vm1);

        rhs[k][4][j][i] = rhs[k][4][j][i] + dy5ty1 * 
          (u[k][4][j+1][i] - 2.0*u[k][4][j][i] + u[k][4][j-1][i]) +
          yycon3 * (qs[k][j+1][i] - 2.0*qs[k][j][i] + qs[k][j-1][i]) +
          yycon4 * (vp1*vp1       - 2.0*vijk*vijk + vm1*vm1) +
          yycon5 * (u[k][4][j+1][i]*rho_i[k][j+1][i] - 
                  2.0*u[k][4][j][i]*rho_i[k][j][i] +
                    u[k][4][j-1][i]*rho_i[k][j-1][i]) -
          ty2 * ((c1*u[k][4][j+1][i] - c2*square[k][j+1][i]) * vp1 -
                 (c1*u[k][4][j-1][i] - c2*square[k][j-1][i]) * vm1);
      }
    }

    //---------------------------------------------------------------------
    // add fourth order eta-direction dissipation         
    //---------------------------------------------------------------------
    j = 1;
    for (i = 1; i <= nx2; i++) {
      for (m = 0; m < 5; m++) {
        rhs[k][m][j][i] = rhs[k][m][j][i]- dssp * 
          ( 5.0*u[k][m][j][i] - 4.0*u[k][m][j+1][i] + u[k][m][j+2][i]);
      }
    }

    j = 2;
    for (i = 1; i <= nx2; i++) {
      for (m = 0; m < 5; m++) {
        rhs[k][m][j][i] = rhs[k][m][j][i] - dssp * 
          (-4.0*u[k][m][j-1][i] + 6.0*u[k][m][j][i] -
            4.0*u[k][m][j+1][i] + u[k][m][j+2][i]);
      }
    }

    for (j = 3; j <= ny2-2; j++) {
      for (i = 1; i <= nx2; i++) {
        for (m = 0; m < 5; m++) {
          rhs[k][m][j][i] = rhs[k][m][j][i] - dssp * 
            ( u[k][m][j-2][i] - 4.0*u[k][m][j-1][i] + 
            6.0*u[k][m][j][i] - 4.0*u[k][m][j+1][i] + 
              u[k][m][j+2][i] );
        }
      }
    }

    j = ny2-1;
    for (i = 1; i <= nx2; i++) {
      for (m = 0; m < 5; m++) {
        rhs[k][m][j][i] = rhs[k][m][j][i] - dssp *
          ( u[k][m][j-2][i] - 4.0*u[k][m][j-1][i] + 
          6.0*u[k][m][j][i] - 4.0*u[k][m][j+1][i] );
      }
    }

    j = ny2;
    for (i = 1; i <= nx2; i++) {
      for (m = 0; m < 5; m++) {
        rhs[k][m][j][i] = rhs[k][m][j][i] - dssp *
          ( u[k][m][j-2][i] - 4.0*u[k][m][j-1][i] + 5.0*u[k][m][j][i] );
      }
    }
  }
  if (timeron) timer_stop(t_rhsy);

  //---------------------------------------------------------------------
  // compute zeta-direction fluxes 
  //---------------------------------------------------------------------
  if (timeron) timer_start(t_rhsz);
  for (k = 1; k <= nz2; k++) {
    for (j = 1; j <= ny2; j++) {
      for (i = 1; i <= nx2; i++) {
        wijk = ws[k][j][i];
        wp1  = ws[k+1][j][i];
        wm1  = ws[k-1][j][i];

        rhs[k][0][j][i] = rhs[k][0][j][i] + dz1tz1 * 
          (u[k+1][0][j][i] - 2.0*u[k][0][j][i] + u[k-1][0][j][i]) -
          tz2 * (u[k+1][3][j][i] - u[k-1][3][j][i]);

        rhs[k][1][j][i] = rhs[k][1][j][i] + dz2tz1 * 
          (u[k+1][1][j][i] - 2.0*u[k][1][j][i] + u[k-1][1][j][i]) +
          zzcon2 * (us[k+1][j][i] - 2.0*us[k][j][i] + us[k-1][j][i]) -
          tz2 * (u[k+1][1][j][i]*wp1 - u[k-1][1][j][i]*wm1);

        rhs[k][2][j][i] = rhs[k][2][j][i] + dz3tz1 * 
          (u[k+1][2][j][i] - 2.0*u[k][2][j][i] + u[k-1][2][j][i]) +
          zzcon2 * (vs[k+1][j][i] - 2.0*vs[k][j][i] + vs[k-1][j][i]) -
          tz2 * (u[k+1][2][j][i]*wp1 - u[k-1][2][j][i]*wm1);

        rhs[k][3][j][i] = rhs[k][3][j][i] + dz4tz1 * 
          (u[k+1][3][j][i] - 2.0*u[k][3][j][i] + u[k-1][3][j][i]) +
          zzcon2*con43 * (wp1 - 2.0*wijk + wm1) -
          tz2 * (u[k+1][3][j][i]*wp1 - u[k-1][3][j][i]*wm1 +
                (u[k+1][4][j][i] - square[k+1][j][i] - 
                 u[k-1][4][j][i] + square[k-1][j][i]) * c2);

        rhs[k][4][j][i] = rhs[k][4][j][i] + dz5tz1 * 
          (u[k+1][4][j][i] - 2.0*u[k][4][j][i] + u[k-1][4][j][i]) +
          zzcon3 * (qs[k+1][j][i] - 2.0*qs[k][j][i] + qs[k-1][j][i]) +
          zzcon4 * (wp1*wp1 - 2.0*wijk*wijk + wm1*wm1) +
          zzcon5 * (u[k+1][4][j][i]*rho_i[k+1][j][i] - 
                  2.0*u[k][4][j][i]*rho_i[k][j][i] +
                    u[k-1][4][j][i]*rho_i[k-1][j][i]) -
          tz2 * ((c1*u[k+1][4][j][i] - c2*square[k+1][j][i])*wp1 -
                 (c1*u[k-1][4][j][i] - c2*square[k-1][j][i])*wm1);
      }
    }
  }

  //---------------------------------------------------------------------
  // add fourth order zeta-direction dissipation                
  //---------------------------------------------------------------------
  k = 1;
  for (j = 1; j <= ny2; j++) {
    for (i = 1; i <= nx2; i++) {
      for (m = 0; m < 5; m++) {
        rhs[k][m][j][i] = rhs[k][m][j][i]- dssp * 
          (5.0*u[k][m][j][i] - 4.0*u[k+1][m][j][i] + u[k+2][m][j][i]);
      }
    }
  }

  k = 2;
  for (j = 1; j <= ny2; j++) {
    for (i = 1; i <= nx2; i++) {
      for (m = 0; m < 5; m++) {
        rhs[k][m][j][i] = rhs[k][m][j][i] - dssp * 
          (-4.0*u[k-1][m][j][i] + 6.0*u[k][m][j][i] -
            4.0*u[k+1][m][j][i] + u[k+2][m][j][i]);
      }
    }
  }

  for (k = 3; k <= nz2-2; k++) {
    for (j = 1; j <= ny2; j++) {
      for (i = 1; i <= nx2; i++) {
        for (m = 0; m < 5; m++) {
          rhs[k][m][j][i] = rhs[k][m][j][i] - dssp * 
            ( u[k-2][m][j][i] - 4.0*u[k-1][m][j][i] + 
            6.0*u[k][m][j][i] - 4.0*u[k+1][m][j][i] + 
              u[k+2][m][j][i] );
        }
      }
    }
  }

  k = nz2-1;
  for (j = 1; j <= ny2; j++) {
    for (i = 1; i <= nx2; i++) {
      for (m = 0; m < 5; m++) {
        rhs[k][m][j][i] = rhs[k][m][j][i] - dssp *
          ( u[k-2][m][j][i] - 4.0*u[k-1][m][j][i] + 
          6.0*u[k][m][j][i] - 4.0*u[k+1][m][j][i] );
      }
    }
  }

  k = nz2;
  for (j = 1; j <= ny2; j++) {
    for (i = 1; i <= nx2; i++) {
      for (m = 0; m < 5; m++) {
        rhs[k][m][j][i] = rhs[k][m][j][i] - dssp *
          ( u[k-2][m][j][i] - 4.0*u[k-1][m][j][i] + 5.0*u[k][m][j][i] );
      }
    }
  }
  if (timeron) timer_stop(t_rhsz);

  for (k = 1; k <= nz2; k++) {
    for (j = 1; j <= ny2; j++) {
      for (i = 1; i <= nx2; i++) {
        for (m = 0; m < 5; m++) {
          rhs[k][m][j][i] = rhs[k][m][j][i] * dt;
        }
      }
    }
  }
  if (timeron) timer_stop(t_rhs);
}

#endif
