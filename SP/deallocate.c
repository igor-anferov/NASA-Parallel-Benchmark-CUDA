#include "header.h"

#include <assert.h>
#include <stdlib.h>

#define DEALLOCATE(ptr) free(ptr); ptr = NULL

void deallocate()
{
/* common /global/ */
    DEALLOCATE(grid_points);

/* common /fields/ */
    DEALLOCATE(u);
    DEALLOCATE(us);
    DEALLOCATE(vs);
    DEALLOCATE(ws);
    DEALLOCATE(qs);
    DEALLOCATE(rho_i);
    DEALLOCATE(speed);
    DEALLOCATE(square);
    DEALLOCATE(rhs);
    DEALLOCATE(forcing);

/* common /work_1d/ */
    DEALLOCATE(cv);
    DEALLOCATE(rhon);
    DEALLOCATE(rhos);
    DEALLOCATE(rhoq);
    DEALLOCATE(cuf);
    DEALLOCATE(q);
    DEALLOCATE(ue);
    DEALLOCATE(buf);

/* common /work_lhs/ */
    DEALLOCATE(lhs);
    DEALLOCATE(lhsp);
    DEALLOCATE(lhsm);
}
