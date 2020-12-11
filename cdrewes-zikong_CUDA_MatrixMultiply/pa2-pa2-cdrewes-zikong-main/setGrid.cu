
#include "mytypes.h"
#include <stdio.h>


void setGrid(int n, dim3 &blockDim, dim3 &gridDim)
{
    // Split the matrix into (n / BLOCKTILE_M) blocks
    gridDim.y = n / BLOCKTILE_M;

    /*
     * We now are simultanously calculating BLOCKTILE_M cols of c
     * Thus we can subdivided the x direction further past the 
     *    division into BLOCKTILE_N segments
     */
    gridDim.x = n / (BLOCKTILE_M * BLOCKTILE_N);

    // Pad the grid
    if(n % BLOCKTILE_M != 0)
      //gridDim.x++;
      gridDim.y++;
    //if(n % (BLOCKTILE_M * BLOCKTILE_N) != 0)
    if(n % (BLOCKTILE_N * BLOCKTILE_M) != 0)
      //gridDim.y++;
      gridDim.x++;
}
