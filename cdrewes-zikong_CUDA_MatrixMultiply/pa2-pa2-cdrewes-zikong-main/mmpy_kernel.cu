// ;-*- mode: c;-*-
// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "utils.h"
#include "types.h"
#include "mytypes.h"
using namespace std;

#include <stdio.h>

__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ * __restrict__ A, _DOUBLE_ * __restrict__ B) {
    // Using volkov paper as reference
    register int maxIdx = N*N;

    // We calculate BLOCKTILE_M vectors of c simultanously
    register _DOUBLE_ c[BLOCKTILE_M];

    // Holds the columns of a
    register _DOUBLE_ a;

    // As per the paper we keep matrix B in shared on chip memeory
    __shared__ _DOUBLE_ B_local[BLOCKTILE_N][BLOCKTILE_M];
   
    // Zero c vectors
    for (int c_vec = 0; c_vec < BLOCKTILE_M; ++c_vec) {
      c[c_vec] = 0;
    }
    //if ((blockIdx.y * blockDim.y + threadIdx.y) < N 
    // && (blockIdx.x * blockDim.x + threadIdx.x) < N) {  
    // Iterate over blocks in the y direction
    for (int block = 0; block < N; block += BLOCKTILE_N) {
      // Get an array of threads forming a column in our current block
      int B_block_col = (block + threadIdx.y) * N;
      
      // For every block in the y direction get a row of threads
      int B_block_row = (blockIdx.y * BLOCKTILE_M + threadIdx.x);
      if ((B_block_col + B_block_row) < maxIdx) { 
        // For the given block value of this loop iteration, we get a 1D array of 2D blocks as a column
        B_local[threadIdx.y][threadIdx.x] = B[B_block_col + B_block_row];
      } else {
        B_local[threadIdx.y][threadIdx.x] = 0;
      }
      
      // "local barrier" as described by Volkov
      __syncthreads();
     
      // Iterate across the a block
      #pragma unroll
      for (int i = 0; i < BLOCKTILE_N; ++i) {
        /* For every A block in the x direction 
         *  (threadIdx.y * blockDim.x + threadIdx.x) forms a 2D matrix of threadIdx.y x threadIdx.x
         *  blockIdx.x * (BLOCKTILE_M * BLOCKTILE_N) gets a 1D array of these 2D matricies
         */
        int A_block_mat = blockIdx.x * (BLOCKTILE_M * BLOCKTILE_N) + (threadIdx.y * blockDim.x + threadIdx.x);

        /* 
         * The block value takes points to the appropriate block in the y direction
         *  The i is then the particular row within said block
         *  Multiply by N to wrap around the enitiry of matrix A to do column alignment
         */
        int A_shift = (block + i) * N;

        //if ((A_block_mat + A_shift) < maxIdx) {
          a = A[A_block_mat + A_shift];
        //} else {
        //  a = 0;
        //}

        // Iterate through every C vector
        #pragma unroll
        for (int c_vec = 0; c_vec < BLOCKTILE_M; ++c_vec) {
          // Set that vector to be a column times row of B
          c[c_vec] += a * B_local[i][c_vec];
        }
      }
      __syncthreads();
    }

    /* For every C block in the x direction 
     *  (threadIdx.y * blockDim.x + threadIdx.x) forms a 2D matrix of threadIdx.y x threadIdx.x
     *  blockIdx.x * (BLOCKTILE_M * BLOCKTILE_N) gets a 1D array of these 2D matricies
     */
    int C_block_mat = blockIdx.x * (BLOCKTILE_M * BLOCKTILE_N) + (threadIdx.y * blockDim.x + threadIdx.x);

    /* 
     * blockIdx.y gets a list of the blocks in y direction
     * blockIdx.y * N sets threads to point to a column in the y direction
     * blockIdx.y * N * BLOCKTILE_N gets pointers to each block in the y direction
     */
    int C_shift = blockIdx.y * N * BLOCKTILE_M;
  
    // Iterate through every vector in c
    for (int c_vec = 0; c_vec < BLOCKTILE_M; ++c_vec) {
      // C_block_mat + C_shift gets a 1D array of 2D matricies for every row of blocks in the y direction
      // this is effectively just a thread for every item in C
      // c_vec * N gets the appropriate vector in C
      if (C_block_mat < N && c_vec < N && (blockIdx.y * BLOCKTILE_M) < N && ((C_block_mat + C_shift) + c_vec * N) < maxIdx) {
      //if (((C_block_mat + C_shift) + c_vec * N) < maxIdx) {
        C[(C_block_mat + C_shift) + c_vec * N] += c[c_vec];
      }
    }
  //}
}



