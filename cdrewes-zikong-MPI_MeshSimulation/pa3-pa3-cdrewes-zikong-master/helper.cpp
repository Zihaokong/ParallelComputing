/* 
 * Utilities for the Aliev-Panfilov code
 * Scott B. Baden, UCSD
 * Nov 2, 2015
 */

#include <iostream>
#include <assert.h>
// Needed for memalign
#include <malloc.h>


// Colin -- need to import these for MPI
#include<mpi.h>
#include"cblock.h"

extern control_block cb;

double *rec_north, *send_north, *rec_south, *send_south, *rec_east, *send_east, *rec_west, *send_west;
int tile_m, tile_n, rem_m, rem_n, myrank, rank_m, rank_n, size; 

using namespace std;


void printMat(const char mesg[], double *E, int m, int n);

//
// Initialization
//
// We set the right half-plane of E_prev to 1.0, the left half plane to 0
// We set the botthom half-plane of R to 1.0, the top half plane to 0
// These coordinates are in world (global) coordinate and must
// be mapped to appropriate local indices when parallelizing the code
//
void init (double *E,double *E_prev,double *R,int m,int n) {
  // INITIALIZE E AS IN THE STARTER CODE THEN DELEDATE FROM THE RANK 0 TO THE SUBs

  // This is the worker process which sends the IC to the other processors
  if (myrank == 0) {
    /*
     * Iterate through each processor which is just one tile
     * Ignore proc 0 as that is the worker thread
     */
    for (int proc = 0; proc < size; ++proc) {
      // Recalculate these values for the other procs (already done in alloc1D)
      int send_rank_m = proc / cb.px;
      int send_rank_n = proc % cb.px;
      int send_rem_m = (cb.m) % cb.py;
      int send_rem_n = (cb.n) % cb.px;
      int send_tile_m = (send_rank_m < send_rem_m) ? (cb.m / cb.py) + 1 : (cb.m / cb.py);
      int send_tile_n = (send_rank_n < send_rem_n) ? (cb.n / cb.px) + 1 : (cb.n / cb.px);
      //printf("TOTAL %d\n", cb.m);
      // Data "read" from file which worker 0 will be sending to respective proc
      double * send_E_prev = (double*) memalign(16, sizeof(double)*send_tile_m*send_tile_n);
      double * send_R = (double*) memalign(16, sizeof(double)*send_tile_m*send_tile_n);
     
     /*
      * This simultates reading from a file
      * Worker 0 will read that information and pass it to the other workers
      * This is required by the writeup--we must use message passing for ICs
      */
      int raw_n;
      if (send_rank_n < send_rem_n) {
        raw_n = send_rank_n * send_tile_n;
      } else {
        raw_n = send_rem_n * (send_tile_n + 1) + ((send_rank_n - send_rem_n) * send_tile_n);
      }
 //     int E_ones = 0;
      //printf("BLOCK MXN: %dx%d\n", send_tile_m, send_tile_n);
      for (int i = 0; i < send_tile_n; ++i) {
        if ((raw_n + i) >= (cb.n/2)) {
          //printf("PROC %d  ONES AT COL %d\n", proc, i);
          for (int j = 0; j < send_tile_m; ++j) {
 //           E_ones++;
            send_E_prev[i+(j*send_tile_n)] = 1.0;
          }
        } else {
          for (int j = 0; j < send_tile_m; ++j) {
            send_E_prev[i+(j*send_tile_n)] = 0.0;
          }
        }
      }
//      printf("E ONES: %d\n", E_ones);      

      int raw_m;
      if (send_rank_m < send_rem_m) {
        raw_m = send_rank_m * send_tile_m;
      } else {
        raw_m = send_rem_m * (send_tile_m + 1) + ((send_rank_m - send_rem_m) * send_tile_m);
      }
  //    int R_ones = 0;
      for (int j = 0; j < send_tile_m; ++j) {
        if ((raw_m + j) >= (cb.m/2)) {
          //printf("PROC %d  ONES AT ROW %d\n", proc, j);
          for (int i = 0; i < send_tile_n; ++i) {
  //          R_ones++;
            send_R[i+(j*send_tile_n)] = 1.0;
          }
        } else {
          for (int i = 0; i < send_tile_n; ++i) {
            send_R[i+(j*send_tile_n)] = 0.0;
          }
        }
      }
 //     printf("R ONES: %d\n", R_ones);      

      if (proc == 0) {
        /*
         * However, we must still have the worker 0 copy its data to its tile
         * This does not require any sending
         * In this case tile_m = send_tile_m
         * And tile_n = send_tile_n
         */
        for (int i = 0; i < tile_m*tile_n; ++i) {
          E_prev[i] = send_E_prev[i];
          R[i] = send_R[i];
        }
      } else {
       /*
        *
        * Use non-blocking send
        * The size of the object being sent is just the sender tile size
        * Send to the current processor
        * Leave tag as 0 to be the E_prev input
        */
        #ifdef _MPI_
        if (!cb.noComm){
          MPI_Request send_E_prev_request;
          MPI_Isend(send_E_prev, send_tile_m*send_tile_n, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD, &send_E_prev_request);

          // Perform the same process for the R matrix except use tag 1 to organize this data
          MPI_Request send_R_request;
          MPI_Isend(send_R, send_tile_m*send_tile_n, MPI_DOUBLE, proc, 1, MPI_COMM_WORLD, &send_R_request);

          // Get the status of the send operation
          MPI_Status send_E_prev_status;
          MPI_Status send_R_status;

          MPI_Wait(&send_E_prev_request, &send_E_prev_status);
          MPI_Wait(&send_R_request, &send_R_status);
        }
        #endif
      }
    }
  // Otherwise, this core is just recieving
  } else {
    // Source processor will always be 0
    // The E tag is 0 by construction
    #ifdef _MPI_
    if (!cb.noComm){
      MPI_Request rec_E_prev_request;
      MPI_Irecv(E_prev, tile_m*tile_n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &rec_E_prev_request);

      // Again recieve from processor 0
      // The R matrix has tab 1
      MPI_Request rec_R_request;
      MPI_Irecv(R, tile_m*tile_n, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &rec_R_request);

      // Wait for this to complete
      MPI_Status rec_E_prev_status;
      MPI_Status rec_R_status;
      MPI_Wait(&rec_E_prev_request, &rec_E_prev_status);
      MPI_Wait(&rec_R_request, &rec_R_status);
    }
    #endif
  }
}



double *alloc1D(int m,int n){

  /* 
   * Must adjust for the impicit padding hence the -2
   * Divide by the number cores in the y direction to get the number of tiles in the y dir
   */
  tile_m = (m - 2) / cb.py;

 /* 
  * Must adjust for the impicit padding hence the -2
  * Divide by the number cores in the x direction to get the number of tiles in the x dir
  */
  tile_n = (n - 2) / cb.px;

  /*
   * However, the size of input n may not be divisible number of threads
   * So, we determine the number of extra columns and rows we need to buffer
   */
  rem_m = (m-2) % cb.py;
  rem_n = (n-2) % cb.px;

  // Allocation now depends on the current PID
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);

  // Get the cpu index in the y direction
  rank_m = myrank / cb.px;
 
  // Get the cpu index in the x direction
  rank_n = myrank % cb.px;

  /*
   * For the first rem_m we will pad with an extra y layer
   */
  tile_m = (rank_m < rem_m) ? tile_m + 1 : tile_m;

  /*
   * For the first rem_n we will pad with an extra x layer
   */
  tile_n = (rank_n < rem_n) ? tile_n + 1 : tile_n;
  
  /* 
   * Just for testing 
   * MPI_Comm_size(MPI_COMM_WORLD,&size);
   * printf("Process: %d of %d\n", myrank, size);
   */

  /*
   * We then manually insert the all of the send and rec channels
   * The north and south vectors are in x direction, i.e n
   * The east and west vectors are in the y direction, i.e m
   */
  rec_north  = new double[4*tile_n];
  send_north = rec_north + tile_n;
  rec_south  = send_north + tile_n;
  send_south = rec_south + tile_n;

  rec_west  = new double[4*tile_m];
  send_west = rec_west + tile_m;
  rec_east  = send_west + tile_m;
  send_east = rec_east + tile_m;
  
  double *E;
  // Ensures that allocatdd memory is aligned on a 16 byte boundary
  assert(E= (double*) memalign(16, sizeof(double)*tile_m*tile_n));
  // Likely uneeded
  return(E);
}

void printMat(const char mesg[], double *E, int m, int n){
  int i;
#if 0
  if (m>8)
    return;
#else
  if (m>34)
    return;
#endif
  printf("%s\n",mesg);
  for (i=0; i < (m+2)*(n+2); i++){
    int rowIndex = i / (n+2);
    int colIndex = i % (n+2);
    if ((colIndex>0) && (colIndex<n+1))
      if ((rowIndex > 0) && (rowIndex < m+1))
        printf("%6.3f ", E[i]);
    if (colIndex == n+1)
      printf("\n");
  }
}
