/* 
 * Solves the Aliev-Panfilov model  using an explicit numerical scheme.
 * Based on code orginally provided by Xing Cai, Simula Research Laboratory
 * 
 * Modified and  restructured by Scott B. Baden, UCSD
 * 
 */

#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <math.h>
#include "time.h"
#include "apf.h"
#include "Plotting.h"
#include "cblock.h"
#include <emmintrin.h>

#include <mpi.h>
using namespace std;

void repNorms(double l2norm, double mx, double dt, int m,int n, int niter, int stats_freq);
void stats(double *E, int m, int n, double *_mx, double *sumSq);
void stats_adj(double *E, int m, int n, double *_mx, double *sumSq);
void printMat2(const char mesg[], double *E, int m, int n);

extern control_block cb;
extern int tile_m, tile_n, rem_m, rem_n, myrank, rank_m, rank_n, size; 
extern double *rec_north, *send_north, *rec_south, *send_south, *rec_east, *send_east, *rec_west, *send_west;
// #ifdef SSE_VEC
// If you intend to vectorize using SSE instructions, you must
// disable the compiler's auto-vectorizer
// __attribute__((optimize("no-tree-vectorize")))
// #endif 

// The L2 norm of an array is computed by taking sum of the squares
// of each element, normalizing by dividing by the number of points
// and then taking the sequare root of the result
//
double L2Norm(double sumSq){
  double l2norm = sumSq /  (double) ((cb.m)*(cb.n));
  l2norm = sqrt(l2norm);
  return l2norm;
}

void stats_adj(double *E, int m, int n, double *_mx, double *sumSq) {
     double mx = -1;
     double _sumSq = 0;
     int i, j;

     for (i=0; i < m*n; ++i) {
        int rowIndex = i / n;			// gives the current row number in 2D array representation
        int colIndex = i % n;		// gives the base index (first row's) of the current index		

        _sumSq += E[i]*E[i];
        double fe = fabs(E[i]);
        if (fe > mx)
            mx = fe;
    }
    *_mx = mx;
    *sumSq = _sumSq;
}

void solve(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf){

  
    //printf("Starting Solve on Proc: %d\n", myrank);
  //printf("ADDY: %p\n", rec_north);
  // Simulated time is different from the integer timestep number
  double t = 0.0;

  //printf("ADDY: %p\n", *_E);
  double *E = *_E, *E_prev = *_E_prev;
  double *R_tmp = R;
  double *E_tmp = *_E;
  double *E_prev_tmp = *_E_prev;
  double mx, sumSq;
  int niter;
  // Adjust for the starter codes implementation
  int m = tile_m, n=tile_n;
  //int innerBlockRowStartIndex = (n+2)+1;
  //int innerBlockRowEndIndex = (((m+2)*(n+2) - 1) - (n)) - (n+2);


  // Colin -- rewrite
  
  // Iterate over timescale
  for (niter = 0; niter < cb.niters; niter++) {
    if  (cb.debug && (niter==0)){
      stats_adj(E_prev,m,n,&mx,&sumSq);
      double l2norm = L2Norm(sumSq);
      repNorms(l2norm,mx,dt,m,n,-1, cb.stats_freq);
      if (cb.plot_freq)
        plotter->updatePlot(E,  -1, m+1, n+1);
    }
    /*
     * First we need to perform the message passing between stencils
     */
    int bottom_row = (tile_m-1)*tile_n;
    for (int i = 0; i < tile_n; ++i) {
      send_north[i] = E_prev[i]; 
      send_south[i] = E_prev[i + bottom_row]; 
    }
    
    for (int j = 0; j < tile_m; ++j) {
      send_west[j] = E_prev[0+(j*tile_n)]; 
      send_east[j] = E_prev[(tile_n-1)+(j*tile_n)]; 
    }

    int stati_count = 0;
    MPI_Request rec[4];
    MPI_Request sent[4];
    MPI_Status stati[4];

    // If the tile is in the top row
    if (rank_m == 0) {
      // This is the same as starter code execpt just use rec channel as buffer
      for (int i = 0; i < tile_n; ++i) {
        rec_north[i] = E_prev[i + tile_n];
      }
    // Not in the top row -- pass messages as usual
    } else {
      #ifdef _MPI_
      if (!cb.noComm){
        MPI_Irecv(rec_north, tile_n, MPI_DOUBLE, myrank - cb.px, 4, MPI_COMM_WORLD, &rec[stati_count]);
        MPI_Isend(send_north, tile_n, MPI_DOUBLE, myrank - cb.px, 7, MPI_COMM_WORLD, &sent[stati_count]);
        stati_count++;
      }
      #endif
    }

    // If the tile is on the bottom row
    if (rank_m == (cb.py - 1)) {
      int bottom_row = (tile_m - 2) * tile_n;
      for (int i = 0; i < tile_n; ++i) {
        rec_south[i] = E_prev[i + bottom_row];
      }
    } else {
      #ifdef _MPI_
      if (!cb.noComm){
        MPI_Irecv(rec_south, tile_n, MPI_DOUBLE, myrank + cb.px, 7, MPI_COMM_WORLD, &rec[stati_count]);
        MPI_Isend(send_south, tile_n, MPI_DOUBLE, myrank + cb.px, 4, MPI_COMM_WORLD, &sent[stati_count]);
        stati_count++;
      }
      #endif
    }

    // If the tile is at the right edge
    if (rank_n == (cb.px - 1)) {
      for (int i = 0; i < tile_m; ++i) {
        rec_east[i] = E_prev[(i*tile_n) + (tile_n-2)];
      }
    } else {
      #ifdef _MPI_
      if (!cb.noComm){
        MPI_Irecv(rec_east, tile_m, MPI_DOUBLE, myrank + 1, 5, MPI_COMM_WORLD, &rec[stati_count]);
        MPI_Isend(send_east, tile_m, MPI_DOUBLE, myrank + 1, 6, MPI_COMM_WORLD, &sent[stati_count]);
        stati_count++;
      }
      #endif
    }

    // If the tile is at the left edge
    if (rank_n == 0) {
      //for (int i = 0, j = 0; j < tile_m; i+=tile_n, ++j) {
      for (register int i = 0; i < tile_m; ++i) {
        rec_west[i] = E_prev[(i*tile_n)+1];
      }
    } else {
      #ifdef _MPI_
      if (!cb.noComm){
        MPI_Irecv(rec_west, tile_m, MPI_DOUBLE, myrank - 1, 6, MPI_COMM_WORLD, &rec[stati_count]);
        MPI_Isend(send_west, tile_m, MPI_DOUBLE, myrank - 1, 5, MPI_COMM_WORLD, &sent[stati_count]);
        stati_count++;
      }
      #endif
    }
    #ifdef _MPI_
    if (!cb.noComm){
      MPI_Waitall(stati_count, rec, stati);
    }
    #endif

    // No longer have extra layer on every side
    for(int j = 0; j < tile_m; ++j) {
      E_tmp = E + (j*tile_n);
      E_prev_tmp = E_prev + (j*tile_n);
     
      // Top row needs to update 
      for(int i = 0; i < tile_n; ++i) {
        // AVX ME
        double E_prev_tmp_i_p_1 = (i == (tile_n-1)) ? rec_east[j] : E_prev_tmp[i+1];
        double E_prev_tmp_i_m_1 = (i == 0) ? rec_west[j] : E_prev_tmp[i-1];
        double E_prev_tmp_i_p_n = (j == (tile_m-1)) ? rec_south[i] : E_prev_tmp[i+tile_n];
        double E_prev_tmp_i_m_n = (j == 0) ? rec_north[i] : E_prev_tmp[i-tile_n];
        E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp_i_p_1+E_prev_tmp_i_m_1-4*E_prev_tmp[i]+E_prev_tmp_i_p_n+E_prev_tmp_i_m_n);
      }
    }


    for(int j = 0; j < tile_m; ++j) {
      E_tmp = E + (j*tile_n);
      R_tmp = R + (j*tile_n);
      E_prev_tmp = E_prev + (j*tile_n);
      for(int i = 0; i < tile_n; ++i) {
        //ode_c++;
        E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
        R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/(E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
      }
    }

    if (cb.stats_freq){
      if ( !(niter % cb.stats_freq)){
        printf("UPDATED\n");
        stats_adj(E,m,n,&mx,&sumSq);
        double l2norm = L2Norm(sumSq);
        repNorms(l2norm,mx,dt,m,n,niter, cb.stats_freq);
      }
    }

    if (cb.plot_freq){
      if (!(niter % cb.plot_freq)){
        plotter->updatePlot(E,  niter, m, n);
      }
    }

    double *tmp = E; E = E_prev; E_prev = tmp;
  }

  double finalSq, finalLinf;
  stats_adj(E_prev,m,n,&Linf,&sumSq);

  //double finalLinf;
  #ifdef _MPI_
  if (!cb.noComm){
    MPI_Reduce(&sumSq, &finalSq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&Linf, &finalLinf, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    Linf = finalLinf;
    sumSq = finalSq;
  }
  #endif

  L2 = L2Norm(sumSq);
  //printf("\nMXN: %dx%d\n", rank_m, rank_n);
  //printf("\nMXN: %dx%d\n", tile_m, tile_n);
  //printMat2("Rank X Matrix E_prev", E_prev, tile_m,tile_n);  

  *_E = E;
  *_E_prev = E_prev;
}

void printMat2(const char mesg[], double *E, int m, int n){
  int i;
  for (i=0; i < m*n; i++){
    int rowIndex = i / (n);
    int colIndex = i % (n);
    //if ((colIndex))
    //if ((colIndex>0) && (colIndex<n))
    //  if ((rowIndex > 0) && (rowIndex < m))
    printf("%6.3f ", E[i]);
    if (colIndex == n-1)
      printf("\n");
  }
}
