#include <stdio.h>
#include <assert.h>
using namespace std;
#include <iostream>



#define A(i,j) (a[(i)*n+(j)])
#define B(i,j) (b[(i)*n+(j)])
#define C(i,j) (c[(i)*n+(j)])

#define _DOUBLE_ double
__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {


    __shared__ SA[block]
    int I =  blockIdx.y*blockDim.y + threadIdx.y;
    int J =  blockIdx.x*blockDim.x + threadIdx.x;

    if((I < N) && (J < N)){
        _DOUBLE_ _c = 0;
        for (unsigned int k = 0; k < N; k++) {
            _DOUBLE_ a = A[I * N + k];
            _DOUBLE_ b = B[k * N + J];
            _c += a * b;
        }
        C[I * N + J] = _c;
    }
}
#define TRANSPOSE
void square_dgemm (int n, double* A, double* B, double* C)
{
#ifdef TRANSPOSE
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
        double t = B[i*n+j];
        B[i*n+j] = B[j*n+i];
        B[j*n+i] = t; 
  }
#endif


  /* For each row i of A */
  for (int i = 0; i < n; ++i)
    /* For each column j of B */
    for (int j = 0; j < n; ++j) 
    {
      /* Compute C(i,j) */
      double cij = C[i*n+j];
      for( int k = 0; k < n; k++ )
#ifdef TRANSPOSE
	cij += A[i*n+k] * B[j*n+k];
#else
	cij += A[i*n+k] * B[k*n+j];
#endif
      C[i*n+j] = cij;
    }
}

void genMatrix( _DOUBLE_ *a, unsigned int m, unsigned int n)
{
  unsigned int i;

  for ( i=0; i<m*n; i++ ) {
      a[i] = 1;
  }
}


void setGrid(int n, dim3 &blockDim, dim3 &gridDim)
{
   // set your block dimensions and grid dimensions here

   gridDim.x = n / blockDim.x;
   gridDim.y = n / blockDim.y;

   // you can overwrite blockDim here if you like.
   if(n % blockDim.x!= 0)
   	gridDim.x++;
   if(n % blockDim.y != 0)
    	gridDim.y++;
}


void printMatrix( _DOUBLE_ *a, unsigned int m, unsigned int n)
{
  unsigned int i, j;


  for ( i=0; i<m; i++ ) {
    for ( j=0; j<n; j++ ) 
        printf("%.2f  ",a[i*n+j]);
    cout << endl;
  }
}


int main(){

    int n = 5;
    unsigned int n2 = n*n*sizeof(_DOUBLE_);
    int _ntx = 15;
    int _nty = 6;

    dim3 threads(_ntx, _nty,1);
    dim3 grid;
    setGrid(n, threads, grid);


    _DOUBLE_ *h_A = (_DOUBLE_ *) malloc(n2);
    _DOUBLE_ *h_B = (_DOUBLE_ *) malloc(n2);
    _DOUBLE_ *h_C = (_DOUBLE_ *) malloc(n2);
    genMatrix(h_A, n, n);
    genMatrix(h_B, n, n);

    _DOUBLE_ * hostC = (_DOUBLE_ *) malloc(n2);
    square_dgemm(n,h_A,h_B,hostC);

    printMatrix(hostC,n,n);





    _DOUBLE_ * d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A,n2);
    cudaMalloc((void**)&d_B,n2);
    cudaMalloc((void**)&d_C,n2);

    cudaMemcpy(d_A,h_A,n2,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,n2,cudaMemcpyHostToDevice);

    cudaFuncCache Preference = cudaFuncCachePreferShared;
    cudaFuncSetCacheConfig(matMul,Preference);

    cudaSharedMemConfig  shmPreference;
    shmPreference = cudaSharedMemBankSizeEightByte;
    cudaFuncSetSharedMemConfig( matMul, shmPreference);


    matMul<<< grid, threads >>>(n, d_C, d_A, d_B);

    cudaMemcpy(h_C,d_C,n2,cudaMemcpyDeviceToHost);

    printf("\n\ndevice\n\n");
    printMatrix(h_C,n,n);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}