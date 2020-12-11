#include "bl_config.h"
#include "bl_dgemm_kernel.h"

#define a(i, j, ld) a[ (i)*(ld) + (j) ]
#define b(i, j, ld) b[ (i)*(ld) + (j) ]
#define c(i, j, ld) c[ (i)*(ld) + (j) ]

//
// C-based micorkernel
//
void bl_dgemm_ukr(int k,int m, int n, 
      const double * restrict A, 
      const double * restrict B,
      double * restrict C,
      unsigned long long ldc,
      aux_t* data )
{

//avx 512
#ifdef AVX
  register __m512d c00_c07, c08_c015,c016_c023;
  register __m512d c10_c17, c18_c115,c116_c123;
  register __m512d c20_c27, c28_c215,c216_c223;
  register __m512d c30_c37, c38_c315,c316_c323;
  register __m512d c40_c47, c48_c415,c416_c423;
  register __m512d c50_c57, c58_c515,c516_c523;
  register __m512d c60_c67, c68_c615,c616_c623;
  register __m512d c70_c77, c78_c715,c716_c723;

  register __m512d a08;
  register __m512d b08,b18,b28;

  // load registers
  c00_c07 = _mm512_loadu_pd(C + 0*ldc);
  c08_c015 = _mm512_loadu_pd(C + 0*ldc + 8);
  c016_c023 = _mm512_loadu_pd(C + 0*ldc + 16);

  c10_c17 = _mm512_loadu_pd(C + 1*ldc);
  c18_c115 = _mm512_loadu_pd(C + 1*ldc + 8);
  c116_c123 = _mm512_loadu_pd(C + 1*ldc + 16);

  c20_c27 = _mm512_loadu_pd(C + 2*ldc);
  c28_c215 = _mm512_loadu_pd(C + 2*ldc + 8);
  c216_c223 = _mm512_loadu_pd(C + 2*ldc + 16);

  c30_c37 = _mm512_loadu_pd(C + 3*ldc);
  c38_c315 = _mm512_loadu_pd(C + 3*ldc + 8);
  c316_c323 = _mm512_loadu_pd(C + 3*ldc + 16);

  c40_c47 = _mm512_loadu_pd(C + 4*ldc);
  c48_c415 = _mm512_loadu_pd(C + 4*ldc + 8);
  c416_c423 = _mm512_loadu_pd(C + 4*ldc + 16);

  c50_c57 = _mm512_loadu_pd(C + 5*ldc);
  c58_c515 = _mm512_loadu_pd(C + 5*ldc + 8);
  c516_c523 = _mm512_loadu_pd(C + 5*ldc + 16);

  c60_c67 = _mm512_loadu_pd(C + 6*ldc);
  c68_c615 = _mm512_loadu_pd(C + 6*ldc + 8);
  c616_c623 = _mm512_loadu_pd(C + 6*ldc + 16);

  c70_c77 = _mm512_loadu_pd(C + 7*ldc);
  c78_c715 = _mm512_loadu_pd(C + 7*ldc + 8);
  c716_c723 = _mm512_loadu_pd(C + 7*ldc + 16);

    for(int hori_k = 0; hori_k<k;hori_k++){
        //loop unrolling 0
        register __m128d a_sub = {*(A+hori_k*DGEMM_MR+0), 0.0L };
        a08 = _mm512_broadcastsd_pd(a_sub);

        //take 12 double in row hori_k, column 0 to 11 from matrix B 
        b08 = _mm512_loadu_pd(B+hori_k*DGEMM_NR);
        b18 = _mm512_loadu_pd(B+hori_k*DGEMM_NR+8);
        b28 = _mm512_loadu_pd(B+hori_k*DGEMM_NR+16);

        //0
        c00_c07 = _mm512_fmadd_pd(a08,b08,c00_c07);
        c08_c015 = _mm512_fmadd_pd(a08,b18,c08_c015);
        c016_c023 = _mm512_fmadd_pd(a08,b28,c016_c023);


        //1
        a_sub = _mm_set_sd(*(A+hori_k*DGEMM_MR+1));
        a08 = _mm512_broadcastsd_pd(a_sub);

        c10_c17 = _mm512_fmadd_pd(a08,b08,c10_c17);
        c18_c115 = _mm512_fmadd_pd(a08,b18,c18_c115);
        c116_c123 = _mm512_fmadd_pd(a08,b28,c116_c123);

        //2
        a_sub = _mm_set_sd(*(A+hori_k*DGEMM_MR+2));
        a08 = _mm512_broadcastsd_pd(a_sub);

        c20_c27 = _mm512_fmadd_pd(a08,b08,c20_c27);
        c28_c215 = _mm512_fmadd_pd(a08,b18,c28_c215);
        c216_c223 = _mm512_fmadd_pd(a08,b28,c216_c223);

        //3
        a_sub = _mm_set_sd(*(A+hori_k*DGEMM_MR+3));
        a08 = _mm512_broadcastsd_pd(a_sub);

        c30_c37 = _mm512_fmadd_pd(a08,b08,c30_c37);
        c38_c315 = _mm512_fmadd_pd(a08,b18,c38_c315);
        c316_c323 = _mm512_fmadd_pd(a08,b28,c316_c323);

        //4
        a_sub = _mm_set_sd(*(A+hori_k*DGEMM_MR+4));
        a08 = _mm512_broadcastsd_pd(a_sub);

        c40_c47= _mm512_fmadd_pd(a08,b08,c40_c47);
        c48_c415 = _mm512_fmadd_pd(a08,b18,c48_c415);
        c416_c423 = _mm512_fmadd_pd(a08,b28,c416_c423);

        //5
        a_sub = _mm_set_sd(*(A+hori_k*DGEMM_MR+5));
        a08 = _mm512_broadcastsd_pd(a_sub);

        c50_c57 = _mm512_fmadd_pd(a08,b08,c50_c57);
        c58_c515 = _mm512_fmadd_pd(a08,b18,c58_c515);
        c516_c523 = _mm512_fmadd_pd(a08,b28,c516_c523);


        //6
        a_sub = _mm_set_sd(*(A+hori_k*DGEMM_MR+6));
        a08 = _mm512_broadcastsd_pd(a_sub);

        c60_c67 = _mm512_fmadd_pd(a08,b08,c60_c67);
        c68_c615 = _mm512_fmadd_pd(a08,b18,c68_c615);
        c616_c623 = _mm512_fmadd_pd(a08,b28,c616_c623);
        
        
        //7
        a_sub = _mm_set_sd(*(A+hori_k*DGEMM_MR+7));
        a08 = _mm512_broadcastsd_pd(a_sub);

        c70_c77 = _mm512_fmadd_pd(a08,b08,c70_c77);
        c78_c715 = _mm512_fmadd_pd(a08,b18,c78_c715);
        c716_c723 = _mm512_fmadd_pd(a08,b28,c716_c723);

    }

    _mm512_storeu_pd(C + ldc*0, c00_c07);
    _mm512_storeu_pd(C + ldc*0 + 8, c08_c015);
    _mm512_storeu_pd(C + ldc*0 + 16, c016_c023);

    _mm512_storeu_pd(C + ldc*1, c10_c17);
    _mm512_storeu_pd(C + ldc*1 + 8, c18_c115);
    _mm512_storeu_pd(C + ldc*1 + 16, c116_c123);

    _mm512_storeu_pd(C + ldc*2, c20_c27);
    _mm512_storeu_pd(C + ldc*2 + 8, c28_c215);
    _mm512_storeu_pd(C + ldc*2 + 16, c216_c223);

    _mm512_storeu_pd(C + ldc*3, c30_c37);
    _mm512_storeu_pd(C + ldc*3 + 8, c38_c315);
    _mm512_storeu_pd(C + ldc*3 + 16, c316_c323);

    _mm512_storeu_pd(C + ldc*4, c40_c47);
    _mm512_storeu_pd(C + ldc*4 + 8, c48_c415);
    _mm512_storeu_pd(C + ldc*4 + 16, c416_c423);

    _mm512_storeu_pd(C + ldc*5, c50_c57);
    _mm512_storeu_pd(C + ldc*5 + 8, c58_c515);
    _mm512_storeu_pd(C + ldc*5 + 16, c516_c523);

    _mm512_storeu_pd(C + ldc*6, c60_c67);
    _mm512_storeu_pd(C + ldc*6 + 8, c68_c615);
    _mm512_storeu_pd(C + ldc*6 + 16, c616_c623);

    _mm512_storeu_pd(C + ldc*7, c70_c77);
    _mm512_storeu_pd(C + ldc*7 + 8, c78_c715);
    _mm512_storeu_pd(C + ldc*7 + 16, c716_c723);



//avx256
#else
  register __m256d c00_c03, c04_c07, c08_c0b, 
                   c10_c13, c14_c17, c18_c1b, 
                   c20_c23, c24_c27, c28_c2b, 
                   c30_c33, c34_c37, c38_c3b; 
  register __m256d a0;
  register __m256d b0,b1,b2;
  //12+3+1 16 ymm registers 


  // load from C to ymm registers, totally 12
  // a register contain 4 double, total 48 doubles.
  c00_c03 = _mm256_loadu_pd(C + 0*ldc);
  c04_c07 = _mm256_loadu_pd(C + 0*ldc + 4);
  c08_c0b = _mm256_loadu_pd(C + 0*ldc + 8);
  
  c10_c13 = _mm256_loadu_pd(C + 1*ldc);
  c14_c17 = _mm256_loadu_pd(C + 1*ldc + 4);
  c18_c1b = _mm256_loadu_pd(C + 1*(ldc + 8));

  c20_c23 = _mm256_loadu_pd(C + 2*ldc);
  c24_c27 = _mm256_loadu_pd(C + 2*ldc + 4);
  c28_c2b = _mm256_loadu_pd(C + 2*ldc + 8);
  
  c30_c33 = _mm256_loadu_pd(C + 3*ldc);
  c34_c37 = _mm256_loadu_pd(C + 3*ldc + 4);
  c38_c3b = _mm256_loadu_pd(C + 3*ldc + 8);

  // Mr * Nr matrix C
  // | c00 c01 c02 c03 | c04 c05 c06 c07 | c08 c09 c0a c0b
  // | c10 c11 c12 c13 | c14 c15 c16 c17 | c18 c19 c1a c1b
  // | c20 c21 c22 c23 | c24 c25 c26 c27 | c28 c29 c2a c2b
  // | c30 c31 c32 c33 | c34 c35 c36 c37 | c38 c39 c3a c3b



  //loop horizontally along kc
  for(int hori_k = 0; hori_k<k;hori_k++){
    //loop unrolling 0

    //take 4 double in row 0, column hori_k from matrix A
    a0 = _mm256_broadcast_sd(A+hori_k*DGEMM_MR+0);

    //take 12 double in row hori_k, column 0 to 11 from matrix B 
    b0 = _mm256_loadu_pd(B+hori_k*DGEMM_NR);
    b1 = _mm256_loadu_pd(B+hori_k*DGEMM_NR+4);
    b2 = _mm256_loadu_pd(B+hori_k*DGEMM_NR+8);

    //calculate row 0 of C
    c00_c03 = _mm256_fmadd_pd(a0,b0,c00_c03);
    c04_c07 = _mm256_fmadd_pd(a0,b1,c04_c07);
    c08_c0b = _mm256_fmadd_pd(a0,b2,c08_c0b);

  
    //loop unrolling 1

    //take 4 double in row 1, column hori_k from matrix A
    a0 = _mm256_broadcast_sd(A+hori_k*DGEMM_MR+1);

    //calculate row 1 of C
    c10_c13 = _mm256_fmadd_pd(a0,b0,c10_c13);
    c14_c17 = _mm256_fmadd_pd(a0,b1,c14_c17);
    c18_c1b = _mm256_fmadd_pd(a0,b2,c18_c1b);

    
    //loop unrolling 2
    //take 4 double in row 2, column hori_k from matrix A
    a0 = _mm256_broadcast_sd(A+hori_k*DGEMM_MR+2);

    //calculate row 2 of C
    c20_c23 = _mm256_fmadd_pd(a0,b0,c20_c23);
    c24_c27 = _mm256_fmadd_pd(a0,b1,c24_c27);
    c28_c2b = _mm256_fmadd_pd(a0,b2,c28_c2b);

    
    //loop unrolling 3

    //take 4 double in row 3, column hori_k from matrix A
    a0 = _mm256_broadcast_sd(A+hori_k*DGEMM_MR+3);

    //calculate row 3 of C
    c30_c33 = _mm256_fmadd_pd(a0,b0,c30_c33);
    c34_c37 = _mm256_fmadd_pd(a0,b1,c34_c37);
    c38_c3b = _mm256_fmadd_pd(a0,b2,c38_c3b);
  }

  //store 12 ymm register back
  _mm256_storeu_pd(C + ldc*0, c00_c03);
  _mm256_storeu_pd(C + ldc*0 + 4, c04_c07);
  _mm256_storeu_pd(C + ldc*0 + 8, c08_c0b);

  _mm256_storeu_pd(C + ldc*1, c10_c13);
  _mm256_storeu_pd(C + ldc*1 + 4, c14_c17);
  _mm256_storeu_pd(C + ldc*1 + 8, c18_c1b);

  _mm256_storeu_pd(C + ldc*2, c20_c23);
  _mm256_storeu_pd(C + ldc*2 + 4, c24_c27);
  _mm256_storeu_pd(C + ldc*2 + 8, c28_c2b);

  _mm256_storeu_pd(C + ldc*3, c30_c33);
  _mm256_storeu_pd(C + ldc*3 + 4, c34_c37);
  _mm256_storeu_pd(C + ldc*3 + 8, c38_c3b);


#endif
}

