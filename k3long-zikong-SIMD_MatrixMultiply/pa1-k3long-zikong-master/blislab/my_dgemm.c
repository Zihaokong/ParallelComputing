/*
 * --------------------------------------------------------------------------
 * BLISLAB 
 * --------------------------------------------------------------------------
 * Copyright (C) 2016, The University of Texas at Austin
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *  - Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  - Neither the name of The University of Texas nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * bl_dgemm.c
 *
 *
 * Purpose:
 * this is the main file of blislab dgemm.
 *
 * Todo:
 *
 *
 * Modification:
 *      bryan chin - ucsd
 *      changed to row-major order  
 *      handle arbitrary  size C
 * */

#include <stdio.h>

#include "bl_dgemm_kernel.h"
#include "bl_dgemm.h"
#include <string.h>
const char* dgemm_desc = "my blislab ";


/* 
 * pack one subpanel of A
 *
 * pack like this 
 * if A is row major order
 *
 *     a c e g
 *     b d f h
 *     i k m o
 *     j l n p
 *     q r s t
 *     
 * then pack into a sub panel
 * each letter represents sequantial
 * addresses in the packed result
 * (e.g. a, b, c, d are sequential
 * addresses). 
 * - down each column
 * - then next column in sub panel
 * - then next sub panel down (on subseqent call)
 
 *     a c e g  < each call packs one
 *     b d f h  < subpanel
 *     -------
 *     i k m o
 *     j l n p
 *     -------
 *     q r s t
 *     0 0 0 0
 */
static inline
void packA_mcxkc_d(
        int    m,
        int    k,
        double * restrict XA,
        int    ldXA,
        double * restrict packA
        )
{
  //if the array to be packed is less than standard size, fill with 0 first
  if(m < DGEMM_MR || k < DGEMM_KC) memset((double*) packA,0,DGEMM_MR*DGEMM_KC*sizeof(double));

  //packing
  int i,j;
  for(j=0;j<k;j++){
    for(i = 0;i<m;i++){
      packA[j*DGEMM_MR+i] = XA[i*ldXA+j];
    }
  }
}




/*
 * --------------------------------------------------------------------------
 */

/* 
 * pack one subpanel of B
 * 
 * pack like this 
 * if B is 
 *
 * row major order matrix
 *     a b c j k l s t
 *     d e f m n o u v
 *     g h i p q r w x
 *
 * each letter represents sequantial
 * addresses in the packed result
 * (e.g. a, b, c, d are sequential
 * addresses). 
 *
 * Then pack 
 *   - across each row in the subpanel
 *   - then next row in each subpanel
 *   - then next subpanel (on subsequent call)
 *
 *     a b c |  j k l |  s t 0
 *     d e f |  m n o |  u v 0
 *     g h i |  p q r |  w x 0
 *
 *     ^^^^^
 *     each call packs one subpanel
 */
static inline
void packB_kcxnc_d(
        int    n,
        int    k,
        double * restrict XB,
        int    ldXB, // ldXB is the original k
        double * restrict packB
        )
{
  //if array to be packed is less than standard size, fill with 0
  if(n < DGEMM_NR || k < DGEMM_KC) memset((double*) packB,0,DGEMM_NR*DGEMM_KC*sizeof(double));

  //packing
  int i,j;
  for(i = 0;i<k;i++){
    for(j = 0;j<n;j++){
      packB[i*DGEMM_NR+j] = XB[i*ldXB+j];
    }
  }
}

/*
 * --------------------------------------------------------------------------
 */

static
inline
void bl_macro_kernel(
        int    m,
        int    n,
        int    k,
        const double * restrict packA,
        const double * restrict packB,
        double * C,
        int    ldc
        )
{
    aux_t  aux;
    int    i, j;

    for ( i = 0; i < m; i += DGEMM_MR ) {                      // 2-th loop around micro-kernel
      for ( j = 0; j < n; j += DGEMM_NR ) {       
        int vertical_len = min(m-i,DGEMM_MR);
        int horizontal_len = min(n-j,DGEMM_NR);                 // 1-th loop around micro-kernel
	      
        ( *bl_micro_kernel ) (k,vertical_len,horizontal_len, &packA[ i * DGEMM_KC ], &packB[ j * DGEMM_KC ], &C[ i * ldc + j ],(unsigned long long) ldc,&aux);

      }                                                        // 1-th loop around micro-kernel
    }                                                           // 2-th loop around micro-kernel                                                      // 2-th loop around micro-kernel
}


void bl_dgemm(
        int    m,
        int    n,
        int    k,
        double * restrict XA,
        int    lda,
        double * restrict XB,
        int    ldb,
        double * restrict C,       
        int    ldc       
        )
{
    int    ic, ib, jc, jb, pc, pb;

    int Asize = DGEMM_KC* ( (DGEMM_MR + DGEMM_MC - 1)/DGEMM_MR )* DGEMM_MR;
    int Bsize = DGEMM_KC*( (DGEMM_NR + DGEMM_NC - 1)/DGEMM_NR )* DGEMM_NR;
    double* packA  = (double*)malloc( Asize*sizeof(double));
    double* packB  = (double*)malloc( Bsize*sizeof(double));

    for ( ic = 0; ic < m; ic += DGEMM_MC ) {              // 5-th loop around micro-kernel
        ib = min( m - ic, DGEMM_MC );
        for ( pc = 0; pc < k; pc += DGEMM_KC ) {          // 4-th loop around micro-kernel
            pb = min( k - pc, DGEMM_KC );


            //empty packing buffer and packA
            int i, j;
            memset((double*) packA,0,Asize*sizeof(double));
            for ( i = 0; i < ib; i += DGEMM_MR ) {
                packA_mcxkc_d(min( ib - i, DGEMM_MR ), pb, &XA[ pc + lda*(ic + i)], k, &packA[i * DGEMM_KC ]);
            }

            for ( jc = 0; jc < n; jc += DGEMM_NC ) {        // 3-rd loop around micro-kernel
                jb = min( n - jc, DGEMM_NC );

                //empty packing buffer and pack B
                memset((double*) packB,0,Bsize*sizeof(double));
                for ( j = 0; j < jb; j += DGEMM_NR ) {
                    packB_kcxnc_d(min( jb - j, DGEMM_NR ), pb, &XB[ ldb * pc + jc + j],n,&packB[ j * DGEMM_KC ] );
                }

                bl_macro_kernel(ib,jb,pb,packA,packB,&C[ic*ldc+jc],ldc);

            }                                               // End 3.rd loop around micro-kernel
        }                                                 // End 4.th loop around micro-kernel
    }                                                     // End 5.th loop around micro-kernel
  

    free( packA );
    free( packB );
}

void square_dgemm(int lda, double *A, double *B, double *C){
  bl_dgemm(lda, lda, lda, A, lda, B, lda, C,  lda);
}