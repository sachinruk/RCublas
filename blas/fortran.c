/*
 * Copyright 1993-2008 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  Users and possessors of this source code 
 * are hereby granted a nonexclusive, royalty-free license to use this code 
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.   This source code is a "commercial item" as 
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer  software"  and "commercial computer software 
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein. 
 *
 * Any use of this source code in individual and commercial software must 
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

/*
 * This file contains example Fortran bindings for the CUBLAS library, These
 * bindings have been tested with Intel Fortran 9.0 on 32-bit and 64-bit 
 * Windows, and with g77 3.4.5 on 32-bit and 64-bit Linux. They will likely
 * have to be adjusted for other Fortran compilers and platforms.
 */

#include <ctype.h>
#include <stdlib.h>
#if defined(__GNUC__)
#include <stdint.h>
#endif /* __GNUC__ */
#include "CUDA_include/cublas.h"   /* CUBLAS public header file  */
#include "Blas/prop.h"
#include "Blas/print_vec.h"
#include "Blas/truncation.h"
#include "Lapack_blas.h"

#define imin(a,b) (((a)<(b))?(a):(b))
#define imax(a,b) (((a)<(b))?(b):(a))

#define CUBLAS_G77              1
#define CUBLAS_INTEL_FORTRAN    2

#define CUBLAS_FORTRAN_COMPILER CUBLAS_G77
#include "Blas/fortran.h"
#include "Blas/fort_cublas.h"

/*GLOBAL- FLAGS*/
static unsigned char initialised =0;
static unsigned char double_capable=0;

int CUBLAS_INIT (void) 
{	
	cublasStatus status=0;
	initialised =1;
    status=cublasInit ();

	double_capable=_prop();

	if (status==CUBLAS_STATUS_ALLOC_FAILED)
		print_error("failure.txt");

	return(int) status;
}

char is_double_capable (void){
	if (double_capable)
		return 1;
	else return 0;
}

/*
 *  Fortran callable BLAS functions that include GPU memory allocation and
 *  copy-up and copy-down code. These can be called from unmodified Fortran 
 *  code, but they are inefficient due to the data constantly bouncing back 
 *  and forth between CPU and GPU.
 */


/*---------------------------------------------------------------------------*/
/*---------------------------------- BLAS1 ----------------------------------*/
/*---------------------------------------------------------------------------*/

int CUBLAS_ISAMAX (const int *n, const float *x, const int *incx)
{
    float *devPtrx;
    int retVal;
	
    cublasAlloc (*n * abs(*incx), sizeof(x[0]), (void**)&devPtrx);
    cublasSetVector (*n, sizeof(x[0]), x, *incx, devPtrx, *incx);
    retVal = cublasIsamax (*n, devPtrx, *incx);
    cublasFree (devPtrx);
    return retVal;
}

int CUBLAS_ISAMIN (const int *n, const float *x, const int *incx)
{
    float *devPtrx;
    int retVal;

    cublasAlloc (*n * abs(*incx), sizeof(x[0]), (void**)&devPtrx);
    cublasSetVector (*n, sizeof(x[0]), x, *incx, devPtrx, *incx);
    retVal = cublasIsamin (*n, devPtrx, *incx);
    cublasFree (devPtrx);
    return retVal;
}

float CUBLAS_SASUM (const int *n, const float *x, const int *incx)
{
    float *devPtrx;
    float retVal;
	int xelem=0;

	xelem= imax(1,*n * abs(*incx));
    
    cublasAlloc (*n * abs(*incx), sizeof(x[0]), (void**)&devPtrx);
    cublasSetVector (*n, sizeof(x[0]), x, *incx, devPtrx, *incx);
    retVal = cublasSasum (*n, devPtrx, *incx);
    cublasFree (devPtrx);
    return retVal;
}

void CUBLAS_SAXPY (const int *n, double *alpha, const float *x, 
                   const int *incx, float *y, const int *incy)
{
    float *devPtrx, *devPtry;
	
    cublasAlloc (*n * abs(*incx), sizeof(x[0]), (void**)&devPtrx);
    cublasAlloc (*n * abs(*incy), sizeof(y[0]), (void**)&devPtry);
    cublasSetVector (*n, sizeof(x[0]), x, abs(*incx), devPtrx, abs(*incx));
    cublasSetVector (*n, sizeof(y[0]), y, abs(*incy), devPtry, abs(*incy));
    cublasSaxpy (*n, (float)*alpha, devPtrx, *incx, devPtry, *incy);
    cublasGetVector (*n, sizeof(y[0]), devPtry, abs(*incy), y, abs(*incy));
    cublasFree (devPtrx);
    cublasFree (devPtry);
}

void CUBLAS_SCOPY (const int *n, const float *x, const int *incx, float *y,
                   const int *incy)
{
    float *devPtrx, *devPtry;

    cublasAlloc (*n * abs(*incx), sizeof(x[0]), (void**)&devPtrx);
    cublasAlloc (*n * abs(*incy), sizeof(y[0]), (void**)&devPtry);
    cublasSetVector (*n, sizeof(x[0]), x, abs(*incx), devPtrx, abs(*incx));
    cublasSetVector (*n, sizeof(y[0]), y, abs(*incy), devPtry, abs(*incy));
    cublasScopy (*n, devPtrx, *incx, devPtry, *incy);
    cublasGetVector (*n, sizeof(y[0]), devPtry, abs(*incy), y, abs(*incy));
    cublasFree (devPtrx);
    cublasFree (devPtry);
}


float CUBLAS_SDOT (const int *n, const float *x, const int *incx, float *y,
                   const int *incy)
{
    float *devPtrx, *devPtry, retVal;

    cublasAlloc (*n * abs(*incx), sizeof(x[0]), (void**)&devPtrx);
    cublasAlloc (*n * abs(*incy), sizeof(y[0]), (void**)&devPtry);
    cublasSetVector (*n, sizeof(x[0]), x, abs(*incx), devPtrx, abs(*incx));
    cublasSetVector (*n, sizeof(y[0]), y, abs(*incy), devPtry, abs(*incy));
    retVal = cublasSdot (*n, devPtrx, *incx, devPtry, *incy);
    cublasFree (devPtrx);
    cublasFree (devPtry);
    return retVal;
}

float CUBLAS_SNRM2 (const int *n, const float *x, const int *incx)
{
    float *devPtrx;
    float retVal;

    cublasAlloc (*n * abs(*incx), sizeof(x[0]), (void**)&devPtrx);
    cublasSetVector (*n, sizeof(x[0]), x, *incx, devPtrx, *incx);
    retVal = cublasSnrm2 (*n, devPtrx, *incx);
    cublasFree (devPtrx);
    return retVal;
}

void CUBLAS_SROT (const int *n, float *x, const int *incx, float *y, 
                  const int *incy, double *sc, double *ss)
{
    float *devPtrx, *devPtry;
	
    cublasAlloc (*n * abs(*incx), sizeof(x[0]), (void**)&devPtrx);
    cublasAlloc (*n * abs(*incy), sizeof(y[0]), (void**)&devPtry);
    cublasSetVector (*n, sizeof(x[0]), x, abs(*incx), devPtrx, abs(*incx));
    cublasSetVector (*n, sizeof(y[0]), y, abs(*incy), devPtry, abs(*incy));
    cublasSrot (*n, devPtrx, *incx, devPtry, *incy, (float)*sc, (float)*ss);
    cublasGetVector (*n, sizeof(x[0]), devPtrx, abs(*incx), x, abs(*incx));
    cublasGetVector (*n, sizeof(y[0]), devPtry, abs(*incy), y, abs(*incy));
    cublasFree (devPtrx);
    cublasFree (devPtry);
}

void CUBLAS_SROTM (const int *n, float *x, const int *incx, float *y, 
                   const int *incy, const float* sparam)
{
    float *devPtrx, *devPtry;

    cublasAlloc (*n * abs(*incx), sizeof(x[0]), (void**)&devPtrx);
    cublasAlloc (*n * abs(*incy), sizeof(y[0]), (void**)&devPtry);
    cublasSetVector (*n, sizeof(x[0]), x, abs(*incx), devPtrx, abs(*incx));
    cublasSetVector (*n, sizeof(y[0]), y, abs(*incy), devPtry, abs(*incy));
    cublasSrotm (*n, devPtrx, *incx, devPtry, *incy, sparam);
    cublasGetVector (*n, sizeof(x[0]), devPtrx, abs(*incx), x, abs(*incx));
    cublasGetVector (*n, sizeof(y[0]), devPtry, abs(*incy), y, abs(*incy));
    cublasFree (devPtrx);
    cublasFree (devPtry);
}

void CUBLAS_SSCAL (const int *n, double *alpha, float *x, const int *incx)
{
    float *devPtrx;	
    
    cublasAlloc (*n * abs(*incx), sizeof(x[0]), (void**)&devPtrx);
    cublasSetVector (*n, sizeof(x[0]), x, *incx, devPtrx, *incx);
    cublasSscal (*n, (float)*alpha, devPtrx, *incx);
    cublasGetVector (*n, sizeof(x[0]), devPtrx, *incx, x, *incx);
    cublasFree (devPtrx); 
}

void CUBLAS_SSWAP (const int *n, float *x, const int *incx, float *y, 
                   const int *incy)
{
    float *devPtrx, *devPtry;

    cublasAlloc (*n * abs(*incx), sizeof(x[0]), (void**)&devPtrx);
    cublasAlloc (*n * abs(*incy), sizeof(y[0]), (void**)&devPtry);
    cublasSetVector (*n, sizeof(x[0]), x, abs(*incx), devPtrx, abs(*incx));
    cublasSetVector (*n, sizeof(y[0]), y, abs(*incy), devPtry, abs(*incy));
    cublasSswap (*n, devPtrx, *incx, devPtry, *incy);
    cublasGetVector (*n, sizeof(x[0]), devPtrx, abs(*incx), x, abs(*incx));
    cublasGetVector (*n, sizeof(y[0]), devPtry, abs(*incy), y, abs(*incy));
    cublasFree (devPtrx);
    cublasFree (devPtry);
}

void CUBLAS_CAXPY (const int *n, const cuComplex *alpha, const cuComplex *x, 
                   const int *incx, cuComplex *y, const int *incy)
{
    cuComplex *devPtrx, *devPtry;

    cublasAlloc (*n * abs(*incx), sizeof(x[0]), (void**)&devPtrx);
    cublasAlloc (*n * abs(*incy), sizeof(y[0]), (void**)&devPtry);
    cublasSetVector (*n, sizeof(x[0]), x, abs(*incx), devPtrx, abs(*incx));
    cublasSetVector (*n, sizeof(y[0]), y, abs(*incy), devPtry, abs(*incy));
    cublasCaxpy (*n, *alpha, devPtrx, *incx, devPtry, *incy);
    cublasGetVector (*n, sizeof(y[0]), devPtry, abs(*incy), y, abs(*incy));
    cublasFree (devPtrx);
    cublasFree (devPtry);
}

void CUBLAS_CCOPY (const int *n, const cuComplex *x, const int *incx, 
                   cuComplex *y, const int *incy)
{
    cuComplex *devPtrx, *devPtry;

    cublasAlloc (*n * abs(*incx), sizeof(x[0]), (void**)&devPtrx);
    cublasAlloc (*n * abs(*incy), sizeof(y[0]), (void**)&devPtry);
    cublasSetVector (*n, sizeof(x[0]), x, abs(*incx), devPtrx, abs(*incx));
    cublasSetVector (*n, sizeof(y[0]), y, abs(*incy), devPtry, abs(*incy));
    cublasCcopy (*n, devPtrx, *incx, devPtry, *incy);
    cublasGetVector (*n, sizeof(y[0]), devPtry, abs(*incy), y, abs(*incy));
    cublasFree (devPtrx);
    cublasFree (devPtry);
}

void CUBLAS_CROT (const int *n, cuComplex *x, const int *incx, cuComplex *y, 
                   const int *incy, const float *sc, const cuComplex *cs)
{
    cuComplex *devPtrx, *devPtry;

    cublasAlloc (*n * abs(*incx), sizeof(x[0]), (void**)&devPtrx);
    cublasAlloc (*n * abs(*incy), sizeof(y[0]), (void**)&devPtry);
    cublasSetVector (*n, sizeof(x[0]), x, abs(*incx), devPtrx, abs(*incx));
    cublasSetVector (*n, sizeof(y[0]), y, abs(*incy), devPtry, abs(*incy));
    cublasCrot (*n, devPtrx, *incx, devPtry, *incy, *sc, *cs);
    cublasGetVector (*n, sizeof(x[0]), devPtrx, abs(*incx), x, abs(*incx));
    cublasGetVector (*n, sizeof(y[0]), devPtry, abs(*incy), y, abs(*incy));
    cublasFree (devPtrx);
    cublasFree (devPtry);
}

void CUBLAS_CROTG (cuComplex *ca, const cuComplex *cb, float *sc,
                   cuComplex *cs)
{	
    cublasCrotg (ca, *cb, sc, cs);
}

void CUBLAS_CSCAL (const int *n, const cuComplex *alpha, cuComplex *x, 
                   const int *incx)
{
    cuComplex *devPtrx;

    cublasAlloc (*n * abs(*incx), sizeof(x[0]), (void**)&devPtrx);
    cublasSetVector (*n, sizeof(x[0]), x, *incx, devPtrx, *incx);
    cublasCscal (*n, *alpha, devPtrx, *incx);
    cublasGetVector (*n, sizeof(x[0]), devPtrx, *incx, x, *incx);
    cublasFree (devPtrx); 
}

void CUBLAS_CSROT (const int *n, cuComplex *x, const int *incx, cuComplex *y, 
                   const int *incy, const float *sc, const float *ss)
{
    cuComplex *devPtrx, *devPtry;

    cublasAlloc (*n * abs(*incx), sizeof(x[0]), (void**)&devPtrx);
    cublasAlloc (*n * abs(*incy), sizeof(y[0]), (void**)&devPtry);
    cublasSetVector (*n, sizeof(x[0]), x, abs(*incx), devPtrx, abs(*incx));
    cublasSetVector (*n, sizeof(y[0]), y, abs(*incy), devPtry, abs(*incy));
    cublasCsrot (*n, devPtrx, *incx, devPtry, *incy, *sc, *ss);
    cublasGetVector (*n, sizeof(x[0]), devPtrx, abs(*incx), x, abs(*incx));
    cublasGetVector (*n, sizeof(y[0]), devPtry, abs(*incy), y, abs(*incy));
    cublasFree (devPtrx);
    cublasFree (devPtry);
}

void CUBLAS_CSSCAL (const int *n, double *alpha, cuComplex *x, 
                    const int *incx)
{
    cuComplex *devPtrx;

    cublasAlloc (*n * abs(*incx), sizeof(x[0]), (void**)&devPtrx);
    cublasSetVector (*n, sizeof(x[0]), x, *incx, devPtrx, *incx);
    cublasCsscal (*n, (float)*alpha, devPtrx, *incx);
    cublasGetVector (*n, sizeof(x[0]), devPtrx, *incx, x, *incx);
    cublasFree (devPtrx); 
}

void CUBLAS_CSWAP (const int *n, cuComplex *x, const int *incx, cuComplex *y,
                   const int *incy)
{
    cuComplex *devPtrx, *devPtry;

    cublasAlloc (*n * abs(*incx), sizeof(x[0]), (void**)&devPtrx);
    cublasAlloc (*n * abs(*incy), sizeof(y[0]), (void**)&devPtry);
    cublasSetVector (*n, sizeof(x[0]), x, abs(*incx), devPtrx, abs(*incx));
    cublasSetVector (*n, sizeof(y[0]), y, abs(*incy), devPtry, abs(*incy));
    cublasCswap (*n, devPtrx, *incx, devPtry, *incy);
    cublasGetVector (*n, sizeof(x[0]), devPtrx, abs(*incx), x, abs(*incx));
    cublasGetVector (*n, sizeof(y[0]), devPtry, abs(*incy), y, abs(*incy));
    cublasFree (devPtrx);
    cublasFree (devPtry);
}

void CUBLAS_CDOTU (cuComplex *retVal, const int *n, const cuComplex *x, 
                   const int *incx, const cuComplex *y, const int *incy)
{
    cuComplex *devPtrx, *devPtry;

    cublasAlloc (*n * abs(*incx), sizeof(x[0]), (void**)&devPtrx);
    cublasAlloc (*n * abs(*incy), sizeof(y[0]), (void**)&devPtry);
    cublasSetVector (*n, sizeof(x[0]), x, abs(*incx), devPtrx, abs(*incx));
    cublasSetVector (*n, sizeof(y[0]), y, abs(*incy), devPtry, abs(*incy));
    *retVal = cublasCdotu (*n, devPtrx, *incx, devPtry, *incy);
    cublasFree (devPtrx);
    cublasFree (devPtry);
}

void CUBLAS_CDOTC (cuComplex *retVal, const int *n, const cuComplex *x, 
                   const int *incx, const cuComplex *y, const int *incy)
{
    cuComplex *devPtrx, *devPtry;

    cublasAlloc (*n * abs(*incx), sizeof(x[0]), (void**)&devPtrx);
    cublasAlloc (*n * abs(*incy), sizeof(y[0]), (void**)&devPtry);
    cublasSetVector (*n, sizeof(x[0]), x, abs(*incx), devPtrx, abs(*incx));
    cublasSetVector (*n, sizeof(y[0]), y, abs(*incy), devPtry, abs(*incy));
    *retVal = cublasCdotc (*n, devPtrx, *incx, devPtry, *incy);
    cublasFree (devPtrx);
    cublasFree (devPtry);
}

int CUBLAS_ICAMAX (const int *n, const cuComplex *x, const int *incx)
{
    cuComplex *devPtrx;
    int retVal;

    cublasAlloc (*n * abs(*incx), sizeof(x[0]), (void**)&devPtrx);
    cublasSetVector (*n, sizeof(x[0]), x, *incx, devPtrx, *incx);
    retVal = cublasIcamax (*n, devPtrx, *incx);
    cublasFree (devPtrx);
    return retVal;
}

int CUBLAS_ICAMIN (const int *n, const cuComplex *x, const int *incx)
{
    cuComplex *devPtrx;
    int retVal;	

    cublasAlloc (*n * abs(*incx), sizeof(x[0]), (void**)&devPtrx);
    cublasSetVector (*n, sizeof(x[0]), x, *incx, devPtrx, *incx);
    retVal = cublasIcamin (*n, devPtrx, *incx);
    cublasFree (devPtrx);
    return retVal;
}

float CUBLAS_SCASUM (const int *n, const cuComplex *x, const int *incx)
{
    cuComplex *devPtrx;
    float retVal;	
    
    cublasAlloc (*n * abs(*incx), sizeof(x[0]), (void**)&devPtrx);
    cublasSetVector (*n, sizeof(x[0]), x, *incx, devPtrx, *incx);
    retVal = cublasScasum (*n, devPtrx, *incx);
    cublasFree (devPtrx);
    return retVal;
}

float CUBLAS_SCNRM2 (const int *n, const cuComplex *x, const int *incx)
{
    cuComplex *devPtrx;
    float retVal;

    cublasAlloc (*n * abs(*incx), sizeof(x[0]), (void**)&devPtrx);
    cublasSetVector (*n, sizeof(x[0]), x, *incx, devPtrx, *incx);
    retVal = cublasScnrm2 (*n, devPtrx, *incx);
    cublasFree (devPtrx);
    return retVal;
}

int CUBLAS_IDAMAX (const int *n, double *x, const int *incx)
{
    double *devPtrx;
    int retVal;
	unsigned int xelem=0;
	float *cpy_x=NULL;

	//print_func_name("idamax");

	if (double_capable){
		/*print_func_name("idamax");
		print_vector((void *)x, DOUBLE, xelem);
		print_vector((void *)n, INT,1);
		print_vector((void *)incx, INT,1);*/
		cublasAlloc (*n * abs(*incx), sizeof(x[0]), (void**)&devPtrx);
		cublasSetVector (*n, sizeof(x[0]), x, *incx, devPtrx, *incx);
		retVal = cublasIdamax (*n, devPtrx, *incx);
		cublasFree (devPtrx);
	}
	else{
		xelem= imax(1,*n * abs(*incx));
		cpy_x=copy_n_truncate (x, xelem);
		retVal = CUBLAS_ISAMAX (n, cpy_x, incx);
		free((void *)cpy_x);
	}
	//print_vector((void *)&retVal, INT, 1);
	return retVal;
}

int CUBLAS_IDAMIN (const int *n, double *x, const int *incx)
{
    double *devPtrx;
    int retVal;
	unsigned int xelem=0;
	float *cpy_x=NULL;

	//print_func_name("idamin");

	if (double_capable){
		cublasAlloc (*n * abs(*incx), sizeof(x[0]), (void**)&devPtrx);
		cublasSetVector (*n, sizeof(x[0]), x, *incx, devPtrx, *incx);
		retVal = cublasIdamin (*n, devPtrx, *incx);
		cublasFree (devPtrx);
	}
	else {
		xelem= imax(1,*n * abs(*incx));
		cpy_x=copy_n_truncate (x, xelem);
		retVal = CUBLAS_ISAMIN (n, cpy_x, incx);
		free((void *)cpy_x);
	}
	//print_vector((void *)&retVal, INT, 1);
    return retVal;
}

double CUBLAS_DASUM (const int *n, double *x, const int *incx)
{
    double *devPtrx;
    double retVal;
	unsigned int xelem=0;
	float *cpy_x=NULL;

	//print_func_name("dasum");

	if (double_capable){  
		/*print_func_name ("dasum");
		print_vector((void *)x, DOUBLE, xelem);*/
		cublasAlloc (*n * abs(*incx), sizeof(x[0]), (void**)&devPtrx);
		cublasSetVector (*n, sizeof(x[0]), x, *incx, devPtrx, *incx);
		retVal = cublasDasum (*n, devPtrx, *incx);
		cublasFree (devPtrx);
		/*print_vector((void *)&retVal, INT, 1);*/
	}
	else {
		xelem= imax(1,*n * abs(*incx));
		cpy_x=copy_n_truncate (x, xelem);
		retVal = (double)CUBLAS_SASUM (n, cpy_x, incx);
		free((void *)cpy_x);
	}
	
	//print_vector((void *)&retVal, INT, 1);
    return retVal;
}

void CUBLAS_DAXPY (const int *n, double *alpha, const double *x, 
                   const int *incx, double *y, const int *incy)
{
    double *devPtrx, *devPtry;
	int xelem=0, yelem=0;
	double *tempX=NULL, *tempY=NULL;
	float *cpy_x=NULL;

	//print_func_name("daxpy");
	
	if (double_capable){ 
		/*print_func_name ("daxpy");
		print_vector((void *)x, DOUBLE, xelem);
		print_vector((void *)y, DOUBLE, yelem);
		print_vector((void *)alpha, DOUBLE, 1);*/
		cublasAlloc (*n * abs(*incx), sizeof(x[0]), (void**)&devPtrx);
		cublasAlloc (*n * abs(*incy), sizeof(y[0]), (void**)&devPtry);
		cublasSetVector (*n, sizeof(x[0]), x, abs(*incx), devPtrx, abs(*incx));
		cublasSetVector (*n, sizeof(y[0]), y, abs(*incy), devPtry, abs(*incy));
		cublasDaxpy (*n, *alpha, devPtrx, *incx, devPtry, *incy);
		cublasGetVector (*n, sizeof(y[0]), devPtry, abs(*incy), y, abs(*incy));
		cublasFree (devPtrx);
		cublasFree (devPtry);
	}
	else{
		xelem= imax(1,*n * abs(*incx));
		yelem= imax(1,*n * abs(*incy));
		cpy_x=copy_n_truncate(x,xelem);
		truncate ((void *)y, yelem);/*truncate y*/		
		CUBLAS_SAXPY (n, alpha, cpy_x, incx, (float *)y, incy);
		untruncate ((void *)y, yelem);
		free (cpy_x);
	}
	//print_vector((void *)y, DOUBLE, yelem);
}

void CUBLAS_DCOPY (const int *n, const double *x, const int *incx, double *y,
                   const int *incy)
{
    double *devPtrx, *devPtry;
	int xelem=0, yelem=0;
	double *tempX=NULL, *tempY=NULL;
	float *cpy_x=NULL;

	//print_func_name("dcopy");
		
	if (double_capable){ 
		cublasAlloc (*n * abs(*incx), sizeof(x[0]), (void**)&devPtrx);
		cublasAlloc (*n * abs(*incy), sizeof(y[0]), (void**)&devPtry);
		cublasSetVector (*n, sizeof(x[0]), x, abs(*incx), devPtrx, abs(*incx));
		cublasSetVector (*n, sizeof(y[0]), y, abs(*incy), devPtry, abs(*incy));
		cublasDcopy (*n, devPtrx, *incx, devPtry, *incy);
		cublasGetVector (*n, sizeof(y[0]), devPtry, abs(*incy), y, abs(*incy));
		cublasFree (devPtrx);
		cublasFree (devPtry);
	}
	else {
		xelem= imax(1,*n * abs(*incx));
		yelem= imax(1,*n * abs(*incy));
		cpy_x=copy_n_truncate(x,xelem);
		truncate ((void *)y, yelem);/*truncate y*/
		CUBLAS_SCOPY (n, cpy_x, incx, (float *)y, incy);
		untruncate ((void *)y, yelem);
		free (cpy_x);
	}
	//print_vector((void *)y, DOUBLE, yelem);
}

double CUBLAS_DDOT (const int *n, double *x, const int *incx, double *y,
                    const int *incy)
{
    double *devPtrx, *devPtry, retVal;
	int xelem=0, yelem=0;
	double *tempX=NULL, *tempY=NULL;
	float *cpy_x=NULL, *cpy_y=NULL;

	//print_func_name("ddot");

	xelem= imax(1,*n * abs(*incx));
	yelem= imax(1,*n * abs(*incy));
	
	if (double_capable){ 
		cublasAlloc (*n * abs(*incx), sizeof(x[0]), (void**)&devPtrx);
		cublasAlloc (*n * abs(*incy), sizeof(y[0]), (void**)&devPtry);
		cublasSetVector (*n, sizeof(x[0]), x, abs(*incx), devPtrx, abs(*incx));
		cublasSetVector (*n, sizeof(y[0]), y, abs(*incy), devPtry, abs(*incy));
		retVal = cublasDdot (*n, devPtrx, *incx, devPtry, *incy);
		cublasFree (devPtrx);
		cublasFree (devPtry);
	}
	else{
		cpy_x=copy_n_truncate(x,xelem);
		cpy_y=copy_n_truncate(y,yelem);
		retVal=(double)CUBLAS_SDOT (n, cpy_x, incx, cpy_y, incy);
		free((void *)cpy_x);
		free((void *)cpy_y);
	}
	//print_vector(&retVal, INT, 1);
    return retVal;
}

double CUBLAS_DNRM2 (const int *n, double *x, const int *incx)
{
    double *devPtrx;
    double retVal;
	int xelem=0;
	float *cpy_x=NULL;

	//print_func_name("dnrm2");

	xelem= imax(1,*n * abs(*incx));
	
	if (double_capable){ 
		cublasAlloc (*n * abs(*incx), sizeof(x[0]), (void**)&devPtrx);
		cublasSetVector (*n, sizeof(x[0]), x, *incx, devPtrx, *incx);
		retVal = cublasDnrm2 (*n, devPtrx, *incx);
		cublasFree (devPtrx);
	}
	else {
		cpy_x=copy_n_truncate(x,xelem);
		retVal=(double)CUBLAS_SNRM2 (n, cpy_x, incx);
		free((void *)cpy_x);
	}
	//print_vector(&retVal, INT, 1);
    return retVal;
}

void CUBLAS_DROT (const int *n, double *x, const int *incx, double *y, 
                  const int *incy, double *sc, double *ss)
{
    double *devPtrx, *devPtry;
	int xelem=0, yelem=0;
	double *tempX=NULL, *tempY=NULL;
	float *cpy_x=NULL;

	xelem= imax(1,*n * abs(*incx));
	yelem= imax(1,*n * abs(*incy));
	//print_func_name("drot");
	
	if (double_capable){ 
		cublasAlloc (*n * abs(*incx), sizeof(x[0]), (void**)&devPtrx);
		cublasAlloc (*n * abs(*incy), sizeof(y[0]), (void**)&devPtry);
		cublasSetVector (*n, sizeof(x[0]), x, abs(*incx), devPtrx, abs(*incx));
		cublasSetVector (*n, sizeof(y[0]), y, abs(*incy), devPtry, abs(*incy));
		cublasDrot (*n, devPtrx, *incx, devPtry, *incy, *sc, *ss);
		cublasGetVector (*n, sizeof(x[0]), devPtrx, abs(*incx), x, abs(*incx));
		cublasGetVector (*n, sizeof(y[0]), devPtry, abs(*incy), y, abs(*incy));
		cublasFree (devPtrx);
		cublasFree (devPtry);
	}
	else {
		cpy_x=copy_n_truncate(x,xelem);
		truncate(y,yelem);
		CUBLAS_SROT (n, cpy_x, incx, (float *)y, incy, sc, ss);
		untruncate ((void *)y, yelem);
		copy_untruncate(cpy_x, x, xelem);
		free((void *)cpy_x);
	}
	//print_vector((void *)y, DOUBLE, yelem);
	//print_vector((void *)x, DOUBLE, xelem);
}

void CUBLAS_DROTG (double *sa, double *sb, double *sc, double *ss)
{
	float *da, *db, *dc, *ds;
	if (double_capable)
		cublasDrotg (sa, sb, sc, ss);
	else {
		*da=(float)*sa;
		*db=(float)*sb;
		*dc=(float)*sc;
		*ds=(float)*ss;
		cublasSrotg ((float *)sa,(float *)sb, (float *)sc, (float *)ss);
		*sa=(double)*da;
		*sb=(double)*db;
		*sc=(double)*dc;
		*ss=(double)*ds;
	}
	//print_func_name("drotg");
}

void CUBLAS_DROTM (const int *n, double *x, const int *incx, double *y, 
                   const int *incy, double* sparam)
{
    double *devPtrx, *devPtry;
	unsigned int xelem=0, yelem=0;
	double *tempX=NULL, *tempY=NULL;
	float *cpy_x=NULL;

	if (double_capable){
		cublasAlloc (*n * abs(*incx), sizeof(x[0]), (void**)&devPtrx);
		cublasAlloc (*n * abs(*incy), sizeof(y[0]), (void**)&devPtry);
		cublasSetVector (*n, sizeof(x[0]), x, abs(*incx), devPtrx, abs(*incx));
		cublasSetVector (*n, sizeof(y[0]), y, abs(*incy), devPtry, abs(*incy));
		cublasDrotm (*n, devPtrx, *incx, devPtry, *incy, sparam);
		cublasGetVector (*n, sizeof(x[0]), devPtrx, abs(*incx), x, abs(*incx));
		cublasGetVector (*n, sizeof(y[0]), devPtry, abs(*incy), y, abs(*incy));
		cublasFree (devPtrx);
		cublasFree (devPtry);
	}
	else {
		xelem= imax(1,*n * abs(*incx));
		yelem= imax(1,*n * abs(*incy));
		cpy_x=copy_n_truncate(x,xelem);
		truncate(y,yelem);
		CUBLAS_SROTM (n, cpy_x, incx, (float *)y, incy, (float *) sparam);
		untruncate ((void *)y, yelem);
		copy_untruncate(cpy_x, x, xelem);
		free((void *)cpy_x);
	}
	//print_func_name("drotm");
}

void CUBLAS_DROTMG (double *sd1, double *sd2, double *sx1, const double *sy1,
                    double* sparam)
{	
	float *dd1, *dd2, *dx1, *dy1;
	if (double_capable)
		cublasDrotmg (sd1, sd2, sx1, sy1, sparam);
	else {
		*dd1=(float)*sd1;
		*dd2=(float)*sd2;
		*dx1=(float)*dx1;
		*dy1=(float)*dy1;
		truncate ((void *)sparam, 5);
		cublasSrotmg(dd1, dd2, dx1, dy1, (float *) sparam);
		untruncate ((void *)sparam, 5);
		*sx1=(double)*dx1;
		*sd2=(double)*dd2;
		*sd1=(double)*dd1;		
	}
	//print_func_name("drotmg");
}

void CUBLAS_DSCAL (const int *n, double *alpha, double *x, const int *incx)
{
    double *devPtrx;
	int xelem=0;
	
	xelem= imax(1,*n * abs(*incx));
	//print_func_name("dscal");
	if (double_capable){ 
		/*print_func_name ("dscal");
		print_vector((void *)x, DOUBLE, xelem);
		print_vector((void *)alpha, DOUBLE, 1);*/
		cublasAlloc (*n * abs(*incx), sizeof(x[0]), (void**)&devPtrx);
		cublasSetVector (*n, sizeof(x[0]), x, *incx, devPtrx, *incx);
		cublasDscal (*n, *alpha, devPtrx, *incx);
		cublasGetVector (*n, sizeof(x[0]), devPtrx, *incx, x, *incx);
		/*print_vector((void *)x, DOUBLE, xelem);*/
		cublasFree (devPtrx); 		
	}
	else{
		truncate ((void *)x, xelem);
		CUBLAS_SSCAL (n, alpha, (float *)x, incx);
		untruncate ((void *)x, xelem);
	}
	//print_vector((void *)x, DOUBLE, xelem);
}

void CUBLAS_DSWAP (const int *n, double *x, const int *incx, double *y, 
                   const int *incy)
{
    double *devPtrx, *devPtry;
	int xelem=0, yelem=0;
	double *tempX=NULL, *tempY=NULL;
	float *cpy_x=NULL, *cpy_y=NULL;	
	
	if (double_capable){ 
		/*print_func_name("dswap");
		print_vector((void *)incx, INT, 1);
		print_vector((void *)incy, INT, 1);
		print_vector((void *)n, INT, 1);
		print_vector((void *)x, DOUBLE, xelem);
		print_vector((void *)y, DOUBLE, yelem);*/
		cublasAlloc (*n * abs(*incx), sizeof(x[0]), (void**)&devPtrx);
		cublasAlloc (*n * abs(*incy), sizeof(y[0]), (void**)&devPtry);
		cublasSetVector (*n, sizeof(x[0]), x, abs(*incx), devPtrx, abs(*incx));
		cublasSetVector (*n, sizeof(y[0]), y, abs(*incy), devPtry, abs(*incy));
		cublasDswap (*n, devPtrx, *incx, devPtry, *incy);
		cublasGetVector (*n, sizeof(x[0]), devPtrx, abs(*incx), x, abs(*incx));
		cublasGetVector (*n, sizeof(y[0]), devPtry, abs(*incy), y, abs(*incy));
		/*print_vector((void *)x, DOUBLE, xelem);
		print_vector((void *)y, DOUBLE, yelem);*/
		cublasFree (devPtrx);
		cublasFree (devPtry);
	}
	else {
		xelem= imax(1,*n * abs(*incx));
		yelem= imax(1,*n * abs(*incy));
		cpy_x=copy_n_truncate(x,xelem);
		cpy_y=copy_n_truncate(y,yelem);
		CUBLAS_SSWAP (n, cpy_x, incx, cpy_y, incy);
		copy_untruncate(cpy_x,x,xelem);
		copy_untruncate(cpy_y,y,yelem);
		free(cpy_x);
		free(cpy_y);
	}
}


/*---------------------------------------------------------------------------*/
/*---------------------------------- BLAS2 ----------------------------------*/
/*---------------------------------------------------------------------------*/

void CUBLAS_SGBMV (const char *trans, const int *m, const int *n, 
                   const int *kl, const int *ku, double *alpha, 
                   const float *A, const int *lda, const float *x, 
                   const int *incx, double *beta, float *y, 
                   const int *incy)
{
    float *devPtrx, *devPtry, *devPtrA;	

    /*  X      - REAL             array of DIMENSION at least
     *           ( 1 + ( n - 1 )*abs( INCX ) ) when TRANS = 'N' or 'n'
     *           and at least
     *           ( 1 + ( m - 1 )*abs( INCX ) ) otherwise.
     */
    if (toupper(trans[0]) == 'N') {
        cublasAlloc (*n * abs(*incx), sizeof(x[0]), (void**)&devPtrx);
        cublasSetVector (*n, sizeof(x[0]), x, abs(*incx), devPtrx, abs(*incx));
    } else {
        cublasAlloc (*m * abs(*incx), sizeof(x[0]), (void**)&devPtrx);
        cublasSetVector (*m, sizeof(x[0]), x, abs(*incx), devPtrx, abs(*incx));
    }
    /*  Y      - REAL             array of DIMENSION at least
     *           ( 1 + ( m - 1 )*abs( INCY ) ) when TRANS = 'N' or 'n'
     *           and at least
     *           ( 1 + ( n - 1 )*abs( INCY ) ) otherwise.
     */
    if (toupper(trans[0]) == 'N') {
        cublasAlloc (*m * abs(*incy), sizeof(y[0]), (void**)&devPtry);
        cublasSetVector (*m, sizeof(y[0]), y, abs(*incy), devPtry, abs(*incy));
    } else {
        cublasAlloc (*n * abs(*incy), sizeof(y[0]), (void**)&devPtry);
        cublasSetVector (*n, sizeof(y[0]), y, abs(*incy), devPtry, abs(*incy));
    }       
    /* A      - REAL             array of DIMENSION ( LDA, n ). 
     * Before entry, the leading ( kl + ku + 1 ) by n part of the
     * array A must contain the matrix of coefficients, supplied
     */
    cublasAlloc ((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    cublasSetMatrix (imin(*kl+*ku+1,*lda), *n, sizeof(A[0]), A, *lda, devPtrA, 
                     *lda);
    cublasSgbmv (trans[0], *m, *n, *kl, *ku, (float)*alpha, devPtrA, *lda, devPtrx, 
                 *incx, (float)*beta, devPtry, *incy);
    if (toupper(trans[0]) == 'N') {
        cublasGetVector (*m, sizeof(y[0]), devPtry, abs(*incy), y, abs(*incy));
    } else {
        cublasGetVector (*n, sizeof(y[0]), devPtry, abs(*incy), y, abs(*incy));
    }       
    cublasFree (devPtrA);
    cublasFree (devPtry);
    cublasFree (devPtrx);
}

void CUBLAS_SGEMV (const char *trans, const int *m, const int *n,
                   double *alpha, const float *A, const int *lda,
                   const float *x, const int *incx, double *beta,
                   float *y, const int *incy)
{
    float *devPtrA, *devPtrx, *devPtry;	
    
    /*  X      - REAL             array of DIMENSION at least
     *           ( 1 + ( n - 1 )*abs( INCX ) ) when TRANS = 'N' or 'n'
     *           and at least
     *           ( 1 + ( m - 1 )*abs( INCX ) ) otherwise.
     */
    if (toupper(trans[0]) == 'N') {
        cublasAlloc (*n * abs(*incx), sizeof(x[0]), (void**)&devPtrx);
        cublasSetVector (*n, sizeof(x[0]), x, abs(*incx), devPtrx, abs(*incx));
    } else {
        cublasAlloc (*m * abs(*incx), sizeof(x[0]), (void**)&devPtrx);
        cublasSetVector (*m, sizeof(x[0]), x, abs(*incx), devPtrx, abs(*incx));
    }
    /*  Y      - REAL             array of DIMENSION at least
     *           ( 1 + ( m - 1 )*abs( INCY ) ) when TRANS = 'N' or 'n'
     *           and at least
     *           ( 1 + ( n - 1 )*abs( INCY ) ) otherwise.
     */
    if (toupper(trans[0]) == 'N') {
        cublasAlloc (*m * abs(*incy), sizeof(y[0]), (void**)&devPtry);
        cublasSetVector (*m, sizeof(y[0]), y, abs(*incy), devPtry, abs(*incy));
    } else {
        cublasAlloc (*n * abs(*incy), sizeof(y[0]), (void**)&devPtry);
        cublasSetVector (*n, sizeof(y[0]), y, abs(*incy), devPtry, abs(*incy));
    }       
    /*  A      - REAL             array of DIMENSION ( LDA, n ).
     *           Before entry, the leading m by n part of the array A must
     *           contain the matrix of coefficients.
     */
    cublasAlloc ((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    cublasSetMatrix (imin(*m,*lda), *n, sizeof(A[0]), A, *lda, devPtrA, *lda);
    cublasSgemv (trans[0], *m, *n, (float)*alpha, devPtrA, *lda, devPtrx, *incx,
                 (float) *beta, devPtry, *incy);
    if (toupper(trans[0]) == 'N') {
        cublasGetVector (*m, sizeof(y[0]), devPtry, abs(*incy), y, abs(*incy));
    } else {
        cublasGetVector (*n, sizeof(y[0]), devPtry, abs(*incy), y, abs(*incy));
    }       
    cublasFree (devPtrA);
    cublasFree (devPtry);
    cublasFree (devPtrx);
}

void CUBLAS_SGER (const int *m, const int *n, double *alpha, 
                  const float *x, const int *incx, const float *y,
                  const int *incy, float *A, const int *lda)
{
    float *devPtrA, *devPtrx, *devPtry;	

    cublasAlloc (*m * abs(*incx), sizeof(devPtrx[0]), (void**)&devPtrx);
    cublasAlloc (*n * abs(*incy), sizeof(devPtry[0]), (void**)&devPtry);
    cublasSetVector (*m* abs(*incx), sizeof(x[0]), x, 1, devPtrx, 1);
    cublasSetVector (*n* abs(*incy), sizeof(y[0]), y, 1, devPtry, 1);

    // REAL array of DIMENSION ( LDA, n ).
    //      Before entry, the leading m by n part of the array A must
    //      contain the matrix of coefficients. On exit, A is
    cublasAlloc ((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    cublasSetMatrix (imin(*m,*lda), *n, sizeof(A[0]), A, *lda, devPtrA, *lda);
    cublasSger (*m, *n, (float)*alpha, devPtrx, *incx, devPtry, *incy, devPtrA, *lda);
    cublasGetMatrix (imin(*m,*lda), *n, sizeof(A[0]), devPtrA, *lda, A, *lda);
    cublasFree (devPtrA);
    cublasFree (devPtrx);
    cublasFree (devPtry);
}

void CUBLAS_SSBMV (const char *uplo, const int *n, const int *k, 
                   double *alpha, const float *A, const int *lda,
                   const float *x, const int *incx, double *beta, 
                   float *y, const int *incy)
{
    float *devPtrA, *devPtrx, *devPtry;

    /*  X      - REAL             array of DIMENSION at least
     *           ( 1 + ( n - 1 )*abs( INCX ) ).
     */
    cublasAlloc (*n * abs(*incx), sizeof(devPtrx[0]), (void**)&devPtrx);
    cublasSetVector (*n* abs(*incx), sizeof(x[0]), x, 1, devPtrx, 1);
    /*  Y      - REAL             array of DIMENSION at least
     *           ( 1 + ( n - 1 )*abs( INCY ) ).
     */
    cublasAlloc (*n * abs(*incy), sizeof(devPtry[0]), (void**)&devPtry);
    cublasSetVector (*n* abs(*incy), sizeof(y[0]), y, 1, devPtry, 1);    
    /*  A      - REAL             array of DIMENSION ( LDA, n ).
     *           Before entry with UPLO = 'U' or 'u', the leading ( k + 1 )
     *           by n part of the array A must contain the upper triangular
     */
    cublasAlloc ((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    cublasSetMatrix (imin(*k+1,*lda), *n, sizeof(A[0]), A, *lda, devPtrA, *lda);
    cublasSsbmv (uplo[0], *n, *k, (float)*alpha, devPtrA, *lda, devPtrx, *incx, (float) *beta,
                 devPtry, *incy);
    cublasGetVector (*n* abs(*incy), sizeof(y[0]), devPtry, 1, y, 1);
    cublasFree (devPtrA);
    cublasFree (devPtrx);
    cublasFree (devPtry);
}

void CUBLAS_SSPMV (const char *uplo, const int *n, double *alpha,
                   const float *AP, const float *x, const int *incx, 
                   double *beta, float *y, const int *incy)
{
    float *devPtrAP, *devPtrx, *devPtry;

    /*  X      - REAL             array of dimension at least
     *           ( 1 + ( n - 1 )*abs( INCX ) ).
     */
    cublasAlloc (*n * abs(*incx), sizeof(devPtrx[0]), (void**)&devPtrx);
    cublasSetVector (*n* abs(*incx), sizeof(x[0]), x, 1, devPtrx, 1);
    /*  Y      - REAL             array of dimension at least
     *           ( 1 + ( n - 1 )*abs( INCY ) ).
     */
    cublasAlloc (*n * abs(*incy), sizeof(devPtry[0]), (void**)&devPtry);
    cublasSetVector (*n* abs(*incy), sizeof(y[0]), y, 1, devPtry, 1);
    /*  AP     - REAL             array of DIMENSION at least
     *           ( ( n*( n + 1 ) )/2 ).
     *           Before entry with UPLO = 'U' or 'u', the array AP must
     *           contain the upper triangular part of the symmetric matrix
     */
    cublasAlloc (((*n)*(*n+1))/2, sizeof(devPtrAP[0]), (void**)&devPtrAP);
    cublasSetVector (((*n)*(*n+1))/2, sizeof(AP[0]), AP, 1, devPtrAP, 1);
    cublasSspmv (*uplo, *n, (float)*alpha, devPtrAP, devPtrx, *incx, (float) *beta, devPtry,
                 *incy);
    cublasGetVector ((*n) * abs(*incy), sizeof(y[0]), devPtry, 1, y, 1);
    cublasFree (devPtrAP);
    cublasFree (devPtrx);
    cublasFree (devPtry);
}

void CUBLAS_SSPR (const char *uplo, const int *n, double *alpha, 
                    const float *x, const int *incx, float *AP)
{
    float *devPtrAP, *devPtrx;

    /*  X      - REAL             array of dimension at least
     *           ( 1 + ( n - 1 )*abs( INCX ) ).
     */
    cublasAlloc (*n * abs(*incx), sizeof(devPtrx[0]), (void**)&devPtrx);
    cublasSetVector (*n* abs(*incx), sizeof(x[0]), x, 1, devPtrx, 1);
    /*  AP     - REAL             array of DIMENSION at least
     *           ( ( n*( n + 1 ) )/2 ).
     */
    cublasAlloc (((*n) * (*n+1))/2, sizeof(devPtrAP[0]), (void**)&devPtrAP);
    cublasSetVector (((*n) * (*n+1))/2, sizeof(AP[0]), AP, 1, devPtrAP, 1);
    cublasSspr (uplo[0], *n, (float)*alpha, devPtrx, *incx, devPtrAP);
    cublasGetVector (((*n) * (*n+1))/2, sizeof(AP[0]), devPtrAP, 1, AP, 1);
    cublasFree (devPtrAP);
    cublasFree (devPtrx);
}

void CUBLAS_SSPR2 (const char *uplo, const int *n, double *alpha,
                   const float *x, const int *incx, const float *y, 
                   const int *incy, float *AP)
{
    float *devPtrAP, *devPtrx, *devPtry;

    /*  X      - REAL             array of dimension at least
     *           ( 1 + ( n - 1 )*abs( INCX ) ).
     */
    cublasAlloc (*n * abs(*incx), sizeof(devPtrx[0]), (void**)&devPtrx);
    cublasSetVector (*n* abs(*incx), sizeof(x[0]), x, 1, devPtrx, 1);
    /*  Y      - REAL             array of dimension at least
     *           ( 1 + ( n - 1 )*abs( INCY ) ).
     */
    cublasAlloc (*n * abs(*incy), sizeof(devPtry[0]), (void**)&devPtry);
    cublasSetVector (*n* abs(*incy), sizeof(y[0]), y, 1, devPtry, 1);
   /*  AP     - REAL             array of DIMENSION at least
    *           ( ( n*( n + 1 ) )/2 ).
    *           Before entry with  UPLO = 'U' or 'u', the array AP must
    *           contain the upper triangular part of the symmetric matrix
    */
    cublasAlloc (((*n) * (*n+1))/2, sizeof(devPtrAP[0]), (void**)&devPtrAP);
    cublasSetVector (((*n) * (*n+1))/2, sizeof(AP[0]), AP, 1, devPtrAP, 1);
    cublasSspr2 (uplo[0], *n, (float)*alpha, devPtrx, *incx, devPtry, *incy,devPtrAP);
    cublasGetVector (((*n) * (*n+1))/2, sizeof(AP[0]), devPtrAP, 1, AP, 1);
    cublasFree (devPtrAP);
    cublasFree (devPtrx);
    cublasFree (devPtry);
}

void CUBLAS_SSYMV (const char *uplo, const int *n, double *alpha,
                   const float *A, const int *lda, const float *x, 
                   const int *incx, double *beta, float *y, 
                   const int *incy)
{
    float *devPtrA, *devPtrx, *devPtry;

    /*  X      - REAL             array of dimension at least
     *           ( 1 + ( n - 1 )*abs( INCX ) ).
     */
    cublasAlloc (*n * abs(*incx), sizeof(devPtrx[0]), (void**)&devPtrx);
    cublasSetVector (*n* abs(*incx), sizeof(x[0]), x, 1, devPtrx, 1);
    /*  Y      - REAL             array of dimension at least
     *           ( 1 + ( n - 1 )*abs( INCY ) ).
     */
    cublasAlloc (*n * abs(*incy), sizeof(devPtry[0]), (void**)&devPtry);
    cublasSetVector (*n* abs(*incy), sizeof(y[0]), y, 1, devPtry, 1);
    /*  A      - REAL             array of DIMENSION ( LDA, n ).
     *           Before entry with  UPLO = 'U' or 'u', the leading n by n
     *           upper triangular part of the array A must contain the upper
     */
    cublasAlloc ((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    cublasSetMatrix (imin(*n,*lda), *n, sizeof(A[0]), A, *lda, devPtrA, *lda);
    cublasSsymv (uplo[0], *n, (float)*alpha, devPtrA, *lda, devPtrx, *incx, (float) *beta,
                 devPtry, *incy);
    cublasGetVector (*n * abs(*incy), sizeof(y[0]), devPtry, 1, y, 1);
    cublasFree (devPtrA);
    cublasFree (devPtrx);
    cublasFree (devPtry);
}

void CUBLAS_SSYR (const char *uplo, const int *n, double *alpha, 
                  const float *x, const int *incx, float *A, const int *lda)
{
    float *devPtrA, *devPtrx;

    /*  X      - REAL             array of dimension at least
     *           ( 1 + ( n - 1 )*abs( INCX ) ).
     */
    cublasAlloc (*n * abs(*incx), sizeof(devPtrx[0]), (void**)&devPtrx);
    cublasSetVector (*n* abs(*incx), sizeof(x[0]), x, 1, devPtrx, 1);
    /*  A      - REAL             array of DIMENSION ( LDA, n ).
     *           Before entry with  UPLO = 'U' or 'u', the leading n by n
     *           upper triangular part of the array A must contain the upper
     */
    cublasAlloc ((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    cublasSetMatrix (imin(*n,*lda), *n, sizeof(A[0]), A, *lda, devPtrA, *lda);
    cublasSsyr (uplo[0], *n, (float)*alpha, devPtrx, *incx, devPtrA, *lda);
    cublasGetMatrix (imin(*n,*lda), *n, sizeof(A[0]), devPtrA, *lda, A, *lda);
    cublasFree (devPtrA);
    cublasFree (devPtrx);
}

void CUBLAS_SSYR2 (const char *uplo, const int *n, double *alpha,
                   const float *x, const int *incx, const float *y,
                   const int *incy, float *A, const int *lda)
{
    float *devPtrA, *devPtrx, *devPtry;

    /*  X      - REAL             array of dimension at least
     *           ( 1 + ( n - 1 )*abs( INCX ) ).
     */ 
    cublasAlloc (*n * abs(*incx), sizeof(devPtrx[0]), (void**)&devPtrx);
    cublasSetVector (*n * abs(*incx), sizeof(x[0]), x, 1, devPtrx, 1);
    /*
     *  Y      - REAL             array of dimension at least
     *           ( 1 + ( n - 1 )*abs( INCY ) ).
     */
    cublasAlloc (*n * abs(*incy), sizeof(devPtry[0]), (void**)&devPtry);
    cublasSetVector (*n* abs(*incy), sizeof(y[0]), y, 1, devPtry, 1);
    /*  A      - REAL             array of DIMENSION ( LDA, n ).
     *           Before entry with  UPLO = 'U' or 'u', the leading n by n
     *           upper triangular part of the array A must contain the upper
     */
    cublasAlloc ((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    cublasSetMatrix (imin(*n,*lda), *n, sizeof(A[0]), A, *lda, devPtrA, *lda);
    cublasSsyr2 (uplo[0], *n, (float)*alpha, devPtrx, *incx, devPtry, *incy, devPtrA,
                 *lda);
    cublasGetMatrix (imin(*n,*lda), *n, sizeof(A[0]), devPtrA, *lda, A, *lda);
    cublasFree (devPtrA);
    cublasFree (devPtrx);
    cublasFree (devPtry);
}

void CUBLAS_STBMV (const char *uplo, const char *trans, const char *diag,
                   const int *n, const int *k, const float *A, const int *lda,
                   float *x, const int *incx)
{
    float *devPtrA, *devPtrx;

    /*  X      - REAL             array of dimension at least
     *           ( 1 + ( n - 1 )*abs( INCX ) ).
     */
    cublasAlloc (*n * abs(*incx), sizeof(devPtrx[0]), (void**)&devPtrx);
    cublasSetVector (*n* abs(*incx), sizeof(x[0]), x, 1, devPtrx, 1);
    /*  A      - REAL             array of DIMENSION ( LDA, n ).
     *           Before entry with UPLO = 'U' or 'u', the leading ( k + 1 )
     *           by n part of the array A must contain the upper triangular
     */
    cublasAlloc ((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    cublasSetMatrix (imin(*k+1,*lda), *n, sizeof(A[0]), A, *lda, devPtrA, *lda);
    cublasStbmv (uplo[0], trans[0], diag[0], *n, *k, devPtrA, *lda, devPtrx, 
                 *incx);
    cublasGetVector (*n* abs(*incx), sizeof(x[0]), devPtrx, 1, x, 1);
    cublasFree (devPtrA);
    cublasFree (devPtrx);
}

void CUBLAS_STBSV (const char *uplo, const char *trans, const char *diag,
                   const int *n, const int *k, const float *A, const int *lda,
                   float *x, const int *incx)
{
    float *devPtrA, *devPtrx;

    /*  X      - REAL             array of dimension at least
     *           ( 1 + ( n - 1 )*abs( INCX ) ).
     */
    cublasAlloc (*n * abs(*incx), sizeof(devPtrx[0]), (void**)&devPtrx);
    cublasSetVector (*n * abs(*incx), sizeof(x[0]), x, 1, devPtrx, 1);
    /*  A      - REAL             array of DIMENSION ( LDA, n ).
     *           Before entry with UPLO = 'U' or 'u', the leading ( k + 1 )
     *           by n part of the array A must contain the upper triangular
     */
    cublasAlloc ((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    cublasSetMatrix (imin(*k+1,*lda), *n, sizeof(A[0]), A, *lda, devPtrA, *lda);
    cublasStbsv (uplo[0], trans[0], diag[0], *n, *k, devPtrA, *lda, devPtrx,
                 *incx);
    cublasGetVector (*n * abs(*incx), sizeof(x[0]), devPtrx, 1, x, 1);
    cublasFree (devPtrA);
    cublasFree (devPtrx);
}

void CUBLAS_STPMV (const char *uplo, const char *trans, const char *diag,
                   const int *n, const float *AP, float *x, const int *incx)
{
    float *devPtrAP, *devPtrx;

    /*  X      - REAL             array of dimension at least
     *           ( 1 + ( n - 1 )*abs( INCX ) ).
     */
    cublasAlloc (*n * abs(*incx), sizeof(devPtrx[0]), (void**)&devPtrx);
    cublasSetVector (*n* abs(*incx), sizeof(x[0]), x, 1, devPtrx, 1);
    /*  AP     - REAL             array of DIMENSION at least
     *           ( ( n*( n + 1 ) )/2 ).
     *           Before entry with  UPLO = 'U' or 'u', the array AP must
     */
    cublasAlloc (((*n) * (*n+1))/2, sizeof(devPtrAP[0]), (void**)&devPtrAP);
    cublasSetVector (((*n) * (*n+1))/2, sizeof(AP[0]), AP, 1, devPtrAP, 1);
    cublasStpmv (uplo[0], trans[0], diag[0], *n, devPtrAP, devPtrx, *incx);
    cublasGetVector (*n* abs(*incx), sizeof(x[0]), devPtrx, 1, x, 1);
    cublasFree (devPtrAP);
    cublasFree (devPtrx);
}

void CUBLAS_STPSV (const char *uplo, const char *trans, const char *diag,
                   const int *n, const float *AP, float *x, const int *incx)
{
    float *devPtrAP, *devPtrx;

     /*  X      - REAL             array of dimension at least
      *           ( 1 + ( n - 1 )*abs( INCX ) ).
      */
    cublasAlloc (*n * abs(*incx), sizeof(devPtrx[0]), (void**)&devPtrx);
    cublasSetVector (*n* abs(*incx), sizeof(x[0]), x, 1, devPtrx, 1);
    /*  AP     - REAL             array of DIMENSION at least
     *           ( ( n*( n + 1 ) )/2 ).
     *           Before entry with  UPLO = 'U' or 'u', the array AP must
     */
    cublasAlloc (((*n) * (*n+1))/2, sizeof(devPtrAP[0]), (void**)&devPtrAP);
    cublasSetVector (((*n) * (*n+1))/2, sizeof(AP[0]), AP, 1, devPtrAP, 1);
    cublasStpsv (uplo[0], trans[0], diag[0], *n, devPtrAP, devPtrx, *incx);
    cublasGetVector (*n* abs(*incx), sizeof(x[0]), devPtrx, 1, x, 1);
    cublasFree (devPtrAP);
    cublasFree (devPtrx);
}

void CUBLAS_STRMV (const char *uplo, const char *trans,
                            const char *diag, const int *n, const float *A,
                            const int *lda, float *x, const int *incx)
{
    float *devPtrA, *devPtrx;

    /*  X      - REAL             array of dimension at least
     *           ( 1 + ( n - 1 )*abs( INCX ) ).
     */
    cublasAlloc (*n * abs(*incx), sizeof(devPtrx[0]), (void**)&devPtrx);
    cublasSetVector (*n* abs(*incx), sizeof(x[0]), x, 1, devPtrx, 1);
    /*  A      - REAL             array of DIMENSION ( LDA, n ).
     *           Before entry with  UPLO = 'U' or 'u', the leading n by n
     *           upper triangular part of the array A must contain the upper
     */
    cublasAlloc ((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    cublasSetMatrix (imin(*n,*lda), *n, sizeof(A[0]), A, *lda, devPtrA, *lda);
    cublasStrmv (uplo[0], trans[0], diag[0], *n, devPtrA, *lda, devPtrx,*incx);
    cublasGetVector (*n* abs(*incx), sizeof(x[0]), devPtrx, 1, x, 1);
    cublasFree (devPtrA);
    cublasFree (devPtrx);
}

void CUBLAS_STRSV (const char *uplo, const char *trans, const char *diag, 
                   const int *n, const float *A, const int *lda, float *x, 
                   const int *incx)
{
    float *devPtrA, *devPtrx;

    /*  X      - REAL             array of dimension at least
     *           ( 1 + ( n - 1 )*abs( INCX ) ).
     */
    cublasAlloc (*n * abs(*incx), sizeof(devPtrx[0]), (void**)&devPtrx);
    cublasSetVector (*n* abs(*incx), sizeof(x[0]), x, 1, devPtrx, 1);
    /*  A      - REAL             array of DIMENSION ( LDA, n ).
     *           Before entry with  UPLO = 'U' or 'u', the leading n by n
     *           upper triangular part of the array A must contain the upper
     */
    cublasAlloc ((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
    cublasSetMatrix (imin(*n,*lda), *n, sizeof(A[0]), A, *lda, devPtrA, *lda);
    cublasStrsv (uplo[0], trans[0], diag[0], *n, devPtrA, *lda, devPtrx,*incx);
    cublasGetVector (*n* abs(*incx), sizeof(x[0]), devPtrx, 1, x, 1);
    cublasFree (devPtrA);
    cublasFree (devPtrx);
}

void CUBLAS_DGEMV (const char *trans, const int *m, const int *n,
                   double *alpha, double *A, const int *lda,
                   double *x, const int *incx, double *beta,
                   double *y, const int *incy)
{
    double *devPtrA, *devPtrx, *devPtry;
	int xelem=0,yelem=0, elementsA;
	double *tempX=NULL, *tempY=NULL;
	float *cpy_A=NULL, *cpy_x=NULL;

	//print_func_name("dgemv");

	if (double_capable){
		/*  X      - REAL             array of DIMENSION at least
		 *           ( 1 + ( n - 1 )*abs( INCX ) ) when TRANS = 'N' or 'n'
		 *           and at least
		 *           ( 1 + ( m - 1 )*abs( INCX ) ) otherwise.
		 */
		if (toupper(trans[0]) == 'N') {
			cublasAlloc (*n * abs(*incx), sizeof(x[0]), (void**)&devPtrx);
			cublasSetVector (*n, sizeof(x[0]), x, abs(*incx), devPtrx, abs(*incx));
		} else {
			cublasAlloc (*m * abs(*incx), sizeof(x[0]), (void**)&devPtrx);
			cublasSetVector (*m, sizeof(x[0]), x, abs(*incx), devPtrx, abs(*incx));
		}
		/*  Y      - REAL             array of DIMENSION at least
		 *           ( 1 + ( m - 1 )*abs( INCY ) ) when TRANS = 'N' or 'n'
		 *           and at least
		 *           ( 1 + ( n - 1 )*abs( INCY ) ) otherwise.
		 */
		if (toupper(trans[0]) == 'N') {
			cublasAlloc (*m * abs(*incy), sizeof(y[0]), (void**)&devPtry);
			cublasSetVector (*m, sizeof(y[0]), y, abs(*incy), devPtry, abs(*incy));
		} else {
			cublasAlloc (*n * abs(*incy), sizeof(y[0]), (void**)&devPtry);
			cublasSetVector (*n, sizeof(y[0]), y, abs(*incy), devPtry, abs(*incy));
		}       
		/*  A      - REAL             array of DIMENSION ( LDA, n ).
		 *           Before entry, the leading m by n part of the array A must
		 *           contain the matrix of coefficients.
		 */
		cublasAlloc ((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
		cublasSetMatrix (imin(*m,*lda), *n, sizeof(A[0]), A, *lda, devPtrA, *lda);
		cublasDgemv (trans[0], *m, *n, *alpha, devPtrA, *lda, devPtrx, *incx, *beta, devPtry, *incy);
		if (toupper(trans[0]) == 'N') {
			cublasGetVector (*m, sizeof(y[0]), devPtry, abs(*incy), y, abs(*incy));
		} else {
			cublasGetVector (*n, sizeof(y[0]), devPtry, abs(*incy), y, abs(*incy));
		}       
		cublasFree (devPtrA);
		cublasFree (devPtry);
		cublasFree (devPtrx);
	}
	else {
		elementsA=(*lda) * (*n);
		if (*trans=='N' || *trans=='n'){
			xelem= imax(1,*n * abs(*incx));
			yelem= imax(1,*m * abs(*incy));
		}
		else{
			xelem= imax(1,*m * abs(*incx));
			yelem= imax(1,*n * abs(*incy));
		}
		cpy_A=copy_n_truncate(A, elementsA);
		cpy_x=copy_n_truncate(x, xelem);
		truncate ((void *)y, xelem);/*truncate*/
		CUBLAS_SGEMV (trans, m, n, alpha, cpy_A, lda, cpy_x, incx, beta, (float *)y, incy);
		untruncate ((void *)y, yelem);
		free((void *)cpy_A);
		free((void *)cpy_x);
	}
	//print_vector((void *)y, DOUBLE, yelem);
}

void CUBLAS_DGER (const int *m, const int *n, double *alpha, 
                  double *x, const int *incx, double *y,
                  const int *incy, double *A, const int *lda)
{
    double *devPtrA, *devPtrx, *devPtry;
	int xelem=0,yelem=0, elementsA;
	double *tempX=NULL, *tempY=NULL;
	float *cpy_x=NULL, *cpy_y=NULL;

	//print_func_name("dger");

	if (double_capable){
		/*print_func_name ("dger");
		print_matrix((void *)A, sizeof(double),imin(*m,*lda), (*lda) * (*n));
		print_vector((void *)m, INT,1);
		print_vector((void *)n, INT,1);
		print_vector((void *)incx, INT,1);
		print_vector((void *)incy, INT, 1);
		print_vector((void *)x, DOUBLE, xelem);
		print_vector((void *)y, DOUBLE, yelem);
		print_vector((void *)alpha, DOUBLE, 1);*/
		cublasAlloc (*m * abs(*incx), sizeof(devPtrx[0]), (void**)&devPtrx);
		cublasAlloc (*n * abs(*incy), sizeof(devPtry[0]), (void**)&devPtry);
		cublasSetVector (*m* abs(*incx), sizeof(x[0]), x, 1, devPtrx, 1);
		cublasSetVector (*n* abs(*incy), sizeof(y[0]), y, 1, devPtry, 1);

		// REAL array of DIMENSION ( LDA, n ).
		//      Before entry, the leading m by n part of the array A must
		//      contain the matrix of coefficients. On exit, A is
		cublasAlloc ((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
		cublasSetMatrix (imin(*m,*lda), *n, sizeof(A[0]), A, *lda, devPtrA, *lda);
		cublasDger (*m, *n, *alpha, devPtrx, *incx, devPtry, *incy, devPtrA, *lda);
		cublasGetMatrix (imin(*m,*lda), *n, sizeof(A[0]), devPtrA, *lda, A, *lda);
		/*print_matrix((void *)A, sizeof(double),imin(*m,*lda), (*lda) * (*n));*/
		cublasFree (devPtrA);
		cublasFree (devPtrx);
		cublasFree (devPtry);
	}
	else{
		xelem = imax(1,*m * abs(*incx));
		yelem = imax(1,*n * abs(*incy));;
		elementsA=(*lda) * (*n);
		cpy_x=copy_n_truncate(x,xelem);
		cpy_y=copy_n_truncate(y,yelem);
		truncate ((void *)A, elementsA);
		CUBLAS_SGER (m, n, alpha, cpy_x, incx, cpy_y, incy, (float *)A, lda);
		untruncate ((void *)A, elementsA);
		free((void *)cpy_x);
		free((void *)cpy_y);
	}
	//print_matrix((void *)A,sizeof(double),*lda,elementsA);
}

void CUBLAS_DSYR (const char *uplo, const int *n, double *alpha, 
                  double *x, const int *incx, double *A, const int *lda)
{
    double *devPtrA, *devPtrx;
	int xelem=0, elementsA=0;
	double *tempX=NULL, *tempY=NULL;
	float * cpy_x=NULL;

	//print_func_name("dsyr");

	if (double_capable){
		/*  X      - REAL             array of dimension at least
		 *           ( 1 + ( n - 1 )*abs( INCX ) ).
		 */
		cublasAlloc (*n * abs(*incx), sizeof(devPtrx[0]), (void**)&devPtrx);
		cublasSetVector (*n* abs(*incx), sizeof(x[0]), x, 1, devPtrx, 1);
		/*  A      - REAL             array of DIMENSION ( LDA, n ).
		 *           Before entry with  UPLO = 'U' or 'u', the leading n by n
		 *           upper triangular part of the array A must contain the upper
		 */
		cublasAlloc ((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
		cublasSetMatrix (imin(*n,*lda), *n, sizeof(A[0]), A, *lda, devPtrA, *lda);
		cublasDsyr (uplo[0], *n, *alpha, devPtrx, *incx, devPtrA, *lda);
		cublasGetMatrix (imin(*n,*lda), *n, sizeof(A[0]), devPtrA, *lda, A, *lda);
		cublasFree (devPtrA);
		cublasFree (devPtrx);
	}
	else {
		xelem = imax(1,*n * abs(*incx));
		elementsA=(*lda) * (*n);
		cpy_x=copy_n_truncate(x,xelem);	
		truncate ((void *)A, elementsA);
		CUBLAS_SSYR (uplo, n, alpha, cpy_x, incx, (float *)A, lda);
		untruncate ((void *)A, elementsA);
		free((void *)cpy_x);
	}
	//print_matrix((void *)A, sizeof(double),*lda,elementsA);
}

void CUBLAS_DTRSV (const char *uplo, const char *trans, const char *diag, 
                   const int *n, double *A, const int *lda, double *x, 
                   const int *incx)
{
    double *devPtrA, *devPtrx;
	int xelem=0, elementsA;
	double *tempX=NULL, *tempY=NULL;
	float *cpy_A=NULL;

	//print_func_name("dtrsv");

	if (double_capable){
		/*  X      - REAL             array of dimension at least
		 *           ( 1 + ( n - 1 )*abs( INCX ) ).
		 */
		cublasAlloc (*n * abs(*incx), sizeof(devPtrx[0]), (void**)&devPtrx);
		cublasSetVector (*n* abs(*incx), sizeof(x[0]), x, 1, devPtrx, 1);
		/*  A      - REAL             array of DIMENSION ( LDA, n ).
		 *           Before entry with  UPLO = 'U' or 'u', the leading n by n
		 *           upper triangular part of the array A must contain the upper
		 */
		cublasAlloc ((*lda) * (*n), sizeof(devPtrA[0]), (void**)&devPtrA);
		cublasSetMatrix (imin(*n,*lda), *n, sizeof(A[0]), A, *lda, devPtrA, *lda);
		cublasDtrsv (uplo[0], trans[0], diag[0], *n, devPtrA, *lda, devPtrx,*incx);
		cublasGetVector (*n* abs(*incx), sizeof(x[0]), devPtrx, 1, x, 1);
		cublasFree (devPtrA);
		cublasFree (devPtrx);
	}
	else {
		elementsA=(*lda) * (*n);
		xelem = imax(1,*n * abs(*incx));
		cpy_A=copy_n_truncate (A, elementsA);
		truncate((void *)x,xelem);
		CUBLAS_STRSV (uplo, trans, diag, n, cpy_A, lda, (float *)x, incx);
		untruncate((void *)x,xelem);
		free((void *)cpy_A);
	}
	//print_vector(x,DOUBLE,xelem);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------- BLAS3 ----------------------------------*/
/*---------------------------------------------------------------------------*/

void CUBLAS_SGEMM (const char *transa, const char *transb, const int *m,
                   const int *n, const int *k, double *alpha,
                   const float *A, const int *lda, const float *B,
                   const int *ldb, double *beta, float *C, const int *ldc)
{
    int ka, kb;
    float *devPtrA, *devPtrB, *devPtrC;

    /*  A      - REAL             array of DIMENSION ( LDA, ka ), where ka is
     *           k  when  TRANSA = 'N' or 'n',  and is  m  otherwise.
     *           Before entry with  TRANSA = 'N' or 'n',  the leading  m by k
     *           part of the array  A  must contain the matrix  A,  otherwise
     *           the leading  k by m  part of the array  A  must contain  the
     *           matrix A.
     */
    ka = (toupper(transa[0]) == 'N') ? *k : *m;
    cublasAlloc (*lda * ka, sizeof(devPtrA[0]), (void**)&devPtrA);
    if (toupper(transa[0]) == 'N') {
        cublasSetMatrix (imin(*m,*lda), *k, sizeof(float), A, *lda, devPtrA, 
                         *lda);
    } else {
        cublasSetMatrix (imin(*k,*lda), *m, sizeof(float), A, *lda, devPtrA, 
                         *lda);
    }

    /*  B      - REAL             array of DIMENSION ( LDB, kb ), where kb is
     *           n  when  TRANSB = 'N' or 'n',  and is  k  otherwise.
     *           Before entry with  TRANSB = 'N' or 'n',  the leading  k by n
     *           part of the array  B  must contain the matrix  B,  otherwise
     *           the leading  n by k  part of the array  B  must contain  the
     *           matrix B.
     */
    kb = (toupper(transb[0]) == 'N') ? *n : *k;
    cublasAlloc (*ldb * kb, sizeof(devPtrB[0]), (void**)&devPtrB);
    if (toupper(transb[0]) == 'N') {
        cublasSetMatrix (imin(*k,*ldb), *n, sizeof(float), B, *ldb, devPtrB, 
                         *ldb);
    } else {
        cublasSetMatrix (imin(*n,*ldb), *k, sizeof(float), B, *ldb, devPtrB,
                         *ldb);
    }
    
    /*  C      - REAL             array of DIMENSION ( LDC, n ).
     *           Before entry, the leading  m by n  part of the array  C must
     *           contain the matrix  C,  except when  beta  is zero, in which
     *           case C need not be set on entry.
     *           On exit, the array  C  is overwritten by the  m by n  matrix
     */
    cublasAlloc ((*ldc) * (*n), sizeof(devPtrC[0]), (void**)&devPtrC);
    cublasSetMatrix (imin(*m,*ldc), *n, sizeof(float), C, *ldc, devPtrC, *ldc);

    cublasSgemm (transa[0], transb[0], *m, *n, *k, ((float)(*alpha)), devPtrA, *lda, 
                 devPtrB, *ldb, ((float) (*beta)), devPtrC, *ldc);

    cublasGetMatrix (imin(*m,*ldc), *n, sizeof(float), devPtrC, *ldc, C, *ldc);
    cublasFree (devPtrA);
    cublasFree (devPtrB);
    cublasFree (devPtrC);
	
}

void CUBLAS_SSYMM (const char *side, const char *uplo, const int *m, 
                   const int *n, double *alpha, const float *A, 
                   const int *lda, const float *B, const int *ldb, 
                   double *beta, float *C, const int *ldc)
{
    int ka;
    float *devPtrA, *devPtrB, *devPtrC;

    /*  A      - REAL             array of DIMENSION ( LDA, ka ), where ka is
     *           m  when  SIDE = 'L' or 'l'  and is  n otherwise.
     *           Before entry  with  SIDE = 'L' or 'l',  the  m by m  part of
     *           the array  A  must contain the  symmetric matrix, [..]
     *           Before entry  with  SIDE = 'R' or 'r',  the  n by n  part of
     *           the array  A  must contain the  symmetric matrix, [..]
     */
    ka = (toupper(side[0]) == 'L') ? *m : *n;
    cublasAlloc ((*lda) * ka, sizeof(devPtrA[0]), (void**)&devPtrA);
    if (toupper(side[0]) == 'L') {
        cublasSetMatrix (imin(*m,*lda), *m, sizeof(A[0]), A, *lda, devPtrA, 
                         *lda);
    } else {
        cublasSetMatrix (imin(*n,*lda), *n, sizeof(A[0]), A, *lda, devPtrA,
                         *lda);
    }

    /*  B      - REAL             array of DIMENSION ( LDB, n ).
     *           Before entry, the leading  m by n part of the array  B  must
     *           contain the matrix B.
     */
    cublasAlloc ((*ldb) * (*n), sizeof(devPtrB[0]), (void**)&devPtrB);
    cublasSetMatrix (imin(*m,*ldb), *n, sizeof(B[0]), B, *ldb, devPtrB, *ldb);

    /*  C      - REAL             array of DIMENSION ( LDC, n ).
     *           Before entry, the leading  m by n  part of the array  C must
     *           contain the matrix  C,  except when  beta  is zero, in which
     *           case C need not be set on entry.
     */
    cublasAlloc ((*ldc) * (*n), sizeof(devPtrC[0]), (void**)&devPtrC);
    cublasSetMatrix (imin(*m,*ldc), *n, sizeof(C[0]), C, *ldc, devPtrC, *ldc);
    
    cublasSsymm (side[0], uplo[0], *m, *n, (float)*alpha, devPtrA, *lda, devPtrB,
                 *ldb, (float) *beta, devPtrC, *ldc);

    cublasGetMatrix (imin(*m,*ldc), *n, sizeof(C[0]), devPtrC, *ldc, C, *ldc);
    cublasFree (devPtrA);
    cublasFree (devPtrB);
    cublasFree (devPtrC);
}

void CUBLAS_SSYR2K (const char *uplo, const char *trans, const int *n,
                    const int *k, double *alpha, const float *A, 
                    const int *lda, const float *B, const int *ldb, 
                    double *beta, float *C, const int *ldc)
{
    int ka, kb;
    float *devPtrA, *devPtrB, *devPtrC;

    /*  A      - REAL             array of DIMENSION ( LDA, ka ), where ka is
     *           k  when  TRANS = 'N' or 'n',  and is  n  otherwise.
     *           Before entry with  TRANS = 'N' or 'n',  the leading  n by k
     *           part of the array  A  must contain the matrix  A,  otherwise
     *           the leading  k by n  part of the array  A  must contain  the
     *           matrix A.
     */
    ka = (toupper(trans[0]) == 'N') ? *k : *n;
    cublasAlloc (*lda * ka, sizeof(devPtrA[0]), (void**)&devPtrA);
    if (toupper(trans[0]) == 'N') {
        cublasSetMatrix (imin(*n,*lda), *k, sizeof(A[0]), A, *lda, devPtrA,
                         *lda);
    } else {
        cublasSetMatrix (imin(*k,*lda), *n, sizeof(A[0]), A, *lda, devPtrA,
                         *lda);
    }

    /*  B      - REAL             array of DIMENSION ( LDB, kb ), where kb is
     *           k  when  TRANS = 'N' or 'n',  and is  n  otherwise.
     *           Before entry with  TRANS = 'N' or 'n',  the leading  n by k
     *           part of the array  B  must contain the matrix  B,  otherwise
     *           the leading  k by n  part of the array  B  must contain  the
     *           matrix B.
     */
    kb = (toupper(trans[0]) == 'N') ? *k : *n;
    cublasAlloc ((*ldb) * kb, sizeof(devPtrB[0]), (void**)&devPtrB);
    if (toupper(trans[0]) == 'N') {
        cublasSetMatrix (imin(*n,*ldb), *k, sizeof(B[0]), B, *ldb, devPtrB,
                         *ldb);
    } else {
        cublasSetMatrix (imin(*k,*ldb), *n, sizeof(B[0]), B, *ldb, devPtrB,
                         *ldb);
    }

    /* C      single precision array of dimensions (ldc, n). If uplo == 'U' or
     *        'u', the leading n x n triangular part of the array C must 
     *        contain the upper triangular part of the symmetric matrix C and 
     *        the strictly lower triangular part of C is not referenced. On 
     *        exit, the upper 
     */
    cublasAlloc ((*ldc) * (*n), sizeof(devPtrC[0]), (void**)&devPtrC);
    cublasSetMatrix (imin(*n,*ldc), *n, sizeof(C[0]), C, *ldc, devPtrC, *ldc);

    cublasSsyr2k (uplo[0], trans[0], *n, *k, (float)*alpha, devPtrA, *lda, devPtrB, 
           *ldb, (float) *beta, devPtrC, *ldc);

    cublasGetMatrix (imin(*n,*ldc), *n, sizeof(C[0]), devPtrC, *ldc, C, *ldc);
    cublasFree (devPtrA);
    cublasFree (devPtrB);
    cublasFree (devPtrC);
}

void CUBLAS_SSYRK (const char *uplo, const char *trans, const int *n, 
                   const int *k, double *alpha, const float *A, 
                   const int *lda, double *beta, float *C, const int *ldc)
{
    int ka;
    float *devPtrA, *devPtrC;

    /* A      single precision array of dimensions (lda, ka), where ka is k 
     *        when trans == 'N' or 'n', and is n otherwise. When trans == 'N' 
     *        or 'n', the leading n x k part of array A must contain the matrix
     *        A, otherwise the leading k x n part of the array must contain the
     *        matrix A.
     */
    ka = (toupper(trans[0]) == 'N') ? *k : *n;
    cublasAlloc (*lda * ka, sizeof(devPtrA[0]), (void**)&devPtrA);
    if (toupper(trans[0]) == 'N') {
        cublasSetMatrix (imin(*n,*lda), *k, sizeof(A[0]), A, *lda, devPtrA,
                         *lda);
    } else {
        cublasSetMatrix (imin(*k,*lda), *n, sizeof(A[0]), A, *lda, devPtrA,
                         *lda);
    }
    
    /* C      single precision array of dimensions (ldc, n). If uplo='U'or'u',
     *        the leading n x n triangular part of the array C must contain the
     *        upper triangular part of the symmetric matrix C and the strictly 
     */
    cublasAlloc ((*ldc) * (*n), sizeof(devPtrC[0]), (void**)&devPtrC);
    cublasSetMatrix (imin(*n,*ldc), *n, sizeof(C[0]), C, *ldc, devPtrC, *ldc);
    
    cublasSsyrk (uplo[0], trans[0], *n, *k, (float)*alpha, devPtrA, *lda, (float) *beta,
                 devPtrC, *ldc);

    cublasGetMatrix (imin(*n,*ldc), *n, sizeof(C[0]), devPtrC, *ldc, C, *ldc);
    cublasFree (devPtrA);
    cublasFree (devPtrC);
}

void CUBLAS_STRMM (const char *side, const char *uplo, const char *transa,
                   const char *diag, const int *m, const int *n,
                   double *alpha, const float *A, const int *lda,
                   float *B, const int *ldb)
{
    int k;
    float *devPtrA, *devPtrB;

    /* A      single precision array of dimensions (lda, k). k = m if side =
     *        'L' or 'l', k = n if side = 'R' or 'r'. If uplo = 'U' or 'u'
     *        the leading k x k upper triangular part of the array A must
     *        contain the upper triangular matrix, and the strictly lower
     *        triangular part of A is not referenced. If uplo = 'L' or 'l'
     *        the leading k x k lower triangular part of the array A must
     *        contain the lower triangular matrix, and the strictly upper
     */
    k = (toupper(side[0]) == 'L') ? *m : *n;
    cublasAlloc (*lda * k, sizeof(devPtrA[0]), (void**)&devPtrA);
    if (toupper(side[0]) == 'L') {
        cublasSetMatrix (imin(k,*lda), k, sizeof(A[0]), A, *lda, devPtrA, *lda);
    } else {
        cublasSetMatrix (imin(k,*lda), k, sizeof(A[0]), A, *lda, devPtrA, *lda);
    }

    /* B      single precision array of dimensions (ldb, n). On entry, the 
     *        leading m x n part of the array contains the matrix B. It is
     *        overwritten with the transformed matrix on exit.
     */
    cublasAlloc ((*ldb) * (*n), sizeof(devPtrB[0]), (void**)&devPtrB);
    cublasSetMatrix (imin(*m,*ldb), *n, sizeof(B[0]), B, *ldb, devPtrB, *ldb);

    cublasStrmm (side[0], uplo[0], transa[0], diag[0], *m, *n, (float)*alpha, devPtrA,
           *lda, devPtrB, *ldb);

    cublasGetMatrix (imin(*m,*ldb), *n, sizeof(B[0]), devPtrB, *ldb, B, *ldb);
    
    cublasFree (devPtrA);
    cublasFree (devPtrB);
}

void CUBLAS_STRSM (const char *side, const char *uplo, const char *transa, 
                   const char *diag, const int *m, const int *n, 
                   double *alpha, const float *A, const int *lda,
                   float *B, const int *ldb)
{
    float *devPtrA, *devPtrB;
    int k;

    //  A      - REAL             array of DIMENSION ( LDA, k ), where k is m
    //           when  SIDE = 'L' or 'l'  and is  n  when  SIDE = 'R' or 'r'.
    //           Before entry  with  UPLO = 'U' or 'u',  the  leading  k by k
    //           upper triangular part of the array  A must contain the upper
    k = (toupper(side[0]) == 'L') ? *m : *n;
    cublasAlloc (*lda * k, sizeof(devPtrA[0]), (void**)&devPtrA);
    cublasSetMatrix (imin(k,*lda), k, sizeof(A[0]), A, *lda, devPtrA, *lda);

    //  B      - REAL             array of DIMENSION ( LDB, n ).
    //           Before entry,  the leading  m by n part of the array  B must
    //           contain  the  right-hand  side  matrix  B,  and  on exit  is
    cublasAlloc ((*ldb) * (*n), sizeof(devPtrB[0]), (void**)&devPtrB);
    cublasSetMatrix (imin(*m,*ldb), *n, sizeof(B[0]), B, *ldb, devPtrB, *ldb);
    cublasStrsm (side[0], uplo[0], transa[0], diag[0], *m, *n, (float)*alpha, devPtrA,
                 *lda, devPtrB, *ldb);
    cublasGetMatrix (imin(*m,*ldb), *n, sizeof(B[0]), devPtrB, *ldb, B, *ldb);
    cublasFree (devPtrA);
    cublasFree (devPtrB);
}

void CUBLAS_CGEMM (const char *transa, const char *transb, const int *m,
                   const int *n, const int *k, const cuComplex *alpha,
                   const cuComplex *A, const int *lda, const cuComplex *B,
                   const int *ldb, const cuComplex *beta, cuComplex *C, 
                   const int *ldc)
{
    int ka, kb;
    cuComplex *devPtrA, *devPtrB, *devPtrC;

	
		

    /*  A      - COMPLEX          array of DIMENSION ( LDA, ka ), where ka is
     *           k  when  TRANSA = 'N' or 'n',  and is  m  otherwise.
     *           Before entry with  TRANSA = 'N' or 'n',  the leading  m by k
     *           part of the array  A  must contain the matrix  A,  otherwise
     *           the leading  k by m  part of the array  A  must contain  the
     *           matrix A.
     */

    ka = (toupper(transa[0]) == 'N') ? *k : *m;
    cublasAlloc (*lda * ka, sizeof(devPtrA[0]), (void**)&devPtrA);
    if (toupper(transa[0]) == 'N') {
        cublasSetMatrix (imin(*m,*lda), *k, sizeof(A[0]), A, *lda, devPtrA, 
                         *lda);
    } else {
        cublasSetMatrix (imin(*k,*lda), *m, sizeof(A[0]), A, *lda, devPtrA, 
                         *lda);
    }

    /*  B      - COMPLEX          array of DIMENSION ( LDB, kb ), where kb is
     *           n  when  TRANSB = 'N' or 'n',  and is  k  otherwise.
     *           Before entry with  TRANSB = 'N' or 'n',  the leading  k by n
     *           part of the array  B  must contain the matrix  B,  otherwise
     *           the leading  n by k  part of the array  B  must contain  the
     *           matrix B.
     */

    kb = (toupper(transb[0]) == 'N') ? *n : *k;
    cublasAlloc (*ldb * kb, sizeof(devPtrB[0]), (void**)&devPtrB);
    if (toupper(transb[0]) == 'N') {
        cublasSetMatrix (imin(*k,*ldb), *n, sizeof(B[0]), B, *ldb, devPtrB, 
                         *ldb);
    } else {
        cublasSetMatrix (imin(*n,*ldb), *k, sizeof(B[0]), B, *ldb, devPtrB,
                         *ldb);
    }

    /*  C      - COMPLEX          array of DIMENSION ( LDC, n ).
     *           Before entry, the leading  m by n  part of the array  C must
     *           contain the matrix  C,  except when  beta  is zero, in which
     *           case C need not be set on entry.
     *           On exit, the array  C  is overwritten by the  m by n  matrix
     */

    cublasAlloc ((*ldc) * (*n), sizeof(devPtrC[0]), (void**)&devPtrC);
    cublasSetMatrix (imin(*m,*ldc), *n, sizeof(C[0]), C, *ldc, devPtrC, *ldc);

    cublasCgemm (transa[0], transb[0], *m, *n, *k, *alpha, devPtrA, *lda, 
                 devPtrB, *ldb, *beta, devPtrC, *ldc);

    cublasGetMatrix (imin(*m,*ldc), *n, sizeof(C[0]), devPtrC, *ldc, C, *ldc);
    cublasFree (devPtrA);
    cublasFree (devPtrB);
    cublasFree (devPtrC);
}

void CUBLAS_DGEMM (const char *transa, const char *transb, const int *m,
                   const int *n, const int *k, double *alpha,
                   double *A, const int *lda, double *B,
                   const int *ldb, double *beta, double *C,
                   const int *ldc)
{
    int ka, kb ;
    double *devPtrA=NULL, *devPtrB=NULL, *devPtrC=NULL;
	unsigned int elementsA=0, elementsB=0, elementsC=0;
	double *tempA=NULL, *tempB=NULL, *tempC=NULL;
	float *cpy_B=NULL, *cpy_A=NULL;

	ka = (toupper(transa[0]) == 'N') ? *k : *m;
	kb = (toupper(transb[0]) == 'N') ? *n : *k;	
	elementsA=(unsigned int)(*lda * ka);
	elementsB=(unsigned int)(*ldb * kb);
	elementsC=(unsigned int)(*ldc * (*n));
	//print_func_name("dgemm");

	if (double_capable){

		/*  A      - REAL             array of DIMENSION ( LDA, ka ), where ka is
		 *           k  when  TRANSA = 'N' or 'n',  and is  m  otherwise.
		 *           Before entry with  TRANSA = 'N' or 'n',  the leading  m by k
		 *           part of the array  A  must contain the matrix  A,  otherwise
		 *           the leading  k by m  part of the array  A  must contain  the
		 *           matrix A.
		 */
		
		cublasAlloc (*lda * ka, sizeof(devPtrA[0]), (void**)&devPtrA);
		if (toupper(transa[0]) == 'N') {
			cublasSetMatrix (imin(*m,*lda), *k, sizeof(A[0]), A, *lda, devPtrA, 
							 *lda);
		} else {
			cublasSetMatrix (imin(*k,*lda), *m, sizeof(A[0]), A, *lda, devPtrA, 
							 *lda);
		}

		/*  B      - REAL             array of DIMENSION ( LDB, kb ), where kb is
		 *           n  when  TRANSB = 'N' or 'n',  and is  k  otherwise.
		 *           Before entry with  TRANSB = 'N' or 'n',  the leading  k by n
		 *           part of the array  B  must contain the matrix  B,  otherwise
		 *           the leading  n by k  part of the array  B  must contain  the
		 *           matrix B.
		 */
		
		cublasAlloc (*ldb * kb, sizeof(devPtrB[0]), (void**)&devPtrB);
		if (toupper(transb[0]) == 'N') {
			cublasSetMatrix (imin(*k,*ldb), *n, sizeof(B[0]), B, *ldb, devPtrB, 
							 *ldb);
		} else {
			cublasSetMatrix (imin(*n,*ldb), *k, sizeof(B[0]), B, *ldb, devPtrB,
							 *ldb);
		}
	    
		/*  C      - REAL             array of DIMENSION ( LDC, n ).
		 *           Before entry, the leading  m by n  part of the array  C must
		 *           contain the matrix  C,  except when  beta  is zero, in which
		 *           case C need not be set on entry.
		 *           On exit, the array  C  is overwritten by the  m by n  matrix
		 */
		cublasAlloc ((*ldc) * (*n), sizeof(devPtrC[0]), (void**)&devPtrC);
		cublasSetMatrix (imin(*m,*ldc), *n, sizeof(C[0]), C, *ldc, devPtrC, *ldc);
		cublasDgemm (transa[0], transb[0], *m, *n, *k, *alpha, devPtrA, *lda, 
					 devPtrB, *ldb, *beta, devPtrC, *ldc);
		cublasGetMatrix (imin(*m,*ldc), *n, sizeof(C[0]), devPtrC, *ldc, C, *ldc);
		cublasFree (devPtrA);
		cublasFree (devPtrB);
		cublasFree (devPtrC);
	}
	else {
		cpy_B=copy_n_truncate (B, elementsB);	
		cpy_A=copy_n_truncate (A, elementsA);
		truncate ((void*)C,elementsC);/*truncate C*/		
		CUBLAS_SGEMM (transa, transb, m, n, k, alpha, cpy_A, lda, cpy_B, ldb, beta, (float *)C, ldc);
		untruncate ((void*)C, elementsC);/*untruncate C*/
		free((void *)cpy_B);
		free((void *)cpy_A);
	}
	//print_matrix(C,sizeof(double),*ldc,elementsC);
}

void CUBLAS_DSYMM (const char *side, const char *uplo, const int *m, 
                   const int *n, double *alpha, double *A, 
                   const int *lda, double *B, const int *ldb, 
                   double *beta, double *C, const int *ldc)
{
    int ka;
    double *devPtrA, *devPtrB, *devPtrC;
	unsigned int elementsA=0, elementsB=0, elementsC=0;
	double *tempA=NULL, *tempB=NULL, *tempC=NULL;
	float *cpy_A=NULL, *cpy_B=NULL;
    
    /*  A      - REAL             array of DIMENSION ( LDA, ka ), where ka is
     *           m  when  SIDE = 'L' or 'l'  and is  n otherwise.
     *           Before entry  with  SIDE = 'L' or 'l',  the  m by m  part of
     *           the array  A  must contain the  symmetric matrix, [..]
     *           Before entry  with  SIDE = 'R' or 'r',  the  n by n  part of
     *           the array  A  must contain the  symmetric matrix, [..]
     */
	ka = (toupper(side[0]) == 'L') ? *m : *n;
	elementsA=(*lda) * ka;
	elementsB=(*ldb) * (*n);
	elementsC=elementsC;
	//print_func_name("dsymm");

	if (double_capable){		
		cublasAlloc (elementsA, sizeof(devPtrA[0]), (void**)&devPtrA);
		if (toupper(side[0]) == 'L') {
			cublasSetMatrix (imin(*m,*lda), *m, sizeof(A[0]), A, *lda, devPtrA, 
							 *lda);
		} else {
			cublasSetMatrix (imin(*n,*lda), *n, sizeof(A[0]), A, *lda, devPtrA,
							 *lda);
		}

		/*  B      - REAL             array of DIMENSION ( LDB, n ).
		 *           Before entry, the leading  m by n part of the array  B  must
		 *           contain the matrix B.
		 */
		cublasAlloc (elementsB, sizeof(devPtrB[0]), (void**)&devPtrB);
		cublasSetMatrix (imin(*m,*ldb), *n, sizeof(B[0]), B, *ldb, devPtrB, *ldb);

		/*  C      - REAL             array of DIMENSION ( LDC, n ).
		 *           Before entry, the leading  m by n  part of the array  C must
		 *           contain the matrix  C,  except when  beta  is zero, in which
		 *           case C need not be set on entry.
		 */
		cublasAlloc (elementsC, sizeof(devPtrC[0]), (void**)&devPtrC);
		cublasSetMatrix (imin(*m,*ldc), *n, sizeof(C[0]), C, *ldc, devPtrC, *ldc);
	    
		cublasDsymm (side[0], uplo[0], *m, *n, *alpha, devPtrA, *lda, devPtrB,
					 *ldb, *beta, devPtrC, *ldc);

		cublasGetMatrix (imin(*m,*ldc), *n, sizeof(C[0]), devPtrC, *ldc, C, *ldc);
		cublasFree (devPtrA);
		cublasFree (devPtrB);
		cublasFree (devPtrC);
	}

	else{
		cpy_B=copy_n_truncate(B, elementsB);/*truncate B*/
		cpy_A=copy_n_truncate (A, elementsA);/*truncate A*/
		truncate ((void *)C, elementsC);/*truncate C*/
		CUBLAS_SSYMM (side, uplo, m, n, alpha, cpy_A, lda, cpy_B, ldb, beta, (float *)C, ldc);
		untruncate ((void *)C, elementsC);/*truncate C*/
		free((void *)cpy_B);
		free((void *)cpy_A);
	}
	//print_matrix(C,sizeof(double),*ldc,elementsC);
}

void CUBLAS_DSYR2K (const char *uplo, const char *trans, const int *n,
                    const int *k, double *alpha, double *A, 
                    const int *lda, double *B, const int *ldb, 
                    double *beta, double *C, const int *ldc)
{
    int ka;
    double *devPtrA, *devPtrB, *devPtrC;
	double *tempA=NULL, *tempB=NULL, *tempC=NULL;
	unsigned int elementsA=0, elementsB=0, elementsC=0;
	float *cpy_A=NULL, *cpy_B=NULL;

	ka = (toupper(trans[0]) == 'N') ? *k : *n;
	elementsA=*lda * ka;
	elementsB=elementsA;/*Matrix A and B are identical in size*/
	elementsC=(*ldc) * (*n);
	//print_func_name("dsyr2k");

    /*  A      - REAL             array of DIMENSION ( LDA, ka ), where ka is
     *           k  when  TRANS = 'N' or 'n',  and is  n  otherwise.
     *           Before entry with  TRANS = 'N' or 'n',  the leading  n by k
     *           part of the array  A  must contain the matrix  A,  otherwise
     *           the leading  k by n  part of the array  A  must contain  the
     *           matrix A.
     */
	if (double_capable){		
		cublasAlloc (elementsA, sizeof(devPtrA[0]), (void**)&devPtrA);
		if (toupper(trans[0]) == 'N') {
			cublasSetMatrix (imin(*n,*lda), *k, sizeof(A[0]), A, *lda, devPtrA,
							 *lda);
		} else {
			cublasSetMatrix (imin(*k,*lda), *n, sizeof(A[0]), A, *lda, devPtrA,
							 *lda);
		}

		/*  B      - REAL             array of DIMENSION ( LDB, kb ), where kb is
		 *           k  when  TRANS = 'N' or 'n',  and is  n  otherwise.
		 *           Before entry with  TRANS = 'N' or 'n',  the leading  n by k
		 *           part of the array  B  must contain the matrix  B,  otherwise
		 *           the leading  k by n  part of the array  B  must contain  the
		 *           matrix B.
		 */
		
		cublasAlloc (elementsB, sizeof(devPtrB[0]), (void**)&devPtrB);
		if (toupper(trans[0]) == 'N') {
			cublasSetMatrix (imin(*n,*ldb), *k, sizeof(B[0]), B, *ldb, devPtrB,
							 *ldb);
		} else {
			cublasSetMatrix (imin(*k,*ldb), *n, sizeof(B[0]), B, *ldb, devPtrB,
							 *ldb);
		}

		/* C      single precision array of dimensions (ldc, n). If uplo == 'U' or
		 *        'u', the leading n x n triangular part of the array C must 
		 *        contain the upper triangular part of the symmetric matrix C and 
		 *        the strictly lower triangular part of C is not referenced. On 
		 *        exit, the upper 
		 */
		cublasAlloc (elementsC, sizeof(devPtrC[0]), (void**)&devPtrC);
		cublasSetMatrix (imin(*n,*ldc), *n, sizeof(C[0]), C, *ldc, devPtrC, *ldc);

		cublasDsyr2k (uplo[0], trans[0], *n, *k, *alpha, devPtrA, *lda, devPtrB, 
			   *ldb, *beta, devPtrC, *ldc);

		cublasGetMatrix (imin(*n,*ldc), *n, sizeof(C[0]), devPtrC, *ldc, C, *ldc);
		cublasFree (devPtrA);
		cublasFree (devPtrB);
		cublasFree (devPtrC);
	}

	else {
		cpy_B=copy_n_truncate(B, elementsB);/*truncate B*/
		cpy_A=copy_n_truncate (A, elementsA);/*truncate A*/
		truncate ((void *)C, elementsC);/*truncate C*/
		CUBLAS_SSYR2K (uplo, trans, n, k,  alpha, cpy_A, lda, cpy_B, ldb, beta, (float *)C, ldc);
		untruncate ((void *)C, elementsC);/*truncate C*/
		free((void *)cpy_B);
		free((void *)cpy_A);
	}
	//print_matrix(C,sizeof(double),*ldc,elementsC);
}

void CUBLAS_DSYRK (const char *uplo, const char *trans, const int *n, 
                   const int *k, double *alpha, double *A, 
                   const int *lda, double *beta, double *C, 
                   const int *ldc)
{
    int ka, elements;
    double *devPtrA, *devPtrC;
	double *tempA=NULL, *tempC=NULL;
	unsigned int elementsA=0, elementsC=0;	
	float *cpy_A=NULL;

    /* A      single precision array of dimensions (lda, ka), where ka is k 
     *        when trans == 'N' or 'n', and is n otherwise. When trans == 'N' 
     *        or 'n', the leading n x k part of array A must contain the matrix
     *        A, otherwise the leading k x n part of the array must contain the
     *        matrix A.
     */
	ka = (toupper(trans[0]) == 'N') ? *k : *n;
	elementsA=*lda * ka;
	elementsC=(*ldc) * (*n);
	//print_func_name("dsyrk");

	if (double_capable){		
		cublasAlloc (elementsA, sizeof(devPtrA[0]), (void**)&devPtrA);
		if (toupper(trans[0]) == 'N') {
			cublasSetMatrix (imin(*n,*lda), *k, sizeof(A[0]), A, *lda, devPtrA,
							 *lda);
		} else {
			cublasSetMatrix (imin(*k,*lda), *n, sizeof(A[0]), A, *lda, devPtrA,
							 *lda);
		}
	    
		/* C      single precision array of dimensions (ldc, n). If uplo='U'or'u',
		 *        the leading n x n triangular part of the array C must contain the
		 *        upper triangular part of the symmetric matrix C and the strictly 
		 */
		cublasAlloc (elementsC, sizeof(devPtrC[0]), (void**)&devPtrC);
		cublasSetMatrix (imin(*n,*ldc), *n, sizeof(C[0]), C, *ldc, devPtrC, *ldc);
	    
		cublasDsyrk (uplo[0], trans[0], *n, *k, *alpha, devPtrA, *lda, *beta,
					 devPtrC, *ldc);

		cublasGetMatrix (imin(*n,*ldc), *n, sizeof(C[0]), devPtrC, *ldc, C, *ldc);
		cublasFree (devPtrA);
		cublasFree (devPtrC);
	}

	else{
		cpy_A=copy_n_truncate (A, elementsA);/*truncate A*/
		truncate ((void *)C, elementsC);/*truncate C*/
		CUBLAS_SSYRK (uplo, trans, n, k, alpha, cpy_A, lda, beta, (float *)C, ldc);
		untruncate ((void *)C, elementsC);/*truncate C*/
		free((void *)cpy_A);
	}
	//print_matrix(C,sizeof(double),*ldc,elementsC);
}

void CUBLAS_DTRMM (const char *side, const char *uplo, const char *transa,
                   const char *diag, const int *m, const int *n,
                   double *alpha, double *A, const int *lda,
                   double *B, const int *ldb)
{
    int k;
    double *devPtrA, *devPtrB;
	double *tempA=NULL, *tempB=NULL;
	unsigned int elementsA=0, elementsB=0;
	float *cpy_A=NULL;

	//print_func_name("dtrmm");

    /* A      single precision array of dimensions (lda, k). k = m if side =
     *        'L' or 'l', k = n if side = 'R' or 'r'. If uplo = 'U' or 'u'
     *        the leading k x k upper triangular part of the array A must
     *        contain the upper triangular matrix, and the strictly lower
     *        triangular part of A is not referenced. If uplo = 'L' or 'l'
     *        the leading k x k lower triangular part of the array A must
     *        contain the lower triangular matrix, and the strictly upper
     */
	k = (toupper(side[0]) == 'L') ? *m : *n;
	elementsA=*lda * k;
	elementsB=(*ldb) * (*n);
	if (double_capable){		
		cublasAlloc (elementsA, sizeof(devPtrA[0]), (void**)&devPtrA);
		if (toupper(side[0]) == 'L') {
			cublasSetMatrix (imin(k,*lda), k, sizeof(A[0]), A, *lda, devPtrA,*lda);
		} else {
			cublasSetMatrix (imin(k,*lda), k, sizeof(A[0]), A, *lda, devPtrA,*lda);
		}

		/* B      single precision array of dimensions (ldb, n). On entry, the 
		 *        leading m x n part of the array contains the matrix B. It is
		 *        overwritten with the transformed matrix on exit.
		 */
		cublasAlloc (elementsB, sizeof(devPtrB[0]), (void**)&devPtrB);
		cublasSetMatrix (imin(*m,*ldb), *n, sizeof(B[0]), B, *ldb, devPtrB, *ldb);

		cublasDtrmm (side[0], uplo[0], transa[0], diag[0], *m, *n, *alpha, devPtrA,
			   *lda, devPtrB, *ldb);

		cublasGetMatrix (imin(*m,*ldb), *n, sizeof(B[0]), devPtrB, *ldb, B, *ldb);
	    
		cublasFree (devPtrA);
		cublasFree (devPtrB);
	}
	else{
		cpy_A=copy_n_truncate (A, elementsA);/*truncate A*/
		truncate ((void *)B, elementsB);/*truncate C*/
		CUBLAS_STRMM (side, uplo, transa, diag, m, n, alpha, cpy_A, lda, (float *)B, ldb);
		untruncate ((void *)B, elementsB);/*truncate C*/
		free((void *)cpy_A);
	}	
	//print_matrix(B,sizeof(double),*ldb,elementsB);
}

void CUBLAS_DTRSM (const char *side, const char *uplo, const char *transa, 
                   const char *diag, const int *m, const int *n, 
                   double *alpha, double *A, const int *lda,
                   double *B, const int *ldb)
{
    double *devPtrA, *devPtrB;
    int k;
	double *tempA=NULL, *tempB=NULL;
	unsigned int elementsA=0, elementsB=0;
	float *cpy_A=NULL;

	
	//print_func_name("dtrsm");

    //  A      - REAL             array of DIMENSION ( LDA, k ), where k is m
    //           when  SIDE = 'L' or 'l'  and is  n  when  SIDE = 'R' or 'r'.
    //           Before entry  with  UPLO = 'U' or 'u',  the  leading  k by k
    //           upper triangular part of the array  A must contain the upper
	k = (toupper(side[0]) == 'L') ? *m : *n;
	elementsA=*lda * k;
	elementsB=(*ldb) * (*n);
	if (double_capable){
		cublasAlloc (elementsA, sizeof(devPtrA[0]), (void**)&devPtrA);
		cublasSetMatrix (imin(k,*lda), k, sizeof(A[0]), A, *lda, devPtrA, *lda);
		//print_vector((void *)alpha,DOUBLE, 1);
		//print_matrix((void *)A, sizeof(double),*lda, elementsA);
		//print_matrix((void *)B, sizeof(double),*ldb, elementsB);
		//  B      - REAL             array of DIMENSION ( LDB, n ).
		//           Before entry,  the leading  m by n part of the array  B must
		//           contain  the  right-hand  side  matrix  B,  and  on exit  is
		cublasAlloc (elementsB, sizeof(devPtrB[0]), (void**)&devPtrB);
		cublasSetMatrix (imin(*m,*ldb), *n, sizeof(B[0]), B, *ldb, devPtrB, *ldb);
		cublasDtrsm (side[0], uplo[0], transa[0], diag[0], *m, *n, *alpha, devPtrA,
					 *lda, devPtrB, *ldb);
		cublasGetMatrix (imin(*m,*ldb), *n, sizeof(B[0]), devPtrB, *ldb, B, *ldb);
		//print_matrix((void *)B, sizeof(double),*ldb, elementsB);
		cublasFree (devPtrA);
		cublasFree (devPtrB);
	}
	else {
		cpy_A=copy_n_truncate (A, elementsA);/*truncate A*/
		truncate ((void *)B, elementsB);/*truncate C*/
		CUBLAS_STRSM (side, uplo, transa, diag, m, n, alpha, cpy_A, lda, (float *)B, ldb);
		untruncate ((void *)B, elementsB);/*truncate C*/
		free((void *)cpy_A);
	}
	//print_matrix(B,sizeof(double),*ldb,elementsB);
}

void CUBLAS_ZGEMM (const char *transa, const char *transb, const int *m,
                   const int *n, const int *k, const cuDoubleComplex *alpha,
                   const cuDoubleComplex *A, const int *lda, 
                   const cuDoubleComplex *B, const int *ldb, 
                   const cuDoubleComplex *beta, cuDoubleComplex *C, 
                   const int *ldc)
{
    int ka, kb;
    cuDoubleComplex *devPtrA, *devPtrB, *devPtrC;

	
		

    /*  A      - COMPLEX          array of DIMENSION ( LDA, ka ), where ka is
     *           k  when  TRANSA = 'N' or 'n',  and is  m  otherwise.
     *           Before entry with  TRANSA = 'N' or 'n',  the leading  m by k
     *           part of the array  A  must contain the matrix  A,  otherwise
     *           the leading  k by m  part of the array  A  must contain  the
     *           matrix A.
     */

    ka = (toupper(transa[0]) == 'N') ? *k : *m;
    cublasAlloc (*lda * ka, sizeof(devPtrA[0]), (void**)&devPtrA);
    if (toupper(transa[0]) == 'N') {
        cublasSetMatrix (imin(*m,*lda), *k, sizeof(A[0]), A, *lda, devPtrA, 
                         *lda);
    } else {
        cublasSetMatrix (imin(*k,*lda), *m, sizeof(A[0]), A, *lda, devPtrA, 
                         *lda);
    }

    /*  B      - COMPLEX          array of DIMENSION ( LDB, kb ), where kb is
     *           n  when  TRANSB = 'N' or 'n',  and is  k  otherwise.
     *           Before entry with  TRANSB = 'N' or 'n',  the leading  k by n
     *           part of the array  B  must contain the matrix  B,  otherwise
     *           the leading  n by k  part of the array  B  must contain  the
     *           matrix B.
     */

    kb = (toupper(transb[0]) == 'N') ? *n : *k;
    cublasAlloc (*ldb * kb, sizeof(devPtrB[0]), (void**)&devPtrB);
    if (toupper(transb[0]) == 'N') {
        cublasSetMatrix (imin(*k,*ldb), *n, sizeof(B[0]), B, *ldb, devPtrB, 
                         *ldb);
    } else {
        cublasSetMatrix (imin(*n,*ldb), *k, sizeof(B[0]), B, *ldb, devPtrB,
                         *ldb);
    }

    /*  C      - COMPLEX          array of DIMENSION ( LDC, n ).
     *           Before entry, the leading  m by n  part of the array  C must
     *           contain the matrix  C,  except when  beta  is zero, in which
     *           case C need not be set on entry.
     *           On exit, the array  C  is overwritten by the  m by n  matrix
     */

    cublasAlloc ((*ldc) * (*n), sizeof(devPtrC[0]), (void**)&devPtrC);
    cublasSetMatrix (imin(*m,*ldc), *n, sizeof(C[0]), C, *ldc, devPtrC, *ldc);

    cublasZgemm (transa[0], transb[0], *m, *n, *k, *alpha, devPtrA, *lda, 
                 devPtrB, *ldb, *beta, devPtrC, *ldc);

    cublasGetMatrix (imin(*m,*ldc), *n, sizeof(C[0]), devPtrC, *ldc, C, *ldc);
    cublasFree (devPtrA);
    cublasFree (devPtrB);
    cublasFree (devPtrC);
}




