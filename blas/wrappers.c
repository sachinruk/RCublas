#include <inttypes.h>
#include "CUDA_include/cublas.h"
#include "Blas/wrappers.h"

int CUBLAS_INIT (void) 
{
    return (int)cublasInit ();
}

int CUBLAS_SHUTDOWN (void) 
{
    return (int)cublasShutdown ();
}

int CUBLAS_ALLOC (const int n, const int elemSize, void **devicePtr)
{    
    return (int)cublasAlloc (n, elemSize, devicePtr);
}

int CUBLAS_FREE (const void *devicePtr)
{
    
    return (int)cublasFree (devicePtr);
}

int CUBLAS_SET_VECTOR (const int n, const int elemSize, const void *x,
                       const int incx, void *y, const int incy)
{
    return (int)cublasSetVector (n, elemSize, x, incx, y, incy);
}

int CUBLAS_GET_VECTOR (const int n, const int elemSize, const void *x,
                       const int incx, void *y, const int incy)
{
    return (int)cublasGetVector (n, elemSize, (void *)x, incx, y, incy);
}

int CUBLAS_SET_MATRIX (const int rows, const int cols, const int elemSize,
                       const void *A, const int lda, void *B, 
                       const int ldb)
{
    return (int)cublasSetMatrix (rows, cols, elemSize, A, lda, B,ldb);
}

int CUBLAS_GET_MATRIX (const int rows, const int cols, const int elemSize,
                       const void *A, const int lda, void *B, 
                       const int ldb)
{
    return (int)cublasGetMatrix (rows, cols, elemSize, A, lda, B, ldb);
}

int CUBLAS_GET_ERROR (void)
{
    return (int)cublasGetError();
}

int CUBLAS_ISAMAX (const int n, const void *devPtrx, const int incx)
{
    return cublasIsamax (n, (float *)devPtrx, incx);    
}

int CUBLAS_ISAMIN (const int n, const void *devPtrx, const int incx)
{
    return cublasIsamin (n, (float *)devPtrx, incx);
}

double CUBLAS_SASUM (const int n, const void *devPtrx, const int incx)
{
    return cublasSasum (n, (float *)devPtrx, incx);
}

void CUBLAS_SAXPY (const int n, const float alpha, float *devPtrx, 
                   const int incx, float *devPtry, const int incy)
{
    cublasSaxpy (n, alpha, devPtrx, incx, devPtry, incy);
}

void CUBLAS_SCOPY (const int n, float *devPtrx, const int incx, 
                   float *devPtry, const int incy)
{
    cublasScopy (n, devPtrx, incx, devPtry, incy);
}

double CUBLAS_SDOT (const int n, float *devPtrx, const int incx, 
                    float *devPtry, const int incy)
{
    
    return cublasSdot (n, devPtrx, incx, devPtry, incy);
}

double CUBLAS_SNRM2 (const int n, float *devPtrx, const int incx)
{
    return cublasSnrm2 (n, devPtrx, incx);
}

void CUBLAS_SROT (const int n, const void *devPtrx, const int incx, 
                  const void *devPtry, const int incy, float *sc, 
                  float *ss)
{
    float *x = (float *)(devPtrx);
    float *y = (float *)(devPtry);
    cublasSrot (n, x, incx, y, incy, *sc, *ss);
}

void CUBLAS_SROTG (float *sa, float *sb, float *sc, float *ss)
{
    cublasSrotg (sa, sb, sc, ss);
}

void CUBLAS_SROTM (const int n, const void *devPtrx, const int incx, 
                   const void *devPtry, const int incy, 
                   const float* sparam) 
{
    float *x = (float *)(devPtrx);
    float *y = (float *)(devPtry);
    cublasSrotm (n, x, incx, y, incy, sparam);
}

void CUBLAS_SROTMG (float *sd1, float *sd2, float *sx1, float *sy1,
                    float* sparam)
{
    cublasSrotmg (sd1, sd2, sx1, sy1, sparam);
}

void CUBLAS_SSCAL (const int n, const float alpha, const void *devPtrx,
                   const int incx)
{
    float *x = (float *)(devPtrx);
    cublasSscal (n, alpha, x, incx);
}

void CUBLAS_SSWAP (const int n, const void *devPtrx, const int incx, 
                   const void *devPtry, const int incy)
{
    float *x = (float *)(devPtrx);
    float *y = (float *)(devPtry);
    cublasSswap (n, x, incx, y, incy);
}

int CUBLAS_IDAMAX (const int n, const void *devPtrx, const int incx)
{
    double *x = (double *)(devPtrx);
    int retVal;
    retVal = cublasIdamax (n, x, incx);
    return retVal;
}

int CUBLAS_IDAMIN (const int n, const void *devPtrx, const int incx)
{
    double *x = (double *)(devPtrx);
    int retVal;
    retVal = cublasIdamin (n, x, incx);
    return retVal;
}

double CUBLAS_DASUM (const int n, const void *devPtrx, const int incx)
{
    double *x = (double *)(devPtrx);
    double retVal;
    retVal = cublasDasum (n, x, incx);
    return retVal;
}

void CUBLAS_DAXPY (const int n, const double alpha, const void *devPtrx, 
                   const int incx, const void *devPtry, const int incy)
{
    double *x = (double *)(devPtrx);
    double *y = (double *)(devPtry);
    cublasDaxpy (n, alpha, x, incx, y, incy);
}

void CUBLAS_DCOPY (const int n, const void *devPtrx, const int incx, 
                   const void *devPtry, const int incy)
{
    double *x = (double *)(devPtrx);
    double *y = (double *)(devPtry);
    cublasDcopy (n, x, incx, y, incy);
}

double CUBLAS_DDOT (const int n, const void *devPtrx, const int incx, 
                    const void *devPtry, const int incy)
{
    double *x = (double *)(devPtrx);
    double *y = (double *)(devPtry);
    return cublasDdot (n, x, incx, y, incy);
}

double CUBLAS_DNRM2 (const int n, const void *devPtrx, const int incx)
{
    double *x = (double *)(devPtrx);
    return cublasDnrm2 (n, x, incx);
}

void CUBLAS_DROT (const int n, const void *devPtrx, const int incx, 
                  const void *devPtry, const int incy, double *sc, 
                  double *ss)
{
    double *x = (double *)(devPtrx);
    double *y = (double *)(devPtry);
    cublasDrot (n, x, incx, y, incy, *sc, *ss);
}

void CUBLAS_DROTG (double *sa, double *sb, double *sc, double *ss)
{
    cublasDrotg (sa, sb, sc, ss);
}

void CUBLAS_DROTM (const int n, const void *devPtrx, const int incx, 
                   const void *devPtry, const int incy, 
                   const double* sparam) 
{
    double *x = (double *)(devPtrx);
    double *y = (double *)(devPtry);
    cublasDrotm (n, x, incx, y, incy, sparam);
}

void CUBLAS_DROTMG (double *sd1, double *sd2, double *sx1, double *sy1,
                    double* sparam)
{
    cublasDrotmg (sd1, sd2, sx1, sy1, sparam);
}

void CUBLAS_DSCAL (const int n, const double alpha, const void *devPtrx,
                   const int incx)
{
    double *x = (double *)(devPtrx);
    cublasDscal (n, alpha, x, incx);
}

void CUBLAS_DSWAP (const int n, const void *devPtrx, const int incx, 
                   const void *devPtry, const int incy)
{
    double *x = (double *)(devPtrx);
    double *y = (double *)(devPtry);
    cublasDswap (n, x, incx, y, incy);
}


/*---------------------------------------------------------------------------*/
/*---------------------------------- BLAS2 ----------------------------------*/
/*---------------------------------------------------------------------------*/

void CUBLAS_SGBMV (const char trans, const int m, const int n, 
                   const int kl, const int ku, const float alpha,
                   const void *devPtrA, const int lda, 
                   const void *devPtrx, const int incx, const float beta,
                   const void *devPtry, const int incy)
{
    float *A = (float *)(devPtrA);
    float *x = (float *)(devPtrx);
    float *y = (float *)(devPtry);
    cublasSgbmv (trans, m, n, kl, ku, alpha, A, lda, x, incx, beta,
                 y, incy);
}

void CUBLAS_SGEMV (const char trans, const int m, const int n, 
                   const float alpha, const void *devPtrA, const int lda,
                   const void *devPtrx, const int incx, const float beta,
                   const void *devPtry, const int incy)
{
    float *A = (float *)(devPtrA);
    float *x = (float *)(devPtrx);
    float *y = (float *)(devPtry);
    cublasSgemv (trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

void CUBLAS_SGER (const int m, const int n, const float alpha, 
                  const void *devPtrx, const int incx,
                  const void *devPtry, const int incy,
                  const void *devPtrA, const int lda)
{
    float *A = (float *)(devPtrA);
    float *x = (float *)(devPtrx);
    float *y = (float *)(devPtry);    
    cublasSger (m, n, alpha, x, incx, y, incy, A, lda);
}

void CUBLAS_SSBMV (const char uplo, const int n, const int k,
                   const float alpha, const void *devPtrA, const int lda,
                   const void *devPtrx, const int incx, const float beta,
                   const void *devPtry, const int incy)
{
    float *A = (float *)(devPtrA);
    float *x = (float *)(devPtrx);
    float *y = (float *)(devPtry);    
    cublasSsbmv (uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
}

void CUBLAS_SSPMV (const char uplo, const int n, const float alpha,
                   const void *devPtrAP, const void *devPtrx,
                   const int incx, const float beta, const void *devPtry,
                   const int incy)
{
    float *AP = (float *)(devPtrAP);
    float *x = (float *)(devPtrx);
    float *y = (float *)(devPtry);    
    cublasSspmv (uplo, n, alpha, AP, x, incx, beta, y, incy);
}

void CUBLAS_SSPR (const char uplo, const int n, const float alpha, 
                  const void *devPtrx, const int incx,
                  const void *devPtrAP)
{
    float *AP = (float *)(devPtrAP);
    float *x = (float *)(devPtrx);
    cublasSspr (uplo, n, alpha, x, incx, AP);
}

void CUBLAS_SSPR2 (const char uplo, const int n, const float alpha,
                   const void *devPtrx, const int incx, 
                   const void *devPtry, const int incy,
                   const void *devPtrAP)
{
    float *AP = (float *)(devPtrAP);
    float *x = (float *)(devPtrx);
    float *y = (float *)(devPtry);    
    cublasSspr2 (uplo, n, alpha, x, incx, y, incy, AP);
}

void CUBLAS_SSYMV (const char uplo, const int n, const float alpha,
                   const void *devPtrA, const int lda, 
                   const void *devPtrx, const int incx, const float beta,
                   const void *devPtry,
                   const int incy)
{
    float *A = (float *)(devPtrA);
    float *x = (float *)(devPtrx);
    float *y = (float *)(devPtry);    
    cublasSsymv (uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

void CUBLAS_SSYR (const char uplo, const int n, const float alpha,
                  const void *devPtrx, const int incx,
                  const void *devPtrA, const int lda)
{
    float *A = (float *)(devPtrA);
    float *x = (float *)(devPtrx);    
    cublasSsyr (uplo, n, alpha, x, incx, A, lda);
}

void CUBLAS_SSYR2 (const char uplo, const int n, const float alpha,
                   const void *devPtrx, const int incx,
                   const void *devPtry, const int incy, 
                   const void *devPtrA, const int lda)
{
    float *A = (float *)(devPtrA);
    float *x = (float *)(devPtrx);
    float *y = (float *)(devPtry);    
    cublasSsyr2 (uplo, n, alpha, x, incx, y, incy, A, lda);
}

void CUBLAS_STBMV (const char uplo, const char trans, const char diag,
                   const int n, const int k, const void *devPtrA, 
                   const int lda, const void *devPtrx, const int incx)
{
    float *A = (float *)(devPtrA);
    float *x = (float *)(devPtrx);    
    cublasStbmv (uplo, trans, diag, n, k, A, lda, x, incx);
}

void CUBLAS_STBSV (const char uplo, const char trans, const char diag,
                   const int n, const int k, const void *devPtrA, 
                   const int lda, const void *devPtrx, const int incx)
{
    float *A = (float *)(devPtrA);
    float *x = (float *)(devPtrx);       
    cublasStbsv (uplo, trans, diag, n, k, A, lda, x, incx);
}

void CUBLAS_STPMV (const char uplo, const char trans, const char diag,
                   const int n,  const void *devPtrAP, 
                   const void *devPtrx, const int incx)
{
    float *AP = (float *)(devPtrAP);
    float *x = (float *)(devPtrx);       
    cublasStpmv (uplo, trans, diag, n, AP, x, incx);
}

void CUBLAS_STPSV (const char uplo, const char trans, const char diag,
                   const int n, const void *devPtrAP, 
                   const void *devPtrx, const int incx)
{
    float *AP = (float *)(devPtrAP);
    float *x = (float *)(devPtrx);       
    cublasStpsv (uplo, trans, diag, n, AP, x, incx);
}

void CUBLAS_STRMV (const char uplo, const char trans, const char diag,
                   const int n, const void *devPtrA, const int lda,
                   const void *devPtrx, const int incx)
{
    float *A = (float *)(devPtrA);
    float *x = (float *)(devPtrx);       
    cublasStrmv (uplo, trans, diag, n, A, lda, x, incx);
}

void CUBLAS_STRSV (const char uplo, const char trans, const char diag,
                   const int n, const void *devPtrA, const int lda,
                   const void *devPtrx, const int incx)
{
    float *A = (float *)(devPtrA);
    float *x = (float *)(devPtrx);       
    cublasStrsv (uplo, trans, diag, n, A, lda, x, incx);
}

void CUBLAS_DGEMV (const char trans, const int m, const int n, 
                   const double alpha, const void *devPtrA,
                   const int lda, const void *devPtrx, const int incx,
                   const double beta, const void *devPtry,
                   const int incy)
{
    double *A = (double *)(devPtrA);
    double *x = (double *)(devPtrx);
    double *y = (double *)(devPtry);
    cublasDgemv (trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

void CUBLAS_DGER (const int m, const int n, const double alpha, 
                  const void *devPtrx, const int incx,
                  const void *devPtry, const int incy,
                  const void *devPtrA, const int lda)
{
    double *A = (double *)(devPtrA);
    double *x = (double *)(devPtrx);
    double *y = (double *)(devPtry);    
    cublasDger (m, n, alpha, x, incx, y, incy, A, lda);
}

void CUBLAS_DSYR (const char uplo, const int n, const double alpha,
                  const void *devPtrx, const int incx,
                  const void *devPtrA, const int lda)
{
    double *A = (double *)(devPtrA);
    double *x = (double *)(devPtrx);    
    cublasDsyr (uplo, n, alpha, x, incx, A, lda);
}

void CUBLAS_DTRSV (const char uplo, const char trans, const char diag,
                   const int n, const void *devPtrA, const int lda,
                   const void *devPtrx, const int incx)
{
    double *A = (double *)(devPtrA);
    double *x = (double *)(devPtrx);       
    cublasDtrsv (uplo, trans, diag, n, A, lda, x, incx);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------- BLAS3 ----------------------------------*/
/*---------------------------------------------------------------------------*/

void CUBLAS_SGEMM (const char transa, const char transb, const int m,
                   const int n, const int k, const float alpha,
                   const void *devPtrA, const int lda, 
                   const void *devPtrB, const int ldb, const float beta,
                   const void *devPtrC, const int ldc)
{
    float *A = (float *)(devPtrA);
    float *B = (float *)(devPtrB);
    float *C = (float *)(devPtrC);
    cublasSgemm (transa, transb, m, n, k, alpha, A, lda, 
                 B, ldb, beta, C, ldc);
}

void CUBLAS_SSYMM (const char side, const char uplo, const int m, 
                   const int n, const float alpha, const void *devPtrA,
                   const int lda, const void *devPtrB, const int ldb, 
                   const float beta, const void *devPtrC, const int ldc)
{
    float *A = (float *)(devPtrA);
    float *B = (float *)(devPtrB);
    float *C = (float *)(devPtrC);
    cublasSsymm (side, uplo, m, m, alpha, A, lda, B, ldb, beta, C,
                 ldc);
}

void CUBLAS_SSYR2K (const char uplo, const char trans, const int n,
                    const int k, const float alpha, const void *devPtrA,
                    const int lda, const void *devPtrB, const int ldb, 
                    const float beta, const void *devPtrC, const int ldc)
{
    float *A = (float *)(devPtrA);
    float *B = (float *)(devPtrB);
    float *C = (float *)(devPtrC);
    cublasSsyr2k (uplo, trans, n, k, alpha, A, lda, B, ldb, beta, 
                  C, ldc);
}

void CUBLAS_SSYRK (const char uplo, const char trans, const int n, 
                   const int k, const float alpha, const void *devPtrA, 
                   const int lda, const float beta, const void *devPtrC,
                   const int ldc)
{
    float *A = (float *)(devPtrA);
    float *C = (float *)(devPtrC);
    cublasSsyrk (uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

void CUBLAS_STRMM (const char side, const char uplo, const char transa,
                   const char diag, const int m, const int n,
                   const float alpha, const void *devPtrA, const int lda,
                   const void *devPtrB, const int ldb)
{
    float *A = (float *)(devPtrA);
    float *B = (float *)(devPtrB);
    cublasStrmm (side, uplo, transa, diag, m, n, alpha, A, lda, B,ldb);
}

void CUBLAS_STRSM (const char side, const char uplo, const char transa,
                   const char diag, const int m, const int n, 
                   const float alpha, const void *devPtrA, const int lda,
                   const void *devPtrB, const int ldb)
{
    float *A = (float *)devPtrA;
    float *B = (float *)devPtrB;
    cublasStrsm (side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
}

void CUBLAS_DGEMM (const char transa, const char transb, const int m,
                   const int n, const int k, const double alpha,
                   const void *devPtrA, const int lda, 
                   const void *devPtrB, const int ldb, const double beta,
                   const void *devPtrC, const int ldc)
{
    double *A = (double *)(devPtrA);
    double *B = (double *)(devPtrB);
    double *C = (double *)(devPtrC);
    cublasDgemm (transa, transb, m, n, k, alpha, A, lda, 
                 B, ldb, beta, C, ldc);
}

void CUBLAS_DSYMM (const char side, const char uplo, const int m, 
                   const int n, const double alpha, const void *devPtrA,
                   const int lda, const void *devPtrB, const int ldb, 
                   const double beta, const void *devPtrC, const int ldc)
{
    double *A = (double *)(devPtrA);
    double *B = (double *)(devPtrB);
    double *C = (double *)(devPtrC);
    cublasDsymm (side, uplo, m, m, alpha, A, lda, B, ldb, beta, C,
                 ldc);
}

void CUBLAS_DSYR2K (const char uplo, const char trans, const int n,
                    const int k, const double alpha, const void *devPtrA,
                    const int lda, const void *devPtrB, const int ldb, 
                    const double beta, const void *devPtrC,
                    const int ldc)
{
    double *A = (double *)(devPtrA);
    double *B = (double *)(devPtrB);
    double *C = (double *)(devPtrC);
    cublasDsyr2k (uplo, trans, n, k, alpha, A, lda, B, ldb, beta, 
                  C, ldc);
}

void CUBLAS_DSYRK (const char uplo, const char trans, const int n, 
                   const int k, const double alpha, const void *devPtrA, 
                   const int lda, const double beta, const void *devPtrC,
                   const int ldc)
{
    double *A = (double *)(devPtrA);
    double *C = (double *)(devPtrC);
    cublasDsyrk (uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

void CUBLAS_DTRMM (const char side, const char uplo, const char transa,
                   const char diag, const int m, const int n,
                   const double alpha, const void *devPtrA, 
                   const int lda, const void *devPtrB, const int ldb)
{
    double *A = (double *)(devPtrA);
    double *B = (double *)(devPtrB);
    cublasDtrmm (side, uplo, transa, diag, m, n, alpha, A, lda, B,
                 ldb);
}

void CUBLAS_DTRSM (const char side, const char uplo, const char transa,
                   const char diag, const int m, const int n, 
                   const double alpha, const void *devPtrA,
                   const int lda, const void *devPtrB, const int ldb)
{
    double *A = (double *)devPtrA;
    double *B = (double *)devPtrB;
    cublasDtrsm (side, uplo, transa, diag, m, n, alpha,
                 A, lda, B, ldb);
}


