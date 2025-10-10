/*
 *  Accelerate Framework Optimizations for DTFE
 *
 *  This header provides Apple Accelerate framework optimized implementations
 *  of common mathematical operations for improved performance on macOS/iOS.
 *
 *  When USE_ACCELERATE is defined, these functions use vDSP and BLAS routines
 *  optimized for Apple Silicon (M1/M2/M3) and Intel processors.
 */

#ifndef ACCELERATE_MATH_H
#define ACCELERATE_MATH_H

#include "define.h"

#ifdef USE_ACCELERATE
// CBLAS enums (must be declared before function declarations)
enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102};
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};
enum CBLAS_UPLO {CblasUpper=121, CblasLower=122};
enum CBLAS_DIAG {CblasNonUnit=131, CblasUnit=132};
enum CBLAS_SIDE {CblasLeft=141, CblasRight=142};

// Forward declare BLAS/LAPACK functions to avoid header conflicts with CGAL
extern "C" {
    // BLAS Level 3: Matrix-matrix multiplication
    void cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                     const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                     const int K, const float alpha, const float *A,
                     const int lda, const float *B, const int ldb,
                     const float beta, float *C, const int ldc);

    void cblas_dgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                     const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                     const int K, const double alpha, const double *A,
                     const int lda, const double *B, const int ldb,
                     const double beta, double *C, const int ldc);

    // BLAS Level 2: Matrix-vector multiplication
    void cblas_sgemv(const enum CBLAS_ORDER order,
                     const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                     const float alpha, const float *A, const int lda,
                     const float *X, const int incX, const float beta,
                     float *Y, const int incY);

    void cblas_dgemv(const enum CBLAS_ORDER order,
                     const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                     const double alpha, const double *A, const int lda,
                     const double *X, const int incX, const double beta,
                     double *Y, const int incY);

    // BLAS Level 1: Vector operations
    float cblas_snrm2(const int N, const float *X, const int incX);
    double cblas_dnrm2(const int N, const double *X, const int incX);

    // vDSP operations
    void vDSP_dotpr(const float *__A, long __IA,
                   const float *__B, long __IB,
                   float *__C, unsigned long __N);

    void vDSP_dotprD(const double *__A, long __IA,
                    const double *__B, long __IB,
                    double *__C, unsigned long __N);

    void vDSP_vadd(const float *__A, long __IA,
                  const float *__B, long __IB,
                  float *__C, long __IC,
                  unsigned long __N);

    void vDSP_vaddD(const double *__A, long __IA,
                   const double *__B, long __IB,
                   double *__C, long __IC,
                   unsigned long __N);

    void vDSP_vsub(const float *__B, long __IB,
                  const float *__A, long __IA,
                  float *__C, long __IC,
                  unsigned long __N);

    void vDSP_vsubD(const double *__B, long __IB,
                   const double *__A, long __IA,
                   double *__C, long __IC,
                   unsigned long __N);

    void vDSP_vsmul(const float *__A, long __IA,
                   const float *__B,
                   float *__C, long __IC,
                   unsigned long __N);

    void vDSP_vsmulD(const double *__A, long __IA,
                    const double *__B,
                    double *__C, long __IC,
                    unsigned long __N);

    void vDSP_sve(const float *__A, long __IA,
                 float *__C,
                 unsigned long __N);

    void vDSP_sveD(const double *__A, long __IA,
                  double *__C,
                  unsigned long __N);

    // LAPACK routines for matrix operations
    int dgetrf_(int *m, int *n, double *a, int *lda, int *ipiv, int *info);
    int dgetri_(int *n, double *a, int *lda, int *ipiv, double *work, int *lwork, int *info);
}

// Accelerate-optimized matrix inverse for 2x2 and 3x3 matrices
inline void matrixInverse_accelerate(double matrix[][NO_DIM], Real result[][NO_DIM])
{
#if NO_DIM == 2
    // For 2x2 matrices, use direct formula (Accelerate overhead not worth it for 2x2)
    double det = matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0];
    double temp = fabs(matrix[0][0]) + fabs(matrix[0][1]) + fabs(matrix[1][0]) + fabs(matrix[1][1]);

    if (std::fabs(det/temp) > 1.e-6) {
        result[0][0] = Real(matrix[1][1] / det);
        result[0][1] = Real(-matrix[0][1] / det);
        result[1][0] = Real(-matrix[1][0] / det);
        result[1][1] = Real(matrix[0][0] / det);
    } else {
        result[0][0] = result[0][1] = result[1][0] = result[1][1] = Real(0.);
    }

#elif NO_DIM == 3
    // For 3x3 matrices, use LAPACK's optimized inversion
    int n = 3;
    int lda = 3;
    int ipiv[3];
    int info;

    // Copy matrix to workspace (LAPACK works in-place)
    double work_matrix[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            work_matrix[i * 3 + j] = matrix[i][j];
        }
    }

    // LU factorization
    dgetrf_(&n, &n, work_matrix, &lda, ipiv, &info);

    if (info == 0) {
        // Compute inverse from LU factorization
        int lwork = 9;
        double work[9];
        dgetri_(&n, work_matrix, &lda, ipiv, work, &lwork, &info);

        if (info == 0) {
            // Copy result back (transposed due to column-major storage)
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    result[i][j] = Real(work_matrix[j * 3 + i]);
                }
            }
        } else {
            // Inversion failed - set to zero
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    result[i][j] = Real(0.);
                }
            }
        }
    } else {
        // LU factorization failed - matrix is singular
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                result[i][j] = Real(0.);
            }
        }
    }
#endif
}

// Accelerate-optimized matrix multiplication M1 * M2 -> result
template <size_t N>
inline void matrixMultiplication_accelerate(Real M1[][NO_DIM], Real M2[][N], Real result[][N])
{
    if (sizeof(Real) == sizeof(float)) {
        // Single precision
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    NO_DIM, N, NO_DIM,
                    1.0f,
                    (float*)M1, NO_DIM,
                    (float*)M2, N,
                    0.0f,
                    (float*)result, N);
    } else {
        // Double precision
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    NO_DIM, N, NO_DIM,
                    1.0,
                    (double*)M1, NO_DIM,
                    (double*)M2, N,
                    0.0,
                    (double*)result, N);
    }
}

// Accelerate-optimized matrix-vector multiplication M1 * M2 -> result
inline void matrixMultiplication_accelerate(Real M1[][NO_DIM], Real *M2, Real *result)
{
    if (sizeof(Real) == sizeof(float)) {
        // Single precision
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    NO_DIM, NO_DIM,
                    1.0f,
                    (float*)M1, NO_DIM,
                    (float*)M2, 1,
                    0.0f,
                    (float*)result, 1);
    } else {
        // Double precision
        cblas_dgemv(CblasRowMajor, CblasNoTrans,
                    NO_DIM, NO_DIM,
                    1.0,
                    (double*)M1, NO_DIM,
                    (double*)M2, 1,
                    0.0,
                    (double*)result, 1);
    }
}

// Accelerate-optimized vector operations
inline Real vectorDotProduct_accelerate(const Real *v1, const Real *v2, size_t n)
{
    if (sizeof(Real) == sizeof(float)) {
        float result;
        vDSP_dotpr((float*)v1, 1, (float*)v2, 1, &result, n);
        return Real(result);
    } else {
        double result;
        vDSP_dotprD((double*)v1, 1, (double*)v2, 1, &result, n);
        return Real(result);
    }
}

inline Real vectorNorm_accelerate(const Real *v, size_t n)
{
    if (sizeof(Real) == sizeof(float)) {
        return Real(cblas_snrm2(n, (float*)v, 1));
    } else {
        return Real(cblas_dnrm2(n, (double*)v, 1));
    }
}

// Accelerate-optimized vector addition: result = v1 + v2
inline void vectorAdd_accelerate(const Real *v1, const Real *v2, Real *result, size_t n)
{
    if (sizeof(Real) == sizeof(float)) {
        vDSP_vadd((float*)v1, 1, (float*)v2, 1, (float*)result, 1, n);
    } else {
        vDSP_vaddD((double*)v1, 1, (double*)v2, 1, (double*)result, 1, n);
    }
}

// Accelerate-optimized vector subtraction: result = v1 - v2
inline void vectorSub_accelerate(const Real *v1, const Real *v2, Real *result, size_t n)
{
    if (sizeof(Real) == sizeof(float)) {
        vDSP_vsub((float*)v2, 1, (float*)v1, 1, (float*)result, 1, n);
    } else {
        vDSP_vsubD((double*)v2, 1, (double*)v1, 1, (double*)result, 1, n);
    }
}

// Accelerate-optimized scalar multiplication: result = scalar * v
inline void vectorScale_accelerate(Real scalar, const Real *v, Real *result, size_t n)
{
    if (sizeof(Real) == sizeof(float)) {
        float s = (float)scalar;
        vDSP_vsmul((float*)v, 1, &s, (float*)result, 1, n);
    } else {
        double s = (double)scalar;
        vDSP_vsmulD((double*)v, 1, &s, (double*)result, 1, n);
    }
}

// Accelerate-optimized array sum
inline Real vectorSum_accelerate(const Real *v, size_t n)
{
    if (sizeof(Real) == sizeof(float)) {
        float sum;
        vDSP_sve((float*)v, 1, &sum, n);
        return Real(sum);
    } else {
        double sum;
        vDSP_sveD((double*)v, 1, &sum, n);
        return Real(sum);
    }
}

#endif // USE_ACCELERATE

#endif // ACCELERATE_MATH_H
