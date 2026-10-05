/* Real <-> half-complex FFT plans with 64-bit sizes (3D, and any rank for the interlacing).

   FFTW's ordinary planners (fftw_plan_dft_r2c_3d and kin) take int sizes, so a grid of more than
   2^31 cells (1291^3) cannot be described to them. The 64-bit guru interface (FFTW manual,
   "64-bit Guru Interface") describes the same row-major transform -- real (nx, ny, nz) to
   half-complex (nx, ny, nz/2+1) and back -- with ptrdiff_t sizes and strides. For grids the
   ordinary planners accept it is the same problem, planned the same way. Used by the T-web
   (DTFE.cpp) and the interlacing (interlacing.cc), in double and in float. */

#ifndef DTFE_FFTW_PLANS64_HEADER
#define DTFE_FFTW_PLANS64_HEADER

#include <cstddef>
#include <fftw3.h>

namespace fftwPlans64 {

// dims of the real array (is/os: real strides in, complex strides out) for r2c; c2r swaps them
inline void dims3(size_t nx, size_t ny, size_t nz, bool r2c, fftw_iodim64 d[3])
{
    ptrdiff_t const nzh = ptrdiff_t(nz/2 + 1);
    ptrdiff_t const rs[3] = { ptrdiff_t(ny*nz), ptrdiff_t(nz), 1 };       // real strides
    ptrdiff_t const cs[3] = { ptrdiff_t(ny)*nzh, nzh, 1 };                // half-complex strides
    ptrdiff_t const n[3]  = { ptrdiff_t(nx), ptrdiff_t(ny), ptrdiff_t(nz) };
    for (int i = 0; i < 3; ++i)
    {
        d[i].n  = n[i];
        d[i].is = r2c ? rs[i] : cs[i];
        d[i].os = r2c ? cs[i] : rs[i];
    }
}

inline fftw_plan r2c(size_t nx, size_t ny, size_t nz, double *in, fftw_complex *out, unsigned flags)
{
    fftw_iodim64 d[3]; dims3( nx, ny, nz, true, d );
    return fftw_plan_guru64_dft_r2c( 3, d, 0, nullptr, in, out, flags );
}
inline fftw_plan c2r(size_t nx, size_t ny, size_t nz, fftw_complex *in, double *out, unsigned flags)
{
    fftw_iodim64 d[3]; dims3( nx, ny, nz, false, d );
    return fftw_plan_guru64_dft_c2r( 3, d, 0, nullptr, in, out, flags );
}
inline fftwf_plan r2c(size_t nx, size_t ny, size_t nz, float *in, fftwf_complex *out, unsigned flags)
{
    fftwf_iodim64 d[3]; dims3( nx, ny, nz, true, d );      // (one iodim64 struct for every precision)
    return fftwf_plan_guru64_dft_r2c( 3, d, 0, nullptr, in, out, flags );
}
inline fftwf_plan c2r(size_t nx, size_t ny, size_t nz, fftwf_complex *in, float *out, unsigned flags)
{
    fftwf_iodim64 d[3]; dims3( nx, ny, nz, false, d );
    return fftwf_plan_guru64_dft_c2r( 3, d, 0, nullptr, in, out, flags );
}

// Any rank (the interlacing runs in 2D builds too): n[0..rank-1], the last axis the half-complex one.
inline void dimsN(int rank, size_t const n[], bool r2c, fftw_iodim64 d[])
{
    ptrdiff_t rs = 1, cs = 1;
    for (int i = rank - 1; i >= 0; --i)
    {
        d[i].n  = ptrdiff_t(n[i]);
        d[i].is = r2c ? rs : cs;
        d[i].os = r2c ? cs : rs;
        rs *= ptrdiff_t(n[i]);
        cs *= ( i == rank - 1 ) ? ptrdiff_t(n[i]/2 + 1) : ptrdiff_t(n[i]);
    }
}
inline fftw_plan r2c(int rank, size_t const n[], double *in, fftw_complex *out, unsigned flags)
{
    fftw_iodim64 d[3]; dimsN( rank, n, true, d );
    return fftw_plan_guru64_dft_r2c( rank, d, 0, nullptr, in, out, flags );
}
inline fftw_plan c2r(int rank, size_t const n[], fftw_complex *in, double *out, unsigned flags)
{
    fftw_iodim64 d[3]; dimsN( rank, n, false, d );
    return fftw_plan_guru64_dft_c2r( rank, d, 0, nullptr, in, out, flags );
}
inline fftwf_plan r2c(int rank, size_t const n[], float *in, fftwf_complex *out, unsigned flags)
{
    fftwf_iodim64 d[3]; dimsN( rank, n, true, d );
    return fftwf_plan_guru64_dft_r2c( rank, d, 0, nullptr, in, out, flags );
}
inline fftwf_plan c2r(int rank, size_t const n[], fftwf_complex *in, float *out, unsigned flags)
{
    fftwf_iodim64 d[3]; dimsN( rank, n, false, d );
    return fftwf_plan_guru64_dft_c2r( rank, d, 0, nullptr, in, out, flags );
}

}   // namespace fftwPlans64

#endif
