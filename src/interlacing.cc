/*
 *  Interlacing (FFTW3): F_out(k) = 0.5*(F1(k) + F2(k)*exp(-i*k.dx/2)) then IFFT back; the
 *  phase term on the half-cell-offset grid cancels the leading-order aliasing.
 */

#include "interlacing.h"
#include "message.h"
#include <cmath>
#include <complex>

#include <fftw3.h>

// FFTW precision follows the Real type
#ifdef DOUBLE
    #define FFTW_PLAN       fftw_plan
    #define FFTW_COMPLEX    fftw_complex
    #define FFTW_ALLOC_REAL fftw_alloc_real
    #define FFTW_ALLOC_COMPLEX fftw_alloc_complex
    #define FFTW_PLAN_R2C   fftw_plan_dft_r2c_3d
    #define FFTW_PLAN_C2R   fftw_plan_dft_c2r_3d
    #define FFTW_EXECUTE    fftw_execute
    #define FFTW_DESTROY    fftw_destroy_plan
    #define FFTW_FREE       fftw_free
#else
    #define FFTW_PLAN       fftwf_plan
    #define FFTW_COMPLEX    fftwf_complex
    #define FFTW_ALLOC_REAL fftwf_alloc_real
    #define FFTW_ALLOC_COMPLEX fftwf_alloc_complex
    #define FFTW_PLAN_R2C   fftwf_plan_dft_r2c_3d
    #define FFTW_PLAN_C2R   fftwf_plan_dft_c2r_3d
    #define FFTW_EXECUTE    fftwf_execute
    #define FFTW_DESTROY    fftwf_destroy_plan
    #define FFTW_FREE       fftwf_free
#endif


void applyInterlacing(std::vector<Real> &field1,
                      std::vector<Real> &field2,
                      size_t const *nGrid,
                      Real const *dx)
{
#if NO_DIM != 3
    throwError("Interlacing is currently only implemented for 3D grids.");
#else
    int const Nx = static_cast<int>(nGrid[0]);
    int const Ny = static_cast<int>(nGrid[1]);
    int const Nz = static_cast<int>(nGrid[2]);
    size_t const N = static_cast<size_t>(Nx) * Ny * Nz;
    size_t const Ncomplex = static_cast<size_t>(Nx) * Ny * (Nz/2 + 1);

    Real *in1 = FFTW_ALLOC_REAL(N);
    Real *in2 = FFTW_ALLOC_REAL(N);
    FFTW_COMPLEX *out1 = FFTW_ALLOC_COMPLEX(Ncomplex);
    FFTW_COMPLEX *out2 = FFTW_ALLOC_COMPLEX(Ncomplex);

    for (size_t i = 0; i < N; ++i)
    {
        in1[i] = field1[i];
        in2[i] = field2[i];
    }

    // forward FFT of both grids
    FFTW_PLAN plan_fwd1 = FFTW_PLAN_R2C(Nx, Ny, Nz, in1, out1, FFTW_ESTIMATE);
    FFTW_PLAN plan_fwd2 = FFTW_PLAN_R2C(Nx, Ny, Nz, in2, out2, FFTW_ESTIMATE);
    FFTW_EXECUTE(plan_fwd1);
    FFTW_EXECUTE(plan_fwd2);
    FFTW_DESTROY(plan_fwd1);
    FFTW_DESTROY(plan_fwd2);

    // Average in Fourier space with the half-cell-shift phase correction.
    // Phase = exp(-i*(kx*dx[0]/2 + ...)) with kx = 2*pi*ix/(Nx*dx[0]), so each term
    // reduces to pi*ix/Nx: Phase = exp(-i*pi*(ix/Nx + iy/Ny + iz/Nz)).
    Real const pi = Real(M_PI);

    for (int ix = 0; ix < Nx; ++ix)
    {
        Real kx_phase = pi * Real(ix) / Real(Nx);

        for (int iy = 0; iy < Ny; ++iy)
        {
            Real ky_phase = pi * Real(iy) / Real(Ny);

            for (int iz = 0; iz <= Nz/2; ++iz)
            {
                Real kz_phase = pi * Real(iz) / Real(Nz);

                size_t idx = static_cast<size_t>(ix) * Ny * (Nz/2 + 1) + iy * (Nz/2 + 1) + iz;

                Real phase = -(kx_phase + ky_phase + kz_phase);
                Real cos_p = std::cos(phase);
                Real sin_p = std::sin(phase);

                // F2_corrected = F2 * exp(i*phase)
                Real f2_re = out2[idx][0];
                Real f2_im = out2[idx][1];
                Real f2c_re = f2_re * cos_p - f2_im * sin_p;
                Real f2c_im = f2_re * sin_p + f2_im * cos_p;

                // Average: F_out = 0.5 * (F1 + F2_corrected)
                out1[idx][0] = Real(0.5) * (out1[idx][0] + f2c_re);
                out1[idx][1] = Real(0.5) * (out1[idx][1] + f2c_im);
            }
        }
    }

    // inverse FFT back to real space
    FFTW_PLAN plan_inv = FFTW_PLAN_C2R(Nx, Ny, Nz, out1, in1, FFTW_ESTIMATE);
    FFTW_EXECUTE(plan_inv);
    FFTW_DESTROY(plan_inv);

    // FFTW c2r does not normalize
    Real norm = Real(1.) / Real(N);
    for (size_t i = 0; i < N; ++i)
        field1[i] = in1[i] * norm;

    FFTW_FREE(in1);
    FFTW_FREE(in2);
    FFTW_FREE(out1);
    FFTW_FREE(out2);
#endif
}
