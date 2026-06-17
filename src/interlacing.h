/*
 *  Interlacing: Fourier-space averaging of two half-cell-offset density grids to
 *  cancel leading-order aliasing in power spectrum measurements. Requires FFTW3
 *  (single-precision fftwf unless DOUBLE is defined).
 */

#ifndef INTERLACING_HEADER
#define INTERLACING_HEADER

#include <vector>
#include "define.h"
#include "user_options.h"

/* Average two density grids in Fourier space (interlacing). field1: original grid, modified
   in-place with the result; field2: half-cell-offset grid; nGrid, dx: NO_DIM elements each. */
void applyInterlacing(std::vector<Real> &field1,
                      std::vector<Real> &field2,
                      size_t const *nGrid,
                      Real const *dx);

#endif
