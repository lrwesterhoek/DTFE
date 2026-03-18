/*
 *  Interlacing: Fourier-space averaging of two half-cell-offset density grids
 *  to cancel leading-order aliasing artifacts in power spectrum measurements.
 *
 *  Requires FFTW3. Uses single-precision (fftwf) or double-precision (fftw)
 *  depending on whether DOUBLE is defined.
 */

#ifndef INTERLACING_HEADER
#define INTERLACING_HEADER

#include <vector>
#include "define.h"
#include "user_options.h"

/* Apply interlacing by averaging two density grids in Fourier space.
   field1: density on original grid (modified in-place with result)
   field2: density on half-cell-offset grid
   nGrid:  grid dimensions (NO_DIM elements)
   dx:     cell spacing (NO_DIM elements)
*/
void applyInterlacing(std::vector<Real> &field1,
                      std::vector<Real> &field2,
                      size_t const *nGrid,
                      Real const *dx);

#endif
