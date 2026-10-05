/* Unit check of applyInterlacing (src/interlacing.cc), compiled by tests/interlacing_check.sh.
 *
 * Interlacing averages the grid with a second one computed half a cell further along every axis
 * (main.cpp shifts the region by +dx/2), after undoing that shift in Fourier space. For a band-limited
 * field the two grids hold EXACT samples, f(x_j) and f(x_j + dx/2), and the result must be f(x_j)
 * itself. The field mixes Fourier modes whose frequency is negative along x only, along y only, along
 * both, and along neither: FFTW stores a full axis's negative frequencies at index > N/2, so a phase
 * computed from the raw index is off by pi for them -- wrong for the first two kinds, right by
 * accident for the third (two sign flips). Exit status 0 = recovered to float rounding. */
#include <cmath>
#include <cstdio>
#include <vector>
#include "interlacing.h"

int main()
{
#if NO_DIM == 3
    size_t const n[3] = { 24, 20, 16 };
    double const L[3] = { 100., 80., 64. };
    int const modes[][3] = { {2, -3, 1}, {-3, 0, 2}, {-2, -1, 3}, {1, 2, 0}, {0, 0, 5} };
#else
    size_t const n[2] = { 24, 20 };
    double const L[2] = { 100., 80. };
    int const modes[][2] = { {2, -3}, {-3, 1}, {-2, -1}, {1, 2}, {0, 7} };
#endif
    int const nModes = int( sizeof(modes) / sizeof(modes[0]) );
    Real dx[NO_DIM];
    size_t N = 1;
    for (int d = 0; d < NO_DIM; ++d) { dx[d] = Real( L[d] / double(n[d]) ); N *= n[d]; }

    auto field = [&](double const x[NO_DIM]) {
        double f = 0.;
        for (int m = 0; m < nModes; ++m)
        {
            double arg = 0.3 + 0.7 * m;                  // a phase per mode
            for (int d = 0; d < NO_DIM; ++d) arg += 2. * M_PI * modes[m][d] * x[d] / L[d];
            f += (1. + 0.25 * m) * std::cos( arg );
        }
        return f;
    };
    std::vector<Real> f1( N ), f2( N );
    std::vector<double> want( N );
    double fmax = 0.;
    for (size_t j = 0; j < N; ++j)
    {
        size_t rem = j;
        double x[NO_DIM], xs[NO_DIM];
        for (int d = NO_DIM - 1; d >= 0; --d)            // the last axis runs fastest, as the grids do
        {
            size_t const i = rem % n[d]; rem /= n[d];
            x[d]  = (double(i) + 0.5) * double(dx[d]);
            xs[d] = x[d] + 0.5 * double(dx[d]);
        }
        want[j] = field( x );
        f1[j] = Real( want[j] );
        f2[j] = Real( field( xs ) );
        fmax = std::max( fmax, std::fabs( want[j] ) );
    }
    applyInterlacing( f1, f2, n, dx );
    double err = 0.;
    for (size_t j = 0; j < N; ++j) err = std::max( err, std::fabs( double(f1[j]) - want[j] ) );
    double const rel = err / fmax, tol = sizeof(Real) == 4 ? 2e-5 : 1e-10;
    std::printf( "   %s  %dD interlacing recovers a band-limited field (max relative error %.2e, tolerance %.0e)\n",
                 rel <= tol ? "PASS" : "FAIL", NO_DIM, rel, tol );
    return rel <= tol ? 0 : 1;
}
