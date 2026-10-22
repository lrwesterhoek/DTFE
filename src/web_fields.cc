/*
 *  Copyright (c) 2011       Marius Cautun
 *
 *                           Kapteyn Astronomical Institute
 *                           University of Groningen, the Netherlands
 *
 *  Copyright (c) 2026       Luuk Westerhoek
 *
 *                           Kapteyn Astronomical Institute
 *                           University of Groningen, the Netherlands
 *
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

/* Derived velocity fields and the cosmic-web classification, moved out of DTFE.cpp: the velocity divergence,
   shear and vorticity from the velocity gradient, the V-web (velocity shear tensor) and the T-web (tidal tensor
   of the density grid, FFT Poisson solve), and postProcessWebFields, which runs them after the interpolation. */

#include <vector>
#include <cmath>
#include <chrono>
#include <utility>
#include <fftw3.h>   // T-web tidal tensor (FFT Poisson solve of the density grid)
#include "fftw_plans64.h"   // ... planned with 64-bit sizes
#include <new>              // std::align_val_t (the aligned FFT buffers)
#include <complex>          // ... into std::complex buffers (FFTW guarantees the layout of fftw_complex)

#include "define.h"
#include "user_options.h"
#include "quantities.h"
#include "message.h"
#include "web_fields.h"


using namespace std;


// Post-processing shared by both DTFE() overloads: derive velocity divergence/shear/vorticity
// and the V-web / T-web labels for the unaveraged and averaged quantities.
void postProcessWebFields(User_options &userOptions,
                          Quantities *uQuantities,
                          Quantities *aQuantities)
{
    computeDivergenceShearVorticity( userOptions.uField, userOptions.verboseLevel, uQuantities );
    computeDivergenceShearVorticity( userOptions.aField, userOptions.verboseLevel, aQuantities );
    computeWebClassification( userOptions.uField, userOptions.verboseLevel, userOptions.lambda_th, userOptions.hubbleParam, userOptions.scaleFactor, uQuantities );
    computeWebClassification( userOptions.aField, userOptions.verboseLevel, userOptions.lambda_th, userOptions.hubbleParam, userOptions.scaleFactor, aQuantities );
    computeTidalWebClassification( userOptions.uField, userOptions, uQuantities );
    computeTidalWebClassification( userOptions.aField, userOptions, aQuantities );
}



// Computes the velocity divergence from the velocity gradient.
Real velocityDivergence(Pvector<Real,noGradComp> &velGrad)
{
    Real div = 0.;
    for (int d=0; d<NO_DIM; ++d)
        div += velGrad[d*NO_DIM+d];
    return div;
}

// Computes the (traceless symmetric) velocity shear from the velocity gradient.
Pvector<Real,noShearComp> velocityShear(Pvector<Real,noGradComp> &velGrad)
{
    Pvector<Real,noShearComp> temp;
    size_t index = 0;
    for (int i=0; i<NO_DIM-1; ++i)
        for (int j=i; j<NO_DIM; ++j)
            temp[index++] = (velGrad[i*NO_DIM+j] + velGrad[j*NO_DIM+i]) / 2.;

    // remove the trace (the divergence) so only the shear remains; the shear tensor is traceless by definition
    Real trace = velGrad[(NO_DIM-1)*NO_DIM+(NO_DIM-1)];
    for (int d=0; d<NO_DIM-1; ++d)
        trace += temp[d*NO_DIM - d*(d-1)/2];
    trace /= NO_DIM;
    for (int d=0; d<NO_DIM-1; ++d)
        temp[d*NO_DIM - d*(d-1)/2] -= trace;
    return temp;
}

// Computes the velocity vorticity from the velocity gradient.
Pvector<Real,noVortComp> velocityVorticity(Pvector<Real,noGradComp> &velGrad)
{
    Pvector<Real,noVortComp> temp;
    size_t index = 0;
    for (int i=0; i<NO_DIM; ++i)
        for (int j=i+1; j<NO_DIM; ++j)
            temp[index++] = (velGrad[i*NO_DIM+j] - velGrad[j*NO_DIM+i]) / 2.;
    return temp;
}

// Eigenvalues of a 3x3 symmetric matrix via Cardano's formula, sorted descending (lambda1 >= lambda2 >= lambda3).
// Computed in DOUBLE whatever Real is: where two eigenvalues (nearly) coincide -- every cell of a field
// that varies along one axis, and the axis-symmetric parts of walls and filaments -- r = det(B)/2 sits
// at +-1, where acos has an infinite slope, so its rounding error eps turns into sqrt(eps) in the
// eigenvalues: in float, 6e-4 for a cell with delta = 3 (the T-web of a 1D pancake reads {3.096, 6e-4,
// -6e-4} for {3.096, 0, 0}; tests/tweb_check.sh). In double, ~1e-8.
Pvector<Real,NO_DIM> symmetricEigenvalues3x3(Real a00r, Real a01r, Real a02r,
                                              Real a11r, Real a12r, Real a22r)
{
    Pvector<Real,NO_DIM> eigenvalues;
#if NO_DIM == 3
    double const a00 = a00r, a01 = a01r, a02 = a02r, a11 = a11r, a12 = a12r, a22 = a22r;
    double ev[3];
    double const p1 = a01*a01 + a02*a02 + a12*a12;
    double const q = (a00 + a11 + a22) / 3.;  // trace / 3

    double const b00 = a00 - q, b11 = a11 - q, b22 = a22 - q;
    double const p2 = b00*b00 + b11*b11 + b22*b22 + 2.*p1;
    double const p = std::sqrt(p2 / 6.);

    if ( p < 1.e-15 )
    {
        // matrix proportional to identity, already diagonal
        ev[0] = a00; ev[1] = a11; ev[2] = a22;
    }
    else
    {
        double const inv_p = 1. / p;
        // B = (1/p) * (A - q*I), compute det(B)
        double const B00 = b00*inv_p, B01 = a01*inv_p, B02 = a02*inv_p;
        double const B11 = b11*inv_p, B12 = a12*inv_p, B22 = b22*inv_p;

        double const detB = B00*(B11*B22 - B12*B12) - B01*(B01*B22 - B12*B02) + B02*(B01*B12 - B11*B02);
        double r = detB / 2.;

        // clamp to [-1, 1] for numerical safety before acos
        if (r <= -1.) r = -1.;
        else if (r >= 1.) r = 1.;

        double const phi = std::acos(r) / 3.;

        ev[0] = q + 2. * p * std::cos(phi);
        ev[2] = q + 2. * p * std::cos(phi + 2. * M_PI / 3.);
        ev[1] = 3. * q - ev[0] - ev[2]; // trace identity
    }

    // sort descending
    if (ev[0] < ev[1]) std::swap(ev[0], ev[1]);
    if (ev[1] < ev[2]) std::swap(ev[1], ev[2]);
    if (ev[0] < ev[1]) std::swap(ev[0], ev[1]);
    for (int i = 0; i < 3; ++i) eigenvalues[i] = Real(ev[i]);
#elif NO_DIM == 2
    // 2x2 symmetric matrix: (a00 + a11)/2 -+ |((a00 - a11)/2, a01)|. The half-difference form, not
    // trace^2/4 - det: that one cancels where the eigenvalues (nearly) coincide and leaves sqrt(eps).
    double const a00 = a00r, a01 = a01r, a11 = a11r;
    (void)a02r; (void)a12r; (void)a22r;
    double const mid = (a00 + a11) / 2.;
    double const disc = std::hypot( (a00 - a11) / 2., a01 );
    eigenvalues[0] = Real(mid + disc);
    eigenvalues[1] = Real(mid - disc);
#endif
    return eigenvalues;
}


// Classifies a cell by counting eigenvalues above lambda_th: 0=void, 1=wall, 2=filament, 3=node
// (2D: 0=void, 1=filament, 2=node -- one collapsed axis is a line there).
int classifyWeb(Pvector<Real,NO_DIM> const &eigenvalues, Real lambda_th)
{
    int label = 0;
    for (int d=0; d<NO_DIM; ++d)
        if (eigenvalues[d] > lambda_th) ++label;
    return label;
}


// V-web classification from the velocity shear tensor Sigma_ij = -(1/(2 H0))(dv_i/dx_j + dv_j/dx_i)
// (Hoffman et al. 2012). The physical T-web (tidal tensor of the potential, Poisson-solved from
// the density grid) is computed by computeTidalWebClassification below.
void computeWebClassification(Field &fields,
                               int const verboseLevel,
                               Real lambda_th,
                               Real hubbleParam,
                               Real scaleFactor,
                               Quantities *q)
{
    if ( q->velocity_gradient.empty() ) return;
    if ( not fields.velocity_vweb ) return;

    // normalize the gradient by H0 = 100 h km/s/Mpc so the threshold lambda_th is dimensionless
    Real H0_norm = Real(1.);
    if ( hubbleParam > Real(0.) )
        H0_norm = Real(100.) * hubbleParam;  // H0 in km/s/Mpc
    // The gradient is in Gadget u-units (u = v_pec/sqrt(a)): multiply by sqrt(a) to reach
    // peculiar km/s, or the eigenvalues come out 1/sqrt(a) inflated at high z (4.6x at z=20)
    // and lambda_th is effectively miscalibrated there. sqrt(1) == 1 exactly, so z=0 results
    // are bit-identical to an uncorrected gradient. -1 = unknown (no header value): warn, treat as 1.
    Real sqrtA = Real(1.);
    if ( scaleFactor > Real(0.) )
        sqrtA = std::sqrt( scaleFactor );
    else
    {
        MESSAGE::Warning warning( verboseLevel );
        warning << "V-web: no scale factor available (header lacked 'Time' and no '--scale-factor' given); "
                << "assuming a = 1. At high redshift the eigenvalues are then 1/sqrt(a) inflated.\n"
                << MESSAGE::EndWarning;
    }

    size_t const N = q->velocity_gradient.size();

    // V-web: shear tensor Sigma_ij
    if ( fields.velocity_vweb )
    {
        q->velocity_vweb.reserve( N );
        q->velocity_vweb_eigenvalues.reserve( N );
        for (size_t i=0; i<N; ++i)
        {
            Pvector<Real,noGradComp> &g = q->velocity_gradient[i];
            Real norm = -sqrtA / (Real(2.) * H0_norm);   // sqrtA: u-units -> peculiar km/s
            Real s00 = norm * (g[0*NO_DIM+0] + g[0*NO_DIM+0]);
            Real s11 = norm * (g[1*NO_DIM+1] + g[1*NO_DIM+1]);
            Real s01 = norm * (g[0*NO_DIM+1] + g[1*NO_DIM+0]);
#if NO_DIM == 3
            Real s22 = norm * (g[2*NO_DIM+2] + g[2*NO_DIM+2]);
            Real s02 = norm * (g[0*NO_DIM+2] + g[2*NO_DIM+0]);
            Real s12 = norm * (g[1*NO_DIM+2] + g[2*NO_DIM+1]);
            Pvector<Real,NO_DIM> eig = symmetricEigenvalues3x3(s00, s01, s02, s11, s12, s22);
#elif NO_DIM == 2
            Pvector<Real,NO_DIM> eig = symmetricEigenvalues3x3(s00, s01, Real(0.), s11, Real(0.), Real(0.));
#endif
            q->velocity_vweb_eigenvalues.push_back( eig );
            q->velocity_vweb.push_back( Real(classifyWeb(eig, lambda_th)) );
        }
    }

    // free the gradient once the web classification no longer needs it; swap-with-empty because
    // clear() keeps the capacity resident (36 B/cell -- ~4.8 GB at 512^3)
    if ( not fields.velocity_gradient )
        std::vector< Pvector<Real,noGradComp> >().swap( q->velocity_gradient );
}


// T-web classification from the TIDAL tensor of the gravitational potential (Hahn et al. 2007;
// Forero-Romero et al. 2009), Poisson-solved from the density grid by FFT. Dimensionless
// convention: T_ij = F^-1[ (k_i k_j / k^2) delta_hat ] so that tr T = delta exactly and
// lambda_th is dimensionless (literature thresholds ~0.2-0.4 apply directly; the same
// convention was validated against an independent Python implementation, trace identity ~1e-6).
// Computed from the RAW density grid -- NO smoothing here by design (the binary outputs raw
// fields only; smoothing is a plot-time choice in python/plot_PS_DTFE.py). Note the labels are
// therefore sensitive to FFT ringing from point-like halo density spikes; labels are categorical,
// so a smoothed CLASSIFICATION (not just a smoothed picture) requires re-deriving the tensor from
// a smoothed density, i.e. a rerun. Periodic, full-box grids only, 2D or 3D (the kernel k_i k_j / k^2
// is the same in any dimension; 2D has three tensor components and two eigenvalues); assumes square
// grid cells (k ratios use per-cell wavenumbers). Runs on the merged full grid after normalization,
// so it is independent of the partition path.
// The FFT buffers of the T-web: as aligned as fftw_alloc makes them (64 B covers every SIMD width FFTW
// uses), through the aligned operator new -- which '--scratch-dir' replaces too (scratch_alloc.cc), so
// above its threshold they live in the scratch files like the grids. fftw_alloc would bypass it.
template <class T>
struct FftwAligned
{
    using value_type = T;
    FftwAligned() = default;
    template <class U> FftwAligned(FftwAligned<U> const &) {}
    T *allocate(size_t n) { return static_cast<T*>( ::operator new( n * sizeof(T), std::align_val_t(64) ) ); }
    void deallocate(T *p, size_t) noexcept { ::operator delete( p, std::align_val_t(64) ); }
    template <class U> bool operator==(FftwAligned<U> const &) const { return true; }
    template <class U> bool operator!=(FftwAligned<U> const &) const { return false; }
};

void computeTidalWebClassification(Field &fields,
                                   User_options const &userOptions,
                                   Quantities *q)
{
    if ( not fields.velocity_tweb ) return;

    MESSAGE::Message message( userOptions.verboseLevel );

    auto skip = [&](char const* why)
    {
        MESSAGE::Warning warning( userOptions.verboseLevel );
        warning << "Skipping the T-web: " << why << "\n" << MESSAGE::EndWarning;
        fields.velocity_tweb = false;   // writer then skips the (empty) T-web output
    };

    if ( not userOptions.periodic ) { skip("the tidal tensor is FFT-Poisson-solved, which needs a periodic box."); return; }
    size_t n[NO_DIM];
    size_t N = 1;
    for (int d = 0; d < NO_DIM; ++d) { n[d] = userOptions.gridSize[d]; N *= n[d]; }
    if ( q->density.size() != N ) { skip("it needs the density field on the full grid (select 'density'/'density_a' too)."); return; }
    // Above 2^31 cells (~1290^3) FFTW's ordinary planners overflow their int sizes: the plans are made
    // with the 64-bit guru interface (fftw_plans64.h). Every buffer below is an ordinary allocation
    // (std::vector), so '--scratch-dir' backs it with disk like the grids: the chain needs ~48 bytes
    // per cell at its peak (the real work array, two half-spectra, the six float tensor components).

    auto const tidalStart = std::chrono::steady_clock::now();
    message << "\nComputing the T-web from the tidal tensor of the raw density grid (lambda_th = "
            << userOptions.lambda_th << ") ... " << MESSAGE::Flush;

    // density contrast (double precision for the FFT chain)
    double mean = 0.;
    for (size_t i = 0; i < N; ++i) mean += double(q->density[i]);
    mean /= double(N);
    if ( mean <= 0. ) { skip("the density grid has non-positive mean."); return; }

    std::vector<double, FftwAligned<double> > work(N);
    for (size_t i = 0; i < N; ++i) work[i] = double(q->density[i]) / mean - 1.;

    // the half-spectra: std::complex<double> has fftw_complex's layout (FFTW manual, "Complex numbers")
    using Spectrum = std::vector< std::complex<double>, FftwAligned< std::complex<double> > >;
    size_t const nLast = n[NO_DIM-1], nzh = nLast/2 + 1;    // the last axis is the half-complex one
    size_t const nOuter = N / nLast;                         // the full axes' cells, flattened
    Spectrum deltaKBuf( nOuter * nzh ), tensKBuf( nOuter * nzh );
    fftw_complex *deltaK = reinterpret_cast<fftw_complex*>( deltaKBuf.data() );
    fftw_complex *tensK  = reinterpret_cast<fftw_complex*>( tensKBuf.data() );
    // 64-bit sizes (fftw_plans64.h): the ordinary planners stop at 2^31 cells
    fftw_plan planF = fftwPlans64::r2c(NO_DIM, n, work.data(), deltaK, FFTW_ESTIMATE);
    if ( planF == nullptr ) { skip("FFTW could not plan the forward transform."); return; }
    fftw_execute(planF);
    fftw_destroy_plan(planF);

    // per-cell angular wavenumbers k_d = 2 pi m_d / n_d (signed m); ratios k_i k_j / k^2 are the
    // dimensionless tidal kernel, and the Gaussian smoothing sigma is naturally in grid cells.
    // The Nyquist SIGN matters for the cross terms (k_a k_b flips with it): full axes treat the
    // Nyquist mode as negative (numpy fftfreq convention) and the rfft half-axis as positive
    // (numpy rfftfreq), matching the validated Python reference implementation exactly.
    auto kFull = [](size_t idx, size_t n) -> double
    {
        long m = long(idx);
        if ( m > (long(n) - 1) / 2 ) m -= long(n);
        return 2. * M_PI * double(m) / double(n);
    };
    auto kHalf = [](size_t idx, size_t n) -> double
    {
        return 2. * M_PI * double(idx) / double(n);
    };

    double const invN   = 1. / double(N);   // FFTW round trip is unnormalized
    // the upper triangle of the symmetric tensor, row by row (symmetricEigenvalues3x3's argument order)
#if NO_DIM == 3
    static int const NPAIRS = 6;
    static int const PAIRS[NPAIRS][2] = { {0,0}, {0,1}, {0,2}, {1,1}, {1,2}, {2,2} };
#else
    static int const NPAIRS = 3;
    static int const PAIRS[NPAIRS][2] = { {0,0}, {0,1}, {1,1} };
#endif
    std::vector<Real> tens[NPAIRS];

    fftw_plan planB = fftwPlans64::c2r(NO_DIM, n, tensK, work.data(), FFTW_ESTIMATE);
    if ( planB == nullptr ) { skip("FFTW could not plan the inverse transform."); return; }
    for (int c = 0; c < NPAIRS; ++c)
    {
        int const a = PAIRS[c][0], b = PAIRS[c][1];
#ifdef OPEN_MP
        #pragma omp parallel for
#endif
        for (size_t o = 0; o < nOuter; ++o)     // (ix, iy) in 3D, ix in 2D: the half-spectrum's rows
        {
            double k[NO_DIM];
            size_t rest = o;
            for (int d = NO_DIM - 2; d >= 0; --d) { k[d] = kFull(rest % n[d], n[d]); rest /= n[d]; }
            for (size_t iz = 0; iz < nzh; ++iz)
            {
                k[NO_DIM-1] = kHalf(iz, nLast);
                double k2 = 0.;
                for (int d = 0; d < NO_DIM; ++d) k2 += k[d]*k[d];
                size_t const idx = o * nzh + iz;
                if ( k2 == 0. ) { tensK[idx][0] = 0.; tensK[idx][1] = 0.; continue; }
                double const f = (k[a] * k[b] / k2) * invN;
                tensK[idx][0] = f * deltaK[idx][0];
                tensK[idx][1] = f * deltaK[idx][1];
            }
        }
        fftw_execute(planB);          // -> work (real tensor component)
        tens[c].resize(N);
        for (size_t i = 0; i < N; ++i) tens[c][i] = Real(work[i]);
    }
    fftw_destroy_plan(planB);
    Spectrum().swap( deltaKBuf );
    Spectrum().swap( tensKBuf );
    work.clear(); work.shrink_to_fit();

    // eigenvalues (descending) and classification
    q->velocity_tweb.assign(N, Real(0.));
    q->velocity_tweb_eigenvalues.assign(N, Pvector<Real,NO_DIM>::zero());
    size_t counts[4] = {0, 0, 0, 0};
#ifdef OPEN_MP
    #pragma omp parallel for reduction(+:counts[:4])
#endif
    for (size_t i = 0; i < N; ++i)
    {
#if NO_DIM == 3
        Pvector<Real,NO_DIM> eig = symmetricEigenvalues3x3( tens[0][i], tens[1][i], tens[2][i],
                                                            tens[3][i], tens[4][i], tens[5][i] );
#else
        Pvector<Real,NO_DIM> eig = symmetricEigenvalues3x3( tens[0][i], tens[1][i], Real(0.),
                                                            tens[2][i], Real(0.), Real(0.) );
#endif
        q->velocity_tweb_eigenvalues[i] = eig;
        int const label = classifyWeb( eig, userOptions.lambda_th );
        q->velocity_tweb[i] = Real(label);
        counts[label] += 1;
    }

    message << "Done.\n";
#if NO_DIM == 3
    static char const* NAMES[4] = { "void", "wall", "filament", "node" };
#else
    static char const* NAMES[3] = { "void", "filament", "node" };     // one collapsed axis is a line in 2D
#endif
    message << "T-web volume fractions:";
    for (int l = 0; l <= NO_DIM; ++l)
        message << "  " << NAMES[l] << " = " << (100. * double(counts[l]) / double(N)) << "\%";
    message << "\n";
    message << "  >>> Time: " << std::chrono::duration<double>(std::chrono::steady_clock::now() - tidalStart).count()
            << " sec. (T-web tidal tensor)\n" << MESSAGE::Flush;
}


// Fills the velocity divergence/shear/vorticity vectors requested in 'fields' from the stored gradient.
void computeDivergenceShearVorticity(Field &fields,
                                     int const verboseLevel,
                                     Quantities *q)
{
    if ( q->velocity_gradient.empty() ) return;

    if ( fields.velocity_divergence )
    {
        q->velocity_divergence.reserve( q->velocity_gradient.size() );
        for (std::vector< Pvector<Real,noGradComp> >::iterator it=q->velocity_gradient.begin(); it!=q->velocity_gradient.end(); ++it )
            q->velocity_divergence.push_back( velocityDivergence(*it) );
    }

    if ( fields.velocity_shear )
    {
        q->velocity_shear.reserve( q->velocity_gradient.size() );
        for (std::vector< Pvector<Real,noGradComp> >::iterator it=q->velocity_gradient.begin(); it!=q->velocity_gradient.end(); ++it )
            q->velocity_shear.push_back( velocityShear(*it) );
    }
    
    if ( fields.velocity_vorticity )
    {
        q->velocity_vorticity.reserve( q->velocity_gradient.size() );
        for (std::vector< Pvector<Real,noGradComp> >::iterator it=q->velocity_gradient.begin(); it!=q->velocity_gradient.end(); ++it )
            q->velocity_vorticity.push_back( velocityVorticity(*it) );
    }
    
    // free the gradient unless the gradient itself or the V-web classification still needs it
    // (the T-web is computed from the density grid, never the gradient); swap-with-empty
    // because clear() keeps the capacity resident
    if ( not fields.velocity_gradient and not fields.velocity_vweb )
        std::vector< Pvector<Real,noGradComp> >().swap( q->velocity_gradient );
}
