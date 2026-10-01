/* Per-tetrahedron caustic stratification for '--ps-caustics'.
 *
 * WHAT THIS COMPUTES. The Lagrangian->Eulerian map of a phase-space tetrahedron is LINEAR, so its
 * deformation tensor J = dx/dq is CONSTANT inside the tetrahedron and exactly recoverable from the
 * two edge matrices the deposit already builds: with rows holding the edge vectors of vertices
 * 1..NO_DIM relative to vertex 0, Ax = Lag * J^T, hence J^T = Lag^-1 * Ax. Eigenvalues are
 * invariant under transposition, so the routine below analyses M = Lag^-1 * Ax directly.
 *
 * The number of NEGATIVE real eigenvalues of J is the number of principal axes along which the
 * sheet has already collapsed and inverted -- the collapse multiplicity k. It refines the fold
 * flag exactly: sign(det J) = (-1)^k, so the existing parity bit is k mod 2, and k additionally
 * separates one-axis collapse (pancakes/walls) from two-axis (filaments) and three-axis (nodes).
 * In 3D k is in 0..3, in 2D in 0..2.
 *
 * DERIVATIVE-FREE OR NOT. The collapse multiplicity above and the D4 (umbilic) indicator are
 * pointwise. D4 is where TWO eigenvalues of J vanish TOGETHER, and both halves matter: a test for
 * coincidence alone is not an umbilic test, it fires across a box with no caustic in it (see
 * UMBILIC_ZERO_BAND).
 *
 * A2/A3/A4 are not pointwise. They are separated by successive directional derivatives of the
 * critical eigenvalue along its own eigenvector (D_n = the n-th derivative of lambda_c along v_c):
 *
 *     A2  lambda_c = 0, D1 != 0                 surface in Lagrangian space (codimension 1)
 *     A3  lambda_c = 0, D1 = 0, D2 != 0         curve                       (codimension 2)
 *     A4  lambda_c = 0, D1 = D2 = 0, D3 != 0    isolated points             (codimension 3)
 *
 * None of those exist inside a tetrahedron where J is constant, so they are fitted across the
 * surrounding tetrahedra (foldDerivatives). BIT_CUSP and BIT_SWALLOWTAIL are therefore indicators at
 * the stencil's resolution, not identifications of the strata -- most sharply for A4, whose points
 * are isolated and so lie inside no tetrahedron generically. The 1-D validation waves are
 * translation-symmetric, which inflates their A3/A4 sets into planes; that is what makes them
 * measurable at all.
 *
 * The per-cell export ORs these per-tetrahedron bits together, exactly like the parity bits, so the
 * result stays independent of the partition and thread schedule (OR is commutative and idempotent).
 */

#ifndef PS_CAUSTIC_CLASS_HEADER
#define PS_CAUSTIC_CLASS_HEADER

#include <algorithm>
#include <cmath>

#include "../define.h"
#include "../math_functions.h"


namespace PSCausticClass
{

/* Wider than a byte on purpose: bit 8 does not fit in an unsigned char. Every accumulator the mask
   is OR-ed into must use this typedef, or bit 8 is truncated with no compiler warning (a narrowing
   OR into an existing byte). The '.causticClass' export stays float32, which is exact to 2^24. */
typedef unsigned short CausticMask;

/* Bit layout of the per-cell mask exported as '.causticClass'. Bits 0..1 repeat the parity bits of
   '.caustic' so the mask is self-contained; bits 2..5 record WHICH collapse multiplicities occur
   among the tetrahedra overlapping the cell; bit 6 is the umbilic-degeneracy indicator; bits 7..8
   are the two Arnold-stratum indicators, both of which need the '--ps-caustic-cusps' stencil. */
enum
{
    BIT_PARITY_POS  = 1 << 0,  // a det(J) > 0 tetrahedron overlaps the cell
    BIT_PARITY_NEG  = 1 << 1,  // a det(J) < 0 tetrahedron overlaps  (both set = fold crossing)
    BIT_COLLAPSE_0  = 1 << 2,  // k = 0 present (no axis collapsed)
    BIT_COLLAPSE_1  = 1 << 3,  // k = 1 present (one axis: wall/pancake)
    BIT_COLLAPSE_2  = 1 << 4,  // k = 2 present (two axes: filament)
    BIT_COLLAPSE_3  = 1 << 5,  // k = 3 present (three axes: node)
    BIT_DEGENERATE  = 1 << 6,  // two eigenvalues coincide (umbilic/D4 INDICATOR, not a D4 label)
    BIT_CUSP        = 1 << 7,  // A3 cusp INDICATOR: D1 = 0, the fold is tangent to its null direction
    BIT_SWALLOWTAIL = 1 << 8,  // A4 INDICATOR: D1 = D2 = 0 as well (see the header note on codim 3)

    /* Everything the GPU deposit can produce: the host builds its per-tet mask without the cusp
       stencil, so bits 7..8 never come back and the merge masks them off. */
    BITS_GPU_MAX    = BIT_PARITY_POS | BIT_PARITY_NEG | BIT_COLLAPSE_0 | BIT_COLLAPSE_1
                    | BIT_COLLAPSE_2 | BIT_COLLAPSE_3 | BIT_DEGENERATE
};

/* D4 (umbilic) thresholds. A D4 point is where two eigenvalues of J vanish TOGETHER, so the test
 * has two halves and needs both: the pair must coincide, and it must sit at zero.
 *
 * Coincidence alone is near-vacuous -- any locally isotropic patch satisfies it. Measured without
 * the zero-band: 97.6% of all touched cells on a weak wave (A*k = 0.15) that has no caustic
 * anywhere, and on crossed waves two thirds of the flagged cells lay on no fold, which an umbilic
 * cannot do.
 *
 * Both tolerances are relative to the tetrahedron's eigenvalue SPREAD, not to the pair's own
 * magnitudes: |l_i - l_j| <= tol*(|l_i| + |l_j|) becomes unsatisfiable exactly where D4 lives. */
static const double DEGENERACY_REL_TOL = 1.e-3;
static const double UMBILIC_ZERO_BAND  = 0.25;

/* A3 (cusp) detection thresholds.
 *
 * A fold is the surface where the critical eigenvalue lambda_c of J passes through zero. It is an
 * ordinary A2 fold where lambda_c varies at first order ALONG its own null direction v_c, and an A3
 * cusp where that variation stops: grad(lambda_c) . v_c = 0. Both quantities are needed only near
 * the fold itself, so a tetrahedron is considered only when |lambda_c| is small compared with the
 * spread of the eigenvalues (FOLD_BAND), and the tangency test is then made scale-free by
 * comparing |grad . v_c| against |grad| (CUSP_REL_TOL).
 *
 * grad(lambda_c) does not exist inside one tetrahedron -- J is constant there -- so it is estimated
 * by least squares from the neighbouring tetrahedra, which makes this an INDICATOR on the
 * tessellation's own resolution scale, not a pointwise identification of the A3 stratum. */
static const double FOLD_BAND    = 0.25;
static const double CUSP_REL_TOL = 0.15;

/* A4 (swallowtail) thresholds, applied to the derivatives from foldDerivatives() below.
 *
 * A4 asks for one more derivative to vanish than A3: D1 = 0 AND D2 = 0. The test is NOT "each
 * derivative is small" -- derivatives of different order have different units and no common scale.
 * (An earlier attempt comparing each Dn against its own |dlambda|/|dq|^n yardstick separated a true
 * A4 from a pure A3 by only a factor 5.)
 *
 * What is actually being asked is that the Taylor series of lambda_c along v_c START at cubic order,
 * i.e. that the cubic term dominate over the stencil. With h the stencil radius the contributions
 *
 *     c1 = |D1| h ,   c2 = |D2| h^2 / 2 ,   c3 = |D3| h^3 / 6
 *
 * are dimensionless once compared with each other, and the strata separate: A2 is dominated by c1,
 * A3 by c2, A4 by c3. TERM_TOL is how far below c3 the two lower terms must sit.
 *
 * SIGNAL_MIN keeps the ratios from being noise where lambda_c is flat, and rejects A5+ (where D3
 * vanishes too). Note it is UNTUNED: none of the three analytic waves exercises it, and sweeping it
 * from 0.15 to 0.90 moves the counts by at most one cell.
 *
 * MEASURED on the three 1-D waves, A4-flagged cells as a fraction of A3-flagged (32^3 particles,
 * 48^3 grid), with the deduplicated nearest-64 stencil of ps_interpolation.cc:
 *
 *     A*k = 1.8, a generic A2 fold                          0 of 7 cells        0.00%
 *     A*k = 1.0, an exact A3 (D2 = k^2 != 0)                10 of 1972 cells    0.51%   <- floor
 *     three-harmonic, an exact A4 (D1 = D2 = 0, D3 != 0)    1473 of 1976 cells 74.54%   <- signal
 *
 * a factor ~146 of separation. TERM_TOL 0.35 holds the floor but drops the signal to 15%; 0.65 buys
 * signal for a 0.66% floor. The signal can exceed 100% at large TERM_TOL because bit 8 is not a
 * subset of bit 7 -- see foldDerivatives.
 *
 * The flagged band has a width no tolerance can sharpen: at Lagrangian distance delta from the A4
 * point the fit sees D2 = 6*C*delta, so c2/c3 = 3*delta/h and the flag survives to delta ~
 * TERM_TOL*h/3 -- a fixed fraction of the STENCIL RADIUS. Hence proximity, not identification. */
static const double SWALLOWTAIL_TERM_TOL   = 0.50;
static const double SWALLOWTAIL_SIGNAL_MIN = 0.25;


/* Real eigenvalues of a 3x3 matrix from its characteristic polynomial, in closed form.
   lambda^3 - p*lambda^2 + q*lambda - r = 0 with p = tr(M), q = sum of principal 2x2 minors,
   r = det(M). Returns the number of real roots found (1 or 3) and fills 'root'. */
inline int realEigenvalues3(double const M[3][3], double root[3])
{
    double const p = M[0][0] + M[1][1] + M[2][2];
    double const q = (M[0][0]*M[1][1] - M[0][1]*M[1][0])
                   + (M[0][0]*M[2][2] - M[0][2]*M[2][0])
                   + (M[1][1]*M[2][2] - M[1][2]*M[2][1]);
    double const r = M[0][0]*(M[1][1]*M[2][2] - M[1][2]*M[2][1])
                   - M[0][1]*(M[1][0]*M[2][2] - M[1][2]*M[2][0])
                   + M[0][2]*(M[1][0]*M[2][1] - M[1][1]*M[2][0]);

    // depress to t^3 + a*t + b with lambda = t + p/3
    double const shift = p / 3.;
    double const a = q - p*p/3.;
    double const b = -2.*p*p*p/27. + p*q/3. - r;

    double const disc = -4.*a*a*a - 27.*b*b;      // > 0: three distinct real roots
    double const eps  = 1.e-300;
    if ( std::fabs(a) < eps and std::fabs(b) < eps )
    {                                              // triple root
        root[0] = root[1] = root[2] = shift;
        return 3;
    }
    /* A DOUBLE root sits exactly at disc == 0, where rounding routinely pushes the computed value
       slightly negative (e.g. -4e-5 against terms of order 1 for eigenvalues (-0.5,-0.5,0.9)) and
       would divert a perfectly real triple into the single-root branch -- losing precisely the
       coincident pair the umbilic indicator is looking for. Anything within rounding of zero is
       therefore treated as three real roots; the trigonometric branch clamps its argument, so a
       marginally out-of-range acos() is safe. */
    double const discScale = std::max( std::fabs(4.*a*a*a), std::fabs(27.*b*b) );
    if ( disc >= -1.e-9 * discScale )
    {                                              // trigonometric (Viete) form, all roots real
        double const m = 2. * std::sqrt( -a/3. );
        double arg = 3.*b / (a*m);
        if ( arg >  1. ) arg =  1.;                // guard the rounding at a double root
        if ( arg < -1. ) arg = -1.;
        double const theta = std::acos( arg ) / 3.;
        for (int i=0; i<3; ++i)
            root[i] = m * std::cos( theta - 2.*M_PI*double(i)/3. ) + shift;
        return 3;
    }
    // one real root (Cardano)
    double const s = std::sqrt( b*b/4. + a*a*a/27. );
    double const u = std::cbrt( -b/2. + s );
    double const v = std::cbrt( -b/2. - s );
    root[0] = u + v + shift;
    return 1;
}


/* The deformation of one tetrahedron: everything the caustic bits are derived from, exposed so the
   A3 test can reuse the critical eigenvalue and its null direction across neighbouring tetrahedra.
   'valid' is false when the Lagrangian simplex is too flat to define J. */
struct TetDeformation
{
    bool   valid    = false;
    double critical = 0.;            // eigenvalue of J closest to zero (the one that folds)
    double spread   = 0.;            // largest |eigenvalue|, the scale |critical| is judged against
    double vcrit[NO_DIM] = {0.};     // unit right eigenvector of 'critical' (the null direction)
    CausticMask bits = 0;            // parity-free bits: collapse multiplicity + degeneracy
};


// Unit null vector of (M - lambda*I) for a 3x3 M: the row pair whose cross product is longest.
#if NO_DIM==3
inline bool nullVector3(double const M[3][3], double lambda, double v[3])
{
    double A[3][3];
    for (int i=0; i<3; ++i)
        for (int j=0; j<3; ++j) A[i][j] = M[i][j] - (i==j ? lambda : 0.);

    double best = 0.;
    for (int a=0; a<3; ++a)
    {
        int const b = (a+1) % 3;
        double const c[3] = { A[a][1]*A[b][2] - A[a][2]*A[b][1],
                              A[a][2]*A[b][0] - A[a][0]*A[b][2],
                              A[a][0]*A[b][1] - A[a][1]*A[b][0] };
        double const n = std::sqrt( c[0]*c[0] + c[1]*c[1] + c[2]*c[2] );
        if ( n > best ) { best = n; v[0]=c[0]/n; v[1]=c[1]/n; v[2]=c[2]/n; }
    }
    return best > 0.;
}
#endif


/* Full deformation analysis of one tetrahedron from its Eulerian (Ax) and Lagrangian (Lag) edge
   matrices. classifyTet() below is the thin wrapper the deposit uses when it only wants the bits.
   Both are built on the two shared primitives declared beneath, so the deformation tensor and the
   characteristic cubic are each solved exactly once per call. */
inline TetDeformation analyzeTet(double const Ax[NO_DIM][NO_DIM],
                                 double const Lag[NO_DIM][NO_DIM]);
inline bool deformationTensor(double const Ax[NO_DIM][NO_DIM],
                              double const Lag[NO_DIM][NO_DIM],
                              double M[NO_DIM][NO_DIM]);
inline CausticMask bitsFromSpectrum(double const ev[], int const nReal);


/* Classification bits for one tetrahedron, from its Eulerian (Ax) and Lagrangian (Lag) edge
   matrices. Returns 0 when the Lagrangian simplex is degenerate (no invertible map, so no
   deformation tensor exists) -- the caller then keeps only the parity bit it already has.
   This is the deposit's hot path: it wants the bits and nothing else, so it stops at the spectrum
   and never forms the null vector that analyzeTet needs. */
inline CausticMask classifyTet(double const Ax[NO_DIM][NO_DIM],
                               double const Lag[NO_DIM][NO_DIM])
{
    double M[NO_DIM][NO_DIM];
    if ( not deformationTensor( Ax, Lag, M ) ) return 0;

#if NO_DIM==3
    double ev[3];
    int const nReal = realEigenvalues3( M, ev );
    return bitsFromSpectrum( ev, nReal );
#elif NO_DIM==2
    // 2x2: lambda^2 - tr*lambda + det = 0; a negative discriminant is a complex pair (nReal = 0)
    double const tr  = M[0][0] + M[1][1];
    double const det = M[0][0]*M[1][1] - M[0][1]*M[1][0];
    double const d2  = tr*tr - 4.*det;
    double ev[2] = {0., 0.};
    int nReal = 0;
    if ( d2 >= 0. )
    {
        double const s = std::sqrt(d2);
        ev[0] = 0.5*(tr + s);
        ev[1] = 0.5*(tr - s);
        nReal = 2;
    }
    return bitsFromSpectrum( ev, nReal );
#endif
}


/* The Lagrangian->Eulerian deformation tensor of one tetrahedron, shared by classifyTet and
   analyzeTet so the inverse and the matrix product are formed exactly once per call site. */
inline bool deformationTensor(double const Ax[NO_DIM][NO_DIM],
                              double const Lag[NO_DIM][NO_DIM],
                              double M[NO_DIM][NO_DIM])
{
    /* The inverse is formed HERE in double precision instead of through math_functions.h's
       matrixInverse(): that one writes a 'Real' result (float in a single-precision build) and
       zeroes it whenever |det| < 1e-6 as an ABSOLUTE threshold, which small Lagrangian simplices
       can legitimately fall below -- a zero inverse would then be silently classified as k = 0.
       The guard below is relative to the matrix scale instead. */
    double scale = 0.;
    for (int i=0; i<NO_DIM; ++i)
        for (int j=0; j<NO_DIM; ++j) scale = std::max( scale, std::fabs(Lag[i][j]) );
    if ( scale <= 0. ) return false;

    double lagCopy[NO_DIM][NO_DIM];
    for (int i=0; i<NO_DIM; ++i)
        for (int j=0; j<NO_DIM; ++j) lagCopy[i][j] = Lag[i][j];
    double const detLag = determinant( lagCopy );
    // a simplex flatter than this carries no usable deformation tensor
    double detScale = 1.;
    for (int i=0; i<NO_DIM; ++i) detScale *= scale;
    if ( not std::isfinite(detLag) or std::fabs(detLag) < 1.e-12 * detScale ) return false;

    double lagInv[NO_DIM][NO_DIM];
#if NO_DIM==3
    lagInv[0][0] =  (Lag[1][1]*Lag[2][2] - Lag[1][2]*Lag[2][1]) / detLag;
    lagInv[0][1] = -(Lag[0][1]*Lag[2][2] - Lag[0][2]*Lag[2][1]) / detLag;
    lagInv[0][2] =  (Lag[0][1]*Lag[1][2] - Lag[0][2]*Lag[1][1]) / detLag;
    lagInv[1][0] = -(Lag[1][0]*Lag[2][2] - Lag[1][2]*Lag[2][0]) / detLag;
    lagInv[1][1] =  (Lag[0][0]*Lag[2][2] - Lag[0][2]*Lag[2][0]) / detLag;
    lagInv[1][2] = -(Lag[0][0]*Lag[1][2] - Lag[0][2]*Lag[1][0]) / detLag;
    lagInv[2][0] =  (Lag[1][0]*Lag[2][1] - Lag[1][1]*Lag[2][0]) / detLag;
    lagInv[2][1] = -(Lag[0][0]*Lag[2][1] - Lag[0][1]*Lag[2][0]) / detLag;
    lagInv[2][2] =  (Lag[0][0]*Lag[1][1] - Lag[0][1]*Lag[1][0]) / detLag;
#elif NO_DIM==2
    lagInv[0][0] =  Lag[1][1] / detLag;
    lagInv[0][1] = -Lag[0][1] / detLag;
    lagInv[1][0] = -Lag[1][0] / detLag;
    lagInv[1][1] =  Lag[0][0] / detLag;
#endif

    // M = Lag^-1 * Ax  (= J^T; same eigenvalues as J)
    for (int i=0; i<NO_DIM; ++i)
        for (int j=0; j<NO_DIM; ++j)
        {
            double s = 0.;
            for (int k=0; k<NO_DIM; ++k) s += lagInv[i][k] * Ax[k][j];
            M[i][j] = s;
        }
    return true;
}


/* The classification bits that follow from an ALREADY-COMPUTED spectrum. Split out so analyzeTet
   can reuse the eigenvalues it has instead of asking classifyTet to solve the cubic a second time
   (which is exactly what it used to do -- see the note there). */
inline CausticMask bitsFromSpectrum(double const ev[], int const nReal)
{
    CausticMask bits = 0;
    int nNeg = 0;
    bool degenerate = false;

#if NO_DIM==3
    for (int i=0; i<nReal; ++i)
        if ( ev[i] < 0. ) ++nNeg;
    if ( nReal == 3 )
    {
        double evSpread = 0.;
        for (int i=0; i<3; ++i) evSpread = std::max( evSpread, std::fabs(ev[i]) );
        for (int i=0; i<3 and not degenerate; ++i)
            for (int j=i+1; j<3 and not degenerate; ++j)
            {
                if ( evSpread <= 0. ) continue;
                if ( std::fabs(ev[i]-ev[j]) > DEGENERACY_REL_TOL*evSpread ) continue;
                // ... and the coincident pair must be AT ZERO, or this is just an isotropic patch
                double const pair = std::max( std::fabs(ev[i]), std::fabs(ev[j]) );
                if ( pair <= UMBILIC_ZERO_BAND*evSpread ) degenerate = true;
            }
    }
#elif NO_DIM==2
    /* nReal is 0 for a complex pair (no collapsed axis to report) and 2 otherwise.
       No umbilic bit in 2D, deliberately. The 3D test asks that a coincident PAIR sit at zero while
       a third eigenvalue sets the scale; with only two eigenvalues the pair IS the whole spectrum,
       so "both at zero relative to the spread" is either vacuous or unsatisfiable depending on what
       one picks for the scale, and there is no honest third choice. That matches the mathematics: a
       corank-2 degeneracy needs the entire differential to vanish, which is codimension 3 and so
       does not occur generically in a 2-parameter Lagrangian family. Nothing claimed. */
    for (int i=0; i<nReal; ++i)
        if ( ev[i] < 0. ) ++nNeg;
#endif

    switch ( nNeg )
    {
        case 0:  bits |= BIT_COLLAPSE_0; break;
        case 1:  bits |= BIT_COLLAPSE_1; break;
        case 2:  bits |= BIT_COLLAPSE_2; break;
        default: bits |= BIT_COLLAPSE_3; break;
    }
    if ( degenerate ) bits |= BIT_DEGENERATE;
    return bits;
}


/* Deformation tensor M = Lag^-1 * Ax (= J^T, same eigenvalues), its bits, and the eigenvalue
   nearest zero with its null direction.
   Shares deformationTensor() + bitsFromSpectrum() with classifyTet so the characteristic cubic is
   solved once per call, not twice. Not a cold path: under '--ps-caustic-cusps' this runs once per
   tetrahedron plus once per ring member of every near-fold one. */
inline TetDeformation analyzeTet(double const Ax[NO_DIM][NO_DIM],
                                 double const Lag[NO_DIM][NO_DIM])
{
    TetDeformation td;
#if NO_DIM==3
    double M[3][3];
    if ( not deformationTensor( Ax, Lag, M ) ) return td;   // degenerate simplex: bits stay 0

    double ev[3];
    int const nReal = realEigenvalues3( M, ev );
    if ( nReal < 1 ) return td;
    td.bits = bitsFromSpectrum( ev, nReal );

    int best = 0;
    for (int i=1; i<nReal; ++i)
        if ( std::fabs(ev[i]) < std::fabs(ev[best]) ) best = i;
    td.critical = ev[best];
    for (int i=0; i<nReal; ++i) td.spread = std::max( td.spread, std::fabs(ev[i]) );
    if ( not nullVector3( M, ev[best], td.vcrit ) ) return td;
    td.valid = true;
#else
    td.bits = classifyTet( Ax, Lag );   // 2D: no critical direction, only the bits
#endif
    return td;
}


/* The A3 (cusp) indicator for one tetrahedron.
 *
 * 'self' is its deformation, 'dq' the Lagrangian offsets of the neighbours from its own centroid
 * and 'dLambda' their critical eigenvalue minus its own. The gradient of lambda_c is the least-
 * squares solution of dq . g = dLambda (normal equations on a 3x3 system -- with four neighbours
 * the system is over-determined, which is what makes it usable at all on an irregular
 * tessellation). The cusp condition is that the gradient carries no component along the null
 * direction: |g . v_c| << |g|.
 *
 * Returns false unless the tetrahedron is BOTH near its fold and tangent there. */
inline bool cuspIndicator(TetDeformation const &self,
                          double const dq[][NO_DIM], double const dLambda[], int n)
{
#if NO_DIM==3
    if ( not self.valid or n < 3 ) return false;
    if ( self.spread <= 0. ) return false;
    if ( std::fabs(self.critical) > FOLD_BAND * self.spread ) return false;   // not near a fold

    double A[3][3] = {{0.,0.,0.},{0.,0.,0.},{0.,0.,0.}}, b[3] = {0.,0.,0.};
    for (int k=0; k<n; ++k)
    {
        for (int i=0; i<3; ++i)
        {
            for (int j=0; j<3; ++j) A[i][j] += dq[k][i] * dq[k][j];
            b[i] += dq[k][i] * dLambda[k];
        }
    }
    double Acopy[3][3];
    for (int i=0; i<3; ++i) for (int j=0; j<3; ++j) Acopy[i][j] = A[i][j];
    double const detA = determinant( Acopy );
    double sc = 0.;
    for (int i=0; i<3; ++i) for (int j=0; j<3; ++j) sc = std::max( sc, std::fabs(A[i][j]) );
    if ( sc <= 0. or not std::isfinite(detA) or std::fabs(detA) < 1.e-12 * sc*sc*sc )
        return false;                       // neighbours too coplanar to fix a gradient

    // Cramer's rule: g = A^-1 b
    double g[3];
    for (int c=0; c<3; ++c)
    {
        double T[3][3];
        for (int i=0; i<3; ++i)
            for (int j=0; j<3; ++j) T[i][j] = (j==c ? b[i] : A[i][j]);
        g[c] = determinant( T ) / detA;
    }
    /* The yardstick is the variation actually MEASURED across the neighbours, not the norm of the
       fitted gradient. |g| is the wrong reference: on an irregular tessellation the transverse
       components of g absorb fit noise, so a tetrahedron whose neighbours barely differ along v_c
       -- a set of neighbours clustered in one slab, say -- gets a small |g.v_c| next to a
       noise-inflated |g| and looks like a cusp. Measured on the 1-D Zel'dovich wave at A*k = 1.8,
       where theory says the fold is generic and there are NO cusps, that mistake flagged 15% of
       the fold cells. The largest observed slope |dlambda|/|dq| is a direct, noise-free scale. */
    double slope = 0.;
    for (int k=0; k<n; ++k)
    {
        double const len = std::sqrt( dq[k][0]*dq[k][0] + dq[k][1]*dq[k][1] + dq[k][2]*dq[k][2] );
        if ( len > 0. ) slope = std::max( slope, std::fabs(dLambda[k]) / len );
    }
    double const gn = slope;
    /* A VANISHING variation is the cusp condition in its strongest form -- at an A3 point lambda_c
       is stationary, so symmetric neighbours legitimately measure no slope (the 1-D Zel'dovich wave
       at A*k = 1 folds at kq = pi, where lambda_c = 1 + cos(kq) has a quadratic minimum) -- but
       ONLY when the tetrahedron is also essentially ON the fold. On a regular lattice plenty of
       tetrahedra have neighbours whose critical eigenvalues happen to coincide while lambda_c
       itself is far from zero; accepting those flagged 15% of the fold cells of the A*k = 1.8 wave,
       which theory says has no cusps at all. FOLD_BAND is deliberately generous so the gradient can
       be measured on the approach to a fold; this branch, having no gradient to lean on, demands
       the much stricter statement that lambda_c has actually reached zero. */
    if ( gn <= 0. ) return std::fabs(self.critical) <= 0.02 * self.spread;
    double const along = std::fabs( g[0]*self.vcrit[0] + g[1]*self.vcrit[1] + g[2]*self.vcrit[2] );
    return along <= CUSP_REL_TOL * gn;
#else
    (void)self; (void)dq; (void)dLambda; (void)n;
    return false;
#endif
}


/* Gaussian elimination with partial pivoting for foldDerivatives()' normal equations. Returns false
   when the system is singular relative to its own scale, i.e. this stencil does not fix the fit. */
inline bool solveSmall(double A[][5], double b[], int n, double x[])
{
    double scale = 0.;
    for (int i=0; i<n; ++i)
        for (int j=0; j<n; ++j) scale = std::max( scale, std::fabs(A[i][j]) );
    if ( scale <= 0. ) return false;

    for (int c=0; c<n; ++c)
    {
        int piv = c;
        for (int r=c+1; r<n; ++r)
            if ( std::fabs(A[r][c]) > std::fabs(A[piv][c]) ) piv = r;
        if ( std::fabs(A[piv][c]) < 1.e-12 * scale ) return false;
        if ( piv != c )
        {
            for (int j=0; j<n; ++j) std::swap( A[c][j], A[piv][j] );
            std::swap( b[c], b[piv] );
        }
        for (int r=c+1; r<n; ++r)
        {
            double const f = A[r][c] / A[c][c];
            if ( f == 0. ) continue;
            for (int j=c; j<n; ++j) A[r][j] -= f * A[c][j];
            b[r] -= f * b[c];
        }
    }
    for (int r=n-1; r>=0; --r)
    {
        double s = b[r];
        for (int j=r+1; j<n; ++j) s -= A[r][j] * x[j];
        x[r] = s / A[r][r];
    }
    for (int i=0; i<n; ++i)
        if ( not std::isfinite(x[i]) ) return false;
    return true;
}


/* The first three derivatives of lambda_c along its own null direction v_c, plus the measured
   yardsticks the A3/A4 tests judge them against.
 *
 * MODEL. With t = dq . v_c the axial coordinate, the fit over the stencil is
 *
 *     dLambda ~= g . dq  +  c * t^2  +  d * t^3        (5 unknowns: g is a full 3-vector)
 *
 * anchored at the tetrahedron's own centroid, so there is no constant term. D1 = g.v_c, D2 = 2c,
 * D3 = 6d. The gradient is kept FULLY three-dimensional while the higher orders are axial only:
 * transverse variation of lambda_c is real and first order, and dropping it would alias straight
 * into the axial coefficients, but transverse second and third order terms are small enough to
 * leave out -- and every extra column is another place for stencil noise to land.
 *
 * THE CUBIC TERM IS NOT OPTIONAL. Near a swallowtail lambda_c ~ C*t^3, and least squares of a
 * purely LINEAR model against a cubic returns g.v_c = C*sum(t^4)/sum(t^2) ~ 0.6*C*h^2 -- about 0.6
 * of the measured slope, four times cuspIndicator's 0.15 threshold. So a genuine A4 point does NOT
 * set BIT_CUSP, and swallowtailIndicator must re-derive D1 from THIS model rather than chain off
 * cuspIndicator. Expect cells flagged A4 but not A3; the two bits are independent measurements. */
struct FoldDerivatives
{
    bool   valid = false;
    double d1 = 0.;      // (v_c . grad) lambda_c
    double d2 = 0.;      // (v_c . grad)^2 lambda_c
    double d3 = 0.;      // (v_c . grad)^3 lambda_c -- the A4 non-degeneracy witness
    double slope = 0.;   // max |dLambda| / |dq| over the stencil: the measured yardstick for d1
    double span  = 0.;   // max |dq|: the stencil radius the term contributions are weighed at
    double signal = 0.;  // max |dLambda|: the variation the fit has to account for
};


inline FoldDerivatives foldDerivatives(TetDeformation const &self,
                                       double const dq[][NO_DIM], double const dLambda[], int n)
{
    FoldDerivatives fd;
#if NO_DIM==3
    // Five unknowns: five samples would fit exactly, with no redundancy, i.e. fit the noise.
    // Under 10 is not worth a second-derivative claim (and excludes the four-face stencil).
    if ( not self.valid or n < 10 ) return fd;

    double A[5][5] = {{0.}}, b[5] = {0.,0.,0.,0.,0.};
    for (int k=0; k<n; ++k)
    {
        double const t = dq[k][0]*self.vcrit[0] + dq[k][1]*self.vcrit[1] + dq[k][2]*self.vcrit[2];
        double const row[5] = { dq[k][0], dq[k][1], dq[k][2], t*t, t*t*t };
        for (int i=0; i<5; ++i)
        {
            for (int j=0; j<5; ++j) A[i][j] += row[i] * row[j];
            b[i] += row[i] * dLambda[k];
        }
        double const len2 = dq[k][0]*dq[k][0] + dq[k][1]*dq[k][1] + dq[k][2]*dq[k][2];
        if ( len2 <= 0. ) continue;
        double const len = std::sqrt( len2 ), aL = std::fabs( dLambda[k] );
        fd.slope  = std::max( fd.slope,  aL / len );
        fd.span   = std::max( fd.span,   len );
        fd.signal = std::max( fd.signal, aL );
    }

    double x[5];
    if ( not solveSmall( A, b, 5, x ) ) return fd;
    fd.d1 = x[0]*self.vcrit[0] + x[1]*self.vcrit[1] + x[2]*self.vcrit[2];
    fd.d2 = 2. * x[3];
    fd.d3 = 6. * x[4];
    fd.valid = true;
#else
    (void)self; (void)dq; (void)dLambda; (void)n;
#endif
    return fd;
}


/* The A4 (swallowtail) indicator for one tetrahedron: the fold is tangent to its null direction AND
   that tangency is itself degenerate. Same stencil and same arguments as cuspIndicator, so the
   caller gathers the ring once and asks both questions. Proximity along the cusp curve, not
   membership of the A4 stratum -- see the codimension note at the top of this file. */
inline bool swallowtailIndicator(TetDeformation const &self,
                                 double const dq[][NO_DIM], double const dLambda[], int n)
{
#if NO_DIM==3
    if ( not self.valid or self.spread <= 0. ) return false;
    if ( std::fabs(self.critical) > FOLD_BAND * self.spread ) return false;   // not near a fold

    FoldDerivatives const fd = foldDerivatives( self, dq, dLambda, n );
    if ( not fd.valid ) return false;
    if ( fd.slope <= 0. or fd.span <= 0. or fd.signal <= 0. ) return false;   // nothing varies

    // Tangency keeps its own MEASURED yardstick, as in cuspIndicator, so a large true slope is
    // rejected before any ratio of fitted terms is trusted.
    if ( std::fabs(fd.d1) > CUSP_REL_TOL * fd.slope ) return false;

    double const h  = fd.span;
    double const c1 = std::fabs(fd.d1) * h;
    double const c2 = std::fabs(fd.d2) * h*h / 2.;
    double const c3 = std::fabs(fd.d3) * h*h*h / 6.;

    // The cubic term must account for a real share of the measured variation, or the ratios below
    // pass trivially wherever lambda_c is flat. Also rejects A5+, where D3 vanishes too.
    if ( c3 < SWALLOWTAIL_SIGNAL_MIN * fd.signal ) return false;

    // and the expansion must START at that cubic term: both lower orders negligible beside it
    return c1 <= SWALLOWTAIL_TERM_TOL * c3 and c2 <= SWALLOWTAIL_TERM_TOL * c3;
#else
    (void)self; (void)dq; (void)dLambda; (void)n;
    return false;
#endif
}

}   // namespace PSCausticClass

#endif
