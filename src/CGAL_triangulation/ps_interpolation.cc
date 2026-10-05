/* Phase-space DTFE grid interpolation. Triangulation is built in Lagrangian space;
   Eulerian simplices can overlap (multi-stream), so we iterate over all cells and
   accumulate each simplex's contribution to every grid point it contains. */

#ifdef PHASE_SPACE

/* 2D phase space (a DIM=2 build: PS-DTFE-2d, 'make PS-DTFE DIM=2'), since 2026-10-02.
 *
 * Until then this was a hard #error, with good reason: the 2D path compiled but was never fed. The
 * Gadget-HDF5 reader reads a whole dataset into a buffer of N*NO_DIM values, so a 3D file overran it,
 * and a purpose-made 2D pancake reported 62 streams over 74% of the grid. What made it work:
 *   * 2D input: an HDF5 file whose Coordinates / Velocities / InitialCoordinates have TWO columns
 *     (tests/generate_ps_test_data.py --dim 2) -- the reader needs no change for that;
 *   * the exact, tie-consistent point-in-TRIANGLE test (ps_exact_inside.h's 2D tiers), so every
 *     sample is counted by exactly one triangle of each stream;
 *   * the vertex degrees through CGAL's 2D face circulator, the beyond-grid sample count as a plain
 *     loop, and the exact deposit through ps_clip2d.h (triangle-cell clipping + polygon moments in
 *     r3d's moment layout), as is the exact folded-overlap flag of '.hidden_streams'.
 * Validated on the analytic 2D pancake (stream counts exact in every column off the caustics,
 * single-stream density converging 0.011 -> 0.004, velocity to 1e-4, the sampled deposit converging
 * to the exact one; mass exact under --ps-vertex-mass, to 2e-4 with the default Lagrangian-area
 * masses, which drop the few triangles squeezed flat on a caustic exactly as in 3D) and on 2D crossed
 * waves (streams {1,3,9} in the separable fractions): tests/ps_2d_check.sh. Also fixed on the way:
 * '--partition' demanded 3 values in every build, and CGAL's 2D binary triangulation I/O does not
 * read back (tessellation_cache.h uses its ASCII mode in 2D). Supported in 2D: the sampled and exact deposits, streams, the hidden-stream
 * flags, velocity / dispersion / gradient moments, --ps-vertex-mass, --ps-volume-weighted,
 * --ps-halo-release, --ps-caustics (parity + collapse multiplicity k = 0..2), --partition, periodic
 * and non-periodic boxes, the tessellation cache. 3D only: the GPU deposit, point evaluation
 * (--sample-points, --serve), --ps-window, --ps-caustic-cusps and --ps-linear-deposit's GPU path.
 */

#include "triangulation_common.h"

#include <algorithm> // sort/unique/nth_element for the deduplicated nearest-N cusp stencil
#include <atomic>    // the slab deposit's dynamic slab pickup
#include <chrono>    // timing of the GPU-deferred CPU deposits
#include <climits>   // LONG_MAX (markFlipped's default slab bounds)
#include <cstring>   // memcpy in the in-place cost-sort permutation
#include <exception> // the deposit threads hand their errors back
#include <thread>    // the CPU deposit's threads (psDepositThreads)
#include <type_traits>
#include <unordered_map>   // PS_DEBUG_DEGREES self-check
#ifdef OPEN_MP
#include <omp.h>
#endif

// gpu_host.h declares the GPU deposit API (METAL=1 / CUDA=1 / HIP=1 builds; the CPU deposit remains the
// fallback) AND defines psWindowHull, the --ps-window hull of the exact deposit, which the CPU loop uses
// in every build -- so it is included unconditionally (declarations only; nothing links against them
// unless PS_GPU is set)
#include "gpu_host.h"
#include "ps_clip2d.h"      // the 2D exact deposit's clipper (DIM=2 builds)

// The finite cells (3D) / faces (2D) around a vertex, appended to 'out'.
template <class Tr, class V>
static inline void psFiniteIncidentCells(Tr const &dt, V const &vh, std::vector<Cell_handle> &out)
{
#if NO_DIM==3
    dt.finite_incident_cells( vh, std::back_inserter(out) );
#else
    auto fc = dt.incident_faces( vh ), done = fc;
    if ( fc == nullptr ) return;
    do { if ( not dt.is_infinite( fc ) ) out.push_back( fc ); } while ( ++fc != done );
#endif
}

// --ps-window pre-filter: does the raw cell span [lo, hi) of a tetrahedron (unclamped; the periodic
// image it lies in) touch the sub-box [sub0, sub0+subN) -- which may itself wrap the axis -- under
// the grid's wrap? A tetrahedron no axis of which touches the window deposits nothing and is skipped
// before any sampling (most of a large box's tetrahedra, for a small window).
static inline bool psSpanTouches(long lo, long hi, long n, bool periodic, long sub0, long subN)
{
    if (hi <= lo) return false;
    if (!periodic)
    {
        lo = std::max(lo, 0L); hi = std::min(hi, n);
        if (hi <= lo) return false;
    }
    else if (hi - lo >= n) return true;
    long A[2][2], B[2][2];
    auto pieces = [n](long a, long b, long (&out)[2][2]) -> int   // [a,b) as residues mod n: one or two runs
    {
        long const len = b - a;
        a = ((a % n) + n) % n; b = a + len;
        if (b <= n) { out[0][0] = a; out[0][1] = b; return 1; }
        out[0][0] = a; out[0][1] = n; out[1][0] = 0; out[1][1] = b - n; return 2;
    };
    int const na = pieces(lo, hi, A), nb = pieces(sub0, sub0 + subN, B);
    for (int i = 0; i < na; ++i)
        for (int j = 0; j < nb; ++j)
            if (A[i][0] < B[j][1] && B[j][0] < A[i][1]) return true;
    return false;
}
// a cell index that marks "counted for the tet's normalization, outside the window: not deposited"
static size_t const PS_NOT_IN_WINDOW = size_t(-1);
#if defined(PS_GPU) && NO_DIM==3
#include "ps_deposit_params.h"   // the GPU per-tet record (psFillTetRecord)
#endif

// --ps-caustics: per-tetrahedron caustic stratification bits. Included UNCONDITIONALLY -- the
// classification is part of the CPU deposit and exists in 2D as well, so it must not sit behind
// the GPU/3D guard above (doing so breaks the CPU-only and the 2D builds).
#include "ps_caustic_class.h"

// exact, tie-consistent sample-in-tetrahedron test (shared with the point evaluation)
#include "ps_exact_inside.h"

// psDepositThreadBytes: the memory the deposit threads' private copies may take (budgeted there too)
#include "../auto_tune.h"

#if NO_DIM==3
// --ps-exact-deposit: analytic tetrahedron-cell intersection moments (Powell & Abel 2015).
// Vendored C library; r3d.h carries its own extern "C" guard. 3D only.
#include "../../third_party/r3d/r3d.h"
#endif

namespace {
/* Does a tetrahedron overlap an axis-aligned box with POSITIVE volume? The separating-axis test
   over the 3 box normals, the 4 face normals and the 18 (edge x box axis) products, which is
   complete for two convex polyhedra. Contact of zero volume (a shared face, edge or vertex) counts
   as separated. Double throughout, in a frame relative to the tetrahedron's vertex 0; the tet's
   projections and the box's projected half-widths depend only on the tet (all boxes are one grid
   cell), so they are set once and each box costs one dot product per axis. Used by the '.hidden_streams'
   multi-stream flag (see flippedFlag). */
struct TetBoxSAT
{
    int n = 0;
    double a[25][3], lo[25], hi[25], rad[25];

    void add(double const v[4][3], double const half[3], double x, double y, double z)
    {
        if ( !(x*x + y*y + z*z > 0.) ) return;        // an edge parallel to a box axis: no new axis
        a[n][0] = x; a[n][1] = y; a[n][2] = z;
        double pmin = x*v[0][0] + y*v[0][1] + z*v[0][2], pmax = pmin;
        for (int k = 1; k < 4; ++k)
        {
            double const p = x*v[k][0] + y*v[k][1] + z*v[k][2];
            pmin = std::min(pmin, p);
            pmax = std::max(pmax, p);
        }
        lo[n] = pmin;
        hi[n] = pmax;
        rad[n] = half[0]*std::fabs(x) + half[1]*std::fabs(y) + half[2]*std::fabs(z);
        ++n;
    }

    TetBoxSAT(double const v[4][3], double const half[3])
    {
        add(v, half, 1., 0., 0.);
        add(v, half, 0., 1., 0.);
        add(v, half, 0., 0., 1.);
        static int const F[4][3] = { {1,2,3}, {0,2,3}, {0,1,3}, {0,1,2} };
        for (int f = 0; f < 4; ++f)
        {
            double e1[3], e2[3];
            for (int d = 0; d < 3; ++d)
            {
                e1[d] = v[F[f][1]][d] - v[F[f][0]][d];
                e2[d] = v[F[f][2]][d] - v[F[f][0]][d];
            }
            add(v, half, e1[1]*e2[2] - e1[2]*e2[1], e1[2]*e2[0] - e1[0]*e2[2], e1[0]*e2[1] - e1[1]*e2[0]);
        }
        static int const E[6][2] = { {0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3} };
        for (int e = 0; e < 6; ++e)
        {
            double const dx = v[E[e][1]][0] - v[E[e][0]][0];
            double const dy = v[E[e][1]][1] - v[E[e][0]][1];
            double const dz = v[E[e][1]][2] - v[E[e][0]][2];
            add(v, half, 0., dz, -dy);          // edge x (1,0,0)
            add(v, half, -dz, 0., dx);          // edge x (0,1,0)
            add(v, half, dy, -dx, 0.);          // edge x (0,0,1)
        }
    }

    bool overlaps(double const c[3]) const     // c = the box centre, in the tet's frame
    {
        for (int i = 0; i < n; ++i)
        {
            double const bc = a[i][0]*c[0] + a[i][1]*c[1] + a[i][2]*c[2];
            if ( hi[i] <= bc - rad[i] || lo[i] >= bc + rad[i] ) return false;
        }
        return true;
    }
};
} // namespace


// Threads for one partition's CPU deposit: every core when this is the only triangulation being
// processed; inside the partitioned loop (an OpenMP team of concurrent partitions) the cores left
// per partition, so the two levels never oversubscribe. DTFE_DEPOSIT_THREADS pins it.
static int psDepositThreads()
{
    if ( const char *env = std::getenv("DTFE_DEPOSIT_THREADS") )
    {
        int const v = std::atoi(env);
        if ( v > 0 ) return v;
    }
    int cores = int( std::thread::hardware_concurrency() );
#ifdef OPEN_MP
    cores = omp_get_max_threads();
    if ( omp_in_parallel() )
        return std::max( 1, cores / std::max( 1, omp_get_num_threads() ) );
#endif
    return std::max( 1, cores );
}

/* Interpolates fields onto a regular grid via PS-DTFE: scatters each finite cell's
   contribution to overlapping grid points so multi-stream regions are handled.
   nSub = sub-samples per axis for volume-averaged fields (1 = unaveraged).
   mayClearDT = this is the last use of an internally-owned dt: clear it right after the
   deposit's last read (GPU-success or CPU-loop end) to free ~650 B/vertex before the
   stats/normalization here and the caller's merge into the shared grids. */
// The samples of a straddling tetrahedron that lie BEYOND a non-periodic grid, counted (and, for the
// linear deposit, their weights summed) for the sampled deposit's share m / (in + beyond). Kept OUT
// of the deposit loop's lambda, not inlined, and handed COPIES of the tetrahedron's state, on purpose:
// written inline, its nested lambdas captured the loop's hoisted locals by reference; as a function
// taking the loop's invD / vertex positions by reference, their addresses escaped and every in-grid
// sample test -- every tetrahedron's hot path -- reloaded them from memory. Either cost the TNG
// region's deposit 30-50 s of CPU, against the ~4 s this counting takes. The function keeps its
// own copy of the exact-test cache (rawPos / exactTet): at most one extra setup per tetrahedron.
struct PSBeyondGrid
{
    size_t nGrid[NO_DIM];
    int    nSub;
    double gridLen[NO_DIM];
    Real   boxLo[NO_DIM];
    Real   dx[NO_DIM];
};
struct PSBeyondTet
{
    int    iMinU[NO_DIM], iMaxU[NO_DIM];   // the unclamped window
    Real   eulerPos[NO_DIM+1][NO_DIM];
    double invD[NO_DIM][NO_DIM];
    double band1;
    Cell_handle cell;
    bool   psLinear, linearProfile;
    Real   denBase, denGrad[NO_DIM];
};
struct PSBeyondCount { size_t n = 0; double w = 0.; };

__attribute__((noinline))
static PSBeyondCount psCountBeyondGrid(PSBeyondGrid const &grid, PSBeyondTet const &T,
                                       psExactInside::Frame const &exactFrame)
{
#if NO_DIM==3
    int const *iMinU = T.iMinU, *iMaxU = T.iMaxU;
    Real const (&eulerPos)[NO_DIM+1][NO_DIM] = T.eulerPos;
    double const (&invD)[NO_DIM][NO_DIM] = T.invD;
    double const band1 = T.band1;
    Cell_handle const &cell = T.cell;
    bool const psLinear = T.psLinear, linearProfile = T.linearProfile;
    Real const denBase = T.denBase;
    Real const (&denGrad)[NO_DIM] = T.denGrad;
    Real rawPos[NO_DIM+1][NO_DIM];
    bool rawPosReady = false;
    psExactInside::TetTest exactTet;
    size_t nOutside = 0;
    double outsideW = 0.;
    // The samples beyond the grid are only COUNTED (and, for the linear deposit, their weights
    // summed), so they are taken one sample COLUMN at a time, along the tet's LONGEST window axis
    // (fewest columns): along a column every barycentric is affine in the position, so the samples
    // that clear tier 1's band by a margin on all four faces form one run, counted in bulk, and so
    // do those that cannot be inside. Only the few in neither -- within the band of a face -- take
    // the per-sample exact test below, so the count is exactly the per-sample one, whichever axis
    // the columns run along. Only columns that HAVE samples beyond the grid and lie inside the tet's
    // projection are visited. Short columns take the per-sample test.
    int ax[NO_DIM] = { 0, 1, 2 };   // ax[2]: the column axis; ax[0] rows, ax[1] the inner loop
    {
        int best = 0;
        for (int d = 1; d < NO_DIM; ++d) if ( iMaxU[d] - iMinU[d] > iMaxU[best] - iMinU[best] ) best = d;
        int k = 0;
        for (int d = 0; d < NO_DIM; ++d) if ( d != best ) ax[k++] = d;
        ax[2] = best;
    }
    int const a0 = ax[0], a1 = ax[1], ac = ax[2];
    long const nS = long(grid.nSub);
    long const S0c = long(iMinU[ac]) * nS, S1c = long(iMaxU[ac]) * nS;
    long const gridC = long(grid.nGrid[ac]) * nS;
    double const e0[NO_DIM] = { double(eulerPos[0][0]), double(eulerPos[0][1]), double(eulerPos[0][2]) };
    auto floorDiv = [nS](long s) -> long { return s >= 0 ? s / nS : -((-s + nS - 1) / nS); };
    // sample s along axis d -> its position, lo + (s + 1/2) h: the samplePosAxis expression up to rounding,
    // with one multiply-add instead of a floor division and a division (the positions only need to be the
    // sample lattice; each sample's inside/outside decision below is exact at the position used)
    double h[NO_DIM], invH[NO_DIM];
    for (int d = 0; d < NO_DIM; ++d)
    {
        h[d] = grid.gridLen[d] / (double(grid.nGrid[d]) * double(grid.nSub));
        invH[d] = 1. / h[d];
    }
    auto samplePos = [&](int d, long s) -> double { return double(grid.boxLo[d]) + (double(s) + 0.5) * h[d]; };
    // the in-grid loop's offset from vertex 0 (Real), for the linear weights
    auto sampleRel = [&](int d, long s) -> Real
    {
        long const k = floorDiv(s);
        Real const fr = (grid.nSub == 1) ? Real(0.5) : ( Real(s - k * nS) + Real(0.5) ) / Real(grid.nSub);
        return grid.boxLo[d] + (Real(k) + fr) * grid.dx[d] - eulerPos[0][d];
    };
    auto linearWeightAt = [&](long s0, long s1, long sc) -> double
    {
        if ( !linearProfile ) return 1.;
        Real w = denBase + denGrad[a0] * sampleRel(a0, s0) + denGrad[a1] * sampleRel(a1, s1)
               + denGrad[ac] * sampleRel(ac, sc);
        return w < Real(0.) ? 0. : double(w);
    };
    // the per-sample exact test of the in-grid loop (tier 1, then the canonical tiers)
    auto exactInside = [&](double const pos[NO_DIM]) -> bool
    {
        double relD[NO_DIM], baryD[NO_DIM];
        for (int d = 0; d < NO_DIM; ++d) relD[d] = pos[d] - e0[d];
        int const fine = psExactInside::fineClassify( invD, relD, band1, baryD );
        if ( fine != 0 ) return fine > 0;
        if ( not rawPosReady )
        {
            for (int v = 0; v <= NO_DIM; ++v)
                for (int d = 0; d < NO_DIM; ++d)
                    rawPos[v][d] = cell->vertex(v)->info().eulerianPosition(d);
            rawPosReady = true;
        }
        double pc[NO_DIM];
        psExactInside::canonicalQuery( exactFrame, pos, pc );
        return psExactInside::canonicalInside( exactFrame, rawPos, exactTet, pc );
    };
    // tier 1's band plus a margin far above the rounding difference between the affine
    // evaluation below and fineClassify's (|bary| terms stay << 1e6 for any kept tet)
    double const tBand = band1 + 1.e-9;
    bool const bulkOK = band1 < 1.e300;   // no inverse: every sample takes the exact tiers
    // only the columns that HAVE samples beyond the grid: when the window stays inside the grid
    // along the column axis, a column inside it along a0 needs just its a1-parts beyond the grid
    bool const cStraddle = S0c < 0 || S1c > gridC;
    long const S10 = long(iMinU[a1]) * nS, S11 = long(iMaxU[a1]) * nS, grid1 = long(grid.nGrid[a1]) * nS;
    // ... and only the columns inside the tet's projection on the (a0, a1) plane: along a row that is
    // one interval, the extent of the six projected edges crossing it, padded by a sample each side
    double V[NO_DIM+1][2];
    for (int v = 0; v <= NO_DIM; ++v)
    {
        V[v][0] = double(eulerPos[v][a0]) - e0[a0];
        V[v][1] = double(eulerPos[v][a1]) - e0[a1];
    }
    double const rowPad = 1.e-7 * double(grid.dx[a0]);
    double const s1Scale = invH[a1];
    double const s1Shift = (e0[a1] - double(grid.boxLo[a1])) * s1Scale - 0.5;
    double const scScale = invH[ac];
    double const scShift = (e0[ac] - double(grid.boxLo[ac])) * scScale - 0.5;
    // the faces' slopes along the column axis are the same for every column: invert them once
    double Gc[4], invGc[4];
    for (int v = 0; v < NO_DIM; ++v) Gc[v] = invD[ac][v];
    Gc[3] = -(Gc[0] + Gc[1] + Gc[2]);
    for (int v = 0; v < 4; ++v) invGc[v] = Gc[v] != 0. ? 1. / Gc[v] : 0.;
    for (long s0 = long(iMinU[a0]) * nS; s0 < long(iMaxU[a0]) * nS; ++s0)
    {
        long const k0 = floorDiv(s0);
        bool const in0 = k0 >= 0 && k0 < long(grid.nGrid[a0]);
        double const rel0 = samplePos(a0, s0) - e0[a0];
        double lo1 = std::numeric_limits<double>::infinity(), hi1 = -lo1;
        for (int i = 0; i < NO_DIM; ++i)
            for (int j = i + 1; j <= NO_DIM; ++j)
            {
                double const xi = V[i][0], xj = V[j][0];
                if ( rel0 < std::min(xi, xj) - rowPad || rel0 > std::max(xi, xj) + rowPad ) continue;
                double y0 = V[i][1], y1 = V[j][1];
                if ( xj != xi )
                {
                    double const t = std::min(1., std::max(0., (rel0 - xi) / (xj - xi)));
                    y0 = y1 = V[i][1] + t * (V[j][1] - V[i][1]);
                }
                lo1 = std::min(lo1, std::min(y0, y1));
                hi1 = std::max(hi1, std::max(y0, y1));
            }
        if ( !(lo1 <= hi1) ) continue;   // this row misses the tet
        long const pLo = std::max(S10, long(std::floor(lo1 * s1Scale + s1Shift)) - 1);
        long const pHi = std::min(S11, long(std::ceil(hi1 * s1Scale + s1Shift)) + 2);
        if ( pLo >= pHi ) continue;
        long rLo[2] = { pLo, 0 }, rHi[2] = { pHi, 0 };
        int nR = 1;
        if ( in0 && !cStraddle )
        {
            nR = 0;
            if ( pLo < 0 )     { rLo[nR] = pLo; rHi[nR] = std::min(pHi, 0L); ++nR; }
            if ( pHi > grid1 ) { rLo[nR] = std::max(pLo, grid1); rHi[nR] = pHi; ++nR; }
        }
        for (int ir = 0; ir < nR; ++ir)
        for (long s1 = rLo[ir]; s1 < rHi[ir]; ++s1)
        {
            long const k1 = floorDiv(s1);
            bool const colIn = in0 && k1 >= 0 && k1 < long(grid.nGrid[a1]);
            // this column's samples beyond the grid: all of it, or its parts below / above the grid
            long outLo[2], outHi[2];
            int nOut = 0;
            if ( !colIn ) { outLo[0] = S0c; outHi[0] = S1c; nOut = 1; }
            else
            {
                if ( S0c < 0 )     { outLo[nOut] = S0c; outHi[nOut] = std::min(S1c, 0L); ++nOut; }
                if ( S1c > gridC ) { outLo[nOut] = std::max(S0c, gridC); outHi[nOut] = S1c; ++nOut; }
            }
            if ( nOut == 0 ) continue;
            double pos[NO_DIM];
            pos[a0] = samplePos(a0, s0);
            pos[a1] = samplePos(a1, s1);
            auto perSample = [&](long u, long v)
            {
                for (long sc = u; sc < v; ++sc)
                {
                    pos[ac] = samplePos(ac, sc);
                    if ( !exactInside(pos) ) continue;
                    ++nOutside;
                    if ( psLinear ) outsideW += linearWeightAt(s0, s1, sc);
                }
            };
            long outLen = 0;
            for (int i = 0; i < nOut; ++i) outLen += outHi[i] - outLo[i];
            if ( !bulkOK || outLen <= 4 )
            {
                for (int i = 0; i < nOut; ++i) perSample(outLo[i], outHi[i]);
                continue;
            }
            // bary_v = A_v + G_v r_c (v = 0..2) and l0 = A_3 + G_3 r_c, each ONE affine expression of
            // the column offset r_c (which rises with the sample index), so each is monotone along it
            double const r0 = pos[a0] - e0[a0], r1 = pos[a1] - e0[a1];
            double A[4];
            for (int v = 0; v < NO_DIM; ++v) A[v] = invD[a0][v] * r0 + invD[a1][v] * r1;
            A[3] = 1. - (A[0] + A[1] + A[2]);
            double const * const G = Gc;
            auto value = [&](int v, long sc) { return A[v] + G[v] * (samplePos(ac, sc) - e0[ac]); };
            // first index of [lo, hi) where a monotone false->true predicate holds (hi if none): two
            // probes at the analytic crossing 'guess' settle it (it is off by rounding only), bisection
            // covers whatever they leave
            auto transition = [](long lo, long hi, double guess, auto pred) -> long
            {
                long L = lo, H = hi;
                long g = (guess > double(lo)) ? ( guess < double(hi) ? long(std::ceil(guess)) : hi ) : lo;
                if ( g > L && g - 1 < H ) { if ( pred(g - 1) ) H = g - 1; else L = g; }
                if ( g >= L && g < H )    { if ( pred(g) ) H = g; else L = g + 1; }
                while ( L < H ) { long const mid = L + (H - L) / 2; if ( pred(mid) ) H = mid; else L = mid + 1; }
                return L;
            };
            // [a, b) = the samples of [S0c, S1c) with value > T on all four faces (one run)
            auto run = [&](double T, long &a, long &b)
            {
                a = S0c; b = S1c;
                for (int v = 0; v < 4 && a < b; ++v)
                {
                    double const guess = G[v] != 0. ? (T - A[v]) * invGc[v] * scScale + scShift : 0.;
                    if ( G[v] > 0. )        // rises along the column: the first sample above T
                        a = transition( a, b, guess, [&](long sc) { return value(v, sc) > T; } );
                    else if ( G[v] < 0. )   // falls: one past the last sample above T
                        b = transition( a, b, guess, [&](long sc) { return !(value(v, sc) > T); } );
                    else if ( !(A[v] > T) ) b = a;
                }
            };
            long aIn, bIn, aPos, bPos;
            run( tBand, aIn, bIn );     // tier 1 decides these inside
            run( -tBand, aPos, bPos );  // the rest lie beyond -band1 on some face: outside
            if ( aIn >= bIn ) aIn = bIn = aPos;   // nothing certain: all of [aPos, bPos) is tested
            auto clip = [](long a, long b, long u, long v, long &c, long &d) { c = std::max(a, u); d = std::min(b, v); return c < d; };
            for (int i = 0; i < nOut; ++i)
            {
                long c, d;
                if ( clip(aIn, bIn, outLo[i], outHi[i], c, d) )
                {
                    nOutside += size_t(d - c);
                    if ( psLinear )
                    {
                        if ( !linearProfile ) outsideW += double(d - c);
                        else
                        {
                            // sum of den0 + denGrad.rel over the run: rel_c is affine in the index
                            double const n = double(d - c);
                            double const mc = double(grid.boxLo[ac]) - e0[ac]
                                            + double(grid.dx[ac]) * ((double(c) + double(d)) * 0.5) / double(grid.nSub);
                            outsideW += n * ( double(denBase) + double(denGrad[a0]) * double(sampleRel(a0, s0))
                                            + double(denGrad[a1]) * double(sampleRel(a1, s1)) + double(denGrad[ac]) * mc );
                        }
                    }
                }
                if ( clip(aPos, aIn, outLo[i], outHi[i], c, d) ) perSample(c, d);
                if ( clip(bIn, bPos, outLo[i], outHi[i], c, d) ) perSample(c, d);
            }
        }
    }
    PSBeyondCount out;
    out.n = nOutside;
    out.w = outsideW;
    return out;
#else
    // 2D: every sample of the unclamped window beyond the grid, one exact test each (the 3D column
    // scheme above is an optimization of exactly this count)
    PSBeyondCount out;
    long const nS = long(grid.nSub);
    double h[NO_DIM];
    long S0[NO_DIM], S1[NO_DIM], G[NO_DIM];
    for (int d = 0; d < NO_DIM; ++d)
    {
        h[d] = grid.gridLen[d] / (double(grid.nGrid[d]) * double(grid.nSub));
        S0[d] = long(T.iMinU[d]) * nS; S1[d] = long(T.iMaxU[d]) * nS; G[d] = long(grid.nGrid[d]) * nS;
    }
    double const e0[NO_DIM] = { double(T.eulerPos[0][0]), double(T.eulerPos[0][1]) };
    auto floorDiv = [nS](long s) -> long { return s >= 0 ? s / nS : -((-s + nS - 1) / nS); };
    auto sampleRel = [&](int d, long s) -> Real
    {
        long const k = floorDiv(s);
        Real const fr = (grid.nSub == 1) ? Real(0.5) : ( Real(s - k * nS) + Real(0.5) ) / Real(grid.nSub);
        return grid.boxLo[d] + (Real(k) + fr) * grid.dx[d] - T.eulerPos[0][d];
    };
    Real rawPos[NO_DIM+1][NO_DIM];
    bool rawPosReady = false;
    psExactInside::TetTest exactTet;
    for (long s0 = S0[0]; s0 < S1[0]; ++s0)
        for (long s1 = S0[1]; s1 < S1[1]; ++s1)
        {
            if ( s0 >= 0 && s0 < G[0] && s1 >= 0 && s1 < G[1] ) continue;   // the in-grid loop's sample
            double const pos[NO_DIM] = { double(grid.boxLo[0]) + (double(s0) + 0.5) * h[0],
                                         double(grid.boxLo[1]) + (double(s1) + 0.5) * h[1] };
            double relD[NO_DIM], baryD[NO_DIM];
            for (int d = 0; d < NO_DIM; ++d) relD[d] = pos[d] - e0[d];
            int const fine = psExactInside::fineClassify( T.invD, relD, T.band1, baryD );
            bool inside = fine > 0;
            if ( fine == 0 )
            {
                if ( not rawPosReady )
                {
                    for (int v = 0; v <= NO_DIM; ++v)
                        for (int d = 0; d < NO_DIM; ++d)
                            rawPos[v][d] = T.cell->vertex(v)->info().eulerianPosition(d);
                    rawPosReady = true;
                }
                double pc[NO_DIM];
                psExactInside::canonicalQuery( exactFrame, pos, pc );
                inside = psExactInside::canonicalInside( exactFrame, rawPos, exactTet, pc );
            }
            if ( not inside ) continue;
            ++out.n;
            if ( T.psLinear )
            {
                if ( not T.linearProfile ) out.w += 1.;
                else
                {
                    Real const w = T.denBase + T.denGrad[0] * sampleRel(0, s0) + T.denGrad[1] * sampleRel(1, s1);
                    out.w += w < Real(0.) ? 0. : double(w);
                }
            }
        }
    return out;
#endif
}

void interpolateGrid_phaseSpace(DT &dt,
                                User_options &userOptions,
                                Quantities *quantities,
                                Field &field,
                                int nSub,
                                bool mayClearDT)
{
    MESSAGE::Message message( userOptions.verboseLevel );
    size_t const *nGrid = &(userOptions.gridSize[0]);
    Box boxCoordinates = userOptions.region;

    // Averaged fields sample each grid cell on an nSub^NO_DIM sub-grid (nSub==1 = cell centre only).
    if ( nSub < 1 ) nSub = 1;
    size_t nSamplesPerCell = 1;
    for (int d = 0; d < NO_DIM; ++d) nSamplesPerCell *= size_t(nSub);

    Real dx[NO_DIM];
    for (int d = 0; d < NO_DIM; ++d)
        dx[d] = (boxCoordinates[2*d+1] - boxCoordinates[2*d]) / nGrid[d];
    Real cellVolume = Real(1.);
    for (int d = 0; d < NO_DIM; ++d) cellVolume *= dx[d];

    size_t totalGrid = 1;
    for (int d = 0; d < NO_DIM; ++d) totalGrid *= nGrid[d];

    // Every pass below walks the finite cells DIRECTLY via the CGAL iterator (all consuming
    // loops are serial, and iteration order is deterministic for a fixed triangulation), so
    // no cellHandles vector is materialized -- that was 8 B x all finite cells (~0.4 GB per
    // concurrent partition at production scale) held for the whole interpolation.
    size_t noTotalCells = 0;
#if NO_DIM==2
    noTotalCells = dt.number_of_faces();
#elif NO_DIM==3
    noTotalCells = dt.number_of_finite_cells();
#endif
#if NO_DIM==2
    #define PS_FOREACH_FINITE_CELL for (DT::Finite_faces_iterator itC = dt.finite_faces_begin(); itC != dt.finite_faces_end(); ++itC)
    #define PS_FINITE_CELLS_BEGIN dt.finite_faces_begin()
    #define PS_FINITE_CELLS_END dt.finite_faces_end()
#elif NO_DIM==3
    #define PS_FOREACH_FINITE_CELL for (DT::Finite_cells_iterator itC = dt.finite_cells_begin(); itC != dt.finite_cells_end(); ++itC)
    #define PS_FINITE_CELLS_BEGIN dt.finite_cells_begin()
    #define PS_FINITE_CELLS_END dt.finite_cells_end()
#endif

    // psUseSubgrid: store only the Eulerian bbox of cells this partition touches, to bound peak
    // memory. Marks per axis which grid planes any kept cell overlaps (same filter as the scatter,
    // so the box covers every written cell). off -> full grid.
    //
    // The box may WRAP a periodic axis: subOrigin+subDims can exceed nGrid, and the local index is
    // then (g - subOrigin + nGrid) % nGrid. Every consumer must apply that same wrap -- the CPU
    // scatter loops below, the GPU kernels (metal/ps_deposit.metal) and the
    // merge-back (Quantities::addFromSubgrid). The wrap is a no-op for an unwrapped box, so the
    // one expression serves both.
    size_t subOrigin[NO_DIM], subDims[NO_DIM];
    for (int d = 0; d < NO_DIM; ++d) { subOrigin[d] = 0; subDims[d] = nGrid[d]; }
    // cells surviving the dummy/ownership/degeneracy filters, counted by the subgrid pass below;
    // used to right-size the Metal tet arrays (reserving for ALL cells over-allocates ~4x, since
    // most padded-partition cells are filtered out). Upper bound when the pass does not run.
    size_t psKeptCells = noTotalCells;
    bool const windowMode = userOptions.psWindowOn;   // --ps-window: the sub-box is a window of the full grid
    if ( userOptions.psUseSubgrid && !windowMode )
    {
        psKeptCells = 0;
        std::vector<char> touched[NO_DIM];
        for (int d = 0; d < NO_DIM; ++d) touched[d].assign(nGrid[d], char(0));
        PS_FOREACH_FINITE_CELL
        {
            Cell_handle cell = itC;
            PSCellGeometry geo;   // dummy/ownership/hull/degeneracy filters (no inverse check here)
            if ( !psFilterCell(cell, userOptions, boxCoordinates, false, NULL, geo) ) continue;
            Real (&ep)[NO_DIM+1][NO_DIM] = geo.eulerPos;
            ++psKeptCells;
            Real eLo[NO_DIM],eHi[NO_DIM];
            for (int d=0;d<NO_DIM;++d){eLo[d]=ep[0][d];eHi[d]=ep[0][d];}
            for (int v=1;v<=NO_DIM;++v) for (int d=0;d<NO_DIM;++d){ if(ep[v][d]<eLo[d])eLo[d]=ep[v][d]; if(ep[v][d]>eHi[d])eHi[d]=ep[v][d]; }
            for (int d=0;d<NO_DIM;++d)
            {
                int iLo=int(floor((eLo[d]-boxCoordinates[2*d])/dx[d]));
                int iHi=int(floor((eHi[d]-boxCoordinates[2*d])/dx[d]))+1;
                if (nSub>1){iLo-=1;iHi+=1;}
                if (!userOptions.periodic){ if(iLo<0)iLo=0; if(iHi>(int)nGrid[d])iHi=(int)nGrid[d]; }
                for (int g=iLo; g<iHi; ++g){ int w = userOptions.periodic ? ((g%(int)nGrid[d]+(int)nGrid[d])%(int)nGrid[d]) : g; if (w>=0 && w<(int)nGrid[d]) touched[d][w]=char(1); }
            }
        }
        for (int d = 0; d < NO_DIM; ++d)
        {
            int lo=-1, hi=-1;
            for (int g=0; g<(int)nGrid[d]; ++g) if (touched[d][g]){ if(lo<0)lo=g; hi=g; }
            if (lo<0) { subOrigin[d]=0; subDims[d]=nGrid[d]; }                                 // no cells touched: keep full axis (harmless)
            else if (touched[d][0] && touched[d][nGrid[d]-1])
            {
                // Touched planes reach both ends. Under --periodic that is the NORMAL case for
                // any partition whose Eulerian footprint straddles the box seam, and keeping the
                // full axis here was what made every boundary-touching Lagrangian slab allocate
                // the whole grid (the documented "each partition writes the WHOLE Eulerian grid").
                // The footprint is really a WRAPPED arc, so crop to it: its complement is the
                // largest run of untouched planes, and because both ends are touched that run is
                // strictly interior and a plain linear scan finds it.
                int bestLen=0, bestStart=-1, curStart=-1;
                for (int g=0; g<(int)nGrid[d]; ++g)
                {
                    if (!touched[d][g]) { if (curStart<0) curStart=g; }
                    else if (curStart>=0)
                    {
                        int const len = g - curStart;
                        if (len > bestLen) { bestLen=len; bestStart=curStart; }
                        curStart = -1;
                    }
                }
                if (bestLen<=0) { subOrigin[d]=0; subDims[d]=nGrid[d]; }                       // every plane touched: genuinely full
                else { subOrigin[d]=(size_t)((bestStart+bestLen)%(int)nGrid[d]); subDims[d]=(size_t)((int)nGrid[d]-bestLen); }
            }
            else { subOrigin[d]=(size_t)lo; subDims[d]=(size_t)(hi-lo+1); }                    // crop to the touched [lo,hi] span
        }
    }
    if ( windowMode )
        for (int d = 0; d < NO_DIM; ++d) { subOrigin[d] = userOptions.psWindowLo[d]; subDims[d] = userOptions.psWindowDims[d]; }
    size_t subTotal = 1; for (int d = 0; d < NO_DIM; ++d) subTotal *= subDims[d];
    totalGrid = subTotal;   // all allocations and loops below use the sub-grid size
    // --ps-window: a tetrahedron whose Eulerian footprint (with the sampled deposit's one-cell margin)
    // misses the window on some axis deposits nothing: skipped before extraction / sampling
    auto windowTouched = [&](Real const (&ep)[NO_DIM+1][NO_DIM]) -> bool
    {
        for (int d = 0; d < NO_DIM; ++d)
        {
            Real eLo = ep[0][d], eHi = ep[0][d];
            for (int v = 1; v <= NO_DIM; ++v) { eLo = std::min(eLo, ep[v][d]); eHi = std::max(eHi, ep[v][d]); }
            long lo = long(std::floor(double(eLo - boxCoordinates[2*d]) / double(dx[d])));
            long hi = long(std::floor(double(eHi - boxCoordinates[2*d]) / double(dx[d]))) + 1;
            if ( nSub > 1 && !userOptions.psExactDeposit ) { lo -= 1; hi += 1; }
            if ( !psSpanTouches(lo, hi, long(nGrid[d]), userOptions.periodic, long(subOrigin[d]), long(subDims[d])) )
                return false;
        }
        return true;
    };

    // The centroid-fallback cell of a tetrahedron: its Eulerian centroid (Real, from the wrapped
    // positions) and that centroid's flat sub-grid index; false when it lies outside this grid/
    // partition region, where the deposit drops the tet. ONE definition for the CPU fallback,
    // the released tets and the GPU record: all three must put the mass in the same cell.
    auto centroidCell = [&](Real const (&ep)[NO_DIM+1][NO_DIM], Real (&centroid)[NO_DIM], size_t &flat) -> bool
    {
        for (int d = 0; d < NO_DIM; ++d)
        {
            centroid[d] = Real(0.);
            for (int v = 0; v <= NO_DIM; ++v) centroid[d] += ep[v][d];
            centroid[d] /= Real(NO_DIM + 1);
        }
        flat = 0;
        for (int d = 0; d < NO_DIM; ++d)
        {
            int raw = int(floor((centroid[d] - boxCoordinates[2*d]) / dx[d]));
            int w = userOptions.periodic ? ((raw % (int)nGrid[d] + (int)nGrid[d]) % (int)nGrid[d]) : raw;
            long loc = (long)w - (long)subOrigin[d];
            if (loc < 0) loc += (long)nGrid[d];        // sub-box may wrap a periodic axis
            if (w < 0 || w >= (int)nGrid[d] || loc < 0 || loc >= (long)subDims[d]) return false;
            flat = flat * subDims[d] + (size_t)loc;
        }
        return true;
    };

    // Exact containment of the sub-samples (ps_exact_inside.h). A float test with a +-1e-6
    // barycentric tolerance counted a sample near a face shared by two tetrahedra in BOTH, and
    // whether it did depended on the rounding of each run's periodic copies: '.streams' at
    // nSub=1 disagreed with the exact point counts in ~51 of 2.1M cells, and serial and
    // --partition runs disagreed with each other at periodic faces. Now every sample is counted
    // by exactly one tetrahedron of each stream it lies in, as in the point evaluation. The
    // GPU kernel reaches the same decisions (metal/ps_deposit.metal: a float test with a rigorous
    // error bound, deferring the undecidable tets to the CPU loop below).
    psExactInside::Frame exactFrame;
    exactFrame.init( userOptions );
    // Sample positions in DOUBLE, per axis: samplePosAxis[d][g * nSub + k] = lo + ((g + (k+0.5)/nSub)
    // * L) / n for the WRAPPED cell index g -- so every tetrahedron (original or periodic image,
    // any partition) sees the same point; at nSub=1 it is the cell centre (i+0.5)*L/n bit for
    // bit, the expression the tests hand to --sample-points. Built once: nGrid*nSub per axis.
    double gridLenD[NO_DIM];
    std::vector<double> samplePosAxis[NO_DIM];
    for (int d = 0; d < NO_DIM; ++d)
    {
        double const lo = double( boxCoordinates[2*d] );
        gridLenD[d] = double( boxCoordinates[2*d+1] ) - lo;
        samplePosAxis[d].resize( nGrid[d] * size_t(nSub) );
        for (size_t g = 0; g < nGrid[d]; ++g)
            for (int k = 0; k < nSub; ++k)
            {
                double const frac = (nSub == 1) ? 0.5 : (double(k) + 0.5) / double(nSub);
                samplePosAxis[d][g * size_t(nSub) + size_t(k)] = lo + ((double(g) + frac) * gridLenD[d]) / double(nGrid[d]);
            }
    }


    // --ps-vertex-mass: chart-independent per-tetrahedron masses. Count, per vertex, the
    // incident cells that will actually DEPOSIT; each tet's mass is then the sum of its
    // vertices' weight/degree shares (computed where tetMass is needed below). Mass follows
    // the particles instead of rho_bar * V_lag, which removes the 1 - D(z_ic)/D(z) contrast
    // suppression when the Lagrangian input is a PERTURBED IC snapshot (see --ps-vertex-mass
    // help). Counted on THIS triangulation with a reset first -- the unaveraged and averaged
    // passes share one dt, so the pass must be idempotent.
    //
    // The degree applies the SAME chart-independent drop filters as the deposit loop (dummy,
    // periodic hull artefact, singular Eulerian inverse) so that sum over deposited tets of
    // sum_v m_v/deg(v) equals sum_v m_v EXACTLY (up to vertices whose every incident tet is
    // dropped): counting a dropped tet in the degree used to send its vertices' shares into
    // the void with it -- a mass deficit tracking the dropped-tet fraction (0.74% at TNG z=0).
    // Ownership is deliberately SKIPPED: it tiles cells across partitions (each cell deposits
    // exactly once globally, and padded-region vertices see their full local neighborhood, so
    // per-partition degrees match the global tessellation).
    bool const psVertexMass = userOptions.psVertexMass;
    // the full degree pass: every finite cell once, its surviving vertices counted
    auto fullDegreePass = [&]()
    {
        for (DT::Finite_vertices_iterator vit = dt.finite_vertices_begin(); vit != dt.finite_vertices_end(); ++vit)
            vit->info().psDegree() = 0;
        size_t degDropped = 0;   // local counter: the deposit loop reports its own
        PS_FOREACH_FINITE_CELL
        {
            Cell_handle cell = itC;
            PSCellGeometry dgeo;
            if ( !psFilterCell(cell, userOptions, boxCoordinates, true, &degDropped, dgeo,
                               /*skipOwnership=*/true) )
                continue;
            for (int v = 0; v <= NO_DIM; ++v)
                cell->vertex(v)->info().psDegree() += 1;
        }
    };
    // --ps-window on the GPU: only the vertices of the tetrahedra that reach the window need their
    // degree, and the GPU extraction loop (single-threaded) visits exactly those -- so the degrees
    // are computed LAZILY there, each from the vertex's own finite incident cells (the same filter,
    // ownership skipped: the same count as the full pass). The full pass walked every cell of the
    // partition: ~47% of a window deposit. -1 = not computed yet. The CPU loop (threaded) keeps the
    // full pass, which is run before it whenever the GPU does not deposit (degreesLazy cleared).
    bool degreesLazy = false;
#if defined(PS_GPU) && NO_DIM==3
    degreesLazy = psVertexMass && windowMode && userOptions.psUseMetal;
#endif
    if ( degreesLazy )
    {
        for (DT::Finite_vertices_iterator vit = dt.finite_vertices_begin(); vit != dt.finite_vertices_end(); ++vit)
            vit->info().psDegree() = -1;
    }
    else if ( psVertexMass )
        fullDegreePass();
    std::vector<Cell_handle> degIncident;     // scratch of lazyDegrees (single-threaded use only)
    auto lazyDegrees = [&](Cell_handle const &cell)
    {
        for (int v = 0; v <= NO_DIM; ++v)
        {
            Vertex_handle const vh = cell->vertex(v);
            if ( vh->info().psDegree() >= 0 ) continue;
            degIncident.clear();
            psFiniteIncidentCells( dt, vh, degIncident );
            int32_t n = 0;
            size_t dropped = 0;
            for (Cell_handle c : degIncident)
            {
                PSCellGeometry g;
                if ( psFilterCell(c, userOptions, boxCoordinates, true, &dropped, g, /*skipOwnership=*/true) ) ++n;
            }
            vh->info().psDegree() = n;
        }
    };
    // per-tet vertex-share mass: sum over the cell's vertices of weight/degree
    auto psVertexShareMass = [](Cell_handle const &cell) -> double
    {
        double m = 0.;
        for (int v = 0; v <= NO_DIM; ++v)
        {
            int32_t const deg = cell->vertex(v)->info().psDegree();
            if (deg > 0)
                m += double(cell->vertex(v)->info().weight()) / double(deg);
        }
        return m;
    };

    // Allocate the requested output fields, zeroed since the cell loop accumulates into them.
    if ( field.density )
        quantities->density.assign(totalGrid, Real(0.));
    // Dispersion needs <v>, so allocate velocity storage even when only dispersion was
    // requested -- EXCEPT under --ps-volume-weighted, where the dispersion carries its OWN
    // mass-weighted mean (disp_velocity/disp_weight below): a dispersion-only volume-weighted
    // run would otherwise allocate, deposit and GPU-round-trip a 12 B/cell grid nothing reads.
    bool const haveVel = field.velocity
                         || (field.velocity_dispersion && !userOptions.psVolumeWeighted);
    // the dispersion always needs the per-sample velocity VALUES (for sum m v_i v_j and its
    // own mean), even when the velocity GRID above is gated away -- keep the two distinct
    bool const needVelValues = haveVel || field.velocity_dispersion;
    if ( haveVel )
        quantities->velocity.assign(totalGrid, Pvector<Real,noVelComp>::zero());
    if ( field.velocity_gradient )
        quantities->velocity_gradient.assign(totalGrid, Pvector<Real,noGradComp>::zero());
    if ( field.velocity_dispersion )
        quantities->velocity_dispersion.assign(totalGrid, Pvector<Real,noDispComp>::zero());
#ifdef SCALAR
    if ( field.scalar )
        quantities->scalar.assign(totalGrid, Pvector<Real,noScalarComp>::zero());
    if ( field.scalar_gradient )
        quantities->scalar_gradient.assign(totalGrid, Pvector<Real,noScalarGradComp>::zero());
#endif

    // Per-grid-point count of overlapping streams (diagnostic; reported as the stream_count field).
    // Sampled deposit: incremented once per in-simplex sample, divided by nSub^NO_DIM at the end
    // -> the cell-mean stream multiplicity, i.e. the quadrature of the point stream count over
    // the cell's sample lattice (at nSub=1: exactly the number of streams at the cell centre).
    // A tetrahedron that contains NO sample (sub-resolution) still deposits its MASS through the
    // centroid fallback, but is no sample of the stream field and adds nothing here -- counting
    // it +1 (the old behaviour) inflated '.streams' by one whole sample per such tet, up to +839
    // in collapsed cells; the quadrature is unbiased without it. Exact deposit: this counts
    // TETRAHEDRA TOUCHING the cell (every nonzero clip, including corner grazes) -- a geometric
    // multiplicity that is NOT a stream count (it runs to hundreds), so it is exported
    // separately as '.tetTouch' and the physical multiplicity is accumulated in exactMult below.
    std::vector<int> streamCount(totalGrid, 0);
    // --ps-halo-release: a released tet is never sampled, so it has no sample count to
    // contribute; it adds its volume fraction V_tet/V_cell instead -- its exact share of the
    // cell-mean multiplicity, the same value the exact deposit uses for it. Allocated only when
    // tets can be released; added to the sampled multiplicity at export.
    std::vector<Real> releasedMult;
    if ( userOptions.psHaloRelease > Real(0.) && !userOptions.psExactDeposit )
        releasedMult.assign(totalGrid, Real(0.));
    // 1 where a tetrahedron deposited MASS here without any of its samples lying here -- the
    // centroid fallback (sub-resolution or released tets). Such a tet is invisible in the sample
    // count above, so this is exported as '.hidden_streams' to keep single-stream masks clean
    // (quantities.h). The GPU kernel reports it in bit 31 of its stream counter.
    std::vector<unsigned char> subSampleMassFlag(totalGrid, 0);
    // The EXACT companion ('.hidden_streams' bit 1): 1 where a negatively oriented (folded) tet
    // overlaps the cell with positive volume. In a periodic box the Lagrangian->Eulerian map has
    // degree 1, so at every point (#positive - #negative tets containing it) = 1 and the stream
    // count is 1 + 2 #negative: a point is multi-stream EXACTLY where a flipped tet covers it. A
    // cell with no flipped overlap is therefore single-stream everywhere, however small its tets
    // (the mass flag above cannot tell a same-stream sub-sample tet from a foreign stream, and so
    // flagged 26% of the pancake's single-stream cells and 4% of TNG100-3 at z=10). Geometric,
    // so it also catches a folded sliver whose mass went to ANOTHER cell, which the mass flag
    // missed. Marked per tet in both deposit paths (markFlipped), OR-merged across partitions;
    // DTFE() then keeps it only where '.streams' reads single-stream (Quantities::finalizeHiddenStreams).
    std::vector<unsigned char> flippedFlag(totalGrid, 0);
    // xLo/xHi: only cells whose local x index lies in [xLo, xHi) -- the slab deposit's thread owns them
    auto markFlipped = [&](PSCellGeometry const &geo, std::vector<unsigned char> &flags,
                           long const xLo = 0, long const xHi = LONG_MAX)
    {
#if NO_DIM==3
        if ( !(geo.cellDet < 0.) ) return;
        Real const (&ep)[NO_DIM+1][NO_DIM] = geo.eulerPos;
        double v[4][3], half[3];
        int wLo[3], wHi[3];
        size_t cells = 1;
        for (int d = 0; d < 3; ++d)
        {
            double eLo = double(ep[0][d]), eHi = eLo;
            for (int k = 0; k < 4; ++k)
            {
                v[k][d] = double(ep[k][d]) - double(ep[0][d]);
                eLo = std::min(eLo, double(ep[k][d]));
                eHi = std::max(eHi, double(ep[k][d]));
            }
            half[d] = 0.5 * double(dx[d]);
            wLo[d] = int(std::floor((eLo - double(boxCoordinates[2*d])) / double(dx[d])));
            wHi[d] = int(std::floor((eHi - double(boxCoordinates[2*d])) / double(dx[d]))) + 1;
            if ( !userOptions.periodic )
            {
                wLo[d] = std::max(wLo[d], 0);
                wHi[d] = std::min(wHi[d], (int)nGrid[d]);
                if ( wLo[d] >= wHi[d] ) return;
            }
            cells *= size_t(wHi[d] - wLo[d]);   // the whole bbox: 'cells == 1' below means the tet fits one cell
        }
        // --ps-window: only the window's cells are stored, so visit only the hull of the bbox with the
        // window's images (as the exact deposit does) -- on a zoom's fine virtual grid a folded void
        // tet's bbox holds millions of cells, and this walk cost ~55 s per partition of a 32^3 test box
        int lLo[3] = { wLo[0], wLo[1], wLo[2] }, lHi[3] = { wHi[0], wHi[1], wHi[2] };
        if ( windowMode )
            for (int d = 0; d < 3; ++d)
            {
                long hl = 0, hh = 0;
                if ( !psWindowHull(wLo[d], wHi[d], long(nGrid[d]), userOptions.periodic,
                                   long(subOrigin[d]), long(subDims[d]), hl, hh) ) return;
                lLo[d] = int(hl); lHi[d] = int(hh);
            }
        TetBoxSAT const sat(v, half);
        for (int gi = lLo[0]; gi < lHi[0]; ++gi)
        for (int gj = lLo[1]; gj < lHi[1]; ++gj)
        for (int gk = lLo[2]; gk < lHi[2]; ++gk)
        {
            int const raw[3] = { gi, gj, gk };
            size_t flat = 0;
            bool inSub = true;
            for (int d = 0; d < 3; ++d)
            {
                int const w = userOptions.periodic ? ((raw[d] % (int)nGrid[d] + (int)nGrid[d]) % (int)nGrid[d]) : raw[d];
                long loc = (long)w - (long)subOrigin[d];
                if (loc < 0) loc += (long)nGrid[d];        // sub-box may wrap a periodic axis
                if (loc < 0 || loc >= (long)subDims[d]) { inSub = false; break; }
                if (d == 0 && (loc < xLo || loc >= xHi)) { inSub = false; break; }   // another slab's cell
                flat = flat * subDims[d] + (size_t)loc;
            }
            if ( !inSub || flags[flat] ) continue;
            if ( cells == 1 ) { flags[flat] = 1; continue; }   // the whole (non-degenerate) tet is in this cell
            double c[3];
            for (int d = 0; d < 3; ++d)
                c[d] = double(boxCoordinates[2*d]) + (raw[d] + 0.5) * double(dx[d]) - double(ep[0][d]);
            if ( sat.overlaps(c) ) flags[flat] = 1;
        }
#else
        // 2D: a folded triangle overlapping the cell with positive area (ps_clip2d.h); under
        // --ps-window only the hull of the bbox with the window's images, as in 3D
        if ( !(geo.cellDet < 0.) ) return;
        Real const (&ep)[NO_DIM+1][NO_DIM] = geo.eulerPos;
        double tri[NO_DIM+1][NO_DIM];
        int wLo[NO_DIM], wHi[NO_DIM];
        for (int d = 0; d < NO_DIM; ++d)
        {
            double eLo = double(ep[0][d]), eHi = eLo;
            for (int k = 0; k <= NO_DIM; ++k)
            {
                tri[k][d] = double(ep[k][d]) - double(ep[0][d]);
                eLo = std::min(eLo, double(ep[k][d]));
                eHi = std::max(eHi, double(ep[k][d]));
            }
            wLo[d] = int(std::floor((eLo - double(boxCoordinates[2*d])) / double(dx[d])));
            wHi[d] = int(std::floor((eHi - double(boxCoordinates[2*d])) / double(dx[d]))) + 1;
            if ( !userOptions.periodic )
            {
                wLo[d] = std::max(wLo[d], 0);
                wHi[d] = std::min(wHi[d], (int)nGrid[d]);
                if ( wLo[d] >= wHi[d] ) return;
            }
        }
        for (int d = 0; d < NO_DIM; ++d) std::swap( tri[1][d], tri[2][d] );   // folded: counter-clockwise now
        if ( windowMode )
            for (int d = 0; d < NO_DIM; ++d)
            {
                long hl = 0, hh = 0;
                if ( !psWindowHull(wLo[d], wHi[d], long(nGrid[d]), userOptions.periodic,
                                   long(subOrigin[d]), long(subDims[d]), hl, hh) ) return;
                wLo[d] = int(hl); wHi[d] = int(hh);
            }
        for (int gi = wLo[0]; gi < wHi[0]; ++gi)
        for (int gj = wLo[1]; gj < wHi[1]; ++gj)
        {
            int const raw[NO_DIM] = { gi, gj };
            size_t flat = 0;
            bool inSub = true;
            for (int d = 0; d < NO_DIM; ++d)
            {
                int const w = userOptions.periodic ? ((raw[d] % (int)nGrid[d] + (int)nGrid[d]) % (int)nGrid[d]) : raw[d];
                long loc = (long)w - (long)subOrigin[d];
                if (loc < 0) loc += (long)nGrid[d];
                if (loc < 0 || loc >= (long)subDims[d]) { inSub = false; break; }
                if (d == 0 && (loc < xLo || loc >= xHi)) { inSub = false; break; }   // another slab's cell
                flat = flat * subDims[d] + (size_t)loc;
            }
            if ( !inSub || flags[flat] ) continue;
            double lo[NO_DIM], hi[NO_DIM], mom[psClip2d::MOMENTS];
            for (int d = 0; d < NO_DIM; ++d)
            {
                lo[d] = double(boxCoordinates[2*d]) + raw[d] * double(dx[d]) - double(ep[0][d]);
                hi[d] = lo[d] + double(dx[d]);
            }
            if ( psClip2d::clipTriangleToCell( tri, lo, hi, mom ) ) flags[flat] = 1;
        }
#endif
    };
    // --ps-exact-deposit: exact cell-mean stream multiplicity = (1/V_cell) * sum_tets V_int,
    // the analytic nSub->infinity limit of the sampled definition (which is the sample-mean
    // multiplicity). On the same scale as the sampled '.streams', so single-stream masking
    // works under either deposit -- but ONLY to a tolerance (|s-1| < 1e-3, see
    // dtfelib.STREAM_TOL): this is a float sum, so single-stream cells land on 1.0 +/- eps,
    // never exactly 1. Grazing slivers contribute ~0 here instead of a full +1, which also
    // makes it far more robust than the raw touch count (float GPU included).
    std::vector<Real> exactMult;
    if ( userOptions.psExactDeposit ) exactMult.assign(totalGrid, Real(0.));

    // --ps-caustics: per-cell orientation mask (bit0 = a det(Ax)>0 stream overlaps the cell,
    // bit1 = det<0). det(Ax)'s sign is the parity of the Lagrangian->Eulerian map (CGAL cells
    // are positively oriented in Lagrangian space), which flips at every fold -- both bits set
    // means a fold caustic surface crosses the cell. Exported as small exact ints in Real.
    bool const psCaustics = userOptions.psCaustics;
    // PSCausticClass::CausticMask, not unsigned char: the A4 indicator lives in bit 8, so a byte
    // accumulator would drop it here without any compiler diagnostic.
    std::vector<PSCausticClass::CausticMask> orientBits;
    if ( psCaustics )
        orientBits.assign(totalGrid, 0);

    // --ps-halo-release: threshold D on the geometric stream density rho_geo/rho_bar =
    // V_lag/V_eul = |det Lag| / |det Ax| (the 1/NO_DIM! factors cancel). Tets beyond D sit
    // deep inside virialized, mixed regions where fold counts explode and the sheet estimate
    // is unconverged anyway (Stuecker et al. 2021): they take the mass-conserving centroid
    // fallback instead of the bbox rasterization. Classified in double in BOTH deposits
    // (partition protocol: CPU and GPU must keep/release identical tets). 0 = off.
    double const psHaloRelease = double(userOptions.psHaloRelease);
    size_t nReleased = 0;

    // --ps-exact-deposit: analytic per-cell intersection moments instead of nSub^3 sampling.
    // The exact deposit is the nSub->infinity limit, so nSub is ignored. '.streams' carries
    // the volume-weighted multiplicity (see exactMult above): a FLOAT, so the partition merge
    // is NOT bit-exact -- the per-tet volume shares sum in thread order. The raw integer
    // tet-touch count, which does merge bit-exactly, is exported separately as '.tetTouch'.
    bool const psExact = userOptions.psExactDeposit;

    // Count of cells dropped because their Eulerian simplex is non-invertible (see the check below).
    size_t nDegenerateInverse = 0;

    // multi-stream velocity/scalar = mass-weighted mean sum(rho_s f_s)/sum(rho_s): accumulate
    // density-weighted moments + mass weight sum(rho_s). Serial path normalizes at function end;
    // psDeferNormalization leaves moments un-normalized so they sum linearly across partitions
    // (the mean is non-linear), normalizing once later.
    bool const deferNorm = userOptions.psDeferNormalization;
    bool const needWeight = field.velocity || field.velocity_gradient || field.velocity_dispersion
#ifdef SCALAR
                            || field.scalar || field.scalar_gradient
#endif
                            ;
    // --ps-volume-weighted: the velocity/scalar moments (and their normalizer) carry equal
    // EULERIAN-VOLUME shares instead of mass shares -> the normalized cell values are volume
    // averages (the standard-DTFE '_a' convention) instead of mass-weighted means. Density,
    // stream count and caustics always keep the mass deposit.
    bool const psVolWeighted = userOptions.psVolumeWeighted;
    // when the density field is selected, its accumulator receives exactly the same per-cell
    // mass sums as massWeight would (both add massShare at the same sites), so the weight
    // ALIASES the density grid instead of allocating a second full grid (4 B/cell). The
    // deferNorm caller reconstructs the weight from the summed density (normalizePhaseSpace
    // scale argument), since density is only converted to rho/rho_bar after weighting here.
    // Volume weighting breaks the aliasing (the moment normalizer is no longer the mass sum),
    // so it always allocates the explicit weight grid.
    bool const weightIsDensity = needWeight && field.density && !psVolWeighted;
    std::vector<Real> massWeight;
    if ( needWeight && !weightIsDensity ) massWeight.assign(totalGrid, Real(0.));
    // --ps-volume-weighted keeps the DISPERSION mass-weighted (sigma_ij is a moment of the
    // phase-space distribution f -- f-weighted by definition) while velocity/gradient go
    // volume-weighted. sigma_ij = <v_i v_j>_m - <v_i>_m<v_j>_m then needs its own mass-weighted
    // mean and normalizer, because 'velocity' now carries the volume-weighted one.
    bool const dispNeedsOwnWeight = psVolWeighted && field.velocity_dispersion;
    std::vector<Real> dispWeight;                      // sum(m_s)
    std::vector< Pvector<Real,noVelComp> > dispVel;    // sum(m_s v_s)
    if ( dispNeedsOwnWeight )
    {
        dispWeight.assign(totalGrid, Real(0.));
        dispVel.assign(totalGrid, Pvector<Real,noVelComp>::zero());
    }

    // Reusable per-tetrahedron scratch for the mass-conserving deposit (see the cell loop): the flat
    // grid index and Eulerian offset (rel = samplePos - vertex0) of every in-simplex sample point.
    std::vector<size_t> insideFlat;
    std::vector<Real>   insideRel;   // insideFlat.size() * NO_DIM, row-major per sample
    // --ps-linear-deposit: per-sample weight = the DTFE-interpolated linear density at the sample
    // (clamped >= 0); PASS 2 renormalizes the weights to the tet mass, so the deposited total
    // still equals tetMass exactly and mass conservation is untouched.
    bool const psLinear = userOptions.psLinearDeposit;
    std::vector<Real>   insideW;
    // --ps-exact-deposit per-tet scratch: 10 order-2 moments and the mass weight per hit cell
    std::vector<double> exMom, exW;

    if ( not userOptions.psSuppressGridStats )
        message << "\n" << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset()
                << " Interpolating fields to grid by iterating over all Delaunay cells ...\n" << MESSAGE::Flush;

    // GPU deposit (--ps-gpu, GPU build): extract flat per-tet arrays with EXACTLY the CPU
    // loop's filters, dispatch metal/ps_deposit.metal::depositFields (validated to mirror the CPU
    // scatter incl. the partition sub-grid), and copy the moment grids back. Any failure (no
    // device, kernel compile, buffer alloc) falls back to the CPU loop below with a warning.
    bool metalDeposited = false;
    // Finite-cell ordinals (ascending) of the tets the GPU DEFERRED: a sample within float rounding
    // of a face, which only the exact test can place. The CPU loop below deposits exactly these.
    std::vector<uint32_t> deferredOrd;
#if defined(PS_GPU) && NO_DIM==3
    bool tryMetal = userOptions.psUseMetal;
#ifdef SCALAR
    if ( tryMetal && (field.scalar || field.scalar_gradient) && noScalarComp != 1 )
    {
        MESSAGE::Warning warning( userOptions.verboseLevel );
        warning << "--ps-gpu supports ONE scalar component (this build has NO_SCALARS=" << noScalarComp << "); using the CPU deposit for this pass.\n" << MESSAGE::EndWarning;
        tryMetal = false;
    }
#endif
    // The kernels index cells with a 32-bit 'uint flat' (metal/ps_deposit.metal).
    // Above 2^32 cells it wraps -- and because the buffers are LARGER than 2^32, the wrapped index
    // is still in bounds: no crash, no OOB, just atomics landing in the wrong cells. Fall back to
    // the CPU instead, mirroring the standard-DTFE guard in averaged_interpolation_1.cc. This is
    // checked on subTotal, the only quantity that knows about the wrap-to-full-axis case above:
    // the cap is on the PARTITION's sub-box, so a finer --partition can bring a big grid back
    // under it. Uniform across field selections -- the moment offsets are already 64-bit.
    if ( tryMetal and noTotalCells > size_t(UINT32_MAX) )
    {   // the deferred tets are mapped back to their cells through 32-bit finite-cell ordinals
        MESSAGE::Warning warning( userOptions.verboseLevel );
        warning << "--ps-gpu does not support more than 2^32 Delaunay cells per triangulation (this one has "
                << noTotalCells << "); using the CPU deposit for this pass. A finer --partition splits them.\n"
                << MESSAGE::EndWarning;
        tryMetal = false;
    }
    if ( tryMetal and subTotal > size_t(UINT32_MAX) )
    {
        MESSAGE::Warning warning( userOptions.verboseLevel );
        warning << "--ps-gpu does not support more than 2^32 cells per partition sub-grid (this one has "
                << subTotal << ", i.e. > " << size_t(UINT32_MAX) << "); using the CPU deposit for this pass. "
                << "A finer --partition would bring the sub-grid back under the cap.\n" << MESSAGE::EndWarning;
        tryMetal = false;
    }
    if ( tryMetal )
    {
        // which moment grids this run needs: unselected ones are neither extracted, allocated
        // (host or GPU; m2 is 24 B/cell, grad 36 B/cell) nor copied back
        bool const fVel  = haveVel;
        bool const fDisp = field.velocity_dispersion;
        bool const fGrad = field.velocity_gradient;
        bool const needsVelArrays = fVel || fDisp || fGrad;
        // the scalar field (one component): the kernel deposits it like the velocity moment
        bool fScal = false, fSGrad = false;
#ifdef SCALAR
        fScal  = field.scalar;
        fSGrad = field.scalar_gradient;
#endif
        bool const needsScalArrays = fScal || fSGrad;

        // reserve for the cells that survive the filters (counted by the subgrid pass), not all
        // finite cells -- the padded-partition majority is filtered out (~4x over-allocation)
        std::vector<float> tetVerts, tetVels, tetMasses, tetDens, tetScal;
        tetVerts.reserve( psKeptCells*PS_TET_STRIDE );
        if ( needsVelArrays )
            tetVels.reserve( psKeptCells*12 );
        if ( needsScalArrays )
            tetScal.reserve( psKeptCells*4 );
        tetMasses.reserve( psKeptCells );
        if ( psLinear or psCaustics )
            tetDens.reserve( psKeptCells*4 );
        // --ps-halo-release: released tets never reach the GPU arrays; their single-cell
        // centroid deposits (same arithmetic as the CPU fallback) are collected here and
        // applied on the host AFTER the GPU grids are copied back (the copy ASSIGNS, so
        // these are the only further accumulations). Discarded if the GPU dispatch fails --
        // the CPU loop then re-handles every cell. momw = the moment weight (V_eul under
        // --ps-volume-weighted, else the mass); orient = the caustic orientation bits.
        struct ReleasedTet { size_t flat; Real mass; Real momw; Real volFrac; PSCausticClass::CausticMask orient; Real vel[noVelComp]; Real grad[noGradComp]; Real scal; Real sgrad[NO_DIM]; };
        std::vector<ReleasedTet> releasedTets;
        // per GPU tet, its finite-cell ordinal (kept in lockstep through the cost sort): maps the
        // kernel's deferred tets back to the cells the CPU loop must deposit
        std::vector<uint32_t> tetOrd;
        tetOrd.reserve( psKeptCells );
        uint32_t cellOrdCounter = 0;
        // grid and canonical frame of the per-tet records (the CPU exact test's own definitions)
        double  recLo[3], recLen[3], recShift[3], recWrap[3];
        int32_t recN[3];
        for (int d = 0; d < 3; ++d)
        {
            recLo[d]    = double( boxCoordinates[2*d] );
            recLen[d]   = gridLenD[d];
            recN[d]     = int32_t( nGrid[d] );
            recShift[d] = exactFrame.periodic ? exactFrame.canonLd[d] : 0.;
            recWrap[d]  = exactFrame.periodic ? exactFrame.canonLd[d] : 0.;
        }
        PS_FOREACH_FINITE_CELL
        {
            uint32_t const cellOrd = cellOrdCounter++;   // counts EVERY finite cell, like the CPU loop
            Cell_handle cell = itC;
            PSCellGeometry geo;   // the CPU loop's filters, including the zero-inverse check
            // --ps-window: a tet that misses the window is dropped on its Eulerian positions, before the
            // filter's determinant and inverse -- the same test this loop applied after the filter, so the
            // extracted set is unchanged (a TNG100-3 partition walks 36M tets for a few thousand inside)
            if ( !psFilterCell(cell, userOptions, boxCoordinates, true, &nDegenerateInverse, geo, false,
                               [&](Real const (&ep)[NO_DIM+1][NO_DIM]) { return windowMode && !windowTouched(ep); }) ) continue;
            markFlipped(geo, flippedFlag);     // '.hidden_streams' bit 1: geometric, so the host does it for the GPU path
            bool const hadBadVertex = geo.useVolumeRatioDensity;
            Real (&ep)[NO_DIM+1][NO_DIM] = geo.eulerPos;
            double Lag[NO_DIM][NO_DIM];
            for (int v=0;v<NO_DIM;++v) for (int i=0;i<NO_DIM;++i) Lag[v][i]=double(cell->vertex(v+1)->point()[i])-double(cell->vertex(0)->point()[i]);
            // absDetLag always computed: the --ps-halo-release classification below stays
            // geometric (V_lag/V_eul) so kept/released sets are identical under either mass
            double const absDetLag = std::fabs(determinant(Lag));
            PSCausticClass::CausticMask const tetBits = psCaustics
                ? (PSCausticClass::CausticMask)( (geo.cellDet > 0. ? PSCausticClass::BIT_PARITY_POS
                                                                   : PSCausticClass::BIT_PARITY_NEG)
                                   | PSCausticClass::classifyTet( geo.Ax, Lag ) )
                : (PSCausticClass::CausticMask)0;
            if ( degreesLazy ) lazyDegrees(cell);   // --ps-window: this tet's vertices only
            float tm = psVertexMass ? float( psVertexShareMass(cell) )
                                    : float( double(userOptions.averageDensity)*absDetLag/factorial(NO_DIM) );
            if (tm<=0.f) continue;
            if ( psHaloRelease > 0. && absDetLag > psHaloRelease * geo.cellAbsDet )
            {
                // identical classification to the CPU loop (double |det Lag| > D*|det Ax|);
                // centroid cell + centroid-evaluated velocity mirror the CPU fallback exactly
                Real centroid[NO_DIM];
                size_t f = 0;
                if ( !centroidCell(ep, centroid, f) ) { continue; }   // outside this grid/partition region: drop (as the CPU fallback does)
                ReleasedTet rt;
                rt.flat = f;
                rt.mass = Real(tm);
                rt.momw = psVolWeighted ? Real( geo.cellAbsDet / factorial(NO_DIM) ) : rt.mass;
                rt.volFrac = Real( geo.cellAbsDet / factorial(NO_DIM) / double(cellVolume) );
                rt.orient = psCaustics ? tetBits : (PSCausticClass::CausticMask)(geo.cellDet > 0. ? 1 : 2);
                if ( needsVelArrays )
                {
                    Real velGrad[NO_DIM][noVelComp], temp[NO_DIM][noVelComp];
                    for (int v = 0; v < NO_DIM; ++v)
                        for (size_t j = 0; j < noVelComp; ++j)
                            temp[v][j] = cell->vertex(v+1)->info().velocity(j) - cell->vertex(0)->info().velocity(j);
                    matrixMultiplication<noVelComp>(geo.posMatInv, temp, velGrad);
                    for (size_t j = 0; j < noVelComp; ++j)
                    {
                        rt.vel[j] = cell->vertex(0)->info().velocity(j);
                        for (int i = 0; i < NO_DIM; ++i)
                            rt.vel[j] += velGrad[i][j] * (centroid[i] - ep[0][i]);
                    }
                    if ( fGrad )
                        for (size_t j = 0; j < noVelComp; ++j)
                            for (int i = 0; i < NO_DIM; ++i)
                                rt.grad[j*NO_DIM+i] = velGrad[i][j];
                }
                rt.scal = Real(0.);
                for (int i = 0; i < NO_DIM; ++i) rt.sgrad[i] = Real(0.);
#ifdef SCALAR
                if ( needsScalArrays )
                {   // the linear scalar at the centroid and its constant gradient, as the CPU fallback deposits them
                    Real sg[NO_DIM][noScalarComp], temp[NO_DIM][noScalarComp];
                    for (int v = 0; v < NO_DIM; ++v)
                        for (size_t j = 0; j < noScalarComp; ++j)
                            temp[v][j] = cell->vertex(v+1)->info().myScalar()[j] - cell->vertex(0)->info().myScalar()[j];
                    matrixMultiplication<noScalarComp>(geo.posMatInv, temp, sg);
                    rt.scal = cell->vertex(0)->info().myScalar()[0];
                    for (int i = 0; i < NO_DIM; ++i)
                    {
                        rt.scal += sg[i][0] * (centroid[i] - ep[0][i]);
                        rt.sgrad[i] = sg[i][0];
                    }
                }
#endif
                releasedTets.push_back(rt);
                ++nReleased;
                continue;
            }
            {
                // the record (ps_deposit_params.h): canonical vertices -> vertex-0 cell + fraction and
                // edges, plus the centroid-fallback cell exactly as the CPU fallback picks it
                float cv[NO_DIM+1][NO_DIM];
                for (int v = 0; v <= NO_DIM; ++v)
                    for (int d = 0; d < NO_DIM; ++d)
                        cv[v][d] = float( psExactInside::canonicalCoord( exactFrame, cell->vertex(v)->info().eulerianPosition(d), d ) );
                Real centroid[NO_DIM];
                size_t cflat = 0;
                uint32_t const cf = centroidCell(ep, centroid, cflat) ? uint32_t(cflat) : 0xFFFFFFFFu;
                size_t const at = tetVerts.size();
                tetVerts.resize( at + PS_TET_STRIDE );
                psFillTetRecord( &tetVerts[at], cv, recLo, recLen, recN, recShift, recWrap, cf );
            }
            if (needsVelArrays)
                for (int v=0;v<=NO_DIM;++v) for (size_t j=0;j<noVelComp;++j) tetVels.push_back(float(cell->vertex(v)->info().velocity(j)));
#ifdef SCALAR
            if (needsScalArrays)
                for (int v=0;v<=NO_DIM;++v) tetScal.push_back(float(cell->vertex(v)->info().myScalar()[0]));
#endif
            if (psLinear)
            {
                // hull cells (non-periodic volume-ratio density) have a constant profile:
                // pass equal weights, which reduces to the uniform deposit for that tet
                if (hadBadVertex)
                    for (int v=0;v<=NO_DIM;++v) tetDens.push_back(1.f);
                else
                    for (int v=0;v<=NO_DIM;++v) tetDens.push_back(float(cell->vertex(v)->info().density()));
            }
            tetMasses.push_back(tm);
            tetOrd.push_back(cellOrd);
            // --ps-caustics without --ps-linear-deposit: the otherwise-unused 'dens' slots carry the
            // caustic mask (slot 0) so the kernel gets the full stratification, not just the parity.
            if ( psCaustics and not psLinear )
            {
                tetDens.push_back( float(tetBits) );
                for (int v=1; v<=NO_DIM; ++v) tetDens.push_back( 0.f );
            }

        }

        // Sort tetrahedra by grid-footprint cost before dispatch. A GPU chunk finishes when its
        // SLOWEST thread finishes: one stretched void tet (spanning thousands of grid cells) mixed
        // among cheap halo tets leaves most of the GPU idle until the chunk boundary -- measured as
        // a ~5x underutilization on real data with watchdog-safe short chunks. Cost-ordering makes
        // every chunk's threads uniformly sized, so all GPU cores stay busy, and the adaptive
        // chunk-size controller tracks a smooth monotone cost curve instead of a noisy mix.
        {
            size_t const nT = tetMasses.size();
            // Counting sort on the log2 cost, quantized to 2048 steps per doubling (0.03%): linear
            // time, and balancing the chunks needs no finer order. (std::sort of (cost, index)
            // pairs took ~1.2 s per 15M tets; this also needs 6 B/tet instead of 8.)
            std::vector<uint16_t> costKey(nT);
            for (size_t t = 0; t < nT; ++t)
            {
                float const* rec = &tetVerts[t*PS_TET_STRIDE];   // edges at [6..14]
                float cells = 1.f;
                for (int d = 0; d < 3; ++d)
                {
                    float lo = 0.f, hi = 0.f;
                    for (int e = 0; e < 3; ++e)
                    {
                        float const cc = rec[6 + e*3 + d];
                        if (cc < lo) lo = cc;
                        if (cc > hi) hi = cc;
                    }
                    cells *= (hi-lo) / float(dx[d]) + (nSub > 1 ? 3.f : 1.f);
                }
                // cells >= 1 (every factor is), so the key is >= 0; 30 doublings fill 61440 < 65536
                costKey[t] = uint16_t( std::min( 65535.f, std::log2(cells) * 2048.f ) );
            }
            std::vector<uint32_t> order(nT);   // order[i] = the tet that goes to position i
            {
                std::vector<size_t> slot(size_t(65536) + 1, 0);
                for (size_t t = 0; t < nT; ++t) ++slot[size_t(costKey[t]) + 1];
                for (size_t k = 0; k < size_t(65536); ++k) slot[k+1] += slot[k];
                for (size_t t = 0; t < nT; ++t) order[ slot[costKey[t]]++ ] = uint32_t(t);
            }
            std::vector<uint16_t>().swap(costKey);
            // apply the permutation IN PLACE by cycle-walking all arrays in lockstep: a gather
            // into a fresh array would transiently hold a second nT*12-float copy (~90 MB per
            // million tets); this needs only one element of scratch plus an nT-byte mask
            {
                std::vector<char> visited(nT, 0);
                bool const haveU = not tetVels.empty();
                bool const haveD = not tetDens.empty();
                bool const haveS = not tetScal.empty();
                float tmpV[PS_TET_STRIDE], tmpU[12], tmpD[4], tmpS[4];
                for (size_t start = 0; start < nT; ++start)
                {
                    if (visited[start]) continue;
                    std::memcpy(tmpV, &tetVerts[start*PS_TET_STRIDE], PS_TET_STRIDE*sizeof(float));
                    if (haveU) std::memcpy(tmpU, &tetVels[start*12], 12*sizeof(float));
                    if (haveD) std::memcpy(tmpD, &tetDens[start*4], 4*sizeof(float));
                    if (haveS) std::memcpy(tmpS, &tetScal[start*4], 4*sizeof(float));
                    float const tmpM = tetMasses[start];
                    uint32_t const tmpO = tetOrd[start];
                    size_t i = start;
                    while (true)
                    {
                        size_t const src = order[i];
                        visited[i] = 1;
                        if (src == start)
                        {
                            std::memcpy(&tetVerts[i*PS_TET_STRIDE], tmpV, PS_TET_STRIDE*sizeof(float));
                            if (haveU) std::memcpy(&tetVels[i*12], tmpU, 12*sizeof(float));
                            if (haveD) std::memcpy(&tetDens[i*4], tmpD, 4*sizeof(float));
                            if (haveS) std::memcpy(&tetScal[i*4], tmpS, 4*sizeof(float));
                            tetMasses[i] = tmpM;
                            tetOrd[i] = tmpO;
                            break;
                        }
                        std::memcpy(&tetVerts[i*PS_TET_STRIDE], &tetVerts[src*PS_TET_STRIDE], PS_TET_STRIDE*sizeof(float));
                        if (haveU) std::memcpy(&tetVels[i*12], &tetVels[src*12], 12*sizeof(float));
                        if (haveD) std::memcpy(&tetDens[i*4], &tetDens[src*4], 4*sizeof(float));
                        if (haveS) std::memcpy(&tetScal[i*4], &tetScal[src*4], 4*sizeof(float));
                        tetMasses[i] = tetMasses[src];
                        tetOrd[i] = tetOrd[src];
                        i = src;
                    }
                }
            }
        }

        double bl[3]  = { double(boxCoordinates[0]), double(boxCoordinates[2]), double(boxCoordinates[4]) };
        double dxv[3] = { double(dx[0]), double(dx[1]), double(dx[2]) };
        size_t const nTetsDeposited = tetMasses.size();   // psGpuDepositFields consumes (frees) the arrays
        PSGpuGrids gpuOut;
        std::vector<uint32_t> deferredTets;
        std::string metalErr;
        if ( psGpuDepositFields(tetVerts, tetVels, tetMasses, tetDens, tetScal, bl, dxv,
                                  nGrid, subOrigin, subDims, nSub,
                                  userOptions.periodic, fVel, fDisp, fGrad, psLinear,
                                  psVolWeighted, psCaustics, psExact, fScal, fSGrad, windowMode, gpuOut, deferredTets, metalErr) )
        {
            // the deferred tets' cells, ascending: the CPU loop below deposits exactly these
            deferredOrd.reserve( deferredTets.size() );
            for (uint32_t const t : deferredTets) deferredOrd.push_back( tetOrd[t] );
            std::sort( deferredOrd.begin(), deferredOrd.end() );
            if ( not userOptions.psSuppressGridStats )
                message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " " << gpuBackendName() << " GPU deposit ("
                        << MESSAGE::cMagenta() << gpuDeviceName() << MESSAGE::cReset() << "), "
                        << MESSAGE::cMagenta() << nTetsDeposited << MESSAGE::cReset() << " tetrahedra ("
                        << MESSAGE::cMagenta() << deferredOrd.size() << MESSAGE::cReset()
                        << " with a sample on a face, deposited by the CPU's exact test).\n" << MESSAGE::Flush;
            for (size_t i = 0; i < totalGrid; ++i)
            {
                if (field.density) quantities->density[i] = Real(gpuOut.mass[i]);
                // moment normalizer: the volume-share sum under --ps-volume-weighted, else the mass
                if (needWeight && !weightIsDensity)
                    massWeight[i] = Real( psVolWeighted ? gpuOut.momw[i] : gpuOut.mass[i] );
                if (haveVel)
                    for (size_t j = 0; j < noVelComp; ++j)
                        quantities->velocity[i][j] = Real(gpuOut.mom[i*3+j]);
                if (field.velocity_gradient)
                    for (size_t q = 0; q < noGradComp; ++q)
                        quantities->velocity_gradient[i][q] = Real(gpuOut.grad[i*9+q]);
                if (field.velocity_dispersion)
                    for (size_t cc = 0; cc < noDispComp; ++cc)
                        quantities->velocity_dispersion[i][cc] = Real(gpuOut.m2[i*6+cc]);
#ifdef SCALAR
                if (fScal)  quantities->scalar[i][0] = Real(gpuOut.scal[i]);
                if (fSGrad) for (int q = 0; q < NO_DIM; ++q) quantities->scalar_gradient[i][q] = Real(gpuOut.sgrad[i*3+q]);
#endif
                if (dispNeedsOwnWeight)
                {   // the dispersion's own mass-weighted mean/normalizer (kernel buffers 13/14)
                    dispWeight[i] = Real(gpuOut.dispw[i]);
                    for (size_t j = 0; j < noVelComp; ++j)
                        dispVel[i][j] = Real(gpuOut.dispvel[i*3+j]);
                }
                streamCount[i] = int(gpuOut.streams[i] & 0x7FFFFFFFu);   // bit 31 = the sub-sample mass flag ('.hidden_streams' bit 2)
                if ( gpuOut.streams[i] >> 31 ) subSampleMassFlag[i] = 1;
                if (psExact)
                    exactMult[i] = Real(gpuOut.streamvol[i]);   // exact cell-mean multiplicity
                // The kernel accumulates the two PARITY bits only, so a GPU deposit reproduces the
                // fold flag EXACTLY (asserted by ps_caustic_class_check.sh) but not the
                // stratification bits -- see the warning below.
                // With the mask riding in 'dens' the kernel ORs the host-built bits verbatim, so the
                // full stratification survives; under --ps-linear-deposit only the parity does.
                if (psCaustics)
                    orientBits[i] = (PSCausticClass::CausticMask)
                        ( gpuOut.caustic[i]
                          & (psLinear ? (unsigned)(PSCausticClass::BIT_PARITY_POS
                                                   | PSCausticClass::BIT_PARITY_NEG)
                                      : (unsigned)PSCausticClass::BITS_GPU_MAX) );
            }
            if (psCaustics and psLinear)
            {
                MESSAGE::Warning warning( userOptions.verboseLevel );
                warning << "--ps-caustics with --ps-linear-deposit on the GPU produces the fold flag "
                           "('.caustic') only: the per-tet caustic mask travels in the 'dens' buffer, "
                           "which --ps-linear-deposit needs for the vertex densities. The stratification "
                           "('.causticClass') comes from the CPU deposit -- drop '--ps-gpu' for it.\n"
                        << MESSAGE::EndWarning;
            }
            // --ps-halo-release: host-side monolithic centroid deposits of the released tets
            // (same accumulation sites and moments as the CPU fallback's single sample)
            for (ReleasedTet const &rt : releasedTets)
            {
                if (field.density) quantities->density[rt.flat] += rt.mass;
                if (needWeight && !weightIsDensity) massWeight[rt.flat] += rt.momw;
                if (haveVel)
                    for (size_t j = 0; j < noVelComp; ++j)
                        quantities->velocity[rt.flat][j] += rt.vel[j] * rt.momw;
                if (field.velocity_gradient)
                    for (size_t q = 0; q < noGradComp; ++q)
                        quantities->velocity_gradient[rt.flat][q] += rt.grad[q] * rt.momw;
#ifdef SCALAR
                if (fScal)  quantities->scalar[rt.flat][0] += rt.scal * rt.momw;
                if (fSGrad) for (int q = 0; q < NO_DIM; ++q) quantities->scalar_gradient[rt.flat][q] += rt.sgrad[q] * rt.momw;
#endif
                if (field.velocity_dispersion)
                {
                    size_t c = 0;
                    for (int a = 0; a < NO_DIM; ++a)
                        for (int b = a; b < NO_DIM; ++b)
                            quantities->velocity_dispersion[rt.flat][c++] += rt.mass * rt.vel[a] * rt.vel[b];
                    if ( dispNeedsOwnWeight )
                    {
                        dispWeight[rt.flat] += rt.mass;
                        for (size_t j = 0; j < noVelComp; ++j)
                            dispVel[rt.flat][j] += rt.vel[j] * rt.mass;
                    }
                }
                subSampleMassFlag[rt.flat] = 1;            // mass from a never-sampled tet
                if (psExact)
                {
                    streamCount[rt.flat]++;             // raw tet-touch count ('.tetTouch')
                    exactMult[rt.flat] += rt.volFrac;   // V_tet / V_cell, as the CPU fallback deposits
                }
                else
                    releasedMult[rt.flat] += rt.volFrac;   // never sampled: its volume fraction
                if (psCaustics)
                    orientBits[rt.flat] |= rt.orient;
            }
            metalDeposited = true;
        }
        else
        {
            MESSAGE::Warning warning( userOptions.verboseLevel );
            warning << "PS-DTFE GPU deposit unavailable (" << metalErr << "); using the CPU deposit.\n" << MESSAGE::EndWarning;
            nReleased = 0;   // releasedTets are discarded; the CPU loop below recounts them
        }
    }
#endif // PS_GPU
    if ( degreesLazy && metalDeposited && std::getenv("PS_DEBUG_DEGREES") )
    {   // self-check of the lazy degrees: every vertex the extraction computed must carry the full pass's count
        std::unordered_map<void const*, int32_t> full;
        size_t degDropped = 0;
        PS_FOREACH_FINITE_CELL
        {
            Cell_handle cell = itC;
            PSCellGeometry dgeo;
            if ( !psFilterCell(cell, userOptions, boxCoordinates, true, &degDropped, dgeo, true) ) continue;
            for (int v = 0; v <= NO_DIM; ++v) full[ &*cell->vertex(v) ] += 1;
        }
        size_t checked = 0, bad = 0;
        for (DT::Finite_vertices_iterator vit = dt.finite_vertices_begin(); vit != dt.finite_vertices_end(); ++vit)
        {
            int32_t const lazy = vit->info().psDegree();
            if ( lazy < 0 ) continue;
            ++checked;
            auto const it = full.find( &*vit );
            if ( lazy != ( it == full.end() ? 0 : it->second ) ) ++bad;
        }
        std::fprintf( stderr, "PS_DEBUG_DEGREES: %zu lazy vertex degrees checked against the full pass, %zu differ\n", checked, bad );
        if ( bad ) throwError( "PS_DEBUG_DEGREES: the lazy vertex degrees differ from the full pass." );
    }
    if ( degreesLazy && not metalDeposited )
    {   // the CPU loop deposits every cell: it needs every degree
        fullDegreePass();
        degreesLazy = false;
    }
#if defined(PS_GPU) && NO_DIM==3
#else
    if ( userOptions.psUseMetal )
    {
        static bool warned = false;
        if ( !warned )
        {
            warned = true;
            MESSAGE::Warning warning( userOptions.verboseLevel );
            warning << "--ps-gpu requested but this binary was built without GPU support (rebuild with METAL=1 on a Mac, CUDA=1 or HIP=1 on Linux; or NO_DIM!=3); using the CPU deposit.\n" << MESSAGE::EndWarning;
        }
    }
#endif // PS_GPU

    // The per-cell scatter. The CPU deposit runs on several threads (depositThreads, capped by the
    // memory its private accumulators may take): thread t takes every T-th block of 256 cells and
    // accumulates into its own copies, which are then summed in thread order -- deterministic for
    // a given thread count (DTFE_DEPOSIT_THREADS pins it). Thread 0 writes the real arrays. Serial
    // before: a small box on a fine grid (0.26M particles, 256^3) took 145 s where the GPU took 25.
    // GPU-deferred cells and --ps-caustic-cusps (CGAL's incident_cells marks visited cells, so it is
    // not re-entrant) stay on one thread. After a GPU deposit only the cells it deferred run here:
    // deferredOrd holds their finite-cell ordinals in the GPU extraction loop's order.
    struct DepositThread
    {
        Quantities *q = nullptr;
        std::vector<int> *streamCount = nullptr;
        std::vector<Real> *releasedMult = nullptr, *exactMult = nullptr, *dispWeight = nullptr, *massWeight = nullptr;
        std::vector< Pvector<Real,noVelComp> > *dispVel = nullptr;
        std::vector<PSCausticClass::CausticMask> *orientBits = nullptr;
        std::vector<unsigned char> *subSampleMassFlag = nullptr, *flippedFlag = nullptr;
        // the private copies of threads > 0 (thread 0 points at the real arrays)
        Quantities ownQ;
        std::vector<int> ownStreamCount;
        std::vector<Real> ownReleasedMult, ownExactMult, ownDispWeight, ownMassWeight;
        std::vector< Pvector<Real,noVelComp> > ownDispVel;
        std::vector<PSCausticClass::CausticMask> ownOrientBits;
        std::vector<unsigned char> ownSubSampleMassFlag, ownFlippedFlag;
        // per-tetrahedron scratch, reused
        std::vector<size_t> insideFlat;
        std::vector<Real>   insideRel, insideW;
        std::vector<double> exMom, exW;
        size_t nDegenerateInverse = 0, nReleased = 0;
    };
    // bytes of one private copy of every accumulator this run uses
    auto vecBytes = [](auto const &v) -> double { return double(v.size()) * double(sizeof(v[0])); };
    double accBytes = vecBytes(streamCount) + vecBytes(releasedMult) + vecBytes(exactMult) + vecBytes(dispWeight)
                    + vecBytes(massWeight)
                    + vecBytes(dispVel) + vecBytes(orientBits) + vecBytes(subSampleMassFlag) + vecBytes(flippedFlag)
                    + vecBytes(quantities->density) + vecBytes(quantities->velocity)
                    + vecBytes(quantities->velocity_gradient) + vecBytes(quantities->velocity_dispersion);
#ifdef SCALAR
    accBytes += vecBytes(quantities->scalar) + vecBytes(quantities->scalar_gradient);
#endif
    int nDepThreads = 1;
    // SLAB MODE. The private copies are what a fine grid cannot afford: on 1024^3 one copy of the
    // density+velocity accumulators is 24 GB, so the copy budget left ONE thread (245 s where four
    // would take ~70). When the copies would cut the threads below what the cores allow, each thread
    // instead OWNS a slab of local x planes and writes only those cells, straight into the real
    // arrays. A thread takes a whole slab at a time and walks its tetrahedra in the loop's own order,
    // so every cell receives its contributions in exactly the order the single-threaded loop gives
    // them: the result is bit-identical to DTFE_DEPOSIT_THREADS=1, at ANY thread count. The price is
    // that a tetrahedron straddling k slabs is classified k times (its normalization needs all its
    // samples), so slabs are kept several tetrahedron extents wide. DTFE_DEPOSIT_SLABS=1 forces the
    // mode (the tests: on small grids the copies always fit), 0 forbids it.
    bool slabMode = false;
    int nSlabs = 0;
    std::vector<long> slabXLo, slabXHi;                     // each slab's local x range [lo, hi)
    std::vector< std::vector<uint32_t> > slabTetLists;      // per slab: its tetrahedra (ordinals, loop order)
    std::vector<Cell_handle> slabTets;                      // ordinal -> cell
    std::vector<uint8_t> slabFirst;                         // ordinal -> its first slab (per-tet tallies)
    if ( !metalDeposited && !userOptions.psCausticCusps )
    {
        int const wantThreads = psDepositThreads();
        nDepThreads = wantThreads;
        double const extra = psDepositThreadBytes();
        if ( accBytes > 0. )
            nDepThreads = std::max( 1, std::min( nDepThreads, 1 + int( extra / accBytes ) ) );
        size_t const blocks = noTotalCells / 256 + 1;                      // at least a few blocks per thread (DEP_BLOCK)
        int const blockCap = std::max( 1, int( blocks / 4 ) + 1 );
        nDepThreads = std::max( 1, std::min( nDepThreads, blockCap ) );
        int const slabThreads = std::max( 1, std::min( wantThreads, blockCap ) );
        char const *slabEnv = std::getenv("DTFE_DEPOSIT_SLABS");
        int const slabSetting = slabEnv ? std::atoi(slabEnv) : -1;     // 1 force, 0 never, else automatic
        // automatic: only when the copies leave fewer than HALF the threads -- a slab run classifies a
        // straddling tet once per slab and filters every tet once more to bin it (~1.6x the work): on a
        // clustered TNG sub-cube at 512^3 (2 cells per tet, copies for 5 of 10 threads) 10 slab threads
        // took 46.8 s, the 5 copy threads 42.5 s; on 768^3 (copies for 2) slabs took 28 s instead of 64
        if ( slabSetting != 0 and slabThreads > 1 and subDims[0] >= 2
             and ( slabSetting == 1 or 2 * nDepThreads < slabThreads ) and noTotalCells < size_t(UINT32_MAX) )
        {
            slabMode = true;
            nDepThreads = slabThreads;
        }
    }
    if ( slabMode )
    {
        // binning: each tetrahedron's x range of global cells -- the bounding box of its Eulerian
        // vertices, minimum-image wrapped EXACTLY as psFilterCell wraps them (same Real operations, so the
        // same positions), widened by two cells: a superset of every cell the deposit writes (its sample
        // window adds one, the centroid fallback lies inside the box). No filter here: running it once
        // more per tetrahedron cost a full extra pass (a third of the slab run on a clustered box, where
        // a tet covers ~2 cells); the deposit filters, and counts a rejection in the tet's first slab.
        slabTets.reserve( noTotalCells );
        for (auto it = PS_FINITE_CELLS_BEGIN; it != PS_FINITE_CELLS_END; ++it) slabTets.push_back( it );
        size_t const nTets = slabTets.size();
        std::vector<int32_t> gLo( nTets ), gHi( nTets );         // its global x cells [gLo, gHi)
        std::vector<float> cost( nTets, 0.f );                    // its work: bbox cells + a per-tet overhead
        std::vector<double> extentSum( size_t(nDepThreads), 0. );
        std::vector<size_t> extentN( size_t(nDepThreads), 0 );
        auto bin = [&](size_t th)
        {
            size_t const lo = nTets * th / size_t(nDepThreads), hi = nTets * (th + 1) / size_t(nDepThreads);
            for (size_t i = lo; i < hi; ++i)
            {
                Cell_handle cell = slabTets[i];
                Real ep[NO_DIM+1][NO_DIM];
                for (int v = 0; v <= NO_DIM; ++v)
                    for (int d = 0; d < NO_DIM; ++d)
                        ep[v][d] = cell->vertex(v)->info().eulerianPosition(d);
                if ( userOptions.periodic )
                    for (int v = 1; v <= NO_DIM; ++v)
                        for (int d = 0; d < NO_DIM; ++d)
                        {
                            Real const boxLen = boxCoordinates[2*d+1] - boxCoordinates[2*d];
                            Real const diff = ep[v][d] - ep[0][d];
                            if (diff >  boxLen * Real(0.5)) ep[v][d] -= boxLen;
                            if (diff < -boxLen * Real(0.5)) ep[v][d] += boxLen;
                        }
                double cells = 1.;
                for (int d = 0; d < NO_DIM; ++d)
                {
                    double eMin = double(ep[0][d]), eMax = eMin;
                    for (int v = 1; v <= NO_DIM; ++v)
                    {
                        eMin = std::min( eMin, double(ep[v][d]) );
                        eMax = std::max( eMax, double(ep[v][d]) );
                    }
                    double const x0 = double(boxCoordinates[2*d]), dxd = double(dx[d]);
                    double const lo = std::floor((eMin - x0) / dxd), hi = std::floor((eMax - x0) / dxd) + 1.;
                    cells *= hi - lo;
                    if ( d == 0 )
                    {
                        gLo[i] = int32_t( lo ) - 2;
                        gHi[i] = int32_t( hi ) + 2;
                        extentSum[th] += hi - lo;
                        extentN[th] += 1;
                    }
                }
                cost[i] = float( 2. + cells );
            }
        };
        {
            std::vector<std::thread> pool;
            for (int th = 1; th < nDepThreads; ++th) pool.emplace_back( bin, size_t(th) );
            bin( 0 );
            for (std::thread &x : pool) x.join();
        }
        double eSum = 0.; size_t eN = 0;
        for (int th = 0; th < nDepThreads; ++th)
        { eSum += extentSum[size_t(th)]; eN += extentN[size_t(th)]; }
        double const meanExtent = eN ? eSum / double(eN) : 1.;
        long const nx = long(subDims[0]);
        bool const periodicX = userOptions.periodic;
        long const nG = long(nGrid[0]), sO = long(subOrigin[0]);
        // a tetrahedron's global x cells, mapped to local planes exactly as the scatter maps them
        auto forEachPlane = [&](size_t i, auto const &f)
        {
            long a = gLo[i], b = gHi[i];
            if ( periodicX and b - a > nG ) { a = 0; b = nG; }       // spans the whole period
            for (long g = a; g < b; ++g)
            {
                long w = g;
                if ( periodicX ) w = ((g % nG) + nG) % nG;
                else if ( g < 0 || g >= nG ) continue;
                long loc = w - sO;
                if ( loc < 0 ) loc += nG;
                if ( loc < 0 || loc >= nx ) continue;
                f( loc );
            }
        };
        // The slab boundaries balance the WORK, not the width: halos put most tetrahedra into a few
        // planes of a clustered box, where equal-width slabs left the thread holding them alone (TNG
        // sub-cube: no faster than five copy threads). Each tet's cost (bbox cells + overhead) is spread
        // over its planes, the slabs cut at equal cumulative cost. Their number is a multiple of the
        // threads (k slabs each, picked up dynamically), k ~ the planes per thread over four mean
        // tetrahedron extents (a straddler then costs ~25% extra), at least 2, at most 64 slabs.
        std::vector<double> planeCost( size_t(nx), 0. );
        {
            std::vector< std::vector<double> > part( size_t(nDepThreads), std::vector<double>( size_t(nx), 0. ) );
            auto hist = [&](size_t th)
            {
                size_t const lo = nTets * th / size_t(nDepThreads), hi = nTets * (th + 1) / size_t(nDepThreads);
                std::vector<double> &h = part[th];
                for (size_t i = lo; i < hi; ++i)
                {
                    double const share = double(cost[i]) / double( std::max<long>( 1, gHi[i] - gLo[i] ) );
                    forEachPlane( i, [&](long loc) { h[size_t(loc)] += share; } );
                }
            };
            std::vector<std::thread> pool;
            for (int th = 1; th < nDepThreads; ++th) pool.emplace_back( hist, size_t(th) );
            hist( 0 );
            for (std::thread &x : pool) x.join();
            for (int th = 0; th < nDepThreads; ++th)        // summed in thread order: the same cuts every run
                for (long x = 0; x < nx; ++x) planeCost[size_t(x)] += part[size_t(th)][size_t(x)];
        }
        std::vector<float>().swap( cost );
        int const perThread = std::max( 2, std::min( 64 / std::max( 1, nDepThreads ),
                              int( std::lround( double(nx) / ( 4. * std::max( 1., meanExtent ) * double(nDepThreads) ) ) ) ) );
        nSlabs = int( std::min<long>( nx, std::min( 64L, long(perThread) * long(nDepThreads) ) ) );
        if ( const char *env = std::getenv("DTFE_DEPOSIT_NSLABS") )
            if ( std::atoi(env) > 0 ) nSlabs = std::min( int(nx), std::min( 64, std::atoi(env) ) );
        slabXLo.assign( size_t(nSlabs), 0 ); slabXHi.assign( size_t(nSlabs), 0 );
        std::vector<int> slabOfX( static_cast<size_t>(nx), 0 );
        {
            double total = 0.;
            for (long x = 0; x < nx; ++x) total += planeCost[size_t(x)];
            long x = 0;
            double acc = 0.;
            for (int sl = 0; sl < nSlabs; ++sl)
            {
                slabXLo[size_t(sl)] = x;
                // the last slab takes the rest; every slab keeps at least one plane, and leaves one for each after it
                long const maxHi = nx - long(nSlabs - 1 - sl);
                double const target = total * double(sl + 1) / double(nSlabs);
                long hi = x + 1;
                acc += planeCost[size_t(x)];
                while ( hi < maxHi and ( sl == nSlabs - 1 or acc + 0.5 * planeCost[size_t(hi)] <= target ) )
                { acc += planeCost[size_t(hi)]; ++hi; }
                if ( sl == nSlabs - 1 ) hi = nx;
                slabXHi[size_t(sl)] = hi;
                for (long y = x; y < hi; ++y) slabOfX[size_t(y)] = sl;
                x = hi;
            }
        }
        std::vector<uint64_t> mask( nTets, 0 );
        auto toSlabs = [&](size_t th)
        {
            size_t const lo = nTets * th / size_t(nDepThreads), hi = nTets * (th + 1) / size_t(nDepThreads);
            for (size_t i = lo; i < hi; ++i)
            {
                uint64_t m = 0;
                forEachPlane( i, [&](long loc) { m |= uint64_t(1) << slabOfX[size_t(loc)]; } );
                // reaching no plane (e.g. the padding around a --box region): it deposits nothing, but is
                // still filtered and counted once -- spread over the slabs, not all on slab 0, which the work
                // balance does not see (a TNG --box region slowed by 7% when they piled up there)
                mask[i] = m ? m : uint64_t(1) << ( i % size_t(nSlabs) );
            }
        };
        {
            std::vector<std::thread> pool;
            for (int th = 1; th < nDepThreads; ++th) pool.emplace_back( toSlabs, size_t(th) );
            toSlabs( 0 );
            for (std::thread &x : pool) x.join();
        }
        std::vector<int32_t>().swap( gLo ); std::vector<int32_t>().swap( gHi );
        slabTetLists.resize( size_t(nSlabs) );
        slabFirst.assign( nTets, 0 );
        {
            std::vector<size_t> count( size_t(nSlabs), 0 );
            for (size_t i = 0; i < nTets; ++i)
                for (uint64_t m = mask[i]; m; m &= m - 1) ++count[size_t(__builtin_ctzll(m))];
            for (int sl = 0; sl < nSlabs; ++sl) slabTetLists[size_t(sl)].reserve( count[size_t(sl)] );
        }
        size_t entries = 0;
        for (size_t i = 0; i < nTets; ++i)
        {
            if ( mask[i] ) slabFirst[i] = uint8_t( __builtin_ctzll(mask[i]) );
            for (uint64_t m = mask[i]; m; m &= m - 1)
            { slabTetLists[size_t(__builtin_ctzll(m))].push_back( uint32_t(i) ); ++entries; }
        }
        size_t kept = 0;
        for (size_t i = 0; i < nTets; ++i) kept += mask[i] ? 1 : 0;
        if ( not userOptions.psSuppressGridStats or userOptions.verboseLevel >= 3 )   // (a partition's at 3)
            message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " CPU deposit in SLAB mode: "
                    << nDepThreads << " threads over " << nSlabs << " slabs of the x axis (the accumulator copies "
                    << "would not fit), " << ( kept ? double(entries) / double(kept) : 0. )
                    << " slabs per tetrahedron (mean extent " << meanExtent << " cells).\n" << MESSAGE::Flush;
    }
    std::vector<DepositThread> depT( static_cast<size_t>(nDepThreads) );
    for (int th = 0; th < nDepThreads; ++th)
    {
        DepositThread &D = depT[size_t(th)];
        if ( th == 0 or slabMode )      // slab mode: every thread writes its own cells of the real arrays
        {
            D.q = quantities; D.streamCount = &streamCount; D.releasedMult = &releasedMult;
            D.exactMult = &exactMult; D.dispWeight = &dispWeight; D.dispVel = &dispVel; D.massWeight = &massWeight;
            D.orientBits = &orientBits; D.subSampleMassFlag = &subSampleMassFlag; D.flippedFlag = &flippedFlag;
            D.insideFlat.swap( insideFlat ); D.insideRel.swap( insideRel ); D.insideW.swap( insideW );
            D.exMom.swap( exMom ); D.exW.swap( exW );
            continue;
        }
        auto zeroLike = [](auto &own, auto const &real) { own.assign( real.size(), typename std::decay_t<decltype(real)>::value_type() ); };
        zeroLike( D.ownQ.density, quantities->density );
        zeroLike( D.ownQ.velocity, quantities->velocity );
        zeroLike( D.ownQ.velocity_gradient, quantities->velocity_gradient );
        zeroLike( D.ownQ.velocity_dispersion, quantities->velocity_dispersion );
#ifdef SCALAR
        zeroLike( D.ownQ.scalar, quantities->scalar );
        zeroLike( D.ownQ.scalar_gradient, quantities->scalar_gradient );
#endif
        zeroLike( D.ownStreamCount, streamCount );         zeroLike( D.ownReleasedMult, releasedMult );
        zeroLike( D.ownExactMult, exactMult );             zeroLike( D.ownDispWeight, dispWeight );
        zeroLike( D.ownDispVel, dispVel );                 zeroLike( D.ownOrientBits, orientBits );
        zeroLike( D.ownMassWeight, massWeight );
        zeroLike( D.ownSubSampleMassFlag, subSampleMassFlag ); zeroLike( D.ownFlippedFlag, flippedFlag );
        D.q = &D.ownQ; D.streamCount = &D.ownStreamCount; D.releasedMult = &D.ownReleasedMult;
        D.exactMult = &D.ownExactMult; D.dispWeight = &D.ownDispWeight; D.dispVel = &D.ownDispVel;
        D.massWeight = &D.ownMassWeight;
        D.orientBits = &D.ownOrientBits; D.subSampleMassFlag = &D.ownSubSampleMassFlag; D.flippedFlag = &D.ownFlippedFlag;
    }

    // One thread's share of the deposit: the loop over the cells, whose body is the serial loop's,
    // unchanged -- its accumulators and scratch are this thread's under the SAME names (locals
    // shadowing the function's), and the do/while(false) keeps its 'continue' meaning "skip the
    // rest of this tetrahedron". Thread th of nT takes every nT-th block of 256 cells; after a GPU
    // deposit (deferred) it walks the deferred ordinals instead. The loop sits INSIDE the lambda:
    // a call per tetrahedron cost the single-thread path 19%.
    size_t const DEP_BLOCK = 256;
    auto depositLoop = [&](DepositThread &TH, size_t th, size_t nT, bool deferred, int const slab)
    {
        Quantities *quantities = TH.q;
        std::vector<int> &streamCount = *TH.streamCount;
        std::vector<Real> &releasedMult = *TH.releasedMult;
        std::vector<Real> &exactMult = *TH.exactMult;
        std::vector<Real> &dispWeight = *TH.dispWeight;
        std::vector<Real> &massWeight = *TH.massWeight;
        std::vector< Pvector<Real,noVelComp> > &dispVel = *TH.dispVel;
        std::vector<PSCausticClass::CausticMask> &orientBits = *TH.orientBits;
        std::vector<unsigned char> &subSampleMassFlag = *TH.subSampleMassFlag;
        std::vector<unsigned char> &flippedFlag = *TH.flippedFlag;
        std::vector<size_t> &insideFlat = TH.insideFlat;
        std::vector<Real> &insideRel = TH.insideRel;
        std::vector<Real> &insideW = TH.insideW;
        std::vector<double> &exMom = TH.exMom;
        std::vector<double> &exW = TH.exW;
        size_t &nDegenerateInverse = TH.nDegenerateInverse;
        size_t &nReleased = TH.nReleased;
        // The per-sample loops' constants as TRUE locals: read through the lambda's captured
        // references they were reloaded after every push_back/accumulate that might alias them
        // (size_t / Real stores), which made the wrapped loop 42% slower than the inline one.
        bool const periodicL = userOptions.periodic;
        bool const windowModeL = windowMode;
        int const nSubL = nSub;
        size_t const nSamplesL = nSamplesPerCell;
        Real const cellVolumeL = cellVolume;
        size_t nGridL[NO_DIM], subOriginL[NO_DIM], subDimsL[NO_DIM];
        double gridLenL[NO_DIM];
        double const *samplePosL[NO_DIM];
        Real dxL[NO_DIM], boxL[2*NO_DIM];
        for (int d = 0; d < NO_DIM; ++d)
        {
            nGridL[d] = nGrid[d]; subOriginL[d] = subOrigin[d]; subDimsL[d] = subDims[d];
            gridLenL[d] = gridLenD[d]; samplePosL[d] = samplePosAxis[d].data(); dxL[d] = dx[d];
            boxL[2*d] = boxCoordinates[2*d]; boxL[2*d+1] = boxCoordinates[2*d+1];
        }
        PSBeyondGrid beyondGrid;   // (a copy: passing the hoisted locals themselves would escape them)
        beyondGrid.nSub = nSub;
        for (int d = 0; d < NO_DIM; ++d)
        {
            beyondGrid.nGrid[d] = nGrid[d]; beyondGrid.gridLen[d] = gridLenD[d];
            beyondGrid.boxLo[d] = boxCoordinates[2*d]; beyondGrid.dx[d] = dx[d];
        }
        bool const dispNeedsOwnWeightL = dispNeedsOwnWeight;
        bool const haveVelL = haveVel;
        bool const needVelValuesL = needVelValues;
        bool const needWeightL = needWeight;
        bool const psCausticsL = psCaustics;
        bool const psExactL = psExact;
        bool const psLinearL = psLinear;
        bool const psVertexMassL = psVertexMass;
        bool const psVolWeightedL = psVolWeighted;
        bool const weightIsDensityL = weightIsDensity;
        bool const fVGradL = field.velocity_gradient;
        bool const fVDispL = field.velocity_dispersion;
        bool const fSGradL = field.scalar_gradient;
        bool const fDenL = field.density;
        bool const fVelL = field.velocity;
        bool const fScalL = field.scalar;
        // slab mode (slab >= 0): this slab's tetrahedra, in the loop's order, and the local x planes it
        // owns -- every write below skips the cells of other planes
        std::vector<uint32_t> const *slabList = slab >= 0 ? &slabTetLists[size_t(slab)] : nullptr;
        long const slabLoL = slab >= 0 ? slabXLo[size_t(slab)] : 0L;
        long const slabHiL = slab >= 0 ? slabXHi[size_t(slab)] : LONG_MAX;
        size_t planeL = 1;                                   // cells per local x plane
        for (int d = 1; d < NO_DIM; ++d) planeL *= subDims[d];
        size_t slabK = 0, curOrd = 0;
        size_t idx = 0, cellOrdCounter = 0, nextDeferred = 0;
        auto itC = PS_FINITE_CELLS_BEGIN;
        auto const itEnd = PS_FINITE_CELLS_END;
        for (;;)
        {
            Cell_handle cell;
            if ( slabList )
            {
                if ( slabK == slabList->size() ) break;
                curOrd = (*slabList)[slabK++];
                cell = slabTets[curOrd];
            }
            else
            {
                if ( itC == itEnd ) break;
                cell = itC;
                ++itC;
                if ( deferred )
                {
                    size_t const cellOrd = cellOrdCounter++;
                    if ( nextDeferred == deferredOrd.size() ) break;   // every deferred cell done
                    if ( size_t(deferredOrd[nextDeferred]) != cellOrd ) continue;
                    ++nextDeferred;
                }
                else if ( ( (idx++ / DEP_BLOCK) % nT ) != th )
                    continue;
            }
        do {

            // Shared filter chain (ps_cell_filter.h): dummy-vertex skip, Lagrangian-centroid
            // ownership, hull handling, minimum-image wrap, degeneracy + zero-inverse checks.
            PSCellGeometry geo;
            size_t degenerateHere = 0;
            if ( !psFilterCell(cell, userOptions, boxCoordinates, true, &degenerateHere, geo) )
            {
                if ( !slabList || int(slabFirst[curOrd]) == slab ) nDegenerateInverse += degenerateHere;   // once per tet
                continue;
            }
            if ( !metalDeposited ) markFlipped(geo, flippedFlag, slabLoL, slabHiL);   // (GPU-deferred tets: the host loop marked them)
            bool const useVolumeRatioDensity = geo.useVolumeRatioDensity;
            Real (&eulerPos)[NO_DIM+1][NO_DIM] = geo.eulerPos;
            Real (&posMatInv)[NO_DIM][NO_DIM] = geo.posMatInv;

            // PS-DTFE per-tetrahedron MASS (Abel/Hahn/Shandarin tetrahedra method): each Lagrangian flow
            // element carries a constant mass m = averageDensity * V_lag, conserved as it maps to its
            // present-day Eulerian simplex (Eulerian volume V_eul). We deposit this finite MASS onto the
            // grid (mass-conserving; see the scatter below), instead of painting the per-tetrahedron
            // density rho = m / V_eul at grid points: at a caustic V_eul -> 0 so rho -> ~1e7, and
            // point-sampling that density either misses the thin simplex (mass undercounted) or splats
            // ~1e7 over a whole grid cell (the grid-aligned "square" over-densities, worse under
            // sub-sampling because more sample points fall in the cell). Depositing m is exact at any nSubL.
            Real tetMass;
            double absDetLag;
            // --ps-caustics: the tetrahedron's caustic stratification bits (collapse multiplicity and
            // the umbilic-degeneracy indicator). Computed HERE, where both edge matrices exist, because
            // the deformation tensor J = Ax * Lag^-1 is constant inside a linearly mapped tetrahedron;
            // the bits are then OR-ed into every cell this tetrahedron deposits into.
            PSCausticClass::CausticMask tetCausticClass = 0;
            {
                double Lag[NO_DIM][NO_DIM];
                for (int v = 0; v < NO_DIM; ++v)
                    for (int i = 0; i < NO_DIM; ++i)
                        Lag[v][i] = double(cell->vertex(v+1)->point()[i]) - double(cell->vertex(0)->point()[i]);
                // absDetLag always computed: the --ps-halo-release classification stays geometric
                // (V_lag/V_eul) so kept/released sets are identical under either mass convention
                absDetLag = std::fabs(determinant(Lag));
                tetMass = psVertexMassL ? Real( psVertexShareMass(cell) )
                                       : Real( double(userOptions.averageDensity) * absDetLag / factorial(NO_DIM) );  // = rho_bar * V_lag
                if (tetMass < Real(0.)) tetMass = Real(0.);
                if ( psCausticsL )
                {
                    PSCausticClass::TetDeformation const self = PSCausticClass::analyzeTet( geo.Ax, Lag );
                    tetCausticClass = self.bits;

                    /* A3 (cusp) indicator. grad(lambda_c) does not exist inside this tetrahedron -- J is
                       constant here -- so it is estimated from the FACE NEIGHBOURS: their critical
                       eigenvalues against their Lagrangian centroid offsets. Only tetrahedra already
                       near their fold do this work (cuspIndicator re-checks the band, but gathering
                       four neighbours for every tetrahedron in the box would be wasted on the ~99% that
                       are nowhere near one). The neighbours go through psFilterCell with ownership
                       skipped, so their edge matrices use exactly the deposit's min-image convention
                       and a neighbour outside this partition still contributes its gradient sample. */
                    // (2D too since 2026-10-02: the triangles around the three vertices; no A4 bit in 2D)
                    if ( userOptions.psCausticCusps and self.valid and self.spread > 0.
                         and std::fabs(self.critical) <= PSCausticClass::FOLD_BAND * self.spread )
                    {
                        /* STENCIL: every tetrahedron sharing a VERTEX with this one, not just the four
                           sharing a face. Four samples barely determine a 3-D gradient -- if they happen
                           to sit in one slab the fit is ill-conditioned along exactly the direction the
                           cusp test interrogates, and the transverse noise then reads as tangency. The
                           vertex-incident ring gives ~20-40 samples spread in every direction. Measured
                           on the 1-D wave at A*k = 1.8, which has NO cusps: the false-positive rate over
                           the fold cells falls from ~5-7% (face neighbours) to the figure the test now
                           bounds. Duplicates (a face neighbour is reached from three vertices) are left
                           in: for a least-squares fit they are a mild re-weighting, and deduplicating
                           CGAL cell handles portably is not worth it. */
                        static int const MAX_STENCIL = 64;
                        double dq[MAX_STENCIL][NO_DIM], dLam[MAX_STENCIL];
                        int nb = 0;
                        double c0[NO_DIM] = {0.};
                        for (int v = 0; v <= NO_DIM; ++v)
                            for (int d = 0; d < NO_DIM; ++d)
                                c0[d] += double(cell->vertex(v)->point()[d]) / double(NO_DIM+1);

                        std::vector<Cell_handle> ring;
                        ring.reserve( 48 );
                        for (int v = 0; v <= NO_DIM; ++v)
                        {
#if NO_DIM==3
                            dt.incident_cells( cell->vertex(v), std::back_inserter(ring) );
#else
                            auto fc = dt.incident_faces( cell->vertex(v) ), done = fc;
                            if ( fc != nullptr )
                                do { ring.push_back( fc ); } while ( ++fc != done );
#endif
                        }

                        /* Deduplicate, then keep the MAX_STENCIL nearest cells. The raw ring lists a
                           face neighbour once per shared vertex; those repeats re-weight the fit AND
                           fill the size cap, so the gather used to truncate part-way and drop the last
                           vertex's cells entirely -- a lopsided stencil biasing exactly the tangency the
                           A3 test measures. Selection happens before psFilterCell/analyzeTet: the
                           centroid needs only vertex coordinates, so the sort is nearly free and the
                           expensive analysis runs on 64 cells instead of on every duplicate. */
                        std::sort( ring.begin(), ring.end(),
                                   [](Cell_handle a, Cell_handle b){ return &*a < &*b; } );
                        ring.erase( std::unique( ring.begin(), ring.end() ), ring.end() );

                        struct RingCand { double d2; Cell_handle cell; double nc[NO_DIM]; };
                        std::vector<RingCand> cand;
                        cand.reserve( ring.size() );
                        for (size_t ri = 0; ri < ring.size(); ++ri)
                        {
                            Cell_handle nbCell = ring[ri];
                            if ( nbCell == cell or dt.is_infinite(nbCell) ) continue;
                            RingCand rc;
                            rc.cell = nbCell;
                            double d2 = 0.;
                            for (int d = 0; d < NO_DIM; ++d) rc.nc[d] = 0.;
                            for (int v = 0; v <= NO_DIM; ++v)
                                for (int d = 0; d < NO_DIM; ++d)
                                    rc.nc[d] += double(nbCell->vertex(v)->point()[d]) / double(NO_DIM+1);
                            for (int d = 0; d < NO_DIM; ++d)
                            {
                                double const e = rc.nc[d] - c0[d];
                                d2 += e*e;
                            }
                            rc.d2 = d2;
                            cand.push_back( rc );
                        }
                        if ( cand.size() > size_t(MAX_STENCIL) )
                        {
                            std::nth_element( cand.begin(), cand.begin() + MAX_STENCIL, cand.end(),
                                              [](RingCand const &a, RingCand const &b){ return a.d2 < b.d2; } );
                            cand.resize( MAX_STENCIL );
                        }

                        for (size_t ci = 0; ci < cand.size() and nb < MAX_STENCIL; ++ci)
                        {
                            Cell_handle nbCell = cand[ci].cell;
                            PSCellGeometry ngeo;
                            if ( !psFilterCell(nbCell, userOptions, boxCoordinates, true, NULL, ngeo, true) )
                                continue;
                            double nLag[NO_DIM][NO_DIM];
                            for (int v = 0; v < NO_DIM; ++v)
                                for (int i = 0; i < NO_DIM; ++i)
                                    nLag[v][i] = double(nbCell->vertex(v+1)->point()[i])
                                               - double(nbCell->vertex(0)->point()[i]);
                            PSCausticClass::TetDeformation const other =
                                PSCausticClass::analyzeTet( ngeo.Ax, nLag );
                            if ( not other.valid ) continue;
                            for (int d = 0; d < NO_DIM; ++d) dq[nb][d] = cand[ci].nc[d] - c0[d];
                            dLam[nb] = other.critical - self.critical;
                            ++nb;
                        }
                        if ( PSCausticClass::cuspIndicator( self, dq, dLam, nb ) )
                            tetCausticClass |= PSCausticClass::BIT_CUSP;
                        /* A4 from the SAME ring: one extra 5x5 solve, the stencil being the whole
                           cost. Deliberately NOT nested inside the A3 branch -- cuspIndicator's linear
                           fit reads a true swallowtail's cubic as a slope and says "no cusp" (see
                           foldDerivatives), so gating on BIT_CUSP would hide the target cells. */
                        if ( PSCausticClass::swallowtailIndicator( self, dq, dLam, nb ) )
                            tetCausticClass |= PSCausticClass::BIT_SWALLOWTAIL;
                    }
                }
            }

            // --ps-halo-release: released tets skip PASS 1 below (zero-trip window), so the
            // empty sample set drives the mass-conserving centroid fallback -- a monolithic
            // single-cell deposit with the centroid-evaluated velocity.
            bool const released = psHaloRelease > 0. && absDetLag > psHaloRelease * geo.cellAbsDet;
            if (released && (!slabList || int(slabFirst[curOrd]) == slab)) ++nReleased;   // once per tet

            // Eulerian bounding box of this cell (from the wrapped positions).
            Real eMin[NO_DIM], eMax[NO_DIM];
            for (int d = 0; d < NO_DIM; ++d)
            {
                eMin[d] = eulerPos[0][d];
                eMax[d] = eulerPos[0][d];
            }
            for (int v = 1; v <= NO_DIM; ++v)
                for (int d = 0; d < NO_DIM; ++d)
                {
                    if (eulerPos[v][d] < eMin[d]) eMin[d] = eulerPos[v][d];
                    if (eulerPos[v][d] > eMax[d]) eMax[d] = eulerPos[v][d];
                }

            // Grid index range overlapping the bounding box (periodic indices are wrapped below).
            // A non-periodic grid clamps it; iMinU/iMaxU keep the unclamped extent, and 'straddle'
            // marks a tetrahedron that reaches beyond the grid -- one cut by the face of a --box
            // region taken from a larger particle cloud.
            int iMin[NO_DIM], iMax[NO_DIM], iMinU[NO_DIM], iMaxU[NO_DIM];
            bool outsideGrid = false, straddle = false;
            for (int d = 0; d < NO_DIM; ++d)
            {
                iMin[d] = int(floor((eMin[d] - boxL[2*d]) / dxL[d]));
                iMax[d] = int(floor((eMax[d] - boxL[2*d]) / dxL[d])) + 1;
                // straddle is decided on the tet's own bbox, before the sample margin below: a tet whose
                // bbox stays inside the grid has no sample beyond it (the margin alone flagged every tet
                // within a cell of a region face, 14% of the TNG region's tets, all for nothing)
                if (!periodicL && (iMin[d] < 0 || iMax[d] > (int)nGridL[d])) straddle = true;
                // sub-sample points can lie in edge cells; the exact deposit clips the bbox
                // window directly (no sub-samples), so the expansion would only add empty clips
                if (nSubL > 1 && !psExactL) { iMin[d] -= 1; iMax[d] += 1; }
                iMinU[d] = iMin[d]; iMaxU[d] = iMax[d];
                if (!periodicL)
                {
                    if (iMin[d] < 0) iMin[d] = 0;
                    if (iMax[d] > (int)nGridL[d]) iMax[d] = (int)nGridL[d];
                    if (iMin[d] >= (int)nGridL[d] || iMax[d] <= 0) { outsideGrid = true; break; }
                }
            }
            if (outsideGrid) { continue; }
            if ( windowModeL )
            {   // --ps-window: a tetrahedron that misses the window on some axis deposits nothing
                bool touches = true;
                for (int d = 0; d < NO_DIM && touches; ++d)
                    touches = psSpanTouches(iMinU[d], iMaxU[d], long(nGridL[d]), periodicL, long(subOriginL[d]), long(subDimsL[d]));
                if ( !touches ) continue;
            }
            if (released)
                for (int d = 0; d < NO_DIM; ++d) iMax[d] = iMin[d];   // zero-trip PASS 1 -> centroid fallback

            // (density is deposited as the per-tetrahedron mass tetMass computed above; no per-vertex
            //  density gradient is needed for the scatter.)

            // Constant velocity gradient (also needed to evaluate the per-stream velocity for the dispersion).
            Real velGrad[NO_DIM][noVelComp];
            if (fVelL || fVGradL || fVDispL)
            {
                Vertex_handle base = cell->vertex(0);
                Real temp[NO_DIM][noVelComp];
                for (int v = 0; v < NO_DIM; ++v)
                    for (size_t i = 0; i < noVelComp; ++i)
                        temp[v][i] = cell->vertex(v+1)->info().velocity(i) - base->info().velocity(i);
                matrixMultiplication<noVelComp>(posMatInv, temp, velGrad);
            }

            // --ps-linear-deposit: constant gradient of the DTFE vertex densities across this cell
            // (same affine convention as the velocity above). Hull cells (non-periodic volume-ratio
            // density) have a constant profile, which degrades to the uniform deposit below.
            Real denBase = Real(0.), denGrad[NO_DIM] = { Real(0.) };
            bool linearProfile = false;
            if ( psLinearL && !useVolumeRatioDensity )
            {
                denBase = cell->vertex(0)->info().density();
                Real temp[NO_DIM];
                for (int v = 0; v < NO_DIM; ++v)
                    temp[v] = cell->vertex(v+1)->info().density() - denBase;
                matrixMultiplication(posMatInv, temp, denGrad);
                linearProfile = true;
            }

            // Constant scalar gradient across this cell.
    #ifdef SCALAR
            Real sGrad[NO_DIM][noScalarComp];
            if (fScalL || fSGradL)
            {
                Real temp[NO_DIM][noScalarComp];
                for (int v = 0; v < NO_DIM; ++v)
                    for (size_t i = 0; i < noScalarComp; ++i)
                        temp[v][i] = cell->vertex(v+1)->info().myScalar()[i] - cell->vertex(0)->info().myScalar()[i];
                matrixMultiplication<noScalarComp>(posMatInv, temp, sGrad);
            }
    #endif

    #if NO_DIM==3 || NO_DIM==2
            // ===================== exact conservative deposit (--ps-exact-deposit) =====================
            // r3d (Powell & Abel 2015): clip the Eulerian tetrahedron against every grid cell in
            // its bbox window and integrate the moments analytically. All geometry lives in
            // coordinates RELATIVE to (wrapped) vertex 0 -- the same affine frame as velGrad and
            // denGrad, so the moments feed the linear profiles directly. Released tets (--ps-halo-
            // release) fall through to the sampled path below, whose zero-tripped window drives
            // the same monolithic centroid fallback.
            if ( psExactL && !released )
            {
                // the moments of a piece: [1, x.., then the second moments a <= b row by row] --
                // 10 in 3D (r3d's order-2 layout), 6 in 2D (ps_clip2d.h)
                static int const NMOM = 1 + NO_DIM + NO_DIM*(NO_DIM+1)/2;
    #if NO_DIM==3
                r3d_rvec3 tv[NO_DIM+1];
                for (int v = 0; v <= NO_DIM; ++v)
                    for (int d = 0; d < NO_DIM; ++d)
                        tv[v].xyz[d] = double(eulerPos[v][d]) - double(eulerPos[0][d]);
                // r3d_init_tet wants positive orientation; a folded (negative-parity) tet is
                // handed over with vertices 1,2 swapped -- vertex 0 stays the frame origin
                if ( geo.cellDet < 0. )
                {
                    r3d_rvec3 const tmp = tv[1]; tv[1] = tv[2]; tv[2] = tmp;
                }
                r3d_poly tetPoly;
                r3d_init_tet(&tetPoly, tv);
    #else
                // 2D: the triangle in the vertex-0 frame, counter-clockwise (a folded one swapped)
                double tri2[NO_DIM+1][NO_DIM];
                for (int v = 0; v <= NO_DIM; ++v)
                    for (int d = 0; d < NO_DIM; ++d)
                        tri2[v][d] = double(eulerPos[v][d]) - double(eulerPos[0][d]);
                if ( geo.cellDet < 0. )
                    for (int d = 0; d < NO_DIM; ++d) std::swap( tri2[1][d], tri2[2][d] );
    #endif

                // PASS 1 (exact): moments of tet ∩ cell for every window cell with nonzero volume.
                // Order 2 = 10 moments [1, x, y, z, x2, xy, xz, y2, yz, z2] in the vertex-0 frame.
                // --ps-window clips only the cells of the window (the hull of its images in raw
                // indices): the tet is then normalized by its WHOLE weight below, so the pieces
                // outside the window are never needed -- on the fine virtual grid of a zoom the
                // window is a thin slab of a bounding box that may hold millions of cells.
                int cLo[NO_DIM], cHi[NO_DIM];
                bool missesWindow = false;
                for (int d = 0; d < NO_DIM; ++d)
                {
                    cLo[d] = iMin[d]; cHi[d] = iMax[d];
                    if ( !windowModeL ) continue;
                    long hl = 0, hh = 0;
                    if ( !psWindowHull(iMin[d], iMax[d], long(nGridL[d]), periodicL, long(subOriginL[d]), long(subDimsL[d]), hl, hh) )
                    { missesWindow = true; break; }
                    cLo[d] = int(hl); cHi[d] = int(hh);
                }
                if ( missesWindow ) continue;   // next tetrahedron: nothing of it lies in the window
                insideFlat.clear();          // reused scratch: flat sub-grid index per hit cell
                exMom.clear();
                exW.clear();
                double sumW = 0.;            // per-tet weight normalizer (uniform: volume)
    #if NO_DIM==3
                for (int gi = cLo[0]; gi < cHi[0]; ++gi)
                for (int gj = cLo[1]; gj < cHi[1]; ++gj)
                for (int gk = cLo[2]; gk < cHi[2]; ++gk)
                {
                    int const wgi = periodicL ? ((gi % (int)nGridL[0] + (int)nGridL[0]) % (int)nGridL[0]) : gi;
                    int const wgj = periodicL ? ((gj % (int)nGridL[1] + (int)nGridL[1]) % (int)nGridL[1]) : gj;
                    int const wgk = periodicL ? ((gk % (int)nGridL[2] + (int)nGridL[2]) % (int)nGridL[2]) : gk;
                    int const gridIdx[3] = {wgi, wgj, wgk};
                    int const rawIdx[3]  = {gi, gj, gk};
    #else
                for (int gi = cLo[0]; gi < cHi[0]; ++gi)
                for (int gj = cLo[1]; gj < cHi[1]; ++gj)
                {
                    int const wgi = periodicL ? ((gi % (int)nGridL[0] + (int)nGridL[0]) % (int)nGridL[0]) : gi;
                    int const wgj = periodicL ? ((gj % (int)nGridL[1] + (int)nGridL[1]) % (int)nGridL[1]) : gj;
                    int const gridIdx[2] = {wgi, wgj};
                    int const rawIdx[2]  = {gi, gj};
    #endif
                    size_t flatIdx = 0;
                    bool inSub = true;
                    for (int d = 0; d < NO_DIM; ++d)
                    {
                        long loc = (long)gridIdx[d] - (long)subOriginL[d];
                        if (loc < 0) loc += (long)nGridL[d];        // sub-box may wrap a periodic axis
                        if (loc < 0 || loc >= (long)subDimsL[d]) { inSub = false; break; }
                        flatIdx = flatIdx * subDimsL[d] + (size_t)loc;
                    }
                    // outside the sub-box: a partition's box holds every cell its tets touch, and a
                    // window's hull may hold cells outside the window -- neither is clipped
                    if (!inSub) continue;

                    // cell bounds in the vertex-0 frame (RAW indices: the periodic image the
                    // wrapped tet actually intersects), then 6 clip planes n.x + d >= 0
    #if NO_DIM==3
                    r3d_plane planes[6];
                    for (int d = 0; d < NO_DIM; ++d)
                    {
                        double const lo = double(boxL[2*d]) + rawIdx[d] * double(dxL[d]) - double(eulerPos[0][d]);
                        double const hi = lo + double(dxL[d]);
                        for (int q = 0; q < 2; ++q)
                        {
                            planes[2*d+q].n.xyz[0] = 0.; planes[2*d+q].n.xyz[1] = 0.; planes[2*d+q].n.xyz[2] = 0.;
                        }
                        planes[2*d].n.xyz[d]   =  1.; planes[2*d].d   = -lo;
                        planes[2*d+1].n.xyz[d] = -1.; planes[2*d+1].d =  hi;
                    }
                    r3d_poly piece = tetPoly;
                    r3d_clip(&piece, planes, 6);
                    if ( piece.nverts == 0 ) continue;
                    r3d_real mom[10];
                    r3d_reduce(&piece, mom, 2);
                    if ( !(mom[0] > 0.) ) continue;
    #else
                    double lo2[NO_DIM], hi2[NO_DIM];
                    for (int d = 0; d < NO_DIM; ++d)
                    {
                        lo2[d] = double(boxL[2*d]) + rawIdx[d] * double(dxL[d]) - double(eulerPos[0][d]);
                        hi2[d] = lo2[d] + double(dxL[d]);
                    }
                    double mom[psClip2d::MOMENTS];
                    if ( not psClip2d::clipTriangleToCell( tri2, lo2, hi2, mom ) ) continue;
    #endif

                    insideFlat.push_back(flatIdx);
                    for (int m = 0; m < NMOM; ++m) exMom.push_back(double(mom[m]));
                    // per-cell mass weight: exact volume, or the exact integral of the linear
                    // density profile under --ps-linear-deposit (clamped like the sampled path)
                    double w = mom[0];
                    if ( psLinearL && linearProfile )
                    {
                        w = double(denBase) * mom[0];
                        for (int d = 0; d < NO_DIM; ++d)
                            w += double(denGrad[d]) * mom[1+d];
                        if ( w < 0. ) w = 0.;
                    }
                    exW.push_back(w);
                    sumW += w;
                }

                // PASS 2 (exact): renormalize the shares to tetMass -- conservation is exact per
                // tet by construction. A tet that leaves a non-periodic grid is normalized by its
                // WHOLE weight instead (the analytic volume, or the integral of the linear profile
                // over the tet), so the grid keeps exactly the share of its mass inside it:
                // renormalizing over the in-grid pieces piled the outside share onto the faces of
                // a --box region (3x the density in the face layer of a cut TNG region). max()
                // guards float rounding. A --ps-window run normalizes EVERY tet this way: only its
                // window pieces were clipped, and the whole weight equals the full run's sum over
                // all pieces to double rounding (the window agrees with the full run to float
                // rounding, where the sampled deposit agrees bit for bit). An all-empty window
                // (degenerate sliver below r3d's resolution, or a tet wholly beyond the grid) falls
                // through to the sampled path -- except under --ps-window, where it means "no
                // volume inside the window": the tet deposits nothing here.
                if ( sumW > 0. )
                {
                    double norm = sumW;
                    if ( straddle || windowModeL )
                    {
                        double const vol = double(geo.cellAbsDet) / double(factorial(NO_DIM));
                        double wFull = vol;
                        if ( psLinearL && linearProfile )
                        {
                            // integral of den0 + denGrad.x over the tet = V (den0 + denGrad.centroid)
                            wFull = double(denBase);
                            for (int d = 0; d < NO_DIM; ++d)
                            {
                                double c = 0.;
                                for (int v = 1; v <= NO_DIM; ++v) c += double(eulerPos[v][d]) - double(eulerPos[0][d]);
                                wFull += double(denGrad[d]) * c / double(NO_DIM + 1);
                            }
                            wFull *= vol;
                        }
                        norm = std::max( sumW, wFull );
                    }
                    double const shareFac = double(tetMass) / norm;
                    for (size_t c = 0; c < insideFlat.size(); ++c)
                    {
                        size_t const flatIdx = insideFlat[c];
                        if ( slabList ) { long const xl = long(flatIdx / planeL); if ( xl < slabLoL || xl >= slabHiL ) continue; }
                        double const *m = &exMom[c*NMOM];
                        Real const massShare = Real( exW[c] * shareFac );
                        // moment weight: mass share, or the analytic intersection volume V_int
                        // (--ps-volume-weighted, the exact continuum volume share)
                        Real const momShare = psVolWeightedL ? Real( m[0] ) : massShare;
                        if (fDenL) quantities->density[flatIdx] += massShare;
                        if (needWeightL && !weightIsDensityL) massWeight[flatIdx] += momShare;

                        // exact mean of the linear velocity profile over this intersection
                        Pvector<Real,noVelComp> velBar;
                        if (needVelValuesL)
                        {
                            double const invV = 1. / m[0];
                            double cen[NO_DIM];
                            for (int d = 0; d < NO_DIM; ++d) cen[d] = m[1+d]*invV;
                            for (size_t j = 0; j < noVelComp; ++j)
                            {
                                double vb = double( cell->vertex(0)->info().velocity(j) );
                                for (int i = 0; i < NO_DIM; ++i)
                                    vb += double(velGrad[i][j]) * cen[i];
                                velBar[j] = Real(vb);
                            }
                            if (haveVelL)
                                quantities->velocity[flatIdx] += velBar * momShare;
                        }
                        if (fVGradL)
                        {
                            Pvector<Real,noGradComp> grad;
                            for (size_t j = 0; j < noVelComp; ++j)
                                for (int i = 0; i < NO_DIM; ++i)
                                    grad[j*NO_DIM+i] = velGrad[i][j];
                            quantities->velocity_gradient[flatIdx] += grad * momShare;
                        }
                        if (fVDispL)
                        {
                            // exact second moment of the linear profile: <v_i v_j> over the piece
                            // = vbar_i vbar_j + (G^T Cov G)_ij with Cov the piece's position
                            // covariance from the order-2 moments (M2 index map: xx xy xz yy yz zz)
                            double const invV = 1. / m[0];
                            double cen[NO_DIM];
                            for (int d = 0; d < NO_DIM; ++d) cen[d] = m[1+d]*invV;
                            double cov[NO_DIM][NO_DIM];
                            {   // second moments a <= b row by row (3D: xx xy xz yy yz zz, 2D: xx xy yy)
                                int k = 1 + NO_DIM;
                                for (int a = 0; a < NO_DIM; ++a)
                                    for (int b = a; b < NO_DIM; ++b, ++k)
                                        cov[a][b] = cov[b][a] = m[k]*invV - cen[a]*cen[b];
                            }
                            size_t c2 = 0;
                            for (int a = 0; a < NO_DIM; ++a)
                                for (int b = a; b < NO_DIM; ++b)
                                {
                                    double gcg = 0.;
                                    for (int i = 0; i < NO_DIM; ++i)
                                        for (int k = 0; k < NO_DIM; ++k)
                                            gcg += double(velGrad[i][a]) * cov[i][k] * double(velGrad[k][b]);
                                    quantities->velocity_dispersion[flatIdx][c2++] +=
                                        massShare * ( velBar[a] * velBar[b] + Real(gcg) );
                                }
                            if ( dispNeedsOwnWeightL )
                            {
                                dispWeight[flatIdx] += massShare;
                                dispVel[flatIdx]    += velBar * massShare;
                            }
                        }
    #ifdef SCALAR
                        if (fScalL)
                        {
                            // exact mean of the linear scalar profile (same construction as velBar)
                            double const invV = 1. / m[0];
                            double cen[NO_DIM];
                            for (int d = 0; d < NO_DIM; ++d) cen[d] = m[1+d]*invV;
                            Pvector<Real,noScalarComp> scalarVal;
                            for (size_t j = 0; j < noScalarComp; ++j)
                            {
                                double sb = double( cell->vertex(0)->info().myScalar()[j] );
                                for (int i = 0; i < NO_DIM; ++i)
                                    sb += double(sGrad[i][j]) * cen[i];
                                scalarVal[j] = Real(sb);
                            }
                            quantities->scalar[flatIdx] += scalarVal * momShare;
                        }
                        if (fSGradL)
                        {
                            Pvector<Real,noScalarGradComp> sgrad;
                            for (size_t j = 0; j < noScalarComp; ++j)
                                for (int i = 0; i < NO_DIM; ++i)
                                    sgrad[j*NO_DIM+i] = sGrad[i][j];
                            quantities->scalar_gradient[flatIdx] += sgrad * momShare;
                        }
    #endif
                        streamCount[flatIdx]++;   // raw tet-touch count -> '.tetTouch'
                        exactMult[flatIdx] += Real( m[0] / double(cellVolumeL) );   // exact multiplicity share
                        if ( psCausticsL )
                            orientBits[flatIdx] |= (PSCausticClass::CausticMask)((geo.cellDet > 0. ? 1 : 2) | tetCausticClass);
                    }
                    continue;   // next tetrahedron (the sampled PASS 1/2 below is skipped)
                }
                if ( windowModeL ) continue;   // no volume in the window: nothing to deposit (see above)
            }
    #endif  // NO_DIM==3 || NO_DIM==2 (exact deposit)

            // ===================== Mass-conserving deposit of this tetrahedron =====================
            // PASS 1: gather every sub-sample point (nSubL^NO_DIM per grid cell in the Eulerian bbox) that
            // lies inside the Eulerian simplex, recording its grid cell (flatIdx) and offset rel from
            // vertex 0. Sub-sampling probes the simplex volume uniformly.
            insideFlat.clear();
            insideRel.clear();
            insideW.clear();

            // exact-containment state of this tetrahedron: double inverse + tier-1 band on the
            // wrapped geometry; the canonical tiers' state (from the stored, unwrapped vertices)
            // is built only if some sample lands within band1 of a face
            double invD[NO_DIM][NO_DIM];
            double band1 = std::numeric_limits<double>::infinity();   // no inverse: all to the exact tiers
            if ( psExactInside::inverseNd( geo.Ax, invD ) )
                band1 = psExactInside::fineBand( exactFrame, geo.Ax, geo.cellAbsDet );
            else
                for (int a = 0; a < NO_DIM; ++a)
                    for (int b = 0; b < NO_DIM; ++b) invD[a][b] = 0.;
            Real rawPos[NO_DIM+1][NO_DIM];
            bool rawPosReady = false;
            psExactInside::TetTest exactTet;
    #if NO_DIM==2
            for (int gi = iMin[0]; gi < iMax[0]; ++gi)
            for (int gj = iMin[1]; gj < iMax[1]; ++gj)
            {
                int wgi = periodicL ? ((gi % (int)nGridL[0] + (int)nGridL[0]) % (int)nGridL[0]) : gi;
                int wgj = periodicL ? ((gj % (int)nGridL[1] + (int)nGridL[1]) % (int)nGridL[1]) : gj;
                int gridIdx[2] = {wgi, wgj};
                int rawIdx[2]  = {gi, gj};
    #elif NO_DIM==3
            for (int gi = iMin[0]; gi < iMax[0]; ++gi)
            for (int gj = iMin[1]; gj < iMax[1]; ++gj)
            for (int gk = iMin[2]; gk < iMax[2]; ++gk)
            {
                int wgi = periodicL ? ((gi % (int)nGridL[0] + (int)nGridL[0]) % (int)nGridL[0]) : gi;
                int wgj = periodicL ? ((gj % (int)nGridL[1] + (int)nGridL[1]) % (int)nGridL[1]) : gj;
                int wgk = periodicL ? ((gk % (int)nGridL[2] + (int)nGridL[2]) % (int)nGridL[2]) : gk;
                int gridIdx[3] = {wgi, wgj, wgk};
                int rawIdx[3]  = {gi, gj, gk};
    #endif
                // flat index within this partition's sub-grid; inSub guard skips cells outside it --
                // except under --ps-window, where the cell's samples still COUNT (the tet's full-run
                // normalization) and are marked so pass 2 deposits none of them
                size_t flatIdx = 0;
                bool inSub = true;
                for (int d = 0; d < NO_DIM; ++d)
                {
                    long loc = (long)gridIdx[d] - (long)subOriginL[d];
                    if (loc < 0) loc += (long)nGridL[d];            // sub-box may wrap a periodic axis
                    if (loc < 0 || loc >= (long)subDimsL[d]) { inSub = false; break; }
                    flatIdx = flatIdx * subDimsL[d] + (size_t)loc;
                }
                if (!inSub) { if (!windowModeL) continue; flatIdx = PS_NOT_IN_WINDOW; }

                for (size_t sIdx = 0; sIdx < nSamplesL; ++sIdx)
                {
                    // Exact point-in-Eulerian-simplex test (ps_exact_inside.h) at the sample's double
                    // position: tier 1 = barycentrics of its minimum-image offset from (wrapped)
                    // vertex 0, decisive outside band1; the rest go to the canonical tiers.
                    {
                        double pos[NO_DIM], relD[NO_DIM], baryD[NO_DIM];
                        size_t remD = sIdx;
                        for (int d = 0; d < NO_DIM; ++d)
                        {
                            pos[d] = samplePosL[d][ size_t(gridIdx[d]) * size_t(nSubL) + remD % size_t(nSubL) ];
                            remD /= size_t(nSubL);
                            double r = pos[d] - double(eulerPos[0][d]);
                            if ( periodicL )
                            {
                                if (r >  0.5 * gridLenL[d]) r -= gridLenL[d];
                                if (r < -0.5 * gridLenL[d]) r += gridLenL[d];
                            }
                            relD[d] = r;
                        }
                        int const fine = psExactInside::fineClassify( invD, relD, band1, baryD );
                        if ( fine < 0 ) continue;
                        if ( fine == 0 )
                        {
                            if ( not rawPosReady )
                            {
                                for (int v = 0; v <= NO_DIM; ++v)
                                    for (int d = 0; d < NO_DIM; ++d)
                                        rawPos[v][d] = cell->vertex(v)->info().eulerianPosition(d);
                                rawPosReady = true;
                            }
                            double pc[NO_DIM];
                            psExactInside::canonicalQuery( exactFrame, pos, pc );
                            if ( not psExactInside::canonicalInside( exactFrame, rawPos, exactTet, pc ) ) continue;
                        }
                    }

                    // Offset from (wrapped) vertex 0 for the deposited moments: the float expression
                    // the deposit has always used, so every sample kept before still deposits the
                    // same bits (the exact test changes only WHICH samples a tetrahedron keeps).
                    Real rel[NO_DIM];
                    size_t rem = sIdx;
                    for (int d = 0; d < NO_DIM; ++d)
                    {
                        Real frac = (nSubL == 1) ? Real(0.5)
                                  : ( Real(int(rem % size_t(nSubL))) + Real(0.5) ) / Real(nSubL);
                        rem /= size_t(nSubL);
                        Real const gridPos = boxL[2*d] + (rawIdx[d] + frac) * dxL[d];
                        rel[d] = gridPos - eulerPos[0][d];
                    }

                    insideFlat.push_back(flatIdx);
                    for (int d = 0; d < NO_DIM; ++d) insideRel.push_back(rel[d]);
                    if ( psLinearL )
                    {
                        // linear density at the sample (constant 1 for hull cells -> uniform);
                        // a sample on a face can graze negative in float: clamp
                        Real w = Real(1.);
                        if ( linearProfile )
                        {
                            w = denBase;
                            for (int d = 0; d < NO_DIM; ++d)
                                w += denGrad[d] * rel[d];
                            if ( w < Real(0.) ) w = Real(0.);
                        }
                        insideW.push_back(w);
                    }
                }
            }

            // A tet that leaves a non-periodic grid (straddle) also counts its samples BEYOND the grid,
            // with the same exact test: every sample then carries tetMass / (all its samples), so the
            // grid keeps only the share of the mass that lies inside it. Sharing the whole mass among
            // the in-grid samples piled the outside share onto the faces of a --box region cut from a
            // larger cloud (3x the density in the face layer of a cut TNG region, and more inside the
            // region wherever tets span many cells). Out-of-grid sample positions use the expression
            // that builds samplePosAxis, so an in-grid index would give the table's value bit for bit.
            size_t nOutside = 0;
            double outsideW = 0.;
    #if NO_DIM==3
            // a straddling tet with no sample inside the grid and its centroid beyond it deposits nothing,
            // whatever it holds beyond the grid: its count is skipped
            bool needBeyond = straddle && !released;
            if ( needBeyond && insideFlat.empty() )
            {
                Real cBeyond[NO_DIM];
                size_t fBeyond = 0;
                needBeyond = centroidCell(eulerPos, cBeyond, fBeyond);
            }
            if ( needBeyond )
            {
                PSBeyondTet bt;   // copies: nothing of this loop's state may escape (see psCountBeyondGrid)
                for (int d = 0; d < NO_DIM; ++d)
                {
                    bt.iMinU[d] = iMinU[d]; bt.iMaxU[d] = iMaxU[d]; bt.denGrad[d] = denGrad[d];
                    for (int v = 0; v <= NO_DIM; ++v) bt.eulerPos[v][d] = eulerPos[v][d];
                    for (int e = 0; e < NO_DIM; ++e) bt.invD[d][e] = invD[d][e];
                }
                bt.band1 = band1; bt.cell = cell;
                bt.psLinear = psLinearL; bt.linearProfile = linearProfile; bt.denBase = denBase;
                PSBeyondCount const bc = psCountBeyondGrid( beyondGrid, bt, exactFrame );
                nOutside = bc.n;
                outsideW = bc.w;
            }
    #endif
            // every sample of this tet lies beyond the grid: its mass is not the region's
            if ( insideFlat.empty() && nOutside > 0 ) continue;

            // Fallback: a simplex smaller than the sub-sample spacing may catch no interior point. Deposit
            // its whole mass at the centroid's cell (the centroid is always inside) so no mass is ever lost
            // -- this is what makes even nSubL==1 mass-conserving (no thin-stream undercount).
            bool useCentroid = false;
            size_t centroidFlat = 0;
            Real   centroidRel[NO_DIM];
            if (insideFlat.empty())
            {
                Real centroid[NO_DIM];
                size_t f = 0;
                if ( !centroidCell(eulerPos, centroid, f) ) { continue; }   // simplex centroid outside this grid/partition region: drop it
                useCentroid = true;
                centroidFlat = f;
                for (int d = 0; d < NO_DIM; ++d) centroidRel[d] = centroid[d] - eulerPos[0][d];
            }

            // PASS 2: split tetMass among the interior samples and deposit. Equal shares (default)
            // reproduce the mass distribution across the cells the simplex covers; --ps-linear-deposit
            // weights the shares by the linear density profile instead, RENORMALIZED so the total is
            // exactly tetMass either way (at any nSubL and for any V_eul) -- no cell can receive more
            // than the tetrahedron's mass and the caustic fix is untouched.
            size_t const nInside = useCentroid ? size_t(1) : insideFlat.size();
            size_t const nShare  = nInside + nOutside;   // the samples the mass is shared among
            Real   const uniformShare = tetMass / Real(nShare);
            // --ps-volume-weighted: equal Eulerian-volume share per sample for the velocity
            // moments (the centroid fallback carries the whole tet volume); density keeps mass
            Real   const volumeShare  = Real( geo.cellAbsDet / factorial(NO_DIM) ) / Real(nShare);
            double shareFactor = 0.;
            bool useLinearShares = false;
            if ( psLinearL && !useCentroid )
            {
                double sumW = outsideW;
                for (size_t s = 0; s < nInside; ++s) sumW += double(insideW[s]);
                if ( sumW > 0. )
                {
                    shareFactor = double(tetMass) / sumW;
                    useLinearShares = true;
                }
            }
            for (size_t s = 0; s < nInside; ++s)
            {
                size_t flatIdx;
                Real rel[NO_DIM];
                if (useCentroid)
                {
                    flatIdx = centroidFlat;
                    for (int d = 0; d < NO_DIM; ++d) rel[d] = centroidRel[d];
                }
                else
                {
                    flatIdx = insideFlat[s];
                    if (flatIdx == PS_NOT_IN_WINDOW) continue;   // counted in nShare, not the window's
                    for (int d = 0; d < NO_DIM; ++d) rel[d] = insideRel[s*NO_DIM + d];
                }
                if ( slabList ) { long const xl = long(flatIdx / planeL); if ( xl < slabLoL || xl >= slabHiL ) continue; }
                Real const massShare = useLinearShares ? Real(double(insideW[s]) * shareFactor)
                                                       : uniformShare;
                // moment weight: mass share (default, momentum-like means) or equal volume share
                // (--ps-volume-weighted, volume-average means); density always deposits MASS
                Real const momShare = psVolWeightedL ? volumeShare : massShare;

                // density accumulates MASS here; it is divided by the cell volume once at the end.
                if (fDenL) quantities->density[flatIdx] += massShare;
                if (needWeightL && !weightIsDensityL) massWeight[flatIdx] += momShare;

                // Weighted velocity moment sum(w_s v_s); also needed for the dispersion (uses <v>).
                Pvector<Real,noVelComp> velVal;
                if (needVelValuesL)
                {
                    velVal = cell->vertex(0)->info().velocity();
                    for (int i = 0; i < NO_DIM; ++i)
                        for (size_t j = 0; j < noVelComp; ++j)
                            velVal[j] += velGrad[i][j] * rel[i];
                    if (haveVelL)
                        quantities->velocity[flatIdx] += velVal * momShare;
                }

                if (fVGradL)
                {
                    Pvector<Real,noGradComp> grad;
                    for (size_t j = 0; j < noVelComp; ++j)
                        for (int i = 0; i < NO_DIM; ++i)
                            grad[j*NO_DIM+i] = velGrad[i][j];
                    quantities->velocity_gradient[flatIdx] += grad * momShare;
                }

                // velocity dispersion: weighted second moment sum(w_s v_i v_j) (upper triangle).
                // After normalization -> sigma_ij = <v_i v_j> - <v_i><v_j>.
                if (fVDispL)
                {
                    size_t c = 0;
                    for (int i = 0; i < NO_DIM; ++i)
                        for (int j = i; j < NO_DIM; ++j)
                            quantities->velocity_dispersion[flatIdx][c++] += massShare * velVal[i] * velVal[j];
                    if ( dispNeedsOwnWeightL )
                    {
                        dispWeight[flatIdx] += massShare;
                        dispVel[flatIdx]    += velVal * massShare;
                    }
                }

    #ifdef SCALAR
                if (fScalL)
                {
                    Pvector<Real,noScalarComp> scalarVal = cell->vertex(0)->info().myScalar();
                    for (int i = 0; i < NO_DIM; ++i)
                        for (size_t j = 0; j < noScalarComp; ++j)
                            scalarVal[j] += sGrad[i][j] * rel[i];
                    quantities->scalar[flatIdx] += scalarVal * momShare;
                }
                if (fSGradL)
                {
                    Pvector<Real,noScalarGradComp> sgrad;
                    for (size_t j = 0; j < noScalarComp; ++j)
                        for (int i = 0; i < NO_DIM; ++i)
                            sgrad[j*NO_DIM+i] = sGrad[i][j];
                    quantities->scalar_gradient[flatIdx] += sgrad * momShare;
                }
    #endif

                // the stream multiplicity (see streamCount): one count per real SAMPLE; a released
                // tet (never sampled) adds its volume fraction; a sampled tet that caught no sample
                // (useCentroid) moves mass only. Under psExactL the counter is the tet-touch count.
                if ( useCentroid )
                    subSampleMassFlag[flatIdx] = 1;   // mass from a tet with no sample here
                if ( psExactL )
                    streamCount[flatIdx]++;
                else if ( released )
                    releasedMult[flatIdx] += Real( geo.cellAbsDet / factorial(NO_DIM) / double(cellVolumeL) );
                else if ( !useCentroid )
                    streamCount[flatIdx]++;
                // --ps-exact-deposit fall-through (empty clip window, released or sub-resolution
                // tets): the tet's volume is shared equally among its nInside samples, so each
                // carries V_tet/nInside -- for the monolithic centroid fallback (nInside==1) that
                // is exactly V_tet/V_cell, the same volume fraction the exact path deposits.
                if ( psExactL )
                    exactMult[flatIdx] += Real( geo.cellAbsDet / factorial(NO_DIM) / double(cellVolumeL) )
                                          / Real(nShare);
                if ( psCausticsL )
                    // fold parity (sign of det(Ax)) plus this tetrahedron's stratification bits
                    orientBits[flatIdx] |= (PSCausticClass::CausticMask)((geo.cellDet > 0. ? 1 : 2) | tetCausticClass);
            }
        } while (false);
        }   // end cell loop
    };

    auto const tCpuLoop0 = std::chrono::steady_clock::now();
    if ( metalDeposited && !deferredOrd.empty() )
        depositLoop( depT[0], 0, 1, true, -1 );
    else if ( !metalDeposited )
    {
        size_t const nT = size_t(nDepThreads);
        std::vector<std::exception_ptr> errors( nT );
        std::atomic<int> nextSlab( 0 );
        auto run = [&](size_t th)
        {
            try
            {
                if ( slabMode )      // whole slabs, picked up as threads come free
                    for (int sl = nextSlab++; sl < nSlabs; sl = nextSlab++)
                        depositLoop( depT[th], th, nT, false, sl );
                else
                    depositLoop( depT[th], th, nT, false, -1 );
            }
            catch (...) { errors[th] = std::current_exception(); }
        };
        if ( nT <= 1 )
            run( 0 );
        else
        {
            std::vector<std::thread> pool;
            for (size_t th = 1; th < nT; ++th) pool.emplace_back( run, th );
            run( 0 );
            for (std::thread &x : pool) x.join();
        }
        for (std::exception_ptr const &e : errors)
            if ( e ) std::rethrow_exception( e );
        // sum (or OR) the private copies into the real arrays in thread order, cell ranges in parallel
        // (slab mode has none: every thread wrote its own cells)
        if ( nT > 1 && !slabMode )
        {
            auto merge = [&](size_t part)
            {
                size_t const lo = totalGrid * part / nT, hi = totalGrid * (part + 1) / nT;
                auto addInto = [lo, hi](auto &real, auto const &own) { if (!own.empty()) for (size_t c = lo; c < hi; ++c) real[c] += own[c]; };
                auto orInto  = [lo, hi](auto &real, auto const &own) { if (!own.empty()) for (size_t c = lo; c < hi; ++c) real[c] |= own[c]; };
                for (size_t th = 1; th < nT; ++th)
                {
                    DepositThread const &D = depT[th];
                    addInto( quantities->density, D.ownQ.density );
                    addInto( quantities->velocity, D.ownQ.velocity );
                    addInto( quantities->velocity_gradient, D.ownQ.velocity_gradient );
                    addInto( quantities->velocity_dispersion, D.ownQ.velocity_dispersion );
#ifdef SCALAR
                    addInto( quantities->scalar, D.ownQ.scalar );
                    addInto( quantities->scalar_gradient, D.ownQ.scalar_gradient );
#endif
                    addInto( streamCount, D.ownStreamCount );   addInto( releasedMult, D.ownReleasedMult );
                    addInto( exactMult, D.ownExactMult );       addInto( dispWeight, D.ownDispWeight );
                    addInto( dispVel, D.ownDispVel );           addInto( massWeight, D.ownMassWeight );
                    orInto( orientBits, D.ownOrientBits );      orInto( subSampleMassFlag, D.ownSubSampleMassFlag );
                    orInto( flippedFlag, D.ownFlippedFlag );
                }
            };
            std::vector<std::thread> pool;
            for (size_t part = 1; part < nT; ++part) pool.emplace_back( merge, part );
            merge( 0 );
            for (std::thread &x : pool) x.join();
        }
    }
    for (DepositThread const &D : depT)
    {
        nDegenerateInverse += D.nDegenerateInverse;
        nReleased += D.nReleased;
    }
    std::vector<DepositThread>().swap( depT );   // the private copies
    std::vector<Cell_handle>().swap( slabTets );
    std::vector< std::vector<uint32_t> >().swap( slabTetLists );
    std::vector<uint8_t>().swap( slabFirst );
    if ( metalDeposited && !deferredOrd.empty() && not userOptions.psSuppressGridStats )
        message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " deposited the "
                << MESSAGE::cMagenta() << deferredOrd.size() << MESSAGE::cReset() << " GPU-deferred tetrahedra on the CPU in "
                << std::chrono::duration<double>(std::chrono::steady_clock::now() - tCpuLoop0).count() << " s.\n" << MESSAGE::Flush;

    // Whichever deposit ran (GPU success above skipped the CPU loop; a GPU FAILURE fell back
    // to the CPU loop, which needed the triangulation), its last read of dt is behind us.
    // For the final call on an internally-owned dt, free the triangulation NOW -- during the
    // normalization below and the caller-side merge into the shared grids -- instead of at
    // partition scope end. ~650 B/vertex, ~5 GB per concurrent partition at production scale.
    if ( mayClearDT )
        dt.clear();

    if ( not userOptions.psSuppressGridStats )
    {
        message << MESSAGE::cGreen() << "Done.\n" << MESSAGE::cReset() << MESSAGE::Flush;
        if ( nDegenerateInverse > 0 )
            message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " dropped "
                    << MESSAGE::cMagenta() << nDegenerateInverse << MESSAGE::cReset()
                    << " degenerate (non-invertible) Eulerian cells.\n" << MESSAGE::Flush;
        if ( nReleased > 0 )
            message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " released "
                    << MESSAGE::cMagenta() << nReleased << MESSAGE::cReset()
                    << " halo-interior tetrahedra (rho_geo/rho_bar > " << userOptions.psHaloRelease
                    << ") to monolithic centroid deposits.\n" << MESSAGE::Flush;
    }

    // turn density-weighted moments into mass-weighted means sum(rho_s f_s)/sum(rho_s). Serial: here,
    // BEFORE the density grid is converted to rho/rho_bar (it doubles as the weight when the density
    // field is selected -- see weightIsDensity above). deferNorm: hand un-normalized moments to the
    // caller, which sums across partitions and normalizes once (the only order correct for cells
    // spanning partitions); the weight travels as mass_weight, or is reconstructed from the summed
    // density by normalizePhaseSpace when it aliases the density grid.
    Real const invSamples = Real(1.) / Real(nSamplesPerCell);
    if ( needWeight )
    {
        if ( deferNorm )
        {
            if ( !weightIsDensity )
                quantities->mass_weight.swap( massWeight );   // sum(rho_s); normalized later by caller
            if ( dispNeedsOwnWeight )
            {   // the dispersion's mass-weighted mean/normalizer must survive the partition merge
                quantities->disp_weight.swap( dispWeight );
                quantities->disp_velocity.swap( dispVel );
            }
        }
        else
            for (size_t i = 0; i < totalGrid; ++i)
            {
                Real const w = weightIsDensity ? quantities->density[i] : massWeight[i];
                if ( w > Real(0.) )
                {
                    Real const inv = Real(1.) / w;
                    if (haveVel)                 quantities->velocity[i]          *= inv;   // <v>
                    if (field.velocity_gradient) quantities->velocity_gradient[i] *= inv;
#ifdef SCALAR
                    if (field.scalar)            quantities->scalar[i]            *= inv;
                    if (field.scalar_gradient)   quantities->scalar_gradient[i]   *= inv;
#endif
                    if (field.velocity_dispersion && !dispNeedsOwnWeight)
                    {
                        // sigma_ij = <v_i v_j> - <v_i><v_j>; quantities->velocity is now <v>.
                        // (Under --ps-volume-weighted the dispersion has its OWN mass-weighted
                        //  mean/normalizer and is closed out in the separate pass below.)
                        Pvector<Real,noVelComp> const &vbar = quantities->velocity[i];
                        size_t c = 0;
                        for (int a = 0; a < NO_DIM; ++a)
                            for (int b = a; b < NO_DIM; ++b)
                            {
                                Real s = quantities->velocity_dispersion[i][c] * inv - vbar[a]*vbar[b];
                                if (a == b and s < Real(0.)) s = Real(0.);   // variance: clamp FP-noise negatives
                                quantities->velocity_dispersion[i][c] = s;
                                ++c;
                            }
                    }
                }
            }

        // --ps-volume-weighted, serial path: close out the MASS-weighted dispersion against its
        // own mean/normalizer (the loop above normalized 'velocity' by the VOLUME weight, so it
        // is not the right mean here). Mirrors normalizePhaseSpace's second pass.
        if ( !deferNorm && dispNeedsOwnWeight )
            for (size_t i = 0; i < totalGrid; ++i)
            {
                Real const wm = dispWeight[i];
                if ( wm <= Real(0.) ) continue;
                Real const invm = Real(1.) / wm;
                Pvector<Real,noVelComp> const vbar = dispVel[i] * invm;   // <v>_mass
                size_t c = 0;
                for (int a = 0; a < NO_DIM; ++a)
                    for (int b = a; b < NO_DIM; ++b)
                    {
                        Real s = quantities->velocity_dispersion[i][c] * invm - vbar[a]*vbar[b];
                        if (a == b and s < Real(0.)) s = Real(0.);   // variance: clamp FP-noise negatives
                        quantities->velocity_dispersion[i][c] = s;
                        ++c;
                    }
            }
    }

    // density accumulated MASS (sum of per-tetrahedron mass shares); convert to a density by dividing
    // by the grid-cell volume, then normalize by averageDensity so the written field is rho/rho_bar
    // (mean-normalized "density contrast + 1", the SAME convention as standard DTFE).
    // Mass conservation makes the box mean exactly 1 for any nSub. Stream count is averaged over
    // the nSub^NO_DIM sub-samples below.
    if ( field.density )
    {
        Real const invRhoBar = userOptions.averageDensity > Real(0.)
                             ? Real(1.) / userOptions.averageDensity : Real(1.);
        for (size_t i = 0; i < totalGrid; ++i)
            quantities->density[i] *= invRhoBar / cellVolume;
    }

    // multi-stream / coverage statistics. Suppressed in the partition path (each partition covers
    // only part of the grid); DTFE.cpp reports one aggregate from the summed grid.
    if ( not userOptions.psSuppressGridStats )
    {
        Real maxStreams = Real(0.);
        size_t multiStreamCells = 0;
        size_t coveredCells = 0;
        for (size_t i = 0; i < totalGrid; ++i)
        {
            // mean streams over the cell. Under psExact 'streamCount' is the raw tet-TOUCH count
            // ('.tetTouch', up to hundreds), NOT a multiplicity -- the physical one is exactMult,
            // which is what '.streams' exports. Reading streamCount here reported 110 instead of
            // 3.0 and 100% multi-stream instead of 16.67% on the pancake.
            Real const relFrac = releasedMult.empty() ? Real(0.) : releasedMult[i];
            Real const avg = psExact ? exactMult[i] : Real(streamCount[i]) * invSamples + relFrac;
            if (avg > maxStreams) maxStreams = avg;
            // '> 1' to a tolerance: exactMult (and a released volume fraction) is a float sum, so
            // single-stream cells sit at 1.0 +/- eps; a pure sample count is an exact integer
            // multiple of invSamples and compares exactly.
            if ((psExact || relFrac > Real(0.)) ? (avg > Real(1.) + PS_STREAM_TOL)
                                            : (streamCount[i] > (int)nSamplesPerCell)) multiStreamCells++;
            if (streamCount[i] > 0 || relFrac > Real(0.)) coveredCells++;   // a stream reaches the cell
        }
        message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " Max streams at a grid point: "
                << MESSAGE::cMagenta() << maxStreams << MESSAGE::cReset()
                << ", grid points with multi-stream: " << MESSAGE::cMagenta() << multiStreamCells
                << " (" << (100.*multiStreamCells/totalGrid) << "\%)" << MESSAGE::cReset() << "\n" << MESSAGE::Flush;
        message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " grid coverage: "
                << MESSAGE::cMagenta() << coveredCells << "/" << totalGrid
                << " (" << (100.*coveredCells/totalGrid) << "\%)" << MESSAGE::cReset()
                << ( coveredCells < totalGrid ?
                     "  -- uncovered cells suggest insufficient padding or a grid finer than the tessellation" : "" )
                << "\n" << MESSAGE::Flush;
    }

    // Export the per-grid-point stream count: the sampled deposit's sample-mean multiplicity,
    // or -- under --ps-exact-deposit -- the ANALYTIC cell-mean multiplicity (1/V_cell) sum V_int,
    // its exact nSub->infinity limit. Both are the same physical quantity on the same scale, so
    // single-stream cells select the same way under either deposit -- but only TO A TOLERANCE
    // (PS_STREAM_TOL / dtfelib.STREAM_TOL): this is a float, never exactly 1.
    if ( psExact )
        quantities->stream_count.swap(exactMult);   // same type and exactMult is dead below: no copy
    else
    {
        quantities->stream_count.resize(totalGrid);
        for (size_t i = 0; i < totalGrid; ++i)
            quantities->stream_count[i] = Real(streamCount[i]) * invSamples;
        if ( not releasedMult.empty() )   // released (never sampled) tets: their volume fractions
            for (size_t i = 0; i < totalGrid; ++i)
                quantities->stream_count[i] += releasedMult[i];
    }

    // '.hidden_streams' bitmask (quantities.h): bit 1 = the exact multi-stream flag (a flipped tet
    // overlaps the cell; DTFE() clears it where '.streams' already reads multi-stream), bit 2 = the
    // mass flag (a never-sampled tet deposited here: the cell mean includes a centroid-fallback
    // piece). The degree-1 argument behind bit 1 needs the periodic box's complete sheet; a
    // non-periodic sheet has an edge whose image can overlap without folding, so there bit 1
    // keeps the conservative mass flag as well.
    quantities->hidden_streams.resize(totalGrid);
    bool const massIsMulti = !userOptions.periodic;
    for (size_t i = 0; i < totalGrid; ++i)
        quantities->hidden_streams[i] = Real( ((flippedFlag[i] || (massIsMulti && subSampleMassFlag[i])) ? 1 : 0)
                                        | (subSampleMassFlag[i] ? 2 : 0) );

    // --ps-exact-deposit: also export the raw integer tet-touch count ('.tetTouch'), the
    // number of tetrahedra with a nonzero intersection with the cell. Kept as a geometric
    // diagnostic (tessellation multiplicity, hundreds in resolved regions); it is NOT a
    // stream count and is not comparable to the sampled deposit's '.streams'.
    if ( psExact )
    {
        quantities->tet_touch.resize(totalGrid);
        for (size_t i = 0; i < totalGrid; ++i)
            quantities->tet_touch[i] = Real(streamCount[i]);
    }

    // Export the caustic orientation bits (0..3, exact in float); OR-merged across partitions
    // and binarized to the 0/1 fold flag once in DTFE() after the merge.
    if ( psCaustics )
    {
        quantities->caustic_bits.resize(totalGrid);
        for (size_t i = 0; i < totalGrid; ++i)
            quantities->caustic_bits[i] = Real(orientBits[i]);
    }

    // Record the sub-grid box so the caller (addFromSubgrid) can map results back into the full shared grid.
    for (int d = 0; d < NO_DIM; ++d)
    {
        quantities->ps_subOrigin[d] = subOrigin[d];
        quantities->ps_subDims[d]   = subDims[d];
    }
}

#endif // PHASE_SPACE
