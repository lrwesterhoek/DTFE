/* Phase-space DTFE grid interpolation. Triangulation is built in Lagrangian space;
   Eulerian simplices can overlap (multi-stream), so we iterate over all cells and
   accumulate each simplex's contribution to every grid point it contains. */

#ifdef PHASE_SPACE

/* 2D phase space (a DIM=2 build: PS-DTFE-2d, 'make PS-DTFE DIM=2').
 *
 * The Gadget-HDF5 reader reads a whole dataset into a buffer of N*NO_DIM values, so a 2D build must be
 * fed 2D data (a 3D file overruns the buffer). What the 2D path rests on:
 *   * 2D input: an HDF5 file whose Coordinates / Velocities / InitialCoordinates have TWO columns
 *     (tests/generate_ps_test_data.py --dim 2), read by the same reader;
 *   * the exact, tie-consistent point-in-TRIANGLE test (ps_exact_inside.h's 2D tiers), so every
 *     sample is counted by exactly one triangle of each stream;
 *   * the vertex degrees through CGAL's 2D face circulator, the beyond-grid sample count as a plain
 *     loop, and the exact deposit through ps_clip2d.h (triangle-cell clipping + polygon moments in
 *     r3d's moment layout), as is the exact folded-overlap flag of '.hidden_streams'.
 * Validated on the analytic 2D pancake (stream counts exact in every column off the caustics,
 * single-stream density converging 0.011 -> 0.004, velocity to 1e-4, the sampled deposit converging
 * to the exact one; mass exact under --ps-vertex-mass, to 2e-4 with the default Lagrangian-area
 * masses, which drop the few triangles squeezed flat on a caustic exactly as in 3D) and on 2D crossed
 * waves (streams {1,3,9} in the separable fractions): tests/ps_2d_check.sh. CGAL's 2D binary
 * triangulation I/O does not read back (tessellation_cache.h uses its ASCII mode in 2D).
 * Supported in 2D: the sampled and exact deposits, streams, the hidden-stream flags, velocity /
 * dispersion / gradient moments, --ps-vertex-mass, --ps-volume-weighted, --ps-halo-release,
 * --ps-caustics (parity + collapse multiplicity k = 0..2), --partition, periodic and non-periodic
 * boxes, the tessellation cache. 3D only: the GPU deposit, point evaluation
 * (--sample-points, --serve), --ps-window, --ps-caustic-cusps and --ps-linear-deposit's GPU path.
 */

#include "triangulation_common.h"

#include <algorithm> // sort/unique/nth_element for the deduplicated nearest-N cusp stencil
#include <atomic>    // the slab deposit's dynamic slab pickup
#include <chrono>    // timing of the GPU-deferred CPU deposits
#include <climits>   // LONG_MAX (markFlipped's default slab bounds)
#include <cstring>   // memcpy in the in-place cost-sort permutation
#include <exception> // the deposit threads hand their errors back
#include <iomanip>   // the phase-timing line
#include <sstream>   // the phase-timing line
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

// The deposit passes that asked for the GPU and ran on the CPU (ps_point_eval.h psGpuFallbacksTaken).
static std::atomic<int> psGpuFallbacks{ 0 };
int psGpuFallbacksTaken(bool reset)
{
    return reset ? psGpuFallbacks.exchange( 0 ) : psGpuFallbacks.load();
}

// Failed GPU deposit attempts of this run (run_progress.h: the progress sidecar's gpu_retries), counted by
// the GPU hosts (ps_metal_host.cc, ps_gpu_cuda.cu).
std::atomic<int> psGpuRetries{ 0 };
// ... and the same attempts by cause (the sidecar's gpu_retries_by_cause and the run's closing
// 'GPU deposit retries:' line): [0] innocent victim, [1] timeout (a watchdog kill), [2] out of memory, [3] any
// other cause; they sum to psGpuRetries (each host bumps both for every failed buffer / kernel).
std::atomic<int> psGpuRetriesByCause[4] = { { 0 }, { 0 }, { 0 }, { 0 } };

// Wall time of the phases inside 'field interpolation' -- the extraction walk, the cost sort, the GPU call, the
// GPU-deferred tets (whose line partitioned runs suppress), the normalization: the phase keeps roughly one core
// busy beside the GPU deposit, and this says where. Each pass prints its own split (at the verbosity of the '>>> Time:'
// lines, so a partitioned run prints it for the partitions its logging thread deposits), and the run's totals so far
// ride along, which with several concurrent partitions cover the silent ones too. Accumulated in microseconds over
// every pass of the run.
namespace {
enum PSPhase { PS_PH_SETUP, PS_PH_EXTRACT, PS_PH_SORT, PS_PH_GPU, PS_PH_COPY, PS_PH_DEFERRED, PS_PH_CPU, PS_PH_NORM, PS_PH_N };
char const* const psPhaseName[PS_PH_N] = { "setup", "extraction walk", "cost sort", "GPU deposit", "copy-back",
                                           "deferred tets on the CPU", "CPU deposit", "normalization" };
std::atomic<long long> psPhaseTotalUs[PS_PH_N];
std::atomic<long long> psPhaseTotalDeferred{ 0 };
std::atomic<int>       psPhasePasses{ 0 };
}
// The run's phase totals so far (seconds per phase in the order of psPhaseName, the deferred-tet count and the
// number of passes), for a caller that wants one closing line after a partitioned run's loop.
void psFieldPhaseTotals(double sec[8], char const* names[8], long long &deferred, int &passes)
{
    static_assert( PS_PH_N == 8, "psFieldPhaseTotals hands out eight phases" );
    for (int k = 0; k < PS_PH_N; ++k) { sec[k] = double( psPhaseTotalUs[k].load() ) * 1e-6; names[k] = psPhaseName[k]; }
    deferred = psPhaseTotalDeferred.load();
    passes = psPhasePasses.load();
}

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
// the GPU per-tet record (psFillTetRecord), and the fixed-point mass unit and shares (psMassFixedScale,
// psMassFixedShare) that the CPU deposit uses in every build, GPU or not
#include "ps_deposit_params.h"

// One share into the fixed-point MASS grid (see 'massFx' in interpolateGrid_phaseSpace): a plain add
// where one thread owns the cell (the serial loop, a slab), a relaxed atomic add where the threads of the copy mode
// share the grid. Integer adds commute, so the grid is the same exact sum whatever the interleaving.
static inline void psAddMassFixed(int64_t *grid, size_t const flat, int64_t const q, bool const atomic)
{
    if ( q == 0 ) return;
    if ( atomic ) __atomic_fetch_add( grid + flat, q, __ATOMIC_RELAXED );
    else grid[flat] += q;
}
// The deposit's order-free COUNTS and FLAGS, by the same rule: the int stream count (+1), the byte flags
// subSampleMassFlag / flippedFlag (set to 1) and the caustic mask (OR). The copy mode's threads share them as they share
// the fixed-point words -- +1, set and OR commute, so every thread count gives the same arrays and none needs a copy.
static inline void psAddCount(int &c, bool const atomic)
{
    if ( atomic ) __atomic_fetch_add( &c, 1, __ATOMIC_RELAXED );
    else ++c;
}
static inline void psSetFlag(unsigned char &f, bool const atomic)
{
    if ( atomic ) __atomic_store_n( &f, (unsigned char)1, __ATOMIC_RELAXED );
    else f = 1;
}
template <class Mask> static inline void psOrMask(Mask &m, Mask const bits, bool const atomic)
{
    if ( !atomic ) m |= bits;
    else if ( ( __atomic_load_n( &m, __ATOMIC_RELAXED ) & bits ) != bits )   // (mostly set already: skip the RMW)
        __atomic_fetch_or( &m, bits, __ATOMIC_RELAXED );
}

// The moment grids' fixed-point words ('fxg' in interpolateGrid_phaseSpace) and their Real grids: a
// cell element is a Pvector (component c) or a plain Real, and the words lie in the same layout, cell by cell.
template <class Elem> inline Real &psFxComp(Elem &e, size_t const c) { return e[c]; }
inline Real &psFxComp(Real &e, size_t) { return e; }
// The words of a grid become its Real grid, each component rounded once (the mass grid's conversion), and are freed.
template <class Elem>
inline void psFxToReal(std::vector<int64_t> &w, float const scale, std::vector<Elem> &grid, size_t const nCell)
{
    size_t const comps = sizeof(Elem) / sizeof(Real);
    grid.assign( nCell, Elem() );
    for (size_t i = 0; i < nCell; ++i)
        for (size_t c = 0; c < comps; ++c)
            psFxComp( grid[i], c ) = Real( psMassFixedMass( w[i*comps + c], scale ) );
    std::vector<int64_t>().swap( w );
}

// --ps-caustics: per-tetrahedron caustic stratification bits. Included UNCONDITIONALLY -- the
// classification is part of the CPU deposit and exists in 2D as well, so it must not sit behind
// the GPU/3D guard above (doing so breaks the CPU-only and the 2D builds).
#include "ps_caustic_class.h"

// exact, tie-consistent sample-in-tetrahedron test (shared with the point evaluation)
#include "ps_exact_inside.h"

// psDepositThreadBytes: the memory the deposit threads' private copies may take (budgeted there too)
#include "../auto_tune.h"

// PS_MASS_AUDIT=1: the deposit's mass bookkeeping (opt-in diagnostic)
#include "ps_mass_audit.h"

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
// written inline, its nested lambdas would capture the loop's hoisted locals by reference; as a function
// taking the loop's invD / vertex positions by reference, their addresses would escape and every in-grid
// sample test -- every tetrahedron's hot path -- would reload them from memory. Either costs the TNG
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

// The cell iteration of the pass and its phases: the finite faces (2D) / cells (3D) of the triangulation 'dt'.
#if NO_DIM==2
    #define PS_FOREACH_FINITE_CELL for (DT::Finite_faces_iterator itC = dt.finite_faces_begin(); itC != dt.finite_faces_end(); ++itC)
    #define PS_FINITE_CELLS_BEGIN dt.finite_faces_begin()
    #define PS_FINITE_CELLS_END dt.finite_faces_end()
#elif NO_DIM==3
    #define PS_FOREACH_FINITE_CELL for (DT::Finite_cells_iterator itC = dt.finite_cells_begin(); itC != dt.finite_cells_end(); ++itC)
    #define PS_FINITE_CELLS_BEGIN dt.finite_cells_begin()
    #define PS_FINITE_CELLS_END dt.finite_cells_end()
#endif

// The pass's triangulation, options, fields and grid geometry, as references, for the phases around its deposit.
struct PSPassCtx
{
    DT &dt;
    User_options &userOptions;
    Quantities *quantities;
    Field &field;
    MESSAGE::Message &message;
    Box &boxCoordinates;
    size_t const *nGrid;
    Real (&dx)[NO_DIM];
    int nSub;
    size_t noTotalCells;
    bool windowMode;
    size_t (&subOrigin)[NO_DIM];
    size_t (&subDims)[NO_DIM];
};

// The pass's accumulators, its per-tetrahedron scratch and the flags that select them, as references.
struct PSGridsCtx
{
    size_t totalGrid;
    std::vector<int> &streamCount;
    std::vector<Real> &releasedMult;
    std::vector<unsigned char> &subSampleMassFlag;
    std::vector<unsigned char> &flippedFlag;
    std::vector<Real> &exactMult;
    std::vector<PSCausticClass::CausticMask> &orientBits;
    std::vector<Real> &massWeight;
    std::vector<Real> &dispWeight;
    std::vector< Pvector<Real,noVelComp> > &dispVel;
    std::vector<int64_t> &massFx;
    float &massFxScale;
    bool &massFxOn;
    std::vector<size_t> &insideFlat;
    std::vector<Real> &insideRel, &insideW;
    std::vector<double> &exMom, &exW;
    bool const (&fxSel)[PS_FX_CPU_N];
    bool massFxWanted, fxMomWanted, haveVel, needWeight, weightIsDensity, dispNeedsOwnWeight, deferNorm, psExact, psCaustics;
};

// The CPU deposit's thread layout: the thread count, the slab mode and the slabs' tetrahedra, as references.
struct PSLayoutCtx
{
    int &nDepThreads;
    bool &slabMode;
    int &nSlabs;
    std::vector<long> &slabXLo, &slabXHi;
    std::vector< std::vector<uint32_t> > &slabTetLists;
    std::vector<Cell_handle> &slabTets;
    std::vector<uint8_t> &slabFirst;
};

// The pass's sub-grid box: the --ps-window box, or (psUseSubgrid) the Eulerian bbox of the partition's kept cells.
static inline void psSubgridBox(PSPassCtx const &passCtx, size_t &psKeptCells)
{
    DT &dt = passCtx.dt;   // (PS_FOREACH_FINITE_CELL)
    User_options &userOptions = passCtx.userOptions;
    Box &boxCoordinates = passCtx.boxCoordinates;
    size_t const *nGrid = passCtx.nGrid;
    Real (&dx)[NO_DIM] = passCtx.dx;
    int const nSub = passCtx.nSub;
    bool const windowMode = passCtx.windowMode;
    size_t (&subOrigin)[NO_DIM] = passCtx.subOrigin;
    size_t (&subDims)[NO_DIM] = passCtx.subDims;
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
                // full axis here would make every boundary-touching Lagrangian slab allocate
                // the whole grid. The footprint is really a WRAPPED arc, so crop to it: its complement is the
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
}

// The sample positions per axis, in double (samplePosAxis[d][g * nSub + k]), and the grid's lengths.
static inline void psSamplePositions(PSPassCtx const &passCtx, double (&gridLenD)[NO_DIM],
                                     std::vector<double> (&samplePosAxis)[NO_DIM])
{
    Box &boxCoordinates = passCtx.boxCoordinates;
    size_t const *nGrid = passCtx.nGrid;
    int const nSub = passCtx.nSub;
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
}

// PS_MASS_AUDIT: counts what the pass should deposit, before it deposits (a template: the audit is the pass's type).
template <class MassAudit, class FullDegreePass>
static inline void psMassAuditCount(PSPassCtx const &passCtx, bool const massAudit, bool const psVertexMass,
                                    bool &degreesLazy, FullDegreePass const &fullDegreePass, MassAudit &audit)
{
    DT &dt = passCtx.dt;
    User_options &userOptions = passCtx.userOptions;
    Box &boxCoordinates = passCtx.boxCoordinates;
    if ( massAudit )
    {
        if ( degreesLazy ) { fullDegreePass(); degreesLazy = false; }   // the same counts (PS_DEBUG_DEGREES)
        Box &own = userOptions.lagrangianRegion;
        bool const ownsAll = own.isNullBox();
        auto owned = [&](double const (&p)[NO_DIM]) -> bool   // psFilterCell's half-open ownership test
        {
            if ( ownsAll ) return true;
            for (int d = 0; d < NO_DIM; ++d)
                if ( p[d] < own[2*d] || p[d] >= own[2*d+1] ) return false;
            return true;
        };
        for (DT::Finite_vertices_iterator vit = dt.finite_vertices_begin(); vit != dt.finite_vertices_end(); ++vit)
        {
#ifdef TEST_PADDING
            if ( vit->info().isDummy() ) continue;
#endif
            double p[NO_DIM];
            for (int d = 0; d < NO_DIM; ++d) p[d] = double(vit->point()[d]);
            if ( !owned(p) ) continue;
            double const w = double(vit->info().weight());
            audit.owned += w; ++audit.nOwned;
            if ( psVertexMass && vit->info().psDegree() == 0 ) { audit.deg0 += w; ++audit.nDeg0; }
        }
        PS_FOREACH_FINITE_CELL
        {
            Cell_handle cell = itC;
            PSCellGeometry g;
            double Lag[NO_DIM][NO_DIM];
            for (int v = 0; v < NO_DIM; ++v)
                for (int i = 0; i < NO_DIM; ++i)
                    Lag[v][i] = double(cell->vertex(v+1)->point()[i]) - double(cell->vertex(0)->point()[i]);
            double const lagMass = double(userOptions.averageDensity) * std::fabs(determinant(Lag)) / factorial(NO_DIM);
            if ( psFilterCell(cell, userOptions, boxCoordinates, true, NULL, g) )
            {
                audit.tets += psVertexMass ? psVertexShareMass(cell) : lagMass;
                ++audit.nTets;
                if ( psVertexMass )
                    for (int v = 0; v <= NO_DIM; ++v)
                        if ( cell->vertex(v)->info().psDegree() == 0 ) { ++audit.nTetsDeg0; break; }   // must stay 0
                continue;
            }
            bool dummy = false;
#ifdef TEST_PADDING
            for (int v = 0; v <= NO_DIM; ++v) dummy = dummy || cell->vertex(v)->info().isDummy();
#endif
            double cen[NO_DIM];
            for (int d = 0; d < NO_DIM; ++d)
            {
                cen[d] = 0.;
                for (int v = 0; v <= NO_DIM; ++v) cen[d] += double(cell->vertex(v)->point()[d]);
                cen[d] /= double(NO_DIM + 1);
            }
            if ( !dummy && owned(cen) ) { audit.dropped += lagMass; ++audit.nDropped; }
        }
        // PS_MASS_AUDIT=2 on one domain: the per-particle closure -- a particle's shares over the kept tets
        // of ALL its copies (periodic images) must add up to its weight; the worst offenders are listed
        if ( PSMassAudit::level() >= 2 && PSMassAudit::partition < 0 && psVertexMass )
        {
            std::unordered_map<uint64_t, double> share;
            std::unordered_map<uint64_t, std::vector<Vertex_handle> > copies;
            for (DT::Finite_vertices_iterator vit = dt.finite_vertices_begin(); vit != dt.finite_vertices_end(); ++vit)
            {
#ifdef TEST_PADDING
                if ( vit->info().isDummy() ) continue;
#endif
                copies[ vit->info().particleID() ].push_back( vit );
            }
            PS_FOREACH_FINITE_CELL
            {
                Cell_handle cell = itC;
                PSCellGeometry g;
                if ( !psFilterCell(cell, userOptions, boxCoordinates, true, NULL, g) ) continue;
                for (int v = 0; v <= NO_DIM; ++v)
                {
                    int32_t const deg = cell->vertex(v)->info().psDegree();
                    if ( deg > 0 ) share[ cell->vertex(v)->info().particleID() ] += double(cell->vertex(v)->info().weight()) / double(deg);
                }
            }
            std::vector< std::pair<double, uint64_t> > bad;
            double net = 0.;
            for (auto const &c : copies)
            {
                double const w = double( c.second[0]->info().weight() );
                double const dev = share[c.first] - w;
                if ( std::fabs(dev) > 1e-9 * w ) { bad.emplace_back( dev, c.first ); net += dev; }
            }
            std::sort( bad.begin(), bad.end(), [](auto const &a, auto const &b) { return std::fabs(a.first) > std::fabs(b.first); } );
            std::ostringstream s;
            s << std::setprecision(9) << "PS mass audit: per-particle closure: " << bad.size() << " of " << copies.size()
              << " particles off (net " << net << ")";
            for (size_t i = 0; i < bad.size() && i < 6; ++i)
            {
                s << "\n    id " << bad[i].second << " off " << bad[i].first << ", copies (Lagrangian position: degree):";
                for (Vertex_handle const &vh : copies[bad[i].second])
                {
                    s << " (";
                    for (int d = 0; d < NO_DIM; ++d) s << vh->point()[d] << (d + 1 < NO_DIM ? " " : ": ");
                    s << vh->info().psDegree() << ")";
                }
            }
            std::fprintf( stderr, "%s\n", s.str().c_str() );
        }
    }
}

// Allocates the moment grids' Real accumulators of a float-sum pass, or releases them for the fixed-point words.
static inline void psAllocMoments(PSPassCtx const &passCtx, bool const haveVel, bool const fxMomWanted, size_t const totalGrid)
{
    Quantities *quantities = passCtx.quantities;
    Field &field = passCtx.field;
    // (fxMomWanted: allocated when the pass knows whether each one is a fixed-point sum -- fxEnsureReal)
    if ( haveVel && !fxMomWanted )
        quantities->velocity.assign(totalGrid, Pvector<Real,noVelComp>::zero());
    if ( field.velocity_gradient && !fxMomWanted )
        quantities->velocity_gradient.assign(totalGrid, Pvector<Real,noGradComp>::zero());
    if ( field.velocity_dispersion && !fxMomWanted )
        quantities->velocity_dispersion.assign(totalGrid, Pvector<Real,noDispComp>::zero());
#ifdef SCALAR
    if ( field.scalar && !fxMomWanted )
        quantities->scalar.assign(totalGrid, Pvector<Real,noScalarComp>::zero());
    if ( field.scalar_gradient && !fxMomWanted )
        quantities->scalar_gradient.assign(totalGrid, Pvector<Real,noScalarGradComp>::zero());
#endif
    if ( fxMomWanted )
    {   // a grid left over from an earlier use of 'quantities' is released here, as the density is above
        std::vector< Pvector<Real,noVelComp> >().swap( quantities->velocity );
        std::vector< Pvector<Real,noGradComp> >().swap( quantities->velocity_gradient );
        std::vector< Pvector<Real,noDispComp> >().swap( quantities->velocity_dispersion );
#ifdef SCALAR
        std::vector< Pvector<Real,noScalarComp> >().swap( quantities->scalar );
        std::vector< Pvector<Real,noScalarGradComp> >().swap( quantities->scalar_gradient );
#endif
    }
}

#if defined(PS_GPU) && NO_DIM==3
// Clears tryMetal (with a warning) for a pass the GPU cannot deposit: cusps, a multi-component scalar, 2^32 cells.
static inline void psGpuEligible(PSPassCtx const &passCtx, size_t const subTotal, bool &tryMetal)
{
    User_options &userOptions = passCtx.userOptions;
#ifdef SCALAR
    Field &field = passCtx.field;
#endif
    size_t const noTotalCells = passCtx.noTotalCells;
    // The A3/A4 indicators of --ps-caustic-cusps fit each near-fold tetrahedron over its vertex ring (a CGAL
    // neighbour gather), which only the CPU deposit has: a GPU deposit would leave bits 7 and 8 unset in every cell
    // it fills (88 of 11642 cusp cells set on a crossed-wave test). So such a run deposits on the CPU.
    if ( tryMetal && userOptions.psCausticCusps )
    {
        static std::atomic<bool> told{ false };
        if ( !told.exchange( true ) )
        {
            MESSAGE::Warning warning( userOptions.verboseLevel );
            warning << "--ps-caustic-cusps needs the CPU deposit (its stencil gathers the triangulation's neighbours); "
                    << "--ps-gpu is not used for the deposit of this run.\n" << MESSAGE::EndWarning;
        }
        tryMetal = false;
    }
#ifdef SCALAR
    if ( tryMetal && (field.scalar || field.scalar_gradient) && noScalarComp != 1 )
    {
        MESSAGE::Warning warning( userOptions.verboseLevel );
        warning << "--ps-gpu supports ONE scalar component (this build has NO_SCALARS=" << noScalarComp << "); using the CPU deposit for this pass.\n" << MESSAGE::EndWarning;
        tryMetal = false;
        ++psGpuFallbacks;
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
        ++psGpuFallbacks;
    }
    if ( tryMetal and subTotal > size_t(UINT32_MAX) )
    {
        MESSAGE::Warning warning( userOptions.verboseLevel );
        warning << "--ps-gpu does not support more than 2^32 cells per partition sub-grid (this one has "
                << subTotal << ", i.e. > " << size_t(UINT32_MAX) << "); using the CPU deposit for this pass. "
                << "A finer --partition would bring the sub-grid back under the cap.\n" << MESSAGE::EndWarning;
        tryMetal = false;
        ++psGpuFallbacks;
    }
}
#endif

// PS_DEBUG_DEGREES: checks the lazily computed vertex degrees of a GPU --ps-window pass against the full pass.
static inline void psDebugDegreesCheck(PSPassCtx const &passCtx, bool const degreesLazy, bool const metalDeposited)
{
    DT &dt = passCtx.dt;
    User_options &userOptions = passCtx.userOptions;
    Box &boxCoordinates = passCtx.boxCoordinates;
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
}

// The CPU deposit's fixed-point units: the mass accumulator and the moment grids' units (a template: the pass's types).
template <class FxWords, class BoundWalk, class FxUnit, class FxStart, class FxEnsureReal, class FxLogUnits>
static inline void psCpuFixedUnits(PSPassCtx const &passCtx, PSGridsCtx const &gridsCtx, bool const metalDeposited,
                                   FxWords (&fxg)[PS_FX_CPU_N], BoundWalk const &fxBoundWalk, FxUnit const &fxUnit,
                                   FxStart const &fxStart, FxEnsureReal const &fxEnsureReal, FxLogUnits const &fxLogUnits)
{
    DT &dt = passCtx.dt;
    User_options &userOptions = passCtx.userOptions;
    Quantities *quantities = passCtx.quantities;
    Field &field = passCtx.field;
    bool const windowMode = passCtx.windowMode;
    size_t const totalGrid = gridsCtx.totalGrid;
    std::vector<int64_t> &massFx = gridsCtx.massFx;
    float &massFxScale = gridsCtx.massFxScale;
    bool &massFxOn = gridsCtx.massFxOn;
    bool const (&fxSel)[PS_FX_CPU_N] = gridsCtx.fxSel;
    bool const massFxWanted = gridsCtx.massFxWanted, fxMomWanted = gridsCtx.fxMomWanted;
    PSFixedBounds walkBounds;
    bool walked = false;

    // The CPU deposits the whole pass (no GPU, or the GPU declined or gave up): its mass accumulator is allocated
    // here, the density itself (the double build, or the float A/B grid) or the fixed-point grid from zero (massFx).
    // Its unit must hold the pass's mass before the loop sees a tet, so it comes from an upper bound found in one walk
    // over the vertices: under --ps-vertex-mass the kept tets hold at most the vertices' total weight (the degrees
    // count every kept tet); otherwise at most rho_bar times the vertices' Lagrangian bounding box (the cells tile
    // their hull). The larger of the two, over by a factor of a few (padding, periodic copies): a bit or two of the
    // 2^61..2^62 units. Deterministic (the triangulation's vertex order). --ps-window: the window's own total, B[DW]
    // of the bound walk, which a GPU window deposit takes too (PSFixedUnits.massTotal): both deposit the same integers.
    if ( !metalDeposited && field.density && !massFxWanted )
        quantities->density.assign( totalGrid, Real(0.) );   // the float (double build: double) accumulator
    else if ( !metalDeposited && field.density )
    {
        double bound = 0.;
        if ( windowMode )
        {
            walkBounds = fxBoundWalk();
            walked = true;
            bound = walkBounds.B[PS_FX_DW];
        }
        else
        {
            double wSum = 0., lagLo[NO_DIM], lagHi[NO_DIM];
            for (int d = 0; d < NO_DIM; ++d) { lagLo[d] = std::numeric_limits<double>::max(); lagHi[d] = -lagLo[d]; }
            for (DT::Finite_vertices_iterator vit = dt.finite_vertices_begin(); vit != dt.finite_vertices_end(); ++vit)
            {
#ifdef TEST_PADDING
                if ( vit->info().isDummy() ) continue;   // the filters drop every tet that touches one
#endif
                wSum += std::fabs( double(vit->info().weight()) );
                for (int d = 0; d < NO_DIM; ++d)
                {
                    double const p = double( vit->point()[d] );
                    lagLo[d] = std::min( lagLo[d], p );
                    lagHi[d] = std::max( lagHi[d], p );
                }
            }
            double lagVolume = 1.;
            for (int d = 0; d < NO_DIM; ++d) lagVolume *= std::max( 0., lagHi[d] - lagLo[d] );
            bound = std::max( wSum, std::fabs( double(userOptions.averageDensity) ) * lagVolume );
        }
        float scale = 0.f;
        if ( psMassFixedScale( bound, scale ) )
        {
            massFx.assign( totalGrid, 0 );
            massFxScale = scale;
            massFxOn = true;
        }
        else
            quantities->density.assign( totalGrid, Real(0.) );   // (no vertices / no mass: the float grid)
    }

    // The CPU deposits the whole pass and the moment grids are to be fixed-point sums (fxg): their units
    // need bounds of what the loop will deposit, before it deposits. No cheap vertex walk bounds the gradients or the
    // Eulerian volumes, so they come from the bound walk above (fxBoundWalk: under --ps-window the window's tets
    // only, as for the mass). DW is not accumulated when the mass grid is this pass's own
    // fixed-point words (it becomes the density after the loop).
    if ( !metalDeposited && fxMomWanted )
    {
        bool any = false;
        for (int g = 0; g < PS_FX_CPU_N; ++g) any = any || ( fxSel[g] && !( g == PS_FX_DW && massFxOn ) );
        if ( any )
        {
            PSFixedBounds const fb = walked ? walkBounds : fxBoundWalk();
            for (int g = 0; g < PS_FX_CPU_N; ++g)
            {
                if ( !fxSel[g] || ( g == PS_FX_DW && massFxOn ) ) continue;
                float const scale = fxUnit( g, fb.B[g] );
                if ( scale > 0.f ) fxStart( g, scale );
                else fxEnsureReal( g );
            }
        }
        else
            for (int g = 0; g < PS_FX_CPU_N; ++g)
                if ( !( g == PS_FX_DW && massFxOn ) ) fxEnsureReal( g );   // (nothing selected: a no-op)
    }
    if ( !metalDeposited )
    {   // (--verbose 3) the CPU deposit's units, as the GPU's are logged after its deposit
        float sc[PS_FX_N];
        for (int g = 0; g < PS_FX_N; ++g) sc[g] = fxg[g].w.empty() ? 0.f : fxg[g].scale;
        fxLogUnits( "CPU", massFxOn ? massFxScale : 0.f, sc );
    }
}

// The CPU deposit's thread count and mode (private copies, or slabs of the x axis), and the copy mode's line.
static inline void psDepositThreadMode(PSPassCtx const &passCtx, PSLayoutCtx const &layoutCtx, bool const metalDeposited,
                                       double const accBytes, std::vector<uint32_t> &deferredOrd)
{
    User_options &userOptions = passCtx.userOptions;
    MESSAGE::Message &message = passCtx.message;
    size_t const noTotalCells = passCtx.noTotalCells;
    size_t (&subDims)[NO_DIM] = passCtx.subDims;
    int &nDepThreads = layoutCtx.nDepThreads;
    bool &slabMode = layoutCtx.slabMode;
    if ( !metalDeposited && !userOptions.psCausticCusps )
    {
        int const wantThreads = psDepositThreads();
        nDepThreads = wantThreads;
        double const extra = psDepositThreadBytes();
        if ( accBytes > 0. )   // (the quotient capped before the int conversion: a few cells' copy is a few bytes)
            nDepThreads = std::max( 1, std::min( nDepThreads, 1 + int( std::min( double(nDepThreads), extra / accBytes ) ) ) );
        size_t const blocks = noTotalCells / 256 + 1;                      // at least a few blocks per thread (DEP_BLOCK)
        int const blockCap = std::max( 1, int( blocks / 4 ) + 1 );
        nDepThreads = std::max( 1, std::min( nDepThreads, blockCap ) );
        int const slabThreads = std::max( 1, std::min( wantThreads, blockCap ) );
        char const *slabEnv = std::getenv("DTFE_DEPOSIT_SLABS");
        int const slabSetting = slabEnv ? std::atoi(slabEnv) : -1;     // 1 force, 0 never, else automatic
        // automatic: only when the copies leave fewer than HALF the threads -- a slab run classifies a
        // straddling tet once per slab (~1.6x the work): on a clustered TNG sub-cube at 512^3 (2 cells per
        // tet, copies for 5 of 10 threads) 10 slab threads take 46.8 s, the 5 copy threads 42.5 s; on 768^3
        // (copies for 2) slabs take 28 s instead of 64. At 256^3 (copies fit) 10 copy threads are 2x the slab
        // speed. The fixed-point grids, the counts and the flags are shared (psAddCount), so the default
        // single build copies nothing and this switch only meets the float sums (the double build,
        // PS_METAL_FLOAT_MOMENTS=1) on fine grids.
        if ( slabSetting != 0 and slabThreads > 1 and subDims[0] >= 2
             and ( slabSetting == 1 or 2 * nDepThreads < slabThreads ) and noTotalCells < size_t(UINT32_MAX) )
        {
            slabMode = true;
            nDepThreads = slabThreads;
        }
    }
    // The GPU-DEFERRED tetrahedra run in slab mode too, on the deferred cells only: one thread walking every
    // finite cell would outlast the GPU deposit itself (48 s against 27 s on 64^3 crossed waves at 256^3: 194154
    // deferred tets, each a large tet whose samples all take the exact double test). Slab mode writes the real arrays
    // in the single-threaded loop's order, so the grids stay bit-identical to the one-thread loop (DTFE_DEPOSIT_SLABS=0
    // keeps that loop). A few hundred deferred tets are not worth the binning: below DEFERRED_SLAB_MIN the loop stays
    // serial.
    size_t const DEFERRED_SLAB_MIN = 2048;
    if ( metalDeposited && deferredOrd.size() >= DEFERRED_SLAB_MIN && !userOptions.psCausticCusps
         && subDims[0] >= 2 && noTotalCells < size_t(UINT32_MAX) )
    {
        char const *slabEnv = std::getenv("DTFE_DEPOSIT_SLABS");
        int const slabThreads = std::max( 1, std::min( psDepositThreads(), int( deferredOrd.size() / 1024 ) + 1 ) );
        if ( !( slabEnv && std::atoi(slabEnv) == 0 ) && slabThreads > 1 )
        {
            slabMode = true;
            nDepThreads = slabThreads;
        }
    }
    if ( !metalDeposited && !slabMode && ( not userOptions.psSuppressGridStats or userOptions.verboseLevel >= 3 ) )
        message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " CPU deposit on " << nDepThreads
                << ( nDepThreads == 1 ? " thread" : " threads" ) << " (copy mode: " << accBytes / 1.e6
                << " MB of private accumulators per extra thread, budget " << psDepositThreadBytes() / 1.e9
                << " GB; the fixed-point grids, counts and flags are shared).\n" << MESSAGE::Flush;
}

// Slab mode: bins the tetrahedra into slabs of local x planes cut at equal work, and prints the slab line.
static inline void psSlabBinning(PSPassCtx const &passCtx, PSLayoutCtx const &layoutCtx, bool const metalDeposited,
                                 std::vector<uint32_t> &deferredOrd)
{
    DT &dt = passCtx.dt;   // (PS_FINITE_CELLS_BEGIN / END)
    User_options &userOptions = passCtx.userOptions;
    MESSAGE::Message &message = passCtx.message;
    Box &boxCoordinates = passCtx.boxCoordinates;
    size_t const *nGrid = passCtx.nGrid;
    Real (&dx)[NO_DIM] = passCtx.dx;
    size_t const noTotalCells = passCtx.noTotalCells;
    size_t (&subOrigin)[NO_DIM] = passCtx.subOrigin;
    size_t (&subDims)[NO_DIM] = passCtx.subDims;
    bool const slabMode = layoutCtx.slabMode;
    int const nDepThreads = layoutCtx.nDepThreads;
    int &nSlabs = layoutCtx.nSlabs;
    std::vector<long> &slabXLo = layoutCtx.slabXLo, &slabXHi = layoutCtx.slabXHi;
    std::vector< std::vector<uint32_t> > &slabTetLists = layoutCtx.slabTetLists;
    std::vector<Cell_handle> &slabTets = layoutCtx.slabTets;
    std::vector<uint8_t> &slabFirst = layoutCtx.slabFirst;
    if ( slabMode )
    {
        // binning: each tetrahedron's x range of global cells -- the bounding box of its Eulerian
        // vertices, minimum-image wrapped EXACTLY as psFilterCell wraps them (same Real operations, so the
        // same positions), widened by two cells: a superset of every cell the deposit writes (the sample
        // window is the bounding box itself, the centroid fallback lies inside it). No
        // filter here: running it once more per tetrahedron would cost a full extra pass (a third of the slab
        // run on a clustered box, where a tet covers ~2 cells); the deposit filters, and counts a
        // rejection in the tet's first slab. After a GPU deposit: only the deferred cells, in loop order.
        if ( metalDeposited )
        {
            slabTets.reserve( deferredOrd.size() );
            size_t ord = 0, next = 0;
            for (auto it = PS_FINITE_CELLS_BEGIN; it != PS_FINITE_CELLS_END && next < deferredOrd.size(); ++it, ++ord)
                if ( size_t(deferredOrd[next]) == ord ) { slabTets.push_back( it ); ++next; }
        }
        else
        {
            slabTets.reserve( noTotalCells );
            for (auto it = PS_FINITE_CELLS_BEGIN; it != PS_FINITE_CELLS_END; ++it) slabTets.push_back( it );
        }
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
        // planes of a clustered box, where equal-width slabs would leave the thread holding them alone (TNG
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
                // balance does not see (a TNG --box region slows by 7% when they pile up there)
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
            message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset()
                    << ( metalDeposited ? " GPU-deferred tetrahedra on the CPU in SLAB mode: " : " CPU deposit in SLAB mode: " )
                    << nDepThreads << " threads over " << nSlabs << " slabs of the x axis"
                    << ( metalDeposited ? ", " : " (the accumulator copies would not fit), " )
                    << ( kept ? double(entries) / double(kept) : 0. )
                    << " slabs per tetrahedron (mean extent " << meanExtent << " cells).\n" << MESSAGE::Flush;
    }
}

// The deposit threads' accumulators: the real arrays for thread 0 (all threads in slab mode), else zeroed copies.
template <class DepositThread>
static inline void psDepositThreadsInit(PSPassCtx const &passCtx, PSGridsCtx const &gridsCtx,
                                        std::vector<DepositThread> &depT, int const nDepThreads, bool const slabMode)
{
    Quantities *quantities = passCtx.quantities;
    std::vector<int> &streamCount = gridsCtx.streamCount;
    std::vector<Real> &releasedMult = gridsCtx.releasedMult;
    std::vector<unsigned char> &subSampleMassFlag = gridsCtx.subSampleMassFlag;
    std::vector<unsigned char> &flippedFlag = gridsCtx.flippedFlag;
    std::vector<Real> &exactMult = gridsCtx.exactMult;
    std::vector<PSCausticClass::CausticMask> &orientBits = gridsCtx.orientBits;
    std::vector<Real> &massWeight = gridsCtx.massWeight;
    std::vector<Real> &dispWeight = gridsCtx.dispWeight;
    std::vector< Pvector<Real,noVelComp> > &dispVel = gridsCtx.dispVel;
    std::vector<int64_t> &massFx = gridsCtx.massFx;
    bool &massFxOn = gridsCtx.massFxOn;
    std::vector<size_t> &insideFlat = gridsCtx.insideFlat;
    std::vector<Real> &insideRel = gridsCtx.insideRel, &insideW = gridsCtx.insideW;
    std::vector<double> &exMom = gridsCtx.exMom, &exW = gridsCtx.exW;
    for (int th = 0; th < nDepThreads; ++th)
    {
        DepositThread &D = depT[size_t(th)];
        D.massFx = massFxOn ? massFx.data() : nullptr;   // shared by all (its own density copy stays EMPTY)
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
        zeroLike( D.ownReleasedMult, releasedMult );       zeroLike( D.ownExactMult, exactMult );
        zeroLike( D.ownDispWeight, dispWeight );           zeroLike( D.ownDispVel, dispVel );
        zeroLike( D.ownMassWeight, massWeight );
        D.q = &D.ownQ; D.releasedMult = &D.ownReleasedMult;
        D.exactMult = &D.ownExactMult; D.dispWeight = &D.ownDispWeight; D.dispVel = &D.ownDispVel;
        D.massWeight = &D.ownMassWeight;
        // shared by every thread, written with relaxed atomics (countAtomicL in depositLoop)
        D.streamCount = &streamCount;
        D.orientBits = &orientBits; D.subSampleMassFlag = &subSampleMassFlag; D.flippedFlag = &flippedFlag;
    }
}

// The fixed-point words become their Real grids, each cell rounded once: the moment grids', then the density.
template <class FxToReal>
static inline void psFixedToReal(PSPassCtx const &passCtx, PSGridsCtx const &gridsCtx, FxToReal const &fxToReal)
{
    Quantities *quantities = passCtx.quantities;
    size_t const totalGrid = gridsCtx.totalGrid;
    std::vector<int64_t> &massFx = gridsCtx.massFx;
    float &massFxScale = gridsCtx.massFxScale;
    bool &massFxOn = gridsCtx.massFxOn;
    bool const dispNeedsOwnWeight = gridsCtx.dispNeedsOwnWeight;
    std::vector<Real> &dispWeight = gridsCtx.dispWeight;
    // The fixed-point mass grid becomes the float density, once (see massFx): every share is in it,
    // and the normalizations below read the density (the moments' weight under weightIsDensity, then rho/rho_bar).
    // Each cell is rounded once, to nearest. Here, after the private copies, the slab lists and (on the final
    // call) the triangulation are freed, the 4 B float grid and the 8 B words briefly coexist.
    // The moment grids' words become their Real grids the same way, grid by grid (fxg): each cell
    // component rounded once, the words freed before the next grid is converted.
    for (int g = 0; g < PS_FX_CPU_N; ++g) fxToReal( g );
    if ( massFxOn )
    {
        quantities->density.assign( totalGrid, Real(0.) );
        for (size_t i = 0; i < totalGrid; ++i)
            quantities->density[i] = Real( psMassFixedMass( massFx[i], massFxScale ) );
        std::vector<int64_t>().swap( massFx );
        massFxOn = false;
        // --ps-volume-weighted: the dispersion's own mass normalizer (dispWeight) sums the very shares this grid
        // sums, at the same sites (the GPU's dispw likewise), so it takes the exact sums too. The default run's
        // dispersion is normalized by this grid (weightIsDensity), and the volume-weighted one then stays the
        // same mass-weighted arithmetic bit for bit (tests/ps_volume_weighted_check.sh B) -- a float sum in
        // dispWeight would part from it at the last bit.
        // (dispWeight is not even accumulated then -- it may be EMPTY here: dwOffL in the loop)
        if ( dispNeedsOwnWeight )
            dispWeight = quantities->density;
    }
}

// The pass's closing lines: done, the degenerate cells dropped, the halo tetrahedra released.
static inline void psPassMessages(PSPassCtx const &passCtx, size_t const nDegenerateInverse, size_t const nReleased)
{
    User_options &userOptions = passCtx.userOptions;
    MESSAGE::Message &message = passCtx.message;
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
}

// The moments become means (divided by their weight, the dispersion closed out); deferred: the weights handed on.
static inline void psNormalizeMoments(PSPassCtx const &passCtx, PSGridsCtx const &gridsCtx)
{
    Quantities *quantities = passCtx.quantities;
    Field &field = passCtx.field;
    size_t const totalGrid = gridsCtx.totalGrid;
    std::vector<Real> &massWeight = gridsCtx.massWeight;
    std::vector<Real> &dispWeight = gridsCtx.dispWeight;
    std::vector< Pvector<Real,noVelComp> > &dispVel = gridsCtx.dispVel;
    bool const haveVel = gridsCtx.haveVel, needWeight = gridsCtx.needWeight, deferNorm = gridsCtx.deferNorm;
    bool const weightIsDensity = gridsCtx.weightIsDensity, dispNeedsOwnWeight = gridsCtx.dispNeedsOwnWeight;
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
}

// PS_MASS_AUDIT: the pass's audit line and the run's totals (a template: the audit and gridSum are the pass's types).
template <class MassAudit, class GridSum>
static inline void psMassAuditLine(PSPassCtx const &passCtx, bool const massAudit, MassAudit &audit,
                                   GridSum const &gridSum, Real const cellVolume, double const auditGridPre)
{
    User_options &userOptions = passCtx.userOptions;
    Field &field = passCtx.field;
    int const nSub = passCtx.nSub;
    if ( massAudit )
    {   // (g) the pass's grid in mass units, before and after the rho/rho_bar conversion; the line (ps_mass_audit.h)
        double const gridPost = field.density ? gridSum() * double(cellVolume) * double(userOptions.averageDensity) : -1.;
        auto rel = [](double x, double ref) { return ref != 0. ? x / ref - 1. : 0.; };
        std::ostringstream s;
        s << std::setprecision(12) << "PS mass audit: partition ";
        if ( PSMassAudit::partition >= 0 ) s << PSMassAudit::partition; else s << "- (one domain)";
        s << " (nSub " << nSub << (&field == &userOptions.aField ? ", averaged" : ", unaveraged") << " pass):"
          << " (a) owned " << audit.owned << " [" << audit.nOwned << " vertices]"
          << " (b) tets " << audit.tets << " [" << audit.nTets << "]"
          << " (c) deg0 " << audit.deg0 << " [" << audit.nDeg0 << " vertices; kept tets touching one " << audit.nTetsDeg0 << "]"
          << " (d) dropped " << audit.nDropped << " owned tets (rho_bar V_lag " << audit.dropped << "), after the filter "
          << audit.after << " [" << audit.nAfter << "]";
        if ( audit.gpuHanded >= 0. )
            s << " (e) gpu handed " << audit.gpuHanded << " deferred " << audit.gpuDeferred << " fixed-point " << audit.gpuFixed
              << " (handed-deferred-fixed " << ( audit.gpuHanded - audit.gpuDeferred - audit.gpuFixed ) << ")"
              << " float grid " << audit.gpuFloat << " (rel " << rel(audit.gpuFloat, audit.gpuFixed) << ")"
              << " released " << audit.released << " gained " << audit.releasedGain;
        s << " (f) cpu tets " << audit.cpuTets << " [" << audit.nCpuTets << "] shares " << audit.cpuShares
          << " gained " << audit.cpuGain << " (shares/tets-1 " << rel(audit.cpuShares, audit.cpuTets)
          << ", gained/shares-1 " << rel(audit.cpuGain, audit.cpuShares) << ")"
          << " (g) grid " << auditGridPre << " post " << gridPost << " (grid/tets-1 " << rel(auditGridPre, audit.tets)
          << ", post/grid-1 " << rel(gridPost, auditGridPre) << ", grid/owned-1 " << rel(auditGridPre, audit.owned) << ")\n";
        std::fprintf( stderr, "%s", s.str().c_str() );
        std::lock_guard<std::mutex> lock( PSMassAudit::mutex );
        PSMassAudit::Totals &T = PSMassAudit::totals[&field == &userOptions.aField ? 1 : 0];
        T.owned += audit.owned; T.tets += audit.tets; T.deg0 += audit.deg0; T.dropped += audit.dropped;
        T.gridPre += auditGridPre; T.gridPost += gridPost; ++T.passes;
    }
}

// The pass's multi-stream and coverage statistics.
static inline void psStreamStats(PSPassCtx const &passCtx, PSGridsCtx const &gridsCtx, Real const invSamples,
                                 size_t const nSamplesPerCell)
{
    User_options &userOptions = passCtx.userOptions;
    MESSAGE::Message &message = passCtx.message;
    size_t const totalGrid = gridsCtx.totalGrid;
    std::vector<int> &streamCount = gridsCtx.streamCount;
    std::vector<Real> &releasedMult = gridsCtx.releasedMult;
    std::vector<Real> &exactMult = gridsCtx.exactMult;
    bool const psExact = gridsCtx.psExact;
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
            // which is what '.streams' exports. Reading streamCount here would report 110 instead of
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
}

// Exports the per-cell stream fields: '.streams', '.hidden_streams', '.tetTouch' and the caustic bits.
static inline void psExportStreams(PSPassCtx const &passCtx, PSGridsCtx const &gridsCtx, Real const invSamples)
{
    User_options &userOptions = passCtx.userOptions;
    Quantities *quantities = passCtx.quantities;
    size_t const totalGrid = gridsCtx.totalGrid;
    std::vector<int> &streamCount = gridsCtx.streamCount;
    std::vector<Real> &releasedMult = gridsCtx.releasedMult;
    std::vector<unsigned char> &subSampleMassFlag = gridsCtx.subSampleMassFlag;
    std::vector<unsigned char> &flippedFlag = gridsCtx.flippedFlag;
    std::vector<Real> &exactMult = gridsCtx.exactMult;
    std::vector<PSCausticClass::CausticMask> &orientBits = gridsCtx.orientBits;
    bool const psExact = gridsCtx.psExact, psCaustics = gridsCtx.psCaustics;
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
    /* The parity bit 1 (det < 0) is the GEOMETRIC overlap flag.
       orientBits gets its bits only where a tetrahedron DEPOSITS A SAMPLE, so a fold thinner than
       the sub-sample spacing -- every young wall, and the edge of every older one -- sets no
       negative-parity bit in the cells it crosses: per sample, '.caustic' would mark 21.0% of the
       cells of TNG100-3-Dark z=1 where folded tetrahedra overlap 33.6% (about 2x low at z >= 3),
       with a fraction depending on nSub, although the '--ps-caustics' help, ps_caustic_class.h and
       dtfelib promise "a det < 0 tetrahedron OVERLAPS the cell". flippedFlag is exactly that
       statement: the separating-axis tet-box test (TetBoxSAT; 2D: the triangle clip) of every
       folded tetrahedron, marked in both deposit paths (the host loop marks the GPU's tets) and
       still RAW here -- Quantities::finalizeHiddenStreams clears it in multi-stream cells only
       after the merge, and only in '.hidden_streams'. OR-ing it in keeps the export partition-,
       thread- and backend-invariant (OR of two OR-merged geometric masks).
       Only bit 1 comes from the geometry. Bit 0 (det > 0) stays per sample: in a periodic box the
       map has degree 1, so every point of the cell lies in a positive tetrahedron and any sample
       of the cell sets it -- '(bits & 3) == 3' is therefore the geometric fold flag in every cell
       a sample reached. The collapse and indicator bits 2..8 are per sample too, NOT geometric:
       a fold cell whose samples all fell outside the folded tetrahedra carries the parity of the
       fold but only the k of the streams its samples saw (typically k = 0) -- nSub-limited, as
       the README notes. */
    if ( psCaustics )
    {
        quantities->caustic_bits.resize(totalGrid);
        for (size_t i = 0; i < totalGrid; ++i)
            quantities->caustic_bits[i] = Real( orientBits[i]
                | ( flippedFlag[i] ? (PSCausticClass::CausticMask)PSCausticClass::BIT_PARITY_NEG
                                   : (PSCausticClass::CausticMask)0 ) );
    }
}

// The pass's phase split and the run's totals so far (psPhaseTotalUs), one line each.
static inline void psPhaseReport(PSPassCtx const &passCtx, double (&phaseSec)[PS_PH_N], bool const metalDeposited,
                                 std::vector<uint32_t> &deferredOrd)
{
    MESSAGE::Message &message = passCtx.message;
    size_t const nDeferredHere = metalDeposited ? deferredOrd.size() : 0;
    for (int k = 0; k < PS_PH_N; ++k) psPhaseTotalUs[k] += (long long)( phaseSec[k] * 1e6 );
    psPhaseTotalDeferred += (long long)nDeferredHere;
    int const nPasses = ++psPhasePasses;
    {
        std::ostringstream line, total;
        line << std::fixed << std::setprecision(2);
        total << std::fixed << std::setprecision(1);
        bool first = true;
        for (int k = 0; k < PS_PH_N; ++k)
        {
            double const tot = double( psPhaseTotalUs[k].load() ) * 1e-6;
            if ( phaseSec[k] <= 0. && tot <= 0. ) continue;   // a phase this build/run never enters
            line  << (first ? "" : ", ") << psPhaseName[k] << " " << phaseSec[k] << " s";
            total << (first ? "" : ", ") << psPhaseName[k] << " " << tot << " s";
            if ( k == PS_PH_DEFERRED )
            {
                line  << " (" << nDeferredHere << " tets)";
                total << " (" << psPhaseTotalDeferred.load() << " tets)";
            }
            first = false;
        }
        message << MESSAGE::cGreen() << "  >>> field interpolation phases: " << line.str() << "\n"
                << "  >>> ... run total over " << nPasses << " pass(es): " << total.str() << MESSAGE::cReset() << "\n" << MESSAGE::Flush;
    }
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

    // the phase split of this pass (psPhaseTotalUs above): lap(k) books the time since the last lap to phase k
    double phaseSec[PS_PH_N] = { 0. };
    auto phaseMark = std::chrono::steady_clock::now();
    auto lap = [&](int k)
    {
        auto const now = std::chrono::steady_clock::now();
        phaseSec[k] += std::chrono::duration<double>( now - phaseMark ).count();
        phaseMark = now;
    };

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
    // no cellHandles vector is materialized -- it would be 8 B x all finite cells (~0.4 GB per
    // concurrent partition at production scale) held for the whole interpolation.
    size_t noTotalCells = 0;
#if NO_DIM==2
    noTotalCells = dt.number_of_faces();
#elif NO_DIM==3
    noTotalCells = dt.number_of_finite_cells();
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
    PSPassCtx const passCtx{ dt, userOptions, quantities, field, message, boxCoordinates, nGrid, dx, nSub, noTotalCells,
                             windowMode, subOrigin, subDims };
    psSubgridBox( passCtx, psKeptCells );
    size_t subTotal = 1; for (int d = 0; d < NO_DIM; ++d) subTotal *= subDims[d];
    totalGrid = subTotal;   // all allocations and loops below use the sub-grid size
    // --ps-window: a tetrahedron whose Eulerian footprint misses the window on some axis deposits
    // nothing: skipped before extraction / sampling. The one-cell ring kept here for nSub > 1 is only
    // generous (the deposits have no sample margin: no sample lies in it)
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
    // barycentric tolerance would count a sample near a face shared by two tetrahedra in BOTH,
    // depending on the rounding of each run's periodic copies (so serial and --partition runs
    // would disagree at periodic faces). Every sample is counted by exactly one tetrahedron of each stream it lies in,
    // as in the point evaluation. The GPU kernel reaches the same decisions (metal/ps_deposit.metal: a float test with
    // a rigorous error bound, deferring the undecidable tets to the CPU loop below).
    psExactInside::Frame exactFrame;
    exactFrame.init( userOptions );
    // Sample positions in DOUBLE, per axis: samplePosAxis[d][g * nSub + k] = lo + ((g + (k+0.5)/nSub)
    // * L) / n for the WRAPPED cell index g -- so every tetrahedron (original or periodic image,
    // any partition) sees the same point; at nSub=1 it is the cell centre (i+0.5)*L/n bit for
    // bit, the expression the tests hand to --sample-points. Built once: nGrid*nSub per axis.
    double gridLenD[NO_DIM];
    std::vector<double> samplePosAxis[NO_DIM];
    psSamplePositions( passCtx, gridLenD, samplePosAxis );


    // --ps-vertex-mass: chart-independent per-tetrahedron masses. Count, per vertex, the incident cells that DEPOSIT;
    // a tet's mass is then the sum of its vertices' weight/degree shares (psVertexShareMass), so mass follows the
    // particles instead of rho_bar * V_lag (see the --ps-vertex-mass help). The degree applies the deposit loop's
    // chart-independent drop filters, so the deposited tets' masses add up to sum_v m_v exactly (but for vertices
    // whose every incident tet is dropped; a dropped tet counted in the degree would take its shares with it).
    // Ownership is skipped: padded vertices see their full neighbourhood, so per-partition degrees match the global
    // ones. Counted on THIS triangulation after a reset: the unaveraged and averaged passes share one dt.
    bool const psVertexMass = userOptions.psVertexMass;
    // the full degree pass: every finite cell once, its surviving vertices counted (ps_cell_filter.h: the
    // point evaluation runs the same pass)
    auto fullDegreePass = [&]()
    {
        psVertexDegreePass( dt, userOptions, boxCoordinates );
    };
    // --ps-window on the GPU: only the vertices of the tetrahedra that reach the window need their
    // degree, and the GPU extraction loop (single-threaded) visits exactly those -- so the degrees
    // are computed LAZILY there, each from the vertex's own finite incident cells (the same filter,
    // ownership skipped: the same count as the full pass). The full pass walks every cell of the
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
    // per-tet vertex-share mass: sum over the cell's vertices of weight/degree -- psVertexShareMass(cell),
    // ps_cell_filter.h (shared with the point evaluation)

    // The MASS grid is an exact 64-bit fixed-point sum on the GPU (metal/ps_deposit.metal depositMass) and the CPU
    // alike: a float '+=' drops every share below half an ulp of a busy cell's running sum, always downward (1.6e-5
    // of the box mass at TNG50-3-Dark z=0.5, PS_MASS_AUDIT=1), and depends on the thread count. So:
    //   - after a GPU deposit the host keeps its words unconverted (PSGpuGrids::massFx), and the released and the
    //     CPU-deferred tets add rint(share * scale) into them with the GPU's unit (psMassFixedShare);
    //   - a CPU deposit starts from zero, with psMassFixedScale's unit for an upper bound of the pass's mass (below),
    //     so no cell overflows;
    //   - the grid is converted to the float density ONCE, after the loop and before the normalization.
    // Integer adds commute: the grid is independent of the thread count, the slab split and the atomics' order (the
    // copy mode's threads add atomically, the serial loop and the slabs plainly). The moment grids follow the same
    // rule (fxg below); under weightIsDensity their normalizer is this grid. Single-precision build only: in the
    // double build a unit of 2^-62 of the pass's mass is coarser than a double sum in a light cell, so the GPU words
    // become doubles at the copy and the CPU keeps its double '+='. PS_METAL_FLOAT_MASS=1 (CUDA/HIP:
    // PS_GPU_FLOAT_MASS=1) selects the float grid on GPU and CPU, for comparison. Memory: 8 B per sub-grid cell in
    // place of the 4 B float grid, which is allocated only at the conversion (auto_tune.h counts it).
    auto envOn = [](char const *name) { char const *e = std::getenv(name); return e && std::atoi(e) != 0; };
    bool const massFxWanted = std::is_same<Real, float>::value && field.density
                              && !envOn("PS_METAL_FLOAT_MASS") && !envOn("PS_GPU_FLOAT_MASS");
    std::vector<int64_t> massFx;   // totalGrid fixed-point words while massFxOn, else EMPTY
    float massFxScale = 0.f;       // their unit: 1/massFxScale mass (a power of two)
    bool massFxOn = false;

    // The MOMENT grids (ps_deposit_params.h PSFxGrid: the grids, their bounds and units) are exact 64-bit fixed-point
    // sums too, for the same reason: int64 words in units of 1/scale while the pass deposits, shares added as
    // rint(v*w * scale), as the mass:
    //   - a GPU deposit returns the words of every grid it got a unit for (fxUnits, below: from bounds the extraction
    //     loop sums over the tets it hands over plus the released ones); released and deferred tets add with those;
    //   - a CPU deposit takes its units from the same bounds, summed in one extra filter pass in fixed blocks of cells
    //     (thread-count independent, the units a GPU deposit of the pass takes; --ps-window too: fxBoundWalk);
    //   - all threads add into ONE set of words (no private copies), converted once after the loop (fxToReal).
    // A grid's Real accumulator stays EMPTY while its words live (fxEnsureReal allocates it when a grid falls back to
    // float sums). Single-precision build only, like the mass (a 2^-61 unit is coarser than a double sum).
    // PS_METAL_FLOAT_MOMENTS=1 (or PS_GPU_FLOAT_MOMENTS=1) selects the float sums on GPU and CPU, the items' weight
    // totals included, for comparison (PS_METAL_FLOAT_MASS is the mass grid's own switch).
    bool const fxGpuWanted = !envOn("PS_METAL_FLOAT_MOMENTS") && !envOn("PS_GPU_FLOAT_MOMENTS");
    bool const fxMomWanted = fxGpuWanted && std::is_same<Real, float>::value;
    struct FxWords { std::vector<int64_t> w; float scale = 0.f; };
    FxWords fxg[PS_FX_CPU_N];      // grid g's words (EMPTY = its Real accumulator is in use) and their unit

    // PS_MASS_AUDIT=1 (ps_mass_audit.h): what this pass should deposit, counted before the deposit in one
    // extra filter pass -- (a) the owned vertices, (c) those of degree 0, (b) the tets that pass the filter
    // with ownership and their mass, (d) the owned tets the drop filters reject. The deposit paths below
    // add what they hand on and what reaches the grid; the line is printed at the end of the pass.
    bool const massAudit = PSMassAudit::on();
    struct MassAudit
    {
        double owned = 0., deg0 = 0., tets = 0., dropped = 0., after = 0.;
        size_t nOwned = 0, nDeg0 = 0, nTets = 0, nTetsDeg0 = 0, nDropped = 0, nAfter = 0;
        double gpuHanded = -1., gpuDeferred = -1., gpuFixed = -1., gpuFloat = -1., released = 0., releasedGain = 0.;
        double cpuTets = 0., cpuShares = 0., cpuGain = -1.;
        size_t nCpuTets = 0;
    } audit;
    auto gridSum = [&]() -> double   // the mass grid in mass units: the fixed-point words while they live (exact)
    {
        if ( massFxOn )
        {
            int64_t q = 0;
            for (int64_t const w : massFx) q += w;
            return psMassFixedMass( q, massFxScale );
        }
        double s = 0.;
        for (Real const v : quantities->density) s += double(v);
        return s;
    };
    psMassAuditCount( passCtx, massAudit, psVertexMass, degreesLazy, fullDegreePass, audit );

    // Allocate the requested output fields, zeroed since the cell loop accumulates into them. The density not
    // yet: after a GPU deposit it is made from the GPU's mass words at the copy (so it does not sit
    // beside the host's grids during the call), and a CPU deposit allocates it, or the fixed-point mass grid
    // that stands in for it until the conversion (massFx above), just before its loop. A grid left over from
    // an earlier use of 'quantities' is released here.
    if ( field.density )
        std::vector<Real>().swap( quantities->density );
    // Dispersion needs <v>, so allocate velocity storage even when only dispersion was
    // requested -- EXCEPT under --ps-volume-weighted, where the dispersion carries its OWN
    // mass-weighted mean (disp_velocity/disp_weight below): a dispersion-only volume-weighted
    // run would otherwise allocate, deposit and GPU-round-trip a 12 B/cell grid nothing reads.
    bool const haveVel = field.velocity
                         || (field.velocity_dispersion && !userOptions.psVolumeWeighted);
    // the dispersion always needs the per-sample velocity VALUES (for sum m v_i v_j and its
    // own mean), even when the velocity GRID above is gated away -- keep the two distinct
    bool const needVelValues = haveVel || field.velocity_dispersion;
    psAllocMoments( passCtx, haveVel, fxMomWanted, totalGrid );

    // Per-grid-point count of overlapping streams (diagnostic; the stream_count field). Sampled deposit: +1 per
    // in-simplex sample, divided by nSub^NO_DIM at the end: the cell-mean stream multiplicity (at nSub=1 exactly the
    // streams at the cell centre). A tet containing NO sample deposits its mass through the centroid fallback but
    // adds nothing here (+1 per such tet would bias '.streams' upward in collapsed cells). Exact deposit: the
    // TETRAHEDRA TOUCHING the cell (every nonzero clip), a geometric multiplicity that is NOT a stream count, so it is
    // exported as '.tetTouch' and the physical multiplicity is accumulated in exactMult below.
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
    // The EXACT companion ('.hidden_streams' bit 1): 1 where a negatively oriented (folded) tet overlaps the cell with
    // positive volume. In a periodic box the Lagrangian->Eulerian map has degree 1, so (#positive - #negative tets
    // containing a point) = 1 and the stream count is 1 + 2 #negative: a point is multi-stream EXACTLY where a flipped
    // tet covers it, however small the tets (the mass flag above cannot tell a same-stream sub-sample tet from a
    // foreign stream, and misses a folded sliver whose mass went to another cell). Marked per tet in both deposit paths
    // (markFlipped), OR-merged across partitions; DTFE() keeps it only where '.streams' reads single-stream
    // (Quantities::finalizeHiddenStreams). Also the '--ps-caustics' parity bit 1 (det < 0 overlaps the cell): the
    // export below ORs it, still raw, into the caustic mask.
    std::vector<unsigned char> flippedFlag(totalGrid, 0);
    // xLo/xHi: only cells whose local x index lies in [xLo, xHi) -- the slab deposit's thread owns them;
    // shared: the copy mode's threads write the same flags (relaxed atomics, psSetFlag; a flag only goes 0 -> 1)
    auto markFlipped = [&](PSCellGeometry const &geo, std::vector<unsigned char> &flags,
                           long const xLo = 0, long const xHi = LONG_MAX, bool const shared = false)
    {
        auto isSet = [&flags, shared](size_t const f) -> bool
        { return shared ? __atomic_load_n( &flags[f], __ATOMIC_RELAXED ) != 0 : flags[f] != 0; };
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
        // tet's bbox holds millions of cells, and the whole walk would cost ~55 s per partition of a 32^3 test box
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
            if ( !inSub || isSet(flat) ) continue;
            if ( cells == 1 ) { psSetFlag( flags[flat], shared ); continue; }   // the whole (non-degenerate) tet is in this cell
            double c[3];
            for (int d = 0; d < 3; ++d)
                c[d] = double(boxCoordinates[2*d]) + (raw[d] + 0.5) * double(dx[d]) - double(ep[0][d]);
            if ( sat.overlaps(c) ) psSetFlag( flags[flat], shared );
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
            if ( !inSub || isSet(flat) ) continue;
            double lo[NO_DIM], hi[NO_DIM], mom[psClip2d::MOMENTS];
            for (int d = 0; d < NO_DIM; ++d)
            {
                lo[d] = double(boxCoordinates[2*d]) + raw[d] * double(dx[d]) - double(ep[0][d]);
                hi[d] = lo[d] + double(dx[d]);
            }
            if ( psClip2d::clipTriangleToCell( tri, lo, hi, mom ) ) psSetFlag( flags[flat], shared );
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
    if ( userOptions.psExactDeposit && !fxMomWanted ) exactMult.assign(totalGrid, Real(0.));   // (fxMomWanted: fxEnsureReal)

    // --ps-caustics: per-cell orientation mask (bit0 = a det(Ax)>0 stream overlaps the cell,
    // bit1 = det<0). det(Ax)'s sign is the parity of the Lagrangian->Eulerian map (CGAL cells
    // are positively oriented in Lagrangian space), which flips at every fold -- both bits set
    // means a fold caustic surface crosses the cell. Exported as small exact ints in Real.
    // Accumulated per deposited SAMPLE; the export ORs the geometric flippedFlag into bit 1, so
    // bit 1 is the true overlap, while bit 0 and the class bits stay per sample.
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
    // Volume weighting breaks the aliasing (the moment normalizer is not the mass sum),
    // so it always allocates the explicit weight grid.
    bool const weightIsDensity = needWeight && field.density && !psVolWeighted;
    std::vector<Real> massWeight;
    if ( needWeight && !weightIsDensity && !fxMomWanted ) massWeight.assign(totalGrid, Real(0.));   // (fxMomWanted: fxEnsureReal)
    // --ps-volume-weighted keeps the DISPERSION mass-weighted (sigma_ij is a moment of the
    // phase-space distribution f -- f-weighted by definition) while velocity/gradient go
    // volume-weighted. sigma_ij = <v_i v_j>_m - <v_i>_m<v_j>_m then needs its own mass-weighted
    // mean and normalizer, because 'velocity' carries the volume-weighted one.
    bool const dispNeedsOwnWeight = psVolWeighted && field.velocity_dispersion;
    std::vector<Real> dispWeight;                      // sum(m_s)
    std::vector< Pvector<Real,noVelComp> > dispVel;    // sum(m_s v_s)
    if ( dispNeedsOwnWeight && !fxMomWanted )   // (fxMomWanted: fxEnsureReal)
    {
        dispWeight.assign(totalGrid, Real(0.));
        dispVel.assign(totalGrid, Pvector<Real,noVelComp>::zero());
    }

    // ---- the moment grids as fixed-point sums (see fxg above) ----
    // which grids this pass accumulates (ps_deposit_params.h PSFxGrid): MOMW is massWeight (the volume shares, or
    // without a density grid the mass shares), DW dispWeight, RM releasedMult
#ifdef SCALAR
    bool const fxScalSel = field.scalar, fxSGradSel = field.scalar_gradient;
#else
    bool const fxScalSel = false, fxSGradSel = false;
#endif
    bool const fxSel[PS_FX_CPU_N] = { haveVel, bool(field.velocity_dispersion), bool(field.velocity_gradient),
                                      needWeight && !weightIsDensity, psExact, dispNeedsOwnWeight, dispNeedsOwnWeight,
                                      fxScalSel, fxSGradSel, !releasedMult.empty() };
    // the Real accumulator of grid g, allocated (zeroed) if the pass has none yet: a grid that is not a fixed-point
    // sum after all (no unit, or the GPU's float grid)
    auto fxEnsureReal = [&](int const g)
    {
        if ( !fxSel[g] ) return;
        switch ( g )
        {
        case PS_FX_MOM:  if ( quantities->velocity.empty() ) quantities->velocity.assign(totalGrid, Pvector<Real,noVelComp>::zero()); break;
        case PS_FX_M2:   if ( quantities->velocity_dispersion.empty() ) quantities->velocity_dispersion.assign(totalGrid, Pvector<Real,noDispComp>::zero()); break;
        case PS_FX_GRAD: if ( quantities->velocity_gradient.empty() ) quantities->velocity_gradient.assign(totalGrid, Pvector<Real,noGradComp>::zero()); break;
        case PS_FX_MOMW: if ( massWeight.empty() ) massWeight.assign(totalGrid, Real(0.)); break;
        case PS_FX_SV:   if ( exactMult.empty() ) exactMult.assign(totalGrid, Real(0.)); break;
        case PS_FX_DV:   if ( dispVel.empty() ) dispVel.assign(totalGrid, Pvector<Real,noVelComp>::zero()); break;
        case PS_FX_DW:   if ( dispWeight.empty() ) dispWeight.assign(totalGrid, Real(0.)); break;
#ifdef SCALAR
        case PS_FX_SCAL: if ( quantities->scalar.empty() ) quantities->scalar.assign(totalGrid, Pvector<Real,noScalarComp>::zero()); break;
        case PS_FX_SG:   if ( quantities->scalar_gradient.empty() ) quantities->scalar_gradient.assign(totalGrid, Pvector<Real,noScalarGradComp>::zero()); break;
#endif
        case PS_FX_RM:   if ( releasedMult.empty() ) releasedMult.assign(totalGrid, Real(0.)); break;
        default: break;
        }
    };
    // components per cell of grid g (the words' layout)
    auto fxComps = [&](int const g) -> size_t
    {
        switch ( g )
        {
        case PS_FX_MOM: case PS_FX_DV: return noVelComp;
        case PS_FX_M2:   return noDispComp;
        case PS_FX_GRAD: return noGradComp;
#ifdef SCALAR
        case PS_FX_SCAL: return noScalarComp;
        case PS_FX_SG:   return noScalarGradComp;
#endif
        default: return 1;
        }
    };
    // grid g becomes a fixed-point sum in units of 1/scale (its words zeroed, its Real accumulator released)
    auto fxStart = [&](int const g, float const scale)
    {
        fxg[g].scale = scale;
        fxg[g].w.assign( totalGrid * fxComps(g), 0 );
        switch ( g )
        {
        case PS_FX_MOM:  std::vector< Pvector<Real,noVelComp> >().swap( quantities->velocity ); break;
        case PS_FX_M2:   std::vector< Pvector<Real,noDispComp> >().swap( quantities->velocity_dispersion ); break;
        case PS_FX_GRAD: std::vector< Pvector<Real,noGradComp> >().swap( quantities->velocity_gradient ); break;
        case PS_FX_MOMW: std::vector<Real>().swap( massWeight ); break;
        case PS_FX_SV:   std::vector<Real>().swap( exactMult ); break;
        case PS_FX_DV:   std::vector< Pvector<Real,noVelComp> >().swap( dispVel ); break;
        case PS_FX_DW:   std::vector<Real>().swap( dispWeight ); break;
#ifdef SCALAR
        case PS_FX_SCAL: std::vector< Pvector<Real,noScalarComp> >().swap( quantities->scalar ); break;
        case PS_FX_SG:   std::vector< Pvector<Real,noScalarGradComp> >().swap( quantities->scalar_gradient ); break;
#endif
        case PS_FX_RM:   std::vector<Real>().swap( releasedMult ); break;
        default: break;
        }
    };
    // grid g's words become its Real grid, once (after the deposit); a no-op for a grid without words
    auto fxToReal = [&](int const g)
    {
        FxWords &F = fxg[g];
        if ( F.w.empty() ) return;
        switch ( g )
        {
        case PS_FX_MOM:  psFxToReal( F.w, F.scale, quantities->velocity, totalGrid ); break;
        case PS_FX_M2:   psFxToReal( F.w, F.scale, quantities->velocity_dispersion, totalGrid ); break;
        case PS_FX_GRAD: psFxToReal( F.w, F.scale, quantities->velocity_gradient, totalGrid ); break;
        case PS_FX_MOMW: psFxToReal( F.w, F.scale, massWeight, totalGrid ); break;
        case PS_FX_SV:   psFxToReal( F.w, F.scale, exactMult, totalGrid ); break;
        case PS_FX_DV:   psFxToReal( F.w, F.scale, dispVel, totalGrid ); break;
        case PS_FX_DW:   psFxToReal( F.w, F.scale, dispWeight, totalGrid ); break;
#ifdef SCALAR
        case PS_FX_SCAL: psFxToReal( F.w, F.scale, quantities->scalar, totalGrid ); break;
        case PS_FX_SG:   psFxToReal( F.w, F.scale, quantities->scalar_gradient, totalGrid ); break;
#endif
        case PS_FX_RM:   psFxToReal( F.w, F.scale, releasedMult, totalGrid ); break;
        default: break;
        }
    };
    // One kept tetrahedron's terms of the bounds (PSFixedBounds::addTet): the largest |value| it can deposit per
    // grid, from the vertex values and the constant gradients the CPU deposit itself forms (geo.posMatInv, Real).
    // The GPU kernel forms its gradients in float from its own record; near the degeneracy cut they may exceed
    // these by a few per cent, which the factor 4 of headroom in the unit absorbs (psMomentFixedScale).
    bool const fxVelBound = needVelValues || field.velocity_gradient;
    auto fxTetBound = [&](Cell_handle const &cell, PSCellGeometry const &geo, double const tetMass, PSFixedBounds &fb)
    {
        double const V = geo.cellAbsDet / double(factorial(NO_DIM));
        double const wm = psVolWeighted ? V : tetMass;
        double vMax = 0., gMax = 0., sMax = 0., sgMax = 0.;
        if ( fxVelBound )
        {
            for (int v = 0; v <= NO_DIM; ++v)
                for (size_t j = 0; j < noVelComp; ++j)
                    vMax = std::max( vMax, std::fabs( double( cell->vertex(v)->info().velocity(j) ) ) );
            if ( field.velocity_gradient )
            {
                Real temp[NO_DIM][noVelComp], vg[NO_DIM][noVelComp];
                for (int v = 0; v < NO_DIM; ++v)
                    for (size_t j = 0; j < noVelComp; ++j)
                        temp[v][j] = cell->vertex(v+1)->info().velocity(j) - cell->vertex(0)->info().velocity(j);
                matrixMultiplication<noVelComp>( const_cast<Real (&)[NO_DIM][NO_DIM]>(geo.posMatInv), temp, vg );
                for (int i = 0; i < NO_DIM; ++i)
                    for (size_t j = 0; j < noVelComp; ++j) gMax = std::max( gMax, std::fabs( double(vg[i][j]) ) );
            }
        }
#ifdef SCALAR
        if ( fxScalSel || fxSGradSel )
        {
            for (int v = 0; v <= NO_DIM; ++v)
                for (size_t j = 0; j < noScalarComp; ++j)
                    sMax = std::max( sMax, std::fabs( double( cell->vertex(v)->info().myScalar()[j] ) ) );
            if ( fxSGradSel )
            {
                Real temp[NO_DIM][noScalarComp], sg[NO_DIM][noScalarComp];
                for (int v = 0; v < NO_DIM; ++v)
                    for (size_t j = 0; j < noScalarComp; ++j)
                        temp[v][j] = cell->vertex(v+1)->info().myScalar()[j] - cell->vertex(0)->info().myScalar()[j];
                matrixMultiplication<noScalarComp>( const_cast<Real (&)[NO_DIM][NO_DIM]>(geo.posMatInv), temp, sg );
                for (int i = 0; i < NO_DIM; ++i)
                    for (size_t j = 0; j < noScalarComp; ++j) sgMax = std::max( sgMax, std::fabs( double(sg[i][j]) ) );
            }
        }
#endif
        fb.addTet( tetMass, V, wm, vMax, gMax, sMax, sgMax, double(cellVolume) );
    };
    // the unit of grid g from its bound; 0 (and a warning naming the grid) when the bound has none: the grid then
    // keeps the float sums
    auto fxUnit = [&](int const g, double const bound) -> float
    {
        static char const *const name[PS_FX_CPU_N] = { "velocity moment", "dispersion second moment", "velocity-gradient moment",
            "moment weight", "exact multiplicity", "dispersion mean", "dispersion weight", "scalar moment",
            "scalar-gradient moment", "released multiplicity" };
        float scale = 0.f;
        if ( psMomentFixedScale( bound, scale ) ) return scale;
        MESSAGE::Warning warning( userOptions.verboseLevel );
        warning << "PS-DTFE: the " << name[g] << " grid's bound (" << bound << ") gives no fixed-point unit; this pass "
                << "sums it in floating point (not bit-reproducible).\n" << MESSAGE::EndWarning;
        return 0.f;
    };
    // --verbose 3: the pass's fixed-point units, one line -- the mass grid's and each moment grid's
    // 2^k (DW, the mass grid's twin, left out) -- the same for a GPU and a CPU deposit of one pass
    // (tests/ps_window_check.sh compares a GPU and a CPU --ps-window run's lines)
    auto fxLogUnits = [&](char const *path, float const massScale, float const (&scale)[PS_FX_N])
    {
        if ( userOptions.verboseLevel < 3 ) return;
        std::ostringstream line;
        line << "PS-DTFE fixed-point units (" << path << "): mass ";
        if ( massScale > 0.f ) line << "2^" << std::ilogb( massScale ); else line << "float";
        for (int g = 0; g < PS_FX_N; ++g)
            if ( g != PS_FX_DW && scale[g] > 0.f ) line << ", [" << g << "] 2^" << std::ilogb( scale[g] );
        std::fprintf( stderr, "%s\n", line.str().c_str() );
    };

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
    psGpuEligible( passCtx, subTotal, tryMetal );
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
        // the bounds of the moment grids' fixed-point units (fxg): every tet that passes the filter --
        // the ones handed to the GPU (those it will defer to the CPU included), the released ones the host deposits
        // itself, and those of no mass -- summed per block of 256 finite cells and then block by block, EXACTLY as
        // the CPU deposit's bound walk sums them (fxBoundWalk, below), so a GPU and a CPU deposit of the same pass
        // take the same units. Under --ps-window both count only the tets that reach the window (this loop skips
        // the others before the filter; the CPU walk applies the same test, fxBoundWalk says why),
        // and the window's MASS unit is taken from these bounds as well (B[DW], PSFixedUnits.massTotal) -- so the
        // bounds are summed under --ps-window even when the moments are float sums (PS_METAL_FLOAT_MOMENTS=1)
        bool const fxBoundsWanted = fxGpuWanted || windowMode;
        std::vector<PSFixedBounds> gpuBoundPart( fxBoundsWanted ? noTotalCells / 256 + 1 : 0 );
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
        lap( PS_PH_SETUP );
        PS_FOREACH_FINITE_CELL
        {
            uint32_t const cellOrd = cellOrdCounter++;   // counts EVERY finite cell, like the CPU loop
            Cell_handle cell = itC;
            PSCellGeometry geo;   // the CPU loop's filters, including the zero-inverse check
            // --ps-window: a tet that misses the window is dropped on its Eulerian positions, before the
            // filter's determinant and inverse -- the same set as testing after the filter, at a fraction of
            // the cost (a TNG100-3 partition walks 36M tets for a few thousand inside)
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
            if ( fixedShareTestNanEvery() && fixedShareTestNan( double(cell->vertex(0)->point()[0]), double(cell->vertex(0)->point()[1]),
                                                                double(cell->vertex(0)->point()[NO_DIM-1]) ) )
                tm = std::numeric_limits<float>::quiet_NaN();   // DTFE_TEST_NAN_SHARE (ps_deposit_params.h)
            if ( fxBoundsWanted ) fxTetBound( cell, geo, double( tm > 0.f ? tm : 0.f ), gpuBoundPart[cellOrd / 256] );
            if (tm<=0.f) { if ( massAudit ) { audit.after += double(tm); ++audit.nAfter; } continue; }
            if ( psHaloRelease > 0. && absDetLag > psHaloRelease * geo.cellAbsDet )
            {
                // identical classification to the CPU loop (double |det Lag| > D*|det Ax|);
                // centroid cell + centroid-evaluated velocity mirror the CPU fallback exactly
                Real centroid[NO_DIM];
                size_t f = 0;
                if ( !centroidCell(ep, centroid, f) )   // outside this grid/partition region: drop (as the CPU fallback does)
                { if ( massAudit ) { audit.after += double(tm); ++audit.nAfter; } continue; }
                if ( massAudit ) audit.released += double(tm);
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
                // edges, plus the centroid-fallback cell exactly as the CPU fallback picks it. The vertices stay
                // Real: the double build's exact test compares DOUBLE positions (psFillTetRecord says why)
                Real cv[NO_DIM+1][NO_DIM];
                for (int v = 0; v <= NO_DIM; ++v)
                    for (int d = 0; d < NO_DIM; ++d)
                        cv[v][d] = psExactInside::canonicalCoord( exactFrame, cell->vertex(v)->info().eulerianPosition(d), d );
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

        lap( PS_PH_EXTRACT );
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
            // pairs takes ~1.2 s per 15M tets; this also needs 6 B/tet instead of 8.)
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
                    // the window spans extent + [1, 2) cells (no sample margin)
                    cells *= (hi-lo) / float(dx[d]) + 1.f;
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
        if ( massAudit )
        {
            audit.gpuHanded = 0.;
            for (float const m : tetMasses) audit.gpuHanded += double(m);
        }
        PSGpuGrids gpuOut;
        std::vector<uint32_t> deferredTets;
        std::string metalErr;
        // the released tets' mass: the host adds it into the GPU's fixed-point grid below, so the unit covers it
        double releasedMass = 0.;
        for (ReleasedTet const &rt : releasedTets) releasedMass += double(rt.mass);
        // the moment grids' units (fxg): one per grid the kernel deposits, from the bounds above, so the
        // host's later integer adds (the deferred and released tets) stay within them too; dispw's unit only says
        // "the mass grid is it" (the host then skips it when its mass grid is fixed). PS_METAL_FLOAT_MOMENTS=1: none.
        PSFixedUnits fxUnits;
        PSFixedBounds gpuBounds;
        for (PSFixedBounds const &b : gpuBoundPart) gpuBounds.add( b );   // in block order (see gpuBoundPart)
        std::vector<PSFixedBounds>().swap( gpuBoundPart );
        if ( fxGpuWanted )
        {
            bool const dispOwn = psVolWeighted && fDisp;
            bool const gpuSel[PS_FX_N] = { fVel, fDisp, fGrad, psVolWeighted, psExact, dispOwn, dispOwn, fScal, fSGrad };
            for (int g = 0; g < PS_FX_N; ++g)
                if ( gpuSel[g] ) fxUnits.scale[g] = g == PS_FX_DW ? 1.f : fxUnit( g, gpuBounds.B[g] );
            fxUnits.sumW = true;
        }
        // --ps-window: the mass grid's unit from the window's own total, as the CPU window deposit
        // takes it (fxBoundWalk), not from the host's sum of the masses it is handed
        if ( windowMode ) fxUnits.massTotal = gpuBounds.B[PS_FX_DW];
        lap( PS_PH_SORT );
        // (the GPU call's wall time includes any wait for the device: concurrent partitions queue on it)
        bool gpuOk = psGpuDepositFields(tetVerts, tetVels, tetMasses, tetDens, tetScal, bl, dxv,
                                  nGrid, subOrigin, subDims, nSub,
                                  userOptions.periodic, fVel, fDisp, fGrad, psLinear,
                                  psVolWeighted, psCaustics, psExact, fScal, fSGrad, windowMode, releasedMass, fxUnits,
                                  gpuOut, deferredTets, metalErr);
        // A host that chose the fixed-point unit hands back the words and leaves its float grid empty, so words
        // of another size would send the decode below into that empty grid: a failed deposit instead (the CPU
        // deposits the partition). Neither host produces it; this keeps it from reading
        // out of bounds if one ever does.
        if ( gpuOk && gpuOut.massScale > 0.f && gpuOut.massFx.size() != totalGrid )
        {
            gpuOk = false;
            metalErr = "the GPU returned a mass grid of the wrong size";
        }
        for (int g = 0; gpuOk && g < PS_FX_N; ++g)   // the same guard for the moment grids' words
            if ( gpuOut.fxScale[g] > 0.f && gpuOut.fx[g].size() != totalGrid * fxComps(g) )
            {
                gpuOk = false;
                metalErr = "the GPU returned a moment grid of the wrong size";
            }
        lap( PS_PH_GPU );
        if ( gpuOk )
        {
            fxLogUnits( "GPU", gpuOut.massScale, gpuOut.fxScale );   // (--verbose 3)
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
            // The mass grid (see massFx): the GPU's fixed-point words become this pass's mass grid as
            // they are -- the released and deferred tets are added into them below, and they are converted to the
            // float density after the CPU loop. Otherwise (the double build, or the float A/B grid) the density
            // is set here, from the words converted straight to Real or from the float grid.
            bool const gpuFixed = gpuOut.massScale > 0.f && gpuOut.massFx.size() == totalGrid;
            if ( field.density && massFxWanted && gpuFixed )
            {
                massFx.swap( gpuOut.massFx );
                massFxScale = gpuOut.massScale;
                massFxOn = true;
            }
            else if ( field.density )
                quantities->density.assign( totalGrid, Real(0.) );   // (set from the GPU's grid just below)
            std::vector<int64_t> const &gpuFx = massFxOn ? massFx : gpuOut.massFx;
            auto gpuMass = [&](size_t const i) -> double   // the GPU's mass of cell i, whichever form it came in
            {
                return gpuFixed ? psMassFixedMass( gpuFx[i], gpuOut.massScale ) : double( gpuOut.mass[i] );
            };
            // The moment grids (fxg): a grid the GPU returns as words becomes this pass's fixed-point grid
            // as it is (the single build; the deferred and released tets are added into it, and it is converted after
            // the CPU loop, like the mass); otherwise its Real grid is set here, cell by cell, from the GPU's floats,
            // or (the double build) from its words converted to double. The moment normalizer MOMW is the GPU's
            // volume-share grid under --ps-volume-weighted, else (no density selected) the mass words themselves;
            // the dispersion's normalizer DW is the mass grid when the kernel skipped it (dispwIsMass) -- and once
            // the mass grid is this pass's own exact words (massFxOn) DW is not accumulated at all: it becomes the
            // density after the loop (see massFx). MOMW and DW never both take the mass words (DW needs
            // --ps-volume-weighted, MOMW then the volume shares).
            std::vector<float> const *const gpuFloat[PS_FX_N] = { &gpuOut.mom, &gpuOut.m2, &gpuOut.grad, &gpuOut.momw,
                &gpuOut.streamvol, &gpuOut.dispvel, &gpuOut.dispw, &gpuOut.scal, &gpuOut.sgrad };
            bool const momwIsMass = !psVolWeighted, dwIsMass = gpuOut.dispwIsMass;
            bool fxFill[PS_FX_N] = {};    // grid g's Real grid is filled from the GPU in the cell loop below
            for (int g = 0; g < PS_FX_N; ++g)
            {
                if ( !fxSel[g] ) continue;
                if ( g == PS_FX_DW && massFxOn ) continue;            // (the density after the loop)
                bool const fromMass = ( g == PS_FX_MOMW && momwIsMass ) || ( g == PS_FX_DW && dwIsMass );
                if ( fxMomWanted && fromMass && gpuFixed && !massFxOn )
                {   // (copied while the density below still reads them: PS_GPU_FLOAT_MASS=1 on a Metal build)
                    fxg[g].scale = gpuOut.massScale;
                    if ( field.density ) fxg[g].w = gpuOut.massFx;
                    else fxg[g].w.swap( gpuOut.massFx );
                }
                else if ( fxMomWanted && !fromMass && gpuOut.fxScale[g] > 0.f )
                {
                    fxg[g].scale = gpuOut.fxScale[g];
                    fxg[g].w.swap( gpuOut.fx[g] );
                }
                else
                {
                    fxEnsureReal( g );
                    fxFill[g] = true;
                }
            }
            // the released tets' multiplicity (sampled deposit + --ps-halo-release): the host's own sum, in the unit a
            // CPU deposit of this pass would take (the same bounds), so the two write the same '.streams'
            if ( fxMomWanted && fxSel[PS_FX_RM] )
            {
                float const scale = fxUnit( PS_FX_RM, gpuBounds.B[PS_FX_RM] );
                if ( scale > 0.f ) fxStart( PS_FX_RM, scale );
            }
            // grid g's GPU value at component index 'at' (cell x components + component), whichever form it came in
            auto gpuVal = [&](int const g, size_t const at) -> double
            {
                if ( ( g == PS_FX_MOMW && momwIsMass ) || ( g == PS_FX_DW && dwIsMass ) ) return gpuMass(at);
                if ( gpuOut.fxScale[g] > 0.f ) return psMassFixedMass( gpuOut.fx[g][at], gpuOut.fxScale[g] );
                return double( (*gpuFloat[g])[at] );
            };
            for (size_t i = 0; i < totalGrid; ++i)
            {
                if (field.density && !massFxOn) quantities->density[i] = Real( gpuMass(i) );
                // moment normalizer: the volume-share sum under --ps-volume-weighted, else the mass (in the
                // single build the exact words' value, rounded once)
                if (fxFill[PS_FX_MOMW]) massWeight[i] = Real( gpuVal(PS_FX_MOMW, i) );
                if (fxFill[PS_FX_MOM])
                    for (size_t j = 0; j < noVelComp; ++j)
                        quantities->velocity[i][j] = Real( gpuVal(PS_FX_MOM, i*3+j) );
                if (fxFill[PS_FX_GRAD])
                    for (size_t q = 0; q < noGradComp; ++q)
                        quantities->velocity_gradient[i][q] = Real( gpuVal(PS_FX_GRAD, i*9+q) );
                if (fxFill[PS_FX_M2])
                    for (size_t cc = 0; cc < noDispComp; ++cc)
                        quantities->velocity_dispersion[i][cc] = Real( gpuVal(PS_FX_M2, i*6+cc) );
#ifdef SCALAR
                if (fxFill[PS_FX_SCAL]) quantities->scalar[i][0] = Real( gpuVal(PS_FX_SCAL, i) );
                if (fxFill[PS_FX_SG]) for (int q = 0; q < NO_DIM; ++q) quantities->scalar_gradient[i][q] = Real( gpuVal(PS_FX_SG, i*3+q) );
#endif
                // the dispersion's own mass-weighted mean/normalizer (kernel buffers 13/14)
                if (fxFill[PS_FX_DW]) dispWeight[i] = Real( gpuVal(PS_FX_DW, i) );
                if (fxFill[PS_FX_DV])
                    for (size_t j = 0; j < noVelComp; ++j)
                        dispVel[i][j] = Real( gpuVal(PS_FX_DV, i*3+j) );
                streamCount[i] = int(gpuOut.streams[i] & 0x7FFFFFFFu);   // bit 31 = the sub-sample mass flag ('.hidden_streams' bit 2)
                if ( gpuOut.streams[i] >> 31 ) subSampleMassFlag[i] = 1;
                if (fxFill[PS_FX_SV])
                    exactMult[i] = Real( gpuVal(PS_FX_SV, i) );   // exact cell-mean multiplicity
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
            if ( massAudit )
            {
                audit.gpuDeferred = gpuOut.auditDeferredMass;
                audit.gpuFixed = gpuOut.auditFixedTotal;
                if ( field.density ) audit.gpuFloat = gridSum();
            }
            // --ps-halo-release: host-side monolithic centroid deposits of the released tets
            // (same accumulation sites and moments as the CPU fallback's single sample)
            // (each moment add: into the grid's fixed-point words when it has them, else its Real grid --
            // the same float product either way; fxHost)
            auto fxHost = [&](int const g, size_t const at, Real const v) -> bool
            {
                if ( fxg[g].w.empty() ) return false;
                fxg[g].w[at] += psFixedShare( double(v), fxg[g].scale, g );
                return true;
            };
            for (ReleasedTet const &rt : releasedTets)
            {
                if (field.density)
                {
                    if ( massFxOn ) massFx[rt.flat] += psMassFixedShare( double(rt.mass), massFxScale );   // exact
                    else quantities->density[rt.flat] += rt.mass;
                }
                if (needWeight && !weightIsDensity && !fxHost(PS_FX_MOMW, rt.flat, rt.momw)) massWeight[rt.flat] += rt.momw;
                if (haveVel)
                    for (size_t j = 0; j < noVelComp; ++j)
                        if ( !fxHost(PS_FX_MOM, rt.flat*noVelComp + j, rt.vel[j] * rt.momw) )
                            quantities->velocity[rt.flat][j] += rt.vel[j] * rt.momw;
                if (field.velocity_gradient)
                    for (size_t q = 0; q < noGradComp; ++q)
                        if ( !fxHost(PS_FX_GRAD, rt.flat*noGradComp + q, rt.grad[q] * rt.momw) )
                            quantities->velocity_gradient[rt.flat][q] += rt.grad[q] * rt.momw;
#ifdef SCALAR
                if (fScal && !fxHost(PS_FX_SCAL, rt.flat, rt.scal * rt.momw)) quantities->scalar[rt.flat][0] += rt.scal * rt.momw;
                if (fSGrad)
                    for (int q = 0; q < NO_DIM; ++q)
                        if ( !fxHost(PS_FX_SG, rt.flat*NO_DIM + size_t(q), rt.sgrad[q] * rt.momw) )
                            quantities->scalar_gradient[rt.flat][q] += rt.sgrad[q] * rt.momw;
#endif
                if (field.velocity_dispersion)
                {
                    size_t c = 0;
                    for (int a = 0; a < NO_DIM; ++a)
                        for (int b = a; b < NO_DIM; ++b, ++c)
                            if ( !fxHost(PS_FX_M2, rt.flat*noDispComp + c, rt.mass * rt.vel[a] * rt.vel[b]) )
                                quantities->velocity_dispersion[rt.flat][c] += rt.mass * rt.vel[a] * rt.vel[b];
                    if ( dispNeedsOwnWeight )
                    {
                        if ( !massFxOn && !fxHost(PS_FX_DW, rt.flat, rt.mass) )   // (massFxOn: DW is the density)
                            dispWeight[rt.flat] += rt.mass;
                        for (size_t j = 0; j < noVelComp; ++j)
                            if ( !fxHost(PS_FX_DV, rt.flat*noVelComp + j, rt.vel[j] * rt.mass) )
                                dispVel[rt.flat][j] += rt.vel[j] * rt.mass;
                    }
                }
                subSampleMassFlag[rt.flat] = 1;            // mass from a never-sampled tet
                if (psExact)
                {
                    streamCount[rt.flat]++;             // raw tet-touch count ('.tetTouch')
                    if ( !fxHost(PS_FX_SV, rt.flat, rt.volFrac) )
                        exactMult[rt.flat] += rt.volFrac;   // V_tet / V_cell, as the CPU fallback deposits
                }
                else if ( !fxHost(PS_FX_RM, rt.flat, rt.volFrac) )
                    releasedMult[rt.flat] += rt.volFrac;   // never sampled: its volume fraction
                if (psCaustics)
                    orientBits[rt.flat] |= rt.orient;
            }
            if ( massAudit && field.density && !releasedTets.empty() ) audit.releasedGain = gridSum() - audit.gpuFloat;
            metalDeposited = true;
        }
        else
        {
            MESSAGE::Warning warning( userOptions.verboseLevel );
            warning << "PS-DTFE GPU deposit unavailable (" << metalErr << "); using the CPU deposit.\n" << MESSAGE::EndWarning;
            ++psGpuFallbacks;
            nReleased = 0;   // releasedTets are discarded; the CPU loop below recounts them
            if ( massAudit ) { audit.after = audit.released = 0.; audit.nAfter = 0; audit.gpuHanded = -1.; }
        }
    }
#endif // PS_GPU
    psDebugDegreesCheck( passCtx, degreesLazy, metalDeposited );
    if ( degreesLazy && not metalDeposited )
    {   // the CPU loop deposits every cell: it needs every degree
        fullDegreePass();
        degreesLazy = false;
    }
#if defined(PS_GPU) && NO_DIM==3
#else
    if ( userOptions.psUseMetal )
    {
        ++psGpuFallbacks;
        static bool warned = false;
        if ( !warned )
        {
            warned = true;
            MESSAGE::Warning warning( userOptions.verboseLevel );
            warning << "--ps-gpu requested but this binary was built without GPU support (rebuild with METAL=1 on a Mac, CUDA=1 or HIP=1 on Linux; or NO_DIM!=3); using the CPU deposit.\n" << MESSAGE::EndWarning;
        }
    }
#endif // PS_GPU

    // The bounds of the pass's fixed-point units: one walk over the cells with the deposit's own filter (psFilterCell,
    // the inverse included) and tet mass, summing PSFixedBounds per tet (fxTetBound). Each thread takes every nT-th
    // block of 256 cells, as the copy mode does, and the per-block sums are added in block order, so the units are the
    // same at every thread count and the same a GPU deposit of the pass takes (gpuBoundPart sums the same blocks).
    // --ps-window: only the tets that reach the window count (windowTouched, the test the GPU extraction drops the
    // others with), so a GPU and a CPU window deposit of one pass take the same units. That is cheaper than the full
    // run's units (no walk over every tet of the partition, no full --ps-vertex-mass degree pass) and finer in a zoom's
    // small cells. The price: a window matches the full run's cells to the rounding of shares below ~2^-38 of the
    // bound, not byte for byte (tests/ps_window_check.sh); GPU and CPU window grids still differ in the last bits (the
    // kernels form their shares in float). The mass grid's unit follows the same rule under --ps-window (below).
    auto fxBoundWalk = [&]() -> PSFixedBounds
    {
        size_t const BLK = 256;
        size_t const nBlocks = noTotalCells / BLK + 1;
        std::vector<PSFixedBounds> part( nBlocks );
        size_t const nT = size_t( std::max( 1, std::min( psDepositThreads(), int( std::min<size_t>( nBlocks, 1024 ) ) ) ) );
        auto boundRun = [&](size_t const th)
        {
            size_t idx = 0;
            for (auto it = PS_FINITE_CELLS_BEGIN; it != PS_FINITE_CELLS_END; ++it, ++idx)
            {
                size_t const blk = idx / BLK;
                if ( blk % nT != th ) continue;
                Cell_handle cell = it;
                PSCellGeometry geo;
                if ( !psFilterCell(cell, userOptions, boxCoordinates, true, NULL, geo, false,
                                   [&](Real const (&ep)[NO_DIM+1][NO_DIM]) { return windowMode && !windowTouched(ep); }) ) continue;
                double tetMass;
                if ( psVertexMass ) tetMass = double( Real( psVertexShareMass(cell) ) );
                else
                {
                    double Lag[NO_DIM][NO_DIM];
                    for (int v = 0; v < NO_DIM; ++v)
                        for (int i = 0; i < NO_DIM; ++i)
                            Lag[v][i] = double(cell->vertex(v+1)->point()[i]) - double(cell->vertex(0)->point()[i]);
                    tetMass = double( Real( double(userOptions.averageDensity) * std::fabs(determinant(Lag)) / factorial(NO_DIM) ) );
                }
                if ( !(tetMass > 0.) ) tetMass = 0.;   // (the deposit clamps a negative mass to 0)
                fxTetBound( cell, geo, tetMass, part[blk] );
            }
        };
        {
            std::vector<std::thread> pool;
            for (size_t th = 1; th < nT; ++th) pool.emplace_back( boundRun, th );
            boundRun( 0 );
            for (std::thread &x : pool) x.join();
        }
        PSFixedBounds fb;
        for (PSFixedBounds const &p : part) fb.add( p );   // in block order: thread-count independent
        return fb;
    };
    PSGridsCtx const gridsCtx{ totalGrid, streamCount, releasedMult, subSampleMassFlag, flippedFlag, exactMult, orientBits,
                               massWeight, dispWeight, dispVel, massFx, massFxScale, massFxOn, insideFlat, insideRel, insideW,
                               exMom, exW, fxSel, massFxWanted, fxMomWanted, haveVel, needWeight, weightIsDensity,
                               dispNeedsOwnWeight, deferNorm, psExact, psCaustics };
    psCpuFixedUnits( passCtx, gridsCtx, metalDeposited, fxg, fxBoundWalk, fxUnit, fxStart, fxEnsureReal, fxLogUnits );

    // The per-cell scatter. The CPU deposit runs on several threads (depositThreads, capped by the memory its private
    // accumulators may take): thread t takes every T-th block of 256 cells into its own copies, summed in thread order
    // afterwards (deterministic for a given thread count; DTFE_DEPOSIT_THREADS pins it); thread 0 writes the real
    // arrays. Shared by all threads instead: the single build's fixed-point mass (massFx) and moment (fxg) words, and
    // in every build the stream count, the two flags and the caustic mask (relaxed atomics). GPU-deferred cells and
    // --ps-caustic-cusps (CGAL's incident_cells marks visited cells: not re-entrant) stay on one thread. After a GPU
    // deposit only its deferred cells run here (deferredOrd: their finite-cell ordinals in the extraction's order).
    struct DepositThread
    {
        Quantities *q = nullptr;
        std::vector<int> *streamCount = nullptr;
        std::vector<Real> *releasedMult = nullptr, *exactMult = nullptr, *dispWeight = nullptr, *massWeight = nullptr;
        std::vector< Pvector<Real,noVelComp> > *dispVel = nullptr;
        std::vector<PSCausticClass::CausticMask> *orientBits = nullptr;
        std::vector<unsigned char> *subSampleMassFlag = nullptr, *flippedFlag = nullptr;
        // the fixed-point mass grid (massFxOn; else null): ONE grid for every thread, never copied
        int64_t *massFx = nullptr;
        // the private copies of threads > 0 (thread 0 points at the real arrays): the Real accumulators only -- the
        // stream count, the two flags and the caustic mask are the real arrays for every thread (psAddCount)
        Quantities ownQ;
        std::vector<Real> ownReleasedMult, ownExactMult, ownDispWeight, ownMassWeight;
        std::vector< Pvector<Real,noVelComp> > ownDispVel;
        // per-tetrahedron scratch, reused
        std::vector<size_t> insideFlat;
        std::vector<Real>   insideRel, insideW;
        std::vector<double> exMom, exW;
        size_t nDegenerateInverse = 0, nReleased = 0;
        // PS_MASS_AUDIT=1: the mass of the tets this thread deposits (once per tet) and of the shares it writes
        double auditTets = 0., auditShares = 0.;
        size_t auditNTets = 0;
    };
    // bytes of one private copy of every accumulator this run uses: the Real ones only. A grid whose fixed-point
    // words stand in for it is EMPTY (the density under massFx, the moment grids under fxg: the words are shared),
    // and the counts and flags are shared too -- so the default single build copies NOTHING, and only
    // the float sums (PS_METAL_FLOAT_MOMENTS=1 / PS_METAL_FLOAT_MASS=1, the double build) still cost a copy per thread.
    auto vecBytes = [](auto const &v) -> double { return double(v.size()) * double(sizeof(v[0])); };
    double accBytes = vecBytes(releasedMult) + vecBytes(exactMult) + vecBytes(dispWeight) + vecBytes(massWeight)
                    + vecBytes(dispVel) + vecBytes(quantities->density) + vecBytes(quantities->velocity)
                    + vecBytes(quantities->velocity_gradient) + vecBytes(quantities->velocity_dispersion);
#ifdef SCALAR
    accBytes += vecBytes(quantities->scalar) + vecBytes(quantities->scalar_gradient);
#endif
    // (the thread setup below -- slab binning, private copies -- is booked to the deposit it serves)
    lap( metalDeposited ? PS_PH_COPY : PS_PH_SETUP );
    int nDepThreads = 1;
    // SLAB MODE. Private copies are what a fine grid cannot afford (one density+velocity copy is 24 GB at 1024^3).
    // When they would cut the threads below what the cores allow, each thread instead OWNS a slab of local x planes
    // and writes only its cells, straight into the real arrays, walking the slab's tetrahedra in the loop's order:
    // every cell gets its contributions in the single-threaded order, bit-identical to DTFE_DEPOSIT_THREADS=1 at ANY
    // thread count. A tet straddling k slabs is classified k times (its normalization needs all its samples), so
    // slabs are several tet extents wide. DTFE_DEPOSIT_SLABS=1 forces the mode (the tests), 0 forbids it.
    bool slabMode = false;
    int nSlabs = 0;
    std::vector<long> slabXLo, slabXHi;                     // each slab's local x range [lo, hi)
    std::vector< std::vector<uint32_t> > slabTetLists;      // per slab: its tetrahedra (ordinals, loop order)
    std::vector<Cell_handle> slabTets;                      // ordinal -> cell
    std::vector<uint8_t> slabFirst;                         // ordinal -> its first slab (per-tet tallies)
    PSLayoutCtx const layoutCtx{ nDepThreads, slabMode, nSlabs, slabXLo, slabXHi, slabTetLists, slabTets, slabFirst };
    psDepositThreadMode( passCtx, layoutCtx, metalDeposited, accBytes, deferredOrd );
    psSlabBinning( passCtx, layoutCtx, metalDeposited, deferredOrd );
    std::vector<DepositThread> depT( static_cast<size_t>(nDepThreads) );
    psDepositThreadsInit( passCtx, gridsCtx, depT, nDepThreads, slabMode );

    // One thread's share of the deposit: the loop over the cells, whose body is the serial loop's
    // -- its accumulators and scratch are this thread's under the SAME names (locals
    // shadowing the function's), and the do/while(false) keeps its 'continue' meaning "skip the
    // rest of this tetrahedron". Thread th of nT takes every nT-th block of 256 cells; after a GPU
    // deposit (deferred) it walks the deferred ordinals instead. The loop sits INSIDE the lambda:
    // a call per tetrahedron costs the single-thread path 19%.
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
        // references they would be reloaded after every push_back/accumulate that might alias them
        // (size_t / Real stores), which makes the wrapped loop 42% slower than the inline one.
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
        bool const auditL = massAudit;
        // the fixed-point mass grid (null: the float density) and how this thread adds to it: atomically when
        // the copy mode's threads share it, plainly when this thread owns its cells (one thread, or a slab)
        int64_t *const massFxL = TH.massFx;
        float const massFxScaleL = massFxScale;
        bool const massFxAtomicL = nT > 1 && slab < 0;
        // the stream count, the two flags and the caustic mask: the same rule (psAddCount, psSetFlag, psOrMask)
        bool const countAtomicL = massFxAtomicL;
        // the moment grids' fixed-point words (fxg; null: the grid's Real accumulator), ONE set for every
        // thread, added exactly as the mass words are (the same atomic rule); each share is the float product the
        // Real grid would have added. DW is not accumulated while the mass words are this pass's (massFxOn): it
        // becomes the density after the loop.
        int64_t *fxWL[PS_FX_CPU_N];
        float fxSL[PS_FX_CPU_N];
        for (int g = 0; g < PS_FX_CPU_N; ++g) { fxWL[g] = fxg[g].w.empty() ? nullptr : fxg[g].w.data(); fxSL[g] = fxg[g].scale; }
        bool const dwOffL = massFxOn;
        auto fxAddL = [&](int const g, size_t const at, Real const v)
        { psAddMassFixed( fxWL[g], at, psFixedShare( double(v), fxSL[g], g ), massFxAtomicL ); };
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
            if ( !metalDeposited ) markFlipped(geo, flippedFlag, slabLoL, slabHiL, countAtomicL);   // (GPU-deferred tets: the host loop marked them)
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
                if ( fixedShareTestNanEvery() && fixedShareTestNan( double(cell->vertex(0)->point()[0]), double(cell->vertex(0)->point()[1]),
                                                                    double(cell->vertex(0)->point()[NO_DIM-1]) ) )
                    tetMass = std::numeric_limits<Real>::quiet_NaN();   // DTFE_TEST_NAN_SHARE (ps_deposit_params.h)
                if ( auditL && ( !slabList || int(slabFirst[curOrd]) == slab ) )   // once per tet
                { TH.auditTets += double(tetMass); ++TH.auditNTets; }
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
                    // (2D too: the triangles around the three vertices; no A4 bit in 2D)
                    if ( userOptions.psCausticCusps and self.valid and self.spread > 0.
                         and std::fabs(self.critical) <= PSCausticClass::FOLD_BAND * self.spread )
                    {
                        /* STENCIL: every tetrahedron sharing a VERTEX with this one, not just the four
                           sharing a face. Four samples barely determine a 3-D gradient -- if they happen
                           to sit in one slab the fit is ill-conditioned along exactly the direction the
                           cusp test interrogates, and the transverse noise then reads as tangency. The
                           vertex-incident ring gives ~20-40 samples spread in every direction. Measured
                           on the 1-D wave at A*k = 1.8, which has NO cusps: the false-positive rate over
                           the fold cells falls from ~5-7% (face neighbours) to the figure the test
                           bounds. */
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
                           face neighbour once per shared vertex; those repeats would re-weight the fit
                           AND fill the size cap, so the gather would truncate part-way and drop the last
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
                // straddle is decided on the tet's own bbox: a tet whose bbox stays inside the grid has
                // no sample beyond it (a one-cell margin would flag every tet within a cell of a region
                // face, 14% of the TNG region's tets, all for nothing)
                if (!periodicL && (iMin[d] < 0 || iMax[d] > (int)nGridL[d])) straddle = true;
                // No one-cell sample margin (as in the GPU kernels): a sub-sample
                // sits at (k + 0.5)/nSub of its cell, >= 1/(2 nSub) inside it, so every sample in the
                // bbox lies in a cell of this window and a ring's samples would all be outside the tet
                // (each one tested; 1.8-2.7x the window cells of real TNG50 z=0 tets).
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

                // PASS 2 (exact): renormalize the shares to tetMass, exact per tet. A tet that leaves a
                // non-periodic grid is normalized by its WHOLE weight instead (the analytic volume, or the
                // linear profile's integral over the tet; max() guards rounding), so the grid keeps exactly
                // its inside share rather than piling the outside share onto a --box region's faces. A
                // --ps-window run normalizes EVERY tet this way (only its window pieces are clipped; it agrees
                // with the full run to float rounding). An all-empty window (a sliver below r3d's resolution,
                // or a tet wholly beyond the grid) falls through to the sampled path, except under
                // --ps-window: no volume inside the window, so the tet deposits nothing here.
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
                        if (fDenL)
                        {
                            if ( massFxL ) psAddMassFixed( massFxL, flatIdx, psMassFixedShare( double(massShare), massFxScaleL ), massFxAtomicL );
                            else quantities->density[flatIdx] += massShare;
                        }
                        if (auditL) TH.auditShares += double(massShare);
                        if (needWeightL && !weightIsDensityL)
                        {
                            if ( fxWL[PS_FX_MOMW] ) fxAddL( PS_FX_MOMW, flatIdx, momShare );
                            else massWeight[flatIdx] += momShare;
                        }

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
                            {
                                if ( fxWL[PS_FX_MOM] )
                                    for (size_t j = 0; j < noVelComp; ++j) fxAddL( PS_FX_MOM, flatIdx*noVelComp + j, velBar[j] * momShare );
                                else quantities->velocity[flatIdx] += velBar * momShare;
                            }
                        }
                        if (fVGradL)
                        {
                            Pvector<Real,noGradComp> grad;
                            for (size_t j = 0; j < noVelComp; ++j)
                                for (int i = 0; i < NO_DIM; ++i)
                                    grad[j*NO_DIM+i] = velGrad[i][j];
                            if ( fxWL[PS_FX_GRAD] )
                                for (size_t q = 0; q < noGradComp; ++q) fxAddL( PS_FX_GRAD, flatIdx*noGradComp + q, grad[q] * momShare );
                            else quantities->velocity_gradient[flatIdx] += grad * momShare;
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
                                    // (the Real grid's add keeps its one expression: a contracted multiply-add there)
                                    if ( fxWL[PS_FX_M2] ) fxAddL( PS_FX_M2, flatIdx*noDispComp + c2, massShare * ( velBar[a] * velBar[b] + Real(gcg) ) );
                                    else quantities->velocity_dispersion[flatIdx][c2] += massShare * ( velBar[a] * velBar[b] + Real(gcg) );
                                    ++c2;
                                }
                            if ( dispNeedsOwnWeightL )
                            {
                                if ( fxWL[PS_FX_DW] ) fxAddL( PS_FX_DW, flatIdx, massShare );
                                else if ( !dwOffL ) dispWeight[flatIdx] += massShare;
                                if ( fxWL[PS_FX_DV] )
                                    for (size_t j = 0; j < noVelComp; ++j) fxAddL( PS_FX_DV, flatIdx*noVelComp + j, velBar[j] * massShare );
                                else dispVel[flatIdx] += velBar * massShare;
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
                            if ( fxWL[PS_FX_SCAL] )
                                for (size_t j = 0; j < noScalarComp; ++j) fxAddL( PS_FX_SCAL, flatIdx*noScalarComp + j, scalarVal[j] * momShare );
                            else quantities->scalar[flatIdx] += scalarVal * momShare;
                        }
                        if (fSGradL)
                        {
                            Pvector<Real,noScalarGradComp> sgrad;
                            for (size_t j = 0; j < noScalarComp; ++j)
                                for (int i = 0; i < NO_DIM; ++i)
                                    sgrad[j*NO_DIM+i] = sGrad[i][j];
                            if ( fxWL[PS_FX_SG] )
                                for (size_t q = 0; q < noScalarGradComp; ++q) fxAddL( PS_FX_SG, flatIdx*noScalarGradComp + q, sgrad[q] * momShare );
                            else quantities->scalar_gradient[flatIdx] += sgrad * momShare;
                        }
    #endif
                        psAddCount( streamCount[flatIdx], countAtomicL );   // raw tet-touch count -> '.tetTouch'
                        if ( fxWL[PS_FX_SV] ) fxAddL( PS_FX_SV, flatIdx, Real( m[0] / double(cellVolumeL) ) );
                        else exactMult[flatIdx] += Real( m[0] / double(cellVolumeL) );   // exact multiplicity share
                        if ( psCausticsL )
                            psOrMask( orientBits[flatIdx], (PSCausticClass::CausticMask)((geo.cellDet > 0. ? 1 : 2) | tetCausticClass), countAtomicL );
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

                    // Offset from (wrapped) vertex 0 for the deposited moments: the plain float
                    // expression -- the exact test decides only WHICH samples a tetrahedron keeps,
                    // not the bits a kept sample deposits.
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
            // the in-grid samples would pile the outside share onto the faces of a --box region cut from a
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
            // than the tetrahedron's mass.
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

                // density accumulates MASS here; it is divided by the cell volume once at the end. Single build:
                // into the fixed-point grid (massFx), exact; converted to the float density once after the loop.
                if (fDenL)
                {
                    if ( massFxL ) psAddMassFixed( massFxL, flatIdx, psMassFixedShare( double(massShare), massFxScaleL ), massFxAtomicL );
                    else quantities->density[flatIdx] += massShare;
                }
                if (auditL) TH.auditShares += double(massShare);
                if (needWeightL && !weightIsDensityL)
                {
                    if ( fxWL[PS_FX_MOMW] ) fxAddL( PS_FX_MOMW, flatIdx, momShare );
                    else massWeight[flatIdx] += momShare;
                }

                // Weighted velocity moment sum(w_s v_s); also needed for the dispersion (uses <v>).
                Pvector<Real,noVelComp> velVal;
                if (needVelValuesL)
                {
                    velVal = cell->vertex(0)->info().velocity();
                    for (int i = 0; i < NO_DIM; ++i)
                        for (size_t j = 0; j < noVelComp; ++j)
                            velVal[j] += velGrad[i][j] * rel[i];
                    if (haveVelL)
                    {
                        if ( fxWL[PS_FX_MOM] )
                            for (size_t j = 0; j < noVelComp; ++j) fxAddL( PS_FX_MOM, flatIdx*noVelComp + j, velVal[j] * momShare );
                        else quantities->velocity[flatIdx] += velVal * momShare;
                    }
                }

                if (fVGradL)
                {
                    Pvector<Real,noGradComp> grad;
                    for (size_t j = 0; j < noVelComp; ++j)
                        for (int i = 0; i < NO_DIM; ++i)
                            grad[j*NO_DIM+i] = velGrad[i][j];
                    if ( fxWL[PS_FX_GRAD] )
                        for (size_t q = 0; q < noGradComp; ++q) fxAddL( PS_FX_GRAD, flatIdx*noGradComp + q, grad[q] * momShare );
                    else quantities->velocity_gradient[flatIdx] += grad * momShare;
                }

                // velocity dispersion: weighted second moment sum(w_s v_i v_j) (upper triangle).
                // After normalization -> sigma_ij = <v_i v_j> - <v_i><v_j>.
                if (fVDispL)
                {
                    size_t c = 0;
                    for (int i = 0; i < NO_DIM; ++i)
                        for (int j = i; j < NO_DIM; ++j, ++c)
                        {
                            if ( fxWL[PS_FX_M2] ) fxAddL( PS_FX_M2, flatIdx*noDispComp + c, massShare * velVal[i] * velVal[j] );
                            else quantities->velocity_dispersion[flatIdx][c] += massShare * velVal[i] * velVal[j];
                        }
                    if ( dispNeedsOwnWeightL )
                    {
                        if ( fxWL[PS_FX_DW] ) fxAddL( PS_FX_DW, flatIdx, massShare );
                        else if ( !dwOffL ) dispWeight[flatIdx] += massShare;
                        if ( fxWL[PS_FX_DV] )
                            for (size_t j = 0; j < noVelComp; ++j) fxAddL( PS_FX_DV, flatIdx*noVelComp + j, velVal[j] * massShare );
                        else dispVel[flatIdx] += velVal * massShare;
                    }
                }

    #ifdef SCALAR
                if (fScalL)
                {
                    Pvector<Real,noScalarComp> scalarVal = cell->vertex(0)->info().myScalar();
                    for (int i = 0; i < NO_DIM; ++i)
                        for (size_t j = 0; j < noScalarComp; ++j)
                            scalarVal[j] += sGrad[i][j] * rel[i];
                    if ( fxWL[PS_FX_SCAL] )
                        for (size_t j = 0; j < noScalarComp; ++j) fxAddL( PS_FX_SCAL, flatIdx*noScalarComp + j, scalarVal[j] * momShare );
                    else quantities->scalar[flatIdx] += scalarVal * momShare;
                }
                if (fSGradL)
                {
                    Pvector<Real,noScalarGradComp> sgrad;
                    for (size_t j = 0; j < noScalarComp; ++j)
                        for (int i = 0; i < NO_DIM; ++i)
                            sgrad[j*NO_DIM+i] = sGrad[i][j];
                    if ( fxWL[PS_FX_SG] )
                        for (size_t q = 0; q < noScalarGradComp; ++q) fxAddL( PS_FX_SG, flatIdx*noScalarGradComp + q, sgrad[q] * momShare );
                    else quantities->scalar_gradient[flatIdx] += sgrad * momShare;
                }
    #endif

                // the stream multiplicity (see streamCount): one count per real SAMPLE; a released
                // tet (never sampled) adds its volume fraction; a sampled tet that caught no sample
                // (useCentroid) moves mass only. Under psExactL the counter is the tet-touch count.
                if ( useCentroid )
                    psSetFlag( subSampleMassFlag[flatIdx], countAtomicL );   // mass from a tet with no sample here
                if ( psExactL )
                    psAddCount( streamCount[flatIdx], countAtomicL );
                else if ( released )
                {
                    if ( fxWL[PS_FX_RM] ) fxAddL( PS_FX_RM, flatIdx, Real( geo.cellAbsDet / factorial(NO_DIM) / double(cellVolumeL) ) );
                    else releasedMult[flatIdx] += Real( geo.cellAbsDet / factorial(NO_DIM) / double(cellVolumeL) );
                }
                else if ( !useCentroid )
                    psAddCount( streamCount[flatIdx], countAtomicL );
                // --ps-exact-deposit fall-through (empty clip window, released or sub-resolution
                // tets): the tet's volume is shared equally among its nInside samples, so each
                // carries V_tet/nInside -- for the monolithic centroid fallback (nInside==1) that
                // is exactly V_tet/V_cell, the same volume fraction the exact path deposits.
                if ( psExactL )
                {
                    if ( fxWL[PS_FX_SV] ) fxAddL( PS_FX_SV, flatIdx, Real( geo.cellAbsDet / factorial(NO_DIM) / double(cellVolumeL) ) / Real(nShare) );
                    else exactMult[flatIdx] += Real( geo.cellAbsDet / factorial(NO_DIM) / double(cellVolumeL) )
                                               / Real(nShare);
                }
                if ( psCausticsL )
                    // fold parity (sign of det(Ax)) plus this tetrahedron's stratification bits
                    psOrMask( orientBits[flatIdx], (PSCausticClass::CausticMask)((geo.cellDet > 0. ? 1 : 2) | tetCausticClass), countAtomicL );
            }
        } while (false);
        }   // end cell loop
    };

    double const auditGridBefore = ( massAudit && field.density ) ? gridSum() : 0.;   // the CPU loop's starting grid
    auto const tCpuLoop0 = std::chrono::steady_clock::now();
    if ( metalDeposited && !deferredOrd.empty() && !slabMode )
        depositLoop( depT[0], 0, 1, true, -1 );
    else if ( !metalDeposited || slabMode )   // (after a GPU deposit, slab mode means its deferred cells)
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
        // sum the private copies into the real arrays in thread order, cell ranges in parallel (slab mode has none:
        // every thread wrote its own cells; nor has a copy mode whose accumulators are all shared, accBytes == 0)
        if ( nT > 1 && !slabMode && accBytes > 0. )
        {
            auto merge = [&](size_t part)
            {
                size_t const lo = totalGrid * part / nT, hi = totalGrid * (part + 1) / nT;
                auto addInto = [lo, hi](auto &real, auto const &own) { if (!own.empty()) for (size_t c = lo; c < hi; ++c) real[c] += own[c]; };
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
                    addInto( releasedMult, D.ownReleasedMult ); addInto( exactMult, D.ownExactMult );
                    addInto( dispWeight, D.ownDispWeight );     addInto( dispVel, D.ownDispVel );
                    addInto( massWeight, D.ownMassWeight );
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
        audit.cpuTets += D.auditTets; audit.cpuShares += D.auditShares; audit.nCpuTets += D.auditNTets;
    }
    lap( metalDeposited ? PS_PH_DEFERRED : PS_PH_CPU );
    if ( massAudit && field.density ) audit.cpuGain = gridSum() - auditGridBefore;
    std::vector<DepositThread>().swap( depT );   // the private copies
    std::vector<Cell_handle>().swap( slabTets );
    std::vector< std::vector<uint32_t> >().swap( slabTetLists );
    std::vector<uint8_t>().swap( slabFirst );
    if ( metalDeposited && !deferredOrd.empty() && not userOptions.psSuppressGridStats )
        message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " deposited the "
                << MESSAGE::cMagenta() << deferredOrd.size() << MESSAGE::cReset() << " GPU-deferred tetrahedra on the CPU in "
                << std::chrono::duration<double>(std::chrono::steady_clock::now() - tCpuLoop0).count() << " s ("
                << nDepThreads << ( nDepThreads == 1 ? " thread" : " threads" ) << ").\n" << MESSAGE::Flush;

    // Whichever deposit ran (GPU success above skipped the CPU loop; a GPU FAILURE fell back
    // to the CPU loop, which needed the triangulation), its last read of dt is behind us.
    // For the final call on an internally-owned dt, free the triangulation NOW -- during the
    // normalization below and the caller-side merge into the shared grids -- instead of at
    // partition scope end. ~650 B/vertex, ~5 GB per concurrent partition at production scale.
    if ( mayClearDT )
        dt.clear();

    psFixedToReal( passCtx, gridsCtx, fxToReal );

    psPassMessages( passCtx, nDegenerateInverse, nReleased );

    // turn density-weighted moments into mass-weighted means sum(rho_s f_s)/sum(rho_s). Serial: here,
    // BEFORE the density grid is converted to rho/rho_bar (it doubles as the weight when the density
    // field is selected -- see weightIsDensity above). deferNorm: hand un-normalized moments to the
    // caller, which sums across partitions and normalizes once (the only order correct for cells
    // spanning partitions); the weight travels as mass_weight, or is reconstructed from the summed
    // density by normalizePhaseSpace when it aliases the density grid.
    Real const invSamples = Real(1.) / Real(nSamplesPerCell);
    psNormalizeMoments( passCtx, gridsCtx );

    // The density grid holds MASS (the per-tet shares): divide by the cell volume and by averageDensity, so the
    // written field is rho/rho_bar (the SAME convention as standard DTFE). Mass conservation makes the box mean 1 for
    // any nSub, up to the accumulators: the single build's exact fixed-point grid (massFx above) is rounded once per
    // cell to float, the double build's sums are good to ~1e-15, and the float grid of PS_METAL_FLOAT_MASS=1 /
    // PS_GPU_FLOAT_MASS=1 loses shares below half an ulp (~2% at 1e7 deposits into one cell). Tetrahedra the cell
    // filters drop take their mass along in every build. The stream count is averaged over the nSub^NO_DIM
    // sub-samples below.
    double const auditGridPre = ( massAudit && field.density ) ? gridSum() : -1.;
    if ( field.density )
    {
        Real const invRhoBar = userOptions.averageDensity > Real(0.)
                             ? Real(1.) / userOptions.averageDensity : Real(1.);
        for (size_t i = 0; i < totalGrid; ++i)
            quantities->density[i] *= invRhoBar / cellVolume;
    }
    psMassAuditLine( passCtx, massAudit, audit, gridSum, cellVolume, auditGridPre );

    psStreamStats( passCtx, gridsCtx, invSamples, nSamplesPerCell );

    psExportStreams( passCtx, gridsCtx, invSamples );

    // Record the sub-grid box so the caller (addFromSubgrid) can map results back into the full shared grid.
    for (int d = 0; d < NO_DIM; ++d)
    {
        quantities->ps_subOrigin[d] = subOrigin[d];
        quantities->ps_subDims[d]   = subDims[d];
    }

    // this pass's phase split and the run's totals so far (psPhaseTotalUs)
    lap( PS_PH_NORM );
    psPhaseReport( passCtx, phaseSec, metalDeposited, deferredOrd );
}

#endif // PHASE_SPACE
