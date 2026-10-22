#ifndef PS_DEPOSIT_PARAMS_H
#define PS_DEPOSIT_PARAMS_H

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "../env_knobs.h"

// Per-tetrahedron record of the GPU deposit: PS_TET_STRIDE floats per tet in the 'verts'
// buffer, laid out as documented at TET_STRIDE in metal/ps_deposit.metal (which must match).
constexpr int PS_TET_STRIDE = 16;

// Fills one record. cv = the tet's four vertices in the CANONICAL frame of the exact inside
// test (ps_exact_inside.h canonicalCoord: every periodic image of a particle at one position),
// in the build's own Real -- the very coordinates that test compares in; gridLo/gridLen/nGrid =
// the deposit grid, as the CPU test forms its sample positions
// lo + ((g + frac) * len) / n; canonShift = how far the canonical frame sits above the grid (the
// box length when periodic, else 0); wrapL = the box length for the minimum-image edge wrap
// (0 = no wrap); centroidFlat = the flat sub-grid index of the centroid-fallback cell as the
// CPU deposit computes it (0xFFFFFFFF = outside this partition, dropped).
// Vertex 0 is stored as a cell index plus an in-cell fraction and the edges as double
// differences rounded once to float, so the kernel forms every sample offset from small numbers.
// The double build passes its DOUBLE vertices: rounding them to float would move a vertex by up to
// half a float ulp of its canonical coordinate (~8e-6 Mpc in a 100 Mpc box), which the kernel's
// error bound (the record's own roundings) does not cover, so a sub-sample that close to a face
// could be counted on the wrong side of the double CPU test (tests/ps_double_check.sh D5 compares
// the stream counts with the double CPU run). A double edge difference rounds by 2^-53 of the edge
// before its float rounding, far inside the bound's safety factor of two; float vertices give the
// same bytes either way.
template <typename V>
inline void psFillTetRecord(float *rec, V const cv[4][3], double const gridLo[3],
                            double const gridLen[3], int32_t const nGrid[3],
                            double const canonShift[3], double const wrapL[3],
                            uint32_t const centroidFlat)
{
    for (int d = 0; d < 3; ++d)
    {
        double const a0 = (double(cv[0][d]) - (gridLo[d] + canonShift[d])) * double(nGrid[d]) / gridLen[d];
        double const ic = std::floor(a0);
        int32_t const ic0 = int32_t(ic);
        std::memcpy(&rec[d], &ic0, sizeof(ic0));
        rec[3 + d] = float(a0 - ic);
    }
    for (int e = 0; e < 3; ++e)
        for (int d = 0; d < 3; ++d)
        {
            double r = double(cv[e+1][d]) - double(cv[0][d]);   // exact for float vertices
            if (wrapL[d] > 0.)
            {
                if (r >  0.5 * wrapL[d]) r -= wrapL[d];
                else if (r < -0.5 * wrapL[d]) r += wrapL[d];
            }
            rec[6 + e*3 + d] = float(r);
        }
    std::memcpy(&rec[15], &centroidFlat, sizeof(centroidFlat));
}

// The deposit's MOMENT grids are exact 64-bit fixed-point sums, like the mass grid: float sums would depend on
// the order the GPU atomics land in (two runs would differ in the last bits of every velocity, dispersion and
// gradient cell) and, on the CPU, on the thread count (thread-private float copies summed in thread order).
// Each share v*w becomes the integer rint(v*w * scale[g]) (scale = 2^k, so the product is
// exact and both sides round it the same way: psMassFixedShare's rule) and the integers are added -- in any
// order, by any number of threads, with the same result. Grid g of this list ([g] = the index into
// PSDepositParams.fxScale, PSGpuGrids::fx and the CPU's own words):
//   MOM   sum(wm v_j)             the velocity moment (3 components)
//   M2    sum(w v_i v_j)          the dispersion's second moment (6, upper triangle)
//   GRAD  sum(wm dv_i/dx_j)       the velocity-gradient moment (9)
//   MOMW  sum(wm)                 the moment normalizer: the volume shares under --ps-volume-weighted, else (CPU,
//                                 no density selected) the mass shares
//   SV    sum(V_int)/V_cell       the exact deposit's cell-mean multiplicity (1)
//   DV    sum(w v_j)              the dispersion's own mass-weighted mean under --ps-volume-weighted (3)
//   DW    sum(w)                  its normalizer (1). On the GPU it is NOT deposited when the mass grid is fixed:
//                                 it sums the very shares the mass grid sums, so the caller takes the mass words
//   SCAL  sum(wm s)               the scalar moment (one component)
//   SG    sum(wm ds/dx_i)         the scalar-gradient moment (3)
//   RM    sum(V_tet)/V_cell       --ps-halo-release on the sampled deposit: the released tets' multiplicity (CPU
//                                 accumulator only; the GPU never sees a released tet)
// w = the mass share, wm = the moment weight (w, or the Eulerian-volume share under --ps-volume-weighted).
enum PSFxGrid { PS_FX_MOM = 0, PS_FX_M2, PS_FX_GRAD, PS_FX_MOMW, PS_FX_SV, PS_FX_DV, PS_FX_DW, PS_FX_SCAL, PS_FX_SG,
                PS_FX_N,                 // the grids a GPU deposit can hold as words (PSDepositParams.fxScale)
                PS_FX_RM = PS_FX_N,      // + the CPU-only one
                PS_FX_CPU_N };
// PSDepositParams.fxMask: bit g (1u << g) = grid g is a fixed-point sum (its words bound at buffer 22 + its slot,
// metal/ps_deposit.metal), except bit PS_FX_DW, which means "dispw is the mass grid: not deposited"; and
// PS_FX_SUMW_BIT = the exact items' per-tet weight totals are fixed-point sums too (see depositExactItems).
static uint32_t const PS_FX_SUMW_BIT = 1u << 16;

// The unit of each moment grid: scale[g] = 2^k from a bound B of the sum of |share| over everything
// the pass can deposit into grid g -- no cell, holding part of that sum, can exceed it. psMassFixedScale (the mass
// grid's rule) puts 2B at 2^61..2^62 units, so B sits at 2^60..2^61: a factor 4 below the 2^63 of the signed words,
// for the GPU's float rounding of a value (the kernel's float inverse of a tetrahedron near the degeneracy cut
// amplifies it; a few per cent at worst, as for the standard deposit: dtfe_gpu_fixed.h). The bound is a sum
// over the tetrahedra of (its weight total) x (the largest |value| it can deposit) -- PSFixedBounds::addTet. The
// resolution this leaves: a cell holding 2^-40 of B keeps 2^20 units, a few times finer than its float.
// False (the grid keeps the float sums) for a bound that is not finite or gives no usable power of two; an empty
// bound (nothing to deposit) gets unit 1.
inline bool psMassFixedScale(double totalMass, float &scale);   // (below)
inline bool psMomentFixedScale(double bound, float &scale)
{
    scale = 1.f;
    if ( !std::isfinite( bound ) || bound < 0. ) return false;
    if ( bound == 0. ) return true;
    return psMassFixedScale( 2. * bound, scale );
}

// The bounds B[g] of one pass, summed tetrahedron by tetrahedron. w = the tet's mass (its shares
// add up to it), V = its Eulerian volume |det|/NO_DIM!, wm = V under --ps-volume-weighted else w; vMax = the
// largest |velocity component| at its vertices (a sample, a piece's centroid or the centroid fallback lies in
// the tet, where the linear profile is a convex combination of the vertex values; the exact deposit's second
// moment <v_a v_b> over a piece is bounded by vMax^2 likewise), gMax = the largest |dv_i/dx_j| of its constant
// gradient, sMax / sgMax the same for the scalar; cellVol = the grid cell's volume.
struct PSFixedBounds
{
    double B[PS_FX_CPU_N] = {};
    void addTet(double w, double V, double wm, double vMax, double gMax, double sMax, double sgMax, double cellVol)
    {
        B[PS_FX_MOM]  += wm * vMax;
        B[PS_FX_M2]   += w * vMax * vMax;
        B[PS_FX_GRAD] += wm * gMax;
        B[PS_FX_MOMW] += wm;
        B[PS_FX_SV]   += V / cellVol;
        B[PS_FX_DV]   += w * vMax;
        B[PS_FX_DW]   += w;
        B[PS_FX_SCAL] += wm * sMax;
        B[PS_FX_SG]   += wm * sgMax;
        B[PS_FX_RM]   += V / cellVol;
    }
    void add(PSFixedBounds const &o) { for (int g = 0; g < PS_FX_CPU_N; ++g) B[g] += o.B[g]; }
};

// The units a pass's caller hands the GPU host: scale[g] > 0 = grid g as a fixed-point sum in units
// of 1/scale[g] (the host still deposits only the grids its flags select), 0 = float atomics
// (PS_GPU_FLOAT_MOMENTS=1, or a bound without a unit); sumW = the exact items' per-tet
// weight totals as fixed-point sums as well.
struct PSFixedUnits
{
    float scale[PS_FX_N] = {};
    bool  sumW = false;
    // the total the mass grid's unit is taken from (psMassFixedScale) when > 0 (a --ps-window pass, so
    // that the GPU and the CPU window deposits take the same unit -- ps_interpolation.cc fxBoundWalk); 0 = the host's
    // own sum of the masses it is handed plus massExtra
    double massTotal = 0.;
};

// The non-finite guard of the fixed-point shares. rint(share * scale) -> int64 is UNSPECIFIED for a
// NaN, an infinity or a product beyond the signed 64-bit range (C++ llrint / the GPU's float->long conversion:
// arm64 gives 0 for a NaN and saturates the rest, x86-64 gives 0x8000000000000000 for all of them), so one such
// share would silently corrupt an exact sum, where a float sum would show a NaN. The policy, the same
// on the CPU, in Metal and in CUDA/HIP (and its emulation): a share whose product is not below 2^63 in magnitude
// -- NaN, +-inf and an overflow alike -- adds NOTHING and is COUNTED, per grid slot (below); the host reports the
// run's counts once (fixedShareDropsReact, DTFE.cpp). Hot path: one compare. The kernels test the float product's
// bits, (bits & 0x7fffffff) < 0x5f000000 (= 2^63), as fast math cannot fold an integer compare away; the CPU
// tests the double product, which is the same float times the same 2^k, exactly. A finite in-range share takes
// the same integer as without the guard, so normal data is unaffected bit for bit.
// The count slots: PS-DTFE g = PSFxGrid (0 .. PS_FX_CPU_N-1), the mass grid, the exact items' per-tet weight totals;
// standard DTFE 0..4 = its grids (dtfe_gpu_fixed.h: density, velocity, velocity gradient, scalar, its gradient).
enum FixedShareSlot { FX_DROP_MASS = 10, FX_DROP_SUMW = 11, FX_DROP_SLOTS = 16 };
static_assert( int(PS_FX_CPU_N) <= int(FX_DROP_MASS), "a moment grid's slot would be the mass grid's" );
// The run's dropped shares by slot: every pass (CPU, or a GPU host after a deposit it kept) adds its counts here
inline std::atomic<unsigned long long> fixedShareDrops[FX_DROP_SLOTS] = {};
// Counts n dropped shares of a slot (the cold path of the guard; also the GPU hosts' per-pass totals)
inline void fixedShareDropped(int const slot, unsigned long long const n = 1)
{
    fixedShareDrops[slot].fetch_add( n, std::memory_order_relaxed );
}
// The guard itself: the share's integer in q, or false (q = 0, counted in 'slot') for a product that has none
inline bool psFixedShareOk(double const product, int const slot, int64_t &q)
{
    if ( !( std::fabs( product ) < 9223372036854775808.0 ) )   // 2^63; false for a NaN too
    {
        fixedShareDropped( slot );
        q = 0;
        return false;
    }
    q = int64_t( std::llrint( product ) );
    return true;
}

// The guard's test hook: DTFE_TEST_NAN_SHARE=<k> makes the deposit's input NaN for about one
// tetrahedron in k -- PS-DTFE its mass (the GPU extraction and the CPU loop alike, so every share of it is NaN), the
// standard GPU deposit its vertex densities -- picked by a hash of its first vertex's coordinates (fixedShareTestNan),
// which every path sees the same way. The units and bounds leave those values out (a NaN is no maximum, and the GPU
// hosts' mass total skips it), so the rest of the grid stays exact and the guard drops and counts the NaN shares.
// 0 / unset: off (tests/ps_fixed_moments_check.sh, tests/dtfe_gpu_retry_check.sh use it).
inline unsigned long fixedShareTestNanEvery()
{
    static unsigned long const k = [] { long const v = Env::integer( "DTFE_TEST_NAN_SHARE", 0 );
                                        return v > 0 ? (unsigned long)v : 0ul; }();
    return k;
}
inline bool fixedShareTestNan(double const x, double const y, double const z)
{
    unsigned long const k = fixedShareTestNanEvery();
    if ( k == 0 ) return false;
    uint64_t h = 0x9e3779b97f4a7c15ull;
    for (double const c : { x, y, z })
    {
        uint64_t b;
        std::memcpy( &b, &c, sizeof b );
        h = ( h ^ b ) * 0xbf58476d1ce4e5b9ull;
        h ^= h >> 31;
    }
    return h % k == 0;
}

// One share of a moment grid in fixed-point units: the float product v*w the deposit formed (the float a
// float sum would add), times the 2^k unit -- exact in double -- rounded to nearest even: the integer rint(v*w *
// scale) the kernels add (metal/ps_deposit.metal addFixed). psMassFixedShare's rule, for any grid. 'slot' = the
// grid's PSFxGrid index: a share without an integer (the guard above) adds 0 and is counted there.
inline int64_t psFixedShare(double share, float scale, int slot)
{
    int64_t q;
    psFixedShareOk( share * double(scale), slot, q );
    return q;
}

// Host-side parameter block of the PS-DTFE GPU deposit, shared by the Metal host
// (ps_metal_host.cc) and metal/validate_deposit.cpp. MUST match the MSL
// 'DepositParams' struct in metal/ps_deposit.metal byte-for-byte -- that copy is the one
// intentional duplicate (MSL cannot include this header).
struct PSDepositParams
{
    float    boxLo[3];
    float    dx[3];
    int32_t  nGrid[3];
    int32_t  nSub;
    int32_t  periodic;
    int32_t  subOrigin[3];   // partition sub-grid box (full grid: origin 0, dims = nGrid)
    int32_t  subDims[3];
    // field flags: when 0 the corresponding moment grid is a 4-byte dummy the kernel never
    // touches (mass and streams are always deposited)
    int32_t  fVel;      // velocity moments (velocity or dispersion selected)
    int32_t  fDisp;     // second moments (24 B per cell)
    int32_t  fGrad;     // velocity-gradient moments (36 B per cell)
    int32_t  fLinear;   // --ps-linear-deposit: density-weighted (renormalized) sample shares
    int32_t  fVolW;     // --ps-volume-weighted: moment weight = V_eul shares (mass grid unchanged); momw normalizer grid deposited
    int32_t  fCaustic;  // --ps-caustics: atomic-OR the per-tet orientation bits (1 = det>0, 2 = det<0) into the caustic grid
    int32_t  fExact;    // --ps-exact-deposit: analytic tet-cell clipping (float32 r3d port in the kernel) instead of nSub^3 sampling; nSub ignored
    uint32_t nTet;
    int32_t  fScal;     // scalar field (one component): vertex scalars in buffer 18, the moment grid in 19
    int32_t  fSGrad;    // scalar gradient (3 components): its moment grid in buffer 20
    int32_t  windowMode; // --ps-window: the sub-box is a WINDOW of the full grid -- count every sample / clipped
                         // piece of a tet (the full run's normalization), deposit only those inside the sub-box
    int32_t  massFixed;  // the mass grid as an exact 64-bit fixed-point sum (buffer 21) instead of float atomics (buffer 3)
    float    massScale;  // its unit, 1/massScale mass: a power of two (psMassFixedScale)
    // the moment grids as exact 64-bit fixed-point sums (PSFxGrid above). fxMask bit g = grid g is one
    // (its words at buffer 22 + its slot, see metal/ps_deposit.metal), in units of 1/fxScale[g]; PS_FX_SUMW_BIT =
    // the exact items' weight totals too. 0 = every moment grid a float sum (the harness's default).
    uint32_t fxMask;
    float    fxScale[PS_FX_N];
};

// The unit of the mass grid's fixed-point sum (the kernel's DepositParams.massFixed says why):
// scale = 2^k with the mass handed to the deposit (a partition's tets) at 2^61..2^62 units -- the finest unit
// at which no cell overflows the signed 64-bit sum, even holding all of it plus every share's rounding. The
// unit must be fine: with the mean TET mass at 2^32 units, the 3 kpc cells of a --ps-window zoom
// (pieces of ~1e-8 of a tet) would come out 5e-4 off. Here a cell holding 2^-40 of the partition's
// mass still has 2^21 units. False when the mass gives no usable power of two (the caller keeps the float grid).
inline bool psMassFixedScale(double totalMass, float &scale)
{
    if ( !(totalMass > 0.) || !std::isfinite( totalMass ) ) return false;
    int eTotal = 0;
    std::frexp( totalMass, &eTotal );                   // total < 2^eTotal
    int const k = 62 - eTotal;
    if ( k < -120 || k > 120 ) return false;           // far outside any mass unit: float range
    scale = std::ldexp( 1.f, k );
    return true;
}

// The 64-bit integer of a cell from its two fixed-point words (low, high; two's complement).
inline int64_t psMassFixedWord(uint32_t lo, uint32_t hi)
{
    return int64_t( ( uint64_t(hi) << 32 ) | uint64_t(lo) );
}

// The mass of one cell from its fixed-point sum.
inline double psMassFixedMass(int64_t q, float scale)
{
    return double(q) / double(scale);
}

// The mass of one cell from its two fixed-point words.
inline double psMassFixedValue(uint32_t lo, uint32_t hi, float scale)
{
    return psMassFixedMass( psMassFixedWord(lo, hi), scale );
}

// One mass share in fixed-point units: the integer the kernels add, rint(w * massScale) in metal/
// ps_deposit.metal depositMass (ps_gpu_cuda.cu the same). The scale is 2^k, so the float product there and the
// double product here are both exact, and rint (round to nearest even) gives the same integer: a share adds the
// same units whichever side deposits it. The CPU deposit (ps_interpolation.cc) adds these into the partition's
// fixed-point grid -- the GPU's own words when the GPU deposited the partition -- so the mass grid is ONE exact
// integer sum, independent of the order and the thread count, and is converted to float once.
inline int64_t psMassFixedShare(double share, float scale)
{
    int64_t q;
    psFixedShareOk( share * double(scale), FX_DROP_MASS, q );   // (the non-finite guard above)
    return q;
}

// The GPU hosts hand the kernels' (low, high) uint32 word pairs over as int64_t by copying the bytes
// (ps_metal_host.cc, ps_gpu_cuda.cu): that IS the integer on a little-endian host, which every Metal (Apple
// Silicon) and CUDA/HIP (x86-64, aarch64) host is.
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "the GPU hosts copy the fixed-point mass words as little-endian int64_t (ps_deposit_params.h)"
#endif

#endif
