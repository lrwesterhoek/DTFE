#ifndef PS_DEPOSIT_PARAMS_H
#define PS_DEPOSIT_PARAMS_H

#include <cmath>
#include <cstdint>
#include <cstring>

// Per-tetrahedron record of the GPU deposit: PS_TET_STRIDE floats per tet in the 'verts'
// buffer, laid out as documented at TET_STRIDE in metal/ps_deposit.metal (which must match).
constexpr int PS_TET_STRIDE = 16;

// Fills one record. cv = the tet's four vertices in the CANONICAL frame of the exact inside
// test (ps_exact_inside.h canonicalCoord: every periodic image of a particle at one float
// position); gridLo/gridLen/nGrid = the deposit grid, as the CPU test forms its sample positions
// lo + ((g + frac) * len) / n; canonShift = how far the canonical frame sits above the grid (the
// box length when periodic, else 0); wrapL = the box length for the minimum-image edge wrap
// (0 = no wrap); centroidFlat = the flat sub-grid index of the centroid-fallback cell as the
// CPU deposit computes it (0xFFFFFFFF = outside this partition, dropped).
// Vertex 0 is stored as a cell index plus an in-cell fraction and the edges as exact double
// differences rounded once, so the kernel forms every sample offset from small numbers.
inline void psFillTetRecord(float *rec, float const cv[4][3], double const gridLo[3],
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
            double r = double(cv[e+1][d]) - double(cv[0][d]);   // exact: a difference of floats
            if (wrapL[d] > 0.)
            {
                if (r >  0.5 * wrapL[d]) r -= wrapL[d];
                else if (r < -0.5 * wrapL[d]) r += wrapL[d];
            }
            rec[6 + e*3 + d] = float(r);
        }
    std::memcpy(&rec[15], &centroidFlat, sizeof(centroidFlat));
}

// The deposit's MOMENT grids as exact 64-bit fixed-point sums (2026-10-14), like the mass grid since
// 2026-10-08/12. Before, every grid but the mass was a float sum: on the GPU float atomics, whose order (and so
// whose rounding) changed run to run -- two runs differed in the last bits of every velocity, dispersion and
// gradient cell (tests/gpu_retry_check.sh had to allow 1e-6 of the peak, and that tolerance flaked under GPU
// load) -- and on the CPU thread-private float copies summed in thread order, so the grids depended on the
// thread count. Now each share v*w becomes the integer rint(v*w * scale[g]) (scale = 2^k, so the product is
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

// The unit of each moment grid (2026-10-14): scale[g] = 2^k from a bound B of the sum of |share| over everything
// the pass can deposit into grid g -- no cell, holding part of that sum, can exceed it. psMassFixedScale (the mass
// grid's rule) puts 2B at 2^61..2^62 units, so B sits at 2^60..2^61: a factor 4 below the 2^63 of the signed words,
// for the GPU's float rounding of a value (the kernel's float inverse of a tetrahedron near the degeneracy cut
// amplifies it; a few per cent at worst, as dtfe_gpu_fixed.h found for the standard deposit). The bound is a sum
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

// The bounds B[g] of one pass, summed tetrahedron by tetrahedron (2026-10-14). w = the tet's mass (its shares
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

// The units a pass's caller hands the GPU host (2026-10-14): scale[g] > 0 = grid g as a fixed-point sum in units
// of 1/scale[g] (the host still deposits only the grids its flags select), 0 = the float atomics of before
// (PS_METAL_FLOAT_MOMENTS=1 / PS_GPU_FLOAT_MOMENTS=1, or a bound without a unit); sumW = the exact items' per-tet
// weight totals as fixed-point sums as well.
struct PSFixedUnits
{
    float scale[PS_FX_N] = {};
    bool  sumW = false;
};

// One share of a moment grid in fixed-point units: the float product v*w the deposit formed (the float the old
// float sum added), times the 2^k unit -- exact in double -- rounded to nearest even: the integer rint(v*w *
// scale) the kernels add (metal/ps_deposit.metal addFixed). psMassFixedShare's rule, for any grid.
inline int64_t psFixedShare(double share, float scale)
{
    return int64_t( std::llrint( share * double(scale) ) );
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
    // 2026-10-14: the moment grids as exact 64-bit fixed-point sums (PSFxGrid above). fxMask bit g = grid g is one
    // (its words at buffer 22 + its slot, see metal/ps_deposit.metal), in units of 1/fxScale[g]; PS_FX_SUMW_BIT =
    // the exact items' weight totals too. 0 = every moment grid a float sum, as before (the harness's default).
    uint32_t fxMask;
    float    fxScale[PS_FX_N];
};

// The unit of the mass grid's fixed-point sum (2026-10-08; the kernel's DepositParams.massFixed says why):
// scale = 2^k with the mass handed to the deposit (a partition's tets) at 2^61..2^62 units -- the finest unit
// at which no cell overflows the signed 64-bit sum, even holding all of it plus every share's rounding. The
// unit must be fine: a first version put the mean TET mass at 2^32 units, and the 3 kpc cells of a
// --ps-window zoom (pieces of ~1e-8 of a tet) came out 5e-4 off. Here a cell holding 2^-40 of the partition's
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

// One mass share in fixed-point units (2026-10-12): the integer the kernels add, rint(w * massScale) in metal/
// ps_deposit.metal depositMass (ps_gpu_cuda.cu the same). The scale is 2^k, so the float product there and the
// double product here are both exact, and rint (round to nearest even) gives the same integer: a share adds the
// same units whichever side deposits it. The CPU deposit (ps_interpolation.cc) adds these into the partition's
// fixed-point grid -- the GPU's own words when the GPU deposited the partition -- so the mass grid is ONE exact
// integer sum, independent of the order and the thread count, and is converted to float once.
inline int64_t psMassFixedShare(double share, float scale)
{
    return int64_t( std::llrint( share * double(scale) ) );
}

// The GPU hosts hand the kernels' (low, high) uint32 word pairs over as int64_t by copying the bytes
// (ps_metal_host.cc, ps_gpu_cuda.cu): that IS the integer on a little-endian host, which every Metal (Apple
// Silicon) and CUDA/HIP (x86-64, aarch64) host is.
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "the GPU hosts copy the fixed-point mass words as little-endian int64_t (ps_deposit_params.h)"
#endif

#endif
