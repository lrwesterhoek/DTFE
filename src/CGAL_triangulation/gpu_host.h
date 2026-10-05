/* Backend-neutral host interface to the GPU deposit kernels, implemented by the Metal
   backend (ps_metal_host.cc / dtfe_metal_host.cc, 'make <target> METAL=1' on macOS/Apple
   Silicon) and by the CUDA/HIP backend (ps_gpu_cuda.cu / dtfe_gpu_cuda.cu, 'CUDA=1' or
   'HIP=1' on Linux; single-source ports of the Metal kernels). The callers never see which.

   A GPU build defines PS_GPU (PS-DTFE) / DTFE_GPU (standard); the callers in
   ps_interpolation.cc and averaged_interpolation_1.cc guard every use on those and fall
   back to the CPU path on any failure. The GPU results match the CPU path to float
   rounding (atomic summation order) -- the parity contract is validated by
   tests/dtfe_metal_check.sh and metal/validate_deposit.cpp (Metal; the CUDA/HIP port is
   compiled and fallback-tested on Linux, not yet validated on hardware).

   This header is self-contained (no DTFE/CGAL/backend includes). */

#ifndef GPU_HOST_HEADER
#define GPU_HOST_HEADER

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

// Name of the backend compiled into this binary ("Metal", "CUDA" or "HIP").
std::string gpuBackendName();

// Name of the GPU device (empty if unavailable); initializes the device on first call.
std::string gpuDeviceName();


// ---------------------------------------------------------------- PS-DTFE deposit
// Output grids, all sized to the partition sub-grid (subDims product = subTotal). Only the
// grids selected by the fVel/fDisp/fGrad flags are allocated (mom/m2/grad stay EMPTY when
// their flag is off -- the m2 and grad grids alone are 24+36 B per sub-grid cell); mass and
// streams are always produced.
struct PSGpuGrids
{
    std::vector<float>    mass;     // subTotal      accumulated MASS (density = mass / cellVolume)
    std::vector<float>    mom;      // subTotal*3    weighted velocity moment  [i*3+j]   (fVel)
    std::vector<float>    m2;       // subTotal*6    weighted 2nd moment, upper triangle [i*6+c] (fDisp)
    std::vector<float>    grad;     // subTotal*9    weighted velocity-gradient moment [i*9+j*3+i'] (fGrad)
    std::vector<uint32_t> streams;  // subTotal      interior-sample count (streams = count / nSub^3)
    std::vector<float>    momw;     // subTotal      moment-weight normalizer sum (fVolW; else EMPTY -- 'mass' is the normalizer)
    std::vector<uint32_t> caustic;  // subTotal      orientation bits, OR-accumulated (fCaustic; else EMPTY)
    std::vector<float>    dispvel;  // subTotal*3    dispersion's own mass-weighted mean sum(m_s v_s) (fVolW+fDisp; else EMPTY)
    std::vector<float>    dispw;    // subTotal      its normalizer sum(m_s)                (fVolW+fDisp; else EMPTY)
    std::vector<float>    streamvol;// subTotal      exact cell-mean multiplicity sum(V_int)/V_cell (fExact; else EMPTY). 'streams' then carries the raw integer tet-touch count.
    std::vector<float>    scal;     // subTotal      scalar moment sum(wm s), one component (fScal; else EMPTY)
    std::vector<float>    sgrad;    // subTotal*3    scalar-gradient moment sum(wm ds/dx_i) [i*3+d] (fSGrad; else EMPTY)
};

// Runs the mass-conserving tetrahedral deposit on the default GPU device. 'verts' holds one
// PS_TET_STRIDE-float record per tet (ps_deposit_params.h psFillTetRecord: vertex 0 as cell
// index + fraction and the edges, in the canonical frame of the exact inside test).
// Thread-safe (serialized on one device queue/stream).
// The kernel decides which samples each tet contains exactly as the CPU deposit does
// (ps_exact_inside.h): in float, with a rigorous error bound. A tet with a sample inside that
// bound is NOT deposited; its index (into the arrays as passed) is returned in 'deferred', and
// the caller must deposit it with the CPU's exact test.
// fVel = velocity moments needed (velocity or dispersion selected), fDisp = second moments
// (dispersion), fGrad = velocity-gradient moments; unselected grids are neither allocated on
// the device (4-byte dummies) nor on the host. 'vels' may be EMPTY when all three are off.
// fLinear (--ps-linear-deposit) weights the interior samples by the linear density built
// from the per-tet vertex densities in 'dens' (nTet*4; may be EMPTY when fLinear is off),
// renormalized per tet so the deposited total still equals the tet mass exactly.
// fVolW (--ps-volume-weighted) switches the mom/m2/grad moment weights to equal Eulerian-
// volume shares (|det|/6 per tet, computed in-kernel) and fills out.momw with their
// normalizer; the mass grid is unaffected. Mutually exclusive with fLinear (rejected at
// option parsing). fCaustic (--ps-caustics) ORs each tet's orientation bits (1 = det>0,
// 2 = det<0) into out.caustic at every deposited sample -- OR is order-invariant, so the
// result matches the CPU deposit's bits (float-vs-double det sign only differs below the
// degeneracy cut, where cells are dropped).
// fExact (--ps-exact-deposit) replaces the nSub^3 sampling by the kernel's float32 r3d port
// (analytic tet-cell clipping + order-2 moments); nSub is then ignored. The CPU path stays
// the DOUBLE-precision reference -- the GPU exact deposit agrees to float rounding, like
// every other GPU/CPU field pair.
// fScal / fSGrad: the scalar field (ONE component, NO_SCALARS=1 builds; the caller declines the GPU
// otherwise): 'scal' holds the four vertex scalars per tet (nTet*4; EMPTY when both are off), and
// the kernel deposits the linear profile's value (and the constant gradient) times the moment
// weight, exactly as the velocity moment, into out.scal / out.sgrad.
// windowMode (--ps-window): the sub-box is a window of the full grid, not a partition's own box --
// every sample / clipped piece of a tet is counted for its normalization as in the full run, only
// those inside the sub-box are deposited.
// Returns false with 'err' set if there is no device / setup fails; outputs untouched in
// that case, zeroed otherwise.
// CONSUMES verts/vels/masses/dens/scal (several GB per partition at scale): the backend frees each one
// right after copying it to the device, so they may be empty when the call returns --
// the CPU fallback deposits from the triangulation, not from these arrays.
bool psGpuDepositFields(std::vector<float>& verts,   // nTet*PS_TET_STRIDE: the per-tet records
                        std::vector<float>& vels,    // nTet*12: 4 vertex velocities (empty if unused)
                        std::vector<float>& masses,  // nTet: tet mass (rho_bar * V_lag, or the --ps-vertex-mass shares)
                        std::vector<float>& dens,    // nTet*4: vertex densities (empty unless fLinear)
                        std::vector<float>& scal,    // nTet*4: vertex scalars (empty unless fScal / fSGrad)
                        const double boxLo[3], const double dx[3],
                        const size_t nGrid[3], const size_t subOrigin[3], const size_t subDims[3],
                        int nSub, bool periodic,
                        bool fVel, bool fDisp, bool fGrad, bool fLinear,
                        bool fVolW, bool fCaustic, bool fExact, bool fScal, bool fSGrad, bool windowMode,
                        PSGpuGrids& out, std::vector<uint32_t>& deferred, std::string& err);


// ---------------------------------------------------------------- standard-DTFE method-1 deposit
// Output grids, sized to the interpolation grid (nGrid product = nCell). Each holds the
// accumulated value*volume contributions; the caller divides by the grid-cell volume
// (identical to the CPU loop). Only the requested fields are allocated; the rest stay empty.
struct DTFEGpuGrids
{
    std::vector<float> density;   // nCell      accumulated density * volume
    std::vector<float> velocity;  // nCell*3    accumulated velocity * volume     [i*3+j]
    std::vector<float> grad;      // nCell*9    accumulated velocity gradient * volume [i*9 + j*3 + i'],
                                  //            layout matching the CPU velocityGradient() Pvector
    std::vector<float> scalar;    // nCell      accumulated scalar * volume (one component; fScal)
    std::vector<float> scalar_grad; // nCell*3  accumulated scalar gradient * volume [i*3 + d] (fSGrad)
};

// Runs the method-1 volume-averaged deposit on the default GPU device. Vertex positions
// must be relative to the region's lower corner; vols/cnts/flats carry the CPU-computed
// tetrahedron volume, sample count (0 = single-grid-cell fast path), and fast-path flat
// grid index. 'sobol' is the shared quasi-random barycentric table (maxNN*3, every count
// is <= maxNN). Thread-safe (serialized).
// Returns false with 'err' set on any failure; outputs untouched in the "no device" case.
// CONSUMES verts/dens/vels/vols/cnts/flats (see psGpuDepositFields); the CPU fallback
// interpolates from the triangulation, not from these arrays.
// fScal / fSGrad: the scalar field (ONE component, NO_SCALARS=1 builds; the caller declines the GPU
// otherwise): 'scal' holds the four vertex scalars per tet (empty when both are off); the kernels
// deposit it like the density (linear over the tet) and its constant gradient like the velocity
// gradient.
// exact (--exact-average): the analytic volume average instead of the samples -- every tetrahedron
// is clipped against each grid cell of its bounding box (the kernels' float32 r3d port) and
// deposits the linear interpolant's integral over each piece; 'cnts' then only marks the
// single-cell tetrahedra (0), which take the centroid fast path as on the CPU. Matches the CPU's
// double-precision r3d to float rounding.
bool dtfeGpuDepositAveraged1(std::vector<float>& verts,      // nTet*12: 4 vertices, region-relative
                             std::vector<float>& dens,       // nTet*4:  vertex densities
                             std::vector<float>& vels,       // nTet*12: 4 vertex velocities (empty if unused)
                             std::vector<float>& scal,       // nTet*4:  vertex scalars (empty if unused)
                             std::vector<float>& vols,       // nTet:    tetrahedron volumes
                             std::vector<uint32_t>& cnts,    // nTet:    sample counts (0 = fast path)
                             std::vector<uint32_t>& flats,   // nTet:    fast-path flat grid indices
                             const std::vector<float>& sobol,// maxNN*3 barycentric table (not consumed)
                             const double dx[3], const size_t nGrid[3],
                             bool fDen, bool fVel, bool fGrad, bool fScal, bool fSGrad, bool exact,
                             DTFEGpuGrids& out, std::string& err);

// ---- shared accumulator: the sub-domains of one DTFE_parallel call deposit into ONE full grid.
// Begin (main thread, before the parallel region) allocates and zeroes the full grids on the
// GPU -- false when memory would not allow it or there is no device, and then every thread
// takes the per-call path above as before. Each thread then extracts its tetrahedra exactly
// as for dtfeGpuDepositAveraged1 (positions and fast-path flat indices relative to ITS
// sub-grid) and hands them over with the sub-grid's origin 'subOff' in the full grid; the
// deposit of samples inside the sub-grid lands at full-grid cells subOff + cell, so every
// cell has one owner and the batches never overlap. Dispatches are serialized on the device
// queue (the chunk controller stays warm across batches); uploads run concurrently. A failed
// batch leaves its owner box zeroed and returns false: the caller deposits that sub-domain on
// the CPU (or via the per-call path) into its own sub-grid, and the final merge ADDS the GPU
// grids into the main grid -- a cell is non-zero in exactly one of the two.
// Result gives read pointers (accumulated value*volume, full-grid layout; a pointer is null for
// a field not requested) that stay valid until End; End frees the GPU grids.
bool dtfeGpuSharedBegin(const size_t fullGrid[3], bool fDen, bool fVel, bool fGrad, bool fScal, bool fSGrad, std::string& err);
bool dtfeGpuSharedActive();
bool dtfeGpuSharedDeposit(std::vector<float>& verts, std::vector<float>& dens, std::vector<float>& vels,
                          std::vector<float>& scal,
                          std::vector<float>& vols, std::vector<uint32_t>& cnts, std::vector<uint32_t>& flats,
                          const std::vector<float>& sobol, const double dx[3], const size_t nGrid[3],
                          const size_t subOff[3], std::string& err);
bool dtfeGpuSharedResult(const float** den, const float** vel, const float** grad, const float** scal, const float** sgrad);
void dtfeGpuSharedEnd();

// --ps-window, the exact deposit (CPU loop and both GPU hosts' item bounds; the kernels carry a twin):
// the contiguous hull, in RAW (unwrapped) cell indices, of the part of a tetrahedron's span [lo, hi)
// that lies in the window [o, o+m) or one of its periodic images (o+m may exceed n: a window wrapping
// the seam). False when the span misses the window. A tetrahedron is shorter than the box, so at
// most two images meet its span; their hull can hold a gap of in-grid cells outside the window,
// which the per-cell window test skips cheaply. A non-periodic span is clamped to the grid first.
inline bool psWindowHull(long lo, long hi, long n, bool periodic, long o, long m, long& outLo, long& outHi)
{
    if (hi <= lo || m <= 0 || n <= 0) return false;
    if (!periodic)
    {
        lo = lo < 0 ? 0 : lo;  hi = hi > n ? n : hi;
        if (hi <= lo) return false;
    }
    bool any = false;
    long hlo = 0, hhi = 0;
    int const kMax = periodic ? 2 : 0;
    for (int k = -kMax; k <= kMax; ++k)
    {
        long const a = lo > o + k*n ? lo : o + k*n;
        long const b = hi < o + m + k*n ? hi : o + m + k*n;
        if (b <= a) continue;
        if (!any) { hlo = a; hhi = b; any = true; }
        else { if (a < hlo) hlo = a; if (b > hhi) hhi = b; }
    }
    if (!any) return false;
    outLo = hlo; outHi = hhi;
    return true;
}

#endif
