/* Backend-neutral host interface to the GPU deposit kernels. Exactly one backend
   implements it per binary, selected at build time:

     make <target> METAL=1   macOS/Apple Silicon  (ps_metal_host.cc / dtfe_metal_host.cc,
                             kernels in metal/*.metal, runtime-compiled)
     make <target> CUDA=1    Linux/NVIDIA         (ps_gpu_cuda.cu / dtfe_gpu_cuda.cu via nvcc)
     make <target> HIP=1     Linux/AMD ROCm       (same .cu sources via hipcc; the sources
                             use a small cuda-vs-hip macro shim and compile for both)

   Any backend defines PS_GPU (PS-DTFE build) / DTFE_GPU (standard build); the callers in
   ps_interpolation.cc and averaged_interpolation_1.cc guard every use on those and fall
   back to the CPU path on any failure. The GPU results match the CPU path to float
   rounding (atomic summation order) -- the parity contract validated for the Metal
   backend (tests/dtfe_metal_check.sh, metal/validate_deposit.cpp) applies to every
   backend, since all kernels implement the same algorithm on float32.

   This header is self-contained (no DTFE/CGAL/backend includes). */

#ifndef GPU_HOST_HEADER
#define GPU_HOST_HEADER

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

// Name of the backend compiled into this binary ("Metal", "CUDA", "HIP").
std::string gpuBackendName();

// Name of the GPU device (empty if unavailable); initializes the device on first call.
std::string gpuDeviceName();


// ---------------------------------------------------------------- PS-DTFE deposit
// Output grids, all sized to the partition sub-grid (subDims product = subTotal).
struct PSGpuGrids
{
    std::vector<float>    mass;     // subTotal      accumulated MASS (density = mass / cellVolume)
    std::vector<float>    mom;      // subTotal*3    mass-weighted velocity moment  [i*3+j]
    std::vector<float>    m2;       // subTotal*6    mass-weighted 2nd moment, upper triangle [i*6+c]
    std::vector<float>    grad;     // subTotal*9    mass-weighted velocity-gradient moment [i*9+j*3+i']
    std::vector<uint32_t> streams;  // subTotal      interior-sample count (streams = count / nSub^3)
};

// Runs the mass-conserving tetrahedral deposit on the default GPU device. Vertices must
// already be min-image-wrapped relative to vertex 0 (as in the CPU deposit). Thread-safe
// (serialized on one device queue/stream).
// Returns false with 'err' set if there is no device / setup fails; outputs untouched in
// that case, zeroed otherwise.
// CONSUMES verts/vels/masses (several GB per partition at scale): Metal frees each one
// right after copying it into unified memory; CUDA/HIP stream them to the device in
// chunks and free them when the dispatch loop ends. Either way they may be empty when the
// call returns -- the CPU fallback deposits from the triangulation, not from these arrays.
bool psGpuDepositFields(std::vector<float>& verts,   // nTet*12: 4 wrapped Eulerian vertices
                        std::vector<float>& vels,    // nTet*12: 4 vertex velocities
                        std::vector<float>& masses,  // nTet: tet mass rho_bar * V_lag
                        const double boxLo[3], const double dx[3],
                        const size_t nGrid[3], const size_t subOrigin[3], const size_t subDims[3],
                        int nSub, bool periodic,
                        PSGpuGrids& out, std::string& err);


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
};

// Runs the method-1 volume-averaged deposit on the default GPU device. Vertex positions
// must be relative to the region's lower corner; vols/cnts/flats carry the CPU-computed
// tetrahedron volume, sample count (0 = single-grid-cell fast path), and fast-path flat
// grid index. 'sobol' is the shared quasi-random barycentric table (maxNN*3, every count
// is <= maxNN). Thread-safe (serialized).
// Returns false with 'err' set on any failure; outputs untouched in the "no device" case.
// CONSUMES verts/dens/vels/vols/cnts/flats (see psGpuDepositFields); the CPU fallback
// interpolates from the triangulation, not from these arrays.
bool dtfeGpuDepositAveraged1(std::vector<float>& verts,      // nTet*12: 4 vertices, region-relative
                             std::vector<float>& dens,       // nTet*4:  vertex densities
                             std::vector<float>& vels,       // nTet*12: 4 vertex velocities (empty if unused)
                             std::vector<float>& vols,       // nTet:    tetrahedron volumes
                             std::vector<uint32_t>& cnts,    // nTet:    sample counts (0 = fast path)
                             std::vector<uint32_t>& flats,   // nTet:    fast-path flat grid indices
                             const std::vector<float>& sobol,// maxNN*3 barycentric table (not consumed)
                             const double dx[3], const size_t nGrid[3],
                             bool fDen, bool fVel, bool fGrad,
                             DTFEGpuGrids& out, std::string& err);

#endif
