/* Host-side interface to the PS-DTFE Metal deposit kernel (metal/ps_deposit.metal,
   'depositFields'). Compiled only in METAL=1 builds (-DPS_METAL); the caller in
   ps_interpolation.cc guards every use and falls back to the CPU deposit on any failure.
   This header is self-contained (no DTFE/CGAL/metal-cpp includes). */

#ifndef PS_METAL_HOST_HEADER
#define PS_METAL_HOST_HEADER

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

// Output grids, all sized to the partition sub-grid (subDims product = subTotal).
struct PSMetalGrids
{
    std::vector<float>    mass;     // subTotal      accumulated MASS (density = mass / cellVolume)
    std::vector<float>    mom;      // subTotal*3    mass-weighted velocity moment  [i*3+j]
    std::vector<float>    m2;       // subTotal*6    mass-weighted 2nd moment, upper triangle [i*6+c]
    std::vector<float>    grad;     // subTotal*9    mass-weighted velocity-gradient moment [i*9+j*3+i']
    std::vector<uint32_t> streams;  // subTotal      interior-sample count (streams = count / nSub^3)
};

// Runs the mass-conserving tetrahedral deposit on the default Metal device. Vertices must already
// be min-image-wrapped relative to vertex 0 (as in the CPU deposit). Thread-safe (serialized).
// Returns false with 'err' set if there is no device / kernel compile fails; outputs untouched.
bool psMetalDepositFields(const std::vector<float>& verts,   // nTet*12: 4 wrapped Eulerian vertices
                          const std::vector<float>& vels,    // nTet*12: 4 vertex velocities
                          const std::vector<float>& masses,  // nTet: tet mass rho_bar * V_lag
                          const double boxLo[3], const double dx[3],
                          const size_t nGrid[3], const size_t subOrigin[3], const size_t subDims[3],
                          int nSub, bool periodic,
                          PSMetalGrids& out, std::string& err);

// Name of the Metal device (empty if unavailable); initializes the device on first call.
std::string psMetalDeviceName();

#endif
