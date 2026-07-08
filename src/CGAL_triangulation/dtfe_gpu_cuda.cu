/* CUDA/HIP host + kernel for the standard-DTFE method-1 deposit (see gpu_host.h).
   Compiled only in CUDA=1 (nvcc) or HIP=1 (hipcc) builds; a direct port of the validated
   Metal kernel metal/dtfe_deposit.metal::depositAveraged1 -- same algorithm, same
   zero-inverse degeneracy semantics (a collapsed tetrahedron still deposits its base
   value, it is NOT dropped), same float32 arithmetic, so the CPU-parity contract carries
   over. See ps_gpu_cuda.cu for the discrete-memory streaming rationale (outputs
   device-resident, inputs streamed per chunk; DTFE_GPU_CHUNK env overrides). */

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "gpu_host.h"
#include "gpu_cuda_compat.h"

namespace {

struct DTFEDepositParams
{
    float    dx[3];
    int      nGrid[3];
    unsigned nTet;
    int      fDen;      // deposit density
    int      fVel;      // deposit velocity
    int      fGrad;     // deposit velocity gradient
};

__device__ inline float det3(const float A[3][3])
{
    return A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])
         - A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])
         + A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
}

// Inverse of 3x3, mirroring the CPU matrixInverse(): |det| < 1e-6 -> ZERO matrix. The
// CPU then interpolates a constant field from the base vertex; it does NOT drop the
// tetrahedron, so neither do we (unlike the PS deposit, which conserves mass and drops).
__device__ inline void inverse3zero(const float A[3][3], float inv[3][3])
{
    float d = det3(A);
    if (fabsf(d) < 1.0e-6f)
    {
        for (int i=0;i<3;++i) for (int j=0;j<3;++j) inv[i][j]=0.0f;
        return;
    }
    float invd = 1.0f / d;
    inv[0][0] =  (A[1][1]*A[2][2]-A[1][2]*A[2][1])*invd;
    inv[1][0] = -(A[1][0]*A[2][2]-A[1][2]*A[2][0])*invd;
    inv[2][0] =  (A[1][0]*A[2][1]-A[1][1]*A[2][0])*invd;
    inv[0][1] = -(A[0][1]*A[2][2]-A[0][2]*A[2][1])*invd;
    inv[1][1] =  (A[0][0]*A[2][2]-A[0][2]*A[2][0])*invd;
    inv[2][1] = -(A[0][0]*A[2][1]-A[0][1]*A[2][0])*invd;
    inv[0][2] =  (A[0][1]*A[1][2]-A[0][2]*A[1][1])*invd;
    inv[1][2] = -(A[0][0]*A[1][2]-A[0][2]*A[1][0])*invd;
    inv[2][2] =  (A[0][0]*A[1][1]-A[0][1]*A[1][0])*invd;
}

// One deposit of (den, vel, gradFlat) * w into grid cell 'flat'. Offsets in 64-bit:
// flat*9 overflows 32-bit uint for grids >= ~782^3.
__device__ inline void depositCell(float* denGrid, float* velGrid, float* gradGrid,
                                   const DTFEDepositParams& P,
                                   unsigned long long flat, float den,
                                   const float vel[3], const float gradFlat[9], float w)
{
    if (P.fDen)
        atomicAdd(&denGrid[flat], den*w);
    if (P.fVel)
        for (int j=0; j<3; ++j)
            atomicAdd(&velGrid[flat*3ull + (unsigned long long)j], vel[j]*w);
    if (P.fGrad)
        for (int q=0; q<9; ++q)
            atomicAdd(&gradGrid[flat*9ull + (unsigned long long)q], gradFlat[q]*w);
}

__global__ void dtfeDepositAveraged1Kernel(const float* verts, const float* dens, const float* vels,
                                           const float* vols, const unsigned* cnts, const unsigned* flats,
                                           const float* sobol,
                                           float* denGrid, float* velGrid, float* gradGrid,
                                           DTFEDepositParams P)
{
    unsigned tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid >= P.nTet) return;

    float base[3] = { verts[tid*12+0], verts[tid*12+1], verts[tid*12+2] };
    float Ax[3][3];   // vertex-difference matrix, rows = vertex k+1 - vertex 0
    for (int v=0; v<3; ++v)
        for (int d=0; d<3; ++d)
            Ax[v][d] = verts[tid*12+(v+1)*3+d] - base[d];

    float posInv[3][3];
    inverse3zero(Ax, posInv);

    bool const needVel = (P.fVel != 0) || (P.fGrad != 0);
    float u0[3] = {0.0f,0.0f,0.0f};
    float vG[3][3];                       // vG[i][j] = d(v_j)/d(x_i), like the CPU velocityGrad
    float gradFlat[9] = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
    if (needVel)
    {
        for (int j=0; j<3; ++j) u0[j] = vels[tid*12+j];
        for (int i=0; i<3; ++i)
            for (int j=0; j<3; ++j)
            {
                float s = 0.0f;
                for (int k=0; k<3; ++k) s += posInv[i][k]*(vels[tid*12+(k+1)*3+j]-u0[j]);
                vG[i][j] = s;
            }
        for (int j=0; j<3; ++j)           // CPU velocityGradient() layout: out[j*3+i] = vG[i][j]
            for (int i=0; i<3; ++i)
                gradFlat[j*3+i] = vG[i][j];
    }

    float const vol = vols[tid];
    unsigned const n = cnts[tid];

    if (n == 0u)   // whole tetrahedron inside one (in-range) grid cell: centroid value * volume
    {
        float denC = 0.0f;
        for (int v=0; v<4; ++v) denC += dens[tid*4+v];
        denC *= 0.25f;
        float velC[3] = {0.0f,0.0f,0.0f};
        if (P.fVel)
        {
            for (int v=0; v<4; ++v) for (int j=0; j<3; ++j) velC[j] += vels[tid*12+v*3+j];
            for (int j=0; j<3; ++j) velC[j] *= 0.25f;
        }
        depositCell(denGrid, velGrid, gradGrid, P, (unsigned long long)flats[tid], denC, velC, gradFlat, vol);
        return;
    }

    float den0 = dens[tid*4+0];
    float densGrad[3];
    for (int i=0; i<3; ++i)
    {
        float s = 0.0f;
        for (int j=0; j<3; ++j) s += posInv[i][j]*(dens[tid*4+j+1]-den0);
        densGrad[i] = s;
    }

    float const factor = vol / float(n);
    for (unsigned s=0; s<n; ++s)
    {
        float q0=sobol[s*3+0], q1=sobol[s*3+1], q2=sobol[s*3+2];
        float p[3];   // sample point relative to the base vertex (CPU quasiRandomPointsInCell)
        for (int d=0; d<3; ++d) p[d] = q0*Ax[0][d] + q1*Ax[1][d] + q2*Ax[2][d];
        int pos[3]; bool in=true;
        for (int d=0; d<3; ++d)
        {
            pos[d] = int(floorf((base[d]+p[d])/P.dx[d]));
            if (pos[d]<0 || pos[d]>=P.nGrid[d]) { in=false; break; }
        }
        if (!in) continue;   // sample outside the region of interest
        unsigned long long flat = ((unsigned long long)pos[0]*(unsigned long long)P.nGrid[1]
                                   + (unsigned long long)pos[1])*(unsigned long long)P.nGrid[2]
                                   + (unsigned long long)pos[2];
        float denS = den0;
        for (int i=0; i<3; ++i) denS += p[i]*densGrad[i];
        float velS[3] = {0.0f,0.0f,0.0f};
        if (P.fVel)
            for (int j=0; j<3; ++j)
            {
                velS[j] = u0[j];
                for (int i=0; i<3; ++i) velS[j] += vG[i][j]*p[i];
            }
        depositCell(denGrid, velGrid, gradGrid, P, flat, denS, velS, gradFlat, factor);
    }
}


struct Ctx
{
    bool        ready = false;
    std::string err;
    std::string name;
};

std::mutex& ctxMutex() { static std::mutex m; return m; }

Ctx& ctx()
{
    static Ctx c;
    static bool tried = false;
    if (tried) return c;
    tried = true;
    int n = 0;
    if (gpuGetDeviceCount(&n) != gpuSuccess || n < 1) { c.err = "no " GPU_BACKEND_STRING " device"; return c; }
    gpuDeviceProp prop;
    std::memset(&prop, 0, sizeof(prop));
    if (gpuGetDeviceProperties(&prop, 0) != gpuSuccess) { c.err = "device query failed"; return c; }
    c.name = prop.name;
    c.ready = true;
    return c;
}

} // namespace


std::string gpuBackendName() { return GPU_BACKEND_STRING; }

std::string gpuDeviceName()
{
    std::lock_guard<std::mutex> lock(ctxMutex());
    Ctx& c = ctx();
    return c.ready ? c.name : std::string();
}


bool dtfeGpuDepositAveraged1(std::vector<float>& verts,
                             std::vector<float>& dens,
                             std::vector<float>& vels,
                             std::vector<float>& vols,
                             std::vector<uint32_t>& cnts,
                             std::vector<uint32_t>& flats,
                             const std::vector<float>& sobol,
                             const double dx[3], const size_t nGrid[3],
                             bool fDen, bool fVel, bool fGrad,
                             DTFEGpuGrids& out, std::string& err)
{
    std::lock_guard<std::mutex> lock(ctxMutex());   // serialize dispatches (single device)
    Ctx& c = ctx();
    if (!c.ready) { err = c.err; return false; }

    size_t const nCell = nGrid[0] * nGrid[1] * nGrid[2];
    out.density.assign(fDen ? nCell : 0, 0.f);
    out.velocity.assign(fVel ? nCell * 3 : 0, 0.f);
    out.grad.assign(fGrad ? nCell * 9 : 0, 0.f);
    if (cnts.empty()) return true;    // nothing to deposit (no cells in the region)

    bool const needVel = fVel || fGrad;
    DTFEDepositParams P;
    std::memset(&P, 0, sizeof(P));
    for (int d = 0; d < 3; ++d)
    {
        P.dx[d]    = float(dx[d]);
        P.nGrid[d] = int(nGrid[d]);
    }
    P.fDen  = fDen ? 1 : 0;
    P.fVel  = fVel ? 1 : 0;
    P.fGrad = fGrad ? 1 : 0;

    size_t const nTetTotal = cnts.size();

    // outputs device-resident; inputs streamed per chunk (see ps_gpu_cuda.cu)
    size_t chunk = 2000000;
    if (const char* env = getenv("DTFE_GPU_CHUNK"))
    { long v = atol(env); if (v > 0) chunk = size_t(v); }
    if (chunk > nTetTotal) chunk = nTetTotal;
    size_t const denBytes  = fDen  ? nCell * 4  : 4;
    size_t const velBytes  = fVel  ? nCell * 12 : 4;
    size_t const gradBytes = fGrad ? nCell * 36 : 4;
    size_t const outBytes  = denBytes + velBytes + gradBytes;
    size_t const perTetBytes = (12 + 4 + (needVel ? 12 : 0) + 1 + 1 + 1) * 4;
    size_t const slack = size_t(64) << 20;
    size_t freeB = 0, totalB = 0;
    if (gpuMemGetInfo(&freeB, &totalB) == gpuSuccess)
    {
        while (chunk > 50000 && outBytes + chunk*perTetBytes + sobol.size()*4 + slack > freeB)
            chunk /= 2;
        if (outBytes + chunk*perTetBytes + sobol.size()*4 + slack > freeB)
        {
            err = "insufficient GPU memory for the output grids (need > " +
                  std::to_string((outBytes+slack)>>20) + " MB free)";
            return false;
        }
    }

    float *dV=nullptr, *dD=nullptr, *dU=nullptr, *dVol=nullptr, *dS=nullptr;
    unsigned *dCnt=nullptr, *dFlt=nullptr;
    float *dDen=nullptr, *dVel=nullptr, *dGrad=nullptr;
    auto freeAll = [&]()
    {
        for (void* p : { (void*)dV,(void*)dD,(void*)dU,(void*)dVol,(void*)dS,
                         (void*)dCnt,(void*)dFlt,(void*)dDen,(void*)dVel,(void*)dGrad })
            if (p) gpuFree(p);
    };

    bool ok = gpuCheck(gpuMalloc((void**)&dV,   chunk*12*sizeof(float)), "gpuMalloc verts", err)
          &&  gpuCheck(gpuMalloc((void**)&dD,   chunk*4*sizeof(float)),  "gpuMalloc dens",  err)
          &&  gpuCheck(gpuMalloc((void**)&dU,   needVel ? chunk*12*sizeof(float) : sizeof(float)), "gpuMalloc vels", err)
          &&  gpuCheck(gpuMalloc((void**)&dVol, chunk*sizeof(float)),    "gpuMalloc vols",  err)
          &&  gpuCheck(gpuMalloc((void**)&dCnt, chunk*sizeof(unsigned)), "gpuMalloc cnts",  err)
          &&  gpuCheck(gpuMalloc((void**)&dFlt, chunk*sizeof(unsigned)), "gpuMalloc flats", err)
          &&  gpuCheck(gpuMalloc((void**)&dS,   sobol.size()*sizeof(float)), "gpuMalloc sobol", err)
          &&  gpuCheck(gpuMalloc((void**)&dDen,  denBytes),  "gpuMalloc density grid",  err)
          &&  gpuCheck(gpuMalloc((void**)&dVel,  velBytes),  "gpuMalloc velocity grid", err)
          &&  gpuCheck(gpuMalloc((void**)&dGrad, gradBytes), "gpuMalloc gradient grid", err)
          &&  gpuCheck(gpuMemcpy(dS, sobol.data(), sobol.size()*sizeof(float), gpuMemcpyHostToDevice), "copy sobol", err)
          &&  gpuCheck(gpuMemset(dDen,  0, denBytes),  "gpuMemset density",  err)
          &&  gpuCheck(gpuMemset(dVel,  0, velBytes),  "gpuMemset velocity", err)
          &&  gpuCheck(gpuMemset(dGrad, 0, gradBytes), "gpuMemset gradient", err);

    int const tpb = 256;
    for (size_t start = 0; start < nTetTotal && ok; start += chunk)
    {
        unsigned const n = unsigned(std::min(chunk, nTetTotal - start));
        P.nTet = n;
        ok = gpuCheck(gpuMemcpy(dV,   verts.data() + start*12, size_t(n)*12*sizeof(float), gpuMemcpyHostToDevice), "copy verts", err)
         &&  gpuCheck(gpuMemcpy(dD,   dens.data()  + start*4,  size_t(n)*4*sizeof(float),  gpuMemcpyHostToDevice), "copy dens",  err)
         &&  (not needVel ||
              gpuCheck(gpuMemcpy(dU,  vels.data()  + start*12, size_t(n)*12*sizeof(float), gpuMemcpyHostToDevice), "copy vels", err))
         &&  gpuCheck(gpuMemcpy(dVol, vols.data()  + start,    size_t(n)*sizeof(float),    gpuMemcpyHostToDevice), "copy vols",  err)
         &&  gpuCheck(gpuMemcpy(dCnt, cnts.data()  + start,    size_t(n)*sizeof(unsigned), gpuMemcpyHostToDevice), "copy cnts",  err)
         &&  gpuCheck(gpuMemcpy(dFlt, flats.data() + start,    size_t(n)*sizeof(unsigned), gpuMemcpyHostToDevice), "copy flats", err);
        if (!ok) break;
        dtfeDepositAveraged1Kernel<<< (n + tpb - 1)/tpb, tpb >>>(dV, dD, dU, dVol, dCnt, dFlt, dS, dDen, dVel, dGrad, P);
        ok = gpuCheck(gpuGetLastError(), "kernel launch", err)
         &&  gpuCheck(gpuDeviceSynchronize(), "kernel execution", err);
    }

    // inputs are consumed by this call (the streaming loop was their last consumer)
    std::vector<float>().swap(verts);
    std::vector<float>().swap(dens);
    std::vector<float>().swap(vels);
    std::vector<float>().swap(vols);
    std::vector<uint32_t>().swap(cnts);
    std::vector<uint32_t>().swap(flats);

    if (ok && fDen)
        ok = gpuCheck(gpuMemcpy(out.density.data(),  dDen,  nCell*4,  gpuMemcpyDeviceToHost), "copy density back",  err);
    if (ok && fVel)
        ok = gpuCheck(gpuMemcpy(out.velocity.data(), dVel,  nCell*12, gpuMemcpyDeviceToHost), "copy velocity back", err);
    if (ok && fGrad)
        ok = gpuCheck(gpuMemcpy(out.grad.data(),     dGrad, nCell*36, gpuMemcpyDeviceToHost), "copy gradient back", err);

    freeAll();
    return ok;
}
