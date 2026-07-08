/* CUDA/HIP host + kernel for the PS-DTFE deposit (see gpu_host.h). Compiled only in
   CUDA=1 (nvcc) or HIP=1 (hipcc) builds; a direct port of the validated Metal kernel
   metal/ps_deposit.metal::depositFields -- same algorithm, same degeneracy filters, same
   float32 arithmetic and relaxed atomics, so the CPU-parity contract carries over.

   Differences from the Metal host (unified vs discrete memory):
   - output grids live device-resident for the whole call; INPUT tet arrays are streamed
     to the GPU in chunks, so VRAM needs are outputs + one chunk (checked up front and the
     chunk shrunk to fit; PS_GPU_CHUNK env overrides the starting chunk size);
   - host input arrays stay alive until the dispatch loop ends (they are the streaming
     source) and are freed before the results are copied back;
   - no watchdog choreography: Linux compute contexts are not display-starved like macOS,
     and a failed call is not retried (CUDA/HIP errors poison the context) -- any failure
     falls back to the CPU deposit in the caller. */

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

struct PSDepositParams
{
    float    boxLo[3];
    float    dx[3];
    int      nGrid[3];
    int      nSub;
    int      periodic;
    int      subOrigin[3];
    int      subDims[3];
    unsigned nTet;
};

__device__ inline float det3(const float A[3][3])
{
    return A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])
         - A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])
         + A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
}

// Inverse of 3x3; false if |det| < 1e-6 (matches the CPU matrixInverse() sentinel so a
// collapsed simplex is dropped rather than bounding-box filled).
__device__ inline bool inverse3(const float A[3][3], float inv[3][3])
{
    float d = det3(A);
    if (fabsf(d) < 1.0e-6f) return false;
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
    return true;
}

__device__ inline int wrapIdx(int g, int n) { return ((g % n) + n) % n; }

// Sub-sample position for local sample index s (base-nSub digits) in cell 'raw' along axis dd.
__device__ inline float samplePos(unsigned s, int dd, int nSub, const int raw[3], const PSDepositParams& P)
{
    unsigned rem = s;
    for (int k=0; k<dd; ++k) rem /= unsigned(nSub);
    float fr = (nSub==1) ? 0.5f : (float(int(rem % unsigned(nSub))) + 0.5f)/float(nSub);
    return P.boxLo[dd] + (float(raw[dd])+fr)*P.dx[dd];
}

// One sample deposit; 64-bit offsets because flat*9 overflows 32-bit uint for grids >= ~782^3.
__device__ inline void depositSample(float* mass, float* mom, float* m2, float* grad,
                                     unsigned* streams, unsigned flat,
                                     const float rel[3], float w,
                                     const float u0[3], const float vG[3][3])
{
    unsigned long long base = flat;
    atomicAdd(&mass[flat], w);
    float vv[3];
    for (int j=0; j<3; ++j)
    {
        vv[j] = u0[j];
        for (int i=0; i<3; ++i) vv[j] += vG[i][j]*rel[i];
        atomicAdd(&mom[base*3ull + (unsigned long long)j], vv[j]*w);
    }
    unsigned long long c = 0;
    for (int i=0; i<3; ++i)
        for (int j=i; j<3; ++j)
            atomicAdd(&m2[base*6ull + c++], w*vv[i]*vv[j]);
    for (int j=0; j<3; ++j)
        for (int i=0; i<3; ++i)
            atomicAdd(&grad[base*9ull + (unsigned long long)(j*3+i)], vG[i][j]*w);
    atomicAdd(&streams[flat], 1u);
}

__global__ void psDepositFieldsKernel(const float* verts, const float* vels, const float* masses,
                                      float* massGrid, float* momGrid, float* m2Grid, float* gradGrid,
                                      unsigned* strGrid, PSDepositParams P)
{
    unsigned tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid >= P.nTet) return;

    float v0[3] = { verts[tid*12+0], verts[tid*12+1],  verts[tid*12+2]  };
    float v1[3] = { verts[tid*12+3], verts[tid*12+4],  verts[tid*12+5]  };
    float v2[3] = { verts[tid*12+6], verts[tid*12+7],  verts[tid*12+8]  };
    float v3[3] = { verts[tid*12+9], verts[tid*12+10], verts[tid*12+11] };
    float m = masses[tid];
    if (m <= 0.0f) return;

    float Ax[3][3];
    for (int i=0; i<3; ++i) { Ax[0][i]=v1[i]-v0[i]; Ax[1][i]=v2[i]-v0[i]; Ax[2][i]=v3[i]-v0[i]; }

    // relative-determinant degeneracy filter, identical to the CPU/Metal paths
    float d = det3(Ax);
    float avgEdge2 = 0.0f;
    for (int v=0; v<3; ++v) { float l2=0.0f; for (int i=0;i<3;++i) l2+=Ax[v][i]*Ax[v][i]; avgEdge2+=l2; }
    avgEdge2 /= 3.0f;
    if (fabsf(d) < 1.0e-6f * avgEdge2 * sqrtf(avgEdge2)) return;

    float posInv[3][3];
    if (!inverse3(Ax, posInv)) return;

    // constant per-tet velocity gradient: vG = posInv * (u_{k+1}-u_0)
    float u0[3] = { vels[tid*12+0], vels[tid*12+1], vels[tid*12+2] };
    float vG[3][3];
    {
        float dV[3][3];
        for (int e=0; e<3; ++e)
            for (int j=0; j<3; ++j)
                dV[e][j] = vels[tid*12+(e+1)*3+j] - u0[j];
        for (int i=0; i<3; ++i)
            for (int j=0; j<3; ++j)
            {
                float s = 0.0f;
                for (int k=0; k<3; ++k) s += posInv[i][k]*dV[k][j];
                vG[i][j] = s;
            }
    }

    // Eulerian bbox -> raw cell range (+/-1 margin when sub-sampling; clamp only non-periodic)
    int iMin[3], iMax[3];
    for (int dd=0; dd<3; ++dd)
    {
        float lo = fminf(fminf(v0[dd],v1[dd]), fminf(v2[dd],v3[dd]));
        float hi = fmaxf(fmaxf(v0[dd],v1[dd]), fmaxf(v2[dd],v3[dd]));
        iMin[dd] = int(floorf((lo-P.boxLo[dd])/P.dx[dd]));
        iMax[dd] = int(floorf((hi-P.boxLo[dd])/P.dx[dd])) + 1;
        if (P.nSub>1) { iMin[dd]-=1; iMax[dd]+=1; }
        if (!P.periodic) { if (iMin[dd]<0) iMin[dd]=0; if (iMax[dd]>P.nGrid[dd]) iMax[dd]=P.nGrid[dd]; }
    }
    int nSub = P.nSub;
    unsigned nSamp = unsigned(nSub)*unsigned(nSub)*unsigned(nSub);

    // Two identical passes over (cell, sample): pass 0 counts N, pass 1 deposits m/N. The cell's
    // wrap + sub-grid mapping is resolved BEFORE the sample loop (like the CPU inSub guard), so N
    // counts exactly the samples that pass 1 deposits -- no mass is ever lost to invalid cells.
    unsigned N = 0;
    float share = 0.0f;
    for (int pass=0; pass<2; ++pass)
    {
        if (pass==1)
        {
            if (N==0u) break;      // centroid fallback below
            share = m / float(N);
        }
        for (int gi=iMin[0]; gi<iMax[0]; ++gi)
        for (int gj=iMin[1]; gj<iMax[1]; ++gj)
        for (int gk=iMin[2]; gk<iMax[2]; ++gk)
        {
            int raw[3] = {gi,gj,gk};
            int wg0 = P.periodic ? wrapIdx(gi,P.nGrid[0]) : gi;
            int wg1 = P.periodic ? wrapIdx(gj,P.nGrid[1]) : gj;
            int wg2 = P.periodic ? wrapIdx(gk,P.nGrid[2]) : gk;
            if (wg0<0||wg0>=P.nGrid[0]||wg1<0||wg1>=P.nGrid[1]||wg2<0||wg2>=P.nGrid[2]) continue;
            int l0=wg0-P.subOrigin[0], l1=wg1-P.subOrigin[1], l2=wg2-P.subOrigin[2];
            if (l0<0||l0>=P.subDims[0]||l1<0||l1>=P.subDims[1]||l2<0||l2>=P.subDims[2]) continue;
            unsigned flat = (unsigned(l0)*unsigned(P.subDims[1])+unsigned(l1))*unsigned(P.subDims[2])+unsigned(l2);
            for (unsigned s=0; s<nSamp; ++s)
            {
                float rel[3] = { samplePos(s,0,nSub,raw,P)-v0[0],
                                 samplePos(s,1,nSub,raw,P)-v0[1],
                                 samplePos(s,2,nSub,raw,P)-v0[2] };
                float sum=0.0f; bool inside=true;
                for (int vv=0; vv<3; ++vv)
                {
                    float bc=0.0f;
                    for (int i=0; i<3; ++i) bc += posInv[i][vv]*rel[i];
                    if (bc<-1.0e-6f) { inside=false; break; }
                    sum += bc;
                }
                if (inside && sum<=1.0f+1.0e-6f)
                {
                    if (pass==0) N++;
                    else         depositSample(massGrid,momGrid,m2Grid,gradGrid,strGrid, flat, rel, share, u0, vG);
                }
            }
        }
    }

    if (N==0u)   // centroid fallback (mass + fields at the centroid position)
    {
        float cen[3], crel[3];
        int loc[3];
        for (int dd=0; dd<3; ++dd) cen[dd] = 0.25f*(v0[dd]+v1[dd]+v2[dd]+v3[dd]);
        for (int dd=0; dd<3; ++dd)
        {
            int cg = int(floorf((cen[dd]-P.boxLo[dd])/P.dx[dd]));
            int wg = P.periodic ? wrapIdx(cg,P.nGrid[dd]) : cg;
            if (wg<0||wg>=P.nGrid[dd]) return;
            loc[dd] = wg-P.subOrigin[dd];
            if (loc[dd]<0||loc[dd]>=P.subDims[dd]) return;   // centroid outside this partition's box: drop
        }
        unsigned flat = (unsigned(loc[0])*unsigned(P.subDims[1])+unsigned(loc[1]))*unsigned(P.subDims[2])+unsigned(loc[2]);
        for (int dd=0; dd<3; ++dd) crel[dd] = cen[dd]-v0[dd];
        depositSample(massGrid,momGrid,m2Grid,gradGrid,strGrid, flat, crel, m, u0, vG);
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


bool psGpuDepositFields(std::vector<float>& verts,
                        std::vector<float>& vels,
                        std::vector<float>& masses,
                        const double boxLo[3], const double dx[3],
                        const size_t nGrid[3], const size_t subOrigin[3], const size_t subDims[3],
                        int nSub, bool periodic,
                        PSGpuGrids& out, std::string& err)
{
    std::lock_guard<std::mutex> lock(ctxMutex());   // serialize dispatches (single device)
    Ctx& c = ctx();
    if (!c.ready) { err = c.err; return false; }

    size_t const nCell = subDims[0] * subDims[1] * subDims[2];
    out.mass.assign(nCell, 0.f);
    out.mom.assign(nCell * 3, 0.f);
    out.m2.assign(nCell * 6, 0.f);
    out.grad.assign(nCell * 9, 0.f);
    out.streams.assign(nCell, 0u);
    if (masses.empty()) return true;    // nothing to deposit (empty partition)

    PSDepositParams P;
    std::memset(&P, 0, sizeof(P));
    for (int d = 0; d < 3; ++d)
    {
        P.boxLo[d]     = float(boxLo[d]);
        P.dx[d]        = float(dx[d]);
        P.nGrid[d]     = int(nGrid[d]);
        P.subOrigin[d] = int(subOrigin[d]);
        P.subDims[d]   = int(subDims[d]);
    }
    P.nSub     = nSub < 1 ? 1 : nSub;
    P.periodic = periodic ? 1 : 0;

    size_t const nTetTotal = masses.size();

    // Outputs stay device-resident for the whole call; inputs stream in chunks so VRAM needs
    // are outputs + one chunk. Shrink the chunk until it fits (with slack for the runtime).
    size_t chunk = 2000000;
    if (const char* env = getenv("PS_GPU_CHUNK"))
    { long v = atol(env); if (v > 0) chunk = size_t(v); }
    if (chunk > nTetTotal) chunk = nTetTotal;
    size_t const outBytes = nCell * (4 + 12 + 24 + 36 + 4);
    size_t const slack = size_t(64) << 20;
    size_t freeB = 0, totalB = 0;
    if (gpuMemGetInfo(&freeB, &totalB) == gpuSuccess)
    {
        while (chunk > 50000 && outBytes + chunk*25*sizeof(float) + slack > freeB)
            chunk /= 2;
        if (outBytes + chunk*25*sizeof(float) + slack > freeB)
        {
            err = "insufficient GPU memory for the output grids (need > " +
                  std::to_string((outBytes+slack)>>20) + " MB free)";
            return false;
        }
    }

    float *dV=nullptr, *dU=nullptr, *dM=nullptr, *dMass=nullptr, *dMom=nullptr, *dM2=nullptr, *dGrad=nullptr;
    unsigned *dStr=nullptr;
    auto freeAll = [&]()
    {
        for (void* p : { (void*)dV,(void*)dU,(void*)dM,(void*)dMass,(void*)dMom,(void*)dM2,(void*)dGrad,(void*)dStr })
            if (p) gpuFree(p);
    };

    bool ok = gpuCheck(gpuMalloc((void**)&dV,   chunk*12*sizeof(float)), "gpuMalloc verts",  err)
          &&  gpuCheck(gpuMalloc((void**)&dU,   chunk*12*sizeof(float)), "gpuMalloc vels",   err)
          &&  gpuCheck(gpuMalloc((void**)&dM,   chunk*sizeof(float)),    "gpuMalloc masses", err)
          &&  gpuCheck(gpuMalloc((void**)&dMass, nCell*4),  "gpuMalloc mass grid", err)
          &&  gpuCheck(gpuMalloc((void**)&dMom,  nCell*12), "gpuMalloc mom grid",  err)
          &&  gpuCheck(gpuMalloc((void**)&dM2,   nCell*24), "gpuMalloc m2 grid",   err)
          &&  gpuCheck(gpuMalloc((void**)&dGrad, nCell*36), "gpuMalloc grad grid", err)
          &&  gpuCheck(gpuMalloc((void**)&dStr,  nCell*4),  "gpuMalloc streams",   err)
          &&  gpuCheck(gpuMemset(dMass, 0, nCell*4),  "gpuMemset mass", err)
          &&  gpuCheck(gpuMemset(dMom,  0, nCell*12), "gpuMemset mom",  err)
          &&  gpuCheck(gpuMemset(dM2,   0, nCell*24), "gpuMemset m2",   err)
          &&  gpuCheck(gpuMemset(dGrad, 0, nCell*36), "gpuMemset grad", err)
          &&  gpuCheck(gpuMemset(dStr,  0, nCell*4),  "gpuMemset streams", err);

    int const tpb = 256;
    for (size_t start = 0; start < nTetTotal && ok; start += chunk)
    {
        unsigned const n = unsigned(std::min(chunk, nTetTotal - start));
        P.nTet = n;
        ok = gpuCheck(gpuMemcpy(dV, verts.data()  + start*12, size_t(n)*12*sizeof(float), gpuMemcpyHostToDevice), "copy verts",  err)
         &&  gpuCheck(gpuMemcpy(dU, vels.data()   + start*12, size_t(n)*12*sizeof(float), gpuMemcpyHostToDevice), "copy vels",   err)
         &&  gpuCheck(gpuMemcpy(dM, masses.data() + start,    size_t(n)*sizeof(float),    gpuMemcpyHostToDevice), "copy masses", err);
        if (!ok) break;
        psDepositFieldsKernel<<< (n + tpb - 1)/tpb, tpb >>>(dV, dU, dM, dMass, dMom, dM2, dGrad, dStr, P);
        ok = gpuCheck(gpuGetLastError(), "kernel launch", err)
         &&  gpuCheck(gpuDeviceSynchronize(), "kernel execution", err);
    }

    // inputs are consumed by this call (the streaming loop was their last consumer)
    std::vector<float>().swap(verts);
    std::vector<float>().swap(vels);
    std::vector<float>().swap(masses);

    if (ok)
        ok = gpuCheck(gpuMemcpy(out.mass.data(),    dMass, nCell*4,  gpuMemcpyDeviceToHost), "copy mass back",    err)
         &&  gpuCheck(gpuMemcpy(out.mom.data(),     dMom,  nCell*12, gpuMemcpyDeviceToHost), "copy mom back",     err)
         &&  gpuCheck(gpuMemcpy(out.m2.data(),      dM2,   nCell*24, gpuMemcpyDeviceToHost), "copy m2 back",      err)
         &&  gpuCheck(gpuMemcpy(out.grad.data(),    dGrad, nCell*36, gpuMemcpyDeviceToHost), "copy grad back",    err)
         &&  gpuCheck(gpuMemcpy(out.streams.data(), dStr,  nCell*4,  gpuMemcpyDeviceToHost), "copy streams back", err);

    freeAll();
    return ok;
}
