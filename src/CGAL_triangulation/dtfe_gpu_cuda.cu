/* CUDA/HIP backend of the standard-DTFE method-1 deposit (implements gpu_host.h). Compiled only
   in CUDA=1 (nvcc) or HIP=1 (hipcc, '-x hip') builds, which define DTFE_GPU for the callers.

   The kernels are a port of metal/dtfe_deposit.metal: depositAveraged1 (one thread per
   tetrahedron, one atomic per field per sample; DTFE_GPU_ITEMS=0) and the work-item kernels
   (one thread per (tetrahedron, run of samples) with the per-thread cell cache, the warp
   reduction of the flushes and the per-block cell table for the evictions). Metal's function
   constants FC_DEN/FC_VEL/FC_GRAD become template parameters (the host picks the instantiation
   per field set), its SIMD-group operations the warp shuffles of gpu_cuda_compat.h, its
   threadgroup memory dynamic shared memory. One difference of discipline: a CUDA warp
   collective needs EVERY lane, so a thread without an item (the chunk's tail) runs through the
   item body with nothing pending instead of returning early -- it then takes no part in any sum.

   The host mirrors dtfe_metal_host.cc: the same Batch / runBatch machinery, the chunked dispatch
   controller, the owner-box re-zero before a retry, and the shared full-grid accumulator (its
   result is copied to host vectors at Result, since the contract hands out pointers). Where
   the Metal host reads unified memory in place, this one copies explicitly. Also mirrored:
   its retry policy (chunk checkpoints, the cause of a failed kernel, the process-wide pace;
   gpu_cuda_retry.cuh, ps_gpu_cuda.cu's) and its exact sums (64-bit fixed-point grids, the Metal
   kernel's header; the item kernels' FX template parameter is Metal's FC_FIXED).

   Compiled and its CPU fallback exercised on Linux; NOT yet validated on NVIDIA/AMD hardware
   (metal/README.md, "CUDA / HIP"). */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <vector>
#include <unistd.h>
extern std::atomic<int> dtfeGpuRetries;            // averaged_interpolation_1.cc (run_progress.h)
extern std::atomic<int> dtfeGpuRetriesByCause[4];  // ... by cause: innocent, timeout, oom, other

#include "gpu_host.h"
#include "../message.h"         // ANSI colour helpers only (the retry warnings)
#include "gpu_cuda_compat.h"
#include "dtfe_gpu_fixed.h"     // batchFixedScales + psMassFixedMass: the exact sums' units (shared with the Metal host)

#define ITEM_BLOCK_SHIFT 5      // log2 of the items per entry of the block table (kernel + host)

namespace {

#include "gpu_exact_clip.cuh"   // V3, det3/inverse3, the r3d clip helpers (shared with ps_gpu_cuda.cu)

// must match the Metal DepositParams (dtfe_deposit.metal) field for field
struct DepositParams
{
    float    dx[3];
    int32_t  nGrid[3];      // the batch's sub-grid
    uint32_t nTet;          // threads in this dispatch (tetrahedra or work items)
    int32_t  fDen;
    int32_t  fVel;
    int32_t  fGrad;
    uint32_t itemSamples;   // item kernel: samples per work item
    uint32_t nTetTotal;     // item kernel: tetrahedra in the batch
    int32_t  subOff[3];     // the sub-grid's origin in the accumulated grid
    uint32_t itemBase;      // item kernel: first item of this dispatch
    int32_t  fullGrid[3];   // the accumulated grid
    uint32_t debugNoAtomic; // DTFE_GPU_DEBUG_NOATOMIC: compute everything, skip the atomics (item kernel)
    float    invDx[3];
    uint32_t tableEntries;  // item kernel: shared-memory cell-table entries (power of two; 0 = none)
    int32_t  fScal;         // the scalar field (ONE component): runtime flags in both kernels here (Metal
    int32_t  fSGrad;        // specializes the item pipelines on them; 96 instantiations would be too many)
    uint32_t fixedSum;      // the grids as exact 64-bit fixed-point sums (batchFixedScales)
    float    fxScale[5];    // their units 2^k: density, velocity, velocity gradient, scalar, scalar gradient
    // the one field the Metal struct has not: the non-finite guard's counters (one word per grid; fxq),
    // which Metal binds as a buffer of its own (21). Here they travel in the by-value parameters, which every
    // device function that converts a share already takes.
    unsigned* bad;
};

// ---- the exact sums (metal/dtfe_deposit.metal's header says why and how). Under fixedSum a grid
// buffer holds two unsigned words per component (low, high: the int64 on the little-endian hosts) instead of a
// float: every share is the integer rint(v * scale) (the scale is 2^k, so the product is exact and rintf rounds
// to nearest even, as Metal's rint), added word by word with the carry done by hand -- ps_gpu_cuda.cu's
// depositMass -- so a cell's total does not depend on the order of the atomics.
// The non-finite guard (metal/dtfe_deposit.metal fxq has the same test, ps_deposit_params.h
// psFixedShareOk says why): a product not below 2^63 in magnitude -- NaN, inf, overflow -- has no int64; it gives 0
// (every add skips it) and is counted in grid g's word of P.bad. The float's bits are tested, as in Metal.
__device__ __forceinline__ ull fxq(float v, const DepositParams& P, int g)
{
    float const p = v * P.fxScale[g];
    if ((__float_as_uint(p) & 0x7fffffffu) < 0x5f000000u) return ull((long long)(rintf(p)));
    atomicAdd(&P.bad[g], 1u);
    return 0ull;
}
__device__ __forceinline__ void addWordsLH(unsigned* g, ull at, unsigned lo, unsigned hi)
{
    if (lo == 0u && hi == 0u) return;
    unsigned const old = atomicAdd(&g[at * 2ull], lo);
    unsigned const up = hi + ((old + lo < old) ? 1u : 0u);
    if (up != 0u)
        atomicAdd(&g[at * 2ull + 1ull], up);
}
__device__ __forceinline__ void addWords(unsigned* g, ull at, ull q) { addWordsLH(g, at, unsigned(q), unsigned(q >> 32)); }
// one share of grid component 'at' (= flat * comps + component): the words when fx, else the float atomic
__device__ __forceinline__ void gridAdd(float* g, bool fx, const DepositParams& P, int slot, ull at, float v)
{
    if (fx) addWords(reinterpret_cast<unsigned*>(g), at, fxq(v, P, slot));
    else atomicAdd(&g[at], v);
}

// ================================================================================================
// kernels
// ================================================================================================

// full-grid flat index of a sub-grid cell (the host guarantees fullGrid cells < 2^32)
__device__ __forceinline__ unsigned fullFlat(const DepositParams& P, const int pos[3]) {
    return (unsigned(pos[0] + P.subOff[0]) * unsigned(P.fullGrid[1]) + unsigned(pos[1] + P.subOff[1])) * unsigned(P.fullGrid[2])
         + unsigned(pos[2] + P.subOff[2]);
}

// Inverse of 3x3, mirroring the CPU matrixInverse(): |det| <= 1e-6 x the product of the row
// lengths -> ZERO matrix (the tetrahedron is NOT dropped: a constant field from the base vertex).
__device__ __forceinline__ void inverse3zero(const float A[3][3], float inv[3][3]) {
    float d = det3(A);
    float rn = 1.0f;
    for (int i=0; i<3; ++i) rn *= sqrtf(A[i][0]*A[i][0] + A[i][1]*A[i][1] + A[i][2]*A[i][2]);
    if (!(fabsf(d) > 1.0e-6f * rn)) {
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

// One deposit of (den, vel, gradFlat) * w into grid cell 'flat' (64-bit offsets: flat*9 would
// overflow 32 bits above ~782^3).
__device__ __forceinline__ void depositCell(float* denGrid, float* velGrid, float* gradGrid,
                                            float* scalGrid, float* sgGrid,
                                            const DepositParams& P, ull flat, float den,
                                            const float vel[3], const float gradFlat[9],
                                            float sc, const float sG[3], float w)
{
    bool const fx = P.fixedSum != 0u;
    if (P.fDen)
        gridAdd(denGrid, fx, P, 0, flat, den*w);
    if (P.fVel)
        for (int j=0;j<3;++j)
            gridAdd(velGrid, fx, P, 1, flat*3ull+ull(j), vel[j]*w);
    if (P.fGrad)
        for (int q=0;q<9;++q)
            gridAdd(gradGrid, fx, P, 2, flat*9ull+ull(q), gradFlat[q]*w);
    if (P.fScal)
        gridAdd(scalGrid, fx, P, 3, flat, sc*w);
    if (P.fSGrad)
        for (int i=0;i<3;++i)
            gridAdd(sgGrid, fx, P, 4, flat*3ull+ull(i), sG[i]*w);
}

__global__ void __launch_bounds__(256)
depositAveraged1(const float* __restrict__ verts, const float* __restrict__ dens, const float* __restrict__ vels,
                 const float* __restrict__ vols, const unsigned* __restrict__ cnts, const unsigned* __restrict__ flats,
                 const float4* __restrict__ sobol,
                 float* denGrid, float* velGrid, float* gradGrid, const DepositParams P,
                 const float* __restrict__ scal, float* scalGrid, float* sgGrid)
{
    unsigned const tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= P.nTet) return;

    float base[3] = { verts[tid*12+0], verts[tid*12+1], verts[tid*12+2] };
    float Ax[3][3];
    for (int v=0; v<3; ++v)
        for (int d=0; d<3; ++d)
            Ax[v][d] = verts[tid*12+(v+1)*3+d] - base[d];

    float posInv[3][3];
    inverse3zero(Ax, posInv);

    bool const needVel = (P.fVel != 0) || (P.fGrad != 0);
    float u0[3] = {0.0f,0.0f,0.0f};
    float vG[3][3];
    float gradFlat[9] = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
    if (needVel) {
        for (int j=0;j<3;++j) u0[j] = vels[tid*12+j];
        for (int i=0;i<3;++i)
            for (int j=0;j<3;++j) {
                float s=0.0f;
                for (int k=0;k<3;++k) s += posInv[i][k]*(vels[tid*12+(k+1)*3+j]-u0[j]);
                vG[i][j]=s;
            }
        for (int j=0;j<3;++j)
            for (int i=0;i<3;++i)
                gradFlat[j*3+i] = vG[i][j];
    }
    // the scalar: linear over the tetrahedron, its gradient constant (the CPU scalarGrad())
    bool const needScal = (P.fScal != 0) || (P.fSGrad != 0);
    float s0 = 0.0f, sG[3] = {0.0f,0.0f,0.0f};
    if (needScal) {
        s0 = scal[tid*4+0];
        for (int i=0;i<3;++i) {
            float s=0.0f;
            for (int j=0;j<3;++j) s += posInv[i][j]*(scal[tid*4+j+1]-s0);
            sG[i]=s;
        }
    }

    float const vol = vols[tid];
    unsigned const n = cnts[tid];

    if (n == 0u) {
        float denC = 0.0f;
        for (int v=0; v<4; ++v) denC += dens[tid*4+v];
        denC *= 0.25f;
        float velC[3] = {0.0f,0.0f,0.0f};
        if (P.fVel) {
            for (int v=0;v<4;++v) for (int j=0;j<3;++j) velC[j] += vels[tid*12+v*3+j];
            for (int j=0;j<3;++j) velC[j] *= 0.25f;
        }
        float scC = 0.0f;
        if (P.fScal) { for (int v=0;v<4;++v) scC += scal[tid*4+v]; scC *= 0.25f; }
        depositCell(denGrid, velGrid, gradGrid, scalGrid, sgGrid, P, ull(flats[tid]), denC, velC, gradFlat, scC, sG, vol);
        return;
    }

    float den0 = dens[tid*4+0];
    float densGrad[3];
    for (int i=0;i<3;++i) {
        float s=0.0f;
        for (int j=0;j<3;++j) s += posInv[i][j]*(dens[tid*4+j+1]-den0);
        densGrad[i]=s;
    }

    float const factor = vol / float(n);
    for (unsigned s=0; s<n; ++s) {
        float4 const q = sobol[s];
        float const q0 = q.x, q1 = q.y, q2 = q.z;
        float p[3];
        for (int d=0;d<3;++d) p[d] = q0*Ax[0][d] + q1*Ax[1][d] + q2*Ax[2][d];
        int pos[3]; bool in=true;
        for (int d=0;d<3;++d) {
            pos[d] = int(floorf((base[d]+p[d])*P.invDx[d]));
            if (pos[d]<0 || pos[d]>=P.nGrid[d]) { in=false; break; }
        }
        if (!in) continue;
        ull flat = ull(fullFlat(P, pos));
        float denS = den0;
        for (int i=0;i<3;++i) denS += p[i]*densGrad[i];
        float velS[3] = {0.0f,0.0f,0.0f};
        if (P.fVel)
            for (int j=0;j<3;++j) {
                velS[j]=u0[j];
                for (int i=0;i<3;++i) velS[j] += vG[i][j]*p[i];
            }
        float scS = s0;
        if (P.fScal) for (int i=0;i<3;++i) scS += p[i]*sG[i];
        depositCell(denGrid, velGrid, gradGrid, scalGrid, sgGrid, P, flat, denS, velS, gradFlat, scS, sG, factor);
    }
}


// ------------------------------------------------------------------------------------------------
// Work-item kernel (see metal/dtfe_deposit.metal for the design): a K-slot per-thread cell cache,
// evictions into the per-block cell table, the end-of-item flush as a warp reduction by cell.
// ------------------------------------------------------------------------------------------------
template <int K>
struct CellCache {
    unsigned flat[K];
    float den[K];
    float vx[K], vy[K], vz[K];
    float sc[K];        // the scalar's sum (P.fScal)
    float cnt[K];
};

template <int K, bool FD, bool FV, bool FG, bool FX>
__device__ __forceinline__ void flushSlot(float* denGrid, float* velGrid, float* gradGrid,
                                          float* scalGrid, float* sgGrid,
                                          const DepositParams& P, const CellCache<K>& C, int k,
                                          float factor, const float vG[3][3], const float sG[3])
{
    unsigned flat = 0u; float den = 0.0f, vx = 0.0f, vy = 0.0f, vz = 0.0f, sc = 0.0f, cnt = 0.0f;
    #pragma unroll
    for (int j = 0; j < K; ++j) {
        bool const m = (j == k);
        flat = m ? C.flat[j] : flat;
        den  = m ? C.den[j]  : den;
        vx   = m ? C.vx[j]   : vx;
        vy   = m ? C.vy[j]   : vy;
        vz   = m ? C.vz[j]   : vz;
        sc   = m ? C.sc[j]   : sc;
        cnt  = m ? C.cnt[j]  : cnt;
    }
    // (the same products as the table path's tableAdd: an eviction adds the same integers whichever takes it)
    ull const f = ull(flat);
    if (FD)
        gridAdd(denGrid, FX, P, 0, f, den * factor);
    if (FV) {
        gridAdd(velGrid, FX, P, 1, f*3ull+0ull, vx * factor);
        gridAdd(velGrid, FX, P, 1, f*3ull+1ull, vy * factor);
        gridAdd(velGrid, FX, P, 1, f*3ull+2ull, vz * factor);
    }
    if (P.fScal)
        gridAdd(scalGrid, FX, P, 3, f, sc * factor);
    if (FG && P.debugNoAtomic < 3u) {
        float const w = cnt * factor;
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i)
                gridAdd(gradGrid, FX, P, 2, f*9ull+ull(j*3+i), vG[i][j] * w);
    }
    if (P.fSGrad && P.debugNoAtomic < 3u) {
        float const w = cnt * factor;
        for (int i = 0; i < 3; ++i)
            gridAdd(sgGrid, FX, P, 4, f*3ull+ull(i), sG[i] * w);
    }
}

struct Item { unsigned tet, sub, perTet; };

// (tetrahedron, index of this item among the tetrahedron's items, items of the tetrahedron):
// itemStart[t] = first item of tetrahedron t, blockTet[iid >> SHIFT] = the tetrahedron holding
// that block's first item; a binary search over at most one block's worth of tetrahedra.
__device__ __forceinline__ Item itemOf(const unsigned* itemStart, const unsigned* blockTet, unsigned nTet, unsigned iid) {
    unsigned const blk = iid >> ITEM_BLOCK_SHIFT;
    unsigned lo = blockTet[blk];
    unsigned hi = min(blockTet[blk + 1u], nTet - 1u);
    while (hi > lo) {
        unsigned const mid = (lo + hi + 1u) >> 1;
        if (itemStart[mid] <= iid) lo = mid; else hi = mid - 1u;
    }
    unsigned const first = itemStart[lo];
    Item it; it.tet = lo; it.sub = iid - first; it.perTet = itemStart[lo + 1u] - first;
    return it;
}

#define TABLE_EMPTY 0xffffffffu
#define TABLE_PROBES 8u

// values per entry: den, vx, vy, vz, [scalar], [9 gradient], [3 scalar gradient] (== the Metal layout); floats,
// or under FX two unsigned words each (tValsX: value v of an entry at words [2v], [2v+1])
template <bool FG> __device__ __forceinline__ unsigned tableOffScal(const DepositParams&)    { return 4u; }
template <bool FG> __device__ __forceinline__ unsigned tableOffGrad(const DepositParams& P)  { return 4u + (P.fScal ? 1u : 0u); }
template <bool FG> __device__ __forceinline__ unsigned tableOffSGrad(const DepositParams& P) { return tableOffGrad<FG>(P) + (FG ? 9u : 0u); }
template <bool FG> __device__ __forceinline__ unsigned tableStride(const DepositParams& P)   { return tableOffSGrad<FG>(P) + (P.fSGrad ? 3u : 0u); }

// the per-block cell table (open addressing, linear probing, keys claimed by compare-exchange)
template <bool FD, bool FV, bool FG, bool FX>
__device__ __forceinline__ bool tableAdd(unsigned* tKeys, float* tVals, unsigned E,
                                         unsigned key, float den, float vx, float vy, float vz, float sc, float w,
                                         const float vG[3][3], const float sG[3], const DepositParams& P)
{
    unsigned h = (key * 2654435761u) >> 7;
    h &= (E - 1u);
    for (unsigned probe = 0u; probe < TABLE_PROBES; ++probe) {
        unsigned k = tKeys[h];
        if (k == TABLE_EMPTY) {
            unsigned const prev = atomicCAS(&tKeys[h], TABLE_EMPTY, key);
            k = (prev == TABLE_EMPTY) ? key : prev;
        }
        if (k == key) {
            unsigned const base = h * tableStride<FG>(P);
            if (FX) {   // the integers flushSlot would add (the same float products, converted here)
                unsigned* tValsX = reinterpret_cast<unsigned*>(tVals);
                if (FD) addWords(tValsX, base + 0u, fxq(den, P, 0));
                if (FV) {
                    addWords(tValsX, base + 1u, fxq(vx, P, 1));
                    addWords(tValsX, base + 2u, fxq(vy, P, 1));
                    addWords(tValsX, base + 3u, fxq(vz, P, 1));
                }
                if (P.fScal) addWords(tValsX, base + tableOffScal<FG>(P), fxq(sc, P, 3));
                if (FG)
                    for (int j = 0; j < 3; ++j)
                        for (int i = 0; i < 3; ++i)
                            addWords(tValsX, base + tableOffGrad<FG>(P) + unsigned(j*3+i), fxq(vG[i][j] * w, P, 2));
                if (P.fSGrad)
                    for (int i = 0; i < 3; ++i)
                        addWords(tValsX, base + tableOffSGrad<FG>(P) + unsigned(i), fxq(sG[i] * w, P, 4));
                return true;
            }
            if (FD) atomicAdd(&tVals[base + 0u], den);
            if (FV) {
                atomicAdd(&tVals[base + 1u], vx);
                atomicAdd(&tVals[base + 2u], vy);
                atomicAdd(&tVals[base + 3u], vz);
            }
            if (P.fScal) atomicAdd(&tVals[base + tableOffScal<FG>(P)], sc);
            if (FG)
                for (int j = 0; j < 3; ++j)
                    for (int i = 0; i < 3; ++i)
                        atomicAdd(&tVals[base + tableOffGrad<FG>(P) + unsigned(j*3+i)], vG[i][j] * w);
            if (P.fSGrad)
                for (int i = 0; i < 3; ++i)
                    atomicAdd(&tVals[base + tableOffSGrad<FG>(P) + unsigned(i)], sG[i] * w);
            return true;
        }
        h = (h + 1u) & (E - 1u);
    }
    return false;
}

// The item body. 'valid' is false for a thread beyond the chunk (it still takes part in the warp
// reduction, with nothing pending).
template <int K, bool FD, bool FV, bool FG, bool FX>
__device__ __forceinline__ void depositItem(
    const float* __restrict__ verts, const float* __restrict__ dens, const float* __restrict__ vels,
    const float* __restrict__ vols,  const unsigned* __restrict__ cnts, const unsigned* __restrict__ flats,
    const float4* __restrict__ sobol,
    float* denGrid, float* velGrid, float* gradGrid,
    const float* __restrict__ scal, float* scalGrid, float* sgGrid,
    const DepositParams& P, Item const it, bool const valid, unsigned const lane,
    unsigned* tKeys, float* tVals)
{
    unsigned const tid    = valid ? it.tet : 0xffffffffu;
    unsigned const sub    = it.sub;
    unsigned const perTet = it.perTet;

    float vG[3][3] = {{0.0f,0.0f,0.0f},{0.0f,0.0f,0.0f},{0.0f,0.0f,0.0f}};   // vG[i][j] = d(v_j)/d(x_i)
    float sG[3] = {0.0f,0.0f,0.0f};                                          // the scalar's constant gradient
    bool const fScal = P.fScal != 0, fSGrad = P.fSGrad != 0;
    CellCache<K> C;
    #pragma unroll
    for (int k = 0; k < K; ++k) { C.flat[k] = 0u; C.den[k] = 0.0f; C.vx[k] = 0.0f; C.vy[k] = 0.0f; C.vz[k] = 0.0f; C.sc[k] = 0.0f; C.cnt[k] = 0.0f; }
    int used = 0;
    int victim = 0;
    float factor = 0.0f;

    if (valid) {
    float base[3] = { verts[tid*12+0], verts[tid*12+1], verts[tid*12+2] };
    float Ax[3][3];
    for (int v=0; v<3; ++v)
        for (int d=0; d<3; ++d)
            Ax[v][d] = verts[tid*12+(v+1)*3+d] - base[d];

    float posInv[3][3];
    inverse3zero(Ax, posInv);

    bool const needVel = FV || FG;
    float u0[3] = {0.0f,0.0f,0.0f};
    if (needVel) {
        for (int j=0;j<3;++j) u0[j] = vels[tid*12+j];
        for (int i=0;i<3;++i)
            for (int j=0;j<3;++j) {
                float s=0.0f;
                for (int k=0;k<3;++k) s += posInv[i][k]*(vels[tid*12+(k+1)*3+j]-u0[j]);
                vG[i][j]=s;
            }
    }

    float s0 = 0.0f;
    if (fScal || fSGrad) {
        s0 = scal[tid*4+0];
        for (int i=0;i<3;++i) {
            float s=0.0f;
            for (int j=0;j<3;++j) s += posInv[i][j]*(scal[tid*4+j+1]-s0);
            sG[i]=s;
        }
    }

    float const vol = vols[tid];
    unsigned const n = cnts[tid];

    if (n == 0u) {   // single-cell tetrahedron: its only item deposits the centroid value * volume
        float denC = 0.0f;
        for (int v=0; v<4; ++v) denC += dens[tid*4+v];
        denC *= 0.25f;
        float velC[3] = {0.0f,0.0f,0.0f};
        if (FV) {
            for (int v=0;v<4;++v) for (int j=0;j<3;++j) velC[j] += vels[tid*12+v*3+j];
            for (int j=0;j<3;++j) velC[j] *= 0.25f;
        }
        float scC = 0.0f;
        if (fScal) { for (int v=0;v<4;++v) scC += scal[tid*4+v]; scC *= 0.25f; }
        C.flat[0] = flats[tid]; C.den[0] = denC; C.vx[0] = velC[0]; C.vy[0] = velC[1]; C.vz[0] = velC[2]; C.sc[0] = scC; C.cnt[0] = 1.0f;
        used = 1;
        factor = vol;
    } else {
    float den0 = dens[tid*4+0];
    float densGrad[3];
    for (int i=0;i<3;++i) {
        float s=0.0f;
        for (int j=0;j<3;++j) s += posInv[i][j]*(dens[tid*4+j+1]-den0);
        densGrad[i]=s;
    }

    factor = vol / float(n);

    // samples dealt round-robin to the tetrahedron's items (consecutive lanes read consecutive
    // table entries)
    for (unsigned s = sub; s < n; s += perTet) {
        float4 const q = sobol[s];
        float const q0 = q.x, q1 = q.y, q2 = q.z;
        float p[3];
        for (int d=0;d<3;++d) p[d] = q0*Ax[0][d] + q1*Ax[1][d] + q2*Ax[2][d];
        int pos[3]; bool in=true;
        for (int d=0;d<3;++d) {
            pos[d] = int(floorf((base[d]+p[d])*P.invDx[d]));
            if (pos[d]<0 || pos[d]>=P.nGrid[d]) { in=false; break; }
        }
        if (!in) continue;
        unsigned const flat = fullFlat(P, pos);
        float denS = den0;
        for (int i=0;i<3;++i) denS += p[i]*densGrad[i];
        float velS[3] = {0.0f,0.0f,0.0f};
        if (FV)
            for (int j=0;j<3;++j) {
                velS[j]=u0[j];
                for (int i=0;i<3;++i) velS[j] += vG[i][j]*p[i];
            }
        float scS = s0;
        if (fScal) for (int i=0;i<3;++i) scS += p[i]*sG[i];

        int hit = -1;
        #pragma unroll
        for (int k = 0; k < K; ++k)
            hit = (k < used && C.flat[k] == flat) ? k : hit;
        if (hit < 0) {
            if (used < K) { hit = used; ++used; }
            else {
                hit = victim;
                victim = (victim + 1 == K) ? 0 : victim + 1;
                unsigned vKey = 0u; float vDen = 0.0f, vVx = 0.0f, vVy = 0.0f, vVz = 0.0f, vSc = 0.0f, vCnt = 0.0f;
                #pragma unroll
                for (int j = 0; j < K; ++j) {
                    bool const mm = (j == hit);
                    vKey = mm ? C.flat[j] : vKey; vDen = mm ? C.den[j] : vDen; vVx = mm ? C.vx[j] : vVx;
                    vVy = mm ? C.vy[j] : vVy; vVz = mm ? C.vz[j] : vVz; vSc = mm ? C.sc[j] : vSc; vCnt = mm ? C.cnt[j] : vCnt;
                }
                if (P.tableEntries == 0u ||
                    !tableAdd<FD, FV, FG, FX>(tKeys, tVals, P.tableEntries, vKey, vDen * factor, vVx * factor, vVy * factor,
                                              vVz * factor, vSc * factor, vCnt * factor, vG, sG, P))
                    flushSlot<K, FD, FV, FG, FX>(denGrid, velGrid, gradGrid, scalGrid, sgGrid, P, C, hit, factor, vG, sG);
            }
            #pragma unroll
            for (int k = 0; k < K; ++k) {
                bool const m = (k == hit);
                C.flat[k] = m ? flat : C.flat[k];
                C.den[k]  = m ? 0.0f : C.den[k];
                C.vx[k]   = m ? 0.0f : C.vx[k];
                C.vy[k]   = m ? 0.0f : C.vy[k];
                C.vz[k]   = m ? 0.0f : C.vz[k];
                C.sc[k]   = m ? 0.0f : C.sc[k];
                C.cnt[k]  = m ? 0.0f : C.cnt[k];
            }
        }
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            bool const m = (k == hit);
            C.den[k] = m ? C.den[k] + denS    : C.den[k];
            if (FV) {
                C.vx[k]  = m ? C.vx[k]  + velS[0] : C.vx[k];
                C.vy[k]  = m ? C.vy[k]  + velS[1] : C.vy[k];
                C.vz[k]  = m ? C.vz[k]  + velS[2] : C.vz[k];
            }
            if (fScal)
                C.sc[k]  = m ? C.sc[k]  + scS     : C.sc[k];
            C.cnt[k] = m ? C.cnt[k] + 1.0f    : C.cnt[k];
        }
    }
    }   // sampled path
    }   // valid

    // ---- flush with a warp reduction by cell (uniform control flow: every lane of the warp
    // evaluates the warp-any conditions; lanes with nothing pending contribute zeros). Under FX the leader
    // converts its warp sums to integers (a warp is always the same items: the host aligns the chunks).
    #pragma unroll
    for (int k = 0; k < K; ++k) {
        bool  pend = k < used;
        unsigned key = pend ? C.flat[k] : 0xffffffffu;
        float const dW = C.den[k] * factor;
        float const xW = C.vx[k] * factor, yW = C.vy[k] * factor, zW = C.vz[k] * factor;
        float const cW = C.sc[k] * factor;
        float const gW = C.cnt[k] * factor;
        while (gpuWarpAny(pend)) {
            unsigned const leaderKey = gpuWarpMinU(pend ? key : 0xffffffffu);
            bool const m = pend && key == leaderKey;
            unsigned const leader = gpuWarpMinU(m ? lane : unsigned(GPU_WARP));
            float const sd = FD ? gpuWarpSumF(m ? dW : 0.0f) : 0.0f;
            float sx = 0.0f, sy = 0.0f, sz = 0.0f;
            if (FV) {
                sx = gpuWarpSumF(m ? xW : 0.0f);
                sy = gpuWarpSumF(m ? yW : 0.0f);
                sz = gpuWarpSumF(m ? zW : 0.0f);
            }
            float const ss = fScal ? gpuWarpSumF(m ? cW : 0.0f) : 0.0f;   // uniform: P is the same for every lane
            if (lane == leader && !P.debugNoAtomic) {
                ull const f = ull(leaderKey);
                if (FD)
                    gridAdd(denGrid, FX, P, 0, f, sd);
                if (FV) {
                    gridAdd(velGrid, FX, P, 1, f*3ull+0ull, sx);
                    gridAdd(velGrid, FX, P, 1, f*3ull+1ull, sy);
                    gridAdd(velGrid, FX, P, 1, f*3ull+2ull, sz);
                }
                if (fScal)
                    gridAdd(scalGrid, FX, P, 3, f, ss);
            }
            if ((FG || fSGrad) && P.debugNoAtomic < 2u) {
                // the gradient is a per-tetrahedron constant times the cell weight: reduced per
                // (cell, tetrahedron)
                bool mg = m;
                while (gpuWarpAny(mg)) {
                    unsigned const leaderTet = gpuWarpMinU(mg ? tid : 0xffffffffu);
                    bool const mt = mg && tid == leaderTet;
                    unsigned const leaderG = gpuWarpMinU(mt ? lane : unsigned(GPU_WARP));
                    float const sw = gpuWarpSumF(mt ? gW : 0.0f);
                    if (lane == leaderG && !P.debugNoAtomic) {
                        ull const f = ull(leaderKey);
                        if (FG)
                            for (int j = 0; j < 3; ++j)
                                for (int i = 0; i < 3; ++i)
                                    gridAdd(gradGrid, FX, P, 2, f*9ull+ull(j*3+i), vG[i][j] * sw);
                        if (fSGrad)
                            for (int i = 0; i < 3; ++i)
                                gridAdd(sgGrid, FX, P, 4, f*3ull+ull(i), sG[i] * sw);
                    }
                    mg = mg && !mt;
                }
            }
            pend = pend && !m;
        }
    }
}

// Flushes the block's cell table into the grids: one device atomic per field per used entry (two words each,
// integer to integer, under FX).
template <bool FD, bool FV, bool FG, bool FX>
__device__ __forceinline__ void tableFlush(const unsigned* tKeys, const float* tVals, unsigned E,
                                           unsigned ltid, unsigned tgSize,
                                           float* denGrid, float* velGrid, float* gradGrid,
                                           float* scalGrid, float* sgGrid, const DepositParams& P)
{
    const unsigned* tValsX = reinterpret_cast<const unsigned*>(tVals);
    // value t of the integer table into device component 'at' of grid g
    auto flushWords = [&](float* g, ull at, unsigned t) {
        addWordsLH(reinterpret_cast<unsigned*>(g), at, tValsX[2u * t], tValsX[2u * t + 1u]);
    };
    for (unsigned e = ltid; e < E; e += tgSize) {
        unsigned const key = tKeys[e];
        if (key == TABLE_EMPTY) continue;
        ull const f = ull(key);
        unsigned const base = e * tableStride<FG>(P);
        if (P.debugNoAtomic) continue;
        if (FX) {
            if (FD)
                flushWords(denGrid, f, base + 0u);
            if (FV)
                for (unsigned c = 0u; c < 3u; ++c)
                    flushWords(velGrid, f*3ull+ull(c), base + 1u + c);
            if (P.fScal)
                flushWords(scalGrid, f, base + tableOffScal<FG>(P));
            if (FG)
                for (unsigned q = 0u; q < 9u; ++q)
                    flushWords(gradGrid, f*9ull+ull(q), base + tableOffGrad<FG>(P) + q);
            if (P.fSGrad)
                for (unsigned i = 0u; i < 3u; ++i)
                    flushWords(sgGrid, f*3ull+ull(i), base + tableOffSGrad<FG>(P) + i);
            continue;
        }
        if (FD)
            atomicAdd(&denGrid[f], tVals[base + 0u]);
        if (FV)
            for (unsigned c = 0u; c < 3u; ++c)
                atomicAdd(&velGrid[f*3ull+ull(c)], tVals[base + 1u + c]);
        if (P.fScal)
            atomicAdd(&scalGrid[f], tVals[base + tableOffScal<FG>(P)]);
        if (FG)
            for (unsigned q = 0u; q < 9u; ++q)
                atomicAdd(&gradGrid[f*9ull+ull(q)], tVals[base + tableOffGrad<FG>(P) + q]);
        if (P.fSGrad)
            for (unsigned i = 0u; i < 3u; ++i)
                atomicAdd(&sgGrid[f*3ull+ull(i)], tVals[base + tableOffSGrad<FG>(P) + i]);
    }
}

#ifdef GPU_EMU
alignas(16) unsigned char dtfeSmem[65536];     // the emulator runs one block at a time: one shared buffer (aligned like the GPU's)
#endif

template <int K, bool FD, bool FV, bool FG, bool FX>
__global__ void __launch_bounds__(256)
depositItemsKernel(const float* __restrict__ verts, const float* __restrict__ dens, const float* __restrict__ vels,
                   const float* __restrict__ vols, const unsigned* __restrict__ cnts, const unsigned* __restrict__ flats,
                   const float4* __restrict__ sobol,
                   float* denGrid, float* velGrid, float* gradGrid, const DepositParams P,
                   const unsigned* __restrict__ itemStart, const unsigned* __restrict__ blockTet,
                   const float* __restrict__ scal, float* scalGrid, float* sgGrid)
{
    extern __shared__ unsigned char dtfeSmem[];           // cell table: keys, then the values (host-sized)
    unsigned* tKeys = reinterpret_cast<unsigned*>(dtfeSmem);
    float*    tVals = reinterpret_cast<float*>(dtfeSmem + size_t(P.tableEntries) * 4u);
    unsigned const E = P.tableEntries;
    unsigned const stride = tableStride<FG>(P);
    unsigned const ltid = threadIdx.x, tgSize = blockDim.x;
    for (unsigned e = ltid; e < E; e += tgSize) {
        tKeys[e] = TABLE_EMPTY;
        if (FX)   // two words per value (the shared memory is sized for it)
            for (unsigned c = 0u; c < 2u * stride; ++c) reinterpret_cast<unsigned*>(tVals)[e * 2u * stride + c] = 0u;
        else
            for (unsigned c = 0u; c < stride; ++c) tVals[e * stride + c] = 0.0f;
    }
    __syncthreads();
    unsigned const iid = blockIdx.x * blockDim.x + threadIdx.x;
    bool const valid = iid < P.nTet;
    Item it; it.tet = 0xffffffffu; it.sub = 0u; it.perTet = 1u;
    if (valid) it = itemOf(itemStart, blockTet, P.nTetTotal, P.itemBase + iid);
    depositItem<K, FD, FV, FG, FX>(verts, dens, vels, vols, cnts, flats, sobol, denGrid, velGrid, gradGrid,
                                   scal, scalGrid, sgGrid, P, it, valid, gpuLane(), tKeys, tVals);
    __syncthreads();
    tableFlush<FD, FV, FG, FX>(tKeys, tVals, E, ltid, tgSize, denGrid, velGrid, gradGrid, scalGrid, sgGrid, P);
}


// ---------------------------------------------------------------------------------------
// depositExactAverage: --exact-average on the GPU (the port of metal/dtfe_deposit.metal's kernel of
// that name): one thread per item = (tetrahedron, block of P.itemSamples cells of its window);
// each piece inside a cell deposits the linear interpolant's centroid value times the piece
// volume (order-1 r3d moments), the gradients times the piece volume. A single-cell tetrahedron
// (cnts == 0) has one item: the centroid fast path. No per-tet normalization, one dispatch.
// ---------------------------------------------------------------------------------------
__global__ void __launch_bounds__(256)
depositExactAverage(const float* __restrict__ verts, const float* __restrict__ dens, const float* __restrict__ vels,
                    const float* __restrict__ vols, const unsigned* __restrict__ cnts, const unsigned* __restrict__ flats,
                    float* denGrid, float* velGrid, float* gradGrid, const DepositParams P,
                    const unsigned* __restrict__ itemStart, const unsigned* __restrict__ blockTet,
                    const float* __restrict__ scal, float* scalGrid, float* sgGrid)
{
    unsigned const iid = blockIdx.x * blockDim.x + threadIdx.x;
    if (iid >= P.nTet) return;
    Item const it = itemOf(itemStart, blockTet, P.nTetTotal, P.itemBase + iid);
    unsigned const tid = it.tet, sub = it.sub;

    float base[3] = { verts[tid*12+0], verts[tid*12+1], verts[tid*12+2] };
    float Ax[3][3];
    for (int v=0; v<3; ++v)
        for (int d=0; d<3; ++d)
            Ax[v][d] = verts[tid*12+(v+1)*3+d] - base[d];
    float posInv[3][3];
    inverse3zero(Ax, posInv);

    bool const needVel = (P.fVel != 0) || (P.fGrad != 0);
    float u0[3] = {0.0f,0.0f,0.0f};
    float vG[3][3] = {{0.0f,0.0f,0.0f},{0.0f,0.0f,0.0f},{0.0f,0.0f,0.0f}};
    float gradFlat[9] = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
    if (needVel) {
        for (int j=0;j<3;++j) u0[j] = vels[tid*12+j];
        for (int i=0;i<3;++i)
            for (int j=0;j<3;++j) {
                float s=0.0f;
                for (int k=0;k<3;++k) s += posInv[i][k]*(vels[tid*12+(k+1)*3+j]-u0[j]);
                vG[i][j]=s;
            }
        for (int j=0;j<3;++j)
            for (int i=0;i<3;++i)
                gradFlat[j*3+i] = vG[i][j];
    }
    bool const needScal = (P.fScal != 0) || (P.fSGrad != 0);
    float s0 = 0.0f, sG[3] = {0.0f,0.0f,0.0f};
    if (needScal) {
        s0 = scal[tid*4+0];
        for (int i=0;i<3;++i) {
            float s=0.0f;
            for (int j=0;j<3;++j) s += posInv[i][j]*(scal[tid*4+j+1]-s0);
            sG[i]=s;
        }
    }

    if (cnts[tid] == 0u) {   // single-cell tetrahedron: the fast path, by its only item
        if (sub != 0u) return;
        float denC = 0.0f;
        for (int v=0; v<4; ++v) denC += dens[tid*4+v];
        denC *= 0.25f;
        float velC[3] = {0.0f,0.0f,0.0f};
        if (P.fVel) {
            for (int v=0;v<4;++v) for (int j=0;j<3;++j) velC[j] += vels[tid*12+v*3+j];
            for (int j=0;j<3;++j) velC[j] *= 0.25f;
        }
        float scC = 0.0f;
        if (P.fScal) { for (int v=0;v<4;++v) scC += scal[tid*4+v]; scC *= 0.25f; }
        depositCell(denGrid, velGrid, gradGrid, scalGrid, sgGrid, P, ull(flats[tid]), denC, velC, gradFlat, scC, sG, vols[tid]);
        return;
    }

    float den0 = dens[tid*4+0];
    float densGrad[3];
    for (int i=0;i<3;++i) {
        float s=0.0f;
        for (int j=0;j<3;++j) s += posInv[i][j]*(dens[tid*4+j+1]-den0);
        densGrad[i]=s;
    }

    int iLo[3], iHi[3];
    for (int d=0; d<3; ++d) {
        float lo = 0.0f, hi = 0.0f;
        for (int v=0; v<3; ++v) { lo = fminf(lo, Ax[v][d]); hi = fmaxf(hi, Ax[v][d]); }
        iLo[d] = int(floorf((base[d] + lo) * P.invDx[d]));
        iHi[d] = int(floorf((base[d] + hi) * P.invDx[d])) + 1;
        if (iLo[d] < 0) iLo[d] = 0;
        if (iHi[d] > P.nGrid[d]) iHi[d] = P.nGrid[d];
    }
    ull const n1 = ull(max(iHi[1] - iLo[1], 0));
    ull const n2 = ull(max(iHi[2] - iLo[2], 0));
    ull const n12 = n1 * n2;
    ull const nW = ull(max(iHi[0] - iLo[0], 0)) * n12;
    ull w = ull(sub) * ull(P.itemSamples);
    if (w >= nW) return;
    ull const wEnd = (nW - w > ull(P.itemSamples)) ? w + ull(P.itemSamples) : nW;
    int a0 = int(w / n12);
    ull const rem = w - ull(a0) * n12;
    int a1 = int(rem / n2);
    int a2 = int(rem - ull(a1) * n2);

    V3 t1 = v3(Ax[0][0], Ax[0][1], Ax[0][2]);
    V3 t2 = v3(Ax[1][0], Ax[1][1], Ax[1][2]);
    V3 t3 = v3(Ax[2][0], Ax[2][1], Ax[2][2]);
    if (det3(Ax) < 0.0f) { V3 tt = t1; t1 = t2; t2 = tt; }
    TetPlanes const faces = tet_planes(t1, t2, t3);
    V3 const hcell = v3(0.5f*P.dx[0], 0.5f*P.dx[1], 0.5f*P.dx[2]);

    for (; w < wEnd; ++w) {
        int const pos[3] = { iLo[0] + a0, iLo[1] + a1, iLo[2] + a2 };
        if (ull(++a2) == n2) { a2 = 0; if (ull(++a1) == n1) { a1 = 0; ++a0; } }
        V3 const ccen = v3((float(pos[0]) + 0.5f) * P.dx[0] - base[0],
                           (float(pos[1]) + 0.5f) * P.dx[1] - base[1],
                           (float(pos[2]) + 0.5f) * P.dx[2] - base[2]);
        if (box_outside_tet(faces, ccen, hcell)) continue;
        // clipped in the OVERLAP frame (gpu_exact_clip.cuh: origin at the centre of the tet bbox's
        // overlap with the cell, where the piece lies -- no frame far from the piece cancels)
        V3 const org = overlap_origin(ccen, hcell, t1, t2, t3);
        float const cc3[3] = { ccen.x, ccen.y, ccen.z }, hc[3] = { hcell.x, hcell.y, hcell.z };
        float const og[3] = { org.x, org.y, org.z };
        ExPoly piece;
        exact_init_tet4(piece, neg3(org), sub3(t1, org), sub3(t2, org), sub3(t3, org));
        for (int d=0; d<3; ++d) {
            exact_clip_plane(piece, d,  1.0f, og[d] - (cc3[d] - hc[d]));
            exact_clip_plane(piece, d, -1.0f, (cc3[d] + hc[d]) - og[d]);
        }
        if (piece.nv == 0) continue;
        float mo[4];
        exact_reduce1(piece, mo);
        if (!(mo[0] > 0.0f)) continue;
        float const dV = mo[0];
        float const cen[3] = { org.x + mo[1]/mo[0], org.y + mo[2]/mo[0], org.z + mo[3]/mo[0] };
        float denS = den0;
        for (int i=0;i<3;++i) denS += cen[i]*densGrad[i];
        float velS[3] = {0.0f,0.0f,0.0f};
        if (P.fVel)
            for (int j=0;j<3;++j) {
                velS[j]=u0[j];
                for (int i=0;i<3;++i) velS[j] += vG[i][j]*cen[i];
            }
        float scS = s0;
        if (P.fScal) for (int i=0;i<3;++i) scS += cen[i]*sG[i];
        depositCell(denGrid, velGrid, gradGrid, scalGrid, sgGrid, P, ull(fullFlat(P, pos)), denS, velS, gradFlat, scS, sG, dV);
    }
}

// zeroes the cells of the box [off, off+dims) of a full grid with 'comps' floats per cell
__global__ void zeroBoxKernel(float* base, int f1, int f2, int o0, int o1, int o2, int d0, int d1, int d2, unsigned comps)
{
    ull const idx = ull(blockIdx.x) * blockDim.x + threadIdx.x;
    ull const nRow = ull(d0) * ull(d1);
    if (idx >= nRow) return;
    int const i = int(idx / ull(d1)), j = int(idx % ull(d1));
    ull const row = (ull(o0 + i) * ull(f1) + ull(o1 + j)) * ull(f2) + ull(o2);
    float* p = base + row * comps;
    for (ull k = 0; k < ull(d2) * comps; ++k) p[k] = 0.0f;
}


// ================================================================================================
// host
// ================================================================================================
struct Ctx
{
    bool        ready = false;
    std::string err;
    std::string name;
};


std::mutex& ctxMutex() { static std::mutex m; return m; }
// DTFE_GPU_TIMING=1 (alias DTFE_METAL_TIMING): one line per batch
bool timingOn() { static bool t = Env::flag({"DTFE_GPU_TIMING", "DTFE_METAL_TIMING"}); return t; }

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

// the item kernel for a cache size, field set and sum mode (all 48 instantiations are compiled ahead of time;
// FX, the exact sums -- Metal's FC_FIXED)
typedef void (*ItemLaunch)(unsigned blocks, size_t smem,
                           const float*, const float*, const float*, const float*, const unsigned*, const unsigned*,
                           const float4*, float*, float*, float*, const DepositParams&, const unsigned*, const unsigned*,
                           const float*, float*, float*);

template <int K, bool FD, bool FV, bool FG, bool FX>
void launchItemsT(unsigned blocks, size_t smem,
                  const float* v, const float* d, const float* u, const float* vol, const unsigned* cnt, const unsigned* flt,
                  const float4* s, float* den, float* vel, float* grad, const DepositParams& P,
                  const unsigned* g, const unsigned* b, const float* sc, float* scal, float* sgrad)
{
    auto const kfn = depositItemsKernel<K, FD, FV, FG, FX>;
    GPU_LAUNCH(kfn, blocks, 256, smem, v, d, u, vol, cnt, flt, s, den, vel, grad, P, g, b, sc, scal, sgrad);
}

template <int K, bool FX>
ItemLaunch itemLaunchKX(int mask)
{
    switch (mask & 7)
    {
        case 0: return launchItemsT<K, false, false, false, FX>;
        case 1: return launchItemsT<K, true,  false, false, FX>;
        case 2: return launchItemsT<K, false, true,  false, FX>;
        case 3: return launchItemsT<K, true,  true,  false, FX>;
        case 4: return launchItemsT<K, false, false, true,  FX>;
        case 5: return launchItemsT<K, true,  false, true,  FX>;
        case 6: return launchItemsT<K, false, true,  true,  FX>;
        default: return launchItemsT<K, true,  true,  true,  FX>;
    }
}

template <int K>
ItemLaunch itemLaunchK(int mask, bool fFixed)
{
    return fFixed ? itemLaunchKX<K, true>(mask) : itemLaunchKX<K, false>(mask);
}

ItemLaunch itemLaunch(int cacheK, bool fDen, bool fVel, bool fGrad, bool fFixed)
{
    int const mask = (fDen ? 1 : 0) | (fVel ? 2 : 0) | (fGrad ? 4 : 0);
    return cacheK == 2 ? itemLaunchK<2>(mask, fFixed) : cacheK == 8 ? itemLaunchK<8>(mask, fFixed) : itemLaunchK<4>(mask, fFixed);
}

// a zeroed device allocation; null (and 'err' set) on failure
float* devAlloc(size_t bytes, const char* what, std::string& err)
{
    void* p = nullptr;
    if (!gpuCheck(gpuMalloc(&p, bytes ? bytes : 4), what, err)) return nullptr;
    if (!gpuCheck(gpuMemset(p, 0, bytes ? bytes : 4), what, err)) { gpuFree(p); return nullptr; }
    return static_cast<float*>(p);
}

float* devUpload(const void* src, size_t bytes, const char* what, std::string& err)
{
    float* p = devAlloc(bytes, what, err);
    if (!p || !bytes) return p;
    if (!gpuCheck(gpuMemcpy(p, src, bytes, gpuMemcpyHostToDevice), what, err)) { gpuFree(p); return nullptr; }
    return p;
}

inline unsigned blocksFor(size_t n, int tpb) { return unsigned((n + size_t(tpb) - 1) / size_t(tpb)); }


// The checkpoints and the cause-aware retry policy of the GPU deposit (ps_gpu_cuda.cu's, see
// gpu_cuda_retry.cuh): GpuFault, the error codes, gpuRetryCause, the process-wide GpuPace, the DTFE_GPU_FAULT_AT
// test hook and GpuSnapshot (device-to-host copies). The pass loop is runBatch below.
#include "gpu_cuda_retry.cuh"

// the run's retry counters (the progress sidecar's and DTFE.cpp's closing line), by gpuRetryCause's index;
// averaged_interpolation_1.cc defines them
void countGpuRetry(GpuFault f, gpuError_t code)
{
    int const k = gpuRetryCause(f, code);
    ++dtfeGpuRetries;
    ++dtfeGpuRetriesByCause[k];
}

// Chunks start at multiples of this many work items (dtfe_metal_host.cc's rule): with the 256-thread
// blocks also starting there, a warp (32 lanes on NVIDIA, 64 on CDNA) is always the same items, so its float warp
// sum -- the item kernel's flush, which the exact sums convert only after -- does not depend on the chunking.
size_t const CHUNK_ALIGN = 256;
size_t alignChunkUp(size_t n) { return (n + CHUNK_ALIGN - 1) / CHUNK_ALIGN * CHUNK_ALIGN; }


// ---------------------------------------------------------------- the accumulated grids
struct Grids
{
    float* den  = nullptr;   // nCell components (or a 4-byte dummy)
    float* vel  = nullptr;   // nCell*3
    float* grad = nullptr;   // nCell*9
    float* scal = nullptr;   // nCell     the scalar
    float* sgrad = nullptr;  // nCell*3   its gradient
    size_t full[3] = {0, 0, 0};
    bool fDen = false, fVel = false, fGrad = false, fScal = false, fSGrad = false;
    bool fixed = false;      // the components are two-word fixed-point sums (8 B), else floats

    size_t nCell() const { return full[0] * full[1] * full[2]; }
    size_t compBytes() const { return fixed ? 8 : 4; }
    size_t bytes(bool on, size_t comps) const { return on ? nCell() * comps * compBytes() : 4; }
    // the buffers the kernels write, for the checkpoints (GpuSnapshot)
    std::vector<DevBuf> buffers() const
    {
        return { { den, bytes(fDen, 1) }, { vel, bytes(fVel, 3) }, { grad, bytes(fGrad, 9) },
                 { scal, bytes(fScal, 1) }, { sgrad, bytes(fSGrad, 3) } };
    }
    bool alloc(const size_t fullGrid[3], bool d, bool v, bool g, bool s, bool sg, std::string& err, bool fx = false)
    {
        for (int i = 0; i < 3; ++i) full[i] = fullGrid[i];
        fDen = d; fVel = v; fGrad = g; fScal = s; fSGrad = sg;
        fixed = fx;
        den  = devAlloc(bytes(fDen, 1), "density grid", err);
        vel  = den  ? devAlloc(bytes(fVel, 3), "velocity grid", err) : nullptr;
        grad = vel  ? devAlloc(bytes(fGrad, 9), "gradient grid", err) : nullptr;
        scal = grad ? devAlloc(bytes(fScal, 1), "scalar grid", err) : nullptr;
        sgrad = scal ? devAlloc(bytes(fSGrad, 3), "scalar-gradient grid", err) : nullptr;
        return den && vel && grad && scal && sgrad;
    }
    void release()
    {
        for (float** b : {&den, &vel, &grad, &scal, &sgrad}) if (*b) { gpuFree(*b); *b = nullptr; }
    }
    // zero the cells of a sub-grid box [off, off+dims) -- the owner box of one batch
    bool zeroBox(const int32_t off[3], const int32_t dims[3], std::string& err)
    {
        auto zero = [&](float* b, bool on, unsigned comps) -> bool {
            if (!on || !b) return true;
            size_t const nRow = size_t(dims[0]) * size_t(dims[1]);
            if (!nRow) return true;
            GPU_LAUNCH(zeroBoxKernel, blocksFor(nRow, 256), 256, 0, b, int(full[1]), int(full[2]), off[0], off[1], off[2],
                       dims[0], dims[1], dims[2], comps);
            return gpuCheck(gpuGetLastError(), "zero box launch", err) && gpuCheck(gpuDeviceSynchronize(), "zero box", err);
        };
        unsigned const w = fixed ? 2u : 1u;   // floats per component
        return zero(den, fDen, 1 * w) && zero(vel, fVel, 3 * w) && zero(grad, fGrad, 9 * w) && zero(scal, fScal, 1 * w)
            && zero(sgrad, fSGrad, 3 * w);
    }
};

// ---------------------------------------------------------------- one sub-domain's batch
struct Batch
{
    float *bV = nullptr, *bD = nullptr, *bU = nullptr, *bVol = nullptr, *bS = nullptr, *bSc = nullptr;
    float *bCnt = nullptr, *bFlt = nullptr, *bG = nullptr, *bB = nullptr;   // (uint32 payloads)
    DepositParams P{};
    size_t nTet = 0, nItems = 0, nSamples = 0, nSingle = 0;
    bool   needVel = false, needScal = false;
    bool   exact = false;          // --exact-average: depositExactAverage over cell-block items
    DTFEGpuKnobs k;                // dtfe_gpu_fixed.h

    void release()
    {
        for (float** b : {&bV, &bD, &bU, &bVol, &bCnt, &bFlt, &bS, &bG, &bB, &bSc}) if (*b) { gpuFree(*b); *b = nullptr; }
    }
};

// Uploads a tetrahedron set (CONSUMING the host arrays one at a time) and builds the item map.
bool uploadBatch(std::vector<float>& verts, std::vector<float>& dens, std::vector<float>& vels,
                 std::vector<float>& scal,
                 std::vector<float>& vols, std::vector<uint32_t>& cnts, std::vector<uint32_t>& flats,
                 const std::vector<float>& sobol, const double dx[3], const size_t nGrid[3],
                 const size_t subOff[3], const size_t fullGrid[3], bool fDen, bool fVel, bool fGrad,
                 bool fScal, bool fSGrad, bool exact, bool allowFixed, Batch& b, std::string& err)
{
    b.k = dtfeGpuKnobs(true);
    b.exact = exact;
    if (exact) b.k.useItems = true;    // the exact kernel is an item kernel (cell blocks), always
    b.needVel = fVel || fGrad;
    b.needScal = fScal || fSGrad;
    b.nTet = cnts.size();
    // the parameter block, with the exact sums' units from this batch's values (dtfe_gpu_fixed.h, shared with
    // the Metal host)
    DepositParams& P = b.P;
    std::memset(&P, 0, sizeof(P));
    uint32_t const exactCells = dtfeBatchParams(P, b.k, verts, dens, vels, scal, vols, dx, nGrid, subOff, fullGrid,
                                                fDen, fVel, fGrad, fScal, fSGrad, exact, allowFixed);
    dtfeShiftFlats(cnts, flats, nGrid, subOff, fullGrid);
    std::vector<uint32_t> itemStart, blockTet;
    P.nTetTotal = uint32_t(b.nTet);
    if (!dtfeItemMap(verts, cnts, dx, nGrid, b.k.useItems, exact, b.k.itemSamples, exactCells, ITEM_BLOCK_SHIFT,
                     itemStart, blockTet, b.nItems, b.nSamples, b.nSingle, err))
        return false;

    bool ok = true;
    auto up = [&](const void* ptr, size_t bytes, const char* what) -> float* {
        if (!ok) return nullptr;
        float* p = devUpload(ptr, bytes, what, err);
        if (!p) ok = false;
        return p;
    };
    b.bV = up(verts.data(), verts.size() * sizeof(float), "upload vertices");   std::vector<float>().swap(verts);
    b.bD = up(dens.data(),  dens.size()  * sizeof(float), "upload densities");  std::vector<float>().swap(dens);
    b.bU = b.needVel ? up(vels.data(), vels.size() * sizeof(float), "upload velocities") : up(nullptr, 0, "dummy");
    std::vector<float>().swap(vels);
    b.bSc = b.needScal ? up(scal.data(), scal.size() * sizeof(float), "upload scalars") : up(nullptr, 0, "dummy");
    std::vector<float>().swap(scal);
    b.bVol = up(vols.data(),  vols.size()  * sizeof(float), "upload volumes");       std::vector<float>().swap(vols);
    b.bCnt = up(cnts.data(),  cnts.size()  * sizeof(uint32_t), "upload counts");     std::vector<uint32_t>().swap(cnts);
    b.bFlt = up(flats.data(), flats.size() * sizeof(uint32_t), "upload flat indices"); std::vector<uint32_t>().swap(flats);
    std::vector<float> const sobol4 = dtfeSobol4(sobol);
    b.bS = up(sobol4.data(), sobol4.size() * sizeof(float), "upload sample table");
    b.bG = up(itemStart.data(), itemStart.size() * sizeof(uint32_t), "upload item map");
    b.bB = up(blockTet.data(), blockTet.size() * sizeof(uint32_t), "upload block table");
    if (!ok)
    {
        if (err.empty()) err = GPU_BACKEND_STRING " allocation failed (out of device memory?)";
        b.release();
        return false;
    }
    return true;
}

// The chunk controller's chunk, carried from one batch to the next on the shared grid (a per-call batch starts
// afresh). The pace -- the kernels' target length and the gaps -- is the process-wide GpuPace
// (gpu_cuda_retry.cuh), as in dtfe_metal_host.cc.
struct ChunkState
{
    size_t     chunk     = 0;       // 0 = not started: use the kernel's first chunk
};

struct Stats { int chunks = 0, retries = 0; };

// Runs one batch into the grids (CALLER HOLDS ctxMutex). Chunked, serialized launches: the chunk controller
// bounds each kernel's duration (a display-attached Linux GPU kills kernels after a few seconds). The
// retry policy is ps_gpu_cuda.cu's runChunked (dtfe_metal_host.cc runBatch has the rationale):
// a failed kernel restores the batch's last snapshot -- its start (the per-call grids are zero there; the shared
// grid's owner box is re-zeroed) or a checkpoint every ~10 s of GPU work (DTFE_GPU_CHECKPOINT_CHUNKS=<n>
// kernels; per-call only: a device-to-host copy of the batch's grids) -- and continues, at a pace that depends
// on the cause; a STICKY error (the context is lost: CUDA's launch timeout is one), 3 failures of the same chunk
// or 16 in the batch hand the sub-domain to the CPU. Every failed kernel counts into the run's retry counters.
bool runBatch(Batch& b, Grids& g, ChunkState& cs, Stats& st, bool sharedBox, std::string& err)
{
    bool const exact = b.exact;
    bool useItems = b.k.useItems;
    size_t const MIN_CHUNK = exact ? 1024 : useItems ? 4096 : 2048;
    size_t const MAX_CHUNK = exact ? 499968 : useItems ? 4000000 : 499968;   // (multiples of CHUNK_ALIGN)
    size_t firstChunk = exact ? 8192 : useItems ? 65536 : 25088;
    // DTFE_GPU_CHUNK (alias DTFE_METAL_CHUNK): the opening chunk
    if (long const v = Env::integer({"DTFE_GPU_CHUNK", "DTFE_METAL_CHUNK"}, 0); v > 0) firstChunk = size_t(v);
    GpuPace& pace = gpuPace();        // targetSec 0.25 s, no gap until escalated
    // an escalated pace (an earlier batch's failures) opens with a proportionally smaller chunk
    if (pace.targetSec < GPU_PACE_TARGET0)
        firstChunk = std::max(size_t(double(firstChunk) * pace.targetSec / GPU_PACE_TARGET0), MIN_CHUNK);
    if (cs.chunk == 0) cs.chunk = firstChunk;

    size_t const nThreads = useItems ? b.nItems : b.nTet;
    if (nThreads == 0) return true;
    int const tpb = 256;
    bool const fixedSum = b.P.fixedSum != 0;
    // the per-block cell table: as many power-of-two entries as fit 32 KB of shared memory (keys
    // 4 B + 4 or 13 values per entry: floats, or the exact sums' two-word integers), like the Metal kernel's
    // threadgroup memory; DTFE_GPU_TABLE=0 turns it off (evictions go straight to the device)
    uint32_t tableEntries = 0;
    size_t smemBytes = 0;
    ItemLaunch launch = nullptr;
    if (useItems && !exact)
    {   // == the kernel's tableStride(): den, vel, [scalar], [gradient], [scalar gradient]
        size_t const stride = 4 + (b.P.fScal ? 1 : 0) + (b.P.fGrad ? 9 : 0) + (b.P.fSGrad ? 3 : 0);
        size_t const valueBytes = fixedSum ? 8 : 4;
        size_t const budget = 32768;
        uint32_t e = 1;
        while (size_t(e * 2) * (4 + stride * valueBytes) <= budget) e *= 2;
        if (size_t(e) * (4 + stride * valueBytes) > budget) e = 0;
        if (!Env::flag("DTFE_GPU_TABLE", true)) e = 0;
        tableEntries = e >= 64 ? e : 0;
        smemBytes = tableEntries ? size_t(tableEntries) * (4 + stride * valueBytes) : 0;
        launch = itemLaunch(b.k.cacheK, b.P.fDen != 0, b.P.fVel != 0, b.P.fGrad != 0, fixedSum);
    }
    b.P.tableEntries = tableEntries;

    // DTFE_GPU_CHECKPOINT_CHUNKS=<n>: a checkpoint every n kernels (0 = only the batch-start snapshot). Unset:
    // every ~10 s of GPU work (n = 10 s / the current kernel target), as the PS deposit's.
    long ckChunks = -1;
    if (Env::isSet("DTFE_GPU_CHECKPOINT_CHUNKS")) ckChunks = std::max(0L, Env::integer("DTFE_GPU_CHECKPOINT_CHUNKS", 0));
    // DTFE_GPU_PACE_RECOVER_SEC=<s>: the clean GPU work after which an escalated pace steps back (GpuPace)
    double paceRecoverSec = 90.;
    if (double const v = Env::real("DTFE_GPU_PACE_RECOVER_SEC", -1.); v >= 0.) paceRecoverSec = v;
    std::vector<DevBuf> const gridBufs = g.buffers();
    GpuSnapshot snap;
    std::string lastErr;
    // The non-finite guard's counters (fxq), one word per grid (P.bad): zeroed before each kernel and
    // added into badRun after one that succeeded; a restore rolls badRun back to the held snapshot's (badCk), so a
    // failed kernel and the redone ones count once (dtfe_metal_host.cc runBatch). Handed to the run's counters
    // only when the batch succeeds.
    unsigned* dBad = reinterpret_cast<unsigned*>(devAlloc(FX_DROP_SLOTS * 4, "guard counters", err));
    if (!dBad) return false;
    b.P.bad = dBad;
    unsigned long long badRun[FX_DROP_SLOTS] = {}, badCk[FX_DROP_SLOTS] = {};
    struct FreeBad { unsigned* p; ~FreeBad() { gpuFree(p); } } const freeBad{ dBad };   // every return below
    // the batch's start: its grids are zero (per call: freshly zeroed; shared: its owner box), recorded as such
    bool ckOn = sharedBox;
    if (!sharedBox)
    {
        GpuSnapshot::Took const took = snap.take(gridBufs, std::vector<bool>(gridBufs.size(), true), 0, lastErr);
        if (took == GpuSnapshot::DeviceError) { err = lastErr; return false; }
        ckOn = took == GpuSnapshot::Ok;
    }
    bool ckPeriodic = false;           // the held snapshot is a periodic checkpoint (not the batch's start)
    int nFailed = 0, nResumedCk = 0;   // failed kernels of this batch, and how many resumed from a periodic checkpoint
    int const MAX_FAILS_SAME_CHUNK = 3, MAX_FAILS_TOTAL = 16;
    // failures of the 'same chunk': a failure that comes before the redo got past the previous failed kernel's
    // end (chunk boundaries move with the timing controller, so a start offset is no identity)
    int  failsHere = 0;
    size_t failEnd = 0;

    bool success = true;
    size_t chunk = alignChunkUp(std::min(std::max(cs.chunk, MIN_CHUNK), MAX_CHUNK));
    size_t start = 0, sinceCk = 0;
    char const* const kname = exact ? "exact-average" : useItems ? "item" : "per-tetrahedron";
    while (start < nThreads)
    {
        uint32_t const n = uint32_t(std::min(chunk, nThreads - start));
        b.P.nTet = n;
        b.P.itemBase = uint32_t(start);

        if (!gpuCheck(gpuMemset(dBad, 0, FX_DROP_SLOTS * 4), "zero the guard counters", lastErr)) { success = false; break; }
        auto t0 = std::chrono::steady_clock::now();
        if (exact)
            GPU_LAUNCH(depositExactAverage, blocksFor(n, tpb), tpb, 0, b.bV, b.bD, b.bU, b.bVol,
                       reinterpret_cast<const unsigned*>(b.bCnt), reinterpret_cast<const unsigned*>(b.bFlt),
                       g.den, g.vel, g.grad, b.P,
                       reinterpret_cast<const unsigned*>(b.bG), reinterpret_cast<const unsigned*>(b.bB),
                       b.bSc, g.scal, g.sgrad);
        else if (useItems)
            launch(blocksFor(n, tpb), smemBytes, b.bV, b.bD, b.bU, b.bVol,
                   reinterpret_cast<const unsigned*>(b.bCnt), reinterpret_cast<const unsigned*>(b.bFlt),
                   reinterpret_cast<const float4*>(b.bS), g.den, g.vel, g.grad, b.P,
                   reinterpret_cast<const unsigned*>(b.bG), reinterpret_cast<const unsigned*>(b.bB),
                   b.bSc, g.scal, g.sgrad);
        else
            GPU_LAUNCH(depositAveraged1, blocksFor(n, tpb), tpb, 0,
                b.bV + start * 12, b.bD + start * 4, b.needVel ? b.bU + start * 12 : b.bU, b.bVol + start,
                reinterpret_cast<const unsigned*>(b.bCnt) + start, reinterpret_cast<const unsigned*>(b.bFlt) + start,
                reinterpret_cast<const float4*>(b.bS), g.den, g.vel, g.grad, b.P,
                b.needScal ? b.bSc + start * 4 : b.bSc, g.scal, g.sgrad);
        gpuError_t code = gpuGetLastError();
        if (code == gpuSuccess) code = gpuDeviceSynchronize();
        ++st.chunks;
        double const el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        std::string desc;
        GpuFault cause = GpuFault::Resource;
        bool const realError = code != gpuSuccess;
        bool failed = realError;
        if (realError)
        {
            desc = gpuGetErrorString(code);
            cause = code == kGpuErrTimeout ? GpuFault::Watchdog : GpuFault::Resource;
        }
        else failed = gpuFaultInjected(desc, cause, code);   // the test hook (the kernel DID deposit)

        if (failed)
        {
            ++nFailed;
            ++st.retries;
            countGpuRetry(cause, code);   // the run's counters count every failed kernel (run_progress.h)
            lastErr = std::string(GPU_BACKEND_STRING " ") + kname + " kernel failed: " + desc + " [" + gpuFaultName(cause) + "]";
            failsHere = (failsHere > 0 && start < failEnd) ? failsHere + 1 : 1;
            failEnd = std::max(failEnd, start + n);
            if (realError)
            {   // a real error: a sticky one has lost the context, the probe returns it again
                (void)gpuGetLastError();
                if (gpuDeviceSynchronize() != gpuSuccess)
                {
                    cause = GpuFault::Fatal;
                    lastErr += " (sticky: the device context is lost)";
                }
            }
            if (cause == GpuFault::Fatal || !ckOn || failsHere >= MAX_FAILS_SAME_CHUNK || nFailed >= MAX_FAILS_TOTAL)
            {
                if (!ckOn) lastErr += " (no checkpoint held)";
                else if (failsHere >= MAX_FAILS_SAME_CHUNK) lastErr += " (" + std::to_string(failsHere) + " failures of the same chunk)";
                else if (nFailed >= MAX_FAILS_TOTAL) lastErr += " (" + std::to_string(nFailed) + " failed kernels in this batch)";
                success = false;
                break;
            }
            if (cause != GpuFault::Victim)
            {   // escalate (process-wide; GpuPace eases it back): 4x shorter kernels, gaps of 8-50 ms, smaller chunk
                pace.targetSec = std::max(pace.targetSec / 4.0, 0.02);
                pace.gapUs     = std::min(std::max(pace.gapUs * 3, GPU_PACE_GAP_MIN), useconds_t(50000));
                pace.cleanSec  = 0.;
                chunk = alignChunkUp(std::max(chunk / 4, MIN_CHUNK));
            }
            std::string const why = lastErr;
            size_t resumeAt = 0;
            if (sharedBox)
            {   // everything launched before has completed (each chunk synchronizes): re-zero the owner box
                if (!g.zeroBox(b.P.subOff, b.P.nGrid, lastErr)) { lastErr = why + " -- " + lastErr; success = false; break; }
            }
            else
            {
                if (!snap.restore(lastErr)) { lastErr = why + " -- " + lastErr; success = false; break; }
                resumeAt = snap.start;
            }
            std::memcpy(badRun, badCk, sizeof badRun);   // (sharedBox: the batch's start, all zero)
            if (ckPeriodic) ++nResumedCk;
            fprintf(stderr, "%sDTFE %s: %s -- resuming from the %s at thread %zu of %zu "
                            "(redoing %zu; failure %d of this chunk, target %.0f ms kernels, %.0f ms gaps)%s\n",
                    MESSAGE::cYellow(), GPU_BACKEND_STRING, lastErr.c_str(), ckPeriodic ? "checkpoint" : "pass start",
                    resumeAt, nThreads, start + n - resumeAt, failsHere,
                    pace.targetSec * 1e3, pace.gapUs / 1e3, MESSAGE::cReset());
            start = resumeAt;
            sinceCk = 0;
            // an innocent victim retries at once; our own kernels' failures let the device settle first
            if (cause == GpuFault::Resource) sleep(2);
            else if (cause == GpuFault::Watchdog) sleep(1);
            continue;
        }

        start += n;
        ++sinceCk;
        {
            unsigned words[FX_DROP_SLOTS];
            if (!gpuCheck(gpuMemcpy(words, dBad, sizeof words, gpuMemcpyDeviceToHost), "read the guard counters", lastErr))
            { success = false; break; }
            for (int k = 0; k < FX_DROP_SLOTS; ++k) badRun[k] += words[k];
        }
        if (el > 1e-4)
        {
            double scale = pace.targetSec / el;
            if (scale < 0.5) scale = 0.5;
            if (scale > 4.0) scale = 4.0;
            chunk = alignChunkUp(std::min(std::max(size_t(double(chunk) * scale), MIN_CHUNK), MAX_CHUNK));
        }
        if (pace.targetSec < GPU_PACE_TARGET0 || pace.gapUs > GPU_PACE_GAP0)
        {   // an escalated pace steps back after a clean stretch of GPU work (GpuPace)
            pace.cleanSec += el + pace.gapUs / 1e6;
            if (pace.cleanSec >= paceRecoverSec)
            {
                pace.targetSec = std::min(pace.targetSec * 4.0, GPU_PACE_TARGET0);
                pace.gapUs     = pace.gapUs / 3 < GPU_PACE_GAP_MIN ? GPU_PACE_GAP0 : pace.gapUs / 3;
                fprintf(stderr, "DTFE %s: %.0f s of GPU work without a fault -- pace eased to %.0f ms "
                                "kernels, %.0f ms gaps%s\n", GPU_BACKEND_STRING, pace.cleanSec, pace.targetSec * 1e3,
                        pace.gapUs / 1e3,
                        pace.targetSec >= GPU_PACE_TARGET0 && pace.gapUs <= GPU_PACE_GAP0 ? " (the default)" : "");
                pace.cleanSec = 0.;
            }
        }
        long const every = ckChunks >= 0 ? ckChunks : std::max(1L, long(std::lround(10.0 / pace.targetSec)));
        if (!sharedBox && ckOn && every > 0 && long(sinceCk) >= every && start < nThreads)
        {   // synchronized above: the buffers hold exactly [0, start)
            GpuSnapshot::Took const t = snap.take(gridBufs, std::vector<bool>(), start, lastErr);
            if (t == GpuSnapshot::DeviceError) { success = false; break; }
            if (t == GpuSnapshot::Ok) { ++snap.taken; ckPeriodic = true; sinceCk = 0; std::memcpy(badCk, badRun, sizeof badCk); }
            // else: no host memory for the spare copy; the last snapshot stays valid
        }
        if (pace.gapUs) usleep(pace.gapUs);
    }
    cs.chunk = chunk;
    if (nFailed > 0)
        fprintf(stderr, "%sDTFE %s: deposit %s after %d failed kernel(s) (%d resumed from a periodic checkpoint); "
                        "%d checkpoint(s), %.0f MB held, %.3f s copying%s\n",
                success ? MESSAGE::cYellow() : MESSAGE::cRed(), GPU_BACKEND_STRING,
                success ? "recovered on the GPU" : "gave up",
                nFailed, nResumedCk, snap.taken, snap.bytes / 1.e6, snap.sec, MESSAGE::cReset());
    if (!success)
    {   // leave the owner box clean for whoever deposits this sub-domain instead
        std::string ignored;
        if (sharedBox) g.zeroBox(b.P.subOff, b.P.nGrid, ignored);
        err = lastErr;
    }
    else   // the batch is kept: its dropped shares join the run's counts (reported once at the end, DTFE.cpp)
        for (int k = 0; k < FX_DROP_SLOTS; ++k)
            if (badRun[k]) fixedShareDropped(k, badRun[k]);
    return success;
}

// ---------------------------------------------------------------- the shared accumulator
struct Shared
{
    bool active = false;
    Grids grids;
    ChunkState cs;
    std::chrono::steady_clock::time_point began;
    int batches = 0, failed = 0;
    double busy = 0.;
    std::vector<float> hDen, hVel, hGrad, hScal, hSGrad;   // the read-back grids (Result), freed at End
    bool readBack = false;
};
Shared& shared() { static Shared s; return s; }

} // namespace


std::string gpuBackendName() { return GPU_BACKEND_STRING; }

std::string gpuDeviceName()
{
    std::lock_guard<std::mutex> lock(ctxMutex());
    Ctx& c = ctx();
    return c.ready ? c.name : std::string();
}


// ================================================================ per-call path
bool dtfeGpuDepositAveraged1(std::vector<float>& verts,
                             std::vector<float>& dens,
                             std::vector<float>& vels,
                             std::vector<float>& scal,
                             std::vector<float>& vols,
                             std::vector<uint32_t>& cnts,
                             std::vector<uint32_t>& flats,
                             const std::vector<float>& sobol,
                             const double dx[3], const size_t nGrid[3],
                             bool fDen, bool fVel, bool fGrad, bool fScal, bool fSGrad, bool exact,
                             DTFEGpuGrids& out, std::string& err)
{
    Ctx& c = [&]() -> Ctx& { std::lock_guard<std::mutex> lock(ctxMutex()); return ctx(); }();
    if (!c.ready) { err = c.err; return false; }

    size_t const nCell = nGrid[0] * nGrid[1] * nGrid[2];
    dtfeGpuInitGrids(out, nCell, fDen, fVel, fGrad, fScal, fSGrad);
    if (cnts.empty()) return true;

    size_t const zero[3] = {0, 0, 0};
    Batch b;      // uploaded WITHOUT the mutex: the sub-domains' host work overlaps the queue
    if (!uploadBatch(verts, dens, vels, scal, vols, cnts, flats, sobol, dx, nGrid, zero, nGrid, fDen, fVel, fGrad,
                     fScal, fSGrad, exact, true, b, err))
        return false;
    Grids g;
    ChunkState cs;
    Stats st;
    bool ok;
    double el = 0.;
    {
        std::lock_guard<std::mutex> lock(ctxMutex());   // serialize dispatches
        // the grids are allocated under the device mutex (dtfe_metal_host.cc says why: one
        // batch's at a time, not every waiting sub-domain's; the exact sums' words are 8 B per component)
        if (!g.alloc(nGrid, fDen, fVel, fGrad, fScal, fSGrad, err, b.P.fixedSum != 0))
        {
            if (err.empty()) err = GPU_BACKEND_STRING " allocation failed (out of device memory?)";
            ok = false;
        }
        else
        {
            auto const t0 = std::chrono::steady_clock::now();
            ok = runBatch(b, g, cs, st, false, err);
            el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }
    }
    if (timingOn() && g.den) dtfeLogBatch(GPU_BACKEND_STRING, b, st, 0., el, -1.);
    if (ok && g.fixed)
    {   // the exact sums: each grid's words back to the host, every cell rounded once to the float of the contract
        // (psMassFixedMass: q / scale in double), one grid at a time outside the device mutex
        auto toFloat = [&](std::vector<float>& dst, const float* src, float scale, const char* what) -> bool {
            std::vector<int64_t> q(dst.size());
            if (!gpuCheck(gpuMemcpy(q.data(), src, q.size() * 8, gpuMemcpyDeviceToHost), what, err)) return false;
            for (size_t i = 0; i < dst.size(); ++i) dst[i] = float(psMassFixedMass(q[i], scale));
            return true;
        };
        ok = (!fDen   || toFloat(out.density,     g.den,   b.P.fxScale[0], "copy density back"))
          && (!fVel   || toFloat(out.velocity,    g.vel,   b.P.fxScale[1], "copy velocity back"))
          && (!fGrad  || toFloat(out.grad,        g.grad,  b.P.fxScale[2], "copy gradient back"))
          && (!fScal  || toFloat(out.scalar,      g.scal,  b.P.fxScale[3], "copy scalar back"))
          && (!fSGrad || toFloat(out.scalar_grad, g.sgrad, b.P.fxScale[4], "copy scalar gradient back"));
    }
    else if (ok)
        ok = (!fDen  || gpuCheck(gpuMemcpy(out.density.data(),  g.den,  nCell * 4,  gpuMemcpyDeviceToHost), "copy density back", err))
          && (!fVel  || gpuCheck(gpuMemcpy(out.velocity.data(), g.vel,  nCell * 12, gpuMemcpyDeviceToHost), "copy velocity back", err))
          && (!fGrad || gpuCheck(gpuMemcpy(out.grad.data(),     g.grad, nCell * 36, gpuMemcpyDeviceToHost), "copy gradient back", err))
          && (!fScal || gpuCheck(gpuMemcpy(out.scalar.data(),   g.scal, nCell * 4,  gpuMemcpyDeviceToHost), "copy scalar back", err))
          && (!fSGrad || gpuCheck(gpuMemcpy(out.scalar_grad.data(), g.sgrad, nCell * 12, gpuMemcpyDeviceToHost), "copy scalar gradient back", err));
    b.release(); g.release();
    return ok;
}


// ================================================================ shared accumulator
bool dtfeGpuSharedBegin(const size_t fullGrid[3], bool fDen, bool fVel, bool fGrad, bool fScal, bool fSGrad, std::string& err)
{
    std::lock_guard<std::mutex> lock(ctxMutex());
    Shared& s = shared();
    if (s.active) { err = "a shared GPU grid is already active"; return false; }
    Ctx& c = ctx();
    if (!c.ready) { err = c.err; return false; }
    size_t const nCell = fullGrid[0] * fullGrid[1] * fullGrid[2];
    if (nCell > size_t(UINT32_MAX)) { err = "more than 2^32 grid cells"; return false; }
    // OPT-IN (DTFE_GPU_SHARED=1), as in the Metal host: the per-sub-domain grids measured faster there
    if (!dtfeSharedOptedIn(err)) return false;
    // the full grids live in device memory for the whole parallel region: cap them to half of what
    // is free there (DTFE_GPU_SHARED_GB overrides)
    size_t freeB = 0, totalB = 0;
    double capGb = 4.0;
    if (gpuMemGetInfo(&freeB, &totalB) == gpuSuccess) capGb = 0.5 * double(freeB) / 1e9;
    if (!dtfeSharedGridFits(nCell, fDen, fVel, fGrad, fScal, fSGrad, capGb, err)) return false;
    if (!s.grids.alloc(fullGrid, fDen, fVel, fGrad, fScal, fSGrad, err))
    {
        s.grids.release();
        if (err.empty()) err = GPU_BACKEND_STRING " allocation failed for the shared grid";
        return false;
    }
    s.cs = ChunkState();
    s.began = std::chrono::steady_clock::now();
    s.batches = s.failed = 0;
    s.busy = 0.;
    s.readBack = false;
    s.active = true;
    return true;
}

bool dtfeGpuSharedActive()
{
    std::lock_guard<std::mutex> lock(ctxMutex());
    return shared().active;
}

bool dtfeGpuSharedDeposit(std::vector<float>& verts, std::vector<float>& dens, std::vector<float>& vels,
                          std::vector<float>& scal,
                          std::vector<float>& vols, std::vector<uint32_t>& cnts, std::vector<uint32_t>& flats,
                          const std::vector<float>& sobol, const double dx[3], const size_t nGrid[3],
                          const size_t subOff[3], std::string& err)
{
    Shared& s = shared();
    if (!s.active) { err = "no shared GPU grid"; return false; }
    for (int d = 0; d < 3; ++d)
        if (subOff[d] + nGrid[d] > s.grids.full[d]) { err = "sub-grid outside the shared grid"; return false; }
    if (cnts.empty()) return true;

    auto const tReady = std::chrono::steady_clock::now();
    Batch b;      // uploaded WITHOUT the mutex (the float atomics: the shared grid's one unit would have to fit
                  // every batch, which no batch's bound sets -- the exact sums are the per-call path's)
    if (!uploadBatch(verts, dens, vels, scal, vols, cnts, flats, sobol, dx, nGrid, subOff, s.grids.full,
                     s.grids.fDen, s.grids.fVel, s.grids.fGrad, s.grids.fScal, s.grids.fSGrad, false, false, b, err))
        return false;

    bool ok;
    double waitSec, runSec;
    {
        std::lock_guard<std::mutex> lock(ctxMutex());
        auto const tGot = std::chrono::steady_clock::now();
        waitSec = std::chrono::duration<double>(tGot - tReady).count();
        Stats st;
        ok = runBatch(b, s.grids, s.cs, st, true, err);
        runSec = std::chrono::duration<double>(std::chrono::steady_clock::now() - tGot).count();
        s.busy += runSec;
        ++s.batches;
        if (!ok) ++s.failed;
        if (timingOn())
            dtfeLogBatch(GPU_BACKEND_STRING, b, st, waitSec, runSec, std::chrono::duration<double>(tReady - s.began).count());
    }
    b.release();
    return ok;
}

bool dtfeGpuSharedResult(const float** den, const float** vel, const float** grad, const float** scal, const float** sgrad)
{
    std::lock_guard<std::mutex> lock(ctxMutex());
    Shared& s = shared();
    if (!s.active) return false;
    if (!s.readBack)
    {   // the contract hands out host pointers: copy the device grids back once
        size_t const nCell = s.grids.nCell();
        std::string err;
        bool ok = true;
        if (s.grids.fDen)  { s.hDen.resize(nCell);      ok = ok && gpuCheck(gpuMemcpy(s.hDen.data(),  s.grids.den,  nCell * 4,  gpuMemcpyDeviceToHost), "copy density back", err); }
        if (s.grids.fVel)  { s.hVel.resize(nCell * 3);  ok = ok && gpuCheck(gpuMemcpy(s.hVel.data(),  s.grids.vel,  nCell * 12, gpuMemcpyDeviceToHost), "copy velocity back", err); }
        if (s.grids.fGrad) { s.hGrad.resize(nCell * 9); ok = ok && gpuCheck(gpuMemcpy(s.hGrad.data(), s.grids.grad, nCell * 36, gpuMemcpyDeviceToHost), "copy gradient back", err); }
        if (s.grids.fScal) { s.hScal.resize(nCell);     ok = ok && gpuCheck(gpuMemcpy(s.hScal.data(), s.grids.scal, nCell * 4,  gpuMemcpyDeviceToHost), "copy scalar back", err); }
        if (s.grids.fSGrad) { s.hSGrad.resize(nCell * 3); ok = ok && gpuCheck(gpuMemcpy(s.hSGrad.data(), s.grids.sgrad, nCell * 12, gpuMemcpyDeviceToHost), "copy scalar gradient back", err); }
        if (!ok)
        {
            fprintf(stderr, "DTFE %s: shared grid read-back failed (%s)\n", GPU_BACKEND_STRING, err.c_str());
            return false;
        }
        s.readBack = true;
    }
    if (den)  *den  = s.grids.fDen  ? s.hDen.data()  : nullptr;
    if (vel)  *vel  = s.grids.fVel  ? s.hVel.data()  : nullptr;
    if (grad) *grad = s.grids.fGrad ? s.hGrad.data() : nullptr;
    if (scal) *scal = s.grids.fScal ? s.hScal.data() : nullptr;
    if (sgrad) *sgrad = s.grids.fSGrad ? s.hSGrad.data() : nullptr;
    if (timingOn())
        fprintf(stderr, "DTFE %s timing: shared grid: %d batches (%d handed back), device busy %.2f s of the "
                        "%.2f s since begin\n", GPU_BACKEND_STRING, s.batches, s.failed, s.busy,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - s.began).count());
    return true;
}

void dtfeGpuSharedEnd()
{
    std::lock_guard<std::mutex> lock(ctxMutex());
    Shared& s = shared();
    if (!s.active) return;
    s.grids.release();
    std::vector<float>().swap(s.hDen);
    std::vector<float>().swap(s.hVel);
    std::vector<float>().swap(s.hGrad);
    std::vector<float>().swap(s.hScal);
    std::vector<float>().swap(s.hSGrad);
    s.readBack = false;
    s.active = false;
}
