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
   the Metal host reads unified memory in place, this one copies explicitly.

   Compiled and its CPU fallback exercised on Linux; NOT yet validated on NVIDIA/AMD hardware
   (metal/README.md, "CUDA / HIP"). */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>
#include <unistd.h>

#include "gpu_host.h"
#include "gpu_cuda_compat.h"

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
};

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
    if (P.fDen)
        atomicAdd(&denGrid[flat], den*w);
    if (P.fVel)
        for (int j=0;j<3;++j)
            atomicAdd(&velGrid[flat*3ull+ull(j)], vel[j]*w);
    if (P.fGrad)
        for (int q=0;q<9;++q)
            atomicAdd(&gradGrid[flat*9ull+ull(q)], gradFlat[q]*w);
    if (P.fScal)
        atomicAdd(&scalGrid[flat], sc*w);
    if (P.fSGrad)
        for (int i=0;i<3;++i)
            atomicAdd(&sgGrid[flat*3ull+ull(i)], sG[i]*w);
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

template <int K, bool FD, bool FV, bool FG>
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
    ull const f = ull(flat);
    if (FD)
        atomicAdd(&denGrid[f], den * factor);
    if (FV) {
        atomicAdd(&velGrid[f*3ull+0ull], vx * factor);
        atomicAdd(&velGrid[f*3ull+1ull], vy * factor);
        atomicAdd(&velGrid[f*3ull+2ull], vz * factor);
    }
    if (P.fScal)
        atomicAdd(&scalGrid[f], sc * factor);
    if (FG && P.debugNoAtomic < 3u) {
        float const w = cnt * factor;
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i)
                atomicAdd(&gradGrid[f*9ull+ull(j*3+i)], vG[i][j] * w);
    }
    if (P.fSGrad && P.debugNoAtomic < 3u) {
        float const w = cnt * factor;
        for (int i = 0; i < 3; ++i)
            atomicAdd(&sgGrid[f*3ull+ull(i)], sG[i] * w);
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

// floats per entry: den, vx, vy, vz, [scalar], [9 gradient], [3 scalar gradient] (== the Metal layout)
template <bool FG> __device__ __forceinline__ unsigned tableOffScal(const DepositParams&)    { return 4u; }
template <bool FG> __device__ __forceinline__ unsigned tableOffGrad(const DepositParams& P)  { return 4u + (P.fScal ? 1u : 0u); }
template <bool FG> __device__ __forceinline__ unsigned tableOffSGrad(const DepositParams& P) { return tableOffGrad<FG>(P) + (FG ? 9u : 0u); }
template <bool FG> __device__ __forceinline__ unsigned tableStride(const DepositParams& P)   { return tableOffSGrad<FG>(P) + (P.fSGrad ? 3u : 0u); }

// the per-block cell table (open addressing, linear probing, keys claimed by compare-exchange)
template <bool FD, bool FV, bool FG>
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
template <int K, bool FD, bool FV, bool FG>
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
                    !tableAdd<FD, FV, FG>(tKeys, tVals, P.tableEntries, vKey, vDen * factor, vVx * factor, vVy * factor,
                                          vVz * factor, vSc * factor, vCnt * factor, vG, sG, P))
                    flushSlot<K, FD, FV, FG>(denGrid, velGrid, gradGrid, scalGrid, sgGrid, P, C, hit, factor, vG, sG);
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
    // evaluates the warp-any conditions; lanes with nothing pending contribute zeros)
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
                    atomicAdd(&denGrid[f], sd);
                if (FV) {
                    atomicAdd(&velGrid[f*3ull+0ull], sx);
                    atomicAdd(&velGrid[f*3ull+1ull], sy);
                    atomicAdd(&velGrid[f*3ull+2ull], sz);
                }
                if (fScal)
                    atomicAdd(&scalGrid[f], ss);
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
                                    atomicAdd(&gradGrid[f*9ull+ull(j*3+i)], vG[i][j] * sw);
                        if (fSGrad)
                            for (int i = 0; i < 3; ++i)
                                atomicAdd(&sgGrid[f*3ull+ull(i)], sG[i] * sw);
                    }
                    mg = mg && !mt;
                }
            }
            pend = pend && !m;
        }
    }
}

// Flushes the block's cell table into the grids: one device atomic per field per used entry.
template <bool FD, bool FV, bool FG>
__device__ __forceinline__ void tableFlush(const unsigned* tKeys, const float* tVals, unsigned E,
                                           unsigned ltid, unsigned tgSize,
                                           float* denGrid, float* velGrid, float* gradGrid,
                                           float* scalGrid, float* sgGrid, const DepositParams& P)
{
    for (unsigned e = ltid; e < E; e += tgSize) {
        unsigned const key = tKeys[e];
        if (key == TABLE_EMPTY) continue;
        ull const f = ull(key);
        unsigned const base = e * tableStride<FG>(P);
        if (P.debugNoAtomic) continue;
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

template <int K, bool FD, bool FV, bool FG>
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
        for (unsigned c = 0u; c < stride; ++c) tVals[e * stride + c] = 0.0f;
    }
    __syncthreads();
    unsigned const iid = blockIdx.x * blockDim.x + threadIdx.x;
    bool const valid = iid < P.nTet;
    Item it; it.tet = 0xffffffffu; it.sub = 0u; it.perTet = 1u;
    if (valid) it = itemOf(itemStart, blockTet, P.nTetTotal, P.itemBase + iid);
    depositItem<K, FD, FV, FG>(verts, dens, vels, vols, cnts, flats, sobol, denGrid, velGrid, gradGrid,
                               scal, scalGrid, sgGrid, P, it, valid, gpuLane(), tKeys, tVals);
    __syncthreads();
    tableFlush<FD, FV, FG>(tKeys, tVals, E, ltid, tgSize, denGrid, velGrid, gradGrid, scalGrid, sgGrid, P);
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
bool timingOn() { static bool t = getenv("DTFE_GPU_TIMING") != nullptr || getenv("DTFE_METAL_TIMING") != nullptr; return t; }

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

// the item kernel for a cache size and field set (all 24 instantiations are compiled ahead of time)
typedef void (*ItemLaunch)(unsigned blocks, size_t smem,
                           const float*, const float*, const float*, const float*, const unsigned*, const unsigned*,
                           const float4*, float*, float*, float*, const DepositParams&, const unsigned*, const unsigned*,
                           const float*, float*, float*);

template <int K, bool FD, bool FV, bool FG>
void launchItemsT(unsigned blocks, size_t smem,
                  const float* v, const float* d, const float* u, const float* vol, const unsigned* cnt, const unsigned* flt,
                  const float4* s, float* den, float* vel, float* grad, const DepositParams& P,
                  const unsigned* g, const unsigned* b, const float* sc, float* scal, float* sgrad)
{
    auto const kfn = depositItemsKernel<K, FD, FV, FG>;
    GPU_LAUNCH(kfn, blocks, 256, smem, v, d, u, vol, cnt, flt, s, den, vel, grad, P, g, b, sc, scal, sgrad);
}

template <int K>
ItemLaunch itemLaunchK(int mask)
{
    switch (mask & 7)
    {
        case 0: return launchItemsT<K, false, false, false>;
        case 1: return launchItemsT<K, true,  false, false>;
        case 2: return launchItemsT<K, false, true,  false>;
        case 3: return launchItemsT<K, true,  true,  false>;
        case 4: return launchItemsT<K, false, false, true >;
        case 5: return launchItemsT<K, true,  false, true >;
        case 6: return launchItemsT<K, false, true,  true >;
        default: return launchItemsT<K, true,  true,  true >;
    }
}

ItemLaunch itemLaunch(int cacheK, bool fDen, bool fVel, bool fGrad)
{
    int const mask = (fDen ? 1 : 0) | (fVel ? 2 : 0) | (fGrad ? 4 : 0);
    return cacheK == 2 ? itemLaunchK<2>(mask) : cacheK == 8 ? itemLaunchK<8>(mask) : itemLaunchK<4>(mask);
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


// ---------------------------------------------------------------- knobs (see dtfe_metal_host.cc)
struct Knobs
{
    bool     useItems    = true;
    uint32_t itemSamples = 48;
    int      cacheK      = 2;
    bool     noWrite     = false;   // DTFE_GPU_DEBUG_NOWRITE: timing experiment, all the work, no atomics
};

Knobs knobs()
{
    Knobs k;
    if (const char* env = getenv("DTFE_GPU_ITEMS")) k.useItems = atoi(env) != 0;
    if (const char* env = getenv("DTFE_GPU_ITEM_SAMPLES"))
    { long v = atol(env); if (v >= 4 && v <= 100000) k.itemSamples = uint32_t(v); }
    if (const char* env = getenv("DTFE_GPU_CACHE_K")) { int v = atoi(env); k.cacheK = (v == 4 || v == 8) ? v : 2; }
    k.noWrite = getenv("DTFE_GPU_DEBUG_NOWRITE") != nullptr;
    return k;
}


// ---------------------------------------------------------------- the accumulated grids
struct Grids
{
    float* den  = nullptr;   // nCell floats (or a 4-byte dummy)
    float* vel  = nullptr;   // nCell*3
    float* grad = nullptr;   // nCell*9
    float* scal = nullptr;   // nCell     the scalar
    float* sgrad = nullptr;  // nCell*3   its gradient
    size_t full[3] = {0, 0, 0};
    bool fDen = false, fVel = false, fGrad = false, fScal = false, fSGrad = false;

    size_t nCell() const { return full[0] * full[1] * full[2]; }
    bool alloc(const size_t fullGrid[3], bool d, bool v, bool g, bool s, bool sg, std::string& err)
    {
        for (int i = 0; i < 3; ++i) full[i] = fullGrid[i];
        fDen = d; fVel = v; fGrad = g; fScal = s; fSGrad = sg;
        den  = devAlloc(fDen  ? nCell() * 4  : 4, "density grid", err);
        vel  = den  ? devAlloc(fVel  ? nCell() * 12 : 4, "velocity grid", err) : nullptr;
        grad = vel  ? devAlloc(fGrad ? nCell() * 36 : 4, "gradient grid", err) : nullptr;
        scal = grad ? devAlloc(fScal ? nCell() * 4  : 4, "scalar grid", err) : nullptr;
        sgrad = scal ? devAlloc(fSGrad ? nCell() * 12 : 4, "scalar-gradient grid", err) : nullptr;
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
        return zero(den, fDen, 1) && zero(vel, fVel, 3) && zero(grad, fGrad, 9) && zero(scal, fScal, 1) && zero(sgrad, fSGrad, 3);
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
    Knobs  k;

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
                 bool fScal, bool fSGrad, bool exact, Batch& b, std::string& err)
{
    b.k = knobs();
    b.exact = exact;
    if (exact) b.k.useItems = true;    // the exact kernel is an item kernel (cell blocks), always
    b.needVel = fVel || fGrad;
    b.needScal = fScal || fSGrad;
    b.nTet = cnts.size();
    DepositParams& P = b.P;
    std::memset(&P, 0, sizeof(P));
    for (int d = 0; d < 3; ++d)
    {
        P.dx[d]       = float(dx[d]);
        P.invDx[d]    = 1.0f / float(dx[d]);
        P.nGrid[d]    = int32_t(nGrid[d]);
        P.subOff[d]   = int32_t(subOff[d]);
        P.fullGrid[d] = int32_t(fullGrid[d]);
    }
    P.fDen  = fDen ? 1 : 0;
    P.fVel  = fVel ? 1 : 0;
    P.fGrad = fGrad ? 1 : 0;
    P.fScal = fScal ? 1 : 0;
    P.fSGrad = fSGrad ? 1 : 0;
    if (b.k.noWrite) P.fDen = P.fVel = P.fGrad = P.fScal = P.fSGrad = 0;
    P.debugNoAtomic = getenv("DTFE_GPU_DEBUG_NOATOMIC") ? uint32_t(std::max(1, atoi(getenv("DTFE_GPU_DEBUG_NOATOMIC")))) : 0u;
    P.itemSamples = b.k.itemSamples;
    uint32_t exactCells = 16;          // the exact kernel's item = a block of cells (DTFE_GPU_EXACT_CELLS)
    if (exact)
    {
        if (const char* env = getenv("DTFE_GPU_EXACT_CELLS")) { long v = atol(env); if (v > 0) exactCells = uint32_t(v); }
        P.itemSamples = exactCells;
    }

    // the fast path's flat indices arrive in the SUB-grid; shift them to full-grid indices
    bool const shifted = subOff[0] || subOff[1] || subOff[2] || fullGrid[0] != nGrid[0]
                      || fullGrid[1] != nGrid[1] || fullGrid[2] != nGrid[2];
    if (shifted)
        for (size_t t = 0; t < b.nTet; ++t)
        {
            if (cnts[t] != 0) continue;
            size_t const f = flats[t];
            size_t const k = f % nGrid[2], j = (f / nGrid[2]) % nGrid[1], i = f / (nGrid[2] * nGrid[1]);
            flats[t] = uint32_t(((i + subOff[0]) * fullGrid[1] + (j + subOff[1])) * fullGrid[2] + (k + subOff[2]));
        }

    std::vector<uint32_t> itemStart, blockTet;
    P.nTetTotal = uint32_t(b.nTet);
    if (b.k.useItems)
    {
        itemStart.resize(b.nTet + 1);
        for (size_t t = 0; t < b.nTet; ++t)
        {
            uint32_t const n = cnts[t];
            itemStart[t] = uint32_t(b.nItems);
            size_t items = n ? (n + b.k.itemSamples - 1) / b.k.itemSamples : 1u;
            if (exact && n)
            {   // an UPPER bound of the kernel's window (margin for the float/double floor); surplus idles
                uint64_t nw = 1;
                for (int d = 0; d < 3 && nw; ++d)
                {
                    double const base = verts[t*12 + d];
                    double lo = 0., hi = 0.;
                    for (int v = 1; v < 4; ++v)
                    {
                        double const a = double(verts[t*12 + v*3 + d]) - base;
                        lo = std::min(lo, a); hi = std::max(hi, a);
                    }
                    double const vlo = (base + lo) / dx[d], vhi = (base + hi) / dx[d];
                    double const flo = std::floor(vlo - (1.e-3 + 1.e-5 * std::fabs(vlo)));
                    double const fhi = std::floor(vhi + (1.e-3 + 1.e-5 * std::fabs(vhi)));
                    int64_t iLo = std::isfinite(flo) ? int64_t(flo) : 0, iHi = std::isfinite(fhi) ? int64_t(fhi) + 1 : 0;
                    iLo = std::max<int64_t>(iLo, 0); iHi = std::min<int64_t>(iHi, int64_t(nGrid[d]));
                    nw = iHi > iLo ? nw * uint64_t(iHi - iLo) : 0;
                }
                items = nw ? size_t((nw + exactCells - 1) / exactCells) : 1u;
            }
            b.nItems += items;
            b.nSamples += exact ? 0 : n;
            b.nSingle += n == 0;
            if (b.nItems > size_t(UINT32_MAX)) { err = "too many work items for one batch (> 2^32)"; return false; }
        }
        itemStart[b.nTet] = uint32_t(b.nItems);
        size_t const nBlocks = (b.nItems >> ITEM_BLOCK_SHIFT) + 2;
        blockTet.assign(nBlocks, uint32_t(b.nTet ? b.nTet - 1 : 0));
        for (size_t t = 0, blk = 0; t < b.nTet && blk < nBlocks; ++t)
            while (blk < nBlocks && (blk << ITEM_BLOCK_SHIFT) < size_t(itemStart[t + 1])) blockTet[blk++] = uint32_t(t);
    }
    else
        for (size_t t = 0; t < b.nTet; ++t) { b.nSamples += cnts[t]; b.nSingle += cnts[t] == 0; }

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
    std::vector<float> sobol4(sobol.size() / 3 * 4, 0.f);     // (x, y, z, 0): one 16-byte load per sample
    for (size_t i = 0; i < sobol.size() / 3; ++i)
        for (int d = 0; d < 3; ++d) sobol4[i * 4 + d] = sobol[i * 3 + d];
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

// The chunk controller's state; one per accumulated grid so it stays warm across batches.
struct ChunkState
{
    double     targetSec = 0.25;
    useconds_t gapUs     = 0;       // no window server to yield to (the Metal host idles 8 ms)
    size_t     chunk     = 0;       // 0 = not started: use the kernel's first chunk
};

struct Stats { int chunks = 0, retries = 0; };

// Runs one batch into the grids (CALLER HOLDS ctxMutex). Chunked, serialized launches: the chunk
// controller bounds each kernel's duration (a display-attached Linux GPU kills kernels after a
// few seconds); a failed chunk is never retried alone -- the batch's OWNER BOX is re-zeroed and
// the whole batch redone with shorter kernels. (A CUDA kernel timeout is a sticky error: the
// retries then fail as well and the caller deposits this sub-domain on the CPU.)
bool runBatch(Batch& b, Grids& g, ChunkState& cs, Stats& st, std::string& err)
{
    bool const exact = b.exact;
    bool useItems = b.k.useItems;
    size_t const MIN_CHUNK = exact ? 1024 : useItems ? 4096 : 2000;
    size_t const MAX_CHUNK = exact ? 500000 : useItems ? 4000000 : 500000;
    size_t firstChunk = exact ? 8192 : useItems ? 65536 : 25000;
    if (const char* env = getenv("DTFE_GPU_CHUNK"))   { long v = atol(env); if (v > 0) firstChunk = size_t(v); }
    if (const char* env = getenv("DTFE_METAL_CHUNK")) { long v = atol(env); if (v > 0) firstChunk = size_t(v); }
    if (cs.chunk == 0) cs.chunk = firstChunk;

    size_t const nThreads = useItems ? b.nItems : b.nTet;
    if (nThreads == 0) return true;
    int const tpb = 256;
    // the per-block cell table: as many power-of-two entries as fit 32 KB of shared memory (keys
    // 4 B + 4 or 13 floats per entry), like the Metal kernel's threadgroup memory; DTFE_GPU_TABLE=0
    // turns it off (evictions go straight to the device)
    uint32_t tableEntries = 0;
    size_t smemBytes = 0;
    ItemLaunch launch = nullptr;
    if (useItems && !exact)
    {   // == the kernel's tableStride(): den, vel, [scalar], [gradient], [scalar gradient]
        size_t const stride = 4 + (b.P.fScal ? 1 : 0) + (b.P.fGrad ? 9 : 0) + (b.P.fSGrad ? 3 : 0);
        size_t const budget = 32768;
        uint32_t e = 1;
        while (size_t(e * 2) * (4 + stride * 4) <= budget) e *= 2;
        if (size_t(e) * (4 + stride * 4) > budget) e = 0;
        if (const char* env = getenv("DTFE_GPU_TABLE")) { if (atoi(env) == 0) e = 0; }
        tableEntries = e >= 64 ? e : 0;
        smemBytes = tableEntries ? size_t(tableEntries) * (4 + stride * 4) : 0;
        launch = itemLaunch(b.k.cacheK, b.P.fDen != 0, b.P.fVel != 0, b.P.fGrad != 0);
    }
    b.P.tableEntries = tableEntries;

    int const maxAttempts = 3;
    bool success = false;
    std::string lastErr;
    for (int attempt = 0; attempt < maxAttempts && !success; ++attempt)
    {
        if (attempt > 0)
        {   // everything launched before has completed (each chunk synchronizes): re-zero the owner box
            if (!g.zeroBox(b.P.subOff, b.P.nGrid, lastErr)) break;
            ++st.retries;
        }
        success = true;
        size_t chunk = std::min(std::max(cs.chunk, MIN_CHUNK), MAX_CHUNK);
        size_t start = 0;
        while (start < nThreads)
        {
            uint32_t const n = uint32_t(std::min(chunk, nThreads - start));
            b.P.nTet = n;
            b.P.itemBase = uint32_t(start);

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
            bool const ok = gpuCheck(gpuGetLastError(), "kernel launch", lastErr)
                         && gpuCheck(gpuDeviceSynchronize(), exact ? "exact-average kernel" : useItems ? "item kernel" : "per-tetrahedron kernel", lastErr);
            ++st.chunks;
            double const el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (!ok) { success = false; break; }     // restart the WHOLE batch (owner box re-zeroed above)

            start += n;
            if (el > 1e-4)
            {
                double scale = cs.targetSec / el;
                if (scale < 0.5) scale = 0.5;
                if (scale > 4.0) scale = 4.0;
                chunk = std::min(std::max(size_t(double(chunk) * scale), MIN_CHUNK), MAX_CHUNK);
            }
            if (cs.gapUs) usleep(cs.gapUs);
        }
        cs.chunk = chunk;

        if (!success)
        {
            cs.targetSec = std::max(cs.targetSec / 4.0, 0.02);
            cs.gapUs     = std::min(std::max(cs.gapUs * 3, useconds_t(8000)), useconds_t(50000));
            cs.chunk     = std::max(cs.chunk / 2, MIN_CHUNK);
            if (attempt + 1 < maxAttempts)
            {
                fprintf(stderr, "DTFE %s: %s -- retrying the deposit from scratch "
                                "(attempt %d/%d, target %.0f ms kernels, %.0f ms gaps)\n",
                        GPU_BACKEND_STRING, lastErr.c_str(), attempt + 2, maxAttempts,
                        cs.targetSec*1e3, cs.gapUs/1e3);
                sleep(2);
            }
        }
    }
    if (!success)
    {   // leave the owner box clean for whoever deposits this sub-domain instead
        std::string ignored;
        g.zeroBox(b.P.subOff, b.P.nGrid, ignored);
        err = lastErr;
    }
    return success;
}

void logBatch(const Batch& b, const Stats& st, double waitSec, double runSec, double sinceBegin)
{
    fprintf(stderr, "DTFE %s timing: %s kernel, %zu tetrahedra (%zu single-cell), %zu items of <= %u %s, "
                    "%zu samples, table %u, %d chunks, %d retries, %.2f s (%.0f M samples/s)%s",
            GPU_BACKEND_STRING,
            b.exact ? "exact" : b.k.useItems ? (b.k.cacheK == 8 ? "item/K8" : b.k.cacheK == 2 ? "item/K2" : "item/K4") : "per-tet",
            b.nTet, b.nSingle, b.k.useItems ? b.nItems : size_t(0), b.P.itemSamples, b.exact ? "cells" : "samples", b.nSamples, b.P.tableEntries,
            st.chunks, st.retries, runSec, runSec > 0 ? b.nSamples / runSec / 1e6 : 0.0,
            sinceBegin >= 0 ? "" : "\n");
    if (sinceBegin >= 0)
        fprintf(stderr, "; shared grid: batch ready %.2f s after begin, waited %.2f s for the device\n",
                sinceBegin, waitSec);
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
    out.density.assign(fDen ? nCell : 0, 0.f);
    out.velocity.assign(fVel ? nCell * 3 : 0, 0.f);
    out.grad.assign(fGrad ? nCell * 9 : 0, 0.f);
    out.scalar.assign(fScal ? nCell : 0, 0.f);
    out.scalar_grad.assign(fSGrad ? nCell * 3 : 0, 0.f);
    if (cnts.empty()) return true;

    size_t const zero[3] = {0, 0, 0};
    Batch b;      // uploaded WITHOUT the mutex: the sub-domains' host work overlaps the queue
    if (!uploadBatch(verts, dens, vels, scal, vols, cnts, flats, sobol, dx, nGrid, zero, nGrid, fDen, fVel, fGrad,
                     fScal, fSGrad, exact, b, err))
        return false;
    Grids g;
    if (!g.alloc(nGrid, fDen, fVel, fGrad, fScal, fSGrad, err))
    {
        if (err.empty()) err = GPU_BACKEND_STRING " allocation failed (out of device memory?)";
        b.release(); g.release();
        return false;
    }
    ChunkState cs;
    Stats st;
    bool ok;
    double el;
    {
        std::lock_guard<std::mutex> lock(ctxMutex());   // serialize dispatches
        auto const t0 = std::chrono::steady_clock::now();
        ok = runBatch(b, g, cs, st, err);
        el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
    if (timingOn()) logBatch(b, st, 0., el, -1.);
    if (ok)
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
    if (!(getenv("DTFE_GPU_SHARED") && atoi(getenv("DTFE_GPU_SHARED")) != 0))
    {
        err = "off by default (DTFE_GPU_SHARED=1 turns it on; the per-sub-domain grids measured faster)";
        return false;
    }
    // the full grids live in device memory for the whole parallel region: cap them to half of what
    // is free there (DTFE_GPU_SHARED_GB overrides)
    size_t freeB = 0, totalB = 0;
    double capGb = 4.0;
    if (gpuMemGetInfo(&freeB, &totalB) == gpuSuccess) capGb = 0.5 * double(freeB) / 1e9;
    if (const char* env = getenv("DTFE_GPU_SHARED_GB")) { double v = atof(env); if (v > 0) capGb = v; }
    double const needGb = double(nCell) * ((fDen ? 4 : 0) + (fVel ? 12 : 0) + (fGrad ? 36 : 0) + (fScal ? 4 : 0) + (fSGrad ? 12 : 0)) / 1e9;
    if (needGb > capGb)
    {
        char buf[200];
        snprintf(buf, sizeof(buf), "the shared GPU grid would take %.1f GB, over the %.1f GB cap", needGb, capGb);
        err = buf;
        return false;
    }
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
    Batch b;      // uploaded WITHOUT the mutex
    if (!uploadBatch(verts, dens, vels, scal, vols, cnts, flats, sobol, dx, nGrid, subOff, s.grids.full,
                     s.grids.fDen, s.grids.fVel, s.grids.fGrad, s.grids.fScal, s.grids.fSGrad, false, b, err))
        return false;

    bool ok;
    double waitSec, runSec;
    {
        std::lock_guard<std::mutex> lock(ctxMutex());
        auto const tGot = std::chrono::steady_clock::now();
        waitSec = std::chrono::duration<double>(tGot - tReady).count();
        Stats st;
        ok = runBatch(b, s.grids, s.cs, st, err);
        runSec = std::chrono::duration<double>(std::chrono::steady_clock::now() - tGot).count();
        s.busy += runSec;
        ++s.batches;
        if (!ok) ++s.failed;
        if (timingOn())
            logBatch(b, st, waitSec, runSec, std::chrono::duration<double>(tReady - s.began).count());
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
