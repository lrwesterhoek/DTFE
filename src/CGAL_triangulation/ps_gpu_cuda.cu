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
#include "ps_deposit_params.h"   // shared host struct (PSDepositParams)

namespace {

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

// ================================================================================================
// --ps-exact-deposit: float32 r3d port (see metal/ps_deposit.metal for the full commentary;
// this is the same algorithm, kept line-for-line in step with the Metal version).
// ================================================================================================
#define EXV 48

struct ExPoly {
    float pos[EXV][3];
    int   nbr[EXV][3];
    int   nv;
};

__device__ inline void exact_init_tet(ExPoly& T, const float t1[3], const float t2[3], const float t3[3])
{
    T.nv = 4;
    for (int i=0;i<3;++i) { T.pos[0][i]=0.0f; T.pos[1][i]=t1[i]; T.pos[2][i]=t2[i]; T.pos[3][i]=t3[i]; }
    T.nbr[0][0]=1; T.nbr[0][1]=3; T.nbr[0][2]=2;
    T.nbr[1][0]=2; T.nbr[1][1]=3; T.nbr[1][2]=0;
    T.nbr[2][0]=0; T.nbr[2][1]=3; T.nbr[2][2]=1;
    T.nbr[3][0]=1; T.nbr[3][1]=2; T.nbr[3][2]=0;
}

__device__ inline void exact_clip_plane(ExPoly& T, int axis, float sgn, float dd)
{
    if (T.nv <= 0) return;
    float sdists[EXV];
    int   clipped[EXV];
    int const onv = T.nv;
    float smin = 1.0e30f, smax = -1.0e30f;
    for (int v=0; v<onv; ++v) {
        float const s = dd + sgn*T.pos[v][axis];
        sdists[v] = s;
        clipped[v] = (s < 0.0f) ? 1 : 0;
        if (s < smin) smin = s;
        if (s > smax) smax = s;
    }
    if (smin >= 0.0f) return;
    if (smax <= 0.0f) { T.nv = 0; return; }

    for (int vcur=0; vcur<onv; ++vcur) {
        if (clipped[vcur]) continue;
        for (int np=0; np<3; ++np) {
            int const vnext = T.nbr[vcur][np];
            if (!clipped[vnext]) continue;
            if (T.nv == EXV) { T.nv = 0; return; }
            float const wa = -sdists[vnext], wb = sdists[vcur];
            for (int i=0;i<3;++i)
                T.pos[T.nv][i] = (wa*T.pos[vcur][i] + wb*T.pos[vnext][i]) / (wa + wb);
            T.nbr[T.nv][0] = vcur; T.nbr[T.nv][1] = -1; T.nbr[T.nv][2] = -1;
            T.nbr[vcur][np] = T.nv;
            T.nv++;
        }
    }

    for (int vstart=onv; vstart<T.nv; ++vstart) {
        int vcur = vstart;
        int vnext = T.nbr[vcur][0];
        int np = 0;
        do {
            for (np=0; np<3; ++np)
                if (T.nbr[vnext][np] == vcur) break;
            vcur = vnext;
            int const pnext = (np+1)%3;
            vnext = T.nbr[vcur][pnext];
        } while (vcur < onv);
        T.nbr[vstart][2] = vcur;
        T.nbr[vcur][1] = vstart;
    }

    int nun = 0;
    for (int v=0; v<T.nv; ++v) {
        bool const isClipped = (v < onv) && (clipped[v] != 0);
        if (!isClipped) {
            for (int i=0;i<3;++i) T.pos[nun][i] = T.pos[v][i];
            T.nbr[nun][0]=T.nbr[v][0]; T.nbr[nun][1]=T.nbr[v][1]; T.nbr[nun][2]=T.nbr[v][2];
            clipped[v] = nun++;
        }
    }
    T.nv = nun;
    for (int v=0; v<T.nv; ++v)
        for (int np=0; np<3; ++np)
            T.nbr[v][np] = clipped[T.nbr[v][np]];
}

__device__ inline void exact_reduce2(const ExPoly& T, float* mom)
{
    for (int m=0; m<10; ++m) mom[m] = 0.0f;
    if (T.nv <= 0) return;
    bool emk[EXV][3];
    for (int v=0; v<T.nv; ++v) { emk[v][0]=false; emk[v][1]=false; emk[v][2]=false; }

    for (int vstart=0; vstart<T.nv; ++vstart)
    for (int pstart=0; pstart<3; ++pstart) {
        if (emk[vstart][pstart]) continue;
        int pnext = pstart;
        int vcur  = vstart;
        emk[vcur][pnext] = true;
        int vnext = T.nbr[vcur][pnext];
        float const v0x=T.pos[vstart][0], v0y=T.pos[vstart][1], v0z=T.pos[vstart][2];
        int np;
        for (np=0; np<3; ++np)
            if (T.nbr[vnext][np] == vcur) break;
        vcur = vnext;
        pnext = (np+1)%3;
        emk[vcur][pnext] = true;
        vnext = T.nbr[vcur][pnext];
        while (vnext != vstart) {
            float const v2x=T.pos[vcur][0],  v2y=T.pos[vcur][1],  v2z=T.pos[vcur][2];
            float const v1x=T.pos[vnext][0], v1y=T.pos[vnext][1], v1z=T.pos[vnext][2];
            float const sixv = (-v2x*v1y*v0z + v1x*v2y*v0z + v2x*v0y*v1z
                                -v0x*v2y*v1z - v1x*v0y*v2z + v0x*v1y*v2z);
            mom[0] += sixv;
            mom[1] += sixv*(v0x+v1x+v2x);
            mom[2] += sixv*(v0y+v1y+v2y);
            mom[3] += sixv*(v0z+v1z+v2z);
            mom[4] += sixv*(v0x*v0x + v1x*v1x + v2x*v2x + v0x*v1x + v0x*v2x + v1x*v2x);
            mom[5] += sixv*(2.0f*(v0x*v0y + v1x*v1y + v2x*v2y)
                            + v0x*v1y + v0y*v1x + v0x*v2y + v0y*v2x + v1x*v2y + v1y*v2x);
            mom[6] += sixv*(2.0f*(v0x*v0z + v1x*v1z + v2x*v2z)
                            + v0x*v1z + v0z*v1x + v0x*v2z + v0z*v2x + v1x*v2z + v1z*v2x);
            mom[7] += sixv*(v0y*v0y + v1y*v1y + v2y*v2y + v0y*v1y + v0y*v2y + v1y*v2y);
            mom[8] += sixv*(2.0f*(v0y*v0z + v1y*v1z + v2y*v2z)
                            + v0y*v1z + v0z*v1y + v0y*v2z + v0z*v2y + v1y*v2z + v1z*v2y);
            mom[9] += sixv*(v0z*v0z + v1z*v1z + v2z*v2z + v0z*v1z + v0z*v2z + v1z*v2z);
            for (np=0; np<3; ++np)
                if (T.nbr[vnext][np] == vcur) break;
            vcur = vnext;
            pnext = (np+1)%3;
            emk[vcur][pnext] = true;
            vnext = T.nbr[vcur][pnext];
        }
    }
    mom[0] /= 6.0f;
    mom[1] /= 24.0f;  mom[2] /= 24.0f;  mom[3] /= 24.0f;
    mom[4] /= 60.0f;  mom[7] /= 60.0f;  mom[9] /= 60.0f;
    mom[5] /= 120.0f; mom[6] /= 120.0f; mom[8] /= 120.0f;
}

// Order-1 variant for PASS 0 of the exact deposit (mirrors metal/ps_deposit.metal): pass 0
// consumes only the volume (and first moments under fLinear), so the six order-2 accumulations
// -- ~78% of the moment arithmetic -- were computed twice per cell only to be discarded.
// IDENTICAL fan walk and accumulation order for mom[0..3] => pass-0 weights bit-identical.
__device__ inline void exact_reduce1(const ExPoly& T, float* mom)
{
    for (int m=0; m<4; ++m) mom[m] = 0.0f;
    if (T.nv <= 0) return;
    bool emk[EXV][3];
    for (int v=0; v<T.nv; ++v) { emk[v][0]=false; emk[v][1]=false; emk[v][2]=false; }

    for (int vstart=0; vstart<T.nv; ++vstart)
    for (int pstart=0; pstart<3; ++pstart) {
        if (emk[vstart][pstart]) continue;
        int pnext = pstart;
        int vcur  = vstart;
        emk[vcur][pnext] = true;
        int vnext = T.nbr[vcur][pnext];
        float const v0x=T.pos[vstart][0], v0y=T.pos[vstart][1], v0z=T.pos[vstart][2];
        int np;
        for (np=0; np<3; ++np)
            if (T.nbr[vnext][np] == vcur) break;
        vcur = vnext;
        pnext = (np+1)%3;
        emk[vcur][pnext] = true;
        vnext = T.nbr[vcur][pnext];
        while (vnext != vstart) {
            float const v2x=T.pos[vcur][0],  v2y=T.pos[vcur][1],  v2z=T.pos[vcur][2];
            float const v1x=T.pos[vnext][0], v1y=T.pos[vnext][1], v1z=T.pos[vnext][2];
            float const sixv = (-v2x*v1y*v0z + v1x*v2y*v0z + v2x*v0y*v1z
                                -v0x*v2y*v1z - v1x*v0y*v2z + v0x*v1y*v2z);
            mom[0] += sixv;
            mom[1] += sixv*(v0x+v1x+v2x);
            mom[2] += sixv*(v0y+v1y+v2y);
            mom[3] += sixv*(v0z+v1z+v2z);
            for (np=0; np<3; ++np)
                if (T.nbr[vnext][np] == vcur) break;
            vcur = vnext;
            pnext = (np+1)%3;
            emk[vcur][pnext] = true;
            vnext = T.nbr[vcur][pnext];
        }
    }
    mom[0] /= 6.0f;
    mom[1] /= 24.0f;  mom[2] /= 24.0f;  mom[3] /= 24.0f;
}

// --ps-linear-deposit sample weight: linear density at offset rel from vertex 0, clamped >= 0
// (the +-1e-6 barycentric tolerance can graze negative).
__device__ inline float linearWeight(const float rel[3], float d0, const float dG[3])
{
    float w = d0 + dG[0]*rel[0] + dG[1]*rel[1] + dG[2]*rel[2];
    return w < 0.0f ? 0.0f : w;
}

// Sub-sample position for local sample index s (base-nSub digits) in cell 'raw' along axis dd.
__device__ inline float samplePos(unsigned s, int dd, int nSub, const int raw[3], const PSDepositParams& P)
{
    unsigned rem = s;
    for (int k=0; k<dd; ++k) rem /= unsigned(nSub);
    float fr = (nSub==1) ? 0.5f : (float(int(rem % unsigned(nSub))) + 0.5f)/float(nSub);
    return P.boxLo[dd] + (float(raw[dd])+fr)*P.dx[dd];
}

// One sample deposit. Offsets are 64-bit BECAUSE flat*9 WOULD overflow a 32-bit uint above
// ~782^3 -- already fixed, do not re-derive. 'flat' itself is still 32-bit (see the callers),
// capping a sub-grid at 2^32 cells; ps_interpolation.cc guards it with a CPU fallback.
// w = mass share (mass grid), wm = moment weight (== w by default; the V_eul share under
// fVolW), obits = the tet's orientation bits for the caustic OR (fCaustic).
__device__ inline void depositSample(float* mass, float* mom, float* m2, float* grad,
                                     unsigned* streams, float* momw, unsigned* caust, float* sv,
                                     float* dispvel, float* dispw,
                                     unsigned flat,
                                     const float rel[3], float w, float wm, unsigned obits,
                                     const float u0[3], const float vG[3][3],
                                     bool fVel, bool fDisp, bool fGrad, bool fVolW, bool fCaustic,
                                     bool fExact, float svShare)
{
    unsigned long long base = flat;
    atomicAdd(&mass[flat], w);
    if (fVel || fDisp)
    {
        float vv[3];
        for (int j=0; j<3; ++j)
        {
            vv[j] = u0[j];
            for (int i=0; i<3; ++i) vv[j] += vG[i][j]*rel[i];
            if (fVel)
                atomicAdd(&mom[base*3ull + (unsigned long long)j], vv[j]*wm);
        }
        if (fDisp)
        {
            // sigma_ij is a moment of f -> always MASS-weighted (w), even under fVolW
            unsigned long long c = 0;
            for (int i=0; i<3; ++i)
                for (int j=i; j<3; ++j)
                    atomicAdd(&m2[base*6ull + c++], w*vv[i]*vv[j]);
            if (fVolW)
            {
                for (int j=0; j<3; ++j) atomicAdd(&dispvel[base*3ull + (unsigned long long)j], vv[j]*w);
                atomicAdd(&dispw[flat], w);
            }
        }
    }
    if (fGrad)
        for (int j=0; j<3; ++j)
            for (int i=0; i<3; ++i)
                atomicAdd(&grad[base*9ull + (unsigned long long)(j*3+i)], vG[i][j]*wm);
    if (fVolW)
        atomicAdd(&momw[flat], wm);
    if (fCaustic)
        atomicOr(&caust[flat], obits);
    if (fExact)
        atomicAdd(&sv[flat], svShare);
    atomicAdd(&streams[flat], 1u);
}

__global__ void psDepositFieldsKernel(const float* verts, const float* vels, const float* masses,
                                      const float* dens,
                                      float* massGrid, float* momGrid, float* m2Grid, float* gradGrid,
                                      unsigned* strGrid, float* momwGrid, unsigned* caustGrid,
                                      float* svGrid, float* dvGrid, float* dwGrid, PSDepositParams P)
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

    // constant per-tet velocity gradient vG = posInv * (u_{k+1}-u_0); skipped entirely (and
    // the vels buffer never read -- it may be a 4-byte dummy) when no velocity-derived grid
    // is requested (fVel, fDisp and fGrad all 0, i.e. a density-only run)
    bool const fVel = P.fVel != 0, fDisp = P.fDisp != 0, fGrad = P.fGrad != 0;
    bool const needsVel = fVel || fDisp || fGrad;
    float u0[3] = { 0.0f, 0.0f, 0.0f };
    float vG[3][3] = { {0.0f,0.0f,0.0f}, {0.0f,0.0f,0.0f}, {0.0f,0.0f,0.0f} };
    if (needsVel)
    {
        for (int j=0; j<3; ++j) u0[j] = vels[tid*12+j];
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

    // --ps-linear-deposit: constant density gradient across the tet (same affine convention
    // as vG above); the vertex densities arrive per tet in the dens buffer.
    bool const fLinear = P.fLinear != 0;
    float d0 = 0.0f, dG[3] = {0.0f, 0.0f, 0.0f};
    if (fLinear)
    {
        d0 = dens[tid*4+0];
        float dd0[3];
        for (int e=0; e<3; ++e) dd0[e] = dens[tid*4+e+1] - d0;
        for (int i=0; i<3; ++i)
        {
            float s = 0.0f;
            for (int k=0; k<3; ++k) s += posInv[i][k]*dd0[k];
            dG[i] = s;
        }
    }

    // --ps-volume-weighted: the tet's total moment weight is its Eulerian volume |det|/6
    // (matching the CPU volumeShare); default: the moment weight equals the mass share.
    // --ps-caustics: orientation bits from the sign of det(Ax).
    bool const fVolW = P.fVolW != 0, fCaustic = P.fCaustic != 0, fExact = P.fExact != 0;
    float const mw = fVolW ? fabsf(d)/6.0f : m;
    unsigned const obits = d > 0.0f ? 1u : 2u;
    float const invCellVol = 1.0f / (P.dx[0]*P.dx[1]*P.dx[2]);
    float const tetVol = fabsf(d)/6.0f;

    // Eulerian bbox -> raw cell range (+/-1 margin when sub-sampling; clamp only non-periodic)
    int iMin[3], iMax[3];
    for (int dd=0; dd<3; ++dd)
    {
        float lo = fminf(fminf(v0[dd],v1[dd]), fminf(v2[dd],v3[dd]));
        float hi = fmaxf(fmaxf(v0[dd],v1[dd]), fmaxf(v2[dd],v3[dd]));
        iMin[dd] = int(floorf((lo-P.boxLo[dd])/P.dx[dd]));
        iMax[dd] = int(floorf((hi-P.boxLo[dd])/P.dx[dd])) + 1;
        // the exact deposit clips the bbox window directly (no sub-samples)
        if (P.nSub>1 && !fExact) { iMin[dd]-=1; iMax[dd]+=1; }
        if (!P.periodic) { if (iMin[dd]<0) iMin[dd]=0; if (iMax[dd]>P.nGrid[dd]) iMax[dd]=P.nGrid[dd]; }
    }
    int nSub = P.nSub;
    unsigned nSamp = unsigned(nSub)*unsigned(nSub)*unsigned(nSub);

    // ===================== exact conservative deposit (--ps-exact-deposit) =====================
    // Mirrors the CPU exact path (and the Metal kernel): PASS 0 sums the per-cell weights,
    // PASS 1 renormalizes to the tet mass and deposits the exact moments. An all-empty window
    // falls through to the sampled path, whose empty window drives the centroid fallback.
    if (fExact)
    {
        float t1[3], t2[3], t3[3];
        for (int i=0;i<3;++i) { t1[i]=v1[i]-v0[i]; t2[i]=v2[i]-v0[i]; t3[i]=v3[i]-v0[i]; }
        if (d < 0.0f) { for (int i=0;i<3;++i) { float tt=t1[i]; t1[i]=t2[i]; t2[i]=tt; } }

        float sumW = 0.0f, shareFac = 0.0f;
        bool deposited = false;
        for (int pass=0; pass<2; ++pass)
        {
            if (pass==1)
            {
                if (!(sumW > 0.0f)) break;
                shareFac = m / sumW;
                deposited = true;
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
                if (l0<0) l0+=P.nGrid[0];   // sub-box may wrap a periodic axis (see ps_interpolation.cc)
                if (l1<0) l1+=P.nGrid[1];
                if (l2<0) l2+=P.nGrid[2];
                if (l0<0||l0>=P.subDims[0]||l1<0||l1>=P.subDims[1]||l2<0||l2>=P.subDims[2]) continue;
                unsigned flat = (unsigned(l0)*unsigned(P.subDims[1])+unsigned(l1))*unsigned(P.subDims[2])+unsigned(l2);

                ExPoly piece;
                exact_init_tet(piece, t1, t2, t3);
                for (int dd=0; dd<3; ++dd)
                {
                    float const clo = P.boxLo[dd] + float(raw[dd])*P.dx[dd] - v0[dd];
                    float const chi = clo + P.dx[dd];
                    exact_clip_plane(piece, dd,  1.0f, -clo);
                    exact_clip_plane(piece, dd, -1.0f,  chi);
                }
                if (piece.nv == 0) continue;
                float mo[10];
                if (pass==0) exact_reduce1(piece, mo);   // volume+first moments only: pass 0 discards the rest
                else         exact_reduce2(piece, mo);
                if (!(mo[0] > 0.0f)) continue;

                float w = mo[0];
                if (fLinear)
                {
                    w = d0*mo[0] + dG[0]*mo[1] + dG[1]*mo[2] + dG[2]*mo[3];
                    if (w < 0.0f) w = 0.0f;
                }
                if (pass==0) { sumW += w; continue; }

                float const massShare = w * shareFac;
                float const wmEx = fVolW ? mo[0] : massShare;
                unsigned long long const base = flat;
                atomicAdd(&massGrid[flat], massShare);
                float const invV = 1.0f / mo[0];
                float const cen[3] = { mo[1]*invV, mo[2]*invV, mo[3]*invV };
                float vb[3];
                if (fVel || fDisp)
                {
                    for (int j=0; j<3; ++j)
                    {
                        vb[j] = u0[j];
                        for (int i=0; i<3; ++i) vb[j] += vG[i][j]*cen[i];
                        if (fVel) atomicAdd(&momGrid[base*3ull + (unsigned long long)j], vb[j]*wmEx);
                    }
                    if (fDisp)
                    {
                        float cov[3][3];
                        cov[0][0] = mo[4]*invV - cen[0]*cen[0];
                        cov[0][1] = cov[1][0] = mo[5]*invV - cen[0]*cen[1];
                        cov[0][2] = cov[2][0] = mo[6]*invV - cen[0]*cen[2];
                        cov[1][1] = mo[7]*invV - cen[1]*cen[1];
                        cov[1][2] = cov[2][1] = mo[8]*invV - cen[1]*cen[2];
                        cov[2][2] = mo[9]*invV - cen[2]*cen[2];
                        unsigned long long c2 = 0;
                        for (int a=0; a<3; ++a)
                            for (int b=a; b<3; ++b)
                            {
                                float gcg = 0.0f;
                                for (int i=0; i<3; ++i)
                                    for (int k=0; k<3; ++k)
                                        gcg += vG[i][a]*cov[i][k]*vG[k][b];
                                atomicAdd(&m2Grid[base*6ull + c2++], massShare*(vb[a]*vb[b] + gcg));
                            }
                        if (fVolW)
                        {
                            for (int j=0; j<3; ++j) atomicAdd(&dvGrid[base*3ull + (unsigned long long)j], vb[j]*massShare);
                            atomicAdd(&dwGrid[flat], massShare);
                        }
                    }
                }
                if (fGrad)
                    for (int j=0; j<3; ++j)
                        for (int i=0; i<3; ++i)
                            atomicAdd(&gradGrid[base*9ull + (unsigned long long)(j*3+i)], vG[i][j]*wmEx);
                if (fVolW)    atomicAdd(&momwGrid[flat], wmEx);
                if (fCaustic) atomicOr(&caustGrid[flat], obits);
                atomicAdd(&svGrid[flat], mo[0]*invCellVol);
                atomicAdd(&strGrid[flat], 1u);
            }
        }
        if (deposited) return;
    }

    // Two identical passes over (cell, sample): pass 0 counts N (and, for the linear deposit,
    // sums the sample weights), pass 1 deposits m/N -- or m*w/sumW when fLinear, which totals
    // exactly m as well. The cell's wrap + sub-grid mapping is resolved BEFORE the sample loop
    // (like the CPU inSub guard), so N counts exactly the samples that pass 1 deposits -- no
    // mass is ever lost to invalid cells.
    unsigned N = 0;
    float sumW = 0.0f;
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
            if (l0<0) l0+=P.nGrid[0];   // sub-box may wrap a periodic axis (see ps_interpolation.cc)
            if (l1<0) l1+=P.nGrid[1];
            if (l2<0) l2+=P.nGrid[2];
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
                    if (pass==0)
                    {
                        N++;
                        if (fLinear) sumW += linearWeight(rel, d0, dG);
                    }
                    else
                    {
                        float w = share;
                        if (fLinear && sumW > 0.0f)
                            w = m * linearWeight(rel, d0, dG) / sumW;
                        // moment weight: equal V_eul share under fVolW (fLinear is rejected
                        // with fVolW at option parsing), else the mass share w
                        float wm = fVolW ? mw / float(N) : w;
                        float svS = fExact ? (tetVol*invCellVol)/float(N) : 0.0f;
                        depositSample(massGrid,momGrid,m2Grid,gradGrid,strGrid,momwGrid,caustGrid,svGrid,dvGrid,dwGrid,
                                      flat, rel, w, wm, obits, u0, vG, fVel, fDisp, fGrad, fVolW, fCaustic, fExact, svS);
                    }
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
            if (loc[dd]<0) loc[dd]+=P.nGrid[dd];             // sub-box may wrap a periodic axis
            if (loc[dd]<0||loc[dd]>=P.subDims[dd]) return;   // centroid outside this partition's box: drop
        }
        unsigned flat = (unsigned(loc[0])*unsigned(P.subDims[1])+unsigned(loc[1]))*unsigned(P.subDims[2])+unsigned(loc[2]);
        for (int dd=0; dd<3; ++dd) crel[dd] = cen[dd]-v0[dd];
        depositSample(massGrid,momGrid,m2Grid,gradGrid,strGrid,momwGrid,caustGrid,svGrid,dvGrid,dwGrid,
                      flat, crel, m, mw, obits, u0, vG, fVel, fDisp, fGrad, fVolW, fCaustic, fExact, tetVol*invCellVol);
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
                        std::vector<float>& dens,
                        const double boxLo[3], const double dx[3],
                        const size_t nGrid[3], const size_t subOrigin[3], const size_t subDims[3],
                        int nSub, bool periodic,
                        bool fVel, bool fDisp, bool fGrad, bool fLinear,
                        bool fVolW, bool fCaustic, bool fExact,
                        PSGpuGrids& out, std::string& err)
{
    std::lock_guard<std::mutex> lock(ctxMutex());   // serialize dispatches (single device)
    Ctx& c = ctx();
    if (!c.ready) { err = c.err; return false; }

    // only the requested moment grids are allocated, host- and device-side: the m2 (24 B/cell)
    // and grad (36 B/cell) grids dominate the deposit's footprint and most runs need neither
    size_t const nCell = subDims[0] * subDims[1] * subDims[2];
    size_t const momBytes  = fVel  ? nCell * 12 : 4;
    size_t const m2Bytes   = fDisp ? nCell * 24 : 4;
    size_t const gradBytes = fGrad ? nCell * 36 : 4;
    size_t const momwBytes  = fVolW    ? nCell * 4 : 4;
    size_t const caustBytes = fCaustic ? nCell * 4 : 4;
    size_t const svBytes    = fExact   ? nCell * 4 : 4;
    bool   const fDispOwn   = fVolW && fDisp;
    size_t const dvBytes    = fDispOwn ? nCell * 12 : 4;
    size_t const dwBytes    = fDispOwn ? nCell * 4  : 4;
    bool const needsVel = fVel || fDisp || fGrad;
    out.mass.assign(nCell, 0.f);
    out.mom.assign(fVel ? nCell * 3 : 0, 0.f);
    out.m2.assign(fDisp ? nCell * 6 : 0, 0.f);
    out.grad.assign(fGrad ? nCell * 9 : 0, 0.f);
    out.streams.assign(nCell, 0u);
    out.momw.assign(fVolW ? nCell : 0, 0.f);
    out.caustic.assign(fCaustic ? nCell : 0, 0u);
    out.streamvol.assign(fExact ? nCell : 0, 0.f);
    out.dispvel.assign(fDispOwn ? nCell * 3 : 0, 0.f);
    out.dispw.assign(fDispOwn ? nCell : 0, 0.f);
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
    P.fVel     = fVel  ? 1 : 0;
    P.fDisp    = fDisp ? 1 : 0;
    P.fGrad    = fGrad ? 1 : 0;
    P.fLinear  = (fLinear && !dens.empty()) ? 1 : 0;
    P.fVolW    = fVolW    ? 1 : 0;
    P.fCaustic = fCaustic ? 1 : 0;
    P.fExact   = fExact   ? 1 : 0;
    bool const useLin = P.fLinear != 0;

    size_t const nTetTotal = masses.size();

    // Outputs stay device-resident for the whole call; inputs stream in chunks so VRAM needs
    // are outputs + one chunk. Shrink the chunk until it fits (with slack for the runtime).
    size_t chunk = 2000000;
    if (const char* env = getenv("PS_GPU_CHUNK"))
    { long v = atol(env); if (v > 0) chunk = size_t(v); }
    if (chunk > nTetTotal) chunk = nTetTotal;
    size_t const outBytes = nCell * (4 + 4) + momBytes + m2Bytes + gradBytes + momwBytes + caustBytes + svBytes + dvBytes + dwBytes;
    size_t const slack = size_t(64) << 20;
    size_t freeB = 0, totalB = 0;
    if (gpuMemGetInfo(&freeB, &totalB) == gpuSuccess)
    {
        while (chunk > 50000 && outBytes + chunk*29*sizeof(float) + slack > freeB)
            chunk /= 2;
        if (outBytes + chunk*29*sizeof(float) + slack > freeB)
        {
            err = "insufficient GPU memory for the output grids (need > " +
                  std::to_string((outBytes+slack)>>20) + " MB free)";
            return false;
        }
    }

    float *dV=nullptr, *dU=nullptr, *dM=nullptr, *dD=nullptr, *dMass=nullptr, *dMom=nullptr, *dM2=nullptr, *dGrad=nullptr, *dMomW=nullptr;
    unsigned *dStr=nullptr, *dCaust=nullptr;
    float *dSV=nullptr, *dDV=nullptr, *dDW=nullptr;
    auto freeAll = [&]()
    {
        for (void* p : { (void*)dV,(void*)dU,(void*)dM,(void*)dD,(void*)dMass,(void*)dMom,(void*)dM2,(void*)dGrad,(void*)dStr,(void*)dMomW,(void*)dCaust,(void*)dSV,(void*)dDV,(void*)dDW })
            if (p) gpuFree(p);
    };

    bool ok = gpuCheck(gpuMalloc((void**)&dV,   chunk*12*sizeof(float)), "gpuMalloc verts",  err)
          &&  gpuCheck(gpuMalloc((void**)&dU,   needsVel ? chunk*12*sizeof(float) : 4), "gpuMalloc vels", err)
          &&  gpuCheck(gpuMalloc((void**)&dD,   useLin ? chunk*4*sizeof(float) : 4), "gpuMalloc dens", err)
          &&  gpuCheck(gpuMalloc((void**)&dM,   chunk*sizeof(float)),    "gpuMalloc masses", err)
          &&  gpuCheck(gpuMalloc((void**)&dMass, nCell*4),  "gpuMalloc mass grid", err)
          &&  gpuCheck(gpuMalloc((void**)&dMom,  momBytes),  "gpuMalloc mom grid",  err)
          &&  gpuCheck(gpuMalloc((void**)&dM2,   m2Bytes),   "gpuMalloc m2 grid",   err)
          &&  gpuCheck(gpuMalloc((void**)&dGrad, gradBytes), "gpuMalloc grad grid", err)
          &&  gpuCheck(gpuMalloc((void**)&dStr,  nCell*4),  "gpuMalloc streams",   err)
          &&  gpuCheck(gpuMalloc((void**)&dMomW,  momwBytes),  "gpuMalloc momw grid",   err)
          &&  gpuCheck(gpuMalloc((void**)&dCaust, caustBytes), "gpuMalloc caustic grid", err)
          &&  gpuCheck(gpuMemset(dMass, 0, nCell*4),  "gpuMemset mass", err)
          &&  gpuCheck(gpuMemset(dMom,  0, momBytes),  "gpuMemset mom",  err)
          &&  gpuCheck(gpuMemset(dM2,   0, m2Bytes),   "gpuMemset m2",   err)
          &&  gpuCheck(gpuMemset(dGrad, 0, gradBytes), "gpuMemset grad", err)
          &&  gpuCheck(gpuMemset(dStr,  0, nCell*4),  "gpuMemset streams", err)
          &&  gpuCheck(gpuMemset(dMomW,  0, momwBytes),  "gpuMemset momw",    err)
          &&  gpuCheck(gpuMalloc((void**)&dSV, svBytes), "gpuMalloc streamvol", err)
          &&  gpuCheck(gpuMemset(dCaust, 0, caustBytes), "gpuMemset caustic", err)
          &&  gpuCheck(gpuMalloc((void**)&dDV, dvBytes), "gpuMalloc dispvel", err)
          &&  gpuCheck(gpuMalloc((void**)&dDW, dwBytes), "gpuMalloc dispw", err)
          &&  gpuCheck(gpuMemset(dSV,    0, svBytes),    "gpuMemset streamvol", err)
          &&  gpuCheck(gpuMemset(dDV,    0, dvBytes),    "gpuMemset dispvel", err)
          &&  gpuCheck(gpuMemset(dDW,    0, dwBytes),    "gpuMemset dispw", err);

    int const tpb = 256;
    for (size_t start = 0; start < nTetTotal && ok; start += chunk)
    {
        unsigned const n = unsigned(std::min(chunk, nTetTotal - start));
        P.nTet = n;
        ok = gpuCheck(gpuMemcpy(dV, verts.data()  + start*12, size_t(n)*12*sizeof(float), gpuMemcpyHostToDevice), "copy verts",  err)
         &&  (!needsVel || gpuCheck(gpuMemcpy(dU, vels.data() + start*12, size_t(n)*12*sizeof(float), gpuMemcpyHostToDevice), "copy vels", err))
         &&  (!useLin   || gpuCheck(gpuMemcpy(dD, dens.data() + start*4,  size_t(n)*4*sizeof(float),  gpuMemcpyHostToDevice), "copy dens", err))
         &&  gpuCheck(gpuMemcpy(dM, masses.data() + start,    size_t(n)*sizeof(float),    gpuMemcpyHostToDevice), "copy masses", err);
        if (!ok) break;
        psDepositFieldsKernel<<< (n + tpb - 1)/tpb, tpb >>>(dV, dU, dM, dD, dMass, dMom, dM2, dGrad, dStr, dMomW, dCaust, dSV, dDV, dDW, P);
        ok = gpuCheck(gpuGetLastError(), "kernel launch", err)
         &&  gpuCheck(gpuDeviceSynchronize(), "kernel execution", err);
    }

    // inputs are consumed by this call (the streaming loop was their last consumer)
    std::vector<float>().swap(verts);
    std::vector<float>().swap(vels);
    std::vector<float>().swap(masses);
    std::vector<float>().swap(dens);

    if (ok)
        ok = gpuCheck(gpuMemcpy(out.mass.data(),    dMass, nCell*4,  gpuMemcpyDeviceToHost), "copy mass back",    err)
         &&  (!fVel  || gpuCheck(gpuMemcpy(out.mom.data(),  dMom,  nCell*12, gpuMemcpyDeviceToHost), "copy mom back",  err))
         &&  (!fDisp || gpuCheck(gpuMemcpy(out.m2.data(),   dM2,   nCell*24, gpuMemcpyDeviceToHost), "copy m2 back",   err))
         &&  (!fGrad || gpuCheck(gpuMemcpy(out.grad.data(), dGrad, nCell*36, gpuMemcpyDeviceToHost), "copy grad back", err))
         &&  gpuCheck(gpuMemcpy(out.streams.data(), dStr,  nCell*4,  gpuMemcpyDeviceToHost), "copy streams back", err)
         &&  (!fVolW    || gpuCheck(gpuMemcpy(out.momw.data(),    dMomW,  nCell*4, gpuMemcpyDeviceToHost), "copy momw back",    err))
         &&  (!fCaustic || gpuCheck(gpuMemcpy(out.caustic.data(), dCaust, nCell*4, gpuMemcpyDeviceToHost), "copy caustic back", err))
         &&  (!fExact   || gpuCheck(gpuMemcpy(out.streamvol.data(), dSV, nCell*4, gpuMemcpyDeviceToHost), "copy streamvol back", err))
         &&  (!fDispOwn || gpuCheck(gpuMemcpy(out.dispvel.data(), dDV, nCell*12, gpuMemcpyDeviceToHost), "copy dispvel back", err))
         &&  (!fDispOwn || gpuCheck(gpuMemcpy(out.dispw.data(),   dDW, nCell*4,  gpuMemcpyDeviceToHost), "copy dispw back", err));

    freeAll();
    return ok;
}
