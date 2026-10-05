/* CUDA/HIP backend of the PS-DTFE GPU deposit (implements gpu_host.h). Compiled only in
   CUDA=1 (nvcc) or HIP=1 (hipcc, '-x hip') builds, which define PS_GPU for the callers.

   The kernels are a line-by-line port of metal/ps_deposit.metal -- depositFields (one thread
   per tetrahedron: the certified float inside test, the sampled two-pass deposit, the beyond-grid
   column counting, the centroid fallback and the one-thread exact deposit) and depositExactItems
   (the exact deposit split below the tetrahedron) with every helper, same float32 arithmetic,
   same deferral rule -- and the host mirrors ps_metal_host.cc: the same chunked dispatch
   controller, the same items pass + leftovers + deferrals, the same retry discipline. Where the
   Metal host reads its unified-memory buffers in place (the items' window bounds, the leftovers,
   the deferrals), this host keeps a host mirror of the masses and copies explicitly.

   The CPU deposit remains the double-precision reference; this backend carries the same
   parity contract as the Metal one (float rounding, atomic summation order). It has been
   compiled and its CPU fallback exercised on Linux, but NOT yet validated on NVIDIA/AMD
   hardware -- see metal/README.md ("CUDA / HIP"). */

#include <algorithm>
#include <chrono>
#include <cmath>
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
#include "ps_deposit_params.h"

namespace {

using DepositParams = PSDepositParams;   // the shared host struct (ps_deposit_params.h)

#include "gpu_exact_clip.cuh"   // V3, det3/inverse3, the r3d clip helpers (shared with dtfe_gpu_cuda.cu)

// ---- the per-tetrahedron record: PS_TET_STRIDE floats per tet, see ps_deposit_params.h /
// metal/ps_deposit.metal (ic0 as int bits, f0, the three edges, the centroid cell as uint bits)
#define TET_STRIDE PS_TET_STRIDE
#define U_F 5.9604645e-8f   // 2^-24, the float unit roundoff

// Band-certified classification of one sample (see metal/ps_deposit.metal for the derivation):
// +1 inside, -1 outside (both exact), 0 undecidable in float -> the caller defers the tet.
__device__ __forceinline__ int classifySample(const float C[3][3], const float M[3][3], float det, float sg,
                                              const float B[4], float Bdet, const float r[3],
                                              const float t[3], const DepositParams& P)
{
    float n[4];
    n[0] = C[0][0]*r[0] + C[0][1]*r[1] + C[0][2]*r[2];
    n[1] = C[1][0]*r[0] + C[1][1]*r[1] + C[1][2]*r[2];
    n[2] = C[2][0]*r[0] + C[2][1]*r[1] + C[2][2]*r[2];
    n[3] = det - (n[0] + n[1] + n[2]);
    bool sure = true;
    for (int k = 0; k < 4; ++k) {
        float const x = sg * n[k];
        if (x < -B[k]) return -1;
        if (x <= B[k]) sure = false;
    }
    if (sure) return 1;

    // stage 2: this sample's own bounds (7u|r_i| + rho_i = dx_i (u (6 + 12|t_i|) + 1e-9))
    float w[3];
    for (int i = 0; i < 3; ++i) w[i] = P.dx[i] * (U_F * (6.0f + 12.0f*fabsf(t[i])) + 1.0e-9f);
    float Bs[4];
    float nMag = 0.0f;
    for (int a = 0; a < 3; ++a) {
        Bs[a] = 2.0f * (M[a][0]*w[0] + M[a][1]*w[1] + M[a][2]*w[2]);
        nMag += M[a][0]*fabsf(r[0]) + M[a][1]*fabsf(r[1]) + M[a][2]*fabsf(r[2]);
    }
    Bs[3] = Bdet + Bs[0] + Bs[1] + Bs[2] + 2.0f * 3.0f * U_F * (fabsf(det) + nMag);
    sure = true;
    for (int k = 0; k < 4; ++k) {
        float const x = sg * n[k];
        if (x < -Bs[k]) return -1;
        if (x <= Bs[k]) sure = false;
    }
    return sure ? 1 : 0;
}

// ------------------------------------------------------------------------------------------------
// The grids (see metal/ps_deposit.metal FieldGrids) and the per-sample deposit.
// ------------------------------------------------------------------------------------------------
struct FieldGrids {
    float*    mass;
    float*    mom;     // nCell*3   [flat*3+j]
    float*    m2;      // nCell*6   [flat*6+c]
    float*    grad;    // nCell*9   [flat*9+j*3+i]
    unsigned* streams;
    float*    momw;    // nCell     moment-weight normalizer (fVolW; else dummy)
    unsigned* caustic; // nCell     orientation bits, atomic OR (fCaustic; else dummy)
    float*    sv;      // nCell     exact multiplicity sum(V_int)/V_cell (fExact; else dummy)
    float*    dispvel; // nCell*3   dispersion's own mass-weighted mean (fVolW+fDisp; else dummy)
    float*    dispw;   // nCell     its normalizer (fVolW+fDisp; else dummy)
    float*    scal;    // nCell     scalar moment sum(wm s) (fScal; else dummy)
    float*    sgrad;   // nCell*3   scalar-gradient moment (fSGrad; else dummy)
};

__device__ __forceinline__ void depositSample(const FieldGrids& G, unsigned flat,
                                              const float rel[3], float w, float wm, unsigned obits,
                                              const float u0[3], const float vG[3][3],
                                              float s0, const float sG[3],
                                              bool fVel, bool fDisp, bool fGrad, bool fVolW, bool fCaustic,
                                              bool fExact, float svShare, bool countStream, bool unresolved,
                                              bool fScal, bool fSGrad)
{
    // Moment-grid offsets are 64-bit BECAUSE flat*9 WOULD overflow a 32-bit uint above ~782^3.
    ull base = ull(flat);
    atomicAdd(&G.mass[flat], w);
    if (fVel || fDisp) {
        float vv[3];
        for (int j=0; j<3; ++j) {
            vv[j]=u0[j];
            for (int i=0;i<3;++i) vv[j]+=vG[i][j]*rel[i];
            if (fVel)
                atomicAdd(&G.mom[base*3ull+ull(j)], vv[j]*wm);
        }
        if (fDisp) {
            ull c=0;
            for (int i=0;i<3;++i)
                for (int j=i;j<3;++j)
                    atomicAdd(&G.m2[base*6ull + c++], w*vv[i]*vv[j]);
            if (fVolW) {
                for (int j=0;j<3;++j)
                    atomicAdd(&G.dispvel[base*3ull+ull(j)], vv[j]*w);
                atomicAdd(&G.dispw[flat], w);
            }
        }
    }
    if (fGrad)
        for (int j=0;j<3;++j)
            for (int i=0;i<3;++i)
                atomicAdd(&G.grad[base*9ull + ull(j*3+i)], vG[i][j]*wm);
    if (fScal) {   // the linear scalar profile at the sample, moment-weighted like the velocity
        float sv = s0;
        for (int i=0;i<3;++i) sv += sG[i]*rel[i];
        atomicAdd(&G.scal[flat], sv*wm);
    }
    if (fSGrad)
        for (int i=0;i<3;++i)
            atomicAdd(&G.sgrad[base*3ull + ull(i)], sG[i]*wm);
    if (fVolW)
        atomicAdd(&G.momw[flat], wm);
    if (fCaustic)
        atomicOr(&G.caustic[flat], obits);
    if (fExact)
        atomicAdd(&G.sv[flat], svShare);
    if (countStream)
        atomicAdd(&G.streams[flat], 1u);
    if (unresolved)
        atomicOr(&G.streams[flat], 0x80000000u);
}

// A RAW window index triple -> its flat sub-grid index (periodic wrap, then the partition
// sub-box, which may itself wrap a periodic axis). False when outside the grid or the sub-box.
// 0: outside the grid; 1: inside the grid but outside the sub-box (the SAMPLED deposit counts such a
// cell's samples under windowMode); 2: inside
__device__ __forceinline__ int window_flat(const int raw[3], const DepositParams& P, unsigned& flat) {
    int l[3];
    bool inSub = true;
    for (int dd=0; dd<3; ++dd) {
        int const wg = P.periodic ? wrapIdx(raw[dd], P.nGrid[dd]) : raw[dd];
        if (wg < 0 || wg >= P.nGrid[dd]) return 0;
        l[dd] = wg - P.subOrigin[dd];
        if (l[dd] < 0) l[dd] += P.nGrid[dd];
        if (l[dd] < 0 || l[dd] >= P.subDims[dd]) inSub = false;
    }
    if (!inSub) return 1;
    flat = (unsigned(l[0])*unsigned(P.subDims[1]) + unsigned(l[1]))*unsigned(P.subDims[2]) + unsigned(l[2]);
    return 2;
}

// --ps-window, exact deposit: restrict a tet's raw span [lo, hi) along one axis to the hull of the
// window's periodic images it meets (gpu_host.h psWindowHull, which the host's item bound uses --
// keep in sync). False when the span misses the window.
__device__ __forceinline__ bool window_hull(int& lo, int& hi, int n, bool periodic, int o, int m) {
    if (hi <= lo || m <= 0) return false;
    if (!periodic) { lo = max(lo, 0); hi = min(hi, n); if (hi <= lo) return false; }
    bool any = false;
    int hlo = 0, hhi = 0;
    int const kMax = periodic ? 2 : 0;
    for (int k = -kMax; k <= kMax; ++k) {
        int const a = max(lo, o + k*n), b = min(hi, o + m + k*n);
        if (b <= a) continue;
        if (!any) { hlo = a; hhi = b; any = true; }
        else { hlo = min(hlo, a); hhi = max(hhi, b); }
    }
    if (!any) return false;
    lo = hlo; hi = hhi;
    return true;
}

struct ExactTet {
    V3     t1, t2, t3;       // edges from vertex 0, positively oriented (folded tet: 1 and 2 swapped)
    TetPlanes faces;
    V3     hcell;            // half a cell
    int    ic0[3];
    float  f0[3];
    float  u0[3], vG[3][3];
    float  s0, sG[3];        // the linear scalar (fScal / fSGrad; else zero)
    float  d0, dG[3];
    unsigned obits;
    float  invCellVol;
    bool   straddle;
    float  wFull;
};

__device__ __forceinline__ float exact_full_weight(float vol, bool fLinear, float d0, const float dG[3],
                                                   V3 t1, V3 t2, V3 t3)
{
    if (!fLinear) return vol;
    V3 const c = scale3(add3(add3(t1, t2), t3), 0.25f);
    return vol * (d0 + dG[0]*c.x + dG[1]*c.y + dG[2]*c.z);
}

__device__ bool exact_cell_clip(const ExactTet& E, const int raw[3], const DepositParams& P,
                                bool full, bool fLinear, float mo[10], float& w, V3& cc)
{
    V3 const ccen = v3((float(raw[0] - E.ic0[0]) - E.f0[0] + 0.5f) * P.dx[0],
                       (float(raw[1] - E.ic0[1]) - E.f0[1] + 0.5f) * P.dx[1],
                       (float(raw[2] - E.ic0[2]) - E.f0[2] + 0.5f) * P.dx[2]);
    if (box_outside_tet(E.faces, ccen, E.hcell)) return false;

    // clip in the OVERLAP frame (gpu_exact_clip.cuh: origin at the centre of the tet bbox's overlap
    // with the cell, where the piece lies); the moments come out about that origin cc, which the
    // caller adds back to the centroid
    V3 const org = overlap_origin(ccen, E.hcell, E.t1, E.t2, E.t3);
    float const cc3[3] = { ccen.x, ccen.y, ccen.z }, hc[3] = { E.hcell.x, E.hcell.y, E.hcell.z };
    float const og[3] = { org.x, org.y, org.z };
    ExPoly piece;
    exact_init_tet4(piece, neg3(org), sub3(E.t1, org), sub3(E.t2, org), sub3(E.t3, org));
    for (int dd=0; dd<3; ++dd) {
        exact_clip_plane(piece, dd,  1.0f, og[dd] - (cc3[dd] - hc[dd]));   // x_d >= clo - org
        exact_clip_plane(piece, dd, -1.0f, (cc3[dd] + hc[dd]) - og[dd]);   // x_d <= chi - org
    }
    if (piece.nv == 0) return false;
    if (full) exact_reduce2(piece, mo);
    else      exact_reduce1(piece, mo);
    if (!(mo[0] > 0.0f)) return false;

    cc = org;
    w = mo[0];
    if (fLinear) {
        w = (E.d0 + E.dG[0]*org.x + E.dG[1]*org.y + E.dG[2]*org.z)*mo[0]
          + E.dG[0]*mo[1] + E.dG[1]*mo[2] + E.dG[2]*mo[3];
        if (w < 0.0f) w = 0.0f;
    }
    return true;
}

__device__ void exact_cell_deposit(const FieldGrids& G, unsigned flat, const float mo[10],
                                   float massShare, const ExactTet& E,
                                   bool fVel, bool fDisp, bool fGrad, bool fVolW, bool fCaustic,
                                   bool fScal, bool fSGrad, V3 cc)
{
    float const wmEx = fVolW ? mo[0] : massShare;
    ull const base = ull(flat);
    atomicAdd(&G.mass[flat], massShare);
    float const invV = 1.0f / mo[0];
    float const loc[3] = { mo[1]*invV, mo[2]*invV, mo[3]*invV };            // about the clip origin
    float const cen[3] = { cc.x + loc[0], cc.y + loc[1], cc.z + loc[2] };   // vertex-0 frame
    float vb[3];
    if (fVel || fDisp) {
        for (int j=0; j<3; ++j) {
            vb[j] = E.u0[j];
            for (int i=0; i<3; ++i) vb[j] += E.vG[i][j]*cen[i];
            if (fVel)
                atomicAdd(&G.mom[base*3ull+ull(j)], vb[j]*wmEx);
        }
        if (fDisp) {
            float cov[3][3];
            // (translation-invariant: taken about the clip origin, near the piece, where nothing cancels)
            cov[0][0] = mo[4]*invV - loc[0]*loc[0];
            cov[0][1] = cov[1][0] = mo[5]*invV - loc[0]*loc[1];
            cov[0][2] = cov[2][0] = mo[6]*invV - loc[0]*loc[2];
            cov[1][1] = mo[7]*invV - loc[1]*loc[1];
            cov[1][2] = cov[2][1] = mo[8]*invV - loc[1]*loc[2];
            cov[2][2] = mo[9]*invV - loc[2]*loc[2];
            ull c2 = 0;
            for (int a=0; a<3; ++a)
                for (int b=a; b<3; ++b) {
                    float gcg = 0.0f;
                    for (int i=0; i<3; ++i)
                        for (int k=0; k<3; ++k)
                            gcg += E.vG[i][a]*cov[i][k]*E.vG[k][b];
                    atomicAdd(&G.m2[base*6ull + c2++], massShare*(vb[a]*vb[b] + gcg));
                }
            if (fVolW) {
                for (int j=0;j<3;++j)
                    atomicAdd(&G.dispvel[base*3ull+ull(j)], vb[j]*massShare);
                atomicAdd(&G.dispw[flat], massShare);
            }
        }
    }
    if (fGrad)
        for (int j=0; j<3; ++j)
            for (int i=0; i<3; ++i)
                atomicAdd(&G.grad[base*9ull + ull(j*3+i)], E.vG[i][j]*wmEx);
    if (fScal) {   // exact mean of the linear scalar profile over the piece (the velocity's construction)
        float sb = E.s0;
        for (int i=0; i<3; ++i) sb += E.sG[i]*cen[i];
        atomicAdd(&G.scal[flat], sb*wmEx);
    }
    if (fSGrad)
        for (int i=0; i<3; ++i)
            atomicAdd(&G.sgrad[base*3ull + ull(i)], E.sG[i]*wmEx);
    if (fVolW)
        atomicAdd(&G.momw[flat], wmEx);
    if (fCaustic)
        atomicOr(&G.caustic[flat], E.obits);
    atomicAdd(&G.sv[flat], mo[0]*E.invCellVol);
    atomicAdd(&G.streams[flat], 1u);
}

// ------------------------------------------------------------------------------------------------
// Counting a straddling tet's samples BEYOND a non-periodic grid (pass 0 of the sampled deposit),
// one sample column (along z) at a time -- see metal/ps_deposit.metal for the derivation.
// ------------------------------------------------------------------------------------------------
__device__ __forceinline__ float sample_tc(int s, int nSub, float invNSub, int ic0d, float f0d) {
    int const k = (s >= 0) ? s / nSub : -((-s + nSub - 1) / nSub);
    int const sub = s - k * nSub;
    float const fr = (nSub == 1) ? 0.5f : (float(sub) + 0.5f) * invNSub;
    return (float(k - ic0d) - f0d) + fr;
}

struct BeyondColumn {
    float P[4], Q[4];
    int   ic0z; float f0z; float dxz; int nSub; float invNSub;
};

__device__ __forceinline__ float column_value(const BeyondColumn& c, int v, int s) {
    float const rz = sample_tc(s, c.nSub, c.invNSub, c.ic0z, c.f0z) * c.dxz;
    return c.P[v] + c.Q[v] * rz;
}

__device__ __forceinline__ int column_transition(const BeyondColumn& c, int v, float T, bool rise,
                                                 int lo, int hi, float guess) {
    int L = lo, H = hi;
    int const g = (guess > float(lo)) ? ((guess < float(hi)) ? int(ceilf(guess)) : hi) : lo;
    if (g > L && g - 1 < H) { if ((column_value(c, v, g - 1) > T) == rise) H = g - 1; else L = g; }
    if (g >= L && g < H)    { if ((column_value(c, v, g) > T) == rise) H = g; else L = g + 1; }
    while (L < H) { int const mid = L + (H - L) / 2; if ((column_value(c, v, mid) > T) == rise) H = mid; else L = mid + 1; }
    return L;
}

__device__ __forceinline__ void column_run(const BeyondColumn& c, int S0, int S1, const float T[4],
                                           int& a, int& b) {
    a = S0; b = S1;
    for (int v = 0; v < 4 && a < b; ++v) {
        float const guess = (c.Q[v] != 0.0f)
                          ? ((T[v] - c.P[v]) / c.Q[v] / c.dxz + float(c.ic0z) + c.f0z) * float(c.nSub) - 0.5f : 0.0f;
        if (c.Q[v] > 0.0f)      a = column_transition(c, v, T[v], true,  a, b, guess);
        else if (c.Q[v] < 0.0f) b = column_transition(c, v, T[v], false, a, b, guess);
        else if (!(c.P[v] > T[v])) b = a;
    }
}

__device__ bool count_beyond_grid(const float C[3][3], const float M[3][3], const float Ax[3][3],
                                  float detC, float sg, const float B[4], float Bdet,
                                  const int iMin[3], const int iMax[3],
                                  const int ic0[3], const float f0[3],
                                  int nSub, float invNSub, bool fLinear, float d0,
                                  const float dG[3], const DepositParams& P,
                                  unsigned& Nout, float& sumW)
{
    int const S0z = iMin[2] * nSub, S1z = iMax[2] * nSub, gridZ = P.nGrid[2] * nSub;
    float const Tin[4]  = { 2.0f*B[0], 2.0f*B[1], 2.0f*B[2], 2.0f*B[3] };
    float const Tpos[4] = { -2.0f*B[0], -2.0f*B[1], -2.0f*B[2], -2.0f*B[3] };
    BeyondColumn c;
    c.ic0z = ic0[2]; c.f0z = f0[2]; c.dxz = P.dx[2]; c.nSub = nSub; c.invNSub = invNSub;
    bool const zStraddle = S0z < 0 || S1z > gridZ;
    int const Sy0 = iMin[1] * nSub, Sy1 = iMax[1] * nSub, gridY = P.nGrid[1] * nSub;
    float const Vx[4] = { 0.0f, Ax[0][0], Ax[1][0], Ax[2][0] };
    float const Vy[4] = { 0.0f, Ax[0][1], Ax[1][1], Ax[2][1] };
    float const xPad = 1.0e-4f * P.dx[0];
    for (int sx = iMin[0] * nSub; sx < iMax[0] * nSub; ++sx) {
    int const kx = (sx >= 0) ? sx / nSub : -((-sx + nSub - 1) / nSub);
    bool const xInCol = kx >= 0 && kx < P.nGrid[0];
    float const relx = sample_tc(sx, nSub, invNSub, ic0[0], f0[0]) * P.dx[0];
    float yMin = GPU_INF, yMax = -GPU_INF;
    for (int i = 0; i < 3; ++i)
        for (int j = i + 1; j < 4; ++j) {
            float const xi = Vx[i], xj = Vx[j];
            if (relx < fminf(xi, xj) - xPad || relx > fmaxf(xi, xj) + xPad) continue;
            float y0 = Vy[i], y1 = Vy[j];
            if (xj != xi) {
                float const t = fminf(fmaxf((relx - xi) / (xj - xi), 0.0f), 1.0f);
                y0 = y1 = Vy[i] + t * (Vy[j] - Vy[i]);
            }
            yMin = fminf(yMin, fminf(y0, y1));
            yMax = fmaxf(yMax, fmaxf(y0, y1));
        }
    if (!(yMin <= yMax)) continue;
    int const pLo = max(Sy0, int(floorf((yMin / P.dx[1] + float(ic0[1]) + f0[1]) * float(nSub) - 0.5f)) - 2);
    int const pHi = min(Sy1, int(ceilf((yMax / P.dx[1] + float(ic0[1]) + f0[1]) * float(nSub) - 0.5f)) + 3);
    if (pLo >= pHi) continue;
    int yLo[2] = { pLo, 0 }, yHi[2] = { pHi, 0 }, nY = 1;
    if (xInCol && !zStraddle) {
        nY = 0;
        if (pLo < 0)     { yLo[nY] = pLo; yHi[nY] = min(pHi, 0); ++nY; }
        if (pHi > gridY) { yLo[nY] = max(pLo, gridY); yHi[nY] = pHi; ++nY; }
    }
    for (int iy = 0; iy < nY; ++iy)
    for (int sy = yLo[iy]; sy < yHi[iy]; ++sy) {
        int const ky = (sy >= 0) ? sy / nSub : -((-sy + nSub - 1) / nSub);
        bool const colIn = kx >= 0 && kx < P.nGrid[0] && ky >= 0 && ky < P.nGrid[1];
        int outLo[2], outHi[2], nOut = 0;
        if (!colIn) { outLo[0] = S0z; outHi[0] = S1z; nOut = 1; }
        else {
            if (S0z < 0)     { outLo[nOut] = S0z; outHi[nOut] = min(S1z, 0); ++nOut; }
            if (S1z > gridZ) { outLo[nOut] = max(S0z, gridZ); outHi[nOut] = S1z; ++nOut; }
        }
        if (nOut == 0) continue;
        float const tx = sample_tc(sx, nSub, invNSub, ic0[0], f0[0]);
        float const ty = sample_tc(sy, nSub, invNSub, ic0[1], f0[1]);
        float const rx = tx * P.dx[0], ry = ty * P.dx[1];
        int outLen = 0;
        for (int i = 0; i < nOut; ++i) outLen += outHi[i] - outLo[i];
        int aIn = 0, bIn = 0, aPos = 0, bPos = 0;
        bool bulk = outLen > 16;
        if (bulk) {
            for (int a = 0; a < 3; ++a) {
                c.P[a] = sg * (C[a][0]*rx + C[a][1]*ry);
                c.Q[a] = sg * C[a][2];
            }
            c.P[3] = sg * (detC - (C[0][0]*rx + C[0][1]*ry + C[1][0]*rx + C[1][1]*ry + C[2][0]*rx + C[2][1]*ry));
            c.Q[3] = -sg * (C[0][2] + C[1][2] + C[2][2]);
            column_run(c, S0z, S1z, Tin, aIn, bIn);
            column_run(c, S0z, S1z, Tpos, aPos, bPos);
            if (aIn >= bIn) { aIn = aPos; bIn = aPos; }
        }
        for (int i = 0; i < nOut; ++i) {
            int lo1 = outLo[i], hi1 = outHi[i], lo2 = 0, hi2 = 0;
            if (bulk) {
                int const c0 = max(aIn, outLo[i]), c1 = min(bIn, outHi[i]);
                if (c0 < c1) {
                    Nout += unsigned(c1 - c0);
                    if (fLinear) {
                        float const n = float(c1 - c0);
                        float const mz = ((float(c0) + float(c1)) * 0.5f * invNSub - float(ic0[2]) - f0[2]) * P.dx[2];
                        sumW += n * (d0 + dG[0]*rx + dG[1]*ry + dG[2]*mz);
                    }
                }
                lo1 = max(aPos, outLo[i]); hi1 = min(aIn, outHi[i]);
                lo2 = max(bIn, outLo[i]);  hi2 = min(bPos, outHi[i]);
            }
            for (int pass2 = 0; pass2 < 2; ++pass2) {
                int const u = pass2 == 0 ? lo1 : lo2, w = pass2 == 0 ? hi1 : hi2;
                for (int sz = u; sz < w; ++sz) {
                    float const tz = sample_tc(sz, nSub, invNSub, ic0[2], f0[2]);
                    float const t[3] = { tx, ty, tz };
                    float rel[3] = { rx, ry, tz * P.dx[2] };
                    int const cls = classifySample(C, M, detC, sg, B, Bdet, rel, t, P);
                    if (cls == 0) return false;
                    if (cls > 0) { Nout++; if (fLinear) sumW += linearWeight(rel, d0, dG); }
                }
            }
        }
    }
    }
    return true;
}

// ================================================================================================
// depositFields: one thread per tetrahedron (metal/ps_deposit.metal::depositFields).
// ================================================================================================
__global__ void __launch_bounds__(256)
depositFields(const float* __restrict__ verts, const float* __restrict__ vels, float* masses,
              float* massGrid, float* momGrid, float* m2Grid, float* gradGrid, unsigned* strGrid,
              const DepositParams P, const float* __restrict__ dens,
              float* momwGrid, unsigned* caustGrid, float* svGrid, float* dvGrid, float* dwGrid,
              const float* __restrict__ scal, float* scalGrid, float* sgGrid)
{
    unsigned const tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= P.nTet) return;

    float  m  = masses[tid];
    if (m <= 0.0f) return;

    const float* R = verts + ull(tid) * ull(TET_STRIDE);
    int   const ic0[3] = { __float_as_int(R[0]), __float_as_int(R[1]), __float_as_int(R[2]) };
    float const f0[3]  = { R[3], R[4], R[5] };
    float Ax[3][3];
    for (int e=0; e<3; ++e)
        for (int i=0; i<3; ++i)
            Ax[e][i] = R[6 + e*3 + i];
    unsigned const centroidFlat = __float_as_uint(R[15]);

    float d = det3(Ax);
    float avgEdge2 = 0.0f;
    for (int v=0; v<3; ++v) { float l2=0.0f; for (int i=0;i<3;++i) l2+=Ax[v][i]*Ax[v][i]; avgEdge2+=l2; }
    avgEdge2 /= 3.0f;
    float posInv[3][3];
    if (fabsf(d) < 1.0e-6f * avgEdge2 * sqrtf(avgEdge2) || !inverse3(Ax, posInv)) { masses[tid] = -m; return; }

    bool const fVel = P.fVel != 0, fDisp = P.fDisp != 0, fGrad = P.fGrad != 0;
    bool const needsVel = fVel || fDisp || fGrad;
    float u0[3] = { 0.0f, 0.0f, 0.0f };
    float vG[3][3] = { {0.0f,0.0f,0.0f}, {0.0f,0.0f,0.0f}, {0.0f,0.0f,0.0f} };
    if (needsVel) {
        for (int j=0;j<3;++j) u0[j] = vels[tid*12+j];
        float dV[3][3];
        for (int e=0;e<3;++e)
            for (int j=0;j<3;++j)
                dV[e][j] = vels[tid*12+(e+1)*3+j] - u0[j];
        for (int i=0;i<3;++i)
            for (int j=0;j<3;++j) {
                float s=0.0f; for (int k=0;k<3;++k) s+=posInv[i][k]*dV[k][j];
                vG[i][j]=s;
            }
    }

    // the linear scalar profile (its gradient is constant over the tet, like the velocity's)
    bool const fScal = P.fScal != 0, fSGrad = P.fSGrad != 0;
    float s0 = 0.0f, sG[3] = { 0.0f, 0.0f, 0.0f };
    if (fScal || fSGrad) {
        s0 = scal[tid*4+0];
        float ds[3];
        for (int e=0;e<3;++e) ds[e] = scal[tid*4+e+1] - s0;
        for (int i=0;i<3;++i) { float s=0.0f; for (int k=0;k<3;++k) s+=posInv[i][k]*ds[k]; sG[i]=s; }
    }

    FieldGrids G;
    G.mass = massGrid; G.mom = momGrid; G.m2 = m2Grid; G.grad = gradGrid; G.streams = strGrid;
    G.momw = momwGrid; G.caustic = caustGrid; G.sv = svGrid; G.dispvel = dvGrid; G.dispw = dwGrid;
    G.scal = scalGrid; G.sgrad = sgGrid;

    bool const fVolW = P.fVolW != 0, fCaustic = P.fCaustic != 0, fExact = P.fExact != 0;
    float const mw = fVolW ? fabsf(d)/6.0f : m;
    unsigned const obits = (fCaustic && P.fLinear == 0) ? unsigned(dens[tid*4+0]) : (d > 0.0f ? 1u : 2u);
    float const invCellVol = 1.0f / (P.dx[0]*P.dx[1]*P.dx[2]);
    float const tetVol = fabsf(d)/6.0f;

    bool const fLinear = P.fLinear != 0;
    float d0 = 0.0f, dG[3] = {0.0f, 0.0f, 0.0f};
    if (fLinear) {
        d0 = dens[tid*4+0];
        float dd0[3];
        for (int e=0;e<3;++e) dd0[e] = dens[tid*4+e+1] - d0;
        for (int i=0;i<3;++i) { float s=0.0f; for (int k=0;k<3;++k) s+=posInv[i][k]*dd0[k]; dG[i]=s; }
    }

    int iMin[3], iMax[3], iMinC[3], iMaxC[3];
    bool straddle = false;
    for (int dd=0; dd<3; ++dd) {
        float lo = 0.0f, hi = 0.0f;
        for (int e=0; e<3; ++e) { lo = fminf(lo, Ax[e][dd]); hi = fmaxf(hi, Ax[e][dd]); }
        iMin[dd] = ic0[dd] + int(floorf(f0[dd] + lo / P.dx[dd]));
        iMax[dd] = ic0[dd] + int(floorf(f0[dd] + hi / P.dx[dd])) + 1;
        if (!P.periodic && (iMin[dd] < 0 || iMax[dd] > P.nGrid[dd])) straddle = true;
        if (P.nSub>1 && !fExact) { iMin[dd]-=1; iMax[dd]+=1; }
        iMinC[dd] = iMin[dd]; iMaxC[dd] = iMax[dd];
        if (!P.periodic) {
            if (iMinC[dd] < 0)           iMinC[dd] = 0;
            if (iMaxC[dd] > P.nGrid[dd]) iMaxC[dd] = P.nGrid[dd];
            if (iMinC[dd] >= P.nGrid[dd] || iMaxC[dd] <= 0) return;
        }
    }
    int nSub = P.nSub;

    float C[3][3], M[3][3];
    {
        float const A00=Ax[0][0], A01=Ax[0][1], A02=Ax[0][2];
        float const A10=Ax[1][0], A11=Ax[1][1], A12=Ax[1][2];
        float const A20=Ax[2][0], A21=Ax[2][1], A22=Ax[2][2];
        C[0][0]=A11*A22-A12*A21; M[0][0]=fabsf(A11*A22)+fabsf(A12*A21);
        C[0][1]=A12*A20-A10*A22; M[0][1]=fabsf(A12*A20)+fabsf(A10*A22);
        C[0][2]=A10*A21-A11*A20; M[0][2]=fabsf(A10*A21)+fabsf(A11*A20);
        C[1][0]=A02*A21-A01*A22; M[1][0]=fabsf(A02*A21)+fabsf(A01*A22);
        C[1][1]=A00*A22-A02*A20; M[1][1]=fabsf(A00*A22)+fabsf(A02*A20);
        C[1][2]=A01*A20-A00*A21; M[1][2]=fabsf(A01*A20)+fabsf(A00*A21);
        C[2][0]=A01*A12-A02*A11; M[2][0]=fabsf(A01*A12)+fabsf(A02*A11);
        C[2][1]=A02*A10-A00*A12; M[2][1]=fabsf(A02*A10)+fabsf(A00*A12);
        C[2][2]=A00*A11-A01*A10; M[2][2]=fabsf(A00*A11)+fabsf(A01*A10);
    }
    float const detC = Ax[0][0]*C[0][0] + Ax[0][1]*C[0][1] + Ax[0][2]*C[0][2];
    float Rw[3], rho[3];
    for (int i=0; i<3; ++i) {
        float const T = fmaxf(fabsf(float(iMin[i] - ic0[i]) - f0[i]), fabsf(float(iMax[i] - ic0[i]) - f0[i]));
        Rw[i]  = T * P.dx[i];
        rho[i] = P.dx[i] * (U_F * (6.0f + 5.0f*T) + 1.0e-9f);
    }
    float B[4];
    float nMag = 0.0f;
    for (int a=0; a<3; ++a) {
        float b = 0.0f;
        for (int i=0; i<3; ++i) { b += M[a][i] * (7.0f*U_F*Rw[i] + rho[i]); nMag += M[a][i] * Rw[i]; }
        B[a] = 2.0f * b;
    }
    float const Bdet = 2.0f * 8.0f * U_F * (fabsf(Ax[0][0])*M[0][0] + fabsf(Ax[0][1])*M[0][1] + fabsf(Ax[0][2])*M[0][2]);
    B[3] = Bdet + B[0] + B[1] + B[2] + 2.0f * 3.0f * U_F * (fabsf(detC) + nMag);
    bool const orientSure = fabsf(detC) > Bdet;
    float const sg = detC > 0.0f ? 1.0f : -1.0f;
    float const invNSub = 1.0f / float(nSub);

    // ===================== exact conservative deposit (--ps-exact-deposit) =====================
    if (fExact) {
        ExactTet E;
        E.t1 = v3(Ax[0][0], Ax[0][1], Ax[0][2]);
        E.t2 = v3(Ax[1][0], Ax[1][1], Ax[1][2]);
        E.t3 = v3(Ax[2][0], Ax[2][1], Ax[2][2]);
        if (d < 0.0f) { V3 tt = E.t1; E.t1 = E.t2; E.t2 = tt; }
        E.faces = tet_planes(E.t1, E.t2, E.t3);
        E.hcell = v3(0.5f*P.dx[0], 0.5f*P.dx[1], 0.5f*P.dx[2]);
        for (int i=0; i<3; ++i) {
            E.ic0[i] = ic0[i];  E.f0[i] = f0[i];  E.u0[i] = u0[i];  E.dG[i] = dG[i];  E.sG[i] = sG[i];
            for (int j=0; j<3; ++j) E.vG[i][j] = vG[i][j];
        }
        E.d0 = d0;  E.s0 = s0;  E.obits = obits;  E.invCellVol = invCellVol;
        E.straddle = straddle;
        E.wFull = exact_full_weight(tetVol, fLinear, d0, dG, E.t1, E.t2, E.t3);

        // --ps-window: only the window's cells are clipped and the tet is normalized by its whole
        // weight (as a straddling tet is); an empty window then means "no volume inside the
        // window", not a degenerate tet -- nothing to deposit, no sampled fallback
        if (P.windowMode != 0)
            for (int dd=0; dd<3; ++dd)
                if (!window_hull(iMinC[dd], iMaxC[dd], P.nGrid[dd], P.periodic != 0, P.subOrigin[dd], P.subDims[dd])) return;
        float sumW = 0.0f;
        float shareFac = 0.0f;
        bool  deposited = false;
        for (int pass=0; pass<2; ++pass) {
            if (pass==1) {
                if (!(sumW > 0.0f)) { if (P.windowMode != 0) return; break; }
                shareFac = m / ((E.straddle || P.windowMode != 0) ? fmaxf(sumW, E.wFull) : sumW);
                deposited = true;
            }
            for (int gi=iMinC[0]; gi<iMaxC[0]; ++gi)
            for (int gj=iMinC[1]; gj<iMaxC[1]; ++gj)
            for (int gk=iMinC[2]; gk<iMaxC[2]; ++gk) {
                int const raw[3]={gi,gj,gk};
                unsigned flat;
                if (window_flat(raw, P, flat) != 2) continue;   // outside the sub-box / window
                float mo[10], w;
                V3 cc;
                if (!exact_cell_clip(E, raw, P, pass==1, fLinear, mo, w, cc)) continue;
                if (pass==0) { sumW += w; continue; }
                exact_cell_deposit(G, flat, mo, w * shareFac, E, fVel, fDisp, fGrad, fVolW, fCaustic, fScal, fSGrad, cc);
            }
        }
        if (deposited) return;
    }

    // ===================== the sampled two-pass deposit =====================
    if (!orientSure) { masses[tid] = -m; return; }
    unsigned N = 0, Nout = 0;
    float sumW = 0.0f;
    float share = 0.0f;
    for (int pass=0; pass<2; ++pass) {
        if (pass==1) {
            if (straddle && (N > 0u || centroidFlat != 0xFFFFFFFFu)
                && !count_beyond_grid(C, M, Ax, detC, sg, B, Bdet, iMin, iMax, ic0, f0, nSub, invNSub,
                                      fLinear, d0, dG, P, Nout, sumW)) { masses[tid] = -m; return; }
            if (N==0u) break;
            share = m / float(N + Nout);
        }
        for (int gi=iMinC[0]; gi<iMaxC[0]; ++gi)
        for (int gj=iMinC[1]; gj<iMaxC[1]; ++gj)
        for (int gk=iMinC[2]; gk<iMaxC[2]; ++gk) {
            int raw[3]={gi,gj,gk};
            int wg0=P.periodic?wrapIdx(gi,P.nGrid[0]):gi;
            int wg1=P.periodic?wrapIdx(gj,P.nGrid[1]):gj;
            int wg2=P.periodic?wrapIdx(gk,P.nGrid[2]):gk;
            bool const inGrid = !(wg0<0||wg0>=P.nGrid[0]||wg1<0||wg1>=P.nGrid[1]||wg2<0||wg2>=P.nGrid[2]);
            if (!inGrid) continue;
            unsigned flat = 0u;
            bool inSub = true;
            {
                int l0=wg0-P.subOrigin[0], l1=wg1-P.subOrigin[1], l2=wg2-P.subOrigin[2];
                if (l0<0) l0+=P.nGrid[0];
                if (l1<0) l1+=P.nGrid[1];
                if (l2<0) l2+=P.nGrid[2];
                if (l0<0||l0>=P.subDims[0]||l1<0||l1>=P.subDims[1]||l2<0||l2>=P.subDims[2]) inSub = false;
                else flat=(unsigned(l0)*unsigned(P.subDims[1])+unsigned(l1))*unsigned(P.subDims[2])+unsigned(l2);
            }
            // a WINDOW still counts the cell's samples in pass 0 (the full run's normalization), deposits none
            if (!inSub && !(P.windowMode != 0 && pass == 0)) continue;
            float const cb[3] = { float(raw[0] - ic0[0]) - f0[0],
                                  float(raw[1] - ic0[1]) - f0[1],
                                  float(raw[2] - ic0[2]) - f0[2] };
            for (int k2=0; k2<nSub; ++k2)
            for (int k1=0; k1<nSub; ++k1)
            for (int k0=0; k0<nSub; ++k0) {
                float const fr0 = (nSub==1) ? 0.5f : (float(k0) + 0.5f) * invNSub;
                float const fr1 = (nSub==1) ? 0.5f : (float(k1) + 0.5f) * invNSub;
                float const fr2 = (nSub==1) ? 0.5f : (float(k2) + 0.5f) * invNSub;
                float const tc[3] = { cb[0] + fr0, cb[1] + fr1, cb[2] + fr2 };
                float rel[3] = { tc[0] * P.dx[0], tc[1] * P.dx[1], tc[2] * P.dx[2] };
                int const cls = classifySample(C, M, detC, sg, B, Bdet, rel, tc, P);
                if (cls == 0 && pass == 0) { masses[tid] = -m; return; }
                if (cls > 0) {
                    if (pass==0) {
                        N++;
                        if (fLinear) sumW += linearWeight(rel, d0, dG);
                    } else {
                        float w = share;
                        if (fLinear && sumW > 0.0f)
                            w = m * linearWeight(rel, d0, dG) / sumW;
                        float wm = fVolW ? mw / float(N + Nout) : w;
                        float sv = fExact ? (tetVol*invCellVol)/float(N + Nout) : 0.0f;
                        depositSample(G, flat, rel, w, wm, obits, u0, vG, s0, sG, fVel, fDisp, fGrad, fVolW, fCaustic, fExact, sv, true, false, fScal, fSGrad);
                    }
                }
            }
        }
    }

    if (N==0u && Nout > 0u) return;
    if (N==0u) {   // centroid fallback
        if (centroidFlat == 0xFFFFFFFFu) return;
        float crel[3];
        for (int i=0; i<3; ++i) crel[i] = (Ax[0][i] + Ax[1][i] + Ax[2][i]) * 0.25f;
        depositSample(G, centroidFlat, crel, m, mw, obits, u0, vG, s0, sG, fVel, fDisp, fGrad, fVolW, fCaustic, fExact, tetVol*invCellVol, fExact, true, fScal, fSGrad);
    }
}

// ================================================================================================
// depositExactItems: the exact deposit with the work split BELOW the tetrahedron, one thread per
// (tet, block of IP.blockCells window cells); phase 0 sums the weights, phase 1 deposits.
// ================================================================================================
struct ExactItemParams { unsigned nItems; unsigned phase; unsigned blockCells; unsigned pad; };

__device__ bool exact_tet_setup(unsigned tet, const float* verts, const float* vels,
                                const float* dens, const float* scal, const DepositParams& P,
                                ExactTet& E, int iMin[3], int iMax[3])
{
    const float* R = verts + ull(tet) * ull(TET_STRIDE);
    float Ax[3][3];
    for (int e=0; e<3; ++e)
        for (int i=0; i<3; ++i)
            Ax[e][i] = R[6 + e*3 + i];
    for (int i=0; i<3; ++i) { E.ic0[i] = __float_as_int(R[i]); E.f0[i] = R[3+i]; }

    float const d = det3(Ax);
    float avgEdge2 = 0.0f;
    for (int v=0; v<3; ++v) { float l2=0.0f; for (int i=0;i<3;++i) l2+=Ax[v][i]*Ax[v][i]; avgEdge2+=l2; }
    avgEdge2 /= 3.0f;
    float posInv[3][3];
    if (fabsf(d) < 1.0e-6f * avgEdge2 * sqrtf(avgEdge2) || !inverse3(Ax, posInv)) return false;

    for (int i=0; i<3; ++i) { E.u0[i] = 0.0f; for (int j=0; j<3; ++j) E.vG[i][j] = 0.0f; }
    if (P.fVel != 0 || P.fDisp != 0 || P.fGrad != 0) {
        ull const v0 = ull(tet) * 12ull;
        for (int j=0;j<3;++j) E.u0[j] = vels[v0+ull(j)];
        float dV[3][3];
        for (int e=0;e<3;++e)
            for (int j=0;j<3;++j)
                dV[e][j] = vels[v0+ull((e+1)*3+j)] - E.u0[j];
        for (int i=0;i<3;++i)
            for (int j=0;j<3;++j) {
                float s=0.0f; for (int k=0;k<3;++k) s+=posInv[i][k]*dV[k][j];
                E.vG[i][j]=s;
            }
    }
    E.s0 = 0.0f;
    for (int i=0; i<3; ++i) E.sG[i] = 0.0f;
    if (P.fScal != 0 || P.fSGrad != 0) {
        E.s0 = scal[ull(tet)*4ull];
        float ds[3];
        for (int e=0;e<3;++e) ds[e] = scal[ull(tet)*4ull+ull(e+1)] - E.s0;
        for (int i=0;i<3;++i) { float s=0.0f; for (int k=0;k<3;++k) s+=posInv[i][k]*ds[k]; E.sG[i]=s; }
    }
    E.obits = (P.fCaustic != 0 && P.fLinear == 0) ? unsigned(dens[ull(tet)*4ull]) : (d > 0.0f ? 1u : 2u);
    E.invCellVol = 1.0f / (P.dx[0]*P.dx[1]*P.dx[2]);
    E.d0 = 0.0f;
    for (int i=0; i<3; ++i) E.dG[i] = 0.0f;
    if (P.fLinear != 0) {
        E.d0 = dens[ull(tet)*4ull];
        float dd0[3];
        for (int e=0;e<3;++e) dd0[e] = dens[ull(tet)*4ull+ull(e+1)] - E.d0;
        for (int i=0;i<3;++i) { float s=0.0f; for (int k=0;k<3;++k) s+=posInv[i][k]*dd0[k]; E.dG[i]=s; }
    }

    E.straddle = false;
    for (int dd=0; dd<3; ++dd) {
        float lo = 0.0f, hi = 0.0f;
        for (int e=0; e<3; ++e) { lo = fminf(lo, Ax[e][dd]); hi = fmaxf(hi, Ax[e][dd]); }
        iMin[dd] = E.ic0[dd] + int(floorf(E.f0[dd] + lo / P.dx[dd]));
        iMax[dd] = E.ic0[dd] + int(floorf(E.f0[dd] + hi / P.dx[dd])) + 1;
        if (!P.periodic) {
            if (iMin[dd] < 0)           { iMin[dd] = 0;           E.straddle = true; }
            if (iMax[dd] > P.nGrid[dd]) { iMax[dd] = P.nGrid[dd]; E.straddle = true; }
        }
    }

    // --ps-window: only the window's cells (the hull of its images along each axis) are items
    if (P.windowMode != 0)
        for (int dd=0; dd<3; ++dd)
            if (!window_hull(iMin[dd], iMax[dd], P.nGrid[dd], P.periodic != 0, P.subOrigin[dd], P.subDims[dd])) { iMax[dd] = iMin[dd]; }

    E.t1 = v3(Ax[0][0], Ax[0][1], Ax[0][2]);
    E.t2 = v3(Ax[1][0], Ax[1][1], Ax[1][2]);
    E.t3 = v3(Ax[2][0], Ax[2][1], Ax[2][2]);
    if (d < 0.0f) { V3 tt = E.t1; E.t1 = E.t2; E.t2 = tt; }
    E.faces = tet_planes(E.t1, E.t2, E.t3);
    E.hcell = v3(0.5f*P.dx[0], 0.5f*P.dx[1], 0.5f*P.dx[2]);
    E.wFull = exact_full_weight(fabsf(d)/6.0f, P.fLinear != 0, E.d0, E.dG, E.t1, E.t2, E.t3);
    return true;
}

__global__ void __launch_bounds__(256)
depositExactItems(const float* __restrict__ verts, const float* __restrict__ vels, const float* __restrict__ masses,
                  float* massGrid, float* momGrid, float* m2Grid, float* gradGrid, unsigned* strGrid,
                  const DepositParams P, const float* __restrict__ dens,
                  float* momwGrid, unsigned* caustGrid, float* svGrid, float* dvGrid, float* dwGrid,
                  const uint2* __restrict__ items, float* sumW, const ExactItemParams IP,
                  const float* __restrict__ scal, float* scalGrid, float* sgGrid)
{
    unsigned const iid = blockIdx.x * blockDim.x + threadIdx.x;
    if (iid >= IP.nItems) return;
    uint2 const it = items[iid];
    unsigned const tet = it.x;
    float const m = masses[tet];
    if (m <= 0.0f) return;
    bool const deposit = IP.phase == 1u;
    float s = 0.0f;
    if (deposit) {
        s = sumW[tet];               // written by phase 0's kernel, complete before this launch
        if (!(s > 0.0f)) return;
    }

    ExactTet E;
    int iMin[3], iMax[3];
    if (!exact_tet_setup(tet, verts, vels, dens, scal, P, E, iMin, iMax)) return;
    float const shareFac = deposit ? m / ((E.straddle || P.windowMode != 0) ? fmaxf(s, E.wFull) : s) : 0.0f;
    ull const n1 = ull(max(iMax[1] - iMin[1], 0));
    ull const n2 = ull(max(iMax[2] - iMin[2], 0));
    ull const n12 = n1 * n2;
    ull const nW = ull(max(iMax[0] - iMin[0], 0)) * n12;
    ull w = ull(it.y) * ull(IP.blockCells);
    if (w >= nW) return;
    ull const wEnd = (nW - w > ull(IP.blockCells)) ? w + ull(IP.blockCells) : nW;
    int a0 = int(w / n12);
    ull const rem = w - ull(a0) * n12;
    int a1 = int(rem / n2);
    int a2 = int(rem - ull(a1) * n2);

    bool const fVel = P.fVel != 0, fDisp = P.fDisp != 0, fGrad = P.fGrad != 0;
    bool const fVolW = P.fVolW != 0, fCaustic = P.fCaustic != 0, fLinear = P.fLinear != 0;
    bool const fScal = P.fScal != 0, fSGrad = P.fSGrad != 0;
    FieldGrids G;
    G.mass = massGrid; G.mom = momGrid; G.m2 = m2Grid; G.grad = gradGrid; G.streams = strGrid;
    G.momw = momwGrid; G.caustic = caustGrid; G.sv = svGrid; G.dispvel = dvGrid; G.dispw = dwGrid;
    G.scal = scalGrid; G.sgrad = sgGrid;
    float acc = 0.0f;
    for (; w < wEnd; ++w) {
        int const raw[3] = { iMin[0] + a0, iMin[1] + a1, iMin[2] + a2 };
        if (ull(++a2) == n2) { a2 = 0; if (ull(++a1) == n1) { a1 = 0; ++a0; } }
        unsigned flat;
        if (window_flat(raw, P, flat) != 2) continue;   // outside the sub-box / window
        float mo[10], wt;
        V3 cc;
        if (!exact_cell_clip(E, raw, P, deposit, fLinear, mo, wt, cc)) continue;
        if (!deposit) acc += wt;
        else exact_cell_deposit(G, flat, mo, wt * shareFac, E, fVel, fDisp, fGrad, fVolW, fCaustic, fScal, fSGrad, cc);
    }
    if (!deposit && acc > 0.0f)
        atomicAdd(&sumW[tet], acc);
}

// gathers n records of 'stride' floats: dst[k] = src[idx[k]] (the compact leftovers)
__global__ void gatherRecords(const float* __restrict__ src, const unsigned* __restrict__ idx,
                              unsigned n, unsigned stride, float* dst)
{
    unsigned const k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const float* s = src + ull(idx[k]) * ull(stride);
    float* d = dst + ull(k) * ull(stride);
    for (unsigned c = 0; c < stride; ++c) d[c] = s[c];
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

// a device allocation that is zeroed; null (and 'err' set) on failure
float* devAlloc(size_t bytes, const char* what, std::string& err)
{
    void* p = nullptr;
    if (!gpuCheck(gpuMalloc(&p, bytes ? bytes : 4), what, err)) return nullptr;
    if (!gpuCheck(gpuMemset(p, 0, bytes ? bytes : 4), what, err)) { gpuFree(p); return nullptr; }
    return static_cast<float*>(p);
}

// uploads a host array (a 4-byte zeroed dummy when empty); null on failure
float* devUpload(const void* src, size_t bytes, const char* what, std::string& err)
{
    float* p = devAlloc(bytes, what, err);
    if (!p || !bytes) return p;
    if (!gpuCheck(gpuMemcpy(p, src, bytes, gpuMemcpyHostToDevice), what, err)) { gpuFree(p); return nullptr; }
    return p;
}

inline unsigned blocksFor(size_t n, int tpb) { return unsigned((n + size_t(tpb) - 1) / size_t(tpb)); }

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
                        std::vector<float>& scal,
                        const double boxLo[3], const double dx[3],
                        const size_t nGrid[3], const size_t subOrigin[3], const size_t subDims[3],
                        int nSub, bool periodic,
                        bool fVel, bool fDisp, bool fGrad, bool fLinear,
                        bool fVolW, bool fCaustic, bool fExact, bool fScal, bool fSGrad, bool windowMode,
                        PSGpuGrids& out, std::vector<uint32_t>& deferred, std::string& err)
{
    std::lock_guard<std::mutex> lock(ctxMutex());   // serialize dispatches (one device)
    Ctx& c = ctx();
    if (!c.ready) { err = c.err; return false; }

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
    size_t const scalBytes  = fScal  ? nCell * 4  : 4;
    size_t const sgBytes    = fSGrad ? nCell * 12 : 4;
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
    out.scal.assign(fScal ? nCell : 0, 0.f);
    out.sgrad.assign(fSGrad ? nCell * 3 : 0, 0.f);
    deferred.clear();
    if (masses.empty()) return true;
    if (verts.size() != masses.size() * size_t(PS_TET_STRIDE))
    {
        err = "tet record size mismatch (verts must hold PS_TET_STRIDE floats per tet)";
        return false;
    }

    DepositParams P;
    std::memset(&P, 0, sizeof(P));
    for (int d = 0; d < 3; ++d)
    {
        P.boxLo[d]     = float(boxLo[d]);
        P.dx[d]        = float(dx[d]);
        P.nGrid[d]     = int32_t(nGrid[d]);
        P.subOrigin[d] = int32_t(subOrigin[d]);
        P.subDims[d]   = int32_t(subDims[d]);
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
    P.nTet     = uint32_t(masses.size());
    P.fScal    = (fScal  && !scal.empty()) ? 1 : 0;
    P.fSGrad   = (fSGrad && !scal.empty()) ? 1 : 0;
    P.windowMode = windowMode ? 1 : 0;
    size_t const nTetTotal = masses.size();

    // PS_GPU_TIMING=1 (or PS_METAL_TIMING=1): one line per kernel run
    bool const timing = (getenv("PS_GPU_TIMING") && atoi(getenv("PS_GPU_TIMING")) != 0)
                     || (getenv("PS_METAL_TIMING") && atoi(getenv("PS_METAL_TIMING")) != 0);

    // --ps-exact-deposit items (see ps_metal_host.cc): the per-tet window bound comes from the
    // records, read here from the HOST copy before it is uploaded and freed
    bool useItems = fExact;
    if (const char* env = getenv("PS_EXACT_ITEMS")) useItems = useItems && atoi(env) != 0;
    uint32_t blockCells = 16;
    if (const char* env = getenv("PS_EXACT_ITEM_CELLS")) { long v = atol(env); if (v > 0) blockCells = uint32_t(v); }
    std::vector<uint32_t> items;      // (tet, block) pairs
    size_t nItems = 0;
    if (useItems)
    {
        std::vector<uint64_t> bound(nTetTotal);
        double total = 0.;
        for (size_t t = 0; t < nTetTotal; ++t)
        {
            float const* R = verts.data() + t * PS_TET_STRIDE;
            uint64_t nw = 1;
            for (int dd = 0; dd < 3 && nw; ++dd)
            {
                int32_t ic0; std::memcpy(&ic0, &R[dd], sizeof ic0);
                double lo = 0., hi = 0.;
                for (int e = 0; e < 3; ++e) { double const a = R[6 + e*3 + dd]; lo = std::min(lo, a); hi = std::max(hi, a); }
                double const f0 = R[3 + dd], dxd = double(P.dx[dd]);
                double const vlo = f0 + lo / dxd, vhi = f0 + hi / dxd;
                double const flo = std::floor(vlo - (1.e-3 + 1.e-5 * std::fabs(vlo)));
                double const fhi = std::floor(vhi + (1.e-3 + 1.e-5 * std::fabs(vhi)));
                if (!std::isfinite(flo) || !std::isfinite(fhi) || fhi - flo > 1.e7) { nw = 0; break; }
                int64_t iLo = int64_t(ic0) + int64_t(flo);
                int64_t iHi = int64_t(ic0) + int64_t(fhi) + 1;
                if (!periodic) { iLo = std::max<int64_t>(iLo, 0); iHi = std::min<int64_t>(iHi, int64_t(nGrid[dd])); }
                if (windowMode)
                {   // the kernel's items cover the window's hull only (window_hull): bound that
                    long hl = 0, hh = 0;
                    if (!psWindowHull(long(iLo), long(iHi), long(nGrid[dd]), periodic, long(P.subOrigin[dd]), long(P.subDims[dd]), hl, hh)) { nw = 0; break; }
                    iLo = hl; iHi = hh;
                }
                nw = iHi > iLo ? nw * uint64_t(iHi - iLo) : 0;
            }
            bound[t] = nw;
            total += double(nw);
        }
        double const maxItems = 32.e6;
        if (total / blockCells > maxItems) blockCells = uint32_t(std::ceil(total / maxItems));
        for (size_t t = 0; t < nTetTotal; ++t)
            nItems += (bound[t] + blockCells - 1) / blockCells;
        if (timing)
            fprintf(stderr, "PS-DTFE %s: exact items: %zu tets, %zu items of %u cells, window bound %.3g cells\n",
                    GPU_BACKEND_STRING, nTetTotal, nItems, blockCells, total);
        items.reserve(2 * nItems);
        for (size_t t = 0; t < nTetTotal; ++t)
        {
            uint64_t const nb = (bound[t] + blockCells - 1) / blockCells;
            for (uint64_t k = 0; k < nb; ++k) { items.push_back(uint32_t(t)); items.push_back(uint32_t(std::min<uint64_t>(k, 0xFFFFFFFFu))); }
        }
    }

    // Upload the per-tet arrays and free each host array IMMEDIATELY (at most one double-held at
    // a time). The masses keep a host mirror (4 B per tet): the leftovers and the deferrals are
    // read from it, in place of the Metal host's unified-memory reads.
    bool const haveVels = !vels.empty();
    bool const haveDens = !dens.empty();
    bool const haveScal = !scal.empty();
    std::vector<float> hostMass;
    hostMass.swap(masses);
    float* dV = devUpload(verts.data(), verts.size() * sizeof(float), "upload tet records", err);
    std::vector<float>().swap(verts);
    float* dU = dV ? devUpload(vels.data(), vels.size() * sizeof(float), "upload velocities", err) : nullptr;
    std::vector<float>().swap(vels);
    float* dM = dU ? devUpload(hostMass.data(), hostMass.size() * sizeof(float), "upload masses", err) : nullptr;
    float* dD = dM ? devUpload(dens.data(), dens.size() * sizeof(float), "upload densities", err) : nullptr;
    std::vector<float>().swap(dens);
    float* dSc = dD ? devUpload(scal.data(), scal.size() * sizeof(float), "upload scalars", err) : nullptr;
    std::vector<float>().swap(scal);
    float* dMass = dSc ? devAlloc(nCell * 4, "mass grid", err) : nullptr;
    float* dMom  = dMass ? devAlloc(momBytes, "moment grid", err) : nullptr;
    float* dM2   = dMom ? devAlloc(m2Bytes, "second-moment grid", err) : nullptr;
    float* dGrad = dM2 ? devAlloc(gradBytes, "gradient grid", err) : nullptr;
    float* dStr  = dGrad ? devAlloc(nCell * 4, "stream grid", err) : nullptr;
    float* dMomW = dStr ? devAlloc(momwBytes, "moment-weight grid", err) : nullptr;
    float* dCaust = dMomW ? devAlloc(caustBytes, "caustic grid", err) : nullptr;
    float* dSV   = dCaust ? devAlloc(svBytes, "stream-volume grid", err) : nullptr;
    float* dDV   = dSV ? devAlloc(dvBytes, "dispersion-mean grid", err) : nullptr;
    float* dDW   = dDV ? devAlloc(dwBytes, "dispersion-weight grid", err) : nullptr;
    float* dScal = dDW ? devAlloc(scalBytes, "scalar grid", err) : nullptr;
    float* dSG   = dScal ? devAlloc(sgBytes, "scalar-gradient grid", err) : nullptr;
    float* dItems = nullptr;
    float* dSumW  = nullptr;
    if (dSG && useItems)
    {
        dItems = devUpload(items.data(), items.size() * sizeof(uint32_t), "upload exact items", err);
        dSumW  = dItems ? devAlloc(std::max<size_t>(nTetTotal, 1) * sizeof(float), "per-tet weight totals", err) : nullptr;
        if (!dItems || !dSumW)
        {   // too big for the device: the one-thread-per-tet kernel still works
            if (dItems) gpuFree(dItems);
            if (dSumW)  gpuFree(dSumW);
            dItems = dSumW = nullptr;
            useItems = false;
            err.clear();
        }
        std::vector<uint32_t>().swap(items);
    }
    auto freeAll = [&]()
    {
        for (float* p : { dV, dU, dM, dD, dSc, dMass, dMom, dM2, dGrad, dStr, dMomW, dCaust, dSV, dDV, dDW, dScal, dSG, dItems, dSumW })
            if (p) gpuFree(p);
    };
    if (!dSG)
    {
        if (err.empty()) err = GPU_BACKEND_STRING " allocation failed (out of device memory?)";
        freeAll();
        return false;
    }

    // Dispatch in CHUNKS, one short kernel each, serialized. The chunk controller is the one the
    // Metal host uses (ps_metal_host.cc): it bounds the kernel's duration -- a display-attached
    // Linux GPU kills kernels after a few seconds -- and adapts the chunk toward targetSec. A killed
    // kernel may have PARTIALLY deposited its atomics, so a failed chunk is never retried alone:
    // each retry re-zeros the grids and redoes the whole deposit with shorter kernels. (A kernel
    // timeout is a STICKY error in CUDA: the retries then fail too and the caller's CPU fallback
    // takes over for this and every later partition of the process.) The idle gap between
    // kernels, which the Metal host needs for the window server, is zero here.
    // PS_GPU_CHUNK=<n> (or PS_METAL_CHUNK) overrides the starting chunk size of the per-tet kernel.
    size_t const MIN_CHUNK = fExact ? 50 : 2000;
    size_t const MAX_CHUNK = 500000;
    double targetSec = 0.25;
    useconds_t gapUs = 0;
    size_t baseChunk = fExact ? 500 : 25000;
    if (const char* env = getenv("PS_GPU_CHUNK"))   { long v = atol(env); if (v > 0) baseChunk = size_t(v); }
    if (const char* env = getenv("PS_METAL_CHUNK")) { long v = atol(env); if (v > 0) baseChunk = size_t(v); }
    std::string lastErr;
    int const tpb = 256;

    auto runChunked = [&](const char* name, size_t nThreads, size_t firstChunk, size_t minChunk, auto&& launch) -> bool
    {
        size_t chunk = std::max(firstChunk, minChunk);
        size_t start = 0, nBuf = 0;
        double busy = 0.;
        auto const tRun = std::chrono::steady_clock::now();
        while (start < nThreads)
        {
            uint32_t const n = uint32_t(std::min(chunk, nThreads - start));
            auto t0 = std::chrono::steady_clock::now();
            launch(start, n);
            if (!gpuCheck(gpuGetLastError(), "kernel launch", lastErr) || !gpuCheck(gpuDeviceSynchronize(), name, lastErr))
                return false;
            double const el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            start += n;
            ++nBuf;
            busy += el;
            if (el > 1e-4)
            {
                double scale = targetSec / el;
                if (scale < 0.5) scale = 0.5;
                if (scale > 4.0) scale = 4.0;
                chunk = std::min(std::max(size_t(double(chunk) * scale), minChunk), MAX_CHUNK);
            }
            if (gapUs) usleep(gapUs);
        }
        if (timing)
            fprintf(stderr, "PS-DTFE %s: %-17s %10zu threads  %6zu kernels  %8.3f s busy  %8.3f s wall\n",
                    GPU_BACKEND_STRING, name, nThreads, nBuf, busy,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - tRun).count());
        return true;
    };

    // depositFields over the tets [0,nT) of the given records (per-tet buffers at the chunk's offset)
    auto runPerTet = [&](const float* vB, const float* uB, float* mB, const float* dB, const float* sB, size_t nT) -> bool
    {
        return runChunked("depositFields", nT, baseChunk, MIN_CHUNK, [&](size_t start, uint32_t n)
        {
            DepositParams Pc = P;
            Pc.nTet = n;
            GPU_LAUNCH(depositFields, blocksFor(n, tpb), tpb, 0,
                vB + start * PS_TET_STRIDE, haveVels ? uB + start * 12 : uB, mB + start,
                dMass, dMom, dM2, dGrad, reinterpret_cast<unsigned*>(dStr), Pc,
                haveDens ? dB + start * 4 : dB,
                dMomW, reinterpret_cast<unsigned*>(dCaust), dSV, dDV, dDW,
                haveScal ? sB + start * 4 : sB, dScal, dSG);
        });
    };

    auto runItems = [&](uint32_t phase) -> bool
    {
        return runChunked("depositExactItems", nItems, 8192, 1024, [&](size_t start, uint32_t n)
        {
            ExactItemParams const IP{ n, phase, blockCells, 0u };
            GPU_LAUNCH(depositExactItems, blocksFor(n, tpb), tpb, 0,
                dV, dU, dM, dMass, dMom, dM2, dGrad, reinterpret_cast<unsigned*>(dStr), P, dD,
                dMomW, reinterpret_cast<unsigned*>(dCaust), dSV, dDV, dDW,
                reinterpret_cast<const uint2*>(dItems) + start, dSumW, IP, dSc, dScal, dSG);
        });
    };

    // The tets the items pass leaves over (sumW still 0) run through depositFields; their
    // deferrals are written back into the masses (host mirror AND device copy).
    auto runLeftovers = [&]() -> bool
    {
        std::vector<float> sw(nTetTotal);
        if (!gpuCheck(gpuMemcpy(sw.data(), dSumW, nTetTotal * sizeof(float), gpuMemcpyDeviceToHost), "read weight totals", lastErr))
            return false;
        std::vector<uint32_t> left;
        for (size_t t = 0; t < nTetTotal; ++t)
            if (hostMass[t] > 0.f && !(sw[t] > 0.f)) left.push_back(uint32_t(t));
        if (left.empty()) return true;
        size_t const nL = left.size();
        bool const compact = nL <= nTetTotal / 16;
        float *lV = dV, *lU = dU, *lD = dD, *lS = dSc, *lM = nullptr, *dLeft = nullptr;
        bool ok = true;
        if (compact)
        {
            dLeft = devUpload(left.data(), nL * sizeof(uint32_t), "upload leftover indices", lastErr);
            auto gather = [&](const float* src, unsigned stride) -> float* {
                float* b = devAlloc(nL * stride * sizeof(float), "leftover records", lastErr);
                if (!b) return b;
                GPU_LAUNCH(gatherRecords, blocksFor(nL, tpb), tpb, 0, src, reinterpret_cast<const unsigned*>(dLeft), unsigned(nL), stride, b);
                if (!gpuCheck(gpuGetLastError(), "gather launch", lastErr) || !gpuCheck(gpuDeviceSynchronize(), "gather", lastErr))
                { gpuFree(b); return nullptr; }
                return b;
            };
            ok = dLeft != nullptr;
            if (ok) { lV = gather(dV, PS_TET_STRIDE); ok = lV != nullptr; }
            if (ok) { lU = haveVels ? gather(dU, 12) : devAlloc(4, "dummy", lastErr); ok = lU != nullptr; }
            if (ok) { lM = gather(dM, 1); ok = lM != nullptr; }
            if (ok) { lD = haveDens ? gather(dD, 4) : devAlloc(4, "dummy", lastErr); ok = lD != nullptr; }
            if (ok) { lS = haveScal ? gather(dSc, 4) : devAlloc(4, "dummy", lastErr); ok = lS != nullptr; }
        }
        else
        {
            std::vector<float> lm(nTetTotal, 0.f);
            for (uint32_t t : left) lm[t] = hostMass[t];
            lM = devUpload(lm.data(), nTetTotal * sizeof(float), "leftover mass mask", lastErr);
            ok = lM != nullptr;
        }
        if (!ok && lastErr.empty()) lastErr = GPU_BACKEND_STRING " allocation failed (exact-deposit leftovers)";
        if (ok) ok = runPerTet(lV, lU, lM, lD, lS, compact ? nL : nTetTotal);
        if (ok)
        {
            std::vector<float> lm(compact ? nL : nTetTotal);
            ok = gpuCheck(gpuMemcpy(lm.data(), lM, lm.size() * sizeof(float), gpuMemcpyDeviceToHost), "read leftover masses", lastErr);
            if (ok)
            {
                for (size_t k = 0; k < nL; ++k)
                {
                    float const v = lm[compact ? k : size_t(left[k])];
                    if (v < 0.f) hostMass[left[k]] = v;
                }
                ok = gpuCheck(gpuMemcpy(dM, hostMass.data(), nTetTotal * sizeof(float), gpuMemcpyHostToDevice), "write back masses", lastErr);
            }
        }
        if (lM) gpuFree(lM);
        if (dLeft) gpuFree(dLeft);
        if (compact) for (float* b : {lV, lU, lD, lS}) if (b) gpuFree(b);
        return ok;
    };

    int const maxAttempts = 3;
    bool success = false;
    for (int attempt = 0; attempt < maxAttempts && !success; ++attempt)
    {
        bool zeroed = gpuCheck(gpuMemset(dMass, 0, nCell * 4), "zero grids", lastErr)
                   && gpuCheck(gpuMemset(dMom,  0, momBytes), "zero grids", lastErr)
                   && gpuCheck(gpuMemset(dM2,   0, m2Bytes), "zero grids", lastErr)
                   && gpuCheck(gpuMemset(dGrad, 0, gradBytes), "zero grids", lastErr)
                   && gpuCheck(gpuMemset(dStr,  0, nCell * 4), "zero grids", lastErr)
                   && gpuCheck(gpuMemset(dMomW, 0, momwBytes), "zero grids", lastErr)
                   && gpuCheck(gpuMemset(dCaust, 0, caustBytes), "zero grids", lastErr)
                   && gpuCheck(gpuMemset(dSV,   0, svBytes), "zero grids", lastErr)
                   && gpuCheck(gpuMemset(dDV,   0, dvBytes), "zero grids", lastErr)
                   && gpuCheck(gpuMemset(dDW,   0, dwBytes), "zero grids", lastErr)
                   && gpuCheck(gpuMemset(dScal, 0, scalBytes), "zero grids", lastErr)
                   && gpuCheck(gpuMemset(dSG,   0, sgBytes), "zero grids", lastErr);
        if (!zeroed) break;

        if (useItems)
            success = gpuCheck(gpuMemset(dSumW, 0, nTetTotal * sizeof(float)), "zero weight totals", lastErr)
                   && runItems(0) && runItems(1) && runLeftovers();
        else
            success = runPerTet(dV, dU, dM, dD, dSc, nTetTotal);

        if (!success)
        {
            targetSec = std::max(targetSec / 4.0, 0.02);
            gapUs     = std::min(std::max(gapUs * 3, useconds_t(8000)), useconds_t(50000));
            baseChunk = std::max(baseChunk / 2, MIN_CHUNK);
            if (attempt + 1 < maxAttempts)
            {
                fprintf(stderr, "PS-DTFE %s: %s -- retrying the partition deposit from scratch "
                                "(attempt %d/%d, target %.0f ms kernels, %.0f ms gaps)\n",
                        GPU_BACKEND_STRING, lastErr.c_str(), attempt + 2, maxAttempts,
                        targetSec*1e3, gapUs/1e3);
                sleep(2);
            }
        }
    }
    if (!success)
    {
        err = lastErr;
        freeAll();
        return false;
    }

    // deferred tets: the kernel negated their masses (per-tet path: on the device; leftovers:
    // written back into both copies)
    bool ok = gpuCheck(gpuMemcpy(hostMass.data(), dM, nTetTotal * sizeof(float), gpuMemcpyDeviceToHost), "read masses back", err);
    for (size_t t = 0; ok && t < nTetTotal; ++t)
        if (hostMass[t] < 0.f) deferred.push_back(uint32_t(t));

    ok = ok && gpuCheck(gpuMemcpy(out.mass.data(),    dMass, nCell * 4,  gpuMemcpyDeviceToHost), "copy mass back", err)
            && (!fVel  || gpuCheck(gpuMemcpy(out.mom.data(),  dMom,  nCell * 12, gpuMemcpyDeviceToHost), "copy mom back", err))
            && (!fDisp || gpuCheck(gpuMemcpy(out.m2.data(),   dM2,   nCell * 24, gpuMemcpyDeviceToHost), "copy m2 back", err))
            && (!fGrad || gpuCheck(gpuMemcpy(out.grad.data(), dGrad, nCell * 36, gpuMemcpyDeviceToHost), "copy grad back", err))
            && gpuCheck(gpuMemcpy(out.streams.data(), dStr,  nCell * 4,  gpuMemcpyDeviceToHost), "copy streams back", err)
            && (!fVolW    || gpuCheck(gpuMemcpy(out.momw.data(),    dMomW,  nCell * 4,  gpuMemcpyDeviceToHost), "copy momw back", err))
            && (!fCaustic || gpuCheck(gpuMemcpy(out.caustic.data(), dCaust, nCell * 4,  gpuMemcpyDeviceToHost), "copy caustic back", err))
            && (!fExact   || gpuCheck(gpuMemcpy(out.streamvol.data(), dSV,  nCell * 4,  gpuMemcpyDeviceToHost), "copy streamvol back", err))
            && (!fDispOwn || gpuCheck(gpuMemcpy(out.dispvel.data(), dDV,  nCell * 12, gpuMemcpyDeviceToHost), "copy dispvel back", err))
            && (!fDispOwn || gpuCheck(gpuMemcpy(out.dispw.data(),   dDW,  nCell * 4,  gpuMemcpyDeviceToHost), "copy dispw back", err))
            && (!fScal    || gpuCheck(gpuMemcpy(out.scal.data(),    dScal, nCell * 4,  gpuMemcpyDeviceToHost), "copy scalar back", err))
            && (!fSGrad   || gpuCheck(gpuMemcpy(out.sgrad.data(),   dSG,   nCell * 12, gpuMemcpyDeviceToHost), "copy scalar gradient back", err));
    freeAll();
    return ok;
}
