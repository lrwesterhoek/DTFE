/* The standard-DTFE GPU deposit's exact sums: the units of its 64-bit fixed-point grids, shared by
   the two GPU hosts of the standard binary (dtfe_metal_host.cc, dtfe_gpu_cuda.cu). Host-side only, plain C++ (no
   backend includes); the fixed-point rule itself is the phase-space mass grid's (ps_deposit_params.h:
   psMassFixedScale picks the unit, psMassFixedMass converts a cell back). Why the grids are integer sums, and how
   the kernels keep them independent of the order their atomics land in: metal/dtfe_deposit.metal's header.
   Below batchFixedScales: the other host-side steps the two hosts share (the knobs, a batch's parameter block
   and item map, the timing line, the shared grid's checks). */
#ifndef DTFE_GPU_FIXED_H
#define DTFE_GPU_FIXED_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "gpu_host.h"            // DTFEGpuGrids
#include "ps_deposit_params.h"

// The grids are 64-bit fixed-point sums (the kernel's header says why): grid g holds integers in
// units of 1/scale[g], scale = 2^k, set per BATCH from a bound B of the sum of |share| over everything the batch
// can deposit -- no cell, holding part of that sum, can exceed it. psMassFixedScale (the PS mass grid's rule) puts
// 2B at 2^61..2^62 units, so B itself sits at 2^60..2^61: a factor 4 below the 2^63 of the signed words for the
// kernel's float rounding of a sample value (the inverse edge matrix of a tetrahedron near the degeneracy cut
// amplifies it; a few per cent at worst). B, per grid, sums V_tet * max|value| over the batch's tetrahedra: a
// sample, a clipped piece's centroid and the single-cell centroid all lie in the tetrahedron, where the linear
// interpolant is a convex combination of the vertex values as uploaded (the Monte-Carlo centring shift is in
// them already). The two gradients are per-tetrahedron constants: computed here in double from the kernel's own
// float edges and differences, with the degeneracy cut relaxed to half the kernel's (inverse3zero's 1e-6 of the
// row-length product), so a tetrahedron the float kernel keeps is never bounded as zero. The resolution this
// leaves: a 512^3 sub-domain of 1.5e7 cells puts a cell holding 1e-3 of the mean share at ~2^26 units -- 2^-26 of
// its own value, finer than the float (2^-24) the grid is handed over in. False (the batch keeps the float atomics) when a bound is not
// finite or has no usable power of two; an all-zero grid gets unit 1.
inline bool batchFixedScales(const std::vector<float>& verts, const std::vector<float>& dens, const std::vector<float>& vels,
                             const std::vector<float>& scal, const std::vector<float>& vols,
                             bool fDen, bool fVel, bool fGrad, bool fScal, bool fSGrad, float scale[5])
{
    size_t const nTet = vols.size();
    double B[5] = {0., 0., 0., 0., 0.};
    bool const needInv = fGrad || fSGrad;
    for (size_t t = 0; t < nTet; ++t)
    {
        double const V = std::fabs(double(vols[t]));
        if (fDen)
        {
            double m = 0.;
            for (int v = 0; v < 4; ++v) m = std::max(m, std::fabs(double(dens[t*4 + v])));
            B[0] += V * m;
        }
        if (fVel)
        {
            double m = 0.;
            for (int k = 0; k < 12; ++k) m = std::max(m, std::fabs(double(vels[t*12 + k])));
            B[1] += V * m;
        }
        if (fScal)
        {
            double m = 0.;
            for (int v = 0; v < 4; ++v) m = std::max(m, std::fabs(double(scal[t*4 + v])));
            B[3] += V * m;
        }
        if (!needInv) continue;
        // the kernel's edge matrix (float differences, as it forms them), its cofactors and determinant in double;
        // the gradient is cof^T df / det, so its largest component is max|sum_k cof[k][i] df[k]| / |det|. The cut
        // compares squares (no square roots: this loop runs over every tetrahedron of the batch)
        double A[3][3];
        for (int v = 0; v < 3; ++v)
            for (int d = 0; d < 3; ++d)
                A[v][d] = double(float(verts[t*12 + (v+1)*3 + d] - verts[t*12 + d]));
        double const C[3][3] = {   // C[i][k] = the inverse's [i][k] times det (inverse3zero's formulas)
            {  A[1][1]*A[2][2]-A[1][2]*A[2][1], -(A[0][1]*A[2][2]-A[0][2]*A[2][1]),  A[0][1]*A[1][2]-A[0][2]*A[1][1] },
            { -(A[1][0]*A[2][2]-A[1][2]*A[2][0]),  A[0][0]*A[2][2]-A[0][2]*A[2][0], -(A[0][0]*A[1][2]-A[0][2]*A[1][0]) },
            {  A[1][0]*A[2][1]-A[1][1]*A[2][0], -(A[0][0]*A[2][1]-A[0][1]*A[2][0]),  A[0][0]*A[1][1]-A[0][1]*A[1][0] } };
        double const det = A[0][0]*C[0][0] + A[0][1]*C[1][0] + A[0][2]*C[2][0];
        double rn2 = 1.;
        for (int i = 0; i < 3; ++i) rn2 *= A[i][0]*A[i][0] + A[i][1]*A[i][1] + A[i][2]*A[i][2];
        if (!(det * det > 0.25e-12 * rn2)) continue;      // |det| <= 0.5e-6 rn: the kernel zeroes this one's gradients too
        double const Vd = V / std::fabs(det);
        // max over i of |sum_k C[i][k] * df[k]| for the component differences df (the kernel's float ones)
        auto gradMax = [&](const float* f0, size_t stride, int comps) {
            double m = 0.;
            for (int j = 0; j < comps; ++j)
            {
                double df[3];
                for (int k = 0; k < 3; ++k) df[k] = double(float(f0[(k+1)*stride + j] - f0[j]));
                for (int i = 0; i < 3; ++i)
                    m = std::max(m, std::fabs(C[i][0]*df[0] + C[i][1]*df[1] + C[i][2]*df[2]));
            }
            return m;
        };
        if (fGrad)  B[2] += Vd * gradMax(&vels[t*12], 3, 3);
        if (fSGrad) B[4] += Vd * gradMax(&scal[t*4], 1, 1);
    }
    bool const on[5] = { fDen, fVel, fGrad, fScal, fSGrad };
    for (int g = 0; g < 5; ++g)
    {
        scale[g] = 1.f;
        if (!on[g] || B[g] == 0.) continue;
        if (!psMassFixedScale(2. * B[g], scale[g])) return false;
    }
    return true;
}


// ---------------------------------------------------------------- knobs
// Work items (default): every tetrahedron becomes max(1, ceil(cnt / itemSamples)) threads, so no
// thread loops over more than itemSamples samples -- the sample count per tetrahedron reaches
// maxNN = 10000 on fine grids, which makes the per-tet kernel's chunks wildly non-uniform and
// trips the GPU watchdog -- and the kernel's per-thread cell cache turns hundreds of per-sample
// atomics into one set per distinct cell (see the kernel header). DTFE_GPU_ITEMS=0 runs the
// per-tet kernel for A/B comparisons.
// DTFE_GPU_ITEM_SAMPLES=<n> sets the run length and DTFE_GPU_CACHE_K=2|4|8 the cell cache.
// Measured on 2.1M particles at 512^3 (the largest sub-domain, 2.0 G samples; the per-tet kernel
// takes 45 s with watchdog retries): K=2 at items 8/16/32/48/64 = 2.85/2.35/2.05/1.97/1.95 s (K=4 at
// 32: 2.05 s), density only 0.9 s, no atomics at all 1.8 s -- the kernel is compute/occupancy-bound
// at ~1.1 G samples/s, ~2.3 G for density.
struct DTFEGpuKnobs
{
    bool     useItems    = true;
    uint32_t itemSamples = 48;
    int      cacheK      = 2;
    bool     noWrite     = false;   // DTFE_GPU_DEBUG_NOWRITE: timing experiment, all the work, no atomics
    bool     floatGrids  = false;   // DTFE_METAL_FLOAT_GRIDS=1 / DTFE_GPU_FLOAT_GRIDS=1: float atomics (A/B)
};

// The knobs from the environment; itemsOk: the host can run the item kernels at all.
inline DTFEGpuKnobs dtfeGpuKnobs(bool itemsOk)
{
    DTFEGpuKnobs k;
    k.useItems = itemsOk;
    if (const char* env = std::getenv("DTFE_GPU_ITEMS")) k.useItems = k.useItems && std::atoi(env) != 0;
    if (const char* env = std::getenv("DTFE_GPU_ITEM_SAMPLES"))
    { long v = std::atol(env); if (v >= 4 && v <= 100000) k.itemSamples = uint32_t(v); }
    if (const char* env = std::getenv("DTFE_GPU_CACHE_K")) { int v = std::atoi(env); k.cacheK = (v == 4 || v == 8) ? v : 2; }
    k.noWrite = std::getenv("DTFE_GPU_DEBUG_NOWRITE") != nullptr;
    auto envOn = [](const char* n) { const char* e = std::getenv(n); return e && std::atoi(e) != 0; };
    k.floatGrids = envOn("DTFE_METAL_FLOAT_GRIDS") || envOn("DTFE_GPU_FLOAT_GRIDS");
    return k;
}


// ---------------------------------------------------------------- one sub-domain's batch
// The batch's parameter block, in either host's DepositParams (zeroed by the host; the fields named here are
// the same in both): the exact sums' units from this batch's values, before the upload consumes them -- the
// shared full grid (allowFixed false) and DTFE_METAL_FLOAT_GRIDS=1 / DTFE_GPU_FLOAT_GRIDS=1 keep the float
// atomics --, the grids' geometry, the field flags (all off under DTFE_GPU_DEBUG_NOWRITE) and the item length.
// DTFE_GPU_DEBUG_NOATOMIC: 1 = no atomics in the final reduction; 2 = also skip the gradient's reduction loop;
// 3 = also the gradient's eviction atomics (timing experiments only). Returns the exact kernel's item, a block
// of cells of the tetrahedron's window (DTFE_GPU_EXACT_CELLS, default 16), which is then P.itemSamples.
template <class Params>
inline uint32_t dtfeBatchParams(Params& P, DTFEGpuKnobs const& k, const std::vector<float>& verts,
                                const std::vector<float>& dens, const std::vector<float>& vels,
                                const std::vector<float>& scal, const std::vector<float>& vols, const double dx[3],
                                const size_t nGrid[3], const size_t subOff[3], const size_t fullGrid[3], bool fDen,
                                bool fVel, bool fGrad, bool fScal, bool fSGrad, bool exact, bool allowFixed)
{
    P.fixedSum = 0;
    for (int g = 0; g < 5; ++g) P.fxScale[g] = 1.f;
    if (allowFixed && !k.floatGrids && !k.noWrite)
        P.fixedSum = batchFixedScales(verts, dens, vels, scal, vols, fDen, fVel, fGrad, fScal, fSGrad, P.fxScale) ? 1u : 0u;
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
    if (k.noWrite) P.fDen = P.fVel = P.fGrad = P.fScal = P.fSGrad = 0;
    P.debugNoAtomic = std::getenv("DTFE_GPU_DEBUG_NOATOMIC") ? uint32_t(std::max(1, std::atoi(std::getenv("DTFE_GPU_DEBUG_NOATOMIC")))) : 0u;
    P.itemSamples = k.itemSamples;
    uint32_t exactCells = 16;
    if (exact)
    {
        if (const char* env = std::getenv("DTFE_GPU_EXACT_CELLS")) { long v = std::atol(env); if (v > 0) exactCells = uint32_t(v); }
        P.itemSamples = exactCells;
    }
    return exactCells;
}

// The fast path's flat indices arrive in the SUB-grid; the kernels deposit into the accumulated grid, so they
// are shifted to full-grid indices here (the sampled path adds subOff on the device).
inline void dtfeShiftFlats(std::vector<uint32_t> const& cnts, std::vector<uint32_t>& flats, const size_t nGrid[3],
                           const size_t subOff[3], const size_t fullGrid[3])
{
    bool const shifted = subOff[0] || subOff[1] || subOff[2] || fullGrid[0] != nGrid[0]
                      || fullGrid[1] != nGrid[1] || fullGrid[2] != nGrid[2];
    if (shifted)
        for (size_t t = 0; t < cnts.size(); ++t)
        {
            if (cnts[t] != 0) continue;
            size_t const f = flats[t];
            size_t const k = f % nGrid[2], j = (f / nGrid[2]) % nGrid[1], i = f / (nGrid[2] * nGrid[1]);
            flats[t] = uint32_t(((i + subOff[0]) * fullGrid[1] + (j + subOff[1])) * fullGrid[2] + (k + subOff[2]));
        }
}

// The item map (see the kernel's itemOf): itemStart[t] = first item of tetrahedron t (nTet+1 entries) and, per
// block of 2^blockShift items, the tetrahedron holding the block's first. A tetrahedron of n samples takes
// ceil(n / itemSamples) items, a single-cell one (n = 0) one; under 'exact' a sampled one takes
// ceil(window / exactCells) items of an UPPER bound of the kernel's cell window (its float floor can differ from
// this double one only within rounding of an integer, which the margin covers; surplus blocks idle). Without
// items only the counts are taken. Adds the batch's items, samples and single-cell tetrahedra to nItems /
// nSamples / nSingle; false (err set) beyond 2^32 items.
inline bool dtfeItemMap(std::vector<float> const& verts, std::vector<uint32_t> const& cnts, const double dx[3],
                        const size_t nGrid[3], bool useItems, bool exact, uint32_t itemSamples, uint32_t exactCells,
                        int blockShift, std::vector<uint32_t>& itemStart, std::vector<uint32_t>& blockTet,
                        size_t& nItems, size_t& nSamples, size_t& nSingle, std::string& err)
{
    size_t const nTet = cnts.size();
    if (!useItems)
    {
        for (size_t t = 0; t < nTet; ++t) { nSamples += cnts[t]; nSingle += cnts[t] == 0; }
        return true;
    }
    itemStart.resize(nTet + 1);
    for (size_t t = 0; t < nTet; ++t)
    {
        uint32_t const n = cnts[t];
        itemStart[t] = uint32_t(nItems);
        size_t items = n ? (n + itemSamples - 1) / itemSamples : 1u;
        if (exact && n)
        {
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
        nItems += items;
        nSamples += exact ? 0 : n;
        nSingle += n == 0;
        if (nItems > size_t(UINT32_MAX)) { err = "too many work items for one batch (> 2^32)"; return false; }
    }
    itemStart[nTet] = uint32_t(nItems);
    size_t const nBlocks = (nItems >> blockShift) + 2;
    blockTet.assign(nBlocks, uint32_t(nTet ? nTet - 1 : 0));
    for (size_t t = 0, blk = 0; t < nTet && blk < nBlocks; ++t)
        while (blk < nBlocks && (blk << blockShift) < size_t(itemStart[t + 1])) blockTet[blk++] = uint32_t(t);
    return true;
}

// The sample table as (x, y, z, 0) entries: one 16-byte load per sample
inline std::vector<float> dtfeSobol4(std::vector<float> const& sobol)
{
    std::vector<float> sobol4(sobol.size() / 3 * 4, 0.f);
    for (size_t i = 0; i < sobol.size() / 3; ++i)
        for (int d = 0; d < 3; ++d) sobol4[i * 4 + d] = sobol[i * 3 + d];
    return sobol4;
}

// The per-call path's output grids, zeroed at the requested fields' sizes (the rest EMPTY)
inline void dtfeGpuInitGrids(DTFEGpuGrids& out, size_t nCell, bool fDen, bool fVel, bool fGrad, bool fScal, bool fSGrad)
{
    out.density.assign(fDen ? nCell : 0, 0.f);
    out.velocity.assign(fVel ? nCell * 3 : 0, 0.f);
    out.grad.assign(fGrad ? nCell * 9 : 0, 0.f);
    out.scalar.assign(fScal ? nCell : 0, 0.f);
    out.scalar_grad.assign(fSGrad ? nCell * 3 : 0, 0.f);
}

// The batch's timing line (DTFE_METAL_TIMING / DTFE_GPU_TIMING), from either host's Batch and Stats; sinceBegin >= 0
// (the shared grid) adds when the batch was ready and how long it waited for the device.
template <class Batch, class Stats>
inline void dtfeLogBatch(const char* backend, const Batch& b, const Stats& st, double waitSec, double runSec,
                         double sinceBegin)
{
    std::fprintf(stderr, "DTFE %s timing: %s kernel, %zu tetrahedra (%zu single-cell), %zu items of <= %u %s, "
                         "%zu samples, table %u, %s sums, %d chunks, %d retries, %.2f s (%.0f M samples/s)%s",
            backend,
            b.exact ? "exact" : b.k.useItems ? (b.k.cacheK == 8 ? "item/K8" : b.k.cacheK == 2 ? "item/K2" : "item/K4") : "per-tet",
            b.nTet, b.nSingle, b.k.useItems ? b.nItems : size_t(0), b.P.itemSamples, b.exact ? "cells" : "samples", b.nSamples, b.P.tableEntries,
            b.P.fixedSum ? "fixed-point" : "float", st.chunks, st.retries, runSec, runSec > 0 ? b.nSamples / runSec / 1e6 : 0.0,
            sinceBegin >= 0 ? "" : "\n");
    if (sinceBegin >= 0)
        std::fprintf(stderr, "; shared grid: batch ready %.2f s after begin, waited %.2f s for the device\n",
                     sinceBegin, waitSec);
}


// ---------------------------------------------------------------- the shared accumulator
// OPT-IN (DTFE_GPU_SHARED=1): false (err set) otherwise. dtfe_metal_host.cc dtfeGpuSharedBegin has the measurement.
inline bool dtfeSharedOptedIn(std::string& err)
{
    if (!(std::getenv("DTFE_GPU_SHARED") && std::atoi(std::getenv("DTFE_GPU_SHARED")) != 0))
    {
        err = "off by default (DTFE_GPU_SHARED=1 turns it on; the per-sub-domain grids measured faster)";
        return false;
    }
    return true;
}

// The full grids' footprint against capGb, the host's cap (DTFE_GPU_SHARED_GB overrides): false (err set) over it.
inline bool dtfeSharedGridFits(size_t nCell, bool fDen, bool fVel, bool fGrad, bool fScal, bool fSGrad, double capGb,
                               std::string& err)
{
    if (const char* env = std::getenv("DTFE_GPU_SHARED_GB")) { double v = std::atof(env); if (v > 0) capGb = v; }
    double const needGb = double(nCell) * ((fDen ? 4 : 0) + (fVel ? 12 : 0) + (fGrad ? 36 : 0) + (fScal ? 4 : 0) + (fSGrad ? 12 : 0)) / 1e9;
    if (needGb > capGb)
    {
        char buf[200];
        std::snprintf(buf, sizeof(buf), "the shared GPU grid would take %.1f GB, over the %.1f GB cap", needGb, capGb);
        err = buf;
        return false;
    }
    return true;
}

#endif
