/* Metal backend of the PS-DTFE GPU deposit (implements gpu_host.h). Owns the metal-cpp
   implementation symbols for the whole binary; compiled only when METAL=1 (which defines
   PS_GPU for the callers). The kernel source is embedded at build time (o_ps/ps_deposit_msl.h,
   generated from metal/ps_deposit.metal) and runtime-compiled once per process; the pipeline
   state is cached. */

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <deque>
#include <mutex>
#include <unistd.h>

#include "gpu_host.h"
#include "ps_deposit_params.h"
#include "ps_deposit_msl.h"   // generated: static const char PS_DEPOSIT_MSL[]
#include "../message.h"       // ANSI colour helpers only (retry warning below)

namespace {

using DepositParams = PSDepositParams;   // shared host struct (ps_deposit_params.h)

struct Ctx
{
    MTL::Device*               dev  = nullptr;
    MTL::CommandQueue*         q    = nullptr;
    MTL::ComputePipelineState* pso  = nullptr;   // depositFields: one thread per tet
    MTL::ComputePipelineState* psoItems = nullptr;   // depositExactItems: one per (tet, cell block)
    std::string                err;
    bool                       ready = false;
};

std::mutex& ctxMutex() { static std::mutex m; return m; }

Ctx& ctx()
{
    static Ctx c;
    static bool tried = false;
    if (tried) return c;
    tried = true;

    c.dev = MTL::CreateSystemDefaultDevice();
    if (!c.dev) { c.err = "no Metal device"; return c; }
    c.q = c.dev->newCommandQueue();

    NS::Error* nsErr = nullptr;
    MTL::Library* lib = c.dev->newLibrary(
        NS::String::string(PS_DEPOSIT_MSL, NS::UTF8StringEncoding),
        (MTL::CompileOptions*)nullptr, &nsErr);
    if (!lib)
    {
        c.err = std::string("kernel compile failed: ")
              + (nsErr ? nsErr->localizedDescription()->utf8String() : "unknown");
        return c;
    }
    for (auto [name, dst] : { std::pair<const char*, MTL::ComputePipelineState**>{"depositFields", &c.pso},
                              std::pair<const char*, MTL::ComputePipelineState**>{"depositExactItems", &c.psoItems} })
    {
        MTL::Function* fn = lib->newFunction(NS::String::string(name, NS::UTF8StringEncoding));
        if (!fn) { c.err = std::string("kernel '") + name + "' not found"; return c; }
        *dst = c.dev->newComputePipelineState(fn, &nsErr);
        if (!*dst)
        {
            c.err = std::string("pipeline state failed: ")
                  + (nsErr ? nsErr->localizedDescription()->utf8String() : "unknown");
            return c;
        }
    }
    c.ready = true;
    return c;
}

} // namespace


std::string gpuBackendName() { return "Metal"; }

std::string gpuDeviceName()
{
    std::lock_guard<std::mutex> lock(ctxMutex());
    Ctx& c = ctx();
    return c.dev ? std::string(c.dev->name()->utf8String()) : std::string();
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
                          PSGpuGrids& out, std::vector<uint32_t>& deferred, std::string& err)
{
    std::lock_guard<std::mutex> lock(ctxMutex());   // serialize dispatches (single queue)
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
    bool   const fDispOwn   = fVolW && fDisp;   // dispersion keeps its own mass-weighted mean+normalizer
    size_t const dvBytes    = fDispOwn ? nCell * 12 : 4;
    size_t const dwBytes    = fDispOwn ? nCell * 4  : 4;
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
    deferred.clear();
    if (masses.empty()) return true;    // nothing to deposit (empty partition)
    if (verts.size() != masses.size() * size_t(PS_TET_STRIDE))
    {
        err = "tet record size mismatch (verts must hold PS_TET_STRIDE floats per tet)";
        return false;
    }

    DepositParams P{};
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
    P.fLinear  = (fLinear && not dens.empty()) ? 1 : 0;
    P.fVolW    = fVolW    ? 1 : 0;
    P.fCaustic = fCaustic ? 1 : 0;
    P.fExact   = fExact   ? 1 : 0;
    P.nTet     = uint32_t(masses.size());

    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

    // newBuffer(ptr,...) copies the host array into the MTL buffer; free each host array
    // IMMEDIATELY so at most one array is double-held at a time (instead of all three, a
    // multi-GB difference per partition at production scale). On failure the CPU fallback
    // re-deposits from the triangulation, never from these arrays.
    size_t const nTetTotal = masses.size();
    auto zeroBuf = [&](size_t bytes) {
        MTL::Buffer* b = c.dev->newBuffer(bytes, MTL::ResourceStorageModeShared);
        if (b) std::memset(b->contents(), 0, bytes);
        return b;
    };
    MTL::Buffer* bV = c.dev->newBuffer(verts.data(),  verts.size()  * sizeof(float), MTL::ResourceStorageModeShared);
    std::vector<float>().swap(verts);
    // vels is empty for density-only runs (kernel never reads it then): bind a dummy. Whether it
    // is real matters at bind time: the per-chunk offset below must NOT be applied to the 4-byte
    // dummy, or every chunk after the first binds past its end -- undefined behaviour that the
    // kernel gets away with only because it never dereferences the pointer in that case.
    bool const haveVels = not vels.empty();
    MTL::Buffer* bU = vels.empty() ? zeroBuf(4)
                    : c.dev->newBuffer(vels.data(), vels.size() * sizeof(float), MTL::ResourceStorageModeShared);
    std::vector<float>().swap(vels);
    MTL::Buffer* bM = c.dev->newBuffer(masses.data(), masses.size() * sizeof(float), MTL::ResourceStorageModeShared);
    std::vector<float>().swap(masses);
    // per-tet vertex densities (--ps-linear-deposit); dummy when the uniform deposit runs
    // 'dens' carries either the per-tet vertex densities (fLinear) or, when fCaustic runs without
    // fLinear, the per-tet caustic mask in slot 0. Either way the buffer is real and its per-chunk
    // offset must be applied -- keyed on the vector being non-empty, not on fLinear.
    bool const haveDens = not dens.empty();
    MTL::Buffer* bD = (haveDens)
                    ? c.dev->newBuffer(dens.data(), dens.size() * sizeof(float), MTL::ResourceStorageModeShared)
                    : zeroBuf(4);
    std::vector<float>().swap(dens);
    MTL::Buffer* bP = c.dev->newBuffer(&P, sizeof(P), MTL::ResourceStorageModeShared);
    MTL::Buffer* bMass = zeroBuf(nCell * 4);
    MTL::Buffer* bMom  = zeroBuf(momBytes);
    MTL::Buffer* bM2   = zeroBuf(m2Bytes);
    MTL::Buffer* bGrad = zeroBuf(gradBytes);
    MTL::Buffer* bStr  = zeroBuf(nCell * 4);
    MTL::Buffer* bMomW  = zeroBuf(momwBytes);
    MTL::Buffer* bCaust = zeroBuf(caustBytes);
    MTL::Buffer* bSV    = zeroBuf(svBytes);
    MTL::Buffer* bDV    = zeroBuf(dvBytes);
    MTL::Buffer* bDW    = zeroBuf(dwBytes);
    if (!bV || !bU || !bM || !bD || !bP || !bMass || !bMom || !bM2 || !bGrad || !bStr || !bMomW || !bCaust || !bSV || !bDV || !bDW)
    {
        err = "Metal buffer allocation failed (out of GPU-visible memory?)";
        for (MTL::Buffer* b : {bV,bU,bM,bD,bP,bMass,bMom,bM2,bGrad,bStr,bMomW,bCaust,bSV,bDV,bDW}) if (b) b->release();
        pool->release();
        return false;
    }

    // Dispatch in CHUNKS, one short command buffer each, FULLY SERIALIZED with a small host-side
    // gap between buffers. Why chunk + gaps:
    //  - one monolithic buffer runs minutes -> killed by the macOS GPU watchdog ("Impacting
    //    Interactivity") when the display needs the GPU;
    //  - keeping buffers back-to-back (pipelined) sustains 100% GPU queue pressure and gets killed
    //    EVEN WITH ~0.1 s buffers under active display use -- the watchdog reacts to starvation,
    //    not just per-buffer duration. The gaps are what let WindowServer breathe.
    // Chunk size adapts toward targetSec (heavy-tailed per-tet cost on real data); a killed buffer
    // may have PARTIALLY deposited its atomics, so a failed chunk is never retried alone: each
    // retry re-zeros the grids, redoes the whole deposit with 4x shorter buffers and wider gaps,
    // and persistent failure falls back to the CPU deposit in the caller.
    // PS_METAL_CHUNK=<n> overrides the starting chunk size of the one-thread-per-tet kernel.
    size_t const MIN_CHUNK = fExact ? 50 : 2000;
    size_t const MAX_CHUNK = 500000;
    double targetSec = 0.25;
    useconds_t gapUs = 8000;          // ~3% overhead at 0.25 s buffers
    // --ps-exact-deposit costs ~1-2 orders of magnitude more per tet (analytic clipping of
    // every cell in the window vs a barycentric test per sample), so start far smaller: the
    // adaptive controller only retargets AFTER the first chunk, and one oversized opening
    // buffer is exactly what the watchdog kills.
    size_t baseChunk = fExact ? 500 : 25000;
    if (const char* env = getenv("PS_METAL_CHUNK"))
    { long v = atol(env); if (v > 0) baseChunk = size_t(v); }
    std::string lastErr;
    // PS_METAL_TIMING=1: one line per kernel run (threads, buffers, GPU-busy and wall seconds)
    bool const timing = getenv("PS_METAL_TIMING") && atoi(getenv("PS_METAL_TIMING")) != 0;

    // One paced, chunked run of 'pso' over nThreads threads; bind(enc, start, n) binds the buffers
    // for the chunk [start, start+n). False (lastErr set) when a command buffer failed.
    auto runChunked = [&](MTL::ComputePipelineState* pso, size_t nThreads, size_t firstChunk,
                          size_t minChunk, auto&& bind) -> bool
    {
        NS::UInteger tg = pso->maxTotalThreadsPerThreadgroup();
        if (tg > 256) tg = 256;
        size_t chunk = std::max(firstChunk, minChunk);
        size_t start = 0, nBuf = 0;
        double busy = 0.;
        auto const tRun = std::chrono::steady_clock::now();
        while (start < nThreads)
        {
            uint32_t const n = uint32_t(std::min(chunk, nThreads - start));
            auto t0 = std::chrono::steady_clock::now();
            MTL::CommandBuffer* cb = c.q->commandBuffer();
            MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
            enc->setComputePipelineState(pso);
            bind(enc, start, n);
            enc->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(tg, 1, 1));
            enc->endEncoding();
            cb->commit();
            cb->waitUntilCompleted();
            double const el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

            if (cb->status() == MTL::CommandBufferStatusError)
            {
                NS::Error* e = cb->error();
                lastErr = std::string("Metal command buffer failed: ")
                        + (e ? e->localizedDescription()->utf8String() : "unknown");
                return false;
            }

            start += n;
            ++nBuf;
            busy += el;
            if (el > 1e-4)   // retarget the next chunk toward targetSec (bounded rescale)
            {
                double scale = targetSec / el;
                if (scale < 0.5) scale = 0.5;
                if (scale > 4.0) scale = 4.0;
                chunk = std::min(std::max(size_t(double(chunk) * scale), minChunk), MAX_CHUNK);
            }
            usleep(gapUs);   // deliberate idle gap: lets WindowServer take the GPU
        }
        if (timing)
            fprintf(stderr, "PS-DTFE Metal: %-17s %10zu threads  %6zu buffers  %8.3f s busy  %8.3f s wall\n",
                    pso == c.psoItems ? "depositExactItems" : "depositFields", nThreads, nBuf, busy,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - tRun).count());
        return true;
    };

    // depositFields over the tets [0,nT) of the given records. The grids are bound whole; the
    // per-tet buffers at the chunk's offset (P.nTet = chunk size, rewritten per chunk: safe, the
    // previous chunk completed).
    auto runPerTet = [&](MTL::Buffer* vB, MTL::Buffer* uB, MTL::Buffer* mB, MTL::Buffer* dB, size_t nT) -> bool
    {
        return runChunked(c.pso, nT, baseChunk, MIN_CHUNK, [&](MTL::ComputeCommandEncoder* enc, size_t start, uint32_t n)
        {
            P.nTet = n;
            std::memcpy(bP->contents(), &P, sizeof(P));
            enc->setBuffer(vB,    start * PS_TET_STRIDE * sizeof(float), 0);
            enc->setBuffer(uB,    haveVels ? start * 12 * sizeof(float) : 0, 1);
            enc->setBuffer(mB,    start * sizeof(float),      2);
            enc->setBuffer(bMass, 0, 3);
            enc->setBuffer(bMom,  0, 4);
            enc->setBuffer(bM2,   0, 5);
            enc->setBuffer(bGrad, 0, 6);
            enc->setBuffer(bStr,  0, 7);
            enc->setBuffer(bP,    0, 8);
            enc->setBuffer(dB,    haveDens ? start * 4 * sizeof(float) : 0, 9);
            enc->setBuffer(bMomW, 0, 10);
            enc->setBuffer(bCaust,0, 11);
            enc->setBuffer(bSV,   0, 12);
            enc->setBuffer(bDV,   0, 13);
            enc->setBuffer(bDW,   0, 14);
        });
    };

    // --ps-exact-deposit: split the work below the tet (depositExactItems in ps_deposit.metal).
    // An item is (tet, block of blockCells cells of its bbox window); the host only needs an
    // UPPER bound of each window -- the kernel re-derives it in float and idles surplus blocks.
    // The float and double floors can differ only when the value is within rounding of an integer,
    // so only such a bound is widened (by eps, far above the float error): padding every axis by a
    // cell instead inflated a 2-3 cell window up to 8x, and each idle item still pays the setup.
    // The items follow the caller's cost order, keeping a SIMD group's threads on similar work.
    // PS_EXACT_ITEMS=0 keeps the one-thread-per-tet kernel (A/B), PS_EXACT_ITEM_CELLS sets the block.
    bool useItems = fExact;
    if (const char* env = getenv("PS_EXACT_ITEMS")) useItems = useItems && atoi(env) != 0;
    MTL::Buffer* bItems = nullptr;
    MTL::Buffer* bSumW  = nullptr;
    size_t nItems = 0;
    uint32_t blockCells = 16;
    if (const char* env = getenv("PS_EXACT_ITEM_CELLS")) { long v = atol(env); if (v > 0) blockCells = uint32_t(v); }
    if (useItems)
    {
        float const* R0 = static_cast<float const*>(bV->contents());
        std::vector<uint64_t> bound(nTetTotal);
        double total = 0.;
        for (size_t t = 0; t < nTetTotal; ++t)
        {
            float const* R = R0 + t * PS_TET_STRIDE;
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
                nw = iHi > iLo ? nw * uint64_t(iHi - iLo) : 0;
            }
            bound[t] = nw;
            total += double(nw);
        }
        // at most ~32M items (256 MB): coarser blocks beyond that (16-32 cells run within 10%)
        double const maxItems = 32.e6;
        if (total / blockCells > maxItems) blockCells = uint32_t(std::ceil(total / maxItems));
        // a tet whose window lies wholly outside the grid gets no item: it is a leftover (below)
        for (size_t t = 0; t < nTetTotal; ++t)
            nItems += (bound[t] + blockCells - 1) / blockCells;
        if (timing)
            fprintf(stderr, "PS-DTFE Metal: exact items: %zu tets, %zu items of %u cells, window bound %.3g cells\n",
                    nTetTotal, nItems, blockCells, total);
        bItems = c.dev->newBuffer(std::max<size_t>(nItems, 1) * 2 * sizeof(uint32_t), MTL::ResourceStorageModeShared);
        bSumW  = c.dev->newBuffer(std::max<size_t>(nTetTotal, 1) * sizeof(float), MTL::ResourceStorageModeShared);
        if (!bItems || !bSumW)
        {
            // too big for GPU-visible memory: the one-thread-per-tet kernel still works
            if (bItems) bItems->release();
            if (bSumW)  bSumW->release();
            bItems = bSumW = nullptr;
            useItems = false;
        }
        else
        {
            uint32_t* it = static_cast<uint32_t*>(bItems->contents());
            for (size_t t = 0; t < nTetTotal; ++t)
            {
                uint64_t const nb = (bound[t] + blockCells - 1) / blockCells;
                for (uint64_t k = 0; k < nb; ++k) { *it++ = uint32_t(t); *it++ = uint32_t(std::min<uint64_t>(k, 0xFFFFFFFFu)); }
            }
        }
    }
    struct ExactItemParams { uint32_t nItems, phase, blockCells, pad; };   // == the MSL struct
    auto runItems = [&](uint32_t phase) -> bool
    {
        // an item clips at most blockCells cells: far cheaper and more uniform than a whole tet
        return runChunked(c.psoItems, nItems, 8192, 1024, [&](MTL::ComputeCommandEncoder* enc, size_t start, uint32_t n)
        {
            ExactItemParams const IP{ n, phase, blockCells, 0u };
            enc->setBuffer(bV,    0, 0);
            enc->setBuffer(bU,    0, 1);
            enc->setBuffer(bM,    0, 2);
            enc->setBuffer(bMass, 0, 3);
            enc->setBuffer(bMom,  0, 4);
            enc->setBuffer(bM2,   0, 5);
            enc->setBuffer(bGrad, 0, 6);
            enc->setBuffer(bStr,  0, 7);
            enc->setBuffer(bP,    0, 8);
            enc->setBuffer(bD,    0, 9);
            enc->setBuffer(bMomW, 0, 10);
            enc->setBuffer(bCaust,0, 11);
            enc->setBuffer(bSV,   0, 12);
            enc->setBuffer(bDV,   0, 13);
            enc->setBuffer(bDW,   0, 14);
            enc->setBuffer(bItems, start * 2 * sizeof(uint32_t), 15);
            enc->setBuffer(bSumW, 0, 16);
            enc->setBytes(&IP, sizeof IP, 17);
        });
    };

    // The tets the items pass leaves over (sumW still 0: float-degenerate, or a window that clips
    // empty or lies outside the grid) run through depositFields, which falls through to the sampled
    // path / centroid fallback or defers them, exactly as the one-thread path does; their
    // deferrals (negated masses) are copied back into bM. A few leftovers run on compacted copies
    // of their records; many (a --box region cut from a larger cloud leaves most of an outer
    // partition's tets outside) run in place with every other tet's mass masked to 0 (skipped by
    // the kernel), 4 B per tet instead of a ~130 B copy each.
    auto runLeftovers = [&]() -> bool
    {
        float* mb = static_cast<float*>(bM->contents());
        float const* sw = static_cast<float const*>(bSumW->contents());
        std::vector<uint32_t> left;
        for (size_t t = 0; t < nTetTotal; ++t)
            if (mb[t] > 0.f && !(sw[t] > 0.f)) left.push_back(uint32_t(t));
        if (left.empty()) return true;
        size_t const nL = left.size();
        bool const compact = nL <= nTetTotal / 16;
        auto gather = [&](MTL::Buffer* src, size_t stride) {
            MTL::Buffer* b = c.dev->newBuffer(nL * stride * sizeof(float), MTL::ResourceStorageModeShared);
            if (!b) return b;
            float const* s0 = static_cast<float const*>(src->contents());
            float* d0 = static_cast<float*>(b->contents());
            for (size_t k = 0; k < nL; ++k) std::memcpy(d0 + k * stride, s0 + size_t(left[k]) * stride, stride * sizeof(float));
            return b;
        };
        MTL::Buffer* lM = nullptr;
        MTL::Buffer *lV = bV, *lU = bU, *lD = bD;
        if (compact)
        {
            lV = gather(bV, PS_TET_STRIDE);
            lU = haveVels ? gather(bU, 12) : zeroBuf(4);
            lM = gather(bM, 1);
            lD = haveDens ? gather(bD, 4) : zeroBuf(4);
        }
        else if ((lM = zeroBuf(nTetTotal * sizeof(float))))
        {
            float* lm = static_cast<float*>(lM->contents());
            for (uint32_t t : left) lm[t] = mb[t];
        }
        bool ok = lV && lU && lM && lD;
        if (!ok) lastErr = "Metal buffer allocation failed (exact-deposit leftovers)";
        else ok = runPerTet(lV, lU, lM, lD, compact ? nL : nTetTotal);
        if (ok)
        {
            float const* lm = static_cast<float const*>(lM->contents());
            for (size_t k = 0; k < nL; ++k)
            {
                float const v = lm[compact ? k : size_t(left[k])];
                if (v < 0.f) mb[left[k]] = v;
            }
        }
        if (lM) lM->release();
        if (compact) for (MTL::Buffer* b : {lV, lU, lD}) if (b) b->release();
        return ok;
    };

    int const maxAttempts = 3;
    bool success = false;

    for (int attempt = 0; attempt < maxAttempts && !success; ++attempt)
    {
        std::memset(bMass->contents(), 0, nCell * 4);
        std::memset(bMom->contents(),  0, momBytes);
        std::memset(bM2->contents(),   0, m2Bytes);
        std::memset(bGrad->contents(), 0, gradBytes);
        std::memset(bStr->contents(),  0, nCell * 4);
        std::memset(bMomW->contents(),  0, momwBytes);
        std::memset(bCaust->contents(), 0, caustBytes);
        std::memset(bSV->contents(),    0, svBytes);
        std::memset(bDV->contents(),    0, dvBytes);
        std::memset(bDW->contents(),    0, dwBytes);

        if (useItems)
        {
            std::memset(bSumW->contents(), 0, nTetTotal * sizeof(float));
            success = runItems(0) && runItems(1) && runLeftovers();
        }
        else
            success = runPerTet(bV, bU, bM, bD, nTetTotal);

        if (!success)
        {
            // escalate: 4x shorter buffers, wider gaps, smaller starting chunk; settle first
            targetSec = std::max(targetSec / 4.0, 0.02);
            gapUs     = std::min(gapUs * 3, useconds_t(50000));
            baseChunk = std::max(baseChunk / 2, MIN_CHUNK);
            if (attempt + 1 < maxAttempts)
            {
                fprintf(stderr, "%sPS-DTFE Metal: %s -- retrying the partition deposit from scratch "
                                "(attempt %d/%d, target %.0f ms buffers, %.0f ms gaps)%s\n",
                        MESSAGE::cYellow(), lastErr.c_str(), attempt + 2, maxAttempts,
                        targetSec*1e3, gapUs/1e3, MESSAGE::cReset());
                sleep(2);   // let the system settle before hammering the GPU again
            }
        }
    }
    for (MTL::Buffer* b : {bItems, bSumW}) if (b) b->release();
    if (!success)
    {
        err = lastErr;
        for (MTL::Buffer* b : {bV,bU,bM,bD,bP,bMass,bMom,bM2,bGrad,bStr,bMomW,bCaust,bSV,bDV,bDW}) b->release();
        pool->release();
        return false;
    }

    // tets the kernel could not classify exactly (a sample within float rounding of a face):
    // it negated their masses and deposited nothing; the caller deposits them on the CPU
    {
        float const* mb = static_cast<float const*>(bM->contents());
        for (size_t t = 0; t < nTetTotal; ++t)
            if (mb[t] < 0.f) deferred.push_back(uint32_t(t));
    }

    std::memcpy(out.mass.data(),    bMass->contents(), nCell * 4);
    if (fVel)  std::memcpy(out.mom.data(),  bMom->contents(),  nCell * 12);
    if (fDisp) std::memcpy(out.m2.data(),   bM2->contents(),   nCell * 24);
    if (fGrad) std::memcpy(out.grad.data(), bGrad->contents(), nCell * 36);
    std::memcpy(out.streams.data(), bStr->contents(),  nCell * 4);
    if (fVolW)    std::memcpy(out.momw.data(),    bMomW->contents(),  nCell * 4);
    if (fCaustic) std::memcpy(out.caustic.data(), bCaust->contents(), nCell * 4);
    if (fExact)   std::memcpy(out.streamvol.data(), bSV->contents(), nCell * 4);
    if (fDispOwn) {
        std::memcpy(out.dispvel.data(), bDV->contents(), nCell * 12);
        std::memcpy(out.dispw.data(),   bDW->contents(), nCell * 4);
    }

    for (MTL::Buffer* b : {bV,bU,bM,bD,bP,bMass,bMom,bM2,bGrad,bStr,bMomW,bCaust,bSV,bDV,bDW}) b->release();
    pool->release();
    return true;
}
