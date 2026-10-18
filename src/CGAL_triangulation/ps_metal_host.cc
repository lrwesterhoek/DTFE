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
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <vector>
#include <unistd.h>

#include "gpu_host.h"
#include "ps_deposit_params.h"
#include "ps_mass_audit.h"     // PS_MASS_AUDIT=1: the fixed-point grid's exact total
#include "ps_deposit_msl.h"   // generated: static const char PS_DEPOSIT_MSL[]
#include <atomic>
extern std::atomic<int> psGpuRetries;   // ps_interpolation.cc (run_progress.h)
extern std::atomic<int> psGpuRetriesByCause[4];   // ... by cause: innocent, timeout, oom, other (2026-10-11)
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


// ---- chunk-level checkpoints and the cause-aware retry policy (2026-10-11) ----
// A failed command buffer used to redo the WHOLE partition deposit (up to ~2000 s per TNG50 partition) and
// three failures in a row sent it to the CPU. Now every pass keeps a snapshot of the buffers its kernel writes
// (the accumulator grids, the exact items' sumW, the per-tet masses the kernel negates when it defers a tet),
// taken at the pass's start and every DTFE_GPU_CHECKPOINT_CHUNKS chunks; a failed buffer (which may have
// deposited part or all of its atomics) restores the last snapshot and continues from its chunk. The policy's
// pieces (GpuFault, the pace, the fault hook, GpuSnapshot) live in gpu_metal_retry.h since 2026-10-13, shared
// with the standard-DTFE host (dtfe_metal_host.cc); the pass loop is psGpuDepositFields' runChunked below.
namespace {

#include "gpu_metal_retry.h"

// the run's retry counters (ps_interpolation.cc's), by gpuRetryCause's index
void countGpuRetry(GpuFault f, long code, std::string const& desc)
{
    int const k = gpuRetryCause(f, code, desc);
    ++psGpuRetries;
    ++psGpuRetriesByCause[k];
}

} // namespace


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
                          double massExtra, PSFixedUnits const& fxUnits,
                          PSGpuGrids& out, std::vector<uint32_t>& deferred, std::string& err)
{
    std::lock_guard<std::mutex> lock(ctxMutex());   // serialize dispatches (single queue)
    Ctx& c = ctx();
    if (!c.ready) { err = c.err; return false; }

    // only the requested moment grids are allocated, host- and device-side: the m2 (24 B/cell)
    // and grad (36 B/cell) grids dominate the deposit's footprint and most runs need neither.
    // A grid the caller gave a fixed-point unit (fxUnits, 2026-10-14) is a 64-bit integer sum: 8 B per component
    // in place of the float's 4, bound at its float index AND at its word index (22 + slot, ps_deposit.metal).
    size_t const nCell = subDims[0] * subDims[1] * subDims[2];
    bool   const fDispOwn   = fVolW && fDisp;   // dispersion keeps its own mass-weighted mean+normalizer
    bool const sel[PS_FX_N] = { fVel, fDisp, fGrad, fVolW, fExact, fDispOwn, false, fScal, fSGrad };
    bool fxOn[PS_FX_N];
    uint32_t fxMask = 0;
    for (int g = 0; g < PS_FX_N; ++g)
    {
        fxOn[g] = sel[g] && fxUnits.scale[g] > 0.f;
        if (fxOn[g]) fxMask |= 1u << g;
    }
    auto gridBytes4 = [&](int g, size_t comps) -> size_t { return sel[g] ? nCell * comps * (fxOn[g] ? 8 : 4) : 4; };
    size_t const momBytes  = gridBytes4(PS_FX_MOM, 3);
    size_t const m2Bytes   = gridBytes4(PS_FX_M2, 6);
    size_t const gradBytes = gridBytes4(PS_FX_GRAD, 9);
    size_t const momwBytes  = gridBytes4(PS_FX_MOMW, 1);
    size_t const caustBytes = fCaustic ? nCell * 4 : 4;
    size_t const svBytes    = gridBytes4(PS_FX_SV, 1);
    size_t const dvBytes    = gridBytes4(PS_FX_DV, 3);
    size_t const scalBytes  = gridBytes4(PS_FX_SCAL, 1);
    size_t const sgBytes    = gridBytes4(PS_FX_SG, 3);
    // the mass grid as an exact 64-bit fixed-point sum (2026-10-08; DepositParams.massFixed in the kernel
    // says why). PS_METAL_FLOAT_MASS=1 keeps the float atomics of before, for comparison. The unit covers the
    // caller's own deposits too (massExtra, 2026-10-12): it adds the deferred and released tets into these words.
    float massScale = 1.f;
    bool massFixed = false;
    {
        double totalMass = massExtra > 0. ? massExtra : 0.;
        // (a non-finite mass -- bad input, or the DTFE_TEST_NAN_SHARE hook -- is left out of the unit: its shares are
        // the non-finite guard's to drop and count, 2026-10-15, not a reason to lose the whole grid's fixed point)
        for (float const mt : masses) if (std::isfinite(mt)) totalMass += double(mt);
        if (fxUnits.massTotal > 0.) totalMass = fxUnits.massTotal;   // (a --ps-window pass: the caller's total, 2026-10-15)
        bool const floatMass = getenv("PS_METAL_FLOAT_MASS") && atoi(getenv("PS_METAL_FLOAT_MASS")) != 0;
        massFixed = !floatMass && psMassFixedScale(totalMass, massScale);
    }
    // the dispersion's own normalizer sums exactly the mass grid's shares: with both exact it is not deposited
    // at all (the caller reads the mass words; PSGpuGrids::dispwIsMass), else the float atomics of before
    bool const dispwIsMass = fDispOwn && massFixed && fxUnits.scale[PS_FX_DW] > 0.f;
    if (dispwIsMass) fxMask |= 1u << PS_FX_DW;
    size_t const dwBytes    = (fDispOwn && !dispwIsMass) ? nCell * 4 : 4;
    bool const sumWFixed = fExact && fxUnits.sumW;
    if (sumWFixed) fxMask |= PS_FX_SUMW_BIT;
    out.dispwIsMass = dispwIsMass;
    for (int g = 0; g < PS_FX_N; ++g) { out.fx[g].clear(); out.fxScale[g] = fxOn[g] ? fxUnits.scale[g] : 0.f; }
    // fixed point: the words are handed over as they are (out.massFx, filled from the device grid at the end;
    // 'mass' stays empty), else the float grid
    out.mass.assign(massFixed ? 0 : nCell, 0.f);
    out.massFx.clear();
    out.massScale = massFixed ? massScale : 0.f;
    // a fixed-point grid comes back as words (out.fx[g]) and leaves its float vector EMPTY
    out.mom.assign(fVel && !fxOn[PS_FX_MOM] ? nCell * 3 : 0, 0.f);
    out.m2.assign(fDisp && !fxOn[PS_FX_M2] ? nCell * 6 : 0, 0.f);
    out.grad.assign(fGrad && !fxOn[PS_FX_GRAD] ? nCell * 9 : 0, 0.f);
    out.streams.assign(nCell, 0u);
    out.momw.assign(fVolW && !fxOn[PS_FX_MOMW] ? nCell : 0, 0.f);
    out.caustic.assign(fCaustic ? nCell : 0, 0u);
    out.streamvol.assign(fExact && !fxOn[PS_FX_SV] ? nCell : 0, 0.f);
    out.dispvel.assign(fDispOwn && !fxOn[PS_FX_DV] ? nCell * 3 : 0, 0.f);
    out.dispw.assign(fDispOwn && !dispwIsMass ? nCell : 0, 0.f);
    out.scal.assign(fScal && !fxOn[PS_FX_SCAL] ? nCell : 0, 0.f);
    out.sgrad.assign(fSGrad && !fxOn[PS_FX_SG] ? nCell * 3 : 0, 0.f);
    static size_t const fxComps[PS_FX_N] = { 3, 6, 9, 1, 1, 3, 1, 1, 3 };
    deferred.clear();
    if (masses.empty())                 // nothing to deposit (empty partition; the caller may still add released tets)
    {
        if (massFixed) out.massFx.assign(nCell, 0);
        for (int g = 0; g < PS_FX_N; ++g) if (fxOn[g]) out.fx[g].assign(nCell * fxComps[g], 0);
        return true;
    }
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
    P.fScal    = (fScal  && not scal.empty()) ? 1 : 0;
    P.fSGrad   = (fSGrad && not scal.empty()) ? 1 : 0;
    P.windowMode = windowMode ? 1 : 0;
    P.massFixed = massFixed ? 1 : 0;   // (the unit is chosen above)
    P.massScale = massScale;
    P.fxMask = fxMask;                 // the moment grids' fixed-point sums (the caller's units, above)
    for (int g = 0; g < PS_FX_N; ++g) P.fxScale[g] = fxOn[g] ? fxUnits.scale[g] : 0.f;
    size_t const massBytes   = massFixed ? 4 : nCell * 4;   // the float grid (buffer 3), else a dummy
    size_t const massFxBytes = massFixed ? nCell * 8 : 4;   // the fixed-point grid (buffer 21), else a dummy

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
    // the per-tet vertex scalars (fScal / fSGrad); dummy otherwise, so its per-chunk offset is keyed
    // on the vector being real, like 'dens'
    bool const haveScal = not scal.empty();
    MTL::Buffer* bSc = haveScal
                     ? c.dev->newBuffer(scal.data(), scal.size() * sizeof(float), MTL::ResourceStorageModeShared)
                     : zeroBuf(4);
    std::vector<float>().swap(scal);
    MTL::Buffer* bP = c.dev->newBuffer(&P, sizeof(P), MTL::ResourceStorageModeShared);
    MTL::Buffer* bMass = zeroBuf(massBytes);
    MTL::Buffer* bMassFx = zeroBuf(massFxBytes);
    MTL::Buffer* bMom  = zeroBuf(momBytes);
    MTL::Buffer* bM2   = zeroBuf(m2Bytes);
    MTL::Buffer* bGrad = zeroBuf(gradBytes);
    MTL::Buffer* bStr  = zeroBuf(nCell * 4);
    MTL::Buffer* bMomW  = zeroBuf(momwBytes);
    MTL::Buffer* bCaust = zeroBuf(caustBytes);
    MTL::Buffer* bSV    = zeroBuf(svBytes);
    MTL::Buffer* bDV    = zeroBuf(dvBytes);
    MTL::Buffer* bDW    = zeroBuf(dwBytes);
    MTL::Buffer* bScal  = zeroBuf(scalBytes);
    MTL::Buffer* bSG    = zeroBuf(sgBytes);
    // the non-finite guard's counters of depositFields (2026-10-15; ps_deposit.metal FieldGrids.bad), one word per slot
    MTL::Buffer* bBad   = zeroBuf(FX_DROP_SLOTS * 4);
    if (!bV || !bU || !bM || !bD || !bSc || !bP || !bMass || !bMassFx || !bMom || !bM2 || !bGrad || !bStr || !bMomW || !bCaust || !bSV || !bDV || !bDW || !bScal || !bSG || !bBad)
    {
        err = "Metal buffer allocation failed (out of GPU-visible memory?)";
        for (MTL::Buffer* b : {bV,bU,bM,bD,bSc,bP,bMass,bMassFx,bMom,bM2,bGrad,bStr,bMomW,bCaust,bSV,bDV,bDW,bScal,bSG,bBad}) if (b) b->release();
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
    // Chunk size adapts toward targetSec (heavy-tailed per-tet cost on real data). A killed buffer
    // may have PARTIALLY deposited its atomics, so a failed chunk is never retried alone: the pass
    // restores its last snapshot (the pass's start, or a periodic checkpoint) and continues from
    // there (2026-10-11; before, every failure redid the whole partition deposit from zero). The
    // retry depends on the cause (GpuFault above); the partition goes to the CPU deposit in the
    // caller only when one chunk fails 3 times, on a fatal error, or when no snapshot can be held.
    // PS_METAL_CHUNK=<n> overrides the starting chunk size of the one-thread-per-tet kernel.
    size_t const MIN_CHUNK = fExact ? 50 : 2000;
    size_t const MAX_CHUNK = 500000;
    GpuPace& pace = gpuPace();        // targetSec 0.25 s, gapUs 8 ms (~3% overhead) until escalated
    // --ps-exact-deposit costs ~1-2 orders of magnitude more per tet (analytic clipping of
    // every cell in the window vs a barycentric test per sample), so start far smaller: the
    // adaptive controller only retargets AFTER the first chunk, and one oversized opening
    // buffer is exactly what the watchdog kills.
    size_t baseChunk = fExact ? 500 : 25000;
    if (const char* env = getenv("PS_METAL_CHUNK"))
    { long v = atol(env); if (v > 0) baseChunk = size_t(v); }
    // an escalated pace (an earlier partition's failures) opens with a proportionally smaller chunk
    if (pace.targetSec < 0.25)
        baseChunk = std::max(size_t(double(baseChunk) * pace.targetSec / 0.25), MIN_CHUNK);
    std::string lastErr;
    // PS_METAL_TIMING=1: one line per kernel run (threads, buffers, GPU-busy and wall seconds)
    bool const timing = getenv("PS_METAL_TIMING") && atoi(getenv("PS_METAL_TIMING")) != 0;

    // DTFE_GPU_CHECKPOINT_CHUNKS=<n>: a checkpoint every n command buffers (0 = only the pass-start
    // snapshot). Unset: every ~10 s of GPU work (n = 10 s / the current buffer target, so an escalated
    // pace of 0.02 s buffers does not checkpoint 12x as often). A checkpoint memcpy's the accumulator
    // grids (~24 B per cell for density + velocity: ~0.4 GB at 256^3, ~0.1 s) into ONE spare host copy,
    // allocated at the first checkpoint and held until the partition's deposit returns.
    long ckChunks = -1;
    if (const char* env = getenv("DTFE_GPU_CHECKPOINT_CHUNKS")) { ckChunks = atol(env); if (ckChunks < 0) ckChunks = 0; }
    // DTFE_GPU_PACE_RECOVER_SEC=<s>: the clean GPU work after which an escalated pace steps back (GpuPace above)
    double paceRecoverSec = 90.;
    if (const char* env = getenv("DTFE_GPU_PACE_RECOVER_SEC")) { double v = atof(env); if (v >= 0.) paceRecoverSec = v; }
    std::vector<MTL::Buffer*> const gridBufs = { bMass, bMassFx, bMom, bM2, bGrad, bStr, bMomW, bCaust, bSV, bDV, bDW, bScal, bSG };
    GpuSnapshot snap;
    int nFailed = 0, nResumedCk = 0;   // failed buffers of this deposit, and how many resumed from a periodic checkpoint
    // The shares the non-finite guard dropped in this deposit, by slot (2026-10-15; ps_deposit_params.h
    // psFixedShareOk): each pass zeroes its counter words before a command buffer and adds them here after one that
    // succeeded, and a restore rolls these sums back to the held snapshot's -- so a failed buffer (which may have
    // counted part or all of its shares) and the redone ones count once. Handed to the run's counters only when
    // the deposit is kept (a CPU fallback deposits, and counts, the partition itself).
    unsigned long long badRun[FX_DROP_SLOTS] = {};
    int const MAX_FAILS_SAME_CHUNK = 3, MAX_FAILS_TOTAL = 16;

    // One paced, chunked run of 'pso' over nThreads threads; bind(enc, start, n) binds the buffers
    // for the chunk [start, start+n). 'extras' are the non-grid buffers the kernel WRITES (sumW, the
    // per-tet masses it negates); gridsZero: the grids are all zero at the pass's start (snapshotted
    // as such, without a copy); gridsKeepZero: the pass never writes them (the exact items' phase 0),
    // so its periodic checkpoints skip them too. False (lastErr set) when the pass gave up (the caller
    // falls back).
    // badW: this pass's counter words of the non-finite guard (badRun above).
    auto runChunked = [&](MTL::ComputePipelineState* pso, size_t nThreads, size_t firstChunk,
                          size_t minChunk, std::vector<MTL::Buffer*> const& extras, bool gridsZero,
                          bool gridsKeepZero, uint32_t* badW, auto&& bind) -> bool
    {
        NS::UInteger tg = pso->maxTotalThreadsPerThreadgroup();
        if (tg > 256) tg = 256;
        size_t chunk = std::max(firstChunk, minChunk);
        size_t start = 0, nBuf = 0, sinceCk = 0;
        double busy = 0.;
        auto const tRun = std::chrono::steady_clock::now();
        std::vector<MTL::Buffer*> ckBufs = gridBufs;
        ckBufs.insert(ckBufs.end(), extras.begin(), extras.end());
        std::vector<bool> zeroMask(ckBufs.size(), false);
        if (gridsZero) std::fill(zeroMask.begin(), zeroMask.begin() + gridBufs.size(), true);
        bool ckOn = snap.take(ckBufs, zeroMask, 0);
        bool ckPeriodic = false;           // the held snapshot is a periodic checkpoint (not the pass start)
        unsigned long long badCk[FX_DROP_SLOTS];   // badRun at the held snapshot
        std::memcpy(badCk, badRun, sizeof badCk);
        // failures of the 'same chunk': a failure that comes before the redo got past the previous failed
        // buffer's end (chunk boundaries move with the timing controller, so a start offset is no identity)
        int  failsHere = 0;
        size_t failEnd = 0;
        while (start < nThreads)
        {
            uint32_t const n = uint32_t(std::min(chunk, nThreads - start));
            auto t0 = std::chrono::steady_clock::now();
            MTL::CommandBuffer* cb = c.q->commandBuffer();
            MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
            enc->setComputePipelineState(pso);
            std::memset(badW, 0, FX_DROP_SLOTS * 4);   // (the previous buffer completed: waitUntilCompleted below)
            bind(enc, start, n);
            enc->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(tg, 1, 1));
            enc->endEncoding();
            cb->commit();
            cb->waitUntilCompleted();
            double const el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

            std::string desc;
            long code = 0;
            bool failed = false;
            if (cb->status() == MTL::CommandBufferStatusError)
            {
                NS::Error* e = cb->error();
                desc = e ? e->localizedDescription()->utf8String() : "unknown";
                code = e ? long(e->code()) : 0;
                failed = true;
            }
            else failed = gpuFaultInjected(desc, code);   // the test hook (the buffer DID deposit)

            if (failed)
            {
                ++nFailed;
                GpuFault const cause = classifyGpuFault(code, desc);
                countGpuRetry(cause, code, desc);   // the run's counters count every failed buffer (run_progress.h)
                lastErr = std::string("Metal command buffer failed: ") + desc + " [" + gpuFaultName(cause) + "]";
                failsHere = (failsHere > 0 && start < failEnd) ? failsHere + 1 : 1;
                failEnd = std::max(failEnd, start + n);
                if (cause == GpuFault::Fatal || !ckOn || failsHere >= MAX_FAILS_SAME_CHUNK || nFailed >= MAX_FAILS_TOTAL)
                {
                    if (!ckOn) lastErr += " (no checkpoint held)";
                    else if (failsHere >= MAX_FAILS_SAME_CHUNK) lastErr += " (" + std::to_string(failsHere) + " failures of the same chunk)";
                    else if (nFailed >= MAX_FAILS_TOTAL) lastErr += " (" + std::to_string(nFailed) + " failed buffers in this deposit)";
                    return false;
                }
                if (cause != GpuFault::Victim)
                {   // escalate (for the rest of the process): 4x shorter buffers, wider gaps, smaller chunk
                    pace.targetSec = std::max(pace.targetSec / 4.0, 0.02);
                    pace.gapUs     = std::min(pace.gapUs * 3, useconds_t(50000));
                    pace.cleanSec  = 0.;
                    chunk = std::max(chunk / 4, minChunk);
                    baseChunk = std::max(baseChunk / 2, MIN_CHUNK);
                }
                snap.restore();
                std::memcpy(badRun, badCk, sizeof badCk);
                if (ckPeriodic) ++nResumedCk;
                fprintf(stderr, "%sPS-DTFE Metal: %s -- resuming from the %s at thread %zu of %zu "
                                "(redoing %zu; failure %d of this chunk, target %.0f ms buffers, %.0f ms gaps)%s\n",
                        MESSAGE::cYellow(), lastErr.c_str(), ckPeriodic ? "checkpoint" : "pass start",
                        snap.start, nThreads, start + n - snap.start, failsHere,
                        pace.targetSec * 1e3, pace.gapUs / 1e3, MESSAGE::cReset());
                start = snap.start;
                sinceCk = 0;
                // an innocent victim retries at once; our own buffers' failures let the system settle first
                if (cause == GpuFault::Resource) sleep(2);
                else if (cause == GpuFault::Watchdog) sleep(1);
                continue;
            }

            start += n;
            ++nBuf;
            ++sinceCk;
            busy += el;
            for (int k = 0; k < FX_DROP_SLOTS; ++k) badRun[k] += badW[k];
            if (el > 1e-4)   // retarget the next chunk toward targetSec (bounded rescale)
            {
                double scale = pace.targetSec / el;
                if (scale < 0.5) scale = 0.5;
                if (scale > 4.0) scale = 4.0;
                chunk = std::min(std::max(size_t(double(chunk) * scale), minChunk), MAX_CHUNK);
            }
            if (pace.targetSec < GPU_PACE_TARGET0 || pace.gapUs > GPU_PACE_GAP0)
            {   // an escalated pace steps back after a clean stretch of GPU work (GpuPace above)
                pace.cleanSec += el + pace.gapUs / 1e6;
                if (pace.cleanSec >= paceRecoverSec)
                {
                    pace.targetSec = std::min(pace.targetSec * 4.0, GPU_PACE_TARGET0);
                    pace.gapUs     = std::max(pace.gapUs / 3, GPU_PACE_GAP0);
                    fprintf(stderr, "PS-DTFE Metal: %.0f s of GPU work without a fault -- pace eased to %.0f ms "
                                    "buffers, %.0f ms gaps%s\n", pace.cleanSec, pace.targetSec * 1e3, pace.gapUs / 1e3,
                            pace.targetSec >= GPU_PACE_TARGET0 && pace.gapUs <= GPU_PACE_GAP0 ? " (the default)" : "");
                    pace.cleanSec = 0.;
                }
            }
            long const every = ckChunks >= 0 ? ckChunks : std::max(1L, long(std::lround(10.0 / pace.targetSec)));
            if (ckOn && every > 0 && long(sinceCk) >= every && start < nThreads)
            {   // the queue is drained (waitUntilCompleted above): the buffers hold exactly [0, start)
                if (snap.take(ckBufs, gridsKeepZero ? zeroMask : std::vector<bool>(), start))
                { ++snap.taken; ckPeriodic = true; sinceCk = 0; std::memcpy(badCk, badRun, sizeof badCk); }
                // else: no host memory for the spare copy; the last snapshot stays valid
            }
            usleep(pace.gapUs);   // deliberate idle gap: lets WindowServer take the GPU
        }
        if (timing)
            fprintf(stderr, "PS-DTFE Metal: %-17s %10zu threads  %6zu buffers  %8.3f s busy  %8.3f s wall\n",
                    pso == c.psoItems ? "depositExactItems" : "depositFields", nThreads, nBuf, busy,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - tRun).count());
        return true;
    };

    // the moment grids' word views (2026-10-14): each fixed-point grid's buffer again at 22 + its slot, where the
    // kernel reads it as two uint words per component (a float grid's slot gets the same buffer; unused there)
    auto bindWords = [&](MTL::ComputeCommandEncoder* enc)
    {
        enc->setBuffer(bMom,  0, 22);
        enc->setBuffer(bM2,   0, 23);
        enc->setBuffer(bGrad, 0, 24);
        enc->setBuffer(bMomW, 0, 25);
        enc->setBuffer(bSV,   0, 26);
        enc->setBuffer(bDV,   0, 27);
        enc->setBuffer(bScal, 0, 28);
        enc->setBuffer(bSG,   0, 29);
    };

    // depositFields over the tets [0,nT) of the given records. The grids are bound whole; the
    // per-tet buffers at the chunk's offset (P.nTet = chunk size, rewritten per chunk: safe, the
    // previous chunk completed). The kernel negates the masses of the tets it defers, so mB is
    // part of the snapshot: a redo then runs exactly the same code path.
    auto runPerTet = [&](MTL::Buffer* vB, MTL::Buffer* uB, MTL::Buffer* mB, MTL::Buffer* dB, MTL::Buffer* sB, size_t nT,
                         bool gridsZero) -> bool
    {
        return runChunked(c.pso, nT, baseChunk, MIN_CHUNK, { mB }, gridsZero, false, static_cast<uint32_t*>(bBad->contents()),
                          [&](MTL::ComputeCommandEncoder* enc, size_t start, uint32_t n)
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
            enc->setBuffer(sB,    haveScal ? start * 4 * sizeof(float) : 0, 18);
            enc->setBuffer(bScal, 0, 19);
            enc->setBuffer(bSG,   0, 20);
            enc->setBuffer(bMassFx, 0, 21);
            bindWords(enc);
            enc->setBuffer(bBad,  0, 30);
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
        // the per-tet weight totals: a float each, or (sumWFixed, 2026-10-14) two uint words in the tet's own unit
        // (+ the non-finite guard's counter words of the items kernel, at ExactItemParams.badAt: 2026-10-15)
        bSumW  = c.dev->newBuffer(std::max<size_t>(nTetTotal, 1) * (sumWFixed ? 8 : sizeof(float)) + FX_DROP_SLOTS * 4,
                                  MTL::ResourceStorageModeShared);
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
    struct ExactItemParams { uint32_t nItems, phase, blockCells, badAt; };   // == the MSL struct
    uint32_t const badAt = uint32_t(std::max<size_t>(nTetTotal, 1) * (sumWFixed ? 2 : 1));   // past the per-tet totals
    auto runItems = [&](uint32_t phase) -> bool
    {
        // an item clips at most blockCells cells: far cheaper and more uniform than a whole tet.
        // Phase 0 writes only sumW (the grids are still zero), phase 1 only the grids (sumW is read)
        std::vector<MTL::Buffer*> extras;
        if (phase == 0) extras.push_back(bSumW);
        return runChunked(c.psoItems, nItems, 8192, 1024, extras, true, phase == 0,
                          static_cast<uint32_t*>(bSumW->contents()) + badAt,
                          [&](MTL::ComputeCommandEncoder* enc, size_t start, uint32_t n)
        {
            ExactItemParams const IP{ n, phase, blockCells, badAt };
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
            enc->setBuffer(bSc,   0, 18);
            enc->setBuffer(bScal, 0, 19);
            enc->setBuffer(bSG,   0, 20);
            enc->setBuffer(bMassFx, 0, 21);
            bindWords(enc);
            enc->setBuffer(bSumW, 0, 30);   // the same buffer as 16: its words when sumWFixed
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
        int64_t const* swx = static_cast<int64_t const*>(bSumW->contents());   // (sumWFixed: the words, little-endian)
        std::vector<uint32_t> left;
        for (size_t t = 0; t < nTetTotal; ++t)
            if (mb[t] > 0.f && !(sumWFixed ? swx[t] > 0 : sw[t] > 0.f)) left.push_back(uint32_t(t));
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
        MTL::Buffer *lV = bV, *lU = bU, *lD = bD, *lS = bSc;
        if (compact)
        {
            lV = gather(bV, PS_TET_STRIDE);
            lU = haveVels ? gather(bU, 12) : zeroBuf(4);
            lM = gather(bM, 1);
            lD = haveDens ? gather(bD, 4) : zeroBuf(4);
            lS = haveScal ? gather(bSc, 4) : zeroBuf(4);
        }
        else if ((lM = zeroBuf(nTetTotal * sizeof(float))))
        {
            float* lm = static_cast<float*>(lM->contents());
            for (uint32_t t : left) lm[t] = mb[t];
        }
        bool ok = lV && lU && lM && lD && lS;
        if (!ok) lastErr = "Metal buffer allocation failed (exact-deposit leftovers)";
        else ok = runPerTet(lV, lU, lM, lD, lS, compact ? nL : nTetTotal, false);
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
        if (compact) for (MTL::Buffer* b : {lV, lU, lD, lS}) if (b) b->release();
        return ok;
    };

    // one pass sequence; each pass heals its own failed buffers from its snapshots (runChunked), so a
    // false here means the GPU gave up on this partition and the caller deposits it on the CPU
    bool success;
    if (useItems)
    {
        std::memset(bSumW->contents(), 0, nTetTotal * (sumWFixed ? 8 : sizeof(float)));
        success = runItems(0) && runItems(1) && runLeftovers();
    }
    else
        success = runPerTet(bV, bU, bM, bD, bSc, nTetTotal, true);
    if (nFailed > 0)
        fprintf(stderr, "%sPS-DTFE Metal: deposit %s after %d failed buffer(s) (%d resumed from a periodic checkpoint); "
                        "%d checkpoint(s), %.0f MB held, %.3f s copying%s\n",
                success ? MESSAGE::cYellow() : MESSAGE::cRed(), success ? "recovered on the GPU" : "gave up",
                nFailed, nResumedCk, snap.taken, snap.bytes / 1.e6, snap.sec, MESSAGE::cReset());
    else if (timing)
        fprintf(stderr, "PS-DTFE Metal: checkpoints: %d, %.0f MB held, %.3f s copying\n", snap.taken, snap.bytes / 1.e6, snap.sec);
    for (MTL::Buffer* b : {bItems, bSumW}) if (b) b->release();
    if (!success)
    {
        err = lastErr;
        for (MTL::Buffer* b : {bV,bU,bM,bD,bSc,bP,bMass,bMassFx,bMom,bM2,bGrad,bStr,bMomW,bCaust,bSV,bDV,bDW,bScal,bSG,bBad}) b->release();
        pool->release();
        return false;
    }
    // the deposit is kept: its dropped shares join the run's counts (reported once at the end, DTFE.cpp)
    for (int k = 0; k < FX_DROP_SLOTS; ++k)
        if (badRun[k]) fixedShareDropped(k, badRun[k]);

    // tets the kernel could not classify exactly (a sample within float rounding of a face):
    // it negated their masses and deposited nothing; the caller deposits them on the CPU
    bool const massAudit = PSMassAudit::on();
    {
        float const* mb = static_cast<float const*>(bM->contents());
        double deferredMass = 0.;
        for (size_t t = 0; t < nTetTotal; ++t)
            if (mb[t] < 0.f) { deferred.push_back(uint32_t(t)); deferredMass -= double(mb[t]); }
        if (massAudit) out.auditDeferredMass = deferredMass;
    }

    if (massFixed)
    {
        // the words themselves (2026-10-12; they were decoded to float here, and the CPU deposit of the deferred
        // tets then added its shares to those floats -- rounding away every share below half an ulp of a busy
        // cell, always downward: 1.6e-5 of the box mass at TNG50-3-Dark z=0.5, 1.2% of one partition's deferred
        // mass). Low word first: on this little-endian host each pair is the int64 (ps_deposit_params.h).
        out.massFx.resize(nCell);
        std::memcpy(out.massFx.data(), bMassFx->contents(), nCell * 8);
        if (massAudit)
        {   // the exact integer total (the partition's mass is below 2^62 units: no overflow)
            int64_t q = 0;
            for (size_t i = 0; i < nCell; ++i) q += out.massFx[i];
            out.auditFixedTotal = psMassFixedMass(q, massScale);
        }
    }
    else
        std::memcpy(out.mass.data(), bMass->contents(), nCell * 4);
    // each moment grid: its words (fixed point, 2026-10-14; each low/high pair IS the int64 on this little-endian
    // host, as for the mass) or its floats
    MTL::Buffer* const fxBuf[PS_FX_N] = { bMom, bM2, bGrad, bMomW, bSV, bDV, nullptr, bScal, bSG };
    std::vector<float>* const flt[PS_FX_N] = { &out.mom, &out.m2, &out.grad, &out.momw, &out.streamvol, &out.dispvel,
                                               nullptr, &out.scal, &out.sgrad };
    for (int g = 0; g < PS_FX_N; ++g)
    {
        if (!sel[g] || !fxBuf[g]) continue;
        size_t const n = nCell * fxComps[g];
        if (fxOn[g]) { out.fx[g].resize(n); std::memcpy(out.fx[g].data(), fxBuf[g]->contents(), n * 8); }
        else std::memcpy(flt[g]->data(), fxBuf[g]->contents(), n * 4);
    }
    std::memcpy(out.streams.data(), bStr->contents(),  nCell * 4);
    if (fCaustic) std::memcpy(out.caustic.data(), bCaust->contents(), nCell * 4);
    if (fDispOwn && !dispwIsMass) std::memcpy(out.dispw.data(), bDW->contents(), nCell * 4);

    for (MTL::Buffer* b : {bV,bU,bM,bD,bSc,bP,bMass,bMassFx,bMom,bM2,bGrad,bStr,bMomW,bCaust,bSV,bDV,bDW,bScal,bSG,bBad}) b->release();
    pool->release();
    return true;
}
