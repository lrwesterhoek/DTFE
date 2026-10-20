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
extern std::atomic<int> psGpuRetriesByCause[4];   // ... by cause: innocent, timeout, oom, other
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


// ---- chunk-level checkpoints and the cause-aware retry policy ----
// Every pass keeps a snapshot of the buffers its kernel writes (the accumulator grids, the exact items' sumW,
// the per-tet masses the kernel negates when it defers a tet), taken at the pass's start and every
// DTFE_GPU_CHECKPOINT_CHUNKS chunks; a failed buffer (which may have deposited part or all of its atomics)
// restores the last snapshot and continues from its chunk, instead of redoing the WHOLE partition deposit (up
// to ~2000 s per TNG50 partition). The policy's pieces (GpuFault, the pace, the fault hook, GpuSnapshot) live
// in gpu_metal_retry.h, shared with the standard-DTFE host (dtfe_metal_host.cc); the pass loop is
// psGpuDepositFields' runChunked below.
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

    // the plan (the selected grids, their fixed-point form and device sizes; gpu_host.h, shared with ps_gpu_cuda.cu).
    // PS_METAL_FLOAT_MASS=1 keeps float mass atomics, for comparison.
    bool const floatMass = getenv("PS_METAL_FLOAT_MASS") && atoi(getenv("PS_METAL_FLOAT_MASS")) != 0;
    PSGpuPlan const pl = psGpuPlan(subDims, fVel, fDisp, fGrad, fVolW, fCaustic, fExact, fScal, fSGrad, masses,
                                   massExtra, fxUnits, floatMass);
    size_t const nCell = pl.nCell;
    psGpuInitGrids(out, pl, fxUnits);
    deferred.clear();
    if (masses.empty())                 // nothing to deposit (empty partition; the caller may still add released tets)
    {
        psGpuZeroWords(out, pl);
        return true;
    }
    if (verts.size() != masses.size() * size_t(PS_TET_STRIDE))
    {
        err = "tet record size mismatch (verts must hold PS_TET_STRIDE floats per tet)";
        return false;
    }

    DepositParams P = psGpuParams(pl, fxUnits, boxLo, dx, nGrid, subOrigin, subDims, nSub, periodic, fLinear,
                                  windowMode, not dens.empty(), not scal.empty(), masses.size());

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
    MTL::Buffer* bMass = zeroBuf(pl.massBytes);
    MTL::Buffer* bMassFx = zeroBuf(pl.massFxBytes);
    MTL::Buffer* bMom  = zeroBuf(pl.momBytes);
    MTL::Buffer* bM2   = zeroBuf(pl.m2Bytes);
    MTL::Buffer* bGrad = zeroBuf(pl.gradBytes);
    MTL::Buffer* bStr  = zeroBuf(nCell * 4);
    MTL::Buffer* bMomW  = zeroBuf(pl.momwBytes);
    MTL::Buffer* bCaust = zeroBuf(pl.caustBytes);
    MTL::Buffer* bSV    = zeroBuf(pl.svBytes);
    MTL::Buffer* bDV    = zeroBuf(pl.dvBytes);
    MTL::Buffer* bDW    = zeroBuf(pl.dwBytes);
    MTL::Buffer* bScal  = zeroBuf(pl.scalBytes);
    MTL::Buffer* bSG    = zeroBuf(pl.sgBytes);
    // the non-finite guard's counters of depositFields (ps_deposit.metal FieldGrids.bad), one word per slot
    MTL::Buffer* bBad   = zeroBuf(FX_DROP_SLOTS * 4);
    if (!bV || !bU || !bM || !bD || !bSc || !bP || !bMass || !bMassFx || !bMom || !bM2 || !bGrad || !bStr || !bMomW || !bCaust || !bSV || !bDV || !bDW || !bScal || !bSG || !bBad)
    {
        err = "Metal buffer allocation failed (out of GPU-visible memory?)";
        for (MTL::Buffer* b : {bV,bU,bM,bD,bSc,bP,bMass,bMassFx,bMom,bM2,bGrad,bStr,bMomW,bCaust,bSV,bDV,bDW,bScal,bSG,bBad}) if (b) b->release();
        pool->release();
        return false;
    }

    // Dispatch in CHUNKS, one short command buffer each, fully serialized with a host-side gap between
    // buffers: the macOS GPU watchdog ("Impacting Interactivity") kills a long buffer when the display needs
    // the GPU, and back-to-back buffers too, even ~0.1 s ones -- it reacts to starvation; the gaps let
    // WindowServer in. The chunk size adapts toward pace.targetSec. A killed buffer may have deposited part of
    // its atomics, so a failure resumes from the pass's last snapshot (runChunked below); the caller deposits
    // the partition on the CPU only after a fatal error, 3 failures of one chunk or 16 in the deposit, or
    // when no snapshot can be held. PS_METAL_CHUNK=<n> sets the per-tet kernel's starting chunk size.
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
    // The shares the non-finite guard dropped in this deposit, by slot (ps_deposit_params.h
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

    // the moment grids' word views: each fixed-point grid's buffer again at 22 + its slot, where the
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

    // --ps-exact-deposit: split the work below the tet (depositExactItems in ps_deposit.metal), items of
    // blockCells cells of a tet's window (psExactItemBounds in gpu_host.h, shared with ps_gpu_cuda.cu).
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
        std::vector<uint64_t> bound;
        double const total = psExactItemBounds(static_cast<float const*>(bV->contents()), nTetTotal, P, nGrid,
                                               periodic, windowMode, bound, blockCells, nItems);
        if (timing)
            fprintf(stderr, "PS-DTFE Metal: exact items: %zu tets, %zu items of %u cells, window bound %.3g cells\n",
                    nTetTotal, nItems, blockCells, total);
        bItems = c.dev->newBuffer(std::max<size_t>(nItems, 1) * 2 * sizeof(uint32_t), MTL::ResourceStorageModeShared);
        // the per-tet weight totals: a float each, or (sumWFixed) two uint words in the tet's own unit
        // (+ the non-finite guard's counter words of the items kernel, at ExactItemParams.badAt)
        bSumW  = c.dev->newBuffer(std::max<size_t>(nTetTotal, 1) * (pl.sumWFixed ? 8 : sizeof(float)) + FX_DROP_SLOTS * 4,
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
    uint32_t const badAt = uint32_t(std::max<size_t>(nTetTotal, 1) * (pl.sumWFixed ? 2 : 1));   // past the per-tet totals
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
            if (mb[t] > 0.f && !(pl.sumWFixed ? swx[t] > 0 : sw[t] > 0.f)) left.push_back(uint32_t(t));
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
        std::memset(bSumW->contents(), 0, nTetTotal * (pl.sumWFixed ? 8 : sizeof(float)));
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

    if (pl.massFixed)
    {
        // the words themselves, not their float decode: adding the deferred tets' shares into decoded floats
        // would round away every share below half an ulp of a busy cell, always downward (1.6e-5 of the box
        // mass at TNG50-3-Dark z=0.5). Low word first: on this little-endian host each pair is the int64
        // (ps_deposit_params.h).
        out.massFx.resize(nCell);
        std::memcpy(out.massFx.data(), bMassFx->contents(), nCell * 8);
        if (massAudit)
        {   // the exact integer total (the partition's mass is below 2^62 units: no overflow)
            int64_t q = 0;
            for (size_t i = 0; i < nCell; ++i) q += out.massFx[i];
            out.auditFixedTotal = psMassFixedMass(q, pl.massScale);
        }
    }
    else
        std::memcpy(out.mass.data(), bMass->contents(), nCell * 4);
    // each moment grid: its words (fixed point; each low/high pair IS the int64 on this little-endian
    // host, as for the mass) or its floats
    MTL::Buffer* const fxBuf[PS_FX_N] = { bMom, bM2, bGrad, bMomW, bSV, bDV, nullptr, bScal, bSG };
    std::vector<float>* const flt[PS_FX_N] = { &out.mom, &out.m2, &out.grad, &out.momw, &out.streamvol, &out.dispvel,
                                               nullptr, &out.scal, &out.sgrad };
    for (int g = 0; g < PS_FX_N; ++g)
    {
        if (!pl.sel[g] || !fxBuf[g]) continue;
        size_t const n = nCell * PS_FX_COMPS[g];
        if (pl.fxOn[g]) { out.fx[g].resize(n); std::memcpy(out.fx[g].data(), fxBuf[g]->contents(), n * 8); }
        else std::memcpy(flt[g]->data(), fxBuf[g]->contents(), n * 4);
    }
    std::memcpy(out.streams.data(), bStr->contents(),  nCell * 4);
    if (fCaustic) std::memcpy(out.caustic.data(), bCaust->contents(), nCell * 4);
    if (pl.fDispOwn && !pl.dispwIsMass) std::memcpy(out.dispw.data(), bDW->contents(), nCell * 4);

    for (MTL::Buffer* b : {bV,bU,bM,bD,bSc,bP,bMass,bMassFx,bMom,bM2,bGrad,bStr,bMomW,bCaust,bSV,bDV,bDW,bScal,bSG,bBad}) b->release();
    pool->release();
    return true;
}
