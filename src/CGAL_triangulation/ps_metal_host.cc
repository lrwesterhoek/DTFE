/* Metal host for the PS-DTFE deposit (see ps_metal_host.h). Owns the metal-cpp implementation
   symbols for the whole binary; compiled only when METAL=1 (-DPS_METAL). The kernel source is
   embedded at build time (o_ps/ps_deposit_msl.h, generated from metal/ps_deposit.metal) and
   runtime-compiled once per process; the pipeline state is cached. */

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <unistd.h>

#include "ps_metal_host.h"
#include "ps_deposit_msl.h"   // generated: static const char PS_DEPOSIT_MSL[]

namespace {

// must match the MSL DepositParams byte-for-byte
struct DepositParams
{
    float    boxLo[3];
    float    dx[3];
    int32_t  nGrid[3];
    int32_t  nSub;
    int32_t  periodic;
    int32_t  subOrigin[3];
    int32_t  subDims[3];
    uint32_t nTet;
};

struct Ctx
{
    MTL::Device*               dev  = nullptr;
    MTL::CommandQueue*         q    = nullptr;
    MTL::ComputePipelineState* pso  = nullptr;
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
    MTL::Function* fn = lib->newFunction(NS::String::string("depositFields", NS::UTF8StringEncoding));
    if (!fn) { c.err = "kernel 'depositFields' not found"; return c; }
    c.pso = c.dev->newComputePipelineState(fn, &nsErr);
    if (!c.pso)
    {
        c.err = std::string("pipeline state failed: ")
              + (nsErr ? nsErr->localizedDescription()->utf8String() : "unknown");
        return c;
    }
    c.ready = true;
    return c;
}

} // namespace


std::string psMetalDeviceName()
{
    std::lock_guard<std::mutex> lock(ctxMutex());
    Ctx& c = ctx();
    return c.dev ? std::string(c.dev->name()->utf8String()) : std::string();
}


bool psMetalDepositFields(const std::vector<float>& verts,
                          const std::vector<float>& vels,
                          const std::vector<float>& masses,
                          const double boxLo[3], const double dx[3],
                          const size_t nGrid[3], const size_t subOrigin[3], const size_t subDims[3],
                          int nSub, bool periodic,
                          PSMetalGrids& out, std::string& err)
{
    std::lock_guard<std::mutex> lock(ctxMutex());   // serialize dispatches (single queue)
    Ctx& c = ctx();
    if (!c.ready) { err = c.err; return false; }

    size_t const nCell = subDims[0] * subDims[1] * subDims[2];
    out.mass.assign(nCell, 0.f);
    out.mom.assign(nCell * 3, 0.f);
    out.m2.assign(nCell * 6, 0.f);
    out.grad.assign(nCell * 9, 0.f);
    out.streams.assign(nCell, 0u);
    if (masses.empty()) return true;    // nothing to deposit (empty partition)

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
    P.nTet     = uint32_t(masses.size());

    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

    MTL::Buffer* bV = c.dev->newBuffer(verts.data(),  verts.size()  * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* bU = c.dev->newBuffer(vels.data(),   vels.size()   * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* bM = c.dev->newBuffer(masses.data(), masses.size() * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* bP = c.dev->newBuffer(&P, sizeof(P), MTL::ResourceStorageModeShared);
    auto zeroBuf = [&](size_t bytes) {
        MTL::Buffer* b = c.dev->newBuffer(bytes, MTL::ResourceStorageModeShared);
        if (b) std::memset(b->contents(), 0, bytes);
        return b;
    };
    MTL::Buffer* bMass = zeroBuf(nCell * 4);
    MTL::Buffer* bMom  = zeroBuf(nCell * 12);
    MTL::Buffer* bM2   = zeroBuf(nCell * 24);
    MTL::Buffer* bGrad = zeroBuf(nCell * 36);
    MTL::Buffer* bStr  = zeroBuf(nCell * 4);
    if (!bV || !bU || !bM || !bP || !bMass || !bMom || !bM2 || !bGrad || !bStr)
    {
        err = "Metal buffer allocation failed (out of GPU-visible memory?)";
        for (MTL::Buffer* b : {bV,bU,bM,bP,bMass,bMom,bM2,bGrad,bStr}) if (b) b->release();
        pool->release();
        return false;
    }

    // Dispatch in CHUNKS of tetrahedra, one short command buffer each, FULLY SERIALIZED with a
    // small host-side gap between buffers. Rationale (learned the hard way):
    //  - one monolithic buffer runs minutes -> killed by the macOS GPU watchdog ("Impacting
    //    Interactivity") when the display needs the GPU;
    //  - keeping buffers back-to-back (pipelined) sustains 100% GPU queue pressure and gets killed
    //    EVEN WITH ~0.1 s buffers under active display use -- the watchdog reacts to starvation,
    //    not just per-buffer duration. The gaps are what let WindowServer breathe.
    // Chunk size adapts toward targetSec (heavy-tailed per-tet cost on real data); a killed buffer
    // may have PARTIALLY deposited its atomics, so a failed chunk is never retried alone: each
    // retry re-zeros the grids, redoes the whole deposit with 4x shorter buffers and wider gaps,
    // and persistent failure falls back to the CPU deposit in the caller.
    // PS_METAL_CHUNK=<n> overrides the starting chunk size.
    size_t const MIN_CHUNK = 2000;
    size_t const MAX_CHUNK = 500000;
    double targetSec = 0.25;
    useconds_t gapUs = 8000;          // ~3% overhead at 0.25 s buffers
    size_t baseChunk = 25000;
    if (const char* env = getenv("PS_METAL_CHUNK"))
    { long v = atol(env); if (v > 0) baseChunk = size_t(v); }

    NS::UInteger tg = c.pso->maxTotalThreadsPerThreadgroup();
    if (tg > 256) tg = 256;
    size_t const nTetTotal = masses.size();
    int const maxAttempts = 3;
    bool success = false;
    std::string lastErr;

    for (int attempt = 0; attempt < maxAttempts && !success; ++attempt)
    {
        std::memset(bMass->contents(), 0, nCell * 4);
        std::memset(bMom->contents(),  0, nCell * 12);
        std::memset(bM2->contents(),   0, nCell * 24);
        std::memset(bGrad->contents(), 0, nCell * 36);
        std::memset(bStr->contents(),  0, nCell * 4);

        success = true;
        size_t chunk = baseChunk;
        size_t start = 0;
        while (start < nTetTotal)
        {
            uint32_t const n = uint32_t(std::min(chunk, nTetTotal - start));
            P.nTet = n;
            std::memcpy(bP->contents(), &P, sizeof(P));   // safe: previous chunk completed

            auto t0 = std::chrono::steady_clock::now();
            MTL::CommandBuffer* cb = c.q->commandBuffer();
            MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
            enc->setComputePipelineState(c.pso);
            enc->setBuffer(bV,    start * 12 * sizeof(float), 0);
            enc->setBuffer(bU,    start * 12 * sizeof(float), 1);
            enc->setBuffer(bM,    start * sizeof(float),      2);
            enc->setBuffer(bMass, 0, 3);
            enc->setBuffer(bMom,  0, 4);
            enc->setBuffer(bM2,   0, 5);
            enc->setBuffer(bGrad, 0, 6);
            enc->setBuffer(bStr,  0, 7);
            enc->setBuffer(bP,    0, 8);
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
                success = false;
                break;      // restart the WHOLE deposit (grids re-zeroed above)
            }

            start += n;
            if (el > 1e-4)   // retarget the next chunk toward targetSec (bounded rescale)
            {
                double scale = targetSec / el;
                if (scale < 0.5) scale = 0.5;
                if (scale > 4.0) scale = 4.0;
                chunk = std::min(std::max(size_t(double(chunk) * scale), MIN_CHUNK), MAX_CHUNK);
            }
            usleep(gapUs);   // deliberate idle gap: lets WindowServer take the GPU
        }

        if (!success)
        {
            // escalate: 4x shorter buffers, wider gaps, smaller starting chunk; settle first
            targetSec = std::max(targetSec / 4.0, 0.02);
            gapUs     = std::min(gapUs * 3, useconds_t(50000));
            baseChunk = std::max(baseChunk / 2, MIN_CHUNK);
            if (attempt + 1 < maxAttempts)
            {
                fprintf(stderr, "PS-DTFE Metal: %s -- retrying the partition deposit from scratch "
                                "(attempt %d/%d, target %.0f ms buffers, %.0f ms gaps)\n",
                        lastErr.c_str(), attempt + 2, maxAttempts, targetSec*1e3, gapUs/1e3);
                sleep(2);   // let the system settle before hammering the GPU again
            }
        }
    }
    if (!success)
    {
        err = lastErr;
        for (MTL::Buffer* b : {bV,bU,bM,bP,bMass,bMom,bM2,bGrad,bStr}) b->release();
        pool->release();
        return false;
    }

    std::memcpy(out.mass.data(),    bMass->contents(), nCell * 4);
    std::memcpy(out.mom.data(),     bMom->contents(),  nCell * 12);
    std::memcpy(out.m2.data(),      bM2->contents(),   nCell * 24);
    std::memcpy(out.grad.data(),    bGrad->contents(), nCell * 36);
    std::memcpy(out.streams.data(), bStr->contents(),  nCell * 4);

    for (MTL::Buffer* b : {bV,bU,bM,bP,bMass,bMom,bM2,bGrad,bStr}) b->release();
    pool->release();
    return true;
}
