/* Metal backend of the standard-DTFE method-1 deposit (implements gpu_host.h). Owns the
   metal-cpp implementation symbols for the DTFE binary; compiled only when METAL=1 (which
   defines DTFE_GPU for the callers). The kernel source is embedded at build time
   (o/dtfe_deposit_msl.h, generated from metal/dtfe_deposit.metal) and runtime-compiled once
   per process; the pipeline state is cached. The chunked, watchdog-safe dispatch discipline
   is the one validated in ps_metal_host.cc -- see the long comment at the dispatch loop. */

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <unistd.h>

#include "gpu_host.h"
#include "dtfe_deposit_msl.h"   // generated: static const char DTFE_DEPOSIT_MSL[]
#include "../message.h"         // ANSI colour helpers only (retry warning below)

namespace {

// must match the MSL DepositParams byte-for-byte
struct DepositParams
{
    float    dx[3];
    int32_t  nGrid[3];
    uint32_t nTet;
    int32_t  fDen;
    int32_t  fVel;
    int32_t  fGrad;
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
        NS::String::string(DTFE_DEPOSIT_MSL, NS::UTF8StringEncoding),
        (MTL::CompileOptions*)nullptr, &nsErr);
    if (!lib)
    {
        c.err = std::string("kernel compile failed: ")
              + (nsErr ? nsErr->localizedDescription()->utf8String() : "unknown");
        return c;
    }
    MTL::Function* fn = lib->newFunction(NS::String::string("depositAveraged1", NS::UTF8StringEncoding));
    if (!fn) { c.err = "kernel 'depositAveraged1' not found"; return c; }
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


std::string gpuBackendName() { return "Metal"; }

std::string gpuDeviceName()
{
    std::lock_guard<std::mutex> lock(ctxMutex());
    Ctx& c = ctx();
    return c.dev ? std::string(c.dev->name()->utf8String()) : std::string();
}


bool dtfeGpuDepositAveraged1(std::vector<float>& verts,
                               std::vector<float>& dens,
                               std::vector<float>& vels,
                               std::vector<float>& vols,
                               std::vector<uint32_t>& cnts,
                               std::vector<uint32_t>& flats,
                               const std::vector<float>& sobol,
                               const double dx[3], const size_t nGrid[3],
                               bool fDen, bool fVel, bool fGrad,
                               DTFEGpuGrids& out, std::string& err)
{
    std::lock_guard<std::mutex> lock(ctxMutex());   // serialize dispatches (single queue)
    Ctx& c = ctx();
    if (!c.ready) { err = c.err; return false; }

    size_t const nCell = nGrid[0] * nGrid[1] * nGrid[2];
    out.density.assign(fDen ? nCell : 0, 0.f);
    out.velocity.assign(fVel ? nCell * 3 : 0, 0.f);
    out.grad.assign(fGrad ? nCell * 9 : 0, 0.f);
    if (cnts.empty()) return true;    // nothing to deposit (no cells in the region)

    bool const needVel = fVel || fGrad;
    DepositParams P{};
    for (int d = 0; d < 3; ++d)
    {
        P.dx[d]    = float(dx[d]);
        P.nGrid[d] = int32_t(nGrid[d]);
    }
    P.nTet  = uint32_t(cnts.size());
    P.fDen  = fDen ? 1 : 0;
    P.fVel  = fVel ? 1 : 0;
    P.fGrad = fGrad ? 1 : 0;

    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

    // newBuffer(ptr,...) copies the host array into the MTL buffer; free each host array
    // IMMEDIATELY so at most one array is double-held at a time (instead of all six, a
    // multi-GB difference per call at production grids). On failure the CPU fallback
    // re-interpolates from the triangulation, never from these arrays.
    size_t const nTetTotal = cnts.size();
    MTL::Buffer* bV   = c.dev->newBuffer(verts.data(), verts.size() * sizeof(float), MTL::ResourceStorageModeShared);
    std::vector<float>().swap(verts);
    MTL::Buffer* bD   = c.dev->newBuffer(dens.data(),  dens.size()  * sizeof(float), MTL::ResourceStorageModeShared);
    std::vector<float>().swap(dens);
    // a 4-byte dummy stands in for unused inputs/outputs; the kernel never touches them (flags)
    MTL::Buffer* bU   = needVel
                      ? c.dev->newBuffer(vels.data(), vels.size() * sizeof(float), MTL::ResourceStorageModeShared)
                      : c.dev->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);
    std::vector<float>().swap(vels);
    MTL::Buffer* bVol = c.dev->newBuffer(vols.data(),  vols.size()  * sizeof(float),    MTL::ResourceStorageModeShared);
    std::vector<float>().swap(vols);
    MTL::Buffer* bCnt = c.dev->newBuffer(cnts.data(),  cnts.size()  * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    std::vector<uint32_t>().swap(cnts);
    MTL::Buffer* bFlt = c.dev->newBuffer(flats.data(), flats.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    std::vector<uint32_t>().swap(flats);
    MTL::Buffer* bS   = c.dev->newBuffer(sobol.data(), sobol.size() * sizeof(float),    MTL::ResourceStorageModeShared);
    size_t const denBytes  = fDen  ? nCell * 4  : 4;
    size_t const velBytes  = fVel  ? nCell * 12 : 4;
    size_t const gradBytes = fGrad ? nCell * 36 : 4;
    auto zeroBuf = [&](size_t bytes) {
        MTL::Buffer* b = c.dev->newBuffer(bytes, MTL::ResourceStorageModeShared);
        if (b) std::memset(b->contents(), 0, bytes);
        return b;
    };
    MTL::Buffer* bDen  = zeroBuf(denBytes);
    MTL::Buffer* bVel  = zeroBuf(velBytes);
    MTL::Buffer* bGrad = zeroBuf(gradBytes);
    MTL::Buffer* bP    = c.dev->newBuffer(&P, sizeof(P), MTL::ResourceStorageModeShared);
    if (!bV || !bD || !bU || !bVol || !bCnt || !bFlt || !bS || !bDen || !bVel || !bGrad || !bP)
    {
        err = "Metal buffer allocation failed (out of GPU-visible memory?)";
        for (MTL::Buffer* b : {bV,bD,bU,bVol,bCnt,bFlt,bS,bDen,bVel,bGrad,bP}) if (b) b->release();
        pool->release();
        return false;
    }

    // Chunked, serialized dispatch with host-side gaps -- the watchdog-safe discipline
    // validated for the PS deposit (see ps_metal_host.cc for the full rationale). A killed
    // command buffer may have partially deposited its atomics, so a failed chunk is never
    // retried alone: each retry re-zeros the grids and redoes the whole deposit with shorter
    // buffers and wider gaps. DTFE_METAL_CHUNK=<n> overrides the starting chunk size.
    size_t const MIN_CHUNK = 2000;
    size_t const MAX_CHUNK = 500000;
    double targetSec = 0.25;
    useconds_t gapUs = 8000;
    size_t baseChunk = 25000;
    if (const char* env = getenv("DTFE_METAL_CHUNK"))
    { long v = atol(env); if (v > 0) baseChunk = size_t(v); }

    NS::UInteger tg = c.pso->maxTotalThreadsPerThreadgroup();
    if (tg > 256) tg = 256;
    int const maxAttempts = 3;
    bool success = false;
    std::string lastErr;

    for (int attempt = 0; attempt < maxAttempts && !success; ++attempt)
    {
        std::memset(bDen->contents(),  0, denBytes);
        std::memset(bVel->contents(),  0, velBytes);
        std::memset(bGrad->contents(), 0, gradBytes);

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
            enc->setBuffer(bV,   start * 12 * sizeof(float), 0);
            enc->setBuffer(bD,   start * 4 * sizeof(float),  1);
            enc->setBuffer(bU,   needVel ? start * 12 * sizeof(float) : 0, 2);
            enc->setBuffer(bVol, start * sizeof(float),      3);
            enc->setBuffer(bCnt, start * sizeof(uint32_t),   4);
            enc->setBuffer(bFlt, start * sizeof(uint32_t),   5);
            enc->setBuffer(bS,   0, 6);
            enc->setBuffer(bDen, 0, 7);
            enc->setBuffer(bVel, 0, 8);
            enc->setBuffer(bGrad,0, 9);
            enc->setBuffer(bP,   0, 10);
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
            targetSec = std::max(targetSec / 4.0, 0.02);
            gapUs     = std::min(gapUs * 3, useconds_t(50000));
            baseChunk = std::max(baseChunk / 2, MIN_CHUNK);
            if (attempt + 1 < maxAttempts)
            {
                fprintf(stderr, "%sDTFE Metal: %s -- retrying the deposit from scratch "
                                "(attempt %d/%d, target %.0f ms buffers, %.0f ms gaps)%s\n",
                        MESSAGE::cYellow(), lastErr.c_str(), attempt + 2, maxAttempts,
                        targetSec*1e3, gapUs/1e3, MESSAGE::cReset());
                sleep(2);   // let the system settle before hammering the GPU again
            }
        }
    }
    if (!success)
    {
        err = lastErr;
        for (MTL::Buffer* b : {bV,bD,bU,bVol,bCnt,bFlt,bS,bDen,bVel,bGrad,bP}) b->release();
        pool->release();
        return false;
    }

    if (fDen)  std::memcpy(out.density.data(),  bDen->contents(),  nCell * 4);
    if (fVel)  std::memcpy(out.velocity.data(), bVel->contents(),  nCell * 12);
    if (fGrad) std::memcpy(out.grad.data(),     bGrad->contents(), nCell * 36);

    for (MTL::Buffer* b : {bV,bD,bU,bVol,bCnt,bFlt,bS,bDen,bVel,bGrad,bP}) b->release();
    pool->release();
    return true;
}
