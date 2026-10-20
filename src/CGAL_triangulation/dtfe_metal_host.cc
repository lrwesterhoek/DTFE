/* Metal backend of the standard-DTFE method-1 deposit (implements gpu_host.h). Owns the
   metal-cpp implementation symbols for the DTFE binary; compiled only when METAL=1 (which
   defines DTFE_GPU for the callers). The kernel source is embedded at build time
   (o/dtfe_deposit_msl.h, generated from metal/dtfe_deposit.metal) and runtime-compiled once
   per process; the pipeline states are cached.

   Two ways in. dtfeGpuDepositAveraged1() deposits one tetrahedron set into grids of its own
   (the per-call path). The dtfeGpuShared* family lets the sub-domains of one
   DTFE_parallel call deposit into ONE full grid: each thread uploads its batch (concurrently),
   the dispatches queue on the device (serialized by ctxMutex, the chunk controller warm across
   batches), and the main thread reads the full grid back once. Both use the same Batch /
   runBatch machinery below; a batch's owner box is zeroed before a retry from its start, so a
   batch can be redone or handed back to the CPU without touching the other sub-domains' cells.

   The chunked, watchdog-safe dispatch discipline is the one validated in ps_metal_host.cc --
   see the long comment at the dispatch loop -- and so is its retry policy: chunk
   checkpoints, the cause of a failed buffer, the process-wide pace that eases back after a clean
   stretch, the run's retry counters (gpu_metal_retry.h, shared with that host). The grids are
   exact 64-bit fixed-point sums (metal/dtfe_deposit.metal's header; the units at
   batchFixedScales), so two runs write byte-identical '_a' grids. */

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <vector>
#include <unistd.h>
#include <sys/sysctl.h>

#include "gpu_host.h"
#include "dtfe_gpu_fixed.h"     // batchFixedScales + psMassFixedMass: the exact sums' units (shared with dtfe_gpu_cuda.cu)
#include "dtfe_deposit_msl.h"   // generated: static const char DTFE_DEPOSIT_MSL[]
#define ITEM_BLOCK_SHIFT 5      // == the kernel's: log2 of the items per entry of the block table
#include "../message.h"         // ANSI colour helpers only (retry warning below)
extern std::atomic<int> dtfeGpuRetries;            // averaged_interpolation_1.cc (run_progress.h)
extern std::atomic<int> dtfeGpuRetriesByCause[4];  // ... by cause: innocent, timeout, oom, other

namespace {

// must match the MSL DepositParams byte-for-byte
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
    uint32_t tableEntries;  // item kernel: threadgroup cell-table entries (power of two; 0 = none)
    int32_t  fScal;         // the scalar field (per-tet kernel; the item pipelines are specialized on it)
    int32_t  fSGrad;        // its gradient
    uint32_t fixedSum;      // the grids as exact 64-bit fixed-point sums (batchFixedScales)
    float    fxScale[5];    // their units 2^k: density, velocity, velocity gradient, scalar, scalar gradient
};

// The checkpoints and the cause-aware retry policy of the GPU deposit (shared with ps_metal_host.cc, see
// gpu_metal_retry.h): GpuFault, gpuRetryCause, the process-wide GpuPace, the DTFE_GPU_FAULT_AT test hook and
// GpuSnapshot. The pass loop is runBatch below.
#include "gpu_metal_retry.h"

// the run's retry counters (the progress sidecar's gpu_retries / gpu_retries_by_cause and DTFE.cpp's closing
// 'GPU deposit retries:' line), by gpuRetryCause's index; averaged_interpolation_1.cc defines them
void countGpuRetry(GpuFault f, long code, std::string const& desc)
{
    int const k = gpuRetryCause(f, code, desc);
    ++dtfeGpuRetries;
    ++dtfeGpuRetriesByCause[k];
}

// Chunks start at multiples of this many work items: with the threadgroups (<= 256 threads, a
// multiple of the 32-lane SIMD width) also starting there, a SIMD group is always the same 32 items, so its
// float simd_sum -- the item kernel's flush, which the exact sums convert only after -- does not depend on where
// the timing controller cut the chunks.
size_t const CHUNK_ALIGN = 256;
size_t alignChunkUp(size_t n) { return (n + CHUNK_ALIGN - 1) / CHUNK_ALIGN * CHUNK_ALIGN; }

struct Ctx
{
    MTL::Device*               dev  = nullptr;
    MTL::CommandQueue*         q    = nullptr;
    MTL::Library*              lib  = nullptr;
    MTL::ComputePipelineState* pso  = nullptr;        // depositAveraged1: one thread per tetrahedron
    MTL::ComputePipelineState* psoExact = nullptr;    // depositExactAverage: one thread per (tet, cell block)
    // depositAveraged1Items{2,4,8}: one thread per (tet, sample run) with a 2/4/8-slot cell cache,
    // specialized per field set (function constants FC_DEN/FC_VEL/FC_GRAD/FC_SCAL/FC_SGRAD) and on the
    // exact sums (FC_FIXED, mask bit 32): index [K slot][mask]
    MTL::ComputePipelineState* psoItems[3][64] = {{nullptr}};
    bool                       itemsOk = false;
    std::string                err;
    bool                       ready = false;
};

int kSlot(int cacheK) { return cacheK == 2 ? 0 : cacheK == 8 ? 2 : 1; }

std::mutex& ctxMutex() { static std::mutex m; return m; }
bool timingOn() { static bool t = getenv("DTFE_METAL_TIMING") != nullptr; return t; }

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
    c.lib = lib;
    MTL::Function* fn = lib->newFunction(NS::String::string("depositAveraged1", NS::UTF8StringEncoding));
    if (!fn) { c.err = "kernel 'depositAveraged1' not found"; return c; }
    c.pso = c.dev->newComputePipelineState(fn, &nsErr);
    if (!c.pso)
    {
        c.err = std::string("pipeline state failed (depositAveraged1): ")
              + (nsErr ? nsErr->localizedDescription()->utf8String() : "unknown");
        return c;
    }
    MTL::Function* fe = lib->newFunction(NS::String::string("depositExactAverage", NS::UTF8StringEncoding));
    if (!fe) { c.err = "kernel 'depositExactAverage' not found"; return c; }
    c.psoExact = c.dev->newComputePipelineState(fe, &nsErr);
    if (!c.psoExact)
    {
        c.err = std::string("pipeline state failed (depositExactAverage): ")
              + (nsErr ? nsErr->localizedDescription()->utf8String() : "unknown");
        return c;
    }
    c.itemsOk = true;      // the specialized item pipelines are built on first use (itemPso)
    c.ready = true;
    return c;
}

// The item pipeline for a cache size and field set, built on first use (thread-safe). Null when it
// cannot be built: the caller falls back to the per-tet kernel.
MTL::ComputePipelineState* itemPso(Ctx& c, int cacheK, bool fDen, bool fVel, bool fGrad, bool fScal, bool fSGrad,
                                   bool fFixed)
{
    static std::mutex psoMutex;              // uploads (and so first uses) run concurrently
    std::lock_guard<std::mutex> lock(psoMutex);
    int const ks = kSlot(cacheK), mask = (fDen ? 1 : 0) | (fVel ? 2 : 0) | (fGrad ? 4 : 0) | (fScal ? 8 : 0) | (fSGrad ? 16 : 0)
                                         | (fFixed ? 32 : 0);
    if (c.psoItems[ks][mask]) return c.psoItems[ks][mask];
    if (!c.itemsOk || !c.lib) return nullptr;
    char name[64];
    snprintf(name, sizeof(name), "depositAveraged1Items%d", cacheK == 2 ? 2 : cacheK == 8 ? 8 : 4);
    MTL::FunctionConstantValues* fc = MTL::FunctionConstantValues::alloc()->init();
    bool d = fDen, v = fVel, g = fGrad, s = fScal, sg = fSGrad, fx = fFixed;
    fc->setConstantValue(&d, MTL::DataTypeBool, NS::UInteger(0));
    fc->setConstantValue(&v, MTL::DataTypeBool, NS::UInteger(1));
    fc->setConstantValue(&g, MTL::DataTypeBool, NS::UInteger(2));
    fc->setConstantValue(&s, MTL::DataTypeBool, NS::UInteger(3));
    fc->setConstantValue(&sg, MTL::DataTypeBool, NS::UInteger(4));
    fc->setConstantValue(&fx, MTL::DataTypeBool, NS::UInteger(5));
    NS::Error* nsErr = nullptr;
    MTL::Function* fn = c.lib->newFunction(NS::String::string(name, NS::UTF8StringEncoding), fc, &nsErr);
    fc->release();
    if (!fn)
    {
        fprintf(stderr, "DTFE Metal: item kernel %s could not be specialized (%s); using the per-tetrahedron kernel\n",
                name, nsErr ? nsErr->localizedDescription()->utf8String() : "unknown");
        c.itemsOk = false;
        return nullptr;
    }
    MTL::ComputePipelineState* pso = c.dev->newComputePipelineState(fn, &nsErr);
    if (!pso)
    {
        fprintf(stderr, "DTFE Metal: pipeline for %s failed (%s); using the per-tetrahedron kernel\n",
                name, nsErr ? nsErr->localizedDescription()->utf8String() : "unknown");
        c.itemsOk = false;
        return nullptr;
    }
    if (timingOn())   // 1024 = lean kernel; a lower cap means the registers limit occupancy
        fprintf(stderr, "DTFE Metal pipeline %s (den %d vel %d grad %d scal %d sgrad %d fixed %d): max threads per threadgroup %lu (per-tet kernel %lu)\n",
                name, int(d), int(v), int(g), int(s), int(sg), int(fx), (unsigned long)pso->maxTotalThreadsPerThreadgroup(),
                (unsigned long)c.pso->maxTotalThreadsPerThreadgroup());
    c.psoItems[ks][mask] = pso;
    return pso;
}


// ---------------------------------------------------------------- the accumulated grids
struct Grids
{
    MTL::Buffer* den  = nullptr;   // nCell components (or a 4-byte dummy)
    MTL::Buffer* vel  = nullptr;   // nCell*3
    MTL::Buffer* grad = nullptr;   // nCell*9
    MTL::Buffer* scal = nullptr;   // nCell     the scalar
    MTL::Buffer* sgrad = nullptr;  // nCell*3   its gradient
    size_t full[3] = {0, 0, 0};
    bool fDen = false, fVel = false, fGrad = false, fScal = false, fSGrad = false;
    bool fixed = false;            // the components are 8-byte fixed-point words, else floats

    size_t nCell() const { return full[0] * full[1] * full[2]; }
    size_t compBytes() const { return fixed ? 8 : 4; }
    std::vector<MTL::Buffer*> buffers() const { return { den, vel, grad, scal, sgrad }; }
    // gpuZero: zero the buffers with ONE blit on the device (the per-call path allocates under the device mutex,
    // where a CPU memset of a 512^3 sub-domain's exact words -- up to 1.6 GB -- would idle the GPU);
    // a failed blit falls back to the memset
    bool alloc(Ctx& c, const size_t fullGrid[3], bool d, bool v, bool g, bool s, bool sg, bool fx = false,
               bool gpuZero = false)
    {
        for (int i = 0; i < 3; ++i) full[i] = fullGrid[i];
        fDen = d; fVel = v; fGrad = g; fScal = s; fSGrad = sg;
        fixed = fx;
        size_t const cb = compBytes();
        auto newBuf = [&](size_t bytes) {
            MTL::Buffer* b = c.dev->newBuffer(bytes, MTL::ResourceStorageModeShared);
            if (b && !gpuZero) std::memset(b->contents(), 0, bytes);
            return b;
        };
        den  = newBuf(fDen  ? nCell() * cb     : 4);
        vel  = newBuf(fVel  ? nCell() * cb * 3 : 4);
        grad = newBuf(fGrad ? nCell() * cb * 9 : 4);
        scal = newBuf(fScal ? nCell() * cb     : 4);
        sgrad = newBuf(fSGrad ? nCell() * cb * 3 : 4);
        bool const ok = den && vel && grad && scal && sgrad;
        if (ok && gpuZero)
        {
            MTL::CommandBuffer* cbuf = c.q->commandBuffer();
            MTL::BlitCommandEncoder* blit = cbuf->blitCommandEncoder();
            for (MTL::Buffer* b : buffers()) blit->fillBuffer(b, NS::Range::Make(0, b->length()), 0);
            blit->endEncoding();
            cbuf->commit();
            cbuf->waitUntilCompleted();
            if (cbuf->status() == MTL::CommandBufferStatusError)
                for (MTL::Buffer* b : buffers()) std::memset(b->contents(), 0, b->length());
        }
        return ok;
    }
    void release()
    {
        for (MTL::Buffer** b : {&den, &vel, &grad, &scal, &sgrad}) if (*b) { (*b)->release(); *b = nullptr; }
    }
    // zero the cells of a sub-grid box [off, off+dims) -- the owner box of one batch
    void zeroBox(const int32_t off[3], const int32_t dims[3])
    {
        size_t const cb = compBytes();
        auto zero = [&](MTL::Buffer* b, bool on, size_t comps) {
            if (!on || !b) return;
            char* base = static_cast<char*>(b->contents());
            for (int32_t i = 0; i < dims[0]; ++i)
                for (int32_t j = 0; j < dims[1]; ++j)
                {
                    size_t const row = (size_t(off[0] + i) * full[1] + size_t(off[1] + j)) * full[2] + size_t(off[2]);
                    std::memset(base + row * comps * cb, 0, size_t(dims[2]) * comps * cb);
                }
        };
        zero(den, fDen, 1); zero(vel, fVel, 3); zero(grad, fGrad, 9); zero(scal, fScal, 1); zero(sgrad, fSGrad, 3);
    }
};

// ---------------------------------------------------------------- one sub-domain's batch
struct Batch
{
    MTL::Buffer *bV = nullptr, *bD = nullptr, *bU = nullptr, *bVol = nullptr, *bCnt = nullptr,
                *bFlt = nullptr, *bS = nullptr, *bG = nullptr, *bB = nullptr, *bSc = nullptr;
    DepositParams P{};
    size_t nTet = 0, nItems = 0, nSamples = 0, nSingle = 0;
    bool   needVel = false, needScal = false;
    bool   exact = false;          // --exact-average: depositExactAverage over cell-block items
    DTFEGpuKnobs k;                // dtfe_gpu_fixed.h

    void release()
    {
        for (MTL::Buffer** b : {&bV, &bD, &bU, &bVol, &bCnt, &bFlt, &bS, &bG, &bB, &bSc}) if (*b) { (*b)->release(); *b = nullptr; }
    }
};

// Uploads a tetrahedron set (CONSUMING the host arrays one at a time, so at most one is double-
// held) and builds the item group table. No device queue work: safe to run concurrently.
bool uploadBatch(Ctx& c, std::vector<float>& verts, std::vector<float>& dens, std::vector<float>& vels,
                 std::vector<float>& scal,
                 std::vector<float>& vols, std::vector<uint32_t>& cnts, std::vector<uint32_t>& flats,
                 const std::vector<float>& sobol, const double dx[3], const size_t nGrid[3],
                 const size_t subOff[3], const size_t fullGrid[3], bool fDen, bool fVel, bool fGrad,
                 bool fScal, bool fSGrad, bool exact, bool allowFixed, Batch& b, std::string& err)
{
    b.k = dtfeGpuKnobs(c.itemsOk);
    b.exact = exact;
    // the parameter block, with the exact sums' units from this batch's values (dtfe_gpu_fixed.h, shared with
    // dtfe_gpu_cuda.cu)
    DepositParams& P = b.P;
    uint32_t const exactCells = dtfeBatchParams(P, b.k, verts, dens, vels, scal, vols, dx, nGrid, subOff, fullGrid,
                                                fDen, fVel, fGrad, fScal, fSGrad, exact, allowFixed);
    if (exact) b.k.useItems = true;    // the exact kernel is an item kernel (cell blocks), always
    else if (b.k.useItems && !itemPso(c, b.k.cacheK, fDen && !b.k.noWrite, fVel && !b.k.noWrite, fGrad && !b.k.noWrite,
                                      fScal && !b.k.noWrite, fSGrad && !b.k.noWrite, P.fixedSum != 0))
        b.k.useItems = false;      // (built here, before the upload, so a failure falls back cleanly)
    b.needVel = fVel || fGrad;
    b.needScal = fScal || fSGrad;
    b.nTet = cnts.size();
    dtfeShiftFlats(cnts, flats, nGrid, subOff, fullGrid);
    std::vector<uint32_t> itemStart, blockTet;
    P.nTetTotal = uint32_t(b.nTet);
    if (!dtfeItemMap(verts, cnts, dx, nGrid, b.k.useItems, exact, b.k.itemSamples, exactCells, ITEM_BLOCK_SHIFT,
                     itemStart, blockTet, b.nItems, b.nSamples, b.nSingle, err))
        return false;

    // newBuffer(ptr,...) copies the host array into the MTL buffer; free each host array
    // IMMEDIATELY so at most one array is double-held at a time (a multi-GB difference per
    // sub-domain at production grids). On failure the CPU fallback re-interpolates from the
    // triangulation, never from these arrays.
    auto upload = [&](const void* ptr, size_t bytes) {
        return bytes ? c.dev->newBuffer(ptr, bytes, MTL::ResourceStorageModeShared)
                     : c.dev->newBuffer(4, MTL::ResourceStorageModeShared);
    };
    b.bV = upload(verts.data(), verts.size() * sizeof(float));   std::vector<float>().swap(verts);
    b.bD = upload(dens.data(),  dens.size()  * sizeof(float));   std::vector<float>().swap(dens);
    b.bU = b.needVel ? upload(vels.data(), vels.size() * sizeof(float)) : upload(nullptr, 0);
    std::vector<float>().swap(vels);
    b.bSc = b.needScal ? upload(scal.data(), scal.size() * sizeof(float)) : upload(nullptr, 0);
    std::vector<float>().swap(scal);
    b.bVol = upload(vols.data(),  vols.size()  * sizeof(float));    std::vector<float>().swap(vols);
    b.bCnt = upload(cnts.data(),  cnts.size()  * sizeof(uint32_t)); std::vector<uint32_t>().swap(cnts);
    b.bFlt = upload(flats.data(), flats.size() * sizeof(uint32_t)); std::vector<uint32_t>().swap(flats);
    std::vector<float> const sobol4 = dtfeSobol4(sobol);
    b.bS   = upload(sobol4.data(), sobol4.size() * sizeof(float));
    b.bG   = upload(itemStart.data(), itemStart.size() * sizeof(uint32_t));
    b.bB   = upload(blockTet.data(), blockTet.size() * sizeof(uint32_t));
    if (!b.bV || !b.bD || !b.bU || !b.bSc || !b.bVol || !b.bCnt || !b.bFlt || !b.bS || !b.bG || !b.bB)
    {
        err = "Metal buffer allocation failed (out of GPU-visible memory?)";
        b.release();
        return false;
    }
    return true;
}

// The chunk controller's chunk, carried from one batch to the next on the shared grid (a per-call batch starts
// afresh). The pace -- the buffers' target length and the gaps -- is the process-wide GpuPace (gpu_metal_retry.h):
// a per-call pace would die with its sub-domain, and the next one, on the same busy display, would meet the
// watchdog at the full pace again.
struct ChunkState
{
    size_t     chunk     = 0;       // 0 = not started: use the kernel's first chunk
};

struct Stats { int chunks = 0, retries = 0; };

// Runs one batch into the grids (CALLER HOLDS ctxMutex). Chunked, serialized dispatch with host-side gaps --
// the watchdog-safe discipline validated for the PS deposit (see ps_metal_host.cc for the full rationale) --
// and that host's retry policy (its runChunked, here for one pass): a killed command buffer
// may have deposited part or all of its atomics, so it is never retried alone; the batch restores its last
// snapshot and continues from there.
//   - the snapshot: the per-call grids (all zero at the start: no copy) and a checkpoint every ~10 s of GPU work
//     (DTFE_GPU_CHECKPOINT_CHUNKS=<n> buffers); 'sharedBox' (the opt-in shared full grid) restores its owner box
//     to zero, the batch's start, without periodic checkpoints (a copy would be the whole 2 GB grid);
//   - the cause (gpu_metal_retry.h GpuFault): an innocent victim retries at once at the same pace; a watchdog
//     kill, out of memory or another resource error escalates the process-wide pace (4x shorter buffers, wider
//     gaps) and lets the system settle; a fatal error, 3 failures of the same chunk or 16 in the batch hand the
//     sub-domain to the CPU (the caller); the pace eases back after DTFE_GPU_PACE_RECOVER_SEC (90 s) of clean GPU
//     work;
//   - every failed buffer counts into the run's retry counters by cause (countGpuRetry).
// Work items are uniform (<= itemSamples samples each), so their chunks can be larger and the controller never
// meets a 100x jump in per-thread work; the per-tet kernel keeps more cautious sizes. Every chunk
// starts at a multiple of CHUNK_ALIGN items (the exact sums need it, see there).
bool runBatch(Ctx& c, Batch& b, Grids& g, ChunkState& cs, Stats& st, bool sharedBox, std::string& err)
{
    bool const exact = b.exact;
    bool useItems = b.k.useItems;
    // the exact kernel's items clip up to 16 cells each (far more than a sample run): the chunks of
    // the phase-space exact items
    size_t const MIN_CHUNK = exact ? 1024 : useItems ? 4096 : 2048;
    size_t const MAX_CHUNK = exact ? 499968 : useItems ? 4000000 : 499968;   // (multiples of CHUNK_ALIGN)
    size_t firstChunk = exact ? 8192 : useItems ? 65536 : 25088;
    if (const char* env = getenv("DTFE_METAL_CHUNK"))
    { long v = atol(env); if (v > 0) firstChunk = size_t(v); }
    GpuPace& pace = gpuPace();        // targetSec 0.25 s, gapUs 8 ms (~3% overhead) until escalated
    // an escalated pace (an earlier batch's failures) opens with a proportionally smaller chunk
    if (pace.targetSec < GPU_PACE_TARGET0)
        firstChunk = std::max(size_t(double(firstChunk) * pace.targetSec / GPU_PACE_TARGET0), MIN_CHUNK);
    if (cs.chunk == 0) cs.chunk = firstChunk;

    MTL::ComputePipelineState* pso = exact ? c.psoExact
                                   : useItems ? itemPso(c, b.k.cacheK, b.P.fDen != 0, b.P.fVel != 0, b.P.fGrad != 0,
                                                        b.P.fScal != 0, b.P.fSGrad != 0, b.P.fixedSum != 0) : c.pso;
    if (!pso)
    {   // the specialized item kernel is unavailable: the per-tet kernel needs the per-tet item count
        if (exact) { err = "the exact-average kernel is unavailable"; return false; }
        pso = c.pso;
        useItems = false;
    }
    size_t const nThreads = useItems ? b.nItems : b.nTet;
    if (nThreads == 0) return true;
    NS::UInteger tg = pso->maxTotalThreadsPerThreadgroup();
    if (tg > 256) tg = 256;
    // the threadgroup cell table: as many power-of-two entries as fit the 32 KB threadgroup memory
    // (keys 4 B + 4 or 13 values per entry, floats or -- the exact sums -- two-word integers, next to the other
    // mode's 16-byte dummy); DTFE_GPU_TABLE=0 turns it off (evictions go straight to the device)
    uint32_t tableEntries = 0;
    size_t keyBytes = 16, valBytes = 16, valXBytes = 16;
    bool const fixedSum = b.P.fixedSum != 0;
    if (useItems && !exact)
    {   // == the kernel's tableStride(): den, vel, [scalar], [gradient], [scalar gradient]
        size_t const stride = 4 + (b.P.fScal ? 1 : 0) + (b.P.fGrad ? 9 : 0) + (b.P.fSGrad ? 3 : 0);
        size_t const valueBytes = fixedSum ? 8 : 4;
        size_t const budget = std::min<size_t>(pso->staticThreadgroupMemoryLength() < 32768 - 16
                                               ? 32768 - 16 - pso->staticThreadgroupMemoryLength() : 0, 32768);
        uint32_t e = 1;
        while (size_t(e * 2) * (4 + stride * valueBytes) <= budget) e *= 2;
        if (size_t(e) * (4 + stride * valueBytes) > budget) e = 0;
        if (const char* env = getenv("DTFE_GPU_TABLE")) { if (atoi(env) == 0) e = 0; }
        tableEntries = e >= 64 ? e : 0;
        if (tableEntries)
        {
            keyBytes = size_t(tableEntries) * 4;
            (fixedSum ? valXBytes : valBytes) = size_t(tableEntries) * stride * valueBytes;
        }
    }
    b.P.tableEntries = tableEntries;
    MTL::Buffer* bP = c.dev->newBuffer(sizeof(DepositParams), MTL::ResourceStorageModeShared);
    if (!bP) { err = "Metal buffer allocation failed (params)"; return false; }
    // The non-finite guard's counters (metal/dtfe_deposit.metal fxq), one word per grid at buffer 21:
    // zeroed before each command buffer and added into badRun after one that succeeded; a restore rolls badRun
    // back to the held snapshot's (badCk), so a failed buffer and the redone ones count once. Handed to the run's
    // counters (ps_deposit_params.h fixedShareDrops) only when the batch succeeds.
    MTL::Buffer* bBad = c.dev->newBuffer(FX_DROP_SLOTS * 4, MTL::ResourceStorageModeShared);
    if (!bBad) { bP->release(); err = "Metal buffer allocation failed (guard counters)"; return false; }
    uint32_t* const badW = static_cast<uint32_t*>(bBad->contents());
    unsigned long long badRun[FX_DROP_SLOTS] = {}, badCk[FX_DROP_SLOTS] = {};

    // DTFE_GPU_CHECKPOINT_CHUNKS=<n>: a checkpoint every n command buffers (0 = only the batch-start snapshot).
    // Unset: every ~10 s of GPU work (n = 10 s / the current buffer target), as the PS deposit's.
    long ckChunks = -1;
    if (const char* env = getenv("DTFE_GPU_CHECKPOINT_CHUNKS")) { ckChunks = atol(env); if (ckChunks < 0) ckChunks = 0; }
    // DTFE_GPU_PACE_RECOVER_SEC=<s>: the clean GPU work after which an escalated pace steps back (GpuPace)
    double paceRecoverSec = 90.;
    if (const char* env = getenv("DTFE_GPU_PACE_RECOVER_SEC")) { double v = atof(env); if (v >= 0.) paceRecoverSec = v; }
    std::vector<MTL::Buffer*> const gridBufs = g.buffers();
    GpuSnapshot snap;
    // the batch's start: its grids are zero (per call: freshly zeroed; shared: its owner box, never written by
    // another batch), recorded as such without a copy
    bool const ckOn = sharedBox || snap.take(gridBufs, std::vector<bool>(gridBufs.size(), true), 0);
    bool ckPeriodic = false;           // the held snapshot is a periodic checkpoint (not the batch's start)
    int nFailed = 0, nResumedCk = 0;   // failed buffers of this batch, and how many resumed from a periodic checkpoint
    int const MAX_FAILS_SAME_CHUNK = 3, MAX_FAILS_TOTAL = 16;
    // failures of the 'same chunk': a failure that comes before the redo got past the previous failed buffer's
    // end (chunk boundaries move with the timing controller, so a start offset is no identity)
    int  failsHere = 0;
    size_t failEnd = 0;
    // an empty command buffer completes only after everything queued before it: drained before a restore, as the
    // atomics of a killed buffer can land after it otherwise (per-tet runs at 512^3: whole samples in void
    // cells, 2e-2 of max|v|)
    auto drainQueue = [&]() {
        MTL::CommandBuffer* drain = c.q->commandBuffer();
        drain->commit();
        drain->waitUntilCompleted();
    };

    bool success = true;
    std::string lastErr;
    size_t chunk = alignChunkUp(std::min(std::max(cs.chunk, MIN_CHUNK), MAX_CHUNK));
    size_t start = 0, sinceCk = 0;
    while (start < nThreads)
    {
        uint32_t const n = uint32_t(std::min(chunk, nThreads - start));
        b.P.nTet = n;
        b.P.itemBase = uint32_t(start);
        std::memcpy(bP->contents(), &b.P, sizeof(DepositParams));   // safe: previous chunk completed

        std::memset(badW, 0, FX_DROP_SLOTS * 4);                     // (likewise)
        auto t0 = std::chrono::steady_clock::now();
        MTL::CommandBuffer* cb = c.q->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        if (useItems)
        {   // items index the tetrahedron arrays globally; the chunk is [itemBase, itemBase+n)
            enc->setBuffer(b.bV,   0, 0);
            enc->setBuffer(b.bD,   0, 1);
            enc->setBuffer(b.bU,   0, 2);
            enc->setBuffer(b.bVol, 0, 3);
            enc->setBuffer(b.bCnt, 0, 4);
            enc->setBuffer(b.bFlt, 0, 5);
        }
        else
        {
            enc->setBuffer(b.bV,   start * 12 * sizeof(float), 0);
            enc->setBuffer(b.bD,   start * 4 * sizeof(float),  1);
            enc->setBuffer(b.bU,   b.needVel ? start * 12 * sizeof(float) : 0, 2);
            enc->setBuffer(b.bVol, start * sizeof(float),      3);
            enc->setBuffer(b.bCnt, start * sizeof(uint32_t),   4);
            enc->setBuffer(b.bFlt, start * sizeof(uint32_t),   5);
        }
        enc->setBuffer(b.bS,   0, 6);
        enc->setBuffer(g.den,  0, 7);
        enc->setBuffer(g.vel,  0, 8);
        enc->setBuffer(g.grad, 0, 9);
        enc->setBuffer(bP,     0, 10);
        enc->setBuffer(b.bG,   0, 11);
        enc->setBuffer(b.bB,   0, 12);
        enc->setBuffer(b.bSc,  (!useItems && b.needScal) ? start * 4 * sizeof(float) : 0, 13);
        enc->setBuffer(g.scal, 0, 14);
        enc->setBuffer(g.sgrad, 0, 15);
        // the same grids as fixed-point words (the kernel reads one view or the other, by P.fixedSum)
        enc->setBuffer(g.den,  0, 16);
        enc->setBuffer(g.vel,  0, 17);
        enc->setBuffer(g.grad, 0, 18);
        enc->setBuffer(g.scal, 0, 19);
        enc->setBuffer(g.sgrad, 0, 20);
        enc->setBuffer(bBad,   0, 21);
        if (useItems && !exact)
        {
            enc->setThreadgroupMemoryLength(keyBytes, 0);
            enc->setThreadgroupMemoryLength(valBytes, 1);
            enc->setThreadgroupMemoryLength(valXBytes, 2);
        }
        enc->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(tg, 1, 1));
        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
        ++st.chunks;
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
            ++st.retries;
            GpuFault const cause = classifyGpuFault(code, desc);
            countGpuRetry(cause, code, desc);   // the run's counters count every failed buffer (run_progress.h)
            lastErr = std::string("Metal command buffer failed: ") + desc + " [" + gpuFaultName(cause) + "]";
            failsHere = (failsHere > 0 && start < failEnd) ? failsHere + 1 : 1;
            failEnd = std::max(failEnd, start + n);
            if (cause == GpuFault::Fatal || !ckOn || failsHere >= MAX_FAILS_SAME_CHUNK || nFailed >= MAX_FAILS_TOTAL)
            {
                if (!ckOn) lastErr += " (no checkpoint held)";
                else if (failsHere >= MAX_FAILS_SAME_CHUNK) lastErr += " (" + std::to_string(failsHere) + " failures of the same chunk)";
                else if (nFailed >= MAX_FAILS_TOTAL) lastErr += " (" + std::to_string(nFailed) + " failed buffers in this batch)";
                success = false;
                break;
            }
            if (cause != GpuFault::Victim)
            {   // escalate (for the rest of the process): 4x shorter buffers, wider gaps, smaller chunk
                pace.targetSec = std::max(pace.targetSec / 4.0, 0.02);
                pace.gapUs     = std::min(pace.gapUs * 3, useconds_t(50000));
                pace.cleanSec  = 0.;
                chunk = alignChunkUp(std::max(chunk / 4, MIN_CHUNK));
            }
            drainQueue();
            size_t resumeAt = 0;
            if (sharedBox) g.zeroBox(b.P.subOff, b.P.nGrid);
            else { snap.restore(); resumeAt = snap.start; }
            std::memcpy(badRun, badCk, sizeof badRun);   // (sharedBox: the batch's start, all zero)
            if (ckPeriodic) ++nResumedCk;
            fprintf(stderr, "%sDTFE Metal: %s -- resuming from the %s at thread %zu of %zu "
                            "(redoing %zu; failure %d of this chunk, target %.0f ms buffers, %.0f ms gaps)%s\n",
                    MESSAGE::cYellow(), lastErr.c_str(), ckPeriodic ? "checkpoint" : "pass start",
                    resumeAt, nThreads, start + n - resumeAt, failsHere,
                    pace.targetSec * 1e3, pace.gapUs / 1e3, MESSAGE::cReset());
            start = resumeAt;
            sinceCk = 0;
            // an innocent victim retries at once; our own buffers' failures let the system settle first
            if (cause == GpuFault::Resource) sleep(2);
            else if (cause == GpuFault::Watchdog) sleep(1);
            continue;
        }

        start += n;
        ++sinceCk;
        for (int k = 0; k < FX_DROP_SLOTS; ++k) badRun[k] += badW[k];
        if (el > 1e-4)   // retarget the next chunk toward targetSec (bounded rescale)
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
                pace.gapUs     = std::max(pace.gapUs / 3, GPU_PACE_GAP0);
                fprintf(stderr, "DTFE Metal: %.0f s of GPU work without a fault -- pace eased to %.0f ms "
                                "buffers, %.0f ms gaps%s\n", pace.cleanSec, pace.targetSec * 1e3, pace.gapUs / 1e3,
                        pace.targetSec >= GPU_PACE_TARGET0 && pace.gapUs <= GPU_PACE_GAP0 ? " (the default)" : "");
                pace.cleanSec = 0.;
            }
        }
        long const every = ckChunks >= 0 ? ckChunks : std::max(1L, long(std::lround(10.0 / pace.targetSec)));
        if (!sharedBox && ckOn && every > 0 && long(sinceCk) >= every && start < nThreads)
        {   // the queue is drained (waitUntilCompleted above): the buffers hold exactly [0, start)
            if (snap.take(gridBufs, std::vector<bool>(), start))
            { ++snap.taken; ckPeriodic = true; sinceCk = 0; std::memcpy(badCk, badRun, sizeof badCk); }
            // else: no host memory for the spare copy; the last snapshot stays valid
        }
        usleep(pace.gapUs);   // deliberate idle gap: lets WindowServer take the GPU
    }
    cs.chunk = chunk;
    if (nFailed > 0)
        fprintf(stderr, "%sDTFE Metal: deposit %s after %d failed buffer(s) (%d resumed from a periodic checkpoint); "
                        "%d checkpoint(s), %.0f MB held, %.3f s copying%s\n",
                success ? MESSAGE::cYellow() : MESSAGE::cRed(), success ? "recovered on the GPU" : "gave up",
                nFailed, nResumedCk, snap.taken, snap.bytes / 1.e6, snap.sec, MESSAGE::cReset());
    bP->release();
    bBad->release();
    if (!success)
    {   // leave the owner box clean for whoever deposits this sub-domain instead
        drainQueue();
        if (sharedBox) g.zeroBox(b.P.subOff, b.P.nGrid);
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
};
Shared& shared() { static Shared s; return s; }

size_t physicalMemory()
{
    uint64_t mem = 0; size_t len = sizeof(mem);
    if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) != 0 || mem == 0) return size_t(16) << 30;
    return size_t(mem);
}

} // namespace


std::string gpuBackendName() { return "Metal"; }

std::string gpuDeviceName()
{
    std::lock_guard<std::mutex> lock(ctxMutex());
    Ctx& c = ctx();
    return c.dev ? std::string(c.dev->name()->utf8String()) : std::string();
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
    if (cnts.empty()) return true;    // nothing to deposit (no cells in the region)

    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    size_t const zero[3] = {0, 0, 0};
    Batch b;      // uploaded WITHOUT the mutex: the sub-domains' host work overlaps the queue
    if (!uploadBatch(c, verts, dens, vels, scal, vols, cnts, flats, sobol, dx, nGrid, zero, nGrid, fDen, fVel, fGrad,
                     fScal, fSGrad, exact, true, b, err))
    { pool->release(); return false; }
    Grids g;
    ChunkState cs;
    Stats st;
    bool ok;
    double el = 0.;
    {
        std::lock_guard<std::mutex> lock(ctxMutex());   // serialize dispatches (single queue)
        // The grids are allocated HERE, under the device mutex: allocated before it, every sub-domain waiting
        // for the device would hold its zeroed grids (all of them at once while the batches queue up), and the
        // exact sums' words are 8 B per component -- a 512^3 run with the velocity gradient would wait on
        // ~1.6 GB per sub-domain. So one batch's at a time (and the one being converted below), zeroed on the GPU.
        if (!g.alloc(c, nGrid, fDen, fVel, fGrad, fScal, fSGrad, b.P.fixedSum != 0, true))
        {
            err = "Metal buffer allocation failed (out of GPU-visible memory?)";
            ok = false;
        }
        else
        {
            auto const t0 = std::chrono::steady_clock::now();
            ok = runBatch(c, b, g, cs, st, false, err);
            el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }
    }
    if (timingOn() && g.den) dtfeLogBatch("Metal", b, st, 0., el, -1.);
    if (ok && g.fixed)
    {   // the exact sums: every cell's integer, rounded once to the float of the contract (psMassFixedMass:
        // q / scale in double), outside the device mutex
        auto toFloat = [](std::vector<float>& dst, MTL::Buffer* src, float scale) {
            int64_t const* q = static_cast<int64_t const*>(src->contents());
            for (size_t i = 0; i < dst.size(); ++i) dst[i] = float(psMassFixedMass(q[i], scale));
        };
        if (fDen)   toFloat(out.density,     g.den,   b.P.fxScale[0]);
        if (fVel)   toFloat(out.velocity,    g.vel,   b.P.fxScale[1]);
        if (fGrad)  toFloat(out.grad,        g.grad,  b.P.fxScale[2]);
        if (fScal)  toFloat(out.scalar,      g.scal,  b.P.fxScale[3]);
        if (fSGrad) toFloat(out.scalar_grad, g.sgrad, b.P.fxScale[4]);
    }
    else if (ok)
    {
        if (fDen)  std::memcpy(out.density.data(),  g.den->contents(),  nCell * 4);
        if (fVel)  std::memcpy(out.velocity.data(), g.vel->contents(),  nCell * 12);
        if (fGrad) std::memcpy(out.grad.data(),     g.grad->contents(), nCell * 36);
        if (fScal) std::memcpy(out.scalar.data(),   g.scal->contents(), nCell * 4);
        if (fSGrad) std::memcpy(out.scalar_grad.data(), g.sgrad->contents(), nCell * 12);
    }
    b.release(); g.release();
    pool->release();
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
    // OPT-IN (DTFE_GPU_SHARED=1). Measured on 2.1M particles at 512^3 (9 sub-domains, 11.4 G
    // samples): the shared full grid runs the SAME kernel at 450-780 M samples/s against 880-1070 M
    // on the per-call path with its sub-grids -- the atomics' footprint on a 2 GB grid costs more
    // in TLB and cache reach than the per-sub-domain zeroing and read-back it saves, even with the
    // tetrahedra in Morton order (22.2 s against 17-19 s on the per-call path).
    // A tiled (brick-ordered) full grid would be the next step if sharing is ever needed; the
    // machinery (owner boxes, implicit items, batch retries) is in place and tested either way.
    if (!dtfeSharedOptedIn(err)) return false;
    // The full grids live on the GPU for the whole parallel region, next to the main grid they are
    // added into afterwards and the threads' triangulations: cap them to a share of physical memory
    // (DTFE_GPU_SHARED_GB overrides). Over the cap every sub-domain takes the per-call path, which
    // holds one sub-grid at a time.
    if (!dtfeSharedGridFits(nCell, fDen, fVel, fGrad, fScal, fSGrad, 0.25 * double(physicalMemory()) / 1e9, err))
        return false;
    if (!s.grids.alloc(c, fullGrid, fDen, fVel, fGrad, fScal, fSGrad))
    {
        s.grids.release();
        err = "Metal buffer allocation failed for the shared grid";
        return false;
    }
    s.cs = ChunkState();
    s.began = std::chrono::steady_clock::now();
    s.batches = s.failed = 0;
    s.busy = 0.;
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
    Ctx& c = ctx();          // initialized by Begin; read-only here
    Shared& s = shared();
    if (!s.active) { err = "no shared GPU grid"; return false; }
    for (int d = 0; d < 3; ++d)
        if (subOff[d] + nGrid[d] > s.grids.full[d]) { err = "sub-grid outside the shared grid"; return false; }
    if (cnts.empty()) return true;

    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    auto const tReady = std::chrono::steady_clock::now();
    Batch b;      // uploaded WITHOUT the mutex: the threads' uploads overlap each other and the queue
    // (the float atomics: the shared grid has one unit for every batch, which no batch's bound sets -- the exact
    // sums are the per-call path's)
    if (!uploadBatch(c, verts, dens, vels, scal, vols, cnts, flats, sobol, dx, nGrid, subOff, s.grids.full,
                     s.grids.fDen, s.grids.fVel, s.grids.fGrad, s.grids.fScal, s.grids.fSGrad, false, false, b, err))
    { pool->release(); return false; }

    bool ok;
    double waitSec, runSec;
    {
        std::lock_guard<std::mutex> lock(ctxMutex());
        auto const tGot = std::chrono::steady_clock::now();
        waitSec = std::chrono::duration<double>(tGot - tReady).count();
        Stats st;
        ok = runBatch(c, b, s.grids, s.cs, st, true, err);
        runSec = std::chrono::duration<double>(std::chrono::steady_clock::now() - tGot).count();
        s.busy += runSec;
        ++s.batches;
        if (!ok) ++s.failed;
        if (timingOn())
            dtfeLogBatch("Metal", b, st, waitSec, runSec, std::chrono::duration<double>(tReady - s.began).count());
    }
    b.release();
    pool->release();
    return ok;
}

bool dtfeGpuSharedResult(const float** den, const float** vel, const float** grad, const float** scal, const float** sgrad)
{
    std::lock_guard<std::mutex> lock(ctxMutex());
    Shared& s = shared();
    if (!s.active) return false;
    if (den)  *den  = s.grids.fDen  ? static_cast<const float*>(s.grids.den->contents())  : nullptr;
    if (vel)  *vel  = s.grids.fVel  ? static_cast<const float*>(s.grids.vel->contents())  : nullptr;
    if (grad) *grad = s.grids.fGrad ? static_cast<const float*>(s.grids.grad->contents()) : nullptr;
    if (scal) *scal = s.grids.fScal ? static_cast<const float*>(s.grids.scal->contents()) : nullptr;
    if (sgrad) *sgrad = s.grids.fSGrad ? static_cast<const float*>(s.grids.sgrad->contents()) : nullptr;
    if (timingOn())
        fprintf(stderr, "DTFE Metal timing: shared grid: %d batches (%d handed back), device busy %.2f s of the "
                        "%.2f s since begin\n", s.batches, s.failed, s.busy,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - s.began).count());
    return true;
}

void dtfeGpuSharedEnd()
{
    std::lock_guard<std::mutex> lock(ctxMutex());
    Shared& s = shared();
    if (!s.active) return;
    s.grids.release();
    s.active = false;
}
