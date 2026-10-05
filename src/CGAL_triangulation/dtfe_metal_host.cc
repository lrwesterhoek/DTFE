/* Metal backend of the standard-DTFE method-1 deposit (implements gpu_host.h). Owns the
   metal-cpp implementation symbols for the DTFE binary; compiled only when METAL=1 (which
   defines DTFE_GPU for the callers). The kernel source is embedded at build time
   (o/dtfe_deposit_msl.h, generated from metal/dtfe_deposit.metal) and runtime-compiled once
   per process; the pipeline states are cached.

   Two ways in. dtfeGpuDepositAveraged1() deposits one tetrahedron set into grids of its own
   (the original per-call path). The dtfeGpuShared* family lets the sub-domains of one
   DTFE_parallel call deposit into ONE full grid: each thread uploads its batch (concurrently),
   the dispatches queue on the device (serialized by ctxMutex, the chunk controller warm across
   batches), and the main thread reads the full grid back once. Both use the same Batch /
   runBatch machinery below; a batch's owner box is zeroed before any retry, so a batch can be
   redone or handed back to the CPU without touching the other sub-domains' cells.

   The chunked, watchdog-safe dispatch discipline is the one validated in ps_metal_host.cc --
   see the long comment at the dispatch loop. */

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <unistd.h>
#include <sys/sysctl.h>

#include "gpu_host.h"
#include "dtfe_deposit_msl.h"   // generated: static const char DTFE_DEPOSIT_MSL[]
#define ITEM_BLOCK_SHIFT 5      // == the kernel's: log2 of the items per entry of the block table
#include "../message.h"         // ANSI colour helpers only (retry warning below)

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
};

struct Ctx
{
    MTL::Device*               dev  = nullptr;
    MTL::CommandQueue*         q    = nullptr;
    MTL::Library*              lib  = nullptr;
    MTL::ComputePipelineState* pso  = nullptr;        // depositAveraged1: one thread per tetrahedron
    MTL::ComputePipelineState* psoExact = nullptr;    // depositExactAverage: one thread per (tet, cell block)
    // depositAveraged1Items{2,4,8}: one thread per (tet, sample run) with a 2/4/8-slot cell cache,
    // specialized per field set (function constants FC_DEN/FC_VEL/FC_GRAD/FC_SCAL/FC_SGRAD): index [K slot][mask]
    MTL::ComputePipelineState* psoItems[3][32] = {{nullptr}};
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
MTL::ComputePipelineState* itemPso(Ctx& c, int cacheK, bool fDen, bool fVel, bool fGrad, bool fScal, bool fSGrad)
{
    static std::mutex psoMutex;              // uploads (and so first uses) run concurrently
    std::lock_guard<std::mutex> lock(psoMutex);
    int const ks = kSlot(cacheK), mask = (fDen ? 1 : 0) | (fVel ? 2 : 0) | (fGrad ? 4 : 0) | (fScal ? 8 : 0) | (fSGrad ? 16 : 0);
    if (c.psoItems[ks][mask]) return c.psoItems[ks][mask];
    if (!c.itemsOk || !c.lib) return nullptr;
    char name[64];
    snprintf(name, sizeof(name), "depositAveraged1Items%d", cacheK == 2 ? 2 : cacheK == 8 ? 8 : 4);
    MTL::FunctionConstantValues* fc = MTL::FunctionConstantValues::alloc()->init();
    bool d = fDen, v = fVel, g = fGrad, s = fScal, sg = fSGrad;
    fc->setConstantValue(&d, MTL::DataTypeBool, NS::UInteger(0));
    fc->setConstantValue(&v, MTL::DataTypeBool, NS::UInteger(1));
    fc->setConstantValue(&g, MTL::DataTypeBool, NS::UInteger(2));
    fc->setConstantValue(&s, MTL::DataTypeBool, NS::UInteger(3));
    fc->setConstantValue(&sg, MTL::DataTypeBool, NS::UInteger(4));
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
        fprintf(stderr, "DTFE Metal pipeline %s (den %d vel %d grad %d scal %d sgrad %d): max threads per threadgroup %lu (per-tet kernel %lu)\n",
                name, int(d), int(v), int(g), int(s), int(sg), (unsigned long)pso->maxTotalThreadsPerThreadgroup(),
                (unsigned long)c.pso->maxTotalThreadsPerThreadgroup());
    c.psoItems[ks][mask] = pso;
    return pso;
}


// ---------------------------------------------------------------- knobs
// Work items (default): every tetrahedron becomes max(1, ceil(cnt / itemSamples)) threads, so no
// thread loops over more than itemSamples samples -- the sample count per tetrahedron reaches
// maxNN = 10000 on fine grids, which made the per-tet kernel's chunks wildly non-uniform and
// tripped the GPU watchdog -- and the kernel's per-thread cell cache turns hundreds of per-sample
// atomics into one set per distinct cell (see the kernel header). DTFE_GPU_ITEMS=0 runs the
// per-tet kernel for A/B comparisons.
// DTFE_GPU_ITEM_SAMPLES=<n> sets the run length and DTFE_GPU_CACHE_K=2|4|8 the cell cache.
// Measured on 2.1M particles at 512^3 (the largest sub-domain, 2.0 G samples; the per-tet kernel
// took 45 s with watchdog retries). First kernel (atomics per distinct cell per item, cost-sorted
// tetrahedra): items 512/128/64/32/16/8 with 4 slots = 7.1/4.3/3.6/2.9/2.7/3.1 s; 8 slots slower
// everywhere, 2 slots slower than 4 -- then limited by threads in flight (register-bound: 640 of
// 1024 threads per threadgroup at K=4). Final kernel (Morton-ordered tetrahedra, SIMD-group
// reduction of the flushes, coalesced float4 table reads, field-set specialization): K=2 at items
// 8/16/32/48/64 = 2.85/2.35/2.05/1.97/1.95 s (K=4 at 32: 2.05 s), density only 0.9 s, no atomics
// at all 1.8 s -- the kernel is now compute/occupancy-bound at ~1.1 G samples/s, ~2.3 G for density.
struct Knobs
{
    bool     useItems    = true;
    uint32_t itemSamples = 48;
    int      cacheK      = 2;
    bool     noWrite     = false;   // DTFE_GPU_DEBUG_NOWRITE: timing experiment, all the work, no atomics
};

Knobs knobs(const Ctx& c)
{
    Knobs k;
    k.useItems = c.itemsOk;
    if (const char* env = getenv("DTFE_GPU_ITEMS")) k.useItems = k.useItems && atoi(env) != 0;
    if (const char* env = getenv("DTFE_GPU_ITEM_SAMPLES"))
    { long v = atol(env); if (v >= 4 && v <= 100000) k.itemSamples = uint32_t(v); }
    if (const char* env = getenv("DTFE_GPU_CACHE_K")) { int v = atoi(env); k.cacheK = (v == 4 || v == 8) ? v : 2; }
    k.noWrite = getenv("DTFE_GPU_DEBUG_NOWRITE") != nullptr;
    return k;
}


// ---------------------------------------------------------------- the accumulated grids
struct Grids
{
    MTL::Buffer* den  = nullptr;   // nCell floats (or a 4-byte dummy)
    MTL::Buffer* vel  = nullptr;   // nCell*3
    MTL::Buffer* grad = nullptr;   // nCell*9
    MTL::Buffer* scal = nullptr;   // nCell     the scalar
    MTL::Buffer* sgrad = nullptr;  // nCell*3   its gradient
    size_t full[3] = {0, 0, 0};
    bool fDen = false, fVel = false, fGrad = false, fScal = false, fSGrad = false;

    size_t nCell() const { return full[0] * full[1] * full[2]; }
    bool alloc(Ctx& c, const size_t fullGrid[3], bool d, bool v, bool g, bool s, bool sg)
    {
        for (int i = 0; i < 3; ++i) full[i] = fullGrid[i];
        fDen = d; fVel = v; fGrad = g; fScal = s; fSGrad = sg;
        auto zeroBuf = [&](size_t bytes) {
            MTL::Buffer* b = c.dev->newBuffer(bytes, MTL::ResourceStorageModeShared);
            if (b) std::memset(b->contents(), 0, bytes);
            return b;
        };
        den  = zeroBuf(fDen  ? nCell() * 4  : 4);
        vel  = zeroBuf(fVel  ? nCell() * 12 : 4);
        grad = zeroBuf(fGrad ? nCell() * 36 : 4);
        scal = zeroBuf(fScal ? nCell() * 4  : 4);
        sgrad = zeroBuf(fSGrad ? nCell() * 12 : 4);
        return den && vel && grad && scal && sgrad;
    }
    void release()
    {
        for (MTL::Buffer** b : {&den, &vel, &grad, &scal, &sgrad}) if (*b) { (*b)->release(); *b = nullptr; }
    }
    // zero the cells of a sub-grid box [off, off+dims) -- the owner box of one batch
    void zeroBox(const int32_t off[3], const int32_t dims[3])
    {
        auto zero = [&](MTL::Buffer* b, bool on, size_t comps) {
            if (!on || !b) return;
            float* base = static_cast<float*>(b->contents());
            for (int32_t i = 0; i < dims[0]; ++i)
                for (int32_t j = 0; j < dims[1]; ++j)
                {
                    size_t const row = (size_t(off[0] + i) * full[1] + size_t(off[1] + j)) * full[2] + size_t(off[2]);
                    std::memset(base + row * comps, 0, size_t(dims[2]) * comps * sizeof(float));
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
    Knobs  k;

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
                 bool fScal, bool fSGrad, bool exact, Batch& b, std::string& err)
{
    b.k = knobs(c);
    b.exact = exact;
    if (exact) b.k.useItems = true;    // the exact kernel is an item kernel (cell blocks), always
    else if (b.k.useItems && !itemPso(c, b.k.cacheK, fDen && !b.k.noWrite, fVel && !b.k.noWrite, fGrad && !b.k.noWrite,
                                      fScal && !b.k.noWrite, fSGrad && !b.k.noWrite))
        b.k.useItems = false;      // (built here, before the upload, so a failure falls back cleanly)
    b.needVel = fVel || fGrad;
    b.needScal = fScal || fSGrad;
    b.nTet = cnts.size();
    DepositParams& P = b.P;
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
    // 1 = no atomics in the final reduction; 2 = also skip the gradient's reduction loop; 3 = also the
    // gradient's eviction atomics (timing experiments only)
    P.itemSamples = b.k.itemSamples;
    // the exact kernel's item = a block of cells of the tetrahedron's window (DTFE_GPU_EXACT_CELLS)
    uint32_t exactCells = 16;
    if (exact)
    {
        if (const char* env = getenv("DTFE_GPU_EXACT_CELLS")) { long v = atol(env); if (v > 0) exactCells = uint32_t(v); }
        P.itemSamples = exactCells;
    }

    // the fast path's flat indices arrive in the SUB-grid; the kernels deposit into the accumulated
    // grid, so shift them to full-grid indices here (the sampled path adds subOff on the device)
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

    // the item map (see the kernel's itemOf): itemStart[t] = first item of tetrahedron t (nTet+1
    // entries) and, per block of 2^ITEM_BLOCK_SHIFT items, the tetrahedron holding the block's first
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
            {   // an UPPER bound of the kernel's window (its float floor can differ from this double one
                // only within rounding of an integer, which the margin covers; surplus blocks idle)
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
    std::vector<float> sobol4(sobol.size() / 3 * 4, 0.f);     // (x, y, z, 0): one 16-byte load per sample
    for (size_t i = 0; i < sobol.size() / 3; ++i)
        for (int d = 0; d < 3; ++d) sobol4[i * 4 + d] = sobol[i * 3 + d];
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

// The chunk controller's state; one per accumulated grid so it stays warm across batches.
struct ChunkState
{
    double     targetSec = 0.25;
    useconds_t gapUs     = 8000;
    size_t     chunk     = 0;       // 0 = not started: use the kernel's first chunk
};

struct Stats { int chunks = 0, retries = 0; };

// Runs one batch into the grids (CALLER HOLDS ctxMutex). Chunked, serialized dispatch with
// host-side gaps -- the watchdog-safe discipline validated for the PS deposit (see
// ps_metal_host.cc for the full rationale). A killed command buffer may have partially
// deposited its atomics, so a failed chunk is never retried alone: each retry drains the
// queue, re-zeros the batch's OWNER BOX (only its own cells can hold its deposits) and redoes
// the whole batch with shorter buffers and wider gaps. Work items are uniform (<= itemSamples
// samples each), so their chunks can be larger and the controller never meets a 100x jump in
// per-thread work; the per-tet kernel keeps the original, more cautious sizes.
bool runBatch(Ctx& c, Batch& b, Grids& g, ChunkState& cs, Stats& st, std::string& err)
{
    bool const exact = b.exact;
    bool useItems = b.k.useItems;
    // the exact kernel's items clip up to 16 cells each (far more than a sample run): the chunks of
    // the phase-space exact items
    size_t const MIN_CHUNK = exact ? 1024 : useItems ? 4096 : 2000;
    size_t const MAX_CHUNK = exact ? 500000 : useItems ? 4000000 : 500000;
    size_t firstChunk = exact ? 8192 : useItems ? 65536 : 25000;
    if (const char* env = getenv("DTFE_METAL_CHUNK"))
    { long v = atol(env); if (v > 0) firstChunk = size_t(v); }
    if (cs.chunk == 0) cs.chunk = firstChunk;

    MTL::ComputePipelineState* pso = exact ? c.psoExact
                                   : useItems ? itemPso(c, b.k.cacheK, b.P.fDen != 0, b.P.fVel != 0, b.P.fGrad != 0,
                                                        b.P.fScal != 0, b.P.fSGrad != 0) : c.pso;
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
    // (keys 4 B + 4 or 13 floats per entry); DTFE_GPU_TABLE=0 turns it off (evictions go straight to
    // the device, the old behaviour)
    uint32_t tableEntries = 0;
    size_t keyBytes = 0, valBytes = 0;
    if (useItems && !exact)
    {   // == the kernel's tableStride(): den, vel, [scalar], [gradient], [scalar gradient]
        size_t const stride = 4 + (b.P.fScal ? 1 : 0) + (b.P.fGrad ? 9 : 0) + (b.P.fSGrad ? 3 : 0);
        size_t const budget = std::min<size_t>(pso->staticThreadgroupMemoryLength() < 32768
                                               ? 32768 - pso->staticThreadgroupMemoryLength() : 0, 32768);
        uint32_t e = 1;
        while (size_t(e * 2) * (4 + stride * 4) <= budget) e *= 2;
        if (size_t(e) * (4 + stride * 4) > budget) e = 0;
        if (const char* env = getenv("DTFE_GPU_TABLE")) { if (atoi(env) == 0) e = 0; }
        tableEntries = e >= 64 ? e : 0;
        keyBytes = tableEntries ? size_t(tableEntries) * 4 : 4;
        valBytes = tableEntries ? size_t(tableEntries) * stride * 4 : 4;
    }
    b.P.tableEntries = tableEntries;
    MTL::Buffer* bP = c.dev->newBuffer(sizeof(DepositParams), MTL::ResourceStorageModeShared);
    if (!bP) { err = "Metal buffer allocation failed (params)"; return false; }

    int const maxAttempts = 3;
    bool success = false;
    std::string lastErr;
    for (int attempt = 0; attempt < maxAttempts && !success; ++attempt)
    {
        if (attempt > 0)
        {   // Drain the queue before re-zeroing: the per-tet kernel's retried 512^3 runs differed run to
            // run by a whole sample in void cells (2e-2 of max|v|; 3e-6 without retries), consistent
            // with atomics of the killed buffer landing after the memset. An empty command buffer
            // completes only after everything queued before it.
            MTL::CommandBuffer* drain = c.q->commandBuffer();
            drain->commit();
            drain->waitUntilCompleted();
            g.zeroBox(b.P.subOff, b.P.nGrid);
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
            std::memcpy(bP->contents(), &b.P, sizeof(DepositParams));   // safe: previous chunk completed

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
            if (useItems && !exact)
            {
                enc->setThreadgroupMemoryLength(keyBytes, 0);
                enc->setThreadgroupMemoryLength(valBytes, 1);
            }
            enc->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(tg, 1, 1));
            enc->endEncoding();
            cb->commit();
            cb->waitUntilCompleted();
            ++st.chunks;
            double const el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

            if (cb->status() == MTL::CommandBufferStatusError)
            {
                NS::Error* e = cb->error();
                lastErr = std::string("Metal command buffer failed: ")
                        + (e ? e->localizedDescription()->utf8String() : "unknown");
                success = false;
                break;      // restart the WHOLE batch (owner box re-zeroed above)
            }

            start += n;
            if (el > 1e-4)   // retarget the next chunk toward targetSec (bounded rescale)
            {
                double scale = cs.targetSec / el;
                if (scale < 0.5) scale = 0.5;
                if (scale > 4.0) scale = 4.0;
                chunk = std::min(std::max(size_t(double(chunk) * scale), MIN_CHUNK), MAX_CHUNK);
            }
            usleep(cs.gapUs);   // deliberate idle gap: lets WindowServer take the GPU
        }
        cs.chunk = chunk;

        if (!success)
        {
            cs.targetSec = std::max(cs.targetSec / 4.0, 0.02);
            cs.gapUs     = std::min(cs.gapUs * 3, useconds_t(50000));
            cs.chunk     = std::max(cs.chunk / 2, MIN_CHUNK);
            if (attempt + 1 < maxAttempts)
            {
                fprintf(stderr, "%sDTFE Metal: %s -- retrying the deposit from scratch "
                                "(attempt %d/%d, target %.0f ms buffers, %.0f ms gaps)%s\n",
                        MESSAGE::cYellow(), lastErr.c_str(), attempt + 2, maxAttempts,
                        cs.targetSec*1e3, cs.gapUs/1e3, MESSAGE::cReset());
                sleep(2);   // let the system settle before hammering the GPU again
            }
        }
    }
    bP->release();
    if (!success)
    {   // leave the owner box clean for whoever deposits this sub-domain instead
        MTL::CommandBuffer* drain = c.q->commandBuffer();
        drain->commit();
        drain->waitUntilCompleted();
        g.zeroBox(b.P.subOff, b.P.nGrid);
        err = lastErr;
    }
    return success;
}

void logBatch(const Batch& b, const Stats& st, double waitSec, double runSec, double sinceBegin)
{
    fprintf(stderr, "DTFE Metal timing: %s kernel, %zu tetrahedra (%zu single-cell), %zu items of <= %u %s, "
                    "%zu samples, table %u, %d chunks, %d retries, %.2f s (%.0f M samples/s)%s",
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
    out.density.assign(fDen ? nCell : 0, 0.f);
    out.velocity.assign(fVel ? nCell * 3 : 0, 0.f);
    out.grad.assign(fGrad ? nCell * 9 : 0, 0.f);
    out.scalar.assign(fScal ? nCell : 0, 0.f);
    out.scalar_grad.assign(fSGrad ? nCell * 3 : 0, 0.f);
    if (cnts.empty()) return true;    // nothing to deposit (no cells in the region)

    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    size_t const zero[3] = {0, 0, 0};
    Batch b;      // uploaded and allocated WITHOUT the mutex: the sub-domains' host work overlaps the queue
    if (!uploadBatch(c, verts, dens, vels, scal, vols, cnts, flats, sobol, dx, nGrid, zero, nGrid, fDen, fVel, fGrad,
                     fScal, fSGrad, exact, b, err))
    { pool->release(); return false; }
    Grids g;
    if (!g.alloc(c, nGrid, fDen, fVel, fGrad, fScal, fSGrad))
    {
        err = "Metal buffer allocation failed (out of GPU-visible memory?)";
        b.release(); g.release(); pool->release();
        return false;
    }
    ChunkState cs;
    Stats st;
    bool ok;
    double el;
    {
        std::lock_guard<std::mutex> lock(ctxMutex());   // serialize dispatches (single queue)
        auto const t0 = std::chrono::steady_clock::now();
        ok = runBatch(c, b, g, cs, st, err);
        el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
    if (timingOn()) logBatch(b, st, 0., el, -1.);
    if (ok)
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
    // tetrahedra in Morton order (which took it from 32.8 s to 22.2 s; the per-call path was 17-19 s).
    // A tiled (brick-ordered) full grid would be the next step if sharing is ever needed; the
    // machinery (owner boxes, implicit items, batch retries) is in place and tested either way.
    if (!(getenv("DTFE_GPU_SHARED") && atoi(getenv("DTFE_GPU_SHARED")) != 0))
    {
        err = "off by default (DTFE_GPU_SHARED=1 turns it on; the per-sub-domain grids measured faster)";
        return false;
    }
    // The full grids live on the GPU for the whole parallel region, next to the main grid they are
    // added into afterwards and the threads' triangulations: cap them to a share of physical memory
    // (DTFE_GPU_SHARED_GB overrides). Over the cap every sub-domain takes the per-call path, which
    // holds one sub-grid at a time.
    double capGb = 0.25 * double(physicalMemory()) / 1e9;
    if (const char* env = getenv("DTFE_GPU_SHARED_GB")) { double v = atof(env); if (v > 0) capGb = v; }
    double const needGb = double(nCell) * ((fDen ? 4 : 0) + (fVel ? 12 : 0) + (fGrad ? 36 : 0) + (fScal ? 4 : 0) + (fSGrad ? 12 : 0)) / 1e9;
    if (needGb > capGb)
    {
        char buf[200];
        snprintf(buf, sizeof(buf), "the shared GPU grid would take %.1f GB, over the %.1f GB cap", needGb, capGb);
        err = buf;
        return false;
    }
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
    if (!uploadBatch(c, verts, dens, vels, scal, vols, cnts, flats, sobol, dx, nGrid, subOff, s.grids.full,
                     s.grids.fDen, s.grids.fVel, s.grids.fGrad, s.grids.fScal, s.grids.fSGrad, false, b, err))
    { pool->release(); return false; }

    bool ok;
    double waitSec, runSec;
    {
        std::lock_guard<std::mutex> lock(ctxMutex());
        auto const tGot = std::chrono::steady_clock::now();
        waitSec = std::chrono::duration<double>(tGot - tReady).count();
        Stats st;
        ok = runBatch(c, b, s.grids, s.cs, st, err);
        runSec = std::chrono::duration<double>(std::chrono::steady_clock::now() - tGot).count();
        s.busy += runSec;
        ++s.batches;
        if (!ok) ++s.failed;
        if (timingOn())
            logBatch(b, st, waitSec, runSec, std::chrono::duration<double>(tReady - s.began).count());
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
