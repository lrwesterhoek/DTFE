/* The Metal GPU hosts' chunk-level checkpoints and cause-aware retry policy (2026-10-11 in ps_metal_host.cc;
   moved here 2026-10-13 so the standard-DTFE host, dtfe_metal_host.cc, runs the SAME policy -- the code below is
   ps_metal_host.cc's, unchanged but for the counter split noted at gpuRetryCause).

   A failed command buffer used to redo the WHOLE deposit (a PS partition: up to ~2000 s per TNG50 partition; a
   standard-DTFE sub-domain batch: the whole batch, three times, then the CPU) without saying why. Now every pass
   keeps a snapshot of the buffers its kernel writes (the accumulator grids; PS also the exact items' sumW and the
   per-tet masses the kernel negates when it defers a tet), taken at the pass's start and every
   DTFE_GPU_CHECKPOINT_CHUNKS chunks; a failed buffer (which may have deposited part or all of its atomics)
   restores the last snapshot and continues from its chunk, at a pace that depends on the cause (GpuFault).

   Included by the two Metal hosts INSIDE their anonymous namespace, after <Metal/Metal.hpp> and the std headers
   (<algorithm> <chrono> <cstdlib> <cstring> <memory> <new> <string> <vector> <unistd.h>), like
   gpu_exact_clip.cuh in the CUDA hosts. A binary links ONE Metal host (PS-DTFE: ps_metal_host.cc, DTFE and
   libDTFE: dtfe_metal_host.cc), so the process-wide statics below (the pace, the fault hook) are that binary's.
   The CUDA/HIP twin is gpu_cuda_retry.cuh. */

// What a failed command buffer says about its cause. Both watchdog flavours share Metal's Timeout code (2), so
// the description is read first:
//   Victim    "Innocent Victim": another process's buffer hung the GPU and ours was collateral -> retry at
//             once at the same pace (slowing down would only cost time, our buffers were not the problem)
//   Watchdog  "Impacting Interactivity" / a GPU timeout: our buffers starved the display -> shorter buffers,
//             wider gaps (the pacing the chunking exists for)
//   Resource  out of memory, page fault, invalid resource, internal: escalate as for the watchdog and let the
//             system settle 2 s first
//   Fatal     device removed / not permitted / access revoked: no retry can help -> straight to the CPU
enum class GpuFault { Victim, Watchdog, Resource, Fatal };

char const* gpuFaultName(GpuFault f)
{
    switch (f)
    {
        case GpuFault::Victim:   return "innocent victim";
        case GpuFault::Watchdog: return "watchdog";
        case GpuFault::Resource: return "resource";
        default:                 return "fatal";
    }
}

GpuFault classifyGpuFault(long code, std::string const& desc)
{
    auto has = [&](char const* s) { return desc.find(s) != std::string::npos; };
    if (has("Innocent Victim") || has("InnocentVictim")) return GpuFault::Victim;
    if (has("Impacting Interactivity") || has("ImpactingInteractivity") || has("Timeout")
        || code == long(MTL::CommandBufferErrorTimeout)) return GpuFault::Watchdog;
    if (code == long(MTL::CommandBufferErrorDeviceRemoved) || code == long(MTL::CommandBufferErrorNotPermitted)
        || code == long(MTL::CommandBufferErrorAccessRevoked)) return GpuFault::Fatal;
    return GpuFault::Resource;
}

// G1 (2026-10-11): the run's retry counters (run_progress.h: the sidecar's gpu_retries and
// gpu_retries_by_cause, DTFE.cpp's closing 'GPU deposit retries:' line) by the cause a reader can act on:
// 0 innocent victim, 1 timeout (the watchdog), 2 out of memory -- split off the Resource class, which also
// holds page faults and internal errors -- and 3 everything else (those, and the fatal causes). The index
// only (2026-10-13): each host bumps its own binary's counters with it (countGpuRetry in the host).
int gpuRetryCause(GpuFault f, long code, std::string const& desc)
{
    int k = 3;
    if (f == GpuFault::Victim) k = 0;
    else if (f == GpuFault::Watchdog) k = 1;
    else if (f == GpuFault::Resource && (code == long(MTL::CommandBufferErrorOutOfMemory)
             || desc.find("out of memory") != std::string::npos || desc.find("OutOfMemory") != std::string::npos
             || desc.find("Insufficient Memory") != std::string::npos)) k = 2;
    return k;
}

// The dispatch pace, kept across partitions once escalated (dtfe_metal_host.cc's batches share it since 2026-10-13):
// a machine whose display load killed one partition's buffers will kill the next one's at the same pace.
// But not for the rest of the PROCESS (2026-10-11): one 'Impacting Interactivity' fault -- a moment of display
// use -- then slowed every later partition (72% GPU duty after one escalation, 29% after two: about +2.5 h,
// up to +9.5 h, on the 8 h TNG50 z=0 run, whose runlog has two such faults at partitions 7 and 39). So the pace
// steps back one notch -- the inverse of one escalation, buffers x4 and gaps /3, toward 0.25 s / 8 ms -- after
// every DTFE_GPU_PACE_RECOVER_SEC (default 90) seconds of GPU work (buffers + gaps) without a fault of ours;
// counted in GPU work, not wall time, so the triangulations between deposits are no evidence the display is
// quiet. An escalation within a deposit therefore holds for at least that long. An innocent victim neither
// escalates nor resets the clean count. Guarded by the host's ctxMutex (only touched inside psGpuDepositFields,
// or dtfe_metal_host.cc's runBatch, whose callers hold it).
struct GpuPace { double targetSec = 0.25; useconds_t gapUs = 8000; double cleanSec = 0.; };
double const GPU_PACE_TARGET0 = 0.25;   // the default pace the recovery returns to
useconds_t const GPU_PACE_GAP0 = 8000;
GpuPace& gpuPace() { static GpuPace p; return p; }

// The test hook DTFE_GPU_FAULT_AT=<k>[:<kind>]: the k-th command buffer of the PROCESS (1-based, counted over
// every pass and partition; buffers that failed for real are not counted) runs to completion -- its atomics ARE deposited, the worst case of a buffer killed
// late -- and is then reported as failed, so only a correct restore keeps the grids exact. Kinds: 'innocent'
// (default), 'timeout', 'oom', and 'persist' (every buffer from the k-th on fails as an innocent victim: the
// retry policy must then give the partition to the CPU). One-shot except 'persist'.
struct GpuFaultHook { long at = 0; std::string kind = "innocent"; long count = 0; bool fired = false; };
GpuFaultHook& gpuFaultHook()
{
    static GpuFaultHook h = [] {
        GpuFaultHook r;
        if (const char* env = getenv("DTFE_GPU_FAULT_AT"))
        {
            r.at = atol(env);
            if (const char* colon = strchr(env, ':')) r.kind = colon + 1;
        }
        return r;
    }();
    return h;
}

// true (and the injected error's description in 'desc', its code in 'code') when this buffer is to fail
bool gpuFaultInjected(std::string& desc, long& code)
{
    GpuFaultHook& h = gpuFaultHook();
    if (h.at <= 0) return false;
    ++h.count;
    bool const persist = h.kind == "persist";
    if (h.count < h.at || (h.fired && !persist)) return false;
    h.fired = true;
    code = 0;
    if (h.kind == "timeout")  { desc = "injected fault (DTFE_GPU_FAULT_AT): Impacting Interactivity"; code = long(MTL::CommandBufferErrorTimeout); }
    else if (h.kind == "oom") { desc = "injected fault (DTFE_GPU_FAULT_AT): out of memory"; code = long(MTL::CommandBufferErrorOutOfMemory); }
    else desc = std::string("injected fault (DTFE_GPU_FAULT_AT") + (persist ? ", persistent" : "") + "): Innocent Victim";
    return true;
}

// One snapshot of a set of buffers, host-side. A buffer known to be all zeros is recorded as such (no copy):
// the first pass's grids at its start, so a short deposit never allocates the spare copy at all.
struct GpuSnapshot
{
    struct Blob { std::unique_ptr<char[]> p; size_t n = 0; bool zero = false; };
    std::vector<MTL::Buffer*> bufs;
    std::vector<Blob> blobs;
    size_t start = 0;      // the pass's first thread NOT yet covered by the snapshot
    bool   valid = false;
    int    taken = 0;      // periodic snapshots (not the pass-start ones)
    double sec   = 0.;     // time spent copying
    size_t bytes = 0;      // largest snapshot held

    // copies every buffer (zeroMask[i]: buffer i is known zero, not copied); false when the host memory for the
    // spare copy is not there (the caller then runs without checkpoints)
    bool take(std::vector<MTL::Buffer*> const& bs, std::vector<bool> const& zeroMask, size_t at)
    {
        auto const t0 = std::chrono::steady_clock::now();
        if (blobs.size() < bs.size()) blobs.resize(bs.size());
        // allocate everything BEFORE copying anything: a failed allocation leaves the previous snapshot whole
        for (size_t i = 0; i < bs.size(); ++i)
        {
            Blob& b = blobs[i];
            size_t const n = bs[i]->length();
            if ((i < zeroMask.size() && zeroMask[i]) || (b.p && b.n >= n)) continue;
            b.p.reset(new (std::nothrow) char[n]);
            b.n = b.p ? n : 0;
            if (!b.p) return false;
        }
        bufs = bs;
        size_t held = 0;
        for (size_t i = 0; i < bs.size(); ++i)
        {
            Blob& b = blobs[i];
            b.zero = i < zeroMask.size() && zeroMask[i];
            if (b.zero) continue;
            size_t const n = bs[i]->length();
            std::memcpy(b.p.get(), bs[i]->contents(), n);
            held += n;
        }
        start = at;
        valid = true;
        bytes = std::max(bytes, held);
        sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return true;
    }
    void restore()
    {
        auto const t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < bufs.size(); ++i)
        {
            size_t const n = bufs[i]->length();
            if (blobs[i].zero) std::memset(bufs[i]->contents(), 0, n);
            else std::memcpy(bufs[i]->contents(), blobs[i].p.get(), n);
        }
        sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
};
