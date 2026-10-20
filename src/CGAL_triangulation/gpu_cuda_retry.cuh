/* The CUDA/HIP GPU hosts' chunk-level checkpoints and cause-aware retry policy (ported from ps_metal_host.cc),
   shared so the standard-DTFE host, dtfe_gpu_cuda.cu, runs the SAME policy as ps_gpu_cuda.cu.
   The Metal twin is gpu_metal_retry.h.

   A failed kernel does not redo the WHOLE deposit (a PS partition, or a standard-DTFE sub-domain batch, from
   re-zeroed grids): every pass keeps a host-side snapshot of the device buffers its kernel writes
   (the accumulator grids; PS also the exact items' sumW and the per-tet masses the kernel negates when it defers
   a tet), taken at the pass's start and every DTFE_GPU_CHECKPOINT_CHUNKS kernels; a failed kernel (which may have
   deposited part or all of its atomics) restores the last snapshot and continues from its chunk. The snapshot is
   a device-to-host copy (the Metal host memcpy's its unified memory), so each checkpoint costs a PCIe round of the
   grids.

   Included by the two .cu hosts INSIDE their anonymous namespace (the gpu_exact_clip.cuh pattern), after
   gpu_cuda_compat.h, "../message.h" and the std headers (<algorithm> <chrono> <cstdlib> <cstring> <memory> <new>
   <string> <vector> <unistd.h>). A binary links ONE of them (PS-DTFE: ps_gpu_cuda.cu, DTFE and libDTFE:
   dtfe_gpu_cuda.cu), so the process-wide statics below (the pace, the fault hook) are that binary's. */

// The error codes the policy tells apart, per runtime: the launch timeout is the display watchdog's kill
// (Linux X/Wayland or the Windows TDR), the allocation failure is out of memory. (gpu_cuda_compat.h has no
// names for them; the HIP spellings are hipErrorLaunchTimeOut / hipErrorOutOfMemory.)
#if defined(GPU_EMU)
constexpr gpuError_t kGpuErrTimeout = gpuErrorLaunchTimeout;
constexpr gpuError_t kGpuErrOom     = gpuErrorMemoryAllocation;
#elif defined(__HIP__) || defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
constexpr gpuError_t kGpuErrTimeout = hipErrorLaunchTimeOut;
constexpr gpuError_t kGpuErrOom     = hipErrorOutOfMemory;
#else
constexpr gpuError_t kGpuErrTimeout = cudaErrorLaunchTimeout;
constexpr gpuError_t kGpuErrOom     = cudaErrorMemoryAllocation;
#endif

// What a failed kernel says about its cause (the classes of ps_metal_host.cc):
//   Victim    an injected 'innocent' fault only (DTFE_GPU_FAULT_AT): CUDA/HIP report no such cause, but the
//             test hook keeps the Metal kinds so one suite (tests/gpu_retry_check.sh) runs on both
//   Watchdog  the launch timeout -> shorter kernels, wider gaps
//   Resource  out of memory and any other non-sticky error: escalate as for the watchdog, settle 2 s first
//   Fatal     a STICKY error -- the context is lost (CUDA: a launch timeout, an illegal address, a launch
//             failure; detected by the probe after the failure), so no restore can run: straight to the CPU.
//             On real CUDA a watchdog kill is sticky, so it ends on the CPU; the checkpoints pay off for
//             the non-sticky ones.
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

// The run's retry counters (run_progress.h: the sidecar's gpu_retries and
// gpu_retries_by_cause, DTFE.cpp's closing 'GPU deposit retries:' line): 0 innocent victim, 1 timeout,
// 2 out of memory, 3 anything else. Counted by the error itself, BEFORE the sticky probe: a real CUDA
// timeout is a timeout even though it ends on the CPU. Only the index: each host bumps its own
// binary's counters with it (countGpuRetry in the host).
int gpuRetryCause(GpuFault f, gpuError_t code)
{
    return f == GpuFault::Victim ? 0 : code == kGpuErrTimeout ? 1 : code == kGpuErrOom ? 2 : 3;
}

// The dispatch pace, kept across partitions once escalated (as the Metal host's): a machine whose display load
// killed one partition's kernels will kill the next one's at the same pace. No idle gap until then (the Metal
// host needs one for the window server from the start). Not for the rest of the PROCESS (the
// Metal host's rule, ps_metal_host.cc GpuPace): the pace steps back one notch -- kernels x4 toward 0.25 s,
// gaps /3, and no gap once below 8 ms, this host's default of none -- after every DTFE_GPU_PACE_RECOVER_SEC
// (default 90) seconds of GPU work (kernels + gaps) without a fault of ours; counted in GPU work, not wall
// time, so the triangulations between deposits are no evidence the display is quiet. A watchdog or resource
// fault resets the count; an injected innocent victim neither escalates nor resets it. Guarded by the host's
// ctxMutex (psGpuDepositFields, or dtfe_gpu_cuda.cu's runBatch, whose callers hold it).
struct GpuPace { double targetSec = 0.25; useconds_t gapUs = 0; double cleanSec = 0.; };
double const GPU_PACE_TARGET0 = 0.25;   // the default pace the recovery returns to
useconds_t const GPU_PACE_GAP0 = 0;
useconds_t const GPU_PACE_GAP_MIN = 8000;   // the escalated gap's floor: an eased gap below it drops to none
GpuPace& gpuPace() { static GpuPace p; return p; }

// The test hook DTFE_GPU_FAULT_AT=<k>[:<kind>], as ps_metal_host.cc's: the k-th chunk kernel of the PROCESS
// (1-based, over every pass and partition; kernels that failed for real are not counted) runs to completion --
// its atomics ARE deposited, the worst case of a kernel killed late -- and is then reported as failed, so only
// a correct restore keeps the grids exact. Kinds: 'innocent' (default), 'timeout' (the launch-timeout code,
// but NOT sticky: the restore runs, which a real CUDA timeout would not allow), 'oom', and 'persist' (every
// kernel from the k-th on fails as an innocent victim: the policy must give the partition to the CPU).
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

// true (and the injected error's description, class and code) when this kernel is to fail
bool gpuFaultInjected(std::string& desc, GpuFault& cause, gpuError_t& code)
{
    GpuFaultHook& h = gpuFaultHook();
    if (h.at <= 0) return false;
    ++h.count;
    bool const persist = h.kind == "persist";
    if (h.count < h.at || (h.fired && !persist)) return false;
    h.fired = true;
    code = gpuSuccess;
    cause = GpuFault::Victim;
    if (h.kind == "timeout")  { desc = "injected fault (DTFE_GPU_FAULT_AT): launch timeout"; cause = GpuFault::Watchdog; code = kGpuErrTimeout; }
    else if (h.kind == "oom") { desc = "injected fault (DTFE_GPU_FAULT_AT): out of memory"; cause = GpuFault::Resource; code = kGpuErrOom; }
    else desc = std::string("injected fault (DTFE_GPU_FAULT_AT") + (persist ? ", persistent" : "") + "): innocent victim";
    return true;
}

// One snapshot of a set of device buffers, in host memory. A buffer known to be all zeros is recorded as
// such (no copy, restored by a memset): the first pass's grids at its start, so a short deposit never
// allocates the spare copy at all.
struct DevBuf { void* p; size_t n; };
struct GpuSnapshot
{
    struct Blob { std::unique_ptr<char[]> p; size_t n = 0; bool zero = false; };
    enum Took { Ok, NoHostMemory, DeviceError };
    std::vector<DevBuf> bufs;
    std::vector<Blob> blobs;
    size_t start = 0;      // the pass's first thread NOT yet covered by the snapshot
    bool   valid = false;
    int    taken = 0;      // periodic snapshots (not the pass-start ones)
    double sec   = 0.;     // time spent copying
    size_t bytes = 0;      // largest snapshot held

    // copies every buffer to the host (zeroMask[i]: buffer i is known zero, not copied). NoHostMemory: the
    // spare copy is not there, the previous snapshot stays whole (the caller runs on without a new one);
    // DeviceError: a copy failed half-way, so no snapshot is valid any more ('err' set)
    Took take(std::vector<DevBuf> const& bs, std::vector<bool> const& zeroMask, size_t at, std::string& err)
    {
        auto const t0 = std::chrono::steady_clock::now();
        if (blobs.size() < bs.size()) blobs.resize(bs.size());
        // allocate everything BEFORE copying anything: a failed allocation leaves the previous snapshot whole
        for (size_t i = 0; i < bs.size(); ++i)
        {
            Blob& b = blobs[i];
            if ((i < zeroMask.size() && zeroMask[i]) || (b.p && b.n >= bs[i].n)) continue;
            b.p.reset(new (std::nothrow) char[bs[i].n]);
            b.n = b.p ? bs[i].n : 0;
            if (!b.p) return NoHostMemory;
        }
        bufs = bs;
        size_t held = 0;
        for (size_t i = 0; i < bs.size(); ++i)
        {
            Blob& b = blobs[i];
            b.zero = i < zeroMask.size() && zeroMask[i];
            if (b.zero) continue;
            if (!gpuCheck(gpuMemcpy(b.p.get(), bs[i].p, bs[i].n, gpuMemcpyDeviceToHost), "checkpoint copy", err))
            { valid = false; return DeviceError; }
            held += bs[i].n;
        }
        start = at;
        valid = true;
        bytes = std::max(bytes, held);
        sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return Ok;
    }
    // false ('err' set) when the device refused the copy back: the grids are then in no known state
    bool restore(std::string& err)
    {
        auto const t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < bufs.size(); ++i)
        {
            bool const ok = blobs[i].zero
                ? gpuCheck(gpuMemset(bufs[i].p, 0, bufs[i].n), "checkpoint restore", err)
                : gpuCheck(gpuMemcpy(bufs[i].p, blobs[i].p.get(), bufs[i].n, gpuMemcpyHostToDevice), "checkpoint restore", err);
            if (!ok) { valid = false; return false; }
        }
        sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return true;
    }
};
