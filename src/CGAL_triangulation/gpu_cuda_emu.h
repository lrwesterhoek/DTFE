/* CPU emulation of the CUDA device and runtime API that ps_gpu_cuda.cu / dtfe_gpu_cuda.cu use,
   so the UNCHANGED kernels and hosts of the CUDA/HIP backend run on a machine without an NVIDIA
   or AMD GPU and can be checked against the CPU paths with the same suites the Metal backend
   passes ('make <target> CUDAEMU=1'; selected by -DGPU_EMU, see gpu_cuda_compat.h).

   What it emulates, and how faithfully:
   - a kernel launch runs the grid ONE BLOCK AT A TIME (so one dynamic shared-memory buffer
     serves every block); a block's warps are OS threads and each warp's lanes are coroutines
     run in lockstep at the collectives, like a SIMT unit -- a lane that calls a warp collective
     or __syncthreads() yields, and the warp's scheduler resumes it with the result once every
     lane has arrived there. Non-uniform control flow (lanes of a warp at different collectives,
     a lane exited before one) aborts with a message instead of hanging silently;
   - the warp collectives of gpu_cuda_compat.h (warp-any, warp-min, warp-sum) are computed at
     the COLLECTIVE level over the lanes' values -- the kernels' uniform control flow around them
     is exercised exactly, the shuffle loops themselves are not; GPU_EMU_WARP=64 runs the
     reductions over 64 lanes, the HIP wavefront width;
   - atomics are real atomics (the lanes are real threads); device memory is host memory;
     cudaMemcpy is memcpy; every runtime call succeeds; the "device" reports one CPU.
   It validates the port's transliteration and host logic, NOT the memory model, scheduling or
   numerics of real hardware -- a run on an NVIDIA/AMD machine remains the final word. */

#ifndef GPU_CUDA_EMU_HEADER
#define GPU_CUDA_EMU_HEADER

#include <algorithm>
#include <atomic>
#include <barrier>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#ifdef __APPLE__
#include <sys/sysctl.h>
#else
#include <unistd.h>
#endif

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __launch_bounds__(n)
#define __shared__

struct float4 { float x, y, z, w; };
struct uint2  { unsigned x, y; };


namespace gpuemu {

struct Idx { unsigned x; };
inline thread_local unsigned tThread = 0;    // the lane's index in the block (set before every resume)
inline thread_local unsigned tBlock  = 0;    // the block being run
inline unsigned gBlockDim = 256;
inline unsigned gGridDim  = 1;

inline unsigned warpWidth()
{
    static unsigned w = [] {
        const char* e = getenv("GPU_EMU_WARP");
        long v = e ? atol(e) : 32;
        return unsigned((v == 64) ? 64 : 32);
    }();
    return w;
}

// ---- SIMT emulation. A block of tpb threads is tpb/W warps; each warp is ONE OS thread that runs
// its W lanes as coroutines (a register-only stack switch), in lockstep at the collectives: a lane that calls a warp
// collective or __syncthreads() yields to the warp's scheduler, which resumes the next lane; when
// every live lane of the warp waits at the SAME collective the scheduler computes it (any / min /
// sum over the lanes' values) and resumes them with the result. __syncthreads() additionally joins
// the block's warp threads on a barrier. Non-uniform control flow -- lanes of one warp waiting at
// different collectives, or a lane that exited before one -- is a kernel bug on the GPU too and
// aborts here with a message. Blocks run one after another (one shared-memory buffer).
enum LaneState { L_READY, L_WAIT_WARP, L_WAIT_BLOCK, L_DONE };
enum Collective { C_ANY, C_MINU, C_SUMF };

// gpuemu_switch(&fromSp, toSp): saves the callee-saved registers on the current stack, stores the
// stack pointer into *fromSp, switches to the stack toSp and restores the registers saved there --
// a register-only coroutine switch (the libc routines cost a signal-mask syscall per switch, which
// made a 32^3 test take ten minutes). A fresh stack is prepared so that the first switch "returns"
// into the lane trampoline (prepareStack). The arm64 switch is verified (Apple Silicon); the x86-64
// one is written to the System V convention but has not been run yet.
extern "C" void gpuemu_switch(void** fromSp, void* toSp);
#if defined(__APPLE__)
#define GPUEMU_SYM "_gpuemu_switch"
#else
#define GPUEMU_SYM "gpuemu_switch"
#endif
#if defined(__aarch64__) || defined(__arm64__)
asm(".text\n"
    ".globl " GPUEMU_SYM "\n"
    ".p2align 2\n"
    GPUEMU_SYM ":\n"
    "  sub sp, sp, #0xb0\n"
    "  stp x19, x20, [sp, #0x00]\n"
    "  stp x21, x22, [sp, #0x10]\n"
    "  stp x23, x24, [sp, #0x20]\n"
    "  stp x25, x26, [sp, #0x30]\n"
    "  stp x27, x28, [sp, #0x40]\n"
    "  stp x29, x30, [sp, #0x50]\n"
    "  stp d8,  d9,  [sp, #0x60]\n"
    "  stp d10, d11, [sp, #0x70]\n"
    "  stp d12, d13, [sp, #0x80]\n"
    "  stp d14, d15, [sp, #0x90]\n"
    "  mov x2, sp\n"
    "  str x2, [x0]\n"
    "  mov sp, x1\n"
    "  ldp x19, x20, [sp, #0x00]\n"
    "  ldp x21, x22, [sp, #0x10]\n"
    "  ldp x23, x24, [sp, #0x20]\n"
    "  ldp x25, x26, [sp, #0x30]\n"
    "  ldp x27, x28, [sp, #0x40]\n"
    "  ldp x29, x30, [sp, #0x50]\n"
    "  ldp d8,  d9,  [sp, #0x60]\n"
    "  ldp d10, d11, [sp, #0x70]\n"
    "  ldp d12, d13, [sp, #0x80]\n"
    "  ldp d14, d15, [sp, #0x90]\n"
    "  add sp, sp, #0xb0\n"
    "  ret\n");
#define GPUEMU_FRAME 0xb0
#define GPUEMU_LR_SLOT 0x58
#elif defined(__x86_64__)
asm(".text\n"
    ".globl " GPUEMU_SYM "\n"
    ".p2align 4\n"
    GPUEMU_SYM ":\n"
    "  pushq %rbp\n  pushq %rbx\n  pushq %r12\n  pushq %r13\n  pushq %r14\n  pushq %r15\n"
    "  movq %rsp, (%rdi)\n"
    "  movq %rsi, %rsp\n"
    "  popq %r15\n  popq %r14\n  popq %r13\n  popq %r12\n  popq %rbx\n  popq %rbp\n"
    "  ret\n");
#define GPUEMU_FRAME 0x38
#define GPUEMU_LR_SLOT 0x30
#else
#error "gpu_cuda_emu.h: no context switch for this architecture (arm64 and x86-64 are supported)"
#endif

struct Lane
{
    void* sp = nullptr;                // the saved stack pointer while not running
    std::vector<char> stack;
    LaneState state = L_READY;
    Collective kind = C_ANY;
    float    fval = 0.f, fres = 0.f;
    unsigned uval = 0, ures = 0;
};

struct Warp
{
    unsigned index = 0;                // within the block
    std::vector<Lane> lanes;
    void* schedSp = nullptr;
    const std::function<void()>* body = nullptr;
    Lane* current = nullptr;
    unsigned currentLane = 0;
};

inline thread_local Warp* tWarp = nullptr;

// the first instruction of every lane: runs the kernel body, then hands the stack back for good
inline void laneTrampoline()
{
    Warp* w = tWarp;
    (*w->body)();
    Lane* L = w->current;
    L->state = L_DONE;
    gpuemu_switch(&L->sp, w->schedSp);
    abort();                           // a finished lane is never resumed
}

// a fresh stack whose first switch returns into the trampoline (saved registers all zero)
inline void prepareStack(Lane& L)
{
    uintptr_t top = reinterpret_cast<uintptr_t>(L.stack.data() + L.stack.size()) & ~uintptr_t(15);
#if defined(__x86_64__)
    top -= 8;                          // after 'ret' into the trampoline, rsp must be 8 mod 16, as at a call
#endif
    char* frame = reinterpret_cast<char*>(top - GPUEMU_FRAME);
    std::memset(frame, 0, GPUEMU_FRAME);
    void (*tramp)() = laneTrampoline;
    std::memcpy(frame + GPUEMU_LR_SLOT, &tramp, sizeof(tramp));
    L.sp = frame;
}

inline void resumeLane(Warp& w, unsigned l)
{
    Lane& L = w.lanes[l];
    w.current = &L;
    w.currentLane = l;
    tThread = w.index * warpWidth() + l;
    gpuemu_switch(&w.schedSp, L.sp);
}

[[noreturn]] inline void nonUniform(const char* what)
{
    fprintf(stderr, "gpu_cuda_emu: NON-UNIFORM control flow in a warp (%s): a kernel bug that would hang or "
                    "corrupt a real GPU too\n", what);
    abort();
}

// runs one block's worth of this warp's lanes: returns when every lane is done; 'blockSync' is
// called when all live lanes wait at __syncthreads()
template <class BlockSync>
inline void runWarpBlock(Warp& w, unsigned block, BlockSync&& blockSync)
{
    unsigned const W = warpWidth();
    tBlock = block;
    for (unsigned l = 0; l < W; ++l)
    {
        w.lanes[l].state = L_READY;
        prepareStack(w.lanes[l]);
    }
    while (true)
    {
        unsigned nDone = 0, nWarp = 0, nBlock = 0;
        for (unsigned l = 0; l < W; ++l)
            if (w.lanes[l].state == L_READY) resumeLane(w, l);
        for (unsigned l = 0; l < W; ++l)
        {
            LaneState const s = w.lanes[l].state;
            nDone += s == L_DONE; nWarp += s == L_WAIT_WARP; nBlock += s == L_WAIT_BLOCK;
        }
        if (nDone == W) return;
        if (nWarp && (nBlock || nDone)) nonUniform(nBlock ? "a warp collective against __syncthreads" : "a lane exited before a warp collective");
        if (nWarp == W)
        {
            Collective const kind = w.lanes[0].kind;
            bool any = false; unsigned mn = 0xffffffffu; float sum = 0.f;
            for (unsigned l = 0; l < W; ++l)
            {
                Lane& L = w.lanes[l];
                if (L.kind != kind) nonUniform("different collectives");
                any = any || (L.fval != 0.f); mn = L.uval < mn ? L.uval : mn; sum += L.fval;
            }
            for (unsigned l = 0; l < W; ++l)
            {
                Lane& L = w.lanes[l];
                L.fres = kind == C_ANY ? (any ? 1.f : 0.f) : sum; L.ures = mn;
                L.state = L_READY;
            }
            continue;
        }
        if (nBlock + nDone == W)
        {   // every live lane waits at __syncthreads(): join the block, then resume them
            blockSync();
            for (unsigned l = 0; l < W; ++l)
                if (w.lanes[l].state == L_WAIT_BLOCK) w.lanes[l].state = L_READY;
            continue;
        }
        nonUniform("mixed lane states");
    }
}

inline void launch(unsigned nBlocks, unsigned tpb, size_t /*smem*/, const std::function<void()>& body)
{
    if (nBlocks == 0) return;
    unsigned const W = warpWidth();
    if (tpb % W != 0 || tpb == 0)
    {
        fprintf(stderr, "gpu_cuda_emu: threads per block (%u) must be a multiple of the warp width (%u)\n", tpb, W);
        abort();
    }
    unsigned const nWarps = tpb / W;
    gBlockDim = tpb;
    gGridDim  = nBlocks;
    std::barrier<> blockBar(nWarps);     // __syncthreads() and the end of every block
    std::vector<std::thread> threads;
    for (unsigned wi = 0; wi < nWarps; ++wi)
        threads.emplace_back([&, wi] {
            Warp w;
            w.index = wi;
            w.body = &body;
            w.lanes.resize(W);
            for (Lane& L : w.lanes) L.stack.resize(size_t(256) << 10);   // 256 KB per lane
            tWarp = &w;
            for (unsigned b = 0; b < nBlocks; ++b)
            {
                runWarpBlock(w, b, [&] { blockBar.arrive_and_wait(); });
                blockBar.arrive_and_wait();   // the block is complete before anybody starts the next
            }
        });
    for (auto& t : threads) t.join();
}

inline Idx threadIdxV() { return Idx{tThread}; }
inline Idx blockIdxV()  { return Idx{tBlock}; }
inline Idx blockDimV()  { return Idx{gBlockDim}; }
inline Idx gridDimV()   { return Idx{gGridDim}; }

// ---- the warp collectives and the block barrier, as seen from a lane
inline unsigned lane() { return tThread % warpWidth(); }

inline void yieldCollective(Collective kind, float f, unsigned u)
{
    Warp* w = tWarp;
    Lane* L = w->current;
    L->kind = kind; L->fval = f; L->uval = u; L->state = L_WAIT_WARP;
    gpuemu_switch(&L->sp, w->schedSp);
}
inline bool warpAny(bool p)          { yieldCollective(C_ANY,  p ? 1.f : 0.f, 0u); return tWarp->current->fres != 0.f; }
inline unsigned warpMinU(unsigned v) { yieldCollective(C_MINU, 0.f, v);           return tWarp->current->ures; }
inline float warpSumF(float v)       { yieldCollective(C_SUMF, v, 0u);            return tWarp->current->fres; }
inline void syncThreads()
{
    Warp* w = tWarp;
    Lane* L = w->current;
    L->state = L_WAIT_BLOCK;
    gpuemu_switch(&L->sp, w->schedSp);
}

} // namespace gpuemu

#define threadIdx (gpuemu::threadIdxV())
#define blockIdx  (gpuemu::blockIdxV())
#define blockDim  (gpuemu::blockDimV())
#define gridDim   (gpuemu::gridDimV())
inline void __syncthreads() { gpuemu::syncThreads(); }

// ---- device builtins the kernels use
inline float atomicAdd(float* p, float v)
{
    std::atomic_ref<float> r(*p);
    float old = r.load(std::memory_order_relaxed);
    while (!r.compare_exchange_weak(old, old + v, std::memory_order_relaxed)) {}
    return old;
}
inline unsigned atomicAdd(unsigned* p, unsigned v) { return std::atomic_ref<unsigned>(*p).fetch_add(v, std::memory_order_relaxed); }
inline unsigned atomicOr(unsigned* p, unsigned v)  { return std::atomic_ref<unsigned>(*p).fetch_or(v, std::memory_order_relaxed); }
inline unsigned atomicCAS(unsigned* p, unsigned cmp, unsigned val)
{
    std::atomic_ref<unsigned>(*p).compare_exchange_strong(cmp, val, std::memory_order_relaxed);
    return cmp;      // the old value either way
}
inline int      __float_as_int(float f)   { int i;      std::memcpy(&i, &f, 4); return i; }
inline unsigned __float_as_uint(float f)  { unsigned u; std::memcpy(&u, &f, 4); return u; }
inline float    __int_as_float(int i)     { float f;    std::memcpy(&f, &i, 4); return f; }
inline int      min(int a, int b) { return a < b ? a : b; }
inline int      max(int a, int b) { return a > b ? a : b; }
inline unsigned min(unsigned a, unsigned b) { return a < b ? a : b; }
inline unsigned max(unsigned a, unsigned b) { return a > b ? a : b; }

// ---- the runtime API (the gpu* names of gpu_cuda_compat.h)
enum gpuError_t { gpuSuccess = 0, gpuErrorMemoryAllocation = 2 };
inline const char* gpuGetErrorString(gpuError_t e) { return e == gpuSuccess ? "no error" : "out of memory"; }
inline gpuError_t gpuGetLastError() { return gpuSuccess; }
inline gpuError_t gpuMalloc(void** p, size_t n) { *p = std::malloc(n ? n : 4); return *p ? gpuSuccess : gpuErrorMemoryAllocation; }
inline gpuError_t gpuFree(void* p) { std::free(p); return gpuSuccess; }
#define gpuMemcpyHostToDevice   1
#define gpuMemcpyDeviceToHost   2
#define gpuMemcpyDeviceToDevice 3
inline gpuError_t gpuMemcpy(void* d, const void* s, size_t n, int) { if (n) std::memcpy(d, s, n); return gpuSuccess; }
inline gpuError_t gpuMemset(void* p, int v, size_t n) { if (n) std::memset(p, v, n); return gpuSuccess; }
inline gpuError_t gpuDeviceSynchronize() { return gpuSuccess; }
inline gpuError_t gpuGetDeviceCount(int* n) { *n = 1; return gpuSuccess; }
struct gpuDeviceProp { char name[256]; };
inline gpuError_t gpuGetDeviceProperties(gpuDeviceProp* p, int)
{
    std::snprintf(p->name, sizeof(p->name), "CPU emulation of the CUDA backend, %u-lane warps", gpuemu::warpWidth());
    return gpuSuccess;
}
inline gpuError_t gpuMemGetInfo(size_t* freeB, size_t* totalB)
{
    uint64_t mem = 0;
#ifdef __APPLE__
    size_t len = sizeof(mem);
    if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) != 0) mem = 0;
#else
    long pages = sysconf(_SC_PHYS_PAGES), psz = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && psz > 0) mem = uint64_t(pages) * uint64_t(psz);
#endif
    if (!mem) mem = uint64_t(16) << 30;
    *totalB = size_t(mem);
    *freeB  = size_t(mem / 2);
    return gpuSuccess;
}
#define GPU_BACKEND_STRING "CUDA (CPU emulation)"

// kernel launches: GPU_LAUNCH in gpu_cuda_compat.h expands to this
#define GPU_EMU_LAUNCH(kernel, blocks, tpb, smem, ...) \
    gpuemu::launch((blocks), (tpb), (smem), [&] { kernel(__VA_ARGS__); })

#endif
