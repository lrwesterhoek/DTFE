/* Single-source CUDA/HIP portability shim for the .cu GPU hosts (ps_gpu_cuda.cu,
   dtfe_gpu_cuda.cu). The runtime APIs of CUDA and HIP mirror each other 1:1 and the
   kernel-side syntax (__global__, atomicAdd, blockIdx, ...) is identical, so one source
   compiles under both:
     nvcc  (CUDA=1): __CUDACC__ path, links against cudart
     hipcc (HIP=1):  __HIP__ path (pass '-x hip' so .cu is treated as HIP), links amdhip64
   The kernels are ports of the Metal kernels under metal/; the SIMD-group operations
   those use (simd_any, simd_min, simd_sum; 32 lanes on Apple GPUs) become the warp
   shuffles below -- 32 lanes on NVIDIA, warpSize (64 on CDNA, 32 on RDNA) on AMD. */

#ifndef GPU_CUDA_COMPAT_HEADER
#define GPU_CUDA_COMPAT_HEADER

#if defined(GPU_EMU)

// CPU emulation (make <target> CUDAEMU=1): everything the two branches below provide comes from
// gpu_cuda_emu.h -- the kernels and hosts are compiled unchanged as plain C++ (see that header).
#include "gpu_cuda_emu.h"

#elif defined(__HIP__) || defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_runtime.h>
#define GPU_BACKEND_STRING     "HIP"
#define gpuError_t             hipError_t
#define gpuSuccess             hipSuccess
#define gpuGetErrorString      hipGetErrorString
#define gpuGetLastError        hipGetLastError
#define gpuMalloc              hipMalloc
#define gpuFree(p)             ((void)hipFree(p))   /* only freed in cleanup paths: hipError_t is [[nodiscard]] */
#define gpuMemcpy              hipMemcpy
#define gpuMemcpyHostToDevice  hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost  hipMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define gpuMemset              hipMemset
#define gpuDeviceSynchronize   hipDeviceSynchronize
#define gpuGetDeviceCount      hipGetDeviceCount
#define gpuGetDeviceProperties hipGetDeviceProperties
#define gpuDeviceProp          hipDeviceProp_t
#define gpuMemGetInfo          hipMemGetInfo

#else

#include <cuda_runtime.h>
#define GPU_BACKEND_STRING     "CUDA"
#define gpuError_t             cudaError_t
#define gpuSuccess             cudaSuccess
#define gpuGetErrorString      cudaGetErrorString
#define gpuGetLastError        cudaGetLastError
#define gpuMalloc              cudaMalloc
#define gpuFree(p)             ((void)cudaFree(p))
#define gpuMemcpy              cudaMemcpy
#define gpuMemcpyHostToDevice  cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost  cudaMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuMemset              cudaMemset
#define gpuDeviceSynchronize   cudaDeviceSynchronize
#define gpuGetDeviceCount      cudaGetDeviceCount
#define gpuGetDeviceProperties cudaGetDeviceProperties
#define gpuDeviceProp          cudaDeviceProp
#define gpuMemGetInfo          cudaMemGetInfo

#endif

#include <string>

// A kernel launch: kernel<<<blocks, tpb, smem>>>(args) under nvcc / hipcc, the emulator's pool
// under GPU_EMU (plain C++ cannot parse the triple chevrons). 'kernel' may be a function pointer.
#if defined(GPU_EMU)
#define GPU_LAUNCH(kernel, blocks, tpb, smem, ...) GPU_EMU_LAUNCH(kernel, blocks, tpb, smem, __VA_ARGS__)
#else
#define GPU_LAUNCH(kernel, blocks, tpb, smem, ...) kernel<<<(blocks), (tpb), (smem)>>>(__VA_ARGS__)
#endif

// true on success; on failure stores "<what>: <error>" in 'err'
inline bool gpuCheck(gpuError_t e, const char* what, std::string& err)
{
    if (e != gpuSuccess)
    {
        err = std::string(what) + ": " + gpuGetErrorString(e);
        return false;
    }
    return true;
}

// ---- warp-level collectives (device side), the counterparts of Metal's simd_* functions.
// Every lane of the warp must call them from UNIFORM control flow (the callers keep their
// reductions in loops whose condition is itself a warp-any, exactly like the Metal kernels,
// and never return from a kernel before the reduction -- see dtfe_gpu_cuda.cu).
#if defined(GPU_EMU)
#define GPU_WARP (gpuemu::warpWidth())
inline unsigned gpuLane() { return gpuemu::lane(); }
inline bool gpuWarpAny(bool p) { return gpuemu::warpAny(p); }
inline unsigned gpuWarpMinU(unsigned v) { return gpuemu::warpMinU(v); }
inline float gpuWarpSumF(float v) { return gpuemu::warpSumF(v); }

#elif defined(__CUDACC__) || defined(__HIPCC__) || defined(__HIP__)

#if defined(__HIP__) || defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#define GPU_WARP warpSize                            // 64 on CDNA/GCN, 32 on RDNA (a compile-time constant per target)
__device__ __forceinline__ unsigned gpuLane() { return unsigned(threadIdx.x) % unsigned(GPU_WARP); }
__device__ __forceinline__ bool gpuWarpAny(bool p) { return __any(p ? 1 : 0) != 0; }
__device__ __forceinline__ unsigned gpuWarpMinU(unsigned v)
{
    for (int o = GPU_WARP / 2; o > 0; o >>= 1) { unsigned const w = __shfl_xor(v, o); v = w < v ? w : v; }
    return v;
}
__device__ __forceinline__ float gpuWarpSumF(float v)
{
    for (int o = GPU_WARP / 2; o > 0; o >>= 1) v += __shfl_xor(v, o);
    return v;
}
#else
#define GPU_WARP 32
#define GPU_FULL_MASK 0xffffffffu
__device__ __forceinline__ unsigned gpuLane() { return unsigned(threadIdx.x) & 31u; }
__device__ __forceinline__ bool gpuWarpAny(bool p) { return __any_sync(GPU_FULL_MASK, p ? 1 : 0) != 0; }
__device__ __forceinline__ unsigned gpuWarpMinU(unsigned v)
{
    for (int o = 16; o > 0; o >>= 1) { unsigned const w = __shfl_xor_sync(GPU_FULL_MASK, v, o); v = w < v ? w : v; }
    return v;
}
__device__ __forceinline__ float gpuWarpSumF(float v)
{
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(GPU_FULL_MASK, v, o);
    return v;
}
#endif

#endif // device compiler

#endif
