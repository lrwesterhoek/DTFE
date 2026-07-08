/* Single-source CUDA/HIP portability shim for the .cu GPU hosts. The runtime APIs of CUDA
   and HIP mirror each other 1:1; kernel-side syntax (__global__, atomicAdd, blockIdx, ...)
   is identical in both dialects. Compiling:
     nvcc  (CUDA=1): __CUDACC__ path, links against cudart
     hipcc (HIP=1):  __HIP__ path (pass '-x hip' so .cu is treated as HIP), links amdhip64 */

#ifndef GPU_CUDA_COMPAT_HEADER
#define GPU_CUDA_COMPAT_HEADER

#if defined(__HIP__) || defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_runtime.h>
#define GPU_BACKEND_STRING     "HIP"
#define gpuError_t             hipError_t
#define gpuSuccess             hipSuccess
#define gpuGetErrorString      hipGetErrorString
#define gpuGetLastError        hipGetLastError
#define gpuMalloc              hipMalloc
#define gpuFree                hipFree
#define gpuMemcpy              hipMemcpy
#define gpuMemcpyHostToDevice  hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost  hipMemcpyDeviceToHost
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
#define gpuFree                cudaFree
#define gpuMemcpy              cudaMemcpy
#define gpuMemcpyHostToDevice  cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost  cudaMemcpyDeviceToHost
#define gpuMemset              cudaMemset
#define gpuDeviceSynchronize   cudaDeviceSynchronize
#define gpuGetDeviceCount      cudaGetDeviceCount
#define gpuGetDeviceProperties cudaGetDeviceProperties
#define gpuDeviceProp          cudaDeviceProp
#define gpuMemGetInfo          cudaMemGetInfo

#endif

#include <string>

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

#endif
