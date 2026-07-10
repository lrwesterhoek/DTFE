#ifndef PS_DEPOSIT_PARAMS_H
#define PS_DEPOSIT_PARAMS_H

#include <cstdint>

// Host-side parameter block of the PS-DTFE GPU deposit, shared by every backend host
// (ps_metal_host.cc, ps_gpu_cuda.cu, metal/validate_deposit.cpp). MUST match the MSL
// 'DepositParams' struct in metal/ps_deposit.metal byte-for-byte -- that copy is the one
// intentional duplicate (MSL cannot include this header).
struct PSDepositParams
{
    float    boxLo[3];
    float    dx[3];
    int32_t  nGrid[3];
    int32_t  nSub;
    int32_t  periodic;
    int32_t  subOrigin[3];   // partition sub-grid box (full grid: origin 0, dims = nGrid)
    int32_t  subDims[3];
    // field flags: when 0 the corresponding moment grid is a 4-byte dummy the kernel never
    // touches (mass and streams are always deposited)
    int32_t  fVel;      // velocity moments (velocity or dispersion selected)
    int32_t  fDisp;     // second moments (24 B per cell)
    int32_t  fGrad;     // velocity-gradient moments (36 B per cell)
    int32_t  fLinear;   // --ps-linear-deposit: density-weighted (renormalized) sample shares
    uint32_t nTet;
};

#endif
