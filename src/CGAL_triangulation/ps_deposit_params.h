#ifndef PS_DEPOSIT_PARAMS_H
#define PS_DEPOSIT_PARAMS_H

#include <cmath>
#include <cstdint>
#include <cstring>

// Per-tetrahedron record of the GPU deposit: PS_TET_STRIDE floats per tet in the 'verts'
// buffer, laid out as documented at TET_STRIDE in metal/ps_deposit.metal (which must match).
constexpr int PS_TET_STRIDE = 16;

// Fills one record. cv = the tet's four vertices in the CANONICAL frame of the exact inside
// test (ps_exact_inside.h canonicalCoord: every periodic image of a particle at one float
// position); gridLo/gridLen/nGrid = the deposit grid, as the CPU test forms its sample positions
// lo + ((g + frac) * len) / n; canonShift = how far the canonical frame sits above the grid (the
// box length when periodic, else 0); wrapL = the box length for the minimum-image edge wrap
// (0 = no wrap); centroidFlat = the flat sub-grid index of the centroid-fallback cell as the
// CPU deposit computes it (0xFFFFFFFF = outside this partition, dropped).
// Vertex 0 is stored as a cell index plus an in-cell fraction and the edges as exact double
// differences rounded once, so the kernel forms every sample offset from small numbers.
inline void psFillTetRecord(float *rec, float const cv[4][3], double const gridLo[3],
                            double const gridLen[3], int32_t const nGrid[3],
                            double const canonShift[3], double const wrapL[3],
                            uint32_t const centroidFlat)
{
    for (int d = 0; d < 3; ++d)
    {
        double const a0 = (double(cv[0][d]) - (gridLo[d] + canonShift[d])) * double(nGrid[d]) / gridLen[d];
        double const ic = std::floor(a0);
        int32_t const ic0 = int32_t(ic);
        std::memcpy(&rec[d], &ic0, sizeof(ic0));
        rec[3 + d] = float(a0 - ic);
    }
    for (int e = 0; e < 3; ++e)
        for (int d = 0; d < 3; ++d)
        {
            double r = double(cv[e+1][d]) - double(cv[0][d]);   // exact: a difference of floats
            if (wrapL[d] > 0.)
            {
                if (r >  0.5 * wrapL[d]) r -= wrapL[d];
                else if (r < -0.5 * wrapL[d]) r += wrapL[d];
            }
            rec[6 + e*3 + d] = float(r);
        }
    std::memcpy(&rec[15], &centroidFlat, sizeof(centroidFlat));
}

// Host-side parameter block of the PS-DTFE GPU deposit, shared by the Metal host
// (ps_metal_host.cc) and metal/validate_deposit.cpp. MUST match the MSL
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
    int32_t  fVolW;     // --ps-volume-weighted: moment weight = V_eul shares (mass grid unchanged); momw normalizer grid deposited
    int32_t  fCaustic;  // --ps-caustics: atomic-OR the per-tet orientation bits (1 = det>0, 2 = det<0) into the caustic grid
    int32_t  fExact;    // --ps-exact-deposit: analytic tet-cell clipping (float32 r3d port in the kernel) instead of nSub^3 sampling; nSub ignored
    uint32_t nTet;
    int32_t  fScal;     // scalar field (one component): vertex scalars in buffer 18, the moment grid in 19
    int32_t  fSGrad;    // scalar gradient (3 components): its moment grid in buffer 20
    int32_t  windowMode; // --ps-window: the sub-box is a WINDOW of the full grid -- count every sample / clipped
                         // piece of a tet (the full run's normalization), deposit only those inside the sub-box
};

#endif
