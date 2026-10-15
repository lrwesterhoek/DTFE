/* The standard-DTFE GPU deposit's exact sums: the units of its 64-bit fixed-point grids (2026-10-13), shared by
   the two GPU hosts of the standard binary (dtfe_metal_host.cc, dtfe_gpu_cuda.cu). Host-side only, plain C++ (no
   backend includes); the fixed-point rule itself is the phase-space mass grid's (ps_deposit_params.h:
   psMassFixedScale picks the unit, psMassFixedMass converts a cell back). Why the grids are integer sums, and how
   the kernels keep them independent of the order their atomics land in: metal/dtfe_deposit.metal's header. */
#ifndef DTFE_GPU_FIXED_H
#define DTFE_GPU_FIXED_H

#include <algorithm>
#include <cmath>
#include <vector>

#include "ps_deposit_params.h"

// The grids are 64-bit fixed-point sums (2026-10-13; the kernel's header says why): grid g holds integers in
// units of 1/scale[g], scale = 2^k, set per BATCH from a bound B of the sum of |share| over everything the batch
// can deposit -- no cell, holding part of that sum, can exceed it. psMassFixedScale (the PS mass grid's rule) puts
// 2B at 2^61..2^62 units, so B itself sits at 2^60..2^61: a factor 4 below the 2^63 of the signed words for the
// kernel's float rounding of a sample value (the inverse edge matrix of a tetrahedron near the degeneracy cut
// amplifies it; a few per cent at worst). B, per grid, sums V_tet * max|value| over the batch's tetrahedra: a
// sample, a clipped piece's centroid and the single-cell centroid all lie in the tetrahedron, where the linear
// interpolant is a convex combination of the vertex values as uploaded (the Monte-Carlo centring shift is in
// them already). The two gradients are per-tetrahedron constants: computed here in double from the kernel's own
// float edges and differences, with the degeneracy cut relaxed to half the kernel's (inverse3zero's 1e-6 of the
// row-length product), so a tetrahedron the float kernel keeps is never bounded as zero. The resolution this
// leaves: a 512^3 sub-domain of 1.5e7 cells puts a cell holding 1e-3 of the mean share at ~2^26 units -- 2^-26 of
// its own value, finer than the float (2^-24) the grid is handed over in. False (the batch keeps the float atomics) when a bound is not
// finite or has no usable power of two; an all-zero grid gets unit 1.
inline bool batchFixedScales(const std::vector<float>& verts, const std::vector<float>& dens, const std::vector<float>& vels,
                             const std::vector<float>& scal, const std::vector<float>& vols,
                             bool fDen, bool fVel, bool fGrad, bool fScal, bool fSGrad, float scale[5])
{
    size_t const nTet = vols.size();
    double B[5] = {0., 0., 0., 0., 0.};
    bool const needInv = fGrad || fSGrad;
    for (size_t t = 0; t < nTet; ++t)
    {
        double const V = std::fabs(double(vols[t]));
        if (fDen)
        {
            double m = 0.;
            for (int v = 0; v < 4; ++v) m = std::max(m, std::fabs(double(dens[t*4 + v])));
            B[0] += V * m;
        }
        if (fVel)
        {
            double m = 0.;
            for (int k = 0; k < 12; ++k) m = std::max(m, std::fabs(double(vels[t*12 + k])));
            B[1] += V * m;
        }
        if (fScal)
        {
            double m = 0.;
            for (int v = 0; v < 4; ++v) m = std::max(m, std::fabs(double(scal[t*4 + v])));
            B[3] += V * m;
        }
        if (!needInv) continue;
        // the kernel's edge matrix (float differences, as it forms them), its cofactors and determinant in double;
        // the gradient is cof^T df / det, so its largest component is max|sum_k cof[k][i] df[k]| / |det|. The cut
        // compares squares (no square roots: this loop runs over every tetrahedron of the batch)
        double A[3][3];
        for (int v = 0; v < 3; ++v)
            for (int d = 0; d < 3; ++d)
                A[v][d] = double(float(verts[t*12 + (v+1)*3 + d] - verts[t*12 + d]));
        double const C[3][3] = {   // C[i][k] = the inverse's [i][k] times det (inverse3zero's formulas)
            {  A[1][1]*A[2][2]-A[1][2]*A[2][1], -(A[0][1]*A[2][2]-A[0][2]*A[2][1]),  A[0][1]*A[1][2]-A[0][2]*A[1][1] },
            { -(A[1][0]*A[2][2]-A[1][2]*A[2][0]),  A[0][0]*A[2][2]-A[0][2]*A[2][0], -(A[0][0]*A[1][2]-A[0][2]*A[1][0]) },
            {  A[1][0]*A[2][1]-A[1][1]*A[2][0], -(A[0][0]*A[2][1]-A[0][1]*A[2][0]),  A[0][0]*A[1][1]-A[0][1]*A[1][0] } };
        double const det = A[0][0]*C[0][0] + A[0][1]*C[1][0] + A[0][2]*C[2][0];
        double rn2 = 1.;
        for (int i = 0; i < 3; ++i) rn2 *= A[i][0]*A[i][0] + A[i][1]*A[i][1] + A[i][2]*A[i][2];
        if (!(det * det > 0.25e-12 * rn2)) continue;      // |det| <= 0.5e-6 rn: the kernel zeroes this one's gradients too
        double const Vd = V / std::fabs(det);
        // max over i of |sum_k C[i][k] * df[k]| for the component differences df (the kernel's float ones)
        auto gradMax = [&](const float* f0, size_t stride, int comps) {
            double m = 0.;
            for (int j = 0; j < comps; ++j)
            {
                double df[3];
                for (int k = 0; k < 3; ++k) df[k] = double(float(f0[(k+1)*stride + j] - f0[j]));
                for (int i = 0; i < 3; ++i)
                    m = std::max(m, std::fabs(C[i][0]*df[0] + C[i][1]*df[1] + C[i][2]*df[2]));
            }
            return m;
        };
        if (fGrad)  B[2] += Vd * gradMax(&vels[t*12], 3, 3);
        if (fSGrad) B[4] += Vd * gradMax(&scal[t*4], 1, 1);
    }
    bool const on[5] = { fDen, fVel, fGrad, fScal, fSGrad };
    for (int g = 0; g < 5; ++g)
    {
        scale[g] = 1.f;
        if (!on[g] || B[g] == 0.) continue;
        if (!psMassFixedScale(2. * B[g], scale[g])) return false;
    }
    return true;
}

#endif
