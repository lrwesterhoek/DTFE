//  PS-DTFE mass-conserving density deposit -- Metal compute kernel (prototype).
//
//  One GPU thread per tetrahedron. Mirrors the CPU deposit in ps_interpolation.cc:
//  each Lagrangian flow element carries a constant mass m = rho_bar * V_lag; the thread
//  splits m equally among the Eulerian sub-sample points inside the simplex and scatters
//  the shares into the (mass) grid with atomic float adds. Total deposited == m exactly,
//  so the field is mass-conserving at any nSub and for any V_eul (no caustic spikes).
//
//  Vertices arrive already min-image-wrapped relative to vertex 0 (host side), so the
//  kernel treats them as plain Eulerian coordinates.

#include <metal_stdlib>
using namespace metal;

struct DepositParams {
    float boxLo[3];
    float dx[3];
    int   nGrid[3];
    int   nSub;
    int   periodic;     // 0 / 1
    // Partition sub-grid (production PS-DTFE stores only the Eulerian bbox each Lagrangian
    // partition touches): output arrays are subDims-sized; a wrapped cell w maps to
    // loc = w - subOrigin and cells outside [0,subDims) are skipped BEFORE sample counting,
    // exactly like the inSub guard in ps_interpolation.cc. Full grid: subOrigin=0, subDims=nGrid.
    // Only depositFields honours these; depositDensity is full-grid (prototype use).
    int   subOrigin[3];
    int   subDims[3];
    // Field flags: which moment grids this run actually needs. When a flag is 0 the host binds
    // a 4-byte dummy buffer for that grid and the kernel never touches it (mass and streams are
    // always deposited). fVel = velocity or dispersion selected (mom), fDisp = dispersion (m2,
    // 24 B per cell), fGrad = velocity gradient (grad, 36 B per cell). Only depositFields
    // honours these; depositDensity deposits mass only.
    int   fVel;
    int   fDisp;
    int   fGrad;
    // --ps-linear-deposit: weight the interior samples by the DTFE-interpolated linear density
    // (from the per-tet vertex densities in buffer 9), renormalized per tet so the deposited
    // total still equals the tet mass exactly. 0 = uniform equal shares (buffer 9 is a dummy).
    int   fLinear;
    uint  nTet;
};

static inline float det3(thread const float A[3][3]) {
    return A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])
         - A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])
         + A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
}

// Inverse of 3x3; false if |det| < 1e-6 (matches the CPU matrixInverse() sentinel so a
// collapsed simplex is dropped rather than bounding-box filled).
static inline bool inverse3(thread const float A[3][3], thread float inv[3][3]) {
    float d = det3(A);
    if (fabs(d) < 1.0e-6f) return false;
    float invd = 1.0f / d;
    inv[0][0] =  (A[1][1]*A[2][2]-A[1][2]*A[2][1])*invd;
    inv[1][0] = -(A[1][0]*A[2][2]-A[1][2]*A[2][0])*invd;
    inv[2][0] =  (A[1][0]*A[2][1]-A[1][1]*A[2][0])*invd;
    inv[0][1] = -(A[0][1]*A[2][2]-A[0][2]*A[2][1])*invd;
    inv[1][1] =  (A[0][0]*A[2][2]-A[0][2]*A[2][0])*invd;
    inv[2][1] = -(A[0][0]*A[2][1]-A[0][1]*A[2][0])*invd;
    inv[0][2] =  (A[0][1]*A[1][2]-A[0][2]*A[1][1])*invd;
    inv[1][2] = -(A[0][0]*A[1][2]-A[0][2]*A[1][0])*invd;
    inv[2][2] =  (A[0][0]*A[1][1]-A[0][1]*A[1][0])*invd;
    return true;
}

static inline int wrapIdx(int g, int n) { return ((g % n) + n) % n; }

// --ps-linear-deposit sample weight: linear density at offset rel from vertex 0, clamped >= 0
// (the +-1e-6 barycentric tolerance can graze negative).
static inline float linearWeight(thread const float rel[3], float d0, thread const float dG[3]) {
    float w = d0 + dG[0]*rel[0] + dG[1]*rel[1] + dG[2]*rel[2];
    return w < 0.0f ? 0.0f : w;
}

// Sub-sample position for local sample index s (base-nSub digits) in cell (raw) along axis dd.
static inline float samplePos(uint s, int dd, int nSub, thread const int raw[3], constant DepositParams& P) {
    uint rem = s;
    for (int k=0; k<dd; ++k) rem /= uint(nSub);
    float fr = (nSub==1) ? 0.5f : (float(int(rem % uint(nSub))) + 0.5f)/float(nSub);
    return P.boxLo[dd] + (float(raw[dd])+fr)*P.dx[dd];
}

// ------------------------------------------------------------------------------------------------
// depositFields: full-field variant. Same geometry/mass-share algorithm as depositDensity, but each
// interior sample also deposits the mass-weighted velocity moment sum(w v_j), second moment
// sum(w v_i v_j) (upper triangle xx,xy,xz,yy,yz,zz), velocity-gradient moment sum(w dv_i/dx_j)
// (layout j*3+i, matching the CPU Pvector), and increments the per-sample stream counter.
// The host normalizes: vbar = mom/W, sigma_ij = m2/W - vbar_i vbar_j, density = mass/cellVol,
// streams = count/nSub^3 -- identical to ps_interpolation.cc.
// ------------------------------------------------------------------------------------------------
struct FieldGrids {
    device atomic_float* mass;
    device atomic_float* mom;    // nCell*3   [flat*3+j]
    device atomic_float* m2;     // nCell*6   [flat*6+c]
    device atomic_float* grad;   // nCell*9   [flat*9+j*3+i]
    device atomic_uint*  streams;
};

static inline void depositSample(thread const FieldGrids& G, uint flat,
                                 thread const float rel[3], float w,
                                 thread const float u0[3], thread const float vG[3][3],
                                 bool fVel, bool fDisp, bool fGrad)
{
    // moment-grid offsets in 64-bit: flat*9 overflows 32-bit uint for grids >= ~782^3
    // (production supports those via size_t), silently scattering atomics to low addresses.
    ulong base = ulong(flat);
    atomic_fetch_add_explicit(&G.mass[flat], w, memory_order_relaxed);
    if (fVel || fDisp) {
        float vv[3];
        for (int j=0; j<3; ++j) {
            vv[j]=u0[j];
            for (int i=0;i<3;++i) vv[j]+=vG[i][j]*rel[i];
            if (fVel)
                atomic_fetch_add_explicit(&G.mom[base*3ul+ulong(j)], vv[j]*w, memory_order_relaxed);
        }
        if (fDisp) {
            ulong c=0;
            for (int i=0;i<3;++i)
                for (int j=i;j<3;++j)
                    atomic_fetch_add_explicit(&G.m2[base*6ul + c++], w*vv[i]*vv[j], memory_order_relaxed);
        }
    }
    if (fGrad)
        for (int j=0;j<3;++j)
            for (int i=0;i<3;++i)
                atomic_fetch_add_explicit(&G.grad[base*9ul + ulong(j*3+i)], vG[i][j]*w, memory_order_relaxed);
    atomic_fetch_add_explicit(&G.streams[flat], 1u, memory_order_relaxed);
}

kernel void depositFields(
    device const float*        verts    [[buffer(0)]],  // nTet * 12 (4 verts * xyz)
    device const float*        vels     [[buffer(1)]],  // nTet * 12 (4 verts * vxyz)
    device const float*        masses   [[buffer(2)]],  // nTet
    device atomic_float*       massGrid [[buffer(3)]],
    device atomic_float*       momGrid  [[buffer(4)]],
    device atomic_float*       m2Grid   [[buffer(5)]],
    device atomic_float*       gradGrid [[buffer(6)]],
    device atomic_uint*        strGrid  [[buffer(7)]],
    constant DepositParams&    P        [[buffer(8)]],
    device const float*        dens     [[buffer(9)]],  // nTet * 4 vertex densities (fLinear; else dummy)
    uint tid [[thread_position_in_grid]])
{
    if (tid >= P.nTet) return;

    float3 v0 = float3(verts[tid*12+0],  verts[tid*12+1],  verts[tid*12+2]);
    float3 v1 = float3(verts[tid*12+3],  verts[tid*12+4],  verts[tid*12+5]);
    float3 v2 = float3(verts[tid*12+6],  verts[tid*12+7],  verts[tid*12+8]);
    float3 v3 = float3(verts[tid*12+9],  verts[tid*12+10], verts[tid*12+11]);
    float  m  = masses[tid];
    if (m <= 0.0f) return;

    float Ax[3][3];
    Ax[0][0]=v1.x-v0.x; Ax[0][1]=v1.y-v0.y; Ax[0][2]=v1.z-v0.z;
    Ax[1][0]=v2.x-v0.x; Ax[1][1]=v2.y-v0.y; Ax[1][2]=v2.z-v0.z;
    Ax[2][0]=v3.x-v0.x; Ax[2][1]=v3.y-v0.y; Ax[2][2]=v3.z-v0.z;

    float d = det3(Ax);
    float avgEdge2 = 0.0f;
    for (int v=0; v<3; ++v) { float l2=0.0f; for (int i=0;i<3;++i) l2+=Ax[v][i]*Ax[v][i]; avgEdge2+=l2; }
    avgEdge2 /= 3.0f;
    if (fabs(d) < 1.0e-6f * avgEdge2 * sqrt(avgEdge2)) return;

    float posInv[3][3];
    if (!inverse3(Ax, posInv)) return;

    // constant per-tet velocity gradient vG = posInv * (u_{k+1}-u_0); skipped entirely (and
    // the vels buffer never read -- it may be a 4-byte dummy) when no velocity-derived grid
    // is requested (fVel, fDisp and fGrad all 0, i.e. a density-only run)
    bool const fVel = P.fVel != 0, fDisp = P.fDisp != 0, fGrad = P.fGrad != 0;
    bool const needsVel = fVel || fDisp || fGrad;
    float u0[3] = { 0.0f, 0.0f, 0.0f };
    float vG[3][3] = { {0.0f,0.0f,0.0f}, {0.0f,0.0f,0.0f}, {0.0f,0.0f,0.0f} };
    if (needsVel) {
        for (int j=0;j<3;++j) u0[j] = vels[tid*12+j];
        float dV[3][3];
        for (int e=0;e<3;++e)
            for (int j=0;j<3;++j)
                dV[e][j] = vels[tid*12+(e+1)*3+j] - u0[j];
        for (int i=0;i<3;++i)
            for (int j=0;j<3;++j) {
                float s=0.0f; for (int k=0;k<3;++k) s+=posInv[i][k]*dV[k][j];
                vG[i][j]=s;
            }
    }

    FieldGrids G { massGrid, momGrid, m2Grid, gradGrid, strGrid };

    // --ps-linear-deposit: constant density gradient across the tet (same affine convention
    // as vG above); the vertex densities arrive per tet in the dens buffer.
    bool const fLinear = P.fLinear != 0;
    float d0 = 0.0f, dG[3] = {0.0f, 0.0f, 0.0f};
    if (fLinear) {
        d0 = dens[tid*4+0];
        float dd0[3];
        for (int e=0;e<3;++e) dd0[e] = dens[tid*4+e+1] - d0;
        for (int i=0;i<3;++i) { float s=0.0f; for (int k=0;k<3;++k) s+=posInv[i][k]*dd0[k]; dG[i]=s; }
    }

    float3 lo = min(min(v0,v1), min(v2,v3));
    float3 hi = max(max(v0,v1), max(v2,v3));
    float loA[3]={lo.x,lo.y,lo.z}, hiA[3]={hi.x,hi.y,hi.z};
    int iMin[3], iMax[3];
    for (int dd=0; dd<3; ++dd) {
        iMin[dd] = int(floor((loA[dd]-P.boxLo[dd])/P.dx[dd]));
        iMax[dd] = int(floor((hiA[dd]-P.boxLo[dd])/P.dx[dd])) + 1;
        if (P.nSub>1) { iMin[dd]-=1; iMax[dd]+=1; }
        if (!P.periodic) { if (iMin[dd]<0) iMin[dd]=0; if (iMax[dd]>P.nGrid[dd]) iMax[dd]=P.nGrid[dd]; }
    }
    int nSub = P.nSub;
    uint nSamp = uint(nSub)*uint(nSub)*uint(nSub);

    // Two identical passes over (cell, sample): pass 0 counts N (and, for the linear deposit,
    // sums the sample weights), pass 1 deposits m/N -- or m*w/sumW when fLinear, which totals
    // exactly m as well. The cell's wrap + sub-grid mapping is resolved BEFORE the sample loop
    // (like the CPU inSub guard), so N counts exactly the samples that pass 1 deposits -- no
    // mass is ever lost to invalid cells.
    uint N = 0;
    float sumW = 0.0f;
    float share = 0.0f;
    for (int pass=0; pass<2; ++pass) {
        if (pass==1) {
            if (N==0u) break;      // centroid fallback below
            share = m / float(N);
        }
        for (int gi=iMin[0]; gi<iMax[0]; ++gi)
        for (int gj=iMin[1]; gj<iMax[1]; ++gj)
        for (int gk=iMin[2]; gk<iMax[2]; ++gk) {
            int raw[3]={gi,gj,gk};
            int wg0=P.periodic?wrapIdx(gi,P.nGrid[0]):gi;
            int wg1=P.periodic?wrapIdx(gj,P.nGrid[1]):gj;
            int wg2=P.periodic?wrapIdx(gk,P.nGrid[2]):gk;
            if (wg0<0||wg0>=P.nGrid[0]||wg1<0||wg1>=P.nGrid[1]||wg2<0||wg2>=P.nGrid[2]) continue;
            int l0=wg0-P.subOrigin[0], l1=wg1-P.subOrigin[1], l2=wg2-P.subOrigin[2];
            if (l0<0||l0>=P.subDims[0]||l1<0||l1>=P.subDims[1]||l2<0||l2>=P.subDims[2]) continue;
            uint flat=(uint(l0)*uint(P.subDims[1])+uint(l1))*uint(P.subDims[2])+uint(l2);
            for (uint s=0; s<nSamp; ++s) {
                float rel[3] = { samplePos(s,0,nSub,raw,P)-v0.x,
                                 samplePos(s,1,nSub,raw,P)-v0.y,
                                 samplePos(s,2,nSub,raw,P)-v0.z };
                float sum=0.0f; bool inside=true;
                for (int vv=0; vv<3; ++vv) {
                    float bc=0.0f; for (int i=0;i<3;++i) bc+=posInv[i][vv]*rel[i];
                    if (bc<-1.0e-6f){inside=false;break;} sum+=bc;
                }
                if (inside && sum<=1.0f+1.0e-6f) {
                    if (pass==0) {
                        N++;
                        if (fLinear) sumW += linearWeight(rel, d0, dG);
                    } else {
                        float w = share;
                        if (fLinear && sumW > 0.0f)
                            w = m * linearWeight(rel, d0, dG) / sumW;
                        depositSample(G, flat, rel, w, u0, vG, fVel, fDisp, fGrad);
                    }
                }
            }
        }
    }

    if (N==0u) {   // centroid fallback (mass + fields at the centroid position)
        float3 cen = (v0+v1+v2+v3)*0.25f;
        float cA[3]={cen.x,cen.y,cen.z}; int loc[3];
        for (int dd=0; dd<3; ++dd) {
            int cg=int(floor((cA[dd]-P.boxLo[dd])/P.dx[dd]));
            int wg=P.periodic ? wrapIdx(cg,P.nGrid[dd]) : cg;
            if (wg<0||wg>=P.nGrid[dd]) return;
            loc[dd]=wg-P.subOrigin[dd];
            if (loc[dd]<0||loc[dd]>=P.subDims[dd]) return;   // centroid outside this partition's box: drop
        }
        uint flat=(uint(loc[0])*uint(P.subDims[1])+uint(loc[1]))*uint(P.subDims[2])+uint(loc[2]);
        float crel[3]={cen.x-v0.x, cen.y-v0.y, cen.z-v0.z};
        depositSample(G, flat, crel, m, u0, vG, fVel, fDisp, fGrad);
    }
}

kernel void depositDensity(
    device const float*        verts    [[buffer(0)]],  // nTet * 12 (4 verts * xyz)
    device const float*        masses   [[buffer(1)]],  // nTet
    device atomic_float*       massGrid [[buffer(2)]],  // nGrid[0]*nGrid[1]*nGrid[2]
    constant DepositParams&    P        [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= P.nTet) return;

    float3 v0 = float3(verts[tid*12+0],  verts[tid*12+1],  verts[tid*12+2]);
    float3 v1 = float3(verts[tid*12+3],  verts[tid*12+4],  verts[tid*12+5]);
    float3 v2 = float3(verts[tid*12+6],  verts[tid*12+7],  verts[tid*12+8]);
    float3 v3 = float3(verts[tid*12+9],  verts[tid*12+10], verts[tid*12+11]);
    float  m  = masses[tid];
    if (m <= 0.0f) return;

    float Ax[3][3];
    Ax[0][0]=v1.x-v0.x; Ax[0][1]=v1.y-v0.y; Ax[0][2]=v1.z-v0.z;
    Ax[1][0]=v2.x-v0.x; Ax[1][1]=v2.y-v0.y; Ax[1][2]=v2.z-v0.z;
    Ax[2][0]=v3.x-v0.x; Ax[2][1]=v3.y-v0.y; Ax[2][2]=v3.z-v0.z;

    float d = det3(Ax);
    float avgEdge2 = 0.0f;
    for (int v=0; v<3; ++v) { float l2=0.0f; for (int i=0;i<3;++i) l2+=Ax[v][i]*Ax[v][i]; avgEdge2+=l2; }
    avgEdge2 /= 3.0f;
    float edgeScale = avgEdge2 * sqrt(avgEdge2);
    if (fabs(d) < 1.0e-6f * edgeScale) return;

    float posInv[3][3];
    if (!inverse3(Ax, posInv)) return;

    float3 lo = min(min(v0,v1), min(v2,v3));
    float3 hi = max(max(v0,v1), max(v2,v3));
    float loA[3]={lo.x,lo.y,lo.z}, hiA[3]={hi.x,hi.y,hi.z};
    int iMin[3], iMax[3];
    for (int dd=0; dd<3; ++dd) {
        iMin[dd] = int(floor((loA[dd]-P.boxLo[dd])/P.dx[dd]));
        iMax[dd] = int(floor((hiA[dd]-P.boxLo[dd])/P.dx[dd])) + 1;
        if (P.nSub>1) { iMin[dd]-=1; iMax[dd]+=1; }
        if (!P.periodic) { if (iMin[dd]<0) iMin[dd]=0; if (iMax[dd]>P.nGrid[dd]) iMax[dd]=P.nGrid[dd]; }
    }

    int nSub = P.nSub;
    uint nSamp = uint(nSub)*uint(nSub)*uint(nSub);

    // PASS 1: count interior sub-sample points.
    uint N = 0;
    for (int gi=iMin[0]; gi<iMax[0]; ++gi)
    for (int gj=iMin[1]; gj<iMax[1]; ++gj)
    for (int gk=iMin[2]; gk<iMax[2]; ++gk) {
        int raw[3]={gi,gj,gk};
        for (uint s=0; s<nSamp; ++s) {
            float rel[3] = { samplePos(s,0,nSub,raw,P)-v0.x,
                             samplePos(s,1,nSub,raw,P)-v0.y,
                             samplePos(s,2,nSub,raw,P)-v0.z };
            float sum=0.0f; bool inside=true;
            for (int vv=0; vv<3; ++vv) {
                float bc=0.0f; for (int i=0;i<3;++i) bc+=posInv[i][vv]*rel[i];
                if (bc<-1.0e-6f){inside=false;break;} sum+=bc;
            }
            if (inside && sum<=1.0f+1.0e-6f) N++;
        }
    }

    // Centroid fallback: simplex smaller than the sub-sample spacing -> deposit whole mass at its
    // centroid cell (centroid always inside) so no mass is lost.
    if (N==0u) {
        float3 c = (v0+v1+v2+v3)*0.25f;
        float cA[3]={c.x,c.y,c.z}; int wg[3];
        bool ok=true;
        for (int dd=0; dd<3; ++dd) {
            int cg=int(floor((cA[dd]-P.boxLo[dd])/P.dx[dd]));
            wg[dd]=P.periodic ? wrapIdx(cg,P.nGrid[dd]) : cg;
            if (wg[dd]<0||wg[dd]>=P.nGrid[dd]) { ok=false; break; }
        }
        if (ok) {
            uint flat=(uint(wg[0])*uint(P.nGrid[1])+uint(wg[1]))*uint(P.nGrid[2])+uint(wg[2]);
            atomic_fetch_add_explicit(&massGrid[flat], m, memory_order_relaxed);
        }
        return;
    }

    // PASS 2: deposit m/N at each interior sub-sample point.
    float share = m / float(N);
    for (int gi=iMin[0]; gi<iMax[0]; ++gi)
    for (int gj=iMin[1]; gj<iMax[1]; ++gj)
    for (int gk=iMin[2]; gk<iMax[2]; ++gk) {
        int raw[3]={gi,gj,gk};
        int wg0=P.periodic?wrapIdx(gi,P.nGrid[0]):gi;
        int wg1=P.periodic?wrapIdx(gj,P.nGrid[1]):gj;
        int wg2=P.periodic?wrapIdx(gk,P.nGrid[2]):gk;
        if (wg0<0||wg0>=P.nGrid[0]||wg1<0||wg1>=P.nGrid[1]||wg2<0||wg2>=P.nGrid[2]) continue;
        uint flat=(uint(wg0)*uint(P.nGrid[1])+uint(wg1))*uint(P.nGrid[2])+uint(wg2);
        for (uint s=0; s<nSamp; ++s) {
            float rel[3] = { samplePos(s,0,nSub,raw,P)-v0.x,
                             samplePos(s,1,nSub,raw,P)-v0.y,
                             samplePos(s,2,nSub,raw,P)-v0.z };
            float sum=0.0f; bool inside=true;
            for (int vv=0; vv<3; ++vv) {
                float bc=0.0f; for (int i=0;i<3;++i) bc+=posInv[i][vv]*rel[i];
                if (bc<-1.0e-6f){inside=false;break;} sum+=bc;
            }
            if (inside && sum<=1.0f+1.0e-6f)
                atomic_fetch_add_explicit(&massGrid[flat], share, memory_order_relaxed);
        }
    }
}
