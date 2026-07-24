//  Standard-DTFE volume-averaged (method 1) grid interpolation -- Metal compute kernel.
//
//  One GPU thread per Delaunay tetrahedron, mirroring the CPU loop in
//  averaged_interpolation_1.cc: fields are linear inside each tetrahedron (constant
//  gradient from the inverse vertex-difference matrix). A tetrahedron that fits inside
//  one grid cell deposits its centroid value times its volume into that cell; larger
//  ones scatter value * (V_tet / nSamples) at shared quasi-random (Sobol) sample points.
//  The caller divides the accumulated grids by the grid-cell volume afterwards, exactly
//  like the CPU path.
//
//  The per-tet volume, sample count, and single-cell flat index are precomputed on the
//  CPU with the CPU loop's own helpers, so both paths use the same classification and
//  sample counts; the shared Sobol table gives them the same barycentric coordinates.
//  Remaining CPU/GPU differences are float rounding (double vertex differences on the
//  CPU) and atomic summation order.
//
//  Vertex positions arrive relative to the region's lower corner, so the grid index is
//  floor(pos/dx) directly; samples outside [0,nGrid) are dropped. Periodicity is
//  pre-baked into padded particle copies before triangulation (as on the CPU), so the
//  kernel needs no wrapping.

#include <metal_stdlib>
using namespace metal;

struct DepositParams {
    float dx[3];
    int   nGrid[3];
    uint  nTet;
    int   fDen;     // deposit density
    int   fVel;     // deposit velocity
    int   fGrad;    // deposit velocity gradient
};

static inline float det3(thread const float A[3][3]) {
    return A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])
         - A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])
         + A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
}

// Inverse of 3x3, mirroring the CPU matrixInverse(): |det| < 1e-6 -> ZERO matrix. The
// CPU then interpolates a constant field from the base vertex; it does NOT drop the
// tetrahedron, so neither do we (unlike the PS deposit, which conserves mass and drops).
static inline void inverse3zero(thread const float A[3][3], thread float inv[3][3]) {
    float d = det3(A);
    if (fabs(d) < 1.0e-6f) {
        for (int i=0;i<3;++i) for (int j=0;j<3;++j) inv[i][j]=0.0f;
        return;
    }
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
}

// One deposit of (den, vel, gradFlat) * w into grid cell 'flat'. Offsets are 64-bit BECAUSE
// flat*9 WOULD overflow a 32-bit uint above ~782^3 (learned in the PS deposit) -- already
// fixed, do not re-derive. The 2^32-cell cap on 'flat' is guarded in averaged_interpolation_1.cc.
static inline void depositCell(device atomic_float* denGrid,
                               device atomic_float* velGrid,
                               device atomic_float* gradGrid,
                               constant DepositParams& P,
                               ulong flat, float den,
                               thread const float vel[3],
                               thread const float gradFlat[9], float w)
{
    if (P.fDen)
        atomic_fetch_add_explicit(&denGrid[flat], den*w, memory_order_relaxed);
    if (P.fVel)
        for (int j=0;j<3;++j)
            atomic_fetch_add_explicit(&velGrid[flat*3ul+ulong(j)], vel[j]*w, memory_order_relaxed);
    if (P.fGrad)
        for (int q=0;q<9;++q)
            atomic_fetch_add_explicit(&gradGrid[flat*9ul+ulong(q)], gradFlat[q]*w, memory_order_relaxed);
}

kernel void depositAveraged1(
    device const float*     verts    [[buffer(0)]],   // nTet*12: 4 vertices, relative to region lower corner
    device const float*     dens     [[buffer(1)]],   // nTet*4:  vertex densities
    device const float*     vels     [[buffer(2)]],   // nTet*12: vertex velocities (dummy when fVel==fGrad==0)
    device const float*     vols     [[buffer(3)]],   // nTet:    tetrahedron volume (CPU value)
    device const uint*      cnts     [[buffer(4)]],   // nTet:    sample count; 0 = single-grid-cell fast path
    device const uint*      flats    [[buffer(5)]],   // nTet:    flat grid index for the fast path
    device const float*     sobol    [[buffer(6)]],   // maxNN*3: shared Sobol barycentric table
    device atomic_float*    denGrid  [[buffer(7)]],
    device atomic_float*    velGrid  [[buffer(8)]],
    device atomic_float*    gradGrid [[buffer(9)]],
    constant DepositParams& P        [[buffer(10)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= P.nTet) return;

    float base[3] = { verts[tid*12+0], verts[tid*12+1], verts[tid*12+2] };
    float Ax[3][3];   // vertex-difference matrix, rows = vertex k+1 - vertex 0
    for (int v=0; v<3; ++v)
        for (int d=0; d<3; ++d)
            Ax[v][d] = verts[tid*12+(v+1)*3+d] - base[d];

    float posInv[3][3];
    inverse3zero(Ax, posInv);

    bool const needVel = (P.fVel != 0) || (P.fGrad != 0);
    float u0[3] = {0.0f,0.0f,0.0f};
    float vG[3][3];                       // vG[i][j] = d(v_j)/d(x_i), like the CPU velocityGrad
    float gradFlat[9] = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
    if (needVel) {
        for (int j=0;j<3;++j) u0[j] = vels[tid*12+j];
        for (int i=0;i<3;++i)
            for (int j=0;j<3;++j) {
                float s=0.0f;
                for (int k=0;k<3;++k) s += posInv[i][k]*(vels[tid*12+(k+1)*3+j]-u0[j]);
                vG[i][j]=s;
            }
        for (int j=0;j<3;++j)             // CPU velocityGradient() layout: out[j*3+i] = vG[i][j]
            for (int i=0;i<3;++i)
                gradFlat[j*3+i] = vG[i][j];
    }

    float const vol = vols[tid];
    uint  const n   = cnts[tid];

    if (n == 0u) {   // whole tetrahedron inside one (in-range) grid cell: centroid value * volume
        float denC = 0.0f;
        for (int v=0; v<4; ++v) denC += dens[tid*4+v];
        denC *= 0.25f;
        float velC[3] = {0.0f,0.0f,0.0f};
        if (P.fVel) {
            for (int v=0;v<4;++v) for (int j=0;j<3;++j) velC[j] += vels[tid*12+v*3+j];
            for (int j=0;j<3;++j) velC[j] *= 0.25f;
        }
        depositCell(denGrid, velGrid, gradGrid, P, ulong(flats[tid]), denC, velC, gradFlat, vol);
        return;
    }

    float den0 = dens[tid*4+0];
    float densGrad[3];
    for (int i=0;i<3;++i) {
        float s=0.0f;
        for (int j=0;j<3;++j) s += posInv[i][j]*(dens[tid*4+j+1]-den0);
        densGrad[i]=s;
    }

    float const factor = vol / float(n);
    for (uint s=0; s<n; ++s) {
        float q0=sobol[s*3+0], q1=sobol[s*3+1], q2=sobol[s*3+2];
        float p[3];   // sample point relative to the base vertex (CPU quasiRandomPointsInCell)
        for (int d=0;d<3;++d) p[d] = q0*Ax[0][d] + q1*Ax[1][d] + q2*Ax[2][d];
        int pos[3]; bool in=true;
        for (int d=0;d<3;++d) {
            pos[d] = int(floor((base[d]+p[d])/P.dx[d]));
            if (pos[d]<0 || pos[d]>=P.nGrid[d]) { in=false; break; }
        }
        if (!in) continue;   // sample outside the region of interest
        ulong flat = (ulong(pos[0])*ulong(P.nGrid[1])+ulong(pos[1]))*ulong(P.nGrid[2])+ulong(pos[2]);
        float denS = den0;
        for (int i=0;i<3;++i) denS += p[i]*densGrad[i];
        float velS[3] = {0.0f,0.0f,0.0f};
        if (P.fVel)
            for (int j=0;j<3;++j) {
                velS[j]=u0[j];
                for (int i=0;i<3;++i) velS[j] += vG[i][j]*p[i];
            }
        depositCell(denGrid, velGrid, gradGrid, P, flat, denS, velS, gradFlat, factor);
    }
}
