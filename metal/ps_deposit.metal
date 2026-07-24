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
    // --ps-volume-weighted: the velocity/dispersion/gradient moments carry equal EULERIAN-VOLUME
    // shares |det(Ax)|/6 / N instead of the mass shares (the mass grid always keeps mass), and
    // their normalizer is deposited into the momw grid (buffer 10; dummy when 0). Mutually
    // exclusive with fLinear (rejected at option parsing).
    int   fVolW;
    // --ps-caustics: atomic-OR the tet's orientation bits (1 = det(Ax)>0, 2 = det<0) into the
    // caustic grid (buffer 11; dummy when 0) at every deposited sample. OR is commutative and
    // idempotent, so the result is chunk-, thread- and partition-order invariant.
    int   fCaustic;
    // --ps-exact-deposit: replace the nSub^3 sub-sampled deposit by analytic tet-cell clipping
    // (float32 port of the vendored r3d, see the exact_* functions below). Ignores nSub; a tet
    // whose whole clip window comes up empty (sliver below float resolution) falls through to
    // the sampled path, whose empty window then drives the centroid fallback -- the same
    // fall-through the CPU exact deposit uses.
    int   fExact;
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

// ================================================================================================
// --ps-exact-deposit: float32 port of the vendored r3d (Powell & Abel 2015) specialized to ONE
// tetrahedron clipped by the 6 axis-aligned planes of a grid cell, with order-2 moments.
// Faithful to third_party/r3d/r3d.c (r3d_init_tet, r3d_clip, r3d_reduce): same vertex-graph
// clipping and the same Koehl (2012) moment recursion, hand-unrolled to order 2. Metal has no
// double, so results match the CPU exact deposit to FLOAT rounding -- the same parity contract
// as the sampled GPU deposit; the CPU path remains the double-precision reference. All geometry
// is vertex-0-relative (values of order the tet size), which keeps float32 well-conditioned.
// EXV bounds the vertex buffer: a tet clipped by <= 6 planes has <= 16 final and <= ~40
// transient vertices; on (unreachable) overflow the poly is emptied, dropping that cell's
// share -- the per-tet renormalization then redistributes it, so mass stays conserved.
// ================================================================================================
#define EXV 48

struct ExPoly {
    // packed_float3 (12 B) not float3 (16 B): 48 vertices x 4 padding bytes = 192 B/thread of
    // pure waste at float3, and this struct dominates the kernel's stack (1348 -> 1156 B).
    // Every access is a value read or whole-element store, so the packed type drops in.
    packed_float3 pos[EXV];
    int    nbr[EXV][3];
    int    nv;
};

static inline void exact_init_tet(thread ExPoly& T, float3 t1, float3 t2, float3 t3) {
    T.nv = 4;
    T.pos[0] = float3(0.0f);
    T.pos[1] = t1; T.pos[2] = t2; T.pos[3] = t3;
    T.nbr[0][0]=1; T.nbr[0][1]=3; T.nbr[0][2]=2;
    T.nbr[1][0]=2; T.nbr[1][1]=3; T.nbr[1][2]=0;
    T.nbr[2][0]=0; T.nbr[2][1]=3; T.nbr[2][2]=1;
    T.nbr[3][0]=1; T.nbr[3][1]=2; T.nbr[3][2]=0;
}

// r3d_clip for one axis-aligned plane sgn*pos[axis] + dd >= 0 (keeps the non-negative side).
static inline void exact_clip_plane(thread ExPoly& T, int axis, float sgn, float dd) {
    if (T.nv <= 0) return;
    float sdists[EXV];
    int   clipped[EXV];
    int const onv = T.nv;
    float smin = 1.0e30f, smax = -1.0e30f;
    for (int v=0; v<onv; ++v) {
        float const coord = (axis==0) ? T.pos[v].x : (axis==1 ? T.pos[v].y : T.pos[v].z);
        float const s = dd + sgn*coord;
        sdists[v] = s;
        clipped[v] = (s < 0.0f) ? 1 : 0;
        if (s < smin) smin = s;
        if (s > smax) smax = s;
    }
    if (smin >= 0.0f) return;             // fully inside this plane
    if (smax <= 0.0f) { T.nv = 0; return; }   // fully clipped away

    // insert a new vertex on every inside->outside edge (r3d: single-linked to the inside end)
    for (int vcur=0; vcur<onv; ++vcur) {
        if (clipped[vcur]) continue;
        for (int np=0; np<3; ++np) {
            int const vnext = T.nbr[vcur][np];
            if (!clipped[vnext]) continue;
            if (T.nv == EXV) { T.nv = 0; return; }   // overflow guard (see header comment)
            float const wa = -sdists[vnext], wb = sdists[vcur];
            T.pos[T.nv] = (wa*float3(T.pos[vcur]) + wb*float3(T.pos[vnext])) / (wa + wb);
            T.nbr[T.nv][0] = vcur;
            T.nbr[T.nv][1] = -1;
            T.nbr[T.nv][2] = -1;
            T.nbr[vcur][np] = T.nv;
            T.nv++;
        }
    }

    // walk around each face to doubly-link the new boundary vertices into the clip face
    for (int vstart=onv; vstart<T.nv; ++vstart) {
        int vcur = vstart;
        int vnext = T.nbr[vcur][0];
        int np = 0;
        do {
            for (np=0; np<3; ++np)
                if (T.nbr[vnext][np] == vcur) break;
            vcur = vnext;
            int const pnext = (np+1)%3;
            vnext = T.nbr[vcur][pnext];
        } while (vcur < onv);
        T.nbr[vstart][2] = vcur;
        T.nbr[vcur][1] = vstart;
    }

    // compress out the clipped vertices, reusing clipped[] as the re-index map (r3d-style);
    // vertices >= onv are the new (kept) boundary vertices
    int nun = 0;
    for (int v=0; v<T.nv; ++v) {
        bool const isClipped = (v < onv) && (clipped[v] != 0);
        if (!isClipped) {
            T.pos[nun] = T.pos[v];
            T.nbr[nun][0] = T.nbr[v][0];
            T.nbr[nun][1] = T.nbr[v][1];
            T.nbr[nun][2] = T.nbr[v][2];
            clipped[v] = nun++;
        }
    }
    T.nv = nun;
    for (int v=0; v<T.nv; ++v)
        for (int np=0; np<3; ++np)
            T.nbr[v][np] = clipped[T.nbr[v][np]];
}

// r3d_reduce, order 2: moments [1, x, y, z, x2, xy, xz, y2, yz, z2] of the polyhedron.
// The Koehl (2012) trinomial recursion is hand-unrolled: per triangle-fan triangle (v0,v1,v2),
// S1_a = a0+a1+a2, S2_aa = a0^2+a1^2+a2^2 + a0a1+a0a2+a1a2,
// S2_ab = 2(a0b0+a1b1+a2b2) + a0b1+a1b0 + a0b2+a2b0 + a1b2+a2b1, with the r3d normalizations
// 1/6, 1/24, 1/60 (diagonal) and 1/120 (off-diagonal).
static inline void exact_reduce2(thread const ExPoly& T, thread float* mom) {
    for (int m=0; m<10; ++m) mom[m] = 0.0f;
    if (T.nv <= 0) return;
    bool emk[EXV][3];
    for (int v=0; v<T.nv; ++v) { emk[v][0]=false; emk[v][1]=false; emk[v][2]=false; }

    for (int vstart=0; vstart<T.nv; ++vstart)
    for (int pstart=0; pstart<3; ++pstart) {
        if (emk[vstart][pstart]) continue;
        int pnext = pstart;
        int vcur  = vstart;
        emk[vcur][pnext] = true;
        int vnext = T.nbr[vcur][pnext];
        float3 const v0 = T.pos[vstart];
        int np;
        for (np=0; np<3; ++np)
            if (T.nbr[vnext][np] == vcur) break;
        vcur = vnext;
        pnext = (np+1)%3;
        emk[vcur][pnext] = true;
        vnext = T.nbr[vcur][pnext];
        while (vnext != vstart) {
            float3 const v2 = T.pos[vcur];
            float3 const v1 = T.pos[vnext];
            float const sixv = (-v2.x*v1.y*v0.z + v1.x*v2.y*v0.z + v2.x*v0.y*v1.z
                                -v0.x*v2.y*v1.z - v1.x*v0.y*v2.z + v0.x*v1.y*v2.z);
            mom[0] += sixv;
            float3 const s1 = v0 + v1 + v2;
            mom[1] += sixv*s1.x;  mom[2] += sixv*s1.y;  mom[3] += sixv*s1.z;
            mom[4] += sixv*(v0.x*v0.x + v1.x*v1.x + v2.x*v2.x + v0.x*v1.x + v0.x*v2.x + v1.x*v2.x);
            mom[5] += sixv*(2.0f*(v0.x*v0.y + v1.x*v1.y + v2.x*v2.y)
                            + v0.x*v1.y + v0.y*v1.x + v0.x*v2.y + v0.y*v2.x + v1.x*v2.y + v1.y*v2.x);
            mom[6] += sixv*(2.0f*(v0.x*v0.z + v1.x*v1.z + v2.x*v2.z)
                            + v0.x*v1.z + v0.z*v1.x + v0.x*v2.z + v0.z*v2.x + v1.x*v2.z + v1.z*v2.x);
            mom[7] += sixv*(v0.y*v0.y + v1.y*v1.y + v2.y*v2.y + v0.y*v1.y + v0.y*v2.y + v1.y*v2.y);
            mom[8] += sixv*(2.0f*(v0.y*v0.z + v1.y*v1.z + v2.y*v2.z)
                            + v0.y*v1.z + v0.z*v1.y + v0.y*v2.z + v0.z*v2.y + v1.y*v2.z + v1.z*v2.y);
            mom[9] += sixv*(v0.z*v0.z + v1.z*v1.z + v2.z*v2.z + v0.z*v1.z + v0.z*v2.z + v1.z*v2.z);
            for (np=0; np<3; ++np)
                if (T.nbr[vnext][np] == vcur) break;
            vcur = vnext;
            pnext = (np+1)%3;
            emk[vcur][pnext] = true;
            vnext = T.nbr[vcur][pnext];
        }
    }
    mom[0] /= 6.0f;
    mom[1] /= 24.0f;  mom[2] /= 24.0f;  mom[3] /= 24.0f;
    mom[4] /= 60.0f;  mom[7] /= 60.0f;  mom[9] /= 60.0f;
    mom[5] /= 120.0f; mom[6] /= 120.0f; mom[8] /= 120.0f;
}

// Order-1 variant for PASS 0 of the exact deposit, which consumes only the volume (and the
// first moments under fLinear): the six order-2 accumulations above are ~78% of the moment
// arithmetic and were computed twice per cell only to be discarded. IDENTICAL fan walk and
// accumulation order for mom[0..3], so the pass-0 weights -- and therefore every deposited
// value -- stay BIT-IDENTICAL to the full reduction.
static inline void exact_reduce1(thread const ExPoly& T, thread float* mom) {
    for (int m=0; m<4; ++m) mom[m] = 0.0f;
    if (T.nv <= 0) return;
    bool emk[EXV][3];
    for (int v=0; v<T.nv; ++v) { emk[v][0]=false; emk[v][1]=false; emk[v][2]=false; }

    for (int vstart=0; vstart<T.nv; ++vstart)
    for (int pstart=0; pstart<3; ++pstart) {
        if (emk[vstart][pstart]) continue;
        int pnext = pstart;
        int vcur  = vstart;
        emk[vcur][pnext] = true;
        int vnext = T.nbr[vcur][pnext];
        float3 const v0 = T.pos[vstart];
        int np;
        for (np=0; np<3; ++np)
            if (T.nbr[vnext][np] == vcur) break;
        vcur = vnext;
        pnext = (np+1)%3;
        emk[vcur][pnext] = true;
        vnext = T.nbr[vcur][pnext];
        while (vnext != vstart) {
            float3 const v2 = T.pos[vcur];
            float3 const v1 = T.pos[vnext];
            float const sixv = (-v2.x*v1.y*v0.z + v1.x*v2.y*v0.z + v2.x*v0.y*v1.z
                                -v0.x*v2.y*v1.z - v1.x*v0.y*v2.z + v0.x*v1.y*v2.z);
            mom[0] += sixv;
            float3 const s1 = v0 + v1 + v2;
            mom[1] += sixv*s1.x;  mom[2] += sixv*s1.y;  mom[3] += sixv*s1.z;
            for (np=0; np<3; ++np)
                if (T.nbr[vnext][np] == vcur) break;
            vcur = vnext;
            pnext = (np+1)%3;
            emk[vcur][pnext] = true;
            vnext = T.nbr[vcur][pnext];
        }
    }
    mom[0] /= 6.0f;
    mom[1] /= 24.0f;  mom[2] /= 24.0f;  mom[3] /= 24.0f;
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
    device atomic_float* mom;     // nCell*3   [flat*3+j]
    device atomic_float* m2;      // nCell*6   [flat*6+c]
    device atomic_float* grad;    // nCell*9   [flat*9+j*3+i]
    device atomic_uint*  streams;
    device atomic_float* momw;    // nCell     moment-weight normalizer (fVolW; else dummy)
    device atomic_uint*  caustic; // nCell     orientation bits, atomic OR (fCaustic; else dummy)
    device atomic_float* sv;      // nCell     exact multiplicity sum(V_int)/V_cell (fExact; else dummy)
    device atomic_float* dispvel; // nCell*3   sum(m_s v_s), the dispersion's own MASS-weighted mean (fVolW+fDisp; else dummy)
    device atomic_float* dispw;   // nCell     sum(m_s), its normalizer (fVolW+fDisp; else dummy)
};

// w = mass share (mass grid), wm = moment weight (== w by default; the V_eul share under
// fVolW), obits = the tet's orientation bits for the caustic OR (fCaustic), svShare = this
// sample's exact-multiplicity share (fExact fall-through path only).
static inline void depositSample(thread const FieldGrids& G, uint flat,
                                 thread const float rel[3], float w, float wm, uint obits,
                                 thread const float u0[3], thread const float vG[3][3],
                                 bool fVel, bool fDisp, bool fGrad, bool fVolW, bool fCaustic,
                                 bool fExact, float svShare)
{
    // Moment-grid offsets are 64-bit BECAUSE flat*9 WOULD overflow a 32-bit uint above ~782^3
    // and silently scatter atomics to low addresses. Already fixed -- do not re-derive.
    // The surviving 32-bit quantity is 'flat' ITSELF (built by the callers), which caps a
    // sub-grid at 2^32 cells (~1625^3); ps_interpolation.cc guards that with a CPU fallback.
    ulong base = ulong(flat);
    atomic_fetch_add_explicit(&G.mass[flat], w, memory_order_relaxed);
    if (fVel || fDisp) {
        float vv[3];
        for (int j=0; j<3; ++j) {
            vv[j]=u0[j];
            for (int i=0;i<3;++i) vv[j]+=vG[i][j]*rel[i];
            if (fVel)
                atomic_fetch_add_explicit(&G.mom[base*3ul+ulong(j)], vv[j]*wm, memory_order_relaxed);
        }
        if (fDisp) {
            // sigma_ij is a moment of f -> always MASS-weighted (w), even when the velocity
            // moments carry volume shares (wm) under fVolW.
            ulong c=0;
            for (int i=0;i<3;++i)
                for (int j=i;j<3;++j)
                    atomic_fetch_add_explicit(&G.m2[base*6ul + c++], w*vv[i]*vv[j], memory_order_relaxed);
            if (fVolW) {   // its own mass-weighted mean + normalizer (the mom grid is volume-weighted)
                for (int j=0;j<3;++j)
                    atomic_fetch_add_explicit(&G.dispvel[base*3ul+ulong(j)], vv[j]*w, memory_order_relaxed);
                atomic_fetch_add_explicit(&G.dispw[flat], w, memory_order_relaxed);
            }
        }
    }
    if (fGrad)
        for (int j=0;j<3;++j)
            for (int i=0;i<3;++i)
                atomic_fetch_add_explicit(&G.grad[base*9ul + ulong(j*3+i)], vG[i][j]*wm, memory_order_relaxed);
    if (fVolW)
        atomic_fetch_add_explicit(&G.momw[flat], wm, memory_order_relaxed);
    if (fCaustic)
        atomic_fetch_or_explicit(&G.caustic[flat], obits, memory_order_relaxed);
    if (fExact)
        atomic_fetch_add_explicit(&G.sv[flat], svShare, memory_order_relaxed);
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
    device atomic_float*       momwGrid [[buffer(10)]], // nCell moment-weight normalizer (fVolW; else dummy)
    device atomic_uint*        caustGrid[[buffer(11)]], // nCell orientation-bit OR grid (fCaustic; else dummy)
    device atomic_float*       svGrid   [[buffer(12)]], // nCell exact multiplicity sum(V_int)/V_cell (fExact; else dummy)
    device atomic_float*       dvGrid   [[buffer(13)]], // nCell*3 dispersion's mass-weighted mean (fVolW+fDisp; else dummy)
    device atomic_float*       dwGrid   [[buffer(14)]], // nCell   its normalizer (fVolW+fDisp; else dummy)
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

    FieldGrids G { massGrid, momGrid, m2Grid, gradGrid, strGrid, momwGrid, caustGrid, svGrid, dvGrid, dwGrid };

    // --ps-volume-weighted: the tet's total moment weight is its Eulerian volume |det|/6
    // (matching the CPU volumeShare); default: the moment weight equals the mass share.
    // --ps-caustics: orientation bits from the sign of det(Ax) (the Lagrangian->Eulerian
    // parity; CGAL cells are positively oriented in Lagrangian space).
    bool const fVolW = P.fVolW != 0, fCaustic = P.fCaustic != 0, fExact = P.fExact != 0;
    float const mw = fVolW ? fabs(d)/6.0f : m;
    uint  const obits = d > 0.0f ? 1u : 2u;
    float const invCellVol = 1.0f / (P.dx[0]*P.dx[1]*P.dx[2]);
    float const tetVol = fabs(d)/6.0f;   // exact-multiplicity share of a monolithic deposit

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
        // the exact deposit clips the bbox window directly (no sub-samples), so the +-1
        // sub-sample margin would only add empty clips -- same guard as the CPU path
        if (P.nSub>1 && !fExact) { iMin[dd]-=1; iMax[dd]+=1; }
        if (!P.periodic) { if (iMin[dd]<0) iMin[dd]=0; if (iMax[dd]>P.nGrid[dd]) iMax[dd]=P.nGrid[dd]; }
    }
    int nSub = P.nSub;
    uint nSamp = uint(nSub)*uint(nSub)*uint(nSub);

    // ===================== exact conservative deposit (--ps-exact-deposit) =====================
    // Mirrors the CPU exact path: PASS 0 sums the per-cell weights (V_int, or the exact linear-
    // profile integral under fLinear), PASS 1 renormalizes the shares to the tet mass and
    // deposits the exact moments. A tet whose whole window clips empty (sliver below float
    // resolution) falls through to the sampled two-pass below, whose empty window then drives
    // the centroid fallback -- exactly the CPU fall-through.
    if (fExact) {
        // r3d wants positive orientation: a folded (det<0) tet is handed over with
        // vertices 1,2 swapped; vertex 0 stays the frame origin (same as the CPU path)
        float3 t1 = v1 - v0, t2 = v2 - v0, t3 = v3 - v0;
        if (d < 0.0f) { float3 tt = t1; t1 = t2; t2 = tt; }

        float sumW = 0.0f;
        float shareFac = 0.0f;
        bool  deposited = false;
        for (int pass=0; pass<2; ++pass) {
            if (pass==1) {
                if (!(sumW > 0.0f)) break;   // empty window: sampled fallback below
                shareFac = m / sumW;
                deposited = true;
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
                if (l0<0) l0+=P.nGrid[0];   // sub-box may wrap a periodic axis (see ps_interpolation.cc)
                if (l1<0) l1+=P.nGrid[1];
                if (l2<0) l2+=P.nGrid[2];
                if (l0<0||l0>=P.subDims[0]||l1<0||l1>=P.subDims[1]||l2<0||l2>=P.subDims[2]) continue;
                uint flat=(uint(l0)*uint(P.subDims[1])+uint(l1))*uint(P.subDims[2])+uint(l2);

                // clip the tet against this cell (bounds in the vertex-0 frame, RAW indices)
                ExPoly piece;
                exact_init_tet(piece, t1, t2, t3);
                for (int dd=0; dd<3; ++dd) {
                    float const clo = P.boxLo[dd] + float(raw[dd])*P.dx[dd]
                                    - (dd==0 ? v0.x : (dd==1 ? v0.y : v0.z));
                    float const chi = clo + P.dx[dd];
                    exact_clip_plane(piece, dd,  1.0f, -clo);   //  x_d >= clo
                    exact_clip_plane(piece, dd, -1.0f,  chi);   //  x_d <= chi
                }
                if (piece.nv == 0) continue;
                float mo[10];
                if (pass==0) exact_reduce1(piece, mo);   // volume+first moments only: pass 0 discards the rest
                else         exact_reduce2(piece, mo);
                if (!(mo[0] > 0.0f)) continue;

                float w = mo[0];
                if (fLinear) {
                    w = d0*mo[0] + dG[0]*mo[1] + dG[1]*mo[2] + dG[2]*mo[3];
                    if (w < 0.0f) w = 0.0f;
                }
                if (pass==0) { sumW += w; continue; }

                float const massShare = w * shareFac;
                float const wmEx = fVolW ? mo[0] : massShare;
                ulong const base = ulong(flat);
                atomic_fetch_add_explicit(&G.mass[flat], massShare, memory_order_relaxed);
                float const invV = 1.0f / mo[0];
                float const cen[3] = { mo[1]*invV, mo[2]*invV, mo[3]*invV };
                float vb[3];
                if (fVel || fDisp) {
                    for (int j=0; j<3; ++j) {
                        vb[j] = u0[j];
                        for (int i=0; i<3; ++i) vb[j] += vG[i][j]*cen[i];
                        if (fVel)
                            atomic_fetch_add_explicit(&G.mom[base*3ul+ulong(j)], vb[j]*wmEx, memory_order_relaxed);
                    }
                    if (fDisp) {
                        // exact second moment of the linear profile: <v_a v_b> over the piece
                        // = vb_a vb_b + (G^T Cov G)_ab, Cov from the order-2 position moments
                        float cov[3][3];
                        cov[0][0] = mo[4]*invV - cen[0]*cen[0];
                        cov[0][1] = cov[1][0] = mo[5]*invV - cen[0]*cen[1];
                        cov[0][2] = cov[2][0] = mo[6]*invV - cen[0]*cen[2];
                        cov[1][1] = mo[7]*invV - cen[1]*cen[1];
                        cov[1][2] = cov[2][1] = mo[8]*invV - cen[1]*cen[2];
                        cov[2][2] = mo[9]*invV - cen[2]*cen[2];
                        ulong c2 = 0;
                        for (int a=0; a<3; ++a)
                            for (int b=a; b<3; ++b) {
                                float gcg = 0.0f;
                                for (int i=0; i<3; ++i)
                                    for (int k=0; k<3; ++k)
                                        gcg += vG[i][a]*cov[i][k]*vG[k][b];
                                atomic_fetch_add_explicit(&G.m2[base*6ul + c2++],
                                                          massShare*(vb[a]*vb[b] + gcg), memory_order_relaxed);
                            }
                        if (fVolW) {
                            for (int j=0;j<3;++j)
                                atomic_fetch_add_explicit(&G.dispvel[base*3ul+ulong(j)], vb[j]*massShare, memory_order_relaxed);
                            atomic_fetch_add_explicit(&G.dispw[flat], massShare, memory_order_relaxed);
                        }
                    }
                }
                if (fGrad)
                    for (int j=0; j<3; ++j)
                        for (int i=0; i<3; ++i)
                            atomic_fetch_add_explicit(&G.grad[base*9ul + ulong(j*3+i)], vG[i][j]*wmEx, memory_order_relaxed);
                if (fVolW)
                    atomic_fetch_add_explicit(&G.momw[flat], wmEx, memory_order_relaxed);
                if (fCaustic)
                    atomic_fetch_or_explicit(&G.caustic[flat], obits, memory_order_relaxed);
                // exact cell-mean multiplicity share V_int/V_cell, and the raw tet-touch count
                atomic_fetch_add_explicit(&svGrid[flat], mo[0]*invCellVol, memory_order_relaxed);
                atomic_fetch_add_explicit(&G.streams[flat], 1u, memory_order_relaxed);
            }
        }
        if (deposited) return;   // exact deposit complete; skip the sampled path
    }

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
                        // moment weight: equal V_eul share under fVolW (fLinear is rejected
                        // with fVolW at option parsing), else the mass share w
                        float wm = fVolW ? mw / float(N) : w;
                        // --ps-exact-deposit fall-through: share the tet volume over its N samples
                        float sv = fExact ? (tetVol*invCellVol)/float(N) : 0.0f;
                        depositSample(G, flat, rel, w, wm, obits, u0, vG, fVel, fDisp, fGrad, fVolW, fCaustic, fExact, sv);
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
            if (loc[dd]<0) loc[dd]+=P.nGrid[dd];             // sub-box may wrap a periodic axis
            if (loc[dd]<0||loc[dd]>=P.subDims[dd]) return;   // centroid outside this partition's box: drop
        }
        uint flat=(uint(loc[0])*uint(P.subDims[1])+uint(loc[1]))*uint(P.subDims[2])+uint(loc[2]);
        float crel[3]={cen.x-v0.x, cen.y-v0.y, cen.z-v0.z};
        depositSample(G, flat, crel, m, mw, obits, u0, vG, fVel, fDisp, fGrad, fVolW, fCaustic, fExact, tetVol*invCellVol);
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
