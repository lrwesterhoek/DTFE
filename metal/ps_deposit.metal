//  PS-DTFE mass-conserving deposit -- Metal compute kernel.
//
//  One GPU thread per tetrahedron. Mirrors the CPU deposit in ps_interpolation.cc:
//  each Lagrangian flow element carries a constant mass m = rho_bar * V_lag; the thread
//  splits m equally among the Eulerian sub-sample points inside the simplex and scatters
//  the shares into the (mass) grid with atomic float adds. Total deposited == m exactly,
//  so the field is mass-conserving at any nSub and for any V_eul (no caustic spikes).
//
//  Which samples a tetrahedron contains is decided EXACTLY, i.e. identically to the CPU
//  deposit's test (src/CGAL_triangulation/ps_exact_inside.h): the host hands every tet over in
//  the canonical frame of that test (TET_STRIDE record below), and the kernel classifies each
//  sample in float with a RIGOROUS error bound (classifySample). Outside the bound the float
//  decision provably equals the exact one. A tet with any sample inside the bound -- a sample
//  within float rounding of a face, a few per million -- is not deposited here at all: the
//  kernel DEFERS it by negating its mass, and the host deposits it with the CPU's exact test.

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
    // loc = (w - subOrigin + nGrid) % nGrid -- the box may wrap a periodic axis -- and cells
    // outside [0,subDims) are skipped BEFORE sample counting, exactly like the inSub guard in
    // ps_interpolation.cc. Full grid: subOrigin=0, subDims=nGrid.
    int   subOrigin[3];
    int   subDims[3];
    // Field flags: which moment grids this run actually needs. When a flag is 0 the host binds
    // a 4-byte dummy buffer for that grid and the kernel never touches it (mass and streams are
    // always deposited). fVel = velocity or dispersion selected (mom), fDisp = dispersion (m2,
    // 24 B per cell), fGrad = velocity gradient (grad, 36 B per cell).
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

// Product of the row lengths (Hadamard's bound on |det|).
static inline float rowNormProduct3(thread const float A[3][3]) {
    float p = 1.0f;
    for (int i=0; i<3; ++i) p *= sqrt(A[i][0]*A[i][0] + A[i][1]*A[i][1] + A[i][2]*A[i][2]);
    return p;
}

// Inverse of 3x3; false if |det| <= 1e-6 x the product of the row lengths -- the CPU
// matrixInverse() sentinel (math_functions.h, RELATIVE and unit-free), so a collapsed simplex
// is dropped rather than bounding-box filled while a merely SMALL one is kept.
static inline bool inverse3(thread const float A[3][3], thread float inv[3][3]) {
    float d = det3(A);
    if (!(fabs(d) > 1.0e-6f * rowNormProduct3(A))) return false;
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
// (a sample on a face can graze negative in float).
static inline float linearWeight(thread const float rel[3], float d0, thread const float dG[3]) {
    float w = d0 + dG[0]*rel[0] + dG[1]*rel[1] + dG[2]*rel[2];
    return w < 0.0f ? 0.0f : w;
}

// ---- the per-tetrahedron record (host: ps_interpolation.cc, harness: validate_deposit.cpp) ----
// TET_STRIDE floats per tet in the 'verts' buffer, in the CANONICAL frame of the exact inside
// test (ps_exact_inside.h: every periodic image of a particle mapped to one float position):
//   [0..2]  ic0 -- vertex 0's grid-cell index in the canonical index frame (int bits)
//   [3..5]  f0  -- vertex 0's offset inside that cell, in cell units, in [0,1)
//   [6..14] E   -- the edges vertex_k - vertex_0 (k = 1..3, one row each), Mpc; exact
//                  differences of the canonical floats, rounded once to float
//   [15]    the centroid-fallback cell as a flat sub-grid index, computed by the host exactly as
//           the CPU fallback does (uint bits; 0xFFFFFFFF = outside this partition: dropped)
// Positions relative to vertex 0 are then formed from SMALL numbers -- sample offset in cells
// (g - ic0) + frac - f0 -- which is what makes a tight rounding bound possible.
constant int TET_STRIDE = 16;
constant float U_F = 5.9604645e-8f;   // 2^-24, the float unit roundoff

// Band-certified classification of one sample, at offset r (Mpc) from vertex 0. With C the
// cofactor matrix of the edge matrix E (rows = edges), the barycentric numerators are
// n_a = sum_i C[a][i] r_i (vertex a+1, Cramer's rule) and n_z = det - sum_a n_a (vertex 0); the
// sample is inside iff all four have the sign of det. B[] bound |computed - exact| for each
// (see the per-tet setup), so a numerator beyond its bound has its exact sign. Returns +1
// inside, -1 outside (both exact), 0 when some numerator lies within its bound -- then only
// the exact predicate can decide, and the caller defers the tetrahedron to the host.
// Two stages: B[] holds the per-tet bounds, valid for the worst sample of the window; a sample
// they cannot decide gets bounds built from ITS OWN offset t (cells) -- the same analysis, with
// |t_i| in place of the window maximum T_i -- which are several times tighter for the samples
// that matter (those near the tet, well inside a window padded by a cell on each side).
static inline int classifySample(thread const float C[3][3], thread const float M[3][3], float det, float sg,
                                 thread const float B[4], float Bdet, thread const float r[3],
                                 thread const float t[3], constant DepositParams& P)
{
    float n[4];
    n[0] = C[0][0]*r[0] + C[0][1]*r[1] + C[0][2]*r[2];
    n[1] = C[1][0]*r[0] + C[1][1]*r[1] + C[1][2]*r[2];
    n[2] = C[2][0]*r[0] + C[2][1]*r[1] + C[2][2]*r[2];
    n[3] = det - (n[0] + n[1] + n[2]);
    bool sure = true;
    for (int k = 0; k < 4; ++k) {
        float const x = sg * n[k];
        if (x < -B[k]) return -1;      // exactly outside, whatever the other three are
        if (x <= B[k]) sure = false;
    }
    if (sure) return 1;

    // stage 2: this sample's own bounds (7u|r_i| + rho_i = dx_i (u (6 + 12|t_i|) + 1e-9))
    float w[3];
    for (int i = 0; i < 3; ++i) w[i] = P.dx[i] * (U_F * (6.0f + 12.0f*fabs(t[i])) + 1.0e-9f);
    float Bs[4];
    float nMag = 0.0f;
    for (int a = 0; a < 3; ++a) {
        Bs[a] = 2.0f * (M[a][0]*w[0] + M[a][1]*w[1] + M[a][2]*w[2]);
        nMag += M[a][0]*fabs(r[0]) + M[a][1]*fabs(r[1]) + M[a][2]*fabs(r[2]);
    }
    Bs[3] = Bdet + Bs[0] + Bs[1] + Bs[2] + 2.0f * 3.0f * U_F * (fabs(det) + nMag);
    sure = true;
    for (int k = 0; k < 4; ++k) {
        float const x = sg * n[k];
        if (x < -Bs[k]) return -1;
        if (x <= Bs[k]) sure = false;
    }
    return sure ? 1 : 0;
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

// The tet's four face planes, outward (n.x <= off inside), in the vertex-0 frame: a cell lying
// wholly beyond one of them cannot intersect the tet, so it is skipped before any clipping. A
// stretched (folded, sheet-like) tet's axis-aligned window holds mostly such cells -- a diagonal
// 16x16x0.1 Mpc sheet spans ~40^3 cells of 0.4 Mpc and touches ~40^2 -- and clipping every one of
// them twice made the exact deposit crawl when cells are much smaller than tets (0.26M particles
// on 256^3: >17 min). Skipped cells clip to nothing anyway: same output, to float rounding.
struct TetPlanes { float3 n[4]; float off[4]; };

static inline TetPlanes tet_planes(float3 t1, float3 t2, float3 t3) {
    float3 const P[4] = { float3(0.0f), t1, t2, t3 };
    TetPlanes F;
    for (int k=0; k<4; ++k) {
        float3 const A = P[(k+1)&3], B = P[(k+2)&3], C = P[(k+3)&3];
        float3 n = cross(B - A, C - A);
        float off = dot(n, A);
        if (dot(n, P[k]) - off > 0.0f) { n = -n; off = -off; }   // vertex k on the inner side
        F.n[k] = n; F.off[k] = off;
    }
    return F;
}

// true when the box (centre c, half-widths h) lies strictly outside the tet; the margin keeps a
// box that touches or overlaps it within float rounding
static inline bool box_outside_tet(thread const TetPlanes& F, float3 c, float3 h) {
    for (int k=0; k<4; ++k) {
        float3 const n = F.n[k];
        float const r = h.x*fabs(n.x) + h.y*fabs(n.y) + h.z*fabs(n.z);
        float const margin = 1.0e-4f * (fabs(n.x)*(fabs(c.x)+h.x) + fabs(n.y)*(fabs(c.y)+h.y)
                                      + fabs(n.z)*(fabs(c.z)+h.z) + fabs(F.off[k]));
        if (dot(n, c) - F.off[k] > r + margin) return true;
    }
    return false;
}

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
// depositFields: the deposit. Each interior sample receives the mass share and, per the field
// flags, also the mass-weighted velocity moment sum(w v_j), second moment
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
// unresolved: this is a centroid-fallback deposit (sets bit 31, see the end).
// countStream: whether this deposit adds to the sample-count multiplicity '.streams'. True for
// every interior SAMPLE; false for the centroid fallback of a tet that contains no sample, which
// moves mass but is no sample of the stream field (counting it +1 inflated '.streams' by one
// whole sample per sub-resolution tetrahedron -- up to +839 in collapsed cells). Under fExact
// the counter is the raw tet-touch count instead, so the fallback counts there.
static inline void depositSample(thread const FieldGrids& G, uint flat,
                                 thread const float rel[3], float w, float wm, uint obits,
                                 thread const float u0[3], thread const float vG[3][3],
                                 bool fVel, bool fDisp, bool fGrad, bool fVolW, bool fCaustic,
                                 bool fExact, float svShare, bool countStream, bool unresolved)
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
    if (countStream)
        atomic_fetch_add_explicit(&G.streams[flat], 1u, memory_order_relaxed);
    // '.hidden_streams' bit 2: mass from a tet none of whose samples lies in this cell (the centroid
    // fallback). Bit 31 of the same counter -- the sample count never reaches 2^31 -- so no
    // extra buffer is bound; the host splits the two (ps_interpolation.cc).
    if (unresolved)
        atomic_fetch_or_explicit(&G.streams[flat], 0x80000000u, memory_order_relaxed);
}

// ------------------------------------------------------------------------------------------------
// Exact deposit (--ps-exact-deposit), one window cell at a time. Shared by depositExactItems (the
// production path: many threads per large tet, below) and depositFields (one thread per tet: the
// few tets the items pass leaves over).
// ------------------------------------------------------------------------------------------------

// A RAW window index triple -> its flat sub-grid index: the periodic wrap, then the partition
// sub-box, which may itself wrap a periodic axis (see ps_interpolation.cc). False when the cell
// lies outside the grid or the sub-box.
static inline bool window_flat(thread const int raw[3], constant DepositParams& P, thread uint& flat) {
    int l[3];
    for (int dd=0; dd<3; ++dd) {
        int const wg = P.periodic ? wrapIdx(raw[dd], P.nGrid[dd]) : raw[dd];
        if (wg < 0 || wg >= P.nGrid[dd]) return false;
        l[dd] = wg - P.subOrigin[dd];
        if (l[dd] < 0) l[dd] += P.nGrid[dd];
        if (l[dd] < 0 || l[dd] >= P.subDims[dd]) return false;
    }
    flat = (uint(l[0])*uint(P.subDims[1]) + uint(l[1]))*uint(P.subDims[2]) + uint(l[2]);
    return true;
}

struct ExactTet {
    float3 t1, t2, t3;       // edges from vertex 0, positively oriented (folded tet: 1 and 2 swapped)
    TetPlanes faces;
    float3 hcell;            // half a cell
    int    ic0[3];
    float  f0[3];
    float  u0[3], vG[3][3];  // the linear velocity (zero when no velocity grid is requested)
    float  d0, dG[3];        // the linear density (fLinear)
    uint   obits;
    float  invCellVol;
    // a tet that reaches beyond a non-periodic grid is normalized by its WHOLE weight wFull (not
    // the sum over its in-grid pieces), so the grid keeps exactly the share of its mass inside it
    bool   straddle;
    float  wFull;
};

// The tet's whole deposit weight: its volume, or under fLinear the integral of the linear density
// d0 + dG.x over it = V (d0 + dG.centroid), the centroid (t1+t2+t3)/4 in the vertex-0 frame.
static inline float exact_full_weight(float vol, bool fLinear, float d0, thread const float dG[3],
                                      float3 t1, float3 t2, float3 t3)
{
    if (!fLinear) return vol;
    float3 const c = 0.25f * (t1 + t2 + t3);
    return vol * (d0 + dG[0]*c.x + dG[1]*c.y + dG[2]*c.z);
}

// Clips the tet against the window cell 'raw'. False when they do not overlap; otherwise mo holds
// the piece's moments (volume and first moments only unless 'full': pass 0 discards the rest) and
// w its deposit weight -- V_int, or the exact linear-profile integral under fLinear (clamped at 0).
static inline bool exact_cell_clip(thread const ExactTet& E, thread const int raw[3],
                                   constant DepositParams& P, bool full, bool fLinear,
                                   thread float mo[10], thread float& w)
{
    // a cell wholly outside a face plane cannot hold any of the tet
    float3 const ccen = float3((float(raw[0] - E.ic0[0]) - E.f0[0] + 0.5f) * P.dx[0],
                               (float(raw[1] - E.ic0[1]) - E.f0[1] + 0.5f) * P.dx[1],
                               (float(raw[2] - E.ic0[2]) - E.f0[2] + 0.5f) * P.dx[2]);
    if (box_outside_tet(E.faces, ccen, E.hcell)) return false;

    // clip the tet against this cell (bounds in the vertex-0 frame, RAW indices)
    ExPoly piece;
    exact_init_tet(piece, E.t1, E.t2, E.t3);
    for (int dd=0; dd<3; ++dd) {
        float const clo = (float(raw[dd] - E.ic0[dd]) - E.f0[dd]) * P.dx[dd];
        float const chi = clo + P.dx[dd];
        exact_clip_plane(piece, dd,  1.0f, -clo);   //  x_d >= clo
        exact_clip_plane(piece, dd, -1.0f,  chi);   //  x_d <= chi
    }
    if (piece.nv == 0) return false;
    if (full) exact_reduce2(piece, mo);
    else      exact_reduce1(piece, mo);
    if (!(mo[0] > 0.0f)) return false;

    w = mo[0];
    if (fLinear) {
        w = E.d0*mo[0] + E.dG[0]*mo[1] + E.dG[1]*mo[2] + E.dG[2]*mo[3];
        if (w < 0.0f) w = 0.0f;
    }
    return true;
}

// Deposits one clipped piece (moments mo, from exact_cell_clip with full=true): its mass share,
// the exact first and second velocity moments of the linear profile over the piece, the
// gradient, the caustic bits, the exact multiplicity share V_int/V_cell and the tet-touch count.
static inline void exact_cell_deposit(thread const FieldGrids& G, uint flat, thread const float mo[10],
                                      float massShare, thread const ExactTet& E,
                                      bool fVel, bool fDisp, bool fGrad, bool fVolW, bool fCaustic)
{
    float const wmEx = fVolW ? mo[0] : massShare;
    ulong const base = ulong(flat);
    atomic_fetch_add_explicit(&G.mass[flat], massShare, memory_order_relaxed);
    float const invV = 1.0f / mo[0];
    float const cen[3] = { mo[1]*invV, mo[2]*invV, mo[3]*invV };
    float vb[3];
    if (fVel || fDisp) {
        for (int j=0; j<3; ++j) {
            vb[j] = E.u0[j];
            for (int i=0; i<3; ++i) vb[j] += E.vG[i][j]*cen[i];
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
                            gcg += E.vG[i][a]*cov[i][k]*E.vG[k][b];
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
                atomic_fetch_add_explicit(&G.grad[base*9ul + ulong(j*3+i)], E.vG[i][j]*wmEx, memory_order_relaxed);
    if (fVolW)
        atomic_fetch_add_explicit(&G.momw[flat], wmEx, memory_order_relaxed);
    if (fCaustic)
        atomic_fetch_or_explicit(&G.caustic[flat], E.obits, memory_order_relaxed);
    // exact cell-mean multiplicity share V_int/V_cell, and the raw tet-touch count
    atomic_fetch_add_explicit(&G.sv[flat], mo[0]*E.invCellVol, memory_order_relaxed);
    atomic_fetch_add_explicit(&G.streams[flat], 1u, memory_order_relaxed);
}

// ------------------------------------------------------------------------------------------------
// Counting a straddling tet's samples BEYOND a non-periodic grid (pass 0 of the sampled deposit).
// They are only counted (and, under fLinear, their weights summed), so they are taken one sample
// COLUMN (along z) at a time: along a column each numerator of classifySample is affine in the z
// offset, n_a = P_a + Q_a r_z, and re-evaluated in that grouping it differs from classifySample's
// own value by far less than its bound B. So the samples whose numerators all clear 2B are inside
// (classifySample would say +1), those below -2B on some face are outside (it would say -1), and
// each of the two sets is one run along the column, found by binary search. Only the samples in
// neither -- within ~B of a face -- are classified one by one, and an undecidable one defers the
// tet exactly as before, so the count and the deferral decisions are the per-sample ones. Cost: the
// tet's cross-section beyond the grid instead of its volume. Short columns are classified directly.
// ------------------------------------------------------------------------------------------------

// sample index s along one axis -> its offset from vertex 0 in cells (the sampled loop's expression)
static inline float sample_tc(int s, int nSub, float invNSub, int ic0d, float f0d) {
    int const k = (s >= 0) ? s / nSub : -((-s + nSub - 1) / nSub);
    int const sub = s - k * nSub;
    float const fr = (nSub == 1) ? 0.5f : (float(sub) + 0.5f) * invNSub;
    return (float(k - ic0d) - f0d) + fr;
}

struct BeyondColumn {
    float P[4], Q[4];      // sg * numerator_v = sg * (P_v + Q_v r_z), n_z included as ONE affine form
    int   ic0z; float f0z; float dxz; int nSub; float invNSub;
};

static inline float column_value(thread const BeyondColumn& c, int v, int s) {
    float const rz = sample_tc(s, c.nSub, c.invNSub, c.ic0z, c.f0z) * c.dxz;
    return c.P[v] + c.Q[v] * rz;
}

// First index of [lo, hi) where "value(v) > T" becomes true (rise) or false (fall along the column),
// hi if never. Two probes at the analytic crossing 'guess' settle it (it is off by rounding only);
// bisection covers whatever they leave -- plain bisection alone made this the deposit's hot spot.
static inline int column_transition(thread const BeyondColumn& c, int v, float T, bool rise,
                                    int lo, int hi, float guess) {
    int L = lo, H = hi;
    int const g = (guess > float(lo)) ? ((guess < float(hi)) ? int(ceil(guess)) : hi) : lo;
    if (g > L && g - 1 < H) { if ((column_value(c, v, g - 1) > T) == rise) H = g - 1; else L = g; }
    if (g >= L && g < H)    { if ((column_value(c, v, g) > T) == rise) H = g; else L = g + 1; }
    while (L < H) { int const mid = L + (H - L) / 2; if ((column_value(c, v, mid) > T) == rise) H = mid; else L = mid + 1; }
    return L;
}

// [a, b) = the samples of [S0, S1) whose four values all exceed T[v] (one run: each value is monotone)
static inline void column_run(thread const BeyondColumn& c, int S0, int S1, thread const float T[4],
                              thread int& a, thread int& b) {
    a = S0; b = S1;
    for (int v = 0; v < 4 && a < b; ++v) {
        // the sample index where P + Q r_z crosses T: r_z = tc dx, tc = (s + 0.5)/nSub - ic0 - f0
        float const guess = (c.Q[v] != 0.0f)
                          ? ((T[v] - c.P[v]) / c.Q[v] / c.dxz + float(c.ic0z) + c.f0z) * float(c.nSub) - 0.5f : 0.0f;
        if (c.Q[v] > 0.0f)      a = column_transition(c, v, T[v], true,  a, b, guess);   // rises: first above T
        else if (c.Q[v] < 0.0f) b = column_transition(c, v, T[v], false, a, b, guess);   // falls: one past the last
        else if (!(c.P[v] > T[v])) b = a;
    }
}

// Counts the tet's samples beyond the grid into Nout (and their linear weights into sumW).
// Returns false when one of them is undecidable: the caller defers the tet.
static inline bool count_beyond_grid(thread const float C[3][3], thread const float M[3][3],
                                     thread const float Ax[3][3],
                                     float detC, float sg, thread const float B[4], float Bdet,
                                     thread const int iMin[3], thread const int iMax[3],
                                     thread const int ic0[3], thread const float f0[3],
                                     int nSub, float invNSub, bool fLinear, float d0,
                                     thread const float dG[3], constant DepositParams& P,
                                     thread uint& Nout, thread float& sumW)
{
    int const S0z = iMin[2] * nSub, S1z = iMax[2] * nSub, gridZ = P.nGrid[2] * nSub;
    float const Tin[4]  = { 2.0f*B[0], 2.0f*B[1], 2.0f*B[2], 2.0f*B[3] };
    float const Tpos[4] = { -2.0f*B[0], -2.0f*B[1], -2.0f*B[2], -2.0f*B[3] };
    BeyondColumn c;
    c.ic0z = ic0[2]; c.f0z = f0[2]; c.dxz = P.dx[2]; c.nSub = nSub; c.invNSub = invNSub;
    // only the columns that HAVE samples beyond the grid: when the window stays inside the grid along
    // z, a column inside it along x needs just its y-parts beyond the grid
    bool const zStraddle = S0z < 0 || S1z > gridZ;
    int const Sy0 = iMin[1] * nSub, Sy1 = iMax[1] * nSub, gridY = P.nGrid[1] * nSub;
    // ... and only the columns inside the tet's xy projection: along a row (fixed x) one y-interval,
    // the extent of the six projected edges crossing x, padded by two samples on each side
    float const Vx[4] = { 0.0f, Ax[0][0], Ax[1][0], Ax[2][0] };
    float const Vy[4] = { 0.0f, Ax[0][1], Ax[1][1], Ax[2][1] };
    float const xPad = 1.0e-4f * P.dx[0];
    for (int sx = iMin[0] * nSub; sx < iMax[0] * nSub; ++sx) {
    int const kx = (sx >= 0) ? sx / nSub : -((-sx + nSub - 1) / nSub);
    bool const xInCol = kx >= 0 && kx < P.nGrid[0];
    float const relx = sample_tc(sx, nSub, invNSub, ic0[0], f0[0]) * P.dx[0];
    float yMin = INFINITY, yMax = -INFINITY;
    for (int i = 0; i < 3; ++i)
        for (int j = i + 1; j < 4; ++j) {
            float const xi = Vx[i], xj = Vx[j];
            if (relx < min(xi, xj) - xPad || relx > max(xi, xj) + xPad) continue;
            float y0 = Vy[i], y1 = Vy[j];
            if (xj != xi) {
                float const t = clamp((relx - xi) / (xj - xi), 0.0f, 1.0f);
                y0 = y1 = Vy[i] + t * (Vy[j] - Vy[i]);
            }
            yMin = min(yMin, min(y0, y1));
            yMax = max(yMax, max(y0, y1));
        }
    if (!(yMin <= yMax)) continue;       // this row misses the tet
    int const pLo = max(Sy0, int(floor((yMin / P.dx[1] + float(ic0[1]) + f0[1]) * float(nSub) - 0.5f)) - 2);
    int const pHi = min(Sy1, int(ceil((yMax / P.dx[1] + float(ic0[1]) + f0[1]) * float(nSub) - 0.5f)) + 3);
    if (pLo >= pHi) continue;
    int yLo[2] = { pLo, 0 }, yHi[2] = { pHi, 0 }, nY = 1;
    if (xInCol && !zStraddle) {
        nY = 0;
        if (pLo < 0)     { yLo[nY] = pLo; yHi[nY] = min(pHi, 0); ++nY; }
        if (pHi > gridY) { yLo[nY] = max(pLo, gridY); yHi[nY] = pHi; ++nY; }
    }
    for (int iy = 0; iy < nY; ++iy)
    for (int sy = yLo[iy]; sy < yHi[iy]; ++sy) {
        int const ky = (sy >= 0) ? sy / nSub : -((-sy + nSub - 1) / nSub);
        bool const colIn = kx >= 0 && kx < P.nGrid[0] && ky >= 0 && ky < P.nGrid[1];
        int outLo[2], outHi[2], nOut = 0;
        if (!colIn) { outLo[0] = S0z; outHi[0] = S1z; nOut = 1; }
        else {
            if (S0z < 0)     { outLo[nOut] = S0z; outHi[nOut] = min(S1z, 0); ++nOut; }
            if (S1z > gridZ) { outLo[nOut] = max(S0z, gridZ); outHi[nOut] = S1z; ++nOut; }
        }
        if (nOut == 0) continue;
        float const tx = sample_tc(sx, nSub, invNSub, ic0[0], f0[0]);
        float const ty = sample_tc(sy, nSub, invNSub, ic0[1], f0[1]);
        float const rx = tx * P.dx[0], ry = ty * P.dx[1];
        int outLen = 0;
        for (int i = 0; i < nOut; ++i) outLen += outHi[i] - outLo[i];
        // per-sample classification of [u, v): short columns, and the samples near a face
        // (written out twice below rather than as a lambda: Metal has none)
        int aIn = 0, bIn = 0, aPos = 0, bPos = 0;
        bool bulk = outLen > 16;
        if (bulk) {
            for (int a = 0; a < 3; ++a) {
                c.P[a] = sg * (C[a][0]*rx + C[a][1]*ry);
                c.Q[a] = sg * C[a][2];
            }
            c.P[3] = sg * (detC - (C[0][0]*rx + C[0][1]*ry + C[1][0]*rx + C[1][1]*ry + C[2][0]*rx + C[2][1]*ry));
            c.Q[3] = -sg * (C[0][2] + C[1][2] + C[2][2]);
            column_run(c, S0z, S1z, Tin, aIn, bIn);    // classifySample says +1
            column_run(c, S0z, S1z, Tpos, aPos, bPos); // the rest are below -B on some face: -1
            if (aIn >= bIn) { aIn = aPos; bIn = aPos; }
        }
        for (int i = 0; i < nOut; ++i) {
            // bulk: the certain run; the samples of [aPos, bPos) outside it are classified below
            int lo1 = outLo[i], hi1 = outHi[i], lo2 = 0, hi2 = 0;
            if (bulk) {
                int const c0 = max(aIn, outLo[i]), c1 = min(bIn, outHi[i]);
                if (c0 < c1) {
                    Nout += uint(c1 - c0);
                    if (fLinear) {
                        // sum of d0 + dG.r over the run: r_z = tc_z dx, tc_z = (s + 0.5)/nSub - ic0 - f0
                        float const n = float(c1 - c0);
                        float const mz = ((float(c0) + float(c1)) * 0.5f * invNSub - float(ic0[2]) - f0[2]) * P.dx[2];
                        sumW += n * (d0 + dG[0]*rx + dG[1]*ry + dG[2]*mz);
                    }
                }
                lo1 = max(aPos, outLo[i]); hi1 = min(aIn, outHi[i]);
                lo2 = max(bIn, outLo[i]);  hi2 = min(bPos, outHi[i]);
            }
            for (int pass2 = 0; pass2 < 2; ++pass2) {
                int const u = pass2 == 0 ? lo1 : lo2, w = pass2 == 0 ? hi1 : hi2;
                for (int sz = u; sz < w; ++sz) {
                    float const tz = sample_tc(sz, nSub, invNSub, ic0[2], f0[2]);
                    float const t[3] = { tx, ty, tz };
                    float rel[3] = { rx, ry, tz * P.dx[2] };
                    int const cls = classifySample(C, M, detC, sg, B, Bdet, rel, t, P);
                    if (cls == 0) return false;
                    if (cls > 0) { Nout++; if (fLinear) sumW += linearWeight(rel, d0, dG); }
                }
            }
        }
    }
    }
    return true;
}

kernel void depositFields(
    device const float*        verts    [[buffer(0)]],  // nTet * TET_STRIDE (the record above)
    device const float*        vels     [[buffer(1)]],  // nTet * 12 (4 verts * vxyz)
    device float*              masses   [[buffer(2)]],  // nTet; a DEFERRED tet's mass is negated here
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

    float  m  = masses[tid];
    if (m <= 0.0f) return;          // (a tet deferred by an earlier, retried attempt stays deferred)

    device const float* R = verts + ulong(tid) * ulong(TET_STRIDE);
    int   const ic0[3] = { as_type<int>(R[0]), as_type<int>(R[1]), as_type<int>(R[2]) };
    float const f0[3]  = { R[3], R[4], R[5] };
    float Ax[3][3];
    for (int e=0; e<3; ++e)
        for (int i=0; i<3; ++i)
            Ax[e][i] = R[6 + e*3 + i];
    uint  const centroidFlat = as_type<uint>(R[15]);

    // The host kept this tet (psFilterCell, double precision); should the float recheck call it
    // degenerate, defer it rather than drop it, so the host deposits it exactly and no mass is lost.
    float d = det3(Ax);
    float avgEdge2 = 0.0f;
    for (int v=0; v<3; ++v) { float l2=0.0f; for (int i=0;i<3;++i) l2+=Ax[v][i]*Ax[v][i]; avgEdge2+=l2; }
    avgEdge2 /= 3.0f;
    float posInv[3][3];
    if (fabs(d) < 1.0e-6f * avgEdge2 * sqrt(avgEdge2) || !inverse3(Ax, posInv)) { masses[tid] = -m; return; }

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
    /* Per-tet caustic mask. With fCaustic and NOT fLinear the 'dens' buffer is otherwise unused, so
       the host parks the mask (built in DOUBLE precision from the same deformation tensor the CPU
       deposit uses) in slot 0 of each tet's four floats. That carries the whole stratification --
       collapse multiplicity and the umbilic indicator, not just the parity -- through an EXISTING
       buffer index; adding a sixteenth buffer for it measurably corrupted this grid (see the
       tessellation of attempts in the memory note), whereas reusing buffer 9 does not. Under
       fLinear the slot really holds a density, so the GPU falls back to the parity it can derive
       from 'd' and '.causticClass' is left to the CPU deposit. */
    uint  const obits = (fCaustic && P.fLinear == 0) ? uint(dens[tid*4+0]) : (d > 0.0f ? 1u : 2u);
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

    // Grid window of the tet's bounding box, in the canonical index frame (vertex extent relative
    // to vertex 0, in cells; periodic indices are wrapped per cell below). On a non-periodic grid
    // iMin/iMax (the sampled loop) stay UNCLAMPED -- a tet cut by the grid face (a --box region of
    // a larger cloud) must count its samples beyond it too, see the sampled path -- and iMinC/
    // iMaxC are the clamped window the exact path clips. A tet wholly beyond the grid deposits
    // nothing (the CPU loop's outsideGrid).
    int iMin[3], iMax[3], iMinC[3], iMaxC[3];
    bool straddle = false;
    for (int dd=0; dd<3; ++dd) {
        float lo = 0.0f, hi = 0.0f;
        for (int e=0; e<3; ++e) { lo = min(lo, Ax[e][dd]); hi = max(hi, Ax[e][dd]); }
        iMin[dd] = ic0[dd] + int(floor(f0[dd] + lo / P.dx[dd]));
        iMax[dd] = ic0[dd] + int(floor(f0[dd] + hi / P.dx[dd])) + 1;
        // straddle is decided on the tet's own bbox, before the sample margin below (a tet whose
        // bbox stays inside the grid has no sample beyond it)
        if (!P.periodic && (iMin[dd] < 0 || iMax[dd] > P.nGrid[dd])) straddle = true;
        // the exact deposit clips the bbox window directly (no sub-samples), so the +-1
        // sub-sample margin would only add empty clips -- same guard as the CPU path
        if (P.nSub>1 && !fExact) { iMin[dd]-=1; iMax[dd]+=1; }
        iMinC[dd] = iMin[dd]; iMaxC[dd] = iMax[dd];
        if (!P.periodic) {
            if (iMinC[dd] < 0)           iMinC[dd] = 0;
            if (iMaxC[dd] > P.nGrid[dd]) iMaxC[dd] = P.nGrid[dd];
            if (iMinC[dd] >= P.nGrid[dd] || iMaxC[dd] <= 0) return;   // wholly beyond the grid
        }
    }
    int nSub = P.nSub;

    /* Error bounds of classifySample, per tet. Every sample offset in this window is formed as
       r_i = ((g_i - ic0_i) + frac - f0_i) * dx_i from numbers of order T_i (cells), so it is within
       rho_i = dx_i (u (6 + 5 T_i) + 1e-9) of the exact offset the CPU test uses (the 1e-9 dwarfs the
       double roundings that define that offset). A cofactor C = ab - cd, from edges each rounded
       once to float, is within 4u M of exact, M = |ab| + |cd|; a numerator sum_i C_i r_i is then
       within sum_i M_i (7u R_i + rho_i), R_i = T_i dx_i; det = sum_i E_0i C_0i within
       8u sum_i |E_0i| M_0i; n_z = det - sum_a n_a within the sum of those plus its own 3u rounding.
       Valid under any evaluation order or fma contraction (the bounds use sums of magnitudes);
       doubled for safety. u = 2^-24. */
    float C[3][3], M[3][3];
    {
        float const A00=Ax[0][0], A01=Ax[0][1], A02=Ax[0][2];
        float const A10=Ax[1][0], A11=Ax[1][1], A12=Ax[1][2];
        float const A20=Ax[2][0], A21=Ax[2][1], A22=Ax[2][2];
        C[0][0]=A11*A22-A12*A21; M[0][0]=fabs(A11*A22)+fabs(A12*A21);
        C[0][1]=A12*A20-A10*A22; M[0][1]=fabs(A12*A20)+fabs(A10*A22);
        C[0][2]=A10*A21-A11*A20; M[0][2]=fabs(A10*A21)+fabs(A11*A20);
        C[1][0]=A02*A21-A01*A22; M[1][0]=fabs(A02*A21)+fabs(A01*A22);
        C[1][1]=A00*A22-A02*A20; M[1][1]=fabs(A00*A22)+fabs(A02*A20);
        C[1][2]=A01*A20-A00*A21; M[1][2]=fabs(A01*A20)+fabs(A00*A21);
        C[2][0]=A01*A12-A02*A11; M[2][0]=fabs(A01*A12)+fabs(A02*A11);
        C[2][1]=A02*A10-A00*A12; M[2][1]=fabs(A02*A10)+fabs(A00*A12);
        C[2][2]=A00*A11-A01*A10; M[2][2]=fabs(A00*A11)+fabs(A01*A10);
    }
    float const detC = Ax[0][0]*C[0][0] + Ax[0][1]*C[0][1] + Ax[0][2]*C[0][2];
    float Rw[3], rho[3];
    for (int i=0; i<3; ++i) {
        float const T = max(fabs(float(iMin[i] - ic0[i]) - f0[i]), fabs(float(iMax[i] - ic0[i]) - f0[i]));
        Rw[i]  = T * P.dx[i];
        rho[i] = P.dx[i] * (U_F * (6.0f + 5.0f*T) + 1.0e-9f);
    }
    float B[4];
    float nMag = 0.0f;   // bound on sum_a |n_a| over the window
    for (int a=0; a<3; ++a) {
        float b = 0.0f;
        for (int i=0; i<3; ++i) { b += M[a][i] * (7.0f*U_F*Rw[i] + rho[i]); nMag += M[a][i] * Rw[i]; }
        B[a] = 2.0f * b;
    }
    float const Bdet = 2.0f * 8.0f * U_F * (fabs(Ax[0][0])*M[0][0] + fabs(Ax[0][1])*M[0][1] + fabs(Ax[0][2])*M[0][2]);
    B[3] = Bdet + B[0] + B[1] + B[2] + 2.0f * 3.0f * U_F * (fabs(detC) + nMag);
    // the orientation itself must be certain: a det within its bound leaves every sign open
    bool const orientSure = fabs(detC) > Bdet;
    float const sg = detC > 0.0f ? 1.0f : -1.0f;
    float const invNSub = 1.0f / float(nSub);

    // ===================== exact conservative deposit (--ps-exact-deposit) =====================
    // Mirrors the CPU exact path: PASS 0 sums the per-cell weights (V_int, or the exact linear-
    // profile integral under fLinear), PASS 1 renormalizes the shares to the tet mass and
    // deposits the exact moments. A tet whose whole window clips empty (sliver below float
    // resolution) falls through to the sampled two-pass below, whose empty window then drives
    // the centroid fallback -- exactly the CPU fall-through.
    if (fExact) {
        // r3d wants positive orientation: a folded (det<0) tet is handed over with
        // vertices 1,2 swapped; vertex 0 stays the frame origin (same as the CPU path)
        ExactTet E;
        E.t1 = float3(Ax[0][0], Ax[0][1], Ax[0][2]);
        E.t2 = float3(Ax[1][0], Ax[1][1], Ax[1][2]);
        E.t3 = float3(Ax[2][0], Ax[2][1], Ax[2][2]);
        if (d < 0.0f) { float3 tt = E.t1; E.t1 = E.t2; E.t2 = tt; }
        E.faces = tet_planes(E.t1, E.t2, E.t3);
        E.hcell = float3(0.5f*P.dx[0], 0.5f*P.dx[1], 0.5f*P.dx[2]);
        for (int i=0; i<3; ++i) {
            E.ic0[i] = ic0[i];  E.f0[i] = f0[i];  E.u0[i] = u0[i];  E.dG[i] = dG[i];
            for (int j=0; j<3; ++j) E.vG[i][j] = vG[i][j];
        }
        E.d0 = d0;  E.obits = obits;  E.invCellVol = invCellVol;
        E.straddle = straddle;
        E.wFull = exact_full_weight(tetVol, fLinear, d0, dG, E.t1, E.t2, E.t3);

        float sumW = 0.0f;
        float shareFac = 0.0f;
        bool  deposited = false;
        for (int pass=0; pass<2; ++pass) {
            if (pass==1) {
                if (!(sumW > 0.0f)) break;   // empty window: sampled fallback below
                shareFac = m / (E.straddle ? max(sumW, E.wFull) : sumW);
                deposited = true;
            }
            for (int gi=iMinC[0]; gi<iMaxC[0]; ++gi)
            for (int gj=iMinC[1]; gj<iMaxC[1]; ++gj)
            for (int gk=iMinC[2]; gk<iMaxC[2]; ++gk) {
                int const raw[3]={gi,gj,gk};
                uint flat;
                if (!window_flat(raw, P, flat)) continue;
                float mo[10], w;
                if (!exact_cell_clip(E, raw, P, pass==1, fLinear, mo, w)) continue;
                if (pass==0) { sumW += w; continue; }
                exact_cell_deposit(G, flat, mo, w * shareFac, E, fVel, fDisp, fGrad, fVolW, fCaustic);
            }
        }
        if (deposited) return;   // exact deposit complete; skip the sampled path
    }

    // Two identical passes over (cell, sample): pass 0 counts N (and, for the linear deposit,
    // sums the sample weights), pass 1 deposits m/N -- or m*w/sumW when fLinear, which totals
    // exactly m as well. The cell's wrap + sub-grid mapping is resolved BEFORE the sample loop
    // (like the CPU inSub guard), so N counts exactly the samples that pass 1 deposits -- no
    // mass is ever lost to invalid cells. Pass 0 also DEFERS the tet (before anything was
    // deposited) at its first sample the float classification cannot decide; pass 1 runs the
    // same code on the same numbers, so every sample it meets is decided as in pass 0.
    //
    // A tet cut by a non-periodic grid's face (straddle) also counts, in pass 0, its samples BEYOND
    // the grid (Nout, and their linear weights), with the same certified test: every sample then
    // carries m / (N + Nout), so the grid keeps only the share of the mass inside it -- sharing it
    // among the in-grid samples alone piled the outside share onto the faces of a --box region.
    if (!orientSure) { masses[tid] = -m; return; }
    uint N = 0, Nout = 0;
    float sumW = 0.0f;
    float share = 0.0f;
    for (int pass=0; pass<2; ++pass) {
        if (pass==1) {
            // a straddling tet's samples beyond the grid (they share its mass, but are not deposited)
            // (skipped when nothing lies in the grid and the centroid is beyond it: nothing is deposited then)
            if (straddle && (N > 0u || centroidFlat != 0xFFFFFFFFu)
                && !count_beyond_grid(C, M, Ax, detC, sg, B, Bdet, iMin, iMax, ic0, f0, nSub, invNSub,
                                               fLinear, d0, dG, P, Nout, sumW)) { masses[tid] = -m; return; }
            if (N==0u) break;      // nothing in the grid: centroid fallback (or nothing) below
            share = m / float(N + Nout);
        }
        // the in-grid cells only (the clamped window): the samples beyond the grid are counted by
        // count_beyond_grid, and walking a large straddling tet's whole window here cost up to ~10^6
        // empty iterations per thread. The bounds B stay those of the unclamped window.
        for (int gi=iMinC[0]; gi<iMaxC[0]; ++gi)
        for (int gj=iMinC[1]; gj<iMaxC[1]; ++gj)
        for (int gk=iMinC[2]; gk<iMaxC[2]; ++gk) {
            int raw[3]={gi,gj,gk};
            int wg0=P.periodic?wrapIdx(gi,P.nGrid[0]):gi;
            int wg1=P.periodic?wrapIdx(gj,P.nGrid[1]):gj;
            int wg2=P.periodic?wrapIdx(gk,P.nGrid[2]):gk;
            bool const inGrid = !(wg0<0||wg0>=P.nGrid[0]||wg1<0||wg1>=P.nGrid[1]||wg2<0||wg2>=P.nGrid[2]);
            if (!inGrid) continue;   // beyond the grid: counted column by column (count_beyond_grid)
            uint flat = 0u;
            {
                int l0=wg0-P.subOrigin[0], l1=wg1-P.subOrigin[1], l2=wg2-P.subOrigin[2];
                // the sub-box may wrap a periodic axis (ps_interpolation.cc): without this every
                // cell on the far side of the seam was skipped -- in a periodic --partition run
                // that is most of each partition's cells (their mass went to centroid fallbacks)
                if (l0<0) l0+=P.nGrid[0];
                if (l1<0) l1+=P.nGrid[1];
                if (l2<0) l2+=P.nGrid[2];
                if (l0<0||l0>=P.subDims[0]||l1<0||l1>=P.subDims[1]||l2<0||l2>=P.subDims[2]) continue;
                flat=(uint(l0)*uint(P.subDims[1])+uint(l1))*uint(P.subDims[2])+uint(l2);
            }
            // this cell's offset from vertex 0 in cells, before the sub-sample fraction
            float const cb[3] = { float(raw[0] - ic0[0]) - f0[0],
                                  float(raw[1] - ic0[1]) - f0[1],
                                  float(raw[2] - ic0[2]) - f0[2] };
            for (int k2=0; k2<nSub; ++k2)
            for (int k1=0; k1<nSub; ++k1)
            for (int k0=0; k0<nSub; ++k0) {
                float const fr0 = (nSub==1) ? 0.5f : (float(k0) + 0.5f) * invNSub;
                float const fr1 = (nSub==1) ? 0.5f : (float(k1) + 0.5f) * invNSub;
                float const fr2 = (nSub==1) ? 0.5f : (float(k2) + 0.5f) * invNSub;
                float const tc[3] = { cb[0] + fr0, cb[1] + fr1, cb[2] + fr2 };   // offset in cells
                float rel[3] = { tc[0] * P.dx[0], tc[1] * P.dx[1], tc[2] * P.dx[2] };
                int const cls = classifySample(C, M, detC, sg, B, Bdet, rel, tc, P);
                if (cls == 0 && pass == 0) { masses[tid] = -m; return; }   // nothing deposited yet
                if (cls > 0) {
                    if (pass==0) {
                        N++;
                        if (fLinear) sumW += linearWeight(rel, d0, dG);
                    } else {
                        float w = share;
                        if (fLinear && sumW > 0.0f)
                            w = m * linearWeight(rel, d0, dG) / sumW;
                        // moment weight: equal V_eul share under fVolW (fLinear is rejected
                        // with fVolW at option parsing), else the mass share w
                        float wm = fVolW ? mw / float(N + Nout) : w;
                        // --ps-exact-deposit fall-through: share the tet volume over its samples
                        float sv = fExact ? (tetVol*invCellVol)/float(N + Nout) : 0.0f;
                        depositSample(G, flat, rel, w, wm, obits, u0, vG, fVel, fDisp, fGrad, fVolW, fCaustic, fExact, sv, true, false);
                    }
                }
            }
        }
    }

    if (N==0u && Nout > 0u) return;   // every sample lies beyond the grid: not the grid's mass
    if (N==0u) {   // centroid fallback (mass + fields at the centroid position)
        // the target cell comes from the host, computed exactly as the CPU fallback computes it,
        // so both deposits put this mass (and the sub-sample mass flag) in the same cell
        if (centroidFlat == 0xFFFFFFFFu) return;   // centroid outside this partition's box: drop
        float crel[3];
        for (int i=0; i<3; ++i) crel[i] = (Ax[0][i] + Ax[1][i] + Ax[2][i]) * 0.25f;
        depositSample(G, centroidFlat, crel, m, mw, obits, u0, vG, fVel, fDisp, fGrad, fVolW, fCaustic, fExact, tetVol*invCellVol, fExact, true);
    }
}

// ------------------------------------------------------------------------------------------------
// depositExactItems: the exact deposit with the work split BELOW the tetrahedron. One thread per
// ITEM = (tet, block of IP.blockCells consecutive cells of the tet's window, in depositFields'
// gi-gj-gk order). With one thread per tet the cost per thread spans five orders of magnitude when
// cells are much smaller than tets (a stretched void tet clips ~10^4 cells, a halo tet one), so
// each buffer waited on its few giants and the dispatch controller shrank buffers to 50 tets:
// 0.26M particles on 256^3 ran >17 min. Two dispatches over ALL items:
//   phase 0: each item adds its cells' weights into sumW[tet] (atomic);
//   phase 1: each item deposits its cells with the share m / sumW[tet], so the tet still deposits
//            exactly its mass (to float rounding), as in the one-thread path.
// A tet whose sumW stays 0 -- float-degenerate, or a window that clips empty -- deposits nothing
// here; the host runs depositFields on those few, which falls through to the sampled path and the
// centroid fallback, or defers to the CPU, exactly as before. Nothing here writes 'masses'.
// ------------------------------------------------------------------------------------------------
struct ExactItemParams { uint nItems; uint phase; uint blockCells; uint pad; };

// The per-tet numbers depositFields derives in its preamble, for the exact path (keep in sync).
// False when the float recheck calls the tet degenerate.
static inline bool exact_tet_setup(uint tet, device const float* verts, device const float* vels,
                                   device const float* dens, constant DepositParams& P,
                                   thread ExactTet& E, thread int iMin[3], thread int iMax[3])
{
    device const float* R = verts + ulong(tet) * ulong(TET_STRIDE);
    float Ax[3][3];
    for (int e=0; e<3; ++e)
        for (int i=0; i<3; ++i)
            Ax[e][i] = R[6 + e*3 + i];
    for (int i=0; i<3; ++i) { E.ic0[i] = as_type<int>(R[i]); E.f0[i] = R[3+i]; }

    float const d = det3(Ax);
    float avgEdge2 = 0.0f;
    for (int v=0; v<3; ++v) { float l2=0.0f; for (int i=0;i<3;++i) l2+=Ax[v][i]*Ax[v][i]; avgEdge2+=l2; }
    avgEdge2 /= 3.0f;
    float posInv[3][3];
    if (fabs(d) < 1.0e-6f * avgEdge2 * sqrt(avgEdge2) || !inverse3(Ax, posInv)) return false;

    for (int i=0; i<3; ++i) { E.u0[i] = 0.0f; for (int j=0; j<3; ++j) E.vG[i][j] = 0.0f; }
    if (P.fVel != 0 || P.fDisp != 0 || P.fGrad != 0) {
        ulong const v0 = ulong(tet) * 12ul;
        for (int j=0;j<3;++j) E.u0[j] = vels[v0+ulong(j)];
        float dV[3][3];
        for (int e=0;e<3;++e)
            for (int j=0;j<3;++j)
                dV[e][j] = vels[v0+ulong((e+1)*3+j)] - E.u0[j];
        for (int i=0;i<3;++i)
            for (int j=0;j<3;++j) {
                float s=0.0f; for (int k=0;k<3;++k) s+=posInv[i][k]*dV[k][j];
                E.vG[i][j]=s;
            }
    }
    E.obits = (P.fCaustic != 0 && P.fLinear == 0) ? uint(dens[ulong(tet)*4ul]) : (d > 0.0f ? 1u : 2u);
    E.invCellVol = 1.0f / (P.dx[0]*P.dx[1]*P.dx[2]);
    E.d0 = 0.0f;
    for (int i=0; i<3; ++i) E.dG[i] = 0.0f;
    if (P.fLinear != 0) {
        E.d0 = dens[ulong(tet)*4ul];
        float dd0[3];
        for (int e=0;e<3;++e) dd0[e] = dens[ulong(tet)*4ul+ulong(e+1)] - E.d0;
        for (int i=0;i<3;++i) { float s=0.0f; for (int k=0;k<3;++k) s+=posInv[i][k]*dd0[k]; E.dG[i]=s; }
    }

    // the bbox window, exactly as depositFields forms it on its exact path (no sample margin)
    E.straddle = false;
    for (int dd=0; dd<3; ++dd) {
        float lo = 0.0f, hi = 0.0f;
        for (int e=0; e<3; ++e) { lo = min(lo, Ax[e][dd]); hi = max(hi, Ax[e][dd]); }
        iMin[dd] = E.ic0[dd] + int(floor(E.f0[dd] + lo / P.dx[dd]));
        iMax[dd] = E.ic0[dd] + int(floor(E.f0[dd] + hi / P.dx[dd])) + 1;
        if (!P.periodic) {
            if (iMin[dd] < 0)           { iMin[dd] = 0;           E.straddle = true; }
            if (iMax[dd] > P.nGrid[dd]) { iMax[dd] = P.nGrid[dd]; E.straddle = true; }
        }
    }

    E.t1 = float3(Ax[0][0], Ax[0][1], Ax[0][2]);
    E.t2 = float3(Ax[1][0], Ax[1][1], Ax[1][2]);
    E.t3 = float3(Ax[2][0], Ax[2][1], Ax[2][2]);
    if (d < 0.0f) { float3 tt = E.t1; E.t1 = E.t2; E.t2 = tt; }
    E.faces = tet_planes(E.t1, E.t2, E.t3);
    E.hcell = float3(0.5f*P.dx[0], 0.5f*P.dx[1], 0.5f*P.dx[2]);
    E.wFull = exact_full_weight(fabs(d)/6.0f, P.fLinear != 0, E.d0, E.dG, E.t1, E.t2, E.t3);
    return true;
}

kernel void depositExactItems(
    device const float*        verts    [[buffer(0)]],
    device const float*        vels     [[buffer(1)]],
    device const float*        masses   [[buffer(2)]],  // read only: deferral is depositFields' job
    device atomic_float*       massGrid [[buffer(3)]],
    device atomic_float*       momGrid  [[buffer(4)]],
    device atomic_float*       m2Grid   [[buffer(5)]],
    device atomic_float*       gradGrid [[buffer(6)]],
    device atomic_uint*        strGrid  [[buffer(7)]],
    constant DepositParams&    P        [[buffer(8)]],
    device const float*        dens     [[buffer(9)]],
    device atomic_float*       momwGrid [[buffer(10)]],
    device atomic_uint*        caustGrid[[buffer(11)]],
    device atomic_float*       svGrid   [[buffer(12)]],
    device atomic_float*       dvGrid   [[buffer(13)]],
    device atomic_float*       dwGrid   [[buffer(14)]],
    device const uint2*        items    [[buffer(15)]], // (tet, block), tets indexing buffers 0-2 and 9 unoffset
    device atomic_float*       sumW     [[buffer(16)]], // nTet: per-tet weight total (phase 0 writes, 1 reads)
    constant ExactItemParams&  IP       [[buffer(17)]],
    uint iid [[thread_position_in_grid]])
{
    if (iid >= IP.nItems) return;
    uint2 const it = items[iid];
    uint  const tet = it.x;
    float const m = masses[tet];
    if (m <= 0.0f) return;           // deferred by an earlier, retried attempt
    bool const deposit = IP.phase == 1u;
    float s = 0.0f;
    if (deposit) {
        s = atomic_load_explicit(&sumW[tet], memory_order_relaxed);
        if (!(s > 0.0f)) return;     // left to depositFields (host)
    }

    ExactTet E;
    int iMin[3], iMax[3];
    if (!exact_tet_setup(tet, verts, vels, dens, P, E, iMin, iMax)) return;
    // a tet cut by the grid face keeps only its inside share: normalized by its whole weight
    float const shareFac = deposit ? m / (E.straddle ? max(s, E.wFull) : s) : 0.0f;
    ulong const n1 = ulong(max(iMax[1] - iMin[1], 0));
    ulong const n2 = ulong(max(iMax[2] - iMin[2], 0));
    ulong const n12 = n1 * n2;
    ulong const nW = ulong(max(iMax[0] - iMin[0], 0)) * n12;
    ulong w = ulong(it.y) * ulong(IP.blockCells);
    if (w >= nW) return;             // the host's window bound has a margin: surplus blocks idle
    ulong const wEnd = (nW - w > ulong(IP.blockCells)) ? w + ulong(IP.blockCells) : nW;
    int a0 = int(w / n12);
    ulong const rem = w - ulong(a0) * n12;
    int a1 = int(rem / n2);
    int a2 = int(rem - ulong(a1) * n2);

    bool const fVel = P.fVel != 0, fDisp = P.fDisp != 0, fGrad = P.fGrad != 0;
    bool const fVolW = P.fVolW != 0, fCaustic = P.fCaustic != 0, fLinear = P.fLinear != 0;
    FieldGrids G { massGrid, momGrid, m2Grid, gradGrid, strGrid, momwGrid, caustGrid, svGrid, dvGrid, dwGrid };
    float acc = 0.0f;
    for (; w < wEnd; ++w) {
        int const raw[3] = { iMin[0] + a0, iMin[1] + a1, iMin[2] + a2 };
        if (ulong(++a2) == n2) { a2 = 0; if (ulong(++a1) == n1) { a1 = 0; ++a0; } }
        uint flat;
        if (!window_flat(raw, P, flat)) continue;
        float mo[10], wt;
        if (!exact_cell_clip(E, raw, P, deposit, fLinear, mo, wt)) continue;
        if (!deposit) acc += wt;
        else exact_cell_deposit(G, flat, mo, wt * shareFac, E, fVel, fDisp, fGrad, fVolW, fCaustic);
    }
    if (!deposit && acc > 0.0f)
        atomic_fetch_add_explicit(&sumW[tet], acc, memory_order_relaxed);
}
