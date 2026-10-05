//  The float32 port of the vendored r3d (Powell & Abel 2015): one tetrahedron clipped by the six
//  axis-aligned planes of a grid cell, with order-1 and order-2 moments. Shared by the phase-space
//  deposit (ps_deposit.metal: --ps-exact-deposit) and the standard-DTFE exact volume average
//  (dtfe_deposit.metal: --exact-average); the Makefile PREPENDS this file to both kernel sources
//  when it embeds them (and metal/validate_deposit.cpp does the same when it loads the source).
//  Keep it self-contained: it is compiled before everything else in each kernel source.
#include <metal_stdlib>
using namespace metal;

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

// The exact deposits clip in a frame whose origin is the centre of the OVERLAP of the tet's bounding
// box with the cell, so the clipped piece -- which lies in that overlap -- has coordinates of its own
// size and its volume and moments do not cancel. The piece's volume comes from origin-apex simplices,
// so a piece far from the origin cancels: in the vertex-0 frame a piece a few Mpc from vertex 0 lost
// its volume once cells were small (a zoom's virtual grid: 1.6% per cell at 12 kpc, 20% at 3 kpc),
// and in the cell-centre frame a small tet in a large cell did the same. The overlap is near the
// piece in both regimes.
static inline void exact_init_tet4(thread ExPoly& T, float3 p0, float3 p1, float3 p2, float3 p3);

static inline void exact_init_tet(thread ExPoly& T, float3 t1, float3 t2, float3 t3) {
    T.nv = 4;
    T.pos[0] = float3(0.0f);
    T.pos[1] = t1; T.pos[2] = t2; T.pos[3] = t3;
    T.nbr[0][0]=1; T.nbr[0][1]=3; T.nbr[0][2]=2;
    T.nbr[1][0]=2; T.nbr[1][1]=3; T.nbr[1][2]=0;
    T.nbr[2][0]=0; T.nbr[2][1]=3; T.nbr[2][2]=1;
    T.nbr[3][0]=1; T.nbr[3][1]=2; T.nbr[3][2]=0;
}

// the tet with arbitrary (positively oriented) vertices: the overlap frame above
static inline void exact_init_tet4(thread ExPoly& T, float3 p0, float3 p1, float3 p2, float3 p3) {
    exact_init_tet(T, p1, p2, p3);
    T.pos[0] = p0;
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

