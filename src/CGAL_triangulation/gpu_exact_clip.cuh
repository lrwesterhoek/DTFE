/* The float32 port of the vendored r3d (Powell & Abel 2015) for the CUDA/HIP backend: one
   tetrahedron clipped by the six axis-aligned planes of a grid cell, with order-1 and order-2
   moments, plus the small V3 vector type and the 3x3 helpers both deposits use. Shared by
   ps_gpu_cuda.cu (--ps-exact-deposit) and dtfe_gpu_cuda.cu (--exact-average); the Metal twin is
   metal/exact_clip.metal.inc. Device code only; included inside the backends' anonymous namespace. */

#ifndef GPU_EXACT_CLIP_CUH
#define GPU_EXACT_CLIP_CUH

typedef unsigned long long ull;

// a 12-byte vector with explicit arithmetic (no operators on the runtime's float3: HIP
// defines its own and they would clash)
struct V3 { float x, y, z; };
__device__ __forceinline__ V3 v3(float x, float y, float z) { V3 r; r.x = x; r.y = y; r.z = z; return r; }
__device__ __forceinline__ V3 add3(V3 a, V3 b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ __forceinline__ V3 sub3(V3 a, V3 b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ __forceinline__ V3 neg3(V3 a) { return v3(-a.x, -a.y, -a.z); }
__device__ __forceinline__ V3 scale3(V3 a, float s) { return v3(a.x * s, a.y * s, a.z * s); }
__device__ __forceinline__ V3 div3(V3 a, float s) { return v3(a.x / s, a.y / s, a.z / s); }
__device__ __forceinline__ float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __forceinline__ V3 cross3(V3 a, V3 b)
{ return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
__device__ __forceinline__ float comp3(V3 a, int axis) { return axis == 0 ? a.x : (axis == 1 ? a.y : a.z); }

#define GPU_INF __int_as_float(0x7f800000)

__device__ __forceinline__ float det3(const float A[3][3]) {
    return A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])
         - A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])
         + A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
}

// Product of the row lengths (Hadamard's bound on |det|).
__device__ __forceinline__ float rowNormProduct3(const float A[3][3]) {
    float p = 1.0f;
    for (int i=0; i<3; ++i) p *= sqrtf(A[i][0]*A[i][0] + A[i][1]*A[i][1] + A[i][2]*A[i][2]);
    return p;
}

// Inverse of 3x3; false if |det| <= 1e-6 x the product of the row lengths -- the CPU
// matrixInverse() sentinel (math_functions.h, RELATIVE and unit-free).
__device__ __forceinline__ bool inverse3(const float A[3][3], float inv[3][3]) {
    float d = det3(A);
    if (!(fabsf(d) > 1.0e-6f * rowNormProduct3(A))) return false;
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

__device__ __forceinline__ int wrapIdx(int g, int n) { return ((g % n) + n) % n; }

// --ps-linear-deposit sample weight: linear density at offset rel from vertex 0, clamped >= 0
__device__ __forceinline__ float linearWeight(const float rel[3], float d0, const float dG[3]) {
    float w = d0 + dG[0]*rel[0] + dG[1]*rel[1] + dG[2]*rel[2];
    return w < 0.0f ? 0.0f : w;
}

// ================================================================================================
// --ps-exact-deposit: float32 port of the vendored r3d (Powell & Abel 2015) specialized to ONE
// tetrahedron clipped by the 6 axis-aligned planes of a grid cell, with order-2 moments. Faithful
// to third_party/r3d/r3d.c and identical to the Metal port (metal/ps_deposit.metal).
// ================================================================================================
#define EXV 48

struct ExPoly {
    V3  pos[EXV];
    int nbr[EXV][3];
    int nv;
};

// The tet's four face planes, outward (n.x <= off inside), in the vertex-0 frame.
struct TetPlanes { V3 n[4]; float off[4]; };

__device__ __forceinline__ TetPlanes tet_planes(V3 t1, V3 t2, V3 t3) {
    V3 const P[4] = { v3(0.0f, 0.0f, 0.0f), t1, t2, t3 };
    TetPlanes F;
    for (int k=0; k<4; ++k) {
        V3 const A = P[(k+1)&3], B = P[(k+2)&3], C = P[(k+3)&3];
        V3 n = cross3(sub3(B, A), sub3(C, A));
        float off = dot3(n, A);
        if (dot3(n, P[k]) - off > 0.0f) { n = neg3(n); off = -off; }   // vertex k on the inner side
        F.n[k] = n; F.off[k] = off;
    }
    return F;
}

// true when the box (centre c, half-widths h) lies strictly outside the tet
__device__ __forceinline__ bool box_outside_tet(const TetPlanes& F, V3 c, V3 h) {
    for (int k=0; k<4; ++k) {
        V3 const n = F.n[k];
        float const r = h.x*fabsf(n.x) + h.y*fabsf(n.y) + h.z*fabsf(n.z);
        float const margin = 1.0e-4f * (fabsf(n.x)*(fabsf(c.x)+h.x) + fabsf(n.y)*(fabsf(c.y)+h.y)
                                      + fabsf(n.z)*(fabsf(c.z)+h.z) + fabsf(F.off[k]));
        if (dot3(n, c) - F.off[k] > r + margin) return true;
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
__device__ __forceinline__ void exact_init_tet(ExPoly& T, V3 t1, V3 t2, V3 t3);
__device__ __forceinline__ void exact_init_tet4(ExPoly& T, V3 p0, V3 p1, V3 p2, V3 p3);

__device__ __forceinline__ void exact_init_tet(ExPoly& T, V3 t1, V3 t2, V3 t3) {
    T.nv = 4;
    T.pos[0] = v3(0.0f, 0.0f, 0.0f);
    T.pos[1] = t1; T.pos[2] = t2; T.pos[3] = t3;
    T.nbr[0][0]=1; T.nbr[0][1]=3; T.nbr[0][2]=2;
    T.nbr[1][0]=2; T.nbr[1][1]=3; T.nbr[1][2]=0;
    T.nbr[2][0]=0; T.nbr[2][1]=3; T.nbr[2][2]=1;
    T.nbr[3][0]=1; T.nbr[3][1]=2; T.nbr[3][2]=0;
}

// r3d_clip for one axis-aligned plane sgn*pos[axis] + dd >= 0 (keeps the non-negative side).
// the overlap frame's origin (vertex-0 frame): the centre of the tet bbox's overlap with the cell
// (cell centre c, half-widths h; tet vertices 0, t1, t2, t3)
__device__ __forceinline__ V3 overlap_origin(V3 c, V3 h, V3 t1, V3 t2, V3 t3) {
    float const cs[3] = { c.x, c.y, c.z }, hs[3] = { h.x, h.y, h.z };
    float const a[3] = { t1.x, t1.y, t1.z }, b[3] = { t2.x, t2.y, t2.z }, e[3] = { t3.x, t3.y, t3.z };
    float o[3];
    for (int d = 0; d < 3; ++d) {
        float const tlo = fminf(fminf(0.0f, a[d]), fminf(b[d], e[d]));
        float const thi = fmaxf(fmaxf(0.0f, a[d]), fmaxf(b[d], e[d]));
        o[d] = 0.5f * (fmaxf(cs[d] - hs[d], tlo) + fminf(cs[d] + hs[d], thi));
    }
    return v3(o[0], o[1], o[2]);
}

// the tet with arbitrary (positively oriented) vertices: the overlap frame above
__device__ __forceinline__ void exact_init_tet4(ExPoly& T, V3 p0, V3 p1, V3 p2, V3 p3) {
    exact_init_tet(T, p1, p2, p3);
    T.pos[0] = p0;
}

__device__ void exact_clip_plane(ExPoly& T, int axis, float sgn, float dd) {
    if (T.nv <= 0) return;
    float sdists[EXV];
    int   clipped[EXV];
    int const onv = T.nv;
    float smin = 1.0e30f, smax = -1.0e30f;
    for (int v=0; v<onv; ++v) {
        float const coord = comp3(T.pos[v], axis);
        float const s = dd + sgn*coord;
        sdists[v] = s;
        clipped[v] = (s < 0.0f) ? 1 : 0;
        if (s < smin) smin = s;
        if (s > smax) smax = s;
    }
    if (smin >= 0.0f) return;
    if (smax <= 0.0f) { T.nv = 0; return; }

    for (int vcur=0; vcur<onv; ++vcur) {
        if (clipped[vcur]) continue;
        for (int np=0; np<3; ++np) {
            int const vnext = T.nbr[vcur][np];
            if (!clipped[vnext]) continue;
            if (T.nv == EXV) { T.nv = 0; return; }   // overflow guard
            float const wa = -sdists[vnext], wb = sdists[vcur];
            T.pos[T.nv] = div3(add3(scale3(T.pos[vcur], wa), scale3(T.pos[vnext], wb)), wa + wb);
            T.nbr[T.nv][0] = vcur;
            T.nbr[T.nv][1] = -1;
            T.nbr[T.nv][2] = -1;
            T.nbr[vcur][np] = T.nv;
            T.nv++;
        }
    }

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
__device__ void exact_reduce2(const ExPoly& T, float* mom) {
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
        V3 const v0 = T.pos[vstart];
        int np;
        for (np=0; np<3; ++np)
            if (T.nbr[vnext][np] == vcur) break;
        vcur = vnext;
        pnext = (np+1)%3;
        emk[vcur][pnext] = true;
        vnext = T.nbr[vcur][pnext];
        while (vnext != vstart) {
            V3 const v2 = T.pos[vcur];
            V3 const v1 = T.pos[vnext];
            float const sixv = (-v2.x*v1.y*v0.z + v1.x*v2.y*v0.z + v2.x*v0.y*v1.z
                                -v0.x*v2.y*v1.z - v1.x*v0.y*v2.z + v0.x*v1.y*v2.z);
            mom[0] += sixv;
            V3 const s1 = add3(add3(v0, v1), v2);
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

// Order-1 variant for PASS 0 (identical fan walk and accumulation order for mom[0..3]).
__device__ void exact_reduce1(const ExPoly& T, float* mom) {
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
        V3 const v0 = T.pos[vstart];
        int np;
        for (np=0; np<3; ++np)
            if (T.nbr[vnext][np] == vcur) break;
        vcur = vnext;
        pnext = (np+1)%3;
        emk[vcur][pnext] = true;
        vnext = T.nbr[vcur][pnext];
        while (vnext != vstart) {
            V3 const v2 = T.pos[vcur];
            V3 const v1 = T.pos[vnext];
            float const sixv = (-v2.x*v1.y*v0.z + v1.x*v2.y*v0.z + v2.x*v0.y*v1.z
                                -v0.x*v2.y*v1.z - v1.x*v0.y*v2.z + v0.x*v1.y*v2.z);
            mom[0] += sixv;
            V3 const s1 = add3(add3(v0, v1), v2);
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


#endif
