//  Standard-DTFE volume-averaged (method 1) grid interpolation -- Metal compute kernel.
//
//  Mirrors the CPU loop in averaged_interpolation_1.cc: fields are linear inside each
//  tetrahedron (constant gradient from the inverse vertex-difference matrix). A tetrahedron
//  that fits inside one grid cell deposits its centroid value times its volume into that
//  cell; larger ones scatter value * (V_tet / nSamples) at shared quasi-random (Sobol)
//  sample points. The caller divides the accumulated grids by the grid-cell volume
//  afterwards, exactly like the CPU path.
//
//  Two kernels share that arithmetic:
//    depositAveraged1       one thread per tetrahedron, one atomic per field per SAMPLE (the
//                           original; kept for A/B runs, DTFE_GPU_ITEMS=0).
//    depositAveraged1Items  one thread per WORK ITEM = (tetrahedron, run of at most
//                           itemSamples samples; implicit, see itemOf), with a small per-thread cell cache that
//                           sums the samples landing in the same grid cell and issues one
//                           atomic per field per distinct CELL. On fine grids (512^3 and up)
//                           most tetrahedra span several cells with hundreds of samples
//                           each, so the per-sample kernel issued hundreds of atomics per
//                           tetrahedron -- 54 G atomics for 7.1M particles on 512^3, 10x
//                           slower than the CPU's plain '+=' -- and its unbounded per-thread
//                           loops (up to maxNN = 10000 samples) tripped the GPU watchdog.
//                           Items bound the work per thread, so the chunked dispatch stays
//                           uniform, and the cache cuts the atomics to about one set per
//                           cell a tetrahedron covers. The samples, their positions and
//                           weights are identical; only the summation order changes.
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

// Which fields the item kernels deposit, fixed when the pipeline is built (the host makes one
// pipeline per field set): the compiler drops the velocity and gradient code and their registers
// from a density-only pipeline, and the item kernels are register-bound (K=4: 640 threads per
// threadgroup, K=2: 832, against the 1024 of a lean kernel), so this buys occupancy.
constant bool FC_DEN  [[function_constant(0)]];
constant bool FC_VEL  [[function_constant(1)]];
constant bool FC_GRAD [[function_constant(2)]];
// the scalar field (ONE component, NO_SCALARS=1 builds; the caller declines the GPU otherwise) and
// its gradient: deposited exactly like the density and the velocity gradient -- the scalar is linear
// over the tetrahedron (buffer 13 holds the four vertex values), its gradient a per-tetrahedron
// constant. Grids: buffer 14 (nCell) and 15 (nCell*3, [flat*3+i] = ds/dx_i).
constant bool FC_SCAL  [[function_constant(3)]];
constant bool FC_SGRAD [[function_constant(4)]];

// A batch = one sub-domain's tetrahedra. Positions are relative to the SUB-DOMAIN's lower corner
// and nGrid is its sub-grid (so the cell classification is the CPU loop's, bit for bit); a sample
// inside the sub-grid is owned by this batch and lands at full-grid cell pos + subOff. With
// subOff = 0 and fullGrid = nGrid this is the original single-grid deposit.
struct DepositParams {
    float dx[3];
    int   nGrid[3];      // the batch's sub-grid: samples outside [0, nGrid) are not this batch's
    uint  nTet;          // threads in this dispatch: tetrahedra (per-tet kernel) or items (item kernel)
    int   fDen;          // deposit density
    int   fVel;          // deposit velocity
    int   fGrad;         // deposit velocity gradient
    uint  itemSamples;   // item kernel: samples per work item (every tetrahedron is split so)
    uint  nTetTotal;     // item kernel: tetrahedra in the batch (bounds the block search, see itemOf)
    int   subOff[3];     // the sub-grid's origin in the accumulated grid
    uint  itemBase;      // item kernel: the first item of this dispatch (items are numbered per batch)
    int   fullGrid[3];   // the accumulated grid's dimensions (flat index = full-grid index)
    uint  debugNoAtomic; // timing experiment: compute everything, issue no atomics (grids stay zero)
    float invDx[3];      // 1/dx: the cell of a sample is floor((base + p) * invDx) in both kernels
    uint  tableEntries;  // item kernel: entries of the threadgroup cell table (a power of two; 0 = none)
    int   fScal;         // deposit the scalar (per-tet kernel; the item kernels use FC_SCAL)
    int   fSGrad;        // deposit the scalar gradient (per-tet kernel; the item kernels use FC_SGRAD)
};

// full-grid flat index of a sub-grid cell (the host guarantees fullGrid cells < 2^32)
static inline uint fullFlat(constant DepositParams& P, thread const int pos[3]) {
    return (uint(pos[0] + P.subOff[0]) * uint(P.fullGrid[1]) + uint(pos[1] + P.subOff[1])) * uint(P.fullGrid[2])
         + uint(pos[2] + P.subOff[2]);
}

static inline float det3(thread const float A[3][3]) {
    return A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])
         - A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])
         + A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
}

// Inverse of 3x3, mirroring the CPU matrixInverse(): |det| <= 1e-6 x the product of the row
// lengths (RELATIVE, unit-free -- see math_functions.h) -> ZERO matrix. The CPU then
// interpolates a constant field from the base vertex; it does NOT drop the tetrahedron, so
// neither do we (unlike the PS deposit, which conserves mass and drops).
static inline void inverse3zero(thread const float A[3][3], thread float inv[3][3]) {
    float d = det3(A);
    float rn = 1.0f;
    for (int i=0; i<3; ++i) rn *= sqrt(A[i][0]*A[i][0] + A[i][1]*A[i][1] + A[i][2]*A[i][2]);
    if (!(fabs(d) > 1.0e-6f * rn)) {
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
                               device atomic_float* scalGrid,
                               device atomic_float* sgGrid,
                               constant DepositParams& P,
                               ulong flat, float den,
                               thread const float vel[3],
                               thread const float gradFlat[9],
                               float sc, thread const float sG[3], float w)
{
    if (P.fDen)
        atomic_fetch_add_explicit(&denGrid[flat], den*w, memory_order_relaxed);
    if (P.fVel)
        for (int j=0;j<3;++j)
            atomic_fetch_add_explicit(&velGrid[flat*3ul+ulong(j)], vel[j]*w, memory_order_relaxed);
    if (P.fGrad)
        for (int q=0;q<9;++q)
            atomic_fetch_add_explicit(&gradGrid[flat*9ul+ulong(q)], gradFlat[q]*w, memory_order_relaxed);
    if (P.fScal)
        atomic_fetch_add_explicit(&scalGrid[flat], sc*w, memory_order_relaxed);
    if (P.fSGrad)
        for (int i=0;i<3;++i)
            atomic_fetch_add_explicit(&sgGrid[flat*3ul+ulong(i)], sG[i]*w, memory_order_relaxed);
}

kernel void depositAveraged1(
    device const float*     verts    [[buffer(0)]],   // nTet*12: 4 vertices, relative to region lower corner
    device const float*     dens     [[buffer(1)]],   // nTet*4:  vertex densities
    device const float*     vels     [[buffer(2)]],   // nTet*12: vertex velocities (dummy when fVel==fGrad==0)
    device const float*     vols     [[buffer(3)]],   // nTet:    tetrahedron volume (CPU value)
    device const uint*      cnts     [[buffer(4)]],   // nTet:    sample count; 0 = single-grid-cell fast path
    device const uint*      flats    [[buffer(5)]],   // nTet:    FULL-grid flat index for the fast path (the host adds subOff)
    device const float4*    sobol    [[buffer(6)]],   // maxNN: shared Sobol barycentric table (xyz, pad)
    device atomic_float*    denGrid  [[buffer(7)]],
    device atomic_float*    velGrid  [[buffer(8)]],
    device atomic_float*    gradGrid [[buffer(9)]],
    constant DepositParams& P        [[buffer(10)]],
    device const float*     scal     [[buffer(13)]],   // nTet*4: vertex scalars (dummy when fScal==fSGrad==0)
    device atomic_float*    scalGrid [[buffer(14)]],
    device atomic_float*    sgGrid   [[buffer(15)]],
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
    // the scalar: linear over the tetrahedron, its gradient constant (the CPU scalarGrad())
    bool const needScal = (P.fScal != 0) || (P.fSGrad != 0);
    float s0 = 0.0f, sG[3] = {0.0f,0.0f,0.0f};
    if (needScal) {
        s0 = scal[tid*4+0];
        for (int i=0;i<3;++i) {
            float s=0.0f;
            for (int j=0;j<3;++j) s += posInv[i][j]*(scal[tid*4+j+1]-s0);
            sG[i]=s;
        }
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
        float scC = 0.0f;
        if (P.fScal) { for (int v=0;v<4;++v) scC += scal[tid*4+v]; scC *= 0.25f; }
        depositCell(denGrid, velGrid, gradGrid, scalGrid, sgGrid, P, ulong(flats[tid]), denC, velC, gradFlat, scC, sG, vol);
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
        float4 const q = sobol[s];
        float const q0 = q.x, q1 = q.y, q2 = q.z;
        float p[3];   // sample point relative to the base vertex (CPU quasiRandomPointsInCell)
        for (int d=0;d<3;++d) p[d] = q0*Ax[0][d] + q1*Ax[1][d] + q2*Ax[2][d];
        int pos[3]; bool in=true;
        for (int d=0;d<3;++d) {
            pos[d] = int(floor((base[d]+p[d])*P.invDx[d]));
            if (pos[d]<0 || pos[d]>=P.nGrid[d]) { in=false; break; }
        }
        if (!in) continue;   // sample outside this batch's sub-grid
        ulong flat = ulong(fullFlat(P, pos));
        float denS = den0;
        for (int i=0;i<3;++i) denS += p[i]*densGrad[i];
        float velS[3] = {0.0f,0.0f,0.0f};
        if (P.fVel)
            for (int j=0;j<3;++j) {
                velS[j]=u0[j];
                for (int i=0;i<3;++i) velS[j] += vG[i][j]*p[i];
            }
        float scS = s0;
        if (P.fScal) for (int i=0;i<3;++i) scS += p[i]*sG[i];
        depositCell(denGrid, velGrid, gradGrid, scalGrid, sgGrid, P, flat, denS, velS, gradFlat, scS, sG, factor);
    }
}


// ---------------------------------------------------------------------------------------
// Work-item kernel: thread = (tetrahedron, samples [s0, min(n, s0+itemSamples))).
//
// Per-thread cell cache: K slots of (flat cell, sum den, sum vel, sample count). The slot
// arrays are only ever indexed inside fully unrolled loops over K with a predicate
// (k == hit), so they stay in registers (a dynamically indexed thread array would spill to
// memory). Replacement is round robin; a victim is flushed with one atomic per field. Every
// sum is multiplied by the per-tet weight factor = V_tet/n once, at the flush, and the
// velocity gradient -- constant over the tetrahedron, the matrix vG kept in registers --
// gets vG * (count * factor), the same total the per-sample kernel adds in 'count' pieces.
// The flush at the end of an item is the SIMD-group reduction described there.
// ---------------------------------------------------------------------------------------
template <int K>
struct CellCache {
    uint  flat[K];
    float den[K];
    float vx[K], vy[K], vz[K];
    float sc[K];        // the scalar's sum (FC_SCAL)
    float cnt[K];
};

// vG[i][j] = d(v_j)/d(x_i); the CPU velocityGradient() layout is out[j*3+i] = vG[i][j]
template <int K>
static inline void flushSlot(device atomic_float* denGrid, device atomic_float* velGrid,
                             device atomic_float* gradGrid, device atomic_float* scalGrid,
                             device atomic_float* sgGrid, constant DepositParams& P,
                             thread const CellCache<K>& C, int k, float factor,
                             thread const float vG[3][3], thread const float sG[3])
{
    // gather slot k with predicated selects (keeps the cache in registers)
    uint  flat = 0u; float den = 0.0f, vx = 0.0f, vy = 0.0f, vz = 0.0f, sc = 0.0f, cnt = 0.0f;
    for (int j = 0; j < K; ++j) {
        bool const m = (j == k);
        flat = m ? C.flat[j] : flat;
        den  = m ? C.den[j]  : den;
        vx   = m ? C.vx[j]   : vx;
        vy   = m ? C.vy[j]   : vy;
        vz   = m ? C.vz[j]   : vz;
        sc   = m ? C.sc[j]   : sc;
        cnt  = m ? C.cnt[j]  : cnt;
    }
    ulong const f = ulong(flat);
    if (P.fDen)
        atomic_fetch_add_explicit(&denGrid[f], den * factor, memory_order_relaxed);
    if (P.fVel) {
        atomic_fetch_add_explicit(&velGrid[f*3ul+0ul], vx * factor, memory_order_relaxed);
        atomic_fetch_add_explicit(&velGrid[f*3ul+1ul], vy * factor, memory_order_relaxed);
        atomic_fetch_add_explicit(&velGrid[f*3ul+2ul], vz * factor, memory_order_relaxed);
    }
    if (FC_SCAL)
        atomic_fetch_add_explicit(&scalGrid[f], sc * factor, memory_order_relaxed);
    if (P.fGrad && P.debugNoAtomic < 3u) {
        float const w = cnt * factor;
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i)
                atomic_fetch_add_explicit(&gradGrid[f*9ul+ulong(j*3+i)], vG[i][j] * w, memory_order_relaxed);
    }
    if (FC_SGRAD && P.debugNoAtomic < 3u) {
        float const w = cnt * factor;
        for (int i = 0; i < 3; ++i)
            atomic_fetch_add_explicit(&sgGrid[f*3ul+ulong(i)], sG[i] * w, memory_order_relaxed);
    }
}

// Work items are implicit (no per-item buffer: 180 M items of a 512^3 sub-domain would be
// 1.4 GB, built on the host under the dispatch mutex); see itemOf below.
// itemStart[t] is the first item of tetrahedron t (a prefix sum of max(1, ceil(cnt/itemSamples)),
// nTet+1 entries); blockTet[iid >> ITEM_BLOCK_SHIFT] the tetrahedron holding that block's first
// item. An item's tetrahedron is the last t in [blockTet[blk], blockTet[blk+1]] with
// itemStart[t] <= iid: a binary search over at most one block's worth of tetrahedra (<= 2^shift
// + 1 entries, a handful of dependent loads from a hot table). Any tetrahedron order works --
// the host orders them along a Morton curve so neighbouring threads deposit into neighbouring
// cells (a binary search over the whole prefix array cost a third of the kernel).
#define ITEM_BLOCK_SHIFT 5
// -> (tetrahedron, index of this item among the tetrahedron's items, items of the tetrahedron)
static inline uint3 itemOf(device const uint* itemStart, device const uint* blockTet, uint nTet, uint iid) {
    uint const blk = iid >> ITEM_BLOCK_SHIFT;
    uint lo = blockTet[blk];
    uint hi = min(blockTet[blk + 1u], nTet - 1u);
    while (hi > lo) {
        uint const mid = (lo + hi + 1u) >> 1;
        if (itemStart[mid] <= iid) lo = mid; else hi = mid - 1u;
    }
    uint const first = itemStart[lo];
    return uint3(lo, iid - first, itemStart[lo + 1u] - first);
}

// ---------------------------------------------------------------------------------------
// Threadgroup cell table: where the per-thread cache's EVICTIONS go. In the void regime a
// tetrahedron spans hundreds of cells and its ~10 000 samples never repeat a cell within one
// item, so every sample evicts; the same cell is hit ~20 times, but across the tetrahedron's
// ~200 items -- which sit in the same threadgroup (consecutive items). The table (open
// addressing, linear probing, 32-bit cell keys claimed by compare-exchange, threadgroup float
// atomics for the sums) collects those, and the threadgroup flushes it once at its end with one
// device atomic per field per entry, in place of 13 device atomics per evicted sample. The
// gradient sums vG * w directly (nine floats per entry), since entries mix tetrahedra. A full
// probe sequence falls back to the direct device atomics of flushSlot. Layout per entry:
// key, den, vx, vy, vz, [g0..g8]; the host sizes the memory per pipeline (grad or not).
// ---------------------------------------------------------------------------------------
#define TABLE_EMPTY 0xffffffffu
#define TABLE_PROBES 8u

// floats per entry: den, vx, vy, vz, [scalar], [9 gradient], [3 scalar gradient]
static inline uint tableOffScal()  { return 4u; }
static inline uint tableOffGrad()  { return 4u + (FC_SCAL ? 1u : 0u); }
static inline uint tableOffSGrad() { return tableOffGrad() + (FC_GRAD ? 9u : 0u); }
static inline uint tableStride()   { return tableOffSGrad() + (FC_SGRAD ? 3u : 0u); }

static inline bool tableAdd(threadgroup atomic_uint* tKeys, threadgroup atomic_float* tVals, uint E,
                            uint key, float den, float vx, float vy, float vz, float sc, float w,
                            thread const float vG[3][3], thread const float sG[3])
{
    uint h = (key * 2654435761u) >> 7;
    h &= (E - 1u);
    for (uint probe = 0u; probe < TABLE_PROBES; ++probe) {
        uint k = atomic_load_explicit(&tKeys[h], memory_order_relaxed);
        if (k == TABLE_EMPTY) {
            uint expected = TABLE_EMPTY;
            if (!atomic_compare_exchange_weak_explicit(&tKeys[h], &expected, key, memory_order_relaxed, memory_order_relaxed))
                k = expected;         // somebody else claimed it meanwhile: 'expected' holds their key
            else
                k = key;
        }
        if (k == key) {
            uint const base = h * tableStride();
            if (FC_DEN) atomic_fetch_add_explicit(&tVals[base + 0u], den, memory_order_relaxed);
            if (FC_VEL) {
                atomic_fetch_add_explicit(&tVals[base + 1u], vx, memory_order_relaxed);
                atomic_fetch_add_explicit(&tVals[base + 2u], vy, memory_order_relaxed);
                atomic_fetch_add_explicit(&tVals[base + 3u], vz, memory_order_relaxed);
            }
            if (FC_SCAL) atomic_fetch_add_explicit(&tVals[base + tableOffScal()], sc, memory_order_relaxed);
            if (FC_GRAD)
                for (int j = 0; j < 3; ++j)
                    for (int i = 0; i < 3; ++i)
                        atomic_fetch_add_explicit(&tVals[base + tableOffGrad() + uint(j*3+i)], vG[i][j] * w, memory_order_relaxed);
            if (FC_SGRAD)
                for (int i = 0; i < 3; ++i)
                    atomic_fetch_add_explicit(&tVals[base + tableOffSGrad() + uint(i)], sG[i] * w, memory_order_relaxed);
            return true;
        }
        h = (h + 1u) & (E - 1u);
    }
    return false;
}

// The item kernel's body, templated on the cache size (the kernels below instantiate it: a
// function constant cannot size an array). Fewer slots = fewer registers = more threads in
// flight; more slots = fewer flushes for tetrahedra spanning many cells. Measured on 2.1M
// particles at 512^3 -- see the host for the default.
template <int K>
static inline void depositItem(
    device const float*     verts, device const float* dens, device const float* vels,
    device const float*     vols,  device const uint*  cnts, device const uint*  flats,
    device const float4*    sobol,
    device atomic_float*    denGrid, device atomic_float* velGrid, device atomic_float* gradGrid,
    device const float*     scal, device atomic_float* scalGrid, device atomic_float* sgGrid,
    constant DepositParams& P, uint3 const it, uint const lane,
    threadgroup atomic_uint* tKeys, threadgroup atomic_float* tVals)
{
    uint  const tid    = it.x;
    uint  const sub    = it.y;      // this item's index among the tetrahedron's items
    uint  const perTet = it.z;      // the tetrahedron's items: this one takes samples sub, sub+perTet, ...

    float base[3] = { verts[tid*12+0], verts[tid*12+1], verts[tid*12+2] };
    float Ax[3][3];
    for (int v=0; v<3; ++v)
        for (int d=0; d<3; ++d)
            Ax[v][d] = verts[tid*12+(v+1)*3+d] - base[d];

    float posInv[3][3];
    inverse3zero(Ax, posInv);

    bool const needVel = FC_VEL || FC_GRAD;
    float u0[3] = {0.0f,0.0f,0.0f};
    float vG[3][3] = {{0.0f,0.0f,0.0f},{0.0f,0.0f,0.0f},{0.0f,0.0f,0.0f}};   // vG[i][j] = d(v_j)/d(x_i)
    if (needVel) {
        for (int j=0;j<3;++j) u0[j] = vels[tid*12+j];
        for (int i=0;i<3;++i)
            for (int j=0;j<3;++j) {
                float s=0.0f;
                for (int k=0;k<3;++k) s += posInv[i][k]*(vels[tid*12+(k+1)*3+j]-u0[j]);
                vG[i][j]=s;
            }
    }

    // the scalar: linear over the tetrahedron, its gradient constant
    float s0 = 0.0f, sG[3] = {0.0f,0.0f,0.0f};
    if (FC_SCAL || FC_SGRAD) {
        s0 = scal[tid*4+0];
        for (int i=0;i<3;++i) {
            float s=0.0f;
            for (int j=0;j<3;++j) s += posInv[i][j]*(scal[tid*4+j+1]-s0);
            sG[i]=s;
        }
    }

    float const vol = vols[tid];
    uint  const n   = cnts[tid];

    CellCache<K> C;
    for (int k = 0; k < K; ++k) { C.flat[k] = 0u; C.den[k] = 0.0f; C.vx[k] = 0.0f; C.vy[k] = 0.0f; C.vz[k] = 0.0f; C.sc[k] = 0.0f; C.cnt[k] = 0.0f; }
    int used = 0;      // slots holding a cell
    int victim = 0;    // round-robin replacement pointer
    float factor;      // weight of the slot sums: V/n per sample, or V for the single-cell centroid

    if (n == 0u) {   // single-cell tetrahedron: its only item deposits the centroid value * volume
        float denC = 0.0f;
        for (int v=0; v<4; ++v) denC += dens[tid*4+v];
        denC *= 0.25f;
        float velC[3] = {0.0f,0.0f,0.0f};
        if (FC_VEL) {
            for (int v=0;v<4;++v) for (int j=0;j<3;++j) velC[j] += vels[tid*12+v*3+j];
            for (int j=0;j<3;++j) velC[j] *= 0.25f;
        }
        float scC = 0.0f;
        if (FC_SCAL) { for (int v=0;v<4;++v) scC += scal[tid*4+v]; scC *= 0.25f; }
        C.flat[0] = flats[tid]; C.den[0] = denC; C.vx[0] = velC[0]; C.vy[0] = velC[1]; C.vz[0] = velC[2]; C.sc[0] = scC; C.cnt[0] = 1.0f;
        used = 1;
        factor = vol;
    } else {
    float den0 = dens[tid*4+0];
    float densGrad[3];
    for (int i=0;i<3;++i) {
        float s=0.0f;
        for (int j=0;j<3;++j) s += posInv[i][j]*(dens[tid*4+j+1]-den0);
        densGrad[i]=s;
    }

    factor = vol / float(n);

    // Samples are dealt to a tetrahedron's items round-robin, so at every loop step the items of
    // one tetrahedron (consecutive lanes) read consecutive table entries: one 512-byte run per
    // SIMD group instead of 32 scattered lines. The set of samples per tetrahedron is unchanged.
    for (uint s = sub; s < n; s += perTet) {
        float4 const q = sobol[s];
        float const q0 = q.x, q1 = q.y, q2 = q.z;
        float p[3];
        for (int d=0;d<3;++d) p[d] = q0*Ax[0][d] + q1*Ax[1][d] + q2*Ax[2][d];
        int pos[3]; bool in=true;
        for (int d=0;d<3;++d) {
            pos[d] = int(floor((base[d]+p[d])*P.invDx[d]));
            if (pos[d]<0 || pos[d]>=P.nGrid[d]) { in=false; break; }
        }
        if (!in) continue;
        uint const flat = fullFlat(P, pos);
        float denS = den0;
        for (int i=0;i<3;++i) denS += p[i]*densGrad[i];
        float velS[3] = {0.0f,0.0f,0.0f};
        if (FC_VEL)
            for (int j=0;j<3;++j) {
                velS[j]=u0[j];
                for (int i=0;i<3;++i) velS[j] += vG[i][j]*p[i];
            }
        float scS = s0;
        if (FC_SCAL) for (int i=0;i<3;++i) scS += p[i]*sG[i];

        int hit = -1;
        for (int k = 0; k < K; ++k)
            hit = (k < used && C.flat[k] == flat) ? k : hit;
        if (hit < 0) {
            if (used < K) { hit = used; ++used; }
            else {
                hit = victim;
                victim = (victim + 1 == K) ? 0 : victim + 1;
                // gather the victim slot (predicated, keeps the cache in registers)
                uint  vKey = 0u; float vDen = 0.0f, vVx = 0.0f, vVy = 0.0f, vVz = 0.0f, vSc = 0.0f, vCnt = 0.0f;
                for (int j = 0; j < K; ++j) {
                    bool const mm = (j == hit);
                    vKey = mm ? C.flat[j] : vKey; vDen = mm ? C.den[j] : vDen; vVx = mm ? C.vx[j] : vVx;
                    vVy = mm ? C.vy[j] : vVy; vVz = mm ? C.vz[j] : vVz; vSc = mm ? C.sc[j] : vSc; vCnt = mm ? C.cnt[j] : vCnt;
                }
                if (P.tableEntries == 0u ||
                    !tableAdd(tKeys, tVals, P.tableEntries, vKey, vDen * factor, vVx * factor, vVy * factor,
                              vVz * factor, vSc * factor, vCnt * factor, vG, sG))
                    flushSlot<K>(denGrid, velGrid, gradGrid, scalGrid, sgGrid, P, C, hit, factor, vG, sG);
            }
            for (int k = 0; k < K; ++k) {       // (re)initialise slot 'hit'
                bool const m = (k == hit);
                C.flat[k] = m ? flat : C.flat[k];
                C.den[k]  = m ? 0.0f : C.den[k];
                C.vx[k]   = m ? 0.0f : C.vx[k];
                C.vy[k]   = m ? 0.0f : C.vy[k];
                C.vz[k]   = m ? 0.0f : C.vz[k];
                C.sc[k]   = m ? 0.0f : C.sc[k];
                C.cnt[k]  = m ? 0.0f : C.cnt[k];
            }
        }
        for (int k = 0; k < K; ++k) {           // accumulate into slot 'hit'
            bool const m = (k == hit);
            C.den[k] = m ? C.den[k] + denS    : C.den[k];
            if (FC_VEL) {
                C.vx[k]  = m ? C.vx[k]  + velS[0] : C.vx[k];
                C.vy[k]  = m ? C.vy[k]  + velS[1] : C.vy[k];
                C.vz[k]  = m ? C.vz[k]  + velS[2] : C.vz[k];
            }
            if (FC_SCAL)
                C.sc[k]  = m ? C.sc[k]  + scS     : C.sc[k];
            C.cnt[k] = m ? C.cnt[k] + 1.0f    : C.cnt[k];
        }
    }
    }   // sampled path

    // ---- flush with a SIMD-group reduction by cell. The lanes of a SIMD group are 32 consecutive
    // items; the host orders the tetrahedra along a Morton curve, so they are the items of one
    // tetrahedron and its neighbours and their slots name the same few cells (measured 1.0-1.3
    // distinct base cells per 32 items at 512^3). Per slot: elect the smallest pending key, sum
    // the pre-weighted values of every lane holding it with simd_sum, and let one lane add them
    // -- one atomic per field per distinct cell per SIMD group instead of per lane. Control flow
    // is uniform (simd_any); lanes that exited early (no item, see the kernel) are inactive and
    // take no part. The sums only reorder the float additions.
    for (int k = 0; k < K; ++k) {
        bool  pend = k < used;
        uint  key  = pend ? C.flat[k] : 0xffffffffu;
        float const dW = C.den[k] * factor;
        float const xW = C.vx[k] * factor, yW = C.vy[k] * factor, zW = C.vz[k] * factor;
        float const cW = C.sc[k] * factor;
        float const gW = C.cnt[k] * factor;           // the gradients' weight: they are constant over the tetrahedron
        while (simd_any(pend)) {
            uint const leaderKey = simd_min(pend ? key : 0xffffffffu);
            bool const m = pend && key == leaderKey;
            uint const leader = simd_min(m ? lane : 32u);
            float const sd = FC_DEN ? simd_sum(m ? dW : 0.0f) : 0.0f;
            float sx = 0.0f, sy = 0.0f, sz = 0.0f;
            if (FC_VEL) {
                sx = simd_sum(m ? xW : 0.0f);
                sy = simd_sum(m ? yW : 0.0f);
                sz = simd_sum(m ? zW : 0.0f);
            }
            float const ss = FC_SCAL ? simd_sum(m ? cW : 0.0f) : 0.0f;
            if (lane == leader && !P.debugNoAtomic) {
                ulong const f = ulong(leaderKey);
                if (FC_DEN)
                    atomic_fetch_add_explicit(&denGrid[f], sd, memory_order_relaxed);
                if (FC_VEL) {
                    atomic_fetch_add_explicit(&velGrid[f*3ul+0ul], sx, memory_order_relaxed);
                    atomic_fetch_add_explicit(&velGrid[f*3ul+1ul], sy, memory_order_relaxed);
                    atomic_fetch_add_explicit(&velGrid[f*3ul+2ul], sz, memory_order_relaxed);
                }
                if (FC_SCAL)
                    atomic_fetch_add_explicit(&scalGrid[f], ss, memory_order_relaxed);
            }
            if ((FC_GRAD || FC_SGRAD) && P.debugNoAtomic < 2u) {
                // The gradient is a per-TETRAHEDRON constant times the cell weight, so it is reduced per
                // (cell, tetrahedron): one weight sum per tetrahedron sharing the cell (usually one or
                // two in a SIMD group) instead of nine component sums, and no nine-float array alive.
                bool mg = m;
                while (simd_any(mg)) {
                    uint const leaderTet = simd_min(mg ? tid : 0xffffffffu);
                    bool const mt = mg && tid == leaderTet;
                    uint const leaderG = simd_min(mt ? lane : 32u);
                    float const sw = simd_sum(mt ? gW : 0.0f);
                    if (lane == leaderG && !P.debugNoAtomic) {
                        ulong const f = ulong(leaderKey);
                        if (FC_GRAD)
                            for (int j = 0; j < 3; ++j)
                                for (int i = 0; i < 3; ++i)
                                    atomic_fetch_add_explicit(&gradGrid[f*9ul+ulong(j*3+i)], vG[i][j] * sw, memory_order_relaxed);
                        if (FC_SGRAD)
                            for (int i = 0; i < 3; ++i)
                                atomic_fetch_add_explicit(&sgGrid[f*3ul+ulong(i)], sG[i] * sw, memory_order_relaxed);
                    }
                    mg = mg && !mt;
                }
            }
            pend = pend && !m;
        }
    }
}

// Flushes the threadgroup table into the grids: one device atomic per field per used entry.
static inline void tableFlush(threadgroup atomic_uint* tKeys, threadgroup atomic_float* tVals, uint E,
                              uint ltid, uint tgSize,
                              device atomic_float* denGrid, device atomic_float* velGrid,
                              device atomic_float* gradGrid, device atomic_float* scalGrid,
                              device atomic_float* sgGrid, constant DepositParams& P)
{
    for (uint e = ltid; e < E; e += tgSize) {
        uint const key = atomic_load_explicit(&tKeys[e], memory_order_relaxed);
        if (key == TABLE_EMPTY) continue;
        ulong const f = ulong(key);
        uint const base = e * tableStride();
        if (P.debugNoAtomic) continue;
        if (FC_DEN)
            atomic_fetch_add_explicit(&denGrid[f], atomic_load_explicit(&tVals[base + 0u], memory_order_relaxed), memory_order_relaxed);
        if (FC_VEL)
            for (uint c = 0u; c < 3u; ++c)
                atomic_fetch_add_explicit(&velGrid[f*3ul+ulong(c)], atomic_load_explicit(&tVals[base + 1u + c], memory_order_relaxed), memory_order_relaxed);
        if (FC_SCAL)
            atomic_fetch_add_explicit(&scalGrid[f], atomic_load_explicit(&tVals[base + tableOffScal()], memory_order_relaxed), memory_order_relaxed);
        if (FC_GRAD)
            for (uint q = 0u; q < 9u; ++q)
                atomic_fetch_add_explicit(&gradGrid[f*9ul+ulong(q)], atomic_load_explicit(&tVals[base + tableOffGrad() + q], memory_order_relaxed), memory_order_relaxed);
        if (FC_SGRAD)
            for (uint i = 0u; i < 3u; ++i)
                atomic_fetch_add_explicit(&sgGrid[f*3ul+ulong(i)], atomic_load_explicit(&tVals[base + tableOffSGrad() + i], memory_order_relaxed), memory_order_relaxed);
    }
}

#define DTFE_ITEM_KERNEL(NAME, K)                                                           \
kernel void NAME(                                                                           \
    device const float*     verts    [[buffer(0)]],                                         \
    device const float*     dens     [[buffer(1)]],                                         \
    device const float*     vels     [[buffer(2)]],                                         \
    device const float*     vols     [[buffer(3)]],                                         \
    device const uint*      cnts     [[buffer(4)]],                                         \
    device const uint*      flats    [[buffer(5)]],                                         \
    device const float4*    sobol    [[buffer(6)]],                                         \
    device atomic_float*    denGrid  [[buffer(7)]],                                         \
    device atomic_float*    velGrid  [[buffer(8)]],                                         \
    device atomic_float*    gradGrid [[buffer(9)]],                                         \
    constant DepositParams& P        [[buffer(10)]],                                        \
    device const uint*      itemStart [[buffer(11)]], /* first item of each tetrahedron */  \
    device const uint*      blockTet [[buffer(12)]],  /* tetrahedron of each block's first */ \
    device const float*     scal     [[buffer(13)]],  /* nTet*4 vertex scalars (dummy when unused) */ \
    device atomic_float*    scalGrid [[buffer(14)]],                                        \
    device atomic_float*    sgGrid   [[buffer(15)]],                                        \
    threadgroup atomic_uint*  tKeys  [[threadgroup(0)]],  /* cell table keys   (host-sized) */ \
    threadgroup atomic_float* tVals  [[threadgroup(1)]],  /* cell table values (host-sized) */ \
    uint iid [[thread_position_in_grid]],                                                   \
    uint lane [[thread_index_in_simdgroup]],                                                \
    uint ltid [[thread_position_in_threadgroup]],                                           \
    uint tgSize [[threads_per_threadgroup]])                                                \
{                                                                                           \
    uint const E = P.tableEntries;                                                          \
    for (uint e = ltid; e < E; e += tgSize) {                                               \
        atomic_store_explicit(&tKeys[e], TABLE_EMPTY, memory_order_relaxed);                \
        for (uint c = 0u; c < tableStride(); ++c)                                           \
            atomic_store_explicit(&tVals[e * tableStride() + c], 0.0f, memory_order_relaxed); \
    }                                                                                       \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                        \
    if (iid < P.nTet)                                                                       \
        depositItem<K>(verts, dens, vels, vols, cnts, flats, sobol, denGrid, velGrid, gradGrid, \
                       scal, scalGrid, sgGrid,                                              \
                       P, itemOf(itemStart, blockTet, P.nTetTotal, P.itemBase + iid), lane, \
                       tKeys, tVals);                                                       \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                        \
    tableFlush(tKeys, tVals, E, ltid, tgSize, denGrid, velGrid, gradGrid, scalGrid, sgGrid, P); \
}

DTFE_ITEM_KERNEL(depositAveraged1Items2, 2)
DTFE_ITEM_KERNEL(depositAveraged1Items4, 4)
DTFE_ITEM_KERNEL(depositAveraged1Items8, 8)


// ---------------------------------------------------------------------------------------
// depositExactAverage: --exact-average on the GPU. One thread per ITEM = (tetrahedron, block of
// P.itemSamples consecutive cells of its bounding-box window, in gi-gj-gk order), the phase-space
// depositExactItems' split: each piece of the tetrahedron inside a cell deposits the LINEAR
// interpolant's integral over the piece -- its centroid value times the piece volume (order-1
// moments, Powell & Abel 2015; the r3d port of metal/exact_clip.metal.inc) -- and the gradients
// times the piece volume, exactly as the CPU loop of averaged_interpolation_1.cc. There is no
// per-tetrahedron normalization (the volume average has none), so a single dispatch. A single-cell
// tetrahedron (cnts == 0) has one item, which deposits its centroid value times its volume: the
// CPU fast path, shared with the sampled mode. The items reuse the itemStart/blockTet map
// (itemOf); the host bounds every window from its double-precision copy of the vertices, and a
// surplus block (the kernel's float window is smaller) idles. No cell cache: a piece is visited
// once per (tetrahedron, cell), so every deposit is one atomic per field.
// ---------------------------------------------------------------------------------------
kernel void depositExactAverage(
    device const float*     verts    [[buffer(0)]],
    device const float*     dens     [[buffer(1)]],
    device const float*     vels     [[buffer(2)]],
    device const float*     vols     [[buffer(3)]],
    device const uint*      cnts     [[buffer(4)]],
    device const uint*      flats    [[buffer(5)]],
    device atomic_float*    denGrid  [[buffer(7)]],
    device atomic_float*    velGrid  [[buffer(8)]],
    device atomic_float*    gradGrid [[buffer(9)]],
    constant DepositParams& P        [[buffer(10)]],
    device const uint*      itemStart [[buffer(11)]],
    device const uint*      blockTet [[buffer(12)]],
    device const float*     scal     [[buffer(13)]],
    device atomic_float*    scalGrid [[buffer(14)]],
    device atomic_float*    sgGrid   [[buffer(15)]],
    uint iid [[thread_position_in_grid]])
{
    if (iid >= P.nTet) return;
    uint3 const it = itemOf(itemStart, blockTet, P.nTetTotal, P.itemBase + iid);
    uint const tid = it.x, sub = it.y;

    float base[3] = { verts[tid*12+0], verts[tid*12+1], verts[tid*12+2] };
    float Ax[3][3];
    for (int v=0; v<3; ++v)
        for (int d=0; d<3; ++d)
            Ax[v][d] = verts[tid*12+(v+1)*3+d] - base[d];
    float posInv[3][3];
    inverse3zero(Ax, posInv);

    bool const needVel = (P.fVel != 0) || (P.fGrad != 0);
    float u0[3] = {0.0f,0.0f,0.0f};
    float vG[3][3] = {{0.0f,0.0f,0.0f},{0.0f,0.0f,0.0f},{0.0f,0.0f,0.0f}};
    float gradFlat[9] = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
    if (needVel) {
        for (int j=0;j<3;++j) u0[j] = vels[tid*12+j];
        for (int i=0;i<3;++i)
            for (int j=0;j<3;++j) {
                float s=0.0f;
                for (int k=0;k<3;++k) s += posInv[i][k]*(vels[tid*12+(k+1)*3+j]-u0[j]);
                vG[i][j]=s;
            }
        for (int j=0;j<3;++j)
            for (int i=0;i<3;++i)
                gradFlat[j*3+i] = vG[i][j];
    }
    bool const needScal = (P.fScal != 0) || (P.fSGrad != 0);
    float s0 = 0.0f, sG[3] = {0.0f,0.0f,0.0f};
    if (needScal) {
        s0 = scal[tid*4+0];
        for (int i=0;i<3;++i) {
            float s=0.0f;
            for (int j=0;j<3;++j) s += posInv[i][j]*(scal[tid*4+j+1]-s0);
            sG[i]=s;
        }
    }

    if (cnts[tid] == 0u) {   // single-cell tetrahedron: the fast path, by its only item
        if (sub != 0u) return;
        float denC = 0.0f;
        for (int v=0; v<4; ++v) denC += dens[tid*4+v];
        denC *= 0.25f;
        float velC[3] = {0.0f,0.0f,0.0f};
        if (P.fVel) {
            for (int v=0;v<4;++v) for (int j=0;j<3;++j) velC[j] += vels[tid*12+v*3+j];
            for (int j=0;j<3;++j) velC[j] *= 0.25f;
        }
        float scC = 0.0f;
        if (P.fScal) { for (int v=0;v<4;++v) scC += scal[tid*4+v]; scC *= 0.25f; }
        depositCell(denGrid, velGrid, gradGrid, scalGrid, sgGrid, P, ulong(flats[tid]), denC, velC, gradFlat, scC, sG, vols[tid]);
        return;
    }

    float den0 = dens[tid*4+0];
    float densGrad[3];
    for (int i=0;i<3;++i) {
        float s=0.0f;
        for (int j=0;j<3;++j) s += posInv[i][j]*(dens[tid*4+j+1]-den0);
        densGrad[i]=s;
    }

    // the window of grid cells overlapped by the tetrahedron's bounding box (the CPU's iLo/iHi:
    // sub-grid indices, clamped to the sub-grid)
    int iLo[3], iHi[3];
    for (int d=0; d<3; ++d) {
        float lo = 0.0f, hi = 0.0f;
        for (int v=0; v<3; ++v) { lo = min(lo, Ax[v][d]); hi = max(hi, Ax[v][d]); }
        iLo[d] = int(floor((base[d] + lo) * P.invDx[d]));
        iHi[d] = int(floor((base[d] + hi) * P.invDx[d])) + 1;
        if (iLo[d] < 0) iLo[d] = 0;
        if (iHi[d] > P.nGrid[d]) iHi[d] = P.nGrid[d];
    }
    ulong const n1 = ulong(max(iHi[1] - iLo[1], 0));
    ulong const n2 = ulong(max(iHi[2] - iLo[2], 0));
    ulong const n12 = n1 * n2;
    ulong const nW = ulong(max(iHi[0] - iLo[0], 0)) * n12;
    ulong w = ulong(sub) * ulong(P.itemSamples);
    if (w >= nW) return;             // the host's window bound has a margin: surplus blocks idle
    ulong const wEnd = (nW - w > ulong(P.itemSamples)) ? w + ulong(P.itemSamples) : nW;
    int a0 = int(w / n12);
    ulong const rem = w - ulong(a0) * n12;
    int a1 = int(rem / n2);
    int a2 = int(rem - ulong(a1) * n2);

    // the tetrahedron in its vertex-0 frame, positively oriented for r3d (a negative one has
    // vertices 1 and 2 swapped, as the CPU does)
    float3 t1 = float3(Ax[0][0], Ax[0][1], Ax[0][2]);
    float3 t2 = float3(Ax[1][0], Ax[1][1], Ax[1][2]);
    float3 t3 = float3(Ax[2][0], Ax[2][1], Ax[2][2]);
    if (det3(Ax) < 0.0f) { float3 tt = t1; t1 = t2; t2 = tt; }
    TetPlanes const faces = tet_planes(t1, t2, t3);
    float3 const hcell = float3(0.5f*P.dx[0], 0.5f*P.dx[1], 0.5f*P.dx[2]);
    float3 const tlo = min(min(float3(0.0f), t1), min(t2, t3));   // the tet's bbox (vertex-0 frame)
    float3 const thi = max(max(float3(0.0f), t1), max(t2, t3));

    for (; w < wEnd; ++w) {
        int const pos[3] = { iLo[0] + a0, iLo[1] + a1, iLo[2] + a2 };
        if (ulong(++a2) == n2) { a2 = 0; if (ulong(++a1) == n1) { a1 = 0; ++a0; } }
        // the cell in the vertex-0 frame: a cell wholly outside a face plane holds no piece
        float3 const ccen = float3((float(pos[0]) + 0.5f) * P.dx[0] - base[0],
                                   (float(pos[1]) + 0.5f) * P.dx[1] - base[1],
                                   (float(pos[2]) + 0.5f) * P.dx[2] - base[2]);
        if (box_outside_tet(faces, ccen, hcell)) continue;
        // clipped in the OVERLAP frame (exact_clip.metal.inc: origin at the centre of the tet bbox's
        // overlap with the cell, where the piece lies -- no frame far from the piece cancels)
        float3 const clo = ccen - hcell, chi = ccen + hcell;
        float3 const org = 0.5f * (max(clo, tlo) + min(chi, thi));
        ExPoly piece;
        exact_init_tet4(piece, -org, t1 - org, t2 - org, t3 - org);
        for (int d=0; d<3; ++d) {
            exact_clip_plane(piece, d,  1.0f, org[d] - clo[d]);   // x_d >= clo - org
            exact_clip_plane(piece, d, -1.0f, chi[d] - org[d]);   // x_d <= chi - org
        }
        if (piece.nv == 0) continue;
        float mo[4];
        exact_reduce1(piece, mo);
        if (!(mo[0] > 0.0f)) continue;
        float const dV = mo[0];
        float const cen[3] = { org.x + mo[1]/mo[0], org.y + mo[2]/mo[0], org.z + mo[3]/mo[0] };   // the piece's centroid, vertex-0 frame
        float denS = den0;
        for (int i=0;i<3;++i) denS += cen[i]*densGrad[i];
        float velS[3] = {0.0f,0.0f,0.0f};
        if (P.fVel)
            for (int j=0;j<3;++j) {
                velS[j]=u0[j];
                for (int i=0;i<3;++i) velS[j] += vG[i][j]*cen[i];
            }
        float scS = s0;
        if (P.fScal) for (int i=0;i<3;++i) scS += cen[i]*sG[i];
        depositCell(denGrid, velGrid, gradGrid, scalGrid, sgGrid, P, ulong(fullFlat(P, pos)), denS, velS, gradFlat, scS, sG, dV);
    }
}
