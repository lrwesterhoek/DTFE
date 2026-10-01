/* Arbitrary-point evaluation (--sample-points), shared by BOTH binaries.

   PS-DTFE: evaluates the multi-stream phase-space fields at user-supplied Eulerian points
   instead of (only) on a regular grid. The triangulation is built in Lagrangian space;
   every Delaunay cell maps to a (possibly folded) Eulerian tetrahedron, and a query point
   receives one stream per tetrahedron that contains it. Per point we report the total
   density, the density-weighted mean velocity, the velocity dispersion tensor and the
   stream count; --per-stream additionally records every stream's own density and velocity,
   and --per-stream-ids each stream's identity (the sorted Lagrangian-vertex ParticleIDs).
   --pts-den-grad adds the density gradient: per stream the constant gradient of the linear
   'dtfe' profile (regardless of --ps-stream-density -- the geometric density is piecewise
   constant, so the dtfe-profile gradient is the only well-defined one), per point the sum
   over its streams.

   Standard DTFE (no PHASE_SPACE): the same interface and .pts_* file formats over the
   EULERIAN tessellation -- exactly one containing tetrahedron per point, so the stream
   count is the 0/1 coverage flag, the density is the plain linear DTFE interpolant and the
   dispersion is identically 0. The context, bucket index, reduction and writers below are
   shared verbatim; only the per-triangulation worker differs (interpolatePoints_standard).
   --per-stream/--per-stream-ids and --ps-stream-density are PS-only (rejected at option
   parsing); the legacy Sample_point mechanism is untouched.

   Per-stream density comes in two variants (--ps-stream-density):
     'dtfe'      (default) linear interpolation of the Lagrangian-vertex DTFE densities
                 inside the tetrahedron -- matches the PhaseSpaceDTFE class of the method
                 authors' reference implementation (github.com/jfeldbrugge/PS-DTFE,
                 Python/density.py). Continuous within a stream, but not mass-conserving.
     'geometric' the constant per-tetrahedron density m_tet / V_eul (= rho_bar V_lag/V_eul)
                 -- the density whose cell-averaged, quantized rendering the mass-conserving
                 grid deposit produces. Discontinuous across tetrahedra, conserves mass.

   The candidate search buckets the (static, read-only) QUERY POINTS on a uniform grid and
   streams the partition's tetrahedra through it: each kept tetrahedron looks up only the
   buckets its Eulerian bounding box overlaps. This is the same O(T + P + hits) complexity
   as a hierarchy over the tetrahedron boxes, with the smaller (point) set indexed.

   Partition protocol: the cell filters below are the CPU grid deposit's filters VERBATIM
   (ps_interpolation.cc) -- dummy-vertex skip, Lagrangian-centroid ownership, bad-density
   drop under periodic, minimum-image wrap of vertices 1..NO_DIM to vertex 0, relative-
   determinant degeneracy and the zero-inverse check -- so a stream is contributed by
   exactly one Lagrangian partition and the union over partitions is exactly the serial
   stream set. Per-point stream records accumulate across partitions (the analogue of
   psDeferNormalization: moments are formed only once, in psPointEvalFinalize, from the
   deterministically sorted records, so results do not depend on partition/thread order).

   Containment is EXACT and tie-consistent (pointInTet; the machinery lives in
   ps_exact_inside.h, which the CPU grid deposit shares for its sub-samples): a point is counted by
   exactly one tetrahedron of every stream, including points ON a shared face, edge or
   vertex -- e.g. a query at a particle position, which a tolerance-based test counts once
   per incident tetrahedron (~25 fake streams). Two ingredients make that possible:
     * every periodic image of a particle is mapped to one canonical coordinate
       (canonicalCoord), so tetrahedra sharing a face see bit-identical vertices even when
       one holds the original particle and the other its (float-shifted) periodic copy;
     * the decision uses exact orientation predicates (CGAL, filtered) with a symbolic
       perturbation of the query point (simulation of simplicity) that breaks every tie
       the same way in all tetrahedra.
   A cheap double-precision classification decides every point that is not within a
   rigorous per-tetrahedron uncertainty band of a face; only the band goes to the exact
   predicates, so the common path costs what the old tolerance test did.

   Each triangulation's cells are processed by a small thread pool (pointEvalThreads): all
   cores for a single-triangulation run, cores / concurrent partitions inside the
   partitioned PS loop, so the partition-level parallelism is never oversubscribed. */

#include "../define.h"

#include "../ps_point_eval.h"

#if NO_DIM==3

#include "triangulation_common.h"
#include "ps_caustic_class.h"   // --ps-caustics: per-tetrahedron caustic stratification bits
#include "ps_exact_inside.h"    // exact, tie-consistent containment (shared with the grid deposit)
#include "tessellation_cache.h" // composite --serve: partitions reloaded from '--tessellation-cache'
#include "../auto_tune.h"       // composite --serve: the memory budget that sizes the resident set

#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>   // --serve: dup/dup2/read/write on the protocol channel
#ifdef OPEN_MP
#include <omp.h>
#endif

// terminal-message prefix only -- the machinery is shared by both binaries and the PS
// output must stay byte-identical (including its log lines)
#ifdef PHASE_SPACE
#define PTS_MSG_BINARY "PS-DTFE:"
#else
#define PTS_MSG_BINARY "DTFE:"
#endif


// '--tessellation-cache' load or build of a tessellation plus its vertex densities (triangulation.cpp,
// the first half of DTFE_interpolation); true when it came from the cache. Composite --serve only.
bool buildTessellation(std::vector<Particle_data> *p, User_options &userOptions, DT &dt);

namespace psPointEvalDetail {

using namespace psExactInside;

// One stream (containing tetrahedron) at a query point. Both density variants are kept so
// the estimator choice only affects the reduction, never the collected geometry.
struct StreamRec
{
    double denGeo;              // 'geometric': rho_bar * V_lag / V_eul, constant per tet
    double denDtfe;             // 'dtfe': vertex densities linearly interpolated at the point
    double vel[noVelComp];      // velocity linearly interpolated at the point
    double grad[NO_DIM];        // constant 'dtfe' density-profile gradient of the tet (--pts-den-grad; 0 otherwise)
    double vgrad[NO_DIM*NO_DIM];// constant velocity gradient of the tet, [d*3+j] = dv_j/dx_d (--pts-vel-grad; 0 otherwise)
    uint64_t ids[NO_DIM+1];     // sorted Lagrangian-vertex ParticleIDs (--per-stream-ids; all 0 otherwise)
    PSCausticClass::CausticMask caustic;   // --ps-caustics: this tetrahedron's fold parity + stratification bits (ps_caustic_class.h; 0 otherwise). Point evaluation runs the derivative-free classifyTet only, so bits 7..8 (the A3/A4 indicators, which need the vertex-incident stencil) are never set here -- the type matches the grid deposit's so that adding them later cannot silently truncate
    double scal[noScalarComp];  // --pts-scalar: the stream's linear interpolant of its vertices' scalar values (0 otherwise)
    double lag[NO_DIM];         // --pts-lagrangian (PS-DTFE): the stream's Lagrangian coordinate q(x), wrapped into the box when periodic (0 otherwise)
};

// The STORED form of a StreamRec: only the members this run's options fill. The core (both
// densities, the velocity, the caustic mask) always; the density gradient, velocity gradient and
// vertex IDs only with --pts-den-grad / --pts-vel-grad / --per-stream-ids. The records ARE the
// memory of a large --sample-points run (an 8192^2 plane x ~4 streams at z=0 held ~48 GB of
// 176-B StreamRecs), and packed they take 42 B without those options, 114 B with --pts-vel-grad.
// unpack() zeroes the members that were not stored -- exactly what the record builders put there
// when the option is off -- so finalize() sorts and reduces the SAME records: bit-identical output.
struct RecLayout
{
    bool   grad = false, vgrad = false, ids = false, scal = false, lag = false;
    size_t bytes = 0;
    void init(bool withGrad, bool withVGrad, bool withIds, bool withScal, bool withLag)
    {
        grad = withGrad; vgrad = withVGrad; ids = withIds; scal = withScal; lag = withLag;
        bytes = 2*sizeof(double) + noVelComp*sizeof(double) + sizeof(PSCausticClass::CausticMask)
              + (grad  ? NO_DIM*sizeof(double)        : 0)
              + (vgrad ? NO_DIM*NO_DIM*sizeof(double) : 0)
              + (ids   ? (NO_DIM+1)*sizeof(uint64_t)  : 0)
              + (scal  ? noScalarComp*sizeof(double)  : 0)
              + (lag   ? NO_DIM*sizeof(double)        : 0);
    }
    void pack(std::vector<unsigned char> &out, StreamRec const &r) const
    {
        size_t const at = out.size();
        out.resize( at + bytes );
        unsigned char *p = out.data() + at;
        auto put = [&p](void const *src, size_t n) { std::memcpy( p, src, n ); p += n; };
        put( &r.denGeo, sizeof(r.denGeo) );
        put( &r.denDtfe, sizeof(r.denDtfe) );
        put( r.vel, sizeof(r.vel) );
        put( &r.caustic, sizeof(r.caustic) );
        if ( grad )  put( r.grad, sizeof(r.grad) );
        if ( vgrad ) put( r.vgrad, sizeof(r.vgrad) );
        if ( ids )   put( r.ids, sizeof(r.ids) );
        if ( scal )  put( r.scal, sizeof(r.scal) );
        if ( lag )   put( r.lag, sizeof(r.lag) );
    }
    void unpackOne(unsigned char const *p, StreamRec &r) const
    {
        r = StreamRec();                    // value-initialized: every unstored member is 0
        auto get = [&p](void *dst, size_t k) { std::memcpy( dst, p, k ); p += k; };
        get( &r.denGeo, sizeof(r.denGeo) );
        get( &r.denDtfe, sizeof(r.denDtfe) );
        get( r.vel, sizeof(r.vel) );
        get( &r.caustic, sizeof(r.caustic) );
        if ( grad )  get( r.grad, sizeof(r.grad) );
        if ( vgrad ) get( r.vgrad, sizeof(r.vgrad) );
        if ( ids )   get( r.ids, sizeof(r.ids) );
        if ( scal )  get( r.scal, sizeof(r.scal) );
        if ( lag )   get( r.lag, sizeof(r.lag) );
    }
};

// An Eulerian ownership box (see Context::own). 'on' = false keeps every tetrahedron.
struct OwnBox
{
    bool   on = false;
    double lo[NO_DIM], hi[NO_DIM];
    bool contains(double const c[NO_DIM]) const
    {
        for (int d = 0; d < NO_DIM; ++d)
            if ( c[d] < lo[d] || c[d] >= hi[d] ) return false;
        return true;
    }
};

struct Context
{
    bool   active = false;
    bool   finalized = false;
    bool   perStream = false;
    bool   perStreamIds = false;
    bool   ptsDenGrad = false;
    bool   ptsVelGrad = false;
    bool   psCaustics = false;      // --ps-caustics: also emit the per-point caustic mask
    bool   ptsScalar = false;       // --pts-scalar: also interpolate the vertex scalars per stream
    bool   ptsLagrangian = false;   // --pts-lagrangian (PS): also record each stream's Lagrangian coordinate
    bool   useGeometric = false;    // variant used for totals, weights and per-stream output
    bool   periodic = false;
    double rhoBar = 1.;             // output densities are rho/rhoBar (grid convention)
    double boxLo[NO_DIM], boxLen[NO_DIM];

    // exact containment (see the file header): the canonical-image frame (ps_exact_inside.h)
    Frame  canon;

    // Standard binary: which tetrahedra this triangulation OWNS -- those whose Eulerian centroid
    // (vertices min-image wrapped to vertex 0) lies in [lo, hi). A periodic single triangulation
    // holds boundary tetrahedra twice (original + image in the padding) and owns the primary-box
    // one; a composite server's partition owns its own region. The PS worker's ownership is the
    // Lagrangian-centroid test inside psFilterCell (the partition's lagrangianRegion) instead.
    OwnBox own;

    size_t nPoints = 0;
    std::vector<double> pos;        // 3*nPoints, wrapped into the box when periodic

    // uniform bucket grid over the query points (CSR layout)
    int    nB[NO_DIM];
    double bInvW[NO_DIM];
    std::vector<size_t>   bucketStart;   // nB^3 + 1 offsets
    std::vector<uint32_t> bucketPts;     // point indices grouped by bucket

    // The stream records of every point, appended in BLOCKS by (possibly concurrent) workers.
    // An entry is the uint32 point index followed by the PACKED record (RecLayout: only the
    // members the options fill); finalize() groups the entries by point with a counting sort.
    // Records used to live in one std::vector per point (a heap block each) and every worker
    // held its partition's hits UNPACKED (~200 B each) until one merge at the end: 16.8M points
    // at 2.1M particles peaked at 28 GB. Now a worker holds one block at most.
    std::vector< std::vector<unsigned char> > recBlocks;
    size_t nRecs = 0;
    RecLayout layout;

    // finalized outputs
    std::vector<double>   outDen;        // N
    std::vector<double>   outVel;        // 3N
    std::vector<double>   outDisp;       // 6N, upper triangle row-major: xx xy xz yy yz zz
    std::vector<int32_t>  outStreams;    // N
    std::vector<uint64_t> outOffsets;    // N+1 (only with --per-stream)
    std::vector<double>   outRecords;    // 4 per stream: density, vx, vy, vz
    std::vector<uint64_t> outIds;        // NO_DIM+1 per stream (only with --per-stream-ids)
    std::vector<double>   outDenGrad;    // 3N (only with --pts-den-grad)
    std::vector<double>   outVelGrad;    // 9N (only with --pts-vel-grad)
    std::vector<double>   outRecGrad;    // 3 per stream (only with --pts-den-grad + --per-stream)
    std::vector<int32_t>  outCaustic;    // N (only with --ps-caustics): OR of the containing tetrahedra's masks
    std::vector<double>   outScalar;     // noScalarComp per point (only with --pts-scalar): density-weighted mean over the streams
    std::vector<double>   outRecScalar;  // noScalarComp per stream (--pts-scalar + --per-stream)
    std::vector<double>   outRecLag;     // 3 per stream (only with --pts-lagrangian)
};

static Context g_ctx;
static std::mutex g_mergeMutex;   // guards g_ctx.recBlocks while workers hand over their blocks

// A worker thread's hits: packed at once (point index + RecLayout bytes) and handed to the shared
// store whenever the block reaches FLUSH_BYTES -- a thread never holds more than one block.
struct HitSink
{
    static constexpr size_t FLUSH_BYTES = size_t(16) << 20;
    std::vector<unsigned char> buf;
    size_t n = 0;

    void push_back(std::pair<uint32_t, StreamRec> const &h)
    {
        if ( buf.capacity() == 0 ) buf.reserve( FLUSH_BYTES + 1024 );
        size_t const at = buf.size();
        buf.resize( at + sizeof(uint32_t) );
        std::memcpy( buf.data() + at, &h.first, sizeof(uint32_t) );
        g_ctx.layout.pack( buf, h.second );
        ++n;
        if ( buf.size() >= FLUSH_BYTES ) flush();
    }
    void flush()
    {
        if ( n == 0 ) return;
        buf.shrink_to_fit();
        std::lock_guard<std::mutex> lock( g_mergeMutex );
        g_ctx.recBlocks.push_back( std::move(buf) );
        g_ctx.nRecs += n;
        buf = std::vector<unsigned char>();
        n = 0;
    }
};


// 3x3 double inverse via the adjugate WITHOUT inverse3x3d's absolute |det| guard (an absolute
// threshold in Mpc^3 also rejects perfectly shaped tetrahedra that are merely small, e.g. in
// halo cores). Returns det; the inverse is valid when det is finite and nonzero.
inline double inverse3x3Unguarded(double const m[NO_DIM][NO_DIM], double inv[NO_DIM][NO_DIM])
{
    double const c00 = m[1][1]*m[2][2] - m[1][2]*m[2][1];
    double const c01 = m[1][2]*m[2][0] - m[1][0]*m[2][2];
    double const c02 = m[1][0]*m[2][1] - m[1][1]*m[2][0];
    double const det = m[0][0]*c00 + m[0][1]*c01 + m[0][2]*c02;
    if ( det == 0. ) return 0.;
    double const invDet = 1.0 / det;
    inv[0][0] = c00 * invDet;
    inv[1][0] = c01 * invDet;
    inv[2][0] = c02 * invDet;
    inv[0][1] = (m[0][2]*m[2][1] - m[0][1]*m[2][2]) * invDet;
    inv[1][1] = (m[0][0]*m[2][2] - m[0][2]*m[2][0]) * invDet;
    inv[2][1] = (m[0][1]*m[2][0] - m[0][0]*m[2][1]) * invDet;
    inv[0][2] = (m[0][1]*m[1][2] - m[0][2]*m[1][1]) * invDet;
    inv[1][2] = (m[0][2]*m[1][0] - m[0][0]*m[1][2]) * invDet;
    inv[2][2] = (m[0][0]*m[1][1] - m[0][1]*m[1][0]) * invDet;
    return det;
}


// Eulerian bounding box of a wrapped tetrahedron -> point-bucket range and rel-space bbox
// pre-test bounds. The slack covers the canonical-vs-wrapped geometry discrepancy (epsAbs),
// so no point the exact test could accept is ever culled here. Shared by the PS and standard
// workers so the candidate search is identical in both binaries. Returns false when the
// (non-periodic) range misses every bucket.
inline bool tetBucketRange(Real const eulerPos[NO_DIM+1][NO_DIM],
                           double relLo[NO_DIM], double relHi[NO_DIM],
                           int bLo[NO_DIM], int bHi[NO_DIM])
{
    for (int d = 0; d < NO_DIM; ++d)
    {
        double eMin = double(eulerPos[0][d]);
        double eMax = eMin;
        for (int v = 1; v <= NO_DIM; ++v)
        {
            double const c = double(eulerPos[v][d]);
            if (c < eMin) eMin = c;
            if (c > eMax) eMax = c;
        }
        double const margin = 1.e-5 * (eMax - eMin) + 1.e-12 + g_ctx.canon.epsAbs;
        relLo[d] = eMin - double(eulerPos[0][d]) - margin;
        relHi[d] = eMax - double(eulerPos[0][d]) + margin;

        bLo[d] = int( std::floor( (eMin - margin - g_ctx.boxLo[d]) * g_ctx.bInvW[d] ) );
        bHi[d] = int( std::floor( (eMax + margin - g_ctx.boxLo[d]) * g_ctx.bInvW[d] ) );
        if ( g_ctx.periodic )
        {
            if ( bHi[d] - bLo[d] + 1 >= g_ctx.nB[d] ) { bLo[d] = 0; bHi[d] = g_ctx.nB[d] - 1; }  // spans every bucket: visit each once
        }
        else
        {
            if ( bLo[d] < 0 ) bLo[d] = 0;
            if ( bHi[d] >= g_ctx.nB[d] ) bHi[d] = g_ctx.nB[d] - 1;
            if ( bLo[d] > bHi[d] ) return false;
        }
    }
    return true;
}


// ---- exact, tie-consistent containment (ps_exact_inside.h) --------------------------------

// Query point p in the canonical frame (its stored position is already wrapped into the box).
inline void canonicalQuery(uint32_t p, double pc[NO_DIM])
{
    canonicalQuery( g_ctx.canon, &g_ctx.pos[size_t(p)*NO_DIM], pc );
}


// Minimum-image displacement of query point p to (wrapped) vertex 0 and the rel-space bbox
// pre-test, then the containment decision in three tiers, each deciding only what it provably
// can: (1) the usual barycentrics on the wrapped geometry, outside the tetrahedron's fineBand;
// (2) a double-precision classification in the canonical frame, outside its own (much
// narrower) band; (3) the exact predicate. Every tier-1/2 decision equals the exact one, so
// the whole test IS the exact, tie-consistent one -- at the cost of the old tolerance test
// for all but the points hugging a face. The returned bary[] are the usual interpolation
// weights on the wrapped geometry (Ax stores edges as rows, so bary = inv^T * rel -- note the
// [i][v] transpose). Shared by the PS and standard workers.
inline bool pointInTet(uint32_t p, Real const eulerPos[NO_DIM+1][NO_DIM],
                       Real const rawPos[NO_DIM+1][NO_DIM], TetTest &T, double const band1,
                       double const inv[NO_DIM][NO_DIM],
                       double const relLo[NO_DIM], double const relHi[NO_DIM],
                       double bary[NO_DIM])
{
    double rel[NO_DIM];
    for (int d = 0; d < NO_DIM; ++d)
    {
        double r = g_ctx.pos[size_t(p)*NO_DIM+d] - double(eulerPos[0][d]);
        if ( g_ctx.periodic )
        {
            if (r >  0.5 * g_ctx.boxLen[d]) r -= g_ctx.boxLen[d];
            if (r < -0.5 * g_ctx.boxLen[d]) r += g_ctx.boxLen[d];
        }
        if ( r < relLo[d] || r > relHi[d] ) return false;
        rel[d] = r;
    }

    // tier 1: the wrapped-geometry barycentrics (also the interpolation weights)
    int const fine = fineClassify( inv, rel, band1, bary );
    if ( fine != 0 ) return fine > 0;

    // tiers 2 and 3 in the canonical frame (bary[] already holds the tier-1 weights)
    double pc[NO_DIM];
    canonicalQuery( p, pc );
    return canonicalInside( g_ctx.canon, rawPos, T, pc );
}


// Threads for one triangulation's point evaluation: every core when this is the only
// triangulation being processed; inside the partitioned PS loop (an OpenMP team of
// concurrent partitions), the cores left per partition -- so the two levels never
// oversubscribe. std::thread rather than a nested OpenMP region, which is disabled by default.
inline int pointEvalThreads()
{
    if ( const char *env = std::getenv("DTFE_PTS_THREADS") )   // explicit override (benchmarks, tests)
    {
        int const v = std::atoi(env);
        if ( v > 0 ) return v;
    }
    int cores = int( std::thread::hardware_concurrency() );
#ifdef OPEN_MP
    cores = omp_get_max_threads();
    if ( omp_in_parallel() )
        return std::max( 1, cores / std::max( 1, omp_get_num_threads() ) );
#endif
    return std::max( 1, cores );
}

// Candidate cells of one '--serve' request (psServe's cell index), or null: every finite cell.
static std::vector<Cell_handle> const *g_cellList = nullptr;

// Runs work(cell, hits) over every finite cell of dt -- or only over g_cellList when the server
// has narrowed a request to the cells near its points -- with nThreads threads. Walking dt, each
// thread takes every nThreads-th block of 256 cells, so no index array is materialized; each
// thread's HitSink hands its packed records to the shared store in blocks (record order is
// irrelevant: finalize sorts every point's records, so any order gives bit-identical outputs).
template <typename Work>
void forEachCellParallel(DT &dt, int nThreads, Work work)
{
    size_t const BLOCK = 256;
    if ( g_cellList )   // a small candidate list is not worth a thread per block
        nThreads = int( std::min<size_t>( size_t(std::max(1, nThreads)), g_cellList->size() / (8 * BLOCK) + 1 ) );
    size_t const nT = size_t( std::max(1, nThreads) );
    std::vector<HitSink> hits( nT );
    std::vector<std::exception_ptr> errors( nT );

    auto run = [&](int t)
    {
        try
        {
            if ( g_cellList )
            {
                std::vector<Cell_handle> const &list = *g_cellList;
                for (size_t b = size_t(t) * BLOCK; b < list.size(); b += nT * BLOCK)
                    for (size_t i = b; i < std::min(b + BLOCK, list.size()); ++i)
                    {
                        Cell_handle cell = list[i];
                        work( cell, hits[size_t(t)] );
                    }
                return;
            }
            size_t idx = 0;
            for (DT::Finite_cells_iterator itC = dt.finite_cells_begin(); itC != dt.finite_cells_end(); ++itC, ++idx)
            {
                if ( ( (idx / BLOCK) % nT ) != size_t(t) ) continue;
                Cell_handle cell = itC;
                work( cell, hits[size_t(t)] );
            }
        }
        catch (...) { errors[size_t(t)] = std::current_exception(); }
    };

    if ( nT <= 1 )
        run(0);
    else
    {
        std::vector<std::thread> pool;
        pool.reserve( nT - 1 );
        for (size_t t = 1; t < nT; ++t)
            pool.emplace_back( run, int(t) );
        run(0);
        for (std::thread &th : pool) th.join();
    }
    for (std::exception_ptr const &e : errors)
        if ( e ) std::rethrow_exception( e );

    for (HitSink &h : hits)                 // the last, partial blocks
        h.flush();
}


// Reads the sample-point file: text ("x y z" per line) or raw binary (float64, N x 3,
// row-major, native little-endian, no header), auto-detected from the leading bytes.
static void readSamplePoints(std::string const &filename, std::vector<double> &points)
{
    std::ifstream file( filename.c_str(), std::ios::binary );
    if ( not file )
        throwError( "Cannot open the '--sample-points' file '", filename, "'." );
    file.seekg( 0, std::ios::end );
    size_t const fileSize = size_t( file.tellg() );
    file.seekg( 0, std::ios::beg );
    if ( fileSize == 0 )
        throwError( "The '--sample-points' file '", filename, "' is empty." );

    char head[4096];
    size_t const headSize = std::min( fileSize, sizeof(head) );
    file.read( head, headSize );
    bool isText = true;
    for (size_t i = 0; i < headSize; ++i)
    {
        char const c = head[i];
        bool const numeric = (c >= '0' && c <= '9') || c == '+' || c == '-' || c == '.'
                             || c == 'e' || c == 'E' || c == ' ' || c == '\t'
                             || c == '\n' || c == '\r';
        if ( not numeric ) { isText = false; break; }
    }

    if ( isText )
    {
        file.clear();
        file.seekg( 0, std::ios::beg );
        double x, y, z;
        while ( file >> x )
        {
            if ( not (file >> y >> z) )
                throwError( "The '--sample-points' text file '", filename,
                            "' ends mid-triplet; every line must hold 'x y z'." );
            points.push_back( x );
            points.push_back( y );
            points.push_back( z );
        }
    }
    else
    {
        if ( fileSize % (NO_DIM * sizeof(double)) != 0 )
            throwError( "The '--sample-points' file '", filename,
                        "' is not text and its size is not a multiple of 24 bytes -- expected raw float64 x/y/z triplets (N x 3, no header)." );
        points.resize( fileSize / sizeof(double) );
        file.clear();
        file.seekg( 0, std::ios::beg );
        file.read( reinterpret_cast<char *>(points.data()), fileSize );
        if ( not file )
            throwError( "Failed reading the binary '--sample-points' file '", filename, "'." );
    }

    if ( points.empty() )
        throwError( "No sample points found in '", filename, "'." );
}

void setQueryPoints(std::vector<double> &&pts, int const bucketFloor);   // defined below

} // namespace psPointEvalDetail


using namespace psPointEvalDetail;


bool psPointEvalActive()
{
    return g_ctx.active;
}


void psPointEvalInit(User_options &userOptions)
{
    MESSAGE::Message message( userOptions.verboseLevel );

    g_ctx = Context();  // re-arm cleanly (defensive; DTFE() runs once per process)
    g_ctx.perStream    = userOptions.psPerStream;
    g_ctx.perStreamIds = userOptions.psPerStreamIds;
    g_ctx.ptsDenGrad   = userOptions.psPtsDenGrad;
    g_ctx.ptsVelGrad   = userOptions.psPtsVelGrad;
#ifdef PHASE_SPACE
    g_ctx.psCaustics   = userOptions.psCaustics;
    g_ctx.ptsLagrangian = userOptions.psPtsLagrangian;
#endif
    g_ctx.ptsScalar    = userOptions.psPtsScalar;
    g_ctx.useGeometric = userOptions.psStreamDensityGeometric;
    g_ctx.periodic     = userOptions.periodic;
#ifdef PHASE_SPACE
    g_ctx.rhoBar       = double( userOptions.averageDensity );   // PS vertex densities are physical
#else
    g_ctx.rhoBar       = 1.;   // standard vertexDensity() already normalizes by averageDensity
                               // (triangulation.cpp: factor = (NO_DIM+1)/averageDensity), so the
                               // interpolated values are rho/rho_bar as-is
#endif
    for (int d = 0; d < NO_DIM; ++d)
    {
        g_ctx.boxLo[d]  = double( userOptions.region[2*d] );
        g_ctx.boxLen[d] = double( userOptions.region[2*d+1] ) - double( userOptions.region[2*d] );
    }

    // Canonical-image frame of the exact containment test (ps_exact_inside.h): the box length
    // is formed with the SAME Real expression as every periodic-copy generator, so
    // canonicalCoord replays their exact float additions.
    g_ctx.canon.init( userOptions );

    // standard binary, single triangulation: a periodic run owns the primary-box image of each
    // boundary tetrahedron (a composite server sets each partition's own box per request)
    g_ctx.own.on = g_ctx.periodic;
    for (int d = 0; d < NO_DIM; ++d)
    {
        g_ctx.own.lo[d] = g_ctx.boxLo[d];
        g_ctx.own.hi[d] = g_ctx.boxLo[d] + g_ctx.boxLen[d];
    }

    g_ctx.active = true;
    if ( userOptions.psServe )
    {
        message << "\n" << MESSAGE::cBold() << PTS_MSG_BINARY << MESSAGE::cReset()
                << " point evaluation server armed (--serve): points arrive per request.\n" << MESSAGE::Flush;
        return;     // setQueryPoints() runs per request (psServe)
    }

    std::vector<double> pts;
    readSamplePoints( userOptions.psSamplePointsFile, pts );
    setQueryPoints( std::move(pts), 1 );

    message << "\n" << MESSAGE::cBold() << PTS_MSG_BINARY << MESSAGE::cReset()
            << " point evaluation at " << MESSAGE::cMagenta() << g_ctx.nPoints << MESSAGE::cReset()
            << " sample points from '" << MESSAGE::cBlue() << userOptions.psSamplePointsFile
            << MESSAGE::cReset() << "' (per-stream density: "
            << MESSAGE::cMagenta() << (g_ctx.useGeometric ? "geometric" : "dtfe") << MESSAGE::cReset()
            << (g_ctx.perStream ? ", writing per-stream records" : "")
            << (g_ctx.perStreamIds ? " + stream ids" : "")
            << (g_ctx.ptsDenGrad ? ", writing density gradients" : "")
            << (g_ctx.ptsVelGrad ? ", writing velocity gradients" : "")
            << (g_ctx.ptsScalar ? ", writing scalar values" : "")
            << (g_ctx.ptsLagrangian ? ", writing Lagrangian positions" : "") << ").\n" << MESSAGE::Flush;
}


namespace psPointEvalDetail {

// Installs a query point set: wraps it into the periodic box, buckets it and (re)arms the
// per-point records and outputs. 'bucketFloor' is the minimum buckets per axis -- the
// sparse-request case (--serve) wants buckets about tetrahedron-sized, so that tetrahedra far
// from every point are dropped by the cheap occupancy test instead of the full per-tet work.
void setQueryPoints(std::vector<double> &&pts, int const bucketFloor)
{
    g_ctx.pos = std::move(pts);
    g_ctx.nPoints = g_ctx.pos.size() / NO_DIM;
    if ( g_ctx.nPoints > size_t(0xFFFFFFFFu) )
        throwError( "Point evaluation supports at most 2^32-1 points per request." );

    // periodic box: wrap the query points inside; the per-tetrahedron test below uses the
    // minimum-image displacement to vertex 0, mirroring the grid deposit's index wrapping
    if ( g_ctx.periodic )
        for (size_t i = 0; i < g_ctx.nPoints; ++i)
            for (int d = 0; d < NO_DIM; ++d)
            {
                double v = std::fmod( g_ctx.pos[i*NO_DIM+d] - g_ctx.boxLo[d], g_ctx.boxLen[d] );
                if ( v < 0. ) v += g_ctx.boxLen[d];
                g_ctx.pos[i*NO_DIM+d] = g_ctx.boxLo[d] + v;
            }

    // bucket the points on a uniform grid (~4 points per bucket, capped so empty-bucket
    // overhead stays negligible for small point sets)
    {
        int n = int( std::cbrt( double(g_ctx.nPoints) / 4. ) ) + 1;
        if ( n < bucketFloor ) n = bucketFloor;
        if ( n < 1 )   n = 1;
        if ( n > 256 ) n = 256;
        size_t nTotal = 1;
        for (int d = 0; d < NO_DIM; ++d)
        {
            g_ctx.nB[d]    = n;
            g_ctx.bInvW[d] = double(n) / g_ctx.boxLen[d];
            nTotal *= size_t(n);
        }

        std::vector<size_t> count( nTotal, 0 );
        auto bucketOf = [&](size_t i) -> size_t
        {
            size_t b = 0;
            for (int d = 0; d < NO_DIM; ++d)
            {
                int c = int( std::floor( (g_ctx.pos[i*NO_DIM+d] - g_ctx.boxLo[d]) * g_ctx.bInvW[d] ) );
                if ( c < 0 ) c = 0;                             // non-periodic points may lie outside the box
                if ( c >= g_ctx.nB[d] ) c = g_ctx.nB[d] - 1;
                b = b * size_t(g_ctx.nB[d]) + size_t(c);
            }
            return b;
        };
        for (size_t i = 0; i < g_ctx.nPoints; ++i)
            count[ bucketOf(i) ]++;
        g_ctx.bucketStart.assign( nTotal + 1, 0 );
        for (size_t b = 0; b < nTotal; ++b)
            g_ctx.bucketStart[b+1] = g_ctx.bucketStart[b] + count[b];
        g_ctx.bucketPts.resize( g_ctx.nPoints );
        std::vector<size_t> fill( g_ctx.bucketStart.begin(), g_ctx.bucketStart.end() - 1 );
        for (size_t i = 0; i < g_ctx.nPoints; ++i)
            g_ctx.bucketPts[ fill[ bucketOf(i) ]++ ] = uint32_t(i);
    }

    // fresh records and outputs (finalize fills the out* arrays; the ragged ones append)
    std::vector< std::vector<unsigned char> >().swap( g_ctx.recBlocks );
    g_ctx.nRecs = 0;
    if ( std::getenv("DTFE_PTS_FULL_RECORDS") )     // test switch: store every member, as before
        g_ctx.layout.init( true, true, true, true, true );   // packing (tests/ps_point_eval_check.sh compares)
    else
        g_ctx.layout.init( g_ctx.ptsDenGrad, g_ctx.ptsVelGrad, g_ctx.perStreamIds, g_ctx.ptsScalar, g_ctx.ptsLagrangian );
    g_ctx.outRecords.clear();
    g_ctx.outIds.clear();
    g_ctx.outRecGrad.clear();
    g_ctx.outRecScalar.clear();
    g_ctx.outRecLag.clear();
    g_ctx.outOffsets.clear();
    g_ctx.finalized = false;
}

// True when any query point lies in the bucket range [bLo, bHi] (periodic ranges wrap). The
// cheap test that lets a tetrahedron with no point anywhere near it skip all per-tet work.
inline bool anyPointInBuckets(int const bLo[NO_DIM], int const bHi[NO_DIM])
{
    for (int bi = bLo[0]; bi <= bHi[0]; ++bi)
    for (int bj = bLo[1]; bj <= bHi[1]; ++bj)
    for (int bk = bLo[2]; bk <= bHi[2]; ++bk)
    {
        int const wi = g_ctx.periodic ? ((bi % g_ctx.nB[0] + g_ctx.nB[0]) % g_ctx.nB[0]) : bi;
        int const wj = g_ctx.periodic ? ((bj % g_ctx.nB[1] + g_ctx.nB[1]) % g_ctx.nB[1]) : bj;
        int const wk = g_ctx.periodic ? ((bk % g_ctx.nB[2] + g_ctx.nB[2]) % g_ctx.nB[2]) : bk;
        size_t const b = ( size_t(wi) * size_t(g_ctx.nB[1]) + size_t(wj) ) * size_t(g_ctx.nB[2]) + size_t(wk);
        if ( g_ctx.bucketStart[b+1] > g_ctx.bucketStart[b] ) return true;
    }
    return false;
}

} // namespace psPointEvalDetail


#ifdef PHASE_SPACE

/* Collects this partition's stream contributions for every query point. Called once per
   Lagrangian partition (and once for a single-triangulation run) after vertexDensity().
   The tessellation is read-only here; the cells are split over pointEvalThreads() threads,
   and concurrent workers only synchronize on the short record merge at the end. */
void interpolatePoints_phaseSpace(DT &dt, User_options &userOptions)
{
    if ( not g_ctx.active || dt.number_of_vertices() < NO_DIM + 1 )
        return;

    Box boxCoordinates = userOptions.region;
    double const avgDens = double( userOptions.averageDensity );

    forEachCellParallel( dt, pointEvalThreads(),
                         [&](Cell_handle &cell, HitSink &hits)
    {
        // The CPU grid deposit's filter chain, shared via ps_cell_filter.h (Real/double
        // arithmetic identical to the grid path, so borderline cells classify identically).
        PSCellGeometry geo;
        if ( !psFilterCell(cell, userOptions, boxCoordinates, true, NULL, geo) )
            return;
        bool const useVolumeRatioDensity = geo.useVolumeRatioDensity;
        Real (&eulerPos)[NO_DIM+1][NO_DIM] = geo.eulerPos;
        double (&Ax)[NO_DIM][NO_DIM] = geo.Ax;
        double const cellAbsDet = geo.cellAbsDet;

        // Eulerian bounding box (+ exact-test slack) -> bucket range (shared helper); a
        // tetrahedron with no query point in reach skips everything below
        double relLo[NO_DIM], relHi[NO_DIM];
        int bLo[NO_DIM], bHi[NO_DIM];
        if ( not tetBucketRange(eulerPos, relLo, relHi, bLo, bHi) ) { return; }
        if ( not anyPointInBuckets(bLo, bHi) ) { return; }

        // double-precision inverse for the evaluation (same zero criterion as above)
        double inv[NO_DIM][NO_DIM];
        if ( not inverse3x3d(Ax, inv) ) { return; }

        // 'geometric' density: rho_bar * V_lag / V_eul (= m_tet * d! / |det Ax|), constant per tet
        double denGeo;
        // --ps-caustics: the same per-tetrahedron mask the grid deposit accumulates (fold parity
        // plus the collapse multiplicity from the constant deformation tensor), carried per stream
        // so the reduction can OR the masks of every tetrahedron containing the point.
        PSCausticClass::CausticMask tetCaustic = 0;
        {
            double Lag[NO_DIM][NO_DIM];
            for (int v = 0; v < NO_DIM; ++v)
                for (int i = 0; i < NO_DIM; ++i)
                    Lag[v][i] = double(cell->vertex(v+1)->point()[i]) - double(cell->vertex(0)->point()[i]);
            denGeo = avgDens * std::fabs(determinant(Lag)) / cellAbsDet;
            if (denGeo < 0.) denGeo = 0.;
            if ( g_ctx.psCaustics )
                tetCaustic = (PSCausticClass::CausticMask)
                             ( (geo.cellDet > 0. ? PSCausticClass::BIT_PARITY_POS
                                                 : PSCausticClass::BIT_PARITY_NEG)
                               | PSCausticClass::classifyTet( geo.Ax, Lag ) );
        }

        double rho[NO_DIM+1], vel[NO_DIM+1][noVelComp];
        for (int v = 0; v <= NO_DIM; ++v)
        {
            rho[v] = double( cell->vertex(v)->info().density() );
            for (size_t j = 0; j < noVelComp; ++j)
                vel[v][j] = double( cell->vertex(v)->info().velocity(j) );
        }
        // --pts-scalar: the vertices' scalar values; --pts-lagrangian: their Lagrangian positions
        // (the triangulation's own points). A tetrahedron is compact in Lagrangian space, so the
        // differences need no wrap; only the interpolated q is wrapped into the box.
        double scal[NO_DIM+1][noScalarComp], lagV[NO_DIM+1][NO_DIM];
        if ( g_ctx.ptsScalar )
            for (int v = 0; v <= NO_DIM; ++v)
                for (size_t j = 0; j < noScalarComp; ++j)
                    scal[v][j] = double( cell->vertex(v)->info().scalar(int(j)) );
        if ( g_ctx.ptsLagrangian )
            for (int v = 0; v <= NO_DIM; ++v)
                for (int d = 0; d < NO_DIM; ++d)
                    lagV[v][d] = double( cell->vertex(v)->point()[d] );

        // stored (unwrapped) Eulerian vertex positions: the exact containment test canonicalizes
        // these, so a periodic copy and its original resolve to the same vertex
        Real rawPos[NO_DIM+1][NO_DIM];
        for (int v = 0; v <= NO_DIM; ++v)
            for (int d = 0; d < NO_DIM; ++d)
                rawPos[v][d] = cell->vertex(v)->info().eulerianPosition(d);
        TetTest tt;
        double const band1 = fineBand( g_ctx.canon, Ax, cellAbsDet );

        StreamRec rec;
        rec.denGeo = denGeo;
        rec.caustic = tetCaustic;
        for (size_t j = 0; j < noScalarComp; ++j)
            rec.scal[j] = 0.;
        for (int d = 0; d < NO_DIM; ++d)
            rec.lag[d] = 0.;
        for (int v = 0; v <= NO_DIM; ++v)
            rec.ids[v] = 0;
        if ( g_ctx.perStreamIds )
        {
            // sorted ascending: the quadruple is orientation-independent, and identical for the
            // original cell and any periodic image (copies keep the original's ParticleID)
            for (int v = 0; v <= NO_DIM; ++v)
                rec.ids[v] = cell->vertex(v)->info().particleID();
            std::sort( rec.ids, rec.ids + NO_DIM + 1 );
        }
        for (int i = 0; i < NO_DIM; ++i)
            rec.grad[i] = 0.;
        if ( g_ctx.ptsDenGrad && !useVolumeRatioDensity )
        {
            // constant 'dtfe' gradient of this stream: rho(p) = rho_0 + grad . (p - x_0) with
            // bary = (Ax^-1)^T rel  =>  grad_i = sum_v inv[i][v] (rho_{v+1} - rho_0) -- the same
            // affine convention as the linear deposit's denGrad (ps_interpolation.cc). Hull
            // cells (volume-ratio density) have a constant profile: gradient stays 0.
            for (int i = 0; i < NO_DIM; ++i)
                for (int v = 0; v < NO_DIM; ++v)
                    rec.grad[i] += inv[i][v] * (rho[v+1] - rho[0]);
        }
        for (int q = 0; q < NO_DIM*NO_DIM; ++q)
            rec.vgrad[q] = 0.;
        if ( g_ctx.ptsVelGrad )
        {
            // constant velocity gradient of this stream: dv_j/dx_d = sum_v inv[d][v]
            // (vel_{v+1,j} - vel_{0,j}), same affine convention as the density gradient and
            // the SAME [d*3+j] layout as the grid deposit's velGrad (ps_interpolation.cc
            // matrixMultiplication(posMatInv, dvel)). The velocity profile is linear on every
            // kept cell -- hull cells included -- so there is no volume-ratio gate here.
            for (int d = 0; d < NO_DIM; ++d)
                for (size_t j = 0; j < noVelComp; ++j)
                    for (int v = 0; v < NO_DIM; ++v)
                        rec.vgrad[d*NO_DIM + j] += inv[d][v] * (vel[v+1][j] - vel[0][j]);
        }

        for (int bi = bLo[0]; bi <= bHi[0]; ++bi)
        for (int bj = bLo[1]; bj <= bHi[1]; ++bj)
        for (int bk = bLo[2]; bk <= bHi[2]; ++bk)
        {
            int const wi = g_ctx.periodic ? ((bi % g_ctx.nB[0] + g_ctx.nB[0]) % g_ctx.nB[0]) : bi;
            int const wj = g_ctx.periodic ? ((bj % g_ctx.nB[1] + g_ctx.nB[1]) % g_ctx.nB[1]) : bj;
            int const wk = g_ctx.periodic ? ((bk % g_ctx.nB[2] + g_ctx.nB[2]) % g_ctx.nB[2]) : bk;
            size_t const b = ( size_t(wi) * size_t(g_ctx.nB[1]) + size_t(wj) ) * size_t(g_ctx.nB[2]) + size_t(wk);

            for (size_t s = g_ctx.bucketStart[b]; s < g_ctx.bucketStart[b+1]; ++s)
            {
                uint32_t const p = g_ctx.bucketPts[s];

                // min-image displacement + bbox pre-test + exact containment (shared helper)
                double bary[NO_DIM];
                if ( not pointInTet(p, eulerPos, rawPos, tt, band1, inv, relLo, relHi, bary) ) continue;

                if ( useVolumeRatioDensity )
                    rec.denDtfe = denGeo;   // hull cell: vertex densities are undefined (0)
                else
                {
                    double dd = rho[0];
                    for (int v = 0; v < NO_DIM; ++v)
                        dd += bary[v] * (rho[v+1] - rho[0]);
                    rec.denDtfe = (dd > 0.) ? dd : 0.;   // a point ON a face can graze negative by rounding
                }
                for (size_t j = 0; j < noVelComp; ++j)
                {
                    double vv = vel[0][j];
                    for (int v = 0; v < NO_DIM; ++v)
                        vv += bary[v] * (vel[v+1][j] - vel[0][j]);
                    rec.vel[j] = vv;
                }
                // the scalar and the Lagrangian coordinate follow the SAME affine map as the
                // velocity: q(x) = q_0 + sum_v bary_v (q_{v+1} - q_0) inverts x(q) on this stream
                if ( g_ctx.ptsScalar )
                    for (size_t j = 0; j < noScalarComp; ++j)
                    {
                        double ss = scal[0][j];
                        for (int v = 0; v < NO_DIM; ++v)
                            ss += bary[v] * (scal[v+1][j] - scal[0][j]);
                        rec.scal[j] = ss;
                    }
                if ( g_ctx.ptsLagrangian )
                    for (int d = 0; d < NO_DIM; ++d)
                    {
                        double qq = lagV[0][d];
                        for (int v = 0; v < NO_DIM; ++v)
                            qq += bary[v] * (lagV[v+1][d] - lagV[0][d]);
                        if ( g_ctx.periodic )
                        {
                            double w = std::fmod( qq - g_ctx.boxLo[d], g_ctx.boxLen[d] );
                            if ( w < 0. ) w += g_ctx.boxLen[d];
                            if ( w >= g_ctx.boxLen[d] ) w -= g_ctx.boxLen[d];
                            qq = g_ctx.boxLo[d] + w;
                        }
                        rec.lag[d] = qq;
                    }

                hits.push_back( std::make_pair( p, rec ) );
            }
        }
    } );   // end cell loop
}

#else // !PHASE_SPACE

/* Standard-DTFE worker: the tessellation itself is Eulerian, so a query point has exactly
   one containing tetrahedron (stream count = 0/1 coverage; the exact containment test makes
   that hold for points ON shared faces too, and psPointEvalFinalize keeps a 1-record cap
   as a safety net). Reuses the PS candidate search verbatim (bucketed points,
   tetBucketRange/pointInTet);
   the estimator differences:
     (a) vertex positions come from the Delaunay points themselves, min-image-wrapped to
         vertex 0 under --periodic exactly like the PS Eulerian wrap (ps_cell_filter.h);
     (b) --periodic pads the box with particle COPIES, so a boundary tetrahedron appears as
         several images: exactly ONE is kept (Eulerian centroid inside the primary box) --
         the standard-build analogue of the PS Lagrangian-centroid partition ownership;
     (c) there is no geometric (V_lag/V_eul) density variant: denGeo aliases the 'dtfe'
         value, so the shared sort/reduction machinery behaves identically. */
// The standard worker's per-cell filter, shared with the composite server's occupancy pass:
// the vertex densities/velocities (false when one is not positive -- a degenerate hull
// configuration), the Eulerian vertex positions min-image wrapped to vertex 0 (eulerPos) and as
// stored (rawPos, for the exact test's canonical frame), and ownership: a periodic run holds a
// boundary tetrahedron once per image, a composite partition also its neighbours' tetrahedra in
// its padding, and exactly ONE copy -- the one whose centroid lies in 'own' -- is kept, so a point
// near a boundary receives exactly one record from each physical tetrahedron.
inline bool standardCell(Cell_handle &cell, OwnBox const &own, double rho[NO_DIM+1],
                         double vel[NO_DIM+1][noVelComp], Real eulerPos[NO_DIM+1][NO_DIM],
                         Real rawPos[NO_DIM+1][NO_DIM])
{
#ifdef TEST_PADDING
    // skip cells touching a dummy padding-test vertex
    for (int v = 0; v <= NO_DIM; ++v)
        if (cell->vertex(v)->info().isDummy()) return false;
#endif

    // vertex DTFE densities must be positive to interpolate the linear profile
    for (int v = 0; v <= NO_DIM; ++v)
    {
        rho[v] = double( cell->vertex(v)->info().density() );
        if ( rho[v] <= 0. ) return false;
        for (size_t j = 0; j < noVelComp; ++j)
            vel[v][j] = double( cell->vertex(v)->info().velocity(j) );
    }

    for (int v = 0; v <= NO_DIM; ++v)
        for (int d = 0; d < NO_DIM; ++d)
            rawPos[v][d] = eulerPos[v][d] = Real( cell->vertex(v)->point()[d] );
    if ( g_ctx.periodic )
        for (int v = 1; v <= NO_DIM; ++v)
            for (int d = 0; d < NO_DIM; ++d)
            {
                Real const diff = eulerPos[v][d] - eulerPos[0][d];
                if (diff >  Real(0.5) * Real(g_ctx.boxLen[d])) eulerPos[v][d] -= Real(g_ctx.boxLen[d]);
                if (diff < -Real(0.5) * Real(g_ctx.boxLen[d])) eulerPos[v][d] += Real(g_ctx.boxLen[d]);
            }
    if ( own.on )
    {
        double cen[NO_DIM];
        for (int d = 0; d < NO_DIM; ++d)
        {
            cen[d] = 0.;
            for (int v = 0; v <= NO_DIM; ++v) cen[d] += double(eulerPos[v][d]);
            cen[d] /= double(NO_DIM + 1);
        }
        if ( not own.contains(cen) ) return false;
    }
    return true;
}

void interpolatePoints_standardOwn(DT &dt, User_options &userOptions, OwnBox const &own)
{
    if ( not g_ctx.active || dt.number_of_vertices() < NO_DIM + 1 )
        return;

    (void)userOptions;   // box geometry/periodicity already captured in g_ctx at init

    forEachCellParallel( dt, pointEvalThreads(),
                         [&](Cell_handle &cell, HitSink &hits)
    {
        double rho[NO_DIM+1], vel[NO_DIM+1][noVelComp];
        Real eulerPos[NO_DIM+1][NO_DIM], rawPos[NO_DIM+1][NO_DIM];
        if ( not standardCell( cell, own, rho, vel, eulerPos, rawPos ) ) { return; }

        // Eulerian edge matrix. Unlike the PS worker (whose stream set must equal the grid
        // deposit's, degeneracy drops included), NO tetrahedron is dropped for being small or
        // flat: the Eulerian tessellation tiles the box, so every drop is a hole -- a point in
        // it, or a query AT a vertex whose tie-break lands in it, would come out uncovered.
        // Containment is decided exactly regardless of shape (fineBand grows as the tet
        // flattens, handing the decision to the exact tiers), and a guard-free inverse gives
        // the interpolation weights; only an exactly zero-volume cell holds no points.
        double relLo[NO_DIM], relHi[NO_DIM];
        int bLo[NO_DIM], bHi[NO_DIM];
        if ( not tetBucketRange(eulerPos, relLo, relHi, bLo, bHi) ) { return; }
        if ( not anyPointInBuckets(bLo, bHi) ) { return; }   // no query point in reach

        double Ax[NO_DIM][NO_DIM];
        for (int v = 0; v < NO_DIM; ++v)
            for (int i = 0; i < NO_DIM; ++i)
                Ax[v][i] = double(eulerPos[v+1][i]) - double(eulerPos[0][i]);
        double inv[NO_DIM][NO_DIM];
        double const absDet = std::fabs( inverse3x3Unguarded(Ax, inv) );
        if ( not (absDet > 0.) || not std::isfinite(absDet) ) { return; }
        TetTest tt;
        double const band1 = fineBand( g_ctx.canon, Ax, absDet );

        StreamRec rec;
        rec.caustic = 0;                    // --ps-caustics is PS-only (no fold parity to report here)
        for (int v = 0; v <= NO_DIM; ++v)
            rec.ids[v] = 0;                 // --per-stream-ids is PS-only (rejected at parsing)
        for (int d = 0; d < NO_DIM; ++d)
            rec.lag[d] = 0.;                // --pts-lagrangian is PS-only
        for (size_t j = 0; j < noScalarComp; ++j)
            rec.scal[j] = 0.;
        double scal[NO_DIM+1][noScalarComp];
        if ( g_ctx.ptsScalar )
            for (int v = 0; v <= NO_DIM; ++v)
                for (size_t j = 0; j < noScalarComp; ++j)
                    scal[v][j] = double( cell->vertex(v)->info().scalar(int(j)) );
        for (int i = 0; i < NO_DIM; ++i)
            rec.grad[i] = 0.;
        if ( g_ctx.ptsDenGrad )
        {
            // constant gradient of the linear DTFE profile, same affine convention as PS:
            // rho(p) = rho_0 + grad . (p - x_0), grad_i = sum_v inv[i][v] (rho_{v+1} - rho_0)
            for (int i = 0; i < NO_DIM; ++i)
                for (int v = 0; v < NO_DIM; ++v)
                    rec.grad[i] += inv[i][v] * (rho[v+1] - rho[0]);
        }
        for (int q = 0; q < NO_DIM*NO_DIM; ++q)
            rec.vgrad[q] = 0.;
        if ( g_ctx.ptsVelGrad )
        {
            // the containing tetrahedron's constant velocity gradient, same [d*3+j] layout
            // as the PS worker (exactly one stream per point here, so the 'mean' is it)
            for (int d = 0; d < NO_DIM; ++d)
                for (size_t j = 0; j < noVelComp; ++j)
                    for (int v = 0; v < NO_DIM; ++v)
                        rec.vgrad[d*NO_DIM + j] += inv[d][v] * (vel[v+1][j] - vel[0][j]);
        }

        for (int bi = bLo[0]; bi <= bHi[0]; ++bi)
        for (int bj = bLo[1]; bj <= bHi[1]; ++bj)
        for (int bk = bLo[2]; bk <= bHi[2]; ++bk)
        {
            int const wi = g_ctx.periodic ? ((bi % g_ctx.nB[0] + g_ctx.nB[0]) % g_ctx.nB[0]) : bi;
            int const wj = g_ctx.periodic ? ((bj % g_ctx.nB[1] + g_ctx.nB[1]) % g_ctx.nB[1]) : bj;
            int const wk = g_ctx.periodic ? ((bk % g_ctx.nB[2] + g_ctx.nB[2]) % g_ctx.nB[2]) : bk;
            size_t const b = ( size_t(wi) * size_t(g_ctx.nB[1]) + size_t(wj) ) * size_t(g_ctx.nB[2]) + size_t(wk);

            for (size_t s = g_ctx.bucketStart[b]; s < g_ctx.bucketStart[b+1]; ++s)
            {
                uint32_t const p = g_ctx.bucketPts[s];

                double bary[NO_DIM];
                if ( not pointInTet(p, eulerPos, rawPos, tt, band1, inv, relLo, relHi, bary) ) continue;

                double dd = rho[0];
                for (int v = 0; v < NO_DIM; ++v)
                    dd += bary[v] * (rho[v+1] - rho[0]);
                rec.denDtfe = (dd > 0.) ? dd : 0.;   // a point ON a face can graze negative by rounding
                rec.denGeo  = rec.denDtfe;           // no geometric variant on the Eulerian tessellation
                for (size_t j = 0; j < noVelComp; ++j)
                {
                    double vv = vel[0][j];
                    for (int v = 0; v < NO_DIM; ++v)
                        vv += bary[v] * (vel[v+1][j] - vel[0][j]);
                    rec.vel[j] = vv;
                }
                if ( g_ctx.ptsScalar )
                    for (size_t j = 0; j < noScalarComp; ++j)
                    {
                        double ss = scal[0][j];
                        for (int v = 0; v < NO_DIM; ++v)
                            ss += bary[v] * (scal[v+1][j] - scal[0][j]);
                        rec.scal[j] = ss;
                    }

                hits.push_back( std::make_pair( p, rec ) );
            }
        }
    } );   // end cell loop
}

void interpolatePoints_standard(DT &dt, User_options &userOptions)
{
    interpolatePoints_standardOwn( dt, userOptions, g_ctx.own );
}

#endif // PHASE_SPACE


void psPointEvalFinalize(User_options const &userOptions)
{
    if ( not g_ctx.active || g_ctx.finalized )
        return;

    MESSAGE::Message message( userOptions.verboseLevel );
    size_t const N = g_ctx.nPoints;
    bool const geo = g_ctx.useGeometric;
    double const invRhoBar = g_ctx.rhoBar > 0. ? 1. / g_ctx.rhoBar : 1.;

    // strict total order (selected density descending, full tie-break) -> the outputs are
    // deterministic and independent of the partition split and thread completion order
    auto recLess = [geo](StreamRec const &a, StreamRec const &b) -> bool
    {
        double const a1 = geo ? a.denGeo : a.denDtfe;
        double const b1 = geo ? b.denGeo : b.denDtfe;
        if (a1 != b1) return a1 > b1;
        if (a.denGeo  != b.denGeo)  return a.denGeo  > b.denGeo;
        if (a.denDtfe != b.denDtfe) return a.denDtfe > b.denDtfe;
        for (size_t j = 0; j < noVelComp; ++j)
            if (a.vel[j] != b.vel[j]) return a.vel[j] < b.vel[j];
        for (int i = 0; i < NO_DIM; ++i)        // all 0 unless --pts-den-grad
            if (a.grad[i] != b.grad[i]) return a.grad[i] < b.grad[i];
        for (int q = 0; q < NO_DIM*NO_DIM; ++q) // all 0 unless --pts-vel-grad
            if (a.vgrad[q] != b.vgrad[q]) return a.vgrad[q] < b.vgrad[q];
        for (int v = 0; v <= NO_DIM; ++v)       // all 0 unless --per-stream-ids; only breaks exact
            if (a.ids[v] != b.ids[v]) return a.ids[v] < b.ids[v];   // den+vel ties, so the ids file
                                                                    // is partition-invariant too
        for (size_t j = 0; j < noScalarComp; ++j)   // all 0 unless --pts-scalar
            if (a.scal[j] != b.scal[j]) return a.scal[j] < b.scal[j];
        for (int d = 0; d < NO_DIM; ++d)             // all 0 unless --pts-lagrangian
            if (a.lag[d] != b.lag[d]) return a.lag[d] < b.lag[d];
        return false;
    };

    g_ctx.outDen.assign( N, 0. );
    g_ctx.outVel.assign( N * noVelComp, 0. );
    g_ctx.outDisp.assign( N * noDispComp, 0. );
    g_ctx.outStreams.assign( N, 0 );
    if ( g_ctx.psCaustics )
        g_ctx.outCaustic.assign( N, 0 );
    size_t const totalStreams = g_ctx.nRecs;

    // Group the stored entries by point: a counting sort into 'first' (N+1 offsets) and 'where'
    // (one locator per record: block << 32 | entry within the block). The order of a point's
    // records here is the arrival order; the strict sort below makes the output independent of it.
    size_t const E = sizeof(uint32_t) + g_ctx.layout.bytes;
    std::vector<uint64_t> first( N + 1, 0 ), where( totalStreams );
    for (std::vector<unsigned char> const &blk : g_ctx.recBlocks)
        for (size_t k = 0; k < blk.size() / E; ++k)
        {
            uint32_t pt;
            std::memcpy( &pt, blk.data() + k * E, sizeof(uint32_t) );
            ++first[ size_t(pt) + 1 ];
        }
    for (size_t i = 0; i < N; ++i) first[i+1] += first[i];
    {
        std::vector<uint64_t> next( first.begin(), first.end() - 1 );
        for (size_t b = 0; b < g_ctx.recBlocks.size(); ++b)
        {
            std::vector<unsigned char> const &blk = g_ctx.recBlocks[b];
            for (size_t k = 0; k < blk.size() / E; ++k)
            {
                uint32_t pt;
                std::memcpy( &pt, blk.data() + k * E, sizeof(uint32_t) );
                where[ next[pt]++ ] = ( uint64_t(b) << 32 ) | uint64_t(k);
            }
        }
    }
    if ( g_ctx.perStream )
    {
        g_ctx.outOffsets.assign( N + 1, 0 );
        g_ctx.outRecords.reserve( totalStreams * (1 + noVelComp) );
        if ( g_ctx.perStreamIds )
            g_ctx.outIds.reserve( totalStreams * (NO_DIM + 1) );
        if ( g_ctx.ptsDenGrad )
            g_ctx.outRecGrad.reserve( totalStreams * NO_DIM );
        if ( g_ctx.ptsScalar )
            g_ctx.outRecScalar.reserve( totalStreams * noScalarComp );
        if ( g_ctx.ptsLagrangian )
            g_ctx.outRecLag.reserve( totalStreams * NO_DIM );
    }
    if ( g_ctx.ptsScalar )
        g_ctx.outScalar.assign( N * noScalarComp, 0. );
    if ( g_ctx.ptsDenGrad )
        g_ctx.outDenGrad.assign( N * NO_DIM, 0. );
    if ( g_ctx.ptsVelGrad )
        g_ctx.outVelGrad.assign( N * NO_DIM*NO_DIM, 0. );

    size_t covered = 0, multi = 0;
    size_t maxStreams = 0;
    std::vector<StreamRec> recs;            // one point's records, unpacked (the buffer is reused)
    for (size_t i = 0; i < N; ++i)
    {
        recs.resize( size_t( first[i+1] - first[i] ) );
        for (uint64_t r = first[i]; r < first[i+1]; ++r)
        {
            uint64_t const loc = where[r];
            g_ctx.layout.unpackOne( g_ctx.recBlocks[ size_t(loc >> 32) ].data() + (loc & 0xFFFFFFFFu) * E
                                    + sizeof(uint32_t), recs[ size_t(r - first[i]) ] );
        }
        std::sort( recs.begin(), recs.end(), recLess );

#ifndef PHASE_SPACE
        // Eulerian tessellation: exactly one containing tet per point, which the exact
        // containment test guarantees even ON shared faces. Kept as a safety net: should a
        // second record ever appear, both interpolate the same continuous field, so keep the
        // deterministic sorted-first one and the coverage count stays an honest 0/1.
        if ( recs.size() > 1 )
            recs.resize( 1 );
#endif

        size_t const nS = recs.size();
        g_ctx.outStreams[i] = int32_t( nS );
        // --ps-caustics: OR the masks of every tetrahedron covering this point. OR is commutative
        // and idempotent, so the result is independent of the partition and thread schedule -- the
        // same guarantee the grid '.caustic' has, and the reason a point ON a fold reports both
        // parities exactly as the enclosing grid cell does.
        if ( g_ctx.psCaustics )
        {
            int mask = 0;
            for (size_t s = 0; s < nS; ++s) mask |= int( recs[s].caustic );
            g_ctx.outCaustic[i] = int32_t( mask );
        }
        if (nS > 0) ++covered;
        if (nS > 1) ++multi;
        if (nS > maxStreams) maxStreams = nS;

        // moments in sorted order (the psDeferNormalization pattern: sums are linear across
        // partitions; the density-weighted means/dispersion are formed exactly once, here)
        double W = 0., mom1[noVelComp] = {0.}, mom2[noDispComp] = {0.};
        double momG[NO_DIM*NO_DIM] = {0.};
        double momS[noScalarComp] = {0.};
        for (size_t s = 0; s < nS; ++s)
        {
            double const den = geo ? recs[s].denGeo : recs[s].denDtfe;
            W += den;
            for (size_t j = 0; j < noVelComp; ++j)
                mom1[j] += den * recs[s].vel[j];
            if ( g_ctx.ptsScalar )
                for (size_t j = 0; j < noScalarComp; ++j)
                    momS[j] += den * recs[s].scal[j];
            if ( g_ctx.ptsVelGrad )
                for (int q = 0; q < NO_DIM*NO_DIM; ++q)
                    momG[q] += den * recs[s].vgrad[q];
            size_t c = 0;
            for (int a = 0; a < NO_DIM; ++a)
                for (int b = a; b < NO_DIM; ++b)
                    mom2[c++] += den * recs[s].vel[a] * recs[s].vel[b];
        }
        g_ctx.outDen[i] = W * invRhoBar;
        if ( g_ctx.ptsDenGrad )
        {
            // gradient of the total ('dtfe') density: sum of the streams' constant gradients,
            // accumulated in the same sorted order as the moments (deterministic across splits)
            double gsum[NO_DIM] = {0.};
            for (size_t s = 0; s < nS; ++s)
                for (int d = 0; d < NO_DIM; ++d)
                    gsum[d] += recs[s].grad[d];
            for (int d = 0; d < NO_DIM; ++d)
                g_ctx.outDenGrad[i*NO_DIM + d] = gsum[d] * invRhoBar;
        }
        if ( W > 0. )
        {
            double const invW = 1. / W;
            for (size_t j = 0; j < noVelComp; ++j)
                g_ctx.outVel[i*noVelComp+j] = mom1[j] * invW;
            if ( g_ctx.ptsVelGrad )
                for (int q = 0; q < NO_DIM*NO_DIM; ++q)
                    g_ctx.outVelGrad[i*NO_DIM*NO_DIM + q] = momG[q] * invW;
            // the scalar: the same density-weighted stream mean as the velocity
            if ( g_ctx.ptsScalar )
                for (size_t j = 0; j < noScalarComp; ++j)
                    g_ctx.outScalar[i*noScalarComp + j] = momS[j] * invW;
            // a single stream has zero dispersion by definition; skip the moment difference
            // (it would only return ~1e-9 cancellation noise)
            if ( nS > 1 )
            {
                size_t c = 0;
                for (int a = 0; a < NO_DIM; ++a)
                    for (int b = a; b < NO_DIM; ++b)
                    {
                        double s2 = mom2[c] * invW
                                  - g_ctx.outVel[i*noVelComp+a] * g_ctx.outVel[i*noVelComp+b];
                        if (a == b && s2 < 0.) s2 = 0.;   // variance: clamp FP-noise negatives
                        g_ctx.outDisp[i*noDispComp + c] = s2;
                        ++c;
                    }
            }
        }

        if ( g_ctx.perStream )
        {
            for (size_t s = 0; s < nS; ++s)
            {
                g_ctx.outRecords.push_back( (geo ? recs[s].denGeo : recs[s].denDtfe) * invRhoBar );
                for (size_t j = 0; j < noVelComp; ++j)
                    g_ctx.outRecords.push_back( recs[s].vel[j] );
                if ( g_ctx.perStreamIds )
                    for (int v = 0; v <= NO_DIM; ++v)
                        g_ctx.outIds.push_back( recs[s].ids[v] );
                if ( g_ctx.ptsDenGrad )
                    for (int d = 0; d < NO_DIM; ++d)
                        g_ctx.outRecGrad.push_back( recs[s].grad[d] * invRhoBar );
                if ( g_ctx.ptsScalar )
                    for (size_t j = 0; j < noScalarComp; ++j)
                        g_ctx.outRecScalar.push_back( recs[s].scal[j] );
                if ( g_ctx.ptsLagrangian )
                    for (int d = 0; d < NO_DIM; ++d)
                        g_ctx.outRecLag.push_back( recs[s].lag[d] );
            }
            g_ctx.outOffsets[i+1] = uint64_t( g_ctx.outOffsets[i] ) + uint64_t( nS );
        }

    }
    std::vector< std::vector<unsigned char> >().swap( g_ctx.recBlocks );
    g_ctx.nRecs = 0;
    g_ctx.finalized = true;

    message << "\n" << MESSAGE::cBold() << PTS_MSG_BINARY << MESSAGE::cReset() << " point evaluation -- "
            << MESSAGE::cMagenta() << covered << "/" << N << MESSAGE::cReset() << " points covered, "
            << MESSAGE::cMagenta() << multi << MESSAGE::cReset() << " multi-stream, max streams "
            << MESSAGE::cMagenta() << maxStreams << MESSAGE::cReset() << "."
            << ( covered < N ? "  (uncovered points lie outside the tessellation or in empty regions)" : "" )
            << "\n" << MESSAGE::Flush;
}


void psPointEvalWriteOutputs(User_options const &userOptions)
{
    if ( not g_ctx.active || not g_ctx.finalized )
        return;

    MESSAGE::Message message( userOptions.verboseLevel );

    auto writeRaw = [&userOptions](std::string const &ext, void const *data, size_t bytes,
                                   char const *what)
    {
        std::string const name = userOptions.outputFilename + ext;
        std::ofstream f( name.c_str(), std::ios::binary | std::ios::trunc );
        if ( not f )
            throwError( "Cannot open the point-evaluation output file '", name, "'." );
        f.write( reinterpret_cast<char const *>(data), bytes );
        if ( not f )
            throwError( "Failed writing the point-evaluation output file '", name, "'." );
        MESSAGE::Message m( userOptions.verboseLevel );
        m << "Writing the " << what << " to the file '" << MESSAGE::cBlue() << name
          << MESSAGE::cReset() << "' ... Done.\n" << MESSAGE::Flush;
    };

    message << "\n" << MESSAGE::cBold() << PTS_MSG_BINARY << MESSAGE::cReset()
            << " writing the point-evaluation outputs (raw binary, see README):\n" << MESSAGE::Flush;
    writeRaw( ".pts_den",     g_ctx.outDen.data(),     g_ctx.outDen.size()     * sizeof(double),
              "point densities (float64, rho/rho_bar)" );
    writeRaw( ".pts_vel",     g_ctx.outVel.data(),     g_ctx.outVel.size()     * sizeof(double),
              "point mean velocities (float64 x 3)" );
    writeRaw( ".pts_velDisp", g_ctx.outDisp.data(),    g_ctx.outDisp.size()    * sizeof(double),
              "point velocity dispersion tensors (float64 x 6, xx xy xz yy yz zz)" );
    writeRaw( ".pts_streams", g_ctx.outStreams.data(), g_ctx.outStreams.size() * sizeof(int32_t),
              "point stream counts (int32)" );
    if ( g_ctx.psCaustics )
        writeRaw( ".pts_caustic", g_ctx.outCaustic.data(), g_ctx.outCaustic.size() * sizeof(int32_t),
                  "point caustic masks (int32; bits as in '.causticClass', both parity bits = fold)" );
    if ( g_ctx.ptsDenGrad )
        writeRaw( ".pts_denGrad", g_ctx.outDenGrad.data(), g_ctx.outDenGrad.size() * sizeof(double),
                  "point density gradients (float64 x 3, d(rho/rho_bar)/dx_i, 'dtfe' profile)" );
    if ( g_ctx.ptsVelGrad )
        writeRaw( ".pts_velGrad", g_ctx.outVelGrad.data(), g_ctx.outVelGrad.size() * sizeof(double),
                  "point velocity gradients (float64 x 9, [d*3+j] = dv_j/dx_d, density-weighted stream mean)" );
    if ( g_ctx.ptsScalar )
        writeRaw( ".pts_scalar", g_ctx.outScalar.data(), g_ctx.outScalar.size() * sizeof(double),
                  "point scalar values (float64, density-weighted mean over the streams)" );
    if ( g_ctx.perStream )
    {
        writeRaw( ".pts_stream_offsets", g_ctx.outOffsets.data(),
                  g_ctx.outOffsets.size() * sizeof(uint64_t),
                  "per-stream record offsets (uint64, N+1)" );
        writeRaw( ".pts_stream_records", g_ctx.outRecords.data(),
                  g_ctx.outRecords.size() * sizeof(double),
                  "per-stream records (float64 x 4: density, vx, vy, vz; density-descending)" );
        if ( g_ctx.perStreamIds )
            writeRaw( ".pts_stream_ids", g_ctx.outIds.data(),
                      g_ctx.outIds.size() * sizeof(uint64_t),
                      "per-stream identities (uint64 x 4: sorted Lagrangian-vertex ParticleIDs)" );
        if ( g_ctx.ptsDenGrad )
            writeRaw( ".pts_stream_dengrad", g_ctx.outRecGrad.data(),
                      g_ctx.outRecGrad.size() * sizeof(double),
                      "per-stream density gradients (float64 x 3, 'dtfe' profile)" );
        if ( g_ctx.ptsScalar )
            writeRaw( ".pts_stream_scalar", g_ctx.outRecScalar.data(),
                      g_ctx.outRecScalar.size() * sizeof(double),
                      "per-stream scalar values (float64)" );
        if ( g_ctx.ptsLagrangian )
            writeRaw( ".pts_stream_lagpos", g_ctx.outRecLag.data(),
                      g_ctx.outRecLag.size() * sizeof(double),
                      "per-stream Lagrangian coordinates (float64 x 3, the initial position of each stream's matter)" );
    }
}


/* ---- --serve: interactive point evaluation over stdin/stdout ------------------------------

   The tessellation is built once; every request is then answered from it by exactly the
   --sample-points machinery (same exact containment, same threads, same reduction), so a
   request's answer is bit-identical to a '--sample-points' run on the same points. Client:
   python/dtfelib/estimator.py. All integers little-endian, doubles native (IEEE-754):

   server -> client, once, when the tessellation is ready:
       char[8]  "DTFESRV2"
       uint32   flags: bit0 PS-DTFE build, bit1 --per-stream, bit2 --per-stream-ids,
                       bit3 --pts-den-grad, bit4 --pts-vel-grad, bit5 --ps-caustics,
                       bit6 --ps-stream-density geometric, bit7 periodic,
                       bit8 --pts-scalar, bit9 --pts-lagrangian, bit10 composite (partitioned)
       uint32   S           scalar components per value (--pts-scalar; 0 otherwise)
       uint64   tessellation vertex count (a composite: summed over its partitions)
       double   region[6]   xlo xhi ylo yhi zlo zhi (the box the points are wrapped into)
       uint32   partitions  (1 for a single tessellation)
       uint32   resident    at most this many partitions are held in memory at once
   client -> server, per request:
       uint64   n           number of points; 0 (or closing stdin) ends the session
       double   xyz[3n]
   server -> client, per request -- the '.pts_*' arrays of a --sample-points run, in order:
       char[8]  "DTFEPTS1",  uint64 n,  uint64 R (per-stream records; 0 without --per-stream)
       double den[n], vel[3n], velDisp[6n];  int32 streams[n]
       int32  caustic[n]        (--ps-caustics)       double denGrad[3n]  (--pts-den-grad)
       double velGrad[9n]       (--pts-vel-grad)      uint64 offsets[n+1] (--per-stream)
       double records[4R]       (--per-stream)        uint64 ids[4R]      (--per-stream-ids)
       double recGrad[3R]       (--per-stream + --pts-den-grad)
       double scalar[S n]       (--pts-scalar)        double recScalar[S R] (--pts-scalar + --per-stream)
       double recLag[3R]        (--pts-lagrangian)
   a request the server rejects (non-finite coordinates, too many points, a partition whose
   cached tessellation vanished) is answered with
       char[8]  "DTFEERR1",  uint64 length,  char message[length]
   and the session continues.

   Version 2 (2026-09-30) added the scalar/Lagrangian arrays and the partition counts; the
   header changed with it, so a version-1 client stops at the handshake instead of misreading. */

static int g_serveFd = -1;   // the protocol channel: a private duplicate of the original stdout

void psServeRedirectStdout()
{
    std::cout.flush();
    std::fflush( stdout );
    int const fd = ::dup( STDOUT_FILENO );
    if ( fd < 0 || ::dup2( STDERR_FILENO, STDOUT_FILENO ) < 0 )
    {
        std::fprintf( stderr, "--serve: cannot set up the protocol channel (dup/dup2 failed).\n" );
        std::exit( 1 );
    }
    g_serveFd = fd;
}

namespace psPointEvalDetail {

// A vanished client (EPIPE) ends the server quietly: nobody is left to report to.
static void serveWrite(void const *data, size_t bytes)
{
    char const *p = static_cast<char const *>(data);
    while ( bytes > 0 )
    {
        ssize_t const w = ::write( g_serveFd, p, bytes );
        if ( w < 0 )
        {
            if ( errno == EINTR ) continue;
            std::_Exit( 0 );
        }
        p += w;
        bytes -= size_t(w);
    }
}

// false on end of input (the client closed stdin), which ends the session
static bool serveRead(void *data, size_t bytes)
{
    char *p = static_cast<char *>(data);
    while ( bytes > 0 )
    {
        ssize_t const r = ::read( STDIN_FILENO, p, bytes );
        if ( r < 0 && errno == EINTR ) continue;
        if ( r <= 0 ) return false;
        p += r;
        bytes -= size_t(r);
    }
    return true;
}

template <typename T>
static void serveWriteVec(std::vector<T> const &v)
{
    if ( not v.empty() ) serveWrite( v.data(), v.size() * sizeof(T) );
}

static void serveError(std::string const &msg)
{
    uint64_t const len = msg.size();
    serveWrite( "DTFEERR1", 8 );
    serveWrite( &len, sizeof(len) );
    serveWrite( msg.data(), msg.size() );
}

// The server's cell index: every finite cell registered in the uniform buckets its Eulerian
// bounding box overlaps -- min-image wrapped and padded exactly like tetBucketRange, so a cell
// containing a point is always registered in that point's bucket. Built once; a small request
// then lists only the cells of its points' buckets (deduplicated by a per-request stamp) and
// the workers run on that list instead of the whole tessellation. Identical per-cell work on
// a superset of the containing cells, so the answers stay bit-identical to a full walk.
struct ServeIndex
{
    int    n[NO_DIM];
    double invW[NO_DIM];
    std::vector<Cell_handle> cells;     // every finite cell
    std::vector<size_t>   start;        // CSR over buckets: members[start[b] .. start[b+1])
    std::vector<uint32_t> members;      // indices into 'cells'
    std::vector<uint32_t> stamp;        // per cell: the last request that listed it
    uint32_t request = 0;
};

inline Real storedEulerian(Cell_handle const &c, int v, int d)
{
#ifdef PHASE_SPACE
    return c->vertex(v)->info().eulerianPosition(d);
#else
    return Real( c->vertex(v)->point()[d] );
#endif
}

inline int serveBucketOf(ServeIndex const &I, double x, int d)
{
    int b = int( std::floor( (x - g_ctx.boxLo[d]) * I.invW[d] ) );
    if ( b < 0 ) b = 0;                      // wrapped periodic points never are; others clamp
    if ( b >= I.n[d] ) b = I.n[d] - 1;
    return b;
}

// the cell's bucket range (unwrapped for periodic axes; callers wrap); false = no overlap
static bool serveRange(ServeIndex const &I, Cell_handle const &c, int lo[NO_DIM], int hi[NO_DIM])
{
    for (int d = 0; d < NO_DIM; ++d)
    {
        double const x0 = double( storedEulerian(c, 0, d) );
        double eMin = x0, eMax = x0;
        for (int v = 1; v <= NO_DIM; ++v)
        {
            double x = double( storedEulerian(c, v, d) );
            if ( g_ctx.periodic )
            {
                double const r = x - x0;
                if ( r >  0.5 * g_ctx.boxLen[d] ) x -= g_ctx.boxLen[d];
                if ( r < -0.5 * g_ctx.boxLen[d] ) x += g_ctx.boxLen[d];
            }
            eMin = std::min( eMin, x );
            eMax = std::max( eMax, x );
        }
        double const margin = 1.e-5 * (eMax - eMin) + 1.e-12 + 2. * g_ctx.canon.epsAbs;
        lo[d] = int( std::floor( (eMin - margin - g_ctx.boxLo[d]) * I.invW[d] ) );
        hi[d] = int( std::floor( (eMax + margin - g_ctx.boxLo[d]) * I.invW[d] ) );
        if ( g_ctx.periodic )
        {
            if ( hi[d] - lo[d] + 1 >= I.n[d] ) { lo[d] = 0; hi[d] = I.n[d] - 1; }
        }
        else
        {
            if ( lo[d] < 0 ) lo[d] = 0;
            if ( hi[d] >= I.n[d] ) hi[d] = I.n[d] - 1;
            if ( lo[d] > hi[d] ) return false;
        }
    }
    return true;
}

static void buildServeIndex(DT &dt, ServeIndex &I)
{
    for (DT::Finite_cells_iterator it = dt.finite_cells_begin(); it != dt.finite_cells_end(); ++it)
        I.cells.push_back( it );
    size_t const T = I.cells.size();
    int nb = int( std::cbrt( double(T) / 8. ) );    // buckets about a few tetrahedra wide
    nb = std::max( 1, std::min( nb, 256 ) );
    size_t nTot = 1;
    for (int d = 0; d < NO_DIM; ++d)
    {
        I.n[d] = nb;
        I.invW[d] = double(nb) / g_ctx.boxLen[d];
        nTot *= size_t(nb);
    }

    // two passes (count, fill) over the same per-cell bucket ranges
    I.start.assign( nTot + 1, 0 );
    std::vector<size_t> fill;
    for (int pass = 0; pass < 2; ++pass)
    {
        for (size_t ci = 0; ci < T; ++ci)
        {
            int lo[NO_DIM], hi[NO_DIM];
            if ( not serveRange( I, I.cells[ci], lo, hi ) ) continue;
            for (int bi = lo[0]; bi <= hi[0]; ++bi)
            for (int bj = lo[1]; bj <= hi[1]; ++bj)
            for (int bk = lo[2]; bk <= hi[2]; ++bk)
            {
                int const wi = g_ctx.periodic ? ((bi % I.n[0] + I.n[0]) % I.n[0]) : bi;
                int const wj = g_ctx.periodic ? ((bj % I.n[1] + I.n[1]) % I.n[1]) : bj;
                int const wk = g_ctx.periodic ? ((bk % I.n[2] + I.n[2]) % I.n[2]) : bk;
                size_t const b = ( size_t(wi) * size_t(I.n[1]) + size_t(wj) ) * size_t(I.n[2]) + size_t(wk);
                if ( pass == 0 ) ++I.start[b+1];
                else             I.members[ fill[b]++ ] = uint32_t(ci);
            }
        }
        if ( pass == 0 )
        {
            for (size_t b = 0; b < nTot; ++b) I.start[b+1] += I.start[b];
            I.members.resize( I.start[nTot] );
            fill.assign( I.start.begin(), I.start.end() - 1 );
        }
    }
    I.stamp.assign( T, 0u );
}

// the deduplicated cells registered in the buckets of the current request's points
static void serveCandidates(ServeIndex &I, std::vector<Cell_handle> &list)
{
    if ( ++I.request == 0 )     // stamp wrap-around: reset and start over
    {
        std::fill( I.stamp.begin(), I.stamp.end(), 0u );
        I.request = 1;
    }
    list.clear();
    for (size_t i = 0; i < g_ctx.nPoints; ++i)
    {
        size_t b = 0;
        for (int d = 0; d < NO_DIM; ++d)
            b = b * size_t(I.n[d]) + size_t( serveBucketOf( I, g_ctx.pos[i*NO_DIM+d], d ) );
        for (size_t m = I.start[b]; m < I.start[b+1]; ++m)
        {
            uint32_t const ci = I.members[m];
            if ( I.stamp[ci] == I.request ) continue;
            I.stamp[ci] = I.request;
            list.push_back( I.cells[ci] );
        }
    }
}

} // namespace psPointEvalDetail

namespace psPointEvalDetail {

// The handshake (protocol above): written once the server can answer.
static void serveHandshake(uint64_t const nVert, uint32_t const nPartitions, uint32_t const nResident)
{
    uint32_t flags = 0;
#ifdef PHASE_SPACE
    flags |= 1u;
#endif
    if ( g_ctx.perStream )     flags |= 1u << 1;
    if ( g_ctx.perStreamIds )  flags |= 1u << 2;
    if ( g_ctx.ptsDenGrad )    flags |= 1u << 3;
    if ( g_ctx.ptsVelGrad )    flags |= 1u << 4;
    if ( g_ctx.psCaustics )    flags |= 1u << 5;
    if ( g_ctx.useGeometric )  flags |= 1u << 6;
    if ( g_ctx.periodic )      flags |= 1u << 7;
    if ( g_ctx.ptsScalar )     flags |= 1u << 8;
    if ( g_ctx.ptsLagrangian ) flags |= 1u << 9;
    if ( nPartitions > 1 )     flags |= 1u << 10;
    uint32_t const nScalar = g_ctx.ptsScalar ? uint32_t(noScalarComp) : 0u;
    double region[2*NO_DIM];
    for (int d = 0; d < NO_DIM; ++d)
    {
        region[2*d]   = g_ctx.boxLo[d];
        region[2*d+1] = g_ctx.boxLo[d] + g_ctx.boxLen[d];
    }
    serveWrite( "DTFESRV2", 8 );
    serveWrite( &flags, sizeof(flags) );
    serveWrite( &nScalar, sizeof(nScalar) );
    serveWrite( &nVert, sizeof(nVert) );
    serveWrite( region, sizeof(region) );
    serveWrite( &nPartitions, sizeof(nPartitions) );
    serveWrite( &nResident, sizeof(nResident) );
}

/* The request loop shared by the single and the composite server. 'evaluate' collects the streams
   of the installed query points (the workers append to g_ctx's records) and returns an empty
   string, or an error message -- the request is then refused and the session continues.
   'indexCells' = the cell count a request is compared against: fewer points than 1/64 of it go
   through the servers' cell indices ('useIndex'), more walk the tessellation(s) with buckets about
   'bucketFloor' per axis, so cells far from every point still fail the cheap occupancy test. */
template <typename Evaluate>
[[noreturn]] void serveLoop(User_options &userOptions, size_t const indexCells, int const bucketFloor,
                            Evaluate evaluate)
{
    size_t nRequests = 0;
    for (;;)
    {
        uint64_t n = 0;
        if ( not serveRead( &n, sizeof(n) ) || n == 0 )
            break;
        if ( n > uint64_t(0xFFFFFFFFu) )
        {
            // drain the payload so the stream stays in sync, then refuse
            std::vector<char> sink( size_t(1) << 20 );
            uint64_t left = n * 3 * sizeof(double);
            bool ok = true;
            while ( left > 0 && ok )
            {
                size_t const chunk = size_t( std::min<uint64_t>( left, sink.size() ) );
                ok = serveRead( sink.data(), chunk );
                left -= chunk;
            }
            if ( not ok ) break;
            serveError( "a request holds at most 2^32-1 points; split it" );
            continue;
        }

        std::vector<double> pts( size_t(n) * NO_DIM );
        if ( not serveRead( pts.data(), pts.size() * sizeof(double) ) )
            break;
        bool finite = true;
        for (double const x : pts)
            if ( not std::isfinite(x) ) { finite = false; break; }
        if ( not finite )
        {
            serveError( "non-finite point coordinate (NaN or inf) in the request" );
            continue;
        }

        bool const useIndex = n * 64 < uint64_t(indexCells);
        setQueryPoints( std::move(pts), useIndex ? 1 : bucketFloor );
        std::string const failure = evaluate( useIndex );
        g_cellList = nullptr;
        if ( not failure.empty() )
        {
            serveError( failure );
            continue;
        }
        psPointEvalFinalize( userOptions );

        uint64_t const R = g_ctx.perStream ? uint64_t( g_ctx.outRecords.size() / (1 + noVelComp) ) : 0;
        serveWrite( "DTFEPTS1", 8 );
        serveWrite( &n, sizeof(n) );
        serveWrite( &R, sizeof(R) );
        serveWriteVec( g_ctx.outDen );
        serveWriteVec( g_ctx.outVel );
        serveWriteVec( g_ctx.outDisp );
        serveWriteVec( g_ctx.outStreams );
        if ( g_ctx.psCaustics )   serveWriteVec( g_ctx.outCaustic );
        if ( g_ctx.ptsDenGrad )   serveWriteVec( g_ctx.outDenGrad );
        if ( g_ctx.ptsVelGrad )   serveWriteVec( g_ctx.outVelGrad );
        if ( g_ctx.perStream )
        {
            serveWriteVec( g_ctx.outOffsets );
            serveWriteVec( g_ctx.outRecords );
            if ( g_ctx.perStreamIds ) serveWriteVec( g_ctx.outIds );
            if ( g_ctx.ptsDenGrad )   serveWriteVec( g_ctx.outRecGrad );
        }
        if ( g_ctx.ptsScalar )
        {
            serveWriteVec( g_ctx.outScalar );
            if ( g_ctx.perStream ) serveWriteVec( g_ctx.outRecScalar );
        }
        if ( g_ctx.ptsLagrangian ) serveWriteVec( g_ctx.outRecLag );
        ++nRequests;
    }

    MESSAGE::Message message( userOptions.verboseLevel );
    message << "\n" << MESSAGE::cBold() << PTS_MSG_BINARY << MESSAGE::cReset() << " server done ("
            << nRequests << " requests answered).\n" << MESSAGE::Flush;
    std::cout.flush();
    std::fflush( stdout );
    std::fflush( stderr );
    ::close( g_serveFd );
    // _Exit: nothing is left to write, and tearing down a large triangulation (plus the
    // static state of the libraries) would only delay the client's clean shutdown
    std::_Exit( 0 );
}

} // namespace psPointEvalDetail

[[noreturn]] void psServe(DT &dt, User_options &userOptions)
{
    if ( g_serveFd < 0 )
        throwError( "'--serve': the protocol channel is not set up (main() must call psServeRedirectStdout() before any output)." );
    if ( not g_ctx.active )
        throwError( "'--serve': the point-evaluation context was not initialized." );

    // the cell index answers small requests without touching the rest of the tessellation;
    // large ones walk it all (with buckets about a few tetrahedra wide, so cells far from
    // every point still fail the cheap occupancy test)
    ServeIndex index;
    {
        Timer t;
        t.start();
        buildServeIndex( dt, index );
        printComputationTime( &t, &userOptions, "point-evaluation server cell index" );
    }
    size_t const nCells = index.cells.size();
    int const bucketFloor = int( std::cbrt( double(nCells) / 16. ) );
    std::vector<Cell_handle> candidates;

    serveHandshake( uint64_t( dt.number_of_vertices() ), 1u, 1u );
    {
        MESSAGE::Message message( userOptions.verboseLevel );
        message << "\n" << MESSAGE::cBold() << PTS_MSG_BINARY << MESSAGE::cReset()
                << " serving point-evaluation requests (" << dt.number_of_vertices()
                << " vertices); close stdin to stop.\n" << MESSAGE::Flush;
    }

    serveLoop( userOptions, nCells, bucketFloor, [&](bool const useIndex) -> std::string
    {
        if ( useIndex )
        {
            serveCandidates( index, candidates );
            g_cellList = &candidates;
        }
#ifdef PHASE_SPACE
        interpolatePoints_phaseSpace( dt, userOptions );
#else
        interpolatePoints_standard( dt, userOptions );
#endif
        return std::string();
    } );
}


/* ---- composite '--serve' (--serve + --partition) -------------------------------------------

   CosmoDTFE's CompositeEstimator routes each query to one of several local estimators; here the
   same idea serves boxes that do not fit in memory as ONE tessellation. The partitions are the
   run's usual ones -- Lagrangian sub-boxes in PS-DTFE (a stream is owned by exactly one partition
   via the Lagrangian-centroid test in psFilterCell), Eulerian sub-boxes in the standard binary
   (a tetrahedron is owned by the partition whose region holds its centroid, OwnBox) -- so the
   union of the partitions' owned tetrahedra is exactly the single tessellation's, and a request
   answered partition by partition is bit-identical to one answered by a single server (the
   records are sorted before any reduction).

   Routing: every partition keeps an OCCUPANCY map, one bit per bucket of a coarse OCC_N^3 grid over
   the box, set wherever one of its owned tetrahedra's Eulerian bounding boxes reaches (min-image
   wrapped and padded like tetBucketRange). A tetrahedron containing a point always covers that
   point's bucket, so a request only visits the partitions whose map holds one of its points.

   Out of core: with '--tessellation-cache' each partition's tessellation lives on disk (the batch
   runs' own per-partition cache files: the descriptors match, so a batch run with the cache warms
   the server and vice versa) and its occupancy map beside it ('<cache file>.occ', stamped with the
   same descriptor). At most 'resident' partitions are held in memory; the least recently used one
   is dropped when another must be loaded. A session whose inputs are unchanged registers every
   partition from these files without building or even loading anything. Without the cache all
   partitions stay in memory (an in-memory composite: faster to build, since the partitions are
   triangulated in parallel, but not smaller than one tessellation). */

namespace psPointEvalDetail {

static const int OCC_N = 64;                 // occupancy buckets per axis (32 KB per partition)
static const size_t OCC_WORDS = size_t(OCC_N) * OCC_N * OCC_N / 64;
static const char OCC_MAGIC[8] = { 'D','T','F','E','O','C','C','1' };

struct ServePart
{
    User_options opts;                       // this partition's options: its ownership + cache identity
    OwnBox own;                              // standard binary: the Eulerian box it owns
    std::vector<uint64_t> occ;               // occupancy bits (OCC_WORDS words); empty = owns nothing
    std::unique_ptr<DT> dt;                  // resident tessellation, or null (in the cache only)
    std::unique_ptr<ServeIndex> index;       // its cell index, built on first small request
    uint64_t lastUse = 0;
    uint64_t nVertices = 0;
    bool cached = false;                     // a valid cache file exists: may be dropped and reloaded
};

struct Composite
{
    std::vector<ServePart> parts;
    int verbose = 1;
    std::mutex logMutex;
    int done = 0;
    uint64_t clock = 0;
    size_t maxResident = 1;                  // cap on resident CACHED partitions (uncached ones are pinned)
};
static Composite g_comp;

inline int occBucket(double x, int d)
{
    double const w = double(OCC_N) / g_ctx.boxLen[d];
    int b = int( std::floor( (x - g_ctx.boxLo[d]) * w ) );
    if ( g_ctx.periodic )
    {
        b %= OCC_N;
        if ( b < 0 ) b += OCC_N;
    }
    else
    {
        if ( b < 0 ) b = 0;
        if ( b >= OCC_N ) b = OCC_N - 1;
    }
    return b;
}

// Marks the occupancy buckets of one tetrahedron's (wrapped) Eulerian vertices -- the same bbox
// and slack as tetBucketRange, so no bucket the exact test could accept a point in is missed.
static void markOccupancy(std::vector<uint64_t> &occ, Real const eulerPos[NO_DIM+1][NO_DIM])
{
    int lo[NO_DIM], hi[NO_DIM];
    for (int d = 0; d < NO_DIM; ++d)
    {
        double eMin = double(eulerPos[0][d]), eMax = eMin;
        for (int v = 1; v <= NO_DIM; ++v)
        {
            double const c = double(eulerPos[v][d]);
            eMin = std::min( eMin, c );
            eMax = std::max( eMax, c );
        }
        double const margin = 1.e-5 * (eMax - eMin) + 1.e-12 + 2. * g_ctx.canon.epsAbs;
        double const w = double(OCC_N) / g_ctx.boxLen[d];
        lo[d] = int( std::floor( (eMin - margin - g_ctx.boxLo[d]) * w ) );
        hi[d] = int( std::floor( (eMax + margin - g_ctx.boxLo[d]) * w ) );
        if ( g_ctx.periodic )
        {
            if ( hi[d] - lo[d] + 1 >= OCC_N ) { lo[d] = 0; hi[d] = OCC_N - 1; }
        }
        else
        {
            lo[d] = std::max( 0, std::min( lo[d], OCC_N - 1 ) );
            hi[d] = std::max( 0, std::min( hi[d], OCC_N - 1 ) );
        }
    }
    for (int i = lo[0]; i <= hi[0]; ++i)
    for (int j = lo[1]; j <= hi[1]; ++j)
    for (int k = lo[2]; k <= hi[2]; ++k)
    {
        int const wi = ((i % OCC_N) + OCC_N) % OCC_N, wj = ((j % OCC_N) + OCC_N) % OCC_N, wk = ((k % OCC_N) + OCC_N) % OCC_N;
        size_t const b = ( size_t(wi) * OCC_N + size_t(wj) ) * OCC_N + size_t(wk);
        occ[b >> 6] |= uint64_t(1) << (b & 63);
    }
}

// The partition's occupancy map from its OWNED tetrahedra (the workers' own filters).
static void computeOccupancy(DT &dt, ServePart &P)
{
    P.occ.assign( OCC_WORDS, 0 );
    for (DT::Finite_cells_iterator it = dt.finite_cells_begin(); it != dt.finite_cells_end(); ++it)
    {
        Cell_handle cell = it;
#ifdef PHASE_SPACE
        PSCellGeometry geo;
        Box boxCoordinates = P.opts.region;
        if ( not psFilterCell( cell, P.opts, boxCoordinates, true, NULL, geo ) ) continue;
        markOccupancy( P.occ, geo.eulerPos );
#else
        double rho[NO_DIM+1], vel[NO_DIM+1][noVelComp];
        Real eulerPos[NO_DIM+1][NO_DIM], rawPos[NO_DIM+1][NO_DIM];
        if ( not standardCell( cell, P.own, rho, vel, eulerPos, rawPos ) ) continue;
        markOccupancy( P.occ, eulerPos );
#endif
    }
}

// '<cache file>.occ': the occupancy map, stamped with the tessellation's own descriptor.
static std::string occPath(User_options const &o) { return TessellationCache::cachePath( o ) + ".occ"; }

static void writeOccupancy(ServePart const &P)
{
    std::string const desc = TessellationCache::makeDescriptor( P.opts );
    std::string const path = occPath( P.opts ), tmp = path + ".tmp";
    std::ofstream f( tmp.c_str(), std::ios::binary | std::ios::trunc );
    if ( not f ) return;
    uint32_t const len = uint32_t( desc.size() ), n = uint32_t( OCC_N );
    uint64_t const words = uint64_t( P.occ.size() );
    f.write( OCC_MAGIC, 8 );
    f.write( reinterpret_cast<char const*>(&len), sizeof len );
    f.write( desc.data(), std::streamsize(len) );
    f.write( reinterpret_cast<char const*>(&n), sizeof n );
    f.write( reinterpret_cast<char const*>(&P.nVertices), sizeof P.nVertices );
    f.write( reinterpret_cast<char const*>(&words), sizeof words );
    f.write( reinterpret_cast<char const*>(P.occ.data()), std::streamsize( words * sizeof(uint64_t) ) );
    f.close();
    if ( f ) std::rename( tmp.c_str(), path.c_str() );
    else     std::remove( tmp.c_str() );
}

static bool readOccupancy(ServePart &P)
{
    std::ifstream f( occPath( P.opts ).c_str(), std::ios::binary );
    if ( not f ) return false;
    char magic[8] = {0};
    uint32_t len = 0, n = 0;
    uint64_t nVert = 0, words = 0;
    f.read( magic, 8 );
    f.read( reinterpret_cast<char*>(&len), sizeof len );
    if ( not f or std::memcmp( magic, OCC_MAGIC, 8 ) != 0 or len == 0 or len > (1u<<20) ) return false;
    std::string stored( size_t(len), '\0' );
    f.read( &stored[0], std::streamsize(len) );
    f.read( reinterpret_cast<char*>(&n), sizeof n );
    f.read( reinterpret_cast<char*>(&nVert), sizeof nVert );
    f.read( reinterpret_cast<char*>(&words), sizeof words );
    if ( not f or stored != TessellationCache::makeDescriptor( P.opts ) or n != uint32_t(OCC_N)
         or (words != OCC_WORDS and words != 0) ) return false;
    std::vector<uint64_t> occ( static_cast<size_t>(words) );
    f.read( reinterpret_cast<char*>(occ.data()), std::streamsize( words * sizeof(uint64_t) ) );
    if ( not f ) return false;
    P.occ.swap( occ );
    P.nVertices = nVert;
    return true;
}

static size_t residentCached()
{
    size_t k = 0;
    for (ServePart const &P : g_comp.parts)
        if ( P.dt and P.cached ) ++k;
    return k;
}

// Loads partition p (dropping the least recently used cached ones beyond the cap). False when its
// cached tessellation cannot be read back.
static bool ensureResident(size_t const p, double &loadSeconds)
{
    ServePart &P = g_comp.parts[p];
    if ( P.dt ) return true;
    while ( residentCached() >= g_comp.maxResident )
    {
        size_t victim = g_comp.parts.size();
        for (size_t q = 0; q < g_comp.parts.size(); ++q)
        {
            ServePart const &Q = g_comp.parts[q];
            if ( q == p or not Q.dt or not Q.cached ) continue;
            if ( victim == g_comp.parts.size() or Q.lastUse < g_comp.parts[victim].lastUse ) victim = q;
        }
        if ( victim == g_comp.parts.size() ) break;
        g_comp.parts[victim].index.reset();          // its Cell_handles point into the tessellation
        g_comp.parts[victim].dt.reset();
    }
    auto const t0 = std::chrono::steady_clock::now();
    std::unique_ptr<DT> dt( new DT );
    if ( not TessellationCache::tryLoad( *dt, P.opts ) ) return false;
    loadSeconds += std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
    P.dt = std::move( dt );
    return true;
}

} // namespace psPointEvalDetail


void psServeCompositeBegin(User_options const &userOptions, int const totalPartitions)
{
    if ( g_serveFd < 0 )
        throwError( "'--serve': the protocol channel is not set up (main() must call psServeRedirectStdout() before any output)." );
    if ( not g_ctx.active )
        throwError( "'--serve': the point-evaluation context was not initialized." );
    g_comp.parts.clear();
    g_comp.parts.resize( size_t( std::max( 1, totalPartitions ) ) );
    g_comp.verbose = userOptions.verboseLevel;
    g_comp.done = 0;
    MESSAGE::Message message( userOptions.verboseLevel );
    message << "\n" << MESSAGE::cBold() << PTS_MSG_BINARY << MESSAGE::cReset() << " composite server over "
            << MESSAGE::cMagenta() << totalPartitions << MESSAGE::cReset() << " partition tessellations"
            << ( userOptions.tessellationCacheDir.empty()
                 ? " (no '--tessellation-cache': every partition stays in memory)"
                 : " (out of core: partitions live in '" + userOptions.tessellationCacheDir + "')" )
            << ".\n" << MESSAGE::Flush;
}


void psServeCompositeAddPartition(int const index, User_options &partOptions,
                                  std::function<void(std::vector<Particle_data>&)> const &selectParticles,
                                  double const *ownLo, double const *ownHi)
{
    ServePart &P = g_comp.parts.at( size_t(index) );
    P.opts = partOptions;
    P.own.on = ( ownLo != nullptr );
    for (int d = 0; d < NO_DIM; ++d)
    {
        P.own.lo[d] = ownLo ? ownLo[d] : 0.;
        P.own.hi[d] = ownHi ? ownHi[d] : 0.;
    }
    auto const t0 = std::chrono::steady_clock::now();
    bool const caching = not P.opts.tessellationCacheDir.empty();
    char const *how = "built";

    // an earlier session (or a batch run with the cache) left this partition's tessellation AND
    // its map: register it without loading anything
    if ( caching and TessellationCache::headerMatches( P.opts ) and readOccupancy( P ) )
    {
        P.cached = true;
        how = "cached";
    }
    else
    {
        // the tessellation may be cached without its map (a batch run wrote it): load it; else
        // select the particles and build (and, with the cache, save) it
        std::unique_ptr<DT> dt( new DT );
        bool have = caching and TessellationCache::headerMatches( P.opts )
                    and TessellationCache::tryLoad( *dt, P.opts );
        if ( have )
            how = "loaded";
        else
        {
            std::vector<Particle_data> particles;
            selectParticles( particles );
            if ( not particles.empty() )
            {
                buildTessellation( &particles, P.opts, *dt );
                have = true;
            }
        }
        if ( not have )
        {
            P.occ.clear();              // no particles: owns nothing
            how = "empty";
        }
        else
        {
            P.nVertices = uint64_t( dt->number_of_vertices() );
            computeOccupancy( *dt, P );
            if ( caching and TessellationCache::headerMatches( P.opts ) )
            {
                writeOccupancy( P );
                P.cached = true;
                dt.reset();             // out of core: reloaded when a request needs it
            }
            else
                P.dt = std::move( dt ); // no cache (or it could not be written): stays in memory
        }
    }

    double const sec = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
    std::lock_guard<std::mutex> lock( g_comp.logMutex );
    ++g_comp.done;
    MESSAGE::Message prog( g_comp.verbose );
    prog << MESSAGE::cGreen() << "  [partitions " << g_comp.done << "/" << g_comp.parts.size() << "]"
         << MESSAGE::cReset() << MESSAGE::cDim() << "  partition " << index << ": " << how << ", "
         << P.nVertices << " vertices, " << MESSAGE::formatDuration(sec) << MESSAGE::cReset() << "\n"
         << MESSAGE::Flush;
}


[[noreturn]] void psServeCompositeRun(User_options &userOptions)
{
    MESSAGE::Message message( userOptions.verboseLevel );
    size_t const nParts = g_comp.parts.size();
    uint64_t nVertTotal = 0, nVertMax = 0;
    size_t nCached = 0;
    for (ServePart const &P : g_comp.parts)
    {
        nVertTotal += P.nVertices;
        nVertMax = std::max( nVertMax, P.nVertices );
        if ( P.cached ) ++nCached;
    }

    // how many cached partitions may be in memory at once: --serve-resident, else the auto-tune
    // memory budget over the largest partition (triangulation ~658 B/vertex + its cell index)
    if ( userOptions.serveResident > 0 )
        g_comp.maxResident = size_t( userOptions.serveResident );
    else if ( nCached > 0 )
    {
        std::string why;
        double const budget = autoTuneBudget( autoTunePhysicalRAM(), why );
        double const perPart = 850. * double( std::max<uint64_t>( nVertMax, 1 ) );
        g_comp.maxResident = size_t( std::max( 1., std::floor( budget / perPart ) ) );
    }
    else
        g_comp.maxResident = nParts;
    g_comp.maxResident = std::max<size_t>( 1, std::min( g_comp.maxResident, nParts ) );

    message << "\n" << MESSAGE::cBold() << PTS_MSG_BINARY << MESSAGE::cReset() << " composite server ready: "
            << nParts << " partitions, " << nVertTotal << " vertices in total; "
            << ( nCached > 0 ? std::to_string( g_comp.maxResident ) + " held in memory at once (--serve-resident"
                               + ( userOptions.serveResident > 0 ? "" : ", automatic" ) + "), the others reloaded from the cache on demand"
                             : std::string( "all held in memory" ) )
            << "; close stdin to stop.\n" << MESSAGE::Flush;

    size_t const indexCells = size_t( 6.8 * double( std::max<uint64_t>( nVertMax, 1 ) ) );
    int const bucketFloor = int( std::cbrt( double(indexCells) / 16. ) );
    std::vector<Cell_handle> candidates;
    serveHandshake( nVertTotal, uint32_t(nParts), uint32_t(g_comp.maxResident) );

    serveLoop( userOptions, indexCells, bucketFloor, [&](bool const useIndex) -> std::string
    {
        // the buckets holding a requested point
        std::vector<uint64_t> want( OCC_WORDS, 0 );
        for (size_t i = 0; i < g_ctx.nPoints; ++i)
        {
            size_t b = 0;
            for (int d = 0; d < NO_DIM; ++d)
                b = b * size_t(OCC_N) + size_t( occBucket( g_ctx.pos[i*NO_DIM+d], d ) );
            want[b >> 6] |= uint64_t(1) << (b & 63);
        }
        std::vector<size_t> need;
        for (size_t p = 0; p < nParts; ++p)
        {
            std::vector<uint64_t> const &occ = g_comp.parts[p].occ;
            if ( occ.empty() ) continue;
            for (size_t w = 0; w < OCC_WORDS; ++w)
                if ( occ[w] & want[w] ) { need.push_back( p ); break; }
        }
        // the resident ones first, so a load never evicts a partition this request still needs
        // while an already-loaded one waits
        std::stable_partition( need.begin(), need.end(), [](size_t p) { return bool( g_comp.parts[p].dt ); } );

        size_t loaded = 0;
        double loadSeconds = 0.;
        for (size_t const p : need)
        {
            ServePart &P = g_comp.parts[p];
            bool const wasResident = bool( P.dt );
            if ( not ensureResident( p, loadSeconds ) )
                return "partition " + std::to_string(p) + "'s cached tessellation could not be read back from '"
                       + P.opts.tessellationCacheDir + "' (deleted, or the inputs changed since the server started); restart the server";
            if ( not wasResident ) ++loaded;
            g_cellList = nullptr;
            if ( useIndex )
            {
                if ( not P.index )
                {
                    P.index.reset( new ServeIndex );
                    buildServeIndex( *P.dt, *P.index );
                }
                serveCandidates( *P.index, candidates );
                g_cellList = &candidates;
            }
#ifdef PHASE_SPACE
            interpolatePoints_phaseSpace( *P.dt, P.opts );
#else
            interpolatePoints_standardOwn( *P.dt, P.opts, P.own );
#endif
            g_cellList = nullptr;
            P.lastUse = ++g_comp.clock;
        }
        if ( userOptions.verboseLevel >= 2 or loaded > 0 )
        {
            MESSAGE::Message m( userOptions.verboseLevel );
            m << MESSAGE::cDim() << "  request of " << g_ctx.nPoints << " points: " << need.size()
              << " of " << nParts << " partitions (" << loaded << " loaded from the cache in "
              << MESSAGE::formatDuration(loadSeconds) << ")" << MESSAGE::cReset() << "\n" << MESSAGE::Flush;
        }
        return std::string();
    } );
}


#else // NO_DIM != 3

// Point evaluation is 3D-only. The per-triangulation worker must still LINK in 2D
// (triangulation.cpp references it unconditionally; psPointEvalActive() is always false
// here, so it can never be called).
#include "triangulation_common.h"
#ifdef PHASE_SPACE
void interpolatePoints_phaseSpace(DT &, User_options &) {}
#else
void interpolatePoints_standard(DT &, User_options &) {}
#endif
bool psPointEvalActive() { return false; }
void psPointEvalInit(User_options &)
{
    throwError( "'--sample-points' (point evaluation) is implemented for 3D only." );
}
void psPointEvalFinalize(User_options const &) {}
void psPointEvalWriteOutputs(User_options const &) {}
void psServeRedirectStdout() {}
[[noreturn]] void psServe(DT &, User_options &)
{
    throwError( "'--serve' (interactive point evaluation) is implemented for 3D only." );
    std::exit( 1 );
}
void psServeCompositeBegin(User_options const &, int)
{
    throwError( "'--serve' (interactive point evaluation) is implemented for 3D only." );
}
void psServeCompositeAddPartition(int, User_options &, std::function<void(std::vector<Particle_data>&)> const &,
                                  double const *, double const *) {}
[[noreturn]] void psServeCompositeRun(User_options &)
{
    throwError( "'--serve' (interactive point evaluation) is implemented for 3D only." );
    std::exit( 1 );
}

#endif // NO_DIM


// ======================================================================================
// Auto-tune support: the mean number of streams at a random point of the box, estimated from
// the particles before any triangulation exists (declared in ps_point_eval.h).
//
// Every tetrahedron of the Lagrangian tessellation covers its Eulerian volume exactly once, so
// the mean stream count at a uniformly random point is sum|V_Euler| / sum V_Lagrange over all
// tetrahedra. That ratio is estimated on up to 256 random Lagrangian patches of 20 mean spacings
// on a side: Delaunay-triangulate each patch's Lagrangian positions (the tessellation PS-DTFE
// itself uses), keep the tetrahedra whose Lagrangian centroid lies 3 spacings inside the patch
// (hull tetrahedra are artefacts of the cut), and sum both volumes. MEASURED against the
// '.pts_streams' of a full-box 8192^2 plane, TNG100-3-Dark: estimated 1.00/0.99/1.13/1.6/2.3/
// 3.4-3.9 vs measured 1.00/1.02/1.21/2.30/2.65/4.25 at z = 20/5/2/1/0.5/0; the rest of the gap
// is the PLANE, not the estimate (16 strips of the z=0 plane range 1.45 .. 22.8), which is why
// the tuner budgets twice the multi-stream excess.
// ======================================================================================
#if NO_DIM == 3 && defined(PHASE_SPACE)      // Lagrangian positions exist only in the PS build
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <random>

double psEstimateMeanStreams(std::vector<Particle_data> const &particles, double const boxLoIn[3],
                             double const boxLenIn[3], bool const periodic, int *patchesUsed)
{
    typedef CGAL::Exact_predicates_inexact_constructions_kernel EK;
    typedef CGAL::Triangulation_vertex_base_with_info_3<uint32_t, EK> EVb;
    typedef CGAL::Triangulation_data_structure_3<EVb> ETds;
    typedef CGAL::Delaunay_triangulation_3<EK, ETds> EDT;

    if ( patchesUsed ) *patchesUsed = 0;
    size_t const N = particles.size();
    if ( N < 2000 or N > size_t(4294967295u) ) return -1.;
    // The Lagrangian domain the patches tile. Periodic: the box. Non-periodic: the particles' OWN
    // Lagrangian bounding box -- the output region ('--box') can be smaller than the cloud, and
    // wrapping initial positions into it folded particles from far apart into one patch, whose
    // Delaunay tetrahedra then spanned the whole cloud in Eulerian space: 8.1 million "streams"
    // per point on a TNG100-3 Lagrangian sub-cube with a 31 Mpc region (true mean ~1-2).
    double boxLo[3], boxLen[3];
    for (int d=0; d<3; ++d) { boxLo[d] = boxLoIn[d]; boxLen[d] = boxLenIn[d]; }
    if ( not periodic )
    {
        double lo[3], hi[3];
        for (int d=0; d<3; ++d) { lo[d] = double(particles[0].lagPos[d]); hi[d] = lo[d]; }
        for (size_t i=1; i<N; ++i)
            for (int d=0; d<3; ++d)
            {
                double const q = double(particles[i].lagPos[d]);
                lo[d] = std::min(lo[d], q);
                hi[d] = std::max(hi[d], q);
            }
        for (int d=0; d<3; ++d)
        {
            if ( not (hi[d] > lo[d]) ) return -1.;
            boxLo[d] = lo[d];
            boxLen[d] = (hi[d] - lo[d]) * (1. + 1e-9);          // the maximum stays inside the last cell
        }
    }
    double const spacing = std::cbrt( boxLen[0]*boxLen[1]*boxLen[2] / double(N) );
    int G[3];
    for (int d=0; d<3; ++d)
        G[d] = std::max( 2, int(boxLen[d] / (20.*spacing)) );
    size_t const nCells = size_t(G[0]) * size_t(G[1]) * size_t(G[2]);
    size_t const K = std::min( nCells, size_t(256) );
    std::vector<size_t> order( nCells );
    for (size_t c=0; c<nCells; ++c) order[c] = c;
    std::mt19937_64 rng( 20260929 );                          // fixed: the same data, the same estimate
    std::shuffle( order.begin(), order.end(), rng );
    std::vector<int> patchOf( nCells, -1 );
    for (size_t k=0; k<K; ++k) patchOf[ order[k] ] = int(k);

    // one pass: the particles of the chosen patches, by their Lagrangian position (wrapped into
    // the box: initial positions can stray a little outside it)
    std::vector< std::vector<uint32_t> > members( K );
    auto wrapped = [&](Real v, int d) {                        // (non-periodic: the bounding box holds every position)
        double x = double(v) - boxLo[d];
        if ( periodic ) x -= boxLen[d] * std::floor( x / boxLen[d] );
        return x;
    };
    for (size_t i=0; i<N; ++i)
    {
        size_t c = 0;
        for (int d=0; d<3; ++d)
        {
            int g = int( wrapped(particles[i].lagPos[d], d) / boxLen[d] * G[d] );
            c = c * size_t(G[d]) + size_t( std::min( std::max(g, 0), G[d]-1 ) );
        }
        if ( patchOf[c] >= 0 ) members[ patchOf[c] ].push_back( uint32_t(i) );
    }

    double sumEuler = 0., sumLag = 0.;
    int used = 0;
#pragma omp parallel for schedule(dynamic) reduction(+:sumEuler,sumLag,used)
    for (long k=0; k<long(K); ++k)
    {
        std::vector<uint32_t> const &idx = members[k];
        if ( idx.size() < 64 ) continue;
        size_t cell = order[size_t(k)];
        double lo[3], hi[3];                                  // the patch's inner region (Lagrangian)
        for (int d=2; d>=0; --d)
        {
            int const g = int( cell % size_t(G[d]) );
            cell /= size_t(G[d]);
            double const w = boxLen[d] / G[d];
            lo[d] = g * w + 3.*spacing;
            hi[d] = (g + 1) * w - 3.*spacing;
        }
        std::vector< std::pair<EK::Point_3, uint32_t> > pts;
        pts.reserve( idx.size() );
        for (uint32_t i : idx)
            pts.emplace_back( EK::Point_3( wrapped(particles[i].lagPos[0], 0), wrapped(particles[i].lagPos[1], 1),
                                           wrapped(particles[i].lagPos[2], 2) ), i );
        EDT dt( pts.begin(), pts.end() );
        for (auto c = dt.finite_cells_begin(); c != dt.finite_cells_end(); ++c)
        {
            double q[4][3], x[4][3];
            bool inside = true;
            for (int v=0; v<4; ++v)
            {
                EK::Point_3 const &p = c->vertex(v)->point();
                q[v][0] = p.x(); q[v][1] = p.y(); q[v][2] = p.z();
                Particle_data const &part = particles[ c->vertex(v)->info() ];
                for (int d=0; d<3; ++d) x[v][d] = double( part.pos[d] );
            }
            for (int d=0; d<3 and inside; ++d)
            {
                double const cen = 0.25 * (q[0][d] + q[1][d] + q[2][d] + q[3][d]);
                inside = cen >= lo[d] and cen < hi[d];
            }
            if ( not inside ) continue;
            double eq[3][3], ex[3][3];
            for (int v=1; v<4; ++v)
                for (int d=0; d<3; ++d)
                {
                    eq[v-1][d] = q[v][d] - q[0][d];
                    double e = x[v][d] - x[0][d];
                    if ( periodic ) e -= boxLen[d] * std::nearbyint( e / boxLen[d] );   // minimum image
                    ex[v-1][d] = e;
                }
            auto det = [](double const a[3][3]) {
                return a[0][0]*(a[1][1]*a[2][2] - a[1][2]*a[2][1])
                     - a[0][1]*(a[1][0]*a[2][2] - a[1][2]*a[2][0])
                     + a[0][2]*(a[1][0]*a[2][1] - a[1][1]*a[2][0]);
            };
            sumLag   += std::fabs( det(eq) );
            sumEuler += std::fabs( det(ex) );
        }
        ++used;
    }
    if ( patchesUsed ) *patchesUsed = used;
    return sumLag > 0. ? sumEuler / sumLag : -1.;
}
#else
double psEstimateMeanStreams(std::vector<Particle_data> const &, double const *, double const *, bool, int *patchesUsed)
{
    if ( patchesUsed ) *patchesUsed = 0;
    return -1.;
}
#endif
