/* PS-DTFE arbitrary-point evaluation (--sample-points).

   Evaluates the multi-stream phase-space fields at user-supplied Eulerian points instead of
   (only) on a regular grid. The triangulation is built in Lagrangian space; every Delaunay
   cell maps to a (possibly folded) Eulerian tetrahedron, and a query point receives one
   stream per tetrahedron that contains it. Per point we report the total density, the
   density-weighted mean velocity, the velocity dispersion tensor and the stream count;
   --per-stream additionally records every stream's own density and velocity.

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
   deterministically sorted records, so results do not depend on partition/thread order). */

#include "../define.h"

#ifdef PHASE_SPACE

#include "../ps_point_eval.h"

#if NO_DIM==3

#include "triangulation_common.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <vector>


namespace psPointEvalDetail {

// One stream (containing tetrahedron) at a query point. Both density variants are kept so
// the estimator choice only affects the reduction, never the collected geometry.
struct StreamRec
{
    double denGeo;              // 'geometric': rho_bar * V_lag / V_eul, constant per tet
    double denDtfe;             // 'dtfe': vertex densities linearly interpolated at the point
    double vel[noVelComp];      // velocity linearly interpolated at the point
};

struct Context
{
    bool   active = false;
    bool   finalized = false;
    bool   perStream = false;
    bool   useGeometric = false;    // variant used for totals, weights and per-stream output
    bool   periodic = false;
    double rhoBar = 1.;             // output densities are rho/rhoBar (grid convention)
    double boxLo[NO_DIM], boxLen[NO_DIM];

    size_t nPoints = 0;
    std::vector<double> pos;        // 3*nPoints, wrapped into the box when periodic

    // uniform bucket grid over the query points (CSR layout)
    int    nB[NO_DIM];
    double bInvW[NO_DIM];
    std::vector<size_t>   bucketStart;   // nB^3 + 1 offsets
    std::vector<uint32_t> bucketPts;     // point indices grouped by bucket

    // per-point stream records, appended by (possibly concurrent) partition workers
    std::vector< std::vector<StreamRec> > recs;

    // finalized outputs
    std::vector<double>   outDen;        // N
    std::vector<double>   outVel;        // 3N
    std::vector<double>   outDisp;       // 6N, upper triangle row-major: xx xy xz yy yz zz
    std::vector<int32_t>  outStreams;    // N
    std::vector<uint64_t> outOffsets;    // N+1 (only with --per-stream)
    std::vector<double>   outRecords;    // 4 per stream: density, vx, vy, vz
};

static Context g_ctx;
static std::mutex g_mergeMutex;   // guards g_ctx.recs during the per-partition merges


// 3x3 double inverse via the adjugate -- the double-precision twin of matrixInverse()
// (math_functions.h), same REAL_PRECISSION zero-guard so the kept-cell set is identical.
inline bool inverse3x3d(double const m[NO_DIM][NO_DIM], double inv[NO_DIM][NO_DIM])
{
    double const c00 = m[1][1]*m[2][2] - m[1][2]*m[2][1];
    double const c01 = m[1][2]*m[2][0] - m[1][0]*m[2][2];
    double const c02 = m[1][0]*m[2][1] - m[1][1]*m[2][0];
    double const det = m[0][0]*c00 + m[0][1]*c01 + m[0][2]*c02;
    if ( std::fabs(det) < REAL_PRECISSION )
        return false;
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
    return true;
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
    g_ctx.useGeometric = userOptions.psStreamDensityGeometric;
    g_ctx.periodic     = userOptions.periodic;
    g_ctx.rhoBar       = double( userOptions.averageDensity );
    for (int d = 0; d < NO_DIM; ++d)
    {
        g_ctx.boxLo[d]  = double( userOptions.region[2*d] );
        g_ctx.boxLen[d] = double( userOptions.region[2*d+1] ) - double( userOptions.region[2*d] );
    }

    readSamplePoints( userOptions.psSamplePointsFile, g_ctx.pos );
    g_ctx.nPoints = g_ctx.pos.size() / NO_DIM;
    if ( g_ctx.nPoints > size_t(0xFFFFFFFFu) )
        throwError( "'--sample-points' supports at most 2^32-1 points." );

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

    g_ctx.recs.resize( g_ctx.nPoints );
    g_ctx.active = true;

    message << "\n" << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset()
            << " point evaluation at " << MESSAGE::cMagenta() << g_ctx.nPoints << MESSAGE::cReset()
            << " sample points from '" << MESSAGE::cBlue() << userOptions.psSamplePointsFile
            << MESSAGE::cReset() << "' (per-stream density: "
            << MESSAGE::cMagenta() << (g_ctx.useGeometric ? "geometric" : "dtfe") << MESSAGE::cReset()
            << (g_ctx.perStream ? ", writing per-stream records" : "") << ").\n" << MESSAGE::Flush;
}


/* Collects this partition's stream contributions for every query point. Called once per
   Lagrangian partition (and once for a single-triangulation run) after vertexDensity().
   The tessellation is read-only here; concurrent partition workers only synchronize on the
   short record merge at the end. */
void interpolatePoints_phaseSpace(DT &dt, User_options &userOptions)
{
    if ( not g_ctx.active || dt.number_of_vertices() < NO_DIM + 1 )
        return;

    Box boxCoordinates = userOptions.region;
    double const avgDens = double( userOptions.averageDensity );

    // local hit buffer; merged into the shared per-point lists once, at the end
    std::vector< std::pair<uint32_t, StreamRec> > hits;

    for (DT::Finite_cells_iterator itC = dt.finite_cells_begin(); itC != dt.finite_cells_end(); ++itC)
    {
        Cell_handle cell = itC;

        // ---- the CPU grid deposit's filters, verbatim (see interpolateGrid_phaseSpace) ----
#ifdef TEST_PADDING
        // Skip cells touching a dummy padding vertex.
        bool hasDummy = false;
        for (int v = 0; v <= NO_DIM; ++v)
            if (cell->vertex(v)->info().isDummy()) { hasDummy = true; break; }
        if (hasDummy) { continue; }
#endif

        // Lagrangian-partition ownership by centroid: tiles without double-counting.
        if ( !userOptions.lagrangianRegion.isNullBox() )
        {
            double cen[NO_DIM];
            for (int d = 0; d < NO_DIM; ++d) cen[d] = 0.;
            for (int v = 0; v <= NO_DIM; ++v)
                for (int d = 0; d < NO_DIM; ++d)
                    cen[d] += double(cell->vertex(v)->point()[d]);
            bool owned = true;
            for (int d = 0; d < NO_DIM; ++d)
            {
                cen[d] /= double(NO_DIM + 1);
                if ( cen[d] < userOptions.lagrangianRegion[2*d] || cen[d] >= userOptions.lagrangianRegion[2*d+1] )
                { owned = false; break; }
            }
            if (!owned) { continue; }
        }

        // zero/negative-density vertex: periodic -> hull artefact, drop; non-periodic -> real
        // cloud surface, use the constant volume-ratio density for the 'dtfe' variant too.
        bool useVolumeRatioDensity = false;
        {
            bool hasBadVertex = false;
            for (int v = 0; v <= NO_DIM; ++v)
                if (cell->vertex(v)->info().density() <= Real(0.)) { hasBadVertex = true; break; }
            if (hasBadVertex)
            {
                if ( userOptions.periodic ) { continue; }
                useVolumeRatioDensity = true;
            }
        }

        // Eulerian vertex positions with the deposit's minimum-image wrap (Real arithmetic,
        // so borderline cells classify identically to the grid path).
        Real eulerPos[NO_DIM+1][NO_DIM];
        for (int v = 0; v <= NO_DIM; ++v)
            for (int d = 0; d < NO_DIM; ++d)
                eulerPos[v][d] = cell->vertex(v)->info().eulerianPosition(d);

        if ( userOptions.periodic )
        {
            Real boxLen[NO_DIM];
            for (int d = 0; d < NO_DIM; ++d)
                boxLen[d] = boxCoordinates[2*d+1] - boxCoordinates[2*d];

            for (int v = 1; v <= NO_DIM; ++v)
                for (int d = 0; d < NO_DIM; ++d)
                {
                    Real diff = eulerPos[v][d] - eulerPos[0][d];
                    if (diff >  boxLen[d] * Real(0.5)) eulerPos[v][d] -= boxLen[d];
                    if (diff < -boxLen[d] * Real(0.5)) eulerPos[v][d] += boxLen[d];
                }
        }

        double Ax[NO_DIM][NO_DIM];
        for (int v = 0; v < NO_DIM; ++v)
            for (int i = 0; i < NO_DIM; ++i)
                Ax[v][i] = double(eulerPos[v+1][i]) - double(eulerPos[0][i]);

        double const cellAbsDet = std::fabs( determinant(Ax) );

        double const DEGENERATE_DET_TOL = 1.e-6;
        {
            double avgEdge2 = 0.;
            for (int v = 0; v < NO_DIM; ++v)
            {
                double len2 = 0.;
                for (int i = 0; i < NO_DIM; ++i)
                    len2 += Ax[v][i] * Ax[v][i];
                avgEdge2 += len2;
            }
            avgEdge2 /= NO_DIM;
            double edgeScale = avgEdge2 * std::sqrt(avgEdge2);
            if (cellAbsDet < DEGENERATE_DET_TOL * edgeScale) { continue; }
        }

        {
            Real posMatInv[NO_DIM][NO_DIM];
            matrixInverse(Ax, posMatInv);
            bool singularInverse = true;
            for (int a = 0; a < NO_DIM && singularInverse; ++a)
                for (int b = 0; b < NO_DIM; ++b)
                    if (posMatInv[a][b] != Real(0.)) { singularInverse = false; break; }
            if (singularInverse) { continue; }
        }
        // ---- end of the shared filter chain ----

        // double-precision inverse for the evaluation (same zero criterion as above)
        double inv[NO_DIM][NO_DIM];
        if ( not inverse3x3d(Ax, inv) ) { continue; }

        // 'geometric' density: rho_bar * V_lag / V_eul (= m_tet * d! / |det Ax|), constant per tet
        double denGeo;
        {
            double Lag[NO_DIM][NO_DIM];
            for (int v = 0; v < NO_DIM; ++v)
                for (int i = 0; i < NO_DIM; ++i)
                    Lag[v][i] = double(cell->vertex(v+1)->point()[i]) - double(cell->vertex(0)->point()[i]);
            denGeo = avgDens * std::fabs(determinant(Lag)) / cellAbsDet;
            if (denGeo < 0.) denGeo = 0.;
        }

        double rho[NO_DIM+1], vel[NO_DIM+1][noVelComp];
        for (int v = 0; v <= NO_DIM; ++v)
        {
            rho[v] = double( cell->vertex(v)->info().density() );
            for (size_t j = 0; j < noVelComp; ++j)
                vel[v][j] = double( cell->vertex(v)->info().velocity(j) );
        }

        // Eulerian bounding box (+ slack for the +-1e-6 barycentric tolerance) -> bucket range
        double eMin[NO_DIM], eMax[NO_DIM], relLo[NO_DIM], relHi[NO_DIM];
        int bLo[NO_DIM], bHi[NO_DIM];
        bool bucketRangeEmpty = false;
        for (int d = 0; d < NO_DIM; ++d)
        {
            eMin[d] = double(eulerPos[0][d]);
            eMax[d] = eMin[d];
            for (int v = 1; v <= NO_DIM; ++v)
            {
                double const c = double(eulerPos[v][d]);
                if (c < eMin[d]) eMin[d] = c;
                if (c > eMax[d]) eMax[d] = c;
            }
            double const margin = 1.e-5 * (eMax[d] - eMin[d]) + 1.e-12;
            relLo[d] = eMin[d] - double(eulerPos[0][d]) - margin;
            relHi[d] = eMax[d] - double(eulerPos[0][d]) + margin;

            bLo[d] = int( std::floor( (eMin[d] - margin - g_ctx.boxLo[d]) * g_ctx.bInvW[d] ) );
            bHi[d] = int( std::floor( (eMax[d] + margin - g_ctx.boxLo[d]) * g_ctx.bInvW[d] ) );
            if ( g_ctx.periodic )
            {
                if ( bHi[d] - bLo[d] + 1 >= g_ctx.nB[d] ) { bLo[d] = 0; bHi[d] = g_ctx.nB[d] - 1; }  // spans every bucket: visit each once
            }
            else
            {
                if ( bLo[d] < 0 ) bLo[d] = 0;
                if ( bHi[d] >= g_ctx.nB[d] ) bHi[d] = g_ctx.nB[d] - 1;
                if ( bLo[d] > bHi[d] ) { bucketRangeEmpty = true; break; }
            }
        }
        if ( bucketRangeEmpty ) { continue; }

        StreamRec rec;
        rec.denGeo = denGeo;

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

                // minimum-image displacement to (wrapped) vertex 0 -- the unwrapped point
                // position relative to the tetrahedron, as the deposit's raw grid indices see it
                double rel[NO_DIM];
                bool inBBox = true;
                for (int d = 0; d < NO_DIM; ++d)
                {
                    double r = g_ctx.pos[size_t(p)*NO_DIM+d] - double(eulerPos[0][d]);
                    if ( g_ctx.periodic )
                    {
                        if (r >  0.5 * g_ctx.boxLen[d]) r -= g_ctx.boxLen[d];
                        if (r < -0.5 * g_ctx.boxLen[d]) r += g_ctx.boxLen[d];
                    }
                    if ( r < relLo[d] || r > relHi[d] ) { inBBox = false; break; }
                    rel[d] = r;
                }
                if ( not inBBox ) continue;

                // barycentric point-in-tet test with the deposit kernels' tolerances
                // (Ax stores edges as rows, so bary = inv^T * rel -- note the [i][v] transpose)
                double bary[NO_DIM], sum = 0.;
                bool inside = true;
                for (int v = 0; v < NO_DIM; ++v)
                {
                    double bc = 0.;
                    for (int i = 0; i < NO_DIM; ++i)
                        bc += inv[i][v] * rel[i];
                    if (bc < -1.e-6) { inside = false; break; }
                    bary[v] = bc;
                    sum += bc;
                }
                if ( !inside || sum > 1. + 1.e-6 ) continue;

                if ( useVolumeRatioDensity )
                    rec.denDtfe = denGeo;   // hull cell: vertex densities are undefined (0)
                else
                {
                    double dd = rho[0];
                    for (int v = 0; v < NO_DIM; ++v)
                        dd += bary[v] * (rho[v+1] - rho[0]);
                    rec.denDtfe = (dd > 0.) ? dd : 0.;   // the +-1e-6 tolerance can graze negative
                }
                for (size_t j = 0; j < noVelComp; ++j)
                {
                    double vv = vel[0][j];
                    for (int v = 0; v < NO_DIM; ++v)
                        vv += bary[v] * (vel[v+1][j] - vel[0][j]);
                    rec.vel[j] = vv;
                }

                hits.push_back( std::make_pair( p, rec ) );
            }
        }
    }   // end cell loop

    if ( hits.empty() )
        return;
    std::lock_guard<std::mutex> lock( g_mergeMutex );
    for (size_t h = 0; h < hits.size(); ++h)
        g_ctx.recs[ hits[h].first ].push_back( hits[h].second );
}


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
        return false;
    };

    g_ctx.outDen.assign( N, 0. );
    g_ctx.outVel.assign( N * noVelComp, 0. );
    g_ctx.outDisp.assign( N * noDispComp, 0. );
    g_ctx.outStreams.assign( N, 0 );
    size_t totalStreams = 0;
    for (size_t i = 0; i < N; ++i)
        totalStreams += g_ctx.recs[i].size();
    if ( g_ctx.perStream )
    {
        g_ctx.outOffsets.assign( N + 1, 0 );
        g_ctx.outRecords.reserve( totalStreams * (1 + noVelComp) );
    }

    size_t covered = 0, multi = 0;
    size_t maxStreams = 0;
    for (size_t i = 0; i < N; ++i)
    {
        std::vector<StreamRec> &recs = g_ctx.recs[i];
        std::sort( recs.begin(), recs.end(), recLess );

        size_t const nS = recs.size();
        g_ctx.outStreams[i] = int32_t( nS );
        if (nS > 0) ++covered;
        if (nS > 1) ++multi;
        if (nS > maxStreams) maxStreams = nS;

        // moments in sorted order (the psDeferNormalization pattern: sums are linear across
        // partitions; the density-weighted means/dispersion are formed exactly once, here)
        double W = 0., mom1[noVelComp] = {0.}, mom2[noDispComp] = {0.};
        for (size_t s = 0; s < nS; ++s)
        {
            double const den = geo ? recs[s].denGeo : recs[s].denDtfe;
            W += den;
            for (size_t j = 0; j < noVelComp; ++j)
                mom1[j] += den * recs[s].vel[j];
            size_t c = 0;
            for (int a = 0; a < NO_DIM; ++a)
                for (int b = a; b < NO_DIM; ++b)
                    mom2[c++] += den * recs[s].vel[a] * recs[s].vel[b];
        }
        g_ctx.outDen[i] = W * invRhoBar;
        if ( W > 0. )
        {
            double const invW = 1. / W;
            for (size_t j = 0; j < noVelComp; ++j)
                g_ctx.outVel[i*noVelComp+j] = mom1[j] * invW;
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
            }
            g_ctx.outOffsets[i+1] = uint64_t( g_ctx.outOffsets[i] ) + uint64_t( nS );
        }

        std::vector<StreamRec>().swap( recs );   // free as we go
    }
    std::vector< std::vector<StreamRec> >().swap( g_ctx.recs );
    g_ctx.finalized = true;

    message << "\n" << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " point evaluation -- "
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

    message << "\n" << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset()
            << " writing the point-evaluation outputs (raw binary, see README):\n" << MESSAGE::Flush;
    writeRaw( ".pts_den",     g_ctx.outDen.data(),     g_ctx.outDen.size()     * sizeof(double),
              "point densities (float64, rho/rho_bar)" );
    writeRaw( ".pts_vel",     g_ctx.outVel.data(),     g_ctx.outVel.size()     * sizeof(double),
              "point mean velocities (float64 x 3)" );
    writeRaw( ".pts_velDisp", g_ctx.outDisp.data(),    g_ctx.outDisp.size()    * sizeof(double),
              "point velocity dispersion tensors (float64 x 6, xx xy xz yy yz zz)" );
    writeRaw( ".pts_streams", g_ctx.outStreams.data(), g_ctx.outStreams.size() * sizeof(int32_t),
              "point stream counts (int32)" );
    if ( g_ctx.perStream )
    {
        writeRaw( ".pts_stream_offsets", g_ctx.outOffsets.data(),
                  g_ctx.outOffsets.size() * sizeof(uint64_t),
                  "per-stream record offsets (uint64, N+1)" );
        writeRaw( ".pts_stream_records", g_ctx.outRecords.data(),
                  g_ctx.outRecords.size() * sizeof(double),
                  "per-stream records (float64 x 4: density, vx, vy, vz; density-descending)" );
    }
}


#else // NO_DIM != 3

// PS-DTFE is only validated in 3D; point evaluation is 3D-only.
#include "../user_options.h"
bool psPointEvalActive() { return false; }
void psPointEvalInit(User_options &)
{
    throwError( "'--sample-points' (PS-DTFE point evaluation) is implemented for 3D only." );
}
void psPointEvalFinalize(User_options const &) {}
void psPointEvalWriteOutputs(User_options const &) {}

#endif // NO_DIM

#endif // PHASE_SPACE
