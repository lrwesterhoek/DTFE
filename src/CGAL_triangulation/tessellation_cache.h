/* On-disk cache of the Delaunay tessellation and its per-vertex payload ('--tessellation-cache').
 *
 * PROBLEM: building the tessellation is the dominant fixed cost of a run and it is repeated in
 * full every time. Measured on a 96^3-particle (884736) crossed-wave snapshot, PS-DTFE, grid 16
 * so the deposit is negligible: triangulation 5.08 s + vertex density 2.06 s = 7.13 s of an
 * 8.81 s run (81%; the HDF5 read is only ~0.8 s). Re-querying the same snapshot with a different
 * '--sample-points' set, a different grid, or different field flags pays all 7.13 s again --
 * which is most of the cost of the iterative point-evaluation figure workflow.
 *
 * MECHANISM: after the tessellation is built AND its vertex densities computed, the whole thing
 * is written to '<dir>/<stem>.tess'; a later run whose inputs are identical loads it instead of
 * rebuilding. The structure travels through CGAL's own binary stream operators (which rebuild the
 * TDS adjacencies directly rather than re-inserting points -- measured 3.56x faster than the
 * spatial_sort+insert path at 1.5e6 vertices), followed by the vertexData payload as flat arrays
 * in finite-vertex-iteration order. Nothing downstream changes: both PS consumers (the grid
 * deposit in ps_interpolation.cc and the point evaluation in ps_point_eval.cc) only ever iterate
 * finite cells and vertices, so a loaded tessellation is indistinguishable from a built one.
 *
 * CORRECTNESS: a cache that is silently stale would corrupt the science, so the identity of every
 * input that can change the tessellation or its payload is written into the file as a plain-text
 * DESCRIPTOR and compared VERBATIM on load (see makeDescriptor: the input files' size+mtime, the
 * geometry, the partition, the sub-sampling, and the compile-time configuration). Any difference
 * -- including a different NO_DIM, Real width or PHASE_SPACE/VELOCITY/SCALAR build -- is a miss,
 * and a miss simply rebuilds. The cache is never consulted unless the user passes the flag.
 *
 * COST: the body is DEFLATED, which brings 1.5e6 vertices from 735 MB to 225 MB -- about 0.15 KB
 * per vertex, and the Lagrangian padding multiplies particle counts by ~4x. Reading is no slower
 * (the smaller read pays for the inflation); WRITING costs ~2.6 s more than an uncompressed save at
 * this size, which is why the level is 1 and not 6 (see DEFLATE_LEVEL). The caller still checks free
 * space and declines to write rather than filling the volume.
 */

#ifndef TESSELLATION_CACHE_HEADER
#define TESSELLATION_CACHE_HEADER

#include <CGAL/IO/io.h>

#include <zlib.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <sys/statvfs.h>

#include "../define.h"
#include "../message.h"
#include "../user_options.h"


namespace TessellationCache
{

// bumped whenever the on-disk layout changes, so old files miss instead of misreading
static const uint32_t FORMAT_VERSION = 2;   // 2 = deflated body (v1 files simply miss)
static const char     MAGIC[8]       = { 'D','T','F','E','T','E','S','S' };

/* Measured bytes per tessellation vertex, for the free-space check. The body is deflated, which
   takes 735 MB down to 214 MB at 1.5e6 vertices -- about 150 B/vertex. The figure is deliberately
   rounded up: it gates whether the write is attempted at all, and over-estimating merely declines a
   marginal case, while under-estimating fills the volume. */
static const double BYTES_PER_VERTEX = 176.;

/* zlib level 1, chosen on the WRITE cost. All three axes measured on the 884736-particle snapshot
   (grid 48, density+velocity; the uncached run is 8.77 s):
                     file     write     hit
     level 6         214 MB   39.8 s    3.41 s
     level 1         225 MB   15.4 s    3.43 s
   Level 6 buys 5% of file size for 24 extra seconds on every cache WRITE -- it would make the first
   run nearly 5x the uncached one, which defeats the point of a cache people are meant to leave on.
   Level 1 costs ~2.6 s over an uncompressed save and reads just as fast. (Judging this on size and
   inflate time alone, as an earlier revision of this comment did, picks the wrong level: those two
   axes both favour 6 and neither of them is the one that hurts.) */
static const int    DEFLATE_LEVEL    = 1;
static const size_t ZBUF             = 256u * 1024u;


/* ---------------------------------------------------------------- deflate/inflate stream buffers
   The tessellation body goes through zlib rather than straight to the file. Compression pays twice:
   the file shrinks ~3.4x, and because the read is I/O-bound the HIT gets no slower -- inflating
   214 MB measured 0.86 s against 1.26 s to read the uncompressed 735 MB.
   These are streambufs, not a compress-the-whole-buffer helper, because a whole-file buffer would
   double peak memory exactly on the large runs the cache is for (a 2e7-particle snapshot caches
   tens of GB). Both work in fixed ZBUF chunks, so memory is flat regardless of tessellation size.
   The HEADER stays uncompressed: a staleness check then costs one small read and no inflation. */
class DeflateBuf : public std::streambuf
{
public:
    explicit DeflateBuf(std::ostream &sink) : sink_(sink), in_(ZBUF), out_(ZBUF), ok_(true)
    {
        std::memset(&z_, 0, sizeof z_);
        // 15+16 = gzip wrapper, so a cache file is also readable with gunzip when debugging
        ok_ = deflateInit2(&z_, DEFLATE_LEVEL, Z_DEFLATED, 15 + 16, 8, Z_DEFAULT_STRATEGY) == Z_OK;
        setp(in_.data(), in_.data() + in_.size());
    }
    ~DeflateBuf() override { finish(); if (ok_) deflateEnd(&z_); }

    // Flushes the put area and terminates the deflate stream. Safe to call twice.
    bool finish()
    {
        if (!ok_ || done_) return ok_;
        if (!pump(Z_NO_FLUSH)) return false;
        done_ = true;
        return pump(Z_FINISH);
    }
    bool good() const { return ok_; }

protected:
    int overflow(int c) override
    {
        if (!pump(Z_NO_FLUSH)) return traits_type::eof();
        if (c != traits_type::eof())
        {
            *pptr() = traits_type::to_char_type(c);
            pbump(1);
        }
        return c;
    }
    int sync() override { return pump(Z_NO_FLUSH) ? 0 : -1; }

private:
    // Hands the put area to zlib and writes every full output block to the sink.
    bool pump(int flush)
    {
        if (!ok_) return false;
        z_.next_in  = reinterpret_cast<Bytef*>(pbase());
        z_.avail_in = uInt(pptr() - pbase());
        do {
            z_.next_out  = reinterpret_cast<Bytef*>(out_.data());
            z_.avail_out = uInt(out_.size());
            int const rc = deflate(&z_, flush);
            if (rc == Z_STREAM_ERROR) { ok_ = false; return false; }
            std::streamsize const n = std::streamsize(out_.size() - z_.avail_out);
            if (n > 0) sink_.write(out_.data(), n);
            if (!sink_) { ok_ = false; return false; }
        } while (z_.avail_out == 0 || (flush == Z_FINISH && z_.avail_in > 0));
        setp(in_.data(), in_.data() + in_.size());
        return true;
    }

    std::ostream      &sink_;
    std::vector<char>  in_, out_;
    z_stream           z_{};
    bool               ok_;
    bool               done_ = false;
};


class InflateBuf : public std::streambuf
{
public:
    explicit InflateBuf(std::istream &src) : src_(src), in_(ZBUF), out_(ZBUF), ok_(true)
    {
        std::memset(&z_, 0, sizeof z_);
        ok_ = inflateInit2(&z_, 15 + 16) == Z_OK;      // gzip wrapper, matching DeflateBuf
        setg(out_.data(), out_.data(), out_.data());
    }
    ~InflateBuf() override { if (ok_) inflateEnd(&z_); }
    bool good() const { return ok_; }

protected:
    int underflow() override
    {
        if (!ok_ || eos_) return traits_type::eof();
        for (;;)
        {
            if (z_.avail_in == 0)
            {
                src_.read(in_.data(), std::streamsize(in_.size()));
                z_.avail_in = uInt(src_.gcount());
                z_.next_in  = reinterpret_cast<Bytef*>(in_.data());
                if (z_.avail_in == 0) { eos_ = true; return traits_type::eof(); }
            }
            z_.next_out  = reinterpret_cast<Bytef*>(out_.data());
            z_.avail_out = uInt(out_.size());
            int const rc = inflate(&z_, Z_NO_FLUSH);
            if (rc != Z_OK && rc != Z_STREAM_END) { ok_ = false; return traits_type::eof(); }
            size_t const n = out_.size() - z_.avail_out;
            if (n > 0)
            {
                setg(out_.data(), out_.data(), out_.data() + n);
                return traits_type::to_int_type(*gptr());
            }
            if (rc == Z_STREAM_END) { eos_ = true; return traits_type::eof(); }
        }
    }

private:
    std::istream      &src_;
    std::vector<char>  in_, out_;
    z_stream           z_{};
    bool               ok_;
    bool               eos_ = false;
};


// "<size>:<mtime>" for a file, or "absent" -- cheap identity that changes whenever the input does
inline std::string fileStamp(std::string const &path)
{
    if ( path.empty() ) return "absent";
    struct stat st;
    if ( ::stat( path.c_str(), &st ) != 0 ) return "missing";
    std::ostringstream os;
    os << (long long)st.st_size << ':' << (long long)st.st_mtime;
    return os.str();
}


/* The cache identity. EVERYTHING that can change the tessellation or the per-vertex payload must
   appear here -- a value that is missing is a silently stale cache. Compared verbatim on load. */
inline std::string makeDescriptor(User_options const &userOptions)
{
    std::ostringstream d;
    d << "format="   << FORMAT_VERSION << '\n';
    // compile-time configuration: changes the payload layout and/or the triangulated coordinates
    d << "dim="      << NO_DIM << " real=" << sizeof(Real) << " velComp=" << noVelComp
      << " scalarComp=" << noScalarComp << '\n';
    d << "build="
#ifdef PHASE_SPACE
      << "PS"
#else
      << "std"
#endif
#ifdef VELOCITY
      << "+vel"
#endif
#ifdef SCALAR
      << "+scalar"
#endif
#ifdef TEST_PADDING
      << "+testpad"
#endif
      << '\n';
    // the inputs themselves
    d << "input="    << userOptions.inputFilename << ' ' << fileStamp(userOptions.inputFilename) << '\n';
    d << "type="     << userOptions.inputFileType << '\n';
    // the per-vertex scalar payload comes from this dataset. Written only when set, so the
    // descriptors -- and the caches -- of every run without it stay what they were.
    if ( not userOptions.scalarDataset.empty() )
        d << "scalarDataset=" << userOptions.scalarDataset << '\n';
#ifdef PHASE_SPACE
    d << "lagrange=" << userOptions.lagrangianInputFilename << ' '
      << fileStamp(userOptions.lagrangianInputFilename) << '\n';
#endif
    // geometry: box, region and padding all decide which particles enter this triangulation
    d << "MpcUnit="  << userOptions.MpcValue << '\n';
    // periodic data gets image copies in its padding (standard: Eulerian, PS: Lagrangian), so the
    // same input with and without '--periodic' is a DIFFERENT tessellation. This line was missing
    // until 2026-09-30 -- a cache written by one mode was loaded by the other. Its addition makes
    // every older cache a (harmless) miss.
    d << "periodic=" << (userOptions.periodicInput?1:0) << '\n';
    d << "box=";
    for (size_t i=0; i<userOptions.boxCoordinates.coords.size(); ++i) d << userOptions.boxCoordinates.coords[i] << ',';
    d << '\n';
    d << "region=" << (userOptions.regionOn?1:0) << (userOptions.regionMpcOn?1:0) << ' ';
    for (size_t i=0; i<userOptions.region.coords.size(); ++i) d << userOptions.region.coords[i] << ',';
    d << '\n';
    d << "padding=";
    for (size_t i=0; i<userOptions.paddingLength.coords.size(); ++i) d << userOptions.paddingLength.coords[i] << ',';
    d << '\n';
    // which partition this file holds
    d << "partition=" << (userOptions.partitionOn?1:0) << ' ';
    for (size_t i=0; i<userOptions.partition.size(); ++i) d << userOptions.partition[i] << ',';
    d << " partNo=" << userOptions.partNo << '\n';
#ifdef PHASE_SPACE
    /* THE per-partition identity in the PS path. 'partNo' stays -1 for every partition of a
       Lagrangian split (DTFE.cpp: psPartitioned = partitionOn and partNo<0), so without this line
       all partitions would share one descriptor and partitions 2..N would load the FIRST
       partition's tessellation. The unpadded Lagrangian region is what selects the particles:
       the padded box is lagRegion +- lagPadding, and both the padding and the periodic-copy rule
       are fixed by options already recorded above, so region + globals determine the vertex set
       exactly. The two partition-path switches ride along because they change how the per-tet
       quantities are normalized before they reach the cached vertices. */
    d << "lagRegion=";
    for (size_t i=0; i<userOptions.lagrangianRegion.coords.size(); ++i)
        d << userOptions.lagrangianRegion.coords[i] << ',';
    d << " subgrid=" << (userOptions.psUseSubgrid?1:0)
      << " deferNorm=" << (userOptions.psDeferNormalization?1:0) << '\n';
#endif
    // anything that changes WHICH particles are triangulated, or their coordinates. The random
    // SEED belongs here only when '--randomSample' is on, because that is the one case where it
    // decides which particles enter the triangulation (random.cc); otherwise it merely seeds the
    // method-2/3 Monte-Carlo interpolation, which runs after the cached stages -- and since an
    // unspecified '--seed' is std::rand() per run, including it unconditionally would make every
    // lookup a miss.
    d << "randomSample=" << userOptions.randomSample << '\n';
    if ( userOptions.randomSample >= Real(0.) )
        d << "sampleSeed=" << userOptions.randomSeed << '\n';
    d << "poisson="  << userOptions.poisson << '\n';
    d << "redshiftSpace=" << (userOptions.transformToRedshiftSpaceOn?1:0) << ' ';
    for (size_t i=0; i<userOptions.transformToRedshiftSpace.size(); ++i)
        d << userOptions.transformToRedshiftSpace[i] << ',';
    d << '\n';
    // vertex densities live in the file, and they are normalized by this
    d << "density0=" << userOptions.averageDensity << '\n';
    return d.str();
}


/* '<dir>/tess_<descriptor hash>_p<partNo>.tess'. The name is derived from the DESCRIPTOR, i.e. from
   the inputs -- never from the output name: the whole point is that a second run which writes
   different outputs (a new '--sample-points' set, another grid, other fields) reuses the same
   tessellation. The hash only locates the file; the full descriptor stored inside is still compared
   verbatim before it is used, so a collision cannot produce a wrong load. */
inline std::string cachePath(User_options const &userOptions)
{
    std::string const desc = makeDescriptor( userOptions );
    uint64_t h = 1469598103934665603ull;                  // FNV-1a
    for (size_t i=0; i<desc.size(); ++i)
    {
        h ^= static_cast<unsigned char>( desc[i] );
        h *= 1099511628211ull;
    }
    char hex[17];
    std::snprintf( hex, sizeof hex, "%016llx", (unsigned long long)h );
    std::ostringstream os;
    os << userOptions.tessellationCacheDir << "/tess_" << hex << "_p" << userOptions.partNo << ".tess";
    return os.str();
}


/* Is the cache usable for this run at all?
 *
 * Multi-partition runs ARE cached: each partition triangulates a different subset, and the subset
 * is identified in the descriptor by its unpadded Lagrangian region (see makeDescriptor). One file
 * per partition therefore, keyed by that region rather than by 'partNo' -- which stays -1 for every
 * partition of a Lagrangian split (DTFE.cpp: psPartitioned = partitionOn and partNo<0) and cannot
 * tell them apart.
 *
 * The one case still refused is a partitioned run of the STANDARD binary, where no Lagrangian
 * region exists to distinguish the sub-domains and 'partNo' is only set in the explicit
 * '--partNo' mode. Rather than reason about whether the processor split is fully pinned by the
 * region/padding already in the descriptor, that combination simply does not cache.
 *
 * This must be checked HERE and not only at option parsing: the auto-tuner turns partitioning on
 * at RUNTIME (auto_tune.h), long after the flags are validated.
 */
inline bool usable(User_options const &userOptions)
{
#ifdef PHASE_SPACE
    (void)userOptions;
    return true;
#else
    return not ( userOptions.partitionOn and userOptions.partNo < 0 );
#endif
}


// warn once per run that the flag is inert, so a partitioned run does not look like it cached
inline void warnUnusable(User_options const &userOptions)
{
    static bool warned = false;
    if ( warned ) return;
    warned = true;
    MESSAGE::Warning warning( userOptions.verboseLevel );
    warning << "--tessellation-cache is ignored for this run: the standard binary splits the box "
               "into several triangulations here (--partition, possibly chosen by the auto-tuner) "
               "and they are not separately identified in the cache key. Phase-space runs cache "
               "every partition; the results are unaffected either way.\n" << MESSAGE::EndWarning;
}


// free bytes on the volume holding 'dir' (0 when it cannot be determined)
inline double freeBytes(std::string const &dir)
{
    struct statvfs vfs;
    if ( ::statvfs( dir.c_str(), &vfs ) != 0 ) return 0.;
    return double(vfs.f_bavail) * double(vfs.f_frsize);
}


/* ---------------------------------------------------------------- payload (de)serialization
   vertexData is written through its public accessors as flat per-field arrays in
   finite-vertex-iteration order: one contiguous block per field, so the layout is explicit and
   independent of struct padding. The order is identical on write and read, and CGAL's stream
   round-trip preserves the finite-vertex order (asserted by the count check on load). */

inline void writePayload(std::ostream &os, DT &dt)
{
    size_t const n = dt.number_of_vertices();

    std::vector<Real>     reals;   // weight, density [, velocity][, scalar][, eulerianPos]
    std::vector<uint8_t>  flags;   // bit0 = dummy, bit1 = dummyNeighbor
    size_t perVertex = 2;
#ifdef VELOCITY
    perVertex += noVelComp;
#endif
#ifdef SCALAR
    perVertex += noScalarComp;
#endif
#ifdef PHASE_SPACE
    perVertex += NO_DIM;
    std::vector<uint64_t> ids;
    std::vector<int32_t>  degs;
    ids.reserve( n );
    degs.reserve( n );
#endif
    reals.reserve( n * perVertex );
    flags.reserve( n );

    for (DT::Finite_vertices_iterator v = dt.finite_vertices_begin(); v != dt.finite_vertices_end(); ++v)
    {
        vertexData &info = v->info();
        reals.push_back( info.weight() );
        reals.push_back( info.density() );
#ifdef VELOCITY
        for (size_t c=0; c<noVelComp; ++c) reals.push_back( info.velocity(int(c)) );
#endif
#ifdef SCALAR
        for (size_t c=0; c<noScalarComp; ++c) reals.push_back( info.scalar()[c] );
#endif
#ifdef PHASE_SPACE
        for (int c=0; c<NO_DIM; ++c) reals.push_back( info.eulerianPosition(c) );
        ids.push_back( info.particleID() );
        degs.push_back( info.psDegree() );
#endif
        flags.push_back( uint8_t( (info.isDummy()?1:0) | (info.hasDummyNeighbor()?2:0) ) );
    }

    uint64_t const count = uint64_t( n );
    os.write( reinterpret_cast<char const*>(&count), sizeof count );
    os.write( reinterpret_cast<char const*>(reals.data()), std::streamsize( reals.size()*sizeof(Real) ) );
    os.write( reinterpret_cast<char const*>(flags.data()), std::streamsize( flags.size()*sizeof(uint8_t) ) );
#ifdef PHASE_SPACE
    os.write( reinterpret_cast<char const*>(ids.data()),  std::streamsize( ids.size()*sizeof(uint64_t) ) );
    os.write( reinterpret_cast<char const*>(degs.data()), std::streamsize( degs.size()*sizeof(int32_t) ) );
#endif
}


// returns false when the payload does not match the loaded structure (treated as a cache miss)
inline bool readPayload(std::istream &is, DT &dt)
{
    uint64_t count = 0;
    is.read( reinterpret_cast<char*>(&count), sizeof count );
    if ( !is or count != uint64_t(dt.number_of_vertices()) ) return false;

    size_t perVertex = 2;
#ifdef VELOCITY
    perVertex += noVelComp;
#endif
#ifdef SCALAR
    perVertex += noScalarComp;
#endif
#ifdef PHASE_SPACE
    perVertex += NO_DIM;
#endif
    size_t const nVert = static_cast<size_t>( count );
    std::vector<Real>    reals( nVert * perVertex );
    std::vector<uint8_t> flags( nVert );
    is.read( reinterpret_cast<char*>(reals.data()), std::streamsize( reals.size()*sizeof(Real) ) );
    is.read( reinterpret_cast<char*>(flags.data()), std::streamsize( flags.size()*sizeof(uint8_t) ) );
#ifdef PHASE_SPACE
    std::vector<uint64_t> ids( nVert );
    std::vector<int32_t>  degs( nVert );
    is.read( reinterpret_cast<char*>(ids.data()),  std::streamsize( ids.size()*sizeof(uint64_t) ) );
    is.read( reinterpret_cast<char*>(degs.data()), std::streamsize( degs.size()*sizeof(int32_t) ) );
#endif
    if ( !is ) return false;

    size_t k = 0, r = 0;
    for (DT::Finite_vertices_iterator v = dt.finite_vertices_begin(); v != dt.finite_vertices_end(); ++v, ++k)
    {
        vertexData &info = v->info();
        // restore the dummy flags FIRST: setDummy() zeroes the density, so it must not run after it
        if ( flags[k] & 1 )      info.setDummy();
        else if ( flags[k] & 2 ) info.setDummyNeighbor();

        info.weight()  = reals[r++];
        info.density() = reals[r++];
#ifdef VELOCITY
        for (size_t c=0; c<noVelComp; ++c) info.velocity(int(c)) = reals[r++];
#endif
#ifdef SCALAR
        for (size_t c=0; c<noScalarComp; ++c) info.scalar()[c] = reals[r++];
#endif
#ifdef PHASE_SPACE
        for (int c=0; c<NO_DIM; ++c) info.eulerianPosition(c) = reals[r++];
        info.setParticleID( ids[k] );
        info.psDegree() = degs[k];
#endif
    }
    return k == nVert;
}


/* ---------------------------------------------------------------- the two entry points */

// True when this partition's cache file exists and was written for exactly these inputs: the
// header check of tryLoad, without reading (or inflating) the body. The composite '--serve'
// startup uses it to register a partition without loading its tessellation.
inline bool headerMatches(User_options const &userOptions)
{
    if ( userOptions.tessellationCacheDir.empty() or not usable( userOptions ) ) return false;
    std::ifstream is( cachePath( userOptions ).c_str(), std::ios::binary );
    if ( !is ) return false;
    char magic[8] = {0};
    is.read( magic, 8 );
    if ( !is or std::memcmp( magic, MAGIC, 8 ) != 0 ) return false;
    uint32_t len = 0;
    is.read( reinterpret_cast<char*>(&len), sizeof len );
    if ( !is or len == 0 or len > (1u<<20) ) return false;
    std::string stored( static_cast<size_t>(len), '\0' );
    is.read( &stored[0], std::streamsize(len) );
    return is and stored == makeDescriptor( userOptions );
}


// Loads the tessellation for this partition into 'dt'. Returns false (a miss) whenever the file is
// absent, unreadable, or was written for different inputs -- the caller then builds as usual.
inline bool tryLoad(DT &dt, User_options const &userOptions)
{
    if ( userOptions.tessellationCacheDir.empty() ) return false;
    if ( not usable( userOptions ) ) { warnUnusable( userOptions ); return false; }

    std::string const path = cachePath( userOptions );
    std::ifstream is( path.c_str(), std::ios::binary );
    if ( !is ) return false;

    char magic[8] = {0};
    is.read( magic, 8 );
    if ( !is or std::memcmp( magic, MAGIC, 8 ) != 0 ) return false;

    uint32_t len = 0;
    is.read( reinterpret_cast<char*>(&len), sizeof len );
    if ( !is or len == 0 or len > (1u<<20) ) return false;
    std::string stored( static_cast<size_t>(len), '\0' );
    is.read( &stored[0], std::streamsize(len) );
    if ( !is ) return false;
    std::string const current = makeDescriptor( userOptions );
    if ( stored != current )
    {
        // A miss is normal (different inputs), but "why does my cache never hit?" is otherwise
        // impossible to answer, so name the first line that differs at the highest verbosity.
        MESSAGE::Message message( userOptions.verboseLevel );
        std::istringstream a( stored ), b( current );
        std::string la, lb;
        while ( std::getline( a, la ) )
        {
            if ( not std::getline( b, lb ) or la != lb )
            {
                message << "\t --tessellation-cache MISS: cached '" << la << "' vs current '" << lb << "'\n"
                        << MESSAGE::Flush;
                break;
            }
        }
        return false;   // stale: rebuild
    }

    // Everything past the header is deflated; wrap the file in an inflating stream and read the
    // tessellation and payload from that exactly as before.
    InflateBuf zbuf( is );
    if ( not zbuf.good() ) return false;
    std::istream zin( &zbuf );
    CGAL::IO::set_binary_mode( zin );
    dt.clear();
    zin >> dt;
    if ( !zin or dt.number_of_vertices() == 0 ) { dt.clear(); return false; }
    if ( not readPayload( zin, dt ) )           { dt.clear(); return false; }
    return true;
}


// Writes the tessellation for this partition. Never fatal: on any problem it warns and the run
// continues (the cache is an optimization, not a result).
inline void save(DT &dt, User_options const &userOptions)
{
    if ( userOptions.tessellationCacheDir.empty() ) return;
    if ( not usable( userOptions ) ) { warnUnusable( userOptions ); return; }

    static const double GB = 1024.*1024.*1024.;
    MESSAGE::Warning warning( userOptions.verboseLevel );
    double const need = double(dt.number_of_vertices()) * BYTES_PER_VERTEX;
    double const have = freeBytes( userOptions.tessellationCacheDir );
    if ( have > 0. and need > 0.95*have )
    {
        warning << "--tessellation-cache: not writing ~" << need/GB << " GB for "
                << dt.number_of_vertices() << " vertices -- only " << have/GB
                << " GB free in '" << userOptions.tessellationCacheDir
                << "'. The run is unaffected; free space or drop the flag.\n" << MESSAGE::EndWarning;
        return;
    }

    std::string const path = cachePath( userOptions );
    std::string const tmp  = path + ".part";     // write aside, rename: a crash leaves no half file
    {
        std::ofstream os( tmp.c_str(), std::ios::binary | std::ios::trunc );
        if ( !os )
        {
            warning << "--tessellation-cache: cannot write '" << tmp << "'; continuing without caching.\n"
                    << MESSAGE::EndWarning;
            return;
        }
        std::string const desc = makeDescriptor( userOptions );
        uint32_t const len = uint32_t( desc.size() );
        os.write( MAGIC, 8 );
        os.write( reinterpret_cast<char const*>(&len), sizeof len );
        os.write( desc.data(), std::streamsize(desc.size()) );

        // The header above stays uncompressed so a staleness check costs one small read; the
        // tessellation and payload go through zlib (see DeflateBuf).
        {
            DeflateBuf zbuf( os );
            if ( not zbuf.good() )
            {
                warning << "--tessellation-cache: could not start the compressor; continuing without caching.\n"
                        << MESSAGE::EndWarning;
                os.close();
                ::remove( tmp.c_str() );
                return;
            }
            std::ostream zout( &zbuf );
            CGAL::IO::set_binary_mode( zout );
            zout << dt;
            writePayload( zout, dt );
            zout.flush();
            if ( not zbuf.finish() or not zout )
            {
                warning << "--tessellation-cache: compressing '" << tmp << "' failed; continuing.\n"
                        << MESSAGE::EndWarning;
                os.close();
                ::remove( tmp.c_str() );
                return;
            }
        }
        if ( !os )
        {
            warning << "--tessellation-cache: write of '" << tmp << "' failed (out of space?); continuing.\n"
                    << MESSAGE::EndWarning;
            os.close();
            ::remove( tmp.c_str() );
            return;
        }
    }
    if ( ::rename( tmp.c_str(), path.c_str() ) != 0 )
    {
        ::remove( tmp.c_str() );
        warning << "--tessellation-cache: could not finalize '" << path << "'; continuing.\n"
                << MESSAGE::EndWarning;
    }
}

}   // namespace TessellationCache

#endif
