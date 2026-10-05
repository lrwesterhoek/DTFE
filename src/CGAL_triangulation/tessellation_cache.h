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
#include <CGAL/Unique_hash_map.h>

#include <zlib.h>
#ifdef DTFE_HAVE_LIBDEFLATE
#include <libdeflate.h>
#endif

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <future>
#include <memory>
#include <new>
#include <thread>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <sys/stat.h>
#include <sys/statvfs.h>

#include "../canonical_path.h"
#include "../cache_key_lines.h"   // the build lines and file stamps of the key (shared with auto_tune.h)
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


/* The CHUNKED body (written since 2026-10-02; single-stream files still load). Loading a cached
   partition is inflate + CGAL deserialization, and one zlib stream inflates on one core: ~3 s of the
   8.4 s a 6M-vertex TNG100-3 partition takes. The body is therefore written as independent gzip
   members of CHUNK_BYTES each (compressed in parallel on the write, which also shortens it), followed
   by an index of their sizes in a footer; a load inflates members AHEAD on a few threads into a bounded
   window while the parser drains them in order. Layout after the plaintext header (magic, length,
   descriptor -- unchanged, so header checks, occupancy maps and every '.tess' name stay as they were):
       CHUNK_MARKER (8 bytes)   member_0 ... member_{n-1}   (each a complete gzip member)
       index: uint64 n, then n x (uint64 compressed size, uint64 uncompressed size)
       footer: uint64 index offset (from the file start), CHUNK_FOOTER (8 bytes)
   The members concatenate to a valid multi-member gzip file. DTFE_TESS_CACHE_LEGACY=1 writes the old
   single stream (tests); python/tools/upgrade_tess_cache.py converts old files, byte for byte. */
static const char   CHUNK_MARKER[8] = { 'D','T','F','E','C','H','K','1' };
static const char   CHUNK_FOOTER[8] = { 'D','T','F','E','C','H','K','E' };
static const size_t CHUNK_BYTES     = size_t(16) << 20;

inline std::atomic<int> &activeLoads() { static std::atomic<int> n{ 0 }; return n; }

// threads for one load's or save's chunks: a few, fewer while other loads run (the composite server
// loads up to 8 partitions at once); DTFE_TESS_CACHE_THREADS pins it
inline size_t chunkThreads()
{
    if ( const char *env = std::getenv("DTFE_TESS_CACHE_THREADS") )
    {
        long const v = std::atol( env );
        if ( v > 0 ) return size_t( v );
    }
    unsigned const hw = std::max( 1u, std::thread::hardware_concurrency() );
    int const others = std::max( 0, activeLoads().load() - 1 );
    return std::max<size_t>( 1, std::min<size_t>( 4, size_t(hw) / size_t(1 + others) / 2 ) );
}

/* libdeflate (Makefile: compiled in when installed, LIBDEFLATE=0 forces zlib): the chunks are whole
   buffers of known size, exactly what its one-shot API wants. Measured on a 3.65M-vertex partition
   (1.9 GB of body in 114 chunks): compression at level 1 2.4x faster than zlib and 3% smaller;
   decompression 1.13x faster than macOS's own zlib (Apple tunes it for its chips), more against
   stock zlib. The members stay plain gzip (libdeflate checks the CRC itself), so every build reads
   every cache. DTFE_TESS_CACHE_ZLIB=1 uses zlib for both directions (for comparisons). */
inline bool useLibdeflate()
{
#ifdef DTFE_HAVE_LIBDEFLATE
    static bool const v = not ( std::getenv("DTFE_TESS_CACHE_ZLIB") and std::atoi( std::getenv("DTFE_TESS_CACHE_ZLIB") ) != 0 );
    return v;
#else
    return false;
#endif
}

inline char const *chunkCodecName() { return useLibdeflate() ? "libdeflate" : "zlib"; }

#ifdef DTFE_HAVE_LIBDEFLATE
// one (de)compressor per thread: they are not thread-safe, and cheap to keep
struct LibdeflateFree
{
    void operator()(libdeflate_compressor *c) const { libdeflate_free_compressor( c ); }
    void operator()(libdeflate_decompressor *d) const { libdeflate_free_decompressor( d ); }
};
inline libdeflate_compressor *threadCompressor()
{
    thread_local std::unique_ptr<libdeflate_compressor, LibdeflateFree> c( libdeflate_alloc_compressor( DEFLATE_LEVEL ) );
    return c.get();
}
inline libdeflate_decompressor *threadDecompressor()
{
    thread_local std::unique_ptr<libdeflate_decompressor, LibdeflateFree> d( libdeflate_alloc_decompressor() );
    return d.get();
}
#endif

inline std::vector<char> deflateChunk(std::vector<char> const &in)
{
#ifdef DTFE_HAVE_LIBDEFLATE
    if ( useLibdeflate() )
    {
        std::vector<char> out;
        libdeflate_compressor *c = threadCompressor();
        if ( c == nullptr ) return out;
        out.resize( libdeflate_gzip_compress_bound( c, in.size() ) );
        size_t const n = libdeflate_gzip_compress( c, in.data(), in.size(), out.data(), out.size() );
        if ( n == 0 ) { out.clear(); return out; }
        out.resize( n );
        return out;
    }
#endif
    z_stream z;
    std::memset( &z, 0, sizeof z );
    std::vector<char> out;
    if ( deflateInit2( &z, DEFLATE_LEVEL, Z_DEFLATED, 15 + 16, 8, Z_DEFAULT_STRATEGY ) != Z_OK ) return out;
    out.resize( deflateBound( &z, uLong( in.size() ) ) + 64 );
    z.next_in   = reinterpret_cast<Bytef*>( const_cast<char*>( in.data() ) );
    z.avail_in  = uInt( in.size() );
    z.next_out  = reinterpret_cast<Bytef*>( out.data() );
    z.avail_out = uInt( out.size() );
    int const rc = deflate( &z, Z_FINISH );
    size_t const n = out.size() - z.avail_out;
    deflateEnd( &z );
    if ( rc != Z_STREAM_END ) { out.clear(); return out; }
    out.resize( n );
    return out;
}

// One member, read through its own stream (no shared file position, no pread: the Docker gate is Linux).
inline std::vector<char> inflateChunk(std::string const &path, uint64_t const off, uint64_t const csize,
                                      uint64_t const usize, bool &ok)
{
    ok = false;
    std::vector<char> out;
    std::ifstream f( path.c_str(), std::ios::binary );
    if ( not f ) return out;
    f.seekg( std::streamoff( off ) );
    std::vector<char> in( static_cast<size_t>( csize ) );
    f.read( in.data(), std::streamsize( csize ) );
    if ( not f ) return out;
#ifdef DTFE_HAVE_LIBDEFLATE
    if ( useLibdeflate() )
    {
        libdeflate_decompressor *d = threadDecompressor();
        if ( d == nullptr ) return out;
        out.resize( size_t( usize ) );
        size_t got = 0;
        ok = libdeflate_gzip_decompress( d, in.data(), in.size(), out.data(), out.size(), &got ) == LIBDEFLATE_SUCCESS
             and got == size_t( usize );
        return out;
    }
#endif
    z_stream z;
    std::memset( &z, 0, sizeof z );
    if ( inflateInit2( &z, 15 + 16 ) != Z_OK ) return out;
    out.resize( size_t( usize ) );
    z.next_in   = reinterpret_cast<Bytef*>( in.data() );
    z.avail_in  = uInt( in.size() );
    z.next_out  = reinterpret_cast<Bytef*>( out.data() );
    z.avail_out = uInt( out.size() );
    int const rc = inflate( &z, Z_FINISH );
    ok = ( rc == Z_STREAM_END and z.total_out == usize );
    inflateEnd( &z );
    return out;
}

// The writer: fills CHUNK_BYTES, hands each full chunk to a compressing task (a bounded number in
// flight) and writes the finished members IN ORDER; finish() writes the last one, the index and the footer.
class ChunkDeflateBuf : public std::streambuf
{
public:
    explicit ChunkDeflateBuf(std::ostream &sink) : sink_(sink), cur_(CHUNK_BYTES)
    {
        start_ = uint64_t( sink_.tellp() );
        threads_ = chunkThreads();
        setp( cur_.data(), cur_.data() + cur_.size() );
    }
    ~ChunkDeflateBuf() override { finish(); }
    bool good() const { return ok_; }

    bool finish()
    {
        if ( done_ ) return ok_;
        done_ = true;
        submit();
        while ( not pending_.empty() ) writeFront();
        if ( not ok_ ) return false;
        uint64_t const indexOffset = start_ + written_;
        uint64_t const n = uint64_t( sizes_.size() );
        sink_.write( reinterpret_cast<char const*>(&n), sizeof n );
        for (auto const &cu : sizes_)
        {
            sink_.write( reinterpret_cast<char const*>(&cu.first), sizeof cu.first );
            sink_.write( reinterpret_cast<char const*>(&cu.second), sizeof cu.second );
        }
        sink_.write( reinterpret_cast<char const*>(&indexOffset), sizeof indexOffset );
        sink_.write( CHUNK_FOOTER, 8 );
        if ( not sink_ ) ok_ = false;
        return ok_;
    }

protected:
    int overflow(int c) override
    {
        submit();
        if ( not ok_ ) return traits_type::eof();
        if ( c != traits_type::eof() )
        {
            *pptr() = traits_type::to_char_type( c );
            pbump( 1 );
        }
        return c;
    }
    int sync() override { return ok_ ? 0 : -1; }      // members are whole chunks: a flush emits nothing

private:
    void submit()
    {
        size_t const n = size_t( pptr() - pbase() );
        if ( n == 0 ) return;
        std::vector<char> chunk( cur_.begin(), cur_.begin() + std::ptrdiff_t( n ) );
        size_t const usize = n;
        pending_.push_back( { usize, std::async( std::launch::async, [](std::vector<char> in) { return deflateChunk( in ); },
                                                 std::move( chunk ) ) } );
        while ( pending_.size() > 2 * threads_ ) writeFront();
        setp( cur_.data(), cur_.data() + cur_.size() );
    }
    void writeFront()
    {
        auto item = std::move( pending_.front() );
        pending_.pop_front();
        std::vector<char> const out = item.second.get();
        if ( out.empty() ) { ok_ = false; return; }
        sink_.write( out.data(), std::streamsize( out.size() ) );
        if ( not sink_ ) { ok_ = false; return; }
        sizes_.push_back( { uint64_t( out.size() ), uint64_t( item.first ) } );
        written_ += uint64_t( out.size() );
    }

    std::ostream &sink_;
    std::vector<char> cur_;
    std::deque<std::pair<size_t, std::future<std::vector<char>>>> pending_;
    std::vector<std::pair<uint64_t, uint64_t>> sizes_;       // (compressed, uncompressed) per member
    uint64_t start_ = 0, written_ = 0;
    size_t threads_ = 1;
    bool ok_ = true, done_ = false;
};

// The reader: members inflated ahead (a bounded window of tasks), drained in order.
class ChunkInflateBuf : public std::streambuf
{
public:
    ChunkInflateBuf(std::string path, std::vector<uint64_t> off, std::vector<uint64_t> csize, std::vector<uint64_t> usize)
        : path_(std::move(path)), off_(std::move(off)), csize_(std::move(csize)), usize_(std::move(usize))
    {
        threads_ = chunkThreads();
        setg( nullptr, nullptr, nullptr );
    }
    bool good() const { return ok_; }

protected:
    int underflow() override
    {
        if ( not ok_ ) return traits_type::eof();
        while ( pending_.size() < 2 * threads_ and next_ < off_.size() )
        {
            size_t const i = next_++;
            pending_.push_back( std::async( std::launch::async, [this, i]() -> std::pair<bool, std::vector<char>>
            {
                bool ok = false;
                std::vector<char> v = inflateChunk( path_, off_[i], csize_[i], usize_[i], ok );
                return { ok, std::move( v ) };
            } ) );
        }
        if ( pending_.empty() ) return traits_type::eof();
        auto r = pending_.front().get();
        pending_.pop_front();
        if ( not r.first ) { ok_ = false; return traits_type::eof(); }
        cur_ = std::move( r.second );
        if ( cur_.empty() ) return underflow();
        setg( cur_.data(), cur_.data(), cur_.data() + cur_.size() );
        return traits_type::to_int_type( *gptr() );
    }

private:
    std::string path_;
    std::vector<uint64_t> off_, csize_, usize_;
    std::deque<std::future<std::pair<bool, std::vector<char>>>> pending_;
    std::vector<char> cur_;
    size_t next_ = 0, threads_ = 1;
    bool ok_ = true;
};

// After the descriptor: a chunked body's member table (offsets from the file start), or false for an
// old single-stream body (the stream is then left at the start of the body).
inline bool readChunkIndex(std::istream &is, std::vector<uint64_t> &off, std::vector<uint64_t> &csize,
                           std::vector<uint64_t> &usize)
{
    std::streampos const body = is.tellg();
    char marker[8] = {0};
    is.read( marker, 8 );
    if ( not is or std::memcmp( marker, CHUNK_MARKER, 8 ) != 0 ) { is.clear(); is.seekg( body ); return false; }
    uint64_t const dataStart = uint64_t( body ) + 8;
    is.seekg( -16, std::ios::end );
    uint64_t indexOffset = 0;
    char footer[8] = {0};
    is.read( reinterpret_cast<char*>(&indexOffset), sizeof indexOffset );
    is.read( footer, 8 );
    if ( not is or std::memcmp( footer, CHUNK_FOOTER, 8 ) != 0 ) return false;
    is.seekg( std::streamoff( indexOffset ) );
    uint64_t n = 0;
    is.read( reinterpret_cast<char*>(&n), sizeof n );
    if ( not is or n > (uint64_t(1) << 32) ) return false;
    off.resize( size_t(n) ); csize.resize( size_t(n) ); usize.resize( size_t(n) );
    uint64_t pos = dataStart;
    for (uint64_t i = 0; i < n; ++i)
    {
        is.read( reinterpret_cast<char*>(&csize[size_t(i)]), sizeof(uint64_t) );
        is.read( reinterpret_cast<char*>(&usize[size_t(i)]), sizeof(uint64_t) );
        off[size_t(i)] = pos;
        pos += csize[size_t(i)];
    }
    return bool( is ) and pos == indexOffset;
}


// "<size>:<mtime>" for a file, or "absent" -- cheap identity that changes whenever the input does
inline std::string fileStamp(std::string const &path) { return cacheFileStamp( path ); }


/* The cache identity. EVERYTHING that can change the tessellation or the per-vertex payload must
   appear here -- a value that is missing is a silently stale cache. Compared verbatim on load. */
inline std::string makeDescriptor(User_options const &userOptions)
{
    std::ostringstream d;
    d << "format="   << FORMAT_VERSION << '\n';
#if NO_DIM == 2
    d << "tds2d=ascii17\n";   // 2D triangulations are stored in CGAL's ASCII mode (see tryLoad)
#endif
    // compile-time configuration: changes the payload layout and/or the triangulated coordinates
    // (cache_key_lines.h, shared with the server tuner's scan of this folder)
    d << cacheBuildLines();
    // the inputs themselves
    // canonical paths (canonical_path.h): a relative, symlinked or other-case name of the same file
    // is the same key (it used to be a rebuild)
    d << "input="    << canonicalPath(userOptions.inputFilename) << ' ' << fileStamp(userOptions.inputFilename) << '\n';
    d << "type="     << userOptions.inputFileType << '\n';
    // the per-vertex scalar payload comes from this dataset. Written only when set, so the
    // descriptors -- and the caches -- of every run without it stay what they were.
    if ( not userOptions.scalarDataset.empty() )
        d << "scalarDataset=" << userOptions.scalarDataset << '\n';
#ifdef PHASE_SPACE
    d << "lagrange=" << canonicalPath(userOptions.lagrangianInputFilename) << ' '
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


/* ---------------------------------------------------------------- the triangulation, read in blocks
   CGAL writes a 3D triangulation in binary mode (Triangulation_3.h operator<<, CGAL 6) as: the
   dimension (int), the number of finite vertices n (size_t), each finite vertex's point (three
   doubles; neither the vertex info nor the hierarchy pointers are written), the number of cells m
   (size_t), every cell's four vertex indices (size_t, 0 = the infinite vertex), every cell's four
   neighbour indices (size_t), and nothing per cell after that. Its reader pulls each of those numbers
   through its own istream::read -- a sentry, a mode lookup and a state update per 8 bytes -- which
   made that parsing, not the decompression, most of a load (2.0 of 2.4 s for 3.65M vertices / 24M
   cells; a 14.3M-vertex TNG100-3 partition took ~12 s). This reads the same bytes in blocks and
   builds the data structure through the same calls in the same order (create_vertex in file order,
   create_cell in file order, a vertex's cell = the LAST cell naming it, as CGAL's set_cell leaves
   it), so the loaded triangulation -- its iteration order included -- is CGAL's. Only dimension 3 is
   handled: a cached partition is never flat, and anything else is reported as unreadable (a miss,
   rebuilt). Writing goes the same way (writeTriangulationBlocks: CGAL's bytes, written in blocks).
   DTFE_TESS_CACHE_CGAL_IO=1 uses CGAL's own reader and writer instead (for comparisons). */

template <class Tr> using Tds_of = typename Tr::Triangulation_data_structure;

inline bool useCgalIO()
{
    static bool const v = std::getenv("DTFE_TESS_CACHE_CGAL_IO") and std::atoi( std::getenv("DTFE_TESS_CACHE_CGAL_IO") ) != 0;
    return v;
}

// What CGAL's binary operator<< writes for a 3D triangulation (see above), byte for byte, through
// an 8 MB buffer instead of one ostream::write per number. 1 = written, 0 = not a 3D triangulation
// (nothing written: the caller uses CGAL's writer), -1 = the stream refused bytes.
template <class Tr>
inline int writeTriangulationBlocks(std::streambuf &sb, Tr const &dt)
{
    typedef typename Tr::Vertex_handle Vertex_handle;
    typedef typename Tr::Cell_handle   Cell_handle;
    static_assert( sizeof(std::size_t) == 8, "the cache format stores size_t as 8 bytes" );
    if ( dt.dimension() != 3 ) return 0;

    std::vector<char> buf;
    buf.reserve( size_t(8) << 20 );
    bool good = true;
    auto flush = [&]()
    {
        if ( not buf.empty() and sb.sputn( buf.data(), std::streamsize( buf.size() ) ) != std::streamsize( buf.size() ) ) good = false;
        buf.clear();
    };
    auto put = [&](void const *p, size_t n)
    {
        if ( buf.size() + n > buf.capacity() ) flush();
        char const *c = static_cast<char const*>( p );
        buf.insert( buf.end(), c, c + n );
    };

    Tds_of<Tr> const &tds = dt.tds();
    int const d = dt.dimension();
    std::size_t const n = dt.number_of_vertices();
    put( &d, sizeof d );
    put( &n, sizeof n );
    CGAL::Unique_hash_map<Vertex_handle, std::size_t> V;
    std::size_t i = 0;
    for (auto it = tds.vertices_begin(); it != tds.vertices_end(); ++it, ++i)
    {
        V[it] = i;
        if ( i == 0 ) continue;                           // the infinite vertex: index 0, no point
        double const xyz[3] = { double( it->point().x() ), double( it->point().y() ), double( it->point().z() ) };
        put( xyz, sizeof xyz );
    }
    if ( i != n + 1 or not dt.is_infinite( tds.vertices_begin() ) ) return -1;

    std::size_t const m = tds.number_of_cells();
    put( &m, sizeof m );
    CGAL::Unique_hash_map<Cell_handle, std::size_t> C( 0, m );
    std::size_t j = 0;
    for (auto it = tds.cells_begin(); it != tds.cells_end(); ++it, ++j)
    {
        C[it] = j;
        std::size_t const v4[4] = { V[it->vertex(0)], V[it->vertex(1)], V[it->vertex(2)], V[it->vertex(3)] };
        put( v4, sizeof v4 );
    }
    for (auto it = tds.cells_begin(); it != tds.cells_end(); ++it)
    {
        std::size_t const n4[4] = { C[it->neighbor(0)], C[it->neighbor(1)], C[it->neighbor(2)], C[it->neighbor(3)] };
        put( n4, sizeof n4 );
    }
    flush();
    return good ? 1 : -1;
}

template <class Tr>
inline bool readTriangulationBlocks(std::streambuf &sb, Tr &dt)
{
    typedef typename Tr::Triangulation_data_structure Tds;
    typedef typename Tr::Vertex_handle                Vertex_handle;
    typedef typename Tr::Cell_handle                  Cell_handle;
    typedef typename Tr::Point                        TrPoint;
    static_assert( sizeof(std::size_t) == 8, "the cache format stores size_t as 8 bytes" );

    auto get = [&sb](void *dst, size_t bytes) -> bool
    {
        char *p = static_cast<char*>( dst );
        while ( bytes > 0 )
        {
            std::streamsize const want = std::streamsize( std::min<size_t>( bytes, size_t(1) << 30 ) );
            std::streamsize const got = sb.sgetn( p, want );
            if ( got <= 0 ) return false;
            p += got;
            bytes -= size_t( got );
        }
        return true;
    };

    int d = 0;
    std::size_t n = 0, m = 0;
    if ( not get( &d, sizeof d ) or not get( &n, sizeof n ) ) return false;
    if ( d != 3 or n == 0 or n > ( size_t(1) << 40 ) ) return false;

    Tds &tds = dt.tds();
    tds.clear();                                       // as CGAL's operator>>: the infinite vertex first
    dt.set_infinite_vertex( tds.create_vertex() );
    tds.set_dimension( 3 );

    std::size_t const BLOCK = size_t(1) << 18;         // items per block read (8 MB of cell indices)
    std::vector<Vertex_handle> V( n + 1 );
    V[0] = dt.infinite_vertex();
    {
        std::vector<double> xyz( 3 * std::min( BLOCK, n ) );
        for (std::size_t i = 1; i <= n; )
        {
            std::size_t const k = std::min( BLOCK, n + 1 - i );
            if ( not get( xyz.data(), k * 3 * sizeof(double) ) ) return false;
            for (std::size_t j = 0; j < k; ++j)
            {
                Vertex_handle const v = tds.create_vertex();
                v->set_point( TrPoint( xyz[3*j], xyz[3*j+1], xyz[3*j+2] ) );
                V[i + j] = v;
            }
            i += k;
        }
    }

    if ( not get( &m, sizeof m ) or m == 0 or m > ( size_t(1) << 42 ) ) return false;
    std::vector<Cell_handle> C( m );
    std::vector<std::size_t> idx( 4 * std::min( BLOCK, m ) );
    for (std::size_t i = 0; i < m; )                   // the cells, by their vertices
    {
        std::size_t const k = std::min( BLOCK, m - i );
        if ( not get( idx.data(), k * 4 * sizeof(std::size_t) ) ) return false;
        for (std::size_t j = 0; j < k; ++j)
        {
            Cell_handle const c = tds.create_cell();
            for (int q = 0; q < 4; ++q)
            {
                std::size_t const ik = idx[4*j + q];
                if ( ik > n ) return false;
                c->set_vertex( q, V[ik] );
                V[ik]->set_cell( c );
            }
            C[i + j] = c;
        }
        i += k;
    }
    for (std::size_t i = 0; i < m; )                   // their neighbours
    {
        std::size_t const k = std::min( BLOCK, m - i );
        if ( not get( idx.data(), k * 4 * sizeof(std::size_t) ) ) return false;
        for (std::size_t j = 0; j < k; ++j)
        {
            Cell_handle const c = C[i + j];
            for (int q = 0; q < 4; ++q)
            {
                std::size_t const ik = idx[4*j + q];
                if ( ik >= m ) return false;
                c->set_neighbor( q, C[ik] );
            }
        }
        i += k;
    }
    return true;                                       // the cells carry nothing more in this format
}


// DTFE_TESS_CACHE_FINGERPRINT=1: after a load, print a hash of the loaded STRUCTURE -- every vertex's
// point and the index of the cell it stores, every cell's vertex and neighbour indices, in iteration
// order -- so the tests can show that the block reader builds exactly what CGAL's reader builds (the
// outputs alone could not: a point location gives the same answer from any starting cell).
template <class Tr>
inline void printStructureFingerprint(Tr const &dt)
{
    typedef typename Tr::Vertex_handle Vertex_handle;
    typedef typename Tr::Cell_handle   Cell_handle;
    std::unordered_map<Vertex_handle, uint64_t, CGAL::Handle_hash_function> vi;
    std::unordered_map<Cell_handle, uint64_t, CGAL::Handle_hash_function>   ci;
    uint64_t k = 0;
    for (auto v = dt.all_vertices_begin(); v != dt.all_vertices_end(); ++v) vi[v] = k++;
    k = 0;
    for (auto c = dt.all_cells_begin(); c != dt.all_cells_end(); ++c) ci[c] = k++;
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](void const *p, size_t n)
    {
        unsigned char const *b = static_cast<unsigned char const*>( p );
        for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    };
    for (auto v = dt.all_vertices_begin(); v != dt.all_vertices_end(); ++v)
    {
        if ( not dt.is_infinite( v ) )
        {
            double const xyz[3] = { v->point().x(), v->point().y(), v->point().z() };
            mix( xyz, sizeof xyz );
        }
        uint64_t const c = ci.at( v->cell() );
        mix( &c, sizeof c );
    }
    for (auto c = dt.all_cells_begin(); c != dt.all_cells_end(); ++c)
        for (int q = 0; q < 4; ++q)
        {
            uint64_t const a[2] = { vi.at( c->vertex( q ) ), ci.at( c->neighbor( q ) ) };
            mix( a, sizeof a );
        }
    // on a line of its own: other threads' progress output may have left the current line open
    std::fprintf( stderr, "\nTESS_FINGERPRINT %zu %zu %016llx\n", size_t( dt.number_of_vertices() ),
                  size_t( dt.tds().number_of_cells() ), (unsigned long long)h );
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

    // Everything past the header is deflated: chunked (inflated ahead on a few threads) or one
    // stream (older files); either way the tessellation and payload are parsed from the stream.
    struct LoadCount { LoadCount() { ++activeLoads(); } ~LoadCount() { --activeLoads(); } } loadCount;
    std::vector<uint64_t> off, csize, usize;
    bool const chunked = readChunkIndex( is, off, csize, usize );
    if ( not chunked and is.fail() ) return false;           // a chunk marker with a broken footer/index
    std::unique_ptr<std::streambuf> zbufp;
    if ( chunked )
    {
        is.close();
        zbufp.reset( new ChunkInflateBuf( path, std::move( off ), std::move( csize ), std::move( usize ) ) );
    }
    else
    {
        InflateBuf *ib = new InflateBuf( is );
        zbufp.reset( ib );
        if ( not ib->good() ) return false;
    }
    std::istream zin( zbufp.get() );
    CGAL::IO::set_binary_mode( zin );
    dt.clear();
    bool structureOk = false;
#if NO_DIM == 3     // the block reader is 3D (cells); a 2D build reads through CGAL's own operator
    if ( useCgalIO() )
    {
        zin >> dt;
        structureOk = bool( zin );
    }
    else
    {
        try { structureOk = readTriangulationBlocks( *zbufp, dt ); }
        catch ( std::bad_alloc const & ) { structureOk = false; }
    }
#else
    // 2D: CGAL's Triangulation_data_structure_2 writes its counts with a formatted '<<' even in
    // binary mode ("n m dim" with no separators), so its binary files never read back; the 2D cache
    // uses CGAL's ASCII mode at 17 significant digits instead (an exact round trip for doubles; 2D
    // tessellations are small). The descriptor's 'tds2d=ascii17' line keeps any older 2D file a miss.
    CGAL::IO::set_ascii_mode( zin );
    zin >> dt;
    // the ASCII text ends before its own trailing whitespace: read through the end marker the
    // writer put there, and exactly one '\n' after it, so the binary payload starts in place
    {
        std::string tok;
        zin >> tok;
        structureOk = bool( zin ) and tok == "DTFE2DEND" and zin.get() == '\n';
    }
    CGAL::IO::set_binary_mode( zin );
#endif
    if ( not structureOk or dt.number_of_vertices() == 0 ) { dt.clear(); return false; }
#if NO_DIM == 3
    if ( std::getenv("DTFE_TESS_CACHE_FINGERPRINT") ) printStructureFingerprint( dt );
#endif
    if ( not readPayload( zin, dt ) )           { dt.clear(); return false; }
    return true;
}


// the triangulation part of a cache body: in blocks, or through CGAL (DTFE_TESS_CACHE_CGAL_IO=1, or
// a triangulation that is not 3D); a refused write leaves the stream bad, which save() reports
template <class Tr>
inline void writeTriangulation(std::ostream &zout, Tr const &dt)
{
#if NO_DIM == 3
    int const r = useCgalIO() ? 0 : writeTriangulationBlocks( *zout.rdbuf(), dt );
#else
    int const r = 0;    // 2D: CGAL's own operator<<, in ASCII mode (see tryLoad)
#endif
    if ( r == 0 )
    {
#if NO_DIM == 2
        CGAL::IO::set_ascii_mode( zout );
        std::streamsize const prec = zout.precision( 17 );
        zout << dt;
        zout << "\nDTFE2DEND\n";       // where the binary payload begins (tryLoad reads through it)
        zout.precision( prec );
        CGAL::IO::set_binary_mode( zout );
#else
        zout << dt;
#endif
    }
    else if ( r < 0 ) zout.setstate( std::ios::badbit );
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
        // tessellation and payload go through zlib in independent chunks (ChunkDeflateBuf), or as
        // one stream under DTFE_TESS_CACHE_LEGACY=1 (the format before 2026-10-02, for tests)
        bool const legacy = std::getenv("DTFE_TESS_CACHE_LEGACY") and std::atoi( std::getenv("DTFE_TESS_CACHE_LEGACY") ) != 0;
        if ( not legacy )
        {
            os.write( CHUNK_MARKER, 8 );
            ChunkDeflateBuf zbuf( os );
            std::ostream zout( &zbuf );
            CGAL::IO::set_binary_mode( zout );
            writeTriangulation( zout, dt );
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
        else
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
            writeTriangulation( zout, dt );
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
