/*  Resume for a killed partitioned PS-DTFE run (2026-10-11). The TNG50 z=0 GPU run takes 8 h; a kill at 7 h
    (a restart, the T7 dropping off, an out-of-memory SIGKILL) threw every merged partition away. Now the
    partition loop CHECKPOINTS its merged accumulators: after a partition has been added into the full grids
    (DTFE.cpp, under the merge mutex, so the grids are not changing), the grids and the list of merged
    partitions go to

        <output>.runlog.checkpoint/checkpoint                        (beside the outputs), or, with --scratch-dir,
        <scratch>/<output basename>.<FNV-1a 64 of the absolute output path, 16 hex>.runlog.checkpoint/checkpoint

    (the scratch name carries the whole path's hash: run_ps_dtfe.sh's snapshots all write '<snapdir>/ps_output',
    so a basename would let one snapshot's checkpoint meet another's, and the flattened path itself was longer
    than a file name may be; config.sh ps_checkpoint_dir computes the same name). '--resume' loads it, skips the
    partitions it lists and merges the rest; the directory is removed once the outputs are in place.

    ATOMIC: ONE file holds the manifest (a text header) AND the grids, written as 'checkpoint.tmp.<pid>' and
    renamed over 'checkpoint'. A kill at any moment leaves either the previous checkpoint or the new one,
    never a manifest that disagrees with its grids (a manifest listing fewer partitions than the grids hold
    would make the resume add those partitions twice).

    CHEAP: a checkpoint is written after a merge unless the previous one took more than 5% of the time since
    it was written (the write time is measured), so the overhead stays near 5% at worst whatever the grid
    and partition sizes: a 1024^3 run's multi-GB write happens every few partitions, a small run's after
    every partition. Measured on the 64-partition synthetic run of tests/ps_resume_check.sh: see notes.

    SAFE: the manifest carries the run's SETTINGS KEY: the build configuration and the binary (its revision and
    link time: a rebuilt binary does not resume), the command line minus the
    options that cannot change a grid value (--resume, --ordered-merge, --ps-no-overlap, --verbose,
    --max-concurrent, --scratch-dir, --tessellation-cache), the input files' size+mtime, the resolved
    partition split and the grid. '--resume' refuses, naming the lines that differ, when the key differs.
    It over-refuses rather than under-refuses (a reordered command line is a different key).

    MERGE ORDER: the manifest lists the merged partitions as a SET, so a run merging in completion order
    (the default memory-driven split) can be resumed too. '--resume' turns '--ordered-merge' on for the rest
    of the run; when the killed run merged in index order too ('--ordered-merge', or '--max-concurrent 1'),
    the resumed grids are byte-identical to an uninterrupted ordered run (tests/ps_resume_check.sh) -- at the
    same '--max-concurrent' and thread count: the CPU deposit inside a partition splits its sums over the
    threads a partition gets, so another count changes the grids at float rounding (measured: '--ordered-merge'
    at --max-concurrent 2 and 4 differ in '.den' by ~1e-7 relative, 2026-10-11). The key leaves the two out on
    purpose: a resume at a lower --max-concurrent (the run was killed for memory) must stay possible.

    ROOM: a write needs the grids' bytes + 1 GB free while it goes (the new file beside the old one), and,
    counting the old file's bytes as free, twice the grids' bytes + 2 GB (the checkpoint and the outputs still
    to be written: on the T7 the checkpoint must never be what makes their write fail);
    short of that the write is skipped with a warning (once) and the previous checkpoint stays.

    NOT CHECKPOINTED: '--sample-points' runs (their per-point accumulators live in ps_point_eval.cc), '--interlace'
    runs (two passes, the second on a half-cell-shifted grid, would share one folder and key: 2026-10-11), and
    with DTFE_PS_CHECKPOINT=0. The scratch-backed grids are written to the checkpoint file like any other:
    their mmap files are unlinked at creation (scratch_alloc.cc) and are modified in place by the next merge,
    so they cannot serve as an atomic checkpoint themselves.

    TEST HOOKS: DTFE_TEST_EXIT_AFTER_PARTITIONS=K writes a checkpoint after the K-th partition merged in this
    process and then SIGKILLs the process, which leaves exactly what a kill leaves. DTFE_TEST_CHECKPOINT_FREE_BYTES
    replaces the volume's free bytes in the room rule. */
#ifndef PARTITION_CHECKPOINT_H
#define PARTITION_CHECKPOINT_H
#ifdef PHASE_SPACE   // the PS partition loop only: the accumulators below are PS-DTFE's

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <signal.h>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>
#include <vector>

#include "define.h"
#include "user_options.h"
#include "quantities.h"
#include "message.h"
#include "cache_key_lines.h"
#include "output_staging.h"

// the binary's provenance (src/build_info.cc, linked into DTFE / PS-DTFE; the library does not compile this header)
char const *dtfeBuildRevision();
char const *dtfeBuildTime();

namespace PartitionCheckpoint
{

static char const *const MAGIC = "DTFE-PS-CHECKPOINT 1";

// every accumulator vector of a Quantities, by a stable name (the file's field list)
template<class F>
inline void forEachAccumulator(Quantities &q, F &&f)
{
    f( "density", q.density );                     f( "velocity", q.velocity );
    f( "velocity_gradient", q.velocity_gradient ); f( "velocity_divergence", q.velocity_divergence );
    f( "velocity_shear", q.velocity_shear );       f( "velocity_vorticity", q.velocity_vorticity );
    f( "velocity_std", q.velocity_std );           f( "velocity_dispersion", q.velocity_dispersion );
    f( "scalar", q.scalar );                       f( "scalar_gradient", q.scalar_gradient );
    f( "velocity_tweb", q.velocity_tweb );         f( "velocity_tweb_eigenvalues", q.velocity_tweb_eigenvalues );
    f( "velocity_vweb", q.velocity_vweb );         f( "velocity_vweb_eigenvalues", q.velocity_vweb_eigenvalues );
    f( "stream_count", q.stream_count );           f( "tet_touch", q.tet_touch );
    f( "mass_weight", q.mass_weight );             f( "disp_weight", q.disp_weight );
    f( "disp_velocity", q.disp_velocity );         f( "caustic_bits", q.caustic_bits );
    f( "hidden_streams", q.hidden_streams );
}

// the output path made absolute with the current directory (not resolved: the output need not exist yet)
inline std::string absoluteOutput(std::string const &out)
{
    if ( not out.empty() and out[0] == '/' ) return out;
    char buf[4096];
    if ( ::getcwd( buf, sizeof(buf) ) == nullptr ) return out;
    return std::string( buf ) + "/" + out;
}

inline uint64_t fnv1a(std::string const &s)
{
    uint64_t h = 14695981039346656037ULL;   // the FNV-1a 64 offset basis
    for ( unsigned char c : s ) { h ^= c; h *= 1099511628211ULL; }
    return h;
}

inline std::string hexOf(uint64_t h)
{
    char buf[17];
    std::snprintf( buf, sizeof(buf), "%016llx", (unsigned long long)h );
    return buf;
}

// where this run's checkpoint lives (the header comment); config.sh ps_checkpoint_dir mirrors this
inline std::string directoryFor(User_options const &u)
{
    if ( u.scratchDir.empty() )
        return u.outputFilename + ".runlog.checkpoint";
    std::string const abs = absoluteOutput( u.outputFilename );
    std::string dir = u.scratchDir;
    while ( dir.size() > 1 and dir.back() == '/' ) dir.pop_back();
    return dir + "/" + abs.substr( abs.find_last_of( '/' ) + 1 ) + "." + hexOf( fnv1a( abs ) ) + ".runlog.checkpoint";
}

// the command line as option groups ('--grid 64 64 64' is one line), minus the options that cannot change a
// grid value; positional arguments (the input and output names) keep their own lines
inline std::string commandKeyLines(std::vector<std::string> const &tokens)
{
    static char const *const dropFlag[]  = { "--resume", "--ordered-merge", "--ps-no-overlap" };
    static char const *const dropValue[] = { "--verbose", "-v", "--max-concurrent", "--scratch-dir", "--tessellation-cache" };
    auto isOption = [](std::string const &t)
    {   // '--name' or '-x'; a negative number ('-0.1') is a value
        return t.size() >= 2 and t[0] == '-' and ( t[1] == '-' or std::isalpha( (unsigned char)t[1] ) );
    };
    std::ostringstream out;
    size_t i = 0;
    while ( i < tokens.size() )
    {
        std::string const &t = tokens[i];
        if ( not isOption( t ) ) { out << "arg " << t << '\n'; ++i; continue; }
        std::string const name = t.substr( 0, t.find( '=' ) );
        std::string group = t;
        size_t j = i + 1;
        while ( j < tokens.size() and not isOption( tokens[j] ) ) group += ' ' + tokens[j++];
        bool drop = false;
        for ( char const *d : dropFlag )  if ( name == d ) drop = true;
        for ( char const *d : dropValue ) if ( name == d ) drop = true;
        // '-v2': the short verbose option with its value attached
        if ( t.size() > 2 and t[0] == '-' and t[1] == 'v' and std::isdigit( (unsigned char)t[2] ) ) drop = true;
        if ( not drop ) out << "opt " << group << '\n';
        i = j;
    }
    return out.str();
}

// the run's settings key (the header comment): compared verbatim on '--resume'
inline std::string settingsKey(User_options const &u)
{
    std::ostringstream k;
    k << cacheBuildLines();
    // the BINARY too (2026-10-11): a resume after a rebuild combined partitions deposited by the old code with the
    // new code's, while the run log names only the new 'Build:' -- which the launcher's fix registry reads to tell a
    // grid made before an output-changing fix. The link time as well: every uncommitted rebuild has the same
    // '<hash>-dirty' revision. Over-refusal is the design; the refusal names PS_RESUME=0.
    k << "binary=" << dtfeBuildRevision() << " (built " << dtfeBuildTime() << ")\n";
    k << "output=" << absoluteOutput( u.outputFilename ) << '\n';
    k << commandKeyLines( u.commandTokens );
    k << cacheFileLine( "input", u.inputFilename ) << '\n';
#ifdef PHASE_SPACE
    if ( not u.lagrangianInputFilename.empty() )
        k << cacheFileLine( "lagrange", u.lagrangianInputFilename ) << '\n';
#endif
    if ( not u.configFilename.empty() )
        k << cacheFileLine( "config", u.configFilename ) << '\n';
    k << "partition=";
    for ( size_t p : u.partition ) k << p << ',';
    k << "\ngrid=";
    for ( int d = 0; d < NO_DIM; ++d ) k << u.gridSize[d] << ',';
    k << '\n';
    return k.str();
}

// The manifest: the text header of the checkpoint file
struct Manifest
{
    std::string key;                  // the settings key (verbatim)
    std::vector<size_t> partition;    // the split
    std::vector<char> merged;         // per partition: merged into the saved grids
    struct FieldEntry { std::string name; size_t elemSize = 0, count = 0; };
    std::vector<FieldEntry> fields;   // in file order
    long headerBytes = 0;             // where the grid bytes start
};

// reads the header of '<dir>/checkpoint'; false (with why) when there is none or it is unreadable
inline bool readManifest(std::string const &dir, Manifest &m, std::string &why)
{
    std::string const path = dir + "/checkpoint";
    std::ifstream in( path, std::ios::binary );
    if ( not in ) { why = "there is no checkpoint '" + path + "'"; return false; }
    std::string line;
    if ( not std::getline( in, line ) or line != MAGIC ) { why = "'" + path + "' is not a PS-DTFE checkpoint (or another format)"; return false; }
    bool inKey = false, ended = false;
    while ( std::getline( in, line ) )
    {
        if ( inKey ) { if ( line == "key-end" ) inKey = false; else m.key += line + '\n'; continue; }
        if ( line == "key-begin" ) { inKey = true; continue; }
        if ( line == "header-end" ) { ended = true; break; }
        std::istringstream ls( line );
        std::string tag;
        ls >> tag;
        if ( tag == "partition" ) { size_t p; while ( ls >> p ) m.partition.push_back( p ); }
        else if ( tag == "merged" )
        {
            std::string bits;
            ls >> bits;
            m.merged.assign( bits.size(), 0 );
            for ( size_t i = 0; i < bits.size(); ++i ) m.merged[i] = ( bits[i] == '1' );
        }
        else if ( tag == "field" )
        {
            Manifest::FieldEntry f;
            ls >> f.name >> f.elemSize >> f.count;
            m.fields.push_back( f );
        }
    }
    if ( not ended ) { why = "the header of '" + path + "' is incomplete"; return false; }
    m.headerBytes = long( in.tellg() );
    return true;
}

inline size_t mergedCount(std::vector<char> const &merged)
{
    return size_t( std::count( merged.begin(), merged.end(), char(1) ) );
}

// the lines of one key missing from the other (both ways), for the refusal message
inline std::string keyDifference(std::string const &saved, std::string const &now)
{
    auto lines = [](std::string const &s)
    {
        std::vector<std::string> v;
        std::istringstream is( s );
        std::string l;
        while ( std::getline( is, l ) ) v.push_back( l );
        return v;
    };
    std::vector<std::string> const a = lines( saved ), b = lines( now );
    std::ostringstream out;
    int shown = 0;
    for ( auto const &l : a )
        if ( std::find( b.begin(), b.end(), l ) == b.end() and shown++ < 8 ) out << "\n\t  checkpoint: " << l;
    for ( auto const &l : b )
        if ( std::find( a.begin(), a.end(), l ) == a.end() and shown++ < 16 ) out << "\n\t  this run  : " << l;
    return out.str();
}

// removes the checkpoint directory and everything in it (one level: the checkpoint and stale temps)
inline void removeDirectory(std::string const &dir)
{
    if ( DIR *d = ::opendir( dir.c_str() ) )
    {
        while ( dirent *e = ::readdir( d ) )
        {
            std::string const n = e->d_name;
            if ( n == "." or n == ".." ) continue;
            std::remove( ( dir + "/" + n ).c_str() );
        }
        ::closedir( d );
    }
    ::rmdir( dir.c_str() );
}

// a dead writer's 'checkpoint.tmp.<pid>' (killed during a write) is deleted; a live one's is left alone
inline void removeStaleTemps(std::string const &dir)
{
    DIR *d = ::opendir( dir.c_str() );
    if ( not d ) return;
    std::vector<std::string> stale;
    while ( dirent *e = ::readdir( d ) )
    {
        std::string const n = e->d_name;
        if ( n.rfind( "checkpoint.tmp.", 0 ) != 0 ) continue;
        long const pid = std::atol( n.c_str() + 15 );
        if ( pid <= 0 or pid == long( ::getpid() ) or ( ::kill( pid_t( pid ), 0 ) != 0 and errno == ESRCH ) )
            stale.push_back( dir + "/" + n );
    }
    ::closedir( d );
    for ( auto const &s : stale ) std::remove( s.c_str() );
}

// The run's checkpoint state (one per process: the PS partition loop runs once)
struct Checkpointer
{
    bool on = false;                  // writing checkpoints
    bool resumed = false;             // grids loaded from a checkpoint
    std::string dir, key;
    std::vector<size_t> partition;
    std::vector<char> merged;         // per partition: in the grids
    std::chrono::steady_clock::time_point lastWrite;
    double lastWriteSec = 0.;         // how long the last write took (the throttle)
    bool wroteOnce = false, warned = false;
    int mergesHere = 0;               // partitions merged by THIS process (the test hook counts these)
    int exitAfter = -1;               // DTFE_TEST_EXIT_AFTER_PARTITIONS
    int written = 0;                  // checkpoints written by this process (reported at the end)
    double writeSecTotal = 0.;
};

inline Checkpointer &state()
{
    static Checkpointer c;
    return c;
}

// After a successful run (the outputs renamed into place, output_staging.h) the checkpoint has done its job.
// An exit before that keeps it: an error in the write phase, a rescue to the temp folder, a throwError, and a
// commit whose renames failed (exit 3, the outputs left as '.part.<pid>' files: 2026-10-11, the old test of
// 'begun cleared, no rescue' passed it and deleted the checkpoint). main.cpp is not changed: this runs from
// atexit, and output_staging's committedOk is set only where outputStageCommit returns 0 with the set in place.
inline void removeOnSuccessAtExit()
{
    Checkpointer &c = state();
    if ( c.dir.empty() ) return;
    OutputStaging &s = outputStagingState();
    bool ok;
    {
        std::lock_guard<std::mutex> lock( s.m );
        ok = s.committedOk;
    }
    if ( ok ) removeDirectory( c.dir );
}

inline void armRemovalOnSuccess(std::string const &dir)
{
    Checkpointer &c = state();
    static bool registered = false;
    c.dir = dir;
    if ( registered ) return;
    registered = true;
    outputStagingState();   // constructed BEFORE the handler is registered, so it is destroyed after it runs
    std::atexit( removeOnSuccessAtExit );
}

// writes the grids + manifest atomically; false (and the previous checkpoint stays) on any failure
inline bool write(Quantities &u, Quantities &a, std::string &why)
{
    Checkpointer &c = state();
    std::ostringstream h;
    h << MAGIC << '\n' << "hash " << hexOf( fnv1a( c.key ) ) << '\n' << "partition";
    for ( size_t p : c.partition ) h << ' ' << p;
    h << "\nmerged ";
    for ( char m : c.merged ) h << ( m ? '1' : '0' );
    h << "\nmerged_count " << mergedCount( c.merged ) << " of " << c.merged.size() << '\n';
    auto listFields = [&](char const *side, Quantities &q)
    {
        forEachAccumulator( q, [&](char const *name, auto &v)
        {
            if ( not v.empty() )
                h << "field " << side << '.' << name << ' ' << sizeof( v[0] ) << ' ' << v.size() << '\n';
        } );
    };
    listFields( "u", u );
    listFields( "a", a );
    h << "key-begin\n" << c.key << "key-end\nheader-end\n";
    std::string const header = h.str();
    // room: the new file beside the old one, and the outputs still to come (about the grids' size again: beside
    // the outputs the checkpoint must never be what makes their write fail), plus 2 GB
    double bytes = double( header.size() );
    auto addBytes = [&](Quantities &q)
    {
        forEachAccumulator( q, [&](char const *, auto &v) { bytes += double( v.size() ) * double( sizeof( v[0] ) ); } );
    };
    addBytes( u );
    addBytes( a );
    // the folder is made only once a write really happens (2026-10-11): it was made first, so a run that never had
    // room, or was killed during its first write, left an empty one, and run_ps_dtfe.sh resumed from it -- refused
    // ('nothing to resume') on every rerun. Until it exists, the room is the volume's of the folder it goes in.
    struct stat dirSt;
    bool const dirExists = ::stat( c.dir.c_str(), &dirSt ) == 0;
    struct statvfs vs;
    if ( ::statvfs( ( dirExists ? c.dir : outputStageDirOf( c.dir ) ).c_str(), &vs ) == 0 )
    {
        double freeBytes = double( vs.f_bavail ) * double( vs.f_frsize );
        if ( char const *t = std::getenv( "DTFE_TEST_CHECKPOINT_FREE_BYTES" ) ) freeBytes = std::atof( t );   // test hook
        // the old checkpoint's bytes come back once the new one is renamed over it
        struct stat old;
        double const oldBytes = ::stat( ( c.dir + "/checkpoint" ).c_str(), &old ) == 0 ? double( old.st_size ) : 0.;
        if ( freeBytes < bytes + 1.e9 or freeBytes + oldBytes < 2. * bytes + 2.e9 )
        {
            why = "only " + std::to_string( int( freeBytes / 1.e9 ) ) + " GB free in '" + c.dir + "', a checkpoint needs "
                  "about twice its " + std::to_string( int( bytes / 1.e9 + 0.5 ) ) + " GB plus 2 GB (it and the outputs)";
            return false;
        }
    }
    if ( not dirExists and ::mkdir( c.dir.c_str(), 0755 ) != 0 and errno != EEXIST )
    {
        why = "cannot create '" + c.dir + "': " + std::strerror( errno );
        return false;
    }
    std::string const tmp = c.dir + "/checkpoint.tmp." + std::to_string( long( ::getpid() ) );
    FILE *f = std::fopen( tmp.c_str(), "wb" );
    if ( not f ) { why = "cannot write '" + tmp + "': " + std::strerror( errno ); ::rmdir( c.dir.c_str() ); return false; }
    bool ok = std::fwrite( header.data(), 1, header.size(), f ) == header.size();
    auto writeFields = [&](Quantities &q)
    {
        forEachAccumulator( q, [&](char const *, auto &v)
        {
            if ( ok and not v.empty() )
                ok = std::fwrite( v.data(), sizeof( v[0] ), v.size(), f ) == v.size();
        } );
    };
    writeFields( u );
    writeFields( a );
    ok = ok and std::fflush( f ) == 0 and ::fsync( ::fileno( f ) ) == 0;
    int const err = errno;
    ok = ( std::fclose( f ) == 0 ) and ok;
    if ( not ok or std::rename( tmp.c_str(), ( c.dir + "/checkpoint" ).c_str() ) != 0 )
    {
        why = std::string( "writing '" ) + tmp + "' failed: " + std::strerror( ok ? errno : err );
        std::remove( tmp.c_str() );
        ::rmdir( c.dir.c_str() );   // a folder this write made, now empty, goes again (a previous checkpoint keeps it)
        return false;
    }
    return true;
}

// loads the grids of a checkpoint whose manifest was read and whose key matched; false (with why) when the
// file does not hold what its manifest says. Nothing is changed in u/a unless every field fits.
inline bool load(std::string const &dir, Manifest const &m, Quantities &u, Quantities &a, std::string &why)
{
    std::string const path = dir + "/checkpoint";
    // the file's size must be exactly header + listed fields (a torn file cannot pass: writes are renamed
    // into place whole, but a disk error could still cut one)
    unsigned long long total = (unsigned long long)m.headerBytes;
    for ( auto const &fe : m.fields ) total += (unsigned long long)fe.elemSize * fe.count;
    struct stat st;
    if ( ::stat( path.c_str(), &st ) != 0 or (unsigned long long)st.st_size != total )
    {
        why = "'" + path + "' is not the size its manifest lists (cut or damaged)";
        return false;
    }
    // every listed field must be one of ours with the same element size (another precision is another key,
    // but say it plainly rather than misread)
    for ( auto const &fe : m.fields )
    {
        bool found = false;
        auto check = [&](char const *side, Quantities &q)
        {
            forEachAccumulator( q, [&](char const *name, auto &v)
            {
                if ( fe.name == std::string( side ) + "." + name )
                {
                    found = true;
                    if ( fe.elemSize != sizeof( v[0] ) )
                        why = "field '" + fe.name + "' has " + std::to_string( fe.elemSize ) + "-byte elements, this build "
                              + std::to_string( sizeof( v[0] ) );
                }
            } );
        };
        check( "u", u );
        check( "a", a );
        if ( not found ) why = "the checkpoint holds a field this build does not know: '" + fe.name + "'";
        if ( not why.empty() ) return false;
    }
    FILE *f = std::fopen( path.c_str(), "rb" );
    if ( not f or std::fseek( f, m.headerBytes, SEEK_SET ) != 0 )
    {
        if ( f ) std::fclose( f );
        why = "cannot read '" + path + "'";
        return false;
    }
    bool ok = true;
    for ( auto const &fe : m.fields )
    {
        auto fill = [&](char const *side, Quantities &q)
        {
            forEachAccumulator( q, [&](char const *name, auto &v)
            {
                if ( ok and fe.name == std::string( side ) + "." + name )
                {
                    v.resize( fe.count );
                    ok = std::fread( v.data(), sizeof( v[0] ), v.size(), f ) == v.size();
                }
            } );
        };
        fill( "u", u );
        fill( "a", a );
    }
    std::fclose( f );
    if ( not ok ) why = "reading the grids of '" + path + "' failed";
    return ok;
}

}   // namespace PartitionCheckpoint

#endif   // PHASE_SPACE
#endif
