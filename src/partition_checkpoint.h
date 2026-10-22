/*  Resume for a killed partitioned PS-DTFE run: a large run takes hours, and a kill late in it (a restart, a
    disk dropping off, an out-of-memory SIGKILL) must not throw every merged partition away. The
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
    every partition.

    SAFE: the manifest carries the run's SETTINGS KEY: the build configuration and the binary (its revision and
    link time: a rebuilt binary does not resume), the command line minus the
    options that cannot change a grid value (--resume, --ordered-merge, --completion-order-merge, --ps-no-overlap, --no-overlap, --verbose,
    --max-concurrent, --scratch-dir, --tessellation-cache), the input files' size+mtime, the resolved
    partition split and the grid. '--resume' refuses, naming the lines that differ, when the key differs.
    It over-refuses rather than under-refuses (a reordered command line is a different key).

    MERGE ORDER: the manifest lists the merged partitions as a SET, so a run merging in completion order
    ('--completion-order-merge') can be resumed too. '--resume'
    merges in index order for the rest of the run; when the killed run merged in index order too (the default,
    or '--max-concurrent 1'), the resumed grids are byte-identical to an
    uninterrupted ordered run (tests/ps_resume_check.sh) -- at the
    same '--max-concurrent' and thread count: the CPU deposit inside a partition splits its sums over the
    threads a partition gets, so another count changes the grids at float rounding (--max-concurrent 2 and 4
    differ in '.den' by ~1e-7 relative). The key leaves the two out on
    purpose: a resume at a lower --max-concurrent (the run was killed for memory) must stay possible.

    ROOM: a write needs the grids' bytes + 1 GB free while it goes (the new file beside the old one), and,
    counting the old file's bytes as free, twice the grids' bytes + 2 GB (the checkpoint and the outputs still
    to be written: on the T7 the checkpoint must never be what makes their write fail);
    short of that the write is skipped with a warning (once) and the previous checkpoint stays.

    POINTS: a '--sample-points' run is checkpointed too. Its per-point stream records (ps_point_eval.cc)
    are kept per partition, and a write saves the merged partitions' records not saved yet as 'points.<i>' files
    in the same folder BEFORE the manifest is renamed into place, so a manifest never lists a partition whose
    records are not on disk; '--resume' reads them back with the grids. Each partition's records are written once
    (they do not change after its merge), and they count in the room rule and the 5% throttle like the grids.
    The settings key then also names the sample-point file (size + mtime).

    NOT CHECKPOINTED: '--interlace' runs (two passes, the second on a half-cell-shifted grid, would share one
    folder and key), and with DTFE_PS_CHECKPOINT=0. The scratch-backed grids are written to the
    checkpoint file like any other: their mmap files are unlinked at creation (scratch_alloc.cc) and are modified in place by the next merge,
    so they cannot serve as an atomic checkpoint themselves.

    TEST HOOKS: DTFE_TEST_EXIT_AFTER_PARTITIONS=K writes a checkpoint after the K-th partition merged in this
    process and then SIGKILLs the process, which leaves exactly what a kill leaves. DTFE_TEST_CHECKPOINT_FREE_BYTES
    replaces the volume's free bytes in the room rule.

    STANDARD DTFE TOO. The standard binary's partition loop (DTFE.cpp: serial, one partition at a time
    with its threads inside) checkpoints through this same code -- beginLoop() before the loop, afterMerge() after
    each partition, reportWrites() at its end; the PS loop calls the same three. There each partition ASSIGNS its
    own disjoint sub-grid of the full grids (copySubgridResultsToMain), so the full grids after k partitions plus
    the list of those k are the run's whole state: '--resume' loads them and computes only the partitions the list
    lacks. The T-web and the V-web come after the loop from the full density and velocity-gradient grids
    (postProcessWebFields), which the loop's options force on, so they are in the checkpoint like every other
    accumulator. A resumed CPU run writes outputs byte-identical to an uninterrupted run's AT THE SAME THREAD COUNT
    ('--max-concurrent' is the thread split inside each partition there, and each thread's sub-box carries its own
    padding: left out of the key as for PS, so a run killed for memory can resume with fewer threads, at float
    rounding where the padding is ample -- 2.8e-2 of the peak on the z = 0 demo at padding 5, 1 against 2 threads);
    a '--gpu' run's too, since its sums are exact (64-bit fixed point; with DTFE_GPU_FLOAT_GRIDS=1 within the
    float atomics' run-to-run rounding). The same file layout, key, room rule,
    5% throttle, refusals and test hooks; what differs is named per build below (BUILD): the file's first line,
    the off switch (DTFE_CHECKPOINT=0 for standard), the driver knob a refusal names (run_dtfe.sh DTFE_RESUME=0)
    and the messages' 'DTFE:' label; standard implies no ordered merge (its loop is serial anyway). The progress
    sidecar of a resumed standard run counts the restored partitions as done (DTFE.cpp). '--interlace' runs write no
    checkpoint, as for PS. A standard '--sample-points' run (with '--partition') is checkpointed
    too, exactly as PS-DTFE's (POINTS above): the standard loop tags each partition's point records
    with psPointEvalSetPartition around its DTFE_parallel call -- under '--sample-points' that call triangulates the
    partition once, on the loop's own thread (DTFE_parallel's single-triangulation branch), so the thread-local tag
    reaches the point walk's sinks (ps_point_eval.cc forEachCellParallel) -- and each point is answered by the one
    partition owning its tetrahedron, so a partition's records are complete once it is in the grids; a standard
    run that also writes a 4096^2 image plane is such a run. tests/dtfe_resume_check.sh (D13). */
#ifndef PARTITION_CHECKPOINT_H
#define PARTITION_CHECKPOINT_H
// both binaries' partition loops; the library compiles it too

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <functional>
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
#include "env_knobs.h"
#include "cache_key_lines.h"
#include "output_staging.h"
#include "ps_point_eval.h"   // the '--sample-points' records of the merged partitions

// the binary's provenance (src/build_info.cc, linked into DTFE / PS-DTFE). WEAK fallbacks: libDTFE links
// no build_info.cc but compiles DTFE.cpp, whose standard partition loop includes this header; the binaries'
// strong definitions in build_info.o replace these at link time
__attribute__((weak)) char const *dtfeBuildRevision() { return "unknown"; }
__attribute__((weak)) char const *dtfeBuildTime()     { return "unknown"; }
// ... and whether this is one of the programs (build_info.cc: true): libDTFE writes no checkpoints, since nothing
// there commits the outputs through output_staging.h, which is what removes the folder
__attribute__((weak)) bool dtfeIsProgram() { return false; }

namespace PartitionCheckpoint
{

// BUILD: what differs between the two binaries' checkpoints -- the file's first line (a PS checkpoint is
// not a standard one and the other way round; their settings keys differ too, 'build=PS' / 'build=std'), the label of
// the messages, the environment switch that writes none, and the driver knob a refusal names to start over
#ifdef PHASE_SPACE
static char const *const MAGIC = "DTFE-PS-CHECKPOINT 1";
static char const *const NAME  = "PS-DTFE";
static char const *const OFF_ENV = "DTFE_PS_CHECKPOINT";
static char const *const START_OVER = "run_ps_dtfe.sh: PS_RESUME=0";
#else
static char const *const MAGIC = "DTFE-CHECKPOINT 1";
static char const *const NAME  = "DTFE";
static char const *const OFF_ENV = "DTFE_CHECKPOINT";
static char const *const START_OVER = "run_dtfe.sh: DTFE_RESUME=0";
#endif

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
#ifdef PHASE_SPACE   // the phase-space accumulators (quantities.h has them in the PS build only)
    f( "stream_count", q.stream_count );           f( "tet_touch", q.tet_touch );
    f( "mass_weight", q.mass_weight );             f( "disp_weight", q.disp_weight );
    f( "disp_velocity", q.disp_velocity );         f( "caustic_bits", q.caustic_bits );
    f( "hidden_streams", q.hidden_streams );
#endif
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
    // '--completion-order-merge' like '--ordered-merge': the merge order moves the grids at float rounding
    // only, and a run killed under one order must resume under the other (--resume merges in index order)
    // '--fixed-point-merge' is dropped here because the key's own 'merge=fixed-point' line carries it
    static char const *const dropFlag[]  = { "--resume", "--ordered-merge", "--completion-order-merge", "--ps-no-overlap",
                                             "--no-overlap",   // (the standard look-ahead switch)
                                             "--fixed-point-merge" };
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
    // the BINARY too: a resume after a rebuild would combine partitions deposited by the old code with the
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
    // the points the saved records were evaluated at: the file's identity, not only its name on the
    // command line -- a plane rewritten in place would otherwise pass. Runs without points get no such line.
    if ( not u.psSamplePointsFile.empty() )
        k << cacheFileLine( "sample-points", u.psSamplePointsFile ) << '\n';
#ifdef PHASE_SPACE
    // the merge mode: a '--fixed-point-merge' checkpoint holds integer words instead of the summed Real
    // grids, so neither mode can resume the other's (beginLoop refuses it by name)
    if ( u.psFixedPointMerge )
        k << "merge=fixed-point\n";
#endif
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
    if ( not std::getline( in, line ) or line != MAGIC ) { why = "'" + path + "' is not a " + NAME + " checkpoint (or another format)"; return false; }
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

// a dead writer's 'checkpoint.tmp.<pid>' or 'points.<i>.tmp.<pid>' (killed during a write) is deleted; a live
// one's is left alone
inline void removeStaleTemps(std::string const &dir)
{
    DIR *d = ::opendir( dir.c_str() );
    if ( not d ) return;
    std::vector<std::string> stale;
    while ( dirent *e = ::readdir( d ) )
    {
        std::string const n = e->d_name;
        size_t const at = n.find( ".tmp." );
        if ( at == std::string::npos or ( n.rfind( "checkpoint.tmp.", 0 ) != 0 and n.rfind( "points.", 0 ) != 0 ) ) continue;
        long const pid = std::atol( n.c_str() + at + 5 );
        if ( pid <= 0 or pid == long( ::getpid() ) or ( ::kill( pid_t( pid ), 0 ) != 0 and errno == ESRCH ) )
            stale.push_back( dir + "/" + n );
    }
    ::closedir( d );
    for ( auto const &s : stale ) std::remove( s.c_str() );
}

// A raw array kept beside the grids: the fixed-point merge's integer words, carries and state
// (ps_fixed_merge.h), whose Real grids do not exist during its loop. Listed as 'field x.<name> <elemSize> <count>'
// after the u and a fields (none with count 0); a load calls 'resize' for the listed ones and reads into its bytes.
struct Blob
{
    std::string name;
    size_t elemSize = 0;
    std::function<size_t()> count;                 // the elements to write now
    std::function<void const *()> data;            // their bytes
    std::function<void *(size_t)> resize;          // load: room for n elements, their bytes
};

// The run's checkpoint state (one per process: a partition loop runs once -- PS-DTFE's, or standard DTFE's)
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
    // '--sample-points': the merged partitions' records go beside the grids ('points.<i>')
    bool points = false;              // this run evaluates points: save their records with every checkpoint
    std::vector<char> pointsSaved;    // per partition: its 'points.<i>' is this run's (written, or read on resume)
    bool pointsCleaned = false;       // an earlier run's unlisted 'points.<i>' files removed (after the first write)
    std::vector<Blob> blobs;          // '--fixed-point-merge': its integer grids, beside the Real ones
};

inline Checkpointer &state()
{
    static Checkpointer c;
    return c;
}

// After a successful run (the outputs renamed into place, output_staging.h) the checkpoint has done its job.
// An exit before that keeps it: an error in the write phase, a rescue to the temp folder, a throwError, and a
// commit whose renames failed (exit 3, the outputs left as '.part.<pid>' files: a test of 'begun cleared, no
// rescue' would pass it and delete the checkpoint). This runs from atexit, and output_staging's committedOk is set only
// where outputStageCommit returns 0 with the set in place.
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
    for ( Blob const &b : c.blobs )
        if ( b.count() > 0 ) h << "field x." << b.name << ' ' << b.elemSize << ' ' << b.count() << '\n';
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
    for ( Blob const &b : c.blobs ) bytes += double( b.count() ) * double( b.elemSize );
    // '--sample-points': the records of the partitions merged since the last write (each partition's once)
    if ( c.points )
        bytes += psPointEvalUnsavedBytes( c.merged, c.pointsSaved );
    // the folder is made only once a write really happens: made first, a run that never had room, or was killed
    // during its first write, would leave an empty one, and run_ps_dtfe.sh would resume from it -- refused
    // ('nothing to resume') on every rerun. Until it exists, the room is the volume's of the folder it goes in.
    struct stat dirSt;
    bool const dirExists = ::stat( c.dir.c_str(), &dirSt ) == 0;
    struct statvfs vs;
    if ( ::statvfs( ( dirExists ? c.dir : outputStageDirOf( c.dir ) ).c_str(), &vs ) == 0 )
    {
        double freeBytes = double( vs.f_bavail ) * double( vs.f_frsize );
        freeBytes = Env::real( "DTFE_TEST_CHECKPOINT_FREE_BYTES", freeBytes );   // test hook
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
    // the point records FIRST: the manifest renamed below lists their partitions as merged, so they
    // must be on disk before it is. A failure leaves the previous manifest, whose partitions' files are older.
    if ( c.points and not psPointEvalSavePartitions( c.dir, hexOf( fnv1a( c.key ) ), c.merged, c.pointsSaved, why ) )
    {
        ::rmdir( c.dir.c_str() );   // only if it is empty (a folder this write made)
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
    for ( Blob const &b : c.blobs )
        if ( ok and b.count() > 0 )
            ok = std::fwrite( b.data(), b.elemSize, b.count(), f ) == b.count();
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
    // an earlier run's 'points.<i>' of partitions this manifest does not list: never read, only room taken
    if ( c.points and not c.pointsCleaned )
    {
        psPointEvalRemoveUnsaved( c.dir, c.pointsSaved );
        c.pointsCleaned = true;
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
        for ( Blob const &b : state().blobs )
            if ( fe.name == "x." + b.name )
            {
                found = true;
                if ( fe.elemSize != b.elemSize )
                    why = "field '" + fe.name + "' has " + std::to_string( fe.elemSize ) + "-byte elements, this build "
                          + std::to_string( b.elemSize );
            }
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
        for ( Blob const &b : state().blobs )
            if ( ok and fe.name == "x." + b.name )
                ok = std::fread( b.resize( fe.count ), b.elemSize, fe.count, f ) == fe.count;
    }
    std::fclose( f );
    if ( not ok ) why = "reading the grids of '" + path + "' failed";
    return ok;
}

/* The three steps of a checkpointed partition loop, shared by the PS and the standard loops.

   beginLoop: after the full grids are reserved (sized: a resume reads into them) and before the first partition.
   Sets up state() for this run (folder, key, split, test hook), removes a dead writer's temp file, and under
   '--resume' refuses unless the checkpoint is this run's (manifest readable, key equal, same partition count,
   every field the right size), then LOADS its grids into u/a (PS-DTFE '--sample-points': and the merged partitions'
   point records) and marks its partitions in state().merged -- the
   caller skips those (PS: like a --ps-window skip, still a turn in the ordered merge; standard: not computed at
   all). Says where checkpoints go, or why there are none. Returns the number of partitions restored. */
inline int beginLoop(User_options const &options, int const totalPartitions, Quantities &u, Quantities &a,
                     bool const pointEval, int const verboseLevel)
{
    Checkpointer &checkpoint = state();
    MESSAGE::Message message( verboseLevel );
    bool const off = not Env::flag( OFF_ENV, true );
    // '--sample-points': PS-DTFE checkpoints the per-point records of every merged partition beside the grids
    // ('points.<i>', the header comment's POINTS); its loop tags them with psPointEvalSetPartition. The standard
    // loop tags them too (DTFE.cpp, around its DTFE_parallel call), so both binaries keep them. pointsLost stays
    // for a loop that would not tag them.
    bool const pointsKept = pointEval;
    bool const pointsLost = pointEval and not pointsKept;   // a points run whose records a checkpoint would miss
    std::string const dir = directoryFor( options );
    std::string const key = settingsKey( options );
    removeStaleTemps( dir );
    checkpoint.dir = dir;
    checkpoint.key = key;
    checkpoint.partition = options.partition;
    checkpoint.merged.assign( size_t(totalPartitions), 0 );
    checkpoint.points = pointsKept;
    checkpoint.pointsSaved.assign( size_t(totalPartitions), 0 );
    checkpoint.exitAfter = int( Env::integer( "DTFE_TEST_EXIT_AFTER_PARTITIONS", checkpoint.exitAfter ) );
    struct stat st;
    bool const exists = ::stat( ( dir + "/checkpoint" ).c_str(), &st ) == 0;
    int nDone = 0;
    bool const program = dtfeIsProgram();   // libDTFE: no checkpoints (above)
    if ( options.psResume and not program )
        throwError( "'--resume' continues a run of the DTFE / PS-DTFE programs; the library writes no checkpoints." );
    if ( options.psResume )
    {
        if ( pointsLost )
            throwError( "'--resume' cannot continue a '--sample-points' run: its point accumulators are not "
                        "checkpointed. Run it again without '--resume'." );
        Manifest m;
        std::string why;
        if ( not readManifest( dir, m, why ) )
            throwError( "'--resume': nothing to resume -- ", why, ". Run without '--resume' to start the run." );
#ifdef PHASE_SPACE
        // the merge mode first: the other mode's grids are not this mode's (Real sums / integer words)
        bool const savedFx = m.key.find( "\nmerge=fixed-point\n" ) != std::string::npos;
        if ( savedFx != options.psFixedPointMerge )
            throwError( "'--resume' refused: the checkpoint '", dir, std::string( "' was written by a run " )
                        + ( savedFx ? "with '--fixed-point-merge' (its grids are integer words); this run merges into "
                                      "the float grids (the default merge)"
                                    : "with the default merge (its grids are float sums); this run has '--fixed-point-merge'" )
                        + ". Resume with the killed run's merge mode, or run without '--resume' to start over (" + START_OVER
                        + ").", "" );
#endif
        if ( m.key != key )
            throwError( "'--resume' refused: the checkpoint '", dir, "' was written by a run with other settings. "
                        "The lines that differ:", keyDifference( m.key, key ),
                        std::string( "\n\tRun with the killed run's settings, or without '--resume' to start over (the "
                        "checkpoint is then replaced; " ) + START_OVER + "). A 'binary=' line differs "
                        "after a rebuild: the old binary's partitions cannot be combined with the new one's." );
        if ( m.merged.size() != size_t(totalPartitions) )
            throwError( "'--resume': the checkpoint '", dir, "' lists " + std::to_string( m.merged.size() )
                        + " partitions, this run has " + std::to_string( totalPartitions ) + "." );
        if ( not load( dir, m, u, a, why ) )
            throwError( "'--resume': ", why, ". Run without '--resume' to start over." );
        // the merged partitions' point records (written before the manifest that lists them)
        if ( pointsKept and not psPointEvalLoadPartitions( dir, hexOf( fnv1a( key ) ), m.merged, why ) )
            throwError( "'--resume': ", why, ". Run without '--resume' to start over." );
        if ( pointsKept ) checkpoint.pointsSaved = m.merged;
        checkpoint.merged = m.merged;
        checkpoint.resumed = true;
        nDone = int( mergedCount( checkpoint.merged ) );
        message << MESSAGE::cBold() << NAME << ":" << MESSAGE::cReset() << " --resume: " << MESSAGE::cMagenta()
                << nDone << " of " << totalPartitions << MESSAGE::cReset() << " partitions are already in the "
                << "checkpoint '" << dir << "'; computing the other " << ( totalPartitions - nDone )
#ifdef PHASE_SPACE
                << ( options.psFixedPointMerge ? " (each added as it finishes: --fixed-point-merge).\n"
                                               : " (merged in partition order: --resume always merges in index order).\n" )
                << MESSAGE::Flush;
#else
                << ".\n" << MESSAGE::Flush;
#endif
        if ( pointsKept )
            message << MESSAGE::cDim() << "  their '--sample-points' stream records were read back from the "
                    << "checkpoint's points.<i> files\n" << MESSAGE::cReset() << MESSAGE::Flush;
    }
    // '--interlace' (psNoCheckpoint, user_options.cc): its two passes would share this folder and key
    checkpoint.on = program and not off and not pointsLost and not options.psNoCheckpoint;
    if ( exists and not options.psResume )
        message << MESSAGE::cBold() << NAME << ":" << MESSAGE::cReset() << " an earlier run's checkpoint is in '"
                << dir << "'; this run starts over" << ( checkpoint.on ? " and replaces it" : " (it writes none)" )
                << " ('--resume' continues it).\n" << MESSAGE::Flush;
    if ( checkpoint.on )
    {
        message << MESSAGE::cBold() << NAME << ":" << MESSAGE::cReset() << " checkpoints of the merged grids go to '"
                << dir << "' (at most ~5% of the run; removed when the outputs are written; '--resume' "
                << "continues a killed run from it).\n" << MESSAGE::Flush;
        if ( pointsKept )
            message << MESSAGE::cDim() << "  with the '--sample-points' stream records of every merged partition "
                    << "(points.<i>, each written once)\n" << MESSAGE::cReset() << MESSAGE::Flush;
    }
    else if ( pointsLost )
        message << MESSAGE::cDim() << "  no checkpoints: a '--sample-points' run cannot be resumed\n"
                << MESSAGE::cReset() << MESSAGE::Flush;
    else if ( options.psNoCheckpoint )
        message << MESSAGE::cDim() << "  no checkpoints: an '--interlace' run cannot be resumed (its two passes "
                << "would share one checkpoint)\n" << MESSAGE::cReset() << MESSAGE::Flush;
    if ( checkpoint.on or checkpoint.resumed )
        armRemovalOnSuccess( dir );
    checkpoint.lastWrite = std::chrono::steady_clock::now();
    return nDone;
}

/* afterMerge: partition pi's grids are now in u/a and hold still (the PS loop: under its merge lock; the standard
   loop is serial). Marks it merged and writes the checkpoint unless the last write took more than 5% of the time
   since it (always the first time, and at the test hook). A 'skipped' partition (PS --ps-window, or restored)
   counts for the manifest but not for the hook. The test hook dies right after its write, as a SIGKILL would. */
inline void afterMerge(int const pi, bool const skipped, Quantities &u, Quantities &a, int const verboseLevel)
{
    Checkpointer &checkpoint = state();
    checkpoint.merged[size_t(pi)] = 1;
    if ( not skipped ) ++checkpoint.mergesHere;
    bool const hookNow = checkpoint.exitAfter > 0 and not skipped and checkpoint.mergesHere == checkpoint.exitAfter;
    if ( checkpoint.on )
    {
        auto const now = std::chrono::steady_clock::now();
        double const since = std::chrono::duration<double>( now - checkpoint.lastWrite ).count();
        if ( hookNow or not checkpoint.wroteOnce or since >= 20. * checkpoint.lastWriteSec )
        {
            std::string why;
            bool const ok = write( u, a, why );
            auto const end = std::chrono::steady_clock::now();
            checkpoint.lastWriteSec = std::chrono::duration<double>( end - now ).count();
            checkpoint.lastWrite = end;
            if ( ok ) { checkpoint.wroteOnce = true; ++checkpoint.written; checkpoint.writeSecTotal += checkpoint.lastWriteSec; }
            else if ( not checkpoint.warned )
            {
                checkpoint.warned = true;
                MESSAGE::Warning warning( verboseLevel );
                warning << "the partition checkpoint could not be written (" << why << "); the run goes on, "
                        << "the previous checkpoint (if any) stays, and later merges try again. Said once.\n"
                        << MESSAGE::EndWarning;
            }
        }
    }
    if ( hookNow )
    {   // the test hook: die here as a SIGKILL would, the checkpoint just written
        std::fflush( nullptr );
        ::raise( SIGKILL );
    }
}

// reportWrites: after the loop, what the checkpoints cost (nothing when none was written)
inline void reportWrites(double const loopSec, int const verboseLevel)
{
    Checkpointer const &checkpoint = state();
    if ( checkpoint.written <= 0 ) return;
    MESSAGE::Message message( verboseLevel );
    message << MESSAGE::cDim() << "  checkpoints: " << checkpoint.written << " written in "
            << MESSAGE::formatDuration( checkpoint.writeSecTotal ) << " (" << std::round( 1000. * checkpoint.writeSecTotal
            / std::max( loopSec, 1e-9 ) ) / 10. << "% of the partition loop)\n" << MESSAGE::cReset() << MESSAGE::Flush;
}

}   // namespace PartitionCheckpoint

#endif
