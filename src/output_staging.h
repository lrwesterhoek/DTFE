/*  Crash-safe outputs: every result is in RAM until the very last step, so a disk that drops off during a long
    run must not throw the grids away, and a kill during the write phase must not leave a mixed old/new set:

      * each output file is written as '<file>.part.<pid>' and the whole set is renamed into place only once
        every file is written (outputStageCommit), so the folder only ever shows a complete old or new set;
      * before the first file, the output folder is probed; while it is missing or read-only the run waits for
        it (DTFE_OUTPUT_WAIT seconds, default 300: a USB disk that dropped off and comes back), then writes the
        whole set to the RESCUE folder $TMPDIR/dtfe-rescue/<folder>_<name> instead, says where and how to move
        it, and exits with status 3 -- the hours of computing are kept;
      * a folder that vanishes after some files were written sends the remaining ones to the rescue folder; the
        files staged before it went may still be on the disk as '.part.<pid>' (a USB disk that comes back): the
        run lists them with the mv that puts each in place;
      * at the start of a run, '.part.<pid>' leftovers of THIS output name whose pid no longer runs are deleted --
        except one whose final file is missing: it may be the only copy of that output, so it is kept and its
        place and mv are printed.

    Header-only: the state is one function-local static shared by main.cpp, input_output.cc (the grid
    writers' openOutput*File) and ps_point_eval.cc (the '.pts_*' writer). Before outputStageBegin() every
    path passes through unchanged, so a writer used outside the write phase writes straight to its path. */
#ifndef OUTPUT_STAGING_H
#define OUTPUT_STAGING_H

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <mutex>
#include <signal.h>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

struct OutputStaging
{
    bool begun = false, rescue = false;
    bool committedOk = false;   // outputStageCommit put the whole set in the output folder (it returned 0 after a begin)
    std::string outDir, rescueDir, name;
    std::vector< std::pair<std::string,std::string> > staged;   // (written '.part.<pid>' path, final path)
    std::vector<std::string> rescued;                            // final paths written to the rescue folder
    std::mutex m;
};

inline OutputStaging &outputStagingState()
{
    static OutputStaging s;
    return s;
}

inline std::string outputStageDirOf(std::string const &path)
{
    size_t const slash = path.find_last_of( '/' );
    if ( slash == std::string::npos ) return ".";
    if ( slash == 0 ) return "/";
    return path.substr( 0, slash );
}

inline std::string outputStageBaseOf(std::string const &path)
{
    size_t const slash = path.find_last_of( '/' );
    return slash == std::string::npos ? path : path.substr( slash + 1 );
}

// can a file be created in 'dir' now? (a probe file, removed again)
inline bool outputStageDirWritable(std::string const &dir)
{
    struct stat st;
    if ( ::stat( dir.c_str(), &st ) != 0 or not S_ISDIR( st.st_mode ) ) return false;
    std::string const probe = dir + "/.dtfe-write-probe." + std::to_string( long( ::getpid() ) );
    FILE *f = std::fopen( probe.c_str(), "w" );
    if ( not f ) return false;
    std::fclose( f );
    std::remove( probe.c_str() );
    return true;
}

// '<file>.part.<pid>' -> pid (0 when the name is not of that form)
inline long outputStagePartPid(std::string const &file)
{
    size_t const at = file.rfind( ".part." );
    if ( at == std::string::npos or at + 6 >= file.size() ) return 0;
    char *end = nullptr;
    long const pid = std::strtol( file.c_str() + at + 6, &end, 10 );
    return ( end and *end == '\0' and pid > 0 ) ? pid : 0;
}

// deletes '<output name>.*.part.<pid>' files in the output folder whose writer no longer runs, unless the final
// file is missing (then the part is kept and pointed to); returns how many were deleted
inline int outputStageCleanStale(std::string const &outputFilename)
{
    std::string const dir = outputStageDirOf( outputFilename ), prefix = outputStageBaseOf( outputFilename ) + ".";
    DIR *d = ::opendir( dir.c_str() );
    if ( not d ) return 0;
    int removed = 0;
    while ( struct dirent *e = ::readdir( d ) )
    {
        std::string const f = e->d_name;
        if ( f.compare( 0, prefix.size(), prefix ) != 0 ) continue;
        long const pid = outputStagePartPid( f );
        if ( pid <= 0 or pid == long( ::getpid() ) ) continue;
        if ( ::kill( pid_t( pid ), 0 ) == 0 or errno == EPERM ) continue;   // that writer is alive
        // no final file beside it: a write phase whose rename failed (the disk went away and came back) may
        // have left the only copy here -- never delete that one
        std::string const finalPath = dir + "/" + f.substr( 0, f.rfind( ".part." ) );
        struct stat st;
        if ( ::stat( finalPath.c_str(), &st ) != 0 )
        {
            std::fprintf( stderr, "Kept the partial output '%s/%s' (its writer, pid %ld, is gone, and '%s' does not exist: "
                          "it may be the only copy). If it is complete, put it in place with:\n    mv '%s/%s' '%s'\n",
                          dir.c_str(), f.c_str(), pid, finalPath.c_str(), dir.c_str(), f.c_str(), finalPath.c_str() );
            continue;
        }
        if ( std::remove( ( dir + "/" + f ).c_str() ) == 0 )
        {
            ++removed;
            std::fprintf( stderr, "Removed the stale partial output '%s/%s' (its writer, pid %ld, is gone).\n",
                          dir.c_str(), f.c_str(), pid );
        }
    }
    ::closedir( d );
    return removed;
}

// the rescue folder: $TMPDIR/dtfe-rescue/<parent folder>_<output name> (another '-<pid>' when it exists)
inline std::string outputStageMakeRescueDir(std::string const &outputFilename)
{
    char const *tmp = std::getenv( "TMPDIR" );
    std::string root = ( tmp and tmp[0] ) ? std::string( tmp ) : std::string( "/tmp" );
    while ( root.size() > 1 and root.back() == '/' ) root.pop_back();
    root += "/dtfe-rescue";
    ::mkdir( root.c_str(), 0755 );
    std::string const parent = outputStageBaseOf( outputStageDirOf( outputFilename ) );
    std::string name = ( parent.empty() or parent == "." or parent == "/" ? std::string() : parent + "_" )
                       + outputStageBaseOf( outputFilename );
    std::string dir = root + "/" + name;
    if ( ::mkdir( dir.c_str(), 0755 ) != 0 )
    {
        dir += "-" + std::to_string( long( ::getpid() ) );
        ::mkdir( dir.c_str(), 0755 );
    }
    return dir;
}

// called with the lock held: from now on every file goes to the rescue folder
inline void outputStageEnterRescueLocked(OutputStaging &s, char const *why)
{
    if ( s.rescue ) return;
    s.rescue = true;
    s.rescueDir = outputStageMakeRescueDir( s.name );
    std::fprintf( stderr, "\n~~~ OUTPUT RESCUE ~~~ The output folder '%s' %s. The results are written to the rescue "
                  "folder '%s' instead (nothing computed is lost).\n", s.outDir.c_str(), why, s.rescueDir.c_str() );
}

// the write phase starts: probe the output folder (waiting for it a while), else rescue
inline void outputStageBegin(std::string const &outputFilename)
{
    OutputStaging &s = outputStagingState();
    std::lock_guard<std::mutex> lock( s.m );
    s.begun = true;
    s.committedOk = false;
    s.name = outputFilename;
    s.outDir = outputStageDirOf( outputFilename );
    if ( outputStageDirWritable( s.outDir ) ) return;
    double waitSec = 300.;
    if ( char const *env = std::getenv( "DTFE_OUTPUT_WAIT" ) ) waitSec = std::max( 0., std::atof( env ) );
    if ( waitSec > 0. )
        std::fprintf( stderr, "The output folder '%s' cannot be written now (an unmounted disk?); waiting up to %.0f s "
                      "for it (DTFE_OUTPUT_WAIT) ...\n", s.outDir.c_str(), waitSec );
    auto const t0 = std::chrono::steady_clock::now();
    while ( std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count() < waitSec )
    {
        std::this_thread::sleep_for( std::chrono::seconds( 5 ) );
        if ( outputStageDirWritable( s.outDir ) )
        {
            std::fprintf( stderr, "The output folder '%s' is writable again.\n", s.outDir.c_str() );
            return;
        }
    }
    outputStageEnterRescueLocked( s, "could not be written" );
}

// the path a writer opens for the output file 'finalName'
inline std::string outputStagePath(std::string const &finalName)
{
    OutputStaging &s = outputStagingState();
    std::lock_guard<std::mutex> lock( s.m );
    if ( not s.begun ) return finalName;
    if ( s.rescue )
    {
        s.rescued.push_back( finalName );
        return s.rescueDir + "/" + outputStageBaseOf( finalName );
    }
    std::string const part = finalName + ".part." + std::to_string( long( ::getpid() ) );
    s.staged.emplace_back( part, finalName );
    return part;
}

// the writer could not open the path outputStagePath gave: rescue this and every later file; returns the new
// path ('' when the rescue is unavailable too, or the staging is not active)
inline std::string outputStageRetry(std::string const &finalName)
{
    OutputStaging &s = outputStagingState();
    {
        std::lock_guard<std::mutex> lock( s.m );
        if ( not s.begun or s.rescue ) return std::string();
        if ( not s.staged.empty() and s.staged.back().second == finalName ) s.staged.pop_back();
        outputStageEnterRescueLocked( s, "stopped being writable during the write" );
    }
    return outputStagePath( finalName );
}

// every file is written: rename the set into place. Returns 0 when the set is in the output folder, 3 when
// it is (partly) in the rescue folder.
inline int outputStageCommit()
{
    OutputStaging &s = outputStagingState();
    std::lock_guard<std::mutex> lock( s.m );
    if ( not s.begun ) return 0;
    std::vector< std::pair<std::string,std::string> > unplaced;   // (staged part, final path) whose rename failed
    for ( auto const &pf : s.staged )
        if ( std::rename( pf.first.c_str(), pf.second.c_str() ) != 0 )
            unplaced.push_back( pf );
    s.staged.clear();
    s.begun = false;
    if ( not s.rescue and unplaced.empty() ) { s.committedOk = true; return 0; }
    if ( not unplaced.empty() )
    {
        // these are not lost: a disk that drops off and comes back still holds them as
        // '.part.<pid>' -- the next run keeps a part whose final file is missing (outputStageCleanStale), yet
        // deletes it where an older final file exists, so the mv must happen before the rerun
        std::fprintf( stderr, "~~~ OUTPUT RESCUE ~~~ %zu file(s) written to '%s' before it went away could not be put "
                      "in place. They may still be there as '.part.%ld' files: once the folder is back, and BEFORE "
                      "running this snapshot again, put them in place with:\n", unplaced.size(), s.outDir.c_str(),
                      long( ::getpid() ) );
        for ( auto const &pf : unplaced )
            std::fprintf( stderr, "    mv '%s' '%s'\n", pf.first.c_str(), pf.second.c_str() );
    }
    if ( s.rescue )
        std::fprintf( stderr, "~~~ OUTPUT RESCUE ~~~ %zu output file(s) are in '%s'. Once the output folder is back, "
                      "move them with:\n    mv '%s'/* '%s'/\n", s.rescued.size(), s.rescueDir.c_str(),
                      s.rescueDir.c_str(), s.outDir.c_str() );
    return 3;
}

#endif
