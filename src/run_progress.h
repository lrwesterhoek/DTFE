/*  The run's progress sidecar (2026-10-11): '<output>.runlog.progress', a small JSON file the binary rewrites
    at every phase change and after every partition, so the launcher (and a phone, through the Runs tab) can
    follow a run started from a terminal without parsing a 200 KB log. The run directory contract (C1):

        {"pid":..,"phase":"..","done":..,"total":..,"elapsed_s":..,"eta_s":..,"rss_gb":..,
         "gpu_retries":..,"updated":<unix time>}

    eta_s is -1 when unknown. Written atomically ('.tmp.<pid>' then rename), so a reader never sees half a
    file; removed when the run finishes successfully (main.cpp), so a sidecar that remains belongs to a run
    that is still going or one that died (its pid then no longer exists). Off for '--serve' and
    '--auto-tune-report', which compute no grid, and with DTFE_NO_PROGRESS=1. Header-only: the state is
    one function-local static shared by DTFE.cpp and main.cpp. */
#ifndef RUN_PROGRESS_H
#define RUN_PROGRESS_H

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <mutex>
#include <string>
#include <unistd.h>
#ifdef __APPLE__
#include <mach/mach.h>
#endif

#include "user_options.h"

#ifdef PHASE_SPACE
// failed GPU deposit attempts (each one redoes the partition's deposit, the third falls back to the CPU);
// counted by the GPU hosts (ps_metal_host.cc, ps_gpu_cuda.cu), defined in ps_interpolation.cc
extern std::atomic<int> psGpuRetries;
#endif

struct RunProgress
{
    bool on = false;
    std::string path;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    std::string phase = "starting";
    long done = 0, total = 0;
    double eta = -1.;
    std::mutex m;
};

inline RunProgress &runProgressState()
{
    static RunProgress p;
    return p;
}

// the resident memory of this process now (not the peak), in bytes; -1 when unknown
inline double runProgressRss()
{
#ifdef __APPLE__
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if ( task_info( mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count ) == KERN_SUCCESS )
        return double( info.resident_size );
    return -1.;
#else
    double pages = -1.;
    if ( FILE *f = std::fopen( "/proc/self/statm", "r" ) )
    {
        double size = 0.;
        if ( std::fscanf( f, "%lf %lf", &size, &pages ) != 2 ) pages = -1.;
        std::fclose( f );
    }
    return pages < 0. ? -1. : pages * double( sysconf( _SC_PAGESIZE ) );
#endif
}

inline int runProgressGpuRetries()
{
#ifdef PHASE_SPACE
    return psGpuRetries.load();
#else
    return 0;
#endif
}

// (re)writes the sidecar from the current state; the caller holds the mutex
inline void runProgressWriteLocked(RunProgress &p)
{
    if ( not p.on ) return;
    double const elapsed = std::chrono::duration<double>( std::chrono::steady_clock::now() - p.start ).count();
    double const rss = runProgressRss();
    std::string const tmp = p.path + ".tmp." + std::to_string( long( ::getpid() ) );
    FILE *f = std::fopen( tmp.c_str(), "w" );
    if ( not f ) return;   // the output folder is not writable now: the progress is a convenience, never fatal
    std::fprintf( f, "{\"pid\": %ld, \"phase\": \"%s\", \"done\": %ld, \"total\": %ld, \"elapsed_s\": %.1f, "
                     "\"eta_s\": %.1f, \"rss_gb\": %.3f, \"gpu_retries\": %d, \"updated\": %ld}\n",
                  long( ::getpid() ), p.phase.c_str(), p.done, p.total, elapsed, p.eta,
                  rss < 0. ? -1. : rss / 1.e9, runProgressGpuRetries(), long( std::time( nullptr ) ) );
    bool const ok = std::fflush( f ) == 0;
    std::fclose( f );
    if ( not ok or std::rename( tmp.c_str(), p.path.c_str() ) != 0 )
        std::remove( tmp.c_str() );
}

// arms the sidecar for this run (main.cpp, right after the options are read)
inline void runProgressInit(User_options const &u)
{
    RunProgress &p = runProgressState();
    std::lock_guard<std::mutex> lock( p.m );
    char const *off = std::getenv( "DTFE_NO_PROGRESS" );
    p.on = not u.outputFilename.empty() and not u.psServe and not u.autoTuneReport
           and not ( off and off[0] and off[0] != '0' );
    p.path = u.outputFilename + ".runlog.progress";
    p.start = std::chrono::steady_clock::now();
    runProgressWriteLocked( p );
}

// a new phase ('reading', 'triangulation', 'partitions', 'post-processing', 'writing'); done/total reset
// unless given
inline void runProgressPhase(char const *phase, long done = 0, long total = 0, double eta = -1.)
{
    RunProgress &p = runProgressState();
    std::lock_guard<std::mutex> lock( p.m );
    p.phase = phase;
    p.done = done;
    p.total = total;
    p.eta = eta;
    runProgressWriteLocked( p );
}

// the run finished and its outputs are in place: the sidecar goes
inline void runProgressFinish()
{
    RunProgress &p = runProgressState();
    std::lock_guard<std::mutex> lock( p.m );
    if ( p.on ) std::remove( p.path.c_str() );
    p.on = false;
}

#endif
