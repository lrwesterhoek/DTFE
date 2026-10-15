/*  The run's progress sidecar (2026-10-11): '<output>.runlog.progress', a small JSON file the binary rewrites
    at every phase change and after every partition, so the launcher (and a phone, through the Runs tab) can
    follow a run started from a terminal without parsing a 200 KB log. The run directory contract (C1):

        {"pid":..,"phase":"..","done":..,"total":..,"elapsed_s":..,"eta_s":..,"rss_gb":..,
         "gpu_retries":..,"gpu_retries_by_cause":{"innocent":..,"timeout":..,"oom":..,"other":..},
         "predicted_gb":..,"peak_gb":..,"updated":<unix time>}

    predicted_gb is the auto-tuner's predicted peak (-1 when it decided nothing: '--partition' and
    '--max-concurrent' both given) and peak_gb the measured peak so far -- the peak physical footprint on macOS
    (what /usr/bin/time -l calls 'peak memory footprint': it counts Metal's shared buffers, which RSS can miss),
    the maximum RSS elsewhere -- both in GB of 1e9 bytes, the tuner's unit (2026-10-11: the tuner under-predicted
    13 of 28 production runs by up to 5.7 GB, seen only by reading two logs side by side; main.cpp's SUMMARY now
    prints the pair too). eta_s is -1 when unknown. gpu_retries is the total of gpu_retries_by_cause (G1, 2026-10-11), by construction:
    the sidecar writes the sum of the four counters it prints (psGpuRetries -- dtfeGpuRetries in the standard binary,
    which counts its GPU retries the same way since 2026-10-13 --, the hosts' own total, is for the
    closing log line, written after the partitions with no failure in flight): an innocent
    victim (another process hung the GPU; retried at once), a timeout (a watchdog kill: CUDA's
    cudaErrorLaunchTimeout, Metal's Impacting Interactivity), out of memory, anything else. Written atomically ('.tmp.<pid>' then rename), so a reader never sees half a
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
#include <sys/resource.h>
#ifdef __APPLE__
#include <mach/mach.h>
#endif

#include "user_options.h"

#ifdef PHASE_SPACE
// failed GPU buffers / kernels (each one resumes the pass from its last checkpoint; 3 failures of one chunk send
// the partition to the CPU, 2026-10-11);
// counted by the GPU hosts (ps_metal_host.cc, ps_gpu_cuda.cu), defined in ps_interpolation.cc
extern std::atomic<int> psGpuRetries;
extern std::atomic<int> psGpuRetriesByCause[4];   // innocent, timeout, oom, other (2026-10-11)
#else
// the standard binary's (2026-10-13): the same policy in its GPU hosts (dtfe_metal_host.cc, dtfe_gpu_cuda.cu: a
// failed buffer resumes the sub-domain's batch from its last checkpoint; 3 failures of one chunk send the
// sub-domain to the CPU), defined in averaged_interpolation_1.cc -- zero in a CPU-only build
extern std::atomic<int> dtfeGpuRetries;
extern std::atomic<int> dtfeGpuRetriesByCause[4];   // innocent, timeout, oom, other
// the sub-domain deposits since the last reset that asked for the GPU (--gpu) and ran on the CPU instead
int dtfeGpuFallbacksTaken(bool reset = false);
#endif

struct RunProgress
{
    bool on = false;
    std::string path;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    std::string phase = "starting";
    long done = 0, total = 0;
    double eta = -1.;
    double predicted = -1.;   // the auto-tuner's predicted peak, bytes (-1: none)
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

// the peak memory of this process so far, in bytes; -1 when unknown. macOS: the peak physical footprint (the
// ledger /usr/bin/time -l reports as 'peak memory footprint'; it includes the GPU's shared buffers), else the
// maximum RSS. *maxRss (optional) gets the maximum RSS in bytes either way (ru_maxrss: bytes on macOS, KB on Linux)
inline double runProgressPeak(double *maxRss = nullptr)
{
    double rss = -1.;
    struct rusage ru;
    if ( getrusage( RUSAGE_SELF, &ru ) == 0 )
#ifdef __APPLE__
        rss = double( ru.ru_maxrss );
#else
        rss = double( ru.ru_maxrss ) * 1024.;
#endif
    if ( maxRss ) *maxRss = rss;
#ifdef __APPLE__
    task_vm_info_data_t vm;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if ( task_info( mach_task_self(), TASK_VM_INFO, (task_info_t)&vm, &count ) == KERN_SUCCESS
         and count >= TASK_VM_INFO_REV3_COUNT and vm.ledger_phys_footprint_peak > 0 )
        return double( vm.ledger_phys_footprint_peak );
#endif
    return rss;
}

// the auto-tuner's predicted peak (bytes), for the sidecar and the SUMMARY's predicted-vs-measured line
inline void runProgressSetPredicted(double bytes)
{
    RunProgress &p = runProgressState();
    std::lock_guard<std::mutex> lock( p.m );
    p.predicted = bytes > 0. ? bytes : -1.;
}

inline double runProgressPredicted()
{
    RunProgress &p = runProgressState();
    std::lock_guard<std::mutex> lock( p.m );
    return p.predicted;
}

inline int runProgressGpuRetries()
{
#ifdef PHASE_SPACE
    return psGpuRetries.load();
#else
    return dtfeGpuRetries.load();
#endif
}

// the failed attempts of one cause: 0 innocent victim, 1 timeout, 2 out of memory, 3 other
inline int runProgressGpuRetries(int cause)
{
#ifdef PHASE_SPACE
    return ( cause >= 0 and cause < 4 ) ? psGpuRetriesByCause[cause].load() : 0;
#else
    return ( cause >= 0 and cause < 4 ) ? dtfeGpuRetriesByCause[cause].load() : 0;
#endif
}

// (re)writes the sidecar from the current state; the caller holds the mutex
inline void runProgressWriteLocked(RunProgress &p)
{
    if ( not p.on ) return;
    double const elapsed = std::chrono::duration<double>( std::chrono::steady_clock::now() - p.start ).count();
    double const rss = runProgressRss();
    double const peak = runProgressPeak();
    // the four causes loaded ONCE and the total written as THEIR sum (2026-10-11): a host bumps psGpuRetries and
    // the cause's counter as two increments, and this write may run while another partition's deposit fails
    // (--max-concurrent >= 2), so five separate loads could print a total its causes do not add up to
    int byCause[4];
    for ( int k = 0; k < 4; ++k ) byCause[k] = runProgressGpuRetries( k );
    int const retries = byCause[0] + byCause[1] + byCause[2] + byCause[3];
    std::string const tmp = p.path + ".tmp." + std::to_string( long( ::getpid() ) );
    FILE *f = std::fopen( tmp.c_str(), "w" );
    if ( not f ) return;   // the output folder is not writable now: the progress is a convenience, never fatal
    std::fprintf( f, "{\"pid\": %ld, \"phase\": \"%s\", \"done\": %ld, \"total\": %ld, \"elapsed_s\": %.1f, "
                     "\"eta_s\": %.1f, \"rss_gb\": %.3f, \"gpu_retries\": %d, \"gpu_retries_by_cause\": "
                     "{\"innocent\": %d, \"timeout\": %d, \"oom\": %d, \"other\": %d}, \"predicted_gb\": %.3f, "
                     "\"peak_gb\": %.3f, \"updated\": %ld}\n",
                  long( ::getpid() ), p.phase.c_str(), p.done, p.total, elapsed, p.eta,
                  rss < 0. ? -1. : rss / 1.e9, retries, byCause[0], byCause[1], byCause[2], byCause[3],
                  p.predicted < 0. ? -1. : p.predicted / 1.e9, peak < 0. ? -1. : peak / 1.e9,
                  long( std::time( nullptr ) ) );
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
