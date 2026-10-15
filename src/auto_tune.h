/* Auto-selection of --partition and --max-concurrent from the simulation actually being
   processed (particle count, grid, requested fields) and the machine (physical RAM, cores).
   Called from DTFE_setup once the input is loaded and the options are finalized; it only
   fills in values the user did NOT give -- explicit flags always win, so all existing
   command lines behave exactly as before.

   The peak-memory model is calibrated against measured runs (TNG50-4 512^3 = 45 GB measured
   vs ~50 GB modeled; TNG300-3 5x5x5/mc2 target ~40 GB vs ~36 GB modeled) and deliberately
   errs HIGH so the tuner never picks a configuration that measured tighter than predicted.
   Model constants (bytes/vertex etc.) were pinned with a compiled sizeof probe; see the
   README 'Memory' section for the empirical anchors.

   Memory budget (autoTuneBudget): 60% of physical RAM, and never more than what the OTHER
   programs leave free at start-up minus 4 GB of headroom -- the machine stays usable while a
   run goes (80% of RAM alone let a 64 GB Mac swap 19 GB and lag with a browser, the GUI and
   the desktop open next to a 50 GB run).

   Environment overrides (for tests and unusual machines):
     DTFE_RAM_GB        physical-RAM override in GB (default: queried from the OS); simulates a
                        machine, so the live free-memory term is skipped (tests stay deterministic)
     DTFE_MEM_FRACTION  share of RAM the budget may take (default 0.60; 0.1 .. 0.95)
     DTFE_MEM_BUDGET_GB the budget itself, in GB -- overrides both rules above, but is still capped by what
                        the other programs leave free minus the headroom (unless DTFE_RAM_GB simulates a
                        machine): 2026-10-07 a 52 GB budget on a 64 GB Mac got TNG50-3-Dark z=0 SIGKILLed
     DTFE_FREE_GB       test hook: the memory the other programs leave free, in GB (default: queried); the
                        partition memory gate and the budget cap read it instead of the OS
     DTFE_AUTO_MINN     minimum particle count before auto-partitioning engages (default 2e6;
                        below it a single domain always fits and partition seams are not worth it) */

#ifndef AUTO_TUNE_HEADER
#define AUTO_TUNE_HEADER

#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <limits>
#include <string>
#include <sys/statvfs.h>
#include <sys/stat.h>
#include <fstream>
#include <thread>      // hardware_concurrency: the point-evaluation HitSink blocks, one per thread
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <dirent.h>    // autoTuneServe: the cached partition splits of a snapshot
#include <map>
#include <set>
#include <functional>  // the standard-DTFE split replay (stdTuneSplit)
#include <sstream>
#include "cache_key_lines.h"


#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#include <mach/mach.h>
#else
#include <unistd.h>
#endif
#ifdef OPEN_MP
#include <omp.h>
#endif

#include "user_options.h"
#include "particle_data.h"
#include "ps_point_eval.h"   // psEstimateMeanStreams: the sample-point stream count, from the data
#include "message.h"
// for PSCausticClass::CausticMask, so the --ps-caustics per-cell accumulator is budgeted at its
// real width rather than a literal that would drift the next time a bit is added
#include "CGAL_triangulation/ps_caustic_class.h"


// Physical RAM in bytes (DTFE_RAM_GB env overrides; 8 GB fallback if the query fails).
inline double autoTunePhysicalRAM()
{
    if ( const char *env = std::getenv("DTFE_RAM_GB") )
    {
        double v = std::atof(env);
        if ( v > 0. ) return v * 1.e9;
    }
#ifdef __APPLE__
    int64_t mem = 0;
    size_t len = sizeof(mem);
    if ( sysctlbyname("hw.memsize", &mem, &len, NULL, 0)==0 and mem>0 )
        return double(mem);
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long pageSize = sysconf(_SC_PAGE_SIZE);
    if ( pages>0 and pageSize>0 )
        return double(pages) * double(pageSize);
#endif
    return 8.e9;
}

// The memory the PS-DTFE CPU deposit threads' private accumulator copies may take per partition:
// 20% of physical RAM (at least 2 GB), or DTFE_DEPOSIT_MEM_GB. ps_interpolation.cc lowers its
// thread count to fit (a 256^3 grid with density, velocity and dispersion is ~1 GB per copy, so
// 2 GB allowed only 3 threads); the partition model below budgets the same amount.
inline double psDepositThreadBytes()
{
    if ( const char *env = std::getenv("DTFE_DEPOSIT_MEM_GB") )
    {
        double const v = std::atof(env);
        if ( v >= 0. ) return v * 1.e9;
    }
    return std::max( 2.e9, 0.2 * autoTunePhysicalRAM() );
}

// Bytes free for NEW allocations right now (what every program, this one included, leaves free);
// < 0 when the OS cannot tell. DTFE_FREE_GB overrides it (tests).
//   macOS: RAM minus Activity Monitor's "Memory Used" (app memory = anonymous - purgeable,
//          + wired + compressed); Linux: MemAvailable. File cache counts as free: it is dropped.
inline double autoTuneSystemFree(double *ownBytes = nullptr)
{
    if ( const char *env = std::getenv("DTFE_FREE_GB") )
    {
        double const v = std::atof(env);
        if ( v > 0. ) { if ( ownBytes ) *ownBytes = 0.; return v * 1.e9; }
    }
#ifdef __APPLE__
    int64_t mem = 0;
    size_t len = sizeof(mem);
    vm_size_t page = 0;
    vm_statistics64_data_t vm;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if ( sysctlbyname("hw.memsize", &mem, &len, NULL, 0)!=0 or mem<=0
         or host_page_size(mach_host_self(), &page)!=KERN_SUCCESS
         or host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vm, &count)!=KERN_SUCCESS )
        return -1.;
    double const used = ( double(vm.internal_page_count) - double(vm.purgeable_count)
                          + double(vm.wire_count) + double(vm.compressor_page_count) ) * double(page);
    double own = 0.;
    task_vm_info_data_t ti;
    mach_msg_type_number_t tcount = TASK_VM_INFO_COUNT;
    if ( task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&ti, &tcount)==KERN_SUCCESS )
        own = double(ti.phys_footprint);
    if ( ownBytes ) *ownBytes = own;
    return double(mem) - used;
#else
    double availKB = -1., ownKB = 0.;
    char key[64];
    double value;
    if ( FILE *f = std::fopen("/proc/meminfo", "r") )
    {
        while ( std::fscanf(f, "%63s %lf kB\n", key, &value)==2 )
            if ( std::string(key)=="MemAvailable:" ) { availKB = value; break; }
        std::fclose(f);
    }
    if ( FILE *f = std::fopen("/proc/self/status", "r") )
    {
        char line[256];
        while ( std::fgets(line, sizeof(line), f) )
            if ( std::sscanf(line, "VmRSS: %lf kB", &value)==1 ) { ownKB = value; break; }
        std::fclose(f);
    }
    if ( ownBytes ) *ownBytes = ownKB * 1024.;
    return availKB < 0. ? -1. : availKB * 1024.;
#endif
}

// Bytes the OTHER programs leave free right now, plus what this process already holds (its
// particles etc., which the run's model counts itself); < 0 when the OS cannot tell.
inline double autoTuneFreeForRun()
{
    double own = 0.;
    double const free = autoTuneSystemFree( &own );
    return free < 0. ? -1. : free + own;
}

// The memory budget the tuner plans against, and a one-line account of how it was set.
inline double autoTuneBudget(double ramBytes, std::string &why)
{
    char buf[256];
    bool const simulated = std::getenv("DTFE_RAM_GB") != NULL and std::atof(std::getenv("DTFE_RAM_GB")) > 0.;
    double const headroom = 4.e9;
    if ( const char *env = std::getenv("DTFE_MEM_BUDGET_GB") )
    {
        double v = std::atof(env);
        if ( v > 0. )
        {
            snprintf( buf, sizeof(buf), "%.3g GB set by DTFE_MEM_BUDGET_GB", v );
            why = buf;
            // ... but never more than the other programs leave free minus the headroom: the budget is a
            // plan, the machine's free memory the fact (2026-10-07: 52 GB on a 64 GB Mac, SIGKILL at 82 min).
            // Only a cut of more than 0.5 GB is one: 44 GB free less 4 against 40 asked is the same budget, and
            // the reason line said "asks 40 GB, but ..." for it (2026-10-08)
            double const freeForRun = simulated ? -1. : autoTuneFreeForRun();
            if ( freeForRun > 0. and freeForRun - headroom < v * 1.e9 - 0.5e9 )
            {
                double const capped = std::max( freeForRun - headroom, std::min( v * 1.e9, 1.e9 ) );
                snprintf( buf, sizeof(buf), "%.3g GB: DTFE_MEM_BUDGET_GB asks %.3g GB, but the other programs leave "
                          "%.3g GB free, minus %.3g GB headroom", capped/1.e9, v, freeForRun/1.e9, headroom/1.e9 );
                why = buf;
                return capped;
            }
            return v * 1.e9;
        }
    }
    double frac = 0.60;
    if ( const char *env = std::getenv("DTFE_MEM_FRACTION") )
    {
        double v = std::atof(env);
        if ( v >= 0.1 and v <= 0.95 ) frac = v;
    }
    double budget = frac * ramBytes;
    snprintf( buf, sizeof(buf), "%.3g GB = %.0f%% of %.3g GB RAM", budget/1.e9, 100.*frac, ramBytes/1.e9 );
    why = buf;
    double const freeForRun = simulated ? -1. : autoTuneFreeForRun();
    if ( freeForRun > 0. and freeForRun - headroom < budget )
    {
        budget = std::max( freeForRun - headroom, 1.e9 );
        snprintf( buf, sizeof(buf), "%.3g GB: other programs leave %.3g GB free, minus %.3g GB headroom "
                  "(%.0f%% of RAM would be %.3g GB)", budget/1.e9, freeForRun/1.e9, headroom/1.e9,
                  100.*frac, frac*ramBytes/1.e9 );
        why = buf;
    }
    return budget;
}

inline int autoTuneCores()
{
#ifdef OPEN_MP
    return std::max( omp_get_max_threads(), 1 );
#else
    return 1;
#endif
}

// Full-grid bytes per cell for one field selection, for ONE of the two phases:
//
//   phase 0 = DEPOSIT. What is resident while the partition triangulations run, i.e. the term
//             that coexists with partitionBytes(). Includes the PS weight grids.
//   phase 1 = POST. What is resident afterwards, when the derived fields are computed. The
//             weights have been freed by normalizePhaseSpace (swap-with-empty, quantities.cc)
//             at DTFE.cpp:659 BEFORE computeDivergenceShearVorticity allocates div/shear/vort.
//
// The two peaks are disjoint sets, so the caller takes max(deposit, post) rather than summing.
// A single blended number used to bound the deposit only by luck: for a density+velocity+
// dispersion volume-weighted run the model said 44 B/cell where the deposit really needs 64.
//
// 'ps' carries the PS flags: they gate real full-grid accumulators and are all finalized before
// the DTFE.cpp call site. Byte counts verified against quantities.cc / ps_interpolation.cc with
// a compiled sizeof probe (Real=4, Pvector<Real,n> = 4n exactly, no padding).
// Number of query points a '--sample-points' file holds, WITHOUT reading it (auto-tune runs
// long before psPointEvalInit). Raw binary is N x 3 float64 = 24 B/point; a text file is
// probed the same way readSamplePoints does and estimated from its average line length.
// Returns 0 when the file is absent/unreadable, which switches the model off.
inline size_t autoTuneSamplePointCount(std::string const &file)
{
    if ( file.empty() ) return 0;
    struct stat st;
    if ( stat( file.c_str(), &st ) != 0 || st.st_size <= 0 ) return 0;
    double const bytes = double(st.st_size);

    std::ifstream f( file.c_str(), std::ios::binary );
    if ( not f ) return 0;
    char head[4096];
    f.read( head, sizeof(head) );
    std::streamsize const got = f.gcount();
    bool isText = true;
    for (std::streamsize i = 0; i < got; ++i)
    {
        char const c = head[i];
        bool const numeric = (c >= '0' && c <= '9') || c == '+' || c == '-' || c == '.'
                             || c == 'e' || c == 'E' || c == ' ' || c == '\t'
                             || c == '\n' || c == '\r';
        if ( not numeric ) { isText = false; break; }
    }
    if ( not isText )
        return size_t( bytes / (3. * sizeof(double)) );

    // text: count the newlines in the probe to get an average bytes-per-triplet
    long lines = 0;
    for (std::streamsize i = 0; i < got; ++i) if ( head[i] == '\n' ) ++lines;
    double const perLine = lines > 0 ? double(got) / double(lines) : 30.;
    return size_t( bytes / perLine );
}

// The Eulerian bounding box of the '--sample-points' (2026-10-11), for the stream estimate on the points' own slab
// (psEstimateMeanStreams' region): binary files are sampled at up to 65536 evenly strided points (a plane's bounds
// to a pixel), text files read whole. Periodic: wrapped into the box first, as the evaluation does. False when the
// file cannot be read.
inline bool autoTuneSamplePointBounds(std::string const &file, double const boxLo[NO_DIM], double const boxLen[NO_DIM],
                                      bool const periodic, double lo[NO_DIM], double hi[NO_DIM])
{
    size_t const n = autoTuneSamplePointCount( file );
    if ( n == 0 ) return false;
    std::ifstream f( file.c_str(), std::ios::binary );
    if ( not f ) return false;
    for (int d=0; d<NO_DIM; ++d) { lo[d] = std::numeric_limits<double>::max(); hi[d] = -std::numeric_limits<double>::max(); }
    auto add = [&](double const *v) {
        for (int d=0; d<NO_DIM; ++d)
        {
            double x = v[d];
            if ( periodic and boxLen[d] > 0. ) x -= boxLen[d] * std::floor( (x - boxLo[d]) / boxLen[d] );
            lo[d] = std::min( lo[d], x );
            hi[d] = std::max( hi[d], x );
        }
    };
    char head[4096];
    f.read( head, sizeof(head) );
    std::streamsize const got = f.gcount();
    bool isText = true;
    for (std::streamsize i = 0; i < got and isText; ++i)
    {
        char const c = head[i];
        isText = (c >= '0' && c <= '9') || c == '+' || c == '-' || c == '.' || c == 'e' || c == 'E' || c == ' '
                 || c == '\t' || c == '\n' || c == '\r';
    }
    f.clear();
    size_t seen = 0;
    double v[NO_DIM];
    if ( isText )
    {
        f.seekg( 0, std::ios::beg );
        while ( f >> v[0] )
        {
            bool full = true;
            for (int d=1; d<NO_DIM and full; ++d) full = bool( f >> v[d] );
            if ( not full ) break;
            add( v );
            ++seen;
        }
    }
    else
    {
        size_t const step = std::max( size_t(1), n / 65536 );
        for (size_t i = 0; i < n; i += step)
        {
            f.seekg( std::streamoff( i * NO_DIM * sizeof(double) ), std::ios::beg );
            if ( not f.read( reinterpret_cast<char*>(v), sizeof(v) ) ) break;
            add( v );
            ++seen;
        }
        f.clear();   // ... and the last point, so a strided sample cannot miss the far edge
        f.seekg( std::streamoff( (n - 1) * NO_DIM * sizeof(double) ), std::ios::beg );
        if ( f.read( reinterpret_cast<char*>(v), sizeof(v) ) ) add( v );
    }
    if ( seen == 0 ) return false;
    for (int d=0; d<NO_DIM; ++d) if ( not std::isfinite( lo[d] ) or not std::isfinite( hi[d] ) ) return false;
    return true;
}


struct AutoTunePS
{
    bool phaseSpace   = false;
    bool volumeWeighted = false;   // --ps-volume-weighted
    bool exactDeposit = false;     // --ps-exact-deposit
    bool caustics     = false;     // --ps-caustics
    bool haloRelease  = false;     // --ps-halo-release (sampled deposit: the released-tet multiplicity grid)
};

// The PS deposit's MOMENT grids as 64-bit fixed-point sums (2026-10-14, ps_deposit_params.h PSFxGrid): on by
// default on the GPU (both builds) and on the CPU of the single build; PS_METAL_FLOAT_MOMENTS=1 /
// PS_GPU_FLOAT_MOMENTS=1 bring back the float sums (ps_interpolation.cc fxGpuWanted / fxMomWanted). Read here so the
// model follows an A/B run too.
inline bool autoTunePSMomentsFixed()
{
    for (char const *n : { "PS_METAL_FLOAT_MOMENTS", "PS_GPU_FLOAT_MOMENTS" })
        if ( char const *e = std::getenv(n) ) if ( std::atoi(e) != 0 ) return false;
    return true;
}

// One pass's MOMENT accumulators in the PS deposit, in Real bytes per sub-grid cell (2026-10-14): the grids
// ps_interpolation.cc holds as fixed-point words while the pass deposits ('fxg' -- velocity, gradient,
// dispersion, scalar and its gradient, the moment weight, the dispersion's own mean and weight, the exact and the
// released multiplicities), with the same selection rules as autoTuneBytesPerCell's deposit phase below. In the
// single build the words take 8 B per component where these floats took 4, and the deposit threads share ONE set
// of them (no private copies). The dispersion weight is the density there when one is selected (not accumulated).
inline double autoTuneFxMomentBytes(Field &f, AutoTunePS const &ps)
{
    if ( not f.selected() or not ps.phaseSpace ) return 0.;
    double const R = double(sizeof(Real));
    bool const gradient = f.velocity_gradient or f.velocity_divergence or f.velocity_shear
                          or f.velocity_vorticity or f.velocity_vweb;
    bool const densityGrid = f.density or f.velocity_tweb;
    bool const needWeight = f.velocity or f.velocity_gradient or f.velocity_dispersion
                            or f.velocity_divergence or f.velocity_shear or f.velocity_vorticity
                            or f.velocity_vweb or f.scalar or f.scalar_gradient;
    double b = 0.;
    if ( f.velocity or (f.velocity_dispersion and not ps.volumeWeighted) ) b += 3.*R;
    if ( gradient ) b += 9.*R;
    if ( f.velocity_dispersion ) b += 6.*R;
    if ( f.scalar ) b += noScalarComp*R;
    if ( f.scalar_gradient ) b += noScalarGradComp*R;
    if ( needWeight and not (densityGrid and not ps.volumeWeighted) ) b += R;                 // mass_weight
    if ( ps.volumeWeighted and f.velocity_dispersion ) b += 3.*R + (densityGrid ? 0. : R);   // disp_velocity (+ disp_weight)
    if ( ps.exactDeposit ) b += R;                                                           // the exact multiplicity
    if ( ps.haloRelease and not ps.exactDeposit ) b += R;                                    // the released multiplicity
    return b;
}

inline double autoTuneBytesPerCell(Field &f, AutoTunePS const &ps, int const phase)
{
    if ( not f.selected() ) return 0.;
    double const R = double(sizeof(Real));
    double b = 0.;
    bool const gradient = f.velocity_gradient or f.velocity_divergence or f.velocity_shear
                          or f.velocity_vorticity or f.velocity_vweb;
    // 'density||tweb' mirrors the tweb->density folding at DTFE.cpp:187, which happens AFTER
    // autoTunePartitioning is called -- so use this predicate anywhere the deposit sees
    // field.density, not the raw f.density.
    bool const densityGrid = f.density or f.velocity_tweb;
    if ( densityGrid ) b += R;
    // dispersion allocates the velocity grid too -- EXCEPT volume-weighted, where it carries
    // its own disp_velocity mean instead (counted in the deposit-phase PS block below)
    if ( f.velocity or (f.velocity_dispersion and not (ps.phaseSpace and ps.volumeWeighted)) )
        b += 3.*R;
    if ( gradient ) b += 9.*R;
    if ( f.velocity_dispersion ) b += 6.*R;
    if ( f.scalar ) b += noScalarComp*R;
    if ( f.scalar_gradient ) b += noScalarGradComp*R;

    if ( phase == 1 )
    {
        // derived fields: allocated only after the deposit loop (DTFE.cpp:83, ~1238-1257)
        if ( f.velocity_divergence ) b += R;
        if ( f.velocity_shear ) b += 5.*R;
        if ( f.velocity_vorticity ) b += 3.*R;
        if ( f.velocity_std ) b += R;
        if ( f.velocity_tweb ) b += R + 3.*R;                      // web label + eigenvalues
        if ( f.velocity_vweb ) b += R + 3.*R;
        // T-web tidal stage (computeTidalWebClassification, DTFE.cpp:1126-1200): all live at
        // once -- work(N) 8 B + deltaK/tensK fftw_complex ~16.03 B + tens[0..5] 6*R. 48 B/cell
        // on top of every resident grid; invisible at the 256-512 sizes this was calibrated on
        // (6.5 GB at 512^3) but 51.6 GB at 1024^3.
        if ( f.velocity_tweb ) b += 8. + 16.032 + 6.*R;
    }
    else if ( ps.phaseSpace )
    {
        // ---- deposit-phase PS weight grids (freed at DTFE.cpp:659, before the derived fields) ----
        bool const needWeight = f.velocity or f.velocity_gradient or f.velocity_dispersion
                                or f.velocity_divergence or f.velocity_shear or f.velocity_vorticity
                                or f.velocity_vweb or f.scalar or f.scalar_gradient;
        // ps_interpolation.cc: weightIsDensity = needWeight && field.density && !psVolWeighted.
        // --ps-volume-weighted therefore BREAKS the aliasing: the weight carries Eulerian-volume
        // shares, which are not the density's mass sums, so it needs its own grid even when
        // density is selected. That is the production default, not a corner case.
        if ( needWeight and not (densityGrid and not ps.volumeWeighted) ) b += R;   // mass_weight
        // dispersion stays MASS-weighted under --ps-volume-weighted, so it carries its OWN
        // mean+normalizer (ps_interpolation.cc: dispNeedsOwnWeight)
        if ( ps.volumeWeighted and f.velocity_dispersion ) b += R + 3.*R;           // disp_weight + disp_velocity
    }

    if ( ps.phaseSpace )
    {
        b += R;                                                    // stream_count (both phases)
        b += R;                                                    // hidden_streams (both phases, DTFE.cpp)
        if ( phase == 0 )      b += 2.;                            // subSampleMassFlag + flippedFlag, the deposit's byte accumulators
        if ( ps.caustics )     b += R;                           // caustic_bits (DTFE.cpp:512)
        if ( ps.exactDeposit ) b += R;                             // tet_touch    (DTFE.cpp:519)
        /* orientBits, the deposit's per-cell accumulator (ps_interpolation.cc): a CausticMask,
           not a byte, since the A4 indicator needs bit 8. Deposit phase only -- freed before the
           derived fields are allocated. Counted once per selected Quantities, like caustic_bits;
           when only one of the u/a pair is deposited that over-counts slightly, deliberately,
           since for a memory budget the safe rounding direction is up. */
        if ( ps.caustics and phase == 0 ) b += double(sizeof(PSCausticClass::CausticMask));
        // --ps-halo-release on the sampled deposit: released tets add their volume fraction to
        // a per-cell Real accumulator (releasedMult, ps_interpolation.cc). Deposit phase only.
        if ( ps.haloRelease and not ps.exactDeposit and phase == 0 ) b += R;
    }
    return b;
}


// Chooses --partition / --max-concurrent when the user did not give them. 'metalActive' =
// the GPU deposit will actually run (flag given AND compiled in). Mutates userOptions.
// What the tuner decided and predicted, for '--auto-tune-report' (one machine-readable line; the
// GUI's memory check parses it). valid = false in the modes the tuner does not handle.
struct AutoTuneReport
{
    bool   valid = false;
    int    partition = 1, maxConcurrent = 1;
    double predicted = 0., fixed = 0., budget = 0., pts = 0.;   // bytes
    double perPartition = 0.;   // bytes ONE partition in flight adds (the run's memory gate waits for this much)
    double lookAhead = 0.;      // bytes the partition look-ahead adds to 'predicted' (0: none runs)
    double gpuSnapshot = 0.;    // bytes of the GPU deposit's host snapshot, held by ONE partition at a time (in 'predicted')
    double streamsEstimate = -1., streamsBudgeted = 0.;
    std::string line() const
    {
        if ( not valid ) return "AUTO-TUNE-REPORT unsupported";
        char b[400];
        snprintf( b, sizeof(b), "AUTO-TUNE-REPORT partition=%d mc=%d predicted_gb=%.2f budget_gb=%.2f "
                  "fixed_gb=%.2f pts_gb=%.2f streams_est=%.3f streams_budget=%.3f over_budget=%d",
                  partition, maxConcurrent, predicted/1.e9, budget/1.e9, fixed/1.e9, pts/1.e9,
                  streamsEstimate, streamsBudgeted, predicted > budget ? 1 : 0 );
        return b;
    }
};

/* '--serve' on data too large for ONE tessellation: turn it into a composite server (ps_point_eval.cc)
   of n^3 partitions -- when '--tessellation-cache' lets the partitions live on disk. Without the
   cache a composite holds every partition in memory, which is MORE than one tessellation, so the
   tuner then only warns. Model: a tessellation costs kDT per vertex plus ~160 B/vertex for the
   server's cell index; the particle array (freed once every partition is built) coexists with the
   concurrent builds. Picks the smallest n (2..8) that leaves room for two partitions next to the
   particles, then as many concurrent builds as fit. A user-given --partition is never touched. */
/* A Lagrangian partition's padding on one axis, in units of the partition's own width, for an n^3 split of N
   particles (2026-10-11): with '--ps-lag-padding <k>' (the default since 2026-10-12, k = 8) k mean spacings -- k n /
   N^(1/NO_DIM) partition widths, at most one period (n widths) on a periodic run -- and at least 4.2
   '--ps-alpha-shape' cuts on a non-periodic one, with no floor; with '--ps-lag-padding 0' the Eulerian '--padding'
   fraction floored at 'floorFrac' of a partition (DTFE.cpp's PSPartitionPlan). One definition for every model below. */
inline double autoTunePadWidths(User_options const &u, double const padFracAxis, int const n, double const N,
                                double const floorFrac)
{
#ifdef PHASE_SPACE
    if ( u.psLagPadding > Real(0.) and n > 1 and N > 0. )
    {
        double k = double(u.psLagPadding);
        if ( not u.periodic and u.psAlphaShape > Real(0.) ) k = std::max( k, 4.2 * double(u.psAlphaShape) );
        double const w = k * double(n) / std::pow( N, 1. / double(NO_DIM) );
        return u.periodic ? std::min( w, double(n) ) : w;     // DTFE.cpp caps a periodic padding at one period
    }
#endif
    return std::max( floorFrac, padFracAxis * double(n) );
}

/* The partition splits of THIS snapshot already in the tessellation cache folder: n -> the number of
   cached n^3 partition files. Reads only each file's uncompressed header (magic, length, descriptor) and
   compares the lines that name the snapshot and the build -- exactly the text tessellation_cache.h writes
   (cache_key_lines.h) -- plus the periodicity and length unit; the rest of a partition's key follows from
   the split. Read-only. */
inline std::map<int,int> cachedServeSplits(User_options const &u)
{
    std::map<int,int> out;
    if ( u.tessellationCacheDir.empty() ) return out;
    std::string const wantInput = cacheFileLine( "input", u.inputFilename ) + '\n';
#ifdef PHASE_SPACE
    std::string const wantLag = cacheFileLine( "lagrange", u.lagrangianInputFilename ) + '\n';
    // tessellation_cache.h's 'lagPadding=<k>' line, written by every partition of a k > 0 run
    std::string wantPad;
    if ( u.psLagPadding > Real(0.) )
    {
        std::ostringstream pad;
        pad << "\nlagPadding=" << u.psLagPadding << '\n';
        wantPad = pad.str();
    }
#endif
    std::string const wantBuild = cacheBuildLines();
    std::ostringstream other;
    other << "MpcUnit=" << u.MpcValue << '\n' << "periodic=" << (u.periodicInput ? 1 : 0) << '\n';
    std::string const wantOther = other.str();
    DIR *dir = ::opendir( u.tessellationCacheDir.c_str() );
    if ( dir == nullptr ) return out;
    while ( struct dirent *e = ::readdir( dir ) )
    {
        std::string const name = e->d_name;
        if ( name.size() < 11 or name.compare( 0, 5, "tess_" ) != 0 or name.compare( name.size() - 5, 5, ".tess" ) != 0 )
            continue;
        std::ifstream f( ( u.tessellationCacheDir + "/" + name ).c_str(), std::ios::binary );
        char magic[8];
        uint32_t len = 0;
        if ( not f.read( magic, 8 ) or std::string( magic, 8 ) != "DTFETESS" ) continue;
        if ( not f.read( reinterpret_cast<char*>(&len), 4 ) or len == 0 or len > (1u << 20) ) continue;
        std::string desc( len, '\0' );
        if ( not f.read( &desc[0], len ) ) continue;
        if ( desc.find( wantBuild ) == std::string::npos or desc.find( wantInput ) == std::string::npos
             or desc.find( wantOther ) == std::string::npos )
            continue;
#ifdef PHASE_SPACE
        if ( desc.find( wantLag ) == std::string::npos ) continue;
        // the partition padding (2026-10-12): a split cached under another padding (the Eulerian rule before
        // 2026-10-12 writes no line) would miss in the run, so it must not steer the split either
        if ( wantPad.empty() ? desc.find( "\nlagPadding=" ) != std::string::npos : desc.find( wantPad ) == std::string::npos )
            continue;
#endif
        static char const SPLIT[] = "\npartition=1 ";
        size_t const p = desc.find( SPLIT );
        if ( p == std::string::npos ) continue;
        int a = 0, b = 0, c = 0;
        if ( std::sscanf( desc.c_str() + p + sizeof(SPLIT) - 1, "%d,%d,%d", &a, &b, &c ) == 3 and a == b and ( NO_DIM == 2 or b == c ) and a > 1 )
            ++out[a];
    }
    ::closedir( dir );
    return out;
}

inline void autoTuneServe(User_options &u, size_t const nParticles)
{
    if ( u.partitionOn or u.partitionGiven or u.partNo >= 0 )
        return;
    double const N = double(nParticles);
    std::string budgetWhy;
    double const budget = autoTuneBudget( autoTunePhysicalRAM(), budgetWhy );
    double const partBytes = double(sizeof(Particle_data));
    // transient of the build: the particle copy that coexists with the growing tessellation, or
    // (TBB parallel insertion, PS-DTFE) the sorted point+payload copy plus CGAL's own copy of it
    // while the particles are already freed -- triangulation.cpp
#if defined(PARALLEL_TRIANGULATION) && NO_DIM==3
    double const kBuild = 330.;
#else
    double const kBuild = 160.;
#endif
#ifdef PHASE_SPACE
    double const kDT = 666. + kBuild;
    double const singlePadFloor = 0.10;             // PS: Lagrangian padding floor (DTFE.cpp)
#else
    double const kDT = 648. + kBuild;
    double const singlePadFloor = 0.;
#endif
    double padFrac = 0.;
    for (int d=0; d<NO_DIM; ++d)
    {
        double const boxLen = double(u.boxCoordinates[2*d+1] - u.boxCoordinates[2*d]);
        if ( boxLen > 0. )
            padFrac = std::max( padFrac, double(u.paddingLength[2*d] + u.paddingLength[2*d+1]) / (2.*boxLen) );
    }
    auto tessBytes = [&](int n) -> double
    {
        double f = 1.;
        double const floorFrac = (n == 1) ? singlePadFloor : 0.3;
        for (int d=0; d<NO_DIM; ++d)
            f *= 1. + 2.*autoTunePadWidths( u, padFrac, n, N, floorFrac );
        return N / std::pow( double(n), NO_DIM ) * f * kDT;
    };
    double const single = N*partBytes + tessBytes(1);
    if ( single <= budget )
        return;

    MESSAGE::Message message( u.verboseLevel );
    char buf[640];
    if ( u.tessellationCacheDir.empty() )
    {
        MESSAGE::Warning warning( u.verboseLevel );
        warning << "AUTO-TUNE (serve): one tessellation of these " << nParticles << " particles needs ~"
                << single/1.e9 << " GB, over the memory budget of " << budget/1.e9 << " GB (" << budgetWhy
                << "). Pass '--tessellation-cache DIR' (a local directory) and the server becomes a composite of "
                << "partition tessellations of which only a few are held in memory at once; continuing with "
                << "one tessellation.\n" << MESSAGE::EndWarning;
        return;
    }
    // the coarsest split that leaves room for two partitions next to the particles while they are built
    int nMin = 2;
    while ( nMin < 8 and 2.*tessBytes(nMin) > budget - N*partBytes ) ++nMin;
    /* Which split. A split this snapshot already has in the cache is reused -- the finest complete one, else
       the one with the most cached partitions -- since any other means building (and storing) every
       partition again: TNG100-3-Dark's 4^3 set took 469 s and 52 GB. With nothing cached, the split that
       holds ~8 partitions in memory at once, the number a zoom touches: measured on TNG100-3-Dark z=0
       (2026-10-03), 3^3 held 4 and a cold 512^2 zoom took 19.7 s, 4^3 held 10 and took 9.8 s (repeats
       5.0 vs 4.6 s). Before then the server always took the coarsest split that fit. */
    std::map<int,int> const cached = cachedServeSplits( u );
    int n = -1;
    char const *why = "";
    for (auto it = cached.rbegin(); it != cached.rend() and n < 0; ++it)
        if ( it->first >= nMin and it->second >= int( std::pow( double(it->first), NO_DIM ) ) )
        { n = it->first; why = "the finest split fully in the tessellation cache"; }
    if ( n < 0 )
    {
        int best = 0;
        for (auto const &kv : cached)
            if ( kv.first >= nMin and kv.second > best ) { best = kv.second; n = kv.first; why = "the split with the most cached partitions"; }
    }
    if ( n < 0 )
    {
        // once built the particles are gone, so the budget holds partitions only; stop where a finer split
        // no longer shrinks them (on a small set the padding shells dominate, and 8^3 would only cost)
        n = nMin;
        while ( n < 8 and 8.*tessBytes(n) > budget and tessBytes(n+1) < 0.75 * tessBytes(n) ) ++n;
        why = cached.empty() ? "nothing cached yet: about 8 partitions held in memory, as many as a zoom touches"
                             : "the cached splits are too coarse for the memory budget: about 8 partitions held in memory";
    }
    double const room = std::max( 0., budget - N*partBytes );
    int conc = int( std::floor( room / std::max( tessBytes(n), 1. ) ) );
    conc = std::max( 1, std::min( conc, autoTuneCores() ) );
    u.partition = std::vector<size_t>( NO_DIM, size_t(n) );
    u.partitionOn = true;
    if ( not u.maxConcurrentOn )
        u.maxConcurrent = conc;
    snprintf( buf, sizeof(buf),
              "AUTO-TUNE (serve): one tessellation would need ~%.1f GB, over the %.1f GB budget -> serving a "
              "composite of %d^3 partitions (%s; ~%.1f GB each, %d built at a time); the resident set is sized "
              "once they are built\n", single/1.e9, budget/1.e9, n, why, tessBytes(n)/1.e9,
              u.maxConcurrentOn ? u.maxConcurrent : conc );
    message << buf << MESSAGE::Flush;
}

#ifndef PHASE_SPACE
/* ---- Standard DTFE: what one partition's thread sub-triangulations hold, from the data (2026-10-13) ----------------
   The 2026-10-13 audit measured the standard '--exact-average' run (the production default since that day) at a peak
   footprint 25-30% above the sampled one: TNG100-3-Dark z=1 (3^3, 9 threads) 37.4 vs 30.0 GB, TNG50-3-Dark z=0 (4^3,
   4 threads) 44.5 vs 34.1 GB, and asked for an exact-average term. What the code allocates says there is none:
     - CPU (averaged_interpolation_1.cc): the exact loop clips each tetrahedron against its cell window with r3d on the
       stack -- no per-tetrahedron or per-cell storage that the Monte-Carlo loop does not have;
     - GPU (dtfe_metal_host.cc / dtfe_gpu_cuda.cu): BOTH averages extract the same flat arrays of the tetrahedra that
       touch the thread's region (vertices 48 B, densities 16, velocities 48, volume, count, flat index, Morton key
       4 each), sort them in place, upload them into buffers of the same size, and deposit into the same per-call
       sub-grid (device + host copy). The exact kernel's only extra is its work-item table (4 B per tetrahedron and
       4 B per 32 items), which the sampled item kernel builds as well.
   What changes is WHEN memory is held. The exact kernel is several times slower, and the thread sub-domains of a
   partition queue on the ONE device (ctxMutex): every thread then waits with its triangulation and its uploaded
   batch alive, so all of a partition's sub-triangulations coexist -- the worst case this model always assumed. In a
   sampled run the GPU drains a batch in about a second, the small sub-domains finish and free theirs before the
   largest one is built, and the peak stays below the worst case. So the model needs the worst case to be RIGHT, and
   it was not: it took every sub-box to hold the box-average particle count, (N / P^3) x the padding factor. The
   load-balanced split (optimalPartitionSplit) gives a dense partition a small VOLUME, while the padding is a fixed
   length (k mean spacings), so on clustered data the padding shells hold most of the vertices: in that TNG50 z=0 run
   one partition's 4 threads triangulated 35.4M particles (10.6M in one sub-box) where the model counted 7.8M -- it
   predicted 39.9 GB, and the exact run peaked at 44.5. (TNG50 at z = 127 had 6.5M, and every snapshot's busiest
   partition holds more as structure grows: 8.4 -> 35.4M from z = 127 to 0 at --padding 20.)
   Now the plan replays the run's own split on the particles: a coarse count grid of the box (or, without periodic
   wrap, of the box and its padding) with a summed-area table gives the particles in any box; the partition split
   (optimalSplit along x, then y per slab, then z per column -- optimalPartitionSplit's order, on bins of the run's
   grid cells), the thread grid of each partition (parallelGrid, as DTFE_parallel) and its split are replayed with
   the run's functions, and every padded thread box is counted, periodic images included. A partition costs:
     - per padded vertex (+6% for the padding test's dummy points): its particle copy, which coexists with the
       growing triangulation, + kDT (+ the TBB insert's copies with '--parallel-triangulation');
     - on the GPU, per tetrahedron of the thread's own box (6.77 per particle, x1.3 for the ones crossing its faces):
       the flat arrays, one sort copy and the item table, which turn into the uploaded batch one array at a time;
     - its sub-grids: the threads' copies, the partition's own grids, and the GPU deposit's host and device grids;
     - its padded particle copy (a partition), or the periodic copies of the whole box (one domain);
   and the plan is the busiest partition's sum, on top of the particles and the full-resolution grids ('fixed', as
   before). Replayed on the logs of the audit's runs and of the 2026-10-12/13 production runs
   (38 runs: TNG50/100/300, z = 127 .. 0, sampled and exact, from their 'Thread i had' lines) it bounds every measured
   peak footprint, by 19-90%: TNG50 z=0 exact 44.5 GB -> 58.7 predicted, at --padding 15 40.8 -> 56.2 (the uniform
   count: 39.9 and 37.3, the only two runs it put below their peak -- both exact, both z = 0), TNG100 z=1 exact 37.4
   -> 47.2, TNG50 z=127 exact 31.6 -> 37.6, TNG300 z=2 sampled 33.4 -> 42.9. It stays high because 'fixed' holds the post-processing grids (168 B/cell for the production fields)
   next to the triangulations, and the footprints need that sum: the peak is in the post-processing, where macOS still
   counts much of the freed triangulation memory (TNG300 z=2: 33.4 GB against 22.6 GB of post grids and 27.4 GB for
   the whole deposit phase). On sub-GB runs the same retention can put the footprint up to ~30% above the plan (a
   0.9M-particle crossed-wave set at 4^3 partitions measured 0.71 GB against 0.54 predicted; the uniform count said
   0.24): the footprint ratchets up over the partitions. The counted padded vertices agree with the run's own to ~5% on clustered test
   sets (tests/dtfe_auto_tune_check.sh). The count grid is one pass over the particles (128^NO_DIM cells: 0.07 s for
   1.7e7 particles); a full search is <= 60 plans (~0.5 s, independent of N); a run without particles in memory keeps
   the uniform count. DTFE_AUTO_TUNE_DEBUG=1 prints the busiest partition's predicted padded count per thread (for
   comparison with the run's own 'Thread i had' lines). */

// the run's own split rules, defined in subpartition.h (compiled into DTFE.cpp, the only caller of the tuner)
void optimalSplit(std::vector<size_t> &counts, size_t const noPartitions, size_t *partitionIndices);
void parallelGrid(size_t const noAvailableProcessors, User_options const &userOptions, size_t *grid);

// The particles in any box, from a coarse count grid and its summed-area table: the density is taken as constant
// inside a coarse cell, so the cumulative count is multilinear in each cell and a box costs 2^NO_DIM interpolations.
// Periodic: the grid covers the box and wraps (a box reaching past a face counts the periodic images, as
// findParticlesInBox selects them); otherwise it covers the box plus its padding (a '--region' crop's particles).
struct StdTuneDensity
{
    bool   ok = false, periodic = false;
    size_t m[NO_DIM] = {};               // cells per axis
    double lo[NO_DIM] = {}, cell[NO_DIM] = {};
    std::vector<double> sat;             // (m0+1) x (m1+1) [x (m2+1)]: the particles in [0,i) x [0,j) [x [0,k)]

    void build(std::vector<Particle_data> const &p, User_options const &u)
    {
        periodic = u.periodic;
        // 128 cells across the box per axis (1024 in 2D); without wrap whole cells of the same size continue over the
        // padding, so a box's cells are the same with or without it (a '--region' that is the whole box counts like
        // the periodic box). A padding that would take the grid past maxCells widens every cell instead.
        size_t const perAxis = NO_DIM == 3 ? 128 : 1024;
        double const maxCells = NO_DIM == 3 ? 4.2e6 : 2.1e6;
        double boxLo[NO_DIM], boxLen[NO_DIM], padLo[NO_DIM], padHi[NO_DIM], scale = 1.;
        for (int pass=0; pass<2; ++pass)
        {
            double total = 1.;
            for (int d=0; d<NO_DIM; ++d)
            {
                boxLo[d]  = double(u.boxCoordinates.coords[2*d]);
                boxLen[d] = double(u.boxCoordinates.coords[2*d+1]) - boxLo[d];
                if ( not ( boxLen[d] > 0. ) ) return;
                padLo[d] = periodic ? 0. : std::max( 0., double(u.paddingLength.coords[2*d]) );
                padHi[d] = periodic ? 0. : std::max( 0., double(u.paddingLength.coords[2*d+1]) );
                cell[d] = scale * boxLen[d] / double(perAxis);
                size_t const below = size_t( std::ceil( padLo[d] / cell[d] ) ), above = size_t( std::ceil( padHi[d] / cell[d] ) );
                m[d] = size_t( std::ceil( boxLen[d] / cell[d] - 1.e-9 ) ) + below + above;
                lo[d] = boxLo[d] - double(below) * cell[d];
                total *= double(m[d]);
            }
            if ( total <= maxCells ) break;
            scale *= std::pow( total / maxCells, 1. / double(NO_DIM) ) * 1.01;
        }
        size_t cells = 1, satCells = 1;
        for (int d=0; d<NO_DIM; ++d)
        {
            cells *= m[d];
            satCells *= m[d] + 1;
        }
        std::vector<uint32_t> count( cells, 0u );
        long const n = long( p.size() );
#ifdef OPEN_MP
        #pragma omp parallel for schedule(static)
#endif
        for (long i=0; i<n; ++i)
        {
            size_t k = 0;
            for (int d=0; d<NO_DIM; ++d)
            {
                double t = ( double(p[size_t(i)].pos[d]) - lo[d] ) / cell[d];
                if ( periodic ) t -= double(m[d]) * std::floor( t / double(m[d]) );
                long c = std::isfinite( t ) ? long( std::floor( t ) ) : 0L;
                c = std::max( 0L, std::min( long(m[d]) - 1, c ) );
                k = k * m[d] + size_t(c);
            }
#ifdef OPEN_MP
            #pragma omp atomic
#endif
            ++count[k];
        }
        // the summed-area table: the counts placed at index + 1, then prefix sums along every axis
        sat.assign( satCells, 0. );
        for (size_t k=0; k<cells; ++k)
        {
            size_t r = k, s = 0, stride = 1;
            for (int d=NO_DIM-1; d>=0; --d)
            {
                s += ( r % m[d] + 1 ) * stride;
                r /= m[d];
                stride *= m[d] + 1;
            }
            sat[s] = double( count[k] );
        }
        size_t stride = 1;
        for (int d=NO_DIM-1; d>=0; --d)
        {
            size_t const len = m[d] + 1;
            for (size_t k=0; k<satCells; ++k)
                if ( ( k / stride ) % len != 0 )
                    sat[k] += sat[k - stride];
            stride *= len;
        }
        ok = true;
    }

    // the summed-area table at a fractional cell position (each coordinate in [0, m])
    double table(double const t[NO_DIM]) const
    {
        size_t i0[NO_DIM];
        double f[NO_DIM];
        for (int d=0; d<NO_DIM; ++d)
        {
            double const c = std::max( 0., std::min( double(m[d]), t[d] ) );
            i0[d] = std::min( m[d] - 1, size_t( c ) );
            f[d] = c - double(i0[d]);
        }
        double v = 0.;
        for (int corner=0; corner<(1<<NO_DIM); ++corner)
        {
            double w = 1.;
            size_t s = 0;
            for (int d=0; d<NO_DIM; ++d)
            {
                bool const up = ( corner >> d ) & 1;
                w *= up ? f[d] : 1. - f[d];
                s = s * ( m[d] + 1 ) + i0[d] + ( up ? 1 : 0 );
            }
            if ( w != 0. ) v += w * sat[s];
        }
        return v;
    }

    // the particles with every coordinate below x (periodic: counting the images below the box, negatively, as well)
    double cumulative(double const x[NO_DIM]) const
    {
        double whole[NO_DIM], part[NO_DIM];   // per axis: periods below x, and the position within the period
        for (int d=0; d<NO_DIM; ++d)
        {
            double const t = ( x[d] - lo[d] ) / cell[d];
            whole[d] = periodic ? std::floor( t / double(m[d]) ) : 0.;
            part[d]  = t - whole[d] * double(m[d]);
        }
        double v = 0.;
        for (int combo=0; combo<(1<<NO_DIM); ++combo)
        {
            double w = 1., t[NO_DIM];
            for (int d=0; d<NO_DIM and w != 0.; ++d)
            {
                bool const full = ( combo >> d ) & 1;   // the whole periods below, or the part of the last one
                w *= full ? whole[d] : 1.;
                t[d] = full ? double(m[d]) : part[d];
            }
            if ( w != 0. ) v += w * table( t );
        }
        return v;
    }

    // the particles in the box [a, b)
    double count(double const a[NO_DIM], double const b[NO_DIM]) const
    {
        double v = 0.;
        for (int corner=0; corner<(1<<NO_DIM); ++corner)
        {
            double x[NO_DIM];
            int low = 0;
            for (int d=0; d<NO_DIM; ++d)
            {
                bool const up = ( corner >> d ) & 1;
                x[d] = up ? b[d] : a[d];
                low += up ? 0 : 1;
            }
            v += ( low % 2 ? -1. : 1. ) * cumulative( x );
        }
        return std::max( 0., v );
    }
};

// optimalPartitionSplit's split of a box of the run's grid -- x first, then y within each x slab, then z within each
// column, every cut where optimalSplit puts it -- replayed on the count grid. Ranges are in grid cells relative to
// 'origin' (the box's lower corner) with cell size dx; counts are per bin of up to 128 bins per axis (the coarse grid
// resolves no finer). An axis with fewer cells than pieces (the run refuses that split) is cut into equal pieces.
inline std::vector< std::vector<double> > stdTuneSplit(StdTuneDensity const &dens, double const origin[NO_DIM],
                                                       double const dx[NO_DIM], std::vector<double> const &range,
                                                       size_t const parts[NO_DIM])
{
    std::vector< std::vector<double> > out;
    std::vector<double> cur( range );
    std::function<void(int)> recurse = [&](int d)
    {
        if ( d == NO_DIM ) { out.push_back( cur ); return; }
        double const a = range[2*d], b = range[2*d+1];
        size_t const cells = size_t( std::llround( b - a ) );
        std::vector<double> edges( parts[d] + 1, a );
        if ( parts[d] <= 1 )
            edges[1] = b;
        else if ( cells < parts[d] )
            for (size_t k=0; k<=parts[d]; ++k) edges[k] = a + ( b - a ) * double(k) / double(parts[d]);
        else
        {
            size_t const nb = std::min( cells, size_t(128) );
            std::vector<size_t> binEdge( nb + 1 ), counts( nb ), idx( parts[d] + 1, 0 );
            for (size_t i=0; i<=nb; ++i) binEdge[i] = ( i * cells ) / nb;
            for (size_t i=0; i<nb; ++i)
            {
                double lo[NO_DIM], hi[NO_DIM];
                for (int e=0; e<NO_DIM; ++e)
                {
                    double const ra = ( e == d ) ? a + double(binEdge[i]) : cur[2*e];
                    double const rb = ( e == d ) ? a + double(binEdge[i+1]) : cur[2*e+1];
                    lo[e] = origin[e] + ra * dx[e];
                    hi[e] = origin[e] + rb * dx[e];
                }
                counts[i] = size_t( std::llround( dens.count( lo, hi ) ) );
            }
            optimalSplit( counts, parts[d], idx.data() );
            for (size_t k=0; k<=parts[d]; ++k) edges[k] = a + double( binEdge[ std::min( idx[k], nb ) ] );
            edges[0] = a; edges[parts[d]] = b;
        }
        for (size_t k=0; k<parts[d]; ++k)
        {
            cur[2*d] = edges[k];
            cur[2*d+1] = edges[k+1];
            recurse( d + 1 );
        }
        cur[2*d] = a; cur[2*d+1] = b;
    };
    recurse( 0 );
    return out;
}
#endif

inline AutoTuneReport autoTunePartitioning(User_options &u,
                                 size_t const nParticles,
                                 bool const userSampling,
                                 bool const metalActive,
                                 std::vector<Particle_data> const *particles = nullptr)
{
    if ( u.psServe )
    {
        autoTuneServe( u, nParticles );   // the grid/partition model below does not apply to a server
        return AutoTuneReport();
    }
    // partitionGiven, not only partitionOn: '--partition 1 1 1' asks for one triangulation, which
    // leaves partitionOn false -- reading that as "not given" used to split the data anyway
    bool const autoPart = not (u.partitionOn or u.partitionGiven);
    bool const autoConc = not u.maxConcurrentOn;
    AutoTuneReport report;
    if ( (not autoPart and not autoConc and not u.autoTuneReport) or u.partNo>=0 or u.redshiftConeOn
         or userSampling or u.psServe )
        return report;  // nothing left to decide, or a mode where partitioning is unavailable
                        // (--serve keeps its single triangulation alive for every request)

    double minN = 2.e6;
    if ( const char *env = std::getenv("DTFE_AUTO_MINN") )
    {
        double v = std::atof(env);
        if ( v > 0. ) minN = v;
    }

    double const N = double(nParticles);
    double const ramBytes = autoTunePhysicalRAM();
    std::string budgetWhy;
    double const budget = autoTuneBudget( ramBytes, budgetWhy );   // see the header comment
    int const cores = autoTuneCores();

    // the cells actually allocated and written: with --ps-window its cells, not the (virtual) full grid
    size_t outGrid[NO_DIM];
    u.outputGridSize( outGrid );
    double gridTotal = 1.;
    for (int d=0; d<NO_DIM; ++d)
        gridTotal *= double(outGrid[d]);

    AutoTunePS ps;
#ifdef PHASE_SPACE
    ps.phaseSpace     = true;
    ps.volumeWeighted = u.psVolumeWeighted;
    ps.exactDeposit   = u.psExactDeposit;
    ps.caustics       = u.psCaustics;
    ps.haloRelease    = u.psHaloRelease > Real(0.);
#endif
    // uField and aField each allocate their OWN full grids and coexist (DTFE.cpp:497-508), so
    // summing both is right. The deposit and post phases are disjoint peaks -> max, not sum.
    double const bCellDeposit = autoTuneBytesPerCell(u.uField, ps, 0) + autoTuneBytesPerCell(u.aField, ps, 0);
    double const bCellPost    = autoTuneBytesPerCell(u.uField, ps, 1) + autoTuneBytesPerCell(u.aField, ps, 1);
    double const bCell        = std::max( bCellDeposit, bCellPost );

    // padding fraction of the box per axis (paddingLength was resolved to Mpc by updateEntries)
    double padFrac[NO_DIM];
    for (int d=0; d<NO_DIM; ++d)
    {
        double const boxLen = double(u.boxCoordinates[2*d+1] - u.boxCoordinates[2*d]);
        padFrac[d] = boxLen>0. ? double(u.paddingLength[2*d] + u.paddingLength[2*d+1]) / (2.*boxLen) : 0.;
    }

    MESSAGE::Message message( u.verboseLevel );
    char buf[512];

#ifdef PHASE_SPACE
    // ---- PS-DTFE: Lagrangian partitions run in PARALLEL; concurrency is the throughput knob
    // and each concurrent partition holds a full padded triangulation (the dominant cost). ----
    double const kDT = 666.;                        // bytes/vertex: 112 (incl. the uint64 ParticleID and the double volume accumulator) + 6.77 cells x 72 B + allocator slack
    double const partBytes = double(sizeof(Particle_data));
    // '--parallel-triangulation' (TBB): the sorted point+payload copy and CGAL's own copy of it outlive the particle
    // copy during the insert. MEASURED 2026-10-11 (4.1M particles, 2^3 partitions of 5.83M vertices at mc 3, A B A B):
    // peak footprint 14.88/14.95 -> 18.46/18.48 GB, +203 B per vertex in flight; it was not in this model
    // (autoTuneServe's kBuild 330 holds it there). Counted whenever asked for: with fewer than 2 threads per
    // partition the insert runs sequentially and this errs high.
#if defined(PARALLEL_TRIANGULATION) && NO_DIM==3
    double const kParInsert = u.psParallelTriangulation ? 210. : 0.;
#else
    double const kParInsert = 0.;
#endif
    // --scratch-dir (already armed by DTFE_setup when set): the full-grid accumulators live in
    // mmap'ed files, so they leave the RAM model -- the kernel pages them against the disk and
    // RSS stays bounded. The RAM check then covers only the particles + per-partition terms,
    // and the grid term is checked against the scratch volume's FREE SPACE instead.
    bool const scratchOn = not u.scratchDir.empty();

    // ---- '--sample-points' point evaluation (ps_point_eval.cc) ---------------------------
    // Also IRREDUCIBLE: the query points and their per-point stream records are GLOBAL
    // structures shared by every partition (each partition appends its hits), so --partition
    // does not shrink them. Omitting this was how a points-only run (tiny --grid, so the grid
    // term vanishes) could pick a coarse split against a budget that ignored tens of GB.
    size_t nPts = autoTuneSamplePointCount( u.psSamplePointsFile );
    if ( const char *env = std::getenv("DTFE_AUTO_PTS_N") )      // a plane that does not exist yet
    {
        double const v = std::atof(env);
        if ( v >= 0. ) nPts = size_t(v);
    }
    bool const ptsVelGrad = u.psPtsVelGrad
                            or (std::getenv("DTFE_AUTO_PTS_VELGRAD") and std::string(std::getenv("DTFE_AUTO_PTS_VELGRAD")) == "1");
    // Mean streams (containing tetrahedra) per point. The old fixed 2 (TNG300-3-Dark measured
    // 1.899 at z=0) was far too low for finer mass resolution: TNG100-3-Dark measured 4.25 at
    // z=0 (2.30 at z=1), so a z=0 run with an 8192^2 plane needed 90 GB where the model said ~50.
    // Now ESTIMATED from the particles themselves (psEstimateMeanStreams: Eulerian over Lagrangian
    // volume of the tessellation, on random Lagrangian patches) and budgeted at TWICE the
    // multi-stream excess, 1 + 2 (est - 1): the estimate is the box mean, while one plane varies
    // around it (the z=0 plane's strips range 1.45 .. 22.8). DTFE_AUTO_PTS_STREAMS overrides.
    double ptsMeanStreams = 2.0;
    char ptsStreamsWhy[200] = "2 streams/point assumed";
    if ( const char *env = std::getenv("DTFE_AUTO_PTS_STREAMS") )
    {
        double const v = std::atof(env);
        if ( v >= 1. )
        {
            ptsMeanStreams = v;
            snprintf( ptsStreamsWhy, sizeof(ptsStreamsWhy), "%.3g streams/point from DTFE_AUTO_PTS_STREAMS", v );
        }
    }
    else if ( nPts > 0 and particles != nullptr )
    {
        double boxLo[3], boxLen[3];
        for (int d=0; d<3; ++d)
        {
            boxLo[d]  = double( u.boxCoordinates[2*d] );
            boxLen[d] = double( u.boxCoordinates[2*d+1] ) - boxLo[d];
        }
        int patches = 0;
        // 2026-10-11: the estimate is made on the points' own slab -- their bounding box, widened by one mean
        // particle spacing per side -- not the whole box. The box mean (17.98 on TNG50-3-Dark z=0) budgeted
        // 19.3 GB for a z2048 plane whose written '.pts_streams' average 5.38: the heavy tail sits in a few
        // haloes the plane misses, and the over-reserve pushed the tuner to 5^3 at mc 1. A points set that
        // fills the box gets the box mean as before. DTFE_AUTO_PTS_BOXMEAN=1 restores the box-wide estimate.
        double rLo[3], rHi[3];
        bool haveRegion = false;
        char const *boxMean = std::getenv( "DTFE_AUTO_PTS_BOXMEAN" );
        if ( not ( boxMean and std::string(boxMean) == "1" ) and NO_DIM == 3 )
        {
            double pLo[NO_DIM], pHi[NO_DIM], bLo[NO_DIM], bLen[NO_DIM];
            for (int d=0; d<NO_DIM; ++d) { bLo[d] = boxLo[d]; bLen[d] = boxLen[d]; }
            if ( autoTuneSamplePointBounds( u.psSamplePointsFile, bLo, bLen, u.periodic, pLo, pHi ) )
            {
                double const spacing = std::cbrt( boxLen[0]*boxLen[1]*boxLen[2] / std::max( 1., N ) );
                double vol = 1.;
                for (int d=0; d<NO_DIM; ++d)
                {
                    rLo[d] = pLo[d] - spacing;
                    rHi[d] = pHi[d] + spacing;
                    if ( u.periodic and rHi[d] - rLo[d] >= boxLen[d] ) { rLo[d] = boxLo[d]; rHi[d] = boxLo[d] + boxLen[d]; }
                    vol *= (rHi[d] - rLo[d]) / boxLen[d];
                }
                haveRegion = vol < 0.999;   // a points set that fills the box: the box mean is its slab's
            }
        }
        double const est = psEstimateMeanStreams( *particles, boxLo, boxLen, u.periodic, &patches,
                                                  haveRegion ? rLo : nullptr, haveRegion ? rHi : nullptr );
        if ( est > 0. )
        {
            ptsMeanStreams = std::max( 1., 1. + 2.*(est - 1.) );
            report.streamsEstimate = est;
            snprintf( ptsStreamsWhy, sizeof(ptsStreamsWhy),
                      "%.2f streams/point estimated from %d Lagrangian patches%s, budgeted as %.2f", est, patches,
                      haveRegion ? " on the points' slab" : "", ptsMeanStreams );
        }
    }
    // The STORED per-stream record is packed (ps_point_eval.cc RecLayout): denGeo + denDtfe +
    // vel[3] doubles + the 2-B caustic mask = 42 B, plus grad[3] / vgrad[9] doubles / ids[4]
    // uint64 only with --pts-den-grad / --pts-vel-grad / --per-stream-ids. KEEP IN SYNC.
    double bStreamRec = 5. * double(sizeof(double)) + 2.;
    if ( u.psPtsDenGrad )   bStreamRec += 3. * double(sizeof(double));
    if ( ptsVelGrad )       bStreamRec += 9. * double(sizeof(double));
    if ( u.psPerStreamIds ) bStreamRec += 4. * 8.;
    if ( u.psPtsScalar )    bStreamRec += double(noScalarComp) * double(sizeof(double));
    if ( u.psPtsLagrangian ) bStreamRec += 3. * double(sizeof(double));
    // ... stored as uint32 point index + the packed record in blocks, and grouped at finalize by
    // one uint64 locator per record (ps_point_eval.cc HitSink / psPointEvalFinalize)
    bStreamRec += 4. + 8.;
    // per-point: pos (3 doubles) + bucketPts (uint32) + finalize's offsets and cursor (2 uint64)
    double const bPerPoint = 3. * double(sizeof(double)) + 4. + 2. * 8.;
    // finalized outputs, allocated while the records are still being drained
    double bPtsOut = double(sizeof(double))            // outDen
                   + 3. * double(sizeof(double))       // outVel
                   + 6. * double(sizeof(double))       // outDisp
                   + 4.;                               // outStreams (int32)
    if ( u.psPtsDenGrad ) bPtsOut += 3. * double(sizeof(double));
    if ( ptsVelGrad ) bPtsOut += 9. * double(sizeof(double));
    if ( u.psPerStream )  bPtsOut += ptsMeanStreams * 4. * double(sizeof(double));   // ragged records
    if ( u.psPtsScalar )  bPtsOut += double(noScalarComp) * double(sizeof(double))
                                     * (1. + (u.psPerStream ? ptsMeanStreams : 0.));   // point + per-stream scalars
    if ( u.psPtsLagrangian ) bPtsOut += ptsMeanStreams * 3. * double(sizeof(double));  // per-stream q(x)
    double const ptsBytes = double(nPts) * (bPerPoint + bPtsOut + ptsMeanStreams * bStreamRec);

    // IRREDUCIBLE in RAM unless scratch-backed: originals + the full-grid accumulators
    // + the point-evaluation structures.
    // --partition does NOT reduce this -- partitions sum into the SHARED full grid.
    // a --ps-window run that started without its particles (ps_globals_record.h) holds none: only
    // cached partitions are loaded, so their memory goes to more partitions at once. (A late read, when
    // a needed partition turns out uncached, adds them back: at most the particles' bytes over budget.)
    double const particleBytes = u.psParticlesDeferred ? 0. : partBytes*N;
    double const fixed = particleBytes + (scratchOn ? 0. : gridTotal*bCell) + ptsBytes;
    report.fixed = fixed;
    report.budget = budget;
    report.pts = ptsBytes;
    report.streamsBudgeted = nPts > 0 ? ptsMeanStreams : 0.;
    // NOTE: the over-budget check below must run BEFORE the small-N early return -- the grid
    // term does not depend on N, so a small snapshot on a huge --grid is exactly the case
    // that must not slip through to a silent 60+ GB allocation.

    if ( scratchOn )
    {
        struct statvfs vfs;
        if ( statvfs(u.scratchDir.c_str(), &vfs) == 0 )
        {
            double const freeBytes = double(vfs.f_bavail) * double(vfs.f_frsize);
            double const need = gridTotal*bCell;
            if ( need * 1.05 + 2.e9 > freeBytes )
            {
                MESSAGE::Warning warning( u.verboseLevel );
                warning << "AUTO-TUNE: the scratch volume cannot hold this run's grids: '"
                        << u.scratchDir << "' has " << freeBytes/1.e9 << " GB free but the "
                        << "full-resolution accumulators need ~" << need/1.e9 << " GB ("
                        << MESSAGE::printElements( u.gridSize, "x" ) << " cells x " << bCell
                        << " B/cell). The allocator will fall back to the heap when the disk "
                        << "runs out -- expect swapping. Free up space, use a bigger volume, or "
                        << "reduce --grid / --field.\n" << MESSAGE::EndWarning;
            }
        }
    }

    // ---- Is the IRREDUCIBLE term alone over budget? --------------------------------------
    // If so, no --partition value can help: partitions are summed into the shared full grid.
    // The old code fell through to the search, where (budget-fixed) went negative, cMem was
    // negative for every n, bestN stayed at the loop's first candidate (2 -- the COARSEST split,
    // i.e. the LARGEST per-partition triangulation, exactly backwards) and it warned "even a
    // single 2^3-split partition is predicted to exceed the memory budget ... consider a smaller
    // grid or fewer fields" -- blaming the split and prescribing a remedy that cannot work.
    // which per-tet/per-cell GPU-deposit pieces this field selection actually needs
    // (unselected moment grids are not allocated, and the tet velocity array is only
    // extracted when a velocity-derived grid is requested). The velocity derivatives
    // (divergence etc.) are folded into the gradient AFTER auto-tune runs, so count
    // them here as gradient requests.
    // dispersion implies the velocity/mom grids only when NOT volume-weighted (5d gating)
    bool const psVel  = u.uField.velocity or u.aField.velocity
                        or ((u.uField.velocity_dispersion or u.aField.velocity_dispersion)
                            and not ps.volumeWeighted);
    bool const psDisp = u.uField.velocity_dispersion or u.aField.velocity_dispersion;
    bool const psGrad = u.uField.velocity_gradient or u.uField.selectedVelocityDerivatives()
                        or u.aField.velocity_gradient or u.aField.selectedVelocityDerivatives();
    // flags gating the remaining GPU buffers (ps_metal_host.cc)
    bool const psVolW  = ps.volumeWeighted;
    bool const psCaust = ps.caustics;
    bool const psExact = ps.exactDeposit;

    // Every buffer in PSGpuGrids, per cell and copy (the GPU deposit's host and device grids). mass 4 + streams 4
    // = the 8; the fVolW/fCaustic/fExact terms were missing and fVolW is the PRODUCTION default -- 20 B/cell/copy =
    // 40 after the x2 in partitionBytes(), i.e. 6.7 GB per partition at 1024^3/n=2.
    // The moment grids are 64-bit fixed-point sums since 2026-10-14 (ps_metal_host.cc, fxUnits): 8 B per component
    // on the device and in the host's copy where the float took 4 (mB = 2), and dispw is not deposited (the mass
    // grid stands for it); PS_METAL_FLOAT_MOMENTS=1 keeps the floats (mB = 1).
    bool const momFx = autoTunePSMomentsFixed();
    double const mB = momFx ? 2. : 1.;
    double const gridB  = 8. + mB * ( (psVel ? 12. : 0.) + (psDisp ? 24. : 0.) + (psGrad ? 36. : 0.)
                                    + (psVolW ? 4. : 0.)                  // momw
                                    + (psExact ? 4. : 0.)                 // streamvol
                                    + ((psVolW and psDisp) ? 12. : 0.) )  // dispvel
                        + (psCaust ? 4. : 0.)                             // caustic
                        + ((psVolW and psDisp and not momFx) ? 4. : 0.);  // dispw

    // The single build's fixed-point MASS grid (2026-10-12, ps_interpolation.cc 'massFx'): a pass whose density is
    // selected accumulates its mass as 8 B words per sub-grid cell in place of the 4 B float density, shared by the
    // deposit threads (atomics; no private density copies), and converts them to the float density once after the
    // loop -- when the 8 B words and the 4 B grid coexist for a moment. The double build keeps its double sums.
    // fxCell = the extra bytes per sub-grid cell while the words live (the passes run one after the other: one grid).
    bool const psDensity = u.uField.density or u.uField.velocity_tweb or u.aField.density or u.aField.velocity_tweb;
    double const fxCell  = ( ps.phaseSpace and sizeof(Real) == 4 and psDensity ) ? 4. : 0.;
    // ... and the moment grids the same way (2026-10-14): fxMom = the extra bytes per sub-grid cell while one pass's
    // words live (as many again as its Real moment grids: the larger pass), which also leave the private copies
    double const fxMom   = ( ps.phaseSpace and sizeof(Real) == 4 and momFx )
                         ? std::max( autoTuneFxMomentBytes(u.uField, ps), autoTuneFxMomentBytes(u.aField, ps) ) : 0.;

    // per-partition bytes for an n^3 Lagrangian split
    auto partitionBytes = [&](int n) -> double
    {
        double const nOwn = N / double(n)/double(n)/double(n);
        double fPad = 1.;
        for (int d=0; d<NO_DIM; ++d)
            fPad *= 1. + 2.*autoTunePadWidths( u, padFrac[d], n, N, 0.3 );   // k spacings, or the Eulerian rule's 0.3-cell floor (DTFE.cpp)
        double const nPad = nOwn * fPad;
        double m = nPad * (partBytes + kDT + kParInsert);   // particle copy coexists with the growing triangulation
        // '--sample-points': each worker fills a LOCAL hits buffer (pair<uint32,StreamRec>,
        // padded to 176 B) and merges it under the mutex at the end, so this part DOES scale
        // down with a finer split -- roughly this partition's share of the total records.
        if ( nPts > 0 )
            m += double(std::thread::hardware_concurrency()) * double(16 << 20);   // one HitSink block per thread
        // Eulerian sub-grid this partition allocates (psUseSubgrid). A Lagrangian sub-box maps to
        // a DISPLACED, distorted Eulerian region, so the crop is the touched span + slop -- valid
        // on both periodic and non-periodic runs since ps_interpolation.cc now crops a periodic
        // axis to its WRAPPED span instead of keeping the whole axis. (Before that fix every
        // boundary-touching slab kept the full axis, making this term ~n^3 optimistic.)
        double subCells = 1.;
        for (int d=0; d<NO_DIM; ++d)
            subCells *= std::min( double(outGrid[d]),
                                  double(u.gridSize[d])/double(n) + 40. );   // crop + slop, never above the axis (nor the window)
        if ( metalActive )
        {
            // flat tet arrays (the 16-float record 64 + masses 4 + the finite-cell ordinal 4 B/tet,
            // + vels 48 when velocity-derived grids are requested; the cost-sort permutes IN PLACE,
            // so no gather copy) + the PSGpuGrids host and device copies of the selected grids
            double const perTet = (72. + (psVel or psDisp or psGrad ? 48. : 0.)) * 1.3;   // + allocator slack
            // gridB (above): every buffer in PSGpuGrids, mirrored host+device (the x2 below)
            // + 4 B/cell: the device's mass grid is a 64-bit fixed-point sum (ps_metal_host.cc, 2026-10-08)
            // + 4 B/cell more on the host since 2026-10-12, which receives those 8 B words instead of a 4 B float
            //   grid (PSGpuGrids::massFx) -- unless a density is selected: the caller then allocates its density
            //   sub-grid (4 B; 8 in the double build) only after the call, from the words, so the peak (the
            //   copy-back) is the same as before or lower
            double const hostFx = ( ps.phaseSpace and not psDensity ) ? 4. : 0.;
            m += perTet*6.77*nOwn + 2.*gridB*subCells + (4. + hostFx)*subCells;
            // --ps-exact-deposit runs depositExactItems (ps_metal_host.cc): a per-tet weight total
            // (4 B) + the item list, 8 B per 16 cells of a tet's bbox window -- ~20x the tet's
            // volume in cells on the crossed-waves benchmark, at least one item -- capped at 32M items
            if ( psExact )
            {
                double const nTet = 6.77*nOwn;
                // --ps-window: items cover only the window's cells (the hull of each tet's bbox with
                // the window), so the window's cell count sets the items, not the virtual full grid
                double gridCells = 1.;
                for (int d=0; d<NO_DIM; ++d) gridCells *= double(outGrid[d]);
                double const cellsPerTet = gridCells / (6.*double(N));
                // (the weight totals are two 4 B words per tet since 2026-10-14: an exact per-tet sum)
                m += (momFx ? 8. : 4.)*nTet + 8.*std::min( 32.e6, nTet*(1. + 20.*cellsPerTet/16.) );
            }
        }
        else
        {
            // CPU deposit sub-grids (was a hardcoded 52) + the deposit threads' private copies, as many
            // as the copy budget allows -- unless that leaves fewer than half the threads: the deposit
            // then runs in SLAB mode (ps_interpolation.cc, the same rule) with no copies, holding a cell
            // handle, a first-slab byte and ~2 slab-list entries per tetrahedron (+ the binning arrays).
            // With the fixed-point mass grid (fxCell, 2026-10-12): +4 B/cell for the words during the deposit, a
            // private copy without the 4 B density, and +4 B/cell more at the conversion after the loop -- the
            // copies and slab lists are freed by then, so that moment only counts when it exceeds them.
            // The moment grids' words (fxMom, 2026-10-14) likewise: +fxMom during the deposit, no private copy of
            // them, and their conversion after the loop takes them grid by grid (under the same max).
            double const copyOne = bCellDeposit*subCells;
            double const copyPriv = std::max( 0., bCellDeposit - fxCell - fxMom )*subCells;
            int const copyThreads = std::min( cores, 1 + int( psDepositThreadBytes() / std::max( 1., copyPriv ) ) );
            double const slabBytes = 6.77 * nPad * ( 8. + 1. + 2.*4. + 20. );
            double const helpers = 2 * copyThreads < cores ? slabBytes : double(copyThreads - 1) * copyPriv;
            m += copyOne + (fxCell + fxMom)*subCells + std::max( helpers, (fxCell + fxMom)*subCells );
        }
        return m;
    };

    // The GPU deposit's host snapshot (ps_metal_host.cc GpuSnapshot, 2026-10-11): to resume a failed command buffer
    // the deposit keeps a host copy of every buffer its kernel writes -- the device grids (gridB, with the 8 B
    // fixed-point mass for the 4 B float one: +4 B/cell), the per-tet masses (4 B/tet) and the exact items' weight
    // totals (4 B/tet) -- taken at each pass's start and every ~10 s of GPU work. Left out it was ~4.7 GB per
    // partition at 1024^3 split 2^3 (volume-weighted velocity), past the gate's 4 GB headroom. Not in
    // partitionBytes(): the deposits run one at a time (ctxMutex), so whatever the concurrency there is ONE such
    // copy, added once to every plan below. (The CUDA port keeps none yet: there it is slack; the model errs high.)
    auto snapshotBytes = [&](int n) -> double
    {
        if ( not metalActive ) return 0.;
        double subCells = 1.;   // the same crop + slop as partitionBytes()
        for (int d=0; d<NO_DIM; ++d)
            subCells *= std::min( double(outGrid[d]), double(u.gridSize[d])/double(n) + 40. );
        double const nTet = 6.77 * N / double(n)/double(n)/double(n);
        return (gridB + 4.)*subCells + (psExact ? (momFx ? 12. : 8.) : 4.)*nTet;
    };

    // The partition look-ahead (DTFE.cpp, 2026-10-11): at one partition in flight with the GPU deposit, the NEXT
    // partition's particle copy + tessellation are built during the deposit -- one more partition's
    // triangulation stage on top of the peak (the run skips a look-ahead the free memory cannot hold)
    auto lookAheadBytes = [&](int n, int c) -> double
    {
        if ( not metalActive or u.psNoOverlap or u.psLookAheadOffBudget or c != 1 or n <= 1 ) return 0.;
        double fPad = 1.;
        for (int d=0; d<NO_DIM; ++d)
            fPad *= 1. + 2.*autoTunePadWidths( u, padFrac[d], n, N, 0.3 );
        return N / double(n)/double(n)/double(n) * fPad * (partBytes + kDT + kParInsert);
    };

    // what a given split and concurrency cost; the concurrency the run WILL use (0 = all cores)
    auto finishReport = [&](int n, int c) -> AutoTuneReport
    {
        report.valid = true;
        report.partition = std::max( n, 1 );
        report.maxConcurrent = std::max( c, 1 );
        // partitions in flight: never more than there are (one domain x 10 threads is ONE domain --
        // the old product put a 0.26M-particle run at 52 GB and the GUI's memory check believed it)
        int const inFlight = std::min( report.maxConcurrent, report.partition*report.partition*report.partition );
        report.perPartition = partitionBytes( report.partition );
        report.gpuSnapshot = snapshotBytes( report.partition );   // once: the GPU deposits run one at a time
        double const base = fixed + report.gpuSnapshot + double(inFlight) * report.perPartition;
        report.lookAhead = lookAheadBytes( report.partition, inFlight );
        // 2026-10-11: the split above is chosen without the look-ahead, which used to be added here anyway --
        // a plan that fitted then reported over_budget=1 (the launcher's memory check halved a plane that
        // fits) and the run went past a budget set on purpose. Every return passes here, so this is where the
        // plan, its prediction and the run are made to agree: a look-ahead that does not fit beside one
        // partition is switched off for this run (DTFE.cpp reads psLookAheadOffBudget), and the prediction
        // is then the grids + particles + one partition. Over budget anyway, the look-ahead only adds to it.
        if ( report.lookAhead > 0. and base + report.lookAhead > budget )
        {
            message << MESSAGE::cBold() << "AUTO-TUNE:" << MESSAGE::cReset() << " look-ahead off for this run: one "
                    << "partition (~" << report.perPartition/1.e9 << " GB) plus the next one's tessellation (~"
                    << report.lookAhead/1.e9 << " GB) does not fit the " << (budget - fixed)/1.e9 << " GB the "
                    << "memory budget leaves beside the grids and particles; partitions are prepared in turn.\n"
                    << MESSAGE::Flush;
            u.psLookAheadOffBudget = true;
            report.lookAhead = 0.;
        }
        report.predicted = base + report.lookAhead;
        return report;
    };
    int const userC = u.maxConcurrent > 0 ? u.maxConcurrent : cores;
    int const userN = u.partition.empty() ? 1 : int(u.partition[0]);

    // The GPU kernels number a partition's sub-grid cells with 32 bits: a sub-grid above 2^32 cells
    // (1626^3 unsplit) falls back to the CPU deposit (ps_interpolation.cc, averaged_interpolation_1.cc).
    // With the GPU on, every split chosen below keeps the sub-grid under that cap -- the same crop +
    // slop estimate as partitionBytes() -- so a very fine grid keeps its GPU deposit; a user's own
    // split that does not is told which one would.
    auto gpuSubCells = [&](int n) -> double
    {
        double c = 1.;
        for (int d=0; d<NO_DIM; ++d)
            c *= std::min( double(outGrid[d]), double(u.gridSize[d])/double(n) + 40. );
        return c;
    };
    double const gpuCellCap = 4294967295.;            // UINT32_MAX
    int nGpuMin = 1;
    if ( metalActive )
        while ( nGpuMin < 64 and gpuSubCells(nGpuMin) > gpuCellCap ) ++nGpuMin;
    if ( metalActive and not autoPart and gpuSubCells(userN) > gpuCellCap )
    {
        MESSAGE::Warning warning( u.verboseLevel );
        warning << "AUTO-TUNE: with --partition " << userN << " each partition's sub-grid holds ~" << gpuSubCells(userN)
                << " cells, above the 2^32 the GPU kernels number: the deposit will run on the CPU. "
                << "--partition " << nGpuMin << " (or finer) keeps it on the GPU.\n" << MESSAGE::EndWarning;
    }

    bool overBudget = false;   // set below; the split is then chosen once partitionBytes() exists
    if ( fixed > budget )
    {
        double const gridBytes = scratchOn ? 0. : gridTotal*bCell;
        // largest cubic grid whose accumulators still leave room for the particles and points
        double const room = budget - particleBytes - ptsBytes;
        long   const maxGrid = room > 0. ? long( std::cbrt( room / bCell ) ) : 0L;
        MESSAGE::Warning warning( u.verboseLevel );
        warning << "AUTO-TUNE: this run does not fit in memory, and --partition cannot fix it.\n  ";
        if ( not scratchOn )
            warning << "The full-resolution output grids need " << gridBytes/1.e9 << " GB ("
                    << MESSAGE::printElements( u.gridSize, "x" ) << " cells x " << bCell << " B/cell for the "
                    << "selected fields), plus ";
        else
            warning << "With the grids on scratch disk, ";
        warning << (partBytes*N)/1.e9 << " GB of particles";
        if ( nPts > 0 )
            warning << " and " << ptsBytes/1.e9 << " GB for the " << double(nPts) << " --sample-points ("
                    << ptsStreamsWhy << "; DTFE_AUTO_PTS_STREAMS overrides)";
        warning << " = " << fixed/1.e9 << " GB, against a memory budget of " << budgetWhy << ".\n"
                << "  --partition splits the TRIANGULATION only: every Lagrangian partition deposits into "
                << "the same shared full-resolution grid and point set, so that part is irreducible.\n"
                << "  Options: ";
        if ( not scratchOn )
            warning << "(a) '--scratch-dir <local dir>' backs the grids with disk instead of RAM "
                    << "(needs ~" << gridBytes/1.e9 << " GB free on a local, non-synced volume, e.g. "
                    << "/private/tmp/dtfe-scratch); ";
        if ( maxGrid > 0 )
            warning << "(b) a smaller --grid -- in RAM, the largest that fits at these fields is about "
                    << maxGrid << "^3; ";
        else
            warning << "(b) a smaller --grid cannot help: the particles" << (nPts > 0 ? " and sample points" : "")
                    << " alone exceed the budget; ";
        warning << "(c) fewer --field entries (the gradient family costs 36 B/cell, "
                << "dispersion 24 B/cell); ";
        if ( nPts > 0 )
            warning << "(d) fewer --sample-points (a smaller image plane) or no --pts-vel-grad; ";
        warning << "(e) more RAM. NOTE: '--partNo'/'--region' are NOT a way out -- they crop particles by "
                << "Eulerian position and are rejected by this binary because the PS tessellation is Lagrangian.\n"
                << "  Running anyway with --max-concurrent 1 and the split with the smallest per-partition "
                << "triangulation; expect swapping.\n" << MESSAGE::EndWarning;
        if ( autoConc ) u.maxConcurrent = 1;
        // A small particle set triangulates cheaply in one piece; otherwise fall through to the
        // split choice below -- the old fallback, one UNPARTITIONED triangulation, put the whole
        // tessellation (~100 GB for 94M particles) on top of an already over-budget run.
        if ( N < minN or not autoPart )
        {
            int const nOne = autoPart ? nGpuMin : userN;      // one domain, or the GPU cap's split
            if ( autoPart ) { u.partition.assign( NO_DIM, size_t(nOne) ); u.partitionOn = ( nOne > 1 ); }
            return finishReport( nOne, autoConc ? 1 : userC );
        }
        overBudget = true;
    }

    // A small particle set triangulates cheaply in one piece -- unless that piece's DEPOSIT does not fit:
    // on a very fine grid (1700^3 from 0.26M particles) one domain's sub-grid accumulators and the
    // deposit threads' private copies are tens of GB, and only a split shrinks them. Such a run takes
    // the memory-driven split below instead (it predicted 83 GB unsplit, over budget, before 2026-10-02).
    bool const smallFits = not autoPart or fixed + snapshotBytes( std::max( 1, nGpuMin ) )
                                           + partitionBytes( std::max( 1, nGpuMin ) ) <= budget;
    if ( N < minN and smallFits )     // small particle set AND the grid fits: one domain, or a split for SPEED
    {
        if ( not autoPart )
        {   // the user's split at the concurrency the report states: left unset, a hand-set '--partition 4 4 4'
            // started all 64 partitions on a 10-core machine while the report said mc=10 (2026-10-08)
            if ( autoConc ) u.maxConcurrent = cores;
            return finishReport( userN, autoConc ? cores : userC );
        }
        // The one stage a split parallelises is the sequential triangulation (+ vertex densities):
        // the deposit and the point evaluation already use every core inside one domain. A
        // triangulation-bound run (a slice of a small box: 1.5 s of 3 s) therefore gains from
        // 2^3 partitions, each triangulating a padded eighth on its own core; a deposit-bound one
        // (a 256^3 grid at 0.26M particles: 1.5 s of 18 s) does not, and loses the padding.
        // Predicted wall time per split, from per-unit costs measured on a 10-core Apple Silicon
        // (single-thread seconds; DTFE_AUTO_T_VERTEX/_SAMPLE/_POINT override):
        double tVertex = 3.6e-6, tSample = 3.5e-7, tPoint = 7.5e-7;
        if ( const char *e = std::getenv("DTFE_AUTO_T_VERTEX") ) { double v = std::atof(e); if ( v > 0. ) tVertex = v; }
        if ( const char *e = std::getenv("DTFE_AUTO_T_SAMPLE") ) { double v = std::atof(e); if ( v > 0. ) tSample = v; }
        if ( const char *e = std::getenv("DTFE_AUTO_T_POINT") )  { double v = std::atof(e); if ( v > 0. ) tPoint = v; }
        double fCopy = 1.;                              // the single domain's periodic copies (none without --periodic)
        if ( u.periodic )
            for (int d=0; d<NO_DIM; ++d) fCopy *= 1. + 2.*padFrac[d];
        double const nSubCube = ( u.aField.selected() or psExact )
                              ? std::pow( double(std::max( 1, psExact ? 3 : int(u.psAvgSubsamples) )), double(NO_DIM) ) : 1.;
        double const depositWork = gridTotal * nSubCube * tSample + double(nPts) * tPoint;
        auto predictedWall = [&](int n, int c) -> double
        {
            double const parts = double(n)*double(n)*double(n);
            double fPad = 1.;
            for (int d=0; d<NO_DIM; ++d) fPad *= 1. + 2.*autoTunePadWidths( u, padFrac[d], n, N, 0.3 );
            double const rounds = std::ceil( parts / double(c) );
            double const triangulate = ( n == 1 ? fCopy * N : fPad * N / parts ) * tVertex;
            double const threads = ( n == 1 ) ? double(cores) : double(std::max( 1, cores / c ));
            // a GPU deposit serialises on the one command queue: the split leaves it unchanged
            // and adds each partition's buffer setup, extraction and leftover passes
            double const deposit = metalActive ? depositWork / double(cores) : depositWork / parts / threads;
            double const overhead = ( n == 1 ) ? 0. : parts * ( N * 5.e-9 + 0.02 + ( metalActive ? 0.1 : 0. ) );
            return rounds * ( triangulate + deposit ) + overhead;
        };
        int bestN = 1, bestC = autoConc ? cores : userC;
        double const single = predictedWall( 1, cores );
        double bestT = single;
        if ( nGpuMin > 1 )                  // the GPU's sub-grid cap rules out one domain
        {
            bestN = nGpuMin;
            bestC = std::max( 1, std::min( { nGpuMin*nGpuMin*nGpuMin, cores,
                                               int( std::floor( (budget - fixed - snapshotBytes(nGpuMin)) / partitionBytes(nGpuMin) ) ) } ) );
            if ( not autoConc ) bestC = std::min( bestC, userC > 0 ? userC : cores );
            bestT = predictedWall( nGpuMin, bestC );
        }
        for (int n = std::max( 2, nGpuMin ); n <= std::max( 3, nGpuMin ); ++n)
        {
            int c = std::min( n*n*n, cores );
            int const cMem = int( std::floor( (budget - fixed - snapshotBytes(n)) / partitionBytes(n) ) );
            if ( cMem < 1 ) continue;
            c = std::min( c, cMem );
            if ( not autoConc ) c = std::min( c, userC > 0 ? userC : cores );
            double const t = predictedWall( n, c );
            if ( t < 0.85 * bestT or ( bestN < nGpuMin ) ) { bestN = n; bestC = c; bestT = t; }   // a clear gain only
        }
        if ( bestN > 1 )
        {
            u.partition.assign( NO_DIM, size_t(bestN) );
            u.partitionOn = true;
            u.psOrderedMerge = true;   // uniform small partitions: merge in order, bit-reproducible at no cost
            u.psSpeedSplit = true;     // ... and no memory gate (DTFE.cpp; 2026-10-14: its own flag, the ordered merge
                                       // being every split's default now)
            if ( autoConc ) u.maxConcurrent = bestC;
            if ( bestN == nGpuMin and nGpuMin > 1 )
                snprintf( buf, sizeof(buf),
                          "%s grid on the GPU: --partition %d %d %d --max-concurrent %d keeps each partition's "
                          "sub-grid under the 2^32 cells the GPU kernels number (~%.3g cells; predicted peak ~%.3g GB)",
                          MESSAGE::printElements( u.gridSize, "x" ).c_str(), bestN, bestN, bestN, u.maxConcurrent,
                          gpuSubCells(bestN), (fixed + snapshotBytes(bestN) + double(bestC)*partitionBytes(bestN))/1.e9 );
            else
                snprintf( buf, sizeof(buf),
                          "%.3g particles: triangulation-bound, so --partition %d %d %d --max-concurrent %d "
                          "(predicted %.2g s against %.2g s for one triangulation; predicted peak ~%.3g GB)",
                          N, bestN, bestN, bestN, u.maxConcurrent, bestT, single,
                          (fixed + snapshotBytes(bestN) + double(bestC)*partitionBytes(bestN))/1.e9 );
            message << "\n" << MESSAGE::cBold() << "AUTO-TUNE:" << MESSAGE::cReset() << " " << buf
                    << ". Pass --partition 1 1 1 for one triangulation.\n" << MESSAGE::Flush;
            return finishReport( bestN, autoConc ? bestC : userC );
        }
        return finishReport( 1, autoConc ? cores : userC );
    }


    if ( overBudget )
    {
        // over budget whatever we do (warned above): one partition at a time, split so that the
        // partition in flight is as small as the padding allows -- partitionBytes() stops falling
        // once the padding shells dominate, so take its minimum over the validated range
        int bestN = std::max( 2, nGpuMin );
        for (int n = bestN + 1; n <= std::max( 6, nGpuMin ); ++n)
            if ( partitionBytes(n) < partitionBytes(bestN) ) bestN = n;
        u.partition.assign( NO_DIM, size_t(bestN) );
        u.partitionOn = true;
        snprintf( buf, sizeof(buf),
                  "%.3g particles, %s grid, %.3g GB RAM (budget %.3g GB) -> OVER BUDGET: --partition %d %d %d "
                  "--max-concurrent %d (smallest partition, ~%.3g GB; predicted peak ~%.3g GB%s)",
                  N, MESSAGE::printElements( u.gridSize, "x" ).c_str(), ramBytes/1.e9, budget/1.e9,
                  bestN, bestN, bestN, u.maxConcurrent, partitionBytes(bestN)/1.e9,
                  (fixed + snapshotBytes(bestN) + partitionBytes(bestN))/1.e9, metalActive ? ", GPU deposit" : "" );
        message << "\n" << MESSAGE::cBold() << "AUTO-TUNE:" << MESSAGE::cReset() << " " << buf
                << ". Pass --partition/--max-concurrent to override.\n           memory budget: "
                << budgetWhy << "\n" << MESSAGE::Flush;
        return finishReport( bestN, autoConc ? 1 : userC );
    }

    // Concurrency target = the core count, GPU or not. The GPU deposits do serialize on one
    // queue (ps_metal_host.cc ctxMutex), but the dominant per-partition cost is the CPU work
    // AROUND them -- Lagrangian triangulation + vertex densities, ~7 s of an ~8 s partition at
    // 2.1M particles / 256^3 -- which runs fully in parallel. A cap of 3 (the old rule) ran 8
    // partitions in three waves: 21 s on the GPU vs 9.5 s on the CPU. Memory still bounds it:
    // partitionBytes() counts the GPU buffers a waiting partition holds, and cMem caps c.
    int const targetC = cores;

    int bestN = -1, bestC = 0;
    int const nLo = autoPart ? std::max( 2, nGpuMin ) : int(u.partition[0]);
    int const nHi = autoPart ? std::max( 6, nGpuMin ) : int(u.partition[0]);   // <=6^3 keeps auto within validated territory (unless the GPU cap needs more)
    for (int n=nLo; n<=nHi; ++n)
    {
        double const m = partitionBytes(n);
        int cMem = int( std::floor( (budget - fixed - snapshotBytes(n)) / m ) );
        int c = std::min( { targetC, n*n*n, cMem } );
        if ( c > bestC or bestN<0 )
        {
            bestN = n;
            bestC = c;
        }
        if ( c >= std::min(targetC, n*n*n) )
            break;      // memory is not the binding constraint; smallest such n wins (fewest seams)
    }
    if ( bestC < 1 )
    {
        // 'fixed' fits (checked above), so this is the triangulation genuinely not fitting in
        // what is left -- here the split IS the right lever, and the finest one is the best try.
        bestC = 1;
        bestN = nHi;
        MESSAGE::Warning warning( u.verboseLevel );
        warning << "AUTO-TUNE: even a single " << bestN << "^3-split partition's triangulation is predicted to "
                << "exceed the " << (budget - fixed)/1.e9 << " GB left after the full-grid accumulators ("
                << (gridTotal*bCell)/1.e9 << " GB) and particles (" << (partBytes*N)/1.e9 << " GB); running with "
                << "--max-concurrent 1 at the finest split. Expect swapping; a finer --partition than "
                << nHi << "^3 may still help.\n" << MESSAGE::EndWarning;
    }

    if ( autoPart )
    {
        u.partition.assign( NO_DIM, size_t(bestN) );
        u.partitionOn = (bestN > 1);
    }
    if ( autoConc )
        u.maxConcurrent = bestC;

    // the concurrency the run WILL use: the tuner's, or the user's own (0 = all cores)
    int const effC = autoConc ? bestC : userC;
    // the report first: it decides whether the look-ahead fits the budget, and the message states its prediction
    AutoTuneReport const decided = finishReport( bestN, effC );
    double const predicted = decided.predicted;
    char scratchNote[256] = "";
    if ( scratchOn )
        snprintf( scratchNote, sizeof(scratchNote), "; grid accumulators (%.3g GB) on scratch disk",
                  gridTotal*bCell/1.e9 );
    char ptsNote[320] = "";
    if ( nPts > 0 )
        snprintf( ptsNote, sizeof(ptsNote), "; %.3g sample points (%.3g GB: %s)",
                  double(nPts), ptsBytes/1.e9, ptsStreamsWhy );
    snprintf( buf, sizeof(buf),
              "%.3g particles, %s grid, %.3g GB RAM (budget %.3g GB) -> --partition %d %d %d --max-concurrent %d "
              "(predicted peak ~%.3g GB%s%s%s)",
              N, MESSAGE::printElements( u.gridSize, "x" ).c_str(), ramBytes/1.e9, budget/1.e9,
              int(u.partition[0]), int(u.partition[1]), int(u.partition[2]), u.maxConcurrent,
              predicted/1.e9, metalActive ? ", GPU deposit" : "", scratchNote, ptsNote );
    message << "\n" << MESSAGE::cBold() << "AUTO-TUNE:" << MESSAGE::cReset() << " " << buf
            << ". Pass --partition/--max-concurrent to override.\n           memory budget: "
            << budgetWhy << "\n" << MESSAGE::Flush;
    return decided;

#else
    // ---- standard DTFE: partitions are processed SERIALLY (memory knob); inside one partition
    // the box is split across up to --max-concurrent threads (throughput knob). ----
    // A '--region'/'--regionMpc' run is planned as what it triangulates (2026-10-13): DTFE_setup crops the region
    // first and passes a copy of the options whose box IS the region, with N = the particles inside it (unpadded),
    // so 'padFrac' above is the padding over the region's length -- the pairing a whole-box run has. (It was the
    // whole snapshot's N and box: a launcher zoom of TNG50-3-Dark, 88k particles in its padded region, came out
    // '--partition 3 3 3, 10 thread(s)' from 1.57e8.) The padded copies the crop holds are left out of 'fixed':
    // 36 B each next to the ~700 B per vertex of the triangulation term, which counts them.
    // bytes/vertex: 96 (incl. the double volume accumulator) + 6.77 cells x 72 B + slack; the double build's vertex data
    // (density, velocity, scalar, ...) in 8 B instead of 4: +50 B/vertex measured (2026-10-13, one triangulation of 0.9M
    // and 1.0M points: 609 vs 659 B per added vertex)
    double const kDT = 648. + 10. * ( double(sizeof(Real)) - 4. );
    double const partBytes = double(sizeof(Particle_data));
    // --scratch-dir: the full-grid accumulators are mmap-backed and leave the RAM model. (The process itself -- code,
    // libraries, the GPU runtime: 66 MB of footprint, 84 MB with --gpu -- is left out, as in the PS model: it matters on
    // sub-GB runs only, where it would make the tests' simulated small machines plan around it.)
    double const fixed = partBytes*N + (u.scratchDir.empty() ? gridTotal*bCell : 0.);
    // '--parallel-triangulation' in a TBB build: the insert's sorted copies (+203 B per vertex, measured on PS above)
#if defined(PARALLEL_TRIANGULATION) && NO_DIM==3
    double const kParInsert = u.psParallelTriangulation ? 210. : 0.;
#else
    double const kParInsert = 0.;
#endif
    // per padded vertex: the thread's particle copy (alive while its triangulation grows) + the triangulation, and 6%
    // for the dummy points the padding test adds on the padded box's faces (3-5% of a sub-box at TNG scale)
    double const perVertex = 1.06 * ( partBytes + kDT + kParInsert );

    /* The GPU deposit of the '_a' pass (averaged_interpolation_1.cc -> dtfe_metal_host.cc / dtfe_gpu_cuda.cu), the
       same for the sampled and the exact average (see StdTuneDensity's comment): per tetrahedron of a thread's own box
       the flat arrays (vertices 48, densities 16, velocities 48 when a velocity-derived '_a' grid is asked, the scalar
       16, volume, sample count, flat index and Morton key 4 each), the Morton order pairs (8), one array's sort copy
       (<= 48) and the work-item table (4) -- the arrays become the uploaded batch one at a time -- and per sub-grid
       cell the per-call deposit's device grid and its host copy (bGpuCell each). The model before 2026-10-13 charged
       6.77 x 120 B per PADDED vertex, i.e. as if every tetrahedron of the padding were extracted: only the ones
       touching the thread's own box are. */
    bool gpuVel = false, gpuScal = false;
    double bGpuCell = 0.;
    if ( metalActive and u.aField.selected() )
    {
        bool const den  = u.aField.density or u.aField.velocity_tweb;
        bool const grad = u.aField.velocity_gradient or u.aField.selectedVelocityDerivatives();
        gpuVel  = u.aField.velocity or grad;
        gpuScal = u.aField.scalar or u.aField.scalar_gradient;
        bGpuCell = ( den ? 4. : 0. ) + ( u.aField.velocity ? 12. : 0. ) + ( grad ? 36. : 0. )
                 + ( u.aField.scalar ? 4. : 0. ) + ( u.aField.scalar_gradient ? 12. : 0. );
    }
    double const perTet = bGpuCell > 0. ? 80. + ( gpuVel ? 48. : 0. ) + ( gpuScal ? 16. : 0. ) + 8. + 48. + 4. : 0.;
    double const tetsPerOwn = 6.77 * 1.3;           // per particle of the thread's own box, + the ones crossing its faces

    // the count grid of the particles (StdTuneDensity: one pass); without particles in memory, the uniform count
    auto const tuneStart = std::chrono::steady_clock::now();
    StdTuneDensity dens;
    if ( particles != nullptr and not particles->empty() )
        dens.build( *particles, u );
    double const countSec = std::chrono::duration<double>( std::chrono::steady_clock::now() - tuneStart ).count();
    // '--sample-points' evaluates ONE tessellation per partition (DTFE_parallel runs no thread split for it)
    bool const oneTessellation = not u.psSamplePointsFile.empty();
    double origin[NO_DIM], dxGrid[NO_DIM];
    for (int d=0; d<NO_DIM; ++d)
    {
        origin[d] = double( u.boxCoordinates[2*d] );
        dxGrid[d] = ( double( u.boxCoordinates[2*d+1] ) - origin[d] ) / double( u.gridSize[d] );
    }
    std::vector<size_t> const userSplit = ( not autoPart and u.partition.size() == size_t(NO_DIM) )
                                        ? u.partition : std::vector<size_t>( NO_DIM, 1 );

    // The busiest partition of a split (P per axis, or the user's own) with up to T threads in each: its bytes on top
    // of 'fixed', the padded vertices its sub-triangulations hold, and those per thread (DTFE_AUTO_TUNE_DEBUG)
    struct StdPlan { double bytes = 0., vertices = 0.; int part = -1; size_t parts = 1; std::vector<double> perThread; };
    std::map< std::pair<int,int>, StdPlan > plans;
    User_options pgOptions = u;                     // parallelGrid reads the partition's own grid from it
    pgOptions.verboseLevel = 0;                     // (and would say so for every plan)
    auto planOf = [&](int P, int T) -> StdPlan const &
    {
        int const threads = ( oneTessellation or T <= 1 ) ? 1 : std::min( T, cores );
        std::pair<int,int> const key( P, threads );
        auto const found = plans.find( key );
        if ( found != plans.end() ) return found->second;
        StdPlan plan;
        size_t partsP[NO_DIM];
        for (int d=0; d<NO_DIM; ++d) partsP[d] = autoPart ? size_t(P) : userSplit[d];
        double nParts = 1.;
        for (int d=0; d<NO_DIM; ++d) nParts *= double( partsP[d] );
        plan.parts = size_t( nParts );
        // a partition's grids: its own (DTFE_parallel's, when split), the threads' sub-grid copies, the GPU's two
        auto gridBytes = [&](double cells, size_t nThreads) {
            return ( nParts > 1. ? cells * bCellDeposit : 0. ) + ( nThreads > 1 ? cells * bCellDeposit : 0. )
                   + 2. * cells * bGpuCell; };
        if ( not dens.ok )
        {   // every sub-box holds the box average (the model before 2026-10-13)
            double const tSplit = std::pow( double(threads), 1. / double(NO_DIM) );
            double f = 1.;
            for (int d=0; d<NO_DIM; ++d) f *= 1. + 2.*padFrac[d]*double(partsP[d])*tSplit;
            double const own = N / nParts;
            plan.vertices = own * f;
            plan.bytes = plan.vertices * perVertex + own * tetsPerOwn * perTet
                       + gridBytes( gridTotal / nParts, size_t(threads) );
            plan.part = 0;
        }
        else
        {
            std::vector<double> whole( 2*NO_DIM, 0. );
            for (int d=0; d<NO_DIM; ++d) whole[2*d+1] = double( u.gridSize[d] );
            auto const partitions = stdTuneSplit( dens, origin, dxGrid, whole, partsP );
            for (size_t ip=0; ip<partitions.size(); ++ip)
            {
                std::vector<double> const &pr = partitions[ip];
                size_t pGrid[NO_DIM];
                double cells = 1.;
                for (int d=0; d<NO_DIM; ++d)
                {
                    pGrid[d] = 1;
                    cells *= std::max( 1., pr[2*d+1] - pr[2*d] );
                    pgOptions.gridSize[d] = size_t( std::max( 1LL, (long long)std::llround( pr[2*d+1] - pr[2*d] ) ) );
                }
                if ( threads > 1 )
                    parallelGrid( size_t(threads), pgOptions, pGrid );   // DTFE_parallel's thread grid (2^a 3^b threads)
                size_t nThreads = 1;
                for (int d=0; d<NO_DIM; ++d) nThreads *= pGrid[d];
                std::vector< std::vector<double> > const boxes = nThreads > 1
                    ? stdTuneSplit( dens, origin, dxGrid, pr, pGrid ) : std::vector< std::vector<double> >( 1, pr );
                double bytes = 0., vertices = 0.;
                std::vector<double> perThread;
                auto boxCount = [&](std::vector<double> const &b, bool const padded) {
                    double lo[NO_DIM], hi[NO_DIM];
                    for (int d=0; d<NO_DIM; ++d)
                    {
                        lo[d] = origin[d] + b[2*d]*dxGrid[d]   - ( padded ? double( u.paddingLength[2*d] ) : 0. );
                        hi[d] = origin[d] + b[2*d+1]*dxGrid[d] + ( padded ? double( u.paddingLength[2*d+1] ) : 0. );
                    }
                    return dens.count( lo, hi ); };
                for (std::vector<double> const &b : boxes)
                {
                    double const padded = boxCount( b, true );
                    perThread.push_back( padded );
                    vertices += padded;
                    bytes += padded * perVertex + boxCount( b, false ) * tetsPerOwn * perTet;
                }
                // a partition's padded particle copy (the partition loop's), or one domain's periodic copies
                // (DTFE_preparePadding) / a '--region' crop's padding beyond the N counted in 'fixed'
                double const partPadded = boxCount( pr, true );
                bytes += ( nParts > 1. ? partPadded : std::max( 0., partPadded - N ) ) * partBytes;
                bytes += gridBytes( cells, nThreads );
                if ( bytes > plan.bytes )
                {
                    plan.bytes = bytes;
                    plan.vertices = vertices;
                    plan.part = int(ip);
                    plan.perThread = perThread;
                }
            }
        }
        return plans.emplace( key, plan ).first->second;
    };

    int const pLo = autoPart ? 1 : int(userSplit[0]);
    int const pHi = autoPart ? 6 : int(userSplit[0]);
    // the threads a partition may use: every core, or the user's --max-concurrent (0 = all threads)
    int const userT = u.maxConcurrent > 0 ? std::min( u.maxConcurrent, cores ) : cores;
    int const tMax = autoConc ? cores : userT;
    /* Only a split that FITS competes (2026-10-12). The old rule let a T=1 split that did not fit win the
       'T==bestT and P<bestP' tie, so whenever one thread was all any split allowed it returned the COARSEST split,
       the one needing the most memory: TNG300-3-Dark z=0 at --padding 50 under a 36-38 GB budget printed
       '--partition 1 1 1, 1 thread(s) (predicted peak ~591 GB)' with no warning, where --partition 6 6 6 with one
       thread was predicted to fit. Every choice whose prediction fitted is unchanged (a T>1 split fits by
       construction, and among fitting T=1 splits the coarsest still wins); when no split fits at all, the one
       predicted to need the least memory is taken and a warning says so. A user's --max-concurrent is the thread
       count a split is checked at (2026-10-13; it started at every core whatever the user gave). */
    int bestP = pHi, bestT = 0;
    int leastP = pHi;                      // the split with the smallest predicted peak, for the nothing-fits case
    int const tLeast = autoConc ? 1 : tMax;
    for (int P=pLo; P<=pHi; ++P)
    {
        if ( planOf(P,tLeast).bytes < planOf(leastP,tLeast).bytes ) leastP = P;
        int T = tMax;
        while ( autoConc and T>1 and fixed + planOf(P,T).bytes > budget )
            --T;
        if ( fixed + planOf(P,T).bytes > budget )
            continue;   // not even one thread fits at this split
        if ( T > bestT or (T==bestT and P<bestP) )
        {
            bestP = P;
            bestT = T;
        }
        if ( T == tMax )
        {
            bestP = P;
            bestT = T;
            break;      // full thread count fits at the coarsest split so far: done
        }
    }
    bool const nothingFits = ( bestT == 0 );
    if ( nothingFits )
    {
        bestP = leastP;
        bestT = tLeast;
    }

    /* The report (2026-10-13; it was PS-only, 'AUTO-TUNE-REPORT unsupported' here): the same line PS prints, so the
       launcher's 'Check memory', run_dtfe.sh AUTO_TUNE_REPORT=1 and the SUMMARY's predicted-vs-measured line work for
       standard runs too. mc = the threads of a partition (its concurrent sub-triangulations); the partitions run one
       after the other. perPartition = the busiest partition's bytes on top of 'fixed'. */
    StdPlan const &chosen = planOf( bestP, bestT );
    double const predicted = fixed + chosen.bytes;
    report.valid = true;
    report.partition = bestP;
    report.maxConcurrent = bestT;
    report.predicted = predicted;
    report.fixed = fixed;
    report.budget = budget;
    report.perPartition = chosen.bytes;
    if ( const char *dbg = std::getenv("DTFE_AUTO_TUNE_DEBUG") )
        if ( std::atoi( dbg ) != 0 )
        {
            std::fprintf( stderr, "AUTO-TUNE-DEBUG standard: %s (count grid %.3f s, %zu plans %.3f s); busiest partition "
                          "%d of %zu: %.0f padded vertices, %.3f GB on top of %.3f GB; per thread:",
                          dens.ok ? "counted from the particles" : "uniform", countSec, plans.size(),
                          std::chrono::duration<double>( std::chrono::steady_clock::now() - tuneStart ).count() - countSec,
                          chosen.part + 1, chosen.parts, chosen.vertices, chosen.bytes/1.e9, fixed/1.e9 );
            for (double v : chosen.perThread) std::fprintf( stderr, " %.0f", v );
            std::fprintf( stderr, "\n" );
        }

    // single domain with all threads fits: keep today's defaults (and stay silent)
    if ( bestP==1 and bestT==cores )
        return report;

    if ( autoPart )
    {
        u.partition.assign( NO_DIM, size_t(bestP) );
        u.partitionOn = (bestP > 1);
    }
    if ( autoConc and bestT < cores )
        u.maxConcurrent = bestT;

    std::vector<size_t> splitShown = u.partition;
    if ( splitShown.size() != size_t(NO_DIM) ) splitShown.assign( NO_DIM, size_t(bestP) );
    char vertNote[160] = "";
    if ( dens.ok )
        snprintf( vertNote, sizeof(vertNote), "; the busiest partition triangulates ~%.3g M padded vertices at once",
                  chosen.vertices/1.e6 );
    snprintf( buf, sizeof(buf),
              "%.3g particles, %s grid, %.3g GB RAM (budget %.3g GB) -> --partition %s, %d thread(s) "
              "(predicted peak ~%.3g GB%s%s)",
              N, MESSAGE::printElements( u.gridSize, "x" ).c_str(), ramBytes/1.e9, budget/1.e9,
              MESSAGE::printElements( splitShown, " " ).c_str(),
              bestT, predicted/1.e9, metalActive ? ", GPU interpolation" : "", vertNote );
    message << "\n" << MESSAGE::cBold() << "AUTO-TUNE:" << MESSAGE::cReset() << " " << buf
            << ". Pass --partition/--max-concurrent to override.\n           memory budget: "
            << budgetWhy << "\n" << MESSAGE::Flush;
    if ( nothingFits )
    {
        MESSAGE::Warning warning( u.verboseLevel );
        warning << "AUTO-TUNE: no split fits the memory budget: even the least demanding one (--partition "
                << MESSAGE::printElements( splitShown, " " )
                << " with " << bestT << " thread(s)) is predicted to need "
                << predicted/1.e9 << " GB against a " << budget/1.e9 << " GB budget. The run may swap or be "
                << "killed: free memory, lower --padding (the dummy-point test reports whether it still suffices) "
                << "or --grid, or pass a finer --partition.\n" << MESSAGE::EndWarning;
    }
    return report;
#endif
}

/* The partition MEMORY GATE (2026-10-07). The auto-tuner plans a peak from a model; the machine's free memory
   is the fact, and other programs change it during a run (TNG50-3-Dark z=0: planned 47 GB under a 52 GB budget,
   SIGKILLed by macOS after 82 min with a browser and the launcher open). Before a partition starts, it waits
   -- in 2 s steps -- while the memory the system leaves free, minus the headroom, is below what one partition
   adds (AutoTuneReport::perPartition) AND another partition is in flight (whose finishing frees memory). With
   none in flight it always starts, saying so: the gate can delay, never deadlock. A skipped partition and the
   speed split's ordered merge never pass it; a cache-only one does (its loaded tessellation holds the same
   memory). need = 0 (no auto-tune estimate: a user-given --partition) disables it.
   ORDERED (--ordered-merge, 2026-10-11; the default since 2026-10-14): a finished partition can wait for every
   lower-numbered one to merge while it still holds its memory (since 2026-10-14 only when the staging room is full:
   DTFE.cpp's staged ordered merge), so a partition waiting here for a HIGHER-numbered one to free memory could wait
   forever; in that mode a partition waits only while a lower-numbered one is in flight (those never wait for it).
   The partition entering is always the highest index handed out so far (the loop's dynamic schedule hands them
   out in order), so every partition in flight is lower-numbered and the two modes wait alike. A STAGED partition
   (finished, its sub-grids parked until its turn) has left the gate: its memory shows in the free-memory reading,
   and it was staged only while that reading left room for one more partition. */
struct PartitionMemoryGate
{
    double need = 0., headroom = 4.e9;
    bool ordered = false;
    std::mutex m;
    std::condition_variable cv;
    int inFlight = 0, held = 0;
    std::multiset<int> flying;   // the in-flight partition indices (the ordered mode's blockers)
    bool saidAlone = false;
    explicit PartitionMemoryGate(double needBytes, bool orderedMerge = false) : need(needBytes), ordered(orderedMerge) {}

    // the in-flight partitions whose finishing this one may wait for
    int blockers(int partition) const
    {
        if ( not ordered ) return inFlight;
        return int( std::distance( flying.begin(), flying.lower_bound( partition ) ) );
    }

    void enter(int partition, int verbose)
    {
        std::unique_lock<std::mutex> lock( m );
        if ( need > 0. )
        {
            bool waited = false;
            double free = autoTuneSystemFree();
            while ( free >= 0. and free - headroom < need and blockers( partition ) > 0 )
            {
                if ( not waited )
                {
                    ++held;
                    if ( verbose > 0 )
                        std::fprintf( stderr, "PS-DTFE memory gate: partition %d waits (%.1f GB free, minus %.0f GB "
                                      "headroom, < the %.1f GB it adds; %d in flight)\n", partition, free/1.e9,
                                      headroom/1.e9, need/1.e9, inFlight );
                    waited = true;
                }
                cv.wait_for( lock, std::chrono::seconds(2) );
                free = autoTuneSystemFree();
            }
            if ( free >= 0. and free - headroom < need and blockers( partition ) == 0 and not saidAlone )
            {
                saidAlone = true;
                if ( verbose > 0 )
                    std::fprintf( stderr, "PS-DTFE memory gate: %.1f GB free, minus %.0f GB headroom, is below the "
                                  "%.1f GB a partition adds -- running one partition at a time from now on\n",
                                  free/1.e9, headroom/1.e9, need/1.e9 );
            }
        }
        ++inFlight;
        flying.insert( partition );
    }
    void leave(int partition)
    {
        {
            std::lock_guard<std::mutex> lock( m );
            --inFlight;
            auto const it = flying.find( partition );
            if ( it != flying.end() ) flying.erase( it );
        }
        cv.notify_all();
    }
};

#endif
