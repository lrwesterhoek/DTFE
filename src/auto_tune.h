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
     DTFE_MEM_BUDGET_GB the budget itself, in GB -- overrides both rules above
     DTFE_AUTO_MINN     minimum particle count before auto-partitioning engages (default 2e6;
                        below it a single domain always fits and partition seams are not worth it) */

#ifndef AUTO_TUNE_HEADER
#define AUTO_TUNE_HEADER

#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <string>
#include <sys/statvfs.h>
#include <sys/stat.h>
#include <fstream>
#include <thread>      // hardware_concurrency: the point-evaluation HitSink blocks, one per thread
#include <dirent.h>    // autoTuneServe: the cached partition splits of a snapshot
#include <map>
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

// Bytes the OTHER programs leave free right now, plus what this process already holds (its
// particles etc., which the run's model counts itself); < 0 when the OS cannot tell.
//   macOS: RAM minus Activity Monitor's "Memory Used" (app memory = anonymous - purgeable,
//          + wired + compressed); Linux: MemAvailable. File cache counts as free: it is dropped.
inline double autoTuneFreeForRun()
{
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
    return double(mem) - used + own;
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
    return availKB < 0. ? -1. : (availKB + ownKB) * 1024.;
#endif
}

// The memory budget the tuner plans against, and a one-line account of how it was set.
inline double autoTuneBudget(double ramBytes, std::string &why)
{
    char buf[256];
    if ( const char *env = std::getenv("DTFE_MEM_BUDGET_GB") )
    {
        double v = std::atof(env);
        if ( v > 0. )
        {
            snprintf( buf, sizeof(buf), "%.3g GB set by DTFE_MEM_BUDGET_GB", v );
            why = buf;
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
    bool const simulated = std::getenv("DTFE_RAM_GB") != NULL and std::atof(std::getenv("DTFE_RAM_GB")) > 0.;
    double const freeForRun = simulated ? -1. : autoTuneFreeForRun();
    double const headroom = 4.e9;
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


struct AutoTunePS
{
    bool phaseSpace   = false;
    bool volumeWeighted = false;   // --ps-volume-weighted
    bool exactDeposit = false;     // --ps-exact-deposit
    bool caustics     = false;     // --ps-caustics
    bool haloRelease  = false;     // --ps-halo-release (sampled deposit: the released-tet multiplicity grid)
};

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
            f *= 1. + 2.*std::max( floorFrac, padFrac*double(n) );
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
        double const est = psEstimateMeanStreams( *particles, boxLo, boxLen, u.periodic, &patches );
        if ( est > 0. )
        {
            ptsMeanStreams = std::max( 1., 1. + 2.*(est - 1.) );
            report.streamsEstimate = est;
            snprintf( ptsStreamsWhy, sizeof(ptsStreamsWhy),
                      "%.2f streams/point estimated from %d Lagrangian patches, budgeted as %.2f", est, patches,
                      ptsMeanStreams );
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

    // per-partition bytes for an n^3 Lagrangian split
    auto partitionBytes = [&](int n) -> double
    {
        double const nOwn = N / double(n)/double(n)/double(n);
        double fPad = 1.;
        for (int d=0; d<NO_DIM; ++d)
            fPad *= 1. + 2.*std::max( 0.3, padFrac[d]*double(n) );   // 0.3-partition-cell padding floor (DTFE.cpp)
        double const nPad = nOwn * fPad;
        double m = nPad * (partBytes + kDT);        // particle copy coexists with the growing triangulation
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
            // Every buffer in PSGpuGrids, mirrored host+device (the x2 below). mass 4 + streams 4
            // = the 8; the fVolW/fCaustic/fExact terms were missing and fVolW is the PRODUCTION
            // default -- 20 B/cell/copy = 40 after the x2, i.e. 6.7 GB per partition at 1024^3/n=2.
            double const gridB  = 8. + (psVel ? 12. : 0.) + (psDisp ? 24. : 0.) + (psGrad ? 36. : 0.)
                                + (psVolW ? 4. : 0.)                              // momw
                                + (psCaust ? 4. : 0.)                             // caustic
                                + (psExact ? 4. : 0.)                             // streamvol
                                + ((psVolW and psDisp) ? 16. : 0.);               // dispvel 12 + dispw 4
            m += perTet*6.77*nOwn + 2.*gridB*subCells;
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
                m += 4.*nTet + 8.*std::min( 32.e6, nTet*(1. + 20.*cellsPerTet/16.) );
            }
        }
        else
        {
            // CPU deposit sub-grids (was a hardcoded 52) + the deposit threads' private copies, as many
            // as the copy budget allows -- unless that leaves fewer than half the threads: the deposit
            // then runs in SLAB mode (ps_interpolation.cc, the same rule) with no copies, holding a cell
            // handle, a first-slab byte and ~2 slab-list entries per tetrahedron (+ the binning arrays)
            double const copyOne = bCellDeposit*subCells;
            int const copyThreads = std::min( cores, 1 + int( psDepositThreadBytes() / std::max( 1., copyOne ) ) );
            double const slabBytes = 6.77 * nPad * ( 8. + 1. + 2.*4. + 20. );
            m += copyOne + ( 2 * copyThreads < cores ? slabBytes : double(copyThreads - 1) * copyOne );
        }
        return m;
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
        report.predicted = fixed + double(inFlight) * partitionBytes( report.partition );
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
    bool const smallFits = not autoPart or fixed + partitionBytes( std::max( 1, nGpuMin ) ) <= budget;
    if ( N < minN and smallFits )     // small particle set AND the grid fits: one domain, or a split for SPEED
    {
        if ( not autoPart )
            return finishReport( userN, autoConc ? cores : userC );
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
            for (int d=0; d<NO_DIM; ++d) fPad *= 1. + 2.*std::max( 0.3, padFrac[d]*double(n) );
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
                                               int( std::floor( (budget - fixed) / partitionBytes(nGpuMin) ) ) } ) );
            if ( not autoConc ) bestC = std::min( bestC, userC > 0 ? userC : cores );
            bestT = predictedWall( nGpuMin, bestC );
        }
        for (int n = std::max( 2, nGpuMin ); n <= std::max( 3, nGpuMin ); ++n)
        {
            int c = std::min( n*n*n, cores );
            int const cMem = int( std::floor( (budget - fixed) / partitionBytes(n) ) );
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
            if ( autoConc ) u.maxConcurrent = bestC;
            if ( bestN == nGpuMin and nGpuMin > 1 )
                snprintf( buf, sizeof(buf),
                          "%s grid on the GPU: --partition %d %d %d --max-concurrent %d keeps each partition's "
                          "sub-grid under the 2^32 cells the GPU kernels number (~%.3g cells; predicted peak ~%.3g GB)",
                          MESSAGE::printElements( u.gridSize, "x" ).c_str(), bestN, bestN, bestN, u.maxConcurrent,
                          gpuSubCells(bestN), (fixed + double(bestC)*partitionBytes(bestN))/1.e9 );
            else
                snprintf( buf, sizeof(buf),
                          "%.3g particles: triangulation-bound, so --partition %d %d %d --max-concurrent %d "
                          "(predicted %.2g s against %.2g s for one triangulation; predicted peak ~%.3g GB)",
                          N, bestN, bestN, bestN, u.maxConcurrent, bestT, single,
                          (fixed + double(bestC)*partitionBytes(bestN))/1.e9 );
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
                  (fixed + partitionBytes(bestN))/1.e9, metalActive ? ", GPU deposit" : "" );
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
        int cMem = int( std::floor( (budget - fixed) / m ) );
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
    double const predicted = fixed + double(effC)*partitionBytes(bestN);
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
    return finishReport( bestN, effC );

#else
    // ---- standard DTFE: partitions are processed SERIALLY (memory knob); inside one partition
    // the box is split across up to --max-concurrent threads (throughput knob). ----
    double const kDT = 648.;                        // bytes/vertex: 96 (incl. the double volume accumulator) + 6.77 cells x 72 B + slack
    double const partBytes = double(sizeof(Particle_data));
    // --scratch-dir: the full-grid accumulators are mmap-backed and leave the RAM model
    double const fixed = partBytes*N + (u.scratchDir.empty() ? gridTotal*bCell : 0.);

    // concurrent triangulation bytes for a P^3 serial split with T threads per partition
    auto triangulationBytes = [&](int P, int T) -> double
    {
        double const tSplit = std::cbrt( double(T) );
        double f = 1.;
        for (int d=0; d<NO_DIM; ++d)
            f *= 1. + 2.*padFrac[d]*double(P)*tSplit;    // each thread's sub-box is padded
        double m = (N / double(P)/double(P)/double(P)) * f * (partBytes + kDT);
        if ( metalActive )
            m += (N / double(P)/double(P)/double(P)) * f * 6.77 * 120.;  // flat tet arrays (transient)
        return m;
    };

    int const pLo = autoPart ? 1 : int(u.partition[0]);
    int const pHi = autoPart ? 6 : int(u.partition[0]);
    int bestP = pHi, bestT = 1;
    for (int P=pLo; P<=pHi; ++P)
    {
        int T = cores;
        while ( T>1 and fixed + triangulationBytes(P,T) > budget )
            --T;
        if ( T > bestT or (T==bestT and P<bestP) )
        {
            bestP = P;
            bestT = T;
        }
        if ( T == cores )
        {
            bestP = P;
            bestT = T;
            break;      // full thread count fits at the coarsest split so far: done
        }
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

    double const predicted = fixed + triangulationBytes(bestP, bestT);
    snprintf( buf, sizeof(buf),
              "%.3g particles, %s grid, %.3g GB RAM (budget %.3g GB) -> --partition %d %d %d, %d thread(s) "
              "(predicted peak ~%.3g GB%s)",
              N, MESSAGE::printElements( u.gridSize, "x" ).c_str(), ramBytes/1.e9, budget/1.e9,
              int(u.partition.size()==size_t(NO_DIM) ? u.partition[0] : 1),
              int(u.partition.size()==size_t(NO_DIM) ? u.partition[1] : 1),
              int(u.partition.size()==size_t(NO_DIM) ? u.partition[2] : 1),
              bestT, predicted/1.e9, metalActive ? ", GPU interpolation" : "" );
    message << "\n" << MESSAGE::cBold() << "AUTO-TUNE:" << MESSAGE::cReset() << " " << buf
            << ". Pass --partition/--max-concurrent to override.\n           memory budget: "
            << budgetWhy << "\n" << MESSAGE::Flush;
    return report;     // the report covers PS-DTFE only (valid = false here)
#endif
}

#endif
