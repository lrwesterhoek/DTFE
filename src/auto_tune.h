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

   Environment overrides (for tests and unusual machines):
     DTFE_RAM_GB    physical-RAM override in GB (default: queried from the OS)
     DTFE_AUTO_MINN minimum particle count before auto-partitioning engages (default 2e6;
                    below it a single domain always fits and partition seams are not worth it) */

#ifndef AUTO_TUNE_HEADER
#define AUTO_TUNE_HEADER

#include <cstdlib>
#include <cmath>
#include <algorithm>

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#else
#include <unistd.h>
#endif
#ifdef OPEN_MP
#include <omp.h>
#endif

#include "user_options.h"
#include "particle_data.h"
#include "message.h"


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

inline int autoTuneCores()
{
#ifdef OPEN_MP
    return std::max( omp_get_max_threads(), 1 );
#else
    return 1;
#endif
}

// Peak bytes per grid cell for one field selection (full-grid accumulators plus the
// post-processing coexistence of the gradient with its derived fields). Upper-bound-ish.
inline double autoTuneBytesPerCell(Field &f, bool const phaseSpace)
{
    if ( not f.selected() ) return 0.;
    double const R = double(sizeof(Real));
    double b = 0.;
    bool const gradient = f.velocity_gradient or f.velocity_divergence or f.velocity_shear
                          or f.velocity_vorticity or f.velocity_vweb;
    if ( f.density or f.velocity_tweb ) b += R;                    // T-web needs the density grid
    if ( f.velocity or f.velocity_dispersion ) b += 3.*R;         // dispersion allocates velocity too
    if ( gradient ) b += 9.*R;
    if ( f.velocity_divergence ) b += R;
    if ( f.velocity_shear ) b += 5.*R;
    if ( f.velocity_vorticity ) b += 3.*R;
    if ( f.velocity_std ) b += R;
    if ( f.velocity_dispersion ) b += 6.*R;
    if ( f.scalar ) b += noScalarComp*R;
    if ( f.scalar_gradient ) b += noScalarGradComp*R;
    if ( f.velocity_tweb ) b += R + 3.*R;                          // web label + eigenvalues
    if ( f.velocity_vweb ) b += R + 3.*R;
    if ( phaseSpace )
    {
        b += R;                                                    // stream_count
        // mass_weight is a separate grid only when weighted fields are selected WITHOUT the
        // density field (with density it aliases the density accumulator; 2026-07 change)
        bool const weighted = f.velocity or f.velocity_gradient or f.velocity_dispersion
                              or f.velocity_divergence or f.velocity_shear or f.velocity_vorticity
                              or f.velocity_vweb or f.scalar or f.scalar_gradient;
        if ( weighted and not f.density ) b += R;
    }
    return b;
}


// Chooses --partition / --max-concurrent when the user did not give them. 'metalActive' =
// the GPU deposit will actually run (flag given AND compiled in). Mutates userOptions.
inline void autoTunePartitioning(User_options &u,
                                 size_t const nParticles,
                                 bool const userSampling,
                                 bool const metalActive)
{
    bool const autoPart = not u.partitionOn;
    bool const autoConc = not u.maxConcurrentOn;
    if ( (not autoPart and not autoConc) or u.partNo>=0 or u.redshiftConeOn or userSampling )
        return;     // nothing left to decide, or a mode where partitioning is unavailable

    double minN = 2.e6;
    if ( const char *env = std::getenv("DTFE_AUTO_MINN") )
    {
        double v = std::atof(env);
        if ( v > 0. ) minN = v;
    }

    double const N = double(nParticles);
    double const ramBytes = autoTunePhysicalRAM();
    double const budget = 0.80 * ramBytes;      // leave headroom for the OS and allocator slack
    int const cores = autoTuneCores();

    double gridTotal = 1.;
    for (int d=0; d<NO_DIM; ++d)
        gridTotal *= double(u.gridSize[d]);
    double const bCell = autoTuneBytesPerCell(u.uField,
#ifdef PHASE_SPACE
                                              true) + autoTuneBytesPerCell(u.aField, true);
#else
                                              false) + autoTuneBytesPerCell(u.aField, false);
#endif

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
    if ( N < minN )
        return;     // single domain always fits at this size; keep historical behavior

    double const kDT = 650.;                        // bytes/vertex: 96 + 6.77 cells x 72 B + allocator slack
    double const partBytes = double(sizeof(Particle_data));
    double const fixed = partBytes*N + gridTotal*bCell;    // originals + full-grid accumulators

    // which per-tet/per-cell GPU-deposit pieces this field selection actually needs
    // (2026-07: unselected moment grids are no longer allocated, and the tet velocity
    // array is only extracted when a velocity-derived grid is requested). The velocity
    // derivatives (divergence etc.) are folded into the gradient AFTER auto-tune runs,
    // so count them here as gradient requests.
    bool const psVel  = u.uField.velocity or u.uField.velocity_dispersion
                        or u.aField.velocity or u.aField.velocity_dispersion;
    bool const psDisp = u.uField.velocity_dispersion or u.aField.velocity_dispersion;
    bool const psGrad = u.uField.velocity_gradient or u.uField.selectedVelocityDerivatives()
                        or u.aField.velocity_gradient or u.aField.selectedVelocityDerivatives();

    // per-partition bytes for an n^3 Lagrangian split
    auto partitionBytes = [&](int n) -> double
    {
        double const nOwn = N / double(n)/double(n)/double(n);
        double fPad = 1.;
        for (int d=0; d<NO_DIM; ++d)
            fPad *= 1. + 2.*std::max( 0.3, padFrac[d]*double(n) );   // 0.3-partition-cell padding floor (DTFE.cpp)
        double const nPad = nOwn * fPad;
        double m = nPad * (partBytes + kDT);        // particle copy coexists with the growing triangulation
        double subCells = 1.;
        for (int d=0; d<NO_DIM; ++d)
            subCells *= double(u.gridSize[d])/double(n) + 40.;       // psUseSubgrid crop + slop
        if ( metalActive )
        {
            // flat tet arrays (verts 48 + masses 4 B/tet, + vels 48 when velocity-derived
            // grids are requested; the cost-sort permutes IN PLACE since 2026-07, so no
            // gather copy) + the PSGpuGrids host and device copies of the selected grids
            double const perTet = (52. + (psVel or psDisp or psGrad ? 48. : 0.)) * 1.3;   // + allocator slack
            double const gridB  = 8. + (psVel ? 12. : 0.) + (psDisp ? 24. : 0.) + (psGrad ? 36. : 0.);
            m += perTet*6.77*nOwn + 2.*gridB*subCells;
        }
        else
            m += 52.*subCells;                      // CPU deposit sub-grids
        return m;
    };

    // GPU deposits serialize on one queue, so past ~3 concurrent partitions extra ones only
    // hold memory while waiting; CPU runs scale up to the core count.
    int const targetC = metalActive ? std::min(cores, 3) : cores;

    int bestN = -1, bestC = 0;
    int const nLo = autoPart ? 2 : int(u.partition[0]);
    int const nHi = autoPart ? 6 : int(u.partition[0]);   // <=6^3 keeps auto within validated territory
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
        bestC = 1;      // even one partition at the finest split exceeds the budget: warn, run anyway
        MESSAGE::Warning warning( u.verboseLevel );
        warning << "AUTO-TUNE: even a single " << bestN << "^3-split partition is predicted to exceed the "
                << "memory budget; running with --max-concurrent 1. Expect swapping; consider a smaller grid "
                << "or fewer fields.\n" << MESSAGE::EndWarning;
    }

    if ( autoPart )
    {
        u.partition.assign( NO_DIM, size_t(bestN) );
        u.partitionOn = (bestN > 1);
    }
    if ( autoConc )
        u.maxConcurrent = bestC;

    double const predicted = fixed + double(bestC)*partitionBytes(bestN);
    snprintf( buf, sizeof(buf),
              "%.3g particles, %s grid, %.3g GB RAM (budget %.3g GB) -> --partition %d %d %d --max-concurrent %d "
              "(predicted peak ~%.3g GB%s)",
              N, MESSAGE::printElements( u.gridSize, "x" ).c_str(), ramBytes/1.e9, budget/1.e9,
              int(u.partition[0]), int(u.partition[1]), int(u.partition[2]), u.maxConcurrent,
              predicted/1.e9, metalActive ? ", GPU deposit" : "" );
    message << "\n" << MESSAGE::cBold() << "AUTO-TUNE:" << MESSAGE::cReset() << " " << buf
            << ". Pass --partition/--max-concurrent to override.\n" << MESSAGE::Flush;

#else
    // ---- standard DTFE: partitions are processed SERIALLY (memory knob); inside one partition
    // the box is split across up to --max-concurrent threads (throughput knob). ----
    double const kDT = 640.;                        // bytes/vertex: 88 + 6.77 cells x 72 B + slack
    double const partBytes = double(sizeof(Particle_data));
    double const fixed = partBytes*N + gridTotal*bCell;

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
        return;

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
            << ". Pass --partition/--max-concurrent to override.\n" << MESSAGE::Flush;
#endif
}

#endif
