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
#include <sys/statvfs.h>
#include <sys/stat.h>
#include <fstream>

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
        if ( ps.caustics )     b += R;                             // caustic_bits (DTFE.cpp:512)
        if ( ps.exactDeposit ) b += R;                             // tet_touch    (DTFE.cpp:519)
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

    AutoTunePS ps;
#ifdef PHASE_SPACE
    ps.phaseSpace     = true;
    ps.volumeWeighted = u.psVolumeWeighted;
    ps.exactDeposit   = u.psExactDeposit;
    ps.caustics       = u.psCaustics;
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
    double const kDT = 658.;                        // bytes/vertex: 104 (incl. the uint64 ParticleID) + 6.77 cells x 72 B + allocator slack
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
    size_t const nPts = autoTuneSamplePointCount( u.psSamplePointsFile );
    // Mean streams (containing tetrahedra) per point -- the one quantity that cannot be known
    // before the tessellation exists. MEASURED on TNG300-3-Dark, 8192^2 full-box plane:
    // 1.000 at z=20, 1.899 at z=0. 2.0 is the low-z value rounded up; override for unusual
    // configurations (a plane through a cluster core samples far more streams).
    double ptsMeanStreams = 2.0;
    if ( const char *env = std::getenv("DTFE_AUTO_PTS_STREAMS") )
    {
        double const v = std::atof(env);
        if ( v >= 1. ) ptsMeanStreams = v;
    }
    // StreamRec (ps_point_eval.cc): denGeo + denDtfe + vel[3] + grad[3] + vgrad[9] doubles
    // + ids[4] uint64 = 168 B. It is that size ALWAYS -- the --pts-den-grad/--pts-vel-grad/
    // --per-stream-ids members exist whether or not those flags are set. KEEP IN SYNC.
    double const bStreamRec = 17. * double(sizeof(double)) + 4. * 8.;
    // per-point: pos (3 doubles) + bucketPts (uint32) + the recs vector header
    double const bPerPoint = 3. * double(sizeof(double)) + 4. + double(sizeof(void*)) * 3.;
    // finalized outputs, allocated while the records are still being drained
    double bPtsOut = double(sizeof(double))            // outDen
                   + 3. * double(sizeof(double))       // outVel
                   + 6. * double(sizeof(double))       // outDisp
                   + 4.;                               // outStreams (int32)
    if ( u.psPtsDenGrad ) bPtsOut += 3. * double(sizeof(double));
    if ( u.psPtsVelGrad ) bPtsOut += 9. * double(sizeof(double));
    if ( u.psPerStream )  bPtsOut += ptsMeanStreams * 4. * double(sizeof(double));   // ragged records
    double const ptsBytes = double(nPts) * (bPerPoint + bPtsOut + ptsMeanStreams * bStreamRec);

    // IRREDUCIBLE in RAM unless scratch-backed: originals + the full-grid accumulators
    // + the point-evaluation structures.
    // --partition does NOT reduce this -- partitions sum into the SHARED full grid.
    double const fixed = partBytes*N + (scratchOn ? 0. : gridTotal*bCell) + ptsBytes;
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
    if ( fixed > budget )
    {
        double const gridBytes = gridTotal*bCell;
        // largest cubic grid whose accumulators still leave room for the particles
        double const room = budget - partBytes*N;
        long   const maxGrid = room > 0. ? long( std::cbrt( room / bCell ) ) : 0L;
        MESSAGE::Warning warning( u.verboseLevel );
        warning << "AUTO-TUNE: this run does not fit in memory, and --partition cannot fix it.\n"
                << "  The full-resolution output grids alone need " << gridBytes/1.e9 << " GB ("
                << MESSAGE::printElements( u.gridSize, "x" ) << " cells x " << bCell << " B/cell for the "
                << "selected fields), plus " << (partBytes*N)/1.e9 << " GB of particles = "
                << fixed/1.e9 << " GB, against a budget of " << budget/1.e9 << " GB ("
                << ramBytes/1.e9 << " GB RAM).\n"
                << "  --partition splits the TRIANGULATION only: every Lagrangian partition deposits into "
                << "the same shared full-resolution grid, so that " << gridBytes/1.e9 << " GB is irreducible.\n"
                << "  Options: (a) '--scratch-dir <local dir>' backs the grids with disk instead of RAM "
                << "(needs ~" << gridBytes/1.e9 << " GB free on a local, non-synced volume, e.g. "
                << "/private/tmp/dtfe-scratch); (b) a smaller --grid -- in RAM, the largest that fits at "
                << "these fields is about " << maxGrid << "^3; (c) fewer --field entries (the gradient "
                << "family costs 36 B/cell, dispersion 24 B/cell); (d) more RAM. NOTE: '--partNo'/'--region' "
                << "are NOT a way out -- they crop particles by Eulerian position and are rejected by this "
                << "binary because the PS tessellation is Lagrangian.\n"
                << "  Running anyway with --max-concurrent 1; expect heavy swapping.\n" << MESSAGE::EndWarning;
        if ( autoPart ) { u.partition.assign( NO_DIM, size_t(1) ); u.partitionOn = false; }
        if ( autoConc ) u.maxConcurrent = 1;
        return;
    }

    if ( N < minN )
        return;     // small particle set AND the grid fits: a single domain is fine

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
    // flags gating the remaining GPU buffers (ps_metal_host.cc / ps_gpu_cuda.cu)
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
            m += double(nPts) * ptsMeanStreams / (double(n)*double(n)*double(n))
                 * (bStreamRec + double(sizeof(uint32_t)) + 4.);
        // Eulerian sub-grid this partition allocates (psUseSubgrid). A Lagrangian sub-box maps to
        // a DISPLACED, distorted Eulerian region, so the crop is the touched span + slop -- valid
        // on both periodic and non-periodic runs since ps_interpolation.cc now crops a periodic
        // axis to its WRAPPED span instead of keeping the whole axis. (Before that fix every
        // boundary-touching slab kept the full axis, making this term ~n^3 optimistic.)
        double subCells = 1.;
        for (int d=0; d<NO_DIM; ++d)
            subCells *= std::min( double(u.gridSize[d]),
                                  double(u.gridSize[d])/double(n) + 40. );   // crop + slop, never above the axis
        if ( metalActive )
        {
            // flat tet arrays (verts 48 + masses 4 B/tet, + vels 48 when velocity-derived
            // grids are requested; the cost-sort permutes IN PLACE, so no gather copy)
            // + the PSGpuGrids host and device copies of the selected grids
            double const perTet = (52. + (psVel or psDisp or psGrad ? 48. : 0.)) * 1.3;   // + allocator slack
            // Every buffer in PSGpuGrids, mirrored host+device (the x2 below). mass 4 + streams 4
            // = the 8; the fVolW/fCaustic/fExact terms were missing and fVolW is the PRODUCTION
            // default -- 20 B/cell/copy = 40 after the x2, i.e. 6.7 GB per partition at 1024^3/n=2.
            double const gridB  = 8. + (psVel ? 12. : 0.) + (psDisp ? 24. : 0.) + (psGrad ? 36. : 0.)
                                + (psVolW ? 4. : 0.)                              // momw
                                + (psCaust ? 4. : 0.)                             // caustic
                                + (psExact ? 4. : 0.)                             // streamvol
                                + ((psVolW and psDisp) ? 16. : 0.);               // dispvel 12 + dispw 4
            m += perTet*6.77*nOwn + 2.*gridB*subCells;
        }
        else
            m += bCellDeposit*subCells;             // CPU deposit sub-grids (was a hardcoded 52)
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

    double const predicted = fixed + double(bestC)*partitionBytes(bestN);
    char scratchNote[256] = "";
    if ( scratchOn )
        snprintf( scratchNote, sizeof(scratchNote), "; grid accumulators (%.3g GB) on scratch disk",
                  gridTotal*bCell/1.e9 );
    char ptsNote[160] = "";
    if ( nPts > 0 )
        snprintf( ptsNote, sizeof(ptsNote), "; %.3g sample points (%.3g GB at %.2g streams/point)",
                  double(nPts), ptsBytes/1.e9, ptsMeanStreams );
    snprintf( buf, sizeof(buf),
              "%.3g particles, %s grid, %.3g GB RAM (budget %.3g GB) -> --partition %d %d %d --max-concurrent %d "
              "(predicted peak ~%.3g GB%s%s%s)",
              N, MESSAGE::printElements( u.gridSize, "x" ).c_str(), ramBytes/1.e9, budget/1.e9,
              int(u.partition[0]), int(u.partition[1]), int(u.partition[2]), u.maxConcurrent,
              predicted/1.e9, metalActive ? ", GPU deposit" : "", scratchNote, ptsNote );
    message << "\n" << MESSAGE::cBold() << "AUTO-TUNE:" << MESSAGE::cReset() << " " << buf
            << ". Pass --partition/--max-concurrent to override.\n" << MESSAGE::Flush;

#else
    // ---- standard DTFE: partitions are processed SERIALLY (memory knob); inside one partition
    // the box is split across up to --max-concurrent threads (throughput knob). ----
    double const kDT = 640.;                        // bytes/vertex: 88 + 6.77 cells x 72 B + slack
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
