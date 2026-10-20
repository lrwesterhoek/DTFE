/*
 *  Copyright (c) 2011       Marius Cautun
 *
 *                           Kapteyn Astronomical Institute
 *                           University of Groningen, the Netherlands
 *
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

/* DTFE driver: data setup, padding, partitioning (standard and PS-DTFE), and velocity-derivative/web post-processing.
   The derived velocity fields and the V-web / T-web classification themselves are in web_fields.cc. */

#include <vector>
#include <cmath>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <chrono>   // partition-loop progress + ETA
#include <thread>      // the partition look-ahead
#include <exception>
#include <memory>
#include <map>       // the staged ordered merge's parked sub-grids
#include <functional>   // the standard look-ahead's memory estimate
#ifdef OPEN_MP
    #include <omp.h>
#endif
#include <boost/math/special_functions/fpclassify.hpp>


#include "define.h"
#include "particle_data.h"
#include "user_options.h"
#include "quantities.h"
#include "subpartition.h"
#include "miscellaneous.h"
#include <limits>

#include "message.h"
#include "auto_tune.h"
#include "scratch_alloc.h"
#include "ps_point_eval.h"
#include "ps_globals_record.h"
#include "run_progress.h"   // the '<output>.runlog.progress' sidecar
#include "partition_checkpoint.h"   // checkpoint + '--resume' of a partitioned run (both binaries)
#include "web_fields.h"   // postProcessWebFields: the derived velocity fields and the V-web / T-web (web_fields.cc)
#ifdef PHASE_SPACE
#include "CGAL_triangulation/ps_mass_audit.h"   // PS_MASS_AUDIT=1: the run's mass closure after the merge
#include "ps_merge_spill.h"   // the staged ordered merge's spill of parked sub-grids to disk
#include "ps_fixed_merge.h"   // '--fixed-point-merge': the order-free merge into exact integer grids
#endif
#include "io/io.h"         // readInputData: the late read of a --ps-window run that started without its particles
#include "CGAL_triangulation/ps_deposit_params.h"   // fixedShareDrops: the non-finite guard's counts


using namespace std;

#include "interpolations.h"

// --gpu (METAL=1 build of the standard binary): the shared full-grid deposit of DTFE_parallel
// (the PS binary partitions in Lagrangian space and never comes here; MY_SCALAR runs user C++ per
// sample, which no kernel can; the kernels are 3D)
#if defined(DTFE_GPU) && NO_DIM==3 && !defined(MY_SCALAR) && !defined(PHASE_SPACE)
#define DTFE_GPU_SHARED_ACTIVE
#include "CGAL_triangulation/gpu_host.h"
#endif




// Interpolates the fields, splitting the work across OpenMP threads when enabled.
void DTFE_parallel(vector<Particle_data> *allParticles,
                   vector<Sample_point> &samples,
                   User_options & userOptions,
                   Quantities *uQuantities,
                   Quantities *aQuantities);
#ifdef PHASE_SPACE
// the partition look-ahead's two halves (triangulation.cpp): the tessellation + vertex densities of a
// partition built ahead (an opaque handle: this file does not see CGAL), and its grid passes later
std::shared_ptr<void> psPrebuildTessellation(vector<Particle_data> *p, User_options &userOptions);
void psInterpolatePrebuilt(std::shared_ptr<void> &tessellation, vector<Sample_point> &samples,
                           User_options &userOptions, Quantities *uQuantities, Quantities *aQuantities);
#else
// the standard partition loop's look-ahead (triangulation.cpp): one thread sub-domain's tessellation +
// vertex densities built ahead (an opaque handle), and its interpolation later
std::shared_ptr<void> dtfePrebuildTessellation(vector<Particle_data> *p, User_options &userOptions);
void dtfeInterpolatePrebuilt(std::shared_ptr<void> &tessellation, vector<Sample_point> &samples,
                             User_options &userOptions, Quantities *uQuantities, Quantities *aQuantities);
#endif
// One standard partition's sub-domains as the look-ahead prepared them: DTFE_parallel's thread split of
// the partition (or its single tessellation) and every sub-domain's tessellation, built; DTFE_parallelImpl then only
// interpolates them. (Declared in both builds, used by the standard partition loop only.)
struct DTFEPrepared
{
    bool single = true;                                // DTFE_parallel's single-tessellation branch
    std::vector<size_t> pGrid;                         // else its thread grid, its split and the sub-domains' tessellations
    std::vector< std::vector<size_t> > subgridList;
    std::vector< Box > subgridCoords;
    std::vector< std::shared_ptr<void> > tessellation; // [thread number]: the sub-domain DTFE_parallel's thread i takes
    std::vector<size_t> nParticles;                    // [thread number]: its padded particles (the thread statistics)
    size_t noTotalParticles = 0;
};
static void DTFE_parallelImpl(vector<Particle_data> *allParticles, vector<Sample_point> &samples,
                              User_options &userOptions, Quantities *uQuantities, Quantities *aQuantities,
                              DTFEPrepared *prepared);

// Builds the velocity-space DTFE density g_i used by the approximate phase-space density (--approxPSD).
#if defined(VELOCITY) && defined(SCALAR) && !defined(PHASE_SPACE)
extern void computeVelocitySpaceDensity(vector<Particle_data> &particles,
                                         User_options &userOptions);
#endif


// A partition loop announces how many tessellations it builds at once and how many it has left
// (triangulation.cpp divides the cores of its parallel insertion by the smaller of the two);
// each partition counts itself off when it is done with its build and deposit.
extern std::atomic<int> dtfeTriangulationConcurrency;
extern std::atomic<int> dtfeTriangulationsRemaining;
struct ConcurrentTriangulationPlan
{
    ConcurrentTriangulationPlan(int const concurrency, int const total)
    { dtfeTriangulationConcurrency = concurrency; dtfeTriangulationsRemaining = total; }
    ~ConcurrentTriangulationPlan() { dtfeTriangulationConcurrency = 0; dtfeTriangulationsRemaining = 0; }
};
struct ConcurrentTriangulationGuard
{
    ~ConcurrentTriangulationGuard() { --dtfeTriangulationsRemaining; }
};

// Particle data plus the resolved per-run options produced by the common DTFE setup.
struct DTFE_State {
    vector<Particle_data> particles;
    User_options options;
};

#ifdef PHASE_SPACE
// Global Lagrangian bounding box (min/max of lagPos over all particles) with a tiny
// relative margin so boundary particles are strictly inside.
static void computeLagrangianBoundingBox(vector<Particle_data> const &particles,
                                         Box &lagBox)
{
    for (int d=0; d<NO_DIM; ++d)
    {
        lagBox[2*d]   = particles[0].lagPos[d];
        lagBox[2*d+1] = particles[0].lagPos[d];
    }
    for (size_t i=1; i<particles.size(); ++i)
        for (int d=0; d<NO_DIM; ++d)
        {
            if (particles[i].lagPos[d] < lagBox[2*d])   lagBox[2*d]   = particles[i].lagPos[d];
            if (particles[i].lagPos[d] > lagBox[2*d+1]) lagBox[2*d+1] = particles[i].lagPos[d];
        }
    // tiny margin to include boundary particles
    for (int d=0; d<NO_DIM; ++d)
    {
        Real eps = (lagBox[2*d+1] - lagBox[2*d]) * Real(1.e-6);
        lagBox[2*d]   -= eps;
        lagBox[2*d+1] += eps;
    }
}

// The log line of the periodic seam (subpartition.h psPeriodicSeam): the ownership period per axis
// and whether its copies are exact shifts -- the Lagrangian ones are by construction where 'exact'; the Eulerian
// ones are counted when the particles are in memory (an inexact one leaves a rounding-wide sliver at the seam).
static void psReportSeam(MESSAGE::Message &message, Real const seam[], bool const exact[], Real const eulerLen[],
                         Real const pad[], std::vector<Particle_data> const &particles)
{
    message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " periodic ownership period per axis:";
    for (int d=0; d<NO_DIM; ++d)
    {
        message << " [" << seam[d] - eulerLen[d] << ", " << seam[d] << ")";
        if ( exact[d] )
        {
            if ( not particles.empty() )
            {
                size_t const n = psCountInexactCopies( particles, d, double(seam[d]) - double(pad[d]) - 1.e-6 * double(eulerLen[d]), eulerLen[d] );
                if ( n ) message << " (" << n << " Eulerian copies round)";
            }
        }
        else
            message << " (the top particle plane: copies may round, the padding is too wide for an exact seam)";
    }
    message << ".\n" << MESSAGE::Flush;
}


// The Lagrangian partition geometry of a partitioned PS-DTFE run, shared by the batch partition
// loop and the composite '--serve' builder: both must select EXACTLY the same particles for
// partition pi (the server even reuses the batch runs' per-partition tessellation caches, whose
// identity is the partition's lagrangianRegion). The arithmetic is the batch loop's, verbatim.
struct PSPartitionPlan
{
    Box    lagBox;
    Real   lagLen[NO_DIM], eulerLen[NO_DIM], lagPadding[NO_DIM];
    size_t grid[NO_DIM];
    int    total = 1;
    bool   periodicCopies = false;
        // periodic: the partitions tile the period [seam - L, seam) per axis (subpartition.h psPeriodicSeam);
        // seamExact = every periodic copy an exact shift (else the fallback anchor lagBox_hi)
    Real   seam[NO_DIM];
    bool   seamExact[NO_DIM];

    void init(DTFE_State &state, Box &lagBoxGlobal, bool const hasLagrangianPeriodicCopies,
              MESSAGE::Message &message)
    {
        periodicCopies = hasLagrangianPeriodicCopies;
        total = 1;
        for (int d=0; d<NO_DIM; ++d)
        {
            grid[d] = state.options.partition[d];
            total *= int(grid[d]);
        }

        // reuse the Lagrangian bounding box computed before periodic copies, else recompute it here
        if (!lagBoxGlobal.isNullBox())
            lagBox = lagBoxGlobal;
        else
            computeLagrangianBoundingBox( state.particles, lagBox );
        message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " Lagrangian bounding box = "
                << MESSAGE::cMagenta() << lagBox.print() << MESSAGE::cReset() << "\n" << MESSAGE::Flush;

        for (int d=0; d<NO_DIM; ++d)
            lagLen[d] = lagBox[2*d+1] - lagBox[2*d];

        // Eulerian box length per axis: the period used to shift each partition's deferred copies
        for (int d=0; d<NO_DIM; ++d)
            eulerLen[d] = state.options.boxCoordinates[2*d+1] - state.options.boxCoordinates[2*d];

        // --ps-lag-padding <k> (default 8): k mean particle spacings, (V/N)^(1/NO_DIM) of
        // the Lagrangian domain (the period when periodic, the cloud's bounding box otherwise); N from the recorded
        // globals when the particles are not read yet (ps_globals_record.h)
        double lagSpacing = 0.;
        if ( state.options.psLagPadding > Real(0.) )
        {
            double const nPart = state.particles.empty() ? double(state.options.psPresetN) : double(state.particles.size());
            double vol = 1.;
            for (int d=0; d<NO_DIM; ++d) vol *= double( periodicCopies ? eulerLen[d] : lagLen[d] );
            if ( nPart > 0. and vol > 0. ) lagSpacing = std::pow( vol / nPart, 1. / double(NO_DIM) );
        }
        // Lagrangian padding per axis: k spacings with no floor (--ps-lag-padding, the default) -- or, with
        // '--ps-lag-padding 0' (or no particle count to take a spacing from), the Eulerian fraction floored at 0.30
        // of a partition cell
        for (int d=0; d<NO_DIM; ++d)
        {
            Real padFrac = (state.options.paddingLength[2*d] + state.options.paddingLength[2*d+1]) / (Real(2.) * eulerLen[d]);
            lagPadding[d] = padFrac * lagLen[d];
            if (lagPadding[d] < lagLen[d] / state.options.partition[d] * Real(0.3))
                lagPadding[d] = lagLen[d] / state.options.partition[d] * Real(0.3);
            if ( lagSpacing > 0. )
            {
                lagPadding[d] = Real( double(state.options.psLagPadding) * lagSpacing );
                // periodic: the per-partition copies come from the +-1 images only, so more than one period of
                // padding would ask for particles no copy provides; one period already holds every neighbour (a
                // set with < 8 particles per axis -- tests only)
                if ( periodicCopies and lagPadding[d] > eulerLen[d] ) lagPadding[d] = eulerLen[d];
            }
            // --ps-alpha-shape (non-periodic): a kept tetrahedron (circumradius <= a) owned by this
            // partition has its empty circumsphere within 2a of its centroid, so with >= 2a of
            // padding it is in the partition's triangulation exactly when it is in the global one;
            // --ps-vertex-mass also counts the kept tets around each of its vertices, which reach 4a
            if ( state.options.psLagAlphaRadius > 0. )
            {
                Real const alphaPad = Real( 4.2 * state.options.psLagAlphaRadius );
                if (lagPadding[d] < alphaPad) lagPadding[d] = alphaPad;
            }
        }
        message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " Lagrangian partition padding = "
                << MESSAGE::cMagenta() << MESSAGE::printElements( lagPadding, NO_DIM, ", " )
                << MESSAGE::cReset();
        // name the rule in the log: a default of 8 with no particle count to take a spacing from falls back to the
        // Eulerian rule, and the log must not look like k spacings then
        if ( lagSpacing > 0. ) message << " (--ps-lag-padding " << state.options.psLagPadding << " mean spacings of " << lagSpacing << ")";
        else message << " (the Eulerian --padding rule" << ( state.options.psLagPadding > Real(0.) ? ": no particle count for a spacing)" : ", --ps-lag-padding 0)" );
        message << "\n" << MESSAGE::Flush;

        // periodic: the period the partitions tile, per axis -- [seam - L, seam) with every periodic copy an
        // exact shift where the padding allows it (subpartition.h psPeriodicSeam). A function of
        // the bounding box, the period and the padding only, so the composite server and a --ps-window run
        // started from the recorded globals tile the same way as the batch run whose caches they reuse
        for (int d=0; d<NO_DIM; ++d)
        {
            seam[d] = lagBox[2*d+1];
            seamExact[d] = periodicCopies and psPeriodicSeam( lagBox[2*d], lagBox[2*d+1], eulerLen[d], lagPadding[d], seam[d] );
        }
        if ( periodicCopies )
            psReportSeam( message, seam, seamExact, eulerLen, lagPadding, state.particles );
    }

    // partition pi's unpadded Lagrangian region (its ownership box) and the padded one that
    // selects its particles
    void regions(int const pi, Box &lagRegion, Box &lagPadded)      // read-only (Box has no const operator[])
    {
        // unflatten the linear partition index pi into per-axis partition coordinates
        int idx[NO_DIM];
        {
            int rem = pi;
            for (int d=NO_DIM-1; d>=0; --d)
            {
                idx[d] = rem % grid[d];
                rem /= grid[d];
            }
        }

        for (int d=0; d<NO_DIM; ++d)
        {
            // periodic, exact seam: equal slices of the period [seam - L, seam), whose copies are all
            // exact shifts (subpartition.h psPeriodicSeam); the first and last edges are the period's own bits
            if ( periodicCopies and seamExact[d] )
            {
                Real const base = seam[d] - eulerLen[d];   // exact (psPeriodicSeam)
                lagRegion[2*d]   = idx[d] == 0 ? base : base + eulerLen[d] * idx[d] / grid[d];
                lagRegion[2*d+1] = idx[d] + 1 == int(grid[d]) ? seam[d] : base + eulerLen[d] * (idx[d]+1) / grid[d];
                continue;
            }
            lagRegion[2*d]   = lagBox[2*d] + lagLen[d] * idx[d] / grid[d];
            lagRegion[2*d+1] = lagBox[2*d] + lagLen[d] * (idx[d]+1) / grid[d];
            // periodic: the partition regions must tile one full PERIOD, not just the particle bounding box (image
            // tets can have Lagrangian centroids in the gap between lagBox and the periodic box, owned by NO partition:
            // their mass would be dropped). So the first lower edge sits one period below the last upper edge, which
            // is lagBox_hi itself (lagBox_lo + lagLen * grid / grid rounds, and an upper edge an ulp off lagBox_hi
            // leaves a slab of the period owned twice or never).
            if (periodicCopies && idx[d] == 0)
                lagRegion[2*d] = lagBox[2*d+1] - eulerLen[d];
            if (periodicCopies && idx[d] + 1 == int(grid[d]))
                lagRegion[2*d+1] = lagBox[2*d+1];
        }

        lagPadded = lagRegion;
        for (int d=0; d<NO_DIM; ++d)
        {
            lagPadded[2*d]   -= lagPadding[d];
            lagPadded[2*d+1] += lagPadding[d];
            // clamp to the overall Lagrangian box only when no periodic copies exist
            if (!periodicCopies)
            {
                if (lagPadded[2*d]   < lagBox[2*d])   lagPadded[2*d]   = lagBox[2*d];
                if (lagPadded[2*d+1] > lagBox[2*d+1]) lagPadded[2*d+1] = lagBox[2*d+1];
            }
        }
    }

    // the partition index along axis d of an original Lagrangian coordinate x: the slice of regions() it falls in
    // (periodic, exact seam: x wrapped into the period first -- an original above the seam is owned through its -L
    // copy). For psPartitionCosts' ETA proxy only; ownership itself is psFilterCell's.
    int axisIndex(int const d, Real const x) const
    {
        int i;
        if ( periodicCopies and seamExact[d] )
        {
            double const L = double(eulerLen[d]), base = double(seam[d]) - L;
            double f = ( double(x) - base ) / L;
            f -= std::floor( f );
            i = int( std::floor( f * double(grid[d]) ) );
        }
        else
        {
            Real const f = ( x - lagBox.coords[2*d] ) / lagLen[d];   // regions()' anchor-at-lagBox arithmetic
            i = int( std::floor( f * Real(grid[d]) ) );
        }
        return std::max( 0, std::min( int(grid[d]) - 1, i ) );
    }
};

/* Each partition's predicted COST, for the partition loop's ETA. Partitions hold about the same
   number of particles (a Lagrangian split of a lattice), so the particle count says nothing; what makes one
   partition 30x dearer than another is the collapsed structure in it -- folded tetrahedra inside a halo span
   the halo in Eulerian space, so the deposit visits ~(halo size)^3 cells for each of them. The proxy: each
   particle weighs the number of particles in its cell of a coarse Eulerian count grid with ~1.6 Mpc cells
   (the size of the largest haloes; 8..256 cells per axis), i.e. the sum of the local density over the
   partition's particles -- a pair count that grows with the mass in dense regions. Checked against a TNG50-3-
   Dark z=0 run (125 partitions, 69 s to 2091 s each): log-correlation 0.86 with the measured partition times,
   median |log2 error| of the ETA 0.25 against the partition count's 0.85; finer cells (0.8 Mpc: 0.71) and a
   squared weight per particle (unstable: up to 16x off) do worse. ONE run's evidence. One O(N) pass (~0.5 s
   for 94M particles). Uniform weights when the particles are not in memory. */
static std::vector<double> psPartitionCosts(std::vector<Particle_data> const &particles, PSPartitionPlan &plan,
                                            User_options const &options)
{
    std::vector<double> cost( size_t(plan.total), 1. );
    if ( particles.empty() or plan.total <= 1 ) return cost;
    Real lo[NO_DIM], len[NO_DIM];
    for (int d=0; d<NO_DIM; ++d)
    {
        lo[d]  = options.boxCoordinates.coords[2*d];
        len[d] = options.boxCoordinates.coords[2*d+1] - options.boxCoordinates.coords[2*d];
        if ( not ( len[d] > Real(0.) ) ) return cost;
    }
    size_t nc[NO_DIM];
    size_t cells = 1;
    for (int d=0; d<NO_DIM; ++d)
    {
        nc[d] = size_t( std::max( 8., std::min( 256., std::round( double(len[d]) / 1.6 ) ) ) );   // ~1.6 Mpc cells
        cells *= nc[d];
    }
    auto cellOf = [&](Particle_data const &p) -> size_t
    {
        size_t c = 0;
        for (int d=0; d<NO_DIM; ++d)
        {
            Real x = std::fmod( p.pos[d] - lo[d], len[d] );   // the unwrapped Eulerian position, wrapped
            if ( x < Real(0.) ) x += len[d];
            size_t i = size_t( x / len[d] * Real(nc[d]) );
            if ( i >= nc[d] ) i = nc[d] - 1;
            c = c * nc[d] + i;
        }
        return c;
    };
    std::vector<uint32_t> count( cells, 0 );
    for (auto const &p : particles) ++count[ cellOf( p ) ];
    std::fill( cost.begin(), cost.end(), 0. );
    for (auto const &p : particles)
    {
        int pi = 0;
        for (int d=0; d<NO_DIM; ++d)
        {
            int const i = plan.axisIndex( d, p.lagPos[d] );   // the slice regions() puts it in
            pi = pi * int(plan.grid[d]) + i;
        }
        cost[size_t(pi)] += double( count[ cellOf( p ) ] );
    }
    for (double &c : cost) c = std::max( c, 1. );   // an empty partition still takes a moment
    return cost;
}
#endif


// Common DTFE setup: optional random/subsampled particles, consistency checks, velocity-derivative
// selection, and region/partition extraction. Mutates userOptions (averageDensity, region).
DTFE_State DTFE_setup(vector<Particle_data> *allParticles,
                      vector<Sample_point> &samples,
                      User_options &userOptions)
{
    DTFE_State state;

    if ( userOptions.poisson!=0 )
        randomParticles( allParticles, &userOptions );

    // optionally replace the data with a random subsample of it
    vector<Particle_data> *particlePointer = allParticles;
    vector<Particle_data> particlesRandomSubsample;
    if (userOptions.randomSample>=Real(0.) )
    {
        randomSample( *particlePointer, &particlesRandomSubsample, userOptions );
        particlePointer->clear();
        particlePointer = &particlesRandomSubsample;
    }

    // run consistency checks and finalize derived options; default the density normalization to the box average
    // (a run that started without its particles takes their count and mean density from the globals record)
    size_t const nParticles = userOptions.psParticlesDeferred ? userOptions.psPresetN : particlePointer->size();
    userOptions.updateEntries( nParticles, not samples.empty() );
    if ( userOptions.averageDensity<0. )
    {
        userOptions.averageDensity = averageDensity( *particlePointer, userOptions );
        userOptions.psAverageDensityFromParticles = true;
    }

    // --scratch-dir: arm the out-of-core allocator BEFORE any full-grid allocation and before
    // auto-tune (which excludes the disk-backed grid term from its RAM model when armed). The
    // path itself was validated at option parsing; the arming can still fail in the library
    // build, whose scratch_alloc STUB never hijacks the host program's allocator.
    if ( not userOptions.scratchDir.empty() )
    {
        double thresholdGB = 1.0;
        if ( const char *env = std::getenv("DTFE_SCRATCH_MIN_GB") )
        {
            double v = std::atof(env);
            if ( v > 0. ) thresholdGB = v;
        }
        if ( ScratchAlloc::scratchArm( userOptions.scratchDir.c_str(), size_t(thresholdGB*1.e9) ) )
        {
            MESSAGE::Message message( userOptions.verboseLevel );
            message << MESSAGE::cBold() << "SCRATCH:" << MESSAGE::cReset() << " allocations >= "
                    << thresholdGB << " GB (the full-grid accumulators) are backed by mmap'ed files in '"
                    << userOptions.scratchDir << "' -- unlinked at creation, so they free themselves on any exit.\n"
                    << MESSAGE::Flush;
        }
        else
        {
            MESSAGE::Warning warning( userOptions.verboseLevel );
            warning << "'--scratch-dir' has no effect in the library build (libDTFE must not replace the host "
                    << "program's global allocator); the full grids stay in RAM.\n" << MESSAGE::EndWarning;
            userOptions.scratchDir.clear();   // keep auto-tune's RAM model honest
        }
    }

    // --ps-window: the auto-tuner sizes the grids by the cells actually allocated -- the window's --
    // so its cell range must be known first (the header's box is final here; DTFE() computes it again,
    // identically). Without this it would see the whole virtual grid of a zoom (2759x3197x512 = 524 GB),
    // call the run impossible and fall back to one partition at a time.
    if ( userOptions.psWindowOn )
        userOptions.computePsWindow();

    // '--resume' without '--partition': the split is the checkpoint's. The auto-tuner would pick it
    // from the free memory of THIS moment, which need not be the killed run's choice, and a different split
    // cannot continue the merged partitions; with the split given it still tunes '--max-concurrent'.
    // Both binaries (run_dtfe.sh, like run_ps_dtfe.sh, leaves the split to the auto-tuner).
    if ( userOptions.psResume and not userOptions.partitionGiven and userOptions.partNo < 0 )
    {
        PartitionCheckpoint::Manifest m;
        std::string why;
        std::string const dir = PartitionCheckpoint::directoryFor( userOptions );
        if ( not PartitionCheckpoint::readManifest( dir, m, why ) )
            throwError( "'--resume': nothing to resume -- ", why, ". Run without '--resume' to start the run." );
        if ( m.partition.size() != size_t(NO_DIM) )
            throwError( "'--resume': the checkpoint in '", dir, "' names no partition split of ", NO_DIM, " numbers." );
        userOptions.partition = m.partition;
        size_t n = 1;
        for ( size_t p : m.partition ) n *= p;
        userOptions.partitionOn = n > 1;
        userOptions.partitionGiven = true;
        MESSAGE::Message message( userOptions.verboseLevel );
        message << MESSAGE::cBold() << PartitionCheckpoint::NAME << ":" << MESSAGE::cReset() << " --resume: the partition split ["
                << MESSAGE::printElements( m.partition, "," ) << "] is the checkpoint's.\n" << MESSAGE::Flush;
    }

    /* '--region' / '--regionMpc' (standard DTFE; PS-DTFE rejects them at option parsing): the run triangulates only
       the region plus its padding, so the auto-tuner must plan THAT, not the whole snapshot: planned from the
       snapshot's particle count and box, a launcher exact zoom of a 20^3-cell TNG50-3-Dark region would be split
       3x3x3 with 10 threads each -- 27 x 9 sub-triangulations, each padded again around its tiny sub-box -- about
       21 min, where one domain takes about a minute. So the region is cropped HERE, before the tuner, and the
       tuner is handed a copy of the options whose box is the region, with the particles inside the UNPADDED
       region as its count. That is the same pairing as a whole-box run (the box's particles; the model adds the
       padding on top as a fraction of the box), so one domain predicts one padded region. Only the split and the
       concurrency it chooses are copied back: nothing else of the standard branch (or of autoTuneServe) is read
       later. */
    vector<Particle_data> particlesRegion;   // the padded region's particles, for the crop below
    bool regionCropped = false;
    User_options tuneOptions;                // the options the tuner plans a region run with
    size_t nTune = nParticles;
#ifndef PHASE_SPACE
    if ( userOptions.regionOn )
    {
        tuneOptions = userOptions;
        tuneOptions.region.validSubBox( tuneOptions.boxCoordinates, tuneOptions.periodic );
        tuneOptions.paddedBox = tuneOptions.region;
        tuneOptions.paddedBox.addPadding( tuneOptions.paddingLength );
        findParticlesInBox( *particlePointer, &particlesRegion, tuneOptions );   // periodic images as the crop takes them
        regionCropped = true;
        // the selected copies sit in the padded box's frame (images shifted in), so the unpadded region is a
        // plain box test here, periodic or not
        Box const &r = tuneOptions.region;
        nTune = size_t( std::count_if( particlesRegion.begin(), particlesRegion.end(),
                                       [&r](Particle_data &p) { return r.isParticleInBox( p ); } ) );
        tuneOptions.updateFullBox( tuneOptions.region );   // the box the model takes its padding fraction from
        tuneOptions.periodic = false;                       // the cropped run is a standalone non-periodic box
    }
#endif

    // auto-select --partition / --max-concurrent from the data and machine when not user-given
    {
        bool gpuActive = false;
#if defined(PS_GPU)
        gpuActive = userOptions.psUseMetal;
#elif defined(DTFE_GPU)
        gpuActive = userOptions.useMetal;
#endif
        // the particles the run triangulates: a region run's padded crop (in the region's frame, periodic images
        // shifted in), whose counts the standard tuner replays its split on (auto_tune.h StdTuneDensity)
        AutoTuneReport const report = autoTunePartitioning( regionCropped ? tuneOptions : userOptions, nTune,
                                                            not samples.empty(), gpuActive,
                                                            regionCropped ? &particlesRegion : particlePointer );
        if ( regionCropped )
        {
            userOptions.partition      = tuneOptions.partition;
            userOptions.partitionOn    = tuneOptions.partitionOn;
            userOptions.maxConcurrent  = tuneOptions.maxConcurrent;
#ifndef PHASE_SPACE
            userOptions.stdLookAheadOffBudget = tuneOptions.stdLookAheadOffBudget;
            userOptions.stdLookAheadCounted   = tuneOptions.stdLookAheadCounted;
#endif
        }
        if ( report.valid ) runProgressSetPredicted( report.predicted );   // the SUMMARY's predicted-vs-measured line
#ifndef PHASE_SPACE
        // what the budget leaves for the partition loop's look-ahead beside the busiest partition (as PS's)
        if ( report.valid )
            userOptions.stdLookAheadRoom = report.budget - report.fixed - report.gpuSnapshot - report.perPartition;
#endif
#ifdef PHASE_SPACE
        if ( report.valid )
        {
            // the partition memory gate (auto_tune.h). It compares ONE partition's need with the free memory, so
            // the GPU deposit's host snapshot -- one at a time, allocated during a deposit, not yet in the free
            // memory a partition sees when it starts -- is in each check once
            userOptions.psPartitionBytes = report.perPartition + report.gpuSnapshot;
            // what the budget leaves for the look-ahead beside one partition (checked against the free memory
            // only, a budget set low on purpose would be passed by one tessellation)
            userOptions.psLookAheadRoom = report.budget - report.fixed - report.gpuSnapshot - report.perPartition;
            // what the budget leaves beside the predicted peak for partitions parked by the staged ordered merge:
            // the prediction does not count them, and on macOS the free-memory reading alone does not stop them (the
            // compressor and swap keep 'free' above the spill threshold: TNG100 z=0 could park 74 GB in RAM on a
            // 64 GB machine). DTFE_TEST_PARK_ROOM_GB sets it (tests).
            userOptions.psParkRoom = report.budget - report.predicted;
        }
        if ( char const *room = std::getenv( "DTFE_TEST_PARK_ROOM_GB" ) )
            if ( room[0] != '\0' ) userOptions.psParkRoom = std::max( 0., std::atof( room ) ) * 1.e9;
#endif
        if ( userOptions.autoTuneReport )   // the memory check: the decision and prediction, nothing else
        {
            std::printf( "%s\n", report.line().c_str() );
            std::fflush( stdout );
            std::exit( 0 );
        }
    }

    // only the partition loop checkpoints, so only a partitioned run has anything to resume (both binaries)
    if ( userOptions.psResume and not ( userOptions.partitionOn and userOptions.partNo < 0 ) )
        throwError( "'--resume' continues a partitioned run ('--partition' with more than one partition); this run has "
                    "one triangulation. Run it without '--resume'." );

    // velocity derivatives are computed from the gradient afterwards, so request the gradient now
    // and defer the derived quantities to the post-processing step
    state.options = userOptions;
    if ( userOptions.uField.selectedVelocityDerivatives() )
    {
        state.options.uField.velocity_gradient = true;
        state.options.uField.deselectVelocityDerivatives();
    }
    if ( userOptions.aField.selectedVelocityDerivatives() )
    {
        state.options.aField.velocity_gradient = true;
        state.options.aField.deselectVelocityDerivatives();
    }

    // the T-web is computed from the density grid (tidal tensor): make sure density is interpolated.
    // Drop the tweb flag itself from the internal copy -- the interpolation never fills the T-web
    // grids, so keeping it only makes reserveMemory pre-allocate them (16 B/cell) during the
    // partition loop; computeTidalWebClassification() re-creates them from the summed density.
    if ( userOptions.uField.velocity_tweb ) { state.options.uField.density = true; state.options.uField.velocity_tweb = false; }
    if ( userOptions.aField.velocity_tweb ) { state.options.aField.density = true; state.options.aField.velocity_tweb = false; }

    // restrict the data to the user-defined region (plus padding) and treat it as a standalone non-periodic box
    if ( userOptions.regionOn )
    {
        state.options.region.validSubBox( state.options.boxCoordinates, state.options.periodic );
        state.options.paddedBox = state.options.region;
        state.options.paddedBox.addPadding( state.options.paddingLength );

        if ( not regionCropped )   // (the standard build selected them before the auto-tuner, above)
            findParticlesInBox( *particlePointer, &particlesRegion, state.options );
        state.options.periodic = false;
        state.options.updateFullBox( state.options.region );

        // FREE the snapshot's particles, not just clear() them: clear() keeps the whole array's capacity for the
        // rest of the run -- 5.7 GB of a 1.57e8-particle snapshot (36 B each) under a triangulation of a few
        // million, which the auto-tuner (planning the region) does not count. Callers get the vector back
        // empty either way ('Clears allParticles'); main's '--interlace' keeps its own copy for the second pass.
        std::vector<Particle_data>().swap( *particlePointer );
        state.particles = std::move(particlesRegion);
        particlePointer = &state.particles;
    }

    // single-partition mode (--partNo): keep only this partition's particles and grid sub-box
    if ( userOptions.partitionOn and userOptions.partNo>=0 )
    {
        MESSAGE::Message message( userOptions.verboseLevel );
        message << "The program will interpolate the fields in partition number " << userOptions.partNo << " of partition grid [" << MESSAGE::printElements( userOptions.partition, "," ) << "].\n" << MESSAGE::Flush;

        std::vector< std::vector<size_t> > subgridList;
        std::vector< Box > subgridCoords;
        optimalPartitionSplit( *particlePointer, state.options, state.options.partition, &subgridList, &subgridCoords );
        copySubgridInformation( &state.options, subgridList, subgridCoords );
        userOptions.region = state.options.region;

        state.options.paddedBox = state.options.region;
        state.options.paddedBox.addPadding( state.options.paddingLength );

        vector<Particle_data> particlesPartition;
        findParticlesInBox( *particlePointer, &particlesPartition, state.options );
        state.options.periodic = false;

        particlePointer->clear();
        state.particles = std::move(particlesPartition);
        particlePointer = &state.particles;

        subgrid( userOptions, subgridList );
    }

    // ensure state.particles owns the data when no region/partition move already populated it
    if ( particlePointer != &state.particles )
        state.particles = std::move(*particlePointer);

    return state;
}


// Sets up the box, padded box, and (for standard DTFE) periodic copies for the whole-box, non-partition case.
void DTFE_preparePadding(DTFE_State &state, User_options const &userOptions)
{
    if ( not userOptions.regionOn and not userOptions.partitionOn )
    {
        state.options.region = state.options.boxCoordinates;
        state.options.paddedBox = state.options.region;
        state.options.paddedBox.addPadding( state.options.paddingLength );

#ifndef PHASE_SPACE
        // standard DTFE: add periodic copies based on Eulerian positions
        if ( userOptions.periodic )
        {
            vector<Particle_data> tempPart;
            findParticlesInBox( state.particles, &tempPart, state.options );
            state.particles = std::move(tempPart);
        }
#endif
        // PS-DTFE Lagrangian periodic padding is handled earlier in DTFE()
    }
#ifdef PHASE_SPACE
    // PS-DTFE keeps periodic=true only for grid-index wrapping; data periodicity comes
    // from the Lagrangian copies and Eulerian unwrapping
#else
    state.options.periodic = false;
#endif
}


// Standard binary, PARTITIONED point evaluation: the Eulerian box whose tetrahedra partition i of a
// split owns -- 'range' its grid-index range (optimalPartitionSplit's subgridList[i]) in the split's 'grid', and
// 'region' its sub-box. A face is on the run's outer boundary when its INDEX is 0 or the grid's end (the float cut
// region[0] + dx*n can miss the box face by an ulp); psPointEvalOwnBox resolves those faces (ps_point_eval.cc).
// Shared by the composite '--serve' and the batch '--sample-points --partition' loop, so both own alike. (Compiled
// in PS-DTFE too, whose build keeps the standard partition branch as unreachable code.)
static void standardPartitionOwnBox(std::vector<size_t> const &range, std::vector<size_t> const &grid,
                                    Box const &region, double ownLo[], double ownHi[])
{
    double lo[NO_DIM], hi[NO_DIM];
    bool outerLo[NO_DIM], outerHi[NO_DIM];
    for (int d=0; d<NO_DIM; ++d)
    {
        lo[d] = double( region.coords[2*d] );
        hi[d] = double( region.coords[2*d+1] );
        outerLo[d] = ( range[2*d] == 0 );
        outerHi[d] = ( range[2*d+1] >= grid[d] );
    }
    psPointEvalOwnBox( lo, hi, outerLo, outerHi, ownLo, ownHi );
}


/* DTFE_parallel's thread split, in pieces the standard partition loop's look-ahead shares: it prepares a
   partition's sub-domains EXACTLY as DTFE_parallel would (the same threads, the same split, the same per-thread
   options and particles), so these are the one definition of each. */
// the threads DTFE_parallel may split one call over: every OpenMP thread, at most '--max-concurrent'
static int dtfeParallelThreads(User_options const &userOptions)
{
#ifdef OPEN_MP
    int n = omp_get_max_threads();
    if ( userOptions.maxConcurrent > 0 and userOptions.maxConcurrent < n )
        n = userOptions.maxConcurrent;
    return n;
#else
    (void)userOptions;
    return 1;
#endif
}
// ... and whether it runs ONE tessellation instead: 1 thread, user sampling, redshift cone, --sample-points point
// evaluation (the standard worker evaluates ONE Eulerian tessellation; overlapping padded sub-triangulations would
// double-count boundary tets), or PS-DTFE (which partitions in Lagrangian space)
static bool dtfeParallelSingle(User_options const &userOptions, vector<Sample_point> const &samples, int const threads)
{
#ifdef PHASE_SPACE
    (void)userOptions; (void)samples; (void)threads;
    return true;   // PS-DTFE: bypass spatial partitioning, use single triangulation
#else
    return threads==1 or not samples.empty() or userOptions.redshiftConeOn or psPointEvalActive();
#endif
}
// one thread's sub-domain of the split: its options (sub-grid, region, padded box), 'userOptions' being the call's
static User_options dtfeSubdomainOptions(User_options const &userOptions, int const threadNo, int const noProcessors,
                                         std::vector<size_t> const &pGrid,
                                         std::vector< std::vector<size_t> > &subgridList, std::vector< Box > &subgridCoords)
{
    User_options tempOptions = userOptions;
    tempOptions.noProcessors = noProcessors;
    tempOptions.threadId = threadNo;
    tempOptions.verboseLevel = (userOptions.verboseLevel>0) ? 1 : userOptions.verboseLevel;   // slaves show only errors/warnings; master shows all

    if ( threadNo==0 )
        tempOptions.verboseLevel = userOptions.verboseLevel;
    tempOptions.partNo = threadNo;
    tempOptions.partition.clear();
    for (int i=0; i<NO_DIM; ++i)
        tempOptions.partition.push_back( pGrid[i] );
    // the call's region becomes the sub-domain's full box: for a partition's sub-domain that is the PARTITION's region,
    // which so names the partition in the tessellation-cache key (its 'box=' line, tessellation_cache.h makeDescriptor)
    Box parentRegion = userOptions.region;   // (updateFullBox takes a non-const box)
    tempOptions.updateFullBox( parentRegion );


    // region allocated to this thread: writes subgrid size to tempOptions.gridSize and box to tempOptions.region
    copySubgridInformation( &tempOptions, subgridList, subgridCoords );
    tempOptions.paddedBox = tempOptions.region;
    tempOptions.paddedBox.addPadding( tempOptions.paddingLength );
    return tempOptions;
}

#ifndef PHASE_SPACE
/* The standard look-ahead's PREPARE stage: partition k+1's sub-domains, on the look-ahead thread while
   partition k deposits. The thread count is the CALLER's (dtfeParallelThreads on the loop's thread: a fresh thread's
   OpenMP defaults need not be the loop's), the rest is DTFE_parallel's own code above: the single-tessellation choice,
   parallelGrid, optimalPartitionSplit on the partition's particles, each sub-domain's options and particles, and
   buildTessellation (dtfePrebuildTessellation) -- with the cache when '--tessellation-cache' is on. The sub-domains are
   built on a team of their own (an OpenMP team started from this thread), as many at once as DTFE_parallel runs, and
   every one silently ('verbose 0': the log's progress line reports the look-ahead). Frees the partition's particles. */
static void dtfePrepareSubdomains(vector<Particle_data> *partParticles, vector<Sample_point> &samples,
                                  User_options const &partOptions, int const threads, DTFEPrepared &P)
{
    User_options quiet = partOptions;
    quiet.verboseLevel = 0;
    P.noTotalParticles = partParticles->size();
    P.single = dtfeParallelSingle( quiet, samples, threads );
    if ( P.single )
    {
        P.nParticles.assign( 1, partParticles->size() );
        P.tessellation.assign( 1, std::shared_ptr<void>() );
        P.tessellation[0] = dtfePrebuildTessellation( partParticles, quiet );   // frees the particles
        return;
    }
    P.pGrid.assign( NO_DIM, 0 );
    parallelGrid( threads, quiet, &(P.pGrid[0]) );
    int noProcessors = 1;
    for (int d=0; d<NO_DIM; ++d) noProcessors *= int( P.pGrid[d] );
    optimalPartitionSplit( *partParticles, quiet, P.pGrid, &P.subgridList, &P.subgridCoords );
    P.tessellation.assign( size_t(noProcessors), std::shared_ptr<void>() );
    P.nParticles.assign( size_t(noProcessors), 0 );
    std::vector<std::exception_ptr> errors( static_cast<size_t>( noProcessors ), std::exception_ptr() );
    // over the sub-domain numbers, not the thread numbers: each is built whatever team the runtime gives
#ifdef OPEN_MP
    #pragma omp parallel for schedule(dynamic, 1) num_threads(noProcessors)
#endif
    for (int t=0; t<noProcessors; ++t)
    {
        try
        {
            User_options o = dtfeSubdomainOptions( quiet, t, noProcessors, P.pGrid, P.subgridList, P.subgridCoords );
            vector<Particle_data> particles;
            findParticlesInBox( *partParticles, &particles, o );
            P.nParticles[size_t(t)] = particles.size();
            P.tessellation[size_t(t)] = dtfePrebuildTessellation( &particles, o );
        }
        catch ( ... ) { errors[size_t(t)] = std::current_exception(); }
    }
    vector<Particle_data>().swap( *partParticles );
    for (std::exception_ptr const &e : errors)
        if ( e ) std::rethrow_exception( e );
}
#endif


#ifdef PHASE_SPACE
// PS-DTFE, periodic: unwrap the Eulerian positions so each displacement s = pos - lagPos lies in
// [-L/2, L/2]; otherwise boundary particles get displacements near L and huge spurious simplices.
static void psUnwrapEulerian(std::vector<Particle_data> &particles, User_options &options)
{
    Real eulerLen[NO_DIM];
    for (int d=0; d<NO_DIM; ++d)
        eulerLen[d] = options.boxCoordinates[2*d+1] - options.boxCoordinates[2*d];
    for (size_t i = 0; i < particles.size(); ++i)
        for (int d = 0; d < NO_DIM; ++d)
        {
            Real s = particles[i].pos[d] - particles[i].lagPos[d];
            s = std::fmod(s + eulerLen[d] * Real(0.5), eulerLen[d]);
            if (s < Real(0.)) s += eulerLen[d];
            s -= eulerLen[d] * Real(0.5);
            particles[i].pos[d] = particles[i].lagPos[d] + s;
        }
}
#endif

// The host's reaction to shares the non-finite guard of the exact fixed-point sums dropped (ps_deposit_params.h
// psFixedShareOk: a NaN, an infinity or a product beyond 2^63 adds nothing and is counted, on the CPU and in every
// GPU kernel). THE one place that decides what such a run does: a warning that names each grid and its count,
// once per DTFE() call (the counts are taken and reset here), and the run goes on -- the grids are written,
// without those shares (a float sum would turn their cells NaN). To fail the run instead, throw here
// (throwError). Nothing is printed when nothing was dropped.
static void fixedShareDropsReact(User_options const &userOptions)
{
#ifdef PHASE_SPACE
    // the slots: PSFxGrid, then FX_DROP_MASS / FX_DROP_SUMW (ps_deposit_params.h)
    static char const *const names[FX_DROP_SLOTS] = {
        "velocity moment", "dispersion second moment", "velocity-gradient moment", "moment weight",
        "exact multiplicity", "dispersion mean", "dispersion weight", "scalar moment", "scalar-gradient moment",
        "released multiplicity", "mass (density)", "exact-deposit weight totals", "", "", "", "" };
#else
    // the standard GPU deposit's grids (dtfe_gpu_fixed.h batchFixedScales)
    static char const *const names[FX_DROP_SLOTS] = {
        "density", "velocity", "velocity gradient", "scalar", "scalar gradient", "", "", "", "", "", "", "", "", "", "", "" };
#endif
    unsigned long long total = 0;
    std::string what;
    for (int k = 0; k < FX_DROP_SLOTS; ++k)
    {
        unsigned long long const n = fixedShareDrops[k].exchange( 0, std::memory_order_relaxed );
        if ( n == 0 ) continue;
        total += n;
        what += ( what.empty() ? "" : ", " ) + std::string( names[k][0] ? names[k] : "slot " + std::to_string( k ) )
              + " grid " + std::to_string( n );
    }
    if ( total == 0 ) return;
    MESSAGE::Warning warning( userOptions.verboseLevel );
    warning << total << " share(s) of the exact fixed-point sums were not finite or beyond the 64-bit range and were "
            << "left out (" << what << "). The grids are written without them; a non-finite input value (a NaN or an "
            << "infinite position, velocity or mass) is the usual cause.\n" << MESSAGE::EndWarning;
}

#ifdef PHASE_SPACE
// PS-DTFE, neither periodic nor --region: the cloud's Lagrangian mean density and the --ps-alpha-shape radius.
static void psCloudNormalization(DTFE_State &state, User_options &userOptions)
{
    if ( not userOptions.periodic and not userOptions.regionOn )
    {
        // non-periodic isolated cloud: normalize to the cloud Lagrangian mean N*m/V_lagBox
        // (box mean N*m/L^3 undercounts a cloud filling part of the box; --region keeps the box mean)
        if ( not state.particles.empty() )
        {
            Box lagBoxNP;
            for (int d = 0; d < NO_DIM; ++d)
            { lagBoxNP[2*d] = state.particles[0].lagPos[d]; lagBoxNP[2*d+1] = state.particles[0].lagPos[d]; }
            double totalMass = 0.;
            for (size_t i = 0; i < state.particles.size(); ++i)
            {
                totalMass += double(state.particles[i].weight());
                for (int d = 0; d < NO_DIM; ++d)
                {
                    Real q = state.particles[i].lagPos[d];
                    if (q < lagBoxNP[2*d])   lagBoxNP[2*d]   = q;
                    if (q > lagBoxNP[2*d+1]) lagBoxNP[2*d+1] = q;
                }
            }
            double vLag = 1.;
            for (int d = 0; d < NO_DIM; ++d) vLag *= double(lagBoxNP[2*d+1] - lagBoxNP[2*d]);
            if ( vLag > 0. )
            {
                state.options.averageDensity = Real( totalMass / vLag );
                MESSAGE::Message msg( userOptions.verboseLevel );
                msg << MESSAGE::cBold() << "PS-DTFE (non-periodic):" << MESSAGE::cReset()
                    << " density normalized to the Lagrangian cloud density N*m/V_lagBox = "
                    << MESSAGE::cMagenta() << state.options.averageDensity << MESSAGE::cReset() << ".\n" << MESSAGE::Flush;
                // --ps-alpha-shape: the Lagrangian domain is the cloud's alpha shape (psFilterCell)
                if ( state.options.psAlphaShape > Real(0.) )
                {
                    double const spacing = std::cbrt( vLag / double(state.particles.size()) );
                    state.options.psLagAlphaRadius = double(state.options.psAlphaShape) * spacing;
                    msg << MESSAGE::cBold() << "PS-DTFE (non-periodic):" << MESSAGE::cReset()
                        << " Lagrangian domain = alpha shape, tetrahedra with circumradius <= "
                        << MESSAGE::cMagenta() << state.options.psLagAlphaRadius << MESSAGE::cReset() << " ("
                        << state.options.psAlphaShape << " mean spacings of " << spacing << "; --ps-alpha-shape).\n" << MESSAGE::Flush;
                }
            }
        }
    }
}

// PS-DTFE, periodic: unwraps, finds the Lagrangian box and seam, and adds the periodic copies (or defers them).
static void psLagrangianPeriodicSetup(DTFE_State &state, User_options &userOptions, Box &lagBoxGlobal,
                                      bool &hasLagrangianPeriodicCopies)
{
    if ( userOptions.periodic && !state.particles.empty() )
    {
        MESSAGE::Message msg( userOptions.verboseLevel );

        computeLagrangianBoundingBox( state.particles, lagBoxGlobal );

        // Eulerian box length per axis, used both for unwrapping and for shifting periodic copies
        Real eulerLen[NO_DIM];
        for (int d=0; d<NO_DIM; ++d)
            eulerLen[d] = state.options.boxCoordinates[2*d+1] - state.options.boxCoordinates[2*d];

        // Lagrangian padding: same fraction as the Eulerian padding, with safety floors
        Real lagLength[NO_DIM], lagPad[NO_DIM];
        for (int d=0; d<NO_DIM; ++d)
        {
            lagLength[d] = lagBoxGlobal[2*d+1] - lagBoxGlobal[2*d];
            Real padFrac = (state.options.paddingLength[2*d] + state.options.paddingLength[2*d+1]) / (Real(2.) * eulerLen[d]);
            lagPad[d] = padFrac * lagLength[d];
            // floor at 10% of the box so the tessellation still tiles when the Eulerian padding is tiny
            if (lagPad[d] < lagLength[d] * Real(0.10))
                lagPad[d] = lagLength[d] * Real(0.10);
            // global copies must reach the per-partition padding (0.30 of a cell) so edge partitions stay covered
            if (userOptions.partitionOn && userOptions.partNo < 0)
            {
                Real partPad = lagLength[d] / Real(userOptions.partition[d]) * Real(0.30);
                if (lagPad[d] < partPad)
                    lagPad[d] = partPad;
            }
        }

        // unwrap Eulerian positions so each displacement s = pos - lagPos lies in [-L/2, L/2];
        // otherwise boundary particles get displacements near L and huge spurious simplices
        psUnwrapEulerian( state.particles, state.options );

        // the globals a later --ps-window run can start from without reading the particles
        // (ps_globals_record.h): written by every partitioned run with a tessellation cache whose
        // particles were read as they are, the mean density computed from them
        if ( not state.options.tessellationCacheDir.empty() and userOptions.partitionOn and userOptions.partNo < 0
             and userOptions.psAverageDensityFromParticles and not userOptions.regionOn and userOptions.poisson == 0
             and userOptions.randomSample < Real(0.) )
        {
            PSGlobals::Record rec;
            rec.n = state.particles.size();
            rec.averageDensity = double( state.options.averageDensity );
            for (int i = 0; i < 2*NO_DIM; ++i)
            {
                rec.box[i] = double( state.options.boxCoordinates.coords[i] );
                rec.lagBox[i] = double( lagBoxGlobal.coords[i] );
            }
            rec.hubble = double( state.options.hubbleParam );
            rec.scaleFactor = double( state.options.scaleFactor );
            PSGlobals::write( state.options, rec );
        }

        msg << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset()
            << " Unwrapped Eulerian positions (displacements within [-L/2, L/2]).\n" << MESSAGE::Flush;

        hasLagrangianPeriodicCopies = true;

        // non-partition path builds one triangulation, so it needs all periodic images up front;
        // the partition path generates images per-partition and defers this whole-box copy array
        // (several GB at 512^3) to keep peak memory low
        bool const psPartitioned = ( userOptions.partitionOn and userOptions.partNo < 0 );
        // the one triangulation's ownership period per axis, [seam - L, seam): where its padding allows, the
        // seam whose periodic copies are all EXACT shifts of their originals (subpartition.h psPeriodicSeam --
        // the tessellation around the two edges of the period is then the same up to L, and the centroid
        // ownership tiles it exactly); else the fallback anchor, the top particle plane. The
        // partition path tiles its own (PSPartitionPlan, with the partitions' padding)
        Real seamG[NO_DIM];
        bool seamExactG[NO_DIM];
        for (int d=0; d<NO_DIM; ++d)
            seamExactG[d] = psPeriodicSeam( lagBoxGlobal[2*d], lagBoxGlobal[2*d+1], eulerLen[d], lagPad[d], seamG[d] );
        if ( not psPartitioned )
        {
            // the padded period: [seam - L - pad, seam + pad] -- the originals and the -L copies of those above
            // seam - pad, no +L copy (the copy loop below makes the copies that fall inside it); the fallback seam
            // keeps the bounding box +- pad
            Box lagPaddedGlobal = lagBoxGlobal;
            for (int d=0; d<NO_DIM; ++d)
            {
                if ( seamExactG[d] )
                {
                    lagPaddedGlobal[2*d]   = seamG[d] - eulerLen[d] - lagPad[d];
                    lagPaddedGlobal[2*d+1] = seamG[d] + lagPad[d];
                    continue;
                }
                lagPaddedGlobal[2*d]   -= lagPad[d];
                lagPaddedGlobal[2*d+1] += lagPad[d];
            }
            psReportSeam( msg, seamG, seamExactG, eulerLen, lagPad, state.particles );

            // add periodic copies from all 26 (3D) or 8 (2D) images; Lagrangian and Eulerian
            // positions shift by the same box-length offset for correct simplex geometry
            size_t const noOriginal = state.particles.size();
            size_t const noOffsets = (NO_DIM == 2 ? 9 : 27);
            size_t const centerIdx = (NO_DIM == 2 ? 4 : 13);

            for (size_t n = 0; n < noOffsets; ++n)
            {
                if (n == centerIdx) continue;  // skip zero offset

                Real lagOffset[NO_DIM], eulerOffset[NO_DIM];
                size_t rem = n;
                for (int d = NO_DIM - 1; d >= 0; --d)
                {
                    int sign = int(rem % 3) - 1;  // -1, 0, or +1
                    // both use the period eulerLen; a bounding-box width would under-shift
                    // copies onto boundary originals and corrupt the tiling
                    lagOffset[d]   = Real(sign) * eulerLen[d];
                    eulerOffset[d] = Real(sign) * eulerLen[d];
                    rem /= 3;
                }

                for (size_t i = 0; i < noOriginal; ++i)
                {
                    bool inside = true;
                    for (int d = 0; d < NO_DIM; ++d)
                    {
                        Real shiftedLag = state.particles[i].lagPos[d] + lagOffset[d];
                        if (shiftedLag < lagPaddedGlobal[2*d] || shiftedLag > lagPaddedGlobal[2*d+1])
                        { inside = false; break; }
                    }
                    if (inside)
                    {
                        Particle_data copy = state.particles[i];
                        for (int d = 0; d < NO_DIM; ++d)
                        {
                            copy.lagPos[d] += lagOffset[d];
                            copy.pos[d]    += eulerOffset[d];
                        }
                        state.particles.push_back(copy);
                    }
                }
            }

            msg << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " Lagrangian bounding box = "
                << MESSAGE::cMagenta() << lagBoxGlobal.print() << MESSAGE::cReset() << "\n"
                << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " Added "
                << MESSAGE::cMagenta() << (state.particles.size() - noOriginal) << MESSAGE::cReset()
                << " Lagrangian periodic copies (" << MESSAGE::cMagenta() << noOriginal << MESSAGE::cReset()
                << " original particles).\n" << MESSAGE::Flush;
        }
        else
        {
            msg << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " Lagrangian bounding box = "
                << MESSAGE::cMagenta() << lagBoxGlobal.print() << MESSAGE::cReset() << "\n"
                << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset()
                << " periodic copies are generated per-partition (deferred) to keep peak memory low.\n" << MESSAGE::Flush;
        }

        // primary periodic box: the centroid ownership check keeps one image of each cell and tiles the box;
        // overridden per partition in the partition path. It is ONE PERIOD, [seam - L, seam), not the snapshot box
        // [0, L): on a lattice the box edge lies halfway between particle planes, so a straddling tet's centroid is
        // ON the edge and its float-rounded copies can put both images outside (4.1% of a single-precision 24^3
        // lattice's mass; tests/tweb_check.sh). The period ends at the exact seam (subpartition.h psPeriodicSeam)
        // where the padding allows, at the top particle plane otherwise.
        for (int d=0; d<NO_DIM; ++d)
        {
            state.options.lagrangianRegion[2*d]   = seamG[d] - eulerLen[d];
            state.options.lagrangianRegion[2*d+1] = seamG[d];
        }
    }
    else if ( userOptions.periodic && userOptions.psParticlesDeferred )
    {
        // started without the particles (ps_globals_record.h): the recorded globals stand in for the
        // block above -- the bounding box of the initial positions, and (partitioned runs only) the
        // periodic copies generated per partition when a partition's particles are selected
        lagBoxGlobal = userOptions.psPresetLagBox;
        hasLagrangianPeriodicCopies = true;
        state.options.lagrangianRegion = state.options.boxCoordinates;
        MESSAGE::Message msg( userOptions.verboseLevel );
        msg << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " Lagrangian bounding box = "
            << MESSAGE::cMagenta() << lagBoxGlobal.print() << MESSAGE::cReset() << " (recorded; the particles are not read yet).\n"
            << MESSAGE::Flush;
    }
}

// --ps-window: computes the window's cell range from the final box and grid, for both copies of the options.
static void psResolveWindow(DTFE_State &state, User_options &userOptions)
{
    // --ps-window: its cell range needs the final box and grid (the header's box is known only now).
    // Both the partition path (state.options) and the single triangulation (state.options via
    // DTFE_parallel) read it from state.options; the writers read the caller's copy, so both get it
    if ( state.options.psWindowOn )
    {
        state.options.computePsWindow();
        for (int d = 0; d < NO_DIM; ++d)
        {
            userOptions.psWindowLo[d]   = state.options.psWindowLo[d];
            userOptions.psWindowDims[d] = state.options.psWindowDims[d];
        }
        // NO_DIM entries: a fixed 3 would read past the end of the 2D build's 2-entry arrays
        std::string dims, lo;
        for (int d = 0; d < NO_DIM; ++d)
        {
            dims += ( d ? "x" : "" ) + std::to_string( state.options.psWindowDims[d] );
            lo   += ( d ? ", " : "" ) + std::to_string( state.options.psWindowLo[d] );
        }
        MESSAGE::Message message( userOptions.verboseLevel );
        message << "--ps-window: the output grid is the window of " << dims << " cells at (" << lo
                << ") of the " << MESSAGE::printElements( state.options.gridSize, "x" ) << " grid.\n" << MESSAGE::Flush;
    }
}

// '--serve --partition', PS-DTFE: every Lagrangian partition to the composite server, which serves; no return.
template <class PartitionOptions>
static void psServeComposite(DTFE_State &state, User_options &userOptions, PSPartitionPlan &plan,
                             PartitionOptions &partitionOptions)
{
    int concurrency = plan.total;
    if ( userOptions.maxConcurrent > 0 and userOptions.maxConcurrent < concurrency )
        concurrency = userOptions.maxConcurrent;
#ifdef OPEN_MP
    concurrency = std::max( 1, std::min( concurrency, omp_get_max_threads() ) );
#endif
    psServeCompositeBegin( state.options, plan.total );
    ConcurrentTriangulationPlan triangulationPlan( concurrency, plan.total );
#ifdef OPEN_MP
    #pragma omp parallel for schedule(dynamic, 1) num_threads(concurrency)
#endif
    for (int pi=0; pi<plan.total; ++pi)
    {
    ConcurrentTriangulationGuard concurrentTriangulation;
        Box lagRegion, lagPadded;
        plan.regions( pi, lagRegion, lagPadded );
        // the batch loop's per-partition options (below), so the cache files are shared
        User_options tempOpt = partitionOptions( lagRegion );
        int tnum = 0;
#ifdef OPEN_MP
        tnum = omp_get_thread_num();
#endif
        tempOpt.verboseLevel = (tnum == 0) ? std::min( userOptions.verboseLevel, 1 ) : 0;
        psServeCompositeAddPartition( pi, tempOpt,
            [&](std::vector<Particle_data> &out)
            {
                findParticlesInBoxLagrangianPeriodic( state.particles, &out, lagPadded,
                                                      plan.eulerLen, plan.periodicCopies, 0 );
            }, nullptr, nullptr );
    }
    std::vector<Particle_data>().swap( state.particles );
    psServeCompositeRun( state.options );   // answers requests until stdin closes; never returns
}

// Sizes the main grids and the accumulators reserveMemory leaves out (streams, caustic bits, exact touches).
static void psReserveGrids(DTFE_State &state, Quantities *uQuantities, Quantities *aQuantities)
{
    size_t mainGrid[NO_DIM];                       // the main grid: the --ps-window cells, else the full grid
    state.options.outputGridSize( mainGrid );
    uQuantities->reserveMemory( mainGrid, state.options.uField );
    aQuantities->reserveMemory( mainGrid, state.options.aField );

    // pre-size the stream_count accumulators (reserveMemory does not cover them) so
    // addFromSubgrid does not write past an unsized vector
    size_t totalGrid = 1;
    for (int d=0; d<NO_DIM; ++d) totalGrid *= mainGrid[d];
    if ( state.options.uField.selected() )
    {
        uQuantities->stream_count.assign(totalGrid, Real(0.));
        uQuantities->hidden_streams.assign(totalGrid, Real(0.));
    }
    if ( state.options.aField.selected() )
    {
        aQuantities->stream_count.assign(totalGrid, Real(0.));
        aQuantities->hidden_streams.assign(totalGrid, Real(0.));
    }
    if ( state.options.psCaustics )
    {
        if ( state.options.uField.selected() )
            uQuantities->caustic_bits.assign(totalGrid, Real(0.));
        if ( state.options.aField.selected() )
            aQuantities->caustic_bits.assign(totalGrid, Real(0.));
    }
    if ( state.options.psExactDeposit )
    {
        if ( state.options.uField.selected() )
            uQuantities->tet_touch.assign(totalGrid, Real(0.));
        if ( state.options.aField.selected() )
            aQuantities->tet_touch.assign(totalGrid, Real(0.));
    }
}

// --ps-window with a tessellation cache: skips the partitions whose occupancy maps never reach the window.
template <class PartitionOptions>
static void psWindowSkips(DTFE_State &state, PSPartitionPlan &plan, PartitionOptions &partitionOptions,
                          int &totalPartitions, MESSAGE::Message &message, std::vector<char> &skipPartition,
                          int &nSkipped)
{
    if ( state.options.psWindowOn and not state.options.tessellationCacheDir.empty() )
    {
        int nUnknown = 0;
        for (int pi=0; pi<totalPartitions; ++pi)
        {
            Box lagRegion, lagPadded;
            plan.regions( pi, lagRegion, lagPadded );
            int const t = psOccupancyTouchesWindow( partitionOptions( lagRegion ), state.options );
            if ( t == 0 ) { skipPartition[size_t(pi)] = 1; ++nSkipped; }
            if ( t < 0 ) ++nUnknown;
        }
        message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " --ps-window: "
                << MESSAGE::cMagenta() << nSkipped << MESSAGE::cReset() << " of " << totalPartitions
                << " partitions never reach the window (their occupancy maps in the tessellation cache) and are skipped; "
                << (totalPartitions - nSkipped) << " to process"
                << ( nUnknown > 0 ? " (" + std::to_string(nUnknown) + " without a map yet)" : std::string() )
                << ".\n" << MESSAGE::Flush;
    }
}

// A run started without its particles: reads them when a partition the window needs is not cached.
template <class PartitionOptions>
static void psReadDeferredParticles(DTFE_State &state, User_options &userOptions, PSPartitionPlan &plan,
                                    PartitionOptions &partitionOptions, int &totalPartitions,
                                    MESSAGE::Message &message, std::vector<char> &skipPartition)
{
    // started without the particles: every partition the window needs must be cached, or they are
    // read now (the run is then the ordinary one) -- verified against the record before use
    if ( state.options.psParticlesDeferred )
    {
        int nMissing = 0;
        for (int pi=0; pi<totalPartitions; ++pi)
        {
            if ( skipPartition[size_t(pi)] ) continue;
            Box lagRegion, lagPadded;
            plan.regions( pi, lagRegion, lagPadded );
            if ( not psTessellationCached( partitionOptions( lagRegion ) ) ) ++nMissing;
        }
        if ( nMissing == 0 )
            message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " every partition the window needs is "
                    << "cached: the snapshot is not read (its globals come from the record).\n" << MESSAGE::Flush;
        else
        {
            message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " " << nMissing
                    << " partition(s) the window needs are not cached: reading the particles after all.\n" << MESSAGE::Flush;
            User_options readOpts = userOptions;
            readOpts.psParticlesDeferred = false;
            std::vector<Sample_point> noSamples;
            readInputData( &state.particles, &noSamples, &readOpts );
            bool same = state.particles.size() == userOptions.psPresetN;
            for (int i = 0; i < 2*NO_DIM and same; ++i)
                same = readOpts.boxCoordinates.coords[i] == state.options.boxCoordinates.coords[i];
            if ( not same )
                throwError( "the globals record '", PSGlobals::path( userOptions ), "' does not match the snapshot "
                            "it names (another particle count or box); delete it -- the next run rewrites it." );
            psUnwrapEulerian( state.particles, state.options );
            state.options.psParticlesDeferred = false;
        }
    }
}

// A partition the staged ordered merge parks until its turn (psPartitionedRun).
struct StagedPartition
{
    Quantities u, a;
    bool emptyPartition = true, skipped = false;
    double bytes = 0.;          // the RAM it holds (0 when spilled)
    PSMergeSpill::File spill;   // spilled: u and a are in this file, their vectors empty
};
// where a finished partition parks (psPartitionedRun's parkWhere)
enum { PARK_WAIT = 0, PARK_RAM = 1, PARK_DISK = 2 };
// One partition between its two stages, prepare and deposit (psPartitionedRun).
struct PSPrepared
{
    bool skipped = false, cacheOnly = false, emptyPartition = true;
    User_options opt;
    vector<Particle_data> part;
    std::shared_ptr<void> pre;   // the look-ahead's tessellation (psPrebuildTessellation); none: built in the deposit
    double seconds = 0.;         // the look-ahead's build time
};
// Leaves the memory gate when a gated partition's iteration ends (psRunPartitionLoop).
struct GateRelease { PartitionMemoryGate *g; int pi; ~GateRelease() { if ( g ) g->leave( pi ); } };

// The partition loop's shared state, references to psPartitionedRun's locals: progress, merge, spill, concurrency.
struct PSLoopCtx
{
    DTFE_State &state;
    User_options &userOptions;
    Quantities *uQuantities;
    Quantities *aQuantities;
    MESSAGE::Message &message;
    int &totalPartitions;
    PSPartitionPlan &plan;
    int &psConcurrency;
    std::vector<char> &skipPartition;
    bool const &fixedMerge;
    PSFixedMerge::Merge &fxMerge;
    int const &toProcess;
    std::chrono::steady_clock::time_point const &psLoopStart;
    int &psDone;
    std::vector<double> const &partCost;
    double &costTotal;
    double &costDone;
    bool const &orderedMerge;
    std::mutex &mergeMutex;
    std::condition_variable &mergeCv;
    int &nextMerge;
    std::map<int, StagedPartition> &staged;
    double &stagedBytes;
    double &stagedPeakBytes;
    size_t &stagedPeakCount;
    size_t &stagedRamCount;
    int &nStaged;
    int &nWaitedTurn;
    double &stageCapBytes;
    bool const &staging;
    bool const &gateGuards;
    int &spillMode;
    std::string &spillWhy;
    std::string const &spillDir;
    bool &spillUsable;
    int &nSpilled;
    int &nSpillTurn;
    size_t &spilledCount;
    size_t &spilledPeakCount;
    double &spilledBytes;
    double &spilledPeakBytes;
    double &spillWritten;
    double &spillPendingBytes;
    std::string &parkRoomText;
};

// Says how the partitions merge: in index order with their parking and spill, or into the fixed-point grids.
static void psReportMergeMode(PSLoopCtx &c)
{
    auto &state = c.state;
    auto &message = c.message;
    auto &psConcurrency = c.psConcurrency;
    auto &fixedMerge = c.fixedMerge;
    auto &fxMerge = c.fxMerge;
    auto &orderedMerge = c.orderedMerge;
    auto &stageCapBytes = c.stageCapBytes;
    auto &staging = c.staging;
    auto &gateGuards = c.gateGuards;
    auto &spillMode = c.spillMode;
    auto &spillWhy = c.spillWhy;
    auto &spillDir = c.spillDir;
    auto &spillUsable = c.spillUsable;
    auto &parkRoomText = c.parkRoomText;
    if ( std::isfinite( state.options.psParkRoom ) )
    {
        char rb[96];
        std::snprintf( rb, sizeof(rb), ", in RAM up to the %.1f GB the memory budget leaves beside the plan",
                       std::max( 0., state.options.psParkRoom ) / 1.e9 );
        parkRoomText = rb;
    }
    if ( orderedMerge and psConcurrency > 1 )
        message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " partitions merge in index order "
                << "(bit-reproducible sums); "
                << ( staging ? ( stageCapBytes > 0. ? "one finished before its turn waits in memory, up to "
                                 + std::to_string( int( std::round( stageCapBytes / 1.e9 ) ) ) + " GB in all "
                                 "(DTFE_MERGE_STAGE_GB), while its thread goes on"
                                 : gateGuards ? std::string( "one finished before its turn waits in memory and its "
                                 "thread goes on (the memory gate admits the next partition)" )
                                   + parkRoomText
                                 : std::string( "one finished before its turn waits in memory, while the free memory "
                                 "allows, and its thread goes on" ) )
                             : std::string( "one finished before its turn waits for it (DTFE_MERGE_STAGE_GB=0)" ) )
                << ( not staging ? std::string()
                     : spillUsable ? std::string( spillMode > 0 ? "; every one with sub-grids is spilled to disk"
                                                                : "; when the memory cannot hold it, spilled to disk" )
                                     + " in '" + spillDir + "'" + ( spillMode > 0 ? " (DTFE_MERGE_SPILL=1)" : "" )
                     : "; no spill to disk (" + spillWhy + ")" )
                << ( state.options.psCompletionOrderMergeFlag ? "; --resume overrides --completion-order-merge" : "" )
                << ".\n" << MESSAGE::Flush;
    if ( fixedMerge )
        message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " --fixed-point-merge: each partition is "
                << "added as it finishes into exact integer grids (9 B per summed component and cell; the same bytes "
                << "at any concurrency and completion order; nothing parks or spills). Units from the "
                << fxMerge.unitsText() << ".\n" << MESSAGE::Flush;
}

// A finished partition, under the merge lock: merged in its turn (then the parked ones due), else parked or waits.
template <class SubgridBytes, class ParkWhere>
static inline void psMergeInTurn(PSLoopCtx &c, SubgridBytes &subgridBytes, ParkWhere &parkWhere, int const pi,
                                 PSPrepared &P, Quantities &temp_uQuantities, Quantities &temp_aQuantities)
{
    auto &state = c.state;
    auto &userOptions = c.userOptions;
    auto &uQuantities = c.uQuantities;
    auto &aQuantities = c.aQuantities;
    auto &fixedMerge = c.fixedMerge;
    auto &fxMerge = c.fxMerge;
    auto &toProcess = c.toProcess;
    auto &psLoopStart = c.psLoopStart;
    auto &psDone = c.psDone;
    auto &partCost = c.partCost;
    auto &costTotal = c.costTotal;
    auto &costDone = c.costDone;
    auto &orderedMerge = c.orderedMerge;
    auto &mergeMutex = c.mergeMutex;
    auto &mergeCv = c.mergeCv;
    auto &nextMerge = c.nextMerge;
    auto &staged = c.staged;
    auto &stagedBytes = c.stagedBytes;
    auto &stagedPeakBytes = c.stagedPeakBytes;
    auto &stagedPeakCount = c.stagedPeakCount;
    auto &stagedRamCount = c.stagedRamCount;
    auto &nStaged = c.nStaged;
    auto &nWaitedTurn = c.nWaitedTurn;
    auto &spillWhy = c.spillWhy;
    auto &spillDir = c.spillDir;
    auto &spillUsable = c.spillUsable;
    auto &nSpilled = c.nSpilled;
    auto &nSpillTurn = c.nSpillTurn;
    auto &spilledCount = c.spilledCount;
    auto &spilledPeakCount = c.spilledPeakCount;
    auto &spilledBytes = c.spilledBytes;
    auto &spilledPeakBytes = c.spilledPeakBytes;
    auto &spillWritten = c.spillWritten;
    auto &spillPendingBytes = c.spillPendingBytes;
    std::unique_lock<std::mutex> lock( mergeMutex );
    // one partition's grids into the full ones, then its checkpoint (partition_checkpoint.h), while the grids
    // hold still under the merge lock: after this merge unless the last write took more than 5% of the time
    // since it
    auto mergeOne = [&](int const pj, bool const emptyPartition, bool const skipped, Quantities &tu, Quantities &ta)
    {
        if ( not emptyPartition )
        {   // with --ps-window every partition's sub-grid IS the window: the merge is window-local
            size_t const *wo = state.options.psWindowOn ? state.options.psWindowLo   : nullptr;
            size_t const *wd = state.options.psWindowOn ? state.options.psWindowDims : nullptr;
            if ( fixedMerge )   // the integer grids; the masks still OR into the Real ones
                fxMerge.add( pj, tu, ta, *uQuantities, *aQuantities, &(state.options.gridSize[0]), wo, wd );
            else
            {
                uQuantities->addFromSubgrid( tu, &(state.options.gridSize[0]), wo, wd );
                aQuantities->addFromSubgrid( ta, &(state.options.gridSize[0]), wo, wd );
            }
        }
        PartitionCheckpoint::afterMerge( pj, skipped, *uQuantities, *aQuantities, userOptions.verboseLevel );
    };
    bool stagedHere = false;
    if ( orderedMerge and nextMerge != pi )
    {
        // not its turn: park its sub-grids -- in RAM, or spilled to disk -- and free the thread,
        // else wait (parkWhere above)
        double const bytes = P.emptyPartition ? 0. : subgridBytes( temp_uQuantities ) + subgridBytes( temp_aQuantities );
        int where = parkWhere( bytes );
        if ( where == PARK_DISK )
        {
            // the write runs outside the merge lock (the merges of the others go on); the sub-grids stay in
            // memory until the file is complete, so a failed write loses nothing
            PSMergeSpill::File file;
            std::string why;
            spillPendingBytes += bytes;   // counted against the room of the spills that start meanwhile
            lock.unlock();
            bool const wrote = PSMergeSpill::write( temp_uQuantities, temp_aQuantities, spillDir, file, why );
            lock.lock();
            spillPendingBytes -= bytes;
            if ( nextMerge == pi )
            {   // its turn came during the write: the drain before it found nothing parked here and stopped,
                // so it merges now, from memory, and the file goes (a late park would never be merged)
                ++nSpillTurn;
                where = -1;
            }
            else if ( wrote )
            {
                PSMergeSpill::release( temp_uQuantities );
                PSMergeSpill::release( temp_aQuantities );
                StagedPartition &S = staged[pi];
                S.u = std::move( temp_uQuantities );   // empty vectors; the sub-grid placement stays with them
                S.a = std::move( temp_aQuantities );
                S.emptyPartition = P.emptyPartition;
                S.skipped = P.skipped;
                S.bytes = 0.;
                ++nStaged;
                ++nSpilled;
                ++spilledCount;
                spilledBytes += file.bytes;
                spillWritten += file.bytes;
                spilledPeakCount = std::max( spilledPeakCount, spilledCount );
                spilledPeakBytes = std::max( spilledPeakBytes, spilledBytes );
                S.spill = std::move( file );
                stagedHere = true;
            }
            else
            {   // the disk refused (full, unwritable): no more spills this run; parked in RAM or waiting, as
                // without the spill
                if ( spillUsable )
                {
                    MESSAGE::Warning warning( userOptions.verboseLevel );
                    warning << "the staged ordered merge cannot spill to disk (" << why << "); from now on a "
                            << "partition that finishes before its turn waits in memory or for its turn.\n"
                            << MESSAGE::EndWarning;
                    spillUsable = false;
                    spillWhy = why;
                }
                where = parkWhere( bytes );
            }
        }
        if ( where == PARK_RAM )
        {
            StagedPartition &S = staged[pi];
            S.u = std::move( temp_uQuantities );
            S.a = std::move( temp_aQuantities );
            S.emptyPartition = P.emptyPartition;
            S.skipped = P.skipped;
            S.bytes = bytes;
            stagedBytes += bytes;
            ++nStaged;
            ++stagedRamCount;
            stagedPeakBytes = std::max( stagedPeakBytes, stagedBytes );
            stagedPeakCount = std::max( stagedPeakCount, stagedRamCount );
            stagedHere = true;
        }
        else if ( where == PARK_WAIT )
        {
            if ( not P.skipped ) ++nWaitedTurn;
            mergeCv.wait( lock, [&]{ return nextMerge == pi; } );
        }
    }
    if ( not stagedHere )
    {
        mergeOne( pi, P.emptyPartition, P.skipped, temp_uQuantities, temp_aQuantities );
        if ( orderedMerge )
        {
            nextMerge = pi + 1;
            // the staged partitions whose turn has now come, in index order; a spilled one is read back first
            // (under the lock, beside its merge: one partition's sub-grids at a time)
            for ( auto it = staged.find( nextMerge ); it != staged.end(); it = staged.find( nextMerge ) )
            {
                StagedPartition S = std::move( it->second );
                staged.erase( it );
                if ( S.spill.fd >= 0 )
                {
                    spilledBytes -= S.spill.bytes;
                    --spilledCount;
                    std::string why;
                    if ( not PSMergeSpill::read( S.spill, S.u, S.a, why ) )
                        throwError( "the staged ordered merge: partition ", nextMerge, " was spilled to '" + spillDir
                                    + "' and cannot be merged: " + why );
                }
                else
                {
                    stagedBytes -= S.bytes;
                    --stagedRamCount;
                }
                mergeOne( nextMerge, S.emptyPartition, S.skipped, S.u, S.a );
                ++nextMerge;
            }
            mergeCv.notify_all();
        }
    }

    // progress over the partitions actually processed (the skipped ones cost nothing); a staged partition is
    // done: its work is finished, only its add waits
    if ( not P.skipped )
    {
    ++psDone;
    costDone += partCost[size_t(pi)];
    double const elapsed = std::chrono::duration<double>( std::chrono::steady_clock::now() - psLoopStart ).count();
    double const frac    = double(psDone) / double(toProcess);
    // the time per unit of predicted cost so far, times the predicted cost still to come
    double const eta     = ( costDone > 0. ) ? elapsed * std::max( 0., costTotal - costDone ) / costDone : 0.;
    MESSAGE::Message prog( userOptions.verboseLevel );
    prog << MESSAGE::cGreen() << "  [partitions " << psDone << "/" << toProcess
         << " | " << int(100.*frac + 0.5) << "%]" << MESSAGE::cReset()
         << MESSAGE::cDim() << "  elapsed " << MESSAGE::formatDuration(elapsed)
         << ( psDone<toProcess ? ", ETA ~" + MESSAGE::formatDuration(eta) : ", done" )
         << MESSAGE::cReset() << "\n" << MESSAGE::Flush;
    runProgressPhase( "partitions", psDone, toProcess, psDone < toProcess ? eta : 0. );
    }
}

// Runs the partitions: in turn with the look-ahead (k+1 prepared while k deposits), else in OpenMP behind the gate.
template <class Prepare, class Deposit>
static void psRunPartitionLoop(PSLoopCtx &c, PartitionMemoryGate &memoryGate, Prepare &prepare, Deposit &deposit)
{
    auto &state = c.state;
    auto &userOptions = c.userOptions;
    auto &message = c.message;
    auto &totalPartitions = c.totalPartitions;
    auto &plan = c.plan;
    auto &psConcurrency = c.psConcurrency;
    auto &skipPartition = c.skipPartition;
    auto &toProcess = c.toProcess;
    // The LOOK-AHEAD. At --max-concurrent 1 with the GPU deposit the cores would idle while the
    // GPU works (the deposit is 82% of a TNG50 z=0 run), and each next partition's selection +
    // triangulation + vertex densities (~35 s) would wait behind it. So partition k+1
    // as without the look-ahead (same particles, same sequential insert) and the merge stays in index
    // order, so the outputs are byte-identical (tests/ps_runtime_check.sh). It holds one more partition's
    // particles + tessellation; before each look-ahead the free memory is checked against that (the
    // auto-tuner's prediction counts it too) and the partition is prepared in turn when it does not fit.
    // '--ps-no-overlap' switches it off; DTFE_PS_OVERLAP=1 forces it on a CPU deposit too (the test suite's
    // byte-identity check on a build without a GPU), DTFE_PS_OVERLAP=0 switches it off.
    bool gpuDeposit = false;
#if defined(PS_GPU) && NO_DIM==3
    gpuDeposit = state.options.psUseMetal and not state.options.psLinearDeposit;
#endif
    char const *overlapEnv = std::getenv( "DTFE_PS_OVERLAP" );
    bool const overlapForced = overlapEnv and overlapEnv[0] == '1';
    bool const overlapBanned = overlapEnv and overlapEnv[0] == '0';
    bool overlap = psConcurrency == 1 and toProcess > 1 and not state.options.psNoOverlap
                   and not state.options.psLookAheadOffBudget   // the auto-tuner's: it does not fit the budget
                   and not overlapBanned and ( gpuDeposit or overlapForced );
    // what the look-ahead holds: one partition's padded particle copy + its tessellation (auto_tune.h's
    // model: Particle_data + 666 B per vertex, padding included)
    double aheadBytes = 0.;
    if ( overlap )
    {
        double padVol = 1., ownVol = 1.;
        {
            Box r, rp;
            plan.regions( 0, r, rp );
            for (int d=0; d<NO_DIM; ++d) { ownVol *= double(r[2*d+1]-r[2*d]); padVol *= double(rp[2*d+1]-rp[2*d]); }
        }
        double const nAll = state.options.psParticlesDeferred ? double(userOptions.psPresetN) : double(state.particles.size());
        aheadBytes = nAll / double(totalPartitions) * ( padVol / std::max( ownVol, 1e-300 ) )
                     * ( double(sizeof(Particle_data)) + 666. );
        // the memory budget too, not only the free memory: what it leaves beside the grids,
        // particles and one partition must hold the look-ahead, or the run has none (the same for every
        // partition, so decided once)
        if ( aheadBytes > state.options.psLookAheadRoom )
        {
            overlap = false;
            message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " look-ahead off: the ~"
                    << std::round( aheadBytes/1.e7 )/100. << " GB it holds is more than the ~"
                    << std::round( std::max( 0., state.options.psLookAheadRoom )/1.e7 )/100. << " GB the memory "
                    << "budget leaves beside one partition; partitions are prepared in turn.\n" << MESSAGE::Flush;
        }
    }
    if ( overlap )
    {
        message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " look-ahead on: partition k+1 is "
                << "tessellated on a CPU thread while partition k deposits"
                << ( gpuDeposit ? " on the GPU" : " (DTFE_PS_OVERLAP=1)" ) << " (~" << std::round( aheadBytes/1.e7 )/100.
                << " GB more in memory; '--ps-no-overlap' switches it off).\n" << MESSAGE::Flush;

        PSPrepared cur;
        prepare( 0, false, cur );
        bool saidNoRoom = false;
        for (int pi=0; pi<totalPartitions; ++pi)
        {
            ConcurrentTriangulationGuard concurrentTriangulation;
            PSPrepared next;
            std::thread ahead;
            std::exception_ptr aheadError;
            bool aheadRan = false;
            if ( pi + 1 < totalPartitions )
            {
                double const freeNow = autoTuneSystemFree();
                bool const room = freeNow < 0. or freeNow - 4.e9 >= aheadBytes;
                if ( room and not skipPartition[size_t(pi+1)] )
                {
                    aheadRan = true;
                    ahead = std::thread( [&, pi]()
                    {
                        try { prepare( pi + 1, true, next ); }
                        catch ( ... ) { aheadError = std::current_exception(); }
                    } );
                }
                else if ( not room and not saidNoRoom )
                {
                    saidNoRoom = true;
                    MESSAGE::Message m( userOptions.verboseLevel );
                    m << MESSAGE::cDim() << "  look-ahead paused: " << freeNow/1.e9 << " GB free, minus 4 GB headroom, "
                      << "is below the ~" << aheadBytes/1.e9 << " GB it holds; partitions are prepared in turn "
                      << "until memory frees up\n" << MESSAGE::cReset() << MESSAGE::Flush;
                }
            }
            try
            {
                deposit( pi, cur, userOptions.verboseLevel );
            }
            catch ( ... )
            {
                if ( ahead.joinable() ) ahead.join();
                throw;
            }
            if ( ahead.joinable() ) ahead.join();
            if ( aheadError ) std::rethrow_exception( aheadError );
            if ( aheadRan and userOptions.verboseLevel > 0 and not next.emptyPartition )
            {
                MESSAGE::Message m( userOptions.verboseLevel );
                m << MESSAGE::cDim() << "  look-ahead: partition " << pi + 1 << " prepared in "
                  << MESSAGE::formatDuration( next.seconds ) << " during this deposit\n" << MESSAGE::cReset() << MESSAGE::Flush;
            }
            if ( pi + 1 < totalPartitions )
            {
                if ( not aheadRan ) prepare( pi + 1, false, next );
                cur = std::move( next );
            }
        }
    }
    else
    {
#ifdef OPEN_MP
    #pragma omp parallel for schedule(dynamic, 1) num_threads(psConcurrency)
#endif
    for (int pi=0; pi<totalPartitions; ++pi)
    {
        ConcurrentTriangulationGuard concurrentTriangulation;
        // the memory gate, BEFORE the particle copy (part of a partition's memory): a partition that builds or
        // loads waits while the free memory cannot hold one more and another (under the ordered merge: a
        // lower-numbered one) is in flight; a skipped one passes, and so does the auto-tuner's speed split
        // (tiny uniform partitions). Released at the iteration's end.
        // (the speed split has its own flag, psSpeedSplit, since every split merges in order)
        bool const gated = not skipPartition[size_t(pi)] and ( not state.options.psSpeedSplit or state.options.psOrderedMergeFlag );
        if ( gated )
            memoryGate.enter( pi, userOptions.verboseLevel );
        GateRelease const gateRelease{ gated ? &memoryGate : nullptr, pi };
        PSPrepared P;
        prepare( pi, false, P );
        int tnum = 0;
#ifdef OPEN_MP
        tnum = omp_get_thread_num();
#endif
        deposit( pi, P, (tnum == 0) ? userOptions.verboseLevel : 0 );
    }
    }
}

// After the loop: the GPU fallbacks, the checkpoint writes, the fixed-point grids to Real, the ordered merge's line.
static void psReportLoopEnd(PSLoopCtx &c)
{
    auto &state = c.state;
    auto &userOptions = c.userOptions;
    auto &uQuantities = c.uQuantities;
    auto &aQuantities = c.aQuantities;
    auto &message = c.message;
    auto &psConcurrency = c.psConcurrency;
    auto &fixedMerge = c.fixedMerge;
    auto &fxMerge = c.fxMerge;
    auto &psLoopStart = c.psLoopStart;
    auto &orderedMerge = c.orderedMerge;
    auto &staged = c.staged;
    auto &stagedPeakBytes = c.stagedPeakBytes;
    auto &stagedPeakCount = c.stagedPeakCount;
    auto &nStaged = c.nStaged;
    auto &nWaitedTurn = c.nWaitedTurn;
    auto &staging = c.staging;
    auto &spillWhy = c.spillWhy;
    auto &spillUsable = c.spillUsable;
    auto &nSpilled = c.nSpilled;
    auto &nSpillTurn = c.nSpillTurn;
    auto &spilledPeakCount = c.spilledPeakCount;
    auto &spillWritten = c.spillWritten;
    // A GPU deposit that fell back to the CPU warns from the master thread only (the others run at
    // verbose 0), so with partitions in flight on several threads most fallbacks would be silent: say how
    // many there were
    if ( int const nFallback = psGpuFallbacksTaken( true ) )
    {
        MESSAGE::Warning warning( userOptions.verboseLevel );
        warning << nFallback << " partition deposit pass(es) asked for the GPU and ran on the CPU instead. Only "
                << "the master thread's partitions print the reason (above, when one of them fell back); the "
                << "grids are complete, the run only took longer.\n" << MESSAGE::EndWarning;
    }

    PartitionCheckpoint::reportWrites( std::chrono::duration<double>( std::chrono::steady_clock::now() - psLoopStart ).count(),
                                       userOptions.verboseLevel );
    // '--fixed-point-merge': every integer grid to its Real grid, once (ps_fixed_merge.h), before anything reads
    // them; then what it held and how much room its units left
    if ( fixedMerge )
    {
        PartitionCheckpoint::state().blobs.clear();   // they point into fxMerge, and nothing is checkpointed now
        fxMerge.finish( *uQuantities, *aQuantities );
        message << MESSAGE::cDim() << "  " << fxMerge.reportText() << "\n" << MESSAGE::cReset() << MESSAGE::Flush;
    }
    // what the staged ordered merge did: how many partitions finished before their turn and waited in
    // memory, at most how many / how much at once, and how many waited for their turn holding their thread
    if ( orderedMerge and psConcurrency > 1 )
    {
        if ( not staged.empty() )   // cannot happen: every partition merges once, the last drains the rest
            throwError( "internal: ", staged.size(), " staged partition(s) were never merged" );
        // (of those, how many were spilled to disk, at most how many at once, and the GB written)
        auto gbText = [](double const b) { char t[32]; std::snprintf( t, sizeof(t), "%.2f", b / 1.e9 ); return std::string( t ); };
        message << MESSAGE::cDim() << "  ordered merge: " << nStaged << " partition(s) finished before their turn: "
                << ( nStaged - nSpilled ) << " waited in memory (at most " << stagedPeakCount << " at once, "
                << std::round( stagedPeakBytes / 1.e7 ) / 100. << " GB), " << nSpilled << " spilled to disk"
                << ( nSpilled > 0 ? " (at most " + std::to_string( spilledPeakCount ) + " at once, "
                                    + gbText( spillWritten ) + " GB written)"
                                  : spillUsable or not staging ? std::string() : " (" + spillWhy + ")" )
                << ( nSpillTurn > 0 ? " (+" + std::to_string( nSpillTurn ) + " whose turn came while writing: merged "
                                      "from memory)" : "" )
                << "; " << nWaitedTurn << " waited for their turn "
                << "holding their thread" << ( staging and nWaitedTurn > 0 ? " (no room to stage)" : "" ) << "\n"
                << MESSAGE::cReset() << MESSAGE::Flush;
    }
}

// Normalizes the summed moments once, over all partitions; PS_MASS_AUDIT=1 reports the run's mass closure.
static void psNormalizeMoments(DTFE_State &state, Quantities *uQuantities, Quantities *aQuantities,
                               double &auditInputMass)
{
    // fields were summed as density-weighted moments; normalize once now so cells drawing
    // streams from several partitions (multi-stream regions) come out correct. When the
    // density field is selected the partitions alias the weight to the density grid, so
    // pass the scale that turns the summed rho/rho_bar back into the per-cell mass.
    Real cellVolume = Real(1.);
    for (int d = 0; d < NO_DIM; ++d)
        cellVolume *= (state.options.region[2*d+1] - state.options.region[2*d]) / Real(state.options.gridSize[d]);
    Real const weightScale = cellVolume * state.options.averageDensity;
    uQuantities->normalizePhaseSpace( state.options.uField, weightScale );
    aQuantities->normalizePhaseSpace( state.options.aField, weightScale );

    // PS_MASS_AUDIT=1, (h): the merged grid against the passes' sums and the input mass, per pass kind
    if ( PSMassAudit::on() )
        for (int k = 0; k < 2; ++k)
        {
            PSMassAudit::Totals const &T = PSMassAudit::totals[k];
            std::vector<Real> const &den = ( k ? aQuantities : uQuantities )->density;
            if ( T.passes == 0 or den.empty() ) continue;
            double sum = 0.;
            for (Real const v : den) sum += double(v);
            double const merged = sum * double(cellVolume) * double(state.options.averageDensity);
            auto rel = [](double x, double ref) { return ref != 0. ? x / ref - 1. : 0.; };
            std::fprintf( stderr, "PS mass audit: run (%s pass, %zu partition passes): input %.12g, sum (a) owned %.12g, "
                          "sum (b) tets %.12g, sum (c) deg0 %.12g, sum (d) dropped %.12g, sum (g) grids %.12g (post %.12g), "
                          "merged %.12g, grid mean - 1 = %.6e | owned/input-1 %.3e, (tets+deg0)/input-1 %.3e, "
                          "grids/tets-1 %.3e, merged/grids(post)-1 %.3e, merged/input-1 %.3e\n",
                          k ? "averaged" : "unaveraged", T.passes, auditInputMass, T.owned, T.tets, T.deg0, T.dropped,
                          T.gridPre, T.gridPost, merged, sum / double(den.size()) - 1.,
                          rel(T.owned, auditInputMass), rel(T.tets + T.deg0, auditInputMass), rel(T.gridPre, T.tets),
                          rel(merged, T.gridPost), rel(merged, auditInputMass) );
        }
}

// Reports the coverage and multi-stream statistics of the summed grid over all partitions.
static void psAggregateReport(Quantities *uQuantities, Quantities *aQuantities, int &totalPartitions,
                              MESSAGE::Message &message)
{
    // aggregate coverage / stream statistics over all partitions from the summed grid
    // (per-partition figures were suppressed as misleading); prefer the unaveraged
    // stream_count, falling back to the averaged one if only '_a' fields exist
    std::vector<Real> const *sc = nullptr;
    if      ( not uQuantities->stream_count.empty() ) sc = &uQuantities->stream_count;
    else if ( not aQuantities->stream_count.empty() ) sc = &aQuantities->stream_count;
    if ( sc != nullptr )
    {
        size_t const ncell = sc->size();
        size_t covered = 0, multi = 0;
        Real maxStreams = Real(0.);
        for (size_t i = 0; i < ncell; ++i)
        {
            Real const v = (*sc)[i];
            if (v > Real(0.)) ++covered;
            // multiplicity is a float sum under every deposit -- single-stream cells land
            // on 1.0 +/- eps, so a bare 'v > 1' counts float noise as multi-stream (28.33%
            // vs the true 16.67% on the pancake). See PS_STREAM_TOL in quantities.h.
            if (v > Real(1.) + PS_STREAM_TOL) ++multi;
            if (v > maxStreams) maxStreams = v;
        }
        message << "\n" << MESSAGE::cCyan() << "PS-DTFE: aggregate over " << totalPartitions
                << " partitions -- max streams " << maxStreams
                << ", multi-stream grid points " << multi
                << " (" << (100.*multi/ncell) << "\%), grid coverage "
                << covered << "/" << ncell
                << " (" << (100.*covered/ncell) << "\%)" << MESSAGE::cReset()
                << ( covered < ncell ?
                     "  -- uncovered cells may indicate insufficient padding, a grid finer than the tessellation, or legitimate empty/expanding regions" : "" )
                << "\n" << MESSAGE::Flush;
    }
}

// PS-DTFE --partition: the Lagrangian partitions deposited, merged in order (or fixed point), normalized, reported.
static void psPartitionedRun(DTFE_State &state, vector<Sample_point> &samples, User_options &userOptions,
                             Quantities *uQuantities, Quantities *aQuantities, Box &lagBoxGlobal,
                             bool &hasLagrangianPeriodicCopies)
{
    int totalPartitions = 1;
    for (int d=0; d<NO_DIM; ++d)
        totalPartitions *= state.options.partition[d];
    MESSAGE::Message message( userOptions.verboseLevel );
    message << "\n" << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset()
            << " Lagrangian-space partitioning with " << MESSAGE::cMagenta() << totalPartitions
            << " partitions [" << MESSAGE::printElements( userOptions.partition, "," ) << "]"
            << MESSAGE::cReset() << ".\n" << MESSAGE::Flush;

    // the partition geometry (bounding box, lengths, padding), shared with the composite server
    PSPartitionPlan plan;
    plan.init( state, lagBoxGlobal, hasLagrangianPeriodicCopies, message );

    // PS_MASS_AUDIT=1: the input mass (the global array holds the originals only here; it is freed
    // before the merge is normalized) and fresh run totals (ps_mass_audit.h)
    double auditInputMass = 0.;
    if ( PSMassAudit::on() )
    {
        for (Particle_data &p : state.particles) auditInputMass += double( p.weight() );
        std::lock_guard<std::mutex> lock( PSMassAudit::mutex );
        PSMassAudit::totals[0] = PSMassAudit::totals[1] = PSMassAudit::Totals();
    }

    // Eulerian region/padding for the full box (needed by DTFE_interpolation; also part of every
    // partition's tessellation-cache identity, so it is set before the composite server too)
    if ( not userOptions.regionOn and not (userOptions.partitionOn and userOptions.partNo>=0) )
    {
        state.options.region = state.options.boxCoordinates;
        state.options.paddedBox = state.options.region;
        state.options.paddedBox.addPadding( state.options.paddingLength );
    }

    // a partition's options: its ownership region and the partition-path switches. They are part of
    // its tessellation-cache identity, so the composite server, the --ps-window skip and the batch
    // loop must form them identically -- hence one definition
    auto partitionOptions = [&](Box const &lagRegion) -> User_options
    {
        User_options o = state.options;
        o.lagrangianRegion = lagRegion;    // for cell ownership check
        o.psSuppressGridStats = true;      // per-partition stats misleading; aggregate printed after the loop
        o.psDeferNormalization = true;     // keep fields as moments; normalize once after summing
        o.psUseSubgrid = true;             // store only this partition's Eulerian bounding box
        return o;
    };

    // '--serve --partition': a composite server of the partition tessellations instead of grids
    if ( state.options.psServe )
        psServeComposite( state, userOptions, plan, partitionOptions );   // never returns

    psReserveGrids( state, uQuantities, aQuantities );

    // periodic=true here only drives grid-level index wrapping; data periodicity comes
    // from the Lagrangian copies and Eulerian unwrapping

    // each thread builds, interpolates, and accumulates its own triangulation into the
    // shared grids inside a critical section; peak memory ~ (concurrent partitions) x full grid

    // cap concurrent partition triangulations (the dominant memory cost); --max-concurrent trades cores for RAM, 0 = all threads
    int psConcurrency = totalPartitions;
    if ( userOptions.maxConcurrent > 0 and userOptions.maxConcurrent < psConcurrency )
        psConcurrency = userOptions.maxConcurrent;
#ifdef OPEN_MP
    // never more than the threads, like the serve paths and DTFE_parallel: '--max-concurrent 0' with a
    // hand-set '--partition 5 5 5' would start all 125 partitions at once, with no memory gate
    psConcurrency = std::min( psConcurrency, omp_get_max_threads() );
#endif
    if ( psConcurrency < 1 ) psConcurrency = 1;
    message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " building up to "
            << MESSAGE::cMagenta() << psConcurrency << MESSAGE::cReset() << " partition triangulation(s) concurrently"
            << ( userOptions.maxConcurrent > 0 ? " (--max-concurrent)" : "" ) << ".\n" << MESSAGE::Flush;

    // --ps-window: a partition whose occupancy map (beside its cached tessellation: written by a
    // composite server or an earlier batch run) has no bucket in the window deposits nothing
    // there -- skip it whole: no particle selection, no tessellation load or build. A partition
    // without a usable map is processed (and leaves its map for the next run).
    std::vector<char> skipPartition( size_t(totalPartitions), 0 );
    int nSkipped = 0;
    psWindowSkips( state, plan, partitionOptions, totalPartitions, message, skipPartition, nSkipped );
    psReadDeferredParticles( state, userOptions, plan, partitionOptions, totalPartitions, message, skipPartition );

    // The checkpoint of the merged grids and '--resume' (partition_checkpoint.h beginLoop / afterMerge /
    // reportWrites, which the standard loop below runs too). On resume the checkpoint's grids are loaded into the
    // pre-sized accumulators and its merged partitions are skipped like a --ps-window skip (no particles, no
    // tessellation, still a turn in the ordered merge). '--fixed-point-merge' (opt-in; ps_fixed_merge.h): the
    // summed accumulators become exact integer grids, allocated now so the memory gate sees them (the moment
    // weights come with a partition's first values, as in the default), in units fixed now from global quantities
    // (one pass over the particles for their largest velocity). A finished partition is then added at once (no
    // index order, parking or spill), the grids go back to Real once after the loop, and their words go into the
    // checkpoint as blobs.
    bool const fixedMerge = state.options.psFixedPointMerge;
    PSFixedMerge::Merge fxMerge;
    PartitionCheckpoint::state().blobs.clear();
    if ( fixedMerge )
    {
        if ( state.options.psParticlesDeferred )   // --ps-window is refused with the flag (user_options.cc)
            throwError( "internal: '--fixed-point-merge' needs the particles for its units, and they were not read." );
        fxMerge.setup( *uQuantities, *aQuantities, state.options, state.particles );
        PartitionCheckpoint::state().blobs = fxMerge.blobs();
    }
    PartitionCheckpoint::Checkpointer &checkpoint = PartitionCheckpoint::state();
    PartitionCheckpoint::beginLoop( state.options, totalPartitions, *uQuantities, *aQuantities, psPointEvalActive(),
                                    userOptions.verboseLevel );
    if ( fixedMerge and checkpoint.resumed )
        fxMerge.restore();   // the saved units must be this run's; the overflow bound comes back
    if ( checkpoint.resumed )
        for (int pi=0; pi<totalPartitions; ++pi)
            if ( checkpoint.merged[size_t(pi)] and not skipPartition[size_t(pi)] ) { skipPartition[size_t(pi)] = 1; ++nSkipped; }
    int const toProcess = std::max( 1, totalPartitions - nSkipped );

    // progress + ETA; updated only under mergeMutex below so it is race-free. The ETA weights each partition
    // by a predicted cost instead of counting partitions: the deposit time per partition is
    // heavy-tailed (TNG50 z=0: 40 s to 2049 s, median 73 s) and the costly cluster partitions sit together in
    // index order, so elapsed x remaining/done can be off by 2x.
    auto const psLoopStart = std::chrono::steady_clock::now();
    int psDone = 0;
    std::vector<double> const partCost = psPartitionCosts( state.particles, plan, state.options );
    double costTotal = 0., costDone = 0.;
    for (int pi=0; pi<totalPartitions; ++pi)
        if ( not skipPartition[size_t(pi)] ) costTotal += partCost[size_t(pi)];
    runProgressPhase( "partitions", 0, toProcess );
    // THE ORDERED MERGE (the default of every split; the auto-tuner's speed split always): the full grids receive
    // partition 0, then 1, 2 ..., exactly the adds of a '--max-concurrent 1' run, so the merged grids are
    // byte-identical to those. '--completion-order-merge' adds partitions as they finish instead, so the last bits
    // vary run to run (in the double build and for the GPU mass grid too; tests/gpu_mass_check.sh pins
    // --max-concurrent 1). Under '--fixed-point-merge' (integer adds need no order) orderedMerge is false, also
    // with --resume, and none of this runs.
    // STAGING: a partition that finishes before its turn moves its two sub-grid Quantities into 'staged' (no copy;
    // its particle copy is freed) and its thread takes the next one (idling behind a slow partition can cost a GPU
    // run 35%). Whoever merges the partition whose turn it is merges every staged one after it, in index order
    // under the same lock, writing each checkpoint (afterMerge) in that order: the checkpoint, its 'points.<i>'
    // files and the test hook see an unstaged merge's sequence. Under the MEMORY GATE (an auto-tuned plan with
    // partitions in flight: psPartitionBytes > 0, not the speed split) a finished partition always parks, since
    // the gate admits by the free memory, parked bytes included; without it, only while the free memory less 4 GB
    // holds it, else it waits (counted on the 'ordered merge:' line). DTFE_MERGE_STAGE_GB=X caps the parked bytes
    // at X GB instead (0 = never park). No deadlock: the partition whose turn it is never waits or parks, each
    // partition merges once, the dynamic schedule hands them out in index order, and the gate waits only behind
    // lower-numbered partitions in flight (parked ones have left it).
    // THE SPILL: the gate must admit a partition with no lower one in flight whatever the free memory, so parking
    // only in RAM (16 GB of sub-grids on TNG100) could exceed the machine. A partition that must park goes
    //   1. to RAM while the free memory less 4 GB (under the gate also less one psPartitionBytes) holds it
    //      (DTFE_MERGE_STAGE_GB=X: while the parked RAM stays under X GB); the free-memory reading already
    //      includes the sub-grids, a margin of one sub-grid kept on purpose;
    //   2. else to DISK (ps_merge_spill.h): one unlinked file in --scratch-dir (SCRATCH_DIR), else $TMPDIR, else
    //      /tmp, while that volume keeps 2 GB free (scratch_alloc.cc's margin) beside the spills being written and
    //      the --scratch-dir mappings' full size; at its turn it is read back into the same Quantities and merged
    //      by the same mergeOne (byte-identical to an in-RAM park). Written outside the merge lock; a partition
    //      whose turn comes during its write merges from RAM and drops the file;
    //   3. else, under the gate without a DTFE_MERGE_STAGE_GB cap, to RAM anyway (the same memory as waiting);
    //      otherwise it waits for its turn holding its thread.
    // DTFE_MERGE_SPILL=0 switches the spill off (the gate then parks in RAM always), =1 spills EVERY parked
    // partition with sub-grids (the tests' hook); DTFE_MERGE_STAGE_GB=0 parks nothing, RAM or disk. The 'ordered
    // merge:' line counts the spills (number, most at once, GB written). A spilled partition has left the gate
    // like a parked one; the tuner's model does not count parked sub-grids (the gate keeps them honest).
    bool const orderedMerge = state.options.psOrderedMerge and not fixedMerge;
    std::mutex mergeMutex;
    std::condition_variable mergeCv;
    int nextMerge = 0;
    std::map<int, StagedPartition> staged;      // finished partitions waiting for their turn (under mergeMutex)
    double stagedBytes = 0., stagedPeakBytes = 0.;
    size_t stagedPeakCount = 0, stagedRamCount = 0;   // parked in RAM: at most / now
    int nStaged = 0, nWaitedTurn = 0;                  // parked (RAM or disk); waited for their turn instead
    double stageCapBytes = -1.;                 // DTFE_MERGE_STAGE_GB: a fixed cap; < 0 = the free-memory rule
    if ( char const *cap = std::getenv( "DTFE_MERGE_STAGE_GB" ) )
        if ( cap[0] != '\0' ) stageCapBytes = std::max( 0., std::atof( cap ) ) * 1.e9;
    bool const staging = orderedMerge and psConcurrency > 1 and stageCapBytes != 0.;
    // the bytes a partition's sub-grids hold (every accumulator, as the checkpoint counts them)
    auto subgridBytes = [](Quantities &q)
    {
        double b = 0.;
        PartitionCheckpoint::forEachAccumulator( q, [&](char const *, auto &v) { b += double( v.capacity() ) * double( sizeof( v[0] ) ); } );
        return b;
    };
    // may a partition of 'bytes' be staged now? (under mergeMutex) The gate guards every start when it has an
    // estimate and the partitions are gated (the omp loop below: not the speed split, unless asked for)
    bool const gateGuards = psConcurrency > 1 and state.options.psPartitionBytes > 0.
                            and ( not state.options.psSpeedSplit or state.options.psOrderedMergeFlag );
    // the spill (the comment above): DTFE_MERGE_SPILL=0 off, =1 every parked partition, else the rule
    int spillMode = 0;
    if ( char const *sp = std::getenv( "DTFE_MERGE_SPILL" ) )
        spillMode = sp[0] == '0' ? -1 : sp[0] == '1' ? 1 : 0;
    std::string spillWhy;
    std::string const spillDir = PSMergeSpill::folder( state.options.scratchDir, spillWhy );
    bool spillUsable = staging and spillMode >= 0 and spillWhy.empty();
    if ( spillMode < 0 ) spillWhy = "DTFE_MERGE_SPILL=0";
    int nSpilled = 0, nSpillTurn = 0;
    size_t spilledCount = 0, spilledPeakCount = 0;
    double spilledBytes = 0., spilledPeakBytes = 0., spillWritten = 0.;
    // the spills being written now (outside the lock): their bytes are not on the disk yet, so the room each new
    // one is checked against must leave them out (k writers would pass one reading together)
    double spillPendingBytes = 0.;
    // where a partition of 'bytes' parks now (under mergeMutex): 1 RAM, 2 disk, 0 nowhere (it waits its turn)
    auto parkWhere = [&](double const bytes) -> int
    {
        if ( not staging ) return PARK_WAIT;
        // spoken-for room on the spill disk: the writes in progress, and with '--scratch-dir' (the spill folder
        // then) every scratch mapping in full -- a sparse file whose unwritten pages statvfs reports free
        // (over-counting the pages already written only spills less: the fallback is the park or the wait)
        double const reserved = spillPendingBytes
                                + ( state.options.scratchDir.empty() ? 0. : double( ScratchAlloc::scratchLiveBytes() ) );
        bool const disk = spillUsable and bytes > 0. and PSMergeSpill::room( spillDir, bytes, reserved,
                                                                                PSMergeSpill::margin( state.options.scratchDir ) );
        if ( spillMode > 0 and disk ) return PARK_DISK;   // the test hook
        bool ram;
        if ( stageCapBytes > 0. ) ram = stagedBytes + bytes <= stageCapBytes;
        else
        {
            double const freeNow = autoTuneSystemFree();
            if ( freeNow < 0. ) ram = stagedBytes + bytes <= 2.e9;   // free memory unknown: a small fixed room
            else ram = freeNow - 4.e9 - ( gateGuards ? state.options.psPartitionBytes : 0. ) >= bytes;
            // and within what the memory budget leaves beside the plan's predicted peak (psParkRoom says
            // why: the free memory alone could let TNG100 z=0 park 74 GB). stagedBytes counts RAM-parked ones only
            if ( ram and std::isfinite( state.options.psParkRoom ) )
                ram = stagedBytes + bytes <= std::max( 0., state.options.psParkRoom );
        }
        if ( ram ) return PARK_RAM;
        if ( disk ) return PARK_DISK;
        return ( gateGuards and stageCapBytes < 0. ) ? PARK_RAM : PARK_WAIT;
    };
    // test hook: DTFE_TEST_DELAY_PARTITION=i:s holds partition i for s seconds before its merge, so the partitions
    // after it finish first and the staging is certain to happen (tests/ps_runtime_check.sh R7)
    int delayPartition = -1;
    double delaySeconds = 0.;
    if ( char const *hook = std::getenv( "DTFE_TEST_DELAY_PARTITION" ) )
        if ( std::sscanf( hook, "%d:%lf", &delayPartition, &delaySeconds ) != 2 ) delayPartition = -1;
    std::string parkRoomText;   // the RAM parking's budget room (psParkRoom), when one is known
    PSLoopCtx loop{ state, userOptions, uQuantities, aQuantities, message, totalPartitions, plan, psConcurrency,
                    skipPartition, fixedMerge, fxMerge, toProcess, psLoopStart, psDone, partCost, costTotal, costDone,
                    orderedMerge, mergeMutex, mergeCv, nextMerge, staged, stagedBytes, stagedPeakBytes,
                    stagedPeakCount, stagedRamCount, nStaged, nWaitedTurn, stageCapBytes, staging, gateGuards,
                    spillMode, spillWhy, spillDir, spillUsable, nSpilled, nSpillTurn, spilledCount, spilledPeakCount,
                    spilledBytes, spilledPeakBytes, spillWritten, spillPendingBytes, parkRoomText };
    psReportMergeMode( loop );

    ConcurrentTriangulationPlan triangulationPlan( psConcurrency, totalPartitions );
    psGpuFallbacksTaken( true );   // counted from here; reported once after the loop
    // a partition starts only when the machine has the memory for it (or nothing else is in flight);
    // under the ordered merge it waits only behind lower-numbered partitions (auto_tune.h)
    // (under '--fixed-point-merge' orderedMerge is false -- no partition waits for its turn, so the gate
    // needs no ordered mode: it waits while any partition is in flight, which the dynamic schedule makes the same
    // set, every one in flight being lower-numbered)
    PartitionMemoryGate memoryGate( psConcurrency > 1 ? state.options.psPartitionBytes : 0., orderedMerge );

    // One partition's work, in two stages: PREPARE (select its particles; with 'build', also its
    // tessellation and vertex densities) and DEPOSIT (the grid passes, then the merge). The look-ahead
    // below runs the prepare stage of partition k+1 on another thread while partition k deposits.
    auto prepare = [&](int const pi, bool const build, PSPrepared &P)
    {
        auto const t0 = std::chrono::steady_clock::now();
        // unpadded Lagrangian region for this partition (ownership) and the padded selection box
        Box lagRegion, lagPadded;
        plan.regions( pi, lagRegion, lagPadded );

        // select particles by Lagrangian position; when periodic, this partition's images
        // are generated on the fly so the global array stays originals-only
        P.skipped = skipPartition[size_t(pi)] != 0;   // --ps-window: never reaches the window
        // a partition whose cached tessellation matches its descriptor is loaded, not built: its
        // particles are never needed, so their selection (a scan of every particle, ~3 s for TNG100-3's
        // 94M, plus a copy of the padded share) is skipped -- the load must then succeed (psCacheOnly)
        P.opt = partitionOptions( lagRegion );
        P.cacheOnly = not P.skipped and not state.options.tessellationCacheDir.empty()
                      and psTessellationCached( P.opt );
        if ( not P.skipped and not P.cacheOnly and state.options.psParticlesDeferred )
            throwError( "internal: a partition needs its particles, but they were not read (the globals record path)." );
        if ( not P.skipped and not P.cacheOnly )
            findParticlesInBoxLagrangianPeriodic( state.particles, &P.part, lagPadded,
                                                  plan.eulerLen, plan.periodicCopies, 0 );
        // an empty (or skipped) partition builds nothing but still takes its turn in the ordered merge
        P.emptyPartition = P.skipped or ( not P.cacheOnly and P.part.empty() );
        P.opt.psCacheOnly = P.cacheOnly;
        if ( build and not P.emptyPartition )
        {
            P.opt.verboseLevel = 0;   // the look-ahead thread is silent; the progress line reports it
            P.pre = psPrebuildTessellation( &P.part, P.opt );
        }
        P.seconds = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
    };
    auto deposit = [&](int const pi, PSPrepared &P, int const verbose)
    {
        Quantities temp_uQuantities, temp_aQuantities;
        if ( not P.emptyPartition )
        {
            // per-partition options use the full Eulerian grid; only the master thread logs
            P.opt.verboseLevel = verbose;
            PSMassAudit::partition = pi;   // names the audit lines (this thread runs the passes)
            psPointEvalSetPartition( pi ); // tags the '--sample-points' records (the checkpoint saves them per partition)
            if ( P.pre )
                psInterpolatePrebuilt( P.pre, samples, P.opt, &temp_uQuantities, &temp_aQuantities );
            else   // PS-DTFE: DTFE_parallel calls serial interpolation directly (no nested OpenMP region)
                DTFE_parallel( &P.part, samples, P.opt, &temp_uQuantities, &temp_aQuantities );
            PSMassAudit::partition = -1;
            psPointEvalSetPartition( -1 );
        }

        if ( pi == delayPartition and delaySeconds > 0. )   // the test hook above
            std::this_thread::sleep_for( std::chrono::duration<double>( delaySeconds ) );
        // accumulate into the shared grids: in partition order (the staged ordered merge above, the default),
        // or as they finish under '--completion-order-merge'
        psMergeInTurn( loop, subgridBytes, parkWhere, pi, P, temp_uQuantities, temp_aQuantities );
    };

    psRunPartitionLoop( loop, memoryGate, prepare, deposit );
    psReportLoopEnd( loop );

    // every partition copied its own particles (tempPart); the global array is not needed by
    // the normalization, post-processing or write phases -- free its ~48 B/particle now
    std::vector<Particle_data>().swap( state.particles );

    psNormalizeMoments( state, uQuantities, aQuantities, auditInputMass );

    psAggregateReport( uQuantities, aQuantities, totalPartitions, message );
}
#endif // PHASE_SPACE

#if NO_DIM==3 && !defined(PHASE_SPACE)
// '--serve --partition', standard DTFE: every Eulerian partition to the composite server, which serves; no return.
static void dtfeServeComposite(DTFE_State &state, User_options &userOptions, int &totalPartitions)
{
    // the load-balancing split works on particle counts per grid cell; a server has no
    // grid of its own (2^3), so the split gets a private 64^3 one
    User_options splitOpt = state.options;
    for (int d=0; d<NO_DIM; ++d)
        splitOpt.gridSize[d] = std::max<size_t>( 64, state.options.partition[d] );
    std::vector< std::vector<size_t> > subgridList;
    std::vector< Box > subgridCoords;
    optimalPartitionSplit( state.particles, splitOpt, state.options.partition, &subgridList, &subgridCoords );

    int concurrency = totalPartitions;
    if ( userOptions.maxConcurrent > 0 and userOptions.maxConcurrent < concurrency )
        concurrency = userOptions.maxConcurrent;
#ifdef OPEN_MP
    concurrency = std::max( 1, std::min( concurrency, omp_get_max_threads() ) );
#endif
    psServeCompositeBegin( state.options, totalPartitions );
#ifdef OPEN_MP
    #pragma omp parallel for schedule(dynamic, 1) num_threads(concurrency)
#endif
    for (int i=0; i<totalPartitions; ++i)
    {
        User_options tempOpt = splitOpt;
        tempOpt.partNo = i;
        copySubgridInformation( &tempOpt, subgridList, subgridCoords );
        tempOpt.paddedBox = tempOpt.region;
        tempOpt.paddedBox.addPadding( tempOpt.paddingLength );
        User_options selectOpt = tempOpt;      // periodic as given: findParticlesInBox adds the images
        tempOpt.periodic = false;
        int tnum = 0;
#ifdef OPEN_MP
        tnum = omp_get_thread_num();
#endif
        tempOpt.verboseLevel = (tnum == 0) ? std::min( userOptions.verboseLevel, 1 ) : 0;

        double ownLo[NO_DIM], ownHi[NO_DIM];
        standardPartitionOwnBox( subgridList[size_t(i)], splitOpt.gridSize, tempOpt.region, ownLo, ownHi );
        psServeCompositeAddPartition( i, tempOpt,
            [&](std::vector<Particle_data> &out) { findParticlesInBox( state.particles, &out, selectOpt ); },
            ownLo, ownHi );
    }
    std::vector<Particle_data>().swap( state.particles );
    psServeCompositeRun( state.options );   // answers requests until stdin closes; never returns
}
#endif

// Standard DTFE --partition: the spatial partitions interpolated in turn (with the look-ahead) into the full grids.
static void dtfePartitionedRun(DTFE_State &state, vector<Sample_point> &samples, User_options &userOptions,
                               Quantities *uQuantities, Quantities *aQuantities)
{
    int totalPartitions = 1;
    for (int d=0; d<NO_DIM; ++d)
        totalPartitions *= state.options.partition[d];
    MESSAGE::Message message( userOptions.verboseLevel );
    message << "The program will interpolate the fields in the region of interest using " << totalPartitions << " partitions defined via the grid [" << MESSAGE::printElements( userOptions.partition, "," ) << "].\n" << MESSAGE::Flush;

#if NO_DIM==3 && !defined(PHASE_SPACE)
    // '--serve --partition': a composite server of Eulerian partition tessellations. Each
    // partition owns the tetrahedra whose centroid lies in its (unpadded) region -- the outer
    // faces of a non-periodic box open to infinity, so hull tetrahedra keep an owner.
    if ( state.options.psServe )
        dtfeServeComposite( state, userOptions, totalPartitions );   // never returns
#endif

    uQuantities->reserveMemory( &(state.options.gridSize[0]), state.options.uField );
    aQuantities->reserveMemory( &(state.options.gridSize[0]), state.options.aField );

    std::vector< std::vector<size_t> > subgridList;
    std::vector< Box > subgridCoords;
    optimalPartitionSplit( state.particles, state.options, state.options.partition, &subgridList, &subgridCoords );

    /* Crash resume, as PS-DTFE's and through the same code (partition_checkpoint.h):
       after a partition is copied into the full grids they are checkpointed (unless the last write took more than
       5% of the time since it) with the list of the partitions in them, to '<output>.runlog.checkpoint' or under
       '--scratch-dir'; '--resume' loads them and computes only the partitions the list lacks. That is the whole
       state: each partition ASSIGNS its own disjoint sub-grid (copySubgridResultsToMain), the split is a function
       of the particles and the options (both in the settings key), nothing else carries from one partition to the
       next, and the T-web / V-web are made after the loop from the density / gradient grids saved here. So a
       resumed CPU run's outputs are byte-identical to an uninterrupted run's at the same thread count; '--gpu'
       ones too (its sums are exact; within rounding under DTFE_METAL_FLOAT_GRIDS=1). The
       progress counts the restored partitions as done (the sidecar's done/total and the '[partitions k/N]' line:
       N is every partition) and the ETA from this process's own rate. */
    PartitionCheckpoint::Checkpointer &checkpoint = PartitionCheckpoint::state();
    int const nRestored = PartitionCheckpoint::beginLoop( state.options, totalPartitions, *uQuantities, *aQuantities,
                                                          psPointEvalActive(), userOptions.verboseLevel );
    runProgressPhase( "partitions", nRestored, totalPartitions );
    int nComputed = 0;   // partitions computed by this process

    // '--sample-points': each partition answers the points inside
    // the tetrahedra it OWNS (Eulerian centroid in its sub-box, standardPartitionOwnBox), so every point is answered
    // once, by the tetrahedron the single tessellation would use. DTFE_parallel then builds ONE triangulation per
    // partition (no thread sub-boxes, whose padded overlaps would answer a point twice), as a single-domain run does
    if ( psPointEvalActive() )
        message << MESSAGE::cBold() << "DTFE:" << MESSAGE::cReset() << " point evaluation in " << totalPartitions
                << " partitions: each triangulates its padded sub-box once and answers the points in the "
                << "tetrahedra whose centroid lies in its sub-box.\n" << MESSAGE::Flush;

    auto const partLoopStart = std::chrono::steady_clock::now();

    // One partition's work in two stages (as the PS loop's): PREPARE selects its particles (its padded
    // box, periodic images included) and, for the look-ahead, splits it into DTFE_parallel's thread sub-domains and
    // builds their tessellations; DEPOSIT interpolates (building the tessellations in place when nothing was
    // prepared), copies the partition's sub-grid into the full grids, checkpoints and reports. Without the
    // look-ahead each partition runs prepare + deposit in turn.
    struct StdPrepared
    {
        User_options opt;              // the partition's options (its sub-grid, region and padded box)
        vector<Particle_data> part;    // its particles (none once the look-ahead has built from them)
        bool built = false;            // the look-ahead built its sub-domains (sub)
        DTFEPrepared sub;
        double seconds = 0.;           // the look-ahead's time for it
    };
    int const parallelThreads = dtfeParallelThreads( state.options );   // DTFE_parallel's, read on THIS thread
    auto prepare = [&](int const i, bool const build, StdPrepared &P)
    {
        auto const t0 = std::chrono::steady_clock::now();
        P.opt = state.options;
        P.opt.partNo = i;
        copySubgridInformation( &P.opt, subgridList, subgridCoords );

        P.opt.paddedBox = P.opt.region;
        P.opt.paddedBox.addPadding( P.opt.paddingLength );

        findParticlesInBox( state.particles, &P.part, P.opt );
        P.opt.periodic = false;
#ifndef PHASE_SPACE
        if ( build )
        {
            dtfePrepareSubdomains( &P.part, samples, P.opt, parallelThreads, P.sub );
            P.built = true;
        }
#else
        (void)build;
#endif
        P.seconds = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
    };
    auto deposit = [&](int const i, StdPrepared &P)
    {
        message << MESSAGE::cBold() << "\n<<< Interpolating the fields for partition " << i+1 << "/" << totalPartitions << " ...\n" << MESSAGE::cReset();
        User_options &tempOpt = P.opt;

        if ( psPointEvalActive() )   // this partition's ownership (above); the loop is serial
        {
            double ownLo[NO_DIM], ownHi[NO_DIM];
            standardPartitionOwnBox( subgridList[size_t(i)], state.options.gridSize, tempOpt.region, ownLo, ownHi );
            psPointEvalSetOwnership( ownLo, ownHi );
        }
        Quantities temp_uQuantities, temp_aQuantities;
        // '--sample-points': tag this partition's point records, as the PS loop does, so the checkpoint
        // saves them per partition ('points.<i>', partition_checkpoint.h) and '--resume' reads them back. The tag is
        // thread-local; under point evaluation DTFE_parallel triangulates the partition once ON THIS THREAD (its
        // single-triangulation branch), and the point walk copies the tag into its sinks before it starts its own
        // threads (ps_point_eval.cc forEachCellParallel). A no-op without '--sample-points'. (The look-ahead only
        // BUILDS on its own thread; the interpolation, and so the walk, stays on this one.)
        psPointEvalSetPartition( i );
        DTFE_parallelImpl( &P.part, samples, tempOpt, &temp_uQuantities, &temp_aQuantities, P.built ? &P.sub : nullptr );
        psPointEvalSetPartition( -1 );

        // map this partition's sub-grid results back into the full output grids
        copySubgridResultsToMain( temp_uQuantities, state.options.gridSize, tempOpt.uField, tempOpt, subgridList, uQuantities );
        copySubgridResultsToMain( temp_aQuantities, state.options.gridSize, tempOpt.aField, tempOpt, subgridList, aQuantities );
        // the checkpoint (partition_checkpoint.h): the full grids now hold this partition too
        PartitionCheckpoint::afterMerge( i, false, *uQuantities, *aQuantities, userOptions.verboseLevel );

        // progress + ETA over the serial partition loop
        double const elapsed = std::chrono::duration<double>( std::chrono::steady_clock::now() - partLoopStart ).count();
        ++nComputed;
        int    const done    = nRestored + nComputed;
        // (the split balances the particle counts, so counting partitions is the cost weighting here; the rate
        // is this process's own, the restored partitions cost it nothing)
        double const eta     = elapsed * ( double(totalPartitions - done) / double(nComputed) );
        runProgressPhase( "partitions", done, totalPartitions, eta );
        message << MESSAGE::cGreen() << "  [partitions " << done << "/" << totalPartitions
                << " | " << int(100.*done/double(totalPartitions) + 0.5) << "%]" << MESSAGE::cReset()
                << MESSAGE::cDim() << "  elapsed " << MESSAGE::formatDuration(elapsed)
                << ( done<totalPartitions ? ", ETA ~" + MESSAGE::formatDuration(eta) : ", done" )
                << MESSAGE::cReset() << "\n" << MESSAGE::Flush;
    };

    // the partitions this process computes, in index order (a resumed run's restored ones are in the grids already)
    std::vector<int> toCompute;
    for (int i=0; i<totalPartitions; ++i)
        if ( not checkpoint.merged[size_t(i)] ) toCompute.push_back( i );

    /* The LOOK-AHEAD (the PS loop's, DTFE.cpp above, for the standard loop). Without it, with '--gpu'
       every partition's volume-averaged pass waits for the partition's particle selection and the Delaunay
       triangulations + vertex densities of all its thread sub-boxes, while the GPU idles. So partition k+1 is
       prepared -- selected, split into DTFE_parallel's sub-domains and triangulated (or loaded from
       '--tessellation-cache'), on a team of CPU threads of its own -- while partition k is interpolated, at most
       one ahead. Every sub-domain is built from the same particles with the same options as without the
       look-ahead (dtfePrepareSubdomains: DTFE_parallel's own split and options, the sequential insert), the
       interpolation is unchanged, and each partition still ASSIGNS its own part of the grids in index order, so
       every output is byte-identical (tests/dtfe_resume_check.sh D14). It holds the next partition's particle copy
       and tessellations; before each look-ahead the free memory, less 4 GB of headroom, is checked against that
       (auto_tune.h stdLookAheadBytes, the uniform count of one partition, or the auto-tuner's count when larger),
       and the partition is prepared in turn when it does not fit; the auto-tuner counts it in its prediction and
       switches it off for a run whose budget cannot hold it (stdLookAheadOffBudget). '--no-overlap' switches it off; DTFE_OVERLAP=1 forces it on
       a CPU run too (the tests' byte-identity check on any build), DTFE_OVERLAP=0 switches it off. A '--resume'
       looks ahead to the next partition it computes, not to the next index. */
    bool gpuDeposit = false;
#if defined(DTFE_GPU) && NO_DIM==3 && !defined(PHASE_SPACE)
    gpuDeposit = state.options.useMetal and state.options.method==1 and state.options.aField.selected()
                 and not state.options.redshiftConeOn and not state.options.userDefinedSampling;
#endif
    bool overlap = false;
    double aheadBytesMax = 0.;
    std::function<double(int)> aheadBytes = [](int) { return 0.; };
#ifndef PHASE_SPACE
    {
        char const *overlapEnv = std::getenv( "DTFE_OVERLAP" );
        bool const overlapForced = overlapEnv and overlapEnv[0] == '1';
        bool const overlapBanned = overlapEnv and overlapEnv[0] == '0';
        overlap = toCompute.size() > 1 and state.options.DTFE and not state.options.stdNoOverlap
                  and not state.options.stdLookAheadOffBudget and not overlapBanned and ( gpuDeposit or overlapForced );
    }
    // what one partition's look-ahead holds: its padded particle copy + the tessellations of its sub-domains
    // (auto_tune.h, the model's bytes per padded vertex); the partitions hold about the same particles (the split
    // balances them), so the uniform count of the partition's geometry -- or the auto-tuner's count from the
    // particles, of the partition that needs the most, when it is larger (on clustered data the padding shells
    // of a dense partition hold more than the uniform count: auto_tune.h StdTuneDensity)
    aheadBytes = [&](int const i) -> double
    {
        User_options o = state.options;
        o.partNo = i;
        copySubgridInformation( &o, subgridList, subgridCoords );
        o.verboseLevel = 0;
        size_t tGrid[NO_DIM];
        for (int d=0; d<NO_DIM; ++d) tGrid[d] = 1;
        if ( not dtfeParallelSingle( o, samples, parallelThreads ) )
            parallelGrid( parallelThreads, o, tGrid );
        double len[NO_DIM], pad[NO_DIM];
        for (int d=0; d<NO_DIM; ++d)
        {
            len[d] = double( o.region[2*d+1] - o.region[2*d] );
            pad[d] = double( o.paddingLength[2*d] + o.paddingLength[2*d+1] ) / 2.;
        }
        return std::max( state.options.stdLookAheadCounted,
                         stdLookAheadBytes( state.options, double( state.particles.size() ) / double( totalPartitions ),
                                            len, pad, tGrid ) );
    };
    if ( overlap )
    {
        for (int i : toCompute) aheadBytesMax = std::max( aheadBytesMax, aheadBytes( i ) );
        // the memory budget too, not only the free memory: what it leaves beside the grids, the particles and one
        // partition must hold the look-ahead, or the run has none (the same for every partition, decided once)
        if ( aheadBytesMax > state.options.stdLookAheadRoom )
        {
            overlap = false;
            message << MESSAGE::cBold() << "DTFE:" << MESSAGE::cReset() << " look-ahead off: the ~"
                    << std::round( aheadBytesMax/1.e7 )/100. << " GB it holds is more than the ~"
                    << std::round( std::max( 0., state.options.stdLookAheadRoom )/1.e7 )/100. << " GB the memory "
                    << "budget leaves beside one partition; partitions are prepared in turn.\n" << MESSAGE::Flush;
        }
    }
#endif
    if ( overlap )
    {
        message << MESSAGE::cBold() << "DTFE:" << MESSAGE::cReset() << " look-ahead on: partition k+1 is "
                << "triangulated on CPU threads while partition k is interpolated"
                << ( gpuDeposit ? " on the GPU" : " (DTFE_OVERLAP=1)" ) << " (~" << std::round( aheadBytesMax/1.e7 )/100.
                << " GB more in memory; '--no-overlap' switches it off).\n" << MESSAGE::Flush;

        StdPrepared cur;
        prepare( toCompute[0], false, cur );
        bool saidNoRoom = false;
        for (size_t k=0; k<toCompute.size(); ++k)
        {
            int const i = toCompute[k];
            StdPrepared next;
            std::thread ahead;
            std::exception_ptr aheadError;
            bool aheadRan = false;
            if ( k + 1 < toCompute.size() )
            {
                int const nextI = toCompute[k+1];
                double const need = aheadBytes( nextI );
                double const freeNow = autoTuneSystemFree();
                bool const room = freeNow < 0. or freeNow - 4.e9 >= need;
                if ( room )
                {
                    aheadRan = true;
                    ahead = std::thread( [&, nextI]()
                    {
                        try { prepare( nextI, true, next ); }
                        catch ( ... ) { aheadError = std::current_exception(); }
                    } );
                }
                else if ( not saidNoRoom )
                {
                    saidNoRoom = true;
                    MESSAGE::Message m( userOptions.verboseLevel );
                    m << MESSAGE::cDim() << "  look-ahead paused: " << freeNow/1.e9 << " GB free, minus 4 GB headroom, "
                      << "is below the ~" << need/1.e9 << " GB it holds; partitions are prepared in turn "
                      << "until memory frees up\n" << MESSAGE::cReset() << MESSAGE::Flush;
                }
            }
            try
            {
                deposit( i, cur );
            }
            catch ( ... )
            {
                if ( ahead.joinable() ) ahead.join();
                throw;
            }
            if ( ahead.joinable() ) ahead.join();
            if ( aheadError ) std::rethrow_exception( aheadError );
            if ( aheadRan and userOptions.verboseLevel > 0 )
            {
                MESSAGE::Message m( userOptions.verboseLevel );
                m << MESSAGE::cDim() << "  look-ahead: partition " << toCompute[k+1] + 1 << " prepared in "
                  << MESSAGE::formatDuration( next.seconds ) << " during this one\n" << MESSAGE::cReset() << MESSAGE::Flush;
            }
            if ( k + 1 < toCompute.size() )
            {
                if ( not aheadRan ) prepare( toCompute[k+1], false, next );
                cur = std::move( next );
            }
        }
    }
    else
        for (int i : toCompute)
        {
            StdPrepared P;
            prepare( i, false, P );
            deposit( i, P );
        }
    PartitionCheckpoint::reportWrites( std::chrono::duration<double>( std::chrono::steady_clock::now() - partLoopStart ).count(),
                                       userOptions.verboseLevel );

    // each partition copied its own particles; the global array is dead weight from here on
    std::vector<Particle_data>().swap( state.particles );
    if ( psPointEvalActive() )
        psPointEvalSetOwnership( nullptr, nullptr );   // back to the single-triangulation default
}

// After the interpolation: the point records reduced, the masks finalized, the web fields made, the counts reported.
static void dtfeFinishFields(DTFE_State &state, User_options &userOptions, Quantities *uQuantities,
                             Quantities *aQuantities)
{
    // all partitions (or the single triangulation) have contributed their streams: reduce the
    // per-point records to the final density/velocity/dispersion/stream-count outputs
    if ( psPointEvalActive() )
        psPointEvalFinalize( state.options );

#ifdef PHASE_SPACE
    // --ps-caustics: binarize the orientation bits ONLY here, after every partition has been
    // OR-merged -- a fold straddling a partition boundary gets its two orientations from
    // different partitions, so binarizing any earlier would lose it. bits==3 (both parities
    // overlap) -> 1, else 0. Not idempotent (1 -> 0 on a second pass): DTFE() runs once per
    // invocation for this flag (--interlace is rejected at option parsing).
    /* The caustic mask is deliberately NOT binarized here. It stays full until writeOutputData,
       which writes '.causticClass' from it and only then collapses it in place for '.caustic' --
       one grid instead of two, saving 4 B/cell per Quantities across the post-processing window
       where peak RSS is set. The write order there is load-bearing. */
    // '.hidden_streams' bit 1: likewise only now, with '.streams' summed over every partition
    uQuantities->finalizeHiddenStreams();
    aQuantities->finalizeHiddenStreams();
#endif

    // post-processing: derive velocity divergence/shear/vorticity and cosmic-web labels from the gradient
    runProgressPhase( "post-processing" );
    postProcessWebFields( userOptions, uQuantities, aQuantities );

#ifndef PHASE_SPACE
    // --gpu: the sub-domain deposits that fell back to the CPU (as the PS binary's partition count
    // after its loop): each prints its reason as a warning of its own thread, and this line says how many there
    // were. Once per DTFE() call.
    if ( int const nFallback = dtfeGpuFallbacksTaken( true ) )
    {
        MESSAGE::Warning warning( userOptions.verboseLevel );
        warning << nFallback << " sub-domain deposit(s) asked for the GPU and ran on the CPU instead (the reasons "
                << "are above); the grids are complete, the run only took longer.\n" << MESSAGE::EndWarning;
    }
#endif

    // the failed GPU buffers / kernels of the run by cause, one line the launcher's Runs tab
    // reads (the progress sidecar that carries the same counts is removed when the run succeeds). The counts
    // are cumulative over the process, so '--interlace' (two DTFE() calls) prints the line twice and the last
    // one holds the run's total. Both binaries (the standard GPU hosts count by the same causes); nothing is
    // printed without a failed buffer, so a CPU run's log has no such line.
    if ( int const nRetry = runProgressGpuRetries() )
    {
        MESSAGE::Message retries( userOptions.verboseLevel );
        retries << "GPU deposit retries: " << nRetry << " (innocent " << runProgressGpuRetries( 0 ) << ", timeout "
                << runProgressGpuRetries( 1 ) << ", oom " << runProgressGpuRetries( 2 ) << ", other "
                << runProgressGpuRetries( 3 ) << ")\n" << MESSAGE::Flush;
    }

    // the shares the non-finite guard of the exact sums dropped in this DTFE() call, by grid
    fixedShareDropsReact( userOptions );
}

// Top-level DTFE interpolation onto a grid; dispatches to the standard, partitioned, or PS-DTFE path. Clears allParticles.
void DTFE(vector<Particle_data> *allParticles,
          vector<Sample_point> &samples,
          User_options & userOptions,
          Quantities *uQuantities,
          Quantities *aQuantities)
{
    DTFE_State state = DTFE_setup( allParticles, samples, userOptions );

    // approxPSD: compute velocity-space density g_i for all particles before
    // partitioning/padding so the velocity tessellation uses the global distribution
#if defined(VELOCITY) && defined(SCALAR) && !defined(PHASE_SPACE)
    if ( state.options.approxPSD && !state.particles.empty() )
        computeVelocitySpaceDensity( state.particles, state.options );
#endif

#ifdef PHASE_SPACE
    // PS-DTFE periodic padding: triangulation is built in Lagrangian space, so periodic
    // copies use shifted Lagrangian (not Eulerian) positions.
    Box lagBoxGlobal;  // saved for later use by partitioning
    lagBoxGlobal.assign(Real(0.));
    bool hasLagrangianPeriodicCopies = false;
    psCloudNormalization( state, userOptions );
    psLagrangianPeriodicSetup( state, userOptions, lagBoxGlobal, hasLagrangianPeriodicCopies );
#endif

    // arbitrary-point evaluation (--sample-points), both binaries: load the query points and
    // build their bucket index once, now that region/periodic/averageDensity are final. Each
    // triangulation then contributes its streams; psPointEvalFinalize() reduces them at the end.
    // --serve arms the same context without points; each request supplies its own.
    if ( not state.options.psSamplePointsFile.empty() or state.options.psServe )
        psPointEvalInit( state.options );

#ifdef PHASE_SPACE
    psResolveWindow( state, userOptions );
#endif

#ifdef PHASE_SPACE
    if ( userOptions.partitionOn and userOptions.partNo<0 )
    {
        // PS-DTFE partitions in Lagrangian space (Eulerian partitioning fails because Lagrangian cells
        // map to arbitrary Eulerian locations). Each partition writes the full grid; results are summed.
        psPartitionedRun( state, samples, userOptions, uQuantities, aQuantities, lagBoxGlobal,
                          hasLagrangianPeriodicCopies );
    }
    else
#endif // PHASE_SPACE

    // standard DTFE: process the spatial partitions one at a time to bound peak memory on large data
    if ( userOptions.partitionOn and userOptions.partNo<0 )
    {
        dtfePartitionedRun( state, samples, userOptions, uQuantities, aQuantities );
    }
    else
    {
        // single triangulation over the whole (padded) box
        DTFE_preparePadding( state, userOptions );
        DTFE_parallel( &state.particles, samples, state.options, uQuantities, aQuantities );
    }

    dtfeFinishFields( state, userOptions, uQuantities, aQuantities );
}



// Builds the Delaunay triangulation and interpolates the fields onto the grid (DTFE method).
extern void DTFE_interpolation(vector<Particle_data> *p,
                               vector<Sample_point> &samples,
                               User_options &userOptions,
                               Quantities *uQuantities,
                               Quantities *aQuantities);

// Dispatches to the interpolation method selected in userOptions (DTFE, NGP, CIC, TSC, PCS, SPH).
void interpolate(vector<Particle_data> *allParticles,
                 vector<Sample_point> &samples,
                 User_options & userOptions,
                 Quantities *uQuantities,
                 Quantities *aQuantities)
{
    if ( userOptions.DTFE )
        DTFE_interpolation( allParticles, samples, userOptions, uQuantities, aQuantities );
    else if ( userOptions.NGP )
        NGP_interpolation( allParticles, samples, userOptions, aQuantities );
    else if ( userOptions.CIC )
        CIC_interpolation( allParticles, samples, userOptions, aQuantities );
    else if ( userOptions.TSC )
        TSC_interpolation( allParticles, samples, userOptions, aQuantities );
    else if ( userOptions.PCS )
        PCS_interpolation( allParticles, samples, userOptions, aQuantities );
    else if ( userOptions.SPH )
        SPH_interpolation( allParticles, samples, userOptions, aQuantities );
    else
        throwError( "Unknow interpolation method in function 'interpolate'." );
}




// Splits the data into spatial partitions computed in parallel via OpenMP.
void DTFE_parallel(vector<Particle_data> *allParticles,
                   vector<Sample_point> &samples,
                   User_options & userOptions,
                   Quantities *uQuantities,
                   Quantities *aQuantities)
{
    DTFE_parallelImpl( allParticles, samples, userOptions, uQuantities, aQuantities, nullptr );
}

// ... the body; 'prepared' (the standard partition loop's look-ahead) holds the split and the sub-domains'
// tessellations already built from these options (dtfePrepareSubdomains), so only their interpolation is left
static void DTFE_parallelImpl(vector<Particle_data> *allParticles,
                              vector<Sample_point> &samples,
                              User_options & userOptions,
                              Quantities *uQuantities,
                              Quantities *aQuantities,
                              DTFEPrepared *prepared)
{
#ifndef PHASE_SPACE
    // a prepared single tessellation (as the branches below, which interpolate() it)
    auto interpolatePrepared = [&](size_t const sub, User_options &o, Quantities *u, Quantities *a)
    {
        if ( not userOptions.DTFE )
            throwError( "internal: a look-ahead tessellation reached a non-DTFE interpolation." );
        dtfeInterpolatePrebuilt( prepared->tessellation[sub], samples, o, u, a );
    };
#else
    if ( prepared ) throwError( "internal: the standard look-ahead reached PS-DTFE's DTFE_parallel." );
    auto interpolatePrepared = [&](size_t, User_options &, Quantities *, Quantities *) {};
#endif
#ifndef OPEN_MP
    if ( prepared ) interpolatePrepared( 0, userOptions, uQuantities, aQuantities );
    else interpolate( allParticles, samples, userOptions, uQuantities, aQuantities );  // deletes allParticles
    return;
#else
    // cap concurrent sub-triangulations to bound peak memory (--max-concurrent); 0 = all threads
    int const noAvailableProcessors = dtfeParallelThreads( userOptions );

    // fall back to a single triangulation when spatial partitioning does not apply (dtfeParallelSingle: 1 thread,
    // user sampling, redshift cone, --sample-points point evaluation, PS-DTFE); a prepared partition decided it
    if ( prepared ? prepared->single : dtfeParallelSingle( userOptions, samples, noAvailableProcessors ) )
    {
        if ( prepared ) interpolatePrepared( 0, userOptions, uQuantities, aQuantities );
        else interpolate( allParticles, samples, userOptions, uQuantities, aQuantities );
        return;
    }


    // more than 1 thread: split the data into spatial partitions, one per thread
    std::vector<size_t> pGrid(NO_DIM,0); // the parallel grid
    if ( prepared ) pGrid = prepared->pGrid;
    else parallelGrid( noAvailableProcessors, userOptions, &(pGrid[0]) );
    int noProcessors = 1;
    for (int d=0; d<NO_DIM; ++d) noProcessors *= pGrid[d];    //number of processors actually used (may differ from 'noAvailableProcessors')

    uQuantities->reserveMemory( &(userOptions.gridSize[0]), userOptions.uField );
    aQuantities->reserveMemory( &(userOptions.gridSize[0]), userOptions.aField );
    MESSAGE::Message message( userOptions.verboseLevel );
    message << "From now on only the master thread will show messages on how the computation is going. Not all threads take the same execution time, so there may be a discrepancy between the messages displayed to the user and the computations across all threads.\n\n" << MESSAGE::Flush;
    std::vector<size_t> processorParticles(noProcessors);    // number of particles associated to each processor
    std::vector<Real> processorTime(noProcessors);           // the actual runtime of each thread
    size_t const noTotalParticles = prepared ? prepared->noTotalParticles : allParticles->size();
    

    // choose the partition geometry that balances the particle load across threads
    std::vector< std::vector<size_t> > subgridList;
    std::vector< Box > subgridCoords;
    if ( prepared )
    {
        subgridList = prepared->subgridList;
        subgridCoords = prepared->subgridCoords;
    }
    else
        optimalPartitionSplit( *allParticles, userOptions, pGrid, &subgridList, &subgridCoords );

    // --gpu: the sub-domains deposit into ONE full grid on the device (dtfeGpuShared*, see
    // gpu_host.h) instead of each into a sub-grid of its own -- one accumulator, one read-back,
    // no per-sub-domain zeroing, and the dispatch controller stays warm from one sub-domain to
    // the next. The threads' sub-grids then stay zero and the GPU grid is ADDED below.
    bool gpuShared = false;
#ifdef DTFE_GPU_SHARED_ACTIVE
    bool const sharedScalarOk =
#ifdef SCALAR
        noScalarComp == 1;   // the kernels carry ONE scalar component (the interpolation declines the GPU otherwise)
#else
        true;
#endif
    if ( userOptions.useMetal and userOptions.DTFE and userOptions.method==1 and userOptions.aField.selected()
         and not userOptions.exactAverage and not userOptions.redshiftConeOn and not userOptions.userDefinedSampling
         and samples.empty() and sharedScalarOk )
    {
        size_t const full[3] = { userOptions.gridSize[0], userOptions.gridSize[1], userOptions.gridSize[2] };
        std::string err;
        gpuShared = dtfeGpuSharedBegin( full, userOptions.aField.density, userOptions.aField.velocity,
                                        userOptions.aField.velocity_gradient, userOptions.aField.scalar,
                                        userOptions.aField.scalar_gradient, err );
        if ( not gpuShared and userOptions.verboseLevel>=2 )
            message << "(GPU: no shared grid, " << err << "; each sub-domain deposits on its own)\n" << MESSAGE::Flush;
    }
#endif


#pragma omp parallel num_threads( noProcessors )
    {
        int const threadNo = omp_get_thread_num();
        // region allocated to this thread (dtfeSubdomainOptions: its sub-grid, box and padded box)
        User_options tempOptions = dtfeSubdomainOptions( userOptions, threadNo, noProcessors, pGrid, subgridList, subgridCoords );


        Quantities temp_uQuantities, temp_aQuantities;
        if ( prepared )   // the look-ahead built this sub-domain's tessellation from the same options and particles
        {
            processorParticles[ threadNo ] = prepared->nParticles[ size_t(threadNo) ];
            interpolatePrepared( size_t(threadNo), tempOptions, &temp_uQuantities, &temp_aQuantities );
        }
        else
        {
            vector<Particle_data> particles;
            findParticlesInBox( *allParticles, &particles, tempOptions );
            processorParticles[ threadNo ] = particles.size();
            interpolate( &particles, samples, tempOptions, &temp_uQuantities, &temp_aQuantities ); //this function deletes the vector 'particles'
        }
        processorTime[ threadNo ] = tempOptions.totalTime;


        // copy this partition's results to the main grid
        copySubgridResultsToMain( temp_uQuantities, userOptions.gridSize, tempOptions.uField, tempOptions, subgridList, uQuantities );
        copySubgridResultsToMain( temp_aQuantities, userOptions.gridSize, tempOptions.aField, tempOptions, subgridList, aQuantities );
        if ( threadNo==0 )
            message << "\nWaiting for all threads to finish the computations ... " << MESSAGE::Flush;
    }
    message << "Done.\n";
    allParticles->clear();

#ifdef DTFE_GPU_SHARED_ACTIVE
    if ( gpuShared )
    {   // add the device's full grid (value*volume sums) into the output, divided by the cell volume
        // exactly as the per-thread path does; a sub-domain the device handed back is zero here and
        // its CPU result is already in the main grid
        const float *gDen = nullptr, *gVel = nullptr, *gGrad = nullptr, *gScal = nullptr, *gSGrad = nullptr;
        if ( dtfeGpuSharedResult( &gDen, &gVel, &gGrad, &gScal, &gSGrad ) )
        {
            Real cellVolume = 1.;
            for (int d=0; d<NO_DIM; ++d)
                cellVolume *= (userOptions.region[2*d+1] - userOptions.region[2*d]) / userOptions.gridSize[d];
            size_t const nCell = userOptions.gridSize[0] * userOptions.gridSize[1] * userOptions.gridSize[2];
            if ( gDen and aQuantities->density.size()==nCell )
            {
                #pragma omp parallel for
                for (size_t i=0; i<nCell; ++i) aQuantities->density[i] += Real(gDen[i]) / cellVolume;
            }
            if ( gVel and aQuantities->velocity.size()==nCell )
            {
                #pragma omp parallel for
                for (size_t i=0; i<nCell; ++i)
                    for (size_t j=0; j<noVelComp; ++j) aQuantities->velocity[i][j] += Real(gVel[i*3+j]) / cellVolume;
            }
            if ( gGrad and aQuantities->velocity_gradient.size()==nCell )
            {
                #pragma omp parallel for
                for (size_t i=0; i<nCell; ++i)
                    for (size_t q=0; q<noGradComp; ++q) aQuantities->velocity_gradient[i][q] += Real(gGrad[i*9+q]) / cellVolume;
            }
#ifdef SCALAR
            if ( gScal and aQuantities->scalar.size()==nCell )
            {
                #pragma omp parallel for
                for (size_t i=0; i<nCell; ++i) aQuantities->scalar[i][0] += Real(gScal[i]) / cellVolume;
            }
            if ( gSGrad and aQuantities->scalar_gradient.size()==nCell )
            {
                #pragma omp parallel for
                for (size_t i=0; i<nCell; ++i)
                    for (int q=0; q<NO_DIM; ++q) aQuantities->scalar_gradient[i][q] += Real(gSGrad[i*3+q]) / cellVolume;
            }
#endif
        }
        dtfeGpuSharedEnd();
    }
#endif


    // total runtime is the slowest thread's CPU time, since the run finishes only when all threads do
    userOptions.totalTime += maximum( processorTime.data(), noProcessors );
    approximativeThreadTime( processorTime.data(), noProcessors );
    message << "Statistics of the execution across the " << noProcessors << " threads:\n";
    for (int i=0; i<noProcessors; ++i)
        message << "\t Thread " << i << " had " << processorParticles[i] << " particles (which represent " << setprecision(4) <<  Real(processorParticles[i])/noTotalParticles*100. << "\%) and took " << processorTime[i] << " sec. \n";
    message << MESSAGE::Flush;
    
#endif
}






// shared zero placeholders for the velocity/scalar members that are compiled out
#ifndef VELOCITY
Pvector<Real,noVelComp> Data_structure::_velocity = Pvector<Real,noVelComp>::zero();
#endif
#ifndef SCALAR
Pvector<Real,noScalarComp> Data_structure::_scalar = Pvector<Real,noScalarComp>::zero();
#endif






#ifdef TRIANGULATION
#include "DTFE.h"

// Builds the triangulation and interpolates to grid, also returning the Delaunay triangulation.
extern void DTFE_interpolation(vector<Particle_data> *p,
                               vector<Sample_point> &samples,
                               User_options &userOptions,
                               Quantities *uQuantities,
                               Quantities *aQuantities,
                               DT &delaunay_triangulation);

// DTFE interpolation that also hands back the Delaunay triangulation (single-triangulation only). Clears allParticles.
void DTFE(vector<Particle_data> *allParticles,
          vector<Sample_point> &samples,
          User_options & userOptions,
          Quantities *uQuantities,
          Quantities *aQuantities,
          DT &delaunay_triangulation)
{
    DTFE_State state = DTFE_setup( allParticles, samples, userOptions );

    if ( userOptions.partitionOn and userOptions.partNo<0 )
    {
        throwError( "There is no implementation of the option '--partition' in the absence of option '--partNo' when requering that the 'DTFE' function returns the Delaunay triangulation." );
    }
    else
    {
        DTFE_preparePadding( state, userOptions );
        DTFE_interpolation( &state.particles, samples, state.options, uQuantities, aQuantities, delaunay_triangulation );
    }

    postProcessWebFields( userOptions, uQuantities, aQuantities );
    fixedShareDropsReact( userOptions );   // the non-finite guard's dropped shares, as at the end of DTFE() above
}
#endif

