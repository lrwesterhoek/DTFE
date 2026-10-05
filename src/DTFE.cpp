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

/* DTFE driver: data setup, padding, partitioning (standard and PS-DTFE), and velocity-derivative/web post-processing. */

#include <vector>
#include <cmath>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <chrono>   // partition-loop progress + ETA
#ifdef OPEN_MP
    #include <omp.h>
#endif
#include <boost/math/special_functions/fpclassify.hpp>
#include <fftw3.h>   // T-web tidal tensor (FFT Poisson solve of the density grid)
#include "fftw_plans64.h"   // ... planned with 64-bit sizes
#include <new>              // std::align_val_t (the aligned FFT buffers)
#include <complex>          // ... into std::complex buffers (FFTW guarantees the layout of fftw_complex)


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
#include "io/io.h"         // readInputData: the late read of a --ps-window run that started without its particles


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

// Derives velocity divergence, shear and vorticity from the stored velocity gradient.
void computeDivergenceShearVorticity(Field &fields,
                                     int const verboseLevel,
                                     Quantities *quantities);

// Derives the V-web cosmic-web classification from the velocity gradient.
void computeWebClassification(Field &fields,
                               int const verboseLevel,
                               Real lambda_th,
                               Real hubbleParam,
                               Real scaleFactor,
                               Quantities *q);

// Derives the T-web classification from the tidal tensor of the DENSITY grid (FFT Poisson).
void computeTidalWebClassification(Field &fields,
                                   User_options const &userOptions,
                                   Quantities *q);

// Post-processing shared by both DTFE() overloads: derive velocity divergence/shear/vorticity
// and the V-web / T-web labels for the unaveraged and averaged quantities.
void postProcessWebFields(User_options &userOptions,
                          Quantities *uQuantities,
                          Quantities *aQuantities)
{
    computeDivergenceShearVorticity( userOptions.uField, userOptions.verboseLevel, uQuantities );
    computeDivergenceShearVorticity( userOptions.aField, userOptions.verboseLevel, aQuantities );
    computeWebClassification( userOptions.uField, userOptions.verboseLevel, userOptions.lambda_th, userOptions.hubbleParam, userOptions.scaleFactor, uQuantities );
    computeWebClassification( userOptions.aField, userOptions.verboseLevel, userOptions.lambda_th, userOptions.hubbleParam, userOptions.scaleFactor, aQuantities );
    computeTidalWebClassification( userOptions.uField, userOptions, uQuantities );
    computeTidalWebClassification( userOptions.aField, userOptions, aQuantities );
}

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

        // Lagrangian padding per axis: same fraction as Eulerian, floored at 0.30 of a partition cell
        for (int d=0; d<NO_DIM; ++d)
        {
            Real padFrac = (state.options.paddingLength[2*d] + state.options.paddingLength[2*d+1]) / (Real(2.) * eulerLen[d]);
            lagPadding[d] = padFrac * lagLen[d];
            if (lagPadding[d] < lagLen[d] / state.options.partition[d] * Real(0.3))
                lagPadding[d] = lagLen[d] / state.options.partition[d] * Real(0.3);
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
            lagRegion[2*d]   = lagBox[2*d] + lagLen[d] * idx[d] / grid[d];
            lagRegion[2*d+1] = lagBox[2*d] + lagLen[d] * (idx[d]+1) / grid[d];
            // periodic: the partition regions must tile one full PERIOD, not just the
            // particle bounding box. Image tets can have Lagrangian centroids in the gap
            // between lagBox and the periodic box (half a lattice spacing for cell-centred
            // ICs); with the tiling anchored to lagBox they are owned by NO partition and
            // their mass is silently dropped (uncovered Eulerian slabs at the box edges).
            // Anchoring the first partition's lower edge one period below the last
            // partition's upper edge keeps the tiling exclusive AND exhaustive mod L.
            if (periodicCopies && idx[d] == 0)
                lagRegion[2*d] = lagBox[2*d+1] - eulerLen[d];
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
};
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
    // identically). Without this it saw the whole virtual grid of a zoom (2759x3197x512 = 524 GB),
    // called the run impossible and fell back to one partition at a time.
    if ( userOptions.psWindowOn )
        userOptions.computePsWindow();

    // auto-select --partition / --max-concurrent from the data and machine when not user-given
    {
        bool gpuActive = false;
#if defined(PS_GPU)
        gpuActive = userOptions.psUseMetal;
#elif defined(DTFE_GPU)
        gpuActive = userOptions.useMetal;
#endif
        AutoTuneReport const report = autoTunePartitioning( userOptions, nParticles,
                                                            not samples.empty(), gpuActive, particlePointer );
        if ( userOptions.autoTuneReport )   // the memory check: the decision and prediction, nothing else
        {
            std::printf( "%s\n", report.line().c_str() );
            std::fflush( stdout );
            std::exit( 0 );
        }
    }

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

        vector<Particle_data> particlesRegion;
        findParticlesInBox( *particlePointer, &particlesRegion, state.options );
        state.options.periodic = false;
        state.options.updateFullBox( state.options.region );

        particlePointer->clear();
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
        if ( not psPartitioned )
        {
            Box lagPaddedGlobal = lagBoxGlobal;
            for (int d=0; d<NO_DIM; ++d)
            {
                lagPaddedGlobal[2*d]   -= lagPad[d];
                lagPaddedGlobal[2*d+1] += lagPad[d];
            }

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

        // primary periodic box: the centroid ownership check keeps one image of each cell and tiles
        // the box; overridden per-partition in the partition path. It is ONE PERIOD ending at the top
        // particle plane, [lagBox_hi - L, lagBox_hi), as the partition plan anchors its tiling -- not
        // the snapshot box [0, L): on a lattice the box edge lies halfway between two particle planes,
        // so a tetrahedron straddling it has its centroid exactly ON the edge, and the float-rounded
        // periodic copies put BOTH its images outside (a single-precision 24^3 lattice lost 4.1% of
        // its mass, the cells along the box faces; 2026-10-02, tests/tweb_check.sh). A centroid can
        // reach the top particle plane only for a tetrahedron flat in that plane.
        for (int d=0; d<NO_DIM; ++d)
        {
            state.options.lagrangianRegion[2*d]   = lagBoxGlobal[2*d+1] - eulerLen[d];
            state.options.lagrangianRegion[2*d+1] = lagBoxGlobal[2*d+1];
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
#endif

    // arbitrary-point evaluation (--sample-points), both binaries: load the query points and
    // build their bucket index once, now that region/periodic/averageDensity are final. Each
    // triangulation then contributes its streams; psPointEvalFinalize() reduces them at the end.
    // --serve arms the same context without points; each request supplies its own.
    if ( not state.options.psSamplePointsFile.empty() or state.options.psServe )
        psPointEvalInit( state.options );

#ifdef PHASE_SPACE
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
        MESSAGE::Message message( userOptions.verboseLevel );
        message << "--ps-window: the output grid is the window of "
                << state.options.psWindowDims[0] << "x" << state.options.psWindowDims[1] << "x" << state.options.psWindowDims[2]
                << " cells at (" << state.options.psWindowLo[0] << ", " << state.options.psWindowLo[1] << ", " << state.options.psWindowLo[2]
                << ") of the " << MESSAGE::printElements( state.options.gridSize, "x" ) << " grid.\n" << MESSAGE::Flush;
    }
#endif

#ifdef PHASE_SPACE
    if ( userOptions.partitionOn and userOptions.partNo<0 )
    {
        // PS-DTFE partitions in Lagrangian space (Eulerian partitioning fails because Lagrangian cells
        // map to arbitrary Eulerian locations). Each partition writes the full grid; results are summed.
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

        size_t mainGrid[NO_DIM];                       // the main grid: the --ps-window cells, else the full grid
        state.options.outputGridSize( mainGrid );
        uQuantities->reserveMemory( mainGrid, state.options.uField );
        aQuantities->reserveMemory( mainGrid, state.options.aField );

        // pre-size the stream_count accumulators (reserveMemory does not cover them) so
        // addFromSubgrid does not write past an unsized vector
        {
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

        // periodic=true here only drives grid-level index wrapping; data periodicity comes
        // from the Lagrangian copies and Eulerian unwrapping

        // each thread builds, interpolates, and accumulates its own triangulation into the
        // shared grids inside a critical section; peak memory ~ (concurrent partitions) x full grid

        // cap concurrent partition triangulations (the dominant memory cost); --max-concurrent trades cores for RAM, 0 = all threads
        int psConcurrency = totalPartitions;
        if ( userOptions.maxConcurrent > 0 and userOptions.maxConcurrent < psConcurrency )
            psConcurrency = userOptions.maxConcurrent;
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
        int const toProcess = std::max( 1, totalPartitions - nSkipped );

        // progress + ETA; updated only under mergeMutex below so it is race-free
        auto const psLoopStart = std::chrono::steady_clock::now();
        int psDone = 0;
        // The auto-tuner's SPEED split of a small set (psOrderedMerge) adds the partitions' grids
        // in INDEX order, so the float sums are the same run after run; its partitions are small
        // and uniform, so waiting for the predecessor costs nothing. A memory-driven split keeps
        // merging in completion order: its partitions differ 30x in cost, and threads idling behind
        // a slow one cost the TNG region's GPU grid 35% -- for sums the GPU's atomics make non-
        // reproducible anyway. Waiting cannot deadlock: the dynamic schedule hands iterations out in
        // order, so every lower index is already running or done.
        bool const orderedMerge = state.options.psOrderedMerge;
        std::mutex mergeMutex;
        std::condition_variable mergeCv;
        int nextMerge = 0;

        ConcurrentTriangulationPlan triangulationPlan( psConcurrency, totalPartitions );
#ifdef OPEN_MP
        #pragma omp parallel for schedule(dynamic, 1) num_threads(psConcurrency)
#endif
        for (int pi=0; pi<totalPartitions; ++pi)
        {
            ConcurrentTriangulationGuard concurrentTriangulation;
            // unpadded Lagrangian region for this partition (ownership) and the padded selection box
            Box lagRegion, lagPadded;
            plan.regions( pi, lagRegion, lagPadded );

            // select particles by Lagrangian position; when periodic, this partition's images
            // are generated on the fly so the global array stays originals-only
            bool const skipped = skipPartition[size_t(pi)] != 0;   // --ps-window: never reaches the window
            // a partition whose cached tessellation matches its descriptor is loaded, not built: its
            // particles are never needed, so their selection (a scan of every particle, ~3 s for TNG100-3's
            // 94M, plus a copy of the padded share) is skipped -- the load must then succeed (psCacheOnly)
            User_options tempOpt = partitionOptions( lagRegion );
            bool const cacheOnly = not skipped and not state.options.tessellationCacheDir.empty()
                                   and psTessellationCached( tempOpt );
            vector<Particle_data> tempPart;
            if ( not skipped and not cacheOnly and state.options.psParticlesDeferred )
                throwError( "internal: a partition needs its particles, but they were not read (the globals record path)." );
            if ( not skipped and not cacheOnly )
                findParticlesInBoxLagrangianPeriodic( state.particles, &tempPart, lagPadded,
                                                      plan.eulerLen, plan.periodicCopies, 0 );

            // an empty (or skipped) partition builds nothing but still takes its turn in the ordered merge
            bool const emptyPartition = skipped or ( not cacheOnly and tempPart.empty() );
            Quantities temp_uQuantities, temp_aQuantities;
            if ( not emptyPartition )
            {
            // per-partition options use the full Eulerian grid; only the master thread logs
            tempOpt.psCacheOnly = cacheOnly;
            int tnum = 0;
#ifdef OPEN_MP
            tnum = omp_get_thread_num();
#endif
            tempOpt.verboseLevel = (tnum == 0) ? userOptions.verboseLevel : 0;

            // PS-DTFE: DTFE_parallel calls serial interpolation directly (no nested OpenMP region)
            DTFE_parallel( &tempPart, samples, tempOpt, &temp_uQuantities, &temp_aQuantities );
            }

            // accumulate into the shared grids (in partition order under the speed split)
            {
                std::unique_lock<std::mutex> lock( mergeMutex );
                if ( orderedMerge )
                    mergeCv.wait( lock, [&]{ return nextMerge == pi; } );
                if ( not emptyPartition )
                {   // with --ps-window every partition's sub-grid IS the window: the merge is window-local
                    size_t const *wo = state.options.psWindowOn ? state.options.psWindowLo   : nullptr;
                    size_t const *wd = state.options.psWindowOn ? state.options.psWindowDims : nullptr;
                    uQuantities->addFromSubgrid( temp_uQuantities, &(state.options.gridSize[0]), wo, wd );
                    aQuantities->addFromSubgrid( temp_aQuantities, &(state.options.gridSize[0]), wo, wd );
                }
                if ( orderedMerge )
                {
                    nextMerge = pi + 1;
                    mergeCv.notify_all();
                }

                // progress over the partitions actually processed (the skipped ones cost nothing)
                if ( not skipped )
                {
                ++psDone;
                double const elapsed = std::chrono::duration<double>( std::chrono::steady_clock::now() - psLoopStart ).count();
                double const frac    = double(psDone) / double(toProcess);
                double const eta     = ( psDone>0 ) ? elapsed * ( double(toProcess - psDone) / double(psDone) ) : 0.;
                MESSAGE::Message prog( userOptions.verboseLevel );
                prog << MESSAGE::cGreen() << "  [partitions " << psDone << "/" << toProcess
                     << " | " << int(100.*frac + 0.5) << "%]" << MESSAGE::cReset()
                     << MESSAGE::cDim() << "  elapsed " << MESSAGE::formatDuration(elapsed)
                     << ( psDone<toProcess ? ", ETA ~" + MESSAGE::formatDuration(eta) : ", done" )
                     << MESSAGE::cReset() << "\n" << MESSAGE::Flush;
                }
            }
        }

        // every partition copied its own particles (tempPart); the global array is not needed by
        // the normalization, post-processing or write phases -- free its ~48 B/particle now
        std::vector<Particle_data>().swap( state.particles );

        // fields were summed as density-weighted moments; normalize once now so cells drawing
        // streams from several partitions (multi-stream regions) come out correct. When the
        // density field is selected the partitions alias the weight to the density grid, so
        // pass the scale that turns the summed rho/rho_bar back into the per-cell mass.
        {
            Real cellVolume = Real(1.);
            for (int d = 0; d < NO_DIM; ++d)
                cellVolume *= (state.options.region[2*d+1] - state.options.region[2*d]) / Real(state.options.gridSize[d]);
            Real const weightScale = cellVolume * state.options.averageDensity;
            uQuantities->normalizePhaseSpace( state.options.uField, weightScale );
            aQuantities->normalizePhaseSpace( state.options.aField, weightScale );
        }

        // aggregate coverage / stream statistics over all partitions from the summed grid
        // (per-partition figures were suppressed as misleading); prefer the unaveraged
        // stream_count, falling back to the averaged one if only '_a' fields exist
        {
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
    }
    else
#endif // PHASE_SPACE

    // standard DTFE: process the spatial partitions one at a time to bound peak memory on large data
    if ( userOptions.partitionOn and userOptions.partNo<0 )
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
                for (int d=0; d<NO_DIM; ++d)
                {
                    ownLo[d] = double( tempOpt.region[2*d] );
                    ownHi[d] = double( tempOpt.region[2*d+1] );
                    if ( not state.options.periodic )
                    {
                        if ( tempOpt.region[2*d]   <= state.options.region[2*d] )   ownLo[d] = -std::numeric_limits<double>::infinity();
                        if ( tempOpt.region[2*d+1] >= state.options.region[2*d+1] ) ownHi[d] =  std::numeric_limits<double>::infinity();
                    }
                }
                psServeCompositeAddPartition( i, tempOpt,
                    [&](std::vector<Particle_data> &out) { findParticlesInBox( state.particles, &out, selectOpt ); },
                    ownLo, ownHi );
            }
            std::vector<Particle_data>().swap( state.particles );
            psServeCompositeRun( state.options );   // answers requests until stdin closes; never returns
        }
#endif

        uQuantities->reserveMemory( &(state.options.gridSize[0]), state.options.uField );
        aQuantities->reserveMemory( &(state.options.gridSize[0]), state.options.aField );

        std::vector< std::vector<size_t> > subgridList;
        std::vector< Box > subgridCoords;
        optimalPartitionSplit( state.particles, state.options, state.options.partition, &subgridList, &subgridCoords );

        auto const partLoopStart = std::chrono::steady_clock::now();
        for (int i=0; i<totalPartitions; ++i)
        {
            message << MESSAGE::cBold() << "\n<<< Interpolating the fields for partition " << i+1 << "/" << totalPartitions << " ...\n" << MESSAGE::cReset();

            User_options tempOpt = state.options;
            tempOpt.partNo = i;
            copySubgridInformation( &tempOpt, subgridList, subgridCoords );

            tempOpt.paddedBox = tempOpt.region;
            tempOpt.paddedBox.addPadding( tempOpt.paddingLength );

            vector<Particle_data> tempPart;
            findParticlesInBox( state.particles, &tempPart, tempOpt );
            tempOpt.periodic = false;

            Quantities temp_uQuantities, temp_aQuantities;
            DTFE_parallel( &tempPart, samples, tempOpt, &temp_uQuantities, &temp_aQuantities );

            // map this partition's sub-grid results back into the full output grids
            copySubgridResultsToMain( temp_uQuantities, state.options.gridSize, tempOpt.uField, tempOpt, subgridList, uQuantities );
            copySubgridResultsToMain( temp_aQuantities, state.options.gridSize, tempOpt.aField, tempOpt, subgridList, aQuantities );

            // progress + ETA over the serial partition loop
            double const elapsed = std::chrono::duration<double>( std::chrono::steady_clock::now() - partLoopStart ).count();
            int    const done    = i + 1;
            double const eta     = elapsed * ( double(totalPartitions - done) / double(done) );
            message << MESSAGE::cGreen() << "  [partitions " << done << "/" << totalPartitions
                    << " | " << int(100.*done/double(totalPartitions) + 0.5) << "%]" << MESSAGE::cReset()
                    << MESSAGE::cDim() << "  elapsed " << MESSAGE::formatDuration(elapsed)
                    << ( done<totalPartitions ? ", ETA ~" + MESSAGE::formatDuration(eta) : ", done" )
                    << MESSAGE::cReset() << "\n" << MESSAGE::Flush;
        }

        // each partition copied its own particles; the global array is dead weight from here on
        std::vector<Particle_data>().swap( state.particles );
    }
    else
    {
        // single triangulation over the whole (padded) box
        DTFE_preparePadding( state, userOptions );
        DTFE_parallel( &state.particles, samples, state.options, uQuantities, aQuantities );
    }

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
    postProcessWebFields( userOptions, uQuantities, aQuantities );
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
#ifndef OPEN_MP
    interpolate( allParticles, samples, userOptions, uQuantities, aQuantities );  // deletes allParticles
    return;
#else
    int noAvailableProcessors = omp_get_max_threads();
    // cap concurrent sub-triangulations to bound peak memory (--max-concurrent); 0 = all threads
    if ( userOptions.maxConcurrent > 0 and userOptions.maxConcurrent < noAvailableProcessors )
        noAvailableProcessors = userOptions.maxConcurrent;

    // fall back to a single triangulation when spatial partitioning does not apply:
    // 1 thread, user sampling, redshift cone, --sample-points point evaluation (the standard
    // worker evaluates ONE Eulerian tessellation; overlapping padded sub-triangulations
    // would double-count boundary tets), or PS-DTFE (which partitions in Lagrangian space)
    if ( noAvailableProcessors==1 or not samples.empty() or userOptions.redshiftConeOn
         or psPointEvalActive()
#ifdef PHASE_SPACE
         or true  // PS-DTFE: bypass spatial partitioning, use single triangulation
#endif
       )
    {
        interpolate( allParticles, samples, userOptions, uQuantities, aQuantities );
        return;
    }


    // more than 1 thread: split the data into spatial partitions, one per thread
    std::vector<size_t> pGrid(NO_DIM,0); // the parallel grid
    parallelGrid( noAvailableProcessors, userOptions, &(pGrid[0]) );
    int noProcessors = 1;
    for (int d=0; d<NO_DIM; ++d) noProcessors *= pGrid[d];    //number of processors actually used (may differ from 'noAvailableProcessors')

    uQuantities->reserveMemory( &(userOptions.gridSize[0]), userOptions.uField );
    aQuantities->reserveMemory( &(userOptions.gridSize[0]), userOptions.aField );
    MESSAGE::Message message( userOptions.verboseLevel );
    message << "From now on only the master thread will show messages on how the computation is going. Not all threads take the same execution time, so there may be a discrepancy between the messages displayed to the user and the computations across all threads.\n\n" << MESSAGE::Flush;
    std::vector<size_t> processorParticles(noProcessors);    // number of particles associated to each processor
    std::vector<Real> processorTime(noProcessors);           // the actual runtime of each thread
    size_t const noTotalParticles = allParticles->size();
    

    // choose the partition geometry that balances the particle load across threads
    std::vector< std::vector<size_t> > subgridList;
    std::vector< Box > subgridCoords;
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
        tempOptions.updateFullBox( userOptions.region );


        // region allocated to this thread: writes subgrid size to tempOptions.gridSize and box to tempOptions.region
        copySubgridInformation( &tempOptions, subgridList, subgridCoords );
        tempOptions.paddedBox = tempOptions.region;
        tempOptions.paddedBox.addPadding( tempOptions.paddingLength );


        vector<Particle_data> particles;
        findParticlesInBox( *allParticles, &particles, tempOptions );
        processorParticles[ threadNo ] = particles.size();


        Quantities temp_uQuantities, temp_aQuantities;
        interpolate( &particles, samples, tempOptions, &temp_uQuantities, &temp_aQuantities ); //this function deletes the vector 'particles'
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






// Computes the velocity divergence from the velocity gradient.
Real velocityDivergence(Pvector<Real,noGradComp> &velGrad)
{
    Real div = 0.;
    for (int d=0; d<NO_DIM; ++d)
        div += velGrad[d*NO_DIM+d];
    return div;
}

// Computes the (traceless symmetric) velocity shear from the velocity gradient.
Pvector<Real,noShearComp> velocityShear(Pvector<Real,noGradComp> &velGrad)
{
    Pvector<Real,noShearComp> temp;
    size_t index = 0;
    for (int i=0; i<NO_DIM-1; ++i)
        for (int j=i; j<NO_DIM; ++j)
            temp[index++] = (velGrad[i*NO_DIM+j] + velGrad[j*NO_DIM+i]) / 2.;

    // remove the trace (the divergence) so only the shear remains; the shear tensor is traceless by definition
    Real trace = velGrad[(NO_DIM-1)*NO_DIM+(NO_DIM-1)];
    for (int d=0; d<NO_DIM-1; ++d)
        trace += temp[d*NO_DIM - d*(d-1)/2];
    trace /= NO_DIM;
    for (int d=0; d<NO_DIM-1; ++d)
        temp[d*NO_DIM - d*(d-1)/2] -= trace;
    return temp;
}

// Computes the velocity vorticity from the velocity gradient.
Pvector<Real,noVortComp> velocityVorticity(Pvector<Real,noGradComp> &velGrad)
{
    Pvector<Real,noVortComp> temp;
    size_t index = 0;
    for (int i=0; i<NO_DIM; ++i)
        for (int j=i+1; j<NO_DIM; ++j)
            temp[index++] = (velGrad[i*NO_DIM+j] - velGrad[j*NO_DIM+i]) / 2.;
    return temp;
}

// Eigenvalues of a 3x3 symmetric matrix via Cardano's formula, sorted descending (lambda1 >= lambda2 >= lambda3).
// Computed in DOUBLE whatever Real is: where two eigenvalues (nearly) coincide -- every cell of a field
// that varies along one axis, and the axis-symmetric parts of walls and filaments -- r = det(B)/2 sits
// at +-1, where acos has an infinite slope, so its rounding error eps turns into sqrt(eps) in the
// eigenvalues: in float, 6e-4 for a cell with delta = 3 (the T-web of a 1D pancake read {3.096, 6e-4,
// -6e-4} for {3.096, 0, 0}; tests/tweb_check.sh, 2026-10-02). In double, ~1e-8.
Pvector<Real,NO_DIM> symmetricEigenvalues3x3(Real a00r, Real a01r, Real a02r,
                                              Real a11r, Real a12r, Real a22r)
{
    Pvector<Real,NO_DIM> eigenvalues;
#if NO_DIM == 3
    double const a00 = a00r, a01 = a01r, a02 = a02r, a11 = a11r, a12 = a12r, a22 = a22r;
    double ev[3];
    double const p1 = a01*a01 + a02*a02 + a12*a12;
    double const q = (a00 + a11 + a22) / 3.;  // trace / 3

    double const b00 = a00 - q, b11 = a11 - q, b22 = a22 - q;
    double const p2 = b00*b00 + b11*b11 + b22*b22 + 2.*p1;
    double const p = std::sqrt(p2 / 6.);

    if ( p < 1.e-15 )
    {
        // matrix proportional to identity, already diagonal
        ev[0] = a00; ev[1] = a11; ev[2] = a22;
    }
    else
    {
        double const inv_p = 1. / p;
        // B = (1/p) * (A - q*I), compute det(B)
        double const B00 = b00*inv_p, B01 = a01*inv_p, B02 = a02*inv_p;
        double const B11 = b11*inv_p, B12 = a12*inv_p, B22 = b22*inv_p;

        double const detB = B00*(B11*B22 - B12*B12) - B01*(B01*B22 - B12*B02) + B02*(B01*B12 - B11*B02);
        double r = detB / 2.;

        // clamp to [-1, 1] for numerical safety before acos
        if (r <= -1.) r = -1.;
        else if (r >= 1.) r = 1.;

        double const phi = std::acos(r) / 3.;

        ev[0] = q + 2. * p * std::cos(phi);
        ev[2] = q + 2. * p * std::cos(phi + 2. * M_PI / 3.);
        ev[1] = 3. * q - ev[0] - ev[2]; // trace identity
    }

    // sort descending
    if (ev[0] < ev[1]) std::swap(ev[0], ev[1]);
    if (ev[1] < ev[2]) std::swap(ev[1], ev[2]);
    if (ev[0] < ev[1]) std::swap(ev[0], ev[1]);
    for (int i = 0; i < 3; ++i) eigenvalues[i] = Real(ev[i]);
#elif NO_DIM == 2
    // 2x2 symmetric matrix: (a00 + a11)/2 -+ |((a00 - a11)/2, a01)|. The half-difference form, not
    // trace^2/4 - det: that one cancels where the eigenvalues (nearly) coincide and leaves sqrt(eps).
    double const a00 = a00r, a01 = a01r, a11 = a11r;
    (void)a02r; (void)a12r; (void)a22r;
    double const mid = (a00 + a11) / 2.;
    double const disc = std::hypot( (a00 - a11) / 2., a01 );
    eigenvalues[0] = Real(mid + disc);
    eigenvalues[1] = Real(mid - disc);
#endif
    return eigenvalues;
}


// Classifies a cell by counting eigenvalues above lambda_th: 0=void, 1=wall, 2=filament, 3=node
// (2D: 0=void, 1=filament, 2=node -- one collapsed axis is a line there).
int classifyWeb(Pvector<Real,NO_DIM> const &eigenvalues, Real lambda_th)
{
    int label = 0;
    for (int d=0; d<NO_DIM; ++d)
        if (eigenvalues[d] > lambda_th) ++label;
    return label;
}


// V-web classification from the velocity shear tensor Sigma_ij = -(1/(2 H0))(dv_i/dx_j + dv_j/dx_i)
// (Hoffman et al. 2012). The physical T-web (tidal tensor of the potential, Poisson-solved from
// the density grid) is computed by computeTidalWebClassification below.
void computeWebClassification(Field &fields,
                               int const verboseLevel,
                               Real lambda_th,
                               Real hubbleParam,
                               Real scaleFactor,
                               Quantities *q)
{
    if ( q->velocity_gradient.empty() ) return;
    if ( not fields.velocity_vweb ) return;

    // normalize the gradient by H0 = 100 h km/s/Mpc so the threshold lambda_th is dimensionless
    Real H0_norm = Real(1.);
    if ( hubbleParam > Real(0.) )
        H0_norm = Real(100.) * hubbleParam;  // H0 in km/s/Mpc
    // The gradient is in Gadget u-units (u = v_pec/sqrt(a)): multiply by sqrt(a) to reach
    // peculiar km/s, or the eigenvalues come out 1/sqrt(a) inflated at high z (4.6x at z=20)
    // and lambda_th is effectively miscalibrated there. sqrt(1) == 1 exactly, so z=0 results
    // are bit-identical to the uncorrected code. -1 = unknown (no header value): warn, treat as 1.
    Real sqrtA = Real(1.);
    if ( scaleFactor > Real(0.) )
        sqrtA = std::sqrt( scaleFactor );
    else
    {
        MESSAGE::Warning warning( verboseLevel );
        warning << "V-web: no scale factor available (header lacked 'Time' and no '--scale-factor' given); "
                << "assuming a = 1. At high redshift the eigenvalues are then 1/sqrt(a) inflated.\n"
                << MESSAGE::EndWarning;
    }

    size_t const N = q->velocity_gradient.size();

    // V-web: shear tensor Sigma_ij
    if ( fields.velocity_vweb )
    {
        q->velocity_vweb.reserve( N );
        q->velocity_vweb_eigenvalues.reserve( N );
        for (size_t i=0; i<N; ++i)
        {
            Pvector<Real,noGradComp> &g = q->velocity_gradient[i];
            Real norm = -sqrtA / (Real(2.) * H0_norm);   // sqrtA: u-units -> peculiar km/s
            Real s00 = norm * (g[0*NO_DIM+0] + g[0*NO_DIM+0]);
            Real s11 = norm * (g[1*NO_DIM+1] + g[1*NO_DIM+1]);
            Real s01 = norm * (g[0*NO_DIM+1] + g[1*NO_DIM+0]);
#if NO_DIM == 3
            Real s22 = norm * (g[2*NO_DIM+2] + g[2*NO_DIM+2]);
            Real s02 = norm * (g[0*NO_DIM+2] + g[2*NO_DIM+0]);
            Real s12 = norm * (g[1*NO_DIM+2] + g[2*NO_DIM+1]);
            Pvector<Real,NO_DIM> eig = symmetricEigenvalues3x3(s00, s01, s02, s11, s12, s22);
#elif NO_DIM == 2
            Pvector<Real,NO_DIM> eig = symmetricEigenvalues3x3(s00, s01, Real(0.), s11, Real(0.), Real(0.));
#endif
            q->velocity_vweb_eigenvalues.push_back( eig );
            q->velocity_vweb.push_back( Real(classifyWeb(eig, lambda_th)) );
        }
    }

    // free the gradient once the web classification no longer needs it; swap-with-empty because
    // clear() keeps the capacity resident (36 B/cell -- ~4.8 GB at 512^3)
    if ( not fields.velocity_gradient )
        std::vector< Pvector<Real,noGradComp> >().swap( q->velocity_gradient );
}


// T-web classification from the TIDAL tensor of the gravitational potential (Hahn et al. 2007;
// Forero-Romero et al. 2009), Poisson-solved from the density grid by FFT. Dimensionless
// convention: T_ij = F^-1[ (k_i k_j / k^2) delta_hat ] so that tr T = delta exactly and
// lambda_th is dimensionless (literature thresholds ~0.2-0.4 apply directly; the same
// convention was validated against an independent Python implementation, trace identity ~1e-6).
// Computed from the RAW density grid -- NO smoothing here by design (the binary outputs raw
// fields only; smoothing is a plot-time choice in python/plot_PS_DTFE.py). Note the labels are
// therefore sensitive to FFT ringing from point-like halo density spikes; labels are categorical,
// so a smoothed CLASSIFICATION (not just a smoothed picture) requires re-deriving the tensor from
// a smoothed density, i.e. a rerun. Periodic, full-box grids only, 2D or 3D (the kernel k_i k_j / k^2
// is the same in any dimension; 2D has three tensor components and two eigenvalues); assumes square
// grid cells (k ratios use per-cell wavenumbers). Runs on the merged full grid after normalization,
// so it is independent of the partition path.
// The FFT buffers of the T-web: as aligned as fftw_alloc makes them (64 B covers every SIMD width FFTW
// uses), through the aligned operator new -- which '--scratch-dir' replaces too (scratch_alloc.cc), so
// above its threshold they live in the scratch files like the grids. fftw_alloc would bypass it.
template <class T>
struct FftwAligned
{
    using value_type = T;
    FftwAligned() = default;
    template <class U> FftwAligned(FftwAligned<U> const &) {}
    T *allocate(size_t n) { return static_cast<T*>( ::operator new( n * sizeof(T), std::align_val_t(64) ) ); }
    void deallocate(T *p, size_t) noexcept { ::operator delete( p, std::align_val_t(64) ); }
    template <class U> bool operator==(FftwAligned<U> const &) const { return true; }
    template <class U> bool operator!=(FftwAligned<U> const &) const { return false; }
};

void computeTidalWebClassification(Field &fields,
                                   User_options const &userOptions,
                                   Quantities *q)
{
    if ( not fields.velocity_tweb ) return;

    MESSAGE::Message message( userOptions.verboseLevel );

    auto skip = [&](char const* why)
    {
        MESSAGE::Warning warning( userOptions.verboseLevel );
        warning << "Skipping the T-web: " << why << "\n" << MESSAGE::EndWarning;
        fields.velocity_tweb = false;   // writer then skips the (empty) T-web output
    };

    if ( not userOptions.periodic ) { skip("the tidal tensor is FFT-Poisson-solved, which needs a periodic box."); return; }
    size_t n[NO_DIM];
    size_t N = 1;
    for (int d = 0; d < NO_DIM; ++d) { n[d] = userOptions.gridSize[d]; N *= n[d]; }
    if ( q->density.size() != N ) { skip("it needs the density field on the full grid (select 'density'/'density_a' too)."); return; }
    // Above 2^31 cells (~1290^3) FFTW's ordinary planners overflow their int sizes: the plans are made
    // with the 64-bit guru interface (fftw_plans64.h). Every buffer below is an ordinary allocation
    // (std::vector), so '--scratch-dir' backs it with disk like the grids: the chain needs ~48 bytes
    // per cell at its peak (the real work array, two half-spectra, the six float tensor components).

    auto const tidalStart = std::chrono::steady_clock::now();
    message << "\nComputing the T-web from the tidal tensor of the raw density grid (lambda_th = "
            << userOptions.lambda_th << ") ... " << MESSAGE::Flush;

    // density contrast (double precision for the FFT chain)
    double mean = 0.;
    for (size_t i = 0; i < N; ++i) mean += double(q->density[i]);
    mean /= double(N);
    if ( mean <= 0. ) { skip("the density grid has non-positive mean."); return; }

    std::vector<double, FftwAligned<double> > work(N);
    for (size_t i = 0; i < N; ++i) work[i] = double(q->density[i]) / mean - 1.;

    // the half-spectra: std::complex<double> has fftw_complex's layout (FFTW manual, "Complex numbers")
    using Spectrum = std::vector< std::complex<double>, FftwAligned< std::complex<double> > >;
    size_t const nLast = n[NO_DIM-1], nzh = nLast/2 + 1;    // the last axis is the half-complex one
    size_t const nOuter = N / nLast;                         // the full axes' cells, flattened
    Spectrum deltaKBuf( nOuter * nzh ), tensKBuf( nOuter * nzh );
    fftw_complex *deltaK = reinterpret_cast<fftw_complex*>( deltaKBuf.data() );
    fftw_complex *tensK  = reinterpret_cast<fftw_complex*>( tensKBuf.data() );
    // 64-bit sizes (fftw_plans64.h): the ordinary planners stop at 2^31 cells
    fftw_plan planF = fftwPlans64::r2c(NO_DIM, n, work.data(), deltaK, FFTW_ESTIMATE);
    if ( planF == nullptr ) { skip("FFTW could not plan the forward transform."); return; }
    fftw_execute(planF);
    fftw_destroy_plan(planF);

    // per-cell angular wavenumbers k_d = 2 pi m_d / n_d (signed m); ratios k_i k_j / k^2 are the
    // dimensionless tidal kernel, and the Gaussian smoothing sigma is naturally in grid cells.
    // The Nyquist SIGN matters for the cross terms (k_a k_b flips with it): full axes treat the
    // Nyquist mode as negative (numpy fftfreq convention) and the rfft half-axis as positive
    // (numpy rfftfreq), matching the validated Python reference implementation exactly.
    auto kFull = [](size_t idx, size_t n) -> double
    {
        long m = long(idx);
        if ( m > (long(n) - 1) / 2 ) m -= long(n);
        return 2. * M_PI * double(m) / double(n);
    };
    auto kHalf = [](size_t idx, size_t n) -> double
    {
        return 2. * M_PI * double(idx) / double(n);
    };

    double const invN   = 1. / double(N);   // FFTW round trip is unnormalized
    // the upper triangle of the symmetric tensor, row by row (symmetricEigenvalues3x3's argument order)
#if NO_DIM == 3
    static int const NPAIRS = 6;
    static int const PAIRS[NPAIRS][2] = { {0,0}, {0,1}, {0,2}, {1,1}, {1,2}, {2,2} };
#else
    static int const NPAIRS = 3;
    static int const PAIRS[NPAIRS][2] = { {0,0}, {0,1}, {1,1} };
#endif
    std::vector<Real> tens[NPAIRS];

    fftw_plan planB = fftwPlans64::c2r(NO_DIM, n, tensK, work.data(), FFTW_ESTIMATE);
    if ( planB == nullptr ) { skip("FFTW could not plan the inverse transform."); return; }
    for (int c = 0; c < NPAIRS; ++c)
    {
        int const a = PAIRS[c][0], b = PAIRS[c][1];
#ifdef OPEN_MP
        #pragma omp parallel for
#endif
        for (size_t o = 0; o < nOuter; ++o)     // (ix, iy) in 3D, ix in 2D: the half-spectrum's rows
        {
            double k[NO_DIM];
            size_t rest = o;
            for (int d = NO_DIM - 2; d >= 0; --d) { k[d] = kFull(rest % n[d], n[d]); rest /= n[d]; }
            for (size_t iz = 0; iz < nzh; ++iz)
            {
                k[NO_DIM-1] = kHalf(iz, nLast);
                double k2 = 0.;
                for (int d = 0; d < NO_DIM; ++d) k2 += k[d]*k[d];
                size_t const idx = o * nzh + iz;
                if ( k2 == 0. ) { tensK[idx][0] = 0.; tensK[idx][1] = 0.; continue; }
                double const f = (k[a] * k[b] / k2) * invN;
                tensK[idx][0] = f * deltaK[idx][0];
                tensK[idx][1] = f * deltaK[idx][1];
            }
        }
        fftw_execute(planB);          // -> work (real tensor component)
        tens[c].resize(N);
        for (size_t i = 0; i < N; ++i) tens[c][i] = Real(work[i]);
    }
    fftw_destroy_plan(planB);
    Spectrum().swap( deltaKBuf );
    Spectrum().swap( tensKBuf );
    work.clear(); work.shrink_to_fit();

    // eigenvalues (descending) and classification
    q->velocity_tweb.assign(N, Real(0.));
    q->velocity_tweb_eigenvalues.assign(N, Pvector<Real,NO_DIM>::zero());
    size_t counts[4] = {0, 0, 0, 0};
#ifdef OPEN_MP
    #pragma omp parallel for reduction(+:counts[:4])
#endif
    for (size_t i = 0; i < N; ++i)
    {
#if NO_DIM == 3
        Pvector<Real,NO_DIM> eig = symmetricEigenvalues3x3( tens[0][i], tens[1][i], tens[2][i],
                                                            tens[3][i], tens[4][i], tens[5][i] );
#else
        Pvector<Real,NO_DIM> eig = symmetricEigenvalues3x3( tens[0][i], tens[1][i], Real(0.),
                                                            tens[2][i], Real(0.), Real(0.) );
#endif
        q->velocity_tweb_eigenvalues[i] = eig;
        int const label = classifyWeb( eig, userOptions.lambda_th );
        q->velocity_tweb[i] = Real(label);
        counts[label] += 1;
    }

    message << "Done.\n";
#if NO_DIM == 3
    static char const* NAMES[4] = { "void", "wall", "filament", "node" };
#else
    static char const* NAMES[3] = { "void", "filament", "node" };     // one collapsed axis is a line in 2D
#endif
    message << "T-web volume fractions:";
    for (int l = 0; l <= NO_DIM; ++l)
        message << "  " << NAMES[l] << " = " << (100. * double(counts[l]) / double(N)) << "\%";
    message << "\n";
    message << "  >>> Time: " << std::chrono::duration<double>(std::chrono::steady_clock::now() - tidalStart).count()
            << " sec. (T-web tidal tensor)\n" << MESSAGE::Flush;
}


// Fills the velocity divergence/shear/vorticity vectors requested in 'fields' from the stored gradient.
void computeDivergenceShearVorticity(Field &fields,
                                     int const verboseLevel,
                                     Quantities *q)
{
    if ( q->velocity_gradient.empty() ) return;

    if ( fields.velocity_divergence )
    {
        q->velocity_divergence.reserve( q->velocity_gradient.size() );
        for (std::vector< Pvector<Real,noGradComp> >::iterator it=q->velocity_gradient.begin(); it!=q->velocity_gradient.end(); ++it )
            q->velocity_divergence.push_back( velocityDivergence(*it) );
    }

    if ( fields.velocity_shear )
    {
        q->velocity_shear.reserve( q->velocity_gradient.size() );
        for (std::vector< Pvector<Real,noGradComp> >::iterator it=q->velocity_gradient.begin(); it!=q->velocity_gradient.end(); ++it )
            q->velocity_shear.push_back( velocityShear(*it) );
    }
    
    if ( fields.velocity_vorticity )
    {
        q->velocity_vorticity.reserve( q->velocity_gradient.size() );
        for (std::vector< Pvector<Real,noGradComp> >::iterator it=q->velocity_gradient.begin(); it!=q->velocity_gradient.end(); ++it )
            q->velocity_vorticity.push_back( velocityVorticity(*it) );
    }
    
    // free the gradient unless the gradient itself or the V-web classification still needs it
    // (the T-web is computed from the density grid, never the gradient); swap-with-empty
    // because clear() keeps the capacity resident
    if ( not fields.velocity_gradient and not fields.velocity_vweb )
        std::vector< Pvector<Real,noGradComp> >().swap( q->velocity_gradient );
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
}
#endif

