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
#include <chrono>   // partition-loop progress + ETA
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
#include "message.h"


using namespace std;

#include "interpolations.h"




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

// Derives T-web and/or V-web cosmic-web classification from the velocity gradient.
void computeWebClassification(Field &fields,
                               int const verboseLevel,
                               Real lambda_th,
                               Real hubbleParam,
                               Quantities *q);

// Builds the velocity-space DTFE density g_i used by the approximate phase-space density (--approxPSD).
#if defined(VELOCITY) && defined(SCALAR) && !defined(PHASE_SPACE)
extern void computeVelocitySpaceDensity(vector<Particle_data> &particles,
                                         User_options &userOptions);
#endif


// Particle data plus the resolved per-run options produced by the common DTFE setup.
struct DTFE_State {
    vector<Particle_data> particles;
    User_options options;
};


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
    userOptions.updateEntries( particlePointer->size(), not samples.empty() );
    if ( userOptions.averageDensity<0. )
        userOptions.averageDensity = averageDensity( *particlePointer, userOptions );

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
                msg << "PS-DTFE (non-periodic): density normalized to the Lagrangian cloud density "
                       "N*m/V_lagBox = " << state.options.averageDensity << ".\n" << MESSAGE::Flush;
            }
        }
    }
    if ( userOptions.periodic && !state.particles.empty() )
    {
        MESSAGE::Message msg( userOptions.verboseLevel );

        // global Lagrangian bounding box (min/max of lagPos over all particles)
        for (int d=0; d<NO_DIM; ++d)
        {
            lagBoxGlobal[2*d]   = state.particles[0].lagPos[d];
            lagBoxGlobal[2*d+1] = state.particles[0].lagPos[d];
        }
        for (size_t i=1; i<state.particles.size(); ++i)
            for (int d=0; d<NO_DIM; ++d)
            {
                if (state.particles[i].lagPos[d] < lagBoxGlobal[2*d])   lagBoxGlobal[2*d]   = state.particles[i].lagPos[d];
                if (state.particles[i].lagPos[d] > lagBoxGlobal[2*d+1]) lagBoxGlobal[2*d+1] = state.particles[i].lagPos[d];
            }
        // tiny margin to include boundary particles
        for (int d=0; d<NO_DIM; ++d)
        {
            Real eps = (lagBoxGlobal[2*d+1] - lagBoxGlobal[2*d]) * Real(1.e-6);
            lagBoxGlobal[2*d]   -= eps;
            lagBoxGlobal[2*d+1] += eps;
        }

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
        for (size_t i = 0; i < state.particles.size(); ++i)
            for (int d = 0; d < NO_DIM; ++d)
            {
                Real s = state.particles[i].pos[d] - state.particles[i].lagPos[d];
                s = std::fmod(s + eulerLen[d] * Real(0.5), eulerLen[d]);
                if (s < Real(0.)) s += eulerLen[d];
                s -= eulerLen[d] * Real(0.5);
                state.particles[i].pos[d] = state.particles[i].lagPos[d] + s;
            }

        msg << "PS-DTFE: Unwrapped Eulerian positions (displacements within [-L/2, L/2]).\n" << MESSAGE::Flush;

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

            msg << "PS-DTFE: Lagrangian bounding box = " << lagBoxGlobal.print() << "\n"
                << "PS-DTFE: Added " << (state.particles.size() - noOriginal)
                << " Lagrangian periodic copies (" << noOriginal << " original particles).\n" << MESSAGE::Flush;
        }
        else
        {
            msg << "PS-DTFE: Lagrangian bounding box = " << lagBoxGlobal.print() << "\n"
                << "PS-DTFE: periodic copies are generated per-partition (deferred) to keep peak memory low.\n" << MESSAGE::Flush;
        }

        // primary periodic box: centroid ownership check keeps one image of each cell and
        // tiles the box; overridden per-partition in the partition path
        state.options.lagrangianRegion = state.options.boxCoordinates;
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
        message << "\nPS-DTFE: Lagrangian-space partitioning with " << totalPartitions
                << " partitions [" << MESSAGE::printElements( userOptions.partition, "," ) << "].\n" << MESSAGE::Flush;

        // reuse the Lagrangian bounding box computed before periodic copies, else recompute it here
        Box lagBox;
        if (!lagBoxGlobal.isNullBox())
        {
            lagBox = lagBoxGlobal;
        }
        else
        {
            for (int d=0; d<NO_DIM; ++d)
            {
                lagBox[2*d]   = state.particles[0].lagPos[d];
                lagBox[2*d+1] = state.particles[0].lagPos[d];
            }
            for (size_t i=1; i<state.particles.size(); ++i)
                for (int d=0; d<NO_DIM; ++d)
                {
                    if (state.particles[i].lagPos[d] < lagBox[2*d])   lagBox[2*d]   = state.particles[i].lagPos[d];
                    if (state.particles[i].lagPos[d] > lagBox[2*d+1]) lagBox[2*d+1] = state.particles[i].lagPos[d];
                }
            // tiny margin to include boundary particles
            for (int d=0; d<NO_DIM; ++d)
            {
                Real eps = (lagBox[2*d+1] - lagBox[2*d]) * Real(1.e-6);
                lagBox[2*d]   -= eps;
                lagBox[2*d+1] += eps;
            }
        }
        message << "PS-DTFE: Lagrangian bounding box = " << lagBox.print() << "\n" << MESSAGE::Flush;

        Real lagLen[NO_DIM];
        for (int d=0; d<NO_DIM; ++d)
            lagLen[d] = lagBox[2*d+1] - lagBox[2*d];

        // Eulerian box length per axis: the period used to shift each partition's deferred copies
        Real eulerLenArr[NO_DIM];
        for (int d=0; d<NO_DIM; ++d)
            eulerLenArr[d] = state.options.boxCoordinates[2*d+1] - state.options.boxCoordinates[2*d];

        // Lagrangian padding per axis: same fraction as Eulerian, floored at 0.30 of a partition cell
        Real lagPadding[NO_DIM];
        for (int d=0; d<NO_DIM; ++d)
        {
            Real padFrac = (state.options.paddingLength[2*d] + state.options.paddingLength[2*d+1]) / (Real(2.) * eulerLenArr[d]);
            lagPadding[d] = padFrac * lagLen[d];
            if (lagPadding[d] < lagLen[d] / state.options.partition[d] * Real(0.3))
                lagPadding[d] = lagLen[d] / state.options.partition[d] * Real(0.3);
        }

        uQuantities->reserveMemory( &(state.options.gridSize[0]), state.options.uField );
        aQuantities->reserveMemory( &(state.options.gridSize[0]), state.options.aField );

        // pre-size the stream_count accumulators (reserveMemory does not cover them) so
        // addFromSubgrid does not write past an unsized vector
        {
            size_t totalGrid = 1;
            for (int d=0; d<NO_DIM; ++d) totalGrid *= state.options.gridSize[d];
            if ( state.options.uField.selected() )
                uQuantities->stream_count.assign(totalGrid, Real(0.));
            if ( state.options.aField.selected() )
                aQuantities->stream_count.assign(totalGrid, Real(0.));
        }

        // Eulerian region/padding for the full box (needed by DTFE_interpolation)
        if ( not userOptions.regionOn and not (userOptions.partitionOn and userOptions.partNo>=0) )
        {
            state.options.region = state.options.boxCoordinates;
            state.options.paddedBox = state.options.region;
            state.options.paddedBox.addPadding( state.options.paddingLength );
        }
        // periodic=true here only drives grid-level index wrapping; data periodicity comes
        // from the Lagrangian copies and Eulerian unwrapping

        // each thread builds, interpolates, and accumulates its own triangulation into the
        // shared grids inside a critical section; peak memory ~ (concurrent partitions) x full grid
        size_t partGrid[NO_DIM];
        for (int d=0; d<NO_DIM; ++d)
            partGrid[d] = state.options.partition[d];

        // cap concurrent partition triangulations (the dominant memory cost); --max-concurrent trades cores for RAM, 0 = all threads
        int psConcurrency = totalPartitions;
        if ( userOptions.maxConcurrent > 0 and userOptions.maxConcurrent < psConcurrency )
            psConcurrency = userOptions.maxConcurrent;
        if ( psConcurrency < 1 ) psConcurrency = 1;
        message << "PS-DTFE: building up to " << psConcurrency << " partition triangulation(s) concurrently"
                << ( userOptions.maxConcurrent > 0 ? " (--max-concurrent)" : "" ) << ".\n" << MESSAGE::Flush;

        // progress + ETA; updated only in the critical section below so it is race-free
        auto const psLoopStart = std::chrono::steady_clock::now();
        int psDone = 0;

#ifdef OPEN_MP
        #pragma omp parallel for schedule(dynamic, 1) num_threads(psConcurrency)
#endif
        for (int pi=0; pi<totalPartitions; ++pi)
        {
            // unflatten the linear partition index pi into per-axis partition coordinates
            int idx[NO_DIM];
            {
                int rem = pi;
                for (int d=NO_DIM-1; d>=0; --d)
                {
                    idx[d] = rem % partGrid[d];
                    rem /= partGrid[d];
                }
            }

            // unpadded Lagrangian region for this partition
            Box lagRegion;
            for (int d=0; d<NO_DIM; ++d)
            {
                lagRegion[2*d]   = lagBox[2*d] + lagLen[d] * idx[d] / partGrid[d];
                lagRegion[2*d+1] = lagBox[2*d] + lagLen[d] * (idx[d]+1) / partGrid[d];
            }

            Box lagPadded = lagRegion;
            for (int d=0; d<NO_DIM; ++d)
            {
                lagPadded[2*d]   -= lagPadding[d];
                lagPadded[2*d+1] += lagPadding[d];
                // clamp to the overall Lagrangian box only when no periodic copies exist
                if (!hasLagrangianPeriodicCopies)
                {
                    if (lagPadded[2*d]   < lagBox[2*d])   lagPadded[2*d]   = lagBox[2*d];
                    if (lagPadded[2*d+1] > lagBox[2*d+1]) lagPadded[2*d+1] = lagBox[2*d+1];
                }
            }

            // select particles by Lagrangian position; when periodic, this partition's images
            // are generated on the fly so the global array stays originals-only
            vector<Particle_data> tempPart;
            findParticlesInBoxLagrangianPeriodic( state.particles, &tempPart, lagPadded,
                                                  eulerLenArr, hasLagrangianPeriodicCopies, 0 );

            if (tempPart.empty())
                continue;

            // per-partition options use the full Eulerian grid; only the master thread logs
            User_options tempOpt = state.options;
            tempOpt.lagrangianRegion = lagRegion;  // for cell ownership check
            tempOpt.psSuppressGridStats = true;    // per-partition stats misleading; aggregate printed after the loop
            tempOpt.psDeferNormalization = true;   // keep fields as moments; normalize once after summing
            tempOpt.psUseSubgrid = true;           // store only this partition's Eulerian bounding box
            int tnum = 0;
#ifdef OPEN_MP
            tnum = omp_get_thread_num();
#endif
            tempOpt.verboseLevel = (tnum == 0) ? userOptions.verboseLevel : 0;

            // PS-DTFE: DTFE_parallel calls serial interpolation directly (no nested OpenMP region)
            Quantities temp_uQuantities, temp_aQuantities;
            DTFE_parallel( &tempPart, samples, tempOpt, &temp_uQuantities, &temp_aQuantities );

            // accumulate into the shared grids (serialized)
#ifdef OPEN_MP
            #pragma omp critical
#endif
            {
                uQuantities->addFromSubgrid( temp_uQuantities, &(state.options.gridSize[0]) );
                aQuantities->addFromSubgrid( temp_aQuantities, &(state.options.gridSize[0]) );

                ++psDone;
                double const elapsed = std::chrono::duration<double>( std::chrono::steady_clock::now() - psLoopStart ).count();
                double const frac    = double(psDone) / double(totalPartitions);
                double const eta     = ( psDone>0 ) ? elapsed * ( double(totalPartitions - psDone) / double(psDone) ) : 0.;
                MESSAGE::Message prog( userOptions.verboseLevel );
                prog << MESSAGE::cGreen() << "  [partitions " << psDone << "/" << totalPartitions
                     << " | " << int(100.*frac + 0.5) << "%]" << MESSAGE::cReset()
                     << MESSAGE::cDim() << "  elapsed " << MESSAGE::formatDuration(elapsed)
                     << ( psDone<totalPartitions ? ", ETA ~" + MESSAGE::formatDuration(eta) : ", done" )
                     << MESSAGE::cReset() << "\n" << MESSAGE::Flush;
            }
        }

        // fields were summed as density-weighted moments; normalize once now so cells drawing
        // streams from several partitions (multi-stream regions) come out correct
        uQuantities->normalizePhaseSpace( state.options.uField );
        aQuantities->normalizePhaseSpace( state.options.aField );

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
                    if (v > Real(1.)) ++multi;
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
    }
    else
    {
        // single triangulation over the whole (padded) box
        DTFE_preparePadding( state, userOptions );
        DTFE_parallel( &state.particles, samples, state.options, uQuantities, aQuantities );
    }

    // post-processing: derive velocity divergence/shear/vorticity and cosmic-web labels from the gradient
    computeDivergenceShearVorticity( userOptions.uField, userOptions.verboseLevel, uQuantities );
    computeDivergenceShearVorticity( userOptions.aField, userOptions.verboseLevel, aQuantities );
    computeWebClassification( userOptions.uField, userOptions.verboseLevel, userOptions.lambda_th, userOptions.hubbleParam, uQuantities );
    computeWebClassification( userOptions.aField, userOptions.verboseLevel, userOptions.lambda_th, userOptions.hubbleParam, aQuantities );
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
    // 1 thread, user sampling, redshift cone, or PS-DTFE (which partitions in Lagrangian space instead)
    if ( noAvailableProcessors==1 or not samples.empty() or userOptions.redshiftConeOn
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
Pvector<Real,NO_DIM> symmetricEigenvalues3x3(Real a00, Real a01, Real a02,
                                              Real a11, Real a12, Real a22)
{
    Pvector<Real,NO_DIM> eigenvalues;
#if NO_DIM == 3
    Real p1 = a01*a01 + a02*a02 + a12*a12;
    Real q = (a00 + a11 + a22) / Real(3.);  // trace / 3

    Real b00 = a00 - q, b11 = a11 - q, b22 = a22 - q;
    Real p2 = b00*b00 + b11*b11 + b22*b22 + Real(2.)*p1;
    Real p = std::sqrt(p2 / Real(6.));

    if ( p < Real(1.e-15) )
    {
        // matrix proportional to identity, already diagonal
        eigenvalues[0] = a00; eigenvalues[1] = a11; eigenvalues[2] = a22;
    }
    else
    {
        Real inv_p = Real(1.) / p;
        // B = (1/p) * (A - q*I), compute det(B)
        Real B00 = b00*inv_p, B01 = a01*inv_p, B02 = a02*inv_p;
        Real B11 = b11*inv_p, B12 = a12*inv_p, B22 = b22*inv_p;

        Real detB = B00*(B11*B22 - B12*B12) - B01*(B01*B22 - B12*B02) + B02*(B01*B12 - B11*B02);
        Real r = detB / Real(2.);

        // clamp to [-1, 1] for numerical safety before acos
        if (r <= Real(-1.)) r = Real(-1.);
        else if (r >= Real(1.)) r = Real(1.);

        Real phi = std::acos(r) / Real(3.);

        eigenvalues[0] = q + Real(2.) * p * std::cos(phi);
        eigenvalues[2] = q + Real(2.) * p * std::cos(phi + Real(2.) * M_PI / Real(3.));
        eigenvalues[1] = Real(3.) * q - eigenvalues[0] - eigenvalues[2]; // trace identity
    }

    // sort descending
    if (eigenvalues[0] < eigenvalues[1]) std::swap(eigenvalues[0], eigenvalues[1]);
    if (eigenvalues[1] < eigenvalues[2]) std::swap(eigenvalues[1], eigenvalues[2]);
    if (eigenvalues[0] < eigenvalues[1]) std::swap(eigenvalues[0], eigenvalues[1]);
#elif NO_DIM == 2
    // 2x2 symmetric matrix: eigenvalues from quadratic formula
    Real trace = a00 + a11;
    Real det = a00*a11 - a01*a01;
    Real disc = std::sqrt( std::max(trace*trace / Real(4.) - det, Real(0.)) );
    eigenvalues[0] = trace / Real(2.) + disc;
    eigenvalues[1] = trace / Real(2.) - disc;
#endif
    return eigenvalues;
}


// Classifies a cell by counting eigenvalues above lambda_th: 0=void, 1=wall, 2=filament, 3=node.
int classifyWeb(Pvector<Real,NO_DIM> const &eigenvalues, Real lambda_th)
{
    int label = 0;
    for (int d=0; d<NO_DIM; ++d)
        if (eigenvalues[d] > lambda_th) ++label;
    return label;
}


// V-web classification from the velocity shear tensor Sigma_ij = -(1/(2 H0))(dv_i/dx_j + dv_j/dx_i)
// (Hoffman et al. 2012). NOTE: this function previously also emitted a "T-web" computed from the
// SAME symmetrized velocity gradient -- algebraically identical to the V-web (the outputs were
// byte-identical), i.e. not a T-web at all. The physical T-web classifies the TIDAL tensor of the
// gravitational potential (Poisson-solved from the density grid) and is pure grid post-processing:
// it now lives in python/compute_tweb.py. Requesting 'tweb' here warns and is ignored.
void computeWebClassification(Field &fields,
                               int const verboseLevel,
                               Real lambda_th,
                               Real hubbleParam,
                               Quantities *q)
{
    if ( fields.velocity_tweb )
    {
        static bool warned = false;
        if ( !warned )
        {
            warned = true;
            MESSAGE::Warning warning( verboseLevel );
            warning << "The 'tweb' field was removed: the velocity-gradient \"T-web\" duplicated the "
                       "V-web exactly. Compute the true tidal-tensor T-web from the density grid with "
                       "python/compute_tweb.py; the V-web ('vweb'/'vweb_a') remains available here.\n"
                    << MESSAGE::EndWarning;
        }
        fields.velocity_tweb = false;   // skip downstream allocation checks and file output
    }
    if ( q->velocity_gradient.empty() ) return;
    if ( not fields.velocity_vweb ) return;

    // normalize the gradient by H0 = 100 h km/s/Mpc so the threshold lambda_th is dimensionless
    Real H0_norm = Real(1.);
    if ( hubbleParam > Real(0.) )
        H0_norm = Real(100.) * hubbleParam;  // H0 in km/s/Mpc

    size_t const N = q->velocity_gradient.size();

    // V-web: shear tensor Sigma_ij
    if ( fields.velocity_vweb )
    {
        q->velocity_vweb.reserve( N );
        q->velocity_vweb_eigenvalues.reserve( N );
        for (size_t i=0; i<N; ++i)
        {
            Pvector<Real,noGradComp> &g = q->velocity_gradient[i];
            Real norm = Real(-1.) / (Real(2.) * H0_norm);
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

    // free the gradient once the web classification no longer needs it
    if ( not fields.velocity_gradient )
        q->velocity_gradient.clear();
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
    
    // free the gradient unless the gradient itself or the web classification still needs it
    if ( not fields.velocity_gradient and not fields.velocity_tweb and not fields.velocity_vweb )
        q->velocity_gradient.clear();
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

    computeDivergenceShearVorticity( userOptions.uField, userOptions.verboseLevel, uQuantities );
    computeDivergenceShearVorticity( userOptions.aField, userOptions.verboseLevel, aQuantities );
    computeWebClassification( userOptions.uField, userOptions.verboseLevel, userOptions.lambda_th, userOptions.hubbleParam, uQuantities );
    computeWebClassification( userOptions.aField, userOptions.verboseLevel, userOptions.lambda_th, userOptions.hubbleParam, aQuantities );
}
#endif

