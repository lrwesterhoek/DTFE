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

#include <vector>
#include <cmath>
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




// splits the total data into different computation regions that are done by different threads (if OPEN_MP is enabled)
void DTFE_parallel(vector<Particle_data> *allParticles,
                   vector<Sample_point> &samples,
                   User_options & userOptions,
                   Quantities *uQuantities,
                   Quantities *aQuantities);

// Computes the velocity divergence, shear and vorticity from the velocity gradient
void computeDivergenceShearVorticity(Field &fields,
                                     int const verboseLevel,
                                     Quantities *quantities);

// Computes T-web and/or V-web cosmic web classification from velocity gradient
void computeWebClassification(Field &fields,
                               int const verboseLevel,
                               Real lambda_th,
                               Real hubbleParam,
                               Quantities *q);

// Computes velocity-space density g_i for approximate phase-space density (--approxPSD)
#if defined(VELOCITY) && defined(SCALAR) && !defined(PHASE_SPACE)
extern void computeVelocitySpaceDensity(vector<Particle_data> &particles,
                                         User_options &userOptions);
#endif


/* Holds the prepared particle data and options after common DTFE setup steps. */
struct DTFE_State {
    vector<Particle_data> particles;  // filtered/prepared particles
    User_options options;             // configured options for the computation
};


/* Common DTFE setup: generates random particles if requested, selects random
   subsample, runs consistency checks, configures velocity derivative options,
   selects particles in the region of interest, and handles partition with partNo.
   Modifies userOptions (averageDensity, region) as side effects. */
DTFE_State DTFE_setup(vector<Particle_data> *allParticles,
                      vector<Sample_point> &samples,
                      User_options &userOptions)
{
    DTFE_State state;

    //! If the user requested for a set of random particles, generate them
    if ( userOptions.poisson!=0 )
        randomParticles( allParticles, &userOptions );

    //! Select a random subset of the data if the user asked so
    vector<Particle_data> *particlePointer = allParticles;
    vector<Particle_data> particlesRandomSubsample;
    if (userOptions.randomSample>=Real(0.) )
    {
        randomSample( *particlePointer, &particlesRandomSubsample, userOptions );
        particlePointer->clear();
        particlePointer = &particlesRandomSubsample;
    }

    //! Do consistency checks and update some entries in the 'userOptions' structure
    userOptions.updateEntries( particlePointer->size(), not samples.empty() );
    if ( userOptions.averageDensity<0. )
        userOptions.averageDensity = averageDensity( *particlePointer, userOptions );

    // Configure velocity derivative options
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

    //! Select only the particles in the region of interest
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

    // Ensure state.particles holds the final particle data
    if ( particlePointer != &state.particles )
        state.particles = std::move(*particlePointer);

    return state;
}


/* Prepares padding for the non-partition, non-region case (periodic boxes). */
void DTFE_preparePadding(DTFE_State &state, User_options const &userOptions)
{
    if ( not userOptions.regionOn and not userOptions.partitionOn )
    {
        state.options.region = state.options.boxCoordinates;
        state.options.paddedBox = state.options.region;
        state.options.paddedBox.addPadding( state.options.paddingLength );

#ifndef PHASE_SPACE
        // Standard DTFE: add periodic copies based on Eulerian positions
        if ( userOptions.periodic )
        {
            vector<Particle_data> tempPart;
            findParticlesInBox( state.particles, &tempPart, state.options );
            state.particles = std::move(tempPart);
        }
#endif
        // PS-DTFE: Lagrangian periodic padding is handled earlier in DTFE()
    }
#ifdef PHASE_SPACE
    // PS-DTFE: keep periodic=true so interpolateGrid_phaseSpace does
    // periodic grid index wrapping. The Lagrangian copies and Eulerian
    // unwrapping handle the actual periodicity of the particle data.
#else
    state.options.periodic = false;
#endif
}


/* This function interpolates fields to a grid using the DTFE method.
NOTE: This function clears the vector 'allParticles'. */
void DTFE(vector<Particle_data> *allParticles,
          vector<Sample_point> &samples,
          User_options & userOptions,
          Quantities *uQuantities,
          Quantities *aQuantities)
{
    DTFE_State state = DTFE_setup( allParticles, samples, userOptions );

    // Approximate phase-space density: compute velocity-space density g_i for all
    // particles BEFORE any spatial partitioning or padding. This ensures the
    // velocity tessellation uses the global velocity distribution.
#if defined(VELOCITY) && defined(SCALAR) && !defined(PHASE_SPACE)
    if ( state.options.approxPSD && !state.particles.empty() )
        computeVelocitySpaceDensity( state.particles, state.options );
#endif

#ifdef PHASE_SPACE
    // Add Lagrangian periodic padding for PS-DTFE in periodic boxes.
    // The Delaunay triangulation is built in Lagrangian space, so periodic
    // copies must use shifted Lagrangian positions (not Eulerian).
    Box lagBoxGlobal;  // saved for later use by partitioning
    lagBoxGlobal.assign(Real(0.));
    bool hasLagrangianPeriodicCopies = false;
    if ( userOptions.periodic && !state.particles.empty() )
    {
        MESSAGE::Message msg( userOptions.verboseLevel );

        // Compute Lagrangian bounding box from original particles
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
        // Add tiny margin to include boundary particles
        for (int d=0; d<NO_DIM; ++d)
        {
            Real eps = (lagBoxGlobal[2*d+1] - lagBoxGlobal[2*d]) * Real(1.e-6);
            lagBoxGlobal[2*d]   -= eps;
            lagBoxGlobal[2*d+1] += eps;
        }

        // Compute Eulerian box length (needed for unwrapping and copy shifting)
        Real eulerLen[NO_DIM];
        for (int d=0; d<NO_DIM; ++d)
            eulerLen[d] = state.options.boxCoordinates[2*d+1] - state.options.boxCoordinates[2*d];

        // Compute Lagrangian padding length
        Real lagLength[NO_DIM], lagPad[NO_DIM];
        for (int d=0; d<NO_DIM; ++d)
        {
            lagLength[d] = lagBoxGlobal[2*d+1] - lagBoxGlobal[2*d];
            Real padFrac = (state.options.paddingLength[2*d] + state.options.paddingLength[2*d+1]) / (Real(2.) * eulerLen[d]);
            lagPad[d] = padFrac * lagLength[d];
            // Ensure sufficient padding: the reference Julia implementation uses
            // 5% of box size. Use 10% minimum for robustness in production runs.
            if (lagPad[d] < lagLength[d] * Real(0.10))
                lagPad[d] = lagLength[d] * Real(0.10);
            // When Lagrangian-space partitioning is enabled, each partition uses
            // 30% of its cell size as padding. The global periodic copies must
            // extend at least that far so that edge partitions have enough particles.
            if (userOptions.partitionOn && userOptions.partNo < 0)
            {
                Real partPad = lagLength[d] / Real(userOptions.partition[d]) * Real(0.30);
                if (lagPad[d] < partPad)
                    lagPad[d] = partPad;
            }
        }

        // Unwrap Eulerian positions so that the displacement s = pos - lagPos
        // is within [-L/2, L/2] for each dimension. This follows the reference
        // Julia implementation's unwrap_x_() function. Without this, particles
        // near the periodic boundary can have displacements close to L instead
        // of ~0, causing huge spurious Eulerian simplices.
        for (size_t i = 0; i < state.particles.size(); ++i)
            for (int d = 0; d < NO_DIM; ++d)
            {
                Real s = state.particles[i].pos[d] - state.particles[i].lagPos[d];
                // Wrap displacement to [-L/2, L/2]
                s = std::fmod(s + eulerLen[d] * Real(0.5), eulerLen[d]);
                if (s < Real(0.)) s += eulerLen[d];
                s -= eulerLen[d] * Real(0.5);
                state.particles[i].pos[d] = state.particles[i].lagPos[d] + s;
            }

        msg << "PS-DTFE: Unwrapped Eulerian positions (displacements within [-L/2, L/2]).\n" << MESSAGE::Flush;

        // Create padded Lagrangian box
        Box lagPaddedGlobal = lagBoxGlobal;
        for (int d=0; d<NO_DIM; ++d)
        {
            lagPaddedGlobal[2*d]   -= lagPad[d];
            lagPaddedGlobal[2*d+1] += lagPad[d];
        }

        // Add periodic copies from all 26 (3D) or 8 (2D) images.
        // Following the reference Julia implementation's frame() function,
        // BOTH Lagrangian and Eulerian positions are shifted by the same
        // box-length offset. This ensures that cells connecting original
        // and copy particles have correct Eulerian simplex geometry.
        size_t const noOriginal = state.particles.size();
        size_t const noOffsets = (NO_DIM == 2 ? 9 : 27);
        size_t const centerIdx = (NO_DIM == 2 ? 4 : 13);

        for (size_t n = 0; n < noOffsets; ++n)
        {
            if (n == centerIdx) continue;  // skip zero offset

            // Compute Lagrangian and Eulerian offsets for this periodic image
            Real lagOffset[NO_DIM], eulerOffset[NO_DIM];
            size_t rem = n;
            for (int d = NO_DIM - 1; d >= 0; --d)
            {
                int sign = int(rem % 3) - 1;  // -1, 0, or +1
                lagOffset[d]   = Real(sign) * lagLength[d];
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
        hasLagrangianPeriodicCopies = true;

        // Set lagrangianRegion so the ownership check in interpolateGrid_phaseSpace
        // rejects cells whose vertex 0 is a periodic copy (outside the original box).
        // For the partitioned path this is overridden per-partition with lagRegion.
        state.options.lagrangianRegion = lagBoxGlobal;
    }
#endif

#ifdef PHASE_SPACE
    if ( userOptions.partitionOn and userOptions.partNo<0 )
    {
        // PS-DTFE Lagrangian-space partitioning:
        // Standard Eulerian partitioning doesn't work because Lagrangian cells can
        // map to arbitrary Eulerian locations. Instead, we partition Lagrangian space
        // uniformly, and each partition writes to the FULL Eulerian grid.
        // Results are accumulated across partitions using addFrom().

        int totalPartitions = 1;
        for (int d=0; d<NO_DIM; ++d)
            totalPartitions *= state.options.partition[d];
        MESSAGE::Message message( userOptions.verboseLevel );
        message << "\nPS-DTFE: Lagrangian-space partitioning with " << totalPartitions
                << " partitions [" << MESSAGE::printElements( userOptions.partition, "," ) << "].\n" << MESSAGE::Flush;

        // Use the Lagrangian bounding box computed before periodic copies were added,
        // or compute it now for the non-periodic case.
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
            // Add tiny margin to include boundary particles
            for (int d=0; d<NO_DIM; ++d)
            {
                Real eps = (lagBox[2*d+1] - lagBox[2*d]) * Real(1.e-6);
                lagBox[2*d]   -= eps;
                lagBox[2*d+1] += eps;
            }
        }
        message << "PS-DTFE: Lagrangian bounding box = " << lagBox.print() << "\n" << MESSAGE::Flush;

        // Compute Lagrangian partition boundaries (uniform split)
        Real lagLen[NO_DIM];
        for (int d=0; d<NO_DIM; ++d)
            lagLen[d] = lagBox[2*d+1] - lagBox[2*d];

        // Compute padding length in Lagrangian space (same fraction as Eulerian padding)
        Real lagPadding[NO_DIM];
        for (int d=0; d<NO_DIM; ++d)
        {
            Real eulerLen = state.options.boxCoordinates[2*d+1] - state.options.boxCoordinates[2*d];
            Real padFrac = (state.options.paddingLength[2*d] + state.options.paddingLength[2*d+1]) / (Real(2.) * eulerLen);
            lagPadding[d] = padFrac * lagLen[d];
            if (lagPadding[d] < lagLen[d] / state.options.partition[d] * Real(0.3))
                lagPadding[d] = lagLen[d] / state.options.partition[d] * Real(0.3);
        }

        // Reserve memory for accumulator on the full grid
        uQuantities->reserveMemory( &(state.options.gridSize[0]), state.options.uField );
        aQuantities->reserveMemory( &(state.options.gridSize[0]), state.options.aField );

        // Also initialize stream_count if density is requested
        if ( state.options.uField.density )
        {
            size_t totalGrid = 1;
            for (int d=0; d<NO_DIM; ++d) totalGrid *= state.options.gridSize[d];
            uQuantities->stream_count.assign(totalGrid, Real(0.));
        }

        // Prepare Eulerian region/padding for the full box (needed by DTFE_interpolation)
        if ( not userOptions.regionOn and not (userOptions.partitionOn and userOptions.partNo>=0) )
        {
            state.options.region = state.options.boxCoordinates;
            state.options.paddedBox = state.options.region;
            state.options.paddedBox.addPadding( state.options.paddingLength );
        }
        // Keep periodic=true for PS-DTFE: interpolateGrid_phaseSpace needs it
        // for periodic grid index wrapping. The Lagrangian copies and Eulerian
        // unwrapping already handle the actual periodicity of the particle data;
        // this flag only controls the grid-level wrapping in the interpolation.

        // Iterate over Lagrangian partitions sequentially
        size_t partGrid[NO_DIM];
        for (int d=0; d<NO_DIM; ++d)
            partGrid[d] = state.options.partition[d];

        for (int pi=0; pi<totalPartitions; ++pi)
        {
            // Compute multi-dimensional partition index
            int idx[NO_DIM];
            {
                int rem = pi;
                for (int d=NO_DIM-1; d>=0; --d)
                {
                    idx[d] = rem % partGrid[d];
                    rem /= partGrid[d];
                }
            }

            // Compute Lagrangian region (unpadded) for this partition
            Box lagRegion;
            for (int d=0; d<NO_DIM; ++d)
            {
                lagRegion[2*d]   = lagBox[2*d] + lagLen[d] * idx[d] / partGrid[d];
                lagRegion[2*d+1] = lagBox[2*d] + lagLen[d] * (idx[d]+1) / partGrid[d];
            }

            // Compute padded Lagrangian box
            Box lagPadded = lagRegion;
            for (int d=0; d<NO_DIM; ++d)
            {
                lagPadded[2*d]   -= lagPadding[d];
                lagPadded[2*d+1] += lagPadding[d];
                // Clamp to overall Lagrangian box only when no periodic copies are available
                if (!hasLagrangianPeriodicCopies)
                {
                    if (lagPadded[2*d]   < lagBox[2*d])   lagPadded[2*d]   = lagBox[2*d];
                    if (lagPadded[2*d+1] > lagBox[2*d+1]) lagPadded[2*d+1] = lagBox[2*d+1];
                }
            }

            message << "\n<<< PS-DTFE partition " << pi+1 << "/" << totalPartitions
                    << " | Lagrangian region " << lagRegion.print()
                    << " | padded " << lagPadded.print() << "\n" << MESSAGE::Flush;

            // Select particles by Lagrangian position
            vector<Particle_data> tempPart;
            findParticlesInBoxLagrangian( state.particles, &tempPart, lagPadded, userOptions.verboseLevel );

            if (tempPart.empty())
            {
                message << "  No particles in this partition, skipping.\n" << MESSAGE::Flush;
                continue;
            }

            // Set up options for this partition — uses the FULL Eulerian grid
            User_options tempOpt = state.options;
            tempOpt.lagrangianRegion = lagRegion;  // for cell ownership check

            // Run DTFE interpolation (routes through DTFE_parallel which bypasses
            // OpenMP spatial partitioning for PS-DTFE, so effectively single-threaded)
            Quantities temp_uQuantities, temp_aQuantities;
            DTFE_parallel( &tempPart, samples, tempOpt, &temp_uQuantities, &temp_aQuantities );

            // Accumulate results into main grid
            uQuantities->addFrom( temp_uQuantities );
            aQuantities->addFrom( temp_aQuantities );

            message << ">>> Partition " << pi+1 << " done.\n" << MESSAGE::Flush;
        }
    }
    else
#endif // PHASE_SPACE

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

        for (int i=0; i<totalPartitions; ++i)
        {
            message << "\n<<< Interpolating the fields for partition " << i+1 << " ...\n";

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

            copySubgridResultsToMain( temp_uQuantities, state.options.gridSize, tempOpt.uField, tempOpt, subgridList, uQuantities );
            copySubgridResultsToMain( temp_aQuantities, state.options.gridSize, tempOpt.aField, tempOpt, subgridList, aQuantities );
        }
    }
    else
    {
        DTFE_preparePadding( state, userOptions );
        DTFE_parallel( &state.particles, samples, state.options, uQuantities, aQuantities );
    }

    computeDivergenceShearVorticity( userOptions.uField, userOptions.verboseLevel, uQuantities );
    computeDivergenceShearVorticity( userOptions.aField, userOptions.verboseLevel, aQuantities );
    computeWebClassification( userOptions.uField, userOptions.verboseLevel, userOptions.lambda_th, userOptions.hubbleParam, uQuantities );
    computeWebClassification( userOptions.aField, userOptions.verboseLevel, userOptions.lambda_th, userOptions.hubbleParam, aQuantities );
}



// computes the actual Delaunay triangulation and the interpolation to grid
extern void DTFE_interpolation(vector<Particle_data> *p,
                               vector<Sample_point> &samples,
                               User_options &userOptions,
                               Quantities *uQuantities,
                               Quantities *aQuantities);

/* Calls the function corresponding to the chosen interpolation method. */
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




/* This function splits the data in several partitions that can be used by shared memory processors to do the computations in parallel. It uses the OpenMP directives for doing so. */
void DTFE_parallel(vector<Particle_data> *allParticles,
                   vector<Sample_point> &samples,
                   User_options & userOptions,
                   Quantities *uQuantities,
                   Quantities *aQuantities)
{
#ifndef OPEN_MP
    //directly compute the DTFE interpolation
    interpolate( allParticles, samples, userOptions, uQuantities, aQuantities );  // this function deletes the vector 'allParticles'
    return;
#else
    int const noAvailableProcessors = omp_get_max_threads();
    
    // if only 1 processor is available, do nothing
    // PS-DTFE requires a single global triangulation — spatial partitioning is incompatible
    if ( noAvailableProcessors==1 or not samples.empty() or userOptions.redshiftConeOn
#ifdef PHASE_SPACE
         or true  // PS-DTFE: bypass spatial partitioning, use single triangulation
#endif
       )
    {
        interpolate( allParticles, samples, userOptions, uQuantities, aQuantities );
        return;
    }
    
    
    // if more than 1 processor is available, split the data in partitions
    std::vector<size_t> pGrid(NO_DIM,0); // the parallel grid
    parallelGrid( noAvailableProcessors, userOptions, &(pGrid[0]) );
    int noProcessors = 1;
    for (int d=0; d<NO_DIM; ++d) noProcessors *= pGrid[d];    //number of processors actually used (may be different from 'noAvailableProcessors')
    
    // reserve memory for the quantities of interest
    uQuantities->reserveMemory( &(userOptions.gridSize[0]), userOptions.uField );
    aQuantities->reserveMemory( &(userOptions.gridSize[0]), userOptions.aField );
    // define some variables to keep track of the time and particle numbers associated to each processor
    MESSAGE::Message message( userOptions.verboseLevel );
    message << "From now on only the master thread will show messages on how the computation is going. Not all threads take the same execution time, so there may be a discrepancy between the messages displayed to the user and the computations across all threads.\n\n" << MESSAGE::Flush;
    std::vector<size_t> processorParticles(noProcessors);    // number of particles associated to each processor
    std::vector<Real> processorTime(noProcessors);           // the actual runtime of each thread
    size_t const noTotalParticles = allParticles->size();
    
    
    // determine the optimal grid to split the computational load
    std::vector< std::vector<size_t> > subgridList;
    std::vector< Box > subgridCoords;
    optimalPartitionSplit( *allParticles, userOptions, pGrid, &subgridList, &subgridCoords );
    
    
    // this contains the parallel section
#pragma omp parallel num_threads( noProcessors )
    {
        int const threadNo = omp_get_thread_num(); // the number of the thread - between 0 to noProcessors
        User_options tempOptions = userOptions;
        tempOptions.noProcessors = noProcessors;    // the number of parallel threads
        tempOptions.threadId = threadNo;            // the id of the thread
        tempOptions.verboseLevel = (userOptions.verboseLevel>0) ? 1 : userOptions.verboseLevel;   // the slave processors will not show runtime messages (with exception of errors and warnings)
        
        if ( threadNo==0 )
            tempOptions.verboseLevel = userOptions.verboseLevel;    // only the master will show runtime messages
        tempOptions.partNo = threadNo;
        tempOptions.partition.clear();
        for (int i=0; i<NO_DIM; ++i)
            tempOptions.partition.push_back( pGrid[i] );
        tempOptions.updateFullBox( userOptions.region );    // this is the new full box
        
        
        // compute the region allocated to each processor
        copySubgridInformation( &tempOptions, subgridList, subgridCoords ); // writes the size of the subgrid to tempOptions.gridSize and the boundaries of the subgrid box to tempOptions.region
        tempOptions.paddedBox = tempOptions.region;
        tempOptions.paddedBox.addPadding( tempOptions.paddingLength );
        
        
        // find the particles in the 'paddedBox'
        vector<Particle_data> particles;
        findParticlesInBox( *allParticles, &particles, tempOptions );
        processorParticles[ threadNo ] = particles.size();
        
        
        //compute the DTFE interpolation
        Quantities temp_uQuantities, temp_aQuantities; // temporary variable to store the grid interpolation for each subgrid
        interpolate( &particles, samples, tempOptions, &temp_uQuantities, &temp_aQuantities ); //this function deletes the vector 'particles'
        processorTime[ threadNo ] = tempOptions.totalTime;  // this is the total CPU time for all threads until this thread ended
        
        
        // write the fields from the given partition to the main grid results
        copySubgridResultsToMain( temp_uQuantities, userOptions.gridSize, tempOptions.uField, tempOptions, subgridList, uQuantities );
        copySubgridResultsToMain( temp_aQuantities, userOptions.gridSize, tempOptions.aField, tempOptions, subgridList, aQuantities );  //copy the results from each processor to a main variable result
        if ( threadNo==0 )
            message << "\nWaiting for all threads to finish the computations ... " << MESSAGE::Flush;
    }
    message << "Done.\n";
    allParticles->clear();
    
    
    // show statistics to the user about the execution in parallel
    userOptions.totalTime += maximum( processorTime.data(), noProcessors );    // update total time by the largest value of CPU time
    approximativeThreadTime( processorTime.data(), noProcessors );   // compute approximative CPU times for each thread
    message << "Statistics of the execution across the " << noProcessors << " threads:\n";
    for (int i=0; i<noProcessors; ++i)
        message << "\t Thread " << i << " had " << processorParticles[i] << " particles (which represent " << setprecision(4) <<  Real(processorParticles[i])/noTotalParticles*100. << "\%) and took " << processorTime[i] << " sec. \n";
    message << MESSAGE::Flush;
    
#endif
}






//! Velocity derivative quantities
/* Computes the velocity divergence from the velocity gradient. */
Real velocityDivergence(Pvector<Real,noGradComp> &velGrad)
{
    Real div = 0.;
    for (int d=0; d<NO_DIM; ++d)
        div += velGrad[d*NO_DIM+d];
    return div;
}

/* Computes the velocity shear from the velocity gradient. */
Pvector<Real,noShearComp> velocityShear(Pvector<Real,noGradComp> &velGrad)
{
    Pvector<Real,noShearComp> temp;
    size_t index = 0;
    for (int i=0; i<NO_DIM-1; ++i)
        for (int j=i; j<NO_DIM; ++j)
            temp[index++] = (velGrad[i*NO_DIM+j] + velGrad[j*NO_DIM+i]) / 2.;
    
    // still need to substract the trace since the shear matrix is traceless
    Real trace = velGrad[(NO_DIM-1)*NO_DIM+(NO_DIM-1)];
    for (int d=0; d<NO_DIM-1; ++d)
        trace += temp[d*NO_DIM - d*(d-1)/2];
    trace /= NO_DIM;
    for (int d=0; d<NO_DIM-1; ++d)
        temp[d*NO_DIM - d*(d-1)/2] -= trace;
    return temp;
}

/* Computes the velocity shear from the velocity gradient. */
Pvector<Real,noVortComp> velocityVorticity(Pvector<Real,noGradComp> &velGrad)
{
    Pvector<Real,noVortComp> temp;
    size_t index = 0;
    for (int i=0; i<NO_DIM; ++i)
        for (int j=i+1; j<NO_DIM; ++j)
            temp[index++] = (velGrad[i*NO_DIM+j] - velGrad[j*NO_DIM+i]) / 2.;
    return temp;
}

/* Compute eigenvalues of a 3x3 symmetric matrix using Cardano's analytical formula.
   Input: symmetric matrix elements a[0..5] stored as {a00, a01, a02, a11, a12, a22}.
   Output: eigenvalues sorted in descending order (lambda1 >= lambda2 >= lambda3). */
Pvector<Real,NO_DIM> symmetricEigenvalues3x3(Real a00, Real a01, Real a02,
                                              Real a11, Real a12, Real a22)
{
    Pvector<Real,NO_DIM> eigenvalues;
#if NO_DIM == 3
    // Trace and cofactors
    Real p1 = a01*a01 + a02*a02 + a12*a12;
    Real q = (a00 + a11 + a22) / Real(3.);  // trace / 3

    Real b00 = a00 - q, b11 = a11 - q, b22 = a22 - q;
    Real p2 = b00*b00 + b11*b11 + b22*b22 + Real(2.)*p1;
    Real p = std::sqrt(p2 / Real(6.));

    if ( p < Real(1.e-15) )
    {
        // Matrix is already diagonal (proportional to identity)
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

        // Clamp to [-1, 1] for numerical safety
        if (r <= Real(-1.)) r = Real(-1.);
        else if (r >= Real(1.)) r = Real(1.);

        Real phi = std::acos(r) / Real(3.);

        eigenvalues[0] = q + Real(2.) * p * std::cos(phi);
        eigenvalues[2] = q + Real(2.) * p * std::cos(phi + Real(2.) * M_PI / Real(3.));
        eigenvalues[1] = Real(3.) * q - eigenvalues[0] - eigenvalues[2]; // trace identity
    }

    // Sort descending: lambda1 >= lambda2 >= lambda3
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


/* Classify a grid cell based on eigenvalue threshold:
   count eigenvalues > lambda_th → 0=void, 1=wall, 2=filament, 3=node. */
int classifyWeb(Pvector<Real,NO_DIM> const &eigenvalues, Real lambda_th)
{
    int label = 0;
    for (int d=0; d<NO_DIM; ++d)
        if (eigenvalues[d] > lambda_th) ++label;
    return label;
}


/* Compute T-web and/or V-web classification from velocity gradient tensor.
   T-web uses the symmetrized velocity gradient (tidal tensor approximation):
     T_ij = -(1/H0) * (dv_i/dx_j + dv_j/dx_i) / 2
   V-web uses the velocity shear tensor:
     Sigma_ij = -(1/(2*H0)) * (dv_i/dx_j + dv_j/dx_i)
   Both are identical up to a factor of 2 convention; here we use the same
   symmetric tensor for both (the eigenvalue threshold absorbs the factor). */
void computeWebClassification(Field &fields,
                               int const verboseLevel,
                               Real lambda_th,
                               Real hubbleParam,
                               Quantities *q)
{
    if ( q->velocity_gradient.empty() ) return;
    if ( not fields.velocity_tweb and not fields.velocity_vweb ) return;

    Real H0_norm = Real(1.);
    if ( hubbleParam > Real(0.) )
        H0_norm = Real(100.) * hubbleParam;  // H0 in km/s/Mpc

    size_t const N = q->velocity_gradient.size();

    // T-web classification
    if ( fields.velocity_tweb )
    {
        q->velocity_tweb.reserve( N );
        q->velocity_tweb_eigenvalues.reserve( N );
        for (size_t i=0; i<N; ++i)
        {
            Pvector<Real,noGradComp> &g = q->velocity_gradient[i];
            // Symmetrize and normalize: T_ij = -(1/H0) * (dv_i/dx_j + dv_j/dx_i) / 2
            Real norm = Real(-1.) / H0_norm;
            Real s00 = norm * g[0*NO_DIM+0];
            Real s11 = norm * g[1*NO_DIM+1];
            Real s01 = norm * (g[0*NO_DIM+1] + g[1*NO_DIM+0]) / Real(2.);
#if NO_DIM == 3
            Real s22 = norm * g[2*NO_DIM+2];
            Real s02 = norm * (g[0*NO_DIM+2] + g[2*NO_DIM+0]) / Real(2.);
            Real s12 = norm * (g[1*NO_DIM+2] + g[2*NO_DIM+1]) / Real(2.);
            Pvector<Real,NO_DIM> eig = symmetricEigenvalues3x3(s00, s01, s02, s11, s12, s22);
#elif NO_DIM == 2
            Pvector<Real,NO_DIM> eig = symmetricEigenvalues3x3(s00, s01, Real(0.), s11, Real(0.), Real(0.));
#endif
            q->velocity_tweb_eigenvalues.push_back( eig );
            q->velocity_tweb.push_back( Real(classifyWeb(eig, lambda_th)) );
        }
    }

    // V-web classification (same tensor, possibly different threshold convention)
    if ( fields.velocity_vweb )
    {
        q->velocity_vweb.reserve( N );
        q->velocity_vweb_eigenvalues.reserve( N );
        for (size_t i=0; i<N; ++i)
        {
            Pvector<Real,noGradComp> &g = q->velocity_gradient[i];
            // V-web: Sigma_ij = -(1/(2*H0)) * (dv_i/dx_j + dv_j/dx_i)
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

    // Clear gradient if no one else needs it
    if ( not fields.velocity_gradient )
        q->velocity_gradient.clear();
}


/* This function computes the velocity divergence, shear and vorticity. */
void computeDivergenceShearVorticity(Field &fields,
                                     int const verboseLevel,
                                     Quantities *q)
{
    if ( q->velocity_gradient.empty() ) return;
    
    // compute the velocity divergence
    if ( fields.velocity_divergence )
    {
        q->velocity_divergence.reserve( q->velocity_gradient.size() );
        for (std::vector< Pvector<Real,noGradComp> >::iterator it=q->velocity_gradient.begin(); it!=q->velocity_gradient.end(); ++it )
            q->velocity_divergence.push_back( velocityDivergence(*it) );
        
//        for (int i=0; i<q->velocity_divergence.size(); ++i)
//            if ( not boost::math::isfinite(q->velocity_divergence[i]) )
//            {
//                q->velocity_divergence[i] = Real(0.);
//                std::cout << "<<< Found non-numerical value in velocity divergence at array index " << i << ". The value of the velocity divergence at this grid point will be set to 0." << std::flush;
//            }
    }
    
    // compute the velocity shear
    if ( fields.velocity_shear )
    {
        q->velocity_shear.reserve( q->velocity_gradient.size() );
        for (std::vector< Pvector<Real,noGradComp> >::iterator it=q->velocity_gradient.begin(); it!=q->velocity_gradient.end(); ++it )
            q->velocity_shear.push_back( velocityShear(*it) );
    }
    
    // compute the velocity vorticity
    if ( fields.velocity_vorticity )
    {
        q->velocity_vorticity.reserve( q->velocity_gradient.size() );
        for (std::vector< Pvector<Real,noGradComp> >::iterator it=q->velocity_gradient.begin(); it!=q->velocity_gradient.end(); ++it )
            q->velocity_vorticity.push_back( velocityVorticity(*it) );
    }
    
    if ( not fields.velocity_gradient and not fields.velocity_tweb and not fields.velocity_vweb )
        q->velocity_gradient.clear();
}





// intialize the static members
#ifndef VELOCITY
Pvector<Real,noVelComp> Data_structure::_velocity = Pvector<Real,noVelComp>::zero();
#endif
#ifndef SCALAR
Pvector<Real,noScalarComp> Data_structure::_scalar = Pvector<Real,noScalarComp>::zero();
#endif






#ifdef TRIANGULATION
#include "DTFE.h"

// computes the actual Delaunay triangulation and the interpolation to grid
extern void DTFE_interpolation(vector<Particle_data> *p,
                               vector<Sample_point> &samples,
                               User_options &userOptions,
                               Quantities *uQuantities,
                               Quantities *aQuantities,
                               DT &delaunay_triangulation);

/* This function interpolates fileds to a grid using the DTFE method.
It takes the following arguments:
        allParticles - stores the particle positions, velocities, scalars (if any) and weights
        samples - stores the position of the grid points where samples are taken (if this grid points don't have a uniform distribution)
        userOptions - options specified by the user plus additional program wise variables
        quantities - structure that stores vectors for all the allowed quantities that can be computed in the function
        delaunay_triangulation - returns the Delaunay triangulation of the point distribution

NOTE: This function clears the vector 'allParticles'. */
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

