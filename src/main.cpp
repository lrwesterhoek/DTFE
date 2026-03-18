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

#include "DTFE.h"
#include "io/io.h"
#include "interlacing.h"

using namespace std;


int main( int argc, char *argv[] )
{
    User_options userOptions;		// stores the program options supplied by the user plus additional program wide constants
    userOptions.readOptions( argc, argv );  // read the user supplied options



    // read the input data
    vector<Particle_data> particles;	// vector that keeps track of particle positions and their data
    vector<Sample_point> samplingCoordinates; // keep track of the sampling coordinates if the grid interpolation is done to a non-regular grid
    readInputData( &particles, &samplingCoordinates, &userOptions ); // reads particle positions, non-regular sampling coordinates (if any). Can also set members of the 'User_options' class like the box dimensions or grid size.



    // compute the DTFE (with optional interlacing)
    Quantities uQuantities;	// structure that will store the output quantities - i.e. the fields at the sampling points (NOTE: "uQuantities" stands for "unaveraged quantities" meaning that this fields were NOT averaged over the sampling cell)
    Quantities aQuantities;     // structure that will store the volume averaged output quantities - i.e. the fields volume averaged over the sampling cells (NOTE: "aQuantities" stands for "averaged quantities")

    if ( userOptions.interlace )
    {
        // Interlacing: run interpolation twice (original + half-cell offset), average in Fourier space
        if ( !userOptions.periodic )
        {
            MESSAGE::Warning warning( userOptions.verboseLevel );
            warning << "Interlacing is designed for periodic boundary conditions. Results may be unreliable for non-periodic data." << MESSAGE::EndWarning;
        }

        // Keep a copy of particles for the second pass (DTFE clears them)
        vector<Particle_data> particlesCopy = particles;

        // First pass: normal grid
        DTFE( &particles, samplingCoordinates, userOptions, &uQuantities, &aQuantities );

        // Save the density from the first pass
        vector<Real> density1;
        if ( !aQuantities.density.empty() )
            density1 = aQuantities.density;
        else if ( !uQuantities.density.empty() )
            density1 = uQuantities.density;

        if ( !density1.empty() )
        {
            // Second pass: shift box origin by half a cell
            User_options shiftedOptions = userOptions;
            shiftedOptions.interlace = false;  // prevent infinite recursion in case of future refactoring
            size_t const *nGrid = &(shiftedOptions.gridSize[0]);
            for (int d = 0; d < NO_DIM; ++d)
            {
                Real cellSize = (shiftedOptions.region[2*d+1] - shiftedOptions.region[2*d]) / nGrid[d];
                shiftedOptions.region[2*d]   += cellSize / Real(2.);
                shiftedOptions.region[2*d+1] += cellSize / Real(2.);
            }

            Quantities uQ2, aQ2;
            DTFE( &particlesCopy, samplingCoordinates, shiftedOptions, &uQ2, &aQ2 );

            // Get the density from the second pass
            vector<Real> &density2 = !aQ2.density.empty() ? aQ2.density : uQ2.density;

            // Apply Fourier-space interlacing
            Real dx[NO_DIM];
            for (int d = 0; d < NO_DIM; ++d)
                dx[d] = (userOptions.region[2*d+1] - userOptions.region[2*d]) / nGrid[d];

            MESSAGE::Message message( userOptions.verboseLevel );
            message << "\nApplying interlacing (Fourier-space averaging) ... " << MESSAGE::Flush;
            applyInterlacing( density1, density2, nGrid, dx );
            message << "Done.\n" << MESSAGE::Flush;

            // Store result back
            if ( !aQuantities.density.empty() )
                aQuantities.density = density1;
            else
                uQuantities.density = density1;
        }
    }
    else
    {
        DTFE( &particles, samplingCoordinates, userOptions, &uQuantities, &aQuantities );  // this function deletes the information in 'particles'
    }



    // output the desired quantities to file/files
    writeOutputData( uQuantities, aQuantities, userOptions );
}
