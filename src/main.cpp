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
#include <iomanip>
#include <chrono>           // total wall time for the run summary
#include <sys/resource.h>   // getrusage -> peak RSS report

#include "DTFE.h"
#include "io/io.h"
#include "interlacing.h"

using namespace std;


// Reads options/input, runs the interpolation (with optional interlacing), writes output, prints a run summary.
int main( int argc, char *argv[] )
{
    User_options userOptions;		// program options plus program-wide constants
    userOptions.readOptions( argc, argv );

    // wall-clock (not CPU) timer for the run summary, so it reflects elapsed time across all threads
    auto const wallStart = std::chrono::steady_clock::now();
    MESSAGE::Message phase( userOptions.verboseLevel );



    phase << MESSAGE::banner("READING INPUT") << MESSAGE::Flush;
    vector<Particle_data> particles;
    vector<Sample_point> samplingCoordinates; // sampling coords for non-regular-grid interpolation
    readInputData( &particles, &samplingCoordinates, &userOptions ); // may also set box dimensions / grid size



    Quantities uQuantities;	// unaveraged fields, sampled at the grid points
    Quantities aQuantities;     // fields volume-averaged over each sampling cell

    phase << MESSAGE::banner("DTFE COMPUTATION") << MESSAGE::Flush;
    if ( userOptions.interlace )
    {
        // Interlacing: interpolate twice (original + half-cell offset), average in Fourier space.
        if ( !userOptions.periodic )
        {
            MESSAGE::Warning warning( userOptions.verboseLevel );
            warning << "Interlacing is designed for periodic boundary conditions. Results may be unreliable for non-periodic data." << MESSAGE::EndWarning;
        }

        // DTFE clears the particles, so keep a copy for the second (offset) pass
        vector<Particle_data> particlesCopy = particles;

        DTFE( &particles, samplingCoordinates, userOptions, &uQuantities, &aQuantities );

        // first-pass density; prefer the averaged field, fall back to the unaveraged one
        vector<Real> density1;
        if ( !aQuantities.density.empty() )
            density1 = aQuantities.density;
        else if ( !uQuantities.density.empty() )
            density1 = uQuantities.density;

        if ( !density1.empty() )
        {
            // second pass: shift the box origin by half a cell along every axis
            User_options shiftedOptions = userOptions;
            shiftedOptions.interlace = false;  // guard against future re-entrant refactoring
            size_t const *nGrid = &(shiftedOptions.gridSize[0]);
            for (int d = 0; d < NO_DIM; ++d)
            {
                Real cellSize = (shiftedOptions.region[2*d+1] - shiftedOptions.region[2*d]) / nGrid[d];
                shiftedOptions.region[2*d]   += cellSize / Real(2.);
                shiftedOptions.region[2*d+1] += cellSize / Real(2.);
            }

            Quantities uQ2, aQ2;
            DTFE( &particlesCopy, samplingCoordinates, shiftedOptions, &uQ2, &aQ2 );

            // second-pass density, matched to whichever field the first pass used
            vector<Real> &density2 = !aQ2.density.empty() ? aQ2.density : uQ2.density;

            // cell size per axis, needed for the half-cell phase shift in Fourier space
            Real dx[NO_DIM];
            for (int d = 0; d < NO_DIM; ++d)
                dx[d] = (userOptions.region[2*d+1] - userOptions.region[2*d]) / nGrid[d];

            MESSAGE::Message message( userOptions.verboseLevel );
            message << "\nApplying interlacing (Fourier-space averaging) ... " << MESSAGE::Flush;
            applyInterlacing( density1, density2, nGrid, dx );
            message << "Done.\n" << MESSAGE::Flush;

            // write the interlaced density back into whichever field held the first pass
            if ( !aQuantities.density.empty() )
                aQuantities.density = density1;
            else
                uQuantities.density = density1;
        }
    }
    else
    {
        DTFE( &particles, samplingCoordinates, userOptions, &uQuantities, &aQuantities );  // clears 'particles'
    }



    phase << MESSAGE::banner("WRITING OUTPUT") << MESSAGE::Flush;
    writeOutputData( uQuantities, aQuantities, userOptions );


    // report peak RSS (to tune --max-concurrent / --partition to fit RAM) and total wall time
    if ( userOptions.verboseLevel >= 1 )
    {
        double const wallSec = std::chrono::duration<double>( std::chrono::steady_clock::now() - wallStart ).count();
        phase << MESSAGE::banner("SUMMARY");

        // ru_maxrss units differ by platform (bytes on macOS, kilobytes on Linux); convert to GB accordingly
        struct rusage ru;
        if ( getrusage( RUSAGE_SELF, &ru ) == 0 )
        {
#ifdef __APPLE__
            double const peakGB = double(ru.ru_maxrss) / (1024.0*1024.0*1024.0);  // ru_maxrss is BYTES on macOS
#else
            double const peakGB = double(ru.ru_maxrss) / (1024.0*1024.0);          // ru_maxrss is KILOBYTES on Linux
#endif
            std::cout << MESSAGE::cYellowD() << "   peak memory (RSS) : " << MESSAGE::cReset()
                      << MESSAGE::cMagenta() << std::setprecision(4) << peakGB << " GB" << MESSAGE::cReset() << "\n" << std::flush;
        }
        std::cout << MESSAGE::cYellowD() << "   total wall time   : " << MESSAGE::cReset()
                  << MESSAGE::cGreen() << MESSAGE::formatDuration(wallSec) << MESSAGE::cReset() << "\n"
                  << MESSAGE::cCyan() << MESSAGE::hrule() << MESSAGE::cReset() << "\n" << std::flush;
    }
}
