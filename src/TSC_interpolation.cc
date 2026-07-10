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


/* Triangular-Shaped-Cloud (TSC) mass assignment: quadratic (order-2) interpolation kernel spanning
   3 cells per dimension. Inner particles scatter freely; boundary particles deposit only into in-grid cells. */


#include <vector>
#include <list>
#include <string>
#include <cmath>

#include "define.h"
#include "particle_data.h"
#include "quantities.h"
#include "box.h"
#include "user_options.h"
#include "message.h"
#include "grid_deposit_engine.h"
#include <boost/timer.hpp>

using namespace std;


void TSC_interpolation_regular_grid(vector<Particle_data> &particles,
                                    User_options &userOptions,
                                    Quantities *q);




// TSC (Triangular Shaped Cloud) interpolation of density and velocity to a grid.
void TSC_interpolation(vector<Particle_data> *particles,
                       vector<Sample_point> &samples,
                       User_options &userOptions,
                       Quantities *q)
{
    if ( samples.empty() and not userOptions.redshiftConeOn )
        TSC_interpolation_regular_grid( *particles, userOptions, q );
    else
        throwError( "The TSC method can interpolate the fields only on a regular rectangular grid. No TSC interpolation methods are implemented for redshift cone coordinates or for user defined sample points." );
    particles->clear();
}





// TSC kernel policy for the shared engine: quadratic weights over a 3-cell stencil, with the
// original edge-case handling for boundary particles.
struct TSC_kernel
{
    static int const STENCIL = 3;
    static constexpr double PAD_CELLS = 1.0;
    static char const* name() { return "TSC"; }

    static void interiorWeights( Real temp, Real weight[3] )
    {
        weight[0] = .5 * (0.5-temp) * (0.5-temp);
        weight[1] = .75 - temp * temp;
        weight[2] = .5 * (0.5+temp) * (0.5+temp);
    }

    static void boundaryCells( int tempInt, Real temp, size_t nGridJ, int j,
                               int cell[3], Real weight[3], int &cellCount )
    {
        if ( temp<-0.5 or temp>0.5 ) cout << temp << "\t" << j << "\t" << tempInt << "\n";

        if ( tempInt==-1 )
        {
            cell[0] = 0; cellCount = 1;
            weight[0] = .5 * (0.5+temp) * (0.5+temp);
        }
        else if ( tempInt==0 )
        {
            cell[0] = 0; cell[1] = 1; cellCount = 2;
            weight[0] = .75 - temp * temp;
            weight[1] = .5 * (0.5+temp) * (0.5+temp);
        }
        else if ( tempInt==int(nGridJ)-1 )
        {
            cell[0] = nGridJ-2; cell[1] = nGridJ-1; cellCount = 2;
            weight[0] = .5 * (0.5-temp) * (0.5-temp);
            weight[1] = .75 - temp * temp;
        }
        else if ( tempInt==int(nGridJ) )
        {
            cell[0] = nGridJ-1; cellCount = 1;
            weight[0] = .5 * (0.5-temp) * (0.5-temp);
        }
        else if ( not( tempInt<-1 or tempInt>int(nGridJ) ) )
        {
            cell[0] = tempInt-1;
            cell[1] = tempInt;
            cell[2] = tempInt+1;
            cellCount = 3;
            interiorWeights( temp, weight );
        }
    }
};


// TSC worker on a regular grid (shared engine; see grid_deposit_engine.h).
void TSC_interpolation_regular_grid(vector<Particle_data> &particles,
                                    User_options &userOptions,
                                    Quantities *q)
{
    gridDeposit_regular_grid<TSC_kernel>( particles, userOptions, q );
}




