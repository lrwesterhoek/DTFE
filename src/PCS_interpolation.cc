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


/* Piecewise-Cubic-Spline (PCS) mass assignment: cubic (order-3) interpolation kernel spanning
   4 cells per dimension. Inner particles scatter freely; boundary particles deposit only into in-grid cells. */


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


void PCS_interpolation_regular_grid(vector<Particle_data> &particles,
                                    User_options &userOptions,
                                    Quantities *q);




// PCS (Piecewise Cubic Spline) interpolation of density and velocity to a grid.
void PCS_interpolation(vector<Particle_data> *particles,
                       vector<Sample_point> &samples,
                       User_options &userOptions,
                       Quantities *q)
{
    if ( samples.empty() and not userOptions.redshiftConeOn )
        PCS_interpolation_regular_grid( *particles, userOptions, q );
    else
        throwError( "The PCS method can interpolate the fields only on a regular rectangular grid. No PCS interpolation methods are implemented for redshift cone coordinates or for user defined sample points." );
    particles->clear();
}




// PCS kernel policy for the shared engine. Kernel W(x) = (1/6)(4-6x^2+3|x|^3) for |x|<1,
// (1/6)(2-|x|)^3 for 1<=|x|<2, else 0; support radius 2, so 4 cells per dimension. Boundary
// particles use the same weights with out-of-grid cells clipped (clipStencil, the engine).
struct PCS_kernel
{
    static int const STENCIL = 4;
    static constexpr double PAD_CELLS = 2.0;    // PCS kernel extends 2 cells from center
    // The 4 cells are the host cell f of (position - half a cell), i.e. the cell whose CENTRE is the nearest
    // one to the left, f-1 .. f+2, and s = 0.5 + temp the distance to f's centre -- the distances d0..d3 below.
    // Without the offset f would be the cell holding the particle and s its distance to f's LEFT EDGE:
    // every particle's mass would land half a cell further along every axis (the field moved by +dx/2)
    static constexpr double CELL_OFFSET = 0.5;
    static char const* name() { return "PCS"; }

    static void interiorWeights( Real temp, Real weight[4] )
    {
        // distances to the 4 cell centers (all positive), s = 0.5 + temp
        Real s = Real(0.5) + temp;
        Real d0 = Real(1.) + s;
        Real d1 = s;
        Real d2 = Real(1.) - s;
        Real d3 = Real(2.) - s;

        // PCS kernel weights
        Real one_sixth = Real(1.) / Real(6.);
        weight[0] = one_sixth * (Real(2.) - d0) * (Real(2.) - d0) * (Real(2.) - d0);
        weight[1] = one_sixth * (Real(4.) - Real(6.) * d1 * d1 + Real(3.) * d1 * d1 * d1);
        weight[2] = one_sixth * (Real(4.) - Real(6.) * d2 * d2 + Real(3.) * d2 * d2 * d2);
        weight[3] = one_sixth * (Real(2.) - d3) * (Real(2.) - d3) * (Real(2.) - d3);
    }

    static void boundaryCells( int tempInt, Real temp, size_t nGridJ, int /*j*/,
                               int cell[4], Real weight[4], int &cellCount )
    {
        clipStencil<PCS_kernel>( tempInt, temp, nGridJ, cell, weight, cellCount );
    }
};


// PCS worker on a regular grid (shared engine; see grid_deposit_engine.h).
void PCS_interpolation_regular_grid(vector<Particle_data> &particles,
                                    User_options &userOptions,
                                    Quantities *q)
{
    gridDeposit_regular_grid<PCS_kernel>( particles, userOptions, q );
}



