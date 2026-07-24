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


/* Nearest-Grid-Point (NGP) mass assignment: deposits each particle's mass and momentum
   entirely into its host cell. Also provides NGP_particle_count for load balancing. */


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
#include <boost/timer.hpp>

void printElapsedTime(boost::timer *t, User_options *userOptions,
                      std::string computationQuantityName);

using namespace std;
typedef vector<Particle_data>::iterator     vectorIterator;


void NGP_interpolation_regular_grid(vector<Particle_data> &particles,
                                    User_options &userOptions,
                                    Quantities *q);




// NGP (Nearest Grid Point) interpolation of density and velocity to a grid.
void NGP_interpolation(vector<Particle_data> *particles,
                       vector<Sample_point> &samples,
                       User_options &userOptions,
                       Quantities *q)
{
    if ( samples.empty() and not userOptions.redshiftConeOn )
        NGP_interpolation_regular_grid( *particles, userOptions, q );
    else
        throwError( "The NGP method can interpolate the fields only on a regular rectangular grid. No NGP interpolation methods are implemented for redshift cone coordinates or for user defined sample points." );
    particles->clear();
}





// NGP worker on a regular grid: assigns each particle to its nearest cell, accumulating mass and momentum,
// then divides momentum by cell mass for velocity and normalizes density to the background mean.
void NGP_interpolation_regular_grid(vector<Particle_data> &particles,
                                    User_options &userOptions,
                                    Quantities *q)
{
    size_t const *nGrid = &(userOptions.gridSize[0]);
    MESSAGE::Message message( userOptions.verboseLevel );
    message << "\nInterpolating the fields to the grid using the NGP method. The interpolation takes place inside the box of coordinates " << userOptions.region.print()
            << " on a " << MESSAGE::printElements( nGrid, NO_DIM, "*" ) << " grid ... " << MESSAGE::Flush;
    boost::timer t;
    t.restart();


    size_t reserveSize = 1;
    for (int d=0; d<NO_DIM; ++d) reserveSize *= nGrid[d];
    q->density.assign( reserveSize, Real(0.) );
    if ( userOptions.aField.velocity )
		q->velocity.assign( reserveSize, Pvector<Real,noVelComp>::zero() );


    Box box = userOptions.region;
    Real dx[NO_DIM];
    for (int i=0; i<NO_DIM; ++i)
        dx[i] = (box[2*i+1]-box[2*i]) / nGrid[i];


    // assign each particle to its nearest grid cell
    for (vectorIterator it=particles.begin(); it!=particles.end(); ++it)
	{
		int cell[NO_DIM];
		bool validCell = true;
		for (int j=0; j<NO_DIM; ++j)
		{
			cell[j] = int( floor( (it->position(j)-box[2*j])/dx[j] ) );
			if ( cell[j]<0 or cell[j]>=nGrid[j] )
				validCell = false;
		}
		if ( not validCell ) continue;
		// size_t: an int32 flat index overflows above ~1290^3 cells (2^31) -> heap corruption
		size_t index = 0;
		for (int d=0; d<NO_DIM; ++d) index = index * size_t(nGrid[d]) + size_t(cell[d]);
		q->density[index] += it->weight();
		if ( userOptions.aField.velocity )
			q->velocity[index] += it->velocity() * it->weight();
	}
    
    
    // convert accumulated momentum to velocity by dividing by cell mass
    if ( userOptions.aField.velocity )
        for (size_t i=0; i<reserveSize; ++i) {
            if ( q->density[i]!=Real(0.) ) {
                q->velocity[i] /= q->density[i];
            } else {
                q->velocity[i] = Pvector<Real,noVelComp>::zero();
            }
        }

    // normalize the density to the average background density
    if ( userOptions.aField.density ) {
        Real factor = Real( q->density.size() ) / box.volume() / userOptions.averageDensity;
        for (vector<Real>::iterator it=q->density.begin(); it!=q->density.end(); ++it)
            (*it) *= factor;
    } else {
        q->density.clear();
    }
    message << "Done.\n" << MESSAGE::Flush;
    printElapsedTime( &t, &userOptions, "NGP interpolation" );
}


// Count particles per grid cell (NGP assignment).
void NGP_particle_count(vector<Particle_data> &particles,
                        size_t const *nGrid,
                        Box box,
                        vector<int> *counts)
{
    size_t reserveSize = 1;
    for (int d=0; d<NO_DIM; ++d) reserveSize *= nGrid[d];
    counts->assign( reserveSize, int(0) );
	
    Real dx[NO_DIM];
    for (int i=0; i<NO_DIM; ++i)
        dx[i] = (box[2*i+1]-box[2*i]) / nGrid[i];


    for (vectorIterator it=particles.begin(); it!=particles.end(); ++it)
	{
		int cell[NO_DIM];
		bool validCell = true;
		for (int j=0; j<NO_DIM; ++j)
		{
			cell[j] = int( floor( (it->position(j)-box[2*j])/dx[j] ) );
			if ( cell[j]<0 or cell[j]>=nGrid[j] )
				validCell = false;
		}
		if ( not validCell ) continue;
		// size_t: an int32 flat index overflows above ~1290^3 cells (2^31) -> heap corruption.
		// This function runs during partition setup, BEFORE any split can reduce the grid.
		size_t index = 0;
		for (int d=0; d<NO_DIM; ++d) index = index * size_t(nGrid[d]) + size_t(cell[d]);
		(*counts)[index] += 1;
	}
}



