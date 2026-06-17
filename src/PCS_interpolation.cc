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
#include <boost/timer.hpp>

void printElapsedTime(boost::timer *t, User_options *userOptions,
                      std::string computationQuantityName);

using namespace std;
typedef vector<Particle_data>::iterator     vectorIterator;


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




// PCS worker on a regular grid: scatters each particle's mass and momentum, then divides momentum by
// cell mass for velocity and normalizes density to the background mean. Kernel W(x) = (1/6)(4-6x^2+3|x|^3)
// for |x|<1, (1/6)(2-|x|)^3 for 1<=|x|<2, else 0; support radius 2, so 4 cells per dimension.
void PCS_interpolation_regular_grid(vector<Particle_data> &particles,
                                    User_options &userOptions,
                                    Quantities *q)
{
    size_t const *nGrid = &(userOptions.gridSize[0]);
    MESSAGE::Message message( userOptions.verboseLevel );
    message << "\nInterpolating the fields to the grid using the PCS method. The interpolation takes place inside the box of coordinates " << userOptions.region.print()
            << " on a " << MESSAGE::printElements( nGrid, NO_DIM, "*" ) << " grid ... " << MESSAGE::Flush;
    boost::timer t;
    t.restart();


    size_t reserveSize = 1;
    for (int d=0; d<NO_DIM; ++d) reserveSize *= nGrid[d];
    q->density.assign( reserveSize, Real(0.) );
    q->velocity.assign( reserveSize, Pvector<Real,noVelComp>::zero() );


    // grid spacing, plus inner/outer boxes selecting particles that contribute to the region
    Box box = userOptions.region,
        outerBox = userOptions.region,
        innerBox = userOptions.region;
    Real dx[NO_DIM], outerPadding[2*NO_DIM], innerPadding[2*NO_DIM];
    for (int i=0; i<NO_DIM; ++i)
    {
        dx[i] = (box[2*i+1]-box[2*i]) / nGrid[i];
        outerPadding[2*i] = 2.0 * dx[i];    // PCS kernel extends 2 cells from center
        outerPadding[2*i+1] = 2.0 * dx[i];
    }
    for (int i=0; i<2*NO_DIM; ++i)
        innerPadding[i] = -1.1*outerPadding[i];    // slightly larger than 2 grid cells to account for numerical uncertainties
    outerBox.addPadding( outerPadding );  // only particles in this box give contributions in 'userOptions.region'
    innerBox.addPadding( innerPadding );  // the contribution of particles in this box is limited to the region of interest


    // inner particles contribute only inside the region; outer ones also outside, so cell validity is checked
    list< vectorIterator > innerParticles, outerParticles;
    for (vectorIterator it=particles.begin(); it!=particles.end(); ++it)
        if ( innerBox.isParticleInBox(*it) )
            innerParticles.push_back( it );
        else if ( outerBox.isParticleInBox(*it) )
            outerParticles.push_back( it );



    for (list< vectorIterator >::iterator it=innerParticles.begin(); it!=innerParticles.end(); ++it)
    {
        int cell[NO_DIM][4];    // grid coords of the cells the particle contributes to
        Real weight[NO_DIM][4]; // weight per contributing cell
        for (int j=0; j<NO_DIM; ++j)
        {
            Real temp = ( (*it)->position(j) - box[2*j] ) / dx[j];
            cell[j][0] = int(floor( temp )) - 1;
            cell[j][1] = cell[j][0] + 1;
            cell[j][2] = cell[j][1] + 1;
            cell[j][3] = cell[j][2] + 1;

            temp -= cell[j][1] + 0.5; // distance of particle from the center of its host cell

            // distances to the 4 cell centers (all positive), s = 0.5 + temp
            Real s = Real(0.5) + temp;
            Real d0 = Real(1.) + s;
            Real d1 = s;
            Real d2 = Real(1.) - s;
            Real d3 = Real(2.) - s;

            // PCS kernel weights
            Real one_sixth = Real(1.) / Real(6.);
            weight[j][0] = one_sixth * (Real(2.) - d0) * (Real(2.) - d0) * (Real(2.) - d0);
            weight[j][1] = one_sixth * (Real(4.) - Real(6.) * d1 * d1 + Real(3.) * d1 * d1 * d1);
            weight[j][2] = one_sixth * (Real(4.) - Real(6.) * d2 * d2 + Real(3.) * d2 * d2 * d2);
            weight[j][3] = one_sixth * (Real(2.) - d3) * (Real(2.) - d3) * (Real(2.) - d3);
        }

        // scatter the particle's contribution to its neighboring cells
        {
            size_t const noNeighbors = (NO_DIM==2 ? 16 : 64);
            for (size_t n=0; n<noNeighbors; ++n)
            {
                int ni[NO_DIM]; size_t rem = n;
                for (int d=NO_DIM-1; d>=0; --d) { ni[d] = rem % 4; rem /= 4; }
                int index = 0;
                Real result = (*it)->weight();
                for (int d=0; d<NO_DIM; ++d) { index = index * nGrid[d] + cell[d][ni[d]]; result *= weight[d][ni[d]]; }
                q->density[index] += result;
                q->velocity[index] += (*it)->velocity() * result;
            }
        }
    }


    // boundary particles: only valid (in-grid) cells get a contribution
    for (list< vectorIterator >::iterator it=outerParticles.begin(); it!=outerParticles.end(); ++it)
    {
        int cell[NO_DIM][4];    // grid coords of the cells the particle contributes to
        int cellCount[NO_DIM];  // number of valid cells per dimension
        for (int i=0; i<NO_DIM; ++i) cellCount[i] = 0;
        Real weight[NO_DIM][4]; // weight per contributing cell
        for (int j=0; j<NO_DIM; ++j)
        {
            Real temp = ( (*it)->position(j) - box[2*j] ) / dx[j];
            int tempInt = int(floor( temp ));
            temp -= tempInt + 0.5; // distance of particle from the center of its host cell

            // distances to cell centers, for cells tempInt-1, tempInt, tempInt+1, tempInt+2
            Real s = Real(0.5) + temp;
            Real d0 = Real(1.) + s;
            Real d1 = s;
            Real d2 = Real(1.) - s;
            Real d3 = Real(2.) - s;

            Real one_sixth = Real(1.) / Real(6.);
            Real allWeights[4];
            allWeights[0] = one_sixth * (Real(2.) - d0) * (Real(2.) - d0) * (Real(2.) - d0);
            allWeights[1] = one_sixth * (Real(4.) - Real(6.) * d1 * d1 + Real(3.) * d1 * d1 * d1);
            allWeights[2] = one_sixth * (Real(4.) - Real(6.) * d2 * d2 + Real(3.) * d2 * d2 * d2);
            allWeights[3] = one_sixth * (Real(2.) - d3) * (Real(2.) - d3) * (Real(2.) - d3);

            int allCells[4] = { tempInt - 1, tempInt, tempInt + 1, tempInt + 2 };

            // keep only in-grid cells
            int count = 0;
            for (int k=0; k<4; ++k)
            {
                if ( allCells[k] >= 0 and allCells[k] < int(nGrid[j]) )
                {
                    cell[j][count] = allCells[k];
                    weight[j][count] = allWeights[k];
                    count++;
                }
            }
            cellCount[j] = count;
        }

        // scatter the particle's contribution to its valid neighboring cells
        {
            int totalCount = 1;
            for (int d=0; d<NO_DIM; ++d) totalCount *= cellCount[d];
            for (int n=0; n<totalCount; ++n)
            {
                int ni[NO_DIM]; int rem = n;
                for (int d=NO_DIM-1; d>=0; --d) { ni[d] = rem % cellCount[d]; rem /= cellCount[d]; }
                int index = 0;
                Real result = (*it)->weight();
                for (int d=0; d<NO_DIM; ++d) { index = index * nGrid[d] + cell[d][ni[d]]; result *= weight[d][ni[d]]; }
                q->density[index] += result;
                q->velocity[index] += (*it)->velocity() * result;
            }
        }
    }


    // convert accumulated momentum to velocity by dividing by cell mass
    if ( userOptions.aField.velocity )
    {
        for (size_t i=0; i<reserveSize; ++i)
            if ( q->density[i]!=Real(0.) )
                q->velocity[i] /= q->density[i];
            else
                q->velocity[i] = Pvector<Real,noVelComp>::zero();
    }
    else
        q->velocity.clear();

    // normalize the density to the average background density
    if ( userOptions.aField.density )
    {
        Real factor = Real( q->density.size() ) / box.volume() / userOptions.averageDensity;
        for (vector<Real>::iterator it=q->density.begin(); it!=q->density.end(); ++it)
            (*it) *= factor;
    }
    else
        q->density.clear();

    message << "Done.\n" << MESSAGE::Flush;
    printElapsedTime( &t, &userOptions, "PCS interpolation" );
}



