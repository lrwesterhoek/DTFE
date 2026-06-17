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


/* Cloud-In-Cell (CIC) mass assignment: linear (order-1) interpolation kernel spanning
   2 cells per dimension. Inner particles scatter freely; boundary particles deposit only into in-grid cells. */


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


void CIC_interpolation_regular_grid(vector<Particle_data> &particles,
                                    User_options &userOptions,
                                    Quantities *q);




// CIC (Cloud In Cell) interpolation of density and velocity to a grid.
void CIC_interpolation(vector<Particle_data> *particles,
                       vector<Sample_point> &samples,
                       User_options &userOptions,
                       Quantities *q)
{
    if ( samples.empty() and not userOptions.redshiftConeOn )
        CIC_interpolation_regular_grid( *particles, userOptions, q );
    else
        throwError( "The CIC method can interpolate the fields only on a regular rectangular grid. No CIC interpolation methods are implemented for redshift cone coordinates or for user defined sample points." );
    particles->clear();
}





// CIC worker on a regular grid: scatters each particle's mass and momentum via the linear kernel,
// then divides momentum by cell mass for velocity and normalizes density to the background mean.
void CIC_interpolation_regular_grid(vector<Particle_data> &particles,
                                    User_options &userOptions,
                                    Quantities *q)
{
    size_t const *nGrid = &(userOptions.gridSize[0]);
    MESSAGE::Message message( userOptions.verboseLevel );
    message << "\nInterpolating the fields to the grid using the CIC method. The interpolation takes place inside the box of coordinates " << userOptions.region.print()
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
        outerPadding[2*i] = dx[i];
        outerPadding[2*i+1] = dx[i];
    }
    for (int i=0; i<2*NO_DIM; ++i)
        innerPadding[i] = -1.1*outerPadding[i];    // slightly larger than 1 grid cell to account for numerical uncertanties
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
        int cell[NO_DIM][3];    // grid coords of the cells the particle contributes to
        Real weight[NO_DIM][3]; // weight per contributing cell
        for (int j=0; j<NO_DIM; ++j)
        {
            Real temp = ( (*it)->position(j) - box[2*j] ) / dx[j];
            cell[j][0] = int(floor( temp )) - 1;
            cell[j][1] = cell[j][0] + 1;
            cell[j][2] = cell[j][1] + 1;

            temp -= cell[j][1] + 0.5; // distance of particle from the center of its host cell

            if ( temp < 0.)
            {
				weight[j][0] = -temp;
				weight[j][1] = 1 + temp;
				weight[j][2] = 0.;
			}
			else
            {
				weight[j][0] = 0.;
				weight[j][1] = 1 - temp;
				weight[j][2] = temp;
			}
            
        }
        
        // scatter the particle's contribution to its neighboring cells
        {
            size_t const noNeighbors = (NO_DIM==2 ? 9 : 27);
            for (size_t n=0; n<noNeighbors; ++n)
            {
                int ni[NO_DIM]; size_t rem = n;
                for (int d=NO_DIM-1; d>=0; --d) { ni[d] = rem % 3; rem /= 3; }
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
        int cell[NO_DIM][3];    // grid coords of the cells the particle contributes to
        int cellCount[NO_DIM];  // number of valid cells per dimension
        for (int i=0; i<NO_DIM; ++i) cellCount[i] = 0;
        Real weight[NO_DIM][3]; // weight per contributing cell
        for (int j=0; j<NO_DIM; ++j)
        {
            Real temp = ( (*it)->position(j) - box[2*j] ) / dx[j];
            int tempInt = int(floor( temp ));
            temp -= tempInt + 0.5; // distance of particle from the center of its host cell
            if ( temp<-0.5 or temp>0.5 ) cout << temp << "\t" << j << "\t" << tempInt << "\n";

            if ( tempInt==-1 )
            {
                cell[j][0] = 0; cellCount[j] = 1;
                weight[j][0] = temp;
            }
            else if ( tempInt==0 )
            {
                cell[j][0] = 0; cell[j][1] = 1; cellCount[j] = 2;
                weight[j][0] = 1 - temp;
                weight[j][1] = temp;
            }
            else if ( tempInt==int(nGrid[j])-1 )
            {
                cell[j][0] = nGrid[j]-2; cell[j][1] = nGrid[j]-1; cellCount[j] = 2;
                weight[j][0] = - temp;
                weight[j][1] = 1 + temp;
            }
            else if ( tempInt==int(nGrid[j]) )
            {
                cell[j][0] = nGrid[j]-1; cellCount[j] = 1;
                weight[j][0] = - temp;
            }
            else if ( not( tempInt<-1 or tempInt>int(nGrid[j]) ) )
            {
                cell[j][0] = tempInt-1;
                cell[j][1] = tempInt;
                cell[j][2] = tempInt+1;
                cellCount[j] = 3;
                
                if ( temp < 0.)
				{
					weight[j][0] = -temp;
					weight[j][1] = 1 + temp;
					weight[j][2] = 0.;
				}
				else
				{
					weight[j][0] = 0.;
					weight[j][1] = 1 - temp;
					weight[j][2] = temp;
				}
            }
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
    printElapsedTime( &t, &userOptions, "CIC interpolation" );
}




