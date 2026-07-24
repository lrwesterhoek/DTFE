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

#ifndef GRID_DEPOSIT_ENGINE_H
#define GRID_DEPOSIT_ENGINE_H

/* Shared regular-grid mass-assignment engine for the CIC / TSC / PCS methods (NGP keeps its
   own simpler path). Each method supplies a kernel policy with: STENCIL (cells per dimension),
   PAD_CELLS (kernel reach in cells, sets the outer selection box), name(), interiorWeights()
   and boundaryCells(). The boundary cell/weight selection stays method-specific and verbatim:
   the CIC/TSC edge cases are NOT a generic clip of the interior stencil.

   The engine preserves the original loop nesting and accumulation order exactly, so its
   outputs are bit-identical to the previous per-method copies (validated by an A/B
   byte-compare of the .a_den/.a_vel outputs and the regression suite). */

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


// Scatters each particle's mass and momentum via the kernel, then divides momentum by cell
// mass for velocity and normalizes density to the background mean.
template <typename Kernel>
void gridDeposit_regular_grid(std::vector<Particle_data> &particles,
                              User_options &userOptions,
                              Quantities *q)
{
    typedef std::vector<Particle_data>::iterator vecIt;
    int const S = Kernel::STENCIL;

    size_t const *nGrid = &(userOptions.gridSize[0]);
    MESSAGE::Message message( userOptions.verboseLevel );
    message << "\nInterpolating the fields to the grid using the " << Kernel::name() << " method. The interpolation takes place inside the box of coordinates " << userOptions.region.print()
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
        outerPadding[2*i] = Kernel::PAD_CELLS * dx[i];
        outerPadding[2*i+1] = Kernel::PAD_CELLS * dx[i];
    }
    for (int i=0; i<2*NO_DIM; ++i)
        innerPadding[i] = -1.1*outerPadding[i];    // slightly larger than the kernel reach to account for numerical uncertainties
    outerBox.addPadding( outerPadding );  // only particles in this box give contributions in 'userOptions.region'
    innerBox.addPadding( innerPadding );  // the contribution of particles in this box is limited to the region of interest


    // inner particles contribute only inside the region; outer ones also outside, so cell validity is checked
    std::list< vecIt > innerParticles, outerParticles;
    for (vecIt it=particles.begin(); it!=particles.end(); ++it)
        if ( innerBox.isParticleInBox(*it) )
            innerParticles.push_back( it );
        else if ( outerBox.isParticleInBox(*it) )
            outerParticles.push_back( it );



    for (typename std::list< vecIt >::iterator it=innerParticles.begin(); it!=innerParticles.end(); ++it)
    {
        int cell[NO_DIM][Kernel::STENCIL];    // grid coords of the cells the particle contributes to
        Real weight[NO_DIM][Kernel::STENCIL]; // weight per contributing cell
        for (int j=0; j<NO_DIM; ++j)
        {
            Real temp = ( (*it)->position(j) - box[2*j] ) / dx[j];
            cell[j][0] = int(std::floor( temp )) - 1;
            for (int k=1; k<S; ++k)
                cell[j][k] = cell[j][k-1] + 1;

            temp -= cell[j][1] + 0.5; // distance of particle from the center of its host cell

            Kernel::interiorWeights( temp, weight[j] );
        }

        // scatter the particle's contribution to its neighboring cells
        {
            size_t noNeighbors = 1;
            for (int d=0; d<NO_DIM; ++d) noNeighbors *= S;
            for (size_t n=0; n<noNeighbors; ++n)
            {
                int ni[NO_DIM]; size_t rem = n;
                for (int d=NO_DIM-1; d>=0; --d) { ni[d] = rem % S; rem /= S; }
                // size_t: an int32 flat index overflows above ~1290^3 cells (2^31)
                size_t index = 0;
                Real result = (*it)->weight();
                for (int d=0; d<NO_DIM; ++d) { index = index * size_t(nGrid[d]) + size_t(cell[d][ni[d]]); result *= weight[d][ni[d]]; }
                q->density[index] += result;
                q->velocity[index] += (*it)->velocity() * result;
            }
        }
    }


    // boundary particles: only valid (in-grid) cells get a contribution
    for (typename std::list< vecIt >::iterator it=outerParticles.begin(); it!=outerParticles.end(); ++it)
    {
        int cell[NO_DIM][Kernel::STENCIL];    // grid coords of the cells the particle contributes to
        int cellCount[NO_DIM];  // number of valid cells per dimension
        for (int i=0; i<NO_DIM; ++i) cellCount[i] = 0;
        Real weight[NO_DIM][Kernel::STENCIL]; // weight per contributing cell
        for (int j=0; j<NO_DIM; ++j)
        {
            Real temp = ( (*it)->position(j) - box[2*j] ) / dx[j];
            int tempInt = int(std::floor( temp ));
            temp -= tempInt + 0.5; // distance of particle from the center of its host cell

            Kernel::boundaryCells( tempInt, temp, nGrid[j], j, cell[j], weight[j], cellCount[j] );
        }

        // scatter the particle's contribution to its valid neighboring cells
        {
            int totalCount = 1;
            for (int d=0; d<NO_DIM; ++d) totalCount *= cellCount[d];
            for (int n=0; n<totalCount; ++n)
            {
                int ni[NO_DIM]; int rem = n;
                for (int d=NO_DIM-1; d>=0; --d) { ni[d] = rem % cellCount[d]; rem /= cellCount[d]; }
                // size_t: an int32 flat index overflows above ~1290^3 cells (2^31)
                size_t index = 0;
                Real result = (*it)->weight();
                for (int d=0; d<NO_DIM; ++d) { index = index * size_t(nGrid[d]) + size_t(cell[d][ni[d]]); result *= weight[d][ni[d]]; }
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
        for (std::vector<Real>::iterator it=q->density.begin(); it!=q->density.end(); ++it)
            (*it) *= factor;
    }
    else
        q->density.clear();

    message << "Done.\n" << MESSAGE::Flush;
    printElapsedTime( &t, &userOptions, std::string(Kernel::name()) + " interpolation" );
}

#endif
