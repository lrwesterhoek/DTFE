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



/* Functions that select the particles inside a sub-box of the main data box. */





#ifndef SUBPARTITION_HEADER
#define SUBPARTITION_HEADER


#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <algorithm>
#include <utility>
#include <cmath>

#include "define.h"
#include "particle_data.h"
#include "user_options.h"
#include "quantities.h"
#include "box.h"
#include "miscellaneous.h"
#include "message.h"

#include <boost/timer.hpp>

typedef std::vector<Particle_data>::iterator   vectorIterator;


void insertParticlesInBox(std::vector<Particle_data> &p,
                        std::vector<Particle_data> *output,
                        Box const &box,
                        Real const offset[]);

#ifdef PHASE_SPACE
// Select particles whose Lagrangian position lies in the padded box (PS-DTFE partitions are Lagrangian),
// generating periodic images on the fly (deferred copies, no whole-box array); shifting lagPos and pos
// by the same eulerLen[] offset matches DTFE()'s global copy generator bit-for-bit.
void findParticlesInBoxLagrangianPeriodic(std::vector<Particle_data> &p,
                                          std::vector<Particle_data> *output,
                                          Box const &paddedBox,
                                          Real const eulerLen[],
                                          bool const periodic,
                                          int verboseLevel)
{
    MESSAGE::Message message( verboseLevel );
    message << "Finding particles (with on-the-fly periodic images) with Lagrangian positions in box " << paddedBox.print() << " ... " << MESSAGE::Flush;
    output->reserve( p.size() / 4 );  // rough estimate

    size_t const noOffsets = periodic ? (NO_DIM==2 ? 9 : 27) : 1;
    for (size_t n=0; n<noOffsets; ++n)
    {
        // periodic-image offset for this pass; applied identically to lagPos and pos so simplices keep correct geometry
        Real off[NO_DIM];
        if ( not periodic )
            for (int d=0; d<NO_DIM; ++d) off[d] = Real(0.);
        else
        {
            size_t rem = n;
            for (int d=NO_DIM-1; d>=0; --d)
            { off[d] = Real(int(rem%3) - 1) * eulerLen[d]; rem /= 3; }
        }

        for (vectorIterator it=p.begin(); it!=p.end(); ++it)
        {
            bool inside = true;
            for (size_t i=0; i<NO_DIM; ++i)
            {
                Real const shiftedLag = it->lagPos[i] + off[i];
                if ( shiftedLag<paddedBox.coords[2*i] or shiftedLag>paddedBox.coords[2*i+1] )
                { inside = false; break; }
            }
            if (inside)
            {
                Particle_data copy = *it;
                for (int d=0; d<NO_DIM; ++d)
                { copy.lagPos[d] += off[d]; copy.pos[d] += off[d]; }
                output->push_back( copy );
            }
        }
    }

    message << "Done.\n"
            << "\t Selected " << output->size() << " particles (originals + periodic images) for this partition.\n" << MESSAGE::Flush;
}
#endif




// Prints the elapsed wall-clock time for a step and adds it to userOptions->totalTime.
void printElapsedTime(boost::timer *t, User_options *userOptions,
                      std::string computationQuantityName)
{
    userOptions->totalTime += t->elapsed();
    MESSAGE::Message message( userOptions->verboseLevel );
    message << "  >>> Time: " << t->elapsed()/userOptions->noProcessors << " sec. (" << computationQuantityName << ")\n" << MESSAGE::Flush;
    t->restart();
}



// Selects all particles inside userOptions.paddedBox into 'output', picking up periodic images when periodic.
void findParticlesInBox(std::vector<Particle_data> &p,
                        std::vector<Particle_data> *output,
                        User_options &userOptions)
{
    Box box = userOptions.paddedBox;
    Box fullBox = userOptions.boxCoordinates;
    Real boxLength[NO_DIM];                   // full box length per axis
    for (size_t i=0; i<NO_DIM; ++i)
        boxLength[i] = fullBox[2*i+1] - fullBox[2*i];
    
    
    MESSAGE::Message message( userOptions.verboseLevel );
    message << "Finding the particles that reside in the padded box " << box.print() << " ... " << MESSAGE::Flush;
    size_t const reserveSize = size_t( p.size() * box.volume()/fullBox.volume() * 1.3 ); // reserve 120% of the average number of particles in the given volume
    output->reserve( reserveSize );
    
    if ( not userOptions.periodic )
    {
        for (vectorIterator it=p.begin(); it!=p.end(); ++it)
            if ( box.isParticleInBox(*it) )
                output->push_back( *it );
    }
    else
    {
        // padded box may extend outside the data box: also pick up particles via periodic translation
        size_t const noOffsets = (NO_DIM==2 ? 9:27);
        Real offset[noOffsets][NO_DIM];
        for (size_t n=0; n<noOffsets; ++n)
        {
            size_t idx = n;
            for (int d=NO_DIM-1; d>=0; --d)
            {
                offset[n][d] = Real(int(idx % 3) - 1) * boxLength[d];
                idx /= 3;
            }
        }
        
        for (size_t n=0; n<noOffsets; ++n)
        {
            Box newBox = box;
            newBox.translate( offset[n] );
            if ( newBox.isBoxOverlaping(fullBox) )           // only translations whose box overlaps the full box contribute particles
                insertParticlesInBox( p, output, newBox, offset[n] );
        }
    }
        
    message << "Done.\n"
            << "\t There are " << output->size() << " particles in the above box. These represent " << std::setprecision(4) << output->size()/float(p.size())*100. << "\% of the total particles in the simulation box.\n" << MESSAGE::Flush;
}



// Inserts into 'output' the particles that fall inside 'box' after translation by 'offset' (storing the shifted-back copy).
void insertParticlesInBox(std::vector<Particle_data> &p,
                        std::vector<Particle_data> *output,
                        Box const &box,
                        Real const offset[])
{
    for (vectorIterator it=p.begin(); it!=p.end(); ++it)
        if ( box.isParticleInBox(*it) )
        {
            Particle_data temp = *it;
            for (int j=0; j<NO_DIM; ++j)
                temp.pos[j] -= offset[j];	// translating the sub-box by +offset is equivalent to translating the particle by -offset
            output->push_back( temp );
        }
}


void NGP_particle_count(std::vector<Particle_data> &particles,
                        size_t const *nGrid,
						Box box,
                        std::vector<int> *counts);

// Splits 'counts' into noPartitions contiguous ranges with as equal a particle load as possible; writes range edges to partitionIndices.
void optimalSplit(std::vector<size_t> &counts,
				  size_t const noPartitions,
				  size_t *partitionIndices)
{
	if ( noPartitions<=size_t(1) or counts.size()==size_t(1) )
		return;
	size_t totalCounts = 0;
	for (size_t i=0; i<counts.size(); ++i)
		totalCounts += counts[i];
	size_t load = totalCounts / noPartitions;	//the number of particles for a perfect balance
	
	// find the optimal balance
	size_t remainingCount = totalCounts, temp = 0, partitionIndexCount = 1;
	partitionIndices[0] = 0;
	partitionIndices[noPartitions] = counts.size();
	for (size_t i=0; i<counts.size(); ++i)
	{
		temp += counts[i];
		if (temp>=load)
		{
			size_t temp1 = ((remainingCount+counts[i]) - temp) / (noPartitions-partitionIndexCount);	//remaining load for split at index i-1
			size_t temp2 = (remainingCount - temp) / (noPartitions-partitionIndexCount);				//remaining load for split at index i
			
			// check which split is more balanced: at index i-1 or index i
			if ( i==size_t(0) )				//first partition must have at least one cell
				(void)i; // keep i unchanged
			else if ( i==counts.size()-1 )	//last partition must have at least one cell
			{
				temp -= counts[i];
				--i;
			}
			else if ( partitionIndexCount>size_t(1) and (i-partitionIndices[partitionIndexCount-1])==size_t(1) )	//each partition must have at least one cell
				(void)i; // keep i unchanged
			else  if ( std::abs(int(temp)-int(temp1))<std::abs(int(temp)-int(temp2)) )	//more balanced load when splitting at index i-1
			{
				temp -= counts[i];
				--i;
			}
			
			partitionIndices[partitionIndexCount] = i;
			remainingCount -= temp;			// keep track of how many particles are left
			load = remainingCount / (noPartitions-partitionIndexCount);	// update the perfect load for the remaining partitions
			temp = 0;
			++partitionIndexCount;
			if ( partitionIndexCount==noPartitions )
				break;
		}
	}
}


// Divides the box along interpolation-grid lines (x then y then z) to balance the particle load across CPUs.
void optimalPartitionSplit(std::vector<Particle_data> &particles,
						   User_options &userOptions,
						   std::vector<size_t> const &partitionGrid,
						   std::vector< std::vector<size_t> > *subgrid,
						   std::vector<Box> *subgridCoords)
{
	MESSAGE::Message message( userOptions.verboseLevel );
    message << "\nComputing the optimal box partition to minimize the processor load ..." << MESSAGE::Flush;
	
	size_t const grid[] = { userOptions.gridSize[0], userOptions.gridSize[1], (NO_DIM==2)?1:userOptions.gridSize[2] };
	size_t const partition[] = { partitionGrid[0], partitionGrid[1], (NO_DIM==2)?1:partitionGrid[2] };
	size_t noPartitions = 1;	// total number of partitions
    for (size_t i=0; i<NO_DIM; ++i)
        noPartitions *= partition[i];
    Real dx[NO_DIM]; 				// to compute the region coordinates for each sub-box
	for (int i=0; i<NO_DIM; ++i)
		dx[i] = (userOptions.region[2*i+1] - userOptions.region[2*i]) / grid[i];
	
	// do some error checking
	if ( noPartitions==size_t(0) )
		throwError( "In function 'optimalPartitionSplit'. The total number of partitions is 0 - one of the partition values along an axis is 0. This does not make sense" );
	for(int i=0; i<NO_DIM; ++i)
		if ( partition[i]>grid[i] )
			throwError( "In function 'optimalPartitionSplit'. The size of the main grid along coordinate '", i, "' is smaller than the requested number of partitions along the same direction." );
	
	// assign memory to the output quantities
	std::vector<size_t> dummy; dummy.assign( 2*NO_DIM, size_t(0) );
	subgrid->assign( noPartitions, dummy );
	subgridCoords->assign( noPartitions, Box() );
	
	
	// find how many particles are in each cell of the grid
	std::vector<int> counts;
	NGP_particle_count( particles, grid, userOptions.region, &counts);
	
	// split along the x-direction
	std::vector<size_t> xSplitIndices; xSplitIndices.assign( partition[0]+1, size_t(0) );
	if ( partition[0]>size_t(1) )
	{
		std::vector<size_t> tempCounts;
		tempCounts.assign( grid[0], size_t(0) );
		
		// project the counts onto the x-axis
		for (int i1=0; i1<grid[0]; ++i1)
			for (int i2=0; i2<grid[1]; ++i2)
				for (int i3=0; i3<grid[2]; ++i3)
				{
					size_t index = i1*grid[1]*grid[2] + i2*grid[2] + i3;
					tempCounts[i1] += counts[index];
				}
		
		
		optimalSplit( tempCounts, partition[0], &(xSplitIndices[0]) );	// split the x-axis evenly
	}
	else
	{
		xSplitIndices[0] = 0;
		xSplitIndices[1] = grid[0];
	}
	// copy the results of the split to the book-keeping array
	for (size_t i1=0; i1<partition[0]; ++i1)
		for (size_t i2=0; i2<partition[1]; ++i2)
			for (size_t i3=0; i3<partition[2]; ++i3)
			{
				size_t index = i1*partition[1]*partition[2] + i2*partition[2] + i3;
				subgrid->at(index)[0] = xSplitIndices[i1];
				subgrid->at(index)[1] = xSplitIndices[i1+1];
				subgridCoords->at(index)[0] = userOptions.region[0] + dx[0]*subgrid->at(index)[0];
				subgridCoords->at(index)[1] = userOptions.region[0] + dx[0]*subgrid->at(index)[1];
			}
	
	
	// split along the y-direction
	std::vector<size_t> ySplitIndices; ySplitIndices.assign( partition[1]+1, size_t(0) );
	for (size_t i=0; i<partition[0]; ++i)
	{
		if ( partition[1]>size_t(1) )
		{
			std::vector<size_t> tempCounts;
			tempCounts.assign( grid[1], size_t(0) );
			
			// project the counts onto the y-axis
			for (int i1=xSplitIndices[i]; i1<xSplitIndices[i+1]; ++i1)
				for (int i2=0; i2<grid[1]; ++i2)
					for (int i3=0; i3<grid[2]; ++i3)
					{
						size_t index = i1*grid[1]*grid[2] + i2*grid[2] + i3;
						tempCounts[i2] += counts[index];
					}
			
			
			optimalSplit( tempCounts, partition[1], &(ySplitIndices[0]) );	// split the y-axis evenly
		}
		else
		{
			ySplitIndices[0] = 0;
			ySplitIndices[1] = grid[1];
		}
		for (size_t i2=0; i2<partition[1]; ++i2)
			for (size_t i3=0; i3<partition[2]; ++i3)
			{
				size_t index = i*partition[1]*partition[2] + i2*partition[2] + i3;
				subgrid->at(index)[2] = ySplitIndices[i2];
				subgrid->at(index)[3] = ySplitIndices[i2+1];
				subgridCoords->at(index)[2] = userOptions.region[2] + dx[1]*subgrid->at(index)[2];
				subgridCoords->at(index)[3] = userOptions.region[2] + dx[1]*subgrid->at(index)[3];
			}
	}
	
	
	// split along the z-direction
#if NO_DIM==3
	std::vector<size_t> zSplitIndices; zSplitIndices.assign( partition[2]+1, size_t(0) );
	for (size_t i=0; i<partition[0]; ++i)
		for (size_t j=0; j<partition[1]; ++j)
		{
			if ( partition[2]>size_t(1) )
			{
				std::vector<size_t> tempCounts;
				tempCounts.assign( grid[2], size_t(0) );
				
				// project the counts onto the z-axis
				for (int i1=xSplitIndices[i]; i1<xSplitIndices[i+1]; ++i1)
					for (int i2=ySplitIndices[j]; i2<ySplitIndices[j+1]; ++i2)
						for (int i3=0; i3<grid[2]; ++i3)
						{
							size_t index = i1*grid[1]*grid[2] + i2*grid[2] + i3;
							tempCounts[i3] += counts[index];
						}
				
				
				optimalSplit( tempCounts, partition[2], &(zSplitIndices[0]) );	// split the z-axis evenly
			}
			else
			{
				zSplitIndices[0] = 0;
				zSplitIndices[1] = grid[2];
			}
			for (size_t i3=0; i3<partition[2]; ++i3)
			{
				size_t index = i*partition[1]*partition[2] + j*partition[2] + i3;
				subgrid->at(index)[4] = zSplitIndices[i3];
				subgrid->at(index)[5] = zSplitIndices[i3+1];
				subgridCoords->at(index)[4] = userOptions.region[4] + dx[2]*subgrid->at(index)[4];
				subgridCoords->at(index)[5] = userOptions.region[4] + dx[2]*subgrid->at(index)[5];
			}
		}
#endif


	message << "Done.\n" << MESSAGE::Flush;
}


// Copies the selected partition's subgrid size and region into userOptions for this run.
void copySubgridInformation(User_options *userOptions,
							std::vector< std::vector<size_t> > &subgridList,
							std::vector< Box > &subgridCoords)
{
	for (int i=0; i<NO_DIM; ++i)
		userOptions->gridSize[i] = subgridList[userOptions->partNo][2*i+1] - subgridList[userOptions->partNo][2*i];
	userOptions->region = subgridCoords[userOptions->partNo];
}

// Prints the grid-index range covered by the selected partition.
void subgrid(User_options &userOptions, std::vector< std::vector<size_t> > &subgrid)
{
    MESSAGE::Message message( userOptions.verboseLevel );
    char const axisName[] = "xyz";
    message << "\nYou choose to compute the results only for a partition of the data. The output files will contain only the results for the grid indices with:\n";
    for (int d=0; d<NO_DIM; ++d)
        message << "\t n" << axisName[d] << " = [" << subgrid[userOptions.partNo][2*d] << " to " << subgrid[userOptions.partNo][2*d+1]-1 << "]\n";
}


// Merges a subgrid's field quantities ('--partition' option) into the full-box quantities, with size sanity checks.
void copySubgridResultsToMain(Quantities const &subgridResults,
                              std::vector<size_t> const &mainGrid,
                              Field &field,
                              User_options &userOptions,
							  std::vector< std::vector<size_t> > &subgridList,
                              Quantities *mainGridResults)
{
    if ( not field.selected() ) return;
    
    // get the subgrid and subgrid offset
    std::vector<size_t> subgrid(NO_DIM,0), subgridOffset(NO_DIM,0);
    for (int i=0; i<NO_DIM; ++i)
	{
		subgrid[i] = subgridList[userOptions.partNo][2*i+1] - subgridList[userOptions.partNo][2*i];
		subgridOffset[i] = subgridList[userOptions.partNo][2*i];
	}
    
    
    // do some error checking
    size_t totalGrid = 1, secondaryGrid = 1;
    for (size_t i=0; i<mainGrid.size(); ++i)
    {
            totalGrid *= mainGrid[i];
            secondaryGrid *= userOptions.gridSize[i];
    }
    
    if ( mainGridResults->size()!=totalGrid )
        throwError( "The size of the vectors in the main grid results does not agree with the size of the main grid. Error in function 'copySubgridResultsToMain'." );
    if ( subgridResults.size()!=secondaryGrid )
        throwError( "The size of the vectors in the subgrid results does not agree with the size of the subgrid grid. Error in function 'copySubgridResultsToMain'." );
    
    // copy the subgrid to the main grid
    mainGridResults->copyFromSubgrid( subgridResults, field, mainGrid, subgrid, subgridOffset );
}



// Returns the mean mass density in the full simulation box, used as the reference for per-vertex density.
Real averageDensity(std::vector<Particle_data> &p,
                    User_options const &userOptions)
{
    double totalMass = 0.;
    for (vectorIterator it=p.begin(); it!=p.end(); ++it)
        totalMass += it->weight();

    Real averageDen = totalMass / userOptions.boxCoordinates.volume();
    MESSAGE::Message message( userOptions.verboseLevel );
    message << "Average density in the box: " << averageDen << "\n" << MESSAGE::Flush;
    return averageDen;
}


// Chooses the parallel partition grid for the available processors, using the largest 2^a*3^b <= noAvailableProcessors.
void parallelGrid(size_t const noAvailableProcessors,
                  User_options const &userOptions,
                  size_t *grid)
{
    // largest x2, x3 with 2^x2 <= n and 3^x3 <= n
    size_t n = noAvailableProcessors;
    int x2 = 0, x3 = 0;
    
    while ( n/2!=0 )
    {
        n /= 2;
        ++x2;
    }
    n = noAvailableProcessors;
    while ( n/3!=0 )
    {
        n /= 3;
        ++x3;
    }
    n = noAvailableProcessors;
    
    // best 2^j2 * 3^j3 <= n closest to n
    size_t diff = n;
    int j2 = 0, j3 = 0;
    size_t nApprox = 1;
    for (int i2=0; i2<=x2; ++i2)
    {
        size_t temp = nApprox;
        for (int i3=0; i3<=x3; ++i3)
        {
            if ( (n-temp)>=0 and (n-temp)<diff )  //if temp<n and temp is closer to n than previous estimate, update these values as best choice
            {
                diff = n-temp;
                j2 = i2;
                j3 = i3;
            }
            temp *= 3;
        }
        nApprox *= 2;
    }
    
    x2 = j2;
    x3 = j3;


    // distribute the 2's and 3's across axes as uniformly as possible to minimize overlapping regions
    for (int i=0; i<NO_DIM; ++i)
        grid[i] = 1;
    // first distribute the 3's
    while ( x3-NO_DIM>=0 )
    {
        for (int i=0; i<NO_DIM; ++i)
            grid[i] *= 3;
        x3 -= NO_DIM;
    }
    for (int i=NO_DIM-1; i>=(NO_DIM-x3); --i)    // put the remaining 3's starting with axis NO_DIM-1, NO_DIM-2, ... so on
        grid[i] *= 3;
    // now distribute the 2's such that the grid is maximally balanced
    int const i1 = x2>(NO_DIM-x3) ? (NO_DIM-x3) : x2; // i1 is the minimum(x2, NO_DIM-x3)
    for (int i=0; i<i1; ++i)    // the grid axes which have a deficency of 3's will receive a factor of 2 
        grid[i%(NO_DIM-x3)] *= 2;
    for (int i=0; i<x2-i1; ++i) // distribute the remaining 2's starting with the first grid axis
        grid[i%NO_DIM] *= 2;

    // Check that the parallel split is consistent with the main grid axes
    size_t const *mainGrid = &(userOptions.gridSize[0]);
	for (int i=0; i<NO_DIM; ++i)
		if ( mainGrid[i]>=grid[i] )	//don't do anything if you can split the mainGrid to have at least one cell per parallel partition
			continue;
		else
		{
			size_t next1 = (i+1)%NO_DIM, next2 = (i+2)%NO_DIM;
			while ( grid[i]>mainGrid[i] and grid[i]%3==0 )
			{
				grid[i] /= 3;
				if ( 3*grid[next1]<=mainGrid[next1] )
					grid[next1] *= 3;
				else if ( 3*grid[next2]<=mainGrid[next2] )
					grid[next2] *= 3;
			}
			while ( grid[i]>mainGrid[i] and grid[i]%2==0 )
			{
				grid[i] /= 2;
				if ( 2*grid[next1]<=mainGrid[next1] )
					grid[next1] *= 2;
				else if ( 2*grid[next2]<=mainGrid[next2] )
					grid[next2] *= 2;
			}
		}
	
	
    // show to the user a message with the parallel grid
    size_t const noProcesses = (NO_DIM==2) ? grid[0]*grid[1] : grid[0]*grid[1]*grid[2];
    MESSAGE::Message message( userOptions.verboseLevel );
    if ( noProcesses==noAvailableProcessors )
        message << "\nThe program detected " << noAvailableProcessors << " processors available for the computation. The processors number was split in the grid [" << MESSAGE::printElements( grid, NO_DIM, "," ) << "] to allow for optimal division of the data box.\n" << MESSAGE::Flush;
    else
        message << "\nThe program detected " << noAvailableProcessors << " processors available for the computation. Since this number cannot be written as a product of 2's and 3's, only " << noProcesses << " processors will be used for the parallel computation. The new processors number was split in the grid [" << MESSAGE::printElements( grid, NO_DIM, "," ) << "] to allow for optimal division of the data box.\n" << MESSAGE::Flush;
}





// Approximate per-thread CPU time (threads cannot be timed independently).
void approximativeThreadTime(Real *processorTime,
                             int const noProcessors)
{
    std::vector< std::pair<Real,int> > temp;
    temp.reserve( noProcessors );
    for (int i=0; i<noProcessors; ++i)
        temp.push_back( std::make_pair( processorTime[i], i ) );
    
    // sort the times ascendingly
    std::sort( temp.begin(), temp.end() );
    Real lastTime = Real(0.), lastTotalTime = Real(0.);
    int factor = noProcessors;
    for (int i=0; i<noProcessors; ++i)
    {
        Real temp2 = temp[i].first;
        temp[i].first = lastTime + (temp2-lastTotalTime) / factor--;
        lastTime = temp[i].first;
        lastTotalTime = temp2;
    }
    
    // write the computed values to the output array
    for (int i=0; i<noProcessors; ++i)
        processorTime[ temp[i].second ] = temp[i].first;
}


#endif
