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


#include "quantities.h"
#include "message.h"




/* This function copies one field of 'Quantities' at a time from the subgrid results to the main grid ones. */
template<typename T>
void copyField(T const &subgridResults,
                T *mainGridResults,
                bool toCopy,
                std::vector<size_t> const &mainGrid,
                std::vector<size_t> const &subgrid,
                std::vector<size_t> const &subgridOffset)
{
    if ( not toCopy )
        return;
    
    size_t start[NO_DIM], end[NO_DIM];
    for (size_t i=0; i<NO_DIM; ++i)
    {
        start[i] = subgridOffset[i];
        end[i] = subgridOffset[i] + subgrid[i];
    }
    
    
    size_t totalSubgrid = 1;
    for (size_t d=0; d<NO_DIM; ++d) totalSubgrid *= subgrid[d];

    for (size_t flatIdx=0; flatIdx<totalSubgrid; ++flatIdx)
    {
        size_t subIdx[NO_DIM], rem = flatIdx;
        for (int d=NO_DIM-1; d>=0; --d)
        {
            subIdx[d] = rem % subgrid[d];
            rem /= subgrid[d];
        }
        size_t indexMain = 0, indexSub = 0;
        for (int d=0; d<NO_DIM; ++d)
        {
            if (d > 0) { indexMain *= mainGrid[d]; indexSub *= subgrid[d]; }
            indexMain += subIdx[d] + start[d];
            indexSub += subIdx[d];
        }
        (*mainGridResults)[indexMain] = subgridResults[indexSub];
    }
}

/* Function used to copy the results obtain on a subgrid using the option 'partion' to the results for the full grid.
NOTE: For additional details on how to compute the 'subgrid' and 'subgridOffset' check the function 'subgridAxesSize' in the file 'subpartition.h'. */
void Quantities::copyFromSubgrid(Quantities const &subgridResults,
                                 Field const &field,
                                 std::vector<size_t> const &mainGrid,
                                 std::vector<size_t> const &subgrid,
                                 std::vector<size_t> const &subgridOffset)
{
    copyField( subgridResults.density, &(this->density), field.density,  mainGrid, subgrid, subgridOffset );
    copyField( subgridResults.velocity, &(this->velocity), field.velocity,  mainGrid, subgrid, subgridOffset );
    copyField( subgridResults.velocity_gradient, &(this->velocity_gradient), field.velocity_gradient,  mainGrid, subgrid, subgridOffset );
    copyField( subgridResults.velocity_divergence, &(this->velocity_divergence), field.velocity_divergence,  mainGrid, subgrid, subgridOffset );
    copyField( subgridResults.velocity_shear, &(this->velocity_shear), field.velocity_shear,  mainGrid, subgrid, subgridOffset );
    copyField( subgridResults.velocity_vorticity, &(this->velocity_vorticity), field.velocity_vorticity,  mainGrid, subgrid, subgridOffset );
    copyField( subgridResults.velocity_std, &(this->velocity_std), field.velocity_std,  mainGrid, subgrid, subgridOffset );
    copyField( subgridResults.scalar, &(this->scalar), field.scalar,  mainGrid, subgrid, subgridOffset );
    copyField( subgridResults.scalar_gradient, &(this->scalar_gradient), field.scalar_gradient,  mainGrid, subgrid, subgridOffset );
    copyField( subgridResults.velocity_tweb, &(this->velocity_tweb), field.velocity_tweb,  mainGrid, subgrid, subgridOffset );
    copyField( subgridResults.velocity_tweb_eigenvalues, &(this->velocity_tweb_eigenvalues), field.velocity_tweb,  mainGrid, subgrid, subgridOffset );
    copyField( subgridResults.velocity_vweb, &(this->velocity_vweb), field.velocity_vweb,  mainGrid, subgrid, subgridOffset );
    copyField( subgridResults.velocity_vweb_eigenvalues, &(this->velocity_vweb_eigenvalues), field.velocity_vweb,  mainGrid, subgrid, subgridOffset );
}



/* Compares the size of a vector with the expected size and outputs error if this is not the case. */
template< typename T>
void fieldSize(T const & object,
                size_t *expectedSize)
{
    if( not object.empty() )
    {
        if ( (*expectedSize)!=0 and (*expectedSize)!=object.size() )
            throwError( "Two or more objects of class 'Quantities' have different sizes. All objects in this class should be empty or have the same size." );
        else
            (*expectedSize) = object.size();
    }
}

/* Returns what is the size of any non-empty object.
NOTE: All non-empty objects should have the same size. */
size_t Quantities::size() const
{
    size_t temp = 0;
    fieldSize( this->density, &temp );
    fieldSize( this->velocity, &temp );
    fieldSize( this->velocity_gradient, &temp );
    fieldSize( this->velocity_divergence, &temp );
    fieldSize( this->velocity_shear, &temp );
    fieldSize( this->velocity_vorticity, &temp );
    fieldSize( this->velocity_std, &temp );
    fieldSize( this->scalar, &temp );
    fieldSize( this->scalar_gradient, &temp );
    fieldSize( this->velocity_tweb, &temp );
    fieldSize( this->velocity_tweb_eigenvalues, &temp );
    fieldSize( this->velocity_vweb, &temp );
    fieldSize( this->velocity_vweb_eigenvalues, &temp );
    return temp;
}


/* Element-wise accumulate: add values from 'other' into this Quantities.
   Used by PS-DTFE partitioning where each partition writes to a full grid
   and results are summed across partitions. */
template<typename T>
void addField(std::vector<T> const &src, std::vector<T> *dst)
{
    if (src.empty()) return;
    for (size_t i = 0; i < src.size(); ++i)
        (*dst)[i] += src[i];
}

void Quantities::addFrom(Quantities const &other)
{
    addField(other.density, &this->density);
    addField(other.velocity, &this->velocity);
    addField(other.velocity_gradient, &this->velocity_gradient);
    addField(other.velocity_divergence, &this->velocity_divergence);
    addField(other.velocity_shear, &this->velocity_shear);
    addField(other.velocity_vorticity, &this->velocity_vorticity);
    addField(other.velocity_std, &this->velocity_std);
    addField(other.scalar, &this->scalar);
    addField(other.scalar_gradient, &this->scalar_gradient);
    // Note: T-web/V-web classification is computed post-interpolation, so these are unused here
    // but included for completeness
    addField(other.velocity_tweb_eigenvalues, &this->velocity_tweb_eigenvalues);
    addField(other.velocity_vweb_eigenvalues, &this->velocity_vweb_eigenvalues);
#ifdef PHASE_SPACE
    addField(other.stream_count, &this->stream_count);
#endif
}


/* This function reserves memory for the main grid quantities when using the 'partition' option. */
void Quantities::reserveMemory(size_t *gridSize, Field &field)
{
    size_t totalSize = 1;
    for (size_t i=0; i<NO_DIM; ++i)
        totalSize *= gridSize[i];
    
    if ( field.density )
        this->density.resize( totalSize, Real(0.) );
    if ( field.velocity )
        this->velocity.resize( totalSize, Pvector<Real,noVelComp>::zero() );
    if ( field.velocity_gradient )
        this->velocity_gradient.resize( totalSize, Pvector<Real,noGradComp>::zero() );
    if ( field.velocity_divergence )
        this->velocity_divergence.resize( totalSize, Real(0.) );
    if ( field.velocity_shear )
        this->velocity_shear.resize( totalSize, Pvector<Real,noShearComp>::zero() );
    if ( field.velocity_vorticity )
        this->velocity_vorticity.resize( totalSize, Pvector<Real,noVortComp>::zero() );
    if ( field.velocity_std )
        this->velocity_std.resize( totalSize, Real(0.) );
    if ( field.scalar )
        this->scalar.resize( totalSize, Pvector<Real,noScalarComp>::zero() );
    if ( field.scalar_gradient )
        this->scalar_gradient.resize( totalSize, Pvector<Real,noScalarGradComp>::zero() );
    if ( field.velocity_tweb )
    {
        this->velocity_tweb.resize( totalSize, Real(0.) );
        this->velocity_tweb_eigenvalues.resize( totalSize, Pvector<Real,NO_DIM>::zero() );
    }
    if ( field.velocity_vweb )
    {
        this->velocity_vweb.resize( totalSize, Real(0.) );
        this->velocity_vweb_eigenvalues.resize( totalSize, Pvector<Real,NO_DIM>::zero() );
    }
}


