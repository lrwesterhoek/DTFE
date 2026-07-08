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


/* Quantities member functions: copy/accumulate field results across partitions and (PS-DTFE) turn
   summed density-weighted moments into mass-weighted means. */

#include "quantities.h"
#include "message.h"




// Copies one Quantities field from the subgrid results into the matching block of the main grid.
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

// Copies '--partition' subgrid results into the full-grid results ('subgrid', 'subgridOffset' from subpartition.h).
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
    copyField( subgridResults.velocity_dispersion, &(this->velocity_dispersion), field.velocity_dispersion,  mainGrid, subgrid, subgridOffset );
    copyField( subgridResults.scalar, &(this->scalar), field.scalar,  mainGrid, subgrid, subgridOffset );
    copyField( subgridResults.scalar_gradient, &(this->scalar_gradient), field.scalar_gradient,  mainGrid, subgrid, subgridOffset );
    copyField( subgridResults.velocity_tweb, &(this->velocity_tweb), field.velocity_tweb,  mainGrid, subgrid, subgridOffset );
    copyField( subgridResults.velocity_tweb_eigenvalues, &(this->velocity_tweb_eigenvalues), field.velocity_tweb,  mainGrid, subgrid, subgridOffset );
    copyField( subgridResults.velocity_vweb, &(this->velocity_vweb), field.velocity_vweb,  mainGrid, subgrid, subgridOffset );
    copyField( subgridResults.velocity_vweb_eigenvalues, &(this->velocity_vweb_eigenvalues), field.velocity_vweb,  mainGrid, subgrid, subgridOffset );
}



// Errors out if 'object' is non-empty and its size differs from the running expected size; updates *expectedSize.
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

// Returns the size of any non-empty member (all non-empty members share the same size).
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
    fieldSize( this->velocity_dispersion, &temp );
    fieldSize( this->scalar, &temp );
    fieldSize( this->scalar_gradient, &temp );
    fieldSize( this->velocity_tweb, &temp );
    fieldSize( this->velocity_tweb_eigenvalues, &temp );
    fieldSize( this->velocity_vweb, &temp );
    fieldSize( this->velocity_vweb_eigenvalues, &temp );
#ifdef PHASE_SPACE
    fieldSize( this->stream_count, &temp );
    fieldSize( this->mass_weight, &temp );
#endif
    return temp;
}


// Element-wise adds 'src' into 'dst'; used to sum PS-DTFE partition results across partitions.
template<typename T>
void addField(std::vector<T> const &src, std::vector<T> *dst)
{
    if (src.empty()) return;
    // Grow an unsized dst (zero-filled) so accumulation starts from zero.
    if (dst->size() < src.size())
        dst->resize(src.size(), T());
    for (size_t i = 0; i < src.size(); ++i)
        (*dst)[i] += src[i];
}

// Accumulates every field of 'other' into this object (PS-DTFE: one partition's results summed into the whole).
void Quantities::addFrom(Quantities const &other)
{
    addField(other.density, &this->density);
    addField(other.velocity, &this->velocity);
    addField(other.velocity_gradient, &this->velocity_gradient);
    addField(other.velocity_divergence, &this->velocity_divergence);
    addField(other.velocity_shear, &this->velocity_shear);
    addField(other.velocity_vorticity, &this->velocity_vorticity);
    addField(other.velocity_std, &this->velocity_std);
    addField(other.velocity_dispersion, &this->velocity_dispersion);
    addField(other.scalar, &this->scalar);
    addField(other.scalar_gradient, &this->scalar_gradient);
    // T-web/V-web labels are computed post-interpolation; only eigenvalues accumulate
    addField(other.velocity_tweb_eigenvalues, &this->velocity_tweb_eigenvalues);
    addField(other.velocity_vweb_eigenvalues, &this->velocity_vweb_eigenvalues);
#ifdef PHASE_SPACE
    addField(other.stream_count, &this->stream_count);
    addField(other.mass_weight, &this->mass_weight);
#endif
}


#ifdef PHASE_SPACE
// PS-DTFE partition path: partitions accumulate density-weighted moments sum(rho_s f_s) and per-cell
// mass sum(rho_s); divide once here after all partitions are summed, since averaging is non-linear.
void Quantities::normalizePhaseSpace(Field const &field)
{
    if ( this->mass_weight.empty() ) return;   // no mass-weighted field was deferred
    size_t const n = this->mass_weight.size();
    bool const haveVel = field.velocity || field.velocity_dispersion; // velocity holds sum(rho v)
    for (size_t i = 0; i < n; ++i)
    {
        if ( this->mass_weight[i] <= Real(0.) ) continue;
        Real const inv = Real(1.) / this->mass_weight[i];
        if ( haveVel )                 this->velocity[i]          *= inv;   // <v> = sum(rho v)/sum(rho)
        if ( field.velocity_gradient ) this->velocity_gradient[i] *= inv;
        if ( field.scalar )            this->scalar[i]            *= inv;
        if ( field.scalar_gradient )   this->scalar_gradient[i]   *= inv;
        if ( field.velocity_dispersion )
        {
            // sigma_ij = <v_i v_j> - <v_i><v_j>, using the now-normalized <v>.
            Pvector<Real,noVelComp> const &vbar = this->velocity[i];
            size_t c = 0;
            for (int a = 0; a < NO_DIM; ++a)
                for (int b = a; b < NO_DIM; ++b)
                {
                    Real s = this->velocity_dispersion[i][c] * inv - vbar[a]*vbar[b];
                    if ( a == b and s < Real(0.) ) s = Real(0.);   // variance: clamp FP-noise negatives
                    this->velocity_dispersion[i][c] = s;
                    ++c;
                }
        }
    }

    // mass_weight is internal to this normalization and never written out -- release it for real
    std::vector<Real>().swap( this->mass_weight );
}


// Like addField, but 'src' is a sub-grid (dims m, global origin o) of the full grid 'full';
// each cell is mapped to its global row-major index. o, m, full are in grid-cell units.
template<typename T>
void addFieldSubgrid(std::vector<T> const &src, std::vector<T> *dst,
                     size_t const *o, size_t const *m, size_t const *full)
{
    if (src.empty()) return;
    size_t fullTotal = 1; for (int d = 0; d < NO_DIM; ++d) fullTotal *= full[d];
    if (dst->size() < fullTotal) dst->resize(fullTotal, T());
    size_t subTotal = 1; for (int d = 0; d < NO_DIM; ++d) subTotal *= m[d];
    for (size_t l = 0; l < subTotal; ++l)
    {
        size_t rem = l, c[NO_DIM];
        for (int d = NO_DIM - 1; d >= 0; --d) { c[d] = rem % m[d]; rem /= m[d]; }   // local coords
        size_t g = 0;
        for (int d = 0; d < NO_DIM; ++d) g = g * full[d] + (c[d] + o[d]);           // global row-major flat
        (*dst)[g] += src[l];
    }
}

// Accumulates 'other' (which holds only its Eulerian sub-box) into the full grid, mapping each cell by global index.
void Quantities::addFromSubgrid(Quantities const &other, size_t const *fullGrid)
{
    if (other.ps_subDims[0] == 0) { this->addFrom(other); return; }   // 'other' spans the full grid
    size_t const *o = other.ps_subOrigin;
    size_t const *m = other.ps_subDims;
    addFieldSubgrid(other.density, &this->density, o, m, fullGrid);
    addFieldSubgrid(other.velocity, &this->velocity, o, m, fullGrid);
    addFieldSubgrid(other.velocity_gradient, &this->velocity_gradient, o, m, fullGrid);
    addFieldSubgrid(other.velocity_divergence, &this->velocity_divergence, o, m, fullGrid);
    addFieldSubgrid(other.velocity_shear, &this->velocity_shear, o, m, fullGrid);
    addFieldSubgrid(other.velocity_vorticity, &this->velocity_vorticity, o, m, fullGrid);
    addFieldSubgrid(other.velocity_std, &this->velocity_std, o, m, fullGrid);
    addFieldSubgrid(other.velocity_dispersion, &this->velocity_dispersion, o, m, fullGrid);
    addFieldSubgrid(other.scalar, &this->scalar, o, m, fullGrid);
    addFieldSubgrid(other.scalar_gradient, &this->scalar_gradient, o, m, fullGrid);
    addFieldSubgrid(other.velocity_tweb_eigenvalues, &this->velocity_tweb_eigenvalues, o, m, fullGrid);
    addFieldSubgrid(other.velocity_vweb_eigenvalues, &this->velocity_vweb_eigenvalues, o, m, fullGrid);
    addFieldSubgrid(other.stream_count, &this->stream_count, o, m, fullGrid);
    addFieldSubgrid(other.mass_weight, &this->mass_weight, o, m, fullGrid);
}
#endif


// Reserves and zero-fills main-grid memory for each requested quantity ('--partition' option).
void Quantities::reserveMemory(size_t *gridSize, Field &field)
{
    size_t totalSize = 1;
    for (size_t i=0; i<NO_DIM; ++i)
        totalSize *= gridSize[i];
    
    if ( field.density )
        this->density.resize( totalSize, Real(0.) );
    if ( field.velocity || field.velocity_dispersion )   // dispersion needs <v> too, so allocate velocity even if only dispersion is requested
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
    if ( field.velocity_dispersion )
        this->velocity_dispersion.resize( totalSize, Pvector<Real,noDispComp>::zero() );
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


