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
    fieldSize( this->tet_touch, &temp );
    fieldSize( this->mass_weight, &temp );
    fieldSize( this->disp_weight, &temp );
    fieldSize( this->disp_velocity, &temp );
    fieldSize( this->caustic_bits, &temp );
    fieldSize( this->hidden_streams, &temp );
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

#ifdef PHASE_SPACE
// Bitwise-OR accumulate for bit-mask fields stored as Real (small exact integers, e.g. the
// caustic orientation bits). OR is commutative AND idempotent, so -- unlike '+=' -- the merge
// is invariant under the partitions' completion order and under a stream being seen by the
// deposit any number of times.
void orField(std::vector<Real> const &src, std::vector<Real> *dst)
{
    if (src.empty()) return;
    if (dst->size() < src.size())
        dst->resize(src.size(), Real(0.));
    for (size_t i = 0; i < src.size(); ++i)
        (*dst)[i] = Real( int((*dst)[i]) | int(src[i]) );
}
#endif

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
    addField(other.tet_touch, &this->tet_touch);   // integer tet-touch counts sum linearly across partitions
    addField(other.mass_weight, &this->mass_weight);
    addField(other.disp_weight, &this->disp_weight);
    addField(other.disp_velocity, &this->disp_velocity);
    orField(other.caustic_bits, &this->caustic_bits);   // orientation bits: OR, never '+='
    orField(other.hidden_streams, &this->hidden_streams);       // bitmask: OR
#endif
}

#ifdef PHASE_SPACE
// '.hidden_streams' bit 1 is marked per partition from the geometry alone (a flipped tet overlaps the
// cell); whether the SAMPLES missed it is only known once every partition's samples are summed
// into stream_count. Where '.streams' already reads multi-stream the samples saw it, so the bit
// is cleared: bit 1 then means exactly "multi-stream volume that '.streams' does not show".
void Quantities::finalizeHiddenStreams()
{
    if ( hidden_streams.empty() || stream_count.size() != hidden_streams.size() ) return;
    for (size_t i = 0; i < hidden_streams.size(); ++i)
    {
        int const bits = int(hidden_streams[i]);
        if ( (bits & 1) && stream_count[i] > Real(1.) + PS_STREAM_TOL )
            hidden_streams[i] = Real(bits & ~1);
    }
}
#endif


#ifdef PHASE_SPACE
// PS-DTFE partition path: partitions accumulate density-weighted moments sum(rho_s f_s) and per-cell
// mass sum(rho_s); divide once here after all partitions are summed, since averaging is non-linear.
// The weight is either the explicit mass_weight grid, or -- when the density field was selected and
// the partitions aliased the weight to it -- reconstructed as density * weightFromDensityScale
// (= density * cellVolume * averageDensity, undoing the rho/rho_bar conversion; saves 4 B/cell).
void Quantities::normalizePhaseSpace(Field const &field, Real const weightFromDensityScale)
{
    bool const fromDensity = this->mass_weight.empty();
    if ( fromDensity and (weightFromDensityScale <= Real(0.) or this->density.empty()) )
        return;   // no mass-weighted field was deferred (or nothing to reconstruct it from)
    size_t const n = fromDensity ? this->density.size() : this->mass_weight.size();
    // velocity holds sum(rho v). A dispersion-only VOLUME-WEIGHTED run never allocates it --
    // the dispersion carries its own mean in disp_velocity (nonempty disp_weight is that
    // configuration's signature), so infer the gate from the data actually present.
    bool const haveVel = field.velocity
                         || (field.velocity_dispersion and this->disp_weight.empty());
    if ( haveVel and this->velocity.empty() )
        return;   // no weighted moments were accumulated (e.g. density-only field selection)
    for (size_t i = 0; i < n; ++i)
    {
        Real const w = fromDensity ? this->density[i] * weightFromDensityScale : this->mass_weight[i];
        if ( w <= Real(0.) ) continue;
        Real const inv = Real(1.) / w;
        if ( haveVel )                 this->velocity[i]          *= inv;   // <v> = sum(rho v)/sum(rho)
        if ( field.velocity_gradient ) this->velocity_gradient[i] *= inv;
        if ( field.scalar )            this->scalar[i]            *= inv;
        if ( field.scalar_gradient )   this->scalar_gradient[i]   *= inv;
        if ( field.velocity_dispersion and this->disp_weight.empty() )
        {
            // sigma_ij = <v_i v_j> - <v_i><v_j>, using the now-normalized <v>. Same weighting
            // for both moments here, so 'velocity' IS the right mean.
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

    // --ps-volume-weighted + dispersion: the dispersion carries MASS-weighted moments while
    // 'velocity' above was normalized by the VOLUME weight, so it must be closed out with its
    // own mass-weighted mean -- a separate pass over its own normalizer.
    if ( field.velocity_dispersion and not this->disp_weight.empty() )
    {
        size_t const nd = this->disp_weight.size();
        for (size_t i = 0; i < nd; ++i)
        {
            Real const wm = this->disp_weight[i];
            if ( wm <= Real(0.) ) continue;
            Real const invm = Real(1.) / wm;
            Pvector<Real,noVelComp> const vbar = this->disp_velocity[i] * invm;   // <v>_mass
            size_t c = 0;
            for (int a = 0; a < NO_DIM; ++a)
                for (int b = a; b < NO_DIM; ++b)
                {
                    Real s = this->velocity_dispersion[i][c] * invm - vbar[a]*vbar[b];
                    if ( a == b and s < Real(0.) ) s = Real(0.);   // variance: clamp FP-noise negatives
                    this->velocity_dispersion[i][c] = s;
                    ++c;
                }
        }
    }

    // these three are internal to this normalization and never written out -- release for real
    std::vector<Real>().swap( this->mass_weight );
    std::vector<Real>().swap( this->disp_weight );
    std::vector< Pvector<Real,noVelComp> >().swap( this->disp_velocity );
}


// Like addField, but 'src' is a sub-grid (dims m, global origin o) of the full grid 'full';
// each cell is mapped to its global row-major index. o, m, full are in grid-cell units.
// The destination may itself be a WINDOW of the full grid (--ps-window: mainOrigin/mainDims given):
// a source cell is then mapped to its window-local index and skipped when it lies outside.
inline bool subgridTarget(size_t const *c, size_t const *o, size_t const *full,
                          size_t const *mainOrigin, size_t const *mainDims, size_t &g)
{
    g = 0;
    for (int d = 0; d < NO_DIM; ++d)
    {
        // '% full[d]': the sub-box may WRAP a periodic axis (o[d]+m[d] > full[d]), which is the
        // normal case for a partition straddling the box seam. No-op for an unwrapped box.
        size_t gl = (c[d] + o[d]) % full[d];              // global index
        if (mainOrigin)
        {
            long ml = long(gl) - long(mainOrigin[d]);
            if (ml < 0) ml += long(full[d]);              // the window may wrap too
            if (ml < 0 || ml >= long(mainDims[d])) return false;
            g = g * mainDims[d] + size_t(ml);
        }
        else
            g = g * full[d] + gl;                         // global row-major flat
    }
    return true;
}

template <typename T>
void addFieldSubgrid(std::vector<T> const &src, std::vector<T> *dst,
                     size_t const *o, size_t const *m, size_t const *full,
                     size_t const *mainOrigin = nullptr, size_t const *mainDims = nullptr)
{
    if (src.empty()) return;
    size_t dstTotal = 1; for (int d = 0; d < NO_DIM; ++d) dstTotal *= mainOrigin ? mainDims[d] : full[d];
    if (dst->size() < dstTotal) dst->resize(dstTotal, T());
    size_t subTotal = 1; for (int d = 0; d < NO_DIM; ++d) subTotal *= m[d];
    for (size_t l = 0; l < subTotal; ++l)
    {
        size_t rem = l, c[NO_DIM];
        for (int d = NO_DIM - 1; d >= 0; --d) { c[d] = rem % m[d]; rem /= m[d]; }   // local coords
        size_t g;
        if (!subgridTarget(c, o, full, mainOrigin, mainDims, g)) continue;
        (*dst)[g] += src[l];
    }
}

// The bitwise-OR twin of addFieldSubgrid, for the mask fields (caustic_bits, hidden_streams): same
// sub-grid -> global mapping -- INCLUDING the '% full[d]' wrap of a sub-box that straddles a
// periodic seam; a hand-rolled copy of this loop once lacked it and indexed past the end --
// but OR instead of '+='.
void orFieldSubgrid(std::vector<Real> const &src, std::vector<Real> *dst,
                    size_t const *o, size_t const *m, size_t const *full,
                    size_t const *mainOrigin = nullptr, size_t const *mainDims = nullptr)
{
    if (src.empty()) return;
    size_t dstTotal = 1; for (int d = 0; d < NO_DIM; ++d) dstTotal *= mainOrigin ? mainDims[d] : full[d];
    if (dst->size() < dstTotal) dst->resize(dstTotal, Real(0.));
    size_t subTotal = 1; for (int d = 0; d < NO_DIM; ++d) subTotal *= m[d];
    for (size_t l = 0; l < subTotal; ++l)
    {
        size_t rem = l, c[NO_DIM];
        for (int d = NO_DIM - 1; d >= 0; --d) { c[d] = rem % m[d]; rem /= m[d]; }
        size_t g;
        if (!subgridTarget(c, o, full, mainOrigin, mainDims, g)) continue;
        (*dst)[g] = Real( int((*dst)[g]) | int(src[l]) );
    }
}

// Accumulates 'other' (which holds only its Eulerian sub-box) into the full grid, mapping each cell by
// global index -- or into THIS grid's own window of it (mainOrigin/mainDims, --ps-window).
void Quantities::addFromSubgrid(Quantities const &other, size_t const *fullGrid, size_t const *mainOrigin, size_t const *mainDims)
{
    if (other.ps_subDims[0] == 0) { this->addFrom(other); return; }   // 'other' spans the full grid
    size_t const *o = other.ps_subOrigin;
    size_t const *m = other.ps_subDims;
    if (mainOrigin && mainDims)
    {   // the main grid is a window: every field goes through the window-aware mapping
        #define ADD_W(F) addFieldSubgrid(other.F, &this->F, o, m, fullGrid, mainOrigin, mainDims)
        ADD_W(density); ADD_W(velocity); ADD_W(velocity_gradient); ADD_W(velocity_divergence);
        ADD_W(velocity_shear); ADD_W(velocity_vorticity); ADD_W(velocity_std); ADD_W(velocity_dispersion);
        ADD_W(scalar); ADD_W(scalar_gradient); ADD_W(velocity_tweb_eigenvalues); ADD_W(velocity_vweb_eigenvalues);
        ADD_W(stream_count); ADD_W(tet_touch); ADD_W(mass_weight); ADD_W(disp_weight); ADD_W(disp_velocity);
        #undef ADD_W
        orFieldSubgrid(other.caustic_bits, &this->caustic_bits, o, m, fullGrid, mainOrigin, mainDims);
        orFieldSubgrid(other.hidden_streams, &this->hidden_streams, o, m, fullGrid, mainOrigin, mainDims);
        return;
    }
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
    addFieldSubgrid(other.tet_touch, &this->tet_touch, o, m, fullGrid);
    addFieldSubgrid(other.mass_weight, &this->mass_weight, o, m, fullGrid);
    addFieldSubgrid(other.disp_weight, &this->disp_weight, o, m, fullGrid);
    addFieldSubgrid(other.disp_velocity, &this->disp_velocity, o, m, fullGrid);
    // the mask fields: same sub-grid -> global mapping, but OR instead of '+='
    orFieldSubgrid(other.caustic_bits, &this->caustic_bits, o, m, fullGrid);
    orFieldSubgrid(other.hidden_streams, &this->hidden_streams, o, m, fullGrid);
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


