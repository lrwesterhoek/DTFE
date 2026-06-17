/* Exact volume-averaging support: clips a Delaunay simplex against the grid walls and
   accumulates each grid cell's enclosed sub-volume and volume-weighted centroid. */

#ifndef VOLUME_SPLIT_HEADER
#define VOLUME_SPLIT_HEADER

#include "../Pvector.h"
#include <cmath>


#ifndef VOLUME_TOL
    #define VOLUME_TOL Real(1.e-3)   // default relative volume below which a fragment is accumulated as-is
#endif

#define NO_PAIRS ((NO_DIM+1)*NO_DIM/2)   // number of edges (vertex pairs) of a simplex


namespace VOLUME_SPLIT
{

// A simplex vertex in grid-cell coordinates, caching the grid-index bracket [nMin,nMax] per axis
// so edge/grid-wall crossing tests are integer comparisons.
struct Vertex
{
    Pvector<Real,NO_DIM> coords;
    Pvector<int,NO_DIM>  nMin;   // floor grid index per axis (with tolerance)
    Pvector<int,NO_DIM>  nMax;   // ceil grid index per axis (with tolerance)

    inline Real& operator[](int i)
    {
        return this->coords[i];
    }

    static Vertex zero()
    {
        return Vertex();
    }

    // Refresh the [nMin,nMax] grid-index bracket for axis i from coordinate p, widened by tol.
    inline void update_nMinMax(int const i, Real const p, Real const tol)
    {
        this->nMin[i]  = (int) (p-tol);
        this->nMax[i]  = (int) (p+tol);
    }
};




// A triangle/tetrahedron (NO_DIM+1 vertices) being clipped against the grid.
struct Simplex
{
    std::vector< Vertex > pos;
    Real vol;   // cached simplex volume

    Simplex()
    {
        pos.assign( NO_DIM+1, Vertex::zero() );
    }

    Vertex& operator[](int i)
    {
        return this->pos[i];
    }

    // Copy a Delaunay cell's vertex coordinates into this simplex.
    template <typename T>
    void assign(T const cell)
    {
        for (int i=0; i<NO_DIM+1; ++i)
            for (int j=0; j<NO_DIM; ++j)
                pos[i][j] = cell->vertex(i)->point()[j];
    }

    // Volume from the edge vectors relative to vertex 0.
    Real volume()
    {
        double posDiff[NO_DIM][NO_DIM];
        for (int i=0; i<NO_DIM; ++i)
            for (int j=0; j<NO_DIM; ++j)
                posDiff[i][j] = double(this->pos[i+1][j] - this->pos[0][j]);
        return simplexVolume(posDiff);
    }

    // Centroid (mean of the vertices) into cm.
    void massCenter(Vertex &cm)
    {
        for (int i=0; i<NO_DIM; ++i)
        {
            cm[i] = Real(0.);
            for (int j=0; j<NO_DIM+1; ++j)
                cm[i] += this->pos[j][i];
            cm[i] /= Real(NO_DIM+1);
        }
    }
};








// Clips simplices against the grid and accumulates each cell's enclosed volume and weighted centroid.
struct VolumeSplit
{
    // Fixed for all simplices.
    Real start[NO_DIM];     // origin of the full grid
    size_t  grid[NO_DIM];   // dimensions of the full grid
    Real dx[NO_DIM];        // grid spacing per dimension
    int  pairs[NO_PAIRS][2];// vertex-pair permutations giving all edges of the triangle/tetrahedron
    Real tol;               // tolerance for real-number comparisons
    Real tolVolume;         // fragment volume below which recursion stops and the piece is accumulated
    Real cellVolume;        // volume of one grid cell

    // Reset per simplex.
    Real currentStart[NO_DIM];  // origin of the small grid enclosing the current simplex
    int  newGrid[NO_DIM];       // dimensions of that small grid (same dx as the main grid)
    std::vector< Pvector<Real,NO_DIM+1> > *results;    // per-cell contributions of the small grid

    std::vector< Simplex > simplices;   // pre-allocated work queue for the iterative split

    // Sets up the grid geometry and precomputes the simplex edge list.
    VolumeSplit(Real *startPos, size_t *gridDims, Real *dxValues)
    {
        tol = Real(1.e-5);
        tolVolume = 1.e-3;
        cellVolume = Real(1.);
        for (int i=0; i<NO_DIM; ++i)
        {
            start[i] = startPos[i];
            grid[i] = gridDims[i];
            dx[i] = dxValues[i];
            cellVolume *= dx[i];
        }
         simplices.assign( 50000, Simplex() );
        
        {
            int pairIdx = 0;
            for (int i=0; i<NO_DIM+1; ++i)
                for (int j=i+1; j<NO_DIM+1; ++j)
                {
                    pairs[pairIdx][0] = i;
                    pairs[pairIdx][1] = j;
                    ++pairIdx;
                }
        }
    }
    
    
    // Orders the simplex edges (vertex pairs) by descending length into newPairs, so the longest
    // edge (most likely to straddle a grid wall) is tested first.
    void sortPairs(Simplex &pos,
                   int pairs[NO_PAIRS][2],
                   int newPairs[NO_PAIRS][2])
    {
        Real len[NO_PAIRS];     // squared length of each simplex edge
        int  count[NO_PAIRS];   // rank of each edge in descending length order
        for (int i=0; i<NO_PAIRS; ++i)
        {
            len[i] = Real(0.);
            int i1 = pairs[i][0], i2 = pairs[i][1];
            for (int j=0; j<NO_DIM; ++j)
            {
                Real temp = pos[i1][j] - pos[i2][j];
                len[i] += temp*temp;
            }
            count[i] = 0;
        }
        for (int i=0; i<NO_PAIRS; ++i)
            for (int j=i+1; j<NO_PAIRS; ++j)
                if ( len[i]>=len[j] )
                    count[j] += 1;
                else
                    count[i] += 1;
        // Scatter each edge into newPairs at its rank position.
        for (int i=0; i<NO_PAIRS; ++i)
            for (int j=0; j<2; ++j)
               newPairs[count[i]][j] = pairs[i][j];
    }


    // Flat small-grid index for integer cell coordinates n; -1 if outside the small grid.
    int index_smallGrid(int *n)
    {
        int result = 0;
        for (int d=0; d<NO_DIM; ++d)
        {
            if (n[d]<0 or n[d]>=this->newGrid[d]) return -1;
            result = result * newGrid[d] + n[d];
        }
        return result;
    }

    // Flat small-grid index of a point (truncates coordinates to the containing cell).
    int index_smallGrid(Vertex pos)
    {
        int n[NO_DIM];
        for (int i=0; i<NO_DIM; ++i)
            n[i] = (int) pos[i];
        return index_smallGrid( n );
    }
    
    
    // Accumulates a fully-contained simplex fragment into its grid cell: adds its volume and its
    // volume-weighted centroid. Stores volume in slot [NO_DIM] and weighted position in slots [0..NO_DIM).
    void simplexContribution(Simplex &pos)
    {
        this->simplexContribution( pos, pos.volume() );
    }
    void simplexContribution(Simplex &pos, Real vol)
    {
        // The centroid picks the cell the fragment is assigned to.
        Vertex cm;
        pos.massCenter( cm );
        int index = this->index_smallGrid( cm );
        if (index<0)
            return;
        Real volume = vol * cellVolume;   // convert grid-cell-unit volume back to physical units

        (*this->results)[index][NO_DIM] += volume;
        for (int i=0; i<NO_DIM; ++i)
            (*this->results)[index][i] += ( cm[i]*dx[i] + currentStart[i] ) * volume;
    }


    // Recursive variant: split the simplex at the first crossed grid wall, else accumulate it.
    void simplexIteration(Simplex &pos, Real vol)
    {
        int sortedPairs[NO_PAIRS][2];
        this->sortPairs( pos, this->pairs, sortedPairs );
        
        
        // Find the first edge crossed by a grid wall; that edge defines where to split.
        for (int i=0; i<NO_PAIRS; ++i)
        {
            int i1 = sortedPairs[i][0];
            int i2 = sortedPairs[i][1];

            // Does a grid wall on axis j separate this edge's endpoints?
            for (int j=0; j<NO_DIM; ++j)
            {
                if ( pos[i1].nMin[j]>pos[i2].nMax[j] or pos[i1].nMax[j]<pos[i2].nMin[j] )
                {
                    int const j2 = (j+1) % NO_DIM;// the other axes, by cyclic rotation of j
                    int const j3 = (j+2) % NO_DIM;

                    // Grid plane to split on: just above the lower of the two endpoints along axis j.
                    int nMin = pos[i1].nMax[j];
                    if ( pos[i1].nMin[j]>pos[i2].nMax[j] )
                        nMin = pos[i2].nMax[j];

                    // Edge parameterized in j; slope/intercept give its j2,j3 coords at the split plane.
                    Real temp = pos[i1][j]-pos[i2][j];
                    Real slope2 = (pos[i1][j2]-pos[i2][j2]) / temp;
                    Real constant2 = pos[i1][j2] - slope2*pos[i1][j];
#if NO_DIM==3
                    Real slope3 = (pos[i1][j3]-pos[i2][j3]) / temp;
                    Real constant3 = pos[i1][j3] - slope3*pos[i1][j];
#endif

                    // Intersection point where the edge meets the grid plane j = nMin+1.
                    Vertex newPoint;
                    newPoint[j]  = Real(nMin+1);
                    newPoint[j2] = slope2*newPoint[j] + constant2;
                    newPoint.nMin[j]  = nMin;
                    newPoint.nMax[j]  = nMin+1;
                    newPoint.update_nMinMax( j2, newPoint[j2], this->tol );
#if NO_DIM==3
                    newPoint[j3] = slope3*newPoint[j] + constant3;
                    newPoint.update_nMinMax( j3, newPoint[j3], this->tol );
#endif
                    
                    // Split into two simplices at newPoint and recurse on each (or accumulate tiny ones).
                    Simplex pos2 = pos;
                    pos2[i2] = newPoint;
                    Real vol1 = pos2.volume();
                    if ( vol1<this->tolVolume )
                        this->simplexContribution( pos, vol1 );
                    else
                        this->simplexIteration( pos2, vol1 );
                    pos2[i1] = pos[i2];
                    vol1 = vol - vol1;   // remaining fragment volume (total minus first piece)
                    if ( vol1<this->tolVolume )
                        this->simplexContribution( pos, vol1 );
                    else
                        this->simplexIteration( pos2, vol1 );
                    return;
                }
            }
        }
        
        // No grid wall crosses any edge: the simplex lies within one cell, so accumulate it.
        this->simplexContribution( pos, vol );
    }


    // Iterative (explicit-stack) equivalent of simplexIteration, avoiding deep recursion: splits
    // the queue of simplices until each fits in a single grid cell. Used by findIntersection.
    void simplexIteration2(Simplex &initSimplex, Real volPrev)
    {
        simplices[0] = initSimplex;
        int current = 0, total = 1;
        simplices[0].vol = volPrev;

        // Process the work queue until every fragment fits inside a single grid cell.
        while ( true )
        {
            int sortedPairs[NO_PAIRS][2];
            Simplex *pos = &(simplices[current]);
            this->sortPairs( *pos, this->pairs, sortedPairs );
            bool simplexFinished = false;
            bool simplexSplit = false;
            
            
            // Find the first edge crossed by a grid wall; that edge defines where to split.
            for (int i=0; i<NO_PAIRS; ++i)
            {
                int i1 = sortedPairs[i][0];
                int i2 = sortedPairs[i][1];

                // Does a grid wall on axis j separate this edge's endpoints?
                for (int j=0; j<NO_DIM; ++j)
                {
                    if ( (*pos)[i1].nMin[j]>(*pos)[i2].nMax[j] or (*pos)[i1].nMax[j]<(*pos)[i2].nMin[j] )
                    {
                        int const j2 = (j+1) % NO_DIM;// the other axes, by cyclic rotation of j
                        int const j3 = (j+2) % NO_DIM;

                        // Grid plane to split on: just above the lower of the two endpoints along axis j.
                        int nMin = (*pos)[i1].nMax[j];
                        if ( (*pos)[i1].nMin[j]>(*pos)[i2].nMax[j] )
                            nMin = (*pos)[i2].nMax[j];

                        // Edge parameterized in j; slope/intercept give its j2,j3 coords at the split plane.
                        Real temp = (*pos)[i1][j]-(*pos)[i2][j];
                        Real slope2 = ((*pos)[i1][j2]-(*pos)[i2][j2]) / temp;
                        Real constant2 = (*pos)[i1][j2] - slope2*(*pos)[i1][j];
#if NO_DIM==3
                        Real slope3 = ((*pos)[i1][j3]-(*pos)[i2][j3]) / temp;
                        Real constant3 = (*pos)[i1][j3] - slope3*(*pos)[i1][j];
#endif
                        
                        // Intersection point where the edge meets the grid plane j = nMin+1.
                        Vertex newPoint;
                        newPoint[j]  = Real(nMin+1);
                        newPoint[j2] = slope2*newPoint[j] + constant2;
                        newPoint.nMin[j]  = nMin;
                        newPoint.nMax[j]  = nMin+1;
                        newPoint.update_nMinMax( j2, newPoint[j2], this->tol );
#if NO_DIM==3
                        newPoint[j3] = slope3*newPoint[j] + constant3;
                        newPoint.update_nMinMax( j3, newPoint[j3], this->tol );
#endif

                        // Split at newPoint: the new fragment goes on the queue, *pos keeps the rest.
                        simplices[total] = *pos;
                        simplices[total][i1] = newPoint;
                        simplices[total].vol = simplices[total].volume();
                        (*pos)[i2] = newPoint;
                        pos->vol -= simplices[total].vol;
                        if ( pos->vol<this->tolVolume )
                        {
                            this->simplexContribution( *pos, pos->vol ); // below volume tolerance: accumulate
                            simplexFinished = true;
                        }
                        if ( simplices[total].vol<this->tolVolume )
                            this->simplexContribution( simplices[total], simplices[total].vol ); // below volume tolerance: accumulate
                        else
                            ++total;    // queue the new simplex for further splitting

                        simplexSplit = true;
                        break;
                    }
                }
                if ( simplexSplit ) break;
            }
            
            if ( not simplexSplit ) // no further split: accumulate this simplex
            {
                this->simplexContribution( *pos, pos->vol );
                simplexFinished = true;
            }

            if ( simplexFinished )
                if ( ++current==total )   // no simplices left
                    return;
        }
    }
    
    
    // Main-grid cell index of coordinate x along one axis (floor, including negative x).
    inline int mainGridIndex(Real const x, int const axis)
    {
        Real temp = (x - this->start[axis]) / this->dx[axis];
        if (temp>=Real(0.))
            return int( temp );
        else
            return int( temp ) - 1;
    }

    // Builds the smallest sub-grid enclosing the simplex, maps its cells to main-grid indices, and
    // rescales the simplex coordinates into small-grid units (cells of size 1, origin at currentStart).
    void findBoundingGrid(Simplex &pos,
                          vector<size_t> &indices)
    {
        int nMin[NO_DIM], nMax[NO_DIM];
        for (int j=0; j<NO_DIM; ++j)
        {
            nMin[j] = this->mainGridIndex( pos[0][j], j );
            nMax[j] = nMin[j];
        }
        
        // Expand [nMin,nMax] per axis to cover every vertex.
        for (int i=1; i<NO_DIM+1; ++i)
            for (int j=0; j<NO_DIM; ++j)
            {
                int temp = this->mainGridIndex( pos[i][j], j );
                if (temp<nMin[j]) nMin[j] = temp;
                if (temp>nMax[j]) nMax[j] = temp;
            }

        // Make the upper bound exclusive so the span covers the cell containing nMax.
        for (int j=0; j<NO_DIM; ++j)
            nMax[j] += 1;

        // Set up the small grid: its origin, dimensions, and total cell count.
        int newSize = 1;
        for (int j=0; j<NO_DIM; ++j)
        {
            this->currentStart[j] = this->start[j] + nMin[j] * this->dx[j];
            this->newGrid[j] =  nMax[j] -  nMin[j];
            newSize *= this->newGrid[j];
        }
        
        // map each small-grid cell to its main-grid index (-1 if outside)
        indices.assign( newSize, size_t(-1) );
        for (size_t flatIdx=0; flatIdx<newSize; ++flatIdx)
        {
            size_t rem = flatIdx;
            int cellIdx[NO_DIM];
            bool valid = true;
            for (int d=NO_DIM-1; d>=0; --d) { cellIdx[d] = nMin[d] + int(rem % this->newGrid[d]); rem /= this->newGrid[d]; }
            size_t mainIdx = 0;
            for (int d=0; d<NO_DIM; ++d)
            {
                if (cellIdx[d]<0 or cellIdx[d]>=grid[d]) { valid = false; break; }
                mainIdx = mainIdx * grid[d] + cellIdx[d];
            }
            if (valid)
                indices[flatIdx] = mainIdx;
        }
        
        // Rescale vertex coordinates into small-grid units (origin at currentStart, cell size 1).
        for (int i=0; i<NO_DIM+1; ++i)
            for (int j=0; j<NO_DIM; ++j)
            {
                pos[i][j] = (pos[i][j] - this->currentStart[j]) / this->dx[j];
                pos[i].update_nMinMax( j, pos[i][j], this->tol );
            }
    }


    // Public entry point: clips one simplex across the grid, returning per-small-cell contributions
    // and the small-cell -> main-grid index map. contributions[c] = (weighted centroid, volume).
    void findIntersection(Simplex &pos,
                          vector<size_t> &indices,
                          vector< Pvector<Real,NO_DIM+1> > &contributions)
    {
        this->results = &contributions;

        this->findBoundingGrid( pos, indices );
        contributions.assign( indices.size(), Pvector<Real,NO_DIM+1>::zero() );

        // Scale the stop tolerance to the simplex size, with an absolute floor to bound recursion depth.
        Real vol = pos.volume();
        this->tolVolume = Real(1.)<vol ? VOLUME_TOL : VOLUME_TOL*vol;
        if ( this->tolVolume<Real(1.e-6) )  this->tolVolume = Real(1.e-6);
        this->simplexIteration2( pos, vol );
    }
};


}   //end of namespace VOLUME_SPLIT
#endif
