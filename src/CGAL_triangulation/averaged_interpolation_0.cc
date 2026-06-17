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


/* Volume-averaged grid interpolation, method 0 (= exact geometric intersection of each Delaunay
   simplex with the grid cells). */
#include "volume_split.h"


// Adds each simplex's density contribution to its intersecting grid cells. In every entry of
// 'contributions' the first NO_DIM components are the field-weighted intersection moment and entry
// NO_DIM is the intersection volume, so the result reconstructs the linear field over that volume.
inline void distributeContributions(vector<size_t> &indices,
                                    vector< Pvector<Real,NO_DIM+1> > &contributions,
                                    std::vector<Real> *density,
                                    Real const base_density,
                                    Real const densityGrad[NO_DIM],
                                    Vertex_handle &base)
{
    for (size_t i=0; i<indices.size(); ++i)
    {
        size_t index = indices[i];
        if ( index==size_t(-1) )
            continue;
        Real volume = contributions[i][NO_DIM];
        Real result = base_density * volume;
        for (int j=0; j<NO_DIM; ++j)
            result += densityGrad[j] * (contributions[i][j] - volume*base->point()[j]);
        (*density)[index] += result;
    }
}
// Vector-field overload: distributes a multi-component field with its per-cell gradient the same way.
template <size_t N1, size_t N2>
inline void distributeContributions(vector<size_t> &indices,
                                    vector< Pvector<Real,NO_DIM+1> > &contributions,
                                    std::vector< Pvector<Real,N1> > *field,
                                    Pvector<Real,N1> field_base,
                                    Real fieldGradient[NO_DIM][N2],
                                    Vertex_handle &base)
{
    for (size_t i=0; i<indices.size(); ++i)
    {
        size_t index = indices[i];
        if ( index==size_t(-1) )
            continue;
        Real volume = contributions[i][NO_DIM];
        Pvector<Real,N1> result = field_base * volume;
        for (int j=0; j<NO_DIM; ++j)
            for (int j2=0; j2<N1; ++j2)
                result[j2] += fieldGradient[j][j2] * (contributions[i][j] - volume*base->point()[j]);
        (*field)[index] += result;
    }
}
// Gradient overload: the gradient is constant inside a Delaunay simplex, so it just scales by volume.
template <size_t N1>
inline void distributeContributions(vector<size_t> &indices,
                                    vector< Pvector<Real,NO_DIM+1> > &contributions,
                                    std::vector< Pvector<Real,N1> > *fieldGrad,
                                    Pvector<Real,N1> fieldGradient)
{
    for (size_t i=0; i<indices.size(); ++i)
    {
        size_t index = indices[i];
        if ( index==size_t(-1) )
            continue;
        Real volume = contributions[i][NO_DIM];
        (*fieldGrad)[index] += fieldGradient * volume;
    }
}




// Returns true if the Delaunay cell's bounding box lies entirely outside the region of interest.
inline bool cellOutsideRegion(DT & dt,
                              Finite_cells_iterator &cell,
                              Bbox &fullBox)
{
#if NO_DIM==2
    return not do_overlap( dt.triangle( cell ).bbox(), fullBox );
#elif NO_DIM==3
    return not do_overlap( dt.tetrahedron( cell ).bbox(), fullBox );
#endif
}

// Returns true if the whole Delaunay cell fits inside one grid cell; also fills the base vertex's
// grid cell and box-relative position.
bool simplexInSingleCell(Finite_cells_iterator &cell,
                         Box &regionBox,
                         Real *dx,
                         int *baseGridCell,
                         Real *basePosition)
{
    Point base = cell->vertex(0)->point();
    for (int i=0; i<NO_DIM; ++i)
    {
        basePosition[i] = base[i] - regionBox[2*i]; // base vertex position relative to the box's lower corner
        baseGridCell[i] = int( floor( (basePosition[i])/dx[i] ) ); // grid cell containing the base point
    }
    Box gridCellBox;    // coordinates of the grid cell containing 'base'
    for (int i=0; i<NO_DIM; ++i)
    {
        gridCellBox[2*i] = baseGridCell[i] * dx[i] + regionBox[2*i];
        gridCellBox[2*i+1] = gridCellBox[2*i] + dx[i];
    }
    for (int i=1; i<NO_DIM+1; ++i)
        if ( not gridCellBox.isPointInBox( cell->vertex(i)->point() ) )
            return false;
    return true;
}





// Volume-averaged grid interpolation (method 0): intersects each Delaunay simplex with the grid and
// accumulates per-cell contributions, giving the exact geometric volume average. 'dt' carries vertex
// densities; 'quantities' receives the output grid fields.
void interpolateGrid_averaged_0(DT &dt,
                                User_options &userOptions,
                                Quantities *quantities)
{
    size_t *nGrid = &(userOptions.gridSize[0]); // grid size along each axis
    Box boxCoordinates = userOptions.region;    // box region of interest
    vector<Real> boxLength, startPos;           // box size and origin along each direction
    for (size_t i=0; i<NO_DIM; ++i)
    {
        boxLength.push_back( boxCoordinates[2*i+1]-boxCoordinates[2*i] );
        startPos.push_back( boxCoordinates[2*i] );
    }
#if NO_DIM==2
    Bbox fullBox( boxCoordinates[0], boxCoordinates[2], boxCoordinates[1], boxCoordinates[3] );
#elif NO_DIM==3
    Bbox fullBox( boxCoordinates[0], boxCoordinates[2], boxCoordinates[4], boxCoordinates[1], boxCoordinates[3], boxCoordinates[5] );
#endif
    
    
    // output field vectors
    Field field = userOptions.aField;
    std::vector<Real>                         *density = &(quantities->density);
    std::vector< Pvector<Real,noVelComp> >    *velocity = &(quantities->velocity);
    std::vector< Pvector<Real,noGradComp> >   *velocity_gradient = &(quantities->velocity_gradient);
    std::vector< Pvector<Real,noScalarComp> > *scalar = &(quantities->scalar);
    std::vector< Pvector<Real,noScalarGradComp> > *scalar_gradient = &(quantities->scalar_gradient);


    MESSAGE::Message message( userOptions.verboseLevel );
    message << "\nComputing interpolation of the fields volume averaged over the sampling cell on a regular "
            << MESSAGE::printElements( nGrid, NO_DIM, "*" ) << " grid in the region "
            << boxCoordinates.print()
            << ". The volume average is computed using geometric intersections up to a precision of " << VOLUME_TOL << ".\n"
            << "\t Done: " << MESSAGE::Flush;


    // zero-initialize all grid fields
    size_t const gridSize = (NO_DIM==2) ? nGrid[0]*nGrid[1] : nGrid[0]*nGrid[1]*nGrid[2];
    if ( field.density ) assingZeroValues<Real>( density, gridSize );
    if ( field.velocity ) assingZeroValues< Pvector<Real,noVelComp> >( velocity, gridSize );
    if ( field.velocity_gradient ) assingZeroValues< Pvector<Real,noGradComp> >( velocity_gradient, gridSize );
    if ( field.scalar ) assingZeroValues< Pvector<Real,noScalarComp> >( scalar, gridSize );
    if ( field.scalar_gradient ) assingZeroValues< Pvector<Real,noScalarGradComp> >( scalar_gradient, gridSize );
    if ( field.velocity_std )
    {
        MESSAGE::Warning warning(userOptions.verboseLevel);
        warning << "You cannot use the averaging method 0 to compute the velocity standard deviation inside the grid cell. Please use the averaging method 2 '--method 2' (Monte Carlo sampling inside the grid cell) to compute the velocity standard deviation.\n" << MESSAGE::EndWarning;
    }
    // without cells there is nothing to interpolate
    if ( dt.number_of_vertices()<NO_DIM+1 )
    {
        MESSAGE::Warning warning(1);
        warning << "Because there are less than " << NO_DIM+1 << " vertices in the Delaunay triangulation there is no cell and hence there is no information that can be used to interpolate the fields to a grid. All field values will be initialized to 0.\n" << MESSAGE::EndWarning;
        return;
    }



    Real dx[NO_DIM];        // grid spacing along each axis
    for (size_t i=0; i<NO_DIM; ++i)
        dx[i] = boxLength[i] / nGrid[i];
    Real const gridCellVolume = (NO_DIM==2) ? (dx[0]*dx[1]) : (dx[0]*dx[1]*dx[2]);
#ifdef TEST_PADDING
    vector<size_t> dummyGridCells_d;
    vector<size_t> dummyGridCells;
#endif


    // geometric intersection helper (clips each simplex against the grid into sub-polytopes)
    VOLUME_SPLIT::VolumeSplit volSplit( &(startPos[0]), nGrid, dx );
    VOLUME_SPLIT::Simplex     simplex;      // triangle (2D) or tetrahedron (3D)
    vector<size_t>            indices;      // grid cell indices intersecting the simplex
    vector< Pvector<Real,NO_DIM+1> > contributions;    // per-cell volume + field moment of the simplex


    size_t prev = 0, amount100 = 0, count = 0;
#if NO_DIM==2
    size_t noTotalCells = dt.number_of_faces();
    for(Finite_cells_iterator itC = dt.finite_faces_begin(); itC!= dt.finite_faces_end(); ++itC)
#elif NO_DIM==3
    size_t noTotalCells = dt.number_of_finite_cells();
    for(Finite_cells_iterator itC = dt.finite_cells_begin(); itC!= dt.finite_cells_end(); ++itC)
#endif
    {
        amount100 = (100 * count++)/ noTotalCells;
        if (prev < amount100)
            message.updateProgress( ++prev );


        if ( cellOutsideRegion( dt, itC, fullBox) )
            continue;


        Real posMatrixInverse[NO_DIM][NO_DIM];  // inverse vertex position-difference matrix
        positionMatrix( itC, posMatrixInverse );
        Vertex_handle base = itC->vertex(0);


        // clip this simplex against the grid to get its per-cell volumes and field moments
        simplex.assign( itC );
        volSplit.findIntersection( simplex, indices, contributions );

        // field gradients are constant inside a Delaunay simplex
        Real densGrad[NO_DIM];
        densityGrad( itC, posMatrixInverse, densGrad );
        if (field.density) distributeContributions( indices, contributions, density, base->info().density(), densGrad, base );
#ifdef VELOCITY
        Real velGrad[NO_DIM][noVelComp];
        velocityGrad( itC, posMatrixInverse, velGrad );
        if (field.velocity) distributeContributions( indices, contributions, velocity, base->info().velocity(), velGrad, base );
        if (field.velocity_gradient) distributeContributions( indices, contributions, velocity_gradient, velocityGradient(velGrad) );
#endif
#ifdef SCALAR
        Real sGrad[NO_DIM][noScalarComp];
        scalarGrad( itC, posMatrixInverse, sGrad );
        if (field.scalar) distributeContributions( indices, contributions, scalar, base->info().myScalar(), sGrad, base );
        if (field.scalar_gradient) distributeContributions( indices, contributions, scalar_gradient, scalarGradient(sGrad) );
#endif
        

#ifdef TEST_PADDING
        bool dummyNeighbors = hasDummyNeighbor( itC );
        bool dummyVertices = hasDummyVertex( itC );
        if ( dummyNeighbors ) updateDummyGridCells( indices, &incompleteCells_d );
        if ( dummyVertices ) updateDummyGridCells( indices, &incompleteCells );
#endif
    }



    // divide by grid-cell volume to turn accumulated contributions into volume averages
    if (field.density)
        for (vector<Real>::iterator it=density->begin(); it!=density->end(); ++it)
            *it /= gridCellVolume;
#ifdef VELOCITY
    if (field.velocity)
        for (vector< Pvector<Real,noVelComp> >::iterator it=velocity->begin(); it!=velocity->end(); ++it)
            *it /= gridCellVolume;
    if (field.velocity_gradient)
        for (vector< Pvector<Real,noGradComp> >::iterator it=velocity_gradient->begin(); it!=velocity_gradient->end(); ++it)
            *it /= gridCellVolume;
#endif
#ifdef SCALAR
    if (field.scalar)
        for (vector< Pvector<Real,noScalarComp> >::iterator it=scalar->begin(); it!=scalar->end(); ++it)
            *it /= gridCellVolume;
    if (field.scalar_gradient)
        for (vector< Pvector<Real,noScalarGradComp> >::iterator it=scalar_gradient->begin(); it!=scalar_gradient->end(); ++it)
            *it /= gridCellVolume;
#endif
    
    message << "100%\n" << MESSAGE::Flush;
    
    
#ifdef TEST_PADDING
    if (field.density)
        showCellsContainingDummyPoints( &incompleteCells_d, userOptions, userOptions.outputFilename+"_density", "density" ); // report cells whose density may be wrong (incomplete tessellation)
    if ( field.velocity or field.velocity_gradient or field.scalar or field.scalar_gradient )
        showCellsContainingDummyPoints( &incompleteCells, userOptions, userOptions.outputFilename+"_fields", "fields" ); // report cells whose other fields may be wrong (incomplete tessellation)
#endif
}



