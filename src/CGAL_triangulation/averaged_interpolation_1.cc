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


/* Volume-averaged grid interpolation, method 1 (= sample points chosen inside the
   Delaunay cell, proportional to cell volume). */
#include "triangulation_common.h"
#include <typeinfo>
#include <stdio.h>
#include <gsl/gsl_qrng.h>



// Fills 'arraySize' quasi-random (Sobol) barycentric coordinates inside the unit simplex; samples in
// the unit cube are rejected unless their coordinates sum to <= 1, keeping them within the simplex.
void quasiRandomSequence(Real quasiRandomNumbers[][NO_DIM],
                         size_t const arraySize)
{
    gsl_qrng * q = gsl_qrng_alloc( gsl_qrng_sobol, NO_DIM );

    double randomRoot[NO_DIM];
    for (size_t i=0; i<arraySize; )
    {
        gsl_qrng_get (q, randomRoot);
        Real sum = NO_DIM==2 ? randomRoot[0]+randomRoot[1] : randomRoot[0]+randomRoot[1]+randomRoot[2];
        if ( sum<=1. )
        {
            for (int j=0; j<NO_DIM; ++j)
                quasiRandomNumbers[i][j] = Real( randomRoot[j] );
            ++i;
        }
    }
    gsl_qrng_free (q);
}
// Maps unit-simplex barycentric samples into the Delaunay cell via its vertex-difference matrix 'vM'
// (triangle in 2D, tetrahedron in 3D); writes the resulting Cartesian points to 'randomPoints'.
void quasiRandomPointsInCell(double vM[][NO_DIM],
                             size_t noRandomPoints,
                             Real qR[][NO_DIM],
                             Point *randomPoints)
{
#if NO_DIM==2
    for (size_t i=0; i<noRandomPoints; ++i)
        randomPoints[i] = Point( qR[i][0]*vM[0][0]+qR[i][1]*vM[1][0], qR[i][0]*vM[0][1]+qR[i][1]*vM[1][1] );
#elif NO_DIM==3
    for (size_t i=0; i<noRandomPoints; ++i)
        randomPoints[i] = Point( qR[i][0]*vM[0][0]+qR[i][1]*vM[1][0]+qR[i][2]*vM[2][0], qR[i][0]*vM[0][1]+qR[i][1]*vM[1][1]+qR[i][2]*vM[2][1], qR[i][0]*vM[0][2]+qR[i][1]*vM[1][2]+qR[i][2]*vM[2][2] );
#endif
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

static int const SINGLE_GRID_CELL = 0;
static int const MULTIPLE_GRID_CELLS = 1;
// Classifies whether a Delaunay cell fits in one grid cell or spans several, and fills 'vertexMatrix'
// with each non-base vertex's offset from the base vertex (used to map barycentric samples).
int checkCellPosition(Finite_cells_iterator &cell,
                      Box &regionBox,
                      Real *dx,
                      double vertexMatrix[][NO_DIM],
                      int *baseGridCell,
                      Real *basePosition)
{
    Point base = cell->vertex(0)->point();
    for (int i=0; i<NO_DIM; ++i)
    {
        basePosition[i] = base[i] - regionBox[2*i]; // base vertex position
        baseGridCell[i] = int( floor( (basePosition[i])/dx[i] ) ); // grid cell containing the base point
    }
    Box gridCellBox;    // bounds of the grid cell that contains 'base'
    for (int i=0; i<NO_DIM; ++i)
    {
        gridCellBox[2*i] = baseGridCell[i] * dx[i] + regionBox[2*i];
        gridCellBox[2*i+1] = gridCellBox[2*i] + dx[i];
    }
    bool insideOneCell = true;
    for (int i=1; i<NO_DIM+1; ++i)
        if ( not gridCellBox.isPointInBox( cell->vertex(i)->point() ) )
        {
            insideOneCell = false;
            break;
        }


    // store each non-base vertex position relative to 'base'
    for (int v = 0; v<NO_DIM; ++v)
        for (int i=0; i<NO_DIM; ++i)
            vertexMatrix[v][i] = double(cell->vertex(v+1)->point()[i]) - double(base[i]);

    if ( insideOneCell )
        return SINGLE_GRID_CELL;
    return MULTIPLE_GRID_CELLS;
}



// Returns true if 'gridCell' is in range, writing its flat array index to 'cellIndex'.
inline bool isGridCellIndex(int *gridCell,
                            size_t *totalGridSize,
                            size_t *cellIndex)
{
    for (int i=0; i<NO_DIM; ++i)
        if ( gridCell[i]<0 or gridCell[i]>=int(totalGridSize[i]) )
            return false;

    *cellIndex = gridCellIndex( gridCell, totalGridSize );
    return true;
}
// Locates the grid cell containing a sample point; returns true and writes 'cellIndex' if in range.
inline bool isGridCellIndex(size_t *totalGridSize,
                            Real *dx,
                            Point &samplePoint,
                            Real *basePosition,
                            size_t *cellIndex)
{
    int pos[NO_DIM];
    for (int i=0; i<NO_DIM; ++i)
    {
        pos[i] = int(floor( (basePosition[i] + samplePoint[i])/dx[i] ));
        if ( pos[i]<0 or pos[i]>=int(totalGridSize[i]) )
            return false;
    }

    *cellIndex = gridCellIndex( pos, totalGridSize );
    return true;
}



// Returns the density averaged over the cell's vertices (= its value at the cell centroid).
inline Real averageDensity(Finite_cells_iterator &cell)
{
    Real temp = 0.;
    for (int i=0; i<NO_DIM+1; ++i)
        temp += cell->vertex(i)->info().density();
    return temp / Real(NO_DIM+1);
}
// Returns the velocity averaged over the cell's vertices (= its value at the cell centroid).
inline Pvector<Real,noVelComp> averageVelocity(Finite_cells_iterator &cell)
{
    Pvector<Real,noVelComp> temp = Pvector<Real,noVelComp>::zero();
    for (int i=0; i<NO_DIM+1; ++i)
        temp += cell->vertex(i)->info().velocity();
    return temp / Real(NO_DIM+1);
}
// Returns the scalar averaged over the cell's vertices (= its value at the cell centroid).
inline Pvector<Real,noScalarComp> averageScalar(Finite_cells_iterator &cell)
{
    Pvector<Real,noScalarComp> temp = Pvector<Real,noScalarComp>::zero();
    for (int i=0; i<NO_DIM+1; ++i)
        temp += cell->vertex(i)->info().myScalar();
    return temp / Real(NO_DIM+1);
}





// Volume-averaged grid interpolation (method 1): quasi-Monte-Carlo sampling inside each Delaunay cell,
// with the sample count proportional to cell volume; scatters each sample's field value into its grid cell.
void interpolateGrid_averaged_1(DT &dt,
                                User_options &userOptions,
                                Quantities *quantities)
{
    size_t *nGrid = &(userOptions.gridSize[0]); // grid size along each axis
    size_t const NN = userOptions.noPoints;     // average random points per grid cell
    size_t const minNN = NN/6 + 1;              // minimum sample points per Delaunay cell
    Real const minRatio = 1./6.;                // vol(Delaunay)/vol(grid) below which minNN is used
    size_t const maxNN = 100*NN;                // maximum sample points per Delaunay cell
    Box boxCoordinates = userOptions.region;    // box region of interest
    vector<Real> boxLength;                     // box size along each direction
    for (size_t i=0; i<NO_DIM; ++i)
        boxLength.push_back( boxCoordinates[2*i+1]-boxCoordinates[2*i] );
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
            << ". The volume average is done using Monte Carlo sampling in each of the triangles/tetrahedra of the Delaunay triangulation. " 
            << "There are " << NN << " random sampling points on average in each grid cell of the output fields (the random samples in the Delaunay triangulation's triangles/tetrahedra are proportional to the area/volume of the cell).\n" 
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
        warning << "You cannot use the averaging method 1 (Monte Carlo sampling inside the Delaunay cell) to compute the velocity standard deviation inside the grid cell. Please use the averaging method 2 '--method 2' (Monte Carlo sampling inside the grid cell) to compute the velocity standard deviation.\n" << MESSAGE::EndWarning;
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
    vector<size_t> incompleteCells_d;// grid cells with a dummy-point error in the density field
    vector<size_t> incompleteCells;  // grid cells with a dummy-point error in the other fields
#endif


    // quasi-random (Sobol) sequence shared by all cells
    std::vector<Real> quasiRandomNumbers_buf(maxNN * NO_DIM);
    Real (*quasiRandomNumbers)[NO_DIM] = reinterpret_cast<Real(*)[NO_DIM]>(quasiRandomNumbers_buf.data());
    quasiRandomSequence( quasiRandomNumbers, maxNN );
    
    
    
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


#ifdef TEST_PADDING
        bool dummyNeighbors = hasDummyNeighbor( itC );
        bool dummyVertices = hasDummyVertex( itC );
#endif


        double vertexMatrix[NO_DIM][NO_DIM]; // vertex differences of the Delaunay cell
        int baseGridCell[NO_DIM];  // grid cell containing the base point
        Real basePosition[NO_DIM]; // base point position relative to the box's lower-left corner
        int cellPosition = checkCellPosition( itC, boxCoordinates, dx, vertexMatrix, baseGridCell, basePosition );

        Real cellVolume = volume( dt, itC );
        Real posMatrixInverse[NO_DIM][NO_DIM];  // inverse vertex position-difference matrix
        matrixInverse( vertexMatrix, posMatrixInverse );
        size_t index;

        // cell fits in one grid cell -> add its centroid value times volume, no MC sampling needed
#ifndef MY_SCALAR
        if ( cellPosition==SINGLE_GRID_CELL and isGridCellIndex( baseGridCell, nGrid, &index ) )
        {
            if (field.density) (*density)[index] += averageDensity(itC) * cellVolume;
#ifdef VELOCITY
            if (field.velocity) (*velocity)[index] += averageVelocity(itC) * cellVolume;
            if (field.velocity_gradient)
            {
                Real velGrad[NO_DIM][noVelComp];
                velocityGrad( itC, posMatrixInverse, velGrad );
                (*velocity_gradient)[index] += velocityGradient(velGrad) * cellVolume;
            }
#endif
#ifdef SCALAR
            if (field.scalar) (*scalar)[index] += averageScalar(itC) * cellVolume;
            if (field.scalar_gradient)
            {
                Real sGrad[NO_DIM][noScalarComp];
                scalarGrad( itC, posMatrixInverse, sGrad );
                (*scalar_gradient)[index] += scalarGradient(sGrad) * cellVolume;
            }
#endif
#ifdef TEST_PADDING
            if ( dummyNeighbors ) updateDummyGridCells( index, &incompleteCells_d );
            if ( dummyVertices ) updateDummyGridCells( index, &incompleteCells );
#endif
            continue;
        }
#endif
        
        
        // cell spans multiple grid cells -> MC-sample inside it
        size_t const tempInt = size_t(NN*cellVolume/gridCellVolume) + 1;
        size_t const noRandomPoints = (cellVolume/gridCellVolume>minRatio) ? (tempInt>maxNN ? maxNN:tempInt) : minNN;
        std::vector<Point> randomPoints(noRandomPoints);
        quasiRandomPointsInCell( vertexMatrix, noRandomPoints, quasiRandomNumbers, randomPoints.data() );
        Real factor = cellVolume / noRandomPoints;  // volume per sample point
        Vertex_handle base = itC->vertex(0);

        Real densGrad[NO_DIM];
        densityGrad( itC, posMatrixInverse, densGrad );
#ifdef VELOCITY
        Real velGrad[NO_DIM][noVelComp];
        velocityGrad( itC, posMatrixInverse, velGrad );
#endif
#ifdef SCALAR
        Real sGrad[NO_DIM][noScalarComp];
        scalarGrad( itC, posMatrixInverse, sGrad );
#endif
        for (size_t i=0; i<noRandomPoints; ++i) // scatter each sample point's field value into its grid cell
        {
            if ( not isGridCellIndex( nGrid, dx, randomPoints[i], basePosition, &index ) )
                continue;   // sample outside the region of interest
            
            if (field.density) (*density)[index] += densityValue(densGrad,base,randomPoints[i]) * factor;
#ifdef VELOCITY
            if (field.velocity) (*velocity)[index] += velocityValue(velGrad,base,randomPoints[i]) * factor;
            if (field.velocity_gradient) (*velocity_gradient)[index] += velocityGradient(velGrad) * factor;
#endif
#ifdef SCALAR
#ifdef MY_SCALAR
            (*scalar)[index] +=  customScalar(itC, posMatrixInverse, base, randomPoints[i]) * factor;
#else
            if (field.scalar) (*scalar)[index] += scalarValue(sGrad,base,randomPoints[i]) * factor;
            if (field.scalar_gradient) (*scalar_gradient)[index] += scalarGradient(sGrad) * factor;
#endif
#endif
#ifdef TEST_PADDING
            if ( dummyNeighbors ) updateDummyGridCells( index, &incompleteCells_d );
            if ( dummyVertices ) updateDummyGridCells( index, &incompleteCells );
#endif
        }
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



