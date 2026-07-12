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

// GPU deposit (METAL=1/CUDA=1/HIP=1 builds, 3D only); the CPU loop below remains the fallback. MY_SCALAR
// evaluates arbitrary user C++ per sample, which cannot run on the GPU.
#if defined(DTFE_GPU) && NO_DIM==3 && !defined(MY_SCALAR)
#define DTFE_GPU_ACTIVE
#include <cstdint>
#include <algorithm>
#include <utility>
#include "gpu_host.h"
#endif

// --exact-average: analytic cell∩tet moments (vendored r3d, Powell & Abel 2015; 3D only;
// MY_SCALAR is not linear inside a tet, so it keeps the MC loop -- rejected at parsing).
// r3d.h carries its own extern "C" guard.
#if NO_DIM==3 && !defined(MY_SCALAR)
#include "../../third_party/r3d/r3d.h"
#endif



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
// --exact-average replaces the MC inner loop by analytic cell∩tet integration (r3d; NO_DIM==3 only).
void interpolateGrid_averaged_1(DT &dt,
                                User_options &userOptions,
                                Quantities *quantities)
{
    size_t *nGrid = &(userOptions.gridSize[0]); // grid size along each axis
    size_t const NN = userOptions.noPoints;     // average random points per grid cell
    // --exact-average: analytic cell∩tet integration of the linear interpolant (3D CPU only)
    bool const exactAvg = userOptions.exactAverage;
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
    if ( exactAvg )
        message << "\nComputing interpolation of the fields volume averaged over the sampling cell on a regular "
                << MESSAGE::printElements( nGrid, NO_DIM, "*" ) << " grid in the region "
                << boxCoordinates.print()
                << ". The volume average integrates the linear interpolant EXACTLY over every grid-cell/tetrahedron intersection (--exact-average, r3d; '--samples' is ignored).\n"
                << "\t Done: " << MESSAGE::Flush;
    else
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


    // GPU deposit (--metal / --gpu, GPU builds): extract flat per-tetrahedron arrays using the CPU
    // loop's own helpers (classification, sample counts, volumes -- so both paths make identical
    // decisions), dispatch metal/dtfe_deposit.metal::depositAveraged1, and copy the grids back.
    // Any failure (no device, kernel compile, buffer alloc) falls back to the CPU loop below.
    bool metalDeposited = false;
#ifdef DTFE_GPU_ACTIVE
    bool tryMetal = userOptions.useMetal;
    if ( tryMetal and exactAvg )
    {
        MESSAGE::Warning warning( userOptions.verboseLevel );
        warning << "--exact-average is CPU-only (analytic r3d integration); using the CPU interpolation for this pass.\n" << MESSAGE::EndWarning;
        tryMetal = false;
    }
#ifdef SCALAR
    if ( tryMetal and (field.scalar or field.scalar_gradient) )
    {
        MESSAGE::Warning warning( userOptions.verboseLevel );
        warning << "--metal does not support the scalar fields; using the CPU interpolation for this pass.\n" << MESSAGE::EndWarning;
        tryMetal = false;
    }
#endif
    // the single-cell fast path ships 32-bit flat grid indices to the GPU
    if ( tryMetal and gridSize > size_t(UINT32_MAX) )
    {
        MESSAGE::Warning warning( userOptions.verboseLevel );
        warning << "--metal does not support more than 2^32 grid cells; using the CPU interpolation.\n" << MESSAGE::EndWarning;
        tryMetal = false;
    }
    if ( tryMetal )
    {
        bool const fDen = field.density;
        bool fVel = false, fGrad = false;
#ifdef VELOCITY
        fVel  = field.velocity;
        fGrad = field.velocity_gradient;
#endif
        bool const needVel = fVel or fGrad;

        // reserve for all finite cells (upper bound; a counting pre-pass would cost a full extra
        // CGAL cell walk, which measured comparably to the whole extraction -- the padded fraction
        // outside the region is small for standard-DTFE partitions, unlike the PS Lagrangian ones)
        size_t const nKept = dt.number_of_finite_cells();

        std::vector<float>    tetVerts, tetDens, tetVels, tetVols;
        std::vector<uint32_t> tetCnts, tetFlats;
        tetVerts.reserve( nKept*12 );
        tetDens.reserve( nKept*4 );
        if ( needVel ) tetVels.reserve( nKept*12 );
        tetVols.reserve( nKept );
        tetCnts.reserve( nKept );
        tetFlats.reserve( nKept );

        for (Finite_cells_iterator itC = dt.finite_cells_begin(); itC != dt.finite_cells_end(); ++itC)
        {
            if ( cellOutsideRegion(dt, itC, fullBox) )
                continue;

            double vertexMatrix[NO_DIM][NO_DIM];
            int baseGridCell[NO_DIM];
            Real basePosition[NO_DIM];
            int cellPosition = checkCellPosition( itC, boxCoordinates, dx, vertexMatrix, baseGridCell, basePosition );
            Real cellVolume = volume( dt, itC );

            size_t index;
            uint32_t cnt = 0, flat = 0;
            if ( cellPosition==SINGLE_GRID_CELL and isGridCellIndex( baseGridCell, nGrid, &index ) )
                flat = uint32_t(index);   // cnt 0 = deposit the centroid value at 'flat'
            else
            {
                size_t const tempInt = size_t(NN*cellVolume/gridCellVolume) + 1;
                cnt = uint32_t( (cellVolume/gridCellVolume>minRatio) ? (tempInt>maxNN ? maxNN:tempInt) : minNN );
            }

            // vertex positions relative to the region's lower corner (subtracted in double, so
            // the narrowed floats match the CPU's Real basePosition exactly)
            for (int v=0; v<NO_DIM+1; ++v)
                for (int d=0; d<NO_DIM; ++d)
                    tetVerts.push_back( float( double(itC->vertex(v)->point()[d]) - double(boxCoordinates[2*d]) ) );
            for (int v=0; v<NO_DIM+1; ++v)
                tetDens.push_back( float(itC->vertex(v)->info().density()) );
#ifdef VELOCITY
            if ( needVel )
                for (int v=0; v<NO_DIM+1; ++v)
                    for (size_t j=0; j<noVelComp; ++j)
                        tetVels.push_back( float(itC->vertex(v)->info().velocity(j)) );
#endif
            tetVols.push_back( float(cellVolume) );
            tetCnts.push_back( cnt );
            tetFlats.push_back( flat );

#ifdef TEST_PADDING
            // the GPU deposits this tetrahedron's field values; replicate only the dummy-cell
            // REPORTING here (dummy tetrahedra live near the padded boundary, so this is cheap)
            bool dummyNeighbors = hasDummyNeighbor( itC );
            bool dummyVertices = hasDummyVertex( itC );
            if ( dummyNeighbors or dummyVertices )
            {
                if ( cnt==0 )
                {
                    if ( dummyNeighbors ) updateDummyGridCells( size_t(flat), &incompleteCells_d );
                    if ( dummyVertices ) updateDummyGridCells( size_t(flat), &incompleteCells );
                }
                else
                {
                    std::vector<Point> randomPoints(cnt);
                    quasiRandomPointsInCell( vertexMatrix, cnt, quasiRandomNumbers, randomPoints.data() );
                    for (size_t i=0; i<cnt; ++i)
                        if ( isGridCellIndex( nGrid, dx, randomPoints[i], basePosition, &index ) )
                        {
                            if ( dummyNeighbors ) updateDummyGridCells( index, &incompleteCells_d );
                            if ( dummyVertices ) updateDummyGridCells( index, &incompleteCells );
                        }
                }
            }
#endif
        }

        // Sort tetrahedra by sample count before dispatch. A GPU chunk finishes when its SLOWEST
        // thread finishes: one huge void tet (thousands of samples) mixed among single-cell halo
        // tets leaves most of the GPU idle until the chunk boundary (measured ~5x underutilization
        // for the PS deposit). Cost-ordering makes every chunk's threads uniformly sized; the
        // permutation is applied one array at a time so only a single extra copy is alive at once.
        {
            size_t const nT = tetCnts.size();
            std::vector< std::pair<uint32_t,uint32_t> > order(nT);
            for (size_t t=0; t<nT; ++t)
                order[t] = std::make_pair( tetCnts[t], uint32_t(t) );
            std::sort( order.begin(), order.end() );
            {
                std::vector<float> sv(nT*12);
                for (size_t i=0; i<nT; ++i)
                { size_t const src = order[i].second; for (int k=0; k<12; ++k) sv[i*12+k] = tetVerts[src*12+k]; }
                tetVerts.swap(sv);
            }
            {
                std::vector<float> sd(nT*4);
                for (size_t i=0; i<nT; ++i)
                { size_t const src = order[i].second; for (int k=0; k<4; ++k) sd[i*4+k] = tetDens[src*4+k]; }
                tetDens.swap(sd);
            }
            if ( needVel )
            {
                std::vector<float> su(nT*12);
                for (size_t i=0; i<nT; ++i)
                { size_t const src = order[i].second; for (int k=0; k<12; ++k) su[i*12+k] = tetVels[src*12+k]; }
                tetVels.swap(su);
            }
            {
                std::vector<float> so(nT);
                for (size_t i=0; i<nT; ++i) so[i] = tetVols[order[i].second];
                tetVols.swap(so);
            }
            {
                std::vector<uint32_t> sc(nT);
                for (size_t i=0; i<nT; ++i) sc[i] = tetCnts[order[i].second];
                tetCnts.swap(sc);
            }
            {
                std::vector<uint32_t> sf(nT);
                for (size_t i=0; i<nT; ++i) sf[i] = tetFlats[order[i].second];
                tetFlats.swap(sf);
            }
        }

        std::vector<float> sobolTable( quasiRandomNumbers_buf.begin(), quasiRandomNumbers_buf.end() );
        double dxv[3] = { double(dx[0]), double(dx[1]), double(dx[2]) };
        size_t const nTetsDeposited = tetCnts.size();   // the deposit call consumes (frees) the arrays
        DTFEGpuGrids gpuOut;
        std::string metalErr;
        message << gpuBackendName() << " GPU (" << gpuDeviceName() << ", " << nTetsDeposited << " tetrahedra) ... " << MESSAGE::Flush;
        if ( dtfeGpuDepositAveraged1( tetVerts, tetDens, tetVels, tetVols, tetCnts, tetFlats,
                                        sobolTable, dxv, nGrid, fDen, fVel, fGrad, gpuOut, metalErr ) )
        {
            for (size_t i=0; i<gridSize; ++i)
            {
                if (fDen) (*density)[i] = Real(gpuOut.density[i]);
#ifdef VELOCITY
                if (fVel)
                    for (size_t j=0; j<noVelComp; ++j)
                        (*velocity)[i][j] = Real(gpuOut.velocity[i*3+j]);
                if (fGrad)
                    for (size_t q=0; q<noGradComp; ++q)
                        (*velocity_gradient)[i][q] = Real(gpuOut.grad[i*9+q]);
#endif
            }
            metalDeposited = true;
        }
        else
        {
            MESSAGE::Warning warning( userOptions.verboseLevel );
            warning << "DTFE Metal deposit unavailable (" << metalErr << "); using the CPU interpolation.\n" << MESSAGE::EndWarning;
        }
    }
#else
    if ( userOptions.useMetal )
    {
        static bool warned = false;
        if ( !warned )
        {
            warned = true;
            MESSAGE::Warning warning( userOptions.verboseLevel );
            warning << "--gpu/--metal requested but this binary was built without GPU support (rebuild with METAL=1, CUDA=1 or HIP=1; or NO_DIM!=3 / MY_SCALAR); using the CPU interpolation.\n" << MESSAGE::EndWarning;
        }
    }
#endif // DTFE_GPU_ACTIVE


    if ( not metalDeposited )
    {
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


#if NO_DIM==3 && !defined(MY_SCALAR)
        // ----- exact volume average (--exact-average): analytic cell∩tet integration -----
        // The interpolant is LINEAR inside the tet, so its integral over each intersection
        // piece is the centroid value times the piece volume -- order-1 r3d moments suffice
        // (Powell & Abel 2015). Same vertex-0-relative frame as the sampled path (densGrad,
        // velGrad and the field evaluators all take base-relative coordinates); out-of-region
        // window cells are skipped exactly like out-of-region samples. The single-grid-cell
        // fast path above (already exact for linear fields) is shared with the sampled mode.
        if ( exactAvg )
        {
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

            r3d_rvec3 tv[NO_DIM+1];
            for (int d = 0; d < NO_DIM; ++d) tv[0].xyz[d] = 0.;
            for (int v = 1; v <= NO_DIM; ++v)
                for (int d = 0; d < NO_DIM; ++d)
                    tv[v].xyz[d] = vertexMatrix[v-1][d];
            if ( determinant(vertexMatrix) < 0. )   // r3d wants positive orientation
            { r3d_rvec3 const tmp = tv[1]; tv[1] = tv[2]; tv[2] = tmp; }
            r3d_poly tetPoly;
            r3d_init_tet(&tetPoly, tv);

            // window of grid cells overlapped by the tet bbox (region-frame indices, clamped)
            int iLo[NO_DIM], iHi[NO_DIM];
            for (int d = 0; d < NO_DIM; ++d)
            {
                double lo = 0., hi = 0.;
                for (int v = 0; v < NO_DIM; ++v)
                {
                    if (vertexMatrix[v][d] < lo) lo = vertexMatrix[v][d];
                    if (vertexMatrix[v][d] > hi) hi = vertexMatrix[v][d];
                }
                iLo[d] = int( std::floor( (double(basePosition[d]) + lo) / double(dx[d]) ) );
                iHi[d] = int( std::floor( (double(basePosition[d]) + hi) / double(dx[d]) ) ) + 1;
                if (iLo[d] < 0) iLo[d] = 0;
                if (iHi[d] > int(nGrid[d])) iHi[d] = int(nGrid[d]);
            }

            for (int gi = iLo[0]; gi < iHi[0]; ++gi)
            for (int gj = iLo[1]; gj < iHi[1]; ++gj)
            for (int gk = iLo[2]; gk < iHi[2]; ++gk)
            {
                int const gijk[3] = {gi, gj, gk};
                r3d_plane planes[6];   // the 6 half-spaces of this cell, n.x + d >= 0 kept
                for (int d = 0; d < NO_DIM; ++d)
                {
                    double const lo = gijk[d] * double(dx[d]) - double(basePosition[d]);
                    double const hi = lo + double(dx[d]);
                    for (int q = 0; q < 2; ++q)
                        for (int dd = 0; dd < NO_DIM; ++dd)
                            planes[2*d+q].n.xyz[dd] = 0.;
                    planes[2*d].n.xyz[d]   =  1.; planes[2*d].d   = -lo;
                    planes[2*d+1].n.xyz[d] = -1.; planes[2*d+1].d =  hi;
                }
                r3d_poly piece = tetPoly;
                r3d_clip(&piece, planes, 6);
                if ( piece.nverts == 0 ) continue;
                r3d_real mom[4];
                r3d_reduce(&piece, mom, 1);
                if ( !(mom[0] > 0.) ) continue;

                index = ( size_t(gi)*nGrid[1] + size_t(gj) )*nGrid[2] + size_t(gk);
                Real const dV = Real(mom[0]);
                Point centroid( mom[1]/mom[0], mom[2]/mom[0], mom[3]/mom[0] );
                if (field.density) (*density)[index] += densityValue(densGrad, base, centroid) * dV;
#ifdef VELOCITY
                if (field.velocity) (*velocity)[index] += velocityValue(velGrad, base, centroid) * dV;
                if (field.velocity_gradient) (*velocity_gradient)[index] += velocityGradient(velGrad) * dV;
#endif
#ifdef SCALAR
                if (field.scalar) (*scalar)[index] += scalarValue(sGrad, base, centroid) * dV;
                if (field.scalar_gradient) (*scalar_gradient)[index] += scalarGradient(sGrad) * dV;
#endif
#ifdef TEST_PADDING
                if ( dummyNeighbors ) updateDummyGridCells( index, &incompleteCells_d );
                if ( dummyVertices ) updateDummyGridCells( index, &incompleteCells );
#endif
            }
            continue;
        }
#endif  // NO_DIM==3 (exact average)


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
    }   // end of the CPU interpolation (skipped when the Metal deposit succeeded)



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



