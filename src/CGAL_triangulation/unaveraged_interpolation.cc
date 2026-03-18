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



/* This file contains the functions that interpolate the fields to the sampling points.

NOTE: This file does NOT compute the average of the fields inside the cell associated to each sampling point.
*/

#include "triangulation_common.h"




/* This function interpolates the fields to a uniform grid by computing the field values at the sample points.
The arguments are:
    'dt' - the periodic Delaunay triangulation; with the density at the vertex points already computed
    'userOptions' - structure which store the number of grid points along each axis and also gives information about the size of the box along each axis. It also contains information about what quantity/qunatities to compute.
    'quantities' - vectors storing the interpolated grid fields

NOTE: this does NOT take volume average the fields inside the sampling cell.
*/


/* This function interpolates the fields to a uniform grid by computing the field values at the sample points.
The arguments are:
    'dt' - the periodic Delaunay triangulation; with the density at the vertex points already computed
    'userOptions' - structure which store the number of grid points along each axis and also gives information about the size of the box along each axis. It also contains information about what quantity/qunatities to compute. 
    'quantities' - vectors storing the interpolated grid fields

NOTE: this does NOT take volume average the fields inside the sampling cell.
*/
void interpolateGrid(DT &dt,
                     User_options &userOptions,
                     Quantities *quantities)
{
    // define some variables for easy access of program wide variables
    size_t *nGrid = &(userOptions.gridSize[0]); // size of the density grid along each axis
    Box boxCoordinates = userOptions.region;    // the box coordinates in which the density is computed
    vector<Real> boxLength;                     // the size of the particle box of interest along each direction
    for (size_t i=0; i<NO_DIM; ++i)
        boxLength.push_back( boxCoordinates[2*i+1]-boxCoordinates[2*i] );
    
    // define pointers to the vectors storing the output fields
    Field field = userOptions.uField;
    std::vector<Real>                         *density = &(quantities->density);
    std::vector< Pvector<Real,noVelComp> >    *velocity = &(quantities->velocity);
    std::vector< Pvector<Real,noGradComp> >   *velocity_gradient = &(quantities->velocity_gradient);
    std::vector< Pvector<Real,noScalarComp> > *scalar = &(quantities->scalar);
    std::vector< Pvector<Real,noScalarGradComp> > *scalar_gradient = &(quantities->scalar_gradient);
    
    
    // show a message to the user
    MESSAGE::Message message( userOptions.verboseLevel );   // structure used to output messages to the user ( the higher the verboseLevel, the more messages will be showns (check 'message.h' for details) )
    message << "Computing interpolation of the fields at the sampling points given by a regular " << 
            MESSAGE::printElements( nGrid, NO_DIM, "*" ) << " grid in the region " << 
            boxCoordinates.print() << ".\n"
            << "\t Done:   " << MESSAGE::Flush;
    
    
    //reserve memory for the output fields
    size_t reserveSize = 1;                     // the number of cell in the grid
    for (size_t i=0; i<NO_DIM; ++i)
        reserveSize *= nGrid[i];
    reserveMemory( density, reserveSize, field.density, dt, "density" );    // reserves memory for the density computations (if any) and initializes the density values to 0 if the Delaunay triangulation is not well defined
    reserveMemory( velocity, reserveSize, field.velocity, dt, "velocity" );
    reserveMemory( velocity_gradient, reserveSize, field.velocity_gradient, dt, "velocity gradient" );
    reserveMemory( scalar, reserveSize, field.scalar, dt, "scalar" );
    reserveMemory( scalar_gradient, reserveSize, field.scalar_gradient, dt, "scalar gradient" );
    if ( dt.number_of_vertices()<NO_DIM+1 ) return;     // stop the computation if the Delaunay triangulation is not well defined
    
    
    
    // temporary variables needed for the interpolation
    Point samplePoint;      // the coordinates of the sampling point
    Real dx[NO_DIM];        // grid spacing along each axis
    for (size_t i=0; i<NO_DIM; ++i)
        dx[i] = boxLength[i] / nGrid[i];
    
#ifdef TEST_PADDING
    vector<size_t> incompleteCells_d;// keep track of grid cells where there is an error in the density field computation (because one of the vertices is a dummy point)
    vector<size_t> incompleteCells;  // keep track of grid cells where there is an error in the field (all fields except density) computation (because one of the vertices is a dummy point)
#endif
    int prev = 0, amount100 = 0;    // variable to show the user about the progress of the computation
    
    
    // parameters for searching in the triangulation
    Locate_type lt;
    int li, lj;
    Cell_handle current;
    
    
    
    Real x = boxCoordinates[0] + dx[0]/2.;
    for (size_t i=0; i<nGrid[0]; ++i)
    {
        // print information to the user about the ongoing computation
        amount100 = (100 * i)/ nGrid[0];
        if (prev < amount100)
            message.updateProgress( ++prev );
        
        Real y = boxCoordinates[2] + dx[1]/2.;
        for (size_t j=0; j<nGrid[1]; ++j)
        {
#if NO_DIM==2
            samplePoint = Point( x, y );
            current = dt.locate( samplePoint, lt, li, current );
#elif NO_DIM==3
            Real z = boxCoordinates[4] + dx[2]/2.;
            for (size_t k=0; k<nGrid[2]; ++k)
            {
                samplePoint = Point( x, y, z );
                current = dt.locate( samplePoint, lt, li, lj, current );
#endif
                
                // check if the cell has a vertex which is at infinity
                if ( dt.is_infinite(current) )
                {
                    if (field.density) density->push_back( Real(0.) );
#ifdef VELOCITY
                    if (field.velocity) velocity->push_back( Pvector<Real,noVelComp>::zero() );
                    if (field.velocity_gradient) velocity_gradient->push_back( Pvector<Real,noGradComp>::zero() );
#endif
#ifdef SCALAR
                    if (field.scalar) scalar->push_back( Pvector<Real,noScalarComp>::zero() );
                    if (field.scalar_gradient) scalar_gradient->push_back( Pvector<Real,noScalarGradComp>::zero() );
#endif
                    continue;
                }
                
                Real posMatrixInverse[NO_DIM][NO_DIM];  //stores the inverse of the vertex position difference matrix
                positionMatrix( current, posMatrixInverse );
                Vertex_handle base = current->vertex(0);  // stores the "base" vertex of the Delaunay cell
                samplePoint = relativeSamplePoint( base, samplePoint ); // now sample point stores the relative position of the sample point with respect to the base vertex
                
                // test for the completness of the padding if required
#ifdef TEST_PADDING
#if NO_DIM==2
                size_t k = 0;
#endif
                Real index = gridCellIndex( i,j,k, nGrid );
                if ( hasDummyNeighbor( current ) ) updateDummyGridCells( index, &incompleteCells_d ); // if one of the vertices is or has dummy neighbors, keep track of this cell for the density computation part
                if ( hasDummyVertex( current ) ) updateDummyGridCells( index, &incompleteCells );     // if one of the vertices is a dummy point, keep track of this grid cell for velocity/scalar field computations
#endif
                
                // now compute the field values at the sample points
                if (field.density)
                {
                    Real densGrad[NO_DIM];
                    densityGrad(current, posMatrixInverse, densGrad);
                    density->push_back( densityValue(densGrad,base,samplePoint) );
                }
#ifdef VELOCITY
                if (field.velocity or field.velocity_gradient)
                {
                    Real velGrad[NO_DIM][noVelComp];
                    velocityGrad( current, posMatrixInverse, velGrad );
                    if (field.velocity) velocity->push_back( velocityValue(velGrad,base,samplePoint) );
                    if (field.velocity_gradient) velocity_gradient->push_back( velocityGradient(velGrad) );
                }
#endif
#ifdef SCALAR
                if (field.scalar or field.scalar_gradient)
                {
#ifdef MY_SCALAR
                    scalar->push_back( customScalar(current, posMatrixInverse, base, samplePoint) );
#else
                    Real sGrad[NO_DIM][noScalarComp];
                    scalarGrad( current, posMatrixInverse, sGrad );
                    if (field.scalar) scalar->push_back( scalarValue(sGrad,base,samplePoint) );
                    if (field.scalar_gradient) scalar_gradient->push_back( scalarGradient(sGrad) );
#endif
                }
#endif
                
#if NO_DIM==3     // end additional z-loop
                z += dx[2];
            }
#endif
            y += dx[1];
        }
        x += dx[0];
    }
    message << "100%\n" << MESSAGE::Flush;
    
    
#ifdef TEST_PADDING
    if (field.density)
        showCellsContainingDummyPoints( &incompleteCells_d, userOptions, userOptions.outputFilename+"_density", "density" ); // check if there are any cells which may have an error in thedensity estimation due to an incomplete Delaunay tesselation over the region of interest
    if ( field.velocity or field.velocity_gradient or field.scalar or field.scalar_gradient )
        showCellsContainingDummyPoints( &incompleteCells, userOptions, userOptions.outputFilename+"_fields", "fields" ); // check if there are any cells which may have an error in the fields estimation (all fields except density) due to an incomplete Delaunay tesselation over the region of interest
#endif
}



/* This function interpolates the fields to a redshift cone grid (usefull for galaxy surveys) by computing the field values at the sample points.
The arguments are:
    'dt' - the periodic Delaunay triangulation; with the density at the vertex points already computed
    'userOptions' - structure which store the number of grid points along each axis and also gives information about the size of the box along each axis. It also contains information about what quantity/qunatities to compute. 
    'quantities' - vectors storing the interpolated grid fields

NOTE: this does NOT take volume average the fields inside the sampling cell.
*/
void interpolateRedshiftCone(DT &dt,
                             User_options &userOptions,
                             Quantities *quantities)
{
    // define some variables for easy access of program wide variables
    size_t *nGrid = &(userOptions.gridSize[0]); // size of the density grid along each axis
    Box boxCoordinates = userOptions.redshiftCone;    // the box coordinates in which the density is computed
    checkAngles( &(boxCoordinates[2]), (NO_DIM-1)*2 );// transform the angles from degrees to radians
    Real *origin = &(userOptions.originPosition[0]);
    vector<Real> boxLength;                     // the size of the particle box of interest along each direction
    for (size_t i=0; i<NO_DIM; ++i)
        boxLength.push_back( boxCoordinates[2*i+1]-boxCoordinates[2*i] );
    
    // define pointers to the vectors storing the output fields
    Field field = userOptions.uField;
    std::vector<Real>                         *density = &(quantities->density);
    std::vector< Pvector<Real,noVelComp> >    *velocity = &(quantities->velocity);
    std::vector< Pvector<Real,noGradComp> >   *velocity_gradient = &(quantities->velocity_gradient);
    std::vector< Pvector<Real,noScalarComp> > *scalar = &(quantities->scalar);
    std::vector< Pvector<Real,noScalarGradComp> > *scalar_gradient = &(quantities->scalar_gradient);
    
    
    // show a message to the user
    MESSAGE::Message message( userOptions.verboseLevel );   // structure used to output messages to the user ( the higher the verboseLevel, the more messages will be showns (check 'message.h' for details) )
    message << "Computing interpolation of the fields at the sampling points given by a regular redshift cone grid with " << (NO_DIM==2 ? "r*psi": "r*theta*psi") << " = " << 
            MESSAGE::printElements( nGrid, NO_DIM, "*" ) << " bins in the region " << (NO_DIM==2 ? "{[r_min,r_max],[psi_min,psi_max]}" : "{[r_min,r_max],[theta_min,theta_max],[psi_min,psi_max]}") << " = " << userOptions.redshiftCone.print() << ".\n"
            << "\t Done:   " << MESSAGE::Flush;
    
    
    //reserve memory for the output fields
    size_t reserveSize = 1;                     // the number of cell in the grid
    for (size_t i=0; i<NO_DIM; ++i)
        reserveSize *= nGrid[i];
    reserveMemory( density, reserveSize, field.density, dt, "density" );    // reserves memory for the density computations (if any) and initializes the density values to 0 if the Delaunay triangulation is not well defined
    reserveMemory( velocity, reserveSize, field.velocity, dt, "velocity" );
    reserveMemory( velocity_gradient, reserveSize, field.velocity_gradient, dt, "velocity gradient" );
    reserveMemory( scalar, reserveSize, field.scalar, dt, "scalar" );
    reserveMemory( scalar_gradient, reserveSize, field.scalar_gradient, dt, "scalar gradient" );
    if ( dt.number_of_vertices()<NO_DIM+1 ) return;     // stop the computation if the Delaunay triangulation is not well defined
    
    
    
    // temporary variables needed for the interpolation
    Point samplePoint;      // the coordinates of the sampling point
    Real dx[NO_DIM];        // grid spacing along each axis
    for (size_t i=0; i<NO_DIM; ++i)
        dx[i] = boxLength[i] / nGrid[i];
    
#ifdef TEST_PADDING
    vector<size_t> incompleteCells_d;// keep track of grid cells where there is an error in the density field computation (because one of the vertices is a dummy point)
    vector<size_t> incompleteCells;  // keep track of grid cells where there is an error in the field (all fields except density) computation (because one of the vertices is a dummy point)
#endif
    int prev = 0, amount100 = 0;    // variable to show the user about the progress of the computation
    
    
    // parameters for searching in the triangulation
    Locate_type lt;
    int li, lj;
    Cell_handle current;
    
    
    
    Real r = boxCoordinates[0] + dx[0]/2.;
    for (size_t i=0; i<nGrid[0]; ++i)
    {
        // print information to the user about the ongoing computation
        amount100 = (100 * i)/ nGrid[0];
        if (prev < amount100)
            message.updateProgress( ++prev );
        
        Real theta = boxCoordinates[2] + dx[1]/2.;
        for (size_t j=0; j<nGrid[1]; ++j)
        {
#if NO_DIM==2
            Real tempX = origin[0] + r*cos(theta);
            Real tempY = origin[1] + r*sin(theta);
            samplePoint = Point(tempX,tempY);
            current = dt.locate( samplePoint, lt, li, current );
#elif NO_DIM==3
            Real psi = boxCoordinates[4] + dx[2]/2.;
            for (size_t k=0; k<nGrid[2]; ++k)
            {
                Real tempX = origin[0] + r*sin(theta)*cos(psi);
                Real tempY = origin[1] + r*sin(theta)*sin(psi);
                Real tempZ = origin[2] + r*cos(theta);
                samplePoint = Point(tempX,tempY,tempZ);
                current = dt.locate( samplePoint, lt, li, lj, current );
#endif
                
                // check if the cell has a vertex which is at infinity
                if ( dt.is_infinite(current) )
                {
                    if (field.density) density->push_back( Real(0.) );
#ifdef VELOCITY
                    if (field.velocity) velocity->push_back( Pvector<Real,noVelComp>::zero() );
                    if (field.velocity_gradient) velocity_gradient->push_back( Pvector<Real,noGradComp>::zero() );
#endif
#ifdef SCALAR
                    if (field.scalar) scalar->push_back( Pvector<Real,noScalarComp>::zero() );
                    if (field.scalar_gradient) scalar_gradient->push_back( Pvector<Real,noScalarGradComp>::zero() );
#endif
                    continue;
                }
                
                Real posMatrixInverse[NO_DIM][NO_DIM];  //stores the inverse of the vertex position difference matrix
                positionMatrix( current, posMatrixInverse );
                Vertex_handle base = current->vertex(0);  // stores the "base" vertex of the Delaunay cell
                samplePoint = relativeSamplePoint( base, samplePoint ); // now sample point stores the relative position of the sample point with respect to the base vertex
                
                // test for the completness of the padding if required
#ifdef TEST_PADDING
#if NO_DIM==2
                size_t k = 0;
#endif
                Real index = gridCellIndex( i,j,k, nGrid );
                if ( hasDummyNeighbor( current ) ) updateDummyGridCells( index, &incompleteCells_d ); // if one of the vertices is or has dummy neighbors, keep track of this cell for the density computation part
                if ( hasDummyVertex( current ) ) updateDummyGridCells( index, &incompleteCells );     // if one of the vertices is a dummy point, keep track of this grid cell for velocity/scalar field computations
#endif
                
                // now compute the field values at the sample points
                if (field.density)
                {
                    Real densGrad[NO_DIM];
                    densityGrad(current, posMatrixInverse, densGrad);
                    density->push_back( densityValue(densGrad,base,samplePoint) );
                }
#ifdef VELOCITY
                if (field.velocity or field.velocity_gradient)
                {
                    Real velGrad[NO_DIM][noVelComp];
                    velocityGrad( current, posMatrixInverse, velGrad );
                    if (field.velocity) velocity->push_back( velocityValue(velGrad,base,samplePoint) );
                    if (field.velocity_gradient) velocity_gradient->push_back( velocityGradient(velGrad) );
                }
#endif
#ifdef SCALAR
                if (field.scalar or field.scalar_gradient)
                {
#ifdef MY_SCALAR
                    scalar->push_back( customScalar(current, posMatrixInverse, base, samplePoint) );
#else
                    Real sGrad[NO_DIM][noScalarComp];
                    scalarGrad( current, posMatrixInverse, sGrad );
                    if (field.scalar) scalar->push_back( scalarValue(sGrad,base,samplePoint) );
                    if (field.scalar_gradient) scalar_gradient->push_back( scalarGradient(sGrad) );
#endif
                }
#endif
                
#if NO_DIM==3     // end additional z-loop
                psi += dx[2];
            }
#endif
            theta += dx[1];
        }
        r += dx[0];
    }
    message << "100%\n" << MESSAGE::Flush;
    
    
#ifdef TEST_PADDING
    if (field.density)
        showCellsContainingDummyPoints( &incompleteCells_d, userOptions, userOptions.outputFilename+"_density", "density" ); // check if there are any cells which may have an error in thedensity estimation due to an incomplete Delaunay tesselation over the region of interest
    if ( field.velocity or field.velocity_gradient or field.scalar or field.scalar_gradient )
        showCellsContainingDummyPoints( &incompleteCells, userOptions, userOptions.outputFilename+"_fields", "fields" ); // check if there are any cells which may have an error in the fields estimation (all fields except density) due to an incomplete Delaunay tesselation over the region of interest
#endif
}




/* This function interpolates the fields at user defined sampling points by computing the field values at the sample points.
The arguments are:
    'dt' - the periodic Delaunay triangulation; with the density at the vertex points already computed
    'userOptions' - structure which store the number of grid points along each axis and also gives information about the size of the box along each axis. It also contains information about what quantity/qunatities to compute. 
    'quantities' - vectors storing the interpolated fields

NOTE: this does NOT take volume average the fields inside the sampling cell.
*/
void interpolateUserSampling(DT &dt,
                             vector<Sample_point> &samples,
                             User_options &userOptions,
                             Quantities *quantities)
{
    // define pointers to the vectors storing the output fields
    Field field = userOptions.uField;
    std::vector<Real>                         *density = &(quantities->density);
    std::vector< Pvector<Real,noVelComp> >    *velocity = &(quantities->velocity);
    std::vector< Pvector<Real,noGradComp> >   *velocity_gradient = &(quantities->velocity_gradient);
    std::vector< Pvector<Real,noScalarComp> > *scalar = &(quantities->scalar);
    std::vector< Pvector<Real,noScalarGradComp> > *scalar_gradient = &(quantities->scalar_gradient);
    
    
    // show a message to the user
    MESSAGE::Message message( userOptions.verboseLevel );   // structure used to output messages to the user ( the higher the verboseLevel, the more messages will be showns (check 'message.h' for details) )
    message << "Computing interpolation of the fields at user given sampling points in the region " << userOptions.region.print() << ".\n"
            << "\t Done:   " << MESSAGE::Flush;
    
    
    //reserve size for the output fields
    size_t reserveSize = samples.size();                   // the number of cell in the grid
    reserveMemory( density, reserveSize, field.density, dt, "density" );    // reserves memory for the density computations (if any) and initializes the density values to 0 if the Delaunay triangulation is not well defined
    reserveMemory( velocity, reserveSize, field.velocity, dt, "velocity" );
    reserveMemory( velocity_gradient, reserveSize, field.velocity_gradient, dt, "velocity gradient" );
    reserveMemory( scalar, reserveSize, field.scalar, dt, "scalar" );
    reserveMemory( scalar_gradient, reserveSize, field.scalar_gradient, dt, "scalar gradient" );
    if ( dt.number_of_vertices()<NO_DIM+1 ) return;     // stop the computation if the Delaunay triangulation is not well defined
    
    
    
    // temporary variables needed for the interpolation
    Point samplePoint;      // the coordinates of the sampling point
    
#ifdef TEST_PADDING
    vector<size_t> incompleteCells_d;// keep track of grid cells where there is an error in the density field computation (because one of the vertices is a dummy point)
    vector<size_t> incompleteCells;  // keep track of grid cells where there is an error in the field (all fields except density) computation (because one of the vertices is a dummy point)
#endif
    int prev = 0, amount100 = 0;    // variable to show the user about the progress of the computation
    
    
    // parameters for searching in the triangulation
    Locate_type lt;
    int li, lj;
    Cell_handle current;
    
    
    
    size_t const noElements = samples.size();
    for (size_t i=0; i<noElements; ++i)
    {
        // print information to the user about the ongoing computation
        amount100 = (100 * i)/ noElements;
        if (prev < amount100)
            message.updateProgress( ++prev );
        
        // check if the sample point is inside the region of interest, otherwise continue to next sample point
        if ( not userOptions.region.isPointInBox(samples[i].position()) )
            continue;
        
        // locate the Delaunay cell that contains the center of the grid point
#if NO_DIM==2
        samplePoint = Point( samples[i].position(0), samples[i].position(1) );
        current = dt.locate( samplePoint, lt, li, current );
#elif NO_DIM==3
        samplePoint = Point( samples[i].position(0), samples[i].position(1), samples[i].position(2) );
        current = dt.locate( samplePoint, lt, li, lj, current );
#endif
        
        // check if the cell has a vertex which is at infinity
        if ( dt.is_infinite(current) )
        {
            if (field.density) density->push_back( Real(0.) );
#ifdef VELOCITY
            if (field.velocity) velocity->push_back( Pvector<Real,noVelComp>::zero() );
            if (field.velocity_gradient) velocity_gradient->push_back( Pvector<Real,noGradComp>::zero() );
#endif
#ifdef SCALAR
            if (field.scalar) scalar->push_back( Pvector<Real,noScalarComp>::zero() );
            if (field.scalar_gradient) scalar_gradient->push_back( Pvector<Real,noScalarGradComp>::zero() );
#endif
            continue;
        }
        
        Real posMatrixInverse[NO_DIM][NO_DIM];  //stores the inverse of the vertex position difference matrix
        positionMatrix( current, posMatrixInverse );
        Vertex_handle base = current->vertex(0);  // stores the "base" vertex of the Delaunay cell
        samplePoint = relativeSamplePoint( base, samplePoint ); // now sample point stores the relative position of the sample point with respect to the base vertex
        
        
        // test for the completness of the padding if required
#ifdef TEST_PADDING
        if ( hasDummyNeighbor( current ) ) updateDummyGridCells( i, &incompleteCells_d ); // if one of the vertices is or has dummy neighbors, keep track of this cell for the density computation part
        if ( hasDummyVertex( current ) ) updateDummyGridCells( i, &incompleteCells );     // if one of the vertices is a dummy point, keep track of this grid cell for velocity/scalar field computations
#endif
        
        // now compute the field values at the sample points
        if (field.density)
        {
            Real densGrad[NO_DIM];
            densityGrad(current, posMatrixInverse, densGrad);
            density->push_back( densityValue(densGrad,base,samplePoint) );
        }
#ifdef VELOCITY
        if (field.velocity or field.velocity_gradient)
        {
            Real velGrad[NO_DIM][noVelComp];
            velocityGrad( current, posMatrixInverse, velGrad );
            if (field.velocity) velocity->push_back( velocityValue(velGrad,base,samplePoint) );
            if (field.velocity_gradient) velocity_gradient->push_back( velocityGradient(velGrad) );
        }
#endif
#ifdef SCALAR
        if (field.scalar or field.scalar_gradient)
        {
            Real sGrad[NO_DIM][noScalarComp];
            scalarGrad( current, posMatrixInverse, sGrad );
            if (field.scalar) scalar->push_back( scalarValue(sGrad,base,samplePoint) );
            if (field.scalar_gradient) scalar_gradient->push_back( scalarGradient(sGrad) );
        }
#endif
    }
    message << "100%\n" << MESSAGE::Flush;
    
    
#ifdef TEST_PADDING
    if (field.density)
        showCellsContainingDummyPoints( &incompleteCells_d, userOptions, userOptions.outputFilename+"_density", "density" ); // check if there are any cells which may have an error in thedensity estimation due to an incomplete Delaunay tesselation over the region of interest
    if ( field.velocity or field.velocity_gradient or field.scalar or field.scalar_gradient )
        showCellsContainingDummyPoints( &incompleteCells, userOptions, userOptions.outputFilename+"_fields", "fields" ); // check if there are any cells which may have an error in the fields estimation (all fields except density) due to an incomplete Delaunay tesselation over the region of interest
#endif
}






