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



/* Field interpolation evaluated at single sample points (point values, NOT cell volume averages):
   the linear DTFE estimate of the Delaunay cell containing each point. */

#include "triangulation_common.h"




// Interpolates fields at the center of each cell of a regular grid (point value, not cell-averaged).
// 'dt' carries vertex densities; 'quantities' receives the output grid fields.
void interpolateGrid(DT &dt,
                     User_options &userOptions,
                     Quantities *quantities)
{
    size_t *nGrid = &(userOptions.gridSize[0]); // grid size along each axis
    Box boxCoordinates = userOptions.region;    // box region of interest
    vector<Real> boxLength;                     // box size along each direction
    for (size_t i=0; i<NO_DIM; ++i)
        boxLength.push_back( boxCoordinates[2*i+1]-boxCoordinates[2*i] );

    // output field vectors
    Field field = userOptions.uField;
    std::vector<Real>                         *density = &(quantities->density);
    std::vector< Pvector<Real,noVelComp> >    *velocity = &(quantities->velocity);
    std::vector< Pvector<Real,noGradComp> >   *velocity_gradient = &(quantities->velocity_gradient);
    std::vector< Pvector<Real,noScalarComp> > *scalar = &(quantities->scalar);
    std::vector< Pvector<Real,noScalarGradComp> > *scalar_gradient = &(quantities->scalar_gradient);


    MESSAGE::Message message( userOptions.verboseLevel );
    message << "Computing interpolation of the fields at the sampling points given by a regular " <<
            MESSAGE::printElements( nGrid, NO_DIM, "*" ) << " grid in the region " <<
            boxCoordinates.print() << ".\n"
            << "\t Done:   " << MESSAGE::Flush;


    // reserve output fields (reserveMemory zero-fills if the triangulation is undefined)
    size_t reserveSize = 1;
    for (size_t i=0; i<NO_DIM; ++i)
        reserveSize *= nGrid[i];
    reserveMemory( density, reserveSize, field.density, dt, "density" );
    reserveMemory( velocity, reserveSize, field.velocity, dt, "velocity" );
    reserveMemory( velocity_gradient, reserveSize, field.velocity_gradient, dt, "velocity gradient" );
    reserveMemory( scalar, reserveSize, field.scalar, dt, "scalar" );
    reserveMemory( scalar_gradient, reserveSize, field.scalar_gradient, dt, "scalar gradient" );
    if ( dt.number_of_vertices()<NO_DIM+1 ) return;     // no interpolation possible without cells



    Point samplePoint;
    Real dx[NO_DIM];        // grid spacing along each axis
    for (size_t i=0; i<NO_DIM; ++i)
        dx[i] = boxLength[i] / nGrid[i];

#ifdef TEST_PADDING
    vector<size_t> incompleteCells_d;// grid cells with a dummy-point error in the density field
    vector<size_t> incompleteCells;  // grid cells with a dummy-point error in the other fields
#endif
    int prev = 0, amount100 = 0;


    // triangulation point-location state
    Locate_type lt;
    int li, lj;
    Cell_handle current;



    Real x = boxCoordinates[0] + dx[0]/2.;
    for (size_t i=0; i<nGrid[0]; ++i)
    {
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
                
                // infinite cell (outside the convex hull) -> zero field
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
                
                Real posMatrixInverse[NO_DIM][NO_DIM];  // inverse vertex position-difference matrix
                positionMatrix( current, posMatrixInverse );
                Vertex_handle base = current->vertex(0);
                samplePoint = relativeSamplePoint( base, samplePoint ); // relative to base vertex

#ifdef TEST_PADDING
#if NO_DIM==2
                size_t k = 0;
#endif
                Real index = gridCellIndex( i,j,k, nGrid );
                if ( hasDummyNeighbor( current ) ) updateDummyGridCells( index, &incompleteCells_d ); // dummy neighbor -> flag for density
                if ( hasDummyVertex( current ) ) updateDummyGridCells( index, &incompleteCells );     // dummy vertex -> flag for other fields
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
        showCellsContainingDummyPoints( &incompleteCells_d, userOptions, userOptions.outputFilename+"_density", "density" ); // report cells whose density may be wrong (incomplete tessellation)
    if ( field.velocity or field.velocity_gradient or field.scalar or field.scalar_gradient )
        showCellsContainingDummyPoints( &incompleteCells, userOptions, userOptions.outputFilename+"_fields", "fields" ); // report cells whose other fields may be wrong (incomplete tessellation)
#endif
}



// Interpolates fields to a redshift-cone grid (for galaxy surveys) at the sample points
// (point value, not cell-averaged). Args as interpolateGrid, on a redshift-cone grid.
void interpolateRedshiftCone(DT &dt,
                             User_options &userOptions,
                             Quantities *quantities)
{
    size_t *nGrid = &(userOptions.gridSize[0]); // grid size along each axis
    Box boxCoordinates = userOptions.redshiftCone;    // redshift-cone region
    checkAngles( &(boxCoordinates[2]), (NO_DIM-1)*2 );// angles degrees -> radians
    Real *origin = &(userOptions.originPosition[0]);
    vector<Real> boxLength;                     // box size along each direction
    for (size_t i=0; i<NO_DIM; ++i)
        boxLength.push_back( boxCoordinates[2*i+1]-boxCoordinates[2*i] );

    // output field vectors
    Field field = userOptions.uField;
    std::vector<Real>                         *density = &(quantities->density);
    std::vector< Pvector<Real,noVelComp> >    *velocity = &(quantities->velocity);
    std::vector< Pvector<Real,noGradComp> >   *velocity_gradient = &(quantities->velocity_gradient);
    std::vector< Pvector<Real,noScalarComp> > *scalar = &(quantities->scalar);
    std::vector< Pvector<Real,noScalarGradComp> > *scalar_gradient = &(quantities->scalar_gradient);


    MESSAGE::Message message( userOptions.verboseLevel );
    message << "Computing interpolation of the fields at the sampling points given by a regular redshift cone grid with " << (NO_DIM==2 ? "r*psi": "r*theta*psi") << " = " <<
            MESSAGE::printElements( nGrid, NO_DIM, "*" ) << " bins in the region " << (NO_DIM==2 ? "{[r_min,r_max],[psi_min,psi_max]}" : "{[r_min,r_max],[theta_min,theta_max],[psi_min,psi_max]}") << " = " << userOptions.redshiftCone.print() << ".\n"
            << "\t Done:   " << MESSAGE::Flush;


    // reserve output fields (reserveMemory zero-fills if the triangulation is undefined)
    size_t reserveSize = 1;
    for (size_t i=0; i<NO_DIM; ++i)
        reserveSize *= nGrid[i];
    reserveMemory( density, reserveSize, field.density, dt, "density" );
    reserveMemory( velocity, reserveSize, field.velocity, dt, "velocity" );
    reserveMemory( velocity_gradient, reserveSize, field.velocity_gradient, dt, "velocity gradient" );
    reserveMemory( scalar, reserveSize, field.scalar, dt, "scalar" );
    reserveMemory( scalar_gradient, reserveSize, field.scalar_gradient, dt, "scalar gradient" );
    if ( dt.number_of_vertices()<NO_DIM+1 ) return;     // no interpolation possible without cells



    Point samplePoint;
    Real dx[NO_DIM];        // grid spacing along each axis
    for (size_t i=0; i<NO_DIM; ++i)
        dx[i] = boxLength[i] / nGrid[i];

#ifdef TEST_PADDING
    vector<size_t> incompleteCells_d;// grid cells with a dummy-point error in the density field
    vector<size_t> incompleteCells;  // grid cells with a dummy-point error in the other fields
#endif
    int prev = 0, amount100 = 0;


    // triangulation point-location state
    Locate_type lt;
    int li, lj;
    Cell_handle current;



    Real r = boxCoordinates[0] + dx[0]/2.;
    for (size_t i=0; i<nGrid[0]; ++i)
    {
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
                
                // infinite cell (outside the convex hull) -> zero field
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
                
                Real posMatrixInverse[NO_DIM][NO_DIM];  // inverse vertex position-difference matrix
                positionMatrix( current, posMatrixInverse );
                Vertex_handle base = current->vertex(0);
                samplePoint = relativeSamplePoint( base, samplePoint ); // relative to base vertex

#ifdef TEST_PADDING
#if NO_DIM==2
                size_t k = 0;
#endif
                Real index = gridCellIndex( i,j,k, nGrid );
                if ( hasDummyNeighbor( current ) ) updateDummyGridCells( index, &incompleteCells_d ); // dummy neighbor -> flag for density
                if ( hasDummyVertex( current ) ) updateDummyGridCells( index, &incompleteCells );     // dummy vertex -> flag for other fields
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
        showCellsContainingDummyPoints( &incompleteCells_d, userOptions, userOptions.outputFilename+"_density", "density" ); // report cells whose density may be wrong (incomplete tessellation)
    if ( field.velocity or field.velocity_gradient or field.scalar or field.scalar_gradient )
        showCellsContainingDummyPoints( &incompleteCells, userOptions, userOptions.outputFilename+"_fields", "fields" ); // report cells whose other fields may be wrong (incomplete tessellation)
#endif
}




// Interpolates fields at user-defined sampling points (point value, not cell-averaged).
// Args as interpolateGrid, with 'samples' giving the sampling points.
void interpolateUserSampling(DT &dt,
                             vector<Sample_point> &samples,
                             User_options &userOptions,
                             Quantities *quantities)
{
    // output field vectors
    Field field = userOptions.uField;
    std::vector<Real>                         *density = &(quantities->density);
    std::vector< Pvector<Real,noVelComp> >    *velocity = &(quantities->velocity);
    std::vector< Pvector<Real,noGradComp> >   *velocity_gradient = &(quantities->velocity_gradient);
    std::vector< Pvector<Real,noScalarComp> > *scalar = &(quantities->scalar);
    std::vector< Pvector<Real,noScalarGradComp> > *scalar_gradient = &(quantities->scalar_gradient);


    MESSAGE::Message message( userOptions.verboseLevel );
    message << "Computing interpolation of the fields at user given sampling points in the region " << userOptions.region.print() << ".\n"
            << "\t Done:   " << MESSAGE::Flush;


    // reserve output fields (reserveMemory zero-fills if the triangulation is undefined)
    size_t reserveSize = samples.size();
    reserveMemory( density, reserveSize, field.density, dt, "density" );
    reserveMemory( velocity, reserveSize, field.velocity, dt, "velocity" );
    reserveMemory( velocity_gradient, reserveSize, field.velocity_gradient, dt, "velocity gradient" );
    reserveMemory( scalar, reserveSize, field.scalar, dt, "scalar" );
    reserveMemory( scalar_gradient, reserveSize, field.scalar_gradient, dt, "scalar gradient" );
    if ( dt.number_of_vertices()<NO_DIM+1 ) return;     // no interpolation possible without cells



    Point samplePoint;

#ifdef TEST_PADDING
    vector<size_t> incompleteCells_d;// grid cells with a dummy-point error in the density field
    vector<size_t> incompleteCells;  // grid cells with a dummy-point error in the other fields
#endif
    int prev = 0, amount100 = 0;


    // triangulation point-location state
    Locate_type lt;
    int li, lj;
    Cell_handle current;



    size_t const noElements = samples.size();
    for (size_t i=0; i<noElements; ++i)
    {
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
        
        // infinite cell (outside the convex hull) -> zero field
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
        
        Real posMatrixInverse[NO_DIM][NO_DIM];  // inverse vertex position-difference matrix
        positionMatrix( current, posMatrixInverse );
        Vertex_handle base = current->vertex(0);
        samplePoint = relativeSamplePoint( base, samplePoint ); // relative to base vertex


#ifdef TEST_PADDING
        if ( hasDummyNeighbor( current ) ) updateDummyGridCells( i, &incompleteCells_d ); // dummy neighbor -> flag for density
        if ( hasDummyVertex( current ) ) updateDummyGridCells( i, &incompleteCells );     // dummy vertex -> flag for other fields
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
        showCellsContainingDummyPoints( &incompleteCells_d, userOptions, userOptions.outputFilename+"_density", "density" ); // report cells whose density may be wrong (incomplete tessellation)
    if ( field.velocity or field.velocity_gradient or field.scalar or field.scalar_gradient )
        showCellsContainingDummyPoints( &incompleteCells, userOptions, userOptions.outputFilename+"_fields", "fields" ); // report cells whose other fields may be wrong (incomplete tessellation)
#endif
}






