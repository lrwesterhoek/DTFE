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


/* Volume-averaged grid interpolation, method 2 (= MC sample points chosen inside the
   grid cell). Method 3 (equidistant sampling) is at the end of the file. */
#include "triangulation_common.h"



// Volume-averaged grid interpolation (method 2): averages the field over NN Monte Carlo points drawn
// inside each grid cell. The only method that can also report the per-cell velocity standard deviation.
void interpolateGrid_averaged_2(DT &dt,
                                User_options &userOptions,
                                Quantities *quantities)
{
    size_t *nGrid = &(userOptions.gridSize[0]); // grid size along each axis
    int const NN = userOptions.noPoints;        // MC samples per grid cell
    Box boxCoordinates = userOptions.region;    // box region of interest
    vector<Real> boxLength;                     // box size along each direction
    for (size_t i=0; i<NO_DIM; ++i)
        boxLength.push_back( boxCoordinates[2*i+1]-boxCoordinates[2*i] );


    // output field vectors
    Field field = userOptions.aField;
    std::vector<Real>                         *density = &(quantities->density);
    std::vector< Pvector<Real,noVelComp> >    *velocity = &(quantities->velocity);
    std::vector< Pvector<Real,noGradComp> >   *velocity_gradient = &(quantities->velocity_gradient);
    std::vector<Real>                         *velocity_std = &(quantities->velocity_std);
    std::vector< Pvector<Real,noScalarComp> > *scalar = &(quantities->scalar);
    std::vector< Pvector<Real,noScalarGradComp> > *scalar_gradient = &(quantities->scalar_gradient);


    MESSAGE::Message message( userOptions.verboseLevel );
    message << "\nComputing interpolation of the fields volume averaged over the sampling cell on a regular "
            << MESSAGE::printElements( nGrid, NO_DIM, "*" ) << " grid in the region "
            << boxCoordinates.print()
            << ". The volume average is done using " << NN << " Monte Carlo samples in each grid cell.\n"
            << "\t Done: " << MESSAGE::Flush;


    // reserve output fields (reserveMemory zero-fills if the triangulation is undefined)
    size_t reserveSize = (NO_DIM==2) ? nGrid[0]*nGrid[1] : nGrid[0]*nGrid[1]*nGrid[2];
    reserveMemory( density, reserveSize, field.density, dt, "density" );
    reserveMemory( velocity, reserveSize, field.velocity, dt, "velocity" );
    reserveMemory( velocity_gradient, reserveSize, field.velocity_gradient, dt, "velocity gradient" );
    reserveMemory( velocity_std, reserveSize, field.velocity_std, dt, "velocity standard deviation" );
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
    Cell_handle current, previous;


    size_t seed = userOptions.randomSeed;
    if ( userOptions.partNo>=0 )
        seed += 1000*userOptions.partNo; // distinct seed per partition
    base_generator_type generator( seed );
    boost::uniform_real<> uni_dist(-0.5,0.5);
    boost::variate_generator<base_generator_type&, boost::uniform_real<> > uni(generator, uni_dist);
    
    
    
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
            previous = dt.locate( Point(x,y) , lt, li, previous );
#elif NO_DIM==3
            Real z = boxCoordinates[4] + dx[2]/2.;
            for (size_t k=0; k<nGrid[2]; ++k)
            {
                previous = dt.locate( Point(x,y,z) , lt, li, lj, previous );
#endif

                // MC-loop accumulators
                bool dummyNeighbors = false, dummyVertices = false;
                Real tempD = Real(0.);
#ifdef VELOCITY
                Pvector<Real,noVelComp> tempV = Pvector<Real,noVelComp>::zero();
                Pvector<Real,noGradComp> tempVG = Pvector<Real,noGradComp>::zero();
                std::vector<Pvector<Real,noVelComp>> tempVelocities(NN);
#endif
#ifdef SCALAR
                Pvector<Real,noScalarComp> tempS = Pvector<Real,noScalarComp>::zero();
                Pvector<Real,noScalarGradComp> tempSG = Pvector<Real,noScalarGradComp>::zero();
#endif
                // cache the last located cell's inverse position matrix; consecutive samples in a
                // grid cell often land in the same Delaunay cell, so this avoids re-inverting it
                Cell_handle cachedCell;
                Real cachedPosMatInv[NO_DIM][NO_DIM];
                Vertex_handle cachedBase;
                bool hasCachedCell = false;

                for (int w=0; w<NN; ++w)        // MC loop over random samples in this grid cell
                {
#if NO_DIM==2
                    samplePoint = Point( uni()*dx[0] + x, uni()*dx[1] + y );
                    current = dt.locate( samplePoint,lt,li,previous );
#elif NO_DIM==3
                    samplePoint = Point( uni()*dx[0] + x, uni()*dx[1] + y, uni()*dx[2] + z );
                    current = dt.locate( samplePoint,lt,li,lj, previous );
#endif

#ifdef TEST_PADDING
                    if ( hasDummyNeighbor( current ) ) dummyNeighbors = true;
                    if ( hasDummyVertex( current ) ) dummyVertices = true;
#endif

                    // infinite cell (sample outside the convex hull) -> skip
                    if ( dt.is_infinite(current) )
                        continue;

                    // reuse the cached inverse position matrix while still in the same cell
                    Real posMatrixInverse[NO_DIM][NO_DIM];
                    Vertex_handle base;
                    if ( hasCachedCell && current == cachedCell )
                    {
                        for (int r=0; r<NO_DIM; ++r)
                            for (int c=0; c<NO_DIM; ++c)
                                posMatrixInverse[r][c] = cachedPosMatInv[r][c];
                        base = cachedBase;
                    }
                    else
                    {
                        positionMatrix( current, posMatrixInverse );
                        base = current->vertex(0);
                        for (int r=0; r<NO_DIM; ++r)
                            for (int c=0; c<NO_DIM; ++c)
                                cachedPosMatInv[r][c] = posMatrixInverse[r][c];
                        cachedBase = base;
                        cachedCell = current;
                        hasCachedCell = true;
                    }
                    samplePoint = relativeSamplePoint( base, samplePoint ); // relative to base vertex
                    
                    if (field.density)
                    {
                        Real densGrad[NO_DIM];
                        densityGrad(current, posMatrixInverse, densGrad);
                        tempD += densityValue(densGrad,base,samplePoint);
                    }
#ifdef VELOCITY
                    if (field.velocity or field.velocity_gradient or field.velocity_std)
                    {
                        Real velGrad[NO_DIM][noVelComp];
                        velocityGrad( current, posMatrixInverse, velGrad );
                        if (field.velocity) tempV += velocityValue(velGrad,base,samplePoint);
                        if (field.velocity_gradient) tempVG += velocityGradient(velGrad);
                        if (field.velocity_std) tempVelocities[w] = velocityValue(velGrad,base,samplePoint);
                    }
#endif
#ifdef SCALAR
                    if (field.scalar or field.scalar_gradient)
                    {
#ifdef MY_SCALAR
                        tempS += customScalar(current, posMatrixInverse, base, samplePoint);
#else
                        Real sGrad[NO_DIM][noScalarComp];
                        scalarGrad( current, posMatrixInverse, sGrad );
                        if (field.scalar) tempS += scalarValue(sGrad,base,samplePoint);
                        if (field.scalar_gradient) tempSG += scalarGradient(sGrad);
#endif
                    }
#endif
                }


                // average and store the field values
                if (field.density) density->push_back( tempD/NN );
#ifdef VELOCITY
                if (field.velocity) velocity->push_back( tempV/NN );
                if (field.velocity_gradient) velocity_gradient->push_back( tempVG/NN );
                if (field.velocity_std) velocity_std->push_back( standardDeviation(tempVelocities.data(),NN) );
#endif
#ifdef SCALAR
                if (field.scalar) scalar->push_back( tempS/NN );
                if (field.scalar_gradient) scalar_gradient->push_back( tempSG/NN );
#endif

#ifdef TEST_PADDING
#if NO_DIM==2
                size_t k = 0;
#endif
                Real index = gridCellIndex( i,j,k, nGrid );
                if ( dummyNeighbors ) updateDummyGridCells( index, &incompleteCells_d ); // dummy neighbor -> flag for density
                if ( dummyVertices ) updateDummyGridCells( index, &incompleteCells ); // dummy vertex -> flag for other fields
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





// Volume-averaged redshift-cone interpolation (method 2): MC sampling of NN points
// inside each cone grid cell. Args as interpolateGrid_averaged_2, on a redshift-cone grid.
void interpolateRedshiftCone_averaged_2(DT &dt,
                                        User_options &userOptions,
                                        Quantities *quantities)
{
    size_t *nGrid = &(userOptions.gridSize[0]); // grid size along each axis
    int const NN = userOptions.noPoints;        // MC samples per grid cell
    Box boxCoordinates = userOptions.redshiftCone;    // redshift-cone region
    checkAngles( &(boxCoordinates[2]), (NO_DIM-1)*2 );// angles degrees -> radians
    Real *origin = &(userOptions.originPosition[0]);
    vector<Real> boxLength;                     // box size along each direction
    for (size_t i=0; i<NO_DIM; ++i)
        boxLength.push_back( boxCoordinates[2*i+1]-boxCoordinates[2*i] );

    // output field vectors
    Field field = userOptions.aField;
    std::vector<Real>                         *density = &(quantities->density);
    std::vector< Pvector<Real,noVelComp> >    *velocity = &(quantities->velocity);
    std::vector< Pvector<Real,noGradComp> >   *velocity_gradient = &(quantities->velocity_gradient);
    std::vector< Pvector<Real,noScalarComp> > *scalar = &(quantities->scalar);
    std::vector< Pvector<Real,noScalarGradComp> > *scalar_gradient = &(quantities->scalar_gradient);


    MESSAGE::Message message( userOptions.verboseLevel );
    message << "Computing interpolation of the fields volume averaged over the sampling cell on a redshift cone grid with " << (NO_DIM==2 ? "r*psi": "r*theta*psi") << " = " <<
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
    Cell_handle current, previous;


    size_t seed = userOptions.randomSeed;
    if ( userOptions.partNo>=0 )
        seed += 1000*userOptions.partNo; // distinct seed per partition
    base_generator_type generator( seed );
    boost::uniform_real<> uni_dist(-0.5,0.5);
    boost::variate_generator<base_generator_type&, boost::uniform_real<> > uni(generator, uni_dist);



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
            previous = dt.locate( samplePoint, lt, li, previous );
#elif NO_DIM==3
            Real psi = boxCoordinates[4] + dx[2]/2.;
            for (size_t k=0; k<nGrid[2]; ++k)
            {
                Real tempX = origin[0] + r*sin(theta)*cos(psi);
                Real tempY = origin[1] + r*sin(theta)*sin(psi);
                Real tempZ = origin[2] + r*cos(theta);
                samplePoint = Point(tempX,tempY,tempZ);
                previous = dt.locate( samplePoint, lt, li, lj, previous );
#endif

                // MC-loop accumulators
                bool dummyNeighbors = false, dummyVertices = false;
                Real tempD = Real(0.);
#ifdef VELOCITY
                Pvector<Real,noVelComp> tempV = Pvector<Real,noVelComp>::zero();
                Pvector<Real,noGradComp> tempVG = Pvector<Real,noGradComp>::zero();
#endif
#ifdef SCALAR
                Pvector<Real,noScalarComp> tempS = Pvector<Real,noScalarComp>::zero();
                Pvector<Real,noScalarGradComp> tempSG = Pvector<Real,noScalarGradComp>::zero();
#endif
                Cell_handle cachedCell2;
                Real cachedPosMatInv2[NO_DIM][NO_DIM];
                Vertex_handle cachedBase2;
                bool hasCachedCell2 = false;

                for (int w=0; w<NN; ++w)        // MC loop over random samples in this grid cell
                {
                    // random sample point in this cone grid cell, in Cartesian coords
#if NO_DIM==2
                    Real tempR = r + uni()*dx[0];
                    Real tempTheta = theta + uni()*dx[1];
                    tempX = origin[0] + tempR*cos(tempTheta);
                    tempY = origin[0] + tempR*sin(tempTheta);
                    samplePoint = Point(tempX,tempY);
                    current = dt.locate( samplePoint,lt,li,previous );
#elif NO_DIM==3
                    Real tempR = r + uni()*dx[0];
                    Real tempTheta = theta + uni()*dx[1];
                    Real tempPsi = psi + uni()*dx[2];
                    tempX = origin[0] + tempR*sin(tempTheta)*cos(tempPsi);
                    tempY = origin[1] + tempR*sin(tempTheta)*sin(tempPsi);
                    tempZ = origin[2] + tempR*cos(tempTheta);
                    samplePoint = Point(tempX,tempY,tempZ);
                    current = dt.locate( samplePoint,lt,li,lj, previous );
#endif

#ifdef TEST_PADDING
                    if ( hasDummyNeighbor( current ) ) dummyNeighbors = true;
                    if ( hasDummyVertex( current ) ) dummyVertices = true;
#endif

                    if ( dt.is_infinite(current) )
                        continue;

                    Real posMatrixInverse[NO_DIM][NO_DIM];
                    Vertex_handle base;
                    if ( hasCachedCell2 && current == cachedCell2 )
                    {
                        for (int r=0; r<NO_DIM; ++r)
                            for (int c=0; c<NO_DIM; ++c)
                                posMatrixInverse[r][c] = cachedPosMatInv2[r][c];
                        base = cachedBase2;
                    }
                    else
                    {
                        positionMatrix( current, posMatrixInverse );
                        base = current->vertex(0);
                        for (int r=0; r<NO_DIM; ++r)
                            for (int c=0; c<NO_DIM; ++c)
                                cachedPosMatInv2[r][c] = posMatrixInverse[r][c];
                        cachedBase2 = base;
                        cachedCell2 = current;
                        hasCachedCell2 = true;
                    }
                    samplePoint = relativeSamplePoint( base, samplePoint ); // relative to base vertex

                    if (field.density)
                    {
                        Real densGrad[NO_DIM];
                        densityGrad(current, posMatrixInverse, densGrad);
                        tempD += densityValue(densGrad,base,samplePoint);
                    }
#ifdef VELOCITY
                    if (field.velocity or field.velocity_gradient)
                    {
                        Real velGrad[NO_DIM][noVelComp];
                        velocityGrad( current, posMatrixInverse, velGrad );
                        if (field.velocity) tempV += velocityValue(velGrad,base,samplePoint);
                        if (field.velocity_gradient) tempVG += velocityGradient(velGrad);
                    }
#endif
#ifdef SCALAR
                    if (field.scalar or field.scalar_gradient)
                    {
#ifdef MY_SCALAR
                        tempS += customScalar(current, posMatrixInverse, base, samplePoint);
#else
                        Real sGrad[NO_DIM][noScalarComp];
                        scalarGrad( current, posMatrixInverse, sGrad );
                        if (field.scalar) tempS += scalarValue(sGrad,base,samplePoint);
                        if (field.scalar_gradient) tempSG += scalarGradient(sGrad);
#endif
                    }
#endif
                }
                
                
                // average and store the field values
                if (field.density) density->push_back( tempD/NN );
#ifdef VELOCITY
                if (field.velocity) velocity->push_back( tempV/NN );
                if (field.velocity_gradient) velocity_gradient->push_back( tempVG/NN );
#endif
#ifdef SCALAR
                if (field.scalar) scalar->push_back( tempS/NN );
                if (field.scalar_gradient) scalar_gradient->push_back( tempSG/NN );
#endif
                
#ifdef TEST_PADDING
#if NO_DIM==2
                size_t k = 0;
#endif
                Real index = gridCellIndex( i,j,k, nGrid );
                if ( dummyNeighbors ) updateDummyGridCells( index, &incompleteCells_d ); // dummy neighbor -> flag for density
                if ( dummyVertices ) updateDummyGridCells( index, &incompleteCells ); // dummy vertex -> flag for other fields
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




// Volume-averaged interpolation at user-defined sampling points (method 2): MC sampling
// of NN points inside each sampling cell. Args as interpolateGrid_averaged_2, with 'samples'.
void interpolateUserSampling_averaged_2(DT &dt,
                                        vector<Sample_point> &samples,
                                        User_options &userOptions,
                                        Quantities *quantities)
{
    int const NN = userOptions.noPoints;        // MC samples per grid cell

    // output field vectors
    Field field = userOptions.aField;
    std::vector<Real>                         *density = &(quantities->density);
    std::vector< Pvector<Real,noVelComp> >    *velocity = &(quantities->velocity);
    std::vector< Pvector<Real,noGradComp> >   *velocity_gradient = &(quantities->velocity_gradient);
    std::vector< Pvector<Real,noScalarComp> > *scalar = &(quantities->scalar);
    std::vector< Pvector<Real,noScalarGradComp> > *scalar_gradient = &(quantities->scalar_gradient);
    
    
    MESSAGE::Message message( userOptions.verboseLevel );
    message << "Computing interpolation of the fields volume averaged over the sampling cell at user given sampling points in the region " << userOptions.region.print() << ".\n"
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
    Cell_handle current, previous;


    size_t seed = userOptions.randomSeed;
    if ( userOptions.partNo>=0 )
        seed += 1000*userOptions.partNo; // distinct seed per partition
    base_generator_type generator( seed );
    boost::uniform_real<> uni_dist(-0.5,0.5);
    boost::variate_generator<base_generator_type&, boost::uniform_real<> > uni(generator, uni_dist);



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
        previous = dt.locate( samplePoint, lt, li, previous );
#elif NO_DIM==3
        samplePoint = Point( samples[i].position(0), samples[i].position(1), samples[i].position(2) );
        previous = dt.locate( samplePoint, lt, li, lj, previous );
#endif
        
        
        
        // MC-loop accumulators
        bool dummyNeighbors = false, dummyVertices = false;
        Real tempD = Real(0.);
#ifdef VELOCITY
        Pvector<Real,noVelComp> tempV = Pvector<Real,noVelComp>::zero();
        Pvector<Real,noGradComp> tempVG = Pvector<Real,noGradComp>::zero();
#endif
#ifdef SCALAR
        Pvector<Real,noScalarComp> tempS = Pvector<Real,noScalarComp>::zero();
        Pvector<Real,noScalarGradComp> tempSG = Pvector<Real,noScalarGradComp>::zero();
#endif
        Cell_handle cachedCell3;
        Real cachedPosMatInv3[NO_DIM][NO_DIM];
        Vertex_handle cachedBase3;
        bool hasCachedCell3 = false;

        for (int w=0; w<NN; ++w)        // MC loop over random samples in this sampling cell
        {
            Real tempX = samples[i].position(0) + uni()*samples[i].delta(0);
            Real tempY = samples[i].position(1) + uni()*samples[i].delta(1);
#if NO_DIM==2
            samplePoint = Point(tempX,tempY);
            current = dt.locate( samplePoint,lt,li,previous );
#elif NO_DIM==3
            Real tempZ = samples[i].position(2) + uni()*samples[i].delta(2);
            samplePoint = Point(tempX,tempY,tempZ);
            current = dt.locate( samplePoint,lt,li,lj, previous );
#endif

#ifdef TEST_PADDING
            if ( hasDummyNeighbor( current ) ) dummyNeighbors = true;
            if ( hasDummyVertex( current ) ) dummyVertices = true;
#endif

            if ( dt.is_infinite(current) )
                continue;

            Real posMatrixInverse[NO_DIM][NO_DIM];
            Vertex_handle base;
            if ( hasCachedCell3 && current == cachedCell3 )
            {
                for (int r=0; r<NO_DIM; ++r)
                    for (int c=0; c<NO_DIM; ++c)
                        posMatrixInverse[r][c] = cachedPosMatInv3[r][c];
                base = cachedBase3;
            }
            else
            {
                positionMatrix( current, posMatrixInverse );
                base = current->vertex(0);
                for (int r=0; r<NO_DIM; ++r)
                    for (int c=0; c<NO_DIM; ++c)
                        cachedPosMatInv3[r][c] = posMatrixInverse[r][c];
                cachedBase3 = base;
                cachedCell3 = current;
                hasCachedCell3 = true;
            }
            samplePoint = relativeSamplePoint( base, samplePoint ); // relative to base vertex
            
            if (field.density)
            {
                Real densGrad[NO_DIM];
                densityGrad(current, posMatrixInverse, densGrad);
                tempD += densityValue(densGrad,base,samplePoint);
            }
#ifdef VELOCITY
            if (field.velocity or field.velocity_gradient)
            {
                Real velGrad[NO_DIM][noVelComp];
                velocityGrad( current, posMatrixInverse, velGrad );
                if (field.velocity) tempV += velocityValue(velGrad,base,samplePoint);
                if (field.velocity_gradient) tempVG += velocityGradient(velGrad);
            }
#endif
#ifdef SCALAR
            if (field.scalar or field.scalar_gradient)
            {
#ifdef MY_SCALAR
                tempS += customScalar(current, posMatrixInverse, base, samplePoint);
#else
                Real sGrad[NO_DIM][noScalarComp];
                scalarGrad( current, posMatrixInverse, sGrad );
                if (field.scalar) tempS += scalarValue(sGrad,base,samplePoint);
                if (field.scalar_gradient) tempSG += scalarGradient(sGrad);
#endif
            }
#endif
        }
        
        
        // average and store the field values
        if (field.density) density->push_back( tempD/NN );
#ifdef VELOCITY
        if (field.velocity) velocity->push_back( tempV/NN );
        if (field.velocity_gradient) velocity_gradient->push_back( tempVG/NN );
#endif
#ifdef SCALAR
        if (field.scalar) scalar->push_back( tempS/NN );
        if (field.scalar_gradient) scalar_gradient->push_back( tempSG/NN );
#endif
        
#ifdef TEST_PADDING
        if ( dummyNeighbors ) updateDummyGridCells( i, &incompleteCells_d ); // dummy neighbor -> flag for density
        if ( dummyVertices ) updateDummyGridCells( i, &incompleteCells ); // dummy vertex -> flag for other fields
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










// Volume-averaged grid interpolation (method 3): averages the field over NN equidistant (not random)
// points laid out on a regular sub-grid inside each grid cell.
void interpolateGrid_averaged_3(DT &dt,
                                User_options &userOptions,
                                Quantities *quantities)
{
    size_t *nGrid = &(userOptions.gridSize[0]); // grid size along each axis
    int const NN = userOptions.noPoints;        // equidistant samples per grid cell
    int const n = rootN( NN, NO_DIM );          // equidistant samples along each axis
    Box boxCoordinates = userOptions.region;    // box region of interest
    vector<Real> boxLength;                     // box size along each direction
    for (size_t i=0; i<NO_DIM; ++i)
        boxLength.push_back( boxCoordinates[2*i+1]-boxCoordinates[2*i] );


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
            << ". The volume average is done using " << NN << " equidistant sampling points in each grid cell.\n"
            << "\t Done: " << MESSAGE::Flush;


    // reserve output fields (reserveMemory zero-fills if the triangulation is undefined)
    size_t reserveSize = (NO_DIM==2) ? nGrid[0]*nGrid[1] : nGrid[0]*nGrid[1]*nGrid[2];
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
    Cell_handle current, previous;


    // equidistant sample offsets relative to the grid-cell center
    std::vector<Real> samplePosition_buf(NN * NO_DIM);
    Real (*samplePosition)[NO_DIM] = reinterpret_cast<Real(*)[NO_DIM]>(samplePosition_buf.data());
    for (int i1=0; i1<n; ++i1)
        for (int i2=0; i2<n; ++i2)
#if NO_DIM==2
        {
            int index = i1*n + i2;
#endif
#if NO_DIM==3
            for (int i3=0; i3<n; ++i3)
            {
                int index = i1*n*n + i2*n + i3;
                samplePosition[index][2] = (i3+0.5)*dx[2]/n - dx[2]/2.;
#endif
                samplePosition[index][0] = (i1+0.5)*dx[0]/n - dx[0]/2.;
                samplePosition[index][1] = (i2+0.5)*dx[1]/n - dx[1]/2.;
            }
    
    
    
    
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
            previous = dt.locate( Point(x,y) , lt, li, previous );
#elif NO_DIM==3     // add additional z-loop
            Real z = boxCoordinates[4] + dx[2]/2.;
            for (size_t k=0; k<nGrid[2]; ++k)
            {
                previous = dt.locate( Point(x,y,z) , lt, li, lj, previous );
#endif


                // per-cell accumulators
                bool dummyNeighbors = false, dummyVertices = false;
                Real tempD = Real(0.);
#ifdef VELOCITY
                Pvector<Real,noVelComp> tempV = Pvector<Real,noVelComp>::zero();
                Pvector<Real,noGradComp> tempVG = Pvector<Real,noGradComp>::zero();
#endif
#ifdef SCALAR
                Pvector<Real,noScalarComp> tempS = Pvector<Real,noScalarComp>::zero();
                Pvector<Real,noScalarGradComp> tempSG = Pvector<Real,noScalarGradComp>::zero();
#endif
                Cell_handle cachedCell4;
                Real cachedPosMatInv4[NO_DIM][NO_DIM];
                Vertex_handle cachedBase4;
                bool hasCachedCell4 = false;

                for (int w=0; w<NN; ++w)        // loop over equidistant sample points in this grid cell
                {
#if NO_DIM==2
                    samplePoint = Point( x + samplePosition[w][0], y + samplePosition[w][1] );
                    current = dt.locate( samplePoint,lt,li,previous );
#elif NO_DIM==3
                    samplePoint = Point( x + samplePosition[w][0], y + samplePosition[w][1], z + samplePosition[w][2] );
                    current = dt.locate( samplePoint,lt,li,lj, previous );
#endif

#ifdef TEST_PADDING
                    if ( hasDummyNeighbor( current ) ) dummyNeighbors = true;
                    if ( hasDummyVertex( current ) ) dummyVertices = true;
#endif

                    if ( dt.is_infinite(current) )
                        continue;

                    Real posMatrixInverse[NO_DIM][NO_DIM];
                    Vertex_handle base;
                    if ( hasCachedCell4 && current == cachedCell4 )
                    {
                        for (int r=0; r<NO_DIM; ++r)
                            for (int c=0; c<NO_DIM; ++c)
                                posMatrixInverse[r][c] = cachedPosMatInv4[r][c];
                        base = cachedBase4;
                    }
                    else
                    {
                        positionMatrix( current, posMatrixInverse );
                        base = current->vertex(0);
                        for (int r=0; r<NO_DIM; ++r)
                            for (int c=0; c<NO_DIM; ++c)
                                cachedPosMatInv4[r][c] = posMatrixInverse[r][c];
                        cachedBase4 = base;
                        cachedCell4 = current;
                        hasCachedCell4 = true;
                    }
                    samplePoint = relativeSamplePoint( base, samplePoint ); // relative to base vertex

                    if (field.density)
                    {
                        Real densGrad[NO_DIM];
                        densityGrad(current, posMatrixInverse, densGrad);
                        tempD += densityValue(densGrad,base,samplePoint);
                    }
#ifdef VELOCITY
                    if (field.velocity or field.velocity_gradient)
                    {
                        Real velGrad[NO_DIM][noVelComp];
                        velocityGrad( current, posMatrixInverse, velGrad );
                        if (field.velocity) tempV += velocityValue(velGrad,base,samplePoint);
                        if (field.velocity_gradient) tempVG += velocityGradient(velGrad);
                    }
#endif
#ifdef SCALAR
                    if (field.scalar or field.scalar_gradient)
                    {
                        Real sGrad[NO_DIM][noScalarComp];
                        scalarGrad( current, posMatrixInverse, sGrad );
                        if (field.scalar) tempS += scalarValue(sGrad,base,samplePoint);
                        if (field.scalar_gradient) tempSG += scalarGradient(sGrad);
                    }
#endif
                }
                
                
                // average and store the field values
                if (field.density) density->push_back( tempD/NN );
#ifdef VELOCITY
                if (field.velocity) velocity->push_back( tempV/NN );
                if (field.velocity_gradient) velocity_gradient->push_back( tempVG/NN );
#endif
#ifdef SCALAR
                if (field.scalar) scalar->push_back( tempS/NN );
                if (field.scalar_gradient) scalar_gradient->push_back( tempSG/NN );
#endif
                
#ifdef TEST_PADDING
#if NO_DIM==2
                size_t k = 0;
#endif
                Real index = gridCellIndex( i,j,k, nGrid );
                if ( dummyNeighbors ) updateDummyGridCells( index, &incompleteCells_d ); // dummy neighbor -> flag for density
                if ( dummyVertices ) updateDummyGridCells( index, &incompleteCells ); // dummy vertex -> flag for other fields
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


