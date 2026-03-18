#ifndef FIELD_COMPUTATION_HEADER
#define FIELD_COMPUTATION_HEADER

/* Shared helper functions for computing field values and gradients
   inside Delaunay cells. Used by unaveraged_interpolation.cc,
   averaged_interpolation_1.cc, and averaged_interpolation_2.cc. */


/* This function returns the inverse of the vertex difference matrix for a given Delaunay cell (triangle/tetrahedron in 2D/3D).*/
inline void positionMatrix(Cell_handle &cell,
                           Real posMatrixInverse[][NO_DIM])
{
    double Ax[NO_DIM][NO_DIM];
    Vertex_handle base = cell->vertex(0);

    // first store in 'Ax' the vertices position differences
    for (int v = 0; v<NO_DIM; ++v)  //loop over vertices != base
        for (int i=0; i<NO_DIM; ++i)    //loop over spatial dimensions
            Ax[v][i] = double(cell->vertex(v+1)->point()[i] - base->point()[i]);
    matrixInverse( Ax, posMatrixInverse );
}

/* Returns the relative position of the sample point. */
inline Point relativeSamplePoint(Vertex_handle const & base,
                                 Point &samplePoint)
{
    Real temp[NO_DIM];
    for (int i=0; i<NO_DIM; ++i)
        temp[i] = samplePoint[i] - base->point()[i];
#if NO_DIM==2
    return Point( temp[0], temp[1] );
#elif NO_DIM==3
    return Point( temp[0], temp[1], temp[2] );
#endif
}



/* Computes the density gradient inside a Delaunay cell. */
template <typename Cell>
void densityGrad(Cell &cell,
                 Real posMatrixInverse[][NO_DIM],
                 Real *densGrad)
{
    Real dens[NO_DIM];  //store density differences
    for (int i=0; i<NO_DIM; ++i)
        dens[i] = cell->vertex(i+1)->info().density() - cell->vertex(0)->info().density();
    matrixMultiplication( posMatrixInverse, dens, densGrad );    //computes the density gradient = posMatrixInverse * dens
}

/* Returns the density associated to a sample point (sample point coordinates are with respect to the base vertex of the Delaunay cell). */
inline Real densityValue(Real *densGrad,
                         Vertex_handle const & base,
                         Point &samplePoint)
{
    Real temp = 0.;
    for (int i=0; i<NO_DIM; ++i)
        temp += samplePoint[i]*densGrad[i];
    return base->info().density() + temp;
}



/* Computes the velocity gradient using the positions and velocities differences in the Delaunay Triangulation cell. */
template <typename Cell>
void velocityGrad(Cell &cell,
                  Real posMatrixInverse[][NO_DIM],
                  Real velGrad[][noVelComp])
{
    Vertex_handle base = cell->vertex(0);
    Real temp[NO_DIM][noVelComp]; // stores the vertex velocities differences
    for (size_t v = 0; v<NO_DIM; ++v)   //loop over vertices != base
        for (size_t i=0; i<noVelComp; ++i)  //loop over velocity components
            temp[v][i] = cell->vertex(v+1)->info().velocity(i) - base->info().velocity(i);
    matrixMultiplication<noVelComp>( posMatrixInverse, temp, velGrad ); //computes velGrad = posMatrixInverse * temp
}

/* Computes the velocity value at the sample point (sample point coordinates are with respect to the base vertex of the Delaunay cell). */
inline Pvector<Real,noVelComp> velocityValue(Real velGrad[][noVelComp],
                                             Vertex_handle const & base,
                                             Point &samplePoint)
{
    Pvector<Real,noVelComp> temp;
    for (int i=0; i<NO_DIM; ++i)
    {
        temp[i] = 0.;
        for (int j=0; j<NO_DIM; ++j)
            temp[i] += velGrad[j][i] * samplePoint[j];
    }

    return temp + base->info().velocity();
}

/* Returns the velocity gradient in a 'Pvector<Real,noGradComp>' object. */
inline Pvector<Real,noGradComp> velocityGradient(Real velGrad[][noVelComp])
{
    Pvector<Real,noGradComp> temp;
    for (size_t j=0; j<noVelComp; ++j)
        for (size_t i=0; i<NO_DIM; ++i)
            temp[j*NO_DIM+i] = velGrad[i][j];
    return temp;
}



/* Computes the scalar gradient using the positions and scalar differences in the Delaunay Triangulation cell. */
template <typename Cell>
void scalarGrad(Cell &cell,
                Real posMatrixInverse[][NO_DIM],
                Real sGrad[][noScalarComp])
{
    Real temp[NO_DIM][noScalarComp];    // stores the vertex scalar differences
    Vertex_handle base = cell->vertex(0);
    for (size_t v = 0; v<NO_DIM; ++v)   //loop over vertices != base
        for (size_t i=0; i<noScalarComp; ++i)   //loop over number of scalar components
            temp[v][i] = cell->vertex(v+1)->info().myScalar()[i] - base->info().myScalar()[i];
    matrixMultiplication<noScalarComp>( posMatrixInverse, temp, sGrad );    //computes scalarGradient = posMatrixInverse * temp
}

/* Computes the scalar value at the sample point (sample point coordinates are with respect to the base vertex of the Delaunay cell). */
inline Pvector<Real,noScalarComp> scalarValue(Real sGrad[][noScalarComp],
                                              Vertex_handle const & base,
                                              Point &samplePoint)
{
    Pvector<Real,noScalarComp> temp;
    for (size_t i=0; i<noScalarComp; ++i)
    {
        temp[i] = 0.;
        for (size_t j=0; j<NO_DIM; ++j)
            temp[i] += sGrad[j][i] * samplePoint[j];
    }

    return temp + base->info().myScalar();
}

/* Returns the scalar gradient in a 'Pvector<Real,noScalarGradComp>' object. */
inline Pvector<Real,noScalarGradComp> scalarGradient(Real sGrad[][noScalarComp])
{
    Pvector<Real,noScalarGradComp> temp;
    for (size_t j=0; j<noScalarComp; ++j)
        for (size_t i=0; i<NO_DIM; ++i)
            temp[j*NO_DIM+i] = sGrad[i][j];
    return temp;
}



/* Function to switch to custom values for the scalar field. */
template <typename Cell>
Pvector<Real,noScalarComp> customScalar(Cell &current,
                                        Real posMatrixInverse[][NO_DIM],
                                        Vertex_handle &base,
                                        Point &samplePoint)
{
    Real densGrad[NO_DIM];
    densityGrad(current, posMatrixInverse, densGrad);
    Real density = densityValue(densGrad,base,samplePoint);

    Real velGrad[NO_DIM][noVelComp];
    velocityGrad( current, posMatrixInverse, velGrad );
    Pvector<Real,noVelComp> velocity = velocityValue(velGrad,base,samplePoint);

    Pvector<Real,noScalarComp> scalar;
    personalizedFunction( samplePoint, density, densGrad, velocity, velGrad, scalar );
    return scalar;
}

#endif
