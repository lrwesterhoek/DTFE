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


#ifndef TRIANGULATION_MISCELLANEOUS_HEADER
#define TRIANGULATION_MISCELLANEOUS_HEADER

#include "../math_functions.h"


/* Print information to the user about the amount of time taken. */
inline void printComputationTime(Timer *t, User_options *userOptions,
                                 string computationQuantityName)
{
    t->stop();
    userOptions->totalTime += t->time();
    MESSAGE::Message message( userOptions->verboseLevel );
    message << "  >>> Time: " << t->time()/userOptions->noProcessors << " sec. (" << computationQuantityName << ")\n" << MESSAGE::Flush;
    t->reset();
}



/* matrixInverse and determinant are now defined in math_functions.h */

#ifdef PHASE_SPACE
/* Computes the volume of a Delaunay cell's Eulerian (deformed) simplex.
   In PS-DTFE mode, the triangulation is in Lagrangian space, but we need
   the Eulerian volume for density estimation. */
inline Real eulerianVolume(const Cell_handle &cell)
{
    double posDiff[NO_DIM][NO_DIM];
    for (int v = 0; v < NO_DIM; ++v)
        for (int i = 0; i < NO_DIM; ++i)
            posDiff[v][i] = double(cell->vertex(v+1)->info().eulerianPosition(i))
                          - double(cell->vertex(0)->info().eulerianPosition(i));
    return simplexVolume(posDiff);
}

/* Computes the Eulerian vertex position difference matrix for a given Delaunay cell. */
inline void eulerianPositionMatrix(Cell_handle &cell,
                                   Real posMatrixInverse[][NO_DIM])
{
    double Ax[NO_DIM][NO_DIM];
    for (int v = 0; v < NO_DIM; ++v)
        for (int i = 0; i < NO_DIM; ++i)
            Ax[v][i] = double(cell->vertex(v+1)->info().eulerianPosition(i))
                      - double(cell->vertex(0)->info().eulerianPosition(i));
    matrixInverse(Ax, posMatrixInverse);
}

/* Tests if a point lies inside the Eulerian simplex of a cell.
   Returns true and sets baryCoords if the point is inside.
   baryCoords[i] = barycentric coordinate for vertex i+1 (vertex 0's coord = 1 - sum). */
inline bool pointInEulerianSimplex(Cell_handle &cell,
                                   Real const *point,
                                   Real posMatrixInverse[][NO_DIM],
                                   Real *baryCoords)
{
    // Compute relative position: point - eulerianPos(vertex 0)
    Real rel[NO_DIM];
    for (int i = 0; i < NO_DIM; ++i)
        rel[i] = point[i] - cell->vertex(0)->info().eulerianPosition(i);

    // Multiply by inverse position matrix to get barycentric coordinates
    matrixMultiplication(posMatrixInverse, rel, baryCoords);

    // Check if all barycentric coords are non-negative and sum <= 1
    Real sum = 0.;
    for (int i = 0; i < NO_DIM; ++i)
    {
        if (baryCoords[i] < Real(-1.e-6)) return false;
        sum += baryCoords[i];
    }
    return (sum <= Real(1. + 1.e-6));
}
#endif

/* Computes the vertex position difference matrix for a given Delaunay cell. */
inline void vertexPositionMatrix(Cell_handle &cell,
                                 Real vertexMatrix[][NO_DIM])
{
    Point base = cell->vertex(0)->point();
    // Now store in 'vertexMatrix' the vertices position with respect to the 'base' vertex
    for (int v = 0; v<NO_DIM; ++v)	//loop over vertices != base
        for (int i=0; i<NO_DIM; ++i)	//loop over spatial dimensions
            vertexMatrix[v][i] = cell->vertex(v+1)->point()[i] - base[i];
}



/* Checks the number of vertices of the Delaunay triangulation. The interpolation computation cannot continue if there are no cellls, so it gives a warning message to the user and initializes all the values to 0. */
template <typename T> inline void assingZeroValues(vector<T> *quant, size_t const noElements )
{
    quant->assign( noElements, T::zero() );
}
template <> inline void assingZeroValues<Real>(vector<Real> *quant, size_t const noElements )
{
    quant->assign( noElements, Real(0.) );
}
template <typename T>
inline bool isTriangulationIncomplete(DT &dt,
                                      size_t gridSize,
                                      vector<T> *quant,
                                      string quantityName,
                                      bool showMessage = true)
{
    if ( dt.number_of_vertices()<NO_DIM+1 )
    {
        if ( showMessage )
        {
            MESSAGE::Warning warning(1);
            warning << "Because there are less than " << NO_DIM+1 << " vertices in the Delaunay triangulation there is no cell and hence there is no information that can be used to interpolate the '" << quantityName << "'. All '" << quantityName << "' values will be initialized to 0.\n" << MESSAGE::EndWarning;
        }
        assingZeroValues( quant, gridSize );
        return true;
    }
    return false;
}



/* Reserves memory for an output field and initializes all the elements to 0 if the Delaunay traingulation is not defined. */
template <typename T>
inline void reserveMemory(T *fieldStorage,
                          size_t const gridSize,
                          bool fieldToBeComputed,
                          DT &dt,
                          string quantityName)
{
    if ( fieldToBeComputed )
    {
        fieldStorage->clear();
        fieldStorage->reserve( gridSize );
        isTriangulationIncomplete( dt, gridSize, fieldStorage, quantityName );
    }
}




/* Checks the values of the angles to give correct interval values and transforms the angles from degrees to radians. */
inline void checkAngles(Real *angles,
                 size_t const size,
                 Real *offset = NULL)
{
    if ( size!=2*(NO_DIM-1) ) throwError( "The function 'checkAngles' in 'density_interpolation.cc' must have the 2nd argument = ", 2*(NO_DIM-1), ", but now that argument is ", size, "." );
    
    // rescale the angles to the interval 0 - 360 (only psi) and theta in the interval -90 - 90
#if NO_DIM==2
    angles[0] -= floor( angles[0]/Real(360.) ) * Real(360.);
    angles[1] -= floor( angles[1]/Real(360.) ) * Real(360.);
#elif NO_DIM==3
    intervalCheck( angles[0], Real(0.), Real(180.), "lower limit of the 'theta' spherical coordinate angle" );
    intervalCheck( angles[1], Real(0.), Real(180.), "upper limit of the 'theta' spherical coordinate angle" );
    angles[2] -= floor( angles[2]/Real(360.) ) * Real(360.);
    angles[3] -= floor( angles[3]/Real(360.) ) * Real(360.);
#endif
    
    // if psi_min>psi_max => psi_max += 360
    if ( offset!=NULL )
{
    offset[0] = Real(0.);
    offset[1] = Real(0.);
}
    if ( angles[size-2]>angles[size-1] )
{
    angles[size-1] += Real(360.);
    if ( offset!=NULL )
    {
        offset[0] = angles[size-2] / Real(2.);
        offset[1] = Real(2.*PI);
    }
}
    
    
    // transform the degrees to radians
    for (size_t i=0; i<size; ++i)
        angles[i] *= Real(PI/180.);
}



/* Computes the area/volume of a Delaunay triangle/tetrahedron in 2D/3D. */
template <typename DTCell>
inline Real volume(DT & dt,
                    DTCell &cell)
{
#if NO_DIM==2
    return dt.triangle( cell ).area();
#elif NO_DIM==3
    return dt.tetrahedron( cell ).volume();
#endif
}



/* Returns the index of the specified grid cell. */
template <typename T>
inline size_t gridCellIndex(T index1, T index2, T index3,
                            T *totalGridSize)
{
    T idx[] = {index1, index2, index3};
    size_t result = 0;
    for (int d=0; d<NO_DIM; ++d)
        result = result * totalGridSize[d] + idx[d];
    return result;
}
template <typename T>
inline size_t gridCellIndex(T *index,
                            size_t *totalGridSize)
{
    size_t result = 0;
    for (int d=0; d<NO_DIM; ++d)
        result = result * totalGridSize[d] + index[d];
    return result;
}



/* Returns the standard deviation of a set of numbers/vectors. */
template <typename T>
inline T standardDeviation(T *data, size_t const size)
{
    T mean = T(0.), result = T(0.);
    for (size_t i=0; i<size; ++i)
        mean += data[i];
    mean /= size;
    for (size_t i=0; i<size; ++i)
    {
        T temp = data[i] - mean;
        result += temp*temp;
    }
    return sqrt( result/size );
}
template <typename T, size_t N>
inline T standardDeviation(Pvector<T,N> *data, size_t const size)
{
    Pvector<T,N> mean = Pvector<T,N>::zero();
    T result = T(0.);
    for (size_t i=0; i<size; ++i)
        mean += data[i];
    mean /= size;
    for (size_t i=0; i<size; ++i)
    {
        Pvector<T,N> temp = data[i] - mean;
        for (size_t j=0; j<N; ++j)
            result += temp[j]*temp[j];
    }
    return sqrt( result/size );
}

#endif
