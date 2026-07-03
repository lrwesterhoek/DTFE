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


/* Inline numeric helpers shared by the triangulation/interpolation code: timing, grid indexing,
   angle normalization, simplex volumes, and (under PHASE_SPACE) Eulerian-simplex geometry. */

#ifndef TRIANGULATION_MISCELLANEOUS_HEADER
#define TRIANGULATION_MISCELLANEOUS_HEADER

#include "../math_functions.h"


// Report elapsed time for a computation stage.
inline void printComputationTime(Timer *t, User_options *userOptions,
                                 string computationQuantityName)
{
    t->stop();
    userOptions->totalTime += t->time();
    MESSAGE::Message message( userOptions->verboseLevel );
    // wall-clock timer: report seconds directly, must not divide by noProcessors
    message << MESSAGE::cGreen() << "  >>> Time: " << t->time() << " sec. (" << computationQuantityName << ")" << MESSAGE::cReset() << "\n" << MESSAGE::Flush;
    t->reset();
}



// matrixInverse and determinant are defined in math_functions.h

#ifdef PHASE_SPACE
// Returns the volume of a cell's Eulerian (deformed) simplex. The triangulation lives in Lagrangian
// space, but the density estimate needs the present-day Eulerian volume.
inline Real eulerianVolume(const Cell_handle &cell)
{
    double posDiff[NO_DIM][NO_DIM];
    for (int v = 0; v < NO_DIM; ++v)
        for (int i = 0; i < NO_DIM; ++i)
            posDiff[v][i] = double(cell->vertex(v+1)->info().eulerianPosition(i))
                          - double(cell->vertex(0)->info().eulerianPosition(i));
    return simplexVolume(posDiff);
}
// NOTE: the per-cell Eulerian edge-matrix inverse and point-in-simplex test used by PS-DTFE live
// inline in ps_interpolation.cc (interpolateGrid_phaseSpace), which correctly transposes the inverse
// for the barycentric coordinates. A previous standalone pointInEulerianSimplex/eulerianPositionMatrix
// pair here used the NON-transposed product (incorrect) and had no call sites; removed to avoid reuse.
#endif

// Vertex position difference matrix (each non-base vertex relative to vertex 0) for a Delaunay cell.
inline void vertexPositionMatrix(Cell_handle &cell,
                                 Real vertexMatrix[][NO_DIM])
{
    Point base = cell->vertex(0)->point();
    for (int v = 0; v<NO_DIM; ++v)
        for (int i=0; i<NO_DIM; ++i)
            vertexMatrix[v][i] = cell->vertex(v+1)->point()[i] - base[i];
}



// Fill a field with zero values (Pvector and scalar Real specializations).
template <typename T> inline void assingZeroValues(vector<T> *quant, size_t const noElements )
{
    quant->assign( noElements, T::zero() );
}
template <> inline void assingZeroValues<Real>(vector<Real> *quant, size_t const noElements )
{
    quant->assign( noElements, Real(0.) );
}
// Returns true (and zeroes the field) if the triangulation has too few vertices to form any cell.
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



// Reserve memory for an output field; zero it if the triangulation is undefined.
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




// Normalize angle limits to valid intervals and convert degrees to radians.
inline void checkAngles(Real *angles,
                 size_t const size,
                 Real *offset = NULL)
{
    if ( size!=2*(NO_DIM-1) ) throwError( "The function 'checkAngles' in 'density_interpolation.cc' must have the 2nd argument = ", 2*(NO_DIM-1), ", but now that argument is ", size, "." );

    // Wrap psi (azimuth) into [0,360); theta (polar) must already lie in [0,180].
#if NO_DIM==2
    angles[0] -= floor( angles[0]/Real(360.) ) * Real(360.);
    angles[1] -= floor( angles[1]/Real(360.) ) * Real(360.);
#elif NO_DIM==3
    intervalCheck( angles[0], Real(0.), Real(180.), "lower limit of the 'theta' spherical coordinate angle" );
    intervalCheck( angles[1], Real(0.), Real(180.), "upper limit of the 'theta' spherical coordinate angle" );
    angles[2] -= floor( angles[2]/Real(360.) ) * Real(360.);
    angles[3] -= floor( angles[3]/Real(360.) ) * Real(360.);
#endif

    // If psi_min > psi_max the range wraps past 360; lift psi_max by +360 and record the offset.
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


    // Convert all limits from degrees to radians.
    for (size_t i=0; i<size; ++i)
        angles[i] *= Real(PI/180.);
}



// Area (2D) or volume (3D) of a Delaunay triangle/tetrahedron.
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



// Flattens multi-dimensional grid indices into a single row-major array index.
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



// Standard deviation of a set of numbers/vectors.
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
