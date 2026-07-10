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

/* Small dense linear-algebra helpers for NO_DIM-sized matrices: determinant, inverse, products, and simplex volume. */

#ifndef MATH_FUNCTIONS_HEADER
#define MATH_FUNCTIONS_HEADER

#include <cmath>
#include <algorithm>
#include "define.h"

#define REAL_PRECISSION 1.e-6


// Determinant of an NxN matrix via Gaussian elimination with partial pivoting.
inline double determinant(double matrix[][NO_DIM])
{
    double temp[NO_DIM][NO_DIM];
    for (int i=0; i<NO_DIM; ++i)
        for (int j=0; j<NO_DIM; ++j)
            temp[i][j] = matrix[i][j];

    double det = 1.0;
    for (int i=0; i<NO_DIM; ++i)
    {
        int maxRow = i;
        double maxVal = std::fabs(temp[i][i]);
        for (int k=i+1; k<NO_DIM; ++k)
            if (std::fabs(temp[k][i]) > maxVal) { maxVal = std::fabs(temp[k][i]); maxRow = k; }

        if (maxRow != i)
        {
            for (int j=0; j<NO_DIM; ++j) std::swap(temp[i][j], temp[maxRow][j]);
            det = -det;
        }

        if (std::fabs(temp[i][i]) < 1e-15) return 0.0;

        det *= temp[i][i];

        for (int k=i+1; k<NO_DIM; ++k)
        {
            double factor = temp[k][i] / temp[i][i];
            for (int j=i+1; j<NO_DIM; ++j)
                temp[k][j] -= factor * temp[i][j];
        }
    }
    return det;
}


// Matrix inverse: closed-form for 2x2/3x3, Gauss-Jordan otherwise; zero result if (near-)singular.
inline void matrixInverse(double matrix[][NO_DIM],
                          Real result[][NO_DIM])
{
#if NO_DIM==2
    double det = matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0];
    if (std::fabs(det) < REAL_PRECISSION)
    {
        result[0][0] = result[0][1] = result[1][0] = result[1][1] = Real(0.);
        return;
    }
    double invDet = 1.0 / det;
    result[0][0] = Real( matrix[1][1] * invDet);
    result[0][1] = Real(-matrix[0][1] * invDet);
    result[1][0] = Real(-matrix[1][0] * invDet);
    result[1][1] = Real( matrix[0][0] * invDet);
#elif NO_DIM==3
    double c00 = matrix[1][1]*matrix[2][2] - matrix[1][2]*matrix[2][1];
    double c01 = matrix[1][2]*matrix[2][0] - matrix[1][0]*matrix[2][2];
    double c02 = matrix[1][0]*matrix[2][1] - matrix[1][1]*matrix[2][0];
    double det = matrix[0][0]*c00 + matrix[0][1]*c01 + matrix[0][2]*c02;
    if (std::fabs(det) < REAL_PRECISSION)
    {
        for (int r=0; r<3; ++r)
            for (int c=0; c<3; ++c)
                result[r][c] = Real(0.);
        return;
    }
    double invDet = 1.0 / det;
    result[0][0] = Real(c00 * invDet);
    result[1][0] = Real(c01 * invDet);
    result[2][0] = Real(c02 * invDet);
    result[0][1] = Real((matrix[0][2]*matrix[2][1] - matrix[0][1]*matrix[2][2]) * invDet);
    result[1][1] = Real((matrix[0][0]*matrix[2][2] - matrix[0][2]*matrix[2][0]) * invDet);
    result[2][1] = Real((matrix[0][1]*matrix[2][0] - matrix[0][0]*matrix[2][1]) * invDet);
    result[0][2] = Real((matrix[0][1]*matrix[1][2] - matrix[0][2]*matrix[1][1]) * invDet);
    result[1][2] = Real((matrix[0][2]*matrix[1][0] - matrix[0][0]*matrix[1][2]) * invDet);
    result[2][2] = Real((matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]) * invDet);
#else
    // general case: Gauss-Jordan elimination with partial pivoting
    double aug[NO_DIM][2*NO_DIM];
    double normSum = 0.0;
    for (int i=0; i<NO_DIM; ++i)
        for (int j=0; j<NO_DIM; ++j)
        {
            aug[i][j] = matrix[i][j];
            aug[i][j+NO_DIM] = (i==j) ? 1.0 : 0.0;
            normSum += std::fabs(matrix[i][j]);
        }

    for (int i=0; i<NO_DIM; ++i)
    {
        int maxRow = i;
        double maxVal = std::fabs(aug[i][i]);
        for (int k=i+1; k<NO_DIM; ++k)
            if (std::fabs(aug[k][i]) > maxVal) { maxVal = std::fabs(aug[k][i]); maxRow = k; }
        if (maxRow != i)
            for (int j=0; j<2*NO_DIM; ++j) std::swap(aug[i][j], aug[maxRow][j]);
        double pivot = aug[i][i];
        if ( not (std::fabs(pivot / normSum) > REAL_PRECISSION) )
        {
            for (int r=0; r<NO_DIM; ++r)
                for (int c=0; c<NO_DIM; ++c)
                    result[r][c] = Real(0.);
            return;
        }
        for (int j=0; j<2*NO_DIM; ++j) aug[i][j] /= pivot;
        for (int k=0; k<NO_DIM; ++k)
        {
            if (k == i) continue;
            double factor = aug[k][i];
            for (int j=0; j<2*NO_DIM; ++j)
                aug[k][j] -= factor * aug[i][j];
        }
    }
    for (int i=0; i<NO_DIM; ++i)
        for (int j=0; j<NO_DIM; ++j)
            result[i][j] = Real(aug[i][j+NO_DIM]);
#endif
}


// Matrix product result = M1 * M2 (M1 is NO_DIM x NO_DIM, M2 and result are NO_DIM x N).
template <size_t N>
inline void matrixMultiplication(Real M1[][NO_DIM],
                                 Real M2[][N],
                                 Real result[][N])
{
    for (int i=0; i<NO_DIM; ++i)
        for (int j=0; j<N; ++j)
        {
            result[i][j] = 0.;
            for (int k=0; k<NO_DIM; ++k)
                result[i][j] += M1[i][k] * M2[k][j];
        }
}

// Matrix-vector product result = M1 * M2 (M1 is NO_DIM x NO_DIM, M2 and result are NO_DIM vectors).
inline void matrixMultiplication(Real M1[][NO_DIM],
                                 Real *M2,
                                 Real *result)
{
    for (int i=0; i<NO_DIM; ++i)
    {
        result[i] = 0.;
        for (int j=0; j<NO_DIM; ++j)
            result[i] += M1[i][j] * M2[j];
    }
}


// Factorial of n.
inline double factorial(int n)
{
    double result = 1.0;
    for (int i=2; i<=n; ++i) result *= i;
    return result;
}


// Simplex volume from the vertex difference matrix: |det(M)| / NO_DIM!.
inline Real simplexVolume(double posDiff[][NO_DIM])
{
    return Real( std::fabs(determinant(posDiff)) / factorial(NO_DIM) );
}


#endif
