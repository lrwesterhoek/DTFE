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


/*
  Fixed-size physical vector of N components, indexed via [] (0..N-1). Supports +, +=, -, -=,
  * and / by a scalar, ==, != and zero(). == is exact equality (no tolerance for float/double).
*/


#ifndef PVECTOR_HEADER
#define PVECTOR_HEADER

// Fixed-size N-component vector of T with value semantics; the workhorse type for positions, velocities and tensors.
template <typename T, size_t N>
struct Pvector
{
    T x[N];

    Pvector() {
        for (size_t i=0; i<N; ++i)
            x[i] = T(0); }

#if NO_DIM==2
    Pvector(T x1, T x2) {
        x[0] = x1;
        x[1] = x2;}
#elif NO_DIM==3
    Pvector(T x1, T x2, T x3) {
        x[0] = x1;
        x[1] = x2;
        x[2] = x3;}
#endif
    Pvector(T *xVector) {
        for (size_t i=0; i<N; ++i)
            x[i] = xVector[i]; }

    T& operator[](size_t i)
    {
        return x[i];
    }
    const T& operator[](size_t i) const
    {
        return x[i];
    }


    Pvector operator +(const Pvector<T,N> &other) const
    {
        Pvector<T,N> result;
        for (size_t i=0; i<N; ++i )
            result.x[i] = x[i] + other.x[i];
        return result;
    }
    Pvector& operator +=(const Pvector<T,N> &other)
    {
        for (size_t i=0; i<N; ++i )
            x[i] += other.x[i];
        return *this;
    }
    Pvector operator -(const Pvector<T,N> &other) const
    {
        Pvector<T,N> result;
        for (size_t i=0; i<N; ++i )
            result.x[i] = x[i] - other.x[i];
        return result;
    }
    Pvector& operator -=(const Pvector<T,N> &other)
    {
        for (size_t i=0; i<N; ++i )
            x[i] -= other.x[i];
        return *this;
    }
    Pvector operator *(T other) const
    {
        Pvector<T,N> result;
        for (size_t i=0; i<N; ++i )
            result.x[i] = x[i] * other;
        return result;
    }
    Pvector& operator *=(T other)
    {
        for (size_t i=0; i<N; ++i )
            x[i] *= other;
        return *this;
    }
    Pvector operator /(T other) const
    {
        Pvector<T,N> result;
        for (size_t i=0; i<N; ++i )
            result.x[i] = x[i] / other;
        return result;
    }
    Pvector& operator /=(T other)
    {
        for (size_t i=0; i<N; ++i )
            x[i] /= other;
        return *this;
    }

    bool operator ==(const Pvector<T,N> &other) const
    {
        for (size_t i=0; i<N; ++i )
            if ( x[i] != other.x[i] )
                return false;
        return true;
    }
    bool operator !=(const Pvector<T,N> &other) const
    {
        return !( (*this)==other );
    }


    // Returns an all-zero vector; handy as a fill value for std::vector::assign/resize.
    static Pvector zero()
    {
        Pvector<T,N> result;
        for (size_t i=0; i<N; ++i )
            result.x[i] = T(0.);
        return result;
    }
};



#endif
