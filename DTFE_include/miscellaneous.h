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



/* General-purpose utilities: range/option consistency checks, array min/max, quicksort, and integer roots. */

#ifndef MISCELLANEOUS_HEADER
#define MISCELLANEOUS_HEADER

#include <string>
#include <cmath>

#include "message.h"



// Error out unless minValue <= target <= maxValue.
template <typename T1>
void intervalCheck(T1 const target,
                   T1 const minValue, T1 const maxValue,
                   std::string errorMessage)
{
    if ( target<minValue or target>maxValue )
    {
        MESSAGE::Error error;
        error << "Some program variable failed a consitency check. The variable " << errorMessage << " has the value " << target << ", but it should be between " << minValue << " to " << maxValue << " ." << MESSAGE::EndError;
    }
}
// Error out unless target >= minValue.
template <typename T1>
void lowerBoundCheck(T1 const target,
                    T1 const minValue,
                    std::string errorMessage)
{
    if ( target<minValue )
    {
        MESSAGE::Error error;
        error << "Some program variable failed a consitency check. The variable " << errorMessage << " has the value " << target << ", but it should be larger or equal than " << minValue << " ." << MESSAGE::EndError;
    }
}
// Error out unless target <= maxValue.
template <typename T1>
void upperBoundCheck(T1 const target,
                    T1 const maxValue,
                    std::string errorMessage)
{
    if ( target>maxValue )
    {
        MESSAGE::Error error;
        error << "Some program variables failed a consitency check. The variable " << errorMessage << " has the value " << target << ", but it should be smaller or equal than " << maxValue << " ." << MESSAGE::EndError;
    }
}


// The next 4 functions require the 'boost/program_options.hpp' library.
// Error out if 'opt1' and 'opt2' are both specified.
template <typename T>
void conflicting_options(const T &vm,
                        const char* opt1, const char* opt2)
{
    if (vm.count(opt1) && !vm[opt1].defaulted() && vm.count(opt2) && !vm[opt2].defaulted())
    {
        MESSAGE::Error error;
        error << "Conflicting options '" << opt1 << "' and '" << opt2 << "'!\n" << MESSAGE::EndError;
    }
}

// Error out if 'for_what' is specified without 'required_option'.
template <typename T>
void option_dependency(const T &vm,
                        const char* for_what, const char* required_option)
{
    if (vm.count(for_what) && !vm[for_what].defaulted())
        if (vm.count(required_option) == 0 || vm[required_option].defaulted())
        {
            MESSAGE::Error error;
            error << "Option '" << for_what << "' requires option '" << required_option << "'!\n" << MESSAGE::EndError;
        }
}


// Error out if 'opt1' is supplied without 'opt2'.
template <typename T>
void superfluous_options(const T &vm,
                        const char* opt1, const char* opt2)
{
    if (vm.count(opt1) && !vm[opt1].defaulted())
        if (vm.count(opt2) == 0 || vm[opt2].defaulted())
        {
            MESSAGE::Error error;
            error << "The option '" << opt1 << "' can be used only in the presence of '" << opt2 << "'. It does not make sense to use '" << opt1 << "' otherwise!\n" << MESSAGE::EndError;
        }
}
// Error out if 'opt1' is supplied while 'optionsOn' is false; 'optionsName' names the required options.
template <typename T>
void superfluous_options(const T &vm,
                        const char* opt1,
                        const bool optionsOn,
                        std::string const optionsName)
{
    if (vm.count(opt1) && !vm[opt1].defaulted())
        if ( not optionsOn )
        {
            MESSAGE::Error error;
            error << "The option '" << opt1 << "' can be used only in the presence of option/s: " << optionsName << ". It does not make sense to use '" << opt1 << "' otherwise!\n" << MESSAGE::EndError;
        }
}




// Minimum and maximum values of an array.
template <typename T, typename T_INT> inline T minimum(T *x, T_INT const size )
{
    T temp = x[0];
    for (T_INT i=1; i<size; ++i)
        if ( temp>x[i] )
            temp = x[i];
    return temp;
}
template <typename T, typename T_INT> inline T maximum(T *x, T_INT const size )
{
    T temp = x[0];
    for (T_INT i=1; i<size; ++i)
        if ( temp<x[i] )
            temp = x[i];
    return temp;
}




// In-place quicksort of values[iMin..iMax] in increasing order.
template <typename T1, typename T2>
void quicksort( T1 values[], T2 iMin, T2 iMax )
{
    if ( iMin>=iMax )
        return;

    T1 temp = values[iMax]; // pivot
    T2 i = iMin, i1=iMin, i2=iMax;
    do
    {
        if ( values[i]>temp )   // larger than pivot -> goes right
        {
            if ( i==i1 )
                values[i2] = values[i];
            --i2;
            i = i2;
        }
        else
        {
            if ( i==i2 )
                values[i1] = values[i];
            ++i1;
            i = i1;
        }
    } while ( i1!=i2 );

    values[i] = temp;   // pivot lands at the free slot i==i1==i2

    quicksort( values, iMin, i-1);
    quicksort( values, i+1, iMax);
}


// Integer rootPower-th root of input: returns result if input==result^rootPower, else terminates.
template <typename T>
T rootN(T const input,
        T const rootPower)
{
    lowerBoundCheck( input, 1, "'input' in function 'rootN'" );
    lowerBoundCheck( rootPower, 0, "'rootPower' in function 'rootN'" );

    T result;
    double temp = pow( input-1., 1./rootPower ); // root of input-1 to stay just below the integer we seek (pow rounding)
    result = T( temp ) + 1;	// integer part + 1


    // check that indeed input = result^rootPower
    T temp2 = 1;
    for (T i=0; i<rootPower; ++i)
        temp2 *= result;
    if (temp2==input)
        return result;
    else
    {
        MESSAGE::Error error;
        error << "The first argument of the function 'rootN' is " << input << " which cannot be written as " << result << "^" << rootPower << "=" << temp <<"." << MESSAGE::EndError;
    }
    return T(0.);
}

#endif

