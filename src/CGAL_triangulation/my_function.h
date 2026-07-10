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



/* User extension point: define your own quantity here for DTFE to interpolate/volume-average. */


#ifdef SCALAR

// Uncomment to route the 'scalar' field through personalizedFunction() below instead of the raw scalar.
//#define MY_SCALAR


// Computes a custom quantity at 'pointPosition' into 'scalar' (selected via '--field scalar / scalar_a').
// Inputs are the cell-interpolated density, densityGradient, velocity, velocityGradient; set the scalar
// component count with '-DNO_SCALARS' (noScalarComp) in the Makefile.
void inline personalizedFunction(Point &pointPosition,
                                 Real density,
                                 Real *densityGradient,
                                 Pvector<Real,noVelComp> &velocity,
                                 Real velocityGradient[][noVelComp],
                                 Pvector<Real,noScalarComp> &scalar)
{
    // Example only; velocityGradient[i][j] = d(v_j)/d(x_i), axes 0=x, 1=y, 2=z.
    scalar[0] = density * densityGradient[1];
    scalar[1] = densityGradient[2] * velocity[2];
}

#endif
