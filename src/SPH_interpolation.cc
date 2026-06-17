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


/* Smoothed-Particle-Hydrodynamics (SPH) interpolation onto a regular grid, user sample points, or a
   redshift cone. Uses an adaptive smoothing length (N-th neighbor) and a symmetric cubic-spline kernel. */


#include <vector>
#include <cmath>

#include "kdtree/kdtree2.hpp"  // uses the boost multi_array library

#include "define.h"
#include "particle_data.h"
#include "quantities.h"
#include "box.h"
#include "user_options.h"
#include "message.h"
#include <boost/timer.hpp>

void printElapsedTime(boost::timer *t, User_options *userOptions,
                      std::string computationQuantityName);




void SPH_interpolation(vector<Particle_data> &p,
                        User_options &userOptions,
                        kdtree2_array &gridPoints,
                        size_t const totalGrid,
                        Quantities *q);




// SPH (smoothed particle hydrodynamics) interpolation: builds the grid sample points, then interpolates.
void SPH_interpolation(vector<Particle_data> *particles,
                       vector<Sample_point> &samples,
                       User_options &userOptions,
                       Quantities *q)
{
    // compute the positions of the grid sampling points
    size_t totalGrid = 0;
    if ( not samples.empty() )
        totalGrid = samples.size();
    else
    {
        size_t const *grid = &(userOptions.gridSize[0]);
        totalGrid = 1;
        for (int d=0; d<NO_DIM; ++d) totalGrid *= grid[d];
    }
    
    kdtree2_array gridPoints(extents[totalGrid][NO_DIM]);
    if ( not samples.empty() )      // user defined sample points
    {
        for (size_t i=0; i<totalGrid; ++i)
            for (size_t j=0; j<NO_DIM; ++j)
                gridPoints[i][j] = samples[i].position(j);
    }
    else if ( not userOptions.redshiftConeOn )     // sample points on a rectangular grid
    {
        size_t const *grid = &(userOptions.gridSize[0]);
        Box box = userOptions.region;
        Real y[NO_DIM];
        Real dy[NO_DIM];
        for (size_t i=0; i<NO_DIM; ++i)
            dy[i] = (box[2*i+1] - box[2*i]) / grid[i];
        
        for (size_t flatIdx=0; flatIdx<totalGrid; ++flatIdx)
        {
            size_t gridIdx[NO_DIM], rem = flatIdx;
            for (int d=NO_DIM-1; d>=0; --d) { gridIdx[d] = rem % grid[d]; rem /= grid[d]; }
            for (size_t j=0; j<NO_DIM; ++j)
                gridPoints[flatIdx][j] = box[2*j] + (gridIdx[j]+0.5)*dy[j];
        }
    }
    else if ( userOptions.redshiftConeOn )     // sample points on a light cone grid
    {
        size_t const *grid = &(userOptions.gridSize[0]);
        Box box = userOptions.redshiftCone;
        Real const *origin = &(userOptions.originPosition[0]);
        Real dx[NO_DIM];
        for (size_t i=0; i<NO_DIM; ++i)
            dx[i] = (box[2*i+1] - box[2*i]) / grid[i];
        
        size_t index = 0;
        Real r = box[0] + dx[0]/2.;
        for (size_t i=0; i<grid[0]; ++i)
        {
            Real theta = box[2] + dx[1]/2.;
            for (size_t j=0; j<grid[1]; ++j)
            {
#if NO_DIM==2
                gridPoints[index][0] = origin[0] + r*std::cos(theta);
                gridPoints[index++][1] = origin[1] + r*std::sin(theta);
#elif NO_DIM==3
                Real phi = box[4] + dx[2]/2.;
                for (size_t k=0; k<grid[2]; ++k)
                {
                    gridPoints[index][0] = origin[0] + r*sin(theta)*cos(phi);
                    gridPoints[index][1] = origin[1] + r*sin(theta)*sin(phi);
                    gridPoints[index++][2] = origin[2] + r*cos(theta);
                    phi += dx[2];
                }
#endif
                theta += dx[1];
            }
            r += dx[0];
        }
    }
    
    
    
    SPH_interpolation( *particles, userOptions, gridPoints, totalGrid, q );
    particles->clear();
}



// SPH smoothing kernel value, without the h^-2 (2D) / h^-3 (3D) normalization factor.
Real SPH_smoothingKernel(Real x)
{
#if NO_DIM==2
    Real factor = Real(10./(7.*PI));
#elif NO_DIM==3
    Real factor = Real(1./PI);
#endif
    
    if ( std::fabs(x)<=1. )
        return factor * Real(1. - 1.5*x*x + 0.75*x*x*x);
    else if ( std::fabs(x)<=2. )
    {
        Real temp = Real( 2.-x );
        return factor * Real(0.25 * temp*temp*temp);
    }
    else
        return Real(0.);
}

// SPH smoothing kernel derivative, without the h^-2 (2D) / h^-3 (3D) normalization factor.
Real SPH_smoothingKernelDerivative(Real x)
{
#if NO_DIM==2
    Real factor = Real(10./(7.*PI));
#elif NO_DIM==3
    Real factor = Real(1./PI);
#endif
    
    if ( std::fabs(x)<=1. )
        return factor * Real(-3.*x + 2.25*x*x);
    else if ( std::fabs(x)<=2. )
    {
        Real temp = Real( 2.-x );
        return factor * Real(-0.75 * temp*temp);
    }
    else
        return Real(0.);
}


// Normalization constant in front of W(r,h); depends only on h.
inline Real hFactor(Real h)
{
    Real result = Real(1.);
    for (int d=0; d<NO_DIM; ++d) result /= h;
    return result;
}


// SPH vector derivative: factor * vec * (r_vec/r), with factor = 1/2 m W_deriv(r,h).
template <typename T, size_t Nvector, size_t Nderiv>
Pvector<T,Nderiv> getSPH_derivative(T const factor,
                                    Pvector<T,Nvector> &vec,
                                    T *pos1,
                                    T *pos2)
{
    Pvector<T,Nderiv> result;
    T vecR[NO_DIM], res1 = T(0.), res2 = T(0.);
    for (int i=0; i<NO_DIM; ++i)
    {
        vecR[i] = pos1[i] - pos2[i];
        res1 += pos1[i]*pos1[i];
        res2 += vecR[i]*vecR[i];
    }
    
    T res = std::sqrt( res2 );
    if ( res2/res1<T(1.e-6) )   // if res2==0
        for (int i=0; i<NO_DIM; ++i)
            vecR[i] = T(0.);
    else
        for (int i=0; i<NO_DIM; ++i)
            vecR[i] /= res;
        
    for (int i=0; i<Nvector; ++i)
        for (int j=0; j<NO_DIM; ++j)
            result[i*NO_DIM+j] = factor * vec[i] * vecR[j];
    return result;
}




// SPH interpolation to a grid. Density uses the symmetric kernel
// rho = sum_j m_j (1/2)[W(r_ij,h_i) + W(r_ij,h_j)], h = R/2 (R = distance to N-th neighbor).
// Steps: (1) per-particle h and density, (2) per-grid-point h and the W(.,h_grid) half,
// (3) scatter the W(.,h_i) half from each particle to nearby grid points via a kdtree.
// Algorithm: http://ciera.northwestern.edu/StarCrash/manual/html/node7.html
void SPH_interpolation(vector<Particle_data> &p,
                        User_options &userOptions,
                        kdtree2_array &gridPoints,
                        size_t const totalGrid,
                        Quantities *q)
{
    MESSAGE::Message message(userOptions.verboseLevel);
    message << "\nInterpolating the fields to a grid using the SPH method with " << userOptions.SPH_neighbors << " neighbors. ";
    if ( userOptions.gridSize.empty() )
        message << "The interpolation takes place at " << totalGrid << " user defined sampling points:\n" << MESSAGE::Flush;
    else if ( not userOptions.redshiftConeOn )
        message << "The interpolation takes place inside the box of coordinates " << userOptions.region.print() << " on a " << MESSAGE::printElements( userOptions.gridSize, "*" ) << " grid:\n" << MESSAGE::Flush;
    else if ( userOptions.redshiftConeOn )
        message << "The interpolation takes place inside the spherical coordinates " << userOptions.region.print() << " on a " << MESSAGE::printElements( userOptions.gridSize, "*" ) << " grid:\n" << MESSAGE::Flush;
    
    
    message << "Computing the kdtree for the SPH interpolation ... " << MESSAGE::Flush;
    boost::timer t;
    t.restart();
    size_t const noParticles = p.size();
    kdtree2_array dataPoints(extents[noParticles][NO_DIM]);
    for (size_t i=0; i<noParticles; ++i)
        for (size_t j=0; j<NO_DIM; ++j)
            dataPoints[i][j] = p[i].position(j);
    kdtree2* tree;
    tree = new kdtree2( dataPoints, false );
    tree->sort_results = true;
    
    message << "Done.\n" << MESSAGE::Flush;
    printElapsedTime( &t, &userOptions, "kdtree construction" );
    
    
    // construct a table with the values of the smoothing kernel
    int const NTable = 1000;
    Real dx = Real( 2. / NTable);
    Real W[NTable+1], W_deriv[NTable+1];
    for (int i=0; i<=NTable; ++i)
    {
        W[i] = SPH_smoothingKernel( dx*(i+0.5) );
        W_deriv[i] = SPH_smoothingKernelDerivative( dx*(i+0.5) );
    }
    
    
    // find the smoothing length and SPH density associated to each particle
    message << "Computing the smoothing scale and density at each particle position.\n\tDone:  " << MESSAGE::Flush;
    t.restart();
    kdtree2_result_vector result;   // nearest-neighbor query results
    int N = int( userOptions.SPH_neighbors );
    std::vector<Real> smoothingLength( noParticles, Real(0.) ); // smoothing length per particle
    std::vector<Real> density( noParticles, Real(0.) );         // density at each particle position
    Real *h = &(smoothingLength[0]);
    Real *d = &(density[0]);
    size_t prev = 0, amount100 = 0;
    for (size_t i=0; i<noParticles; ++i)
    {
        amount100 = (100 * i)/ noParticles;
        if (prev < amount100)
            message.updateProgress( ++prev );

        tree->n_nearest_around_point(i,0,N,result); // neighbors in ascending distance order
        h[i] = Real(std::sqrt( result[N-1].dis ) / 2.); // h chosen so 2h contains N neighbors

        // add only the W(r,h_i) half of the symmetric kernel here
        Real const c1 = hFactor(h[i]);
        for (size_t j=0; j<result.size(); ++j)
        {
            Real const temp = std::sqrt( result[j].dis );
            int const id = result[j].idx;
            int bin1 = int( temp / (h[i]*dx));
            
            d[i] += Real(0.5) * p[id].weight() * c1*W[bin1];
            d[id] += Real(0.5) * p[i].weight() * c1*W[bin1];
        }
    }
    message << "100%\n" << MESSAGE::Flush;
    printElapsedTime( &t, &userOptions, "SPH particle density" );
    
    
    
    message << "Computing the interpolated fields on the grid.\n\tDone:  " << MESSAGE::Flush;
    t.restart();
    q->density.reserve( totalGrid );    // density is always needed
    if ( userOptions.aField.velocity ) q->velocity.reserve( totalGrid );
    if ( userOptions.aField.velocity_gradient ) q->velocity_gradient.reserve( totalGrid );
    if ( userOptions.aField.scalar ) q->scalar.reserve( totalGrid );
    if ( userOptions.aField.scalar_gradient ) q->scalar_gradient.reserve( totalGrid );
    
    
    // per-grid-point smoothing scale + the W(.,h_grid) half of the kernel
    std::vector<float> y( NO_DIM, float(0.) );
    prev = 0; amount100 = 0;
    for (size_t i=0; i<totalGrid; ++i)
    {
        amount100 = (50 * i) / totalGrid;
        if (prev < amount100)
            message.updateProgress( ++prev );
        
        for (size_t j=0; j<NO_DIM; ++j)
            y[j] = gridPoints[i][j];
        tree->n_nearest( y, N, result );
        Real const tempH = Real(std::sqrt( result[N-1].dis ) / 2.);
        Real const c1 = hFactor(tempH);
        
        // accumulators for the different quantities
        Real resDens = Real(0.);
        Pvector<Real,noVelComp> resVel = Pvector<Real,noVelComp>::zero();
        Pvector<Real,noScalarComp> resIntensive = Pvector<Real,noScalarComp>::zero();
        Pvector<Real,noScalarComp> resExtensive = Pvector<Real,noScalarComp>::zero();

        for (size_t j=0; j<result.size(); ++j)
        {
            int const id = result[j].idx;
            Real tempR = std::sqrt( result[j].dis );
            int bin1 = int( tempR / (tempH*dx));

            Real temp1 = Real(0.5) * p[id].weight() * c1*W[bin1];
            resDens += temp1;
            resVel += p[id].velocity() * temp1;
            resIntensive += p[id].scalar() * temp1;
            resExtensive += p[id].scalar() * temp1 / d[id];
        }

        q->density.push_back( resDens );
        if ( userOptions.aField.velocity ) q->velocity.push_back( resVel );
        if ( userOptions.extensive )
        {
            if ( userOptions.aField.scalar ) q->scalar.push_back( resExtensive );
        }
        else
        {
            if ( userOptions.aField.scalar ) q->scalar.push_back( resIntensive );
        }
    }

    // build the kdtree over the grid points
    delete tree;
    tree = new kdtree2( gridPoints, false );
    tree->sort_results = true;
    for (size_t i=0; i<noParticles; ++i)
    {
        amount100 = 50 + (50 * i) / noParticles;
        if (prev < amount100)
            message.updateProgress( ++prev );
        
        for (size_t j=0; j<NO_DIM; ++j)
            y[j] = p[i].position(j);
        Real const distance = Real(4.) * h[i]*h[i]; //(2h)^2
        tree->r_nearest( y, distance, result );    // grid points within the particle's smoothing radius

        Real const c1 = Real(0.5) * p[i].weight() * hFactor(h[i]);
        for (size_t j=0; j<result.size(); ++j)
        {
            int const id = result[j].idx;
            Real y2[NO_DIM];
            for (int i1=0; i1<NO_DIM; ++i1)
                y2[i1] = gridPoints[id][i1];
            Real tempR = std::sqrt( result[j].dis );
            int bin1 = int( tempR / (h[i]*dx));

            Real temp1 = c1*W[bin1];
            q->density[id] += temp1;
            if ( userOptions.aField.velocity ) q->velocity[id] += p[i].velocity() * temp1;
            if ( userOptions.extensive )
            {
                if ( userOptions.aField.scalar ) q->scalar[id] += p[i].scalar() * temp1;
            }
            else
            {
                if ( userOptions.aField.scalar ) q->scalar[id] += p[i].scalar() * temp1 / d[i];
            }
        }
    }
    delete tree;


    // velocity = accumulated momentum / cell density
    if ( userOptions.aField.velocity )
        for (size_t i=0; i<totalGrid; ++i)
            q->velocity[i] /= q->density[i];

    // intensive scalar field: divide by cell density
    if ( userOptions.aField.scalar and not userOptions.extensive )
        for (size_t i=0; i<totalGrid; ++i)
            q->scalar[i] /= q->density[i];

    // density: apply the normalization factor
    if ( userOptions.aField.density )
    {
        Real factor = Real( 1. / userOptions.averageDensity );    //normalization factor for the density
        for (std::vector<Real>::iterator it=q->density.begin(); it!=q->density.end(); ++it)
            (*it) *= factor;
    }
    else q->density.clear();
    
    
    message << "100%\n" << MESSAGE::Flush;
    printElapsedTime( &t, &userOptions, "SPH grid interpolation" );
}

