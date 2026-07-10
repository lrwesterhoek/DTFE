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
  Data_structure (particle data minus position), Particle_data (position + Data_structure),
  Sample_point (user-supplied points to interpolate at). To add properties, modify "noScalarComp".
*/


#ifndef PARTICLE_DATA_HEADER
#define PARTICLE_DATA_HEADER

#include "define.h"
#include "Pvector.h"




// Particle properties except position; split off so they can be carried on the triangulation vertices.
class Data_structure
{
    protected:
        Real                       _weight; // particle weight
        Real                       _density;// filled later by DTFE interpolation at each vertex
#ifdef VELOCITY
    Pvector<Real,noVelComp>    _velocity; // particle velocity
#else
    static Pvector<Real,noVelComp> _velocity; // static: shared instance, no per-particle cost
#endif
#ifdef SCALAR
    Pvector<Real,noScalarComp> _scalar; // extra particle data; size set by "noScalarComp"
#else
    static Pvector<Real,noScalarComp> _scalar; // static: shared instance, no per-particle cost
#endif
    
    public:
    Data_structure()
    {
        _weight = Real(1.);
        _density = Real(0.);
#ifdef VELOCITY
        _velocity = Pvector<Real,noVelComp>::zero();
#endif
#ifdef SCALAR
        _scalar = Pvector<Real,noScalarComp>::zero();
#endif
    }

    inline Real& weight() { return _weight;}
    inline void setWeight(Real const w) { _weight = w;}
    inline Real& density() { return _density;}
    inline void setDensity(Real const d) { _density = d;}
    inline Pvector<Real,noVelComp>& velocity() { return _velocity;}
    inline Real& velocity(int const i) { return _velocity[i];}
    inline void setVelocity(Real * vel) { for(size_t j=0; j<noVelComp; ++j) _velocity[j] = vel[j];}
    inline Pvector<Real,noScalarComp>& scalar() { return _scalar;}
    inline Real& scalar(int const i) { return _scalar[i];}
    inline void setScalar(Real * s) { for(size_t j=0; j<noScalarComp; ++j) _scalar[j] = s[j];}
};



// Carries a particle's position and data between program units. Member must stay named 'pos' or it won't compile.
struct Particle_data : public Data_structure
{
    Pvector<Real,NO_DIM>   pos;    // Eulerian position
#ifdef PHASE_SPACE
    Pvector<Real,NO_DIM>   lagPos; // Lagrangian (initial) position
#endif

    inline Pvector<Real,NO_DIM>& position() { return pos;}
    inline Real& position(int const i) { return pos[i];}
    inline void setPosition(Real *p) { for(size_t j=0; j<NO_DIM; ++j) pos[j] = p[j];}
#ifdef PHASE_SPACE
    inline Pvector<Real,NO_DIM>& lagrangianPosition() { return lagPos;}
    inline Real& lagrangianPosition(int const i) { return lagPos[i];}
    inline void setLagrangianPosition(Real *p) { for(size_t j=0; j<NO_DIM; ++j) lagPos[j] = p[j];}
#endif
};



// User-supplied sample point for interpolating fields at arbitrary locations.
struct Sample_point
{
    Pvector<Real,NO_DIM>    pos;    // sample point position
    Pvector<Real,NO_DIM>    _delta; // grid cell size per axis; density interpolation only (cell averaged to filter Poisson noise)
    
    inline Pvector<Real,NO_DIM>& position() { return pos;}
    inline Real& position(int const i) { return pos[i];}
    inline void setPosition(Real *p) { for(size_t j=0; j<NO_DIM; ++j) pos[j] = p[j];}
    inline Pvector<Real,NO_DIM>& delta() { return _delta;}
    inline Real& delta(int const i) { return _delta[i];}
    inline void setDelta(Real *p) { for(size_t j=0; j<NO_DIM; ++j) _delta[j] = p[j];}
};


#endif
