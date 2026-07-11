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


#ifndef VERTEX_DATA_HEADER
#define VERTEX_DATA_HEADER

#include "../define.h"
#include "../particle_data.h"


/* Per-vertex Delaunay data: the particle properties the vertex represents (see "particle_data.h") plus the dummy flags below. */
struct vertexData : public Data_structure
{
    protected:
    bool   dummy;         // true if vertex is a dummy test point (padding-efficiency test)
    bool   dummyNeighbor; // true if vertex has at least one dummy neighbor (padding test for density)
#ifdef PHASE_SPACE
    Pvector<Real,NO_DIM> _eulerianPos; // Eulerian position (triangulation vertices store Lagrangian coords in PS-DTFE mode)
    uint64_t _particleID;              // snapshot ParticleID (stream identities, --per-stream-ids); +8 B/vertex, see auto_tune.h
#endif


    public:
    vertexData(){ dummy=false; dummyNeighbor=false;
#ifdef PHASE_SPACE
        _particleID=0;
#endif
    }

    // Returns the scalar field for this vertex; the single hook to customize what "scalar" means.
    inline Pvector<Real,noScalarComp> myScalar()
    {
        return scalar();
    }

    // Copies a particle's fields into this vertex (and its Eulerian position in PS-DTFE mode).
    inline void setData(Particle_data &other)
    {
        weight() = other.weight();
        density() = other.density();
#ifdef VELOCITY
        velocity() = other.velocity();
#endif
#ifdef SCALAR
        scalar() = other.scalar();
#endif
#ifdef PHASE_SPACE
        _eulerianPos = other.position();
        _particleID = other.particleID();
#endif
    }

#ifdef PHASE_SPACE
    inline Pvector<Real,NO_DIM>& eulerianPosition() { return _eulerianPos; }      // full Eulerian position
    inline Real& eulerianPosition(int const i) { return _eulerianPos[i]; }        // one Eulerian component
    inline uint64_t particleID() { return _particleID; }                          // snapshot ParticleID (0 if the reader has none)
#endif
    // remaining accessors for 'Data_structure' are in "particle_data.h"

    inline void setDummy() { dummy=true; dummyNeighbor=true; setDensity(0.); }    // mark as a dummy test point
    inline void setDummyNeighbor() { dummyNeighbor=true; }                        // mark as adjacent to a dummy
    inline bool isDummy() { return dummy; }
    inline bool hasDummyNeighbor() { return dummyNeighbor; }
};

#endif
