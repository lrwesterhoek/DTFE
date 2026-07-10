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
  Holds every quantity the DTFE grid interpolation can compute, each in its own vector so memory
  is allocated only for requested quantities. Pvector ("Pvector.h"); component counts in "define.h".
*/



#ifndef QUANTITIES_HEADER
#define QUANTITIES_HEADER

#include <vector>

#include "define.h"
#include "particle_data.h"
#include "user_options.h"




// Container for every interpolated grid field; each vector is allocated only when its field is requested.
struct Quantities
{
    std::vector<Real>                         density;
    std::vector< Pvector<Real,noVelComp> >    velocity;
    std::vector< Pvector<Real,noGradComp> >   velocity_gradient;
    std::vector<Real>                         velocity_divergence;
    std::vector< Pvector<Real,noShearComp> >  velocity_shear;
    std::vector< Pvector<Real,noVortComp> >   velocity_vorticity;
    std::vector<Real>                         velocity_std;         // velocity standard deviation
    std::vector< Pvector<Real,noDispComp> >   velocity_dispersion;  // dispersion tensor sigma_ij (symmetric, upper-triangle row-major; trace = dispersion scalar)
    std::vector< Pvector<Real,noScalarComp> > scalar;
    std::vector< Pvector<Real,noScalarGradComp> > scalar_gradient;
    std::vector<Real>                         velocity_tweb;        // T-web classification label (0=void, 1=wall, 2=filament, 3=node)
    std::vector< Pvector<Real,NO_DIM> >       velocity_tweb_eigenvalues; // T-web eigenvalues (sorted descending)
    std::vector<Real>                         velocity_vweb;        // V-web classification label (0=void, 1=wall, 2=filament, 3=node)
    std::vector< Pvector<Real,NO_DIM> >       velocity_vweb_eigenvalues; // V-web eigenvalues (sorted descending)
#ifdef PHASE_SPACE
    std::vector<Real>                         stream_count;       // number of streams at each grid point
    std::vector<Real>                         mass_weight;        // per-cell summed stream density sum(rho_s); internal to normalizePhaseSpace, never written out
    // PS-DTFE partition sub-grid in global grid-cell units, so addFromSubgrid() can map cells back.
    size_t ps_subOrigin[NO_DIM] = {0};
    size_t ps_subDims[NO_DIM]   = {0};   // [0]==0 means "not a sub-grid" (vectors span the full grid)
#endif

    // if you add members, update the functions below so the 'partition' option keeps working
    void copyFromSubgrid(Quantities const &subgridResults,
                        Field const &field,
                        std::vector<size_t> const &mainGrid,
                        std::vector<size_t> const &subgrid,
                        std::vector<size_t> const &subgridOffset); // copy subgrid ('partition') results into the full grid
    size_t size() const;	// size of any non-empty member vector
    void reserveMemory(size_t *gridSize, Field &field); // reserve main-grid memory for the 'partition' option
    void addFrom(Quantities const &other); // element-wise accumulate from another Quantities (PS-DTFE partitioning)
#ifdef PHASE_SPACE
    // Turn summed density-weighted moments into mass-weighted means (call once, after all
    // partitions added). When the partitions aliased the weight to the density grid (density
    // field selected -> no mass_weight was filled), pass weightFromDensityScale =
    // cellVolume * averageDensity so the per-cell mass is reconstructed from the summed
    // (rho/rho_bar) density; 0 keeps the historical mass_weight-only behaviour.
    void normalizePhaseSpace(Field const &field, Real const weightFromDensityScale = Real(0.));
    void addFromSubgrid(Quantities const &other, size_t const *fullGrid); // like addFrom but 'other' stores only its Eulerian box; maps each cell into the full grid (dims fullGrid)
#endif
};


#endif
