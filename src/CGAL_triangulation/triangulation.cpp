/*
 *  Copyright (c) 2011       Marius Cautun
 *                           Erwin Platen
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

/* Top-level DTFE driver: builds the Delaunay triangulation, estimates per-vertex
   densities, and dispatches field interpolation (standard, Voronoi, or PS-DTFE). */

#include "triangulation_common.h"

// Interpolation functions defined in separate compilation units.
void interpolateGrid(DT &dt, User_options &userOptions, Quantities *quantities);
void interpolateRedshiftCone(DT &dt, User_options &userOptions, Quantities *quantities);
void interpolateUserSampling(DT &dt, vector<Sample_point> &samples, User_options &userOptions, Quantities *quantities);
void interpolateGrid_averaged_1(DT &dt, User_options &userOptions, Quantities *quantities);
void interpolateGrid_averaged_2(DT &dt, User_options &userOptions, Quantities *quantities);
void interpolateGrid_averaged_3(DT &dt, User_options &userOptions, Quantities *quantities);
void interpolateRedshiftCone_averaged_2(DT &dt, User_options &userOptions, Quantities *quantities);
void interpolateUserSampling_averaged_2(DT &dt, vector<Sample_point> &samples, User_options &userOptions, Quantities *quantities);
#ifdef PHASE_SPACE
// mayClearDT: this call is the triangulation's LAST use and dt is internally owned -> the
// deposit may free it early (right after its last read) instead of at partition scope end.
void interpolateGrid_phaseSpace(DT &dt, User_options &userOptions, Quantities *quantities, Field &field, int nSub, bool mayClearDT = false);
#include "../ps_point_eval.h"
void interpolatePoints_phaseSpace(DT &dt, User_options &userOptions);   // ps_point_eval.cc
#endif


void delaunayTriangulation(DT *dt,
                           vector<Particle_data> *p,
                           int const verboseLevel);

void vertexDensity(DT & dt,
                User_options &userOptions);



// Builds the Delaunay triangulation, computes vertex densities, and interpolates the requested fields.
// uQuantities = per-point (unaveraged) fields, aQuantities = cell-volume-averaged fields.
void DTFE_interpolation(vector<Particle_data> *p,
                        vector<Sample_point> &samples,
                        User_options &userOptions,
                        Quantities *uQuantities,
                        Quantities *aQuantities,
                        DT &dt)
{
    intervalCheck( userOptions.method, 1, 3, "'userOptions.method' in function 'DTFE_interpolation' must have values from 1 to 3");
    if ( userOptions.method==3 )
        rootN( userOptions.noPoints, NO_DIM );


    Timer t;
    t.start();
    delaunayTriangulation( &dt, p, userOptions.verboseLevel );
    printComputationTime( &t, &userOptions, "triangulation" );
    vector<Particle_data>().swap(*p);  // data lives in the triangulation now; swap-with-empty actually frees the input array (clear() keeps capacity)

    // Skip the padding test under PS-DTFE: dummy points use Eulerian coords, which corrupt the Lagrangian triangulation.
#if defined(TEST_PADDING) && !defined(PHASE_SPACE)
    if ( userOptions.testPaddedBoundaries )
    {
        t.start();
        insertDummyTestParticles( dt, userOptions );
        printComputationTime( &t, &userOptions, "insertion of dummy points" );
    }
#endif
    
    
    t.start();
    vertexDensity( dt, userOptions );
    printComputationTime( &t, &userOptions, "vertex density computation" );


    // approxPSD: form the phase-space density f_i = rho_i (spatial density) * g_i (velocity-space density, in scalar(0)).
#if defined(SCALAR) && !defined(PHASE_SPACE)
    if ( userOptions.approxPSD )
    {
        for (DT::Finite_vertices_iterator vIT = dt.finite_vertices_begin();
             vIT != dt.finite_vertices_end(); ++vIT)
        {
#ifdef TEST_PADDING
            if ( vIT->info().isDummy() ) continue;
#endif
            Real rho = vIT->info().density();
            Real g   = vIT->info().scalar(0);
            if ( rho > Real(0.) && g > Real(0.) )
                vIT->info().scalar(0) = rho * g;
            else
                vIT->info().scalar(0) = Real(0.);
        }
    }
#endif


#ifdef PHASE_SPACE
    // arbitrary-point evaluation (--sample-points): collect this triangulation's stream
    // contributions for every query point; results accumulate across the Lagrangian
    // partitions and are reduced once by psPointEvalFinalize() (called from DTFE()).
    if ( psPointEvalActive() )
    {
        t.start();
        interpolatePoints_phaseSpace( dt, userOptions );
        printComputationTime( &t, &userOptions, "point evaluation (--sample-points)" );
    }
#endif

    // Voronoi: NGP-assign vertex densities to grid (piecewise-constant; bypasses DTFE linear interpolation)
    if ( userOptions.Voronoi )
    {
        MESSAGE::Message message( userOptions.verboseLevel );
        message << "\nAssigning Voronoi volume densities to grid using NGP ... " << MESSAGE::Flush;
        t.start();

        size_t const *nGrid = &(userOptions.gridSize[0]);
        size_t reserveSize = 1;
        for (int d=0; d<NO_DIM; ++d) reserveSize *= nGrid[d];

        aQuantities->density.assign( reserveSize, Real(0.) );
        std::vector<Real> counts( reserveSize, Real(0.) );

        Box box = userOptions.region;
        Real dx[NO_DIM];
        for (int i=0; i<NO_DIM; ++i)
            dx[i] = (box[2*i+1]-box[2*i]) / nGrid[i];

        for (DT::Finite_vertices_iterator vIT = dt.finite_vertices_begin();
             vIT != dt.finite_vertices_end(); ++vIT)
        {
#ifdef TEST_PADDING
            if ( vIT->info().isDummy() ) continue;
#endif
            if ( vIT->info().density() <= Real(0.) ) continue;

            int cell[NO_DIM];
            bool valid = true;
            for (int d=0; d<NO_DIM; ++d)
            {
#ifdef PHASE_SPACE
                cell[d] = int(floor( (vIT->info().eulerianPosition(d) - box[2*d]) / dx[d] ));
#else
                cell[d] = int(floor( (vIT->point()[d] - box[2*d]) / dx[d] ));
#endif
                if ( cell[d] < 0 || cell[d] >= int(nGrid[d]) ) { valid = false; break; }
            }
            if ( !valid ) continue;

            size_t index = 0;
            for (int d=0; d<NO_DIM; ++d) index = index * nGrid[d] + cell[d];

            aQuantities->density[index] += vIT->info().density();
            counts[index] += Real(1.);
        }

        // Average the density where multiple particles fell in one cell.
        for (size_t i=0; i<reserveSize; ++i)
            if ( counts[i] > Real(0.) )
                aQuantities->density[i] /= counts[i];

        printComputationTime( &t, &userOptions, "Voronoi NGP grid assignment" );
        message << "Done.\n" << MESSAGE::Flush;
        return;  // Voronoi path is complete; skip DTFE linear interpolation.
    }


    if ( userOptions.uField.selected() )    // unaveraged fields at the sampling points
    {
        t.start();
#ifdef PHASE_SPACE
        // PS-DTFE iterates over cells to handle multi-stream regions; the triangulation may
        // be torn down after the LAST of the (up to two) grid passes when dt is internally owned
        if ( not userOptions.redshiftConeOn and not userOptions.userDefinedSampling )
            interpolateGrid_phaseSpace( dt, userOptions, uQuantities, userOptions.uField, 1,
                                        userOptions.psMayClearDT and not userOptions.aField.selected() );
        else
            throwError( "PS-DTFE currently only supports regular grid interpolation (no redshift cone or user-defined sampling)." );
#else
        if ( not userOptions.redshiftConeOn and not userOptions.userDefinedSampling )   // interpolate on a regular grid
            interpolateGrid( dt, userOptions, uQuantities );
        else if ( userOptions.redshiftConeOn and not userOptions.userDefinedSampling )  // interpolate on a redshift cone grid
            interpolateRedshiftCone( dt, userOptions, uQuantities );
        else if ( userOptions.userDefinedSampling )
            interpolateUserSampling( dt, samples, userOptions, uQuantities );
#endif
        printComputationTime( &t, &userOptions, "field interpolation (per-point / unaveraged)" );
    }
    
    if ( userOptions.aField.selected() )    // fields volume-averaged inside the grid cells
    {
        t.start();
#ifdef PHASE_SPACE
        // PS-DTFE averaged fields: sub-sample each grid cell nSub^NO_DIM times (--avg-subsamples, default 3)
        if ( not userOptions.redshiftConeOn and not userOptions.userDefinedSampling )
            interpolateGrid_phaseSpace( dt, userOptions, aQuantities, userOptions.aField, userOptions.psAvgSubsamples,
                                        userOptions.psMayClearDT );
        else
            throwError( "PS-DTFE averaged fields support only regular-grid interpolation (no redshift cone or user-defined sampling)." );
#else
        if ( not userOptions.redshiftConeOn and not userOptions.userDefinedSampling )   // interpolate on a regular grid
        {
            if ( userOptions.method==1 )
                interpolateGrid_averaged_1( dt, userOptions, aQuantities );
            else if ( userOptions.method==2 )
                interpolateGrid_averaged_2( dt, userOptions, aQuantities );
            else if ( userOptions.method==3 )
                interpolateGrid_averaged_3( dt, userOptions, aQuantities );
            else throwError( "Unknown averaging method '", userOptions.method, "' when volume averaging the fields on a regular grid." );
        }
        else if ( userOptions.redshiftConeOn and not userOptions.userDefinedSampling )  // interpolate on a redshift cone grid
        {
            if ( userOptions.method==2 )
                interpolateRedshiftCone_averaged_2( dt, userOptions, aQuantities );
            else throwError( "Unknown averaging method '", userOptions.method, "' when volume averaging the fields on a redshift cone grid. The only available redshift cone grid averaging method is '2'." );
        }
        else if ( userOptions.userDefinedSampling )
        {
            if ( userOptions.method==2 )
                interpolateUserSampling_averaged_2( dt, samples, userOptions, aQuantities );
            else throwError( "Unknown averaging method '", userOptions.method, "' when volume averaging the fields on a user defined grid. The only available user defined grid averaging method is '2'." );
        }
#endif
        printComputationTime( &t, &userOptions, "field interpolation (volume-averaged '_a')" );
    }
}

// Convenience overload for callers that do not need the Delaunay triangulation back.
void DTFE_interpolation(vector<Particle_data> *p,
                        vector<Sample_point> &samples,
                        User_options &userOptions,
                        Quantities *uQuantities,
                        Quantities *aQuantities)
{
    DT dt;
#ifdef PHASE_SPACE
    // dt is owned here, so the last PS grid pass may clear it right after its deposit --
    // that frees the triangulation (~650 B/vertex) during the stats/merge phase instead of
    // holding it until this scope ends (~5 GB per concurrent partition at production scale)
    userOptions.psMayClearDT = true;
#endif
    DTFE_interpolation( p, samples, userOptions, uQuantities, aQuantities, dt );
}




// Inserts every particle into the Delaunay triangulation. Under PS-DTFE the points are the
// Lagrangian (initial-condition) coordinates; otherwise the Eulerian positions.
void delaunayTriangulation(DT *dt,
                           vector<Particle_data> *p,
                           int const verboseLevel)
{
    MESSAGE::Message message( verboseLevel );

#if defined(PARALLEL_TRIANGULATION) && NO_DIM==3
    // TBB-parallel build (opt-in, TBB=1): parallel range insert needs Parallel_tag and a lock grid over the bounding box
    message << "\nConstructing the Delaunay triangulation (parallel, TBB) ... " << MESSAGE::Flush;
    double bb[2*NO_DIM];
    for (int d = 0; d < NO_DIM; ++d) { bb[2*d] = 1.e300; bb[2*d+1] = -1.e300; }
    std::vector< std::pair<Point, vertexData> > pts;
    pts.reserve( p->size() );
    for (vector<Particle_data>::iterator it = p->begin(); it != p->end(); ++it)
    {
#ifdef PHASE_SPACE
        double c[NO_DIM] = { double(it->lagPos[0]), double(it->lagPos[1]), double(it->lagPos[2]) };
#else
        double c[NO_DIM] = { double(it->pos[0]), double(it->pos[1]), double(it->pos[2]) };
#endif
        vertexData info;
        info.setData( *it );
        pts.push_back( std::make_pair( Point(c[0], c[1], c[2]), info ) );
        for (int d = 0; d < NO_DIM; ++d)
        { if (c[d] < bb[2*d]) bb[2*d] = c[d]; if (c[d] > bb[2*d+1]) bb[2*d+1] = c[d]; }
    }
    // Pad the bbox so every point lies strictly inside the lock grid.
    for (int d = 0; d < NO_DIM; ++d)
    { double e = (bb[2*d+1]-bb[2*d])*1.e-3 + 1.e-9; bb[2*d] -= e; bb[2*d+1] += e; }
    Bbox lockBox( bb[0], bb[2], bb[4], bb[1], bb[3], bb[5] );
    DT::Lock_data_structure lock( lockBox, 50 );
    dt->set_lock_data_structure( &lock );
    dt->insert( pts.begin(), pts.end() );
    dt->set_lock_data_structure( nullptr );   // lock grid is only needed during concurrent insertion
    message << "Done.\n"
        << "The triangulation has " << dt->number_of_vertices() << " points.\n" << MESSAGE::Flush;
#else
    // spatial_sort speeds up insertion by keeping consecutive points spatially close (better locality hints).
    message << "\nSorting the points to be spatially close yet randomly distributed ... " << MESSAGE::Flush;
    CGAL::spatial_sort( p->begin(), p->end(), Particle_data_sort_traits() );
    message << "Done.\n";


    message << "Constructing the Delaunay triangulation.\n\t Done: " << MESSAGE::Flush;
    size_t prev = 0, amount100 = 0, count = 0;
    size_t const noPoints = p->size();
    Vertex_handle vh;
    for (vector<Particle_data>::iterator it=p->begin(); it!=p->end(); ++it)
    {
#ifdef PHASE_SPACE
        // PS-DTFE builds the triangulation in Lagrangian (initial-condition) space
#if NO_DIM==2
        vh = dt->insert( Point(it->lagPos[0],it->lagPos[1]) );
#elif NO_DIM==3
        vh = dt->insert( Point(it->lagPos[0],it->lagPos[1],it->lagPos[2]), vh );
#endif
#else
#if NO_DIM==2
        vh = dt->insert( Point(it->pos[0],it->pos[1]) );
#elif NO_DIM==3
        vh = dt->insert( Point(it->pos[0],it->pos[1],it->pos[2]), vh );
#endif
#endif
        vh->info().setData( *it );

        amount100 = (100 * count++)/ noPoints;
        if (prev < amount100)
            message.updateProgress( ++prev );
    }

    message << "100\%.\n"
        << "The triangulation has " << dt->number_of_vertices() << " points.\n" << MESSAGE::Flush;
#endif
}



// Estimates the density at each vertex. Standard DTFE: (mass-conservation factor) / total volume of
// incident Delaunay cells. PS-DTFE: avgDensity * V_Lagrangian / V_Eulerian for the incident cells.
void vertexDensity(DT & dt,
                  User_options &userOptions)
{
    MESSAGE::Message message( userOptions.verboseLevel );
    message << "\nComputing the density at each particle position.\n\t Done: " << MESSAGE::Flush;
    if ( userOptions.averageDensity<=0. )
    {
        MESSAGE::Error error;
        error << "The member 'averageDensity' of class 'User_options' must be positive since it represents the average density. Error found in function 'vertexDensity'." << MESSAGE::EndError;
        printf("%f\n", userOptions.averageDensity);
    }
    if ( dt.number_of_vertices()<NO_DIM+1 )
    {
        MESSAGE::Warning warning(1);
        warning << "Because there are less than " << NO_DIM+1 << " vertices in the Delaunay triangulation there is no cell and hence there is no information that can be used to compute the density associated to each vertex. All vertex density values will be initialized to 0.\n" << MESSAGE::EndWarning;
        return;
        for (DT::Finite_vertices_iterator vIT = dt.finite_vertices_begin(); vIT != dt.finite_vertices_end(); ++vIT )
            vIT->info().setDensity( 0. );
    }
    
    Real const factor = (NO_DIM+1.) / userOptions.averageDensity;  // mass-conservation normalization; NO_DIM+1 = vertices per Delaunay cell
    size_t const noVertices = dt.number_of_vertices();

    // Collect active vertex handles up front (CGAL iterators are not random-access).
    std::vector<Vertex_handle> vertexHandles;
    vertexHandles.reserve( noVertices );
    for (DT::Finite_vertices_iterator vIT = dt.finite_vertices_begin(); vIT != dt.finite_vertices_end(); ++vIT )
    {
#ifdef TEST_PADDING
        if ( vIT->info().isDummy() ) continue;
#endif
        vertexHandles.push_back( vIT );
    }
    size_t const noActiveVertices = vertexHandles.size();

    size_t prev = 0, amount100 = 0;
    for (size_t idx = 0; idx < noActiveVertices; ++idx)
    {
        Vertex_handle vIT = vertexHandles[idx];

        vector<Cell_handle> cells;
#if NO_DIM==2
        DT::Face_circulator fc = dt.incident_faces( vIT );
        cells.push_back( fc++ );
        for (; fc!=cells[0]; ++fc)
            cells.push_back( fc );
#elif NO_DIM==3
        dt.incident_cells( vIT, back_inserter(cells) );
#endif

        Real vol = 0.;
        bool infinite_volume = false;

        Real vol_euler = 0.;
        for ( vector< Cell_handle >::const_iterator itC = cells.begin(); itC!=cells.end(); ++itC )
        {
            if ( !dt.is_infinite(*itC) )
            {
                vol += volume( dt, *itC );
#ifdef PHASE_SPACE
                vol_euler += eulerianVolume( *itC );
#endif
#ifdef TEST_PADDING
                if( hasDummyVertex(*itC) )
                    vIT->info().setDummyNeighbor();
#endif
            }
            else
            {
                // An incident infinite cell means the vertex is on the convex hull: density is undefined.
                vIT->info().setDummyNeighbor();
                infinite_volume = true;
                break;
            }
        }

        if ( not infinite_volume )
        {
#ifdef PHASE_SPACE
            if ( vol_euler > Real(0.) )
                vIT->info().setDensity( userOptions.averageDensity * vol / vol_euler );
            else
                vIT->info().setDensity( 0. );
#else
            vIT->info().setDensity( (vIT->info().weight()) * factor /vol );
#endif
        }
        else
            vIT->info().setDensity( 0. );

        amount100 = (100 * idx) / noActiveVertices;
        if (prev < amount100)
            message.updateProgress( ++prev );
    }

    message << "100\%.\n" << MESSAGE::Flush;
}


#if defined(VELOCITY) && defined(SCALAR) && !defined(PHASE_SPACE)
// Estimates each particle's velocity-space density g_i via a Delaunay tessellation in velocity
// coordinates, storing it in scalar(0). Used by --approxPSD to form f_i = rho_i * g_i.
void computeVelocitySpaceDensity(vector<Particle_data> &particles,
                                  User_options &userOptions)
{
    MESSAGE::Message message( userOptions.verboseLevel );
    message << "\nApproximate PSD: Computing velocity-space density.\n" << MESSAGE::Flush;

    size_t const N = particles.size();
    if (N < NO_DIM + 2)
    {
        MESSAGE::Warning warning(1);
        warning << "Too few particles (" << N << ") for velocity tessellation. "
                << "Setting all velocity-space densities to 0.\n" << MESSAGE::EndWarning;
        for (size_t i = 0; i < N; ++i)
            particles[i].scalar(0) = Real(0.);
        return;
    }

    // Velocity-space bounding box and total mass.
    Real vMin[NO_DIM], vMax[NO_DIM];
    for (int d = 0; d < NO_DIM; ++d)
    {
        vMin[d] = particles[0].velocity(d);
        vMax[d] = particles[0].velocity(d);
    }
    Real totalMass = Real(0.);
    for (size_t i = 0; i < N; ++i)
    {
        totalMass += particles[i].weight();
        for (int d = 0; d < NO_DIM; ++d)
        {
            if (particles[i].velocity(d) < vMin[d]) vMin[d] = particles[i].velocity(d);
            if (particles[i].velocity(d) > vMax[d]) vMax[d] = particles[i].velocity(d);
        }
    }
    // Pad the box with a small margin to avoid a zero-volume (degenerate) bounding box.
    for (int d = 0; d < NO_DIM; ++d)
    {
        Real range = vMax[d] - vMin[d];
        if (range < Real(1.e-10))
        { vMin[d] -= Real(1.); vMax[d] += Real(1.); }
        else
        { vMin[d] -= range * Real(1.e-4); vMax[d] += range * Real(1.e-4); }
    }
    Real velBoxVol = Real(1.);
    for (int d = 0; d < NO_DIM; ++d)
        velBoxVol *= (vMax[d] - vMin[d]);
    Real avgVelDensity = totalMass / velBoxVol;

    message << "  Velocity bounds: ";
    for (int d = 0; d < NO_DIM; ++d)
        message << "[" << vMin[d] << ", " << vMax[d] << "] ";
    message << "\n  Average velocity-space density: " << avgVelDensity << "\n" << MESSAGE::Flush;

    // Build the Delaunay tessellation in velocity space.
    DT velDT;
    std::vector<Vertex_handle> velVertices(N);
    Vertex_handle hint;

    message << "  Building velocity-space Delaunay tessellation.\n\t Done: " << MESSAGE::Flush;
    size_t prev = 0, amount100 = 0;
    for (size_t i = 0; i < N; ++i)
    {
#if NO_DIM==2
        Point vp(particles[i].velocity(0), particles[i].velocity(1));
        hint = velDT.insert(vp);
#elif NO_DIM==3
        Point vp(particles[i].velocity(0), particles[i].velocity(1), particles[i].velocity(2));
        hint = velDT.insert(vp, hint);
#endif
        velVertices[i] = hint;
        hint->info().weight() = particles[i].weight();

        amount100 = (100 * i) / N;
        if (prev < amount100)
            message.updateProgress(++prev);
    }
    message << "100\%.\n  Velocity tessellation has " << velDT.number_of_vertices()
            << " vertices (from " << N << " particles).\n" << MESSAGE::Flush;

    // Reusing vertexDensity yields the correct g_i once avgDensity is set to totalMass/velBoxVol.
    User_options velOpt = userOptions;
    velOpt.averageDensity = avgVelDensity;
    velOpt.verboseLevel = (userOptions.verboseLevel > 0) ? 1 : 0;
    vertexDensity(velDT, velOpt);

    // Copy g_i from the velocity-tessellation vertices back into each particle's scalar field.
    size_t nBoundary = 0;
    for (size_t i = 0; i < N; ++i)
    {
        Real gi = velVertices[i]->info().density();
        particles[i].scalar(0) = gi;
        if (gi <= Real(0.)) ++nBoundary;
    }

    if (nBoundary > 0)
        message << "  Note: " << nBoundary << " particles have g_i=0 (velocity convex hull).\n" << MESSAGE::Flush;

    message << "  Velocity-space tessellation complete.\n" << MESSAGE::Flush;
}
#endif



