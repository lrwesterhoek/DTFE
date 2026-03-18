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

#include "triangulation_common.h"

// Forward declarations of functions defined in separate compilation units
void interpolateGrid(DT &dt, User_options &userOptions, Quantities *quantities);
void interpolateRedshiftCone(DT &dt, User_options &userOptions, Quantities *quantities);
void interpolateUserSampling(DT &dt, vector<Sample_point> &samples, User_options &userOptions, Quantities *quantities);
void interpolateGrid_averaged_1(DT &dt, User_options &userOptions, Quantities *quantities);
void interpolateGrid_averaged_2(DT &dt, User_options &userOptions, Quantities *quantities);
void interpolateGrid_averaged_3(DT &dt, User_options &userOptions, Quantities *quantities);
void interpolateRedshiftCone_averaged_2(DT &dt, User_options &userOptions, Quantities *quantities);
void interpolateUserSampling_averaged_2(DT &dt, vector<Sample_point> &samples, User_options &userOptions, Quantities *quantities);
#ifdef PHASE_SPACE
void interpolateGrid_phaseSpace(DT &dt, User_options &userOptions, Quantities *quantities);
#endif


void delaunayTriangulation(DT *dt,
                           vector<Particle_data> *p,
                           int const verboseLevel);

void vertexDensity(DT & dt,
                User_options &userOptions);



/* This function computes the grid density in a periodic cosmological box of size [0,1] on each axis. If a non-periodic box is used, than points can have coordinates outside the [0,1] box to give a full Delaunay triangulation cover of the box. 
The function arguments are:
        p - vector which stores the particle positions and properties
        samples - vector storing the user defined sampling points, if any
        userOptions - structure to keep track of user given options and parameters for the interpolation
        uQuantities - struture that will hold the output fields, fields interpolated at the sampling points
        aQuantities - struture that will hold the output volume averaged fields, fields volume averaged over the grid cell associated to each sampling point
        dt - the CGAL Delaunay triangulation structure that stores the full triangulation of the point set
*/
void DTFE_interpolation(vector<Particle_data> *p,
                        vector<Sample_point> &samples,
                        User_options &userOptions,
                        Quantities *uQuantities,
                        Quantities *aQuantities,
                        DT &dt)
{
    // check that some conditions are satified (to don't get errors later on after all the heavy work was done)
    intervalCheck( userOptions.method, 1, 3, "'userOptions.method' in function 'DTFE_interpolation' must have values from 1 to 3");
    if ( userOptions.method==3 )
        rootN( userOptions.noPoints, NO_DIM );
    
    
    // Compute the Delaunay triangulation
    Timer t;     // construct a timer
    t.start();
    delaunayTriangulation( &dt, p, userOptions.verboseLevel );   // compute the triangulation
    printComputationTime( &t, &userOptions, "triangulation" );
    p->clear();  // delete the vector storing the particle data - not needed anymore
    
    
    // insert dummy points to test the padding
    // Skip for PS-DTFE: the dummy points use paddedBox (Eulerian) coordinates,
    // but the PS-DTFE triangulation is built in Lagrangian space. Inserting
    // Eulerian-coordinate points into the Lagrangian triangulation corrupts it.
#if defined(TEST_PADDING) && !defined(PHASE_SPACE)
    if ( userOptions.testPaddedBoundaries )
    {
        t.start();
        insertDummyTestParticles( dt, userOptions );
        printComputationTime( &t, &userOptions, "insertion of dummy points" );
    }
#endif
    
    
    // compute the density associated to each Delaunay triangulation vertex (to each particle)
    t.start();
    vertexDensity( dt, userOptions );
    printComputationTime( &t, &userOptions, "vertex density computation" );


    // For approximate PSD: multiply f_i = rho_i * g_i at each vertex.
    // At this point, vertex.density() = rho_i (spatial density from vertexDensity)
    // and vertex.scalar(0) = g_i (velocity-space density from computeVelocitySpaceDensity,
    // which was stored in the particle scalar field before setData() copied it to vertices).
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


    // Voronoi volume density: NGP assignment of vertex densities to grid
    // Bypasses DTFE linear interpolation — gives piecewise-constant density field
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

        // Average density in cells with multiple particles
        for (size_t i=0; i<reserveSize; ++i)
            if ( counts[i] > Real(0.) )
                aQuantities->density[i] /= counts[i];

        printComputationTime( &t, &userOptions, "Voronoi NGP grid assignment" );
        message << "Done.\n" << MESSAGE::Flush;
        return;  // skip DTFE linear interpolation
    }


    // interpolate the required fields
    if ( userOptions.uField.selected() )    // interpolate the fields at the sampling points
    {
        t.start();
#ifdef PHASE_SPACE
        // PS-DTFE uses cell-iteration approach to handle multi-stream regions
        if ( not userOptions.redshiftConeOn and not userOptions.userDefinedSampling )
            interpolateGrid_phaseSpace( dt, userOptions, uQuantities );
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
        printComputationTime( &t, &userOptions, "interpolation to sampling points" );
    }
    
    if ( userOptions.aField.selected() )    // interpolate the fields volume averaged inside the grid cells
    {
#ifdef PHASE_SPACE
        throwError( "PS-DTFE does not support averaged fields (_a suffix). Use unaveraged fields instead (e.g., 'density' instead of 'density_a')." );
#endif
        t.start();
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
        printComputationTime( &t, &userOptions, "interpolation to sampling points" );
    }
}

// Accesses the above 'DTFE_interpolation', but without returning the Delaunay triangulation
void DTFE_interpolation(vector<Particle_data> *p,
                        vector<Sample_point> &samples,
                        User_options &userOptions,
                        Quantities *uQuantities,
                        Quantities *aQuantities)
{
    DT dt;                // the triangulation
    DTFE_interpolation( p, samples, userOptions, uQuantities, aQuantities, dt );
}




/* Constructs the Delaunay triangulation using a set of points. */
void delaunayTriangulation(DT *dt,
                           vector<Particle_data> *p,
                           int const verboseLevel)
{
    // sort the particles to be spatially close - increase speed of Delaunay triangulation 
    MESSAGE::Message message( verboseLevel );
    message << "\nSorting the points to be spatially close yet randomly distributed ... " << MESSAGE::Flush;
    CGAL::spatial_sort( p->begin(), p->end(), Particle_data_sort_traits() );
    message << "Done.\n";
    
    
    // construct the actual triangulation
    message << "Constructing the Delaunay triangulation.\n\t Done: " << MESSAGE::Flush;
    size_t prev = 0, amount100 = 0, count = 0; // variable to show the user about the progress of the computation
    size_t const noPoints = p->size();
    Vertex_handle vh;     // vertex handle - points to each inserted point
    for (vector<Particle_data>::iterator it=p->begin(); it!=p->end(); ++it)
    {
#ifdef PHASE_SPACE
        // In PS-DTFE mode, build triangulation in Lagrangian (initial condition) space
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
        vh->info().setData( *it );      // set vertex quantities
        
        // show the progress of the computation
        amount100 = (100 * count++)/ noPoints;
        if (prev < amount100)
            message.updateProgress( ++prev );
    }
    
    message << "100\%.\n"
        << "The triangulation has " << dt->number_of_vertices() << " points.\n" << MESSAGE::Flush;
}



/* The function computes the volume of all Delaunay tetrahedra for each vertex and than computes the density at each vertex as the inverse of the total tetrahedra volumes incident on the respective vertex. */
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
    
    Real const factor = (NO_DIM+1.) / userOptions.averageDensity;  //factor to normalize the density to the average background density; NO_DIM+1 comes from the number of vertices of a Delaunay tetrahedra - needed for mass conservation
    size_t const noVertices = dt.number_of_vertices();  // total number of finite vertices

    // collect vertex handles for parallel processing (CGAL iterators are not random-access)
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
/* Compute velocity-space density g_i for each particle using a Delaunay
   tessellation in velocity coordinates. The result is stored in the
   scalar(0) field of each particle. Used by --approxPSD for approximate
   phase-space density estimation: f_i = rho_i * g_i. */
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

    // 1. Compute velocity bounding box and total mass
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
    // Add small margin to avoid zero-volume box
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

    // 2. Build Delaunay tessellation in velocity space
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

    // 3. Compute velocity-space density at each vertex
    // Use vertexDensity() with velocity-space normalization.
    // The density formula ρ_i = weight_i * (D+1) / (avgDensity * V_i) gives
    // the correct velocity-space density when avgDensity = totalMass / velBoxVol.
    User_options velOpt = userOptions;
    velOpt.averageDensity = avgVelDensity;
    velOpt.verboseLevel = (userOptions.verboseLevel > 0) ? 1 : 0;
    vertexDensity(velDT, velOpt);

    // 4. Copy g_i from velocity tessellation vertices to particle scalar field
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
    // velDT destroyed on scope exit, freeing memory
}
#endif



