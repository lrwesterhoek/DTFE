/*
 *  Phase Space DTFE grid interpolation.
 *
 *  In PS-DTFE, the Delaunay triangulation is built in Lagrangian space.
 *  Eulerian simplices can overlap (multi-stream regions), so we iterate
 *  over all cells and accumulate contributions from each simplex that
 *  contains a given grid point.
 */

#ifdef PHASE_SPACE

#include "triangulation_common.h"


/* Interpolates fields onto a regular grid using PS-DTFE.
   Unlike standard DTFE (which locates each grid point in the triangulation),
   PS-DTFE iterates over all finite cells and distributes their contributions
   to overlapping grid points. This handles multi-stream regions correctly. */
void interpolateGrid_phaseSpace(DT &dt,
                                User_options &userOptions,
                                Quantities *quantities)
{
    MESSAGE::Message message( userOptions.verboseLevel );
    Field &field = userOptions.uField;
    size_t const *nGrid = &(userOptions.gridSize[0]);
    Box boxCoordinates = userOptions.region;

    // compute grid spacing
    Real dx[NO_DIM];
    for (int d = 0; d < NO_DIM; ++d)
        dx[d] = (boxCoordinates[2*d+1] - boxCoordinates[2*d]) / nGrid[d];

    // compute total grid size
    size_t totalGrid = 1;
    for (int d = 0; d < NO_DIM; ++d) totalGrid *= nGrid[d];

    // reserve output memory (initialize to zero for accumulation)
    if ( field.density )
        quantities->density.assign(totalGrid, Real(0.));
    if ( field.velocity )
        quantities->velocity.assign(totalGrid, Pvector<Real,noVelComp>::zero());
    if ( field.velocity_gradient )
        quantities->velocity_gradient.assign(totalGrid, Pvector<Real,noGradComp>::zero());
#ifdef SCALAR
    if ( field.scalar )
        quantities->scalar.assign(totalGrid, Pvector<Real,noScalarComp>::zero());
    if ( field.scalar_gradient )
        quantities->scalar_gradient.assign(totalGrid, Pvector<Real,noScalarGradComp>::zero());
#endif

    // count number of streams at each grid point (for diagnostics)
    std::vector<int> streamCount(totalGrid, 0);

    message << "\nPS-DTFE: Interpolating fields to grid by iterating over all Delaunay cells.\n\t Done: " << MESSAGE::Flush;

    // count finite cells for progress
    size_t noTotalCells = 0;
#if NO_DIM==2
    noTotalCells = dt.number_of_faces();
#elif NO_DIM==3
    noTotalCells = dt.number_of_finite_cells();
#endif

    size_t prev = 0, amount100 = 0, count = 0;

    // iterate over all finite cells
#if NO_DIM==2
    for (DT::Finite_faces_iterator itC = dt.finite_faces_begin(); itC != dt.finite_faces_end(); ++itC)
#elif NO_DIM==3
    for (DT::Finite_cells_iterator itC = dt.finite_cells_begin(); itC != dt.finite_cells_end(); ++itC)
#endif
    {
        Cell_handle cell = itC;

        // skip cells with dummy vertices
#ifdef TEST_PADDING
        bool hasDummy = false;
        for (int v = 0; v <= NO_DIM; ++v)
            if (cell->vertex(v)->info().isDummy()) { hasDummy = true; break; }
        if (hasDummy) { ++count; continue; }
#endif

        // Cell ownership check for Lagrangian partitioning:
        // When partitioning in Lagrangian space, padding zones overlap between
        // partitions. To avoid double-counting, a cell is "owned" by the partition
        // that contains vertex 0's Lagrangian position (triangulation coordinate).
        if ( !userOptions.lagrangianRegion.isNullBox() )
        {
            Point const &lagPt = cell->vertex(0)->point();
            bool owned = true;
            for (int d = 0; d < NO_DIM; ++d)
            {
                Real coord = lagPt[d];
                if ( coord < userOptions.lagrangianRegion[2*d] || coord >= userOptions.lagrangianRegion[2*d+1] )
                { owned = false; break; }
            }
            if (!owned) { ++count; continue; }
        }

        // Skip cells where any vertex has zero or negative density.
        // These are typically at the convex hull of the Lagrangian triangulation
        // (where vertex density cannot be computed due to infinite incident cells)
        // and produce spurious contributions spanning large Eulerian volumes.
        {
            bool hasBadVertex = false;
            for (int v = 0; v <= NO_DIM; ++v)
                if (cell->vertex(v)->info().density() <= Real(0.)) { hasBadVertex = true; break; }
            if (hasBadVertex) { ++count; continue; }
        }

        // Gather Eulerian positions for this cell's vertices, applying
        // minimum-image convention for periodic boxes so that cells near
        // the periodic boundary don't span the entire box.
        Real eulerPos[NO_DIM+1][NO_DIM];
        for (int v = 0; v <= NO_DIM; ++v)
            for (int d = 0; d < NO_DIM; ++d)
                eulerPos[v][d] = cell->vertex(v)->info().eulerianPosition(d);

        if ( userOptions.periodic )
        {
            Real boxLen[NO_DIM];
            for (int d = 0; d < NO_DIM; ++d)
                boxLen[d] = boxCoordinates[2*d+1] - boxCoordinates[2*d];

            // Wrap vertices 1..NO_DIM relative to vertex 0 using minimum image
            for (int v = 1; v <= NO_DIM; ++v)
                for (int d = 0; d < NO_DIM; ++d)
                {
                    Real diff = eulerPos[v][d] - eulerPos[0][d];
                    if (diff >  boxLen[d] * Real(0.5)) eulerPos[v][d] -= boxLen[d];
                    if (diff < -boxLen[d] * Real(0.5)) eulerPos[v][d] += boxLen[d];
                }
        }

        // Compute Eulerian position matrix inverse from (possibly wrapped) positions
        double Ax[NO_DIM][NO_DIM];
        for (int v = 0; v < NO_DIM; ++v)
            for (int i = 0; i < NO_DIM; ++i)
                Ax[v][i] = double(eulerPos[v+1][i]) - double(eulerPos[0][i]);

        // Compute determinant to check for degenerate / near-singular cells
        double cellDet = determinant(Ax);
        double cellAbsDet = std::fabs(cellDet);

        // Skip near-singular cells: these have nearly-flat Eulerian simplices
        // that produce extreme density gradients and bright streak artifacts.
        // Scale threshold by average edge length cubed to be resolution-independent.
        {
            double avgEdge2 = 0.;
            for (int v = 0; v < NO_DIM; ++v)
            {
                double len2 = 0.;
                for (int i = 0; i < NO_DIM; ++i)
                    len2 += Ax[v][i] * Ax[v][i];
                avgEdge2 += len2;
            }
            avgEdge2 /= NO_DIM;
            double edgeScale = avgEdge2 * std::sqrt(avgEdge2); // avgEdge^3
            if (cellAbsDet < 1.e-6 * edgeScale) { ++count; continue; }
        }

        Real posMatInv[NO_DIM][NO_DIM];
        matrixInverse(Ax, posMatInv);

        // compute Eulerian bounding box of this cell (using wrapped positions)
        Real eMin[NO_DIM], eMax[NO_DIM];
        for (int d = 0; d < NO_DIM; ++d)
        {
            eMin[d] = eulerPos[0][d];
            eMax[d] = eulerPos[0][d];
        }
        for (int v = 1; v <= NO_DIM; ++v)
            for (int d = 0; d < NO_DIM; ++d)
            {
                if (eulerPos[v][d] < eMin[d]) eMin[d] = eulerPos[v][d];
                if (eulerPos[v][d] > eMax[d]) eMax[d] = eulerPos[v][d];
            }

        // For cells with wrapped positions that are now outside [boxMin, boxMax],
        // shift the bounding box back into the grid domain.
        // We may need to process the cell twice if it straddles the boundary.
        // For simplicity we handle this by allowing grid indices to wrap.

        // find grid index range overlapping with bounding box
        int iMin[NO_DIM], iMax[NO_DIM];
        bool outsideGrid = false;
        for (int d = 0; d < NO_DIM; ++d)
        {
            iMin[d] = int(floor((eMin[d] - boxCoordinates[2*d]) / dx[d]));
            iMax[d] = int(floor((eMax[d] - boxCoordinates[2*d]) / dx[d])) + 1;
            if (!userOptions.periodic)
            {
                if (iMin[d] < 0) iMin[d] = 0;
                if (iMax[d] > (int)nGrid[d]) iMax[d] = (int)nGrid[d];
                if (iMin[d] >= (int)nGrid[d] || iMax[d] <= 0) { outsideGrid = true; break; }
            }
        }
        if (outsideGrid) { ++count; continue; }

        // compute density gradient in this cell (using wrapped Eulerian positions)
        Real densGrad[NO_DIM];
        {
            Real dens[NO_DIM];
            for (int i = 0; i < NO_DIM; ++i)
                dens[i] = cell->vertex(i+1)->info().density() - cell->vertex(0)->info().density();
            matrixMultiplication(posMatInv, dens, densGrad);
        }

        // compute velocity gradient if needed
        Real velGrad[NO_DIM][noVelComp];
        if (field.velocity || field.velocity_gradient)
        {
            Vertex_handle base = cell->vertex(0);
            Real temp[NO_DIM][noVelComp];
            for (int v = 0; v < NO_DIM; ++v)
                for (size_t i = 0; i < noVelComp; ++i)
                    temp[v][i] = cell->vertex(v+1)->info().velocity(i) - base->info().velocity(i);
            matrixMultiplication<noVelComp>(posMatInv, temp, velGrad);
        }

        // compute scalar gradient if needed
#ifdef SCALAR
        Real sGrad[NO_DIM][noScalarComp];
        if (field.scalar || field.scalar_gradient)
        {
            Real temp[NO_DIM][noScalarComp];
            for (int v = 0; v < NO_DIM; ++v)
                for (size_t i = 0; i < noScalarComp; ++i)
                    temp[v][i] = cell->vertex(v+1)->info().myScalar()[i] - cell->vertex(0)->info().myScalar()[i];
            matrixMultiplication<noScalarComp>(posMatInv, temp, sGrad);
        }
#endif

        // iterate over candidate grid points using nested loops
#if NO_DIM==2
        for (int gi = iMin[0]; gi < iMax[0]; ++gi)
        for (int gj = iMin[1]; gj < iMax[1]; ++gj)
        {
            // Wrap grid indices for periodic boxes
            int wgi = userOptions.periodic ? ((gi % (int)nGrid[0] + (int)nGrid[0]) % (int)nGrid[0]) : gi;
            int wgj = userOptions.periodic ? ((gj % (int)nGrid[1] + (int)nGrid[1]) % (int)nGrid[1]) : gj;
            int gridIdx[2] = {wgi, wgj};
#elif NO_DIM==3
        for (int gi = iMin[0]; gi < iMax[0]; ++gi)
        for (int gj = iMin[1]; gj < iMax[1]; ++gj)
        for (int gk = iMin[2]; gk < iMax[2]; ++gk)
        {
            // Wrap grid indices for periodic boxes
            int wgi = userOptions.periodic ? ((gi % (int)nGrid[0] + (int)nGrid[0]) % (int)nGrid[0]) : gi;
            int wgj = userOptions.periodic ? ((gj % (int)nGrid[1] + (int)nGrid[1]) % (int)nGrid[1]) : gj;
            int wgk = userOptions.periodic ? ((gk % (int)nGrid[2] + (int)nGrid[2]) % (int)nGrid[2]) : gk;
            int gridIdx[3] = {wgi, wgj, wgk};
#endif

            // compute grid point position (cell center) -- use unwrapped index
            // so the position is consistent with the (possibly wrapped) Eulerian coords
            Real gridPos[NO_DIM];
#if NO_DIM==2
            int rawIdx[2] = {gi, gj};
#elif NO_DIM==3
            int rawIdx[3] = {gi, gj, gk};
#endif
            for (int d = 0; d < NO_DIM; ++d)
                gridPos[d] = boxCoordinates[2*d] + (rawIdx[d] + Real(0.5)) * dx[d];

            // test if grid point is inside the Eulerian simplex
            // Use the wrapped vertex 0 position for the point-in-simplex test
            Real rel[NO_DIM];
            for (int d = 0; d < NO_DIM; ++d)
                rel[d] = gridPos[d] - eulerPos[0][d];
            Real baryCoords[NO_DIM];
            matrixMultiplication(posMatInv, rel, baryCoords);
            Real sum = 0.;
            bool inside = true;
            for (int i = 0; i < NO_DIM; ++i)
            {
                if (baryCoords[i] < Real(-1.e-6)) { inside = false; break; }
                sum += baryCoords[i];
            }
            if (!inside || sum > Real(1. + 1.e-6)) continue;

            // compute flat grid index (from wrapped indices)
            size_t flatIdx = 0;
            for (int d = 0; d < NO_DIM; ++d)
                flatIdx = flatIdx * nGrid[d] + gridIdx[d];

            // accumulate density (clamp to non-negative to avoid unphysical values)
            if (field.density)
            {
                Real densVal = cell->vertex(0)->info().density();
                for (int d = 0; d < NO_DIM; ++d)
                    densVal += rel[d] * densGrad[d];
                if (densVal < Real(0.)) densVal = Real(0.);
                quantities->density[flatIdx] += densVal;
            }

            // accumulate velocity
            if (field.velocity)
            {
                Pvector<Real,noVelComp> velVal = cell->vertex(0)->info().velocity();
                for (int i = 0; i < NO_DIM; ++i)
                    for (size_t j = 0; j < noVelComp; ++j)
                        velVal[j] += velGrad[i][j] * rel[i];
                quantities->velocity[flatIdx] += velVal;
            }

            // accumulate velocity gradient
            if (field.velocity_gradient)
            {
                Pvector<Real,noGradComp> grad;
                for (size_t j = 0; j < noVelComp; ++j)
                    for (int i = 0; i < NO_DIM; ++i)
                        grad[j*NO_DIM+i] = velGrad[i][j];
                quantities->velocity_gradient[flatIdx] += grad;
            }

            // accumulate scalar
#ifdef SCALAR
            if (field.scalar)
            {
                Pvector<Real,noScalarComp> scalarVal = cell->vertex(0)->info().myScalar();
                for (int i = 0; i < NO_DIM; ++i)
                    for (size_t j = 0; j < noScalarComp; ++j)
                        scalarVal[j] += sGrad[i][j] * rel[i];
                quantities->scalar[flatIdx] += scalarVal;
            }

            // accumulate scalar gradient
            if (field.scalar_gradient)
            {
                Pvector<Real,noScalarGradComp> sgrad;
                for (size_t j = 0; j < noScalarComp; ++j)
                    for (int i = 0; i < NO_DIM; ++i)
                        sgrad[j*NO_DIM+i] = sGrad[i][j];
                quantities->scalar_gradient[flatIdx] += sgrad;
            }
#endif

            streamCount[flatIdx]++;
        }

        // show progress
        amount100 = (100 * count++) / noTotalCells;
        if (prev < amount100)
            message.updateProgress( ++prev );
    }

    message << "100\%.\n" << MESSAGE::Flush;

    // report multi-stream statistics
    int maxStreams = 0;
    size_t multiStreamCells = 0;
    for (size_t i = 0; i < totalGrid; ++i)
    {
        if (streamCount[i] > maxStreams) maxStreams = streamCount[i];
        if (streamCount[i] > 1) multiStreamCells++;
    }
    message << "PS-DTFE: Max streams at a grid point: " << maxStreams
            << ", grid points with multi-stream: " << multiStreamCells
            << " (" << (100.*multiStreamCells/totalGrid) << "\%)\n" << MESSAGE::Flush;

    // store stream count in the output quantities
    quantities->stream_count.resize(totalGrid);
    for (size_t i = 0; i < totalGrid; ++i)
        quantities->stream_count[i] = Real(streamCount[i]);
}

#endif // PHASE_SPACE
