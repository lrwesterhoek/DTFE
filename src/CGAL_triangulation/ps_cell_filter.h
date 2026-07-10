/* Shared PS-DTFE per-cell filter chain: dummy-vertex skip, Lagrangian-centroid partition
   ownership, bad-density hull handling, minimum-image wrap of the Eulerian vertices, the
   Eulerian edge matrix with its relative-determinant degeneracy test, and (optionally) the
   matrixInverse() zero-inverse check.

   Used by the CPU grid deposit, the GPU tet extraction and the sub-grid pass in
   ps_interpolation.cc, and by the point evaluation in ps_point_eval.cc. The partition
   protocol requires every consumer to classify borderline cells IDENTICALLY (a stream must
   be contributed by exactly one Lagrangian partition, and the point-eval stream set must
   equal the grid deposit's), so the chain lives here once and every consumer runs the same
   arithmetic in the same precision. */

#ifndef PS_CELL_FILTER_HEADER
#define PS_CELL_FILTER_HEADER

#ifdef PHASE_SPACE   // both consumers (ps_interpolation.cc, ps_point_eval.cc) are PS-only

// Geometry of a cell that passed psFilterCell().
struct PSCellGeometry
{
    Real   eulerPos[NO_DIM+1][NO_DIM]; // Eulerian vertex positions, minimum-image wrapped
    double Ax[NO_DIM][NO_DIM];         // Eulerian edge matrix (rows = vertices 1..NO_DIM relative to vertex 0)
    double cellDet;                    // determinant(Ax)
    double cellAbsDet;                 // |cellDet|
    Real   posMatInv[NO_DIM][NO_DIM];  // matrixInverse(Ax); filled only when checkSingularInverse is true
    bool   useVolumeRatioDensity;      // zero/negative-density hull vertex found (kept: non-periodic cloud surface)
};

// Runs the filter chain on one Delaunay cell. Returns false when the cell must be skipped;
// on success 'g' holds the wrapped positions and edge-matrix geometry every consumer needs.
//
// checkSingularInverse: additionally drop cells whose matrixInverse() is the ALL-ZERO matrix
// (|det| below REAL_PRECISSION; counted in *nDegenerateInverse when given). A zero inverse
// makes every barycentric coordinate below exactly 0, which the point-in-simplex test then
// reads as "inside" for EVERY grid point in the cell's bounding box -- so the cell floods its
// whole axis-aligned bbox. That is the source of the grid-aligned "square" artefacts: worst
// at caustics (flat, collapsed cells cluster there) and amplified by sub-sampling (nSub>1
// puts more sample points inside the bbox). A collapsed cell has no reliable Eulerian
// volume, so drop it; an adjacent non-degenerate stream covers the region. NOTE: this must
// stay consistent with matrixInverse()'s degeneracy criterion -- the relative-determinant
// pre-filter alone does NOT catch every cell that matrixInverse() zeroes.
inline bool psFilterCell(Cell_handle &cell,
                         User_options &userOptions,
                         Box &boxCoordinates,
                         bool const checkSingularInverse,
                         size_t *nDegenerateInverse,
                         PSCellGeometry &g)
{
#ifdef TEST_PADDING
    // Skip cells touching a dummy padding vertex.
    {
        bool hasDummy = false;
        for (int v = 0; v <= NO_DIM; ++v)
            if (cell->vertex(v)->info().isDummy()) { hasDummy = true; break; }
        if (hasDummy) { return false; }
    }
#endif

    // Lagrangian-partition ownership: padding zones overlap, so keep a cell only if its
    // Lagrangian centroid (not an arbitrary vertex) lies in the primary box -> tiles without double-counting
    if ( !userOptions.lagrangianRegion.isNullBox() )
    {
        double cen[NO_DIM];
        for (int d = 0; d < NO_DIM; ++d) cen[d] = 0.;
        for (int v = 0; v <= NO_DIM; ++v)
            for (int d = 0; d < NO_DIM; ++d)
                cen[d] += double(cell->vertex(v)->point()[d]);
        bool owned = true;
        for (int d = 0; d < NO_DIM; ++d)
        {
            cen[d] /= double(NO_DIM + 1);
            if ( cen[d] < userOptions.lagrangianRegion[2*d] || cen[d] >= userOptions.lagrangianRegion[2*d+1] )
            { owned = false; break; }
        }
        if (!owned) { return false; }
    }

    // zero/negative-density vertex sits on the Lagrangian convex hull. Periodic: artefact, drop the
    // cell. Non-periodic: real cloud surface, give it a constant volume-ratio density (avgDensity * V_Lag/V_Eul).
    g.useVolumeRatioDensity = false;
    {
        bool hasBadVertex = false;
        for (int v = 0; v <= NO_DIM; ++v)
            if (cell->vertex(v)->info().density() <= Real(0.)) { hasBadVertex = true; break; }
        if (hasBadVertex)
        {
            if ( userOptions.periodic ) { return false; }
            g.useVolumeRatioDensity = true;
        }
    }

    // Gather the Eulerian vertex positions; the minimum-image convention (below) keeps boundary
    // cells from spuriously spanning the whole periodic box.
    for (int v = 0; v <= NO_DIM; ++v)
        for (int d = 0; d < NO_DIM; ++d)
            g.eulerPos[v][d] = cell->vertex(v)->info().eulerianPosition(d);

    if ( userOptions.periodic )
    {
        Real boxLen[NO_DIM];
        for (int d = 0; d < NO_DIM; ++d)
            boxLen[d] = boxCoordinates[2*d+1] - boxCoordinates[2*d];

        // Wrap vertices 1..NO_DIM to their nearest image of vertex 0.
        for (int v = 1; v <= NO_DIM; ++v)
            for (int d = 0; d < NO_DIM; ++d)
            {
                Real diff = g.eulerPos[v][d] - g.eulerPos[0][d];
                if (diff >  boxLen[d] * Real(0.5)) g.eulerPos[v][d] -= boxLen[d];
                if (diff < -boxLen[d] * Real(0.5)) g.eulerPos[v][d] += boxLen[d];
            }
    }

    // Eulerian edge matrix (rows = vertices 1..NO_DIM relative to vertex 0, possibly wrapped).
    for (int v = 0; v < NO_DIM; ++v)
        for (int i = 0; i < NO_DIM; ++i)
            g.Ax[v][i] = double(g.eulerPos[v+1][i]) - double(g.eulerPos[0][i]);

    g.cellDet = determinant(g.Ax);
    g.cellAbsDet = std::fabs(g.cellDet);

    // drop only degenerate cells (near-zero Eulerian volume -> divergent 1/volume density).
    // |det| < tol*(mean edge)^3 is resolution-independent; caustics stay well above it.
    double const DEGENERATE_DET_TOL = 1.e-6;
    {
        double avgEdge2 = 0.;
        for (int v = 0; v < NO_DIM; ++v)
        {
            double len2 = 0.;
            for (int i = 0; i < NO_DIM; ++i)
                len2 += g.Ax[v][i] * g.Ax[v][i];
            avgEdge2 += len2;
        }
        avgEdge2 /= NO_DIM;
        double edgeScale = avgEdge2 * std::sqrt(avgEdge2); // avgEdge^3
        if (g.cellAbsDet < DEGENERATE_DET_TOL * edgeScale) { return false; }
    }

    if ( checkSingularInverse )
    {
        matrixInverse(g.Ax, g.posMatInv);
        bool singularInverse = true;
        for (int a = 0; a < NO_DIM && singularInverse; ++a)
            for (int b = 0; b < NO_DIM; ++b)
                if (g.posMatInv[a][b] != Real(0.)) { singularInverse = false; break; }
        if (singularInverse)
        {
            if (nDegenerateInverse) ++(*nDegenerateInverse);
            return false;
        }
    }

    return true;
}

#endif  // PHASE_SPACE

#endif
