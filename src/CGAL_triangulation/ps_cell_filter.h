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
// (|det| below REAL_PRECISSION x the product of the edge lengths -- a RELATIVE, unit-free test
// since 2026-09; the old absolute |det| < 1e-6 dropped every small but well-shaped tetrahedron,
// losing their mass; counted in *nDegenerateInverse when given). A zero inverse
// makes every barycentric coordinate below exactly 0, which the point-in-simplex test then
// reads as "inside" for EVERY grid point in the cell's bounding box -- so the cell floods its
// whole axis-aligned bbox. That is the source of the grid-aligned "square" artefacts: worst
// at caustics (flat, collapsed cells cluster there) and amplified by sub-sampling (nSub>1
// puts more sample points inside the bbox). A collapsed cell has no reliable Eulerian
// volume, so drop it; an adjacent non-degenerate stream covers the region. NOTE: this must
// stay consistent with matrixInverse()'s degeneracy criterion. With both tests relative, the
// pre-filter below (|det| < 1e-6 avgEdge^3) already implies it (avgEdge^3 >= the product of
// the edge lengths, by AM-GM), so this check is now a safety net -- keep it anyway: the zero
// sentinel is what the "square" artefact came from, and the Metal kernel mirrors it.
// 'skipOwnership': apply every CHART-INDEPENDENT filter (dummy, hull artefact, degeneracy)
// but NOT the Lagrangian-partition ownership tiling. Used by the --ps-vertex-mass degree
// pass, which must count a vertex's GLOBALLY-depositing incident tets: ownership only decides
// WHICH partition deposits a tet, not WHETHER it deposits, so degrees must ignore it -- while
// the drop filters decide 'whether' and are computed identically in every partition that sees
// the cell (padded vertices carry their full neighborhood).
// No early rejection (the default of psFilterCell's last argument).
struct PSNoEarlyReject { bool operator()(Real const (&)[NO_DIM+1][NO_DIM]) const { return false; } };

// 'earlyReject(eulerPos)' -- given the wrapped Eulerian positions, before the edge matrix,
// determinant and inverse -- lets a caller drop a cell it will discard anyway on its positions
// alone (--ps-window's GPU extraction: a tetrahedron that misses the window), at a fraction of the
// filter's cost. The default rejects nothing.
template <class EarlyReject = PSNoEarlyReject>
inline bool psFilterCell(Cell_handle &cell,
                         User_options &userOptions,
                         Box &boxCoordinates,
                         bool const checkSingularInverse,
                         size_t *nDegenerateInverse,
                         PSCellGeometry &g,
                         bool const skipOwnership = false,
                         EarlyReject const &earlyReject = EarlyReject())
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
    if ( !skipOwnership && !userOptions.lagrangianRegion.isNullBox() )
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

    // Non-periodic clouds (--ps-alpha-shape): keep only the tetrahedra of the cloud's ALPHA SHAPE,
    // whose Lagrangian circumradius is at most psLagAlphaRadius (3 mean spacings by default). Interior
    // tetrahedra of lattice-like initial conditions stay below ~0.9 spacings (measured on jittered
    // lattices and on TNG100-3 ICs); the cut removes the flat slivers on the convex hull and the
    // tetrahedra bridging concavities, which are no flow elements -- a hull sliver joins particles up
    // to a whole cloud face apart, and its stretched Eulerian image put spurious mass and streams
    // across the region. They are also why --partition runs differed from single ones: the hull of a
    // nearly flat face is a GLOBAL object, built differently by each partition, whereas a tetrahedron
    // within the cut is in a partition's Delaunay triangulation exactly when it is in the global one
    // (its empty circumsphere lies inside the padded box: the partition plan pads by >= 4 cuts).
    // A drop filter, so it applies with skipOwnership too; a function of the four vertices only.
    if ( userOptions.psLagAlphaRadius > 0. )
    {
#if NO_DIM==3
        double e[3][3];
        for (int v = 0; v < 3; ++v)
            for (int d = 0; d < 3; ++d)
                e[v][d] = double(cell->vertex(v+1)->point()[d]) - double(cell->vertex(0)->point()[d]);
        auto cross = [](double const *a, double const *b, double *c)
        { c[0] = a[1]*b[2] - a[2]*b[1]; c[1] = a[2]*b[0] - a[0]*b[2]; c[2] = a[0]*b[1] - a[1]*b[0]; };
        double bc[3], ca[3], ab[3];
        cross(e[1], e[2], bc); cross(e[2], e[0], ca); cross(e[0], e[1], ab);
        double const den = 2. * (e[0][0]*bc[0] + e[0][1]*bc[1] + e[0][2]*bc[2]);
        double n2[3];
        for (int v = 0; v < 3; ++v) n2[v] = e[v][0]*e[v][0] + e[v][1]*e[v][1] + e[v][2]*e[v][2];
        double num2 = 0.;
        for (int d = 0; d < 3; ++d)
        {
            double const c = n2[0]*bc[d] + n2[1]*ca[d] + n2[2]*ab[d];
            num2 += c*c;
        }
        // circumradius = |num| / |den|; a flat tetrahedron (den -> 0) is beyond any cut
        double const a = userOptions.psLagAlphaRadius;
        if ( !( num2 <= a*a * den*den ) || den == 0. ) { return false; }
#else
        double const p0x = cell->vertex(0)->point()[0], p0y = cell->vertex(0)->point()[1];
        double const ax = cell->vertex(1)->point()[0] - p0x, ay = cell->vertex(1)->point()[1] - p0y;
        double const bx = cell->vertex(2)->point()[0] - p0x, by = cell->vertex(2)->point()[1] - p0y;
        double const den = 2. * (ax*by - ay*bx);
        double const ux = by*(ax*ax + ay*ay) - ay*(bx*bx + by*by), uy = ax*(bx*bx + by*by) - bx*(ax*ax + ay*ay);
        double const a = userOptions.psLagAlphaRadius;
        if ( !( ux*ux + uy*uy <= a*a * den*den ) || den == 0. ) { return false; }
#endif
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

    if ( earlyReject(g.eulerPos) ) { return false; }

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
