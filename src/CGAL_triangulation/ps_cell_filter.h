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

#include <algorithm>   // std::sort / std::max in the labelling-independent degeneracy scale

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
// pre-filter below (|det| < 1e-6 avgEdge^3, avgEdge the largest vertex-star mean) already implies
// it (avgEdge^3 >= the product of the vertex-0 edge lengths, by AM-GM, for any labelling), so this
// check is now a safety net -- keep it anyway: the zero
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
    //
    // Decided as sum_v (x_v - edge) against 0, per edge (2026-10-13) -- lo <= centroid < hi exactly when
    // sum (x_v - lo) >= 0 and sum (x_v - hi) < 0. Each x_v - edge is a difference of two Reals, exact in double,
    // and where the answer is close (every vertex within a tetrahedron's extent of the edge) so is their sum, in
    // any order CGAL lists the vertices. A tetrahedron and its periodic copy -- an exact shift by L at the
    // periodic seam (subpartition.h psPeriodicSeam), whose period's edges are exactly L apart -- therefore get
    // complementary answers bit for bit: one of them is owned, never both or neither. The old test (the raw
    // coordinates summed, divided by NO_DIM+1, compared) gives the same answers whenever that sum is exact,
    // i.e. unless a coordinate ~2^-29 of the others' size made it round.
    if ( !skipOwnership && !userOptions.lagrangianRegion.isNullBox() )
    {
        bool owned = true;
        for (int d = 0; d < NO_DIM && owned; ++d)
        {
            double const lo = double(userOptions.lagrangianRegion[2*d]);
            double const hi = double(userOptions.lagrangianRegion[2*d+1]);
            double sLo = 0., sHi = 0.;
            for (int v = 0; v <= NO_DIM; ++v)
            {
                double const x = double(cell->vertex(v)->point()[d]);
                sLo += x - lo;
                sHi += x - hi;
            }
            if ( sLo < 0. || sHi >= 0. ) owned = false;
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
    // |det| < tol*(mean edge)^3 is resolution-independent.
    //
    // The edge scale is the LARGEST over the NO_DIM+1 vertices of the mean squared length of the edges
    // meeting there (2026-10-12). It used to be the edges at vertex 0 only -- and which vertex CGAL calls 0
    // follows the insertion history, so it differs between a single triangulation and every --partition /
    // --padding / --ps-lag-padding / parallel-insert variant of it. Folded tetrahedra at a caustic sit right
    // at the cut (a 32^3 crossed-wave test: ratios 2e-7 .. 1.2e-6 for the four labellings of one tet), so ~20
    // of them were kept by one run and dropped by the other: their whole mass (most are centroid-fallback
    // tets) appeared or vanished in one cell -- '.a_den' 0.7 relative in ~40 cells, hidden-stream flags and
    // '.a_streams' samples flipping, and with --ps-vertex-mass the degrees (and so every neighbour's mass)
    // changing too. A function of the tetrahedron, not of its labelling, makes the kept set the same in
    // every run. The maximum (not the 6-edge mean) because it bounds the vertex-0 scale of EVERY labelling:
    // a kept tet then passes matrixInverse()'s relative test (product of the vertex-0 rows <= their mean^3,
    // AM-GM) and the GPU kernels' float re-check (vertex-0 mean) whatever order it arrives in, so neither
    // can reintroduce the order dependence. Only the order-ambiguous tets change: kept under every
    // labelling -> still kept, dropped under every labelling -> still dropped.
    double const DEGENERATE_DET_TOL = 1.e-6;
    {
        // squared edge lengths, each unordered pair once (wrapped positions, the frame of Ax)
        double e2[NO_DIM+1][NO_DIM+1];
        for (int v = 0; v <= NO_DIM; ++v)
        {
            e2[v][v] = 0.;
            for (int w = v + 1; w <= NO_DIM; ++w)
            {
                double len2 = 0.;
                for (int i = 0; i < NO_DIM; ++i)
                {
                    double const d = double(g.eulerPos[w][i]) - double(g.eulerPos[v][i]);
                    len2 += d * d;
                }
                e2[v][w] = e2[w][v] = len2;
            }
        }
        double maxAvgEdge2 = 0.;
        for (int v = 0; v <= NO_DIM; ++v)
        {
            // the star's lengths summed in ascending order: the same bits whatever the labelling
            double star[NO_DIM];
            int n = 0;
            for (int w = 0; w <= NO_DIM; ++w)
                if (w != v) star[n++] = e2[v][w];
            std::sort(star, star + NO_DIM);
            double avgEdge2 = 0.;
            for (int k = 0; k < NO_DIM; ++k) avgEdge2 += star[k];
            avgEdge2 /= NO_DIM;
            maxAvgEdge2 = std::max(maxAvgEdge2, avgEdge2);
        }
        double edgeScale = maxAvgEdge2 * std::sqrt(maxAvgEdge2); // (max over vertices of the mean edge)^3
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


/* --ps-vertex-mass, shared by the grid deposit (ps_interpolation.cc) and the point evaluation (ps_point_eval.cc,
   2026-10-13). The grid deposit's degree pass and share mass moved here unchanged, so that both read one
   definition.

   psForEachDepositingCell(dt, ..., f) calls f(cell, geometry) for every finite cell that DEPOSITS -- every
   chart-independent drop filter of psFilterCell, the Lagrangian ownership skipped (it decides WHICH partition
   deposits a cell, not WHETHER; see the 'skipOwnership' note above) -- in the tessellation's cell order.
   psVertexDegreePass counts those cells per vertex after a reset, so a second pass over the same tessellation
   (the unaveraged and averaged grid passes share one) gives the same counts. A tetrahedron's mass is then the sum
   of its vertices' weight/degree shares (psVertexShareMass), so the masses of the deposited tetrahedra add up to
   the particles' masses exactly. */
template <class F>
inline void psForEachDepositingCell(DT &dt, User_options &userOptions, Box &boxCoordinates, F &&f)
{
    size_t degDropped = 0;   // local counter: the deposit loop reports its own
#if NO_DIM==2
    for (DT::Finite_faces_iterator itC = dt.finite_faces_begin(); itC != dt.finite_faces_end(); ++itC)
#else
    for (DT::Finite_cells_iterator itC = dt.finite_cells_begin(); itC != dt.finite_cells_end(); ++itC)
#endif
    {
        Cell_handle cell = itC;
        PSCellGeometry dgeo;
        if ( !psFilterCell(cell, userOptions, boxCoordinates, true, &degDropped, dgeo,
                           /*skipOwnership=*/true) )
            continue;
        f( cell, dgeo );
    }
}

inline void psVertexDegreePass(DT &dt, User_options &userOptions, Box &boxCoordinates)
{
    for (DT::Finite_vertices_iterator vit = dt.finite_vertices_begin(); vit != dt.finite_vertices_end(); ++vit)
        vit->info().psDegree() = 0;
    psForEachDepositingCell( dt, userOptions, boxCoordinates, [](Cell_handle &cell, PSCellGeometry const &)
    {
        for (int v = 0; v <= NO_DIM; ++v)
            cell->vertex(v)->info().psDegree() += 1;
    } );
}

// per-tet vertex-share mass: sum over the cell's vertices of weight/degree (a degree that is not positive -- not
// computed, or every incident cell dropped -- contributes nothing)
inline double psVertexShareMass(Cell_handle const &cell)
{
    double m = 0.;
    for (int v = 0; v <= NO_DIM; ++v)
    {
        int32_t const deg = cell->vertex(v)->info().psDegree();
        if (deg > 0)
            m += double(cell->vertex(v)->info().weight()) / double(deg);
    }
    return m;
}

#endif  // PHASE_SPACE

#endif
