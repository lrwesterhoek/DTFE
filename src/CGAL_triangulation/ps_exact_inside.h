/* Exact, tie-consistent point-in-tetrahedron test, shared by the point evaluation
   (ps_point_eval.cc, both binaries) and the CPU grid deposit (ps_interpolation.cc).

   A point must be counted by exactly one tetrahedron of every stream, including a point ON a
   shared face, edge or vertex -- which a float test with a tolerance gets wrong both ways: it
   counts a point near a shared face in both neighbours, and whether it does depends on the
   rounding of each run's periodic copies. Two ingredients make the decision exact:
     * every periodic image of a particle is mapped to one canonical coordinate
       (canonicalCoord), so tetrahedra sharing a face see bit-identical vertices even when one
       holds the original particle and the other its (float-shifted) periodic copy;
     * the decision uses exact orientation predicates (CGAL, filtered) with a symbolic
       perturbation of the query point (simulation of simplicity) that breaks every tie the
       same way in all tetrahedra.
   Three tiers keep it cheap, each deciding only what it provably can: (1) barycentrics on the
   wrapped ('fine') geometry, outside the tetrahedron's fineBand; (2) a double-precision
   classification in the canonical frame, outside its own much narrower band; (3) the exact
   predicate. Every tier-1/2 decision equals the exact one, so the whole test IS the exact one,
   at the cost of a plain barycentric test for all but the points hugging a face.

   The consistency argument needs every periodic copy to be ONE Real +-L addition to the
   original (DTFE.cpp's global copies, subpartition.h's per-partition copies, the standard
   binary's findParticlesInBox) -- keep it that way.

   2D (a DIM=2 phase-space build, 2026-10-02): the same three tiers for a point in a TRIANGLE. The
   frame, the canonical images and tier 1 are dimension-free; the exact predicate is the 2D
   orientation, and its simulation of simplicity reduces to exact comparisons of coordinates. */

#ifndef PS_EXACT_INSIDE_HEADER
#define PS_EXACT_INSIDE_HEADER

#if NO_DIM==3 || NO_DIM==2

#include <algorithm>
#include <cmath>
#include <limits>

namespace psExactInside {

// The canonical-image frame. canonHi/canonL are Real and computed exactly like the
// periodic-copy generators' box length, so canonicalCoord replays the very float additions
// that created the copies.
struct Frame
{
    bool   periodic = false;
    Real   canonHi[NO_DIM], canonL[NO_DIM];
    double canonLd[NO_DIM];
    double epsAbs = 0.;     // bound on |wrapped Eulerian geometry - canonical geometry| per vertex
    double roundAbs = 0.;   // double rounding scale of a canonical relative coordinate

    void init(User_options &userOptions)   // non-const: Box has no const operator[]
    {
        periodic = userOptions.periodic;
        // the box length is formed with the SAME Real expression as every periodic-copy
        // generator (DTFE.cpp eulerLen/eulerLenArr, subpartition.h findParticlesInBox)
        double scale = 0.;
        for (int d = 0; d < NO_DIM; ++d)
        {
            canonHi[d] = userOptions.boxCoordinates[2*d+1];
            canonL[d]  = userOptions.boxCoordinates[2*d+1] - userOptions.boxCoordinates[2*d];
            canonLd[d] = double( canonL[d] );
            scale = std::max( scale, std::fabs(double(userOptions.boxCoordinates[2*d]))
                                     + std::fabs(double(userOptions.boxCoordinates[2*d+1]))
                                     + 2. * std::fabs(canonLd[d]) );
        }
        // relative coordinates in the canonical frame are formed from values up to ~scale:
        // a handful of double roundings, bounded generously
        roundAbs = 8. * std::numeric_limits<double>::epsilon() * scale;
        // per-vertex gap between the wrapped Eulerian geometry (psFilterCell / the standard
        // worker's wrap: one Real +-L) and the canonical one (up to two Real +L): at most
        // three half-ulp roundings at magnitude <= 2.5 L, i.e. <= 1.25 eps_Real * scale;
        // taken with a 3x margin. Zero without periodicity (no wraps, no copies).
        epsAbs = periodic
               ? 4. * double( std::numeric_limits<Real>::epsilon() ) * scale + 4. * roundAbs
               : 4. * roundAbs;
    }
};


#if NO_DIM==3
// 3x3 double inverse via the adjugate -- the double-precision twin of matrixInverse()
// (math_functions.h), same relative singularity test so the kept-cell set is identical.
inline bool inverse3x3d(double const m[NO_DIM][NO_DIM], double inv[NO_DIM][NO_DIM])
{
    double const c00 = m[1][1]*m[2][2] - m[1][2]*m[2][1];
    double const c01 = m[1][2]*m[2][0] - m[1][0]*m[2][2];
    double const c02 = m[1][0]*m[2][1] - m[1][1]*m[2][0];
    double const det = m[0][0]*c00 + m[0][1]*c01 + m[0][2]*c02;
    if ( isRelativelySingular(det, m) )
        return false;
    double const invDet = 1.0 / det;
    inv[0][0] = c00 * invDet;
    inv[1][0] = c01 * invDet;
    inv[2][0] = c02 * invDet;
    inv[0][1] = (m[0][2]*m[2][1] - m[0][1]*m[2][2]) * invDet;
    inv[1][1] = (m[0][0]*m[2][2] - m[0][2]*m[2][0]) * invDet;
    inv[2][1] = (m[0][1]*m[2][0] - m[0][0]*m[2][1]) * invDet;
    inv[0][2] = (m[0][1]*m[1][2] - m[0][2]*m[1][1]) * invDet;
    inv[1][2] = (m[0][2]*m[1][0] - m[0][0]*m[1][2]) * invDet;
    inv[2][2] = (m[0][0]*m[1][1] - m[0][1]*m[1][0]) * invDet;
    return true;
}
inline bool inverseNd(double const m[NO_DIM][NO_DIM], double inv[NO_DIM][NO_DIM]) { return inverse3x3d( m, inv ); }
#else
// 2x2 double inverse, the same relative singularity test as matrixInverse()
inline bool inverseNd(double const m[NO_DIM][NO_DIM], double inv[NO_DIM][NO_DIM])
{
    double const det = m[0][0]*m[1][1] - m[0][1]*m[1][0];
    if ( isRelativelySingular(det, m) )
        return false;
    double const invDet = 1.0 / det;
    inv[0][0] =  m[1][1] * invDet;
    inv[0][1] = -m[0][1] * invDet;
    inv[1][0] = -m[1][0] * invDet;
    inv[1][1] =  m[0][0] * invDet;
    return true;
}
#endif


// Canonical image of a stored Eulerian coordinate. Periodic copies are made by ONE Real
// addition of +-L to the original, and '+L' drops low bits when the original is small.
// Replaying '+L' until the value reaches the top of the box maps the original and all of its
// copies to the same float: a +L copy IS the first step of the original's replay, and a -L
// copy (exact by Sterbenz -- such originals lie in the upper half of the box) returns to the
// original exactly on its first step. Non-periodic runs have no copies: identity.
inline Real canonicalCoord(Frame const &F, Real v, int d)
{
    if ( not F.periodic ) return v;
    for (int k = 0; k < 8 && v < F.canonHi[d]; ++k)
        v = v + F.canonL[d];
    return v;
}

// minimum-image wrap of a relative coordinate in the canonical frame
inline double wrapRel(Frame const &F, double r, int d)
{
    if ( F.periodic )
    {
        double const L = F.canonLd[d];
        if (r >  0.5 * L) r -= L;
        else if (r < -0.5 * L) r += L;
    }
    return r;
}

// A query point, wrapped into the box, in the canonical frame (periodic: shifted by +L like
// the vertices). The same double arithmetic for every tetrahedron, so shared vertices get
// bit-identical relative coordinates.
inline void canonicalQuery(Frame const &F, double const pos[NO_DIM], double pc[NO_DIM])
{
    for (int d = 0; d < NO_DIM; ++d)
        pc[d] = pos[d] + (F.periodic ? F.canonLd[d] : 0.);
}

#if NO_DIM==3
// Exact sign of det[a;b;c] -- the orientation of the triangle abc as seen from the query
// point, which sits at the origin of these coordinates -- with the query point symbolically
// perturbed by delta = (e, e^2, e^3), e -> 0+ (simulation of simplicity), so it is never 0.
// Moving the point by +delta moves the vertices by -delta:
//     det[a-delta; b-delta; c-delta] = det[a;b;c] - delta . n,   n = (b-a) x (c-a),
// so a zero determinant takes the sign of -n_x, else -n_y, else -n_z; each n component is the
// exact 2D orientation of the triangle projected onto the other two axes.
inline int orient0Sos(double const a[NO_DIM], double const b[NO_DIM], double const c[NO_DIM])
{
    typedef K::Point_3 P3;
    typedef K::Point_2 P2;
    CGAL::Orientation const o = CGAL::orientation( P3(0., 0., 0.), P3(a[0], a[1], a[2]),
                                                   P3(b[0], b[1], b[2]), P3(c[0], c[1], c[2]) );
    if ( o != CGAL::COPLANAR ) return (o == CGAL::POSITIVE) ? 1 : -1;
    static int const proj[NO_DIM][2] = { {1, 2}, {2, 0}, {0, 1} };   // n_x, n_y, n_z
    for (int k = 0; k < NO_DIM; ++k)
    {
        int const i = proj[k][0], j = proj[k][1];
        CGAL::Orientation const o2 = CGAL::orientation( P2(a[i], a[j]), P2(b[i], b[j]), P2(c[i], c[j]) );
        if ( o2 != CGAL::COLLINEAR ) return (o2 == CGAL::POSITIVE) ? -1 : 1;
    }
    return 1;   // collinear face: only a zero-volume tetrahedron has one, and none is ever kept
}

// Exact containment of the (perturbed) query point pc in the canonical tetrahedron cv. With
// the point at the origin, replacing vertex i by it gives the sub-tetrahedron orientation
// (-1)^i det[the other three, in index order]; the point is inside iff all four match the
// tetrahedron's own orientation. Folded (negatively oriented) streams work unchanged.
inline bool exactInside(Frame const &F, Real const cv[NO_DIM+1][NO_DIM], double const pc[NO_DIM])
{
    typedef K::Point_3 P3;
    double dv[NO_DIM+1][NO_DIM];
    for (int v = 0; v <= NO_DIM; ++v)
        for (int d = 0; d < NO_DIM; ++d)
            dv[v][d] = wrapRel( F, double(cv[v][d]) - pc[d], d );

    CGAL::Orientation const ot = CGAL::orientation( P3(dv[0][0], dv[0][1], dv[0][2]),
                                                    P3(dv[1][0], dv[1][1], dv[1][2]),
                                                    P3(dv[2][0], dv[2][1], dv[2][2]),
                                                    P3(dv[3][0], dv[3][1], dv[3][2]) );
    if ( ot == CGAL::COPLANAR ) return false;   // flat in the canonical frame: holds no volume
    int const o = (ot == CGAL::POSITIVE) ? 1 : -1;

    static int const face[NO_DIM+1][NO_DIM] = { {1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2} };
    for (int i = 0; i <= NO_DIM; ++i)
    {
        int s = orient0Sos( dv[face[i][0]], dv[face[i][1]], dv[face[i][2]] );
        if ( i & 1 ) s = -s;
        if ( s != o ) return false;
    }
    return true;
}
#else
// Exact sign of det[a;b] -- the orientation of the segment ab as seen from the query point at the
// origin -- with the point perturbed by delta = (e, e^2), e -> 0+ (simulation of simplicity):
//     det[a-delta; b-delta] = det[a;b] - e (b_y - a_y) + e^2 (b_x - a_x) + O(e^3 terms that cancel),
// so a zero determinant takes the sign of a_y - b_y, else of b_x - a_x: exact comparisons of doubles.
inline int orient0Sos(double const a[NO_DIM], double const b[NO_DIM])
{
    typedef K::Point_2 P2;
    CGAL::Orientation const o = CGAL::orientation( P2(0., 0.), P2(a[0], a[1]), P2(b[0], b[1]) );
    if ( o != CGAL::COLLINEAR ) return (o == CGAL::POSITIVE) ? 1 : -1;
    if ( a[1] != b[1] ) return ( a[1] > b[1] ) ? 1 : -1;
    if ( a[0] != b[0] ) return ( b[0] > a[0] ) ? 1 : -1;
    return 1;   // a zero-length edge: only a zero-area triangle has one, and none is ever kept
}

// Exact containment of the (perturbed) query point pc in the canonical triangle cv: with the point
// at the origin, the sub-triangles (O,v1,v2), (O,v2,v0), (O,v0,v1) must all share the triangle's
// orientation; an edge shared by two triangles is seen in opposite directions, so a point on it
// goes to exactly one of them. Folded (negatively oriented) streams work unchanged.
inline bool exactInside(Frame const &F, Real const cv[NO_DIM+1][NO_DIM], double const pc[NO_DIM])
{
    typedef K::Point_2 P2;
    double dv[NO_DIM+1][NO_DIM];
    for (int v = 0; v <= NO_DIM; ++v)
        for (int d = 0; d < NO_DIM; ++d)
            dv[v][d] = wrapRel( F, double(cv[v][d]) - pc[d], d );
    CGAL::Orientation const ot = CGAL::orientation( P2(dv[0][0], dv[0][1]), P2(dv[1][0], dv[1][1]),
                                                    P2(dv[2][0], dv[2][1]) );
    if ( ot == CGAL::COLLINEAR ) return false;   // flat in the canonical frame: holds no area
    int const o = (ot == CGAL::POSITIVE) ? 1 : -1;
    static int const edge[NO_DIM+1][2] = { {1, 2}, {2, 0}, {0, 1} };
    for (int i = 0; i <= NO_DIM; ++i)
        if ( orient0Sos( dv[edge[i][0]], dv[edge[i][1]] ) != o ) return false;
    return true;
}
#endif

// Per-tetrahedron state of the canonical tiers, built lazily on the first undecided point.
struct TetTest
{
    Real   cv[NO_DIM+1][NO_DIM];   // canonical vertex coordinates (the exact test's geometry)
    double invC[NO_DIM][NO_DIM];   // inverse of the canonical edge matrix (fast classification)
    double band = 0.;              // barycentric uncertainty of the fast classification
    bool   ready = false;
    bool   fast = false;           // false: every candidate goes to the exact test
};

// rawPos: the STORED (unwrapped) Eulerian vertex positions -- the ones canonicalCoord maps to
// a single image per particle.
#if NO_DIM==3
inline void tetTestSetup(Frame const &F, Real const rawPos[NO_DIM+1][NO_DIM], TetTest &T)
{
    for (int v = 0; v <= NO_DIM; ++v)
        for (int d = 0; d < NO_DIM; ++d)
            T.cv[v][d] = canonicalCoord( F, rawPos[v][d], d );

    // canonical edges (rows), exact up to the wrap: float differences are exact in double
    double E[NO_DIM][NO_DIM];
    for (int v = 0; v < NO_DIM; ++v)
        for (int d = 0; d < NO_DIM; ++d)
            E[v][d] = wrapRel( F, double(T.cv[v+1][d]) - double(T.cv[0][d]), d );

    auto cross = [](double const a[NO_DIM], double const b[NO_DIM], double n[NO_DIM])
    {
        n[0] = a[1]*b[2] - a[2]*b[1];
        n[1] = a[2]*b[0] - a[0]*b[2];
        n[2] = a[0]*b[1] - a[1]*b[0];
    };
    auto norm2 = [](double const a[NO_DIM]) { return a[0]*a[0] + a[1]*a[1] + a[2]*a[2]; };

    // adjugate inverse, no zero guard: tiny tetrahedra just get a wide band (or none)
    double const c00 = E[1][1]*E[2][2] - E[1][2]*E[2][1];
    double const c01 = E[1][2]*E[2][0] - E[1][0]*E[2][2];
    double const c02 = E[1][0]*E[2][1] - E[1][1]*E[2][0];
    double const det = E[0][0]*c00 + E[0][1]*c01 + E[0][2]*c02;

    // largest face (|n| = 2 x area) -> smallest height h = |det| / |n|, and the longest edge
    double e3[NO_DIM], e4[NO_DIM], e5[NO_DIM], n[NO_DIM];
    for (int d = 0; d < NO_DIM; ++d)
    {
        e3[d] = E[1][d] - E[0][d];
        e4[d] = E[2][d] - E[0][d];
        e5[d] = E[2][d] - E[1][d];
    }
    double nMax2 = 0.;
    cross(E[1], E[2], n); nMax2 = std::max( nMax2, norm2(n) );
    cross(E[0], E[2], n); nMax2 = std::max( nMax2, norm2(n) );
    cross(E[0], E[1], n); nMax2 = std::max( nMax2, norm2(n) );
    cross(e3, e4, n);     nMax2 = std::max( nMax2, norm2(n) );
    double const edge2 = std::max( { norm2(E[0]), norm2(E[1]), norm2(E[2]), norm2(e3), norm2(e4), norm2(e5) } );

    T.fast = false;
    double const absDet = std::fabs(det);
    if ( absDet > 0. && nMax2 > 0. )
    {
        double const hMin  = absDet / std::sqrt(nMax2);
        double const edge  = std::sqrt(edge2);
        // relative-coordinate rounding (roundAbs) and the adjugate's own rounding (~eps x
        // edge^3/|det|), each with a wide safety factor; the 1e-8 floor dwarfs both for any
        // reasonably shaped tetrahedron
        double const band = 1.e-8 + 4. * F.roundAbs / hMin
                          + 64. * std::numeric_limits<double>::epsilon() * edge * edge * edge / absDet;
        if ( band < 0.25 )
        {
            double const invDet = 1. / det;
            T.invC[0][0] = c00 * invDet;
            T.invC[1][0] = c01 * invDet;
            T.invC[2][0] = c02 * invDet;
            T.invC[0][1] = (E[0][2]*E[2][1] - E[0][1]*E[2][2]) * invDet;
            T.invC[1][1] = (E[0][0]*E[2][2] - E[0][2]*E[2][0]) * invDet;
            T.invC[2][1] = (E[0][1]*E[2][0] - E[0][0]*E[2][1]) * invDet;
            T.invC[0][2] = (E[0][1]*E[1][2] - E[0][2]*E[1][1]) * invDet;
            T.invC[1][2] = (E[0][2]*E[1][0] - E[0][0]*E[1][2]) * invDet;
            T.invC[2][2] = (E[0][0]*E[1][1] - E[0][1]*E[1][0]) * invDet;
            T.band = band;
            T.fast = true;
        }
    }
    T.ready = true;
}
#else
inline void tetTestSetup(Frame const &F, Real const rawPos[NO_DIM+1][NO_DIM], TetTest &T)
{
    for (int v = 0; v <= NO_DIM; ++v)
        for (int d = 0; d < NO_DIM; ++d)
            T.cv[v][d] = canonicalCoord( F, rawPos[v][d], d );
    double E[NO_DIM][NO_DIM];
    for (int v = 0; v < NO_DIM; ++v)
        for (int d = 0; d < NO_DIM; ++d)
            E[v][d] = wrapRel( F, double(T.cv[v+1][d]) - double(T.cv[0][d]), d );
    double const det = E[0][0]*E[1][1] - E[0][1]*E[1][0];
    double const e2[3] = { E[0][0]*E[0][0] + E[0][1]*E[0][1], E[1][0]*E[1][0] + E[1][1]*E[1][1],
                           (E[1][0]-E[0][0])*(E[1][0]-E[0][0]) + (E[1][1]-E[0][1])*(E[1][1]-E[0][1]) };
    double const edge2 = std::max( { e2[0], e2[1], e2[2] } );
    T.fast = false;
    double const absDet = std::fabs(det);
    if ( absDet > 0. && edge2 > 0. )
    {
        double const edge = std::sqrt(edge2);
        double const hMin = absDet / edge;            // smallest height: twice the area over the longest edge
        double const band = 1.e-8 + 4. * F.roundAbs / hMin
                          + 64. * std::numeric_limits<double>::epsilon() * edge2 / absDet;
        if ( band < 0.25 )
        {
            double const invDet = 1. / det;
            // the plain inverse of the edge matrix (rows = edges), as in 3D; canonicalInside reads
            // bary = inv^T * rel
            T.invC[0][0] =  E[1][1] * invDet;
            T.invC[0][1] = -E[0][1] * invDet;
            T.invC[1][0] = -E[1][0] * invDet;
            T.invC[1][1] =  E[0][0] * invDet;
            T.band = band;
            T.fast = true;
        }
    }
    T.ready = true;
}
#endif

// Barycentric margin within which the wrapped-geometry ('fine') barycentrics cannot decide
// containment on their own: the fine and canonical geometries differ by <= epsAbs per vertex,
// which moves a barycentric coordinate by <= 4 epsAbs / h (h = smallest height >= |det| /
// longest-edge^2), plus the adjugate rounding. Cheap per tetrahedron, from the edge matrix.
inline double fineBand(Frame const &F, double const Ax[NO_DIM][NO_DIM], double const absDet)
{
    double edge2 = 0.;
    for (int a = 0; a < NO_DIM; ++a)
    {
        double l2 = 0.;
        for (int i = 0; i < NO_DIM; ++i) l2 += Ax[a][i] * Ax[a][i];
        edge2 = std::max( edge2, l2 );
        for (int b = a + 1; b < NO_DIM; ++b)
        {
            double m2 = 0.;
            for (int i = 0; i < NO_DIM; ++i) { double const t = Ax[b][i] - Ax[a][i]; m2 += t * t; }
            edge2 = std::max( edge2, m2 );
        }
    }
    if ( not (absDet > 0.) ) return std::numeric_limits<double>::infinity();
#if NO_DIM==3
    return 1.e-8 + 4. * (F.epsAbs + F.roundAbs) * edge2 / absDet
         + 64. * std::numeric_limits<double>::epsilon() * edge2 * std::sqrt(edge2) / absDet;
#else
    // 2D: the smallest height is >= |det| / longest edge, and the adjugate's rounding ~ edge^2 / |det|
    return 1.e-8 + 4. * (F.epsAbs + F.roundAbs) * std::sqrt(edge2) / absDet
         + 64. * std::numeric_limits<double>::epsilon() * edge2 / absDet;
#endif
}

// Tier 1: barycentrics of rel (the query's minimum-image offset from wrapped vertex 0) on the
// fine geometry. Returns +1 / -1 when they decide inside / outside beyond band1, 0 when the
// point is within band1 of a face. Ax stores edges as rows, so bary = inv^T * rel -- note the
// [i][v] transpose. bary[] is filled (as the interpolation weights) whenever the result is not -1.
inline int fineClassify(double const inv[NO_DIM][NO_DIM], double const rel[NO_DIM],
                        double const band1, double bary[NO_DIM])
{
    double sum = 0.;
    bool unsure = false;
    for (int v = 0; v < NO_DIM; ++v)
    {
        double bc = 0.;
        for (int i = 0; i < NO_DIM; ++i)
            bc += inv[i][v] * rel[i];
        if ( bc < -band1 ) return -1;
        if ( bc <= band1 ) unsure = true;
        bary[v] = bc;
        sum += bc;
    }
    double const l0 = 1. - sum;
    if ( l0 < -band1 ) return -1;
    if ( l0 <= band1 ) unsure = true;
    return unsure ? 0 : 1;
}

// Tiers 2 and 3, for a point tier 1 could not decide: pc = the query in the canonical frame
// (canonicalQuery), rawPos = the tetrahedron's stored vertex positions.
inline bool canonicalInside(Frame const &F, Real const rawPos[NO_DIM+1][NO_DIM], TetTest &T,
                            double const pc[NO_DIM])
{
    if ( not T.ready ) tetTestSetup( F, rawPos, T );

    if ( T.fast )
    {
        double r[NO_DIM];
        for (int d = 0; d < NO_DIM; ++d)
            r[d] = wrapRel( F, pc[d] - double(T.cv[0][d]), d );
        double sum = 0.;
        bool unsure = false;
        for (int v = 0; v < NO_DIM; ++v)
        {
            double bc = 0.;
            for (int i = 0; i < NO_DIM; ++i)
                bc += T.invC[i][v] * r[i];
            if ( bc < -T.band ) return false;
            if ( bc <= T.band ) unsure = true;
            sum += bc;
        }
        double const l0 = 1. - sum;
        if ( l0 < -T.band ) return false;
        if ( l0 <= T.band ) unsure = true;
        if ( not unsure ) return true;
    }
    return exactInside( F, T.cv, pc );
}

}  // namespace psExactInside

#endif  // NO_DIM==3 || NO_DIM==2

#endif  // PS_EXACT_INSIDE_HEADER
