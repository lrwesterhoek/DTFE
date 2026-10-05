/* The 2D counterpart of r3d for '--ps-exact-deposit' in a DIM=2 phase-space build: a triangle clipped
   to an axis-aligned grid cell, and the moments of the piece up to second order.

   r3d (third_party/r3d, Powell & Abel 2015) clips a tetrahedron against a cell's six planes and
   integrates [1, x, y, z, xx, xy, xz, yy, yz, zz] over the piece. In 2D the piece of a triangle
   inside a square is a convex polygon of at most 7 vertices, found by clipping against the four
   edges in turn (Sutherland-Hodgman), and its moments [1, x, y, xx, xy, yy] follow exactly from the
   vertices (Green's theorem over the boundary). The moment layout matches r3d's -- order 0, the
   first moments, then the second moments a <= b row by row -- so the deposit reads both the same way.

   The triangle must be counter-clockwise (positive area); the caller swaps two vertices of a
   folded one, as it does for r3d. */

#ifndef PS_CLIP2D_HEADER
#define PS_CLIP2D_HEADER

namespace psClip2d {

static int const MOMENTS = 6;          // [1, x, y, xx, xy, yy]

struct Polygon
{
    double x[16], y[16];
    int n = 0;
};

// keep the part with sign * (coordinate[axis] - bound) >= 0
inline void clipHalfPlane(Polygon const &in, int const axis, double const bound, double const sign, Polygon &out)
{
    out.n = 0;
    if ( in.n == 0 ) return;
    auto value = [&](int i) { return sign * ( (axis == 0 ? in.x[i] : in.y[i]) - bound ); };
    for (int i = 0; i < in.n; ++i)
    {
        int const j = (i + 1) % in.n;
        double const vi = value(i), vj = value(j);
        bool const inI = vi >= 0., inJ = vj >= 0.;
        if ( inI ) { out.x[out.n] = in.x[i]; out.y[out.n] = in.y[i]; ++out.n; }
        if ( inI != inJ )
        {
            double const t = vi / (vi - vj);
            out.x[out.n] = in.x[i] + t * (in.x[j] - in.x[i]);
            out.y[out.n] = in.y[i] + t * (in.y[j] - in.y[i]);
            if ( axis == 0 ) out.x[out.n] = bound; else out.y[out.n] = bound;   // exactly on the edge
            ++out.n;
        }
    }
}

/* tri: the triangle's vertices (counter-clockwise), lo/hi: the cell; mom: the moments of the
   intersection. Returns false when the intersection is empty. */
inline bool clipTriangleToCell(double const tri[3][2], double const lo[2], double const hi[2],
                               double mom[MOMENTS])
{
    Polygon a, b;
    a.n = 3;
    for (int v = 0; v < 3; ++v) { a.x[v] = tri[v][0]; a.y[v] = tri[v][1]; }
    clipHalfPlane( a, 0, lo[0],  1., b );
    clipHalfPlane( b, 0, hi[0], -1., a );
    clipHalfPlane( a, 1, lo[1],  1., b );
    clipHalfPlane( b, 1, hi[1], -1., a );
    for (int m = 0; m < MOMENTS; ++m) mom[m] = 0.;
    if ( a.n < 3 ) return false;
    // Green's theorem over the boundary, edge (i, j): c = x_i y_j - x_j y_i
    for (int i = 0; i < a.n; ++i)
    {
        int const j = (i + 1) % a.n;
        double const xi = a.x[i], yi = a.y[i], xj = a.x[j], yj = a.y[j];
        double const c = xi * yj - xj * yi;
        mom[0] += c;
        mom[1] += (xi + xj) * c;
        mom[2] += (yi + yj) * c;
        mom[3] += (xi*xi + xi*xj + xj*xj) * c;
        mom[4] += (xi*yj + 2.*xi*yi + 2.*xj*yj + xj*yi) * c;
        mom[5] += (yi*yi + yi*yj + yj*yj) * c;
    }
    mom[0] /= 2.;  mom[1] /= 6.;  mom[2] /= 6.;
    mom[3] /= 12.; mom[4] /= 24.; mom[5] /= 12.;
    return mom[0] > 0.;
}

}   // namespace psClip2d

#endif
