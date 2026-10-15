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



/* Functions that select the particles inside a sub-box of the main data box. */





#ifndef SUBPARTITION_HEADER
#define SUBPARTITION_HEADER


#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <algorithm>
#include <utility>
#include <cmath>

#include "define.h"
#include "particle_data.h"
#include "user_options.h"
#include "quantities.h"
#include "box.h"
#include "miscellaneous.h"
#include "message.h"
#include <limits>
#ifdef OPEN_MP
    #include <omp.h>
#endif

#include <boost/timer.hpp>

typedef std::vector<Particle_data>::iterator   vectorIterator;


void insertParticlesInBox(std::vector<Particle_data> &p,
                        std::vector<Particle_data> *output,
                        Box const &box,
                        Real const offset[]);

#ifdef PHASE_SPACE
// Select particles whose Lagrangian position lies in the padded box (PS-DTFE partitions are Lagrangian),
// generating periodic images on the fly (deferred copies, no whole-box array); shifting lagPos and pos
// by the same eulerLen[] offset matches DTFE()'s global copy generator bit-for-bit.
void findParticlesInBoxLagrangianPeriodic(std::vector<Particle_data> &p,
                                          std::vector<Particle_data> *output,
                                          Box const &paddedBox,
                                          Real const eulerLen[],
                                          bool const periodic,
                                          int verboseLevel)
{
    MESSAGE::Message message( verboseLevel );
    message << "Finding particles (with on-the-fly periodic images) with Lagrangian positions in box " << paddedBox.print() << " ... " << MESSAGE::Flush;
    output->reserve( p.size() / 4 );  // rough estimate

    size_t const noOffsets = periodic ? (NO_DIM==2 ? 9 : 27) : 1;
    for (size_t n=0; n<noOffsets; ++n)
    {
        // periodic-image offset for this pass; applied identically to lagPos and pos so simplices keep correct geometry
        Real off[NO_DIM];
        if ( not periodic )
            for (int d=0; d<NO_DIM; ++d) off[d] = Real(0.);
        else
        {
            size_t rem = n;
            for (int d=NO_DIM-1; d>=0; --d)
            { off[d] = Real(int(rem%3) - 1) * eulerLen[d]; rem /= 3; }
        }

        for (vectorIterator it=p.begin(); it!=p.end(); ++it)
        {
            bool inside = true;
            for (size_t i=0; i<NO_DIM; ++i)
            {
                Real const shiftedLag = it->lagPos[i] + off[i];
                if ( shiftedLag<paddedBox.coords[2*i] or shiftedLag>paddedBox.coords[2*i+1] )
                { inside = false; break; }
            }
            if (inside)
            {
                Particle_data copy = *it;
                for (int d=0; d<NO_DIM; ++d)
                { copy.lagPos[d] += off[d]; copy.pos[d] += off[d]; }
                output->push_back( copy );
            }
        }
    }

    message << "Done.\n"
            << "\t Selected " << output->size() << " particles (originals + periodic images) for this partition.\n" << MESSAGE::Flush;
}


/* The periodic SEAM of the Lagrangian ownership tiling (2026-10-13).

   A periodic PS-DTFE run tiles ONE period [s - L, s) of Lagrangian space per axis: a tetrahedron is
   deposited by the copy (the original or a periodic image) whose Lagrangian centroid lies in that period, or
   in the partition region that tiles it (psFilterCell). That is exclusive and exhaustive only if the
   tessellation around the two edges of the period is the SAME up to the shift L -- the copies that sit
   beyond the edges must be exact translates of their originals. They were not. The period used to end at the
   top particle plane, s = lagBox_hi (2026-10-02), so the padding above it came from the +L copies of the
   bottom particles, made in Real arithmetic: y + L of a y near 0 keeps only ulp(L) (2^-7 kpc/h at L = 75
   Mpc/h), so those copies sat up to half an ulp(L) off the translates of the originals they stand for. On
   near-lattice initial conditions (TNG's IC positions are a lattice displaced by a small fraction of a
   spacing) the Delaunay tessellation is near-degenerate in every cube, and that offset flips some of its
   choices: the tetrahedra above the seam are not the translates of those below it, so the tiles of the two
   frames overlap or leave gaps -- whole tetrahedra deposited twice or never, and under --ps-vertex-mass a
   particle whose copies have different degrees deposits more or less than its weight. That is the likely
   source of the 2026-10-13 audit's seam defect on TNG (up to 394 double-covered and tens of uncovered
   '.a_streams' cells within a few cells of the box faces, moving with the flow; a +-1e-7 box-mass offset that
   is constant in z) -- inferred, not measured on TNG. Measured on synthetic periodic boxes (PS_MASS_AUDIT=2,
   tests/ps_periodic_seam_check.sh): a 32^3 lattice jittered by 1e-5 spacings has ~400 seam particles whose
   shares do not close (their copies' degrees 22 vs 26) and +6e-5 of the box mass; a box across the float
   binade 2^17 (its +L copies round to 2^-6) has 177 uncovered and 18 double-covered cells and -6.4e-5 of the
   mass. Quantizing its coordinates so that every shift is exact removes all of it, and no ownership rule
   alone does: the min-particle-ID vertex instead of the centroid still left 68 holes and 65 doubles -- the
   tessellations differ, not just the test.

   The fix moves the seam to where every copy IS exact: s = lagBox_lo + L - pad (less a margin), as high as
   the padding allows without any +L copy, so the padded period [s - L - pad, s + pad] holds the originals and
   the -L copies of the originals above s - pad only, and the generators (DTFE()'s global copies and
   findParticlesInBoxLagrangianPeriodic) make no other. A float y - L is exact whenever its result is no
   coarser than y and L (psShiftDownExact; Sterbenz's lemma for y in [L/2, 2L], i.e. any snapshot box [0, L)
   with pad < L/4): the copies are then translates bit for bit, the tessellation of the padded period is the
   periodic one, and the centroid ownership tiles it exactly (psFilterCell's relative centroid test). Nothing
   about a tetrahedron's own geometry or its keep/drop decision changes; only WHICH copy of a seam tetrahedron
   deposits it, and the partition boundaries (they tile [s - L, s) now: every partition's tessellation cache
   identity changes once). When no such s exists (a padding above ~L/4 -- tiny test boxes, '--ps-lag-padding
   0' on one partition -- or a box whose coordinates do not allow exact shifts), the axis keeps the
   2026-10-02 seam, as before. */

// The granularity of a Real: the ulp of the binade of x (x > 0), as a double.
inline double psRealUlp(double const x)
{
    int e = 0;
    std::frexp( x, &e );
    return std::ldexp( 1., e - std::numeric_limits<Real>::digits );
}

// True when y - L is exactly representable for every Real y in [A, B] (0 < A <= B, L > 0): the result's
// coarsest ulp (that of its largest magnitude) is no coarser than the inputs' common granularity -- the ulp of
// A (the finest of the range) and the lowest set bit of L. Every such y - L is a multiple of that granularity
// and below 2^digits of its own ulp, so it is a Real. (Sterbenz's lemma is the case A >= L/2, B <= 2L.)
inline bool psShiftDownExact(double const A, double const B, Real const L)
{
    if ( !( A > 0. ) or !( B >= A ) or !( L > Real(0.) ) ) return false;
    double const Ld = double(L);
    double lsb = psRealUlp( Ld );
    while ( std::fmod( Ld, 2. * lsb ) == 0. ) lsb *= 2.;
    double const maxRes = std::max( std::fabs( A - Ld ), std::fabs( B - Ld ) );
    if ( maxRes == 0. ) return true;
    return psRealUlp( maxRes ) <= std::min( psRealUlp( A ), lsb );
}

// The top edge s of the ownership period [s - L, s) on one axis (see above), for the global Lagrangian bounding
// box [lagLo, lagHi] (computeLagrangianBoundingBox: the particles' extent plus a 1e-6 margin), the period L and
// the padding 'pad' of the copies around the period (the single triangulation's, or the largest a partition
// adds). Returns true and the exact seam, or false and the old anchor lagHi.
inline bool psPeriodicSeam(Real const lagLo, Real const lagHi, Real const L, Real const pad, Real &seam)
{
    seam = lagHi;
    double const Ld = double(L), padD = double(pad);
    if ( !( Ld > 0. ) or !( padD >= 0. ) or !( lagHi >= lagLo ) ) return false;
    // room for the float rounding of the boxes built from s (s - L - pad, s + pad, the partition edges):
    // 4e-6 L, or 8 ulps of the largest coordinate involved when that is coarser (a box far from the origin)
    double const m = std::max( 4.e-6 * Ld, 8. * psRealUlp( std::max( std::fabs( double(lagLo) ), std::fabs( double(lagHi) ) ) + Ld ) );
    // as high as possible below the lowest +L copy, lagLo + L
    Real const s = Real( double(lagLo) + Ld - padD - m );
    double const sD = double(s);
    if ( !( sD + padD < double(lagLo) + Ld - 0.5 * m ) ) return false;   // a +L copy would join the padding
    if ( !( sD > double(lagHi) - Ld ) ) return false;                   // every original has a copy in the period
    // the -L copies: of every original above the padded bottom edge (with m/2 for the boxes' rounding), and
    // the bottom edge s - L itself
    if ( !psShiftDownExact( sD - padD - 0.5 * m, std::max( double(lagHi), sD ), L ) ) return false;
    seam = s;
    return true;
}

// How many of the -L copies of the originals above 'from' (on axis d) are NOT exact shifts, Lagrangian or
// Eulerian (the Lagrangian ones are exact by psPeriodicSeam's construction; the Eulerian position is the
// unwrapped pos = lagPos + displacement, exact when it is no coarser -- for a snapshot box [0, L), when pos
// >= L/2). An inexact Eulerian copy leaves a sliver of rounding width between the deposits of the two frames
// at the seam: reported, not fixed.
inline size_t psCountInexactCopies(std::vector<Particle_data> const &p, int const d, double const from, Real const L)
{
    size_t n = 0;
    double const Ld = double(L);
    for (Particle_data const &q : p)
    {
        if ( double(q.lagPos[d]) < from ) continue;
        Real const lc = q.lagPos[d] - L, ec = q.pos[d] - L;
        if ( double(lc) != double(q.lagPos[d]) - Ld or double(ec) != double(q.pos[d]) - Ld ) ++n;
    }
    return n;
}
#endif




// Prints the elapsed wall-clock time for a step and adds it to userOptions->totalTime.
void printElapsedTime(boost::timer *t, User_options *userOptions,
                      std::string computationQuantityName)
{
    userOptions->totalTime += t->elapsed();
    MESSAGE::Message message( userOptions->verboseLevel );
    message << "  >>> Time: " << t->elapsed()/userOptions->noProcessors << " sec. (" << computationQuantityName << ")\n" << MESSAGE::Flush;
    t->restart();
}



// true when selectParticlesInBoxesOrdered() takes its parallel path for N particles
inline bool selectionRunsParallel(size_t const N)
{
#ifdef OPEN_MP
    return N >= (size_t(1) << 18) and not omp_in_parallel() and omp_get_max_threads() >= 2;
#else
    (void)N;
    return false;
#endif
}

/* Ordered parallel box selection (2026-10-12). Appends to 'output' every particle of 'p' inside boxes[k]
   (stored shifted back by -offsets[k] when 'shift'), BOX-MAJOR and in particle order within each box: exactly
   the vector that calling insertParticlesInBox( p, output, boxes[k], offsets[k] ) for k = 0, 1, ... in turn
   produces, byte for byte (same particles, same order, same 'pos - offset' arithmetic) -- the spatial sort and
   the CGAL insertion downstream see the same input. It used to be serial and, periodic, scanned all N particles
   once per periodic offset (Box::isBoxOverlaping is true for practically every pair, so all 27 offsets): ~2 s per
   partition at 9.4e7 particles with the other cores idle, 4% of a TNG100 run, ~1/3 of a TNG300 one.
   Now: (1) offsets whose box cannot hold a particle are dropped EXACTLY, against the particles' own bounding box
   (closed intervals, the test isParticleInBox makes; a NaN coordinate -- which every box test passes -- keeps
   all boxes); (2) one count pass and one fill pass over contiguous chunks, the fill writing straight into its
   final slot (prefix sums over box-major, chunk-minor counts), so the order never depends on the threads and
   the vector is allocated once at its exact size. Serial (the old loops) inside an enclosing parallel region
   -- DTFE_parallel's threads and the '--serve' composite call this concurrently -- and for small inputs. */
inline void selectParticlesInBoxesOrdered(std::vector<Particle_data> &p,
                                          std::vector<Particle_data> *output,
                                          std::vector<Box> const &boxesIn,
                                          std::vector< std::vector<Real> > const &offsetsIn,
                                          bool const shift)
{
    size_t const N = p.size();
    if ( not selectionRunsParallel( N ) )
    {
        for (size_t k=0; k<boxesIn.size(); ++k)
        {
            if ( shift )
                insertParticlesInBox( p, output, boxesIn[k], offsetsIn[k].data() );
            else
                for (vectorIterator it=p.begin(); it!=p.end(); ++it)
                    if ( boxesIn[k].isParticleInBox(*it) )
                        output->push_back( *it );
        }
        return;
    }
#ifdef OPEN_MP
    // (1) the particles' bounding box, and whether any coordinate is NaN
    Real pMin[NO_DIM], pMax[NO_DIM];
    for (int d=0; d<NO_DIM; ++d) { pMin[d] = std::numeric_limits<Real>::infinity(); pMax[d] = -pMin[d]; }
    int nanSeen = 0;
    #pragma omp parallel
    {
        Real lMin[NO_DIM], lMax[NO_DIM];
        for (int d=0; d<NO_DIM; ++d) { lMin[d] = pMin[d]; lMax[d] = pMax[d]; }
        int lNan = 0;
        #pragma omp for schedule(static) nowait
        for (size_t i=0; i<N; ++i)
            for (int d=0; d<NO_DIM; ++d)
            {
                Real const x = p[i].pos[d];
                if ( x != x ) lNan = 1;
                if ( x < lMin[d] ) lMin[d] = x;
                if ( x > lMax[d] ) lMax[d] = x;
            }
        #pragma omp critical(selectParticlesBBox)
        {
            for (int d=0; d<NO_DIM; ++d) { pMin[d] = std::min( pMin[d], lMin[d] ); pMax[d] = std::max( pMax[d], lMax[d] ); }
            nanSeen |= lNan;
        }
    }
    // the boxes that can hold a particle, kept in their order; bounds copied to flat arrays for the hot loop
    std::vector<size_t> keep;
    for (size_t k=0; k<boxesIn.size(); ++k)
    {
        bool possible = true;
        if ( not nanSeen )
            for (int d=0; d<NO_DIM; ++d)
                if ( boxesIn[k].coords[2*d+1] < pMin[d] or boxesIn[k].coords[2*d] > pMax[d] )
                { possible = false; break; }
        if ( possible ) keep.push_back( k );
    }
    size_t const nb = keep.size();
    if ( nb == 0 ) return;
    std::vector<Real> lo( nb*NO_DIM ), hi( nb*NO_DIM ), off( nb*NO_DIM, Real(0.) );
    for (size_t b=0; b<nb; ++b)
        for (int d=0; d<NO_DIM; ++d)
        {
            lo[b*NO_DIM+d] = boxesIn[keep[b]].coords[2*d];
            hi[b*NO_DIM+d] = boxesIn[keep[b]].coords[2*d+1];
            if ( shift ) off[b*NO_DIM+d] = offsetsIn[keep[b]][d];
        }
    // the same membership test as Box::isParticleInBox
    auto inside = [&](Particle_data const &q, size_t b) -> bool
    {
        for (int d=0; d<NO_DIM; ++d)
            if ( q.pos[d] < lo[b*NO_DIM+d] or q.pos[d] > hi[b*NO_DIM+d] )
                return false;
        return true;
    };

    // (2) count per (chunk, box), prefix sums box-major / chunk-minor, fill
    size_t const chunkLen = size_t(1) << 16;
    size_t const nChunk = (N + chunkLen - 1) / chunkLen;
    std::vector<size_t> pos( nChunk*nb, 0 );          // count, then the chunk's write slot for each box
    #pragma omp parallel for schedule(dynamic, 4)
    for (size_t c=0; c<nChunk; ++c)
    {
        size_t const i1 = std::min( N, (c+1)*chunkLen );
        size_t *cnt = &pos[c*nb];
        for (size_t i=c*chunkLen; i<i1; ++i)
            for (size_t b=0; b<nb; ++b)
                if ( inside( p[i], b ) ) ++cnt[b];
    }
    size_t running = output->size();
    for (size_t b=0; b<nb; ++b)
        for (size_t c=0; c<nChunk; ++c)
        {
            size_t const n = pos[c*nb+b];
            pos[c*nb+b] = running;
            running += n;
        }
    output->resize( running );
    Particle_data *out = output->data();
    #pragma omp parallel for schedule(dynamic, 4)
    for (size_t c=0; c<nChunk; ++c)
    {
        size_t const i1 = std::min( N, (c+1)*chunkLen );
        size_t *slot = &pos[c*nb];
        for (size_t i=c*chunkLen; i<i1; ++i)
            for (size_t b=0; b<nb; ++b)
                if ( inside( p[i], b ) )
                {
                    Particle_data &dst = out[ slot[b]++ ];
                    dst = p[i];
                    if ( shift )
                        for (int j=0; j<NO_DIM; ++j)
                            dst.pos[j] -= off[b*NO_DIM+j];
                }
    }
#endif
}


// Selects all particles inside userOptions.paddedBox into 'output', picking up periodic images when periodic.
void findParticlesInBox(std::vector<Particle_data> &p,
                        std::vector<Particle_data> *output,
                        User_options &userOptions)
{
    Box box = userOptions.paddedBox;
    Box fullBox = userOptions.boxCoordinates;
    Real boxLength[NO_DIM];                   // full box length per axis
    for (size_t i=0; i<NO_DIM; ++i)
        boxLength[i] = fullBox[2*i+1] - fullBox[2*i];
    
    
    MESSAGE::Message message( userOptions.verboseLevel );
    message << "Finding the particles that reside in the padded box " << box.print() << " ... " << MESSAGE::Flush;
    // reserve ~130% of the average number of particles in the given volume -- a hint only, so CAPPED at
    // p.size(): DTFE_parallel's threads call this after updateFullBox(region), so 'fullBox' is the user
    // region there, not the data box, and a thin --regionMpc (1 x 1 x 0.002 Mpc, the exact-zoom shape)
    // made the padded/region volume ratio ~5e6 = a multi-TB reserve. macOS overcommits it (untouched
    // pages), Linux refuses it with std::bad_alloc (2026-10-11). '!(est < cap)' also catches a NaN/inf
    // from a zero-volume region, and the double is clamped BEFORE the size_t conversion (UB if too big).
    // (The ordered parallel selection sizes the vector exactly itself; the hint serves the serial path.)
    double estimate = double( p.size() ) * double( box.volume() ) / double( fullBox.volume() ) * 1.3;
    if ( not (estimate < double(p.size())) ) estimate = double( p.size() );
    if ( not (estimate > 0.) ) estimate = 0.;
    
    std::vector<Box> boxes;
    std::vector< std::vector<Real> > offsets;
    if ( not userOptions.periodic )
    {
        boxes.push_back( box );
        offsets.push_back( std::vector<Real>( NO_DIM, Real(0.) ) );
    }
    else
    {
        // padded box may extend outside the data box: also pick up particles via periodic translation
        size_t const noOffsets = (NO_DIM==2 ? 9:27);
        Real offset[noOffsets][NO_DIM];
        for (size_t n=0; n<noOffsets; ++n)
        {
            size_t idx = n;
            for (int d=NO_DIM-1; d>=0; --d)
            {
                offset[n][d] = Real(int(idx % 3) - 1) * boxLength[d];
                idx /= 3;
            }
        }
        
        for (size_t n=0; n<noOffsets; ++n)
        {
            Box newBox = box;
            newBox.translate( offset[n] );
            if ( newBox.isBoxOverlaping(fullBox) )           // only translations whose box overlaps the full box contribute particles
            {
                boxes.push_back( newBox );
                offsets.push_back( std::vector<Real>( offset[n], offset[n]+NO_DIM ) );
            }
        }
    }
    if ( not selectionRunsParallel( p.size() ) )
        output->reserve( size_t(estimate) );
    selectParticlesInBoxesOrdered( p, output, boxes, offsets, userOptions.periodic );
        
    message << "Done.\n"
            << "\t There are " << output->size() << " particles in the above box. These represent " << std::setprecision(4) << output->size()/float(p.size())*100. << "\% of the total particles in the simulation box.\n" << MESSAGE::Flush;
}



// Inserts into 'output' the particles that fall inside 'box' after translation by 'offset' (storing the shifted-back copy).
void insertParticlesInBox(std::vector<Particle_data> &p,
                        std::vector<Particle_data> *output,
                        Box const &box,
                        Real const offset[])
{
    for (vectorIterator it=p.begin(); it!=p.end(); ++it)
        if ( box.isParticleInBox(*it) )
        {
            Particle_data temp = *it;
            for (int j=0; j<NO_DIM; ++j)
                temp.pos[j] -= offset[j];	// translating the sub-box by +offset is equivalent to translating the particle by -offset
            output->push_back( temp );
        }
}


void NGP_particle_count(std::vector<Particle_data> &particles,
                        size_t const *nGrid,
						Box box,
                        std::vector<int> *counts);

// Splits 'counts' into noPartitions contiguous ranges with as equal a particle load as possible; writes range edges to partitionIndices.
void optimalSplit(std::vector<size_t> &counts,
				  size_t const noPartitions,
				  size_t *partitionIndices)
{
	if ( noPartitions<=size_t(1) or counts.size()==size_t(1) )
		return;
	size_t totalCounts = 0;
	for (size_t i=0; i<counts.size(); ++i)
		totalCounts += counts[i];
	size_t load = totalCounts / noPartitions;	//the number of particles for a perfect balance
	
	// find the optimal balance
	size_t remainingCount = totalCounts, temp = 0, partitionIndexCount = 1;
	partitionIndices[0] = 0;
	partitionIndices[noPartitions] = counts.size();
	for (size_t i=0; i<counts.size(); ++i)
	{
		temp += counts[i];
		if (temp>=load)
		{
			size_t temp1 = ((remainingCount+counts[i]) - temp) / (noPartitions-partitionIndexCount);	//remaining load for split at index i-1
			size_t temp2 = (remainingCount - temp) / (noPartitions-partitionIndexCount);				//remaining load for split at index i
			
			// check which split is more balanced: at index i-1 or index i
			if ( i==size_t(0) )				//first partition must have at least one cell
				(void)i; // keep i unchanged
			else if ( i==counts.size()-1 )	//last partition must have at least one cell
			{
				temp -= counts[i];
				--i;
			}
			else if ( partitionIndexCount>size_t(1) and (i-partitionIndices[partitionIndexCount-1])==size_t(1) )	//each partition must have at least one cell
				(void)i; // keep i unchanged
			else  if ( std::abs(int(temp)-int(temp1))<std::abs(int(temp)-int(temp2)) )	//more balanced load when splitting at index i-1
			{
				temp -= counts[i];
				--i;
			}
			
			partitionIndices[partitionIndexCount] = i;
			remainingCount -= temp;			// keep track of how many particles are left
			load = remainingCount / (noPartitions-partitionIndexCount);	// update the perfect load for the remaining partitions
			temp = 0;
			++partitionIndexCount;
			if ( partitionIndexCount==noPartitions )
				break;
		}
	}

	// Every partition keeps at least one cell (2026-10-11). The balance above can leave two edges equal -- an EMPTY
	// range: the first edge stays at 0 when the first cell alone reaches the load, which happens always with more
	// partitions than half the cells (-g 8 --partition 1 5 5 gave an '8*0*0' grid and TSC/CIC/PCS segfaulted) and
	// on a cloud whose first slab holds a partition's load (DTFE_parallel's split over the threads too: a small
	// grid left threads with empty boxes). Push each edge right of the previous one, then left of what the
	// partitions after it need (possible: optimalPartitionSplit refuses more partitions than cells). A split that
	// already gave every partition a cell is unchanged, so only runs that had an empty range get new edges.
	size_t const n = counts.size();
	for (size_t k=1; k<noPartitions; ++k)
		if ( partitionIndices[k] < partitionIndices[k-1]+1 )
			partitionIndices[k] = partitionIndices[k-1]+1;
	for (size_t k=noPartitions-1; k>=1; --k)
		if ( partitionIndices[k] > n-(noPartitions-k) )
			partitionIndices[k] = n-(noPartitions-k);
}


// Divides the box along interpolation-grid lines (x then y then z) to balance the particle load across CPUs.
void optimalPartitionSplit(std::vector<Particle_data> &particles,
						   User_options &userOptions,
						   std::vector<size_t> const &partitionGrid,
						   std::vector< std::vector<size_t> > *subgrid,
						   std::vector<Box> *subgridCoords)
{
	MESSAGE::Message message( userOptions.verboseLevel );
    message << "\nComputing the optimal box partition to minimize the processor load ..." << MESSAGE::Flush;
	
	size_t const grid[] = { userOptions.gridSize[0], userOptions.gridSize[1], (NO_DIM==2)?1:userOptions.gridSize[2] };
	size_t const partition[] = { partitionGrid[0], partitionGrid[1], (NO_DIM==2)?1:partitionGrid[2] };
	size_t noPartitions = 1;	// total number of partitions
    for (size_t i=0; i<NO_DIM; ++i)
        noPartitions *= partition[i];
    Real dx[NO_DIM]; 				// to compute the region coordinates for each sub-box
	for (int i=0; i<NO_DIM; ++i)
		dx[i] = (userOptions.region[2*i+1] - userOptions.region[2*i]) / grid[i];
	
	// do some error checking
	if ( noPartitions==size_t(0) )
		throwError( "In function 'optimalPartitionSplit'. The total number of partitions is 0 - one of the partition values along an axis is 0. This does not make sense" );
	for(int i=0; i<NO_DIM; ++i)
		if ( partition[i]>grid[i] )
			throwError( "In function 'optimalPartitionSplit'. The size of the main grid along coordinate '", i, "' is smaller than the requested number of partitions along the same direction." );
	
	// assign memory to the output quantities
	std::vector<size_t> dummy; dummy.assign( 2*NO_DIM, size_t(0) );
	subgrid->assign( noPartitions, dummy );
	subgridCoords->assign( noPartitions, Box() );
	
	
	// find how many particles are in each cell of the grid
	std::vector<int> counts;
	NGP_particle_count( particles, grid, userOptions.region, &counts);
	
	// split along the x-direction
	std::vector<size_t> xSplitIndices; xSplitIndices.assign( partition[0]+1, size_t(0) );
	if ( partition[0]>size_t(1) )
	{
		std::vector<size_t> tempCounts;
		tempCounts.assign( grid[0], size_t(0) );
		
		// project the counts onto the x-axis
		for (int i1=0; i1<grid[0]; ++i1)
			for (int i2=0; i2<grid[1]; ++i2)
				for (int i3=0; i3<grid[2]; ++i3)
				{
					size_t index = i1*grid[1]*grid[2] + i2*grid[2] + i3;
					tempCounts[i1] += counts[index];
				}
		
		
		optimalSplit( tempCounts, partition[0], &(xSplitIndices[0]) );	// split the x-axis evenly
	}
	else
	{
		xSplitIndices[0] = 0;
		xSplitIndices[1] = grid[0];
	}
	// copy the results of the split to the book-keeping array
	for (size_t i1=0; i1<partition[0]; ++i1)
		for (size_t i2=0; i2<partition[1]; ++i2)
			for (size_t i3=0; i3<partition[2]; ++i3)
			{
				size_t index = i1*partition[1]*partition[2] + i2*partition[2] + i3;
				subgrid->at(index)[0] = xSplitIndices[i1];
				subgrid->at(index)[1] = xSplitIndices[i1+1];
				subgridCoords->at(index)[0] = userOptions.region[0] + dx[0]*subgrid->at(index)[0];
				subgridCoords->at(index)[1] = userOptions.region[0] + dx[0]*subgrid->at(index)[1];
			}
	
	
	// split along the y-direction
	std::vector<size_t> ySplitIndices; ySplitIndices.assign( partition[1]+1, size_t(0) );
	for (size_t i=0; i<partition[0]; ++i)
	{
		if ( partition[1]>size_t(1) )
		{
			std::vector<size_t> tempCounts;
			tempCounts.assign( grid[1], size_t(0) );
			
			// project the counts onto the y-axis
			for (int i1=xSplitIndices[i]; i1<xSplitIndices[i+1]; ++i1)
				for (int i2=0; i2<grid[1]; ++i2)
					for (int i3=0; i3<grid[2]; ++i3)
					{
						size_t index = i1*grid[1]*grid[2] + i2*grid[2] + i3;
						tempCounts[i2] += counts[index];
					}
			
			
			optimalSplit( tempCounts, partition[1], &(ySplitIndices[0]) );	// split the y-axis evenly
		}
		else
		{
			ySplitIndices[0] = 0;
			ySplitIndices[1] = grid[1];
		}
		for (size_t i2=0; i2<partition[1]; ++i2)
			for (size_t i3=0; i3<partition[2]; ++i3)
			{
				size_t index = i*partition[1]*partition[2] + i2*partition[2] + i3;
				subgrid->at(index)[2] = ySplitIndices[i2];
				subgrid->at(index)[3] = ySplitIndices[i2+1];
				subgridCoords->at(index)[2] = userOptions.region[2] + dx[1]*subgrid->at(index)[2];
				subgridCoords->at(index)[3] = userOptions.region[2] + dx[1]*subgrid->at(index)[3];
			}
	}
	
	
	// split along the z-direction
#if NO_DIM==3
	std::vector<size_t> zSplitIndices; zSplitIndices.assign( partition[2]+1, size_t(0) );
	for (size_t i=0; i<partition[0]; ++i)
		for (size_t j=0; j<partition[1]; ++j)
		{
			if ( partition[2]>size_t(1) )
			{
				std::vector<size_t> tempCounts;
				tempCounts.assign( grid[2], size_t(0) );
				
				// project the counts onto the z-axis
				for (int i1=xSplitIndices[i]; i1<xSplitIndices[i+1]; ++i1)
					for (int i2=ySplitIndices[j]; i2<ySplitIndices[j+1]; ++i2)
						for (int i3=0; i3<grid[2]; ++i3)
						{
							size_t index = i1*grid[1]*grid[2] + i2*grid[2] + i3;
							tempCounts[i3] += counts[index];
						}
				
				
				optimalSplit( tempCounts, partition[2], &(zSplitIndices[0]) );	// split the z-axis evenly
			}
			else
			{
				zSplitIndices[0] = 0;
				zSplitIndices[1] = grid[2];
			}
			for (size_t i3=0; i3<partition[2]; ++i3)
			{
				size_t index = i*partition[1]*partition[2] + j*partition[2] + i3;
				subgrid->at(index)[4] = zSplitIndices[i3];
				subgrid->at(index)[5] = zSplitIndices[i3+1];
				subgridCoords->at(index)[4] = userOptions.region[4] + dx[2]*subgrid->at(index)[4];
				subgridCoords->at(index)[5] = userOptions.region[4] + dx[2]*subgrid->at(index)[5];
			}
		}
#endif


	message << "Done.\n" << MESSAGE::Flush;
}


// Copies the selected partition's subgrid size and region into userOptions for this run.
void copySubgridInformation(User_options *userOptions,
							std::vector< std::vector<size_t> > &subgridList,
							std::vector< Box > &subgridCoords)
{
	userOptions->subgridOffset.assign( NO_DIM, 0 );
	for (int i=0; i<NO_DIM; ++i)
	{
		userOptions->gridSize[i] = subgridList[userOptions->partNo][2*i+1] - subgridList[userOptions->partNo][2*i];
		userOptions->subgridOffset[i] = subgridList[userOptions->partNo][2*i];
	}
	userOptions->region = subgridCoords[userOptions->partNo];
}

// Prints the grid-index range covered by the selected partition.
void subgrid(User_options &userOptions, std::vector< std::vector<size_t> > &subgrid)
{
    MESSAGE::Message message( userOptions.verboseLevel );
    char const axisName[] = "xyz";
    message << "\nYou choose to compute the results only for a partition of the data. The output files will contain only the results for the grid indices with:\n";
    for (int d=0; d<NO_DIM; ++d)
        message << "\t n" << axisName[d] << " = [" << subgrid[userOptions.partNo][2*d] << " to " << subgrid[userOptions.partNo][2*d+1]-1 << "]\n";
}


// Merges a subgrid's field quantities ('--partition' option) into the full-box quantities, with size sanity checks.
void copySubgridResultsToMain(Quantities const &subgridResults,
                              std::vector<size_t> const &mainGrid,
                              Field &field,
                              User_options &userOptions,
							  std::vector< std::vector<size_t> > &subgridList,
                              Quantities *mainGridResults)
{
    if ( not field.selected() ) return;
    
    // get the subgrid and subgrid offset
    std::vector<size_t> subgrid(NO_DIM,0), subgridOffset(NO_DIM,0);
    for (int i=0; i<NO_DIM; ++i)
	{
		subgrid[i] = subgridList[userOptions.partNo][2*i+1] - subgridList[userOptions.partNo][2*i];
		subgridOffset[i] = subgridList[userOptions.partNo][2*i];
	}
    
    
    // do some error checking
    size_t totalGrid = 1, secondaryGrid = 1;
    for (size_t i=0; i<mainGrid.size(); ++i)
    {
            totalGrid *= mainGrid[i];
            secondaryGrid *= userOptions.gridSize[i];
    }
    
    if ( mainGridResults->size()!=totalGrid )
        throwError( "The size of the vectors in the main grid results does not agree with the size of the main grid. Error in function 'copySubgridResultsToMain'." );
    if ( subgridResults.size()!=secondaryGrid )
        throwError( "The size of the vectors in the subgrid results does not agree with the size of the subgrid grid. Error in function 'copySubgridResultsToMain'." );
    
    // copy the subgrid to the main grid
    mainGridResults->copyFromSubgrid( subgridResults, field, mainGrid, subgrid, subgridOffset );
}



// Returns the mean mass density in the full simulation box, used as the reference for per-vertex density.
Real averageDensity(std::vector<Particle_data> &p,
                    User_options const &userOptions)
{
    double totalMass = 0.;
    for (vectorIterator it=p.begin(); it!=p.end(); ++it)
        totalMass += it->weight();

    Real averageDen = totalMass / userOptions.boxCoordinates.volume();
    MESSAGE::Message message( userOptions.verboseLevel );
    message << "Average density in the box: " << averageDen << "\n" << MESSAGE::Flush;
    return averageDen;
}


// Chooses the parallel partition grid for the available processors, using the largest 2^a*3^b <= noAvailableProcessors.
void parallelGrid(size_t const noAvailableProcessors,
                  User_options const &userOptions,
                  size_t *grid)
{
    // largest x2, x3 with 2^x2 <= n and 3^x3 <= n
    size_t n = noAvailableProcessors;
    int x2 = 0, x3 = 0;
    
    while ( n/2!=0 )
    {
        n /= 2;
        ++x2;
    }
    n = noAvailableProcessors;
    while ( n/3!=0 )
    {
        n /= 3;
        ++x3;
    }
    n = noAvailableProcessors;
    
    // best 2^j2 * 3^j3 <= n closest to n
    size_t diff = n;
    int j2 = 0, j3 = 0;
    size_t nApprox = 1;
    for (int i2=0; i2<=x2; ++i2)
    {
        size_t temp = nApprox;
        for (int i3=0; i3<=x3; ++i3)
        {
            if ( (n-temp)>=0 and (n-temp)<diff )  //if temp<n and temp is closer to n than previous estimate, update these values as best choice
            {
                diff = n-temp;
                j2 = i2;
                j3 = i3;
            }
            temp *= 3;
        }
        nApprox *= 2;
    }
    
    x2 = j2;
    x3 = j3;


    // distribute the 2's and 3's across axes as uniformly as possible to minimize overlapping regions
    for (int i=0; i<NO_DIM; ++i)
        grid[i] = 1;
    // first distribute the 3's
    while ( x3-NO_DIM>=0 )
    {
        for (int i=0; i<NO_DIM; ++i)
            grid[i] *= 3;
        x3 -= NO_DIM;
    }
    for (int i=NO_DIM-1; i>=(NO_DIM-x3); --i)    // put the remaining 3's starting with axis NO_DIM-1, NO_DIM-2, ... so on
        grid[i] *= 3;
    // now distribute the 2's such that the grid is maximally balanced
    int const i1 = x2>(NO_DIM-x3) ? (NO_DIM-x3) : x2; // i1 is the minimum(x2, NO_DIM-x3)
    for (int i=0; i<i1; ++i)    // the grid axes which have a deficency of 3's will receive a factor of 2 
        grid[i%(NO_DIM-x3)] *= 2;
    for (int i=0; i<x2-i1; ++i) // distribute the remaining 2's starting with the first grid axis
        grid[i%NO_DIM] *= 2;

    // Check that the parallel split is consistent with the main grid axes
    size_t const *mainGrid = &(userOptions.gridSize[0]);
	for (int i=0; i<NO_DIM; ++i)
		if ( mainGrid[i]>=grid[i] )	//don't do anything if you can split the mainGrid to have at least one cell per parallel partition
			continue;
		else
		{
			size_t next1 = (i+1)%NO_DIM, next2 = (i+2)%NO_DIM;
			while ( grid[i]>mainGrid[i] and grid[i]%3==0 )
			{
				grid[i] /= 3;
				if ( 3*grid[next1]<=mainGrid[next1] )
					grid[next1] *= 3;
				else if ( 3*grid[next2]<=mainGrid[next2] )
					grid[next2] *= 3;
			}
			while ( grid[i]>mainGrid[i] and grid[i]%2==0 )
			{
				grid[i] /= 2;
				if ( 2*grid[next1]<=mainGrid[next1] )
					grid[next1] *= 2;
				else if ( 2*grid[next2]<=mainGrid[next2] )
					grid[next2] *= 2;
			}
		}
	
	
    // show to the user a message with the parallel grid
    size_t const noProcesses = (NO_DIM==2) ? grid[0]*grid[1] : grid[0]*grid[1]*grid[2];
    MESSAGE::Message message( userOptions.verboseLevel );
    if ( noProcesses==noAvailableProcessors )
        message << "\nThe program detected " << noAvailableProcessors << " processors available for the computation. The processors number was split in the grid [" << MESSAGE::printElements( grid, NO_DIM, "," ) << "] to allow for optimal division of the data box.\n" << MESSAGE::Flush;
    else
        message << "\nThe program detected " << noAvailableProcessors << " processors available for the computation. Since this number cannot be written as a product of 2's and 3's, only " << noProcesses << " processors will be used for the parallel computation. The new processors number was split in the grid [" << MESSAGE::printElements( grid, NO_DIM, "," ) << "] to allow for optimal division of the data box.\n" << MESSAGE::Flush;
}





// Approximate per-thread CPU time (threads cannot be timed independently).
void approximativeThreadTime(Real *processorTime,
                             int const noProcessors)
{
    std::vector< std::pair<Real,int> > temp;
    temp.reserve( noProcessors );
    for (int i=0; i<noProcessors; ++i)
        temp.push_back( std::make_pair( processorTime[i], i ) );
    
    // sort the times ascendingly
    std::sort( temp.begin(), temp.end() );
    Real lastTime = Real(0.), lastTotalTime = Real(0.);
    int factor = noProcessors;
    for (int i=0; i<noProcessors; ++i)
    {
        Real temp2 = temp[i].first;
        temp[i].first = lastTime + (temp2-lastTotalTime) / factor--;
        lastTime = temp[i].first;
        lastTotalTime = temp2;
    }
    
    // write the computed values to the output array
    for (int i=0; i<noProcessors; ++i)
        processorTime[ temp[i].second ] = temp[i].first;
}


#endif
