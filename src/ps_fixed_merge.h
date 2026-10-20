/* THE FIXED-POINT MERGE of the PS-DTFE partition loop ('--fixed-point-merge', DTFE.cpp; OPT-IN).

   The default merge (the staged ordered merge, DTFE.cpp) adds each partition's sub-grid -- Real values, each an exact
   fixed-point sum of the deposit rounded to Real once, so a partition's values do not depend on the run -- into float
   full grids in partition INDEX order: float adds do not commute, so only one order gives the same bits, and a
   partition that finishes early must park (in RAM, or spilled to disk) until its turn. This mode adds every partition
   AS IT FINISHES into exact INTEGER full-grid accumulators instead; integer adds commute, so the result cannot depend
   on the completion order, the concurrency or the thread count, and nothing parks. It is opt-in, for comparing its
   memory and speed against the default merge.

   THE ACCUMULATORS. Every accumulator addFromSubgrid SUMS (forEachSummed below: its '+=' list; caustic_bits and
   hidden_streams are OR-merged, order-free already, and stay Real grids, as do the T-web / V-web labels, which no
   partition fills) becomes, per component and cell, an int64 WORD plus an int8 CARRY in a global unit 2^-K of that
   accumulator: 9 B where the float took 4. A partition value v becomes q = llrint(v * 2^K) -- v * 2^K is exact (a
   power of two), llrint rounds half to even, so q is a property of the partition alone -- and is added to the word
   with __builtin_add_overflow; on a signed overflow the carry takes the sign of q. A cell is then carry * 2^64 + word,
   an exact 72-bit integer whatever the order of the adds. The integer arrays replace the pre-sized Real grids when
   the merge is set up, before the first partition starts (so the memory gate's free-memory reading holds them, as
   the default's holds its Real grids), except those of the grids no partition fills (derived
   after the merge); an accumulator the default grows at its first values (the moment weights) gets its arrays at a
   partition's first value of it (the units are fixed before, so the allocation order changes nothing); the Real full
   grids do not exist during the loop (the pre-sized ones are freed when the merge is set up). Allocations of >= 1 GB
   go through scratch_alloc.cc like the float grids, so '--scratch-dir' backs these too.

   THE CONVERSION, once, after the loop, accumulator by accumulator (its Real grid allocated, its integer arrays freed
   right after): the 72-bit integer is converted to the nearest float ONCE (round half to even: the compiler's
   __int128 -> float conversion), then scaled by 2^-K (ldexp: exact while the result is a normal float). Deterministic
   per cell. The result differs from the ordered merge's float sums at about 1e-7 relative: each partition value is
   rounded to the unit (below), and the float sums of the default round at every add.

   THE UNITS are fixed BEFORE the first merge from GLOBAL quantities only -- never from the partitions that finish
   first, which would make them order-dependent. Per accumulator a natural per-cell scale S (cellMass = the particles'
   total mass / the grid's cells, cellVol = the grid cell's volume, Sw = the moment weight's scale: cellVol under
   '--ps-volume-weighted', else cellMass; vmax = the particles' largest |velocity component|, smax = their largest
   |scalar component|, dx = the smallest cell side; one O(N) pass over the particles):
       density                                  1             (the sub-grid holds rho/rho_bar already)
       mass_weight                              Sw            (the mass shares, or the volume shares)
       disp_weight                              cellMass
       velocity, velocity_std                   Sw   * vmax
       disp_velocity                            cellMass * vmax
       velocity_dispersion                      cellMass * vmax^2   (the second moment is always mass-weighted)
       velocity_gradient, _divergence, _shear,  Sw   * vmax / dx
         _vorticity, velocity_vweb_eigenvalues
       scalar                                   Sw   * smax
       scalar_gradient                          Sw   * smax / dx
       stream_count, tet_touch                  1             (counts: sample-mean multiplicity, tet touches)
       velocity_tweb_eigenvalues                1
   (the derived fields and the eigenvalues are in addFromSubgrid's list, but no partition fills them: they are made
   after the merge.) A scale that is not positive and finite (vmax = 0: no velocities) falls back to its weight's, or
   1. K is chosen so that S maps to 2^36..2^37 units (S in [2^(e-1), 2^e): K = 37 - e): a value >= ~1e-4 S keeps its
   float's 24 bits, a single value may reach 2^26 S (2^63 units), and a sum ~2^34 S thanks to the carry.
   DTFE_TEST_FXMERGE_SHIFT=n makes every unit 2^n finer (the test suite's overflow hook; a negative n coarser).

   OVERFLOW, order-free and FATAL (a wrapped cell cannot be repaired): (a) a value that is not finite or has
   |v * 2^K| >= 2^63 stops the run naming the accumulator, the partition and the value's ratio to S -- a property of
   one partition, so it fires whatever the order; (b) per accumulator the sum over the merged partitions of each one's
   largest |q| is kept (an unsigned __int128); it bounds every cell's |sum|, and it must stay below 2^71 - 2^63 (the
   int8 carry with a signed word holds [-2^71 - 2^63, 2^71 - 2^63)). It only grows, so checking it at every merge
   fails a run exactly when its total over all partitions reaches the limit, whatever the order; the message names
   the accumulator and the bound, not the partition at which it crossed (that one is order-dependent). Never the
   running sum's own overflow (order-dependent).
   Whether a run stops is a property of its partitions; which partition (a) names when several hold such values is
   the first to finish among them. DTFE_TEST_FXMERGE_SUM_LIMIT=n lowers the limit of (b) to 2^n (its test).

   CHECKPOINT ('--resume', partition_checkpoint.h): the words and carries go into the checkpoint as raw blobs
   ('x.fxw.<side>.<name>', 'x.fxc.<side>.<name>') with the per-accumulator state ('x.fxmerge.state': K, the overflow
   bound (b), the largest |q| and the cells), and the settings key carries 'merge=fixed-point', so a checkpoint of the
   other merge mode is refused with a clear message. A resumed run's integers are the killed run's plus the rest,
   so its outputs are byte-identical to an uninterrupted run (tests/ps_fixed_merge_check.sh). The units are compared
   on resume (a changed DTFE_TEST_FXMERGE_SHIFT, or another binary, is refused). */
#ifndef PS_FIXED_MERGE_H
#define PS_FIXED_MERGE_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#include "define.h"
#include "message.h"
#include "particle_data.h"
#include "quantities.h"
#include "user_options.h"
#include "partition_checkpoint.h"   // the checkpoint's blob list (Checkpointer::blobs)

namespace PSFixedMerge
{

// the scale kinds (the table in the header comment)
enum Kind { K_ONE = 0, K_WEIGHT, K_MASS, K_VEL_W, K_VEL_M, K_VEL2_M, K_GRAD_W, K_SCAL_W, K_SGRAD_W, K_N };

// every accumulator addFromSubgrid SUMS ('+='), with its scale kind -- quantities.cc's list, in its order. The names
// are PartitionCheckpoint::forEachAccumulator's. If you add a summed accumulator there, add it here.
template<class F>
inline void forEachSummed(Quantities &q, F &&f)
{
    f( "density", K_ONE, q.density );                         f( "velocity", K_VEL_W, q.velocity );
    f( "velocity_gradient", K_GRAD_W, q.velocity_gradient );  f( "velocity_divergence", K_GRAD_W, q.velocity_divergence );
    f( "velocity_shear", K_GRAD_W, q.velocity_shear );        f( "velocity_vorticity", K_GRAD_W, q.velocity_vorticity );
    f( "velocity_std", K_VEL_W, q.velocity_std );             f( "velocity_dispersion", K_VEL2_M, q.velocity_dispersion );
    f( "scalar", K_SCAL_W, q.scalar );                        f( "scalar_gradient", K_SGRAD_W, q.scalar_gradient );
    f( "velocity_tweb_eigenvalues", K_ONE, q.velocity_tweb_eigenvalues );
    f( "velocity_vweb_eigenvalues", K_GRAD_W, q.velocity_vweb_eigenvalues );
    f( "stream_count", K_ONE, q.stream_count );               f( "tet_touch", K_ONE, q.tet_touch );
    f( "mass_weight", K_WEIGHT, q.mass_weight );              f( "disp_weight", K_MASS, q.disp_weight );
    f( "disp_velocity", K_VEL_M, q.disp_velocity );
}

// the sum bound (b): an int8 carry over a signed int64 word holds [-2^71 - 2^63, 2^71 - 2^63)
static unsigned __int128 const SUM_LIMIT = ( (unsigned __int128)1 << 71 ) - ( (unsigned __int128)1 << 63 );
static double const VALUE_LIMIT = 9223372036854775808.;   // 2^63: one value's |q| must stay below it

// one summed accumulator of u or a
struct Slot
{
    std::string name;              // "u.density"
    int kind = K_ONE;
    size_t comps = 1;              // Real components per cell
    double S = 1.;                 // its natural per-cell scale
    int K = 0;                     // its unit is 2^-K
    size_t presized = 0;           // cells of the Real grid before the loop (freed at setup; 0: it had none)
    size_t cells = 0;              // cells of the integer arrays (0: not allocated yet)
    std::vector<int64_t> w;        // the words, comps per cell
    std::vector<int8_t> c;         // the carries
    unsigned __int128 bound = 0;   // (b): the sum over the merged partitions of each one's largest |q|
    uint64_t maxQ = 0;             // the largest |q| of one value merged
};

// the saved state of one slot in the checkpoint ('x.fxmerge.state')
struct SavedSlot
{
    int64_t K = 0;
    uint64_t boundLo = 0, boundHi = 0, maxQ = 0, cells = 0;
};

inline double log2u128(unsigned __int128 v) { return v == 0 ? -1.e300 : std::log2( double( v ) ); }

struct Merge
{
    bool on = false;
    std::vector<Slot> slots;              // u's, then a's, in forEachSummed order
    double cellMass = 0., cellVol = 0., vmax = 0., smax = 0., dxMin = 0.;
    int shift = 0;                        // DTFE_TEST_FXMERGE_SHIFT
    unsigned __int128 sumLimit = SUM_LIMIT;   // (b); DTFE_TEST_FXMERGE_SUM_LIMIT=n lowers it to 2^n (the test of (b))
    int sumLimitLog2 = -1;                    // n, or -1
    double peakBytes = 0., liveBytes = 0.;
    std::vector<SavedSlot> loadedState;   // filled by the checkpoint load (restore() checks it)

    // the slot of the i-th summed accumulator of side 0 (u) / 1 (a)
    size_t perSide() const { return slots.size() / 2; }

    // the units, from the global quantities; replaces the pre-sized Real grids of the summed accumulators by their
    // integer arrays (freed only, for the grids no partition fills)
    void setup(Quantities &u, Quantities &a, User_options const &opt, std::vector<Particle_data> &particles)
    {
        on = true;
        double totalMass = 0.;
        double vm = 0., sm = 0.;
        for (Particle_data &p : particles)
        {
            totalMass += double( p.weight() );
            for (size_t j = 0; j < noVelComp; ++j) vm = std::max( vm, std::fabs( double( p.velocity( int(j) ) ) ) );
#ifdef SCALAR
            for (size_t j = 0; j < noScalarComp; ++j) sm = std::max( sm, std::fabs( double( p.scalar( int(j) ) ) ) );
#endif
        }
        vmax = vm;
        smax = sm;
        double fullCells = 1.;
        cellVol = 1.;
        dxMin = HUGE_VAL;
        for (int d = 0; d < NO_DIM; ++d)
        {
            double const dx = double( opt.region.coords[2*d+1] - opt.region.coords[2*d] ) / double( opt.gridSize[d] );
            cellVol *= dx;
            dxMin = std::min( dxMin, dx );
            fullCells *= double( opt.gridSize[d] );
        }
        cellMass = totalMass / fullCells;
        if ( char const *e = std::getenv( "DTFE_TEST_FXMERGE_SHIFT" ) )
            if ( e[0] != '\0' ) shift = std::atoi( e );
        if ( char const *e = std::getenv( "DTFE_TEST_FXMERGE_SUM_LIMIT" ) )
            if ( e[0] != '\0' and std::atoi( e ) > 0 and std::atoi( e ) < 71 )
            {
                sumLimitLog2 = std::atoi( e );
                sumLimit = (unsigned __int128)1 << sumLimitLog2;
            }
        double const sw = opt.psVolumeWeighted ? cellVol : cellMass;
        auto scaleOf = [&](int const kind) -> double
        {
            auto pick = [](double const s, double const fallback)
            {
                if ( s > 0. and std::isfinite( s ) ) return s;
                return ( fallback > 0. and std::isfinite( fallback ) ) ? fallback : 1.;
            };
            switch ( kind )
            {
            case K_WEIGHT:  return pick( sw, 1. );
            case K_MASS:    return pick( cellMass, 1. );
            case K_VEL_W:   return pick( sw * vmax, sw );
            case K_VEL_M:   return pick( cellMass * vmax, cellMass );
            case K_VEL2_M:  return pick( cellMass * vmax * vmax, cellMass );
            case K_GRAD_W:  return pick( sw * vmax / dxMin, sw );
            case K_SCAL_W:  return pick( sw * smax, sw );
            case K_SGRAD_W: return pick( sw * smax / dxMin, sw );
            default:        return 1.;
            }
        };
        slots.clear();
        for (int side = 0; side < 2; ++side)
            forEachSummed( side ? a : u, [&](char const *name, int const kind, auto &v)
            {
                using V = typename std::decay<decltype( v )>::type;
                using T = typename V::value_type;
                Slot s;
                s.name = std::string( side ? "a." : "u." ) + name;
                s.kind = kind;
                s.comps = sizeof( T ) / sizeof( Real );
                s.S = scaleOf( kind );
                int e = 0;
                std::frexp( s.S, &e );   // S in [2^(e-1), 2^e)
                s.K = 37 - e + shift;    // S * 2^K in [2^36, 2^37)
                s.presized = v.size();
                V().swap( v );           // the Real grid: none during the loop
                // its integer twin NOW, before any partition starts: allocated at the first merge instead, the
                // memory gate would admit the first partitions with a free-memory reading that does not hold the
                // 9 B grids yet (the default's Real grids are pre-sized before the loop, so its gate sees them) --
                // a hand-set --max-concurrent could then overcommit. Not for the grids no partition
                // fills (derived after the merge); those not pre-sized (the moment weights) still come with their
                // first values, as the default's do.
                if ( s.presized > 0 and not derivedAfterMerge( name ) )
                {
                    s.w.assign( s.presized * s.comps, 0 );
                    s.c.assign( s.presized * s.comps, 0 );
                    s.cells = s.presized;
                }
                slots.push_back( std::move( s ) );
            } );
        account();
    }

    // the summed accumulators no partition fills: made after the merge from the summed ones (DTFE.cpp)
    static bool derivedAfterMerge(char const *name)
    {
        for (char const *d : { "velocity_divergence", "velocity_shear", "velocity_vorticity", "velocity_std",
                               "velocity_tweb_eigenvalues", "velocity_vweb_eigenvalues" })
            if ( std::strcmp( name, d ) == 0 ) return true;
        return false;
    }

    // the checkpoint's blobs: every slot's words and carries, and the state (registered before beginLoop: a resume
    // loads into them)
    std::vector<PartitionCheckpoint::Blob> blobs()
    {
        std::vector<PartitionCheckpoint::Blob> b;
        for (size_t i = 0; i < slots.size(); ++i)
        {
            Slot *s = &slots[i];
            b.push_back( { "fxw." + s->name, sizeof( int64_t ),
                           [s]() { return s->w.size(); },
                           [s]() { return static_cast<void const *>( s->w.data() ); },
                           [this, s](size_t const n) { s->w.assign( n, 0 ); s->cells = n / s->comps; account(); return static_cast<void *>( s->w.data() ); } } );
            b.push_back( { "fxc." + s->name, sizeof( int8_t ),
                           [s]() { return s->c.size(); },
                           [s]() { return static_cast<void const *>( s->c.data() ); },
                           [this, s](size_t const n) { s->c.assign( n, 0 ); account(); return static_cast<void *>( s->c.data() ); } } );
        }
        b.push_back( { "fxmerge.state", sizeof( SavedSlot ),
                       [this]() { return slots.size(); },
                       [this]()
                       {
                           loadedState.assign( slots.size(), SavedSlot() );
                           for (size_t i = 0; i < slots.size(); ++i)
                           {
                               SavedSlot &t = loadedState[i];
                               t.K = slots[i].K;
                               t.boundLo = uint64_t( slots[i].bound );
                               t.boundHi = uint64_t( slots[i].bound >> 64 );
                               t.maxQ = slots[i].maxQ;
                               t.cells = slots[i].cells;
                           }
                           return static_cast<void const *>( loadedState.data() );
                       },
                       [this](size_t const n) { loadedState.assign( n, SavedSlot() ); return static_cast<void *>( loadedState.data() ); } } );
        return b;
    }

    // after a checkpoint was loaded ('--resume'): the saved units must be this run's; restores the bound (b) and the
    // largest |q| (they cannot be recomputed from the summed words)
    void restore()
    {
        if ( loadedState.size() != slots.size() )
            throwError( "'--resume': the checkpoint holds no fixed-point merge state for this run's accumulators (",
                        loadedState.size(), " saved, " + std::to_string( slots.size() ) + " here). Run without '--resume' "
                        "to start over (run_ps_dtfe.sh: PS_RESUME=0)." );
        for (size_t i = 0; i < slots.size(); ++i)
        {
            Slot &s = slots[i];
            SavedSlot const &t = loadedState[i];
            if ( t.K != int64_t( s.K ) )
                throwError( "'--resume': the checkpoint's fixed-point unit of ", s.name, " is 2^-" + std::to_string( t.K )
                            + ", this run's 2^-" + std::to_string( s.K ) + " (DTFE_TEST_FXMERGE_SHIFT or the particles "
                            "changed): its integers cannot be added to this run's. Run without '--resume' to start over." );
            if ( ( t.cells == 0 ) != s.w.empty() or ( t.cells != 0 and ( s.w.size() != t.cells * s.comps
                                                                         or s.c.size() != s.w.size() ) ) )
                throwError( "'--resume': the checkpoint's fixed-point words of ", s.name, " do not match its state (cut or "
                            "damaged). Run without '--resume' to start over." );
            s.bound = ( (unsigned __int128)t.boundHi << 64 ) | (unsigned __int128)t.boundLo;
            s.maxQ = t.maxQ;
            s.cells = size_t( t.cells );
        }
        std::vector<SavedSlot>().swap( loadedState );
    }

    void account()
    {
        double b = 0.;
        for (Slot const &s : slots) b += double( s.w.capacity() ) * 8. + double( s.c.capacity() );
        liveBytes = b;
        peakBytes = std::max( peakBytes, b );
    }

    // one accumulator of one partition into its slot (under the merge lock)
    template<class T>
    void addOne(Slot &s, std::vector<T> const &src, Quantities const &sub, size_t const *fullGrid,
                size_t const *mainOrigin, size_t const *mainDims, int const partition)
    {
        if ( src.empty() ) return;
        size_t dstTotal = 1;
        for (int d = 0; d < NO_DIM; ++d) dstTotal *= mainOrigin ? mainDims[d] : fullGrid[d];
        bool const isSub = sub.ps_subDims[0] != 0;   // else 'sub' spans the main grid, cell for cell (addFrom)
        if ( not isSub ) dstTotal = std::max( dstTotal, src.size() );
        if ( s.cells == 0 )
        {
            s.w.assign( dstTotal * s.comps, 0 );
            s.c.assign( dstTotal * s.comps, 0 );
            s.cells = dstTotal;
            account();
        }
        if ( s.cells < dstTotal )
            throwError( "internal: the fixed-point merge's ", s.name, " holds " + std::to_string( s.cells ) + " cells, a "
                        "partition delivers " + std::to_string( dstTotal ) );
        size_t const comps = s.comps;
        Real const *v = reinterpret_cast<Real const *>( src.data() );
        size_t subTotal = src.size();
        size_t const *o = sub.ps_subOrigin, *m = sub.ps_subDims;
        uint64_t pmax = 0;
        int64_t *w = s.w.data();
        int8_t *c = s.c.data();
        double const unit = std::ldexp( 1., s.K );
        for (size_t l = 0; l < subTotal; ++l)
        {
            size_t g = l;
            if ( isSub )
            {
                size_t rem = l, cc[NO_DIM];
                for (int d = NO_DIM - 1; d >= 0; --d) { cc[d] = rem % m[d]; rem /= m[d]; }   // local coords
                if ( not subgridTarget( cc, o, fullGrid, mainOrigin, mainDims, g ) ) continue;
            }
            for (size_t k = 0; k < comps; ++k)
            {
                double const x = double( v[l*comps + k] ) * unit;   // exact: a power of two
                if ( not ( std::fabs( x ) < VALUE_LIMIT ) )   // NaN and inf too
                    valueError( s, partition, double( v[l*comps + k] ) );
                int64_t const q = int64_t( std::llrint( x ) );
                uint64_t const aq = q < 0 ? uint64_t( 0 ) - uint64_t( q ) : uint64_t( q );
                pmax = std::max( pmax, aq );
                size_t const i = g * comps + k;
                int64_t r;
                if ( __builtin_add_overflow( w[i], q, &r ) ) c[i] = int8_t( c[i] + ( q > 0 ? 1 : -1 ) );
                w[i] = r;
            }
        }
        s.bound += pmax;
        s.maxQ = std::max( s.maxQ, pmax );
        if ( s.bound >= sumLimit )
            throwError( "--fixed-point-merge: the merged partitions' largest values of ", s.name, " sum to 2^"
                        + fmt( log2u128( s.bound ), 4 ) + " units (" + fmt( std::ldexp( double( s.bound ), -s.K ) / s.S, 3 )
                        + " times its scale S = " + fmt( s.S, 4 ) + "), the limit " + ( sumLimitLog2 > 0 ? "2^" + std::to_string(
                        sumLimitLog2 ) + " (DTFE_TEST_FXMERGE_SUM_LIMIT)" : std::string( "2^71 - 2^63 of its 72-bit sums" ) ) + ": a cell "
                        "could overflow, which could not be repaired. The run stops. Run with the default (ordered) "
                        "merge, which has no such limit.", "" );
    }

    static std::string fmt(double const x, int const prec)
    {
        char b[64];
        std::snprintf( b, sizeof b, "%.*g", prec, x );
        return b;
    }

    [[noreturn]] void valueError(Slot const &s, int const partition, double const v)
    {
        if ( not std::isfinite( v ) )
            throwError( "--fixed-point-merge: partition ", partition, " holds a non-finite value (" + fmt( v, 6 ) + ") in "
                        + s.name + ", which no integer unit can hold. The run stops." );
        throwError( "--fixed-point-merge: partition ", partition, " holds a value of " + s.name + " of " + fmt( v, 6 )
                    + " = " + fmt( v / s.S, 4 ) + " times its scale S = " + fmt( s.S, 4 ) + "; its unit 2^-"
                    + std::to_string( s.K ) + " holds |values| below " + fmt( std::ldexp( 1., 63 - s.K ) / s.S, 4 )
                    + " S (2^63 units). A wrapped cell could not be repaired: the run stops. Run with the default "
                    "(ordered) merge, which has no such limit.", "" );
        std::abort();   // throwError does not return
    }

    // one partition's two sub-grid Quantities into the integer grids (their summed vectors are released), and their
    // OR-merged masks into the full Real grids by addFromSubgrid (the summed vectors being empty, it does only those)
    void add(int const partition, Quantities &tu, Quantities &ta, Quantities &fu, Quantities &fa,
             size_t const *fullGrid, size_t const *mainOrigin, size_t const *mainDims)
    {
        for (int side = 0; side < 2; ++side)
        {
            Quantities &t = side ? ta : tu;
            size_t i = side * perSide();
            forEachSummed( t, [&](char const *, int, auto &v)
            {
                using V = typename std::decay<decltype( v )>::type;
                addOne( slots[i++], v, t, fullGrid, mainOrigin, mainDims, partition );
                V().swap( v );
            } );
            ( side ? fa : fu ).addFromSubgrid( t, fullGrid, mainOrigin, mainDims );
        }
    }

    // after the loop: every slot to its Real grid, one at a time (its integers freed right after); a pre-sized grid no
    // partition filled comes back as zeros, as the default merge leaves it
    void finish(Quantities &u, Quantities &a)
    {
        for (int side = 0; side < 2; ++side)
        {
            size_t i = side * perSide();
            forEachSummed( side ? a : u, [&](char const *, int, auto &v)
            {
                using V = typename std::decay<decltype( v )>::type;
                using T = typename V::value_type;
                Slot &s = slots[i++];
                if ( s.cells > 0 )
                {
                    v.assign( s.cells, T() );
                    Real *out = reinterpret_cast<Real *>( v.data() );
                    size_t const n = s.cells * s.comps;
                    for (size_t j = 0; j < n; ++j)
                    {
                        __int128 const t = (__int128)s.c[j] * ( (__int128)1 << 64 ) + (__int128)s.w[j];
                        out[j] = std::ldexp( static_cast<Real>( t ), -s.K );   // one rounding, then an exact scale
                    }
                    std::vector<int64_t>().swap( s.w );
                    std::vector<int8_t>().swap( s.c );
                }
                else if ( s.presized > 0 )
                {
                    v.assign( s.presized, T() );
                    std::memset( static_cast<void *>( v.data() ), 0, s.presized * sizeof( T ) );
                }
            } );
        }
        account();
    }

    // the units, said before the loop: the allocated slots are not known yet, so every selected kind once
    std::string unitsText() const
    {
        char b[200];
        std::snprintf( b, sizeof b, "cell mass %.4g, cell volume %.4g, max |velocity component| %.4g%s",
                       cellMass, cellVol, vmax, shift ? ( ", DTFE_TEST_FXMERGE_SHIFT=" + std::to_string( shift ) ).c_str() : "" );
        return b;
    }

    // the report after the loop: the bytes, and per filled accumulator its unit and its headroom (one value's largest
    // |q| below 2^63, the sum bound (b) below 2^71)
    std::string reportText() const
    {
        std::string t;
        char b[400];
        std::snprintf( b, sizeof b, "fixed-point merge: %.2f GB of integer grids at most (9 B per summed component and "
                       "cell); nothing parked or spilled", peakBytes / 1.e9 );
        t = b;
        for (Slot const &s : slots)
        {
            if ( s.maxQ == 0 and s.bound == 0 ) continue;
            std::snprintf( b, sizeof b, "; %s: unit 2^-%d (S %.3g), headroom 2^%.1f per value, 2^%.1f per sum",
                           s.name.c_str(), s.K, s.S, 63. - std::log2( double( std::max<uint64_t>( s.maxQ, 1 ) ) ),
                           71. - std::max( 0., log2u128( s.bound ) ) );
            t += b;
        }
        return t;
    }
};

}   // namespace PSFixedMerge

#endif
