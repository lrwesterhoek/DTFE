/* PS_MASS_AUDIT=1: the PS-DTFE deposit's mass bookkeeping (a diagnostic, OFF by default --
   unset, none of it runs and every output is unchanged). Each grid pass (interpolateGrid_phaseSpace) prints
   one 'PS mass audit: partition i ...' line to stderr, and a partitioned run one 'PS mass audit: run ...'
   line per pass kind after the merge (DTFE.cpp). All sums in double, in mass units:
     (a) owned    the weight of the vertices this pass OWNS (Lagrangian position in the ownership box, the
                  filter's half-open test; no box: every non-dummy vertex)
     (b) tets     the mass of the tetrahedra that pass psFilterCell with ownership (the deposit's input:
                  vertex shares under --ps-vertex-mass, else rho_bar V_lag)
     (c) deg0     the owned vertices of degree 0 (every incident tet dropped): their weight reaches no tet
     (d) dropped  the owned tetrahedra the drop filters reject (their rho_bar V_lag: LOST without
                  --ps-vertex-mass, handed to the neighbours through the degrees with it); 'after' = kept
                  tets the deposit then skips (zero float mass, released centroid beyond the region)
     (e) gpu      the mass handed to the GPU, its deferred part, the fixed-point grid's exact total and that
                  grid as the floats the host keeps
     (f) cpu      the tets the CPU loop deposits (all of them, or the GPU-deferred ones), the shares it
                  writes, and what the float grid actually gained
     (g) grid     the pass's mass grid before the merge (before and after the rho/rho_bar conversion)
     (h) run      Sum over the passes vs the merged grid vs the input mass, and the grid mean - 1
   The closures that must hold: Sum (a) = input; Sum (b) + Sum (c) = input under --ps-vertex-mass (the
   degrees agree across partitions); per pass gpu handed - deferred = fixed-point total, cpu shares = tets
   (less the drops), grid gained = shares; Sum (g) = the merged grid. */

#ifndef PS_MASS_AUDIT_HEADER
#define PS_MASS_AUDIT_HEADER

#include <cstddef>
#include <cstdlib>
#include <mutex>

namespace PSMassAudit
{
    // read once: the switch is per run. 2 adds the per-particle closure (ps_interpolation.cc), on ONE domain
    // only (partition < 0): the auto-tuner splits even small runs into partitions, so pass --partition 1 1 1
    // or --ps-gpu to get one
    inline int level()
    {
        static int const v = [] { char const *e = std::getenv("PS_MASS_AUDIT"); return e ? std::atoi(e) : 0; }();
        return v;
    }
    inline bool on() { return level() >= 1; }

    // the partition the calling thread deposits (DTFE.cpp's batch loop sets it; -1 = one domain)
    inline thread_local int partition = -1;

    // run totals over the passes, [0] the unaveraged pass, [1] the averaged one; under 'mutex'
    struct Totals
    {
        double owned = 0., tets = 0., deg0 = 0., dropped = 0., gridPre = 0., gridPost = 0.;
        size_t passes = 0;
    };
    inline std::mutex mutex;
    inline Totals totals[2];
}

#endif
