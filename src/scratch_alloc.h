/* Out-of-core backing for the full-grid accumulators (--scratch-dir).
 *
 * PROBLEM: the full-resolution output grids are IRREDUCIBLE by --partition (every Lagrangian
 * partition deposits into the same shared grid), so a 1024^3 run with the production field set
 * needs ~146 GB of accumulators against a 64 GB machine. No tuning can fix that in RAM.
 *
 * MECHANISM: scratch_alloc.cc REPLACES the global operator new/delete (a standard C++
 * replaceable-function hook). Once armed, an allocation of at least the threshold (default
 * 1 GB; DTFE_SCRATCH_MIN_GB overrides -- tests set it tiny to exercise the route on small
 * grids) is served from an mmap'ed, immediately-unlinked file in the scratch directory:
 *   - std::vector code is UNTOUCHED -- std::allocator calls ::operator new, so every full-grid
 *     .assign()/.resize() lands here with zero type ripple through quantities.h;
 *   - unlink-after-mmap: the kernel reclaims the disk space when the process exits, INCLUDING
 *     on a crash -- stale scratch files are impossible;
 *   - dirty pages are written back by the kernel under memory pressure, so peak RSS stays
 *     bounded while the arithmetic is bit-identical (it is ordinary memory);
 *   - everything below the threshold (per-partition subgrid temps, CGAL, the works) takes the
 *     ordinary malloc path with one atomic load of overhead.
 *
 * LINKING: only the two BINARIES get scratch_alloc.o (the real hook). libDTFE gets
 * scratch_alloc_stub.o instead, whose scratchArm() refuses -- a library must never hijack its
 * host program's global allocator.
 *
 * The scratch directory must be on a LOCAL volume. This repo lives in iCloud Drive: an mmap
 * scratch file there would sync-upload ~146 GB, so user_options.cc refuses such paths.
 */

#ifndef SCRATCH_ALLOC_HEADER
#define SCRATCH_ALLOC_HEADER

#include <cstddef>

namespace ScratchAlloc
{
    // Arm the scratch route: allocations >= thresholdBytes are served from mmap'ed files in
    // 'dir' from this call on. Returns false when the interposition is not linked into this
    // program (libDTFE stub) -- callers must surface that instead of silently proceeding.
    // 'dir' must already exist and be writable (validated by the caller).
    bool scratchArm(char const *dir, std::size_t thresholdBytes);

    // diagnostics for banners/tests: bytes currently mmap'ed, and mappings served in total
    long scratchLiveBytes();
    long scratchTotalAllocs();
}

#endif
