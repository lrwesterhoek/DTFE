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

/* Arbitrary-point evaluation (--sample-points): CGAL-free entry points, shared by BOTH
   binaries. PS-DTFE evaluates the multi-stream phase-space fields (one stream per folded
   tetrahedron containing the point); the standard binary evaluates the Eulerian DTFE
   interpolant (exactly one containing tetrahedron, stream count = 0/1 coverage). The
   implementation lives in CGAL_triangulation/ps_point_eval.cc; the per-triangulation
   worker -- interpolatePoints_phaseSpace(DT&, User_options&) under PHASE_SPACE,
   interpolatePoints_standard(DT&, User_options&) otherwise -- is declared where the CGAL
   types exist (triangulation.cpp). Everything is a no-op unless --sample-points is given. */

#ifndef PS_POINT_EVAL_HEADER
#define PS_POINT_EVAL_HEADER

#include <functional>
#include <string>
#include <vector>

struct User_options;
struct Particle_data;

// True once psPointEvalInit() has loaded query points (i.e. --sample-points is active).
bool psPointEvalActive();

// Reads the --sample-points file, wraps the points into the periodic box, and builds the
// point bucket index. Call once, after the run options (region, averageDensity, periodic)
// are final and before any interpolation.
void psPointEvalInit(User_options &userOptions);

// Sorts each point's accumulated stream records (density descending) and reduces them to the
// per-point outputs: total density, mass-weighted mean velocity, velocity dispersion tensor,
// stream count. Call once, after ALL partitions have been processed.
void psPointEvalFinalize(User_options const &userOptions);

// Writes the finalized point outputs next to the grid outputs ('<root>.pts_*'). See the
// README (PS-DTFE point evaluation) for the exact file layout.
void psPointEvalWriteOutputs(User_options const &userOptions);

// '--resume' of a partitioned '--sample-points' run (2026-10-13; partition_checkpoint.h). The stream records
// are the point evaluation's only accumulators, and each belongs to the ONE partition that owns its
// tetrahedron, so they are kept per partition and checkpointed as such:
//   psPointEvalSetPartition    the partition the calling thread deposits (DTFE.cpp's batch loop, beside the
//                              mass audit's tag; -1 = none): its records are tagged with it
//   psPointEvalUnsavedBytes    the bytes the next save writes: the records of the merged partitions not saved yet
//   psPointEvalSavePartitions  writes '<dir>/points.<i>' for every merged, unsaved partition i (atomically: a
//                              temp file, fsync, rename) and marks it saved; false (with why) on any failure
//   psPointEvalLoadPartitions  on '--resume': reads the merged partitions' files back into the record store;
//                              false (with why) when one is missing or not this run's (key hash, point count,
//                              record layout, partition)
//   psPointEvalRemoveUnsaved   removes the 'points.<i>' files of partitions not saved by this run (an earlier
//                              run's, which the manifest no longer lists)
// keyHash is the checkpoint key's FNV-1a hash (hex), so a points file cannot be read into another run's grids.
void   psPointEvalSetPartition(int partition);
double psPointEvalUnsavedBytes(std::vector<char> const &merged, std::vector<char> const &saved);
bool   psPointEvalSavePartitions(std::string const &dir, std::string const &keyHash, std::vector<char> const &merged,
                                 std::vector<char> &saved, std::string &why);
bool   psPointEvalLoadPartitions(std::string const &dir, std::string const &keyHash, std::vector<char> const &merged,
                                 std::string &why);
void   psPointEvalRemoveUnsaved(std::string const &dir, std::vector<char> const &saved);

// --serve: moves the binary protocol onto a private duplicate of stdout and points stdout
// itself at stderr, so every log line the run prints lands on stderr and the protocol stream
// stays clean. Call before ANYTHING is printed (main() does, when '--serve' is on the command
// line). The request loop itself, psServe(DT&, User_options&), is declared with the workers.
void psServeRedirectStdout();

// --serve with --partition: a COMPOSITE server of partition tessellations (ps_point_eval.cc).
// DTFE() calls Begin once, AddPartition once per partition (from concurrent OpenMP threads is
// fine: each writes its own slot), then Run, which answers requests until stdin closes.
//   partOptions      the partition's own options, exactly as the batch partition loop sets them
//                    (they are its tessellation-cache identity and, in PS-DTFE, its ownership:
//                    lagrangianRegion)
//   selectParticles  fills the partition's particles (padding + periodic copies included); only
//                    called when the tessellation must actually be built
//   ownLo/ownHi      standard binary: the Eulerian box whose tetrahedra this partition owns
//                    (null in PS-DTFE, which owns by the Lagrangian centroid)
void psServeCompositeBegin(User_options const &userOptions, int totalPartitions);
void psServeCompositeAddPartition(int index, User_options &partOptions,
                                  std::function<void(std::vector<Particle_data>&)> const &selectParticles,
                                  double const *ownLo, double const *ownHi);
[[noreturn]] void psServeCompositeRun(User_options &userOptions);

// Standard binary, PARTITIONED point evaluation (the composite '--serve' above and, since 2026-10-13, a batch
// '--sample-points' run with '--partition'): a tetrahedron belongs to the partition whose Eulerian sub-box holds
// its centroid, so a point is answered by exactly one partition. psPointEvalOwnBox turns a sub-box of the split
// into that ownership box: lo/hi are the sub-box, outerLo/outerHi[d] say that its face on axis d is a face of the
// run's whole box -- the caller decides that from the split's GRID indices, never by comparing float coordinates
// (the last cut lo + dx*n can miss the box face by an ulp). An outer face becomes the single triangulation's own:
// periodic, exactly the primary box's face (the seam is then decided as in an unpartitioned run); non-periodic,
// open to infinity (hull tetrahedra keep an owner). Needs psPointEvalInit() (the box comes from its context).
void psPointEvalOwnBox(double const lo[], double const hi[], bool const outerLo[], bool const outerHi[],
                       double ownLo[], double ownHi[]);

// The ownership the standard worker applies to the triangulation(s) evaluated from now on: one partition's box
// (psPointEvalOwnBox) while DTFE()'s SERIAL standard partition loop evaluates it; null restores the single-
// triangulation default (periodic: the primary box; otherwise every tetrahedron). Not for triangulations evaluated
// concurrently -- the composite server passes each partition's box with the partition instead. A no-op in PS-DTFE,
// which owns by the Lagrangian centroid (the partition's lagrangianRegion).
void psPointEvalSetOwnership(double const *ownLo, double const *ownHi);

// --ps-window in a batch partition run: 1 when partition 'partOpts' can reach the window of
// 'mainOpts' by its occupancy map (beside its cached tessellation), 0 when it cannot (skip it),
// -1 when no usable map exists (process it). The map writer, which needs the tessellation type,
// psWriteOccupancyIfMissing(DT&, User_options&), is declared with the workers (triangulation.cpp).
int psOccupancyTouchesWindow(User_options const &partOpts, User_options const &mainOpts);

// Is this run's / partition's tessellation in '--tessellation-cache' (its header matches the full
// descriptor)? The batch partition loop then skips the particle selection.
bool psTessellationCached(User_options const &opts);

// The deposit passes since the last reset that asked for the GPU (--ps-gpu) and ran on the CPU instead
// (ps_interpolation.cc); reset = true also zeroes the count. Each pass warns itself, but the partition
// loop runs its non-master threads at verbose 0, so DTFE() reports the count after the loop (2026-10-08).
int psGpuFallbacksTaken(bool reset = false);

// Auto-tune: the mean number of streams at a random point of the box -- sum|V_Euler| over
// sum V_Lagrange of the Lagrangian tessellation -- estimated from up to 256 random Lagrangian
// patches, before any triangulation of the run exists. Needs the particles' Lagrangian
// positions (phase-space runs). Returns < 0 when it cannot tell (too few particles, 2D);
// *patchesUsed = the patches that contributed. Deterministic for given data.
// regionLo/regionHi (2026-10-11, optional): the mean over that EULERIAN box instead of the whole box -- the
// slab of a '--sample-points' plane, whose stream count can be 3x below the box mean when it misses the big
// haloes. Only patches whose Eulerian image meets the region are triangulated, and each tetrahedron counts
// the part of its Eulerian extent inside it. Falls back to the box mean when fewer than 16 patches meet it.
double psEstimateMeanStreams(std::vector<Particle_data> const &particles, double const boxLo[3],
                             double const boxLen[3], bool periodic, int *patchesUsed,
                             double const *regionLo = nullptr, double const *regionHi = nullptr);

#endif
