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

// Auto-tune: the mean number of streams at a random point of the box -- sum|V_Euler| over
// sum V_Lagrange of the Lagrangian tessellation -- estimated from up to 256 random Lagrangian
// patches, before any triangulation of the run exists. Needs the particles' Lagrangian
// positions (phase-space runs). Returns < 0 when it cannot tell (too few particles, 2D);
// *patchesUsed = the patches that contributed. Deterministic for given data.
double psEstimateMeanStreams(std::vector<Particle_data> const &particles, double const boxLo[3],
                             double const boxLen[3], bool periodic, int *patchesUsed);

#endif
