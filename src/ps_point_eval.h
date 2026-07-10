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

/* PS-DTFE arbitrary-point evaluation (--sample-points): CGAL-free entry points.
   The implementation lives in CGAL_triangulation/ps_point_eval.cc; the per-partition
   worker interpolatePoints_phaseSpace(DT&, User_options&) is declared where the CGAL
   types exist (triangulation.cpp). Everything is a no-op unless --sample-points is given. */

#ifndef PS_POINT_EVAL_HEADER
#define PS_POINT_EVAL_HEADER

#ifdef PHASE_SPACE

struct User_options;

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

#endif // PHASE_SPACE

#endif
