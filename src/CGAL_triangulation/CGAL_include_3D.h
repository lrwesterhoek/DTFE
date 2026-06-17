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


/* CGAL setup for the 3D build: pulls in the Delaunay headers and defines the kernel and triangulation
   typedefs (DT, Point, Vertex_handle, ...) the rest of the code uses, with an optional TBB-parallel build. */

#include <boost/multiprecision/gmp.hpp>

#include <CGAL/basic.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Real_timer.h>   // wall-clock timer; not CGAL::Timer, whose CPU time sums across partition threads and overstates per-stage seconds
#include <CGAL/Cartesian.h>
#include <CGAL/Bbox_2.h>
#include <CGAL/spatial_sort.h>

#include <vector>
#include <list>
#include <iostream>
#include <string>
#include <stdio.h>
#include <algorithm>

#include <boost/random/uniform_real.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
typedef boost::mt19937 base_generator_type;


// per-vertex data attached to triangulation vertices
#include "vertexData.h"


typedef CGAL::Exact_predicates_inexact_constructions_kernel         K;
typedef CGAL::Triangulation_vertex_base_with_info_3<vertexData, K>  Vb;    // vertex stores a vertexData payload

#ifdef PARALLEL_TRIANGULATION
// TBB-parallel Delaunay build (opt-in: `make ... TBB=1`). Parallel_tag enables
// parallel insert_with_info. Fast_location is dropped (sequential point-location
// hierarchy); harmless since PS-DTFE iterates cells and never locates points.
#include <CGAL/Triangulation_cell_base_3.h>
typedef CGAL::Triangulation_data_structure_3<Vb, CGAL::Triangulation_cell_base_3<K>, CGAL::Parallel_tag> Tds;
typedef CGAL::Delaunay_triangulation_3<K, Tds>                      DT;
#else
typedef CGAL::Triangulation_data_structure_3<Vb>                    Tds;
typedef CGAL::Delaunay_triangulation_3<K, Tds, CGAL::Fast_location> DT;
#endif

typedef DT::Point                   Point;
typedef DT::Locate_type             Locate_type;
typedef DT::Cell_handle             Cell_handle;
typedef DT::Cell_iterator           Cell_iterator;
typedef DT::Finite_cells_iterator   Finite_cells_iterator;
typedef DT::Vertex_handle           Vertex_handle;
typedef CGAL::Real_timer            Timer;   // wall-clock seconds (see Real_timer.h note above)

typedef CGAL::Cartesian<Real>       K2;
typedef K2::Point_2                 Point_22;
typedef CGAL::Circle_2<K2>          Circle_2;
typedef CGAL::Bbox_2                Bbox_2;
typedef CGAL::Bbox_3                Bbox;






