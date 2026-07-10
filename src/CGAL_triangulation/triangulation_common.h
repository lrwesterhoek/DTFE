/* Umbrella header that pulls in the CGAL types and all shared triangulation/interpolation
   helpers; every interpolation compilation unit includes this one file. */

#ifndef TRIANGULATION_COMMON_HEADER
#define TRIANGULATION_COMMON_HEADER

#include "../define.h"
// Pick the 2D or 3D CGAL triangulation types based on the compile-time dimension.
#if NO_DIM==2
    #include "CGAL_include_2D.h"
#elif NO_DIM==3
    #include "CGAL_include_3D.h"
#endif

#include "../user_options.h"
#include "../quantities.h"
#include "../particle_data.h"
#include "particle_data_traits.h"
#include "../box.h"
#include "../message.h"
#include "../miscellaneous.h"

using namespace std;

#include "triangulation_miscellaneous.h"
#include "my_function.h"
#include "padding_test.h"
#include "field_computation.h"
#include "ps_cell_filter.h"

#endif
