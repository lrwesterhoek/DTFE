#ifndef TRIANGULATION_COMMON_HEADER
#define TRIANGULATION_COMMON_HEADER

#include "../define.h"
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

#endif
