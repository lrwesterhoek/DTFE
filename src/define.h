
/*
  Compile-time configuration: dimensionality (NO_DIM), the Real float type, and the per-field
  component counts derived from NO_DIM. Included almost everywhere; edit with care.
*/

#ifndef DEFINE_HEADER
#define DEFINE_HEADER

#include <cstddef>



// number of spatial dimensions (2 or 3)
#ifndef NO_DIM
     #define NO_DIM 3
#endif
#if NO_DIM!=2
     #if NO_DIM!=3
          #error "The preprocessor macro 'NO_DIM' can take only the values 2 or 3!"
     #endif
#endif
#define NO_DIM2 (NO_DIM*NO_DIM)



// Real = float by default, double if DOUBLE is defined
#ifdef DOUBLE
    typedef double                         Real;
#else
    typedef float                          Real;
#endif



// Per-field component counts, fixed by NO_DIM -- do not modify.
static const size_t noVelComp = NO_DIM;                     // velocity components
static const size_t noGradComp = NO_DIM * NO_DIM;           // velocity gradient components
static const size_t noShearComp = (NO_DIM*(NO_DIM+1))/2-1;  // independent velocity shear components (2 in 2D, 5 in 3D); '-1' because shear is traceless
static const size_t noVortComp = ((NO_DIM-1)*NO_DIM)/2;     // independent velocity vorticity components (1 in 2D, 3 in 3D)
static const size_t noDispComp = (NO_DIM*(NO_DIM+1))/2;     // independent velocity-dispersion tensor components (3 in 2D, 6 in 3D), upper-triangle row-major, e.g. 3D = [xx,xy,xz,yy,yz,zz]

#ifndef NO_SCALARS
    #define NO_SCALARS 1
#endif
static const size_t noScalarComp = NO_SCALARS;               // scalar components
static const size_t noScalarGradComp = noScalarComp * NO_DIM;// scalar-field derivatives (NO_DIM per scalar component)



#define PI   3.14159265


// length conversion factor (input units per Mpc)
#ifndef MPC_UNIT
#define MPC_UNIT 1.
#endif



#endif
