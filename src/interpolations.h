/*
 *  Forward declarations for interpolation methods and random sampling functions.
 *  These functions are defined in their respective .cc files and compiled as
 *  separate translation units.
 */

#ifndef INTERPOLATIONS_HEADER
#define INTERPOLATIONS_HEADER

#include <vector>
#include "define.h"
#include "particle_data.h"
#include "quantities.h"
#include "user_options.h"


// NGP (Nearest Grid Point) interpolation - defined in NGP_interpolation.cc
void NGP_interpolation(std::vector<Particle_data> *particles,
                       std::vector<Sample_point> &samples,
                       User_options &userOptions,
                       Quantities *q);

// CIC (Cloud In Cell) interpolation - defined in CIC_interpolation.cc
void CIC_interpolation(std::vector<Particle_data> *particles,
                       std::vector<Sample_point> &samples,
                       User_options &userOptions,
                       Quantities *q);

// TSC (Triangular Shaped Cloud) interpolation - defined in TSC_interpolation.cc
void TSC_interpolation(std::vector<Particle_data> *particles,
                       std::vector<Sample_point> &samples,
                       User_options &userOptions,
                       Quantities *q);

// PCS (Piecewise Cubic Spline) interpolation - defined in PCS_interpolation.cc
void PCS_interpolation(std::vector<Particle_data> *particles,
                       std::vector<Sample_point> &samples,
                       User_options &userOptions,
                       Quantities *q);

// SPH (Smoothed Particle Hydrodynamics) interpolation - defined in SPH_interpolation.cc
void SPH_interpolation(std::vector<Particle_data> *particles,
                       std::vector<Sample_point> &samples,
                       User_options &userOptions,
                       Quantities *q);

// Random sampling functions - defined in random.cc
void randomSample(std::vector<Particle_data> particles,
                  std::vector<Particle_data> *subSample,
                  User_options userOptions);

void randomParticles(std::vector<Particle_data> *particles,
                     User_options *userOptions);


#endif
