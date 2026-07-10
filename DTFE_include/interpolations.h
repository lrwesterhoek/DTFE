/*
 *  Forward declarations for the interpolation methods and random sampling functions
 *  (defined in their respective .cc files).
 */

#ifndef INTERPOLATIONS_HEADER
#define INTERPOLATIONS_HEADER

#include <vector>
#include "define.h"
#include "particle_data.h"
#include "quantities.h"
#include "user_options.h"


// NGP (Nearest Grid Point) interpolation
void NGP_interpolation(std::vector<Particle_data> *particles,
                       std::vector<Sample_point> &samples,
                       User_options &userOptions,
                       Quantities *q);

// CIC (Cloud In Cell) interpolation
void CIC_interpolation(std::vector<Particle_data> *particles,
                       std::vector<Sample_point> &samples,
                       User_options &userOptions,
                       Quantities *q);

// TSC (Triangular Shaped Cloud) interpolation
void TSC_interpolation(std::vector<Particle_data> *particles,
                       std::vector<Sample_point> &samples,
                       User_options &userOptions,
                       Quantities *q);

// PCS (Piecewise Cubic Spline) interpolation
void PCS_interpolation(std::vector<Particle_data> *particles,
                       std::vector<Sample_point> &samples,
                       User_options &userOptions,
                       Quantities *q);

// SPH (Smoothed Particle Hydrodynamics) interpolation
void SPH_interpolation(std::vector<Particle_data> *particles,
                       std::vector<Sample_point> &samples,
                       User_options &userOptions,
                       Quantities *q);

// Random subsample of the input particles (fraction from User_options::randomSample).
void randomSample(std::vector<Particle_data> particles,
                  std::vector<Particle_data> *subSample,
                  User_options userOptions);

// Poisson (uniform-random) particle field in a unit box.
void randomParticles(std::vector<Particle_data> *particles,
                     User_options *userOptions);


#endif
