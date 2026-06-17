#ifndef IO_HEADER
#define IO_HEADER

/* Public interface of the I/O layer: top-level routines to read the input particles and to write
the computed fields. */

#include <vector>
#include <string>
#include "../define.h"
#include "../particle_data.h"
#include "../quantities.h"
#include "../user_options.h"

// Reads the input particles (and any user sampling points) according to the chosen input format.
void readInputData(std::vector<Particle_data> *p,
                   std::vector<Sample_point> *samplingCoordinates,
                   User_options *userOptions);

// Writes the requested unaveraged and volume-averaged fields to the output file(s).
void writeOutputData(Quantities &uQuantities,
                     Quantities &aQuantities,
                     User_options const &userOptions);

#endif
