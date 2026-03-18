#ifndef IO_HEADER
#define IO_HEADER

#include <vector>
#include <string>
#include "../define.h"
#include "../particle_data.h"
#include "../quantities.h"
#include "../user_options.h"

void readInputData(std::vector<Particle_data> *p,
                   std::vector<Sample_point> *samplingCoordinates,
                   User_options *userOptions);

void writeOutputData(Quantities &uQuantities,
                     Quantities &aQuantities,
                     User_options const &userOptions);

#endif
