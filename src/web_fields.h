/* Derived velocity fields and the cosmic-web classification (web_fields.cc): the velocity divergence, shear and
   vorticity from the stored velocity gradient, the V-web and the T-web labels, and postProcessWebFields, which
   runs them all after the interpolation (called by both DTFE() overloads in DTFE.cpp). */

#ifndef DTFE_WEB_FIELDS_HEADER
#define DTFE_WEB_FIELDS_HEADER

#include "define.h"
#include "user_options.h"
#include "quantities.h"

// Derives velocity divergence, shear and vorticity from the stored velocity gradient.
void computeDivergenceShearVorticity(Field &fields,
                                     int const verboseLevel,
                                     Quantities *quantities);

// Derives the V-web cosmic-web classification from the velocity gradient.
void computeWebClassification(Field &fields,
                               int const verboseLevel,
                               Real lambda_th,
                               Real hubbleParam,
                               Real scaleFactor,
                               Quantities *q);

// Derives the T-web classification from the tidal tensor of the DENSITY grid (FFT Poisson).
void computeTidalWebClassification(Field &fields,
                                   User_options const &userOptions,
                                   Quantities *q);

void postProcessWebFields(User_options &userOptions,
                          Quantities *uQuantities,
                          Quantities *aQuantities);

#endif
