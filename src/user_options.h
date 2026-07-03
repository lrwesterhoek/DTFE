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


/* Declares the Field selector and the User_options class that hold every command-line/config program option. */


#ifndef USER_OPTIONS_HEADER
#define USER_OPTIONS_HEADER

#include <string>
#include <vector>
#include <cstdlib>
#include <ctime> 

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include "define.h"
#include "miscellaneous.h"
#include "box.h"
#include "message.h"



// Boolean selectors for which fields to interpolate, with helpers to set/query them by name.
struct Field
{
    bool triangulation;
    bool density;
    bool velocity;
    bool velocity_gradient;
    bool velocity_divergence;
    bool velocity_shear;
    bool velocity_vorticity;
    bool velocity_std;
    bool velocity_dispersion;   // PS-DTFE: multi-stream velocity dispersion tensor (trace + full sigma_ij)
    bool scalar;
    bool scalar_gradient;
    bool velocity_tweb;
    bool velocity_vweb;

    Field()
    { triangulation = false; density = false; velocity = false; velocity_gradient = false; velocity_divergence = false; velocity_shear = false; velocity_vorticity = false; velocity_std = false; velocity_dispersion = false; scalar = false; scalar_gradient = false; velocity_tweb = false; velocity_vweb = false; }

    // Enables the field whose name matches 'choice'; returns false if 'choice' is none of the given names.
    bool updateChoices(std::string choice,
                       std::string str_triang, std::string str_den, std::string str_vel, std::string str_grad, std::string str_div, std::string str_shear, std::string str_vort, std::string str_velstd, std::string str_scalar, std::string str_scalarGrad, std::string str_tweb, std::string str_vweb, std::string str_veldisp="")
    {
        if ( choice.compare(str_triang)==0 ) triangulation = true;
        else if ( choice.compare(str_den)==0 ) density = true;
        else if ( choice.compare(str_vel)==0 ) velocity = true;
        else if ( choice.compare(str_grad)==0 ) velocity_gradient = true;
        else if ( choice.compare(str_div)==0 ) velocity_divergence = true;
        else if ( choice.compare(str_shear)==0 ) velocity_shear = true;
        else if ( choice.compare(str_vort)==0 ) velocity_vorticity = true;
        else if ( choice.compare(str_velstd)==0 ) velocity_std = true;
        else if ( not str_veldisp.empty() and choice.compare(str_veldisp)==0 ) velocity_dispersion = true;
        else if ( choice.compare(str_scalar)==0 ) scalar = true;
        else if ( choice.compare(str_scalarGrad)==0 ) scalar_gradient = true;
        else if ( choice.compare(str_tweb)==0 ) velocity_tweb = true;
        else if ( choice.compare(str_vweb)==0 ) velocity_vweb = true;
        else return false;
        return true;
    }

    // True if any field at all is selected.
    bool selected()
    { return ( density or velocity or velocity_gradient or velocity_divergence or velocity_shear or velocity_vorticity or velocity_std or velocity_dispersion or scalar or scalar_gradient or velocity_tweb or velocity_vweb ); }

    // True if any field derived from the velocity gradient (divergence/shear/vorticity/web) is selected.
    bool selectedVelocityDerivatives()
    { return ( velocity_divergence or velocity_shear or velocity_vorticity or velocity_tweb or velocity_vweb ); }
    void deselectVelocityDerivatives()
    { velocity_divergence = false; velocity_shear = false; velocity_vorticity = false; velocity_tweb = false; velocity_vweb = false; }
    // True if any velocity-related field is selected.
    bool selectedVelocity()
    { return ( velocity or velocity_gradient or velocity_divergence or velocity_shear or velocity_vorticity or velocity_std or velocity_dispersion or velocity_tweb or velocity_vweb ); }
    void deselectVelocity()
    { velocity = false; velocity_gradient = false; velocity_divergence = false; velocity_shear = false; velocity_vorticity = false; velocity_std = false; velocity_dispersion = false; velocity_tweb = false; velocity_vweb = false; }

    // True if any scalar-related field is selected.
    bool selectedScalar()
    { return ( scalar or scalar_gradient ); }
    void deselectScalar()
    { scalar = false; scalar_gradient = false; }
};


// All user-specified program options plus a few values derived at runtime.
struct User_options
{
    std::string inputFilename; // input particle positions file
    std::string outputFilename;// output file/files (root) name
#ifdef PHASE_SPACE
    std::string lagrangianInputFilename; // file with Lagrangian (initial condition) positions for PS-DTFE
#endif


    std::vector<size_t> gridSize;// grid size along each direction
    bool   periodic;      // true if the particle data is in a periodic box
    Box    boxCoordinates;// coordinates of the full particle data box
    bool   userGivenBoxCoordinates;// true if the user supplied the box coordinates
    int    inputFileType;   // input file type
    std::vector<int>   readParticleData;       // which particle properties to read
    std::vector<int>   readParticleSpecies;    // which particle species to read
    int    outputFileType;  // output file type


    Field  uField;        // unaveraged field: interpolated at the sampling points
    Field  aField;        // averaged field: volume average over the grid cell
    
    
    bool   partitionOn;   // true if the user selects the partition option
    std::vector<size_t> partition;// grid along which to split the particle data so the Delaunay triangulation runs on parts (time/memory)
    int    partNo;        // which partition the program should output (index in 'fileGrid')
    int    maxConcurrent; // cap on concurrently-built Delaunay triangulations (peak RAM ~ concurrency x per-triangulation); 0 = all threads. PS-DTFE: concurrent Lagrangian partitions; standard DTFE: concurrent spatial sub-triangulations.
    
    
    bool   paddingOn;     // true if a padding value is set
    bool   paddingMpcOn;  // true if the padding is in Mpc, not relative to box length
    Real   paddingParticles;// number of particles to use for padding when partitioning
    Box    paddingLength; // padding length (Mpc or relative box length) per direction, both edges
    bool   testPaddedBoundaries; // insert dummy boundary points to check the padding fully covers the region of interest
    
    
    bool   regionOn;      // true if computing only in a user-defined region of the box
    bool   regionMpcOn;   // true if region coordinates are in Mpc, not a fraction of box length
    Box    region;        // coordinates of the user-defined region
    
    
    // averaging computation
    int    method;        // volume-averaging method
    int    noPoints;      // sample points per grid cell for the average
    bool   noPointsOn;    // true if the user specified the number of sampling points
#ifdef PHASE_SPACE
    int    psAvgSubsamples; // PS-DTFE: linear sub-sample count nSub for '_a' fields (nSub^NO_DIM sub-grid per cell, cost ~nSub^NO_DIM); 3 = 27 sub-points (default), 1 = cell-centre.
    bool   psUseMetal;      // PS-DTFE: run the grid deposit on the Apple GPU (Metal); needs a METAL=1 build (else falls back to the CPU deposit with a warning).
#endif
    Real   averageDensity;// density normalization (computed as the box average if not given)
    size_t randomSeed;    // random seed for the Monte Carlo interpolation
    
    
    
    // light cone computations
    std::vector<Real> originPosition;   // light cone origin
    bool              redshiftConeOn;   // true if interpolating to a light cone grid
    Box               redshiftCone;     // light cone grid bounds (r_min, r_max, theta_min, theta_max, psi_min, psi_max)
    
    
    // additional options
    std::string configFilename;// config file from which program options were read
    bool   NGP;           // true if to use NGP grid interpolation instead of DTFE
    bool   CIC;           // true if to use CIC grid interpolation instead of DTFE
    bool   TSC;           // true if to use TSC grid interpolation instead of DTFE
    bool   PCS;           // true if to use PCS grid interpolation instead of DTFE
    bool   SPH;           // true if to use SPH grid interpolation instead of DTFE
    bool   Voronoi;       // true if to use Voronoi volume density with NGP grid assignment
    bool   interlace;     // true if to use interlacing to reduce aliasing in density field
    int    SPH_neighbors; // keep track of the number of neighbors used for the SPH grid interpolation
    Real   MpcValue;      // the value of 1Mpc in input units
    bool   extensive;     // true if scalar fields are extensive (else intensive); matters for TSC/SPH
    bool   approxPSD;     // compute approximate phase-space density f = rho * g (velocity-space DTFE)
    Real   lambda_th;    // eigenvalue threshold for T-web/V-web classification (default 0.0)
    Real   hubbleParam;  // Hubble parameter h for T-web/V-web normalization (-1 = read from header)
    int    verboseLevel;  // verbose level (see 'message.h')
    Real   randomSample;  // size of the random subsample if using only a subset of the data
    size_t poisson;       // if !=0, generate this many random particles instead of reading positions
    std::vector<std::string> additionalOptions; // extra options the user can pass to the code
    bool   transformToRedshiftSpaceOn;          // true if transforming position to redshift space
    std::vector<Real> transformToRedshiftSpace; // velocity coords for redshift-space transform (H=100 h km/s/Mpc)
    
    
    
    // set during runtime, not from the command line
#ifdef PHASE_SPACE
    Box    lagrangianRegion; // unpadded Lagrangian partition region (PS-DTFE cell ownership check)
    bool   psSuppressGridStats; // PS-DTFE partition path: silence per-partition coverage/stream stats (each covers ~1/Npart of the grid); aggregate reported once after the loop
    bool   psUseSubgrid;        // PS-DTFE partition path: allocate only each partition's Eulerian bounding box, mapped back via addFromSubgrid -> lower peak memory
    bool   psDeferNormalization; // PS-DTFE partition path: keep fields as density-weighted moments (no per-partition normalize) so they sum linearly and normalize once afterwards; correct for multi-stream cells spanning partitions
#endif
    Box    paddedBox;      // dimensions of the padded box
    std::vector<Real> fullBoxOffset; // full box left side per axis (redundant with 'boxCoordinates')
    std::vector<Real> fullBoxLength; // full box length per axis (redundant with 'boxCoordinates')

    bool   DTFE;           // true if DTFE interpolation method
    bool   userDefinedSampling; // true if the user supplied sampling coordinates
    int    noProcessors;   // number of threads running in parallel
    int    threadId;       // thread id for parallel processes
    Real   totalTime;      // total CPU time used by the program
    std::string programOptions; // all program options supplied by the user
    
    
    
    
    // function bodies are in "user_options.cc"
    User_options();
    void readOptions(int argc, char *argv[], bool getFileNames=true, bool showOptions=true );   // parse the command line/config, validate, and store the options
    void addOptions(po::options_description &allOptions,
                    po::options_description &visibleOptions,
                    po::positional_options_description &p); // register every program option with Boost.Program_options
    void helpInformation( po::options_description &visibleOptions, char *progName ); // print the detailed help and exit
    void shortHelp( char *progName ); // print the summary help and exit
    void printOptions(); // print the resolved configuration the run will use
    void updateFullBox(Box &newBox); // set 'boxCoordinates' and recompute 'fullBoxOffset'/'fullBoxLength'
    void updateEntries(size_t const noTotalParticles,
                       bool userSampling);   // after reading input: update paddedBox/fullBoxOffset/fullBoxLength and error-check members
    void updatePadding(size_t const noParticles); // resolve the padding length (from particle count or box-relative input)
};





#endif


