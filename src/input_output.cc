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

/*!
Here are the functions used for data input and output.
To easily find how to read the input data go to the function:
    

*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>
#include <string>
#include <boost/filesystem.hpp>
namespace bfs=boost::filesystem;

#include "define.h"
#include "particle_data.h"
#include "quantities.h"
#include "user_options.h"
#include "message.h"


// contains the definitions of some classes and fucntions used only for input and output purposes
#include "io/input_output.h"

// different data format readers and writters
#include "io/gadget_reader_header.cc" 
#include "io/gadget_reader_binary.cc"  // reader for binary Gadget files
#include "io/gadget_reader_HDF5.cc" // reader for Gadget snapshots in HDF5 format
#include "io/gadget_reader_HDF5_Cristian.cc" // reader for Gadget HDF5 files using Cristian's format
#include "io/gadget_reader_MOG.cc"  // reader for Gadget snapshots and MOG forces into scalar field
#include "io/hdf5_input_my_DESI.cc" // reader for some personal DESI particle data
#include "io/binary_io.cc"      // writer for binary file format
#include "io/text_io.cc"        // reader/writer for text files
#include "io/my_io.cc"          // reader/writer for text files
#include "io/density_file_io.cc"    // own output file format



void openInputBinaryFile(std::fstream & inputFile,
                         std::string & fileName);
void openOutputBinaryFile(std::fstream & outputFile,
                          std::string & fileName);
void openInputTextFile(std::fstream & inputFile,
                       std::string & fileName);
void openOutputTextFile(std::fstream & outputFile,
                        std::string & fileName);





//! Functions for reading the input data

typedef void (*FunctionReadInputData)(std::string, Read_data<Real> *, User_options *);

/*! This function uses the 'inputFileType' entry in the userOptions class to decide on the function that will be used to read the input data. */
FunctionReadInputData chooseInputDataReadFunction(int const inputFileType)
{
    if ( inputFileType==101 )
        return &readGadgetFile;    // Read the input data from a single/multiple Gadget snapshot file (works only for snapshots type 1 or 2). See the "src/io/gadget_reader.cc" for the definition of this function.
    if ( inputFileType==109 )
        return &readGadgetFile_MOG;    // Read the input data from a single/multiple Gadget snapshot file (works only for snapshots type 1 or 2). Also reads modified gravity theory forces: the gravitational and fifth force. See the "src/io/gadget_reader_MOG.cc" for the definition of this function.
#ifndef DOUBLE
#ifdef HDF5
    else if ( inputFileType==105 )
        return &HDF5_readGadgetFile;        // Read the input data from a HDF5 Gadget snapshot file. See the "src/io/gadget_reader_HDF5.cc" for the definition of this function.
    else if ( inputFileType==106 )
        return &HDF5_readGadgetFile_HI;     // Read the input HI data from a HDF5 Gadget snapshot file. See the "src/io/gadget_reader_HDF5.cc" for the definition of this function.
    else if ( inputFileType==107 )
        return &HDF5_readData_DESI;         // Read the input data from a HDF5 file storing DESI particle data.
    else if ( inputFileType==108 )
        return &HDF5_readGadgetFile_Cristian;     // Read the input HI data from a HDF5 Gadget snapshot file. See the "src/io/gadget_reader_HDF5_Cristian.cc" for the definition of this function.
#endif
    else if ( inputFileType==111 )
        return &readTextFile;               // Read the input data from a text file. See the "src/io/text_io.cc" for the definition of this function.
    else if ( inputFileType==112 )
        return &readTextFile_positions;     // Read the input data from a text file. The text file has only particle positions. See the "src/io/text_io.cc" for the definition of this function.
    else if ( inputFileType==121 )
        return &readBinaryFile;             // Define your custom function to read the input data. Do this in the "src/io/bynary_io.cc" file.
    else if ( inputFileType==122 )
        return &readBinaryFile_StructuredData;  // Function to read the input data. Defined in the "src/io/binary_io.cc" file.
    else if ( inputFileType==131 )
        return &readMyFile;                 // Define your custom function to read the input data. Do this in the "src/io/my_io.cc" file.
#endif
    else
        throwError( "Unknow value for the 'inputFileType' argument in function 'chooseInputDataReadFunction'. The program could not recognize the input data file type." );
    
    return NULL;
}





/*! This function reads the particle data used for the DTFE computation. */
void readInputData(std::vector<Particle_data> *p,
                   std::vector<Sample_point> *samplingCoordinates,
                   User_options *userOptions)
{
    std::string filename = userOptions->inputFilename;  // the name of the input file
    Read_data<Real> readData; // will store the data read from the input file
    
    
    // Read the data from the input file - see the function 'chooseInputDataReadFunction' that selects the input function used to read in the input data file
    FunctionReadInputData functionReadInputData = chooseInputDataReadFunction( userOptions->inputFileType );
    (*functionReadInputData)( filename, &readData, userOptions );
    
    
    // 'userOptions->MpcValue' is the conversion factor from the units in the input data file to Mpc units - do the next computation only if userOptions->MpcValue!=1
    if ( userOptions->MpcValue!=Real(1.) )
    {
        for (size_t i=0; i<userOptions->boxCoordinates.size(); ++i)
            userOptions->boxCoordinates[i] /= userOptions->MpcValue;
        
        size_t noParticles = readData.noParticles();  // returns the number of particles
        Real *positions = readData.position();  //returns pointer to array storing particle positions
        for (size_t i=0; i<noParticles*NO_DIM; ++i)
            positions[i] /= userOptions->MpcValue;
        
        // if the user inserted user defined sampling points, divide the coordinates of those points by the normalization factor
        if ( readData.noSamples()!=size_t(0) )
        {
            Real *sampling = readData.sampling();  // returns pointer to user defined sampling coordinates
            Real *delta = readData.delta();        // returns pointer to user defined sampling cell sizes
            for (size_t i=0; i<readData.noSamples()*NO_DIM; ++i)
            {
                sampling[i] /= userOptions->MpcValue;
                delta[i] /= userOptions->MpcValue;
            }
        }
    }
    
    
    // Handle Lagrangian positions for PS-DTFE
#ifdef PHASE_SPACE
    // If Lagrangian positions were read from the main file (InitialCoordinates dataset), apply MpcValue
    if ( readData._lagrangianPositionPopulated and userOptions->MpcValue!=Real(1.) )
    {
        size_t noParticles2 = readData.noParticles();
        Real *lagPos = readData.lagrangianPosition();
        for (size_t i=0; i<noParticles2*NO_DIM; ++i)
            lagPos[i] /= userOptions->MpcValue;
    }

    // If Lagrangian positions not in main file, read from separate file with ID-based matching
    if ( not readData._lagrangianPositionPopulated )
    {
        if ( userOptions->lagrangianInputFilename.empty() )
            throwError( "PS-DTFE requires Lagrangian (initial condition) positions. The main input file does not contain an 'InitialCoordinates' dataset. Specify an IC snapshot using '--lagrangianInput'." );

#ifdef HDF5
        MESSAGE::Message message( userOptions->verboseLevel );
        size_t const noParticles = readData.noParticles();

        // Read header of Lagrangian file to get file count and particle numbers
        Gadget_header lagHeader;
        std::string lagFilename = userOptions->lagrangianInputFilename;
        std::string lagFirstFile = lagFilename;
        bool lagSingleFile = true;

        if ( not bfs::exists(lagFilename) )
        {
            lagFirstFile = lagHeader.filename(lagFilename, 0);
            lagSingleFile = false;
        }
        HDF5_readGadgetHeader(lagFirstFile, &lagHeader);

        int lagNumFiles = lagSingleFile ? 1 : lagHeader.num_files;

        // Count total particles in Lagrangian file
        size_t lagTotalParticles = 0;
        if (lagSingleFile)
        {
            for (int i=0; i<6; ++i)
                if (userOptions->readParticleSpecies[i])
                    lagTotalParticles += lagHeader.npart[i];
        }
        else
        {
            size_t lagPartCount[6];
            HDF5_countGadgetParticleNumber(lagFilename, lagNumFiles, userOptions->verboseLevel, lagPartCount);
            for (int i=0; i<6; ++i)
                if (userOptions->readParticleSpecies[i])
                    lagTotalParticles += lagPartCount[i];
        }

        if (lagTotalParticles != noParticles)
            throwError( "Particle count mismatch: main file has ", noParticles, " particles, Lagrangian file has ", lagTotalParticles, "." );

        message << "\nReading Lagrangian positions from '" << userOptions->lagrangianInputFilename
                << "' (" << lagTotalParticles << " particles, " << lagNumFiles << " file(s)) with ID matching...\n" << MESSAGE::Flush;

        // Allocate arrays for Lagrangian file data
        std::vector<float> lagCoords(lagTotalParticles * NO_DIM);
        std::vector<uint64_t> lagIDs(lagTotalParticles);

        // Read Coordinates and ParticleIDs from all Lagrangian file chunks
        size_t lagOffset = 0;
        for (int fi = 0; fi < lagNumFiles; ++fi)
        {
            std::string fn = lagSingleFile ? lagFilename : lagHeader.filename(lagFilename, fi);

            Gadget_header chunkHeader;
            HDF5_readGadgetHeader(fn, &chunkHeader);

            H5::H5File lagFile( fn.c_str(), H5F_ACC_RDONLY, H5::FileAccPropList::DEFAULT );

            for (int type = 0; type < 6; ++type)
            {
                if (!userOptions->readParticleSpecies[type]) continue;
                if (chunkHeader.npart[type] <= 0) continue;

                char buf[500];
                snprintf(buf, sizeof(buf), "/PartType%d", type);
                H5::Group grp = lagFile.openGroup(buf);

                H5::DataSet coordDS = grp.openDataSet("Coordinates");
                coordDS.read(&lagCoords[lagOffset * NO_DIM], H5::PredType::NATIVE_FLOAT);

                H5::DataSet idDS = grp.openDataSet("ParticleIDs");
                idDS.read(&lagIDs[lagOffset], H5::PredType::NATIVE_UINT64);

                lagOffset += chunkHeader.npart[type];
            }
        }

        // Match Lagrangian positions to main file particles by ID
        Real *lagPosOut = readData.lagrangianPosition();

        // Check if Lagrangian IDs are contiguous (common for IC files: IDs = 1..N)
        uint64_t minID = *std::min_element(lagIDs.begin(), lagIDs.end());
        uint64_t maxID = *std::max_element(lagIDs.begin(), lagIDs.end());

        if (maxID - minID + 1 == lagTotalParticles)
        {
            // Contiguous IDs — use direct array indexing (O(N), no sorting)
            message << "\t ID range is contiguous (" << minID << ".." << maxID << "), using direct indexing.\n" << MESSAGE::Flush;

            // Reorder lagCoords so index [id - minID] gives the coordinates for that ID
            // The IC file may already be in ID order, but we handle the general case
            // by building a position lookup indexed by (id - minID)
            std::vector<float> lagByID(lagTotalParticles * NO_DIM);
            for (size_t i = 0; i < lagTotalParticles; ++i)
            {
                size_t slot = lagIDs[i] - minID;
                for (int d = 0; d < NO_DIM; ++d)
                    lagByID[slot * NO_DIM + d] = lagCoords[i * NO_DIM + d];
            }

            // Map main file particles to their Lagrangian positions
            for (size_t i = 0; i < noParticles; ++i)
            {
                uint64_t id = readData._particleIDs[i];
                size_t slot = id - minID;
                for (int d = 0; d < NO_DIM; ++d)
                    lagPosOut[NO_DIM*i + d] = lagByID[slot * NO_DIM + d];
            }
        }
        else
        {
            // Non-contiguous IDs — use sort-based matching (O(N log N))
            message << "\t IDs are non-contiguous, using sort-based matching.\n" << MESSAGE::Flush;

            std::vector<size_t> lagSortIdx(lagTotalParticles);
            std::iota(lagSortIdx.begin(), lagSortIdx.end(), size_t(0));
            std::sort(lagSortIdx.begin(), lagSortIdx.end(),
                      [&lagIDs](size_t a, size_t b) { return lagIDs[a] < lagIDs[b]; });

            for (size_t i = 0; i < noParticles; ++i)
            {
                uint64_t id = readData._particleIDs[i];
                auto it = std::lower_bound(lagSortIdx.begin(), lagSortIdx.end(), id,
                            [&lagIDs](size_t idx, uint64_t val) { return lagIDs[idx] < val; });
                if (it == lagSortIdx.end() || lagIDs[*it] != id)
                    throwError( "Particle ID ", id, " from main file not found in Lagrangian file." );
                size_t lagIdx = *it;
                for (int d = 0; d < NO_DIM; ++d)
                    lagPosOut[NO_DIM*i + d] = lagCoords[lagIdx * NO_DIM + d];
            }
        }

        // Apply MpcValue conversion to Lagrangian positions
        if (userOptions->MpcValue != Real(1.))
        {
            for (size_t i = 0; i < noParticles * NO_DIM; ++i)
                lagPosOut[i] /= userOptions->MpcValue;
        }

        readData._lagrangianPositionPopulated = true;
        message << "\t Lagrangian position matching complete.\n" << MESSAGE::Flush;
#else
        throwError( "The '--lagrangianInput' option requires HDF5 support. Please compile with -DHDF5." );
#endif
    }

    // Final check
    if ( not readData._lagrangianPositionPopulated )
        throwError( "PS-DTFE mode requires Lagrangian positions but none were loaded." );
#endif

    // now store the data in the 'Particle_data list'. It also copies the user given sampling coordinates, if any - none in this case.
    readData.transferData( p, samplingCoordinates );
}









//! Functions for writing the output data


//! short class to easily change between writing the output to different file types while the program is running. The user can choose the output file type using the '--output' option.
class OutputData
{
    public:
    int outputFileType;
    
    // class constructor - initializes the type of the output file
    OutputData(int fileType)
    {
        this->outputFileType = fileType;
    }
    
    // the function that calls the function doing the actual writing
    template <typename T>
    void write(T &dataToWrite,
                std::string filename,
                std::string variableName,
                User_options const &userOptions)
    {
        if ( this->outputFileType==101 )
            writeBinaryFile( dataToWrite, filename, variableName, userOptions );        // writes the data to a binary file (see "binary_io.cc" for function definition)
        else if ( this->outputFileType==111 )
            writeTextFile( dataToWrite, filename, variableName, userOptions );          // writes the data to a text file (see "text_io.cc" for function definition)
        else if ( this->outputFileType==112 )
            writeTextFile_gridIndex( dataToWrite, filename, variableName, userOptions );// writes the data to a text file, but on each line writes also the coordinates of the grid cell corresponding to the result being written (see "text_io.cc" for function definition)
        else if ( this->outputFileType==113 )
            writeTextFile_samplingPosition( dataToWrite, filename, variableName, userOptions ); // writes the data to a text file, but on each line writes also the sampling point coordinates corresponding to the result being written  (see "text_io.cc" for function definition)
        else if ( this->outputFileType==114 )
            writeTextFile_redshiftConePosition( dataToWrite, filename, variableName, userOptions ); // writes the data to a text file, but on each line writes also the sampling point coordinates for a redshift cone grid  (see "text_io.cc" for function definition)
        else if ( this->outputFileType==121 )
            writeMyFile( dataToWrite, filename, variableName, userOptions );   // the format that I use where it write to a binary file with a header that describes the data stored in it
        else if ( this->outputFileType==100 )
            writeSpecialFile( dataToWrite, filename, variableName, userOptions );   // the format that I use where it write to a binary file with a header that describes the data stored in it
    }
};


/*! This function writes the output data to a file, each different 'variable' being written to a separate file. 
You can modify this function as you please. */
void writeOutputData(Quantities &uQuantities,
                     Quantities &aQuantities,
                     User_options const &userOptions)
{
    // select the file type for output
    OutputData output( userOptions.outputFileType );
    
    // output the desired quantities to file/files
    // outputs the density
    if ( userOptions.uField.density )
        output.write( uQuantities.density, userOptions.outputFilename + ".den", "density", userOptions );
    if ( userOptions.aField.density )
        output.write( aQuantities.density, userOptions.outputFilename + ".a_den", "volume averaged density", userOptions );
    
    // outputs the velocity
    if ( userOptions.uField.velocity )
        output.write( uQuantities.velocity, userOptions.outputFilename + ".vel", "velocity", userOptions );
    if ( userOptions.aField.velocity )
        output.write( aQuantities.velocity, userOptions.outputFilename + ".a_vel", "volume averaged velocity", userOptions );
    
    // outputs the velocity gradient
    if ( userOptions.uField.velocity_gradient )
        output.write( uQuantities.velocity_gradient, userOptions.outputFilename + ".velGrad", "velocity gradient", userOptions );
    if ( userOptions.aField.velocity_gradient )
        output.write( aQuantities.velocity_gradient, userOptions.outputFilename + ".a_velGrad", "volume averaged velocity gradient", userOptions );
    
    // outputs the velocity divergence
    if ( userOptions.uField.velocity_divergence )
        output.write( uQuantities.velocity_divergence, userOptions.outputFilename + ".velDiv", "velocity divergence", userOptions );
    if ( userOptions.aField.velocity_divergence )
        output.write( aQuantities.velocity_divergence, userOptions.outputFilename + ".a_velDiv", "volume averaged velocity divergence", userOptions );
    
    // outputs the velocity shear
    if ( userOptions.uField.velocity_shear )
        output.write( uQuantities.velocity_shear, userOptions.outputFilename + ".velShear", "velocity shear", userOptions );
    if ( userOptions.aField.velocity_shear )
        output.write( aQuantities.velocity_shear, userOptions.outputFilename + ".a_velShear", "volume averaged velocity shear", userOptions );
    
    // outputs the velocity vorticity
    if ( userOptions.uField.velocity_vorticity )
        output.write( uQuantities.velocity_vorticity, userOptions.outputFilename + ".velVort", "velocity vorticity", userOptions );
    if ( userOptions.aField.velocity_vorticity )
        output.write( aQuantities.velocity_vorticity, userOptions.outputFilename + ".a_velVort", "volume averaged velocity vorticity", userOptions );
    
    // outputs the velocity standard deviation
    if ( userOptions.aField.velocity_std )
        output.write( aQuantities.velocity_std, userOptions.outputFilename + ".a_velStd", "volume averaged velocity standard deviation", userOptions );
    
    // outputs the scalar fields
    if ( userOptions.uField.scalar )
        output.write( uQuantities.scalar, userOptions.outputFilename + ".scalar", "scalar", userOptions );
    if ( userOptions.aField.scalar )
        output.write( aQuantities.scalar, userOptions.outputFilename + ".a_scalar", "volume averaged scalar", userOptions );
    
    // outputs the scalar fields gradient
    if ( userOptions.uField.scalar_gradient )
        output.write( uQuantities.scalar_gradient, userOptions.outputFilename + ".scalarGrad", "scalar gradient", userOptions );
    if ( userOptions.aField.scalar_gradient )
        output.write( aQuantities.scalar_gradient, userOptions.outputFilename + ".a_scalarGrad", "volume averaged scalar gradient", userOptions );

    // outputs the T-web classification and eigenvalues
    if ( userOptions.uField.velocity_tweb )
    {
        output.write( uQuantities.velocity_tweb, userOptions.outputFilename + ".velTweb", "T-web classification", userOptions );
        output.write( uQuantities.velocity_tweb_eigenvalues, userOptions.outputFilename + ".velTwebEig", "T-web eigenvalues", userOptions );
    }
    if ( userOptions.aField.velocity_tweb )
    {
        output.write( aQuantities.velocity_tweb, userOptions.outputFilename + ".a_velTweb", "volume averaged T-web classification", userOptions );
        output.write( aQuantities.velocity_tweb_eigenvalues, userOptions.outputFilename + ".a_velTwebEig", "volume averaged T-web eigenvalues", userOptions );
    }

    // outputs the V-web classification and eigenvalues
    if ( userOptions.uField.velocity_vweb )
    {
        output.write( uQuantities.velocity_vweb, userOptions.outputFilename + ".velVweb", "V-web classification", userOptions );
        output.write( uQuantities.velocity_vweb_eigenvalues, userOptions.outputFilename + ".velVwebEig", "V-web eigenvalues", userOptions );
    }
    if ( userOptions.aField.velocity_vweb )
    {
        output.write( aQuantities.velocity_vweb, userOptions.outputFilename + ".a_velVweb", "volume averaged V-web classification", userOptions );
        output.write( aQuantities.velocity_vweb_eigenvalues, userOptions.outputFilename + ".a_velVwebEig", "volume averaged V-web eigenvalues", userOptions );
    }

    // outputs the stream count (PS-DTFE only)
#ifdef PHASE_SPACE
    if ( not uQuantities.stream_count.empty() )
        output.write( uQuantities.stream_count, userOptions.outputFilename + ".streams", "stream count", userOptions );
#endif
}




















