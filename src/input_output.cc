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

/* Reads particle data and writes result fields, dispatching to a format-specific reader/writer by file-type code. */

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
#include "ps_point_eval.h"


#include "io/input_output.h"

// data format readers and writers
#include "io/gadget_reader_header.cc"
#include "io/gadget_reader_binary.cc"  // binary Gadget files
#include "io/gadget_reader_HDF5.cc" // Gadget snapshots in HDF5 format
#include "io/binary_io.cc"      // binary writer
#include "io/text_io.cc"        // text reader/writer
#include "io/density_file_io.cc"    // own output file format



void openInputBinaryFile(std::fstream & inputFile,
                         std::string & fileName);
void openOutputBinaryFile(std::fstream & outputFile,
                          std::string & fileName);
void openInputTextFile(std::fstream & inputFile,
                       std::string & fileName);
void openOutputTextFile(std::fstream & outputFile,
                        std::string & fileName);





// Functions for reading the input data

typedef void (*FunctionReadInputData)(std::string, Read_data<Real> *, User_options *);

// Pick the reader function for the given 'inputFileType'.
FunctionReadInputData chooseInputDataReadFunction(int const inputFileType)
{
    if ( inputFileType==101 )
        return &readGadgetFile;    // single/multiple Gadget snapshot, type 1 or 2
#ifndef DOUBLE
#ifdef HDF5
    else if ( inputFileType==105 )
        return &HDF5_readGadgetFile;        // HDF5 Gadget snapshot
    else if ( inputFileType==106 )
        return &HDF5_readGadgetFile_HI;     // HI data from HDF5 Gadget snapshot
#endif
    else if ( inputFileType==111 )
        return &readTextFile;               // text file
    else if ( inputFileType==112 )
        return &readTextFile_positions;     // text file, positions only
    else if ( inputFileType==121 )
        return &readBinaryFile;             // custom binary reader
    else if ( inputFileType==122 )
        return &readBinaryFile_StructuredData;  // structured-data binary reader
#endif
    else
        throwError( "Unknow value for the 'inputFileType' argument in function 'chooseInputDataReadFunction'. The program could not recognize the input data file type." );
    
    return NULL;
}





// Read the particle data used for the DTFE computation.
void readInputData(std::vector<Particle_data> *p,
                   std::vector<Sample_point> *samplingCoordinates,
                   User_options *userOptions)
{
    std::string filename = userOptions->inputFilename;
    Read_data<Real> readData; // stores the data read from the input file


    FunctionReadInputData functionReadInputData = chooseInputDataReadFunction( userOptions->inputFileType );
    (*functionReadInputData)( filename, &readData, userOptions );


    // MpcValue is the conversion factor from input-file units to Mpc; rescale only if != 1
    if ( userOptions->MpcValue!=Real(1.) )
    {
        for (size_t i=0; i<userOptions->boxCoordinates.size(); ++i)
            userOptions->boxCoordinates[i] /= userOptions->MpcValue;
        
        size_t noParticles = readData.noParticles();
        Real *positions = readData.position();
        for (size_t i=0; i<noParticles*NO_DIM; ++i)
            positions[i] /= userOptions->MpcValue;

        // rescale any user-defined sampling points too
        if ( readData.noSamples()!=size_t(0) )
        {
            Real *sampling = readData.sampling();  // sampling coordinates
            Real *delta = readData.delta();        // sampling cell sizes
            for (size_t i=0; i<readData.noSamples()*NO_DIM; ++i)
            {
                sampling[i] /= userOptions->MpcValue;
                delta[i] /= userOptions->MpcValue;
            }
        }
    }
    
    
    // PS-DTFE needs Lagrangian (initial-condition) positions
#ifdef PHASE_SPACE
    // already read from the main file's InitialCoordinates dataset: just rescale
    if ( readData._lagrangianPositionPopulated and userOptions->MpcValue!=Real(1.) )
    {
        size_t noParticles2 = readData.noParticles();
        Real *lagPos = readData.lagrangianPosition();
        for (size_t i=0; i<noParticles2*NO_DIM; ++i)
            lagPos[i] /= userOptions->MpcValue;
    }

    // otherwise read from a separate IC file, matching by particle ID
    if ( not readData._lagrangianPositionPopulated )
    {
        if ( userOptions->lagrangianInputFilename.empty() )
            throwError( "PS-DTFE requires Lagrangian (initial condition) positions. The main input file does not contain an 'InitialCoordinates' dataset. Specify an IC snapshot using '--lagrangianInput'." );

#ifdef HDF5
        MESSAGE::Message message( userOptions->verboseLevel );
        size_t const noParticles = readData.noParticles();

        // read Lagrangian file header for file count and particle numbers
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

        // count total particles in the Lagrangian file
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

        message << "\nReading Lagrangian positions from '" << MESSAGE::cBlue() << userOptions->lagrangianInputFilename
                << MESSAGE::cReset() << "' (" << MESSAGE::cMagenta() << lagTotalParticles << MESSAGE::cReset()
                << " particles, " << lagNumFiles << " file(s)) with ID matching...\n" << MESSAGE::Flush;

        std::vector<float> lagCoords(lagTotalParticles * NO_DIM);
        std::vector<uint64_t> lagIDs(lagTotalParticles);

        // read Coordinates and ParticleIDs from all file chunks
        size_t lagOffset = 0;
        for (int fi = 0; fi < lagNumFiles; ++fi)
        {
            std::string fn = lagSingleFile ? lagFilename : lagHeader.filename(lagFilename, fi);

            Gadget_header chunkHeader;
            HDF5_readGadgetHeader(fn, &chunkHeader);

            H5::H5File lagFile( fn.c_str(), H5F_ACC_RDONLY );

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

        // match Lagrangian positions to main-file particles by ID
        Real *lagPosOut = readData.lagrangianPosition();

        // contiguous IDs (IDs = 1..N) allow O(N) direct indexing instead of a sort
        uint64_t minID = *std::min_element(lagIDs.begin(), lagIDs.end());
        uint64_t maxID = *std::max_element(lagIDs.begin(), lagIDs.end());

        if (maxID - minID + 1 == lagTotalParticles)
        {
            message << "\t ID range is contiguous (" << MESSAGE::cMagenta() << minID << ".." << maxID
                    << MESSAGE::cReset() << "), using direct indexing.\n" << MESSAGE::Flush;

            // position lookup indexed by (id - minID); handles unordered IC files
            std::vector<float> lagByID(lagTotalParticles * NO_DIM);
            for (size_t i = 0; i < lagTotalParticles; ++i)
            {
                size_t slot = lagIDs[i] - minID;
                for (int d = 0; d < NO_DIM; ++d)
                    lagByID[slot * NO_DIM + d] = lagCoords[i * NO_DIM + d];
            }
            // the raw chunk arrays are dead once the by-ID table exists; free them before the
            // matching loop (20 B/particle transient, ~5 GB at TNG300-3 scale)
            std::vector<float>().swap(lagCoords);
            std::vector<uint64_t>().swap(lagIDs);

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
            // non-contiguous IDs: sort-based matching, O(N log N)
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

        // rescale Lagrangian positions to Mpc
        if (userOptions->MpcValue != Real(1.))
        {
            for (size_t i = 0; i < noParticles * NO_DIM; ++i)
                lagPosOut[i] /= userOptions->MpcValue;
        }

        readData._lagrangianPositionPopulated = true;
        message << "\t " << MESSAGE::cGreen() << "Lagrangian position matching complete." << MESSAGE::cReset() << "\n" << MESSAGE::Flush;
#else
        throwError( "The '--lagrangianInput' option requires HDF5 support. Please compile with -DHDF5." );
#endif
    }

    if ( not readData._lagrangianPositionPopulated )
        throwError( "PS-DTFE mode requires Lagrangian positions but none were loaded." );
#endif

    // store particles (and any user-given sampling coordinates) in the output lists
    readData.transferData( p, samplingCoordinates );
}


// Functions for writing the output data


// Dispatches output writing to the format chosen via the '--output' option.
class OutputData
{
    public:
    int outputFileType;

    OutputData(int fileType)
    {
        this->outputFileType = fileType;
    }

    template <typename T>
    void write(T &dataToWrite,
                std::string filename,
                std::string variableName,
                User_options const &userOptions)
    {
        if ( this->outputFileType==101 )
            writeBinaryFile( dataToWrite, filename, variableName, userOptions );        // binary file
        else if ( this->outputFileType==111 )
            writeTextFile( dataToWrite, filename, variableName, userOptions );          // text file
        else if ( this->outputFileType==112 )
            writeTextFile_gridIndex( dataToWrite, filename, variableName, userOptions );// text file, each line prefixed with grid-cell coordinates
        else if ( this->outputFileType==113 )
            writeTextFile_samplingPosition( dataToWrite, filename, variableName, userOptions ); // text file, each line prefixed with sampling-point coordinates
        else if ( this->outputFileType==114 )
            writeTextFile_redshiftConePosition( dataToWrite, filename, variableName, userOptions ); // text file, each line prefixed with redshift-cone coordinates
        else if ( this->outputFileType==100 )
            writeSpecialFile( dataToWrite, filename, variableName, userOptions );   // binary with self-describing header
    }
};


// Write the output data, each variable to a separate file.
void writeOutputData(Quantities &uQuantities,
                     Quantities &aQuantities,
                     User_options const &userOptions)
{
    OutputData output( userOptions.outputFileType );

    if ( userOptions.uField.density )
        output.write( uQuantities.density, userOptions.outputFilename + ".den", "density", userOptions );
    if ( userOptions.aField.density )
        output.write( aQuantities.density, userOptions.outputFilename + ".a_den", "volume averaged density", userOptions );

    if ( userOptions.uField.velocity )
        output.write( uQuantities.velocity, userOptions.outputFilename + ".vel", "velocity", userOptions );
    if ( userOptions.aField.velocity )
        output.write( aQuantities.velocity, userOptions.outputFilename + ".a_vel", "volume averaged velocity", userOptions );

    if ( userOptions.uField.velocity_gradient )
        output.write( uQuantities.velocity_gradient, userOptions.outputFilename + ".velGrad", "velocity gradient", userOptions );
    if ( userOptions.aField.velocity_gradient )
        output.write( aQuantities.velocity_gradient, userOptions.outputFilename + ".a_velGrad", "volume averaged velocity gradient", userOptions );

    if ( userOptions.uField.velocity_divergence )
        output.write( uQuantities.velocity_divergence, userOptions.outputFilename + ".velDiv", "velocity divergence", userOptions );
    if ( userOptions.aField.velocity_divergence )
        output.write( aQuantities.velocity_divergence, userOptions.outputFilename + ".a_velDiv", "volume averaged velocity divergence", userOptions );

    if ( userOptions.uField.velocity_shear )
        output.write( uQuantities.velocity_shear, userOptions.outputFilename + ".velShear", "velocity shear", userOptions );
    if ( userOptions.aField.velocity_shear )
        output.write( aQuantities.velocity_shear, userOptions.outputFilename + ".a_velShear", "volume averaged velocity shear", userOptions );

    if ( userOptions.uField.velocity_vorticity )
        output.write( uQuantities.velocity_vorticity, userOptions.outputFilename + ".velVort", "velocity vorticity", userOptions );
    if ( userOptions.aField.velocity_vorticity )
        output.write( aQuantities.velocity_vorticity, userOptions.outputFilename + ".a_velVort", "volume averaged velocity vorticity", userOptions );

    if ( userOptions.aField.velocity_std )
        output.write( aQuantities.velocity_std, userOptions.outputFilename + ".a_velStd", "volume averaged velocity standard deviation", userOptions );

    // PS-DTFE velocity dispersion: full tensor sigma_ij to '.velDispTensor', trace sigma^2 to '.velDisp' (mass-weighted, ~0 single-stream).
#ifdef PHASE_SPACE
    {
        // offset of diagonal element (i,i) in the upper-triangle row-major packing
        size_t diag[NO_DIM];
        for (int i=0, off=0; i<NO_DIM; off += (NO_DIM-i), ++i) diag[i] = size_t(off);
        auto writeDispersion = [&](std::vector< Pvector<Real,noDispComp> > &tens,
                                   std::string const &traceExt, std::string const &tensorExt,
                                   std::string const &traceDesc, std::string const &tensorDesc)
        {
            if ( tens.empty() ) return;
            std::vector<Real> trace( tens.size() );
            for (size_t c=0; c<tens.size(); ++c)
            {
                Real t = Real(0.);
                for (int i=0; i<NO_DIM; ++i) t += tens[c][ diag[i] ];
                trace[c] = t;
            }
            output.write( trace, userOptions.outputFilename + traceExt, traceDesc, userOptions );
            output.write( tens,  userOptions.outputFilename + tensorExt, tensorDesc, userOptions );
        };
        if ( userOptions.uField.velocity_dispersion )
            writeDispersion( uQuantities.velocity_dispersion, ".velDisp", ".velDispTensor",
                             "velocity dispersion (trace sigma^2)", "velocity dispersion tensor" );
        if ( userOptions.aField.velocity_dispersion )
            writeDispersion( aQuantities.velocity_dispersion, ".a_velDisp", ".a_velDispTensor",
                             "volume averaged velocity dispersion (trace sigma^2)", "volume averaged velocity dispersion tensor" );
    }
#endif

    if ( userOptions.uField.scalar )
        output.write( uQuantities.scalar, userOptions.outputFilename + ".scalar", "scalar", userOptions );
    if ( userOptions.aField.scalar )
        output.write( aQuantities.scalar, userOptions.outputFilename + ".a_scalar", "volume averaged scalar", userOptions );

    if ( userOptions.uField.scalar_gradient )
        output.write( uQuantities.scalar_gradient, userOptions.outputFilename + ".scalarGrad", "scalar gradient", userOptions );
    if ( userOptions.aField.scalar_gradient )
        output.write( aQuantities.scalar_gradient, userOptions.outputFilename + ".a_scalarGrad", "volume averaged scalar gradient", userOptions );

    // T-web classification and eigenvalues
    if ( userOptions.uField.velocity_tweb )
    {
        output.write( uQuantities.velocity_tweb, userOptions.outputFilename + ".tweb", "T-web classification", userOptions );
        output.write( uQuantities.velocity_tweb_eigenvalues, userOptions.outputFilename + ".twebEig", "T-web eigenvalues", userOptions );
    }
    if ( userOptions.aField.velocity_tweb )
    {
        output.write( aQuantities.velocity_tweb, userOptions.outputFilename + ".a_tweb", "volume averaged T-web classification", userOptions );
        output.write( aQuantities.velocity_tweb_eigenvalues, userOptions.outputFilename + ".a_twebEig", "volume averaged T-web eigenvalues", userOptions );
    }

    // V-web classification and eigenvalues
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

    // stream count (PS-DTFE only): per-cell for unaveraged fields, cell-averaged for '_a' fields.
#ifdef PHASE_SPACE
    if ( not uQuantities.stream_count.empty() )
        output.write( uQuantities.stream_count, userOptions.outputFilename + ".streams", "stream count", userOptions );
    if ( not aQuantities.stream_count.empty() )
        output.write( aQuantities.stream_count, userOptions.outputFilename + ".a_streams", "averaged stream count", userOptions );

    // arbitrary-point evaluation outputs ('<root>.pts_*'; no-op unless --sample-points was given)
    psPointEvalWriteOutputs( userOptions );
#endif
}
