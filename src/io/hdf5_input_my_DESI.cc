/*
 *  Copyright (c) 2019       Marius Cautun
 *
 *                           ICC, 
 *                           Durham University, UK
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


/* Reader for DESI HDF5 matter snapshots (single common particle mass, "Matter" group layout). */
#ifdef HDF5
#include <H5Cpp.h>
using namespace H5;



// File name for input file 'fileNumber'; fileRoot must contain a '%i' or '%s' for multiple files.
std::string HDF5_filename_DESI( std::string fileRoot, size_t fileNumber, bool checkFileExists=true )
    {
        char buf[500];
        snprintf( buf, sizeof(buf), fileRoot.c_str(), fileNumber );
        std::string fileName( buf );
        if ( not bfs::exists(fileName) and checkFileExists )
            throwError( "The program could not open the input GADGET snapshot file/files: '" + fileName + "'. It cannot find the file/files." );
        return fileName;
    }



// Reads selected header entries of the HDF5 file (particle number, mass, box length, number of files
// per snapshot), not the full header.
void HDF5_readHeader_DESI(std::string filename,
                           size_t *numParticles,
                           double *BoxSize,
                           size_t *numFiles,
                           double *particleMass)
{
    const H5std_string FILE_NAME( filename );

    H5File *file = new H5File( FILE_NAME, H5F_ACC_RDONLY, H5::FileAccPropList::DEFAULT );
    Group *group = new Group( file->openGroup("Header") );


    // read the header attributes one at a time
    std::string name( "NP.Matter" );
    if ( doesAttributeExist( group->getId(), name.c_str() ) )
        group->openAttribute( name.c_str() ).read( PredType::NATIVE_ULONG, numParticles );
    else throwError( "No '" + name + "' attribute found in the HDF5 file '" + filename + "'. Cannot continue with the program!" );
    
    name = "ParticleMass.Matter";
    if ( doesAttributeExist( group->getId(), name.c_str() ) )
        group->openAttribute( name.c_str() ).read( PredType::NATIVE_DOUBLE, particleMass );
    else throwError( "No '" + name + "' attribute found in the HDF5 file '" + filename + "'. Cannot continue with the program!" );
    
    name = "NumFilesPerSnapshot";
    if ( doesAttributeExist( group->getId(), name.c_str() ) )
        group->openAttribute( name.c_str() ).read( PredType::NATIVE_ULONG, numFiles );
    else throwError( "No '" + name + "' attribute found in the HDF5 file '" + filename + "'. Cannot continue with the program!" );
    
    name = "BoxSize";
    if ( doesAttributeExist( group->getId(), name.c_str() ) )
        group->openAttribute( name.c_str() ).read( PredType::NATIVE_DOUBLE, BoxSize );
    
    std::cout << "The code read the following data  (number particles, particle mass, number of files, box size):  " <<  *numParticles << "  " << *particleMass << "  " << *numFiles << "  " << *BoxSize << "\n";

    delete group;
    delete file;
}



// Reads the file header and initializes the corresponding userOptions values.
void HDF5_initialize_DESI(std::string filename,
                           Read_data<float> *readData,
                           User_options *userOptions,
                           size_t *noParticles,
                           size_t *numFiles)
{
    MESSAGE::Message message( userOptions->verboseLevel );
    std::string fileName = filename;
    if ( not bfs::exists(fileName) ) // a missing root file means the input is split across several files
    {
        fileName = HDF5_filename_DESI( filename, 0 );
    }


    double BoxSize;
    double particleMass;
    HDF5_readHeader_DESI( fileName, noParticles, &BoxSize, numFiles, &particleMass );


    // set the box coordinates from the header unless the user supplied them
    if ( not userOptions->userGivenBoxCoordinates )
    {
        for (size_t i=0; i<NO_DIM; ++i)
        {
            userOptions->boxCoordinates[2*i]   = 0.;                // left edge of the full box
            userOptions->boxCoordinates[2*i+1] = BoxSize;           // right edge of the full box
        }
    }
    else
        message << "The box coordinates were set by the user using the program options. The program will keep this values and will NOT use the box length information from the input file!" << MESSAGE::Flush;
    
    
    
    // allocate memory for the particle data
    if ( userOptions->readParticleData[0] )
        readData->position( *noParticles );
    if ( userOptions->readParticleData[1] )
    {
        readData->weight( *noParticles );    // all particles share the same mass from the header
        float *weights = readData->weight();
        for (size_t j=0; j<*noParticles; ++j)
            weights[j] = particleMass;
    }
#ifdef VELOCITY
    if ( userOptions->readParticleData[2] )
        readData->velocity( *noParticles );
#endif
#ifdef SCALAR
    int noScalars = 0;
    for (size_t i=3; i<userOptions->readParticleData.size(); ++i)
        if ( userOptions->readParticleData[i] )
            noScalars += 1;
    if ( noScalars>0 )
        readData->scalar( *noParticles );
#endif
    
    
    
    // check that the 'readParticleData' and 'readParticleSpecies' options make sense
    if ( not userOptions->readParticleData[0] )
        throwError( "The program needs the particle position information to be able to interpolate the fields on a grid. Please add '1' to the integer number giving the data blocks to be read from the input Gadget snapshot." );
    if ( not userOptions->readParticleData[1] )
    {
        MESSAGE::Warning warning( userOptions->verboseLevel );
        warning << "You selected not to read the particle masses. This means that the program will treat all particles as having the same weight (mass)." << MESSAGE::EndWarning;
    }
#ifdef SCALAR
    if ( noScalars>NO_SCALARS )
        throwError( "You asked for too many scalars to be read from the file using the input data option. Either ask for less scalar quantities or increase the number of scalar field components using the Makefile option '-DNO_SCALARS'. You asked for ", noScalars, " scalar components, but 'NO_SCALARS' is only ", NO_SCALARS, "." );
#endif
}



// Reads the DESI data from single or multiple HDF5 files.
void HDF5_readData_DESI(std::string filenameRoot,
                         Read_data<float> *readData,
                         User_options *userOptions)
{
    MESSAGE::Message message( userOptions->verboseLevel );

    size_t numParticles;
    size_t numFiles;
    HDF5_initialize_DESI( filenameRoot, readData, userOptions, &numParticles, &numFiles );


    size_t numParticlesRead = 0;
    size_t numParticlesRead2 = 0;
    for (size_t fileNumber=0; fileNumber<numFiles; ++fileNumber)
    {
        std::string filename = HDF5_filename_DESI( filenameRoot, fileNumber );
        message << "Reading file " << fileNumber << ": '" << filename << "'...\n";

        const H5std_string FILE_NAME( filename );
        H5File *file = new H5File( FILE_NAME, H5F_ACC_RDONLY, H5::FileAccPropList::DEFAULT );
        Group *group;
        group = new Group( file->openGroup("Matter") );


        // read the coordinates
        if ( userOptions->readParticleData[0] )
        {
            float *positions = readData->position();
            size_t dataOffset = numParticlesRead * NO_DIM;
            message << "\t reading the particles positions ... " << MESSAGE::Flush;

            DataSet dataset = group->openDataSet("Position");
            DataSpace dataspace = dataset.getSpace();
            hsize_t dims_out[2];
            dataspace.getSimpleExtentDims( dims_out, NULL );

            dataset.read( &(positions[dataOffset]), PredType::NATIVE_FLOAT );
            numParticlesRead2 = numParticlesRead + (unsigned long)(dims_out[0]);
            message << (unsigned long)(dims_out[0]) << " particles ... Done\n";
        }


        // read the velocities
        if ( userOptions->readParticleData[2] )
        {
            float *velocities = readData->velocity();
            size_t dataOffset = numParticlesRead * NO_DIM;
            message << "\t reading the particles velocities ... " << MESSAGE::Flush;

            DataSet dataset = group->openDataSet("Velocities");
            DataSpace dataspace = dataset.getSpace();
            hsize_t dims_out[2];
            dataspace.getSimpleExtentDims( dims_out, NULL );

            dataset.read( &(velocities[dataOffset]), PredType::NATIVE_FLOAT );
            numParticlesRead2 = numParticlesRead + (unsigned long)(dims_out[0]);
            message << (unsigned long)(dims_out[0]) << " particles ... Done\n";
        }


        numParticlesRead = numParticlesRead2;

        delete group;
        delete file;
    }
}



#endif
