/*
 *  Copyright (c) 2013       Marius Cautun
 *
 *                           Institute for Computational Cosmology
 *                           Durham University, Durham, UK
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



/* Reads a Gadget snapshot together with the matching modified-gravity (MOG/GR) force file. */

// Reads the modified-gravity forces for one file into the scalar block (defined below).
void readForces_MOG(std::string fileName,
                    Read_data<Real> *readData,
                    User_options &userOptions,
                    size_t const noParticlesThisFile,
                    int const noScalars,
                    size_t *numberParticlesRead);



// Reads a Gadget snapshot (single or multiple files) and the GR/MOG forces from a separate file.
void readGadgetFile_MOG(std::string filename,
                        Read_data<Real> *readData,
                        User_options *userOptions)
{
    int gadgetFileType;     // gadget file format (1 or 2)
    int noBytesPos, noBytesVel;     // bytes per value for positions and velocities (4=float, 8=double)
    bool swapEndian;        // true if the data endianness must be swapped
    size_t noParticles;     // total number of particles to read
    Gadget_header gadgetHeader; // header of the first file


    // determine file type, endianness, particle/file counts and reserve memory for the data
    initializeGadget( filename, readData, userOptions, &gadgetHeader, &gadgetFileType, &noBytesPos, &noBytesVel, &swapEndian, &noParticles );


    MESSAGE::Message message( userOptions->verboseLevel );
    std::string fileName = filename;
    std::string fileNameMOG;


    // reserve memory for the scalar particle data
    int noScalars = atoi( userOptions->additionalOptions[1].c_str() );
    if ( noScalars>0  and  noScalars<=noScalarComp )
        readData->scalar( readData->noParticles() );  // particle scalar quantity
    else if ( noScalars>noScalarComp )
    {
        std::cout << "~~~ ERROR ~~~ The file required number of scalar components = " << noScalars << " is larger than the one specified when the program has been compiled = " << noScalarComp << "! The program ended unsucessfully!\n";
        exit( EXIT_FAILURE );
    }
    
    
    size_t numberParticlesRead  = 0;   // running particle count from the snapshot reader
    size_t numberParticlesRead2 = 0;   // running particle count from the force reader

    for (int i=0; i<gadgetHeader.num_files; ++i )
    {
        fileName = gadgetHeader.filename( filename, i );
        message << "Reading GADGET snapshot file '" << fileName << "' which is file " << i+1 << " of " << gadgetHeader.num_files << " files...\n" << MESSAGE::Flush;

        readGadgetData( fileName, readData, *userOptions, gadgetFileType, noBytesPos, noBytesVel, swapEndian, &numberParticlesRead );

        // read the matching forces for this file
        fileNameMOG = gadgetHeader.filename( userOptions->additionalOptions[0], i+1 );
        size_t noParticlesThisFile = numberParticlesRead - numberParticlesRead2;
        if ( noScalars > 0 )
            readForces_MOG( fileNameMOG, readData, *userOptions, noParticlesThisFile, noScalars, &numberParticlesRead2 );
    }


    // swap endianness of all read data if needed
    if ( swapEndian )
    {
        if ( userOptions->readParticleData[0] )
            ByteSwapArray( readData->position(), NO_DIM*noParticles );
        if ( userOptions->readParticleData[1] )
            ByteSwapArray( readData->weight(), noParticles );
#ifdef VELOCITY
        if ( userOptions->readParticleData[2] )
            ByteSwapArray( readData->velocity(), NO_DIM*noParticles );
#endif
#ifdef SCALAR
        if ( noScalars>0 )
            ByteSwapArray( readData->scalar(), NO_DIM*noParticles );
#endif
    }
    
    return;
}








// Reads the MOG/GR forces from a binary file into the scalar block.
void readForces_MOG(std::string fileName,
                    Read_data<Real> *readData,
                    User_options &userOptions,
                    size_t const noParticlesThisFile,
                    int const noScalars,
                    size_t *numberParticlesRead)
{
    MESSAGE::Message message( userOptions.verboseLevel );
    message << "\t reading force data from file '" << fileName << "' ... " << MESSAGE::Flush;

    std::fstream inputFile;
    openInputBinaryFile( inputFile, fileName );
    int buffer1, buffer2;


    // read the data block, bracketed by matching record-size integers
    inputFile.read( reinterpret_cast<char *>(&buffer1), sizeof(buffer1) );

    double *data = new double[ noParticlesThisFile * noScalars ];
    size_t readBytes = noParticlesThisFile * sizeof(double) * noScalars;
    inputFile.read( reinterpret_cast<char *>( data ), readBytes );

    inputFile.read( reinterpret_cast<char *>(&buffer2), sizeof(buffer2) );
    if ( buffer1!=buffer2 )
        throwError( "The integers before and after the particle 'MOG forces' data block in the GADGET file '" + fileName + "' did not match. The GADGET snapshot file is corrupt." );


    // copy into the scalar block (stride noScalarComp, only the first noScalars columns used)
    size_t alreadyRead = (*numberParticlesRead) * noScalarComp;
    Real *scalar = &( readData->scalar()[ alreadyRead ] );
    for (size_t i=0; i<noParticlesThisFile; ++i)
        for (int j=0; j<noScalars; ++j)
            scalar[ i*noScalarComp + j ] = data[ i*noScalars + j ];
    delete[] data;
    message << "Done.";

    inputFile.close();
    message << "\n";
    (*numberParticlesRead) += noParticlesThisFile;
}








