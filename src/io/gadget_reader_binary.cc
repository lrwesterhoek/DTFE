/*
 *  Copyright (c) 2021       Marius Cautun
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



/* Reader for binary Gadget snapshot files; which particles and properties are read is controlled by the program options. */




// Counts the particles per species across a multi-file snapshot (declaration; defined below).
void countGadgetParticleNumber(std::string filenameRoot,
                               int const noFiles,
                               int const gadgetFileType,
                               bool const swapEndian,
                               int const verboseLevel,
                               size_t numberTotalParticles[]);

// Reads one Gadget file's data, called once per file by readGadgetFile (declaration; defined below).
void readGadgetData(std::string fileName,
                    Read_data<Real> *readData,
                    User_options &userOptions,
                    int const gadgetFileType,
                    int const noBytesPos,
                    int const noBytesVel,
                    bool const swapEndian,
                    size_t *numberParticlesRead);




// Reads the Gadget header of one input file and sets everything needed to read the data:
// number of files, header type and endianness, box dimensions, particle count, and memory allocation.
template <typename T>
void initializeGadget(std::string filename,
                      Read_data<T> *readData,
                      User_options *userOptions,
                      Gadget_header *gadgetHeader,
                      int *gadgetFileType,
                      int *noBytesPos,
                      int *noBytesVel,
                      bool *swapEndian,
                      size_t *noParticles)
{
    MESSAGE::Message message( userOptions->verboseLevel );
    std::string fileName = filename;
    bool singleFile = true;
    if ( not bfs::exists(fileName) ) // a missing root file means the input is split across several files
    {
        fileName = gadgetHeader->filename( filename, 0 );
        singleFile = false;
    }


    std::fstream inputFile;
    openInputBinaryFile( inputFile, fileName );


    // detect the Gadget file format (type 1 or 2) and endianness
    int buffer1, buffer2, buffer3, buffer4;       // record-size integers bracketing each data block
    inputFile.read( reinterpret_cast<char *>(&buffer1), sizeof(buffer1) );
    bool validFile = gadgetHeader->detectSnapshotType( buffer1, gadgetFileType, swapEndian );
    if ( not validFile )
        throwError( "Unknown file type for the input Gadget snapshot. Tried Gadget snapshots type 1 and 2 as well as changing endianness, but none worked. Check that you inserted the correct input file." );
    if ( *swapEndian )
        message << "Detected that the input data file has a different endianness than the current system. The program will automatically change endianness for the data!" << MESSAGE::Flush;
    int offset = (*gadgetFileType)==2 ? 16 : 0;      // type 2 prefixes each block with a 16-byte label


    // read the gadget header
    inputFile.seekg( offset, std::ios::beg );
    inputFile.read( reinterpret_cast<char *>(&buffer1), sizeof(buffer1) );
    inputFile.read( reinterpret_cast<char *>(gadgetHeader), sizeof(*gadgetHeader) );
    inputFile.read( reinterpret_cast<char *>(&buffer2), sizeof(buffer2) );
    SWAP_HEADER_ENDIANNESS( *swapEndian, buffer1, buffer2, (*gadgetHeader) );

    // read the record sizes of the position and velocity blocks (to infer their data type)
    inputFile.seekg( offset, std::ios::cur );
    inputFile.read( reinterpret_cast<char *>(&buffer3), sizeof(buffer3) );
    inputFile.seekg( buffer3, std::ios::cur );
    inputFile.seekg( offset, std::ios::cur );
    inputFile.read( reinterpret_cast<char *>(&buffer4), sizeof(buffer4) );
    inputFile.read( reinterpret_cast<char *>(&buffer4), sizeof(buffer4) );
    inputFile.close();

    SWAP_HEADER_ENDIANNESS( *swapEndian, buffer1, buffer2, (*gadgetHeader) );
    if ( buffer1!=buffer2 or buffer1!=256 )
        throwError( "The was an error while reading the header of the GADGET snapshot file. The integers before and after the header do not match the value 256. The GADGET snapshot file is corrupt." );

    // bytes per real value = block size / (3 components * particle count)
    int thisNoParts = 0;
    for (int i=0; i<6; ++i)
        thisNoParts += gadgetHeader->npart[i];
    *noBytesPos = buffer3 / (3*thisNoParts);
    *noBytesVel = buffer4 / (3*thisNoParts);
    std::cout << "!!!! Number of bytes for position and velocity data: " << *noBytesPos << " and  " << *noBytesVel << ", respectively.\n" << std::flush;


    // set the box coordinates from the header unless the user supplied them
    if ( not userOptions->userGivenBoxCoordinates )
    {
        for (size_t i=0; i<NO_DIM; ++i)
        {
            userOptions->boxCoordinates[2*i] = 0.;                    // left edge of the full box
            userOptions->boxCoordinates[2*i+1] = gadgetHeader->BoxSize;// right edge of the full box
        }
    }
    else
        message << "The box coordinates were set by the user using the program options. The program will keep this values and will NOT use the box length information from the Gadget file!" << MESSAGE::Flush;

    // set HubbleParam from header if unset; used only for T-web/V-web normalization, so announce only then
    if ( userOptions->hubbleParam < Real(0.) && gadgetHeader->HubbleParam > 0. )
    {
        userOptions->hubbleParam = Real(gadgetHeader->HubbleParam);
        if ( userOptions->uField.velocity_tweb or userOptions->uField.velocity_vweb
          or userOptions->aField.velocity_tweb or userOptions->aField.velocity_vweb )
            message << "Using HubbleParam = " << userOptions->hubbleParam << " from file header for T-web/V-web normalization.\n" << MESSAGE::Flush;
    }
#ifdef WOJTEK
    if ( userOptions->additionalOptions.size()!=0 ) // if an option was inserted
        gadgetHeader->num_files = atoi( userOptions->additionalOptions[0].c_str() );
    gadgetHeader->print();
#endif


    // total number of particles across the file(s)
    size_t numberTotalParticles[6];
    if ( singleFile )
    {
        gadgetHeader->num_files = 1;
        for (int i=0; i<6; ++i)
            numberTotalParticles[i] = gadgetHeader->npart[i];
    }
    else
        countGadgetParticleNumber( filename, gadgetHeader->num_files, *gadgetFileType, *swapEndian, userOptions->verboseLevel, numberTotalParticles );

    // keep only the species the user requested
    *noParticles = 0;
    for (int i=0; i<6; ++i)
    {
        if ( not userOptions->readParticleSpecies[i] )
            numberTotalParticles[i] = 0;
        *noParticles += numberTotalParticles[i];
    }
    message << "Reading " << *noParticles << " particle data from the input file. These particles are made from the particle species: " << numberTotalParticles[0] << " + "  << numberTotalParticles[1] << " + "  << numberTotalParticles[2] << " + "  << numberTotalParticles[3] << " + "  << numberTotalParticles[4] << " + "  << numberTotalParticles[5] << " .\n" << MESSAGE::Flush;



    // allocate memory for the particle data
    message << "Allocating memory for: positions... " << MESSAGE::Flush;
    if ( userOptions->readParticleData[0] )
        readData->position( *noParticles );
    message << "weights... " << MESSAGE::Flush;
    if ( userOptions->readParticleData[1] )
        readData->weight( *noParticles );    // weights = particle masses
#ifdef VELOCITY
    message << "velocity... " << MESSAGE::Flush;
    if ( userOptions->readParticleData[2] )
        readData->velocity( *noParticles );
#endif
#ifdef SCALAR
    message << "scalars... " << MESSAGE::Flush;
    int noScalars = 0;
    for (size_t i=3; i<userOptions->readParticleData.size(); ++i)
        if ( userOptions->readParticleData[i] )
            noScalars += 1;
    if ( noScalars>0 )
        readData->scalar( *noParticles );
#endif
    message << "Done.\n";

    // check that the 'readParticleData' and 'readParticleSpecies' options make sense
    if ( not userOptions->readParticleData[0] )
        throwError( "The program needs the particle position information to be able to interpolate the fields on a grid. Please add '1' to the integer number giving the data blocks to be read from the input Gadget snapshot." );
    if ( not userOptions->readParticleData[1] )
    {
        MESSAGE::Warning warning( userOptions->verboseLevel );
        warning << "You selected not to read the Gadget particle masses. This means that the program will treat all particles as having the same weight (mass)." << MESSAGE::EndWarning;
    }
    if ( (*noParticles)<=0 )
        throwError( "Please select again the particle species that you would like to read from the Gadget file. There are no particles in the current selection!" );
}



// Reads a Gadget snapshot saved in a single or multiple files.
void readGadgetFile(std::string filename,
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


    size_t numberParticlesRead = 0;   // running particle count across files

    for (int i=0; i<gadgetHeader.num_files; ++i )
    {
        fileName = gadgetHeader.filename( filename, i );
        message << "Reading GADGET snapshot file '" << fileName << "' which is file " << i+1 << " of " << gadgetHeader.num_files << " files...\n" << MESSAGE::Flush;

        readGadgetData( fileName, readData, *userOptions, gadgetFileType, noBytesPos, noBytesVel, swapEndian, &numberParticlesRead );
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
    }

    return;
}




// Counts the particles in a Gadget snapshot split across multiple files.
void countGadgetParticleNumber(std::string filenameRoot,
                               int const noFiles,
                               int const gadgetFileType,
                               bool const swapEndian,
                               int const verboseLevel,
                               size_t numberTotalParticles[])
{
    for (int i=0; i<6; ++i)
        numberTotalParticles[i] = 0;
    int offset = gadgetFileType==2 ? 16 : 0;

    for (int i=0; i<noFiles; ++i)
    {
        Gadget_header gadgetHeader;
        std::string fileName = gadgetHeader.filename( filenameRoot, i );

        std::fstream inputFile;
        openInputBinaryFile( inputFile, fileName );

        // read the header
        int buffer1, buffer2;
        inputFile.seekg( offset, std::ios::beg );
        inputFile.read( reinterpret_cast<char *>(&buffer1), sizeof(buffer1) );
        inputFile.read( reinterpret_cast<char *>(&gadgetHeader), sizeof(gadgetHeader) );
        inputFile.read( reinterpret_cast<char *>(&buffer2), sizeof(buffer2) );
        inputFile.close();
        SWAP_HEADER_ENDIANNESS( swapEndian, buffer1, buffer2, gadgetHeader );
        if ( buffer1!=buffer2 or buffer1!=256 )
            throwError( "The was an error while reading the header of the GADGET snapshot file '" + fileName + "'. The integers before and after the header do not match the value 256. The GADGET snapshot file is corrupt." );

        for (int j=0; j<6; ++j)
            numberTotalParticles[j] += gadgetHeader.npart[j];
    }

    MESSAGE::Message message( verboseLevel );
    message << "The data is in " << noFiles << " files and contains the following number of particles: " << numberTotalParticles[0] << " + "  << numberTotalParticles[1] << " + "  << numberTotalParticles[2] << " + "  << numberTotalParticles[3] << " + "  << numberTotalParticles[4] << " + "  << numberTotalParticles[5] << " .\n" << MESSAGE::Flush;
}



// Reads the gadget data from a single file (called once per file when reading a multi-file snapshot).
void readGadgetData(std::string fileName,
                    Read_data<Real> *readData,
                    User_options &userOptions,
                    int const gadgetFileType,
                    int const noBytesPos,
                    int const noBytesVel,
                    bool const swapEndian,
                    size_t *numberParticlesRead)
{
    MESSAGE::Message message( userOptions.verboseLevel );
    int offset = gadgetFileType==2 ? 16 : 0;


    std::fstream inputFile;
    openInputBinaryFile( inputFile, fileName );


    // read the header
    int buffer1, buffer2;
    Gadget_header tempHeader;
    READ_DELIMETER;
    inputFile.read( reinterpret_cast<char *>(&tempHeader), sizeof(tempHeader) );
    DELIMETER_CONSISTANCY_CHECK("header");
    if ( buffer1!=256 )
        throwError( "The integers before and after the header do not match the value 256. The GADGET snapshot file is corrupt." );


    // read the position block
    READ_DELIMETER;
    if ( userOptions.readParticleData[0] )
    {
        Real *positions = readData->position();
        size_t dataOffset = (*numberParticlesRead) * NO_DIM;
        message << "\t reading positions of the particles... " << MESSAGE::Flush;

        for (int i=0; i<6; ++i)   // read each requested species, seeking past the rest
        {
            if ( tempHeader.npart[i]==0 )
                continue;
            else if ( not userOptions.readParticleSpecies[i] )
            {
                size_t skipBytes = tempHeader.npart[i] * noBytesPos * NO_DIM;
                inputFile.seekg( skipBytes, std::ios::cur );
                continue;
            }

            // read directly if file type matches Real, otherwise read and convert
            size_t readBytes = tempHeader.npart[i] * noBytesPos * NO_DIM;
            if ( sizeof(Real) == noBytesPos )
                inputFile.read( reinterpret_cast<char *>( &(positions[dataOffset]) ), readBytes );
            else if ( noBytesPos==4 )
            {
                float *temp = new float[ tempHeader.npart[i] * NO_DIM ];
                inputFile.read( reinterpret_cast<char *>( temp ), readBytes );
                for (int u=0; u<tempHeader.npart[i] * NO_DIM; ++u)  positions[ dataOffset + u ] = temp[u];
                delete[] temp;
            }
            else if ( noBytesPos==8 )
            {
                double *temp = new double[ tempHeader.npart[i] * NO_DIM ];
                inputFile.read( reinterpret_cast<char *>( temp ), readBytes );
                for (int u=0; u<tempHeader.npart[i] * NO_DIM; ++u)  positions[ dataOffset + u ] = temp[u];
                delete[] temp;
            }
            dataOffset += tempHeader.npart[i] * NO_DIM;
        }

        message << "Done.";
    }
    else    // skip the positions if not needed
    {
        message << "\n\t (skipping positions)" << MESSAGE::Flush;
        size_t skipBytes = 0;
        for (int i=0; i<6; ++i) skipBytes += tempHeader.npart[i] * noBytesPos * NO_DIM;
        inputFile.seekg( skipBytes, std::ios::cur );
    }
    DELIMETER_CONSISTANCY_CHECK("position");


    // read the velocities block
    READ_DELIMETER;
#ifdef VELOCITY
    if ( userOptions.readParticleData[2] )
    {
        Real *velocities = readData->velocity();
        size_t dataOffset = (*numberParticlesRead) * NO_DIM;
        message << "\n\t reading velocities of the particles... " << MESSAGE::Flush;

        for (int i=0; i<6; ++i)   // read each requested species, seeking past the rest
        {
            if ( tempHeader.npart[i]==0 )
                continue;
            else if ( not userOptions.readParticleSpecies[i] )
            {
                size_t skipBytes = tempHeader.npart[i] * noBytesVel * NO_DIM;
                inputFile.seekg( skipBytes, std::ios::cur );
                continue;
            }

            // read directly if file type matches Real, otherwise read and convert
            size_t readBytes = tempHeader.npart[i] * noBytesVel * NO_DIM;
            if ( sizeof(Real) == noBytesVel )
                inputFile.read( reinterpret_cast<char *>( &(velocities[dataOffset]) ), readBytes );
            else if ( noBytesVel==4 )
            {
                float *temp = new float[ tempHeader.npart[i] * NO_DIM ];
                inputFile.read( reinterpret_cast<char *>( temp ), readBytes );
                for (int u=0; u<tempHeader.npart[i] * NO_DIM; ++u)  velocities[ dataOffset + u ] = temp[u];
                delete[] temp;
            }
            else if ( noBytesVel==8 )
            {
                double *temp = new double[ tempHeader.npart[i] * NO_DIM ];
                inputFile.read( reinterpret_cast<char *>( temp ), readBytes );
                for (int u=0; u<tempHeader.npart[i] * NO_DIM; ++u)  velocities[ dataOffset + u ] = temp[u];
                delete[] temp;
            }
            dataOffset += tempHeader.npart[i] * NO_DIM;
        }
        message << "Done.";
    }
    else
    {   // skip the velocities if not needed
        message << "\n\t (skipping velocities)" << MESSAGE::Flush;
        size_t skipBytes = 0;
        for (int i=0; i<6; ++i) skipBytes += tempHeader.npart[i] * noBytesVel * NO_DIM;
        inputFile.seekg( skipBytes, std::ios::cur );
    }
#else
    message << "\n\t (skipping velocities)" << MESSAGE::Flush;
    size_t skipBytes = 0;
    for (int i=0; i<6; ++i)
        skipBytes += tempHeader.npart[i] * noBytesVel * NO_DIM;
    inputFile.seekg( skipBytes, std::ios::cur );
#endif
    DELIMETER_CONSISTANCY_CHECK("velocity");


    // skip the particle ID block
    message << "\n\t (skipping ids)" << MESSAGE::Flush;
    READ_DELIMETER;
    inputFile.seekg( buffer1, std::ios::cur );
    DELIMETER_CONSISTANCY_CHECK("id");


    // the mass block is present only if some present species has a per-particle mass (header mass==0)
    bool massBlockPresent = false;
    for (int i=0; i<6; ++i)
        if ( tempHeader.mass[i]==0. and tempHeader.npart[i]!=0 )
            massBlockPresent = true;

    if ( massBlockPresent )
    {
        READ_DELIMETER;
    }
    if ( userOptions.readParticleData[1] )
    {
        Real *weights = readData->weight();
        size_t dataOffset = (*numberParticlesRead);
        message << "\n\t reading masses of the particles... " << MESSAGE::Flush;

        for (int i=0; i<6; ++i)   // read each requested species, seeking past the rest
        {
            if ( tempHeader.npart[i]==0 )
                continue;
            else if ( tempHeader.mass[i]==0. and not userOptions.readParticleSpecies[i] )
            {
                size_t skipBytes = tempHeader.npart[i] * sizeof(float);
                inputFile.seekg( skipBytes, std::ios::cur );
                continue;
            }

            if ( tempHeader.mass[i]==0. )       // per-particle mass: read directly or convert
            {
                size_t readBytes = tempHeader.npart[i] * noBytesPos;
                if ( sizeof(Real) == noBytesPos )
                    inputFile.read( reinterpret_cast<char *>( &(weights[dataOffset]) ), readBytes );
                else if ( noBytesPos==4 )
                {
                    float *temp = new float[ tempHeader.npart[i] ];
                    inputFile.read( reinterpret_cast<char *>( temp ), readBytes );
                    for (int u=0; u<tempHeader.npart[i]; ++u)  weights[ dataOffset + u ] = temp[u];
                    delete[] temp;
                }
                else if ( noBytesPos==8 )
                {
                    double *temp = new double[ tempHeader.npart[i] ];
                    inputFile.read( reinterpret_cast<char *>( temp ), readBytes );
                    for (int u=0; u<tempHeader.npart[i]; ++u)  weights[ dataOffset + u ] = temp[u];
                    delete[] temp;
                }
            }
            else if ( userOptions.readParticleSpecies[i] ) // common mass for all particles of the species
            {
                float mass = tempHeader.mass[i];
                if ( swapEndian ) BYTESWAP(mass);       // match endianness of the rest of the data
                for (size_t j=dataOffset; j<dataOffset+tempHeader.npart[i]; ++j)
                    weights[j] = mass;
            }

            dataOffset += tempHeader.npart[i];
        }
        message << "Done.\n";
    }
    if ( massBlockPresent )
    {
        DELIMETER_CONSISTANCY_CHECK("mass");
    }


    // read the internal energy (gas particles only)
    size_t noScalarsRead = 0;
    if ( userOptions.readParticleData[3] and tempHeader.npart[0]>0)
    {
        READ_DELIMETER;
        Real *scalar = readData->scalar();
        Real *weights = readData->weight();
        size_t dataOffset = (*numberParticlesRead);
        message << "\t reading internal energy of the particles... " << MESSAGE::Flush;

        size_t readBytes = tempHeader.npart[0] * sizeof(float);
        float *tempData = new float[ tempHeader.npart[0] ];
        inputFile.read( reinterpret_cast<char *>( tempData ), readBytes );

        float mean = 0;
        for (size_t i=0; i<size_t(tempHeader.npart[0]); ++i)
        {
            size_t index1 = dataOffset + i;
            size_t index2 = index1 * NO_SCALARS + noScalarsRead;
            scalar[index2] = weights[index1] * tempData[i]; // U is per unit mass in Gadget, so multiply by mass
            mean += scalar[index2];
        }
        mean /= tempHeader.npart[0];

        delete[] tempData;
        noScalarsRead += 1;
        message << "\r\t reading internal energy of the particles (mean energy: " << mean << ")... Done.\n";
        DELIMETER_CONSISTANCY_CHECK("internal energy U");
    }

    inputFile.close();
    message << "\n";
    for (int i=0; i<6; ++i)
        if ( userOptions.readParticleSpecies[i] )
            (*numberParticlesRead) += tempHeader.npart[i];
}








