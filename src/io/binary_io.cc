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



/* This file contains functions for reading and writing the data to a binary file. */


// Reads input from a binary file: int noParticles; 6 floats box (xMin,xMax,...,zMax);
// positions; weights; velocities. All values single-precision float.
void readBinaryFile(std::string filename,
                    Read_data<Real> *readData,
                    User_options *userOptions)
{
    MESSAGE::Message message( userOptions->verboseLevel );
    message << "Reading the input data from the binary file '" << filename << "' ... " << MESSAGE::Flush;

    std::fstream inputFile;
    openInputBinaryFile( inputFile, filename );


    // read the number of particles and the box coordinates (the whole 2*NO_DIM-float box,
    // not just its first value -- a short read here also misaligned every block after it)
    int noParticles;
    float boxCoordinates[2*NO_DIM];
    inputFile.read( reinterpret_cast<char *>(&noParticles), sizeof(noParticles) );
    inputFile.read( reinterpret_cast<char *>(boxCoordinates), sizeof(float) * 2*NO_DIM );
    for (size_t i=0; i<2*NO_DIM; ++i)
        userOptions->boxCoordinates[i] = boxCoordinates[i];


    // the file stores float32, so the raw reads below require a single-precision build (Real == float)
    Real *positions  = readData->position(noParticles);
    Real *weights    = readData->weight(noParticles);    // weights = particle masses
    Real *velocities = readData->velocity(noParticles);


    size_t dataSize = noParticles * sizeof(float) * NO_DIM;
    inputFile.read( reinterpret_cast<char *>(positions), dataSize );

    dataSize = noParticles * sizeof(float);
    inputFile.read( reinterpret_cast<char *>(weights), dataSize );

    dataSize = noParticles * sizeof(float) * NO_DIM;
    inputFile.read( reinterpret_cast<char *>(velocities), dataSize );

    checkFileOperations( inputFile, "read from" );
    inputFile.close();
    
    message << "Done.\n";
}




// Reads input from a binary file: int noParticles; then for each particle N float
// values X, Y, Z, d1, ..., dN-3 (single precision), stored particle by particle.
void readBinaryFile_StructuredData(std::string filename,
                                   Read_data<Real> *readData,
                                   User_options *userOptions)
{
    MESSAGE::Message message( userOptions->verboseLevel );
    if ( userOptions->boxCoordinates.isNullBox() )
        throwError( "The coordinates of the data box have not been specified. The function 'readBinaryFile_StructuredData' needs the box coordinates to be given as an option to the program since they are not stored in the input file!" );
    

    // --options extras: [0] mass column (-1=uniform), [1] values per point (min 3), [2] threshold column (<0=off), [3] threshold (keep >=).
    int massColumn = -1;    // default: don't read any mass
    if ( userOptions->additionalOptions.size()!=0 )
        massColumn = atoi( userOptions->additionalOptions[0].c_str() );
    bool readMasses = massColumn>=0 ? true : false;

    int noElements = 6;     // default value
    if ( userOptions->additionalOptions.size()>=2 )
        noElements = atoi( userOptions->additionalOptions[1].c_str() );

    int thresholdColumn = -1;   // default: do not use a threshold
    Real threshold = 0.;
    if ( userOptions->additionalOptions.size()>=4 )
    {
        thresholdColumn = atoi( userOptions->additionalOptions[2].c_str() );
        threshold = atof( userOptions->additionalOptions[3].c_str() );
        message << "Selecting only the points that have the values in column " << thresholdColumn << " larger or equal than the threshold value " << threshold << " ... " << MESSAGE::Flush;
    }


    message << "Reading the input data from the binary file '" << filename << "' ... " << MESSAGE::Flush;
    std::fstream inputFile;
    openInputBinaryFile( inputFile, filename );


    int noParticles;
    inputFile.read( reinterpret_cast<char *>(&noParticles), sizeof(noParticles) );


    // read the full data structure
    size_t dataSize = noParticles * noElements;
    float *data = new float[dataSize];
    inputFile.read( reinterpret_cast<char *>(data), dataSize * sizeof(float) );
    inputFile.close();
    message << "Done.\n";
    
    
    // if a threshold is specified, find how many points pass this threshold
    size_t noFinalPoints = 0;
    if ( thresholdColumn>=0 )
    {
        for (int i=0; i<noParticles; ++i)
            if ( data[i*noElements+thresholdColumn]>=threshold )
                ++noFinalPoints;
        message << "After applying the threshold, the program found " << noFinalPoints << " (" << std::setprecision(4) << noFinalPoints*100./noParticles << "%) valid points that will be used for any further computations.\n" << MESSAGE::Flush;
    }
    else
        noFinalPoints = noParticles;
    
    
    // copy the relevant information to the '*readData' structure
    Real *positions = readData->position( noFinalPoints );
    Real *weights;
    if ( readMasses )
        weights     = readData->weight( noFinalPoints );    // weights = particle masses
    size_t counter = 0;
    for (int i=0; i<noParticles; ++i)
    {
        if ( thresholdColumn>=0 and data[i*noElements+thresholdColumn]<threshold )  // skip points below threshold
            continue;
        
        for (int j=0; j<NO_DIM; ++j)
            positions[counter*NO_DIM+j] = data[i*noElements+j];
        if ( readMasses )
            weights[counter] = data[i*noElements+massColumn];
        ++counter;
    }
}




// Writes a container of values to a raw binary file (no header).
template <typename T>
void writeBinaryFile(T const &dataToWrite,
                     std::string filename,
                     std::string variableName,
                     User_options const &userOptions)
{
    MESSAGE::Message message( userOptions.verboseLevel );
    message << "Writing the " << MESSAGE::cYellowD() << variableName << MESSAGE::cReset()
            << " to the binary file '" << MESSAGE::cBlue() << filename << MESSAGE::cReset() << "' ...  " << MESSAGE::Flush;

    std::fstream outputFile;
    openOutputBinaryFile( outputFile, filename );


    // write in blocks of at most 256^3 elements; a single huge write() can fail on large data sets
    size_t maxSize = 256*256*256;
    size_t noRepeats = size_t( dataToWrite.size() / maxSize ), currentPosition = 0;
    size_t tempBuffer = maxSize * sizeof(dataToWrite[0]);
    for (size_t i=0; i<noRepeats; ++i)
    {
        outputFile.write( reinterpret_cast<char const *>(&(dataToWrite[currentPosition])), tempBuffer );
        currentPosition += maxSize;
    }
    tempBuffer = (dataToWrite.size() - currentPosition) * sizeof(dataToWrite[0]);    // write the remainder
    outputFile.write( reinterpret_cast<char const *>(&(dataToWrite[currentPosition])), tempBuffer );

    checkFileOperations( outputFile, "write to" );
    outputFile.close();

    message << MESSAGE::cGreen() << "Done." << MESSAGE::cReset() << "\n";
}


