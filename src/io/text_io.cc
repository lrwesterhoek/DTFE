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



/* This file implements functions for reading and writing the data from/to text files. */
#include <cmath>



// Readers for text input files.

// Example reader. Line 1 = particle count; line 2 = box; line 3+ = "posX posY posZ velX velY velZ
// weight scalar" per particle. Sized Read_data accessors allocate; all arrays must share the same count.
void readTextFile(std::string filename,
                  Read_data<Real> *readData,
                  User_options *userOptions)
{
    MESSAGE::Message message( userOptions->verboseLevel );
    message << "Reading the input data from the text file '" << filename << "' ... " << MESSAGE::Flush;
    std::fstream inputFile;
    openInputTextFile( inputFile, filename );


    // line 1: particle count; line 2: box coordinates
    size_t noParticles;
    inputFile >> noParticles;
    for (int i=0; i<2*NO_DIM; ++i)
        inputFile >> userOptions->boxCoordinates[i];


    // assumes each particle line is: posX, posY, posZ, velX, velY, velZ, weight(=mass), scalar(1 component)
    Real *positions = readData->position(noParticles);
    readData->velocity(noParticles);
    Real *weights = readData->weight(noParticles);    // weights = particle/galaxy masses
    readData->scalar(noParticles);


    for (int i=0; i<noParticles; ++i)
    {
        for (int j=0; j<NO_DIM; ++j)
            inputFile >> positions[NO_DIM*i+j];
        inputFile >> weights[i];
    }

    checkFileOperations( inputFile, "read from" );
    inputFile.close();
    message << "Done.\n";
}


// Reads only the particle positions from a text file that contains only the positions.
void readTextFile_positions(std::string filename,
                            Read_data<Real> *readData,
                            User_options *userOptions)
{
    MESSAGE::Message message( userOptions->verboseLevel );
    message << "Reading the particle position data from the text file '" << filename << "' ... " << MESSAGE::Flush;
    std::fstream inputFile;
    openInputTextFile( inputFile, filename );

    // line 1: particle count; line 2: box coordinates
    size_t noParticles;
    inputFile >> noParticles;
    for (int i=0; i<2*NO_DIM; ++i)
        inputFile >> userOptions->boxCoordinates[i];

    // assumes each particle line is: posX, posY, posZ
    Real *positions = readData->position(noParticles);

    for (int i=0; i<noParticles; ++i)
    {
        for (int j=0; j<NO_DIM; ++j)
            inputFile >> positions[NO_DIM*i+j];
    }

    checkFileOperations( inputFile, "read from" );
    inputFile.close();
    message << "Done.\n";
}



// Writers for text output files.

// Writes the field to a text file, one sampling point per line (one column per field component).
void writeTextFile(std::vector<Real> &dataToWrite,
                    std::string filename,
                    std::string variableName,
                    User_options const &userOptions)
{
    MESSAGE::Message message( userOptions.verboseLevel );
    message << "Writing the " << variableName << " to the text file '" << filename << "' ...  " << MESSAGE::Flush;

    std::fstream outputFile;
    openOutputTextFile( outputFile, filename );

    for (size_t i=0; i<dataToWrite.size(); ++i)
        outputFile << dataToWrite[i] << "\n";

    checkFileOperations( outputFile, "write to" );
    outputFile.close();
    message << "Done.\n";
}
// Vector-field overload: writes the N components of each sampling point as tab-separated columns.
template <size_t N>
void writeTextFile(std::vector< Pvector<Real,N> > &dataToWrite,
                    std::string filename,
                    std::string variableName,
                    User_options const &userOptions)
{
    MESSAGE::Message message( userOptions.verboseLevel );
    message << "Writing the " << variableName << " to the text file '" << filename << "' ...  " << MESSAGE::Flush;

    std::fstream outputFile;
    openOutputTextFile( outputFile, filename );

    for (size_t i=0; i<dataToWrite.size(); ++i)
    {
        for (size_t j=0; j<N; ++j)
            outputFile << dataToWrite[i][j] << "\t";
        outputFile << "\n";
    }

    checkFileOperations( outputFile, "write to" );
    outputFile.close();
    message << "Done.\n";
}



// Writes the field to a text file, one line per grid cell: grid indices followed by the field value
// (one column per component).
void writeTextFile_gridIndex(std::vector<Real> &dataToWrite,
                             std::string filename,
                             std::string variableName,
                             User_options const &userOptions)
{
    MESSAGE::Message message( userOptions.verboseLevel );
    message << "Writing the " << variableName << " to the text file '" << filename << "' ...  " << MESSAGE::Flush;
    
    if ( userOptions.userDefinedSampling )
        throwError( "You cannot use the function 'writeTextFile_gridIndex' to write the data to a text file when using user defined coordinates since there are no grid indices associated to each sampling point." );


    std::fstream outputFile;
    openOutputTextFile( outputFile, filename );


    size_t const *grid = &(userOptions.gridSize[0]);
    size_t totalGrid = 1;
    for (int d=0; d<NO_DIM; ++d) totalGrid *= grid[d];
    for (size_t flatIdx=0; flatIdx<totalGrid; ++flatIdx)
    {
        size_t gridIdx[NO_DIM], rem = flatIdx;
        for (int d=NO_DIM-1; d>=0; --d) { gridIdx[d] = rem % grid[d]; rem /= grid[d]; }
        for (int d=0; d<NO_DIM; ++d) outputFile << gridIdx[d] << "\t";
        outputFile << dataToWrite[flatIdx] << "\n";
    }

    checkFileOperations( outputFile, "write to" );
    outputFile.close();
    message << "Done.\n";
}
// Vector-field overload: grid indices followed by the N components of the cell value, per line.
template <typename T, size_t N>
void writeTextFile_gridIndex(std::vector< Pvector<T,N> > &dataToWrite,
                             std::string filename,
                             std::string variableName,
                             User_options const &userOptions)
{
    MESSAGE::Message message( userOptions.verboseLevel );
    message << "Writing the " << variableName << " to the text file '" << filename << "' ...  " << MESSAGE::Flush;
    
    if ( userOptions.userDefinedSampling )
        throwError( "You cannot use the function 'writeTextFile_gridIndex' to write the data to a text file when using user defined coordinates since there are no grid indices associated to each sampling point." );


    std::fstream outputFile;
    openOutputTextFile( outputFile, filename );


    size_t const *grid = &(userOptions.gridSize[0]);
    size_t totalGrid = 1;
    for (int d=0; d<NO_DIM; ++d) totalGrid *= grid[d];
    for (size_t flatIdx=0; flatIdx<totalGrid; ++flatIdx)
    {
        size_t gridIdx[NO_DIM], rem = flatIdx;
        for (int d=NO_DIM-1; d>=0; --d) { gridIdx[d] = rem % grid[d]; rem /= grid[d]; }
        for (int d=0; d<NO_DIM; ++d) outputFile << gridIdx[d] << "\t";
        for (size_t i1=0; i1<N; ++i1)
            outputFile << dataToWrite[flatIdx][i1] << "\t";
        outputFile << "\n";
    }

    checkFileOperations( outputFile, "write to" );
    outputFile.close();
    message << "Done.\n";
}



// Writes the field to a text file, one line per grid cell: sampling-point coordinates followed by the
// field value (one column per component). Regular rectangular grids only.
void writeTextFile_samplingPosition(std::vector<Real> &dataToWrite,
                                    std::string filename,
                                    std::string variableName,
                                    User_options const &userOptions)
{
    MESSAGE::Message message( userOptions.verboseLevel );
    message << "Writing the " << variableName << " to the text file '" << filename << "' ...  " << MESSAGE::Flush;
    
    if ( userOptions.userDefinedSampling or userOptions.redshiftConeOn )
        throwError( "You cannot use the function 'writeTextFile_samplingPosition' to write the data to a text file when using redshift cone or user defined coordinates since the sampling coordinates are incorrect in this case." );


    std::fstream outputFile;
    openOutputTextFile( outputFile, filename );


    size_t const *grid = &(userOptions.gridSize[0]);
    Box boxCoordinates = userOptions.region;    // boundaries of the box the fields were interpolated on
    Real dx[NO_DIM];
    for (size_t i=0; i<NO_DIM; ++i) dx[i] = (boxCoordinates[2*i+1]-boxCoordinates[2*i]) / grid[i];
    
    size_t totalGrid = 1;
    for (int d=0; d<NO_DIM; ++d) totalGrid *= grid[d];
    for (size_t flatIdx=0; flatIdx<totalGrid; ++flatIdx)
    {
        size_t gridIdx[NO_DIM], rem = flatIdx;
        for (int d=NO_DIM-1; d>=0; --d) { gridIdx[d] = rem % grid[d]; rem /= grid[d]; }
        for (int d=0; d<NO_DIM; ++d)
            outputFile << (boxCoordinates[2*d] + dx[d] * (gridIdx[d]+0.5)) << "\t";
        outputFile << dataToWrite[flatIdx] << "\n";
    }

    checkFileOperations( outputFile, "write to" );
    outputFile.close();
    message << "Done.\n";
}
// Vector-field overload: cell-center coordinates followed by the N components of the value, per line.
template <typename T, size_t N>
void writeTextFile_samplingPosition(std::vector< Pvector<T,N> > &dataToWrite,
                                    std::string filename,
                                    std::string variableName,
                                    User_options const &userOptions)
{
    MESSAGE::Message message( userOptions.verboseLevel );
    message << "Writing the " << variableName << " to the text file '" << filename << "' ...  " << MESSAGE::Flush;
    
    if ( userOptions.userDefinedSampling or userOptions.redshiftConeOn )
        throwError( "You cannot use the function 'writeTextFile_samplingPosition' to write the data to a text file when using redshift cone or user defined coordinates since the sampling coordinates are incorrect in this case." );


    std::fstream outputFile;
    openOutputTextFile( outputFile, filename );


    size_t const *grid = &(userOptions.gridSize[0]);
    Box boxCoordinates = userOptions.region;    // boundaries of the box the fields were interpolated on
    Real dx[NO_DIM];
    for (size_t i=0; i<NO_DIM; ++i) dx[i] = (boxCoordinates[2*i+1]-boxCoordinates[2*i]) / grid[i];
    
    size_t totalGrid = 1;
    for (int d=0; d<NO_DIM; ++d) totalGrid *= grid[d];
    for (size_t flatIdx=0; flatIdx<totalGrid; ++flatIdx)
    {
        size_t gridIdx[NO_DIM], rem = flatIdx;
        for (int d=NO_DIM-1; d>=0; --d) { gridIdx[d] = rem % grid[d]; rem /= grid[d]; }
        for (int d=0; d<NO_DIM; ++d)
            outputFile << (boxCoordinates[2*d] + dx[d] * (gridIdx[d]+0.5)) << "\t";
        for (size_t i1=0; i1<N; ++i1)
            outputFile << dataToWrite[flatIdx][i1] << "\t";
        outputFile << "\n";
    }
    
    checkFileOperations( outputFile, "write to" );
    outputFile.close();
    message << "Done.\n";
}




// Writes the field to a text file, one line per cell: redshift-cone coordinates (toggle between
// (r,theta,phi) and (x,y,z) via the commented-out lines below) followed by the field value.
void writeTextFile_redshiftConePosition(std::vector<Real> &dataToWrite,
                                        std::string filename,
                                        std::string variableName,
                                        User_options const &userOptions)
{
    MESSAGE::Message message( userOptions.verboseLevel );
    message << "Writing the " << variableName << " to the text file '" << filename << "' ...  " << MESSAGE::Flush;

    if ( userOptions.userDefinedSampling )
        throwError( "You cannot use the function 'writeTextFile_redshiftConePosition' to write the data to a text file when using user defined coordinates since the program doesn't know how to compute the redshift cone coordinates." );
    if ( not userOptions.redshiftConeOn )
        throwError( "The function 'writeTextFile_redshiftConePosition' can be used to write the data to a text file only when using redshift cone coordinates since it writes the redshift cone coordinates in the file too." );

    std::fstream outputFile;
    openOutputTextFile( outputFile, filename );


    size_t const *grid = &(userOptions.gridSize[0]);
    Box coneCoordinates = userOptions.redshiftCone;    // redshift cone coordinates
    std::vector<Real> origin = userOptions.originPosition;// origin of the redshift cone
    Real dx[NO_DIM];
    for (size_t i=0; i<NO_DIM; ++i) dx[i] = (coneCoordinates[2*i+1]-coneCoordinates[2*i]) / grid[i];
    Real factor = 3.14/180.; // degrees to radians

#if NO_DIM==2
    for (size_t i=0; i<grid[0]; ++i)
        for (size_t j=0; j<grid[1]; ++j)
        {
            size_t index = i*grid[1] + j;
            Real r = coneCoordinates[0] + dx[0] * (i+0.5);
            Real theta = coneCoordinates[2] + dx[1] * (j+0.5);
            Real x = origin[0] + r*cos(theta*factor);
            Real y = origin[1] + r*sin(theta*factor);

            // write either (r, theta) or (x, y)
            outputFile << r << "\t" << theta << "\t";
//             outputFile << x << "\t" << y << "\t";
            outputFile << dataToWrite[index] << "\n";
        }
#elif NO_DIM==3
    for (size_t i=0; i<grid[0]; ++i)
        for (size_t j=0; j<grid[1]; ++j)
            for(size_t k=0; k<grid[2]; ++k)
            {
                size_t index = i*grid[1]*grid[2] + j*grid[2] + k;
                Real r = coneCoordinates[0] + dx[0] * (i+0.5);
                Real theta = coneCoordinates[2] + dx[1] * (j+0.5);
                Real phi = coneCoordinates[4] + dx[2] * (k+0.5);
                Real x = origin[0] + r*sin(theta*factor)*cos(phi*factor);
                Real y = origin[1] + r*sin(theta*factor)*sin(phi*factor);
                Real z = origin[2] + r*cos(theta*factor);

                // write either (r, theta, phi) or (x, y, z)
//                 outputFile << r << "\t" << theta << "\t" << phi << "\t";
                outputFile << x << "\t" << y << "\t" << z << "\t";
                outputFile << dataToWrite[index] << "\n";
            }
#endif

    checkFileOperations( outputFile, "write to" );
    outputFile.close();
    message << "Done.\n";
}
// Vector-field overload: redshift-cone coordinates followed by the N components of the value, per line.
template <typename T, size_t N>
void writeTextFile_redshiftConePosition(std::vector< Pvector<T,N> > &dataToWrite,
                                        std::string filename,
                                        std::string variableName,
                                        User_options const &userOptions)
{
    MESSAGE::Message message( userOptions.verboseLevel );
    message << "Writing the " << variableName << " to the text file '" << filename << "' ...  " << MESSAGE::Flush;

    if ( userOptions.userDefinedSampling )
        throwError( "You cannot use the function 'writeTextFile_redshiftConePosition' to write the data to a text file when using user defined coordinates since the program doesn't know how to compute the redshift cone coordinates." );
    if ( not userOptions.redshiftConeOn )
        throwError( "The function 'writeTextFile_redshiftConePosition' can be used to write the data to a text file only when using redshift cone coordinates since it writes the redshift cone coordinates in the file too." );

    std::fstream outputFile;
    openOutputTextFile( outputFile, filename );


    size_t const *grid = &(userOptions.gridSize[0]);
    Box coneCoordinates = userOptions.redshiftCone;    // redshift cone coordinates
    std::vector<Real> origin = userOptions.originPosition;// origin of the redshift cone
    Real dx[NO_DIM];
    for (size_t i=0; i<NO_DIM; ++i) dx[i] = (coneCoordinates[2*i+1]-coneCoordinates[2*i]) / grid[i];
    Real factor = 3.14/180.; // degrees to radians

#if NO_DIM==2
    for (size_t i=0; i<grid[0]; ++i)
        for (size_t j=0; j<grid[1]; ++j)
        {
            size_t index = i*grid[1] + j;
            Real r = coneCoordinates[0] + dx[0] * (i+0.5);
            Real theta = coneCoordinates[2] + dx[1] * (j+0.5);
            Real x = origin[0] + r*cos(theta*factor);
            Real y = origin[1] + r*sin(theta*factor);

            // write either (r, theta) or (x, y)
//             outputFile << r << "\t" << theta << "\t";
            outputFile << x << "\t" << y << "\t";
            for (size_t i1=0; i1<N; ++i1)
                outputFile << dataToWrite[index][i1] << "\t";
            outputFile << "\n";
        }
#elif NO_DIM==3
    for (size_t i=0; i<grid[0]; ++i)
        for (size_t j=0; j<grid[1]; ++j)
            for(size_t k=0; k<grid[2]; ++k)
            {
                size_t index = i*grid[1]*grid[2] + j*grid[2] + k;
                Real r = coneCoordinates[0] + dx[0] * (i+0.5);
                Real theta = coneCoordinates[2] + dx[1] * (j+0.5);
                Real phi = coneCoordinates[4] + dx[2] * (k+0.5);
                Real x = origin[0] + r*sin(theta*factor)*cos(phi*factor);
                Real y = origin[1] + r*sin(theta*factor)*sin(phi*factor);
                Real z = origin[2] + r*cos(theta*factor);

                // write either (r, theta, phi) or (x, y, z)
//                 outputFile << r << "\t" << theta << "\t" << phi << "\t";
                outputFile << x << "\t" << y << "\t" << z << "\t";
                for (size_t i1=0; i1<N; ++i1)
                    outputFile << dataToWrite[index][i1] << "\t";
                outputFile << "\n";
            }
#endif

    checkFileOperations( outputFile, "write to" );
    outputFile.close();
    message << "Done.\n";
}
