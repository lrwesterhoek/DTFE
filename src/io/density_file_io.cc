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




/* Header struct stored at the start of a density output file, plus its read/write helpers. */

// how the density was computed
static size_t const DTFE_METHOD = 1;
static size_t const TSC_METHOD = 2;
static size_t const SPH_METHOD = 3;
static size_t const UNKNOWN_METHOD = -1;
// which field the file contains
static int const DENSITY_FILE = 1;
static int const VELOCITY_FILE = 11;
static int const VELOCITY_GRADIENT_FILE = 12;
static int const VELOCITY_DIVERGENCE_FILE = 13;
static int const VELOCITY_SHEAR_FILE = 14;
static int const VELOCITY_VORTICITY_FILE = 15;
static int const VELOCITY_STD_FILE = 16;
static int const SCALAR_FIELD_FILE = 20;
static int const SCALAR_FIELD_GRADIENT_FILE = 21;
static int const UNKNOWN_FILE = -1;


static int const fillSize = 1024 - 13*8 - 8*18 - 8*2;   // pads the header to 1024 bytes

// Fixed 1024-byte header written at the start of a density/field output file, describing the grid,
// the field type, the box, and the source snapshot's cosmology.
struct Density_header
{
    size_t  gridSize[3];    // density grid dimensions along the 3 directions
    size_t  totalGrid;      // total grid size = gridSize[0]*gridSize[1]*gridSize[2]
    int     fileType;       // field type stored: DENSITY_FILE, VELOCITY_FILE, etc.
    int     noDensityFiles; // number of density files for this run
    int     densityFileGrid[3];  // when split across files, grid dimensions of the patch partition
    int     indexDensityFile;    // this file's index in the partition (which region it holds)
    double  box[6];         // box coordinates of the density (xMin, xMax, yMin, yMax, zMin, zMax)


    // mirror of the gadget header: info about the input snapshot used to compute the density
    size_t   npartTotal[6];// total number of particles in the gadget simulation
    double   mass[6];      // particle masses in the N-body code
    double   time;         // expansion parameter 'a' of the snapshot
    double   redshift;     // corresponding redshift
    double   BoxSize;      // box size in kpc
    double   Omega0;       // Omega_matter
    double   OmegaLambda;  // Omega_Lambda
    double   HubbleParam;  // Hubble parameter h (H = 100 h km/s/Mpc)


    size_t  method;        // density method: DTFE_METHOD, TSC_METHOD or SPH_METHOD
    char    fill[fillSize];// pad to 1024 bytes; also stores the command line used to obtain the file
    size_t  FILE_ID;       // unique id for this file type


    // Initializes every field to 0 or to the not-assigned sentinel (-1).
    Density_header()
    {
        for (int i=0; i<3; ++i)
        {
            gridSize[i] = size_t(0);
            densityFileGrid[i] = 1;
        }
        totalGrid = size_t(0);
        fileType = UNKNOWN_FILE;
        noDensityFiles = 1;
        indexDensityFile = -1;
        for (int i=0; i<6; ++i)
        {
            box[i] = 0.;
            npartTotal[i] = 0;
        }
    
        time = -1.; redshift = -1.;
        BoxSize = -1.; Omega0 = -1.; OmegaLambda = -1.; HubbleParam = -1.;
    
        method = UNKNOWN_METHOD;
        FILE_ID = 1;
    }
    
    // Prints the header contents to stdout.
    void print()
    {
        std::string densityMethod = "unknown";
        if ( method==DTFE_METHOD ) densityMethod = "DTFE";
        else if ( method==TSC_METHOD ) densityMethod = "TSC";
        else if ( method==SPH_METHOD ) densityMethod = "SPH";
    
        std::string fileTypeName = "unknown file type";
        if ( fileType==DENSITY_FILE ) fileTypeName = "the file stores a density field";
        else if ( fileType==VELOCITY_FILE ) fileTypeName = "the file stores a velocity field";
        else if ( fileType==VELOCITY_GRADIENT_FILE ) fileTypeName = "the file stores the gradient of a velocity field";
        else if ( fileType==VELOCITY_DIVERGENCE_FILE ) fileTypeName = "the file stores a velocity divergence";
        else if ( fileType==VELOCITY_SHEAR_FILE ) fileTypeName = "the file stores a velocity shear";
        else if ( fileType==VELOCITY_VORTICITY_FILE ) fileTypeName = "the file stores a velocity vorticity";
        else if ( fileType==SCALAR_FIELD_FILE ) fileTypeName = "the file stores a scalar field";
        else if ( fileType==SCALAR_FIELD_GRADIENT_FILE ) fileTypeName = "the file stores the gradient of a scalar field";
    
    
        std::cout << "\nThe header of the density file contains the following info:\n" <<
            "1) Information about the actual density computations:\n"
            << "gridSize      = " << gridSize[0] << "  " << gridSize[1] << "  " << gridSize[2] << "\n"
            << "totalGrid     = " << totalGrid << "\n"
            << "file type     = " << fileTypeName << "\n"
            << "# density file= " << noDensityFiles << "\n";
        if ( noDensityFiles>1 )
            std::cout << "file grid size= " << densityFileGrid[0] << "  " << densityFileGrid[1] << "  " << densityFileGrid[2] << "\n"
                << "file index    = " << indexDensityFile << "\n";
        std::cout << "box coords    = " << box[0] << "  " << box[1] << "  " << box[2] << "  " << box[3] << "  " << box[4] << "  " << box[5] << "\n";
            
        std::cout << "\n2) Information about the snapshot file used to compute the density:\n"
            << "npartTotal[6] =  " << npartTotal[0] << "  " << npartTotal[1] << "  " << npartTotal[2] << "  " << npartTotal[3] << "  " << npartTotal[4] << "  " << npartTotal[5] << "\n"
            << "mass[6]       =  " << mass[0] << "  " << mass[1] << "  " << mass[2] << "  " << mass[3] << "  " << mass[4] << "  " << mass[5] << "\n"
            << "time          =  " << time << "\n"
            << "redshift      =  " << redshift << "\n"
            << "BoxSize       =  " << BoxSize << "\n"
            << "Omega0        =  " << Omega0 << "\n"
            << "OmegaLambda   =  " << OmegaLambda << "\n"
            << "HubbleParam   =  " << HubbleParam << "\n";
    
    
        std::cout << "\n3) Information about files and additional remarks:\n"
            << "method          = " << densityMethod << "\n"
            << "fill            = " << fill << "\n\n";
    }
    
    // Fills the header from the user options and the variable name (grid, box, method, file type).
    void updateDensityHeader(User_options userOptions, std::string variableName)
    {
        if ( userOptions.regionOn and not userOptions.regionMpcOn )
        {
            for (size_t i=0; i<userOptions.region.size(); ++i )
                userOptions.region[i] = userOptions.boxCoordinates[i%2] + userOptions.region[i] * (userOptions.boxCoordinates[i%2+1]-userOptions.boxCoordinates[i%2]);
            userOptions.regionMpcOn = true;
        }
        
        for (int i=0; i<NO_DIM; ++i)        // update the grid dimensions
            this->gridSize[i] = userOptions.gridSize[i];
        if ( userOptions.partNo>=0 )        // update the data coordinates if file was split
            for (int i=0; i<NO_DIM; ++i)
                this->densityFileGrid[i] = userOptions.partition[i]; 
        for (int i=0; i<2*NO_DIM; ++i)      // update the box coordinates
            this->box[i] = userOptions.regionOn ? userOptions.region[i] : userOptions.boxCoordinates[i];
        
#if NO_DIM==2
        this->gridSize[2] = 1;
        if ( userOptions.partNo>=0 )
            this->densityFileGrid[2] = 1;
        this->box[4] = 0.;
        this->box[5] = 1.;
#endif
        this->totalGrid = this->gridSize[0]*this->gridSize[1]*this->gridSize[2];
        
        // store the program options used to get the data in the fill block
        int commandLength = userOptions.programOptions.length();
        for (int i=0; i<userOptions.programOptions.length(); ++i)
            this->fill[i] = userOptions.programOptions[i];
        this->fill[ commandLength+0 ] = ' ';
        this->fill[ commandLength+1 ] = ';';
        this->fill[ commandLength+2 ] = ' ';
        this->fill[ commandLength+3 ] = ' ';
        for (int i=commandLength+4; i<fillSize; ++i)
            this->fill[i] = '\0';

        if ( userOptions.partNo>=0 )
            this->indexDensityFile = userOptions.partNo;
        if ( userOptions.DTFE ) this->method = DTFE_METHOD;
        else if ( userOptions.TSC ) this->method = TSC_METHOD;
        else if ( userOptions.SPH ) this->method = SPH_METHOD;

        // Derive the file type from keywords in the variable name (order matters: check specific before generic).
        if ( variableName.find( "density" )!=std::string::npos ) this->fileType = DENSITY_FILE;
        else if ( variableName.find( "velocity gradient" )!=std::string::npos ) this->fileType = VELOCITY_GRADIENT_FILE;
        else if ( variableName.find( "velocity divergence" )!=std::string::npos ) this->fileType = VELOCITY_DIVERGENCE_FILE;
        else if ( variableName.find( "velocity shear" )!=std::string::npos ) this->fileType = VELOCITY_SHEAR_FILE;
        else if ( variableName.find( "velocity vorticity" )!=std::string::npos ) this->fileType = VELOCITY_VORTICITY_FILE;
        else if ( variableName.find( "velocity standard deviation" )!=std::string::npos ) this->fileType = VELOCITY_STD_FILE;
        else if ( variableName.find( "velocity" )!=std::string::npos ) this->fileType = VELOCITY_FILE;
        else if ( variableName.find( "scalar" )!=std::string::npos ) this->fileType = SCALAR_FIELD_FILE;
        else if ( variableName.find( "scalar gradient" )!=std::string::npos ) this->fileType = SCALAR_FIELD_GRADIENT_FILE;
    }
    
    // Copies snapshot info (particle counts, masses, cosmology) from the input Gadget header into this header.
    void copyGadgetHeaderInfo(User_options userOptions)
    {
        Gadget_header gadgetHeader;
        std::string filename = gadgetHeader.filename( userOptions.inputFilename, 0, false );
        if ( not bfs::exists(filename) ) return;    // file could not be found

        if ( userOptions.inputFileType==101 or userOptions.inputFileType==102 )
        {
            std::fstream inputFile;
            openInputBinaryFile( inputFile, filename );

            // read the header
            int buffer, gadgetFileType;
            bool swapEndian = false;
            inputFile.read( reinterpret_cast<char *>(&buffer), sizeof(buffer) );
            bool validFile = gadgetHeader.detectSnapshotType( buffer, &gadgetFileType, &swapEndian );
            if ( not validFile )
                return;
            int offset = gadgetFileType==2 ? 16+sizeof(buffer) : 0+sizeof(buffer);
            inputFile.seekg( offset, std::ios::beg );
            inputFile.read( reinterpret_cast<char *>(&gadgetHeader), sizeof(gadgetHeader) );
            inputFile.close();
        }
        else if ( userOptions.inputFileType==105 )
            HDF5_readGadgetHeader( filename, &gadgetHeader );
        else
            return;
        
        // copy the info from the gadget header
        for (int i=0; i<6; ++i)
        {
            this->npartTotal[i] = gadgetHeader.npartTotal[i];
            this->mass[i] = gadgetHeader.mass[i];
        }
        this->time = gadgetHeader.time;
        this->redshift = gadgetHeader.redshift;
        this->BoxSize = gadgetHeader.BoxSize;
        this->Omega0 = gadgetHeader.Omega0;
        this->OmegaLambda = gadgetHeader.OmegaLambda;
        this->HubbleParam = gadgetHeader.HubbleParam;
    }
};





// Writes a field to a binary file prefixed by a Density_header that records how the data was produced.
template <typename T>
void writeSpecialFile(T const &dataToWrite,
                        std::string filename,
                        std::string variableName,
                        User_options const &userOptions)
{
    // update the density header information
    Density_header densityHeader;
    densityHeader.updateDensityHeader( userOptions, variableName );
    densityHeader.copyGadgetHeaderInfo( userOptions );
    
    
    MESSAGE::Message message( userOptions.verboseLevel );
    message << "Writing the " << variableName << " to the file '" << filename << "' ...  " << MESSAGE::Flush;

    std::fstream outputFile;
    openOutputBinaryFile( outputFile, filename );


    // write the header
    size_t buffer = sizeof( densityHeader );
    outputFile.write( reinterpret_cast<char *>(&buffer), sizeof(buffer) );
    outputFile.write( reinterpret_cast<char *>(&densityHeader), sizeof(densityHeader) );
    outputFile.write( reinterpret_cast<char *>(&buffer), sizeof(buffer) );

    // write the data, in blocks of at most 256^3 elements (a single huge write() can fail)
    buffer = dataToWrite.size()*sizeof(dataToWrite[0]);
    outputFile.write( reinterpret_cast<char *>(&buffer), sizeof(buffer) );
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
    outputFile.write( reinterpret_cast<char *>(&buffer), sizeof(buffer) );
    
    outputFile.close();
    message << "Done.\n";
}





