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
 
 
// Helper macros for reading Gadget blocks: swap endianness when needed, read the record-size
// delimiter, and verify the integers bracketing a block match (else the file is corrupt).
#define SWAP_HEADER_ENDIANNESS(x1,x2,x3,x4) { if( x1 ) {BYTESWAP( x2 ); BYTESWAP( x3 ); x4.swapBytes();} }
#define SWAP_ENDIANNESS(x1,x2,x3)           { if( x1 ) {BYTESWAP( x2 ); BYTESWAP( x3 );} }
#define READ_DELIMETER \
    inputFile.seekg( offset, std::ios::cur ); \
    inputFile.read( reinterpret_cast<char *>(&buffer1), sizeof(buffer1) )
#define DELIMETER_CONSISTANCY_CHECK(field) \
    inputFile.read( reinterpret_cast<char *>(&buffer2), sizeof(buffer2) ); \
    SWAP_ENDIANNESS( swapEndian, buffer1, buffer2 ); \
    if ( buffer1!=buffer2 ) \
        throwError( "The integers before and after the particle " field " data block in the GADGET file '" + fileName + "' did not match. The GADGET snapshot file is corrupt." )


// The 256-byte Gadget snapshot header (standard layout), with helpers for file naming and endianness.
struct Gadget_header
{
    int      npart[6];
    double   mass[6];
    double   time;
    double   redshift;
    int      flag_sfr;
    int      flag_feedback;
    int      npartTotal[6];
    int      flag_cooling;
    int      num_files;
    double   BoxSize;
    double   Omega0;
    double   OmegaLambda;
    double   HubbleParam;
    char     fill[256- 6*4- 6*8- 2*8- 2*4- 6*4- 2*4 - 4*8];  // fills to 256 Bytes


    // file name for snapshot file 'fileNumber'; fileRoot must contain a '%i' or '%s' for multiple files
    std::string filename(std::string fileRoot, int const fileNumber, bool checkFileExists=true )
    {
        char buf[500];
        snprintf( buf, sizeof(buf), fileRoot.c_str(), fileNumber );
        std::string fileName( buf );
        if ( not bfs::exists(fileName) and checkFileExists )
            throwError( "The program could not open the input GADGET snapshot file/files: '" + fileName + "'. It cannot find the file/files." );
        return fileName;
    }

    // Prints the Gadget header contents to stdout.
    void print()
    {
        std::cout << "\nThe header of the Gadget file contains the following info:\n"
            << "npart[6]     =  " << npart[0] << "  " << npart[1] << "  " << npart[2] << "  " << npart[3] << "  " <<  npart[4] << "  " <<  npart[5] << "\n"
            << "mass[6]      =  " << mass[0] << "  " << mass[1] << "  " << mass[2] << "  " << mass[3] << "  " << mass[4] << "  " << mass[5] << "\n"
            << "time         =  " << time << "\n"
            << "redshift     =  " << redshift << "\n"
            << "flag_sfr     =  " << flag_sfr << "\n"
            << "flag_feedback=  " << flag_feedback << "\n"
            << "npartTotal[6]=  " << npartTotal[0] << "  " << npartTotal[1] << "  " << npartTotal[2] << "  " << npartTotal[3] << "  " << npartTotal[4] << "  " << npartTotal[5] << "  " << "\n"
            << "flag_cooling =  " << flag_cooling << "\n"
            << "num_files    =  " << num_files << "\n"
            << "BoxSize      =  " << BoxSize << "\n"
            << "Omega0       =  " << Omega0 << "\n"
            << "OmegaLambda  =  " << OmegaLambda << "\n"
            << "h            =  " << HubbleParam << "\n\n";
    }

    // Swaps the endianness of every header field in place.
    void swapBytes()
    {
        ByteSwapArray( npart, 6 );
        ByteSwapArray( mass, 6 );
        BYTESWAP( time );
        BYTESWAP( redshift );
        BYTESWAP( flag_sfr );
        BYTESWAP( flag_feedback );
        ByteSwapArray( npartTotal, 6 );
        BYTESWAP( flag_cooling );
        BYTESWAP( num_files );
        BYTESWAP( BoxSize );
        BYTESWAP( Omega0 );
        BYTESWAP( OmegaLambda );
        BYTESWAP( HubbleParam );
    }

    // Detect Gadget file format (1 or 2) from the first record-size integer; sets swapEndian if the
    // value only matches after a byte swap. Returns false if neither format is recognized.
    bool detectSnapshotType(int const bufferValue,
                            int *gadgetFileType,
                            bool *swapEndian)
    {
        int buffer1 = bufferValue;
        *swapEndian = false;

        if ( buffer1 == 8 )             // format 2
            *gadgetFileType = 2;
        else if ( buffer1 == 256 )      // format 1
            *gadgetFileType = 1;
        else                            // retry with swapped endianness
        {
            BYTESWAP( buffer1 );
            *swapEndian = true;
            if ( buffer1 == 8 )
                *gadgetFileType = 2;
            else if ( buffer1 == 256 )
                *gadgetFileType = 1;
            else
                return false;
        }
        return true;
    }
};



// Post-header setup shared by the binary and HDF5 Gadget initializers: default the box
// coordinates from the header unless the user supplied them, and take HubbleParam from the
// header when unset (used only for T-web/V-web normalization, so announce only then).
void gadgetHeaderDefaults(Gadget_header *gadgetHeader,
                          User_options *userOptions,
                          MESSAGE::Message &message)
{
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
}
