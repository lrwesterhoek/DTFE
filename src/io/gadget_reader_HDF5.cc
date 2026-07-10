/*
 *  Copyright (c) 2013       Marius Cautun
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




/* Readers for Gadget snapshots stored in HDF5 (standard and HI-mass variants). */
#ifdef HDF5
#include <H5Cpp.h>
using namespace H5;


// Returns true if the HDF5 object 'obj_id' has an attribute named 'name'.
extern "C"
{
    bool doesAttributeExist(hid_t obj_id, const char* name)
    {
        return( H5Aexists( obj_id, name ) > 0 ? true : false );
    }
}




// Reads selected entries of the Gadget header from an HDF5 file (particle number, mass array,
// box length, number of files per snapshot, cosmology), not the full header.
void HDF5_readGadgetHeader(std::string filename,
                           Gadget_header *gadgetHeader)
{
    const H5std_string FILE_NAME( filename );

    H5File *file = new H5File( FILE_NAME, H5F_ACC_RDONLY );
    Group *group = new Group( file->openGroup("/Header") );


    // read the header attributes one at a time
    std::string name( "NumPart_ThisFile" );
    if ( doesAttributeExist( group->getId(), name.c_str() ) )
        group->openAttribute( name.c_str() ).read( PredType::NATIVE_INT, gadgetHeader->npart );
    else throwError( "No '" + name + "' attribute found in the HDF5 file '" + filename + "'. Cannot continue with the program!" );
    
    name = "MassTable";
    if ( doesAttributeExist( group->getId(), name.c_str() ) )
        group->openAttribute( name.c_str() ).read( PredType::NATIVE_DOUBLE, gadgetHeader->mass );
    else throwError( "No '" + name + "' attribute found in the HDF5 file '" + filename + "'. Cannot continue with the program!" );
    
    name = "NumFilesPerSnapshot";
    if ( doesAttributeExist( group->getId(), name.c_str() ) )
        group->openAttribute( name.c_str() ).read( PredType::NATIVE_INT, &(gadgetHeader->num_files) );
    else throwError( "No '" + name + "' attribute found in the HDF5 file '" + filename + "'. Cannot continue with the program!" );
    
    name = "BoxSize";
    if ( doesAttributeExist( group->getId(), name.c_str() ) )
        group->openAttribute( name.c_str() ).read( PredType::NATIVE_DOUBLE, &(gadgetHeader->BoxSize) );
    
    name = "NumPart_Total";
    if ( doesAttributeExist( group->getId(), name.c_str() ) )
        group->openAttribute( name.c_str() ).read( PredType::NATIVE_UINT, gadgetHeader->npartTotal );
    
    name = "Redshift";
    if ( doesAttributeExist( group->getId(), name.c_str() ) )
        group->openAttribute( name.c_str() ).read( PredType::NATIVE_DOUBLE, &(gadgetHeader->redshift) );
    
    name = "Time";
    if ( doesAttributeExist( group->getId(), name.c_str() ) )
        group->openAttribute( name.c_str() ).read( PredType::NATIVE_DOUBLE, &(gadgetHeader->time) );
    name = "Time_GYR";
    if ( doesAttributeExist( group->getId(), name.c_str() ) )
        group->openAttribute( name.c_str() ).read( PredType::NATIVE_DOUBLE, &(gadgetHeader->time) );
    
    name = "Omega0";
    if ( doesAttributeExist( group->getId(), name.c_str() ) )
        group->openAttribute( name.c_str() ).read( PredType::NATIVE_DOUBLE, &(gadgetHeader->Omega0) );
    
    name = "OmegaLambda";
    if ( doesAttributeExist( group->getId(), name.c_str() ) )
        group->openAttribute( name.c_str() ).read( PredType::NATIVE_DOUBLE, &(gadgetHeader->OmegaLambda) );
    
    name = "HubbleParam";
    if ( doesAttributeExist( group->getId(), name.c_str() ) )
        group->openAttribute( name.c_str() ).read( PredType::NATIVE_DOUBLE, &(gadgetHeader->HubbleParam) );


    delete group;
    delete file;
}



// Reads the Gadget particle data from a single HDF5 file (one of possibly several per snapshot).
void HDF5_readGadgetData(std::string filename,
                         Read_data<float> *readData,
                         User_options &userOptions,
                         int const fileIndex,
                         size_t *numberParticlesRead)
{
    MESSAGE::Message message( userOptions.verboseLevel );

    Gadget_header gadgetHeader;
    HDF5_readGadgetHeader( filename, &gadgetHeader );

    // keep only the species of interest
    for (int i=0; i<6; ++i)
        if ( not userOptions.readParticleSpecies[i] )
            gadgetHeader.npart[i] = 0;


    const H5std_string FILE_NAME( filename );
    H5File *file = new H5File( FILE_NAME, H5F_ACC_RDONLY );
    Group *group;


    // read the coordinates
    if ( userOptions.readParticleData[0] )
    {
        float *positions = readData->position();
        size_t dataOffset = (*numberParticlesRead) * NO_DIM;   // offset (in floats) where new data starts
        message << "\t reading the particles positions ... " << MESSAGE::Flush;
        for(int type=0; type<6; type++)
        {
            if ( gadgetHeader.npart[type]<=0 ) continue;
            char buf[500];
            snprintf( buf, sizeof(buf), "/PartType%d", type );
            group = new Group( file->openGroup(buf) );

            DataSet dataset = group->openDataSet("Coordinates");

            dataset.read( &(positions[dataOffset]), PredType::NATIVE_FLOAT );
            delete group;

            dataOffset += gadgetHeader.npart[type] * NO_DIM;
        }
        message << MESSAGE::cGreen() << "Done" << MESSAGE::cReset() << "\n";
    }


    // read the masses (or weights) if different
    if ( userOptions.readParticleData[1] )
    {
        float *weights = readData->weight();
        size_t dataOffset = (*numberParticlesRead);
        message << "\t reading the particles masses ... " << MESSAGE::Flush;
        for(int type=0; type<6; type++)
        {
            if ( gadgetHeader.npart[type]<=0 ) continue;
            if ( gadgetHeader.mass[type]!=0. )          // common mass given in header
            {
                float mass = gadgetHeader.mass[type];
                for (size_t j=dataOffset; j<dataOffset+gadgetHeader.npart[type]; ++j)
                    weights[j] = mass;
            }
            else                                        // per-particle masses in a dataset
            {
                char buf[500];
                snprintf( buf, sizeof(buf),  "/PartType%d", type );
                group = new Group( file->openGroup(buf) );

                DataSet dataset = group->openDataSet("Mass");

                dataset.read( &(weights[dataOffset]), PredType::NATIVE_FLOAT );
                delete group;
            }
            dataOffset += gadgetHeader.npart[type];
        }
        message << MESSAGE::cGreen() << "Done" << MESSAGE::cReset() << "\n";
    }


    // read the velocities
    if ( userOptions.readParticleData[2] )
    {
        float *velocities = readData->velocity();
        size_t dataOffset = (*numberParticlesRead) * NO_DIM;
        message << "\t reading the particles velocities ... " << MESSAGE::Flush;
        for(int type=0; type<6; type++)
        {
            if ( gadgetHeader.npart[type]<=0 ) continue;
            char buf[500];
            snprintf( buf, sizeof(buf), "/PartType%d", type );
            group = new Group( file->openGroup(buf) );

            DataSet dataset = group->openDataSet("Velocities");

            dataset.read( &(velocities[dataOffset]), PredType::NATIVE_FLOAT );
            delete group;

            dataOffset += gadgetHeader.npart[type] * NO_DIM;
        }
        message << MESSAGE::cGreen() << "Done" << MESSAGE::cReset() << "\n";
    }



#ifdef PHASE_SPACE
    // particle IDs are needed to match particles against a separate Lagrangian file
    {
        size_t dataOffset = *numberParticlesRead;
        message << "\t reading ParticleIDs ... " << MESSAGE::Flush;
        for(int type=0; type<6; type++)
        {
            if ( gadgetHeader.npart[type]<=0 ) continue;
            char buf[500];
            snprintf( buf, sizeof(buf), "/PartType%d", type );
            group = new Group( file->openGroup(buf) );
            DataSet dataset = group->openDataSet("ParticleIDs");
            dataset.read( &(readData->_particleIDs[dataOffset]), PredType::NATIVE_UINT64 );
            delete group;
            dataOffset += gadgetHeader.npart[type];
        }
        message << MESSAGE::cGreen() << "Done" << MESSAGE::cReset() << "\n";
    }

    // InitialCoordinates may be absent if Lagrangian positions come from a separate --lagrangianInput file
    if ( readData->_lagrangianPosition._assigned )
    {
        float *lagPositions = readData->lagrangianPosition();
        size_t dataOffset = (*numberParticlesRead) * NO_DIM;
        bool success = true;
        for(int type=0; type<6; type++)
        {
            if ( gadgetHeader.npart[type]<=0 ) continue;
            char buf[500];
            snprintf( buf, sizeof(buf), "/PartType%d", type );
            group = new Group( file->openGroup(buf) );

            if ( H5Lexists(group->getId(), "InitialCoordinates", H5P_DEFAULT) <= 0 )
            {
                delete group;
                message << "\t No 'InitialCoordinates' dataset found (will use --lagrangianInput).\n" << MESSAGE::Flush;
                success = false;
                break;
            }

            DataSet dataset = group->openDataSet("InitialCoordinates");
            dataset.read( &(lagPositions[dataOffset]), PredType::NATIVE_FLOAT );
            delete group;
            dataOffset += gadgetHeader.npart[type] * NO_DIM;
        }
        if (success)
        {
            message << "\t reading the Lagrangian positions ... " << MESSAGE::cGreen() << "Done" << MESSAGE::cReset() << "\n" << MESSAGE::Flush;
            readData->_lagrangianPositionPopulated = true;
        }
    }
#endif

    int noScalarsRead = 0;
    // read the gas temperature (gas particles only)
    if ( userOptions.readParticleData[3] and gadgetHeader.npart[0]>0 )
    {
        message << "\t reading the gas temperature ..." << MESSAGE::Flush;
        group = new Group( file->openGroup( "/PartType0" ) );
        DataSet dataset = group->openDataSet("Temperature");
        float *tempData = new float[ gadgetHeader.npart[0] ];
        dataset.read( tempData, PredType::NATIVE_FLOAT );
        delete group;

        // store the mass-weighted temperature as a scalar field
        float *scalar = readData->scalar();
        float *weights = readData->weight();
        size_t dataOffset = (*numberParticlesRead);
        for (size_t i=0; i<size_t(gadgetHeader.npart[0]); ++i)
        {
            size_t index1 = dataOffset + i;
            size_t index2 = index1 * NO_SCALARS + noScalarsRead;
            scalar[index2] = weights[index1] * tempData[i];
        }

        delete[] tempData;
        noScalarsRead += 1;
        message << MESSAGE::cGreen() << "Done" << MESSAGE::cReset() << "\n";
    }


    delete file;
    for (int i=0; i<6; ++i)
        (*numberParticlesRead) += gadgetHeader.npart[i];
}






// Counts the particles in a Gadget snapshot split across multiple HDF5 files.
void HDF5_countGadgetParticleNumber(std::string filenameRoot,
                                    int const noFiles,
                                    int const verboseLevel,
                                    size_t numberTotalParticles[])
{
    for (int i=0; i<6; ++i)
        numberTotalParticles[i] = 0;

    for (int i=0; i<noFiles; ++i)
    {
        Gadget_header gadgetHeader;
        std::string fileName = gadgetHeader.filename( filenameRoot, i );

        HDF5_readGadgetHeader( fileName, &gadgetHeader );

        for (int j=0; j<6; ++j)
            numberTotalParticles[j] += gadgetHeader.npart[j];
    }
    
    MESSAGE::Message message( verboseLevel );
    message << "The data is in " << MESSAGE::cMagenta() << noFiles << MESSAGE::cReset()
            << " files and contains the following number of particles: " << MESSAGE::cMagenta()
            << numberTotalParticles[0] << " + "  << numberTotalParticles[1] << " + "  << numberTotalParticles[2] << " + "  << numberTotalParticles[3] << " + "  << numberTotalParticles[4] << " + "  << numberTotalParticles[5]
            << MESSAGE::cReset() << " .\n" << MESSAGE::Flush;
}




// Reads the Gadget header for one snapshot and initializes the corresponding userOptions values.
void HDF5_initializeGadget(std::string filename,
                           Read_data<float> *readData,
                           User_options *userOptions,
                           Gadget_header *gadgetHeader,
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


    HDF5_readGadgetHeader( fileName, gadgetHeader );


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
        HDF5_countGadgetParticleNumber( filename, gadgetHeader->num_files, userOptions->verboseLevel, numberTotalParticles );

    // keep only the species the user requested
    *noParticles = 0;
    for (int i=0; i<6; ++i)
    {
        if ( not userOptions->readParticleSpecies[i] )
            numberTotalParticles[i] = 0;
        *noParticles += numberTotalParticles[i];
    }
    message << "Reading " << MESSAGE::cMagenta() << *noParticles << MESSAGE::cReset()
            << " particle data from the input file. These particles are made from the particle species: "
            << MESSAGE::cMagenta() << numberTotalParticles[0] << " + "  << numberTotalParticles[1] << " + "  << numberTotalParticles[2] << " + "  << numberTotalParticles[3] << " + "  << numberTotalParticles[4] << " + "  << numberTotalParticles[5]
            << MESSAGE::cReset() << " .\n" << MESSAGE::Flush;
    
    
    
    // allocate memory for the particle data
    if ( userOptions->readParticleData[0] )
        readData->position( *noParticles );
    if ( userOptions->readParticleData[1] )
        readData->weight( *noParticles );    // weights = particle masses
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
#ifdef PHASE_SPACE
    readData->lagrangianPosition( *noParticles );
    readData->_particleIDs.resize( *noParticles );  // for ID-based Lagrangian matching
#endif



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
#ifdef SCALAR
    if ( noScalars>NO_SCALARS )
        throwError( "You asked for too many scalars to be read from the file using the input data option. Either ask for less scalar quantities or increase the number of scalar field components using the Makefile option '-DNO_SCALARS'. You asked for ", noScalars, " scalar components, but 'NO_SCALARS' is only ", NO_SCALARS, "." );
#endif
}



// Reads the Gadget data from single or multiple HDF5 files.
void HDF5_readGadgetFile(std::string filename,
                         Read_data<float> *readData,
                         User_options *userOptions)
{
    MESSAGE::Message message( userOptions->verboseLevel );

    Gadget_header gadgetHeader;
    size_t noParticles = 0;
    HDF5_initializeGadget( filename, readData, userOptions, &gadgetHeader, &noParticles );


    size_t numberParticlesRead = 0;
    std::string fileName;
    for (int i=0; i<gadgetHeader.num_files; ++i)
    {
        fileName = gadgetHeader.filename( filename, i );
        message << "Reading GADGET snapshot file '" << MESSAGE::cBlue() << fileName << MESSAGE::cReset()
                << "' which is file " << i+1 << " of " << gadgetHeader.num_files << " files...\n" << MESSAGE::Flush;

        HDF5_readGadgetData( fileName, readData, *userOptions, i, &numberParticlesRead );
    }
}







// Reads HI Gadget data from 2 HDF5 files: the first gives the gas content, the second the HI fraction.
// Reads one of possibly several files per snapshot.
void HDF5_readGadgetData_HI(std::string filename,
                            std::string h1FileName,
                            Read_data<float> *readData,
                            User_options &userOptions,
                            int const fileIndex,
                            size_t *numberParticlesRead)
{
    MESSAGE::Message message( userOptions.verboseLevel );

    Gadget_header gadgetHeader;
    HDF5_readGadgetHeader( filename, &gadgetHeader );

    // keep only the species of interest
    for (int i=0; i<6; ++i)
        if ( not userOptions.readParticleSpecies[i] )
            gadgetHeader.npart[i] = 0;


    H5File *file = new H5File( filename.c_str(), H5F_ACC_RDONLY );   // gas data
    Group *group;
    H5File *file2 = new H5File( h1FileName.c_str(), H5F_ACC_RDONLY ); // HI fraction


    // read the coordinates
    if ( userOptions.readParticleData[0] )
    {
        float *positions = readData->position();
        size_t dataOffset = (*numberParticlesRead) * NO_DIM;
        message << "\t reading the particles positions ... " << MESSAGE::Flush;
        for(int type=0; type<6; type++)
        {
            if ( gadgetHeader.npart[type]<=0 ) continue;
            char buf[500];
            snprintf( buf, sizeof(buf),  "/PartType%d", type );
            group = new Group( file->openGroup(buf) );

            DataSet dataset = group->openDataSet("Coordinates");

            dataset.read( &(positions[dataOffset]), PredType::NATIVE_FLOAT );
            delete group;

            dataOffset += gadgetHeader.npart[type] * NO_DIM;
        }
        message << MESSAGE::cGreen() << "Done" << MESSAGE::cReset() << "\n";
    }


    // read the masses (or weights) if different
    if ( userOptions.readParticleData[1] )
    {
        float *weights = readData->weight();
        size_t dataOffset = (*numberParticlesRead);
        message << "\t reading the particles masses ... " << MESSAGE::Flush;
        for(int type=0; type<6; type++)
        {
            if ( gadgetHeader.npart[type]<=0 ) continue;
            if ( gadgetHeader.mass[type]!=0. )          // common mass given in header
            {
                float mass = gadgetHeader.mass[type];
                for (size_t j=dataOffset; j<dataOffset+gadgetHeader.npart[type]; ++j)
                    weights[j] = mass;
            }
            else                                        // per-particle masses in a dataset
            {
                char buf[500];
                snprintf( buf, sizeof(buf), "/PartType%d", type );
                group = new Group( file->openGroup(buf) );

                DataSet dataset = group->openDataSet("Mass");

                dataset.read( &(weights[dataOffset]), PredType::NATIVE_FLOAT );
                delete group;
            }
            dataOffset += gadgetHeader.npart[type];
        }
        message << MESSAGE::cGreen() << "Done" << MESSAGE::cReset() << "\n";
    }


    // convert the gas mass to HI mass using the HI fraction (gas particles only)
    if ( gadgetHeader.npart[0]>0 )
    {
        message << "\t reading the HI fraction ... " << MESSAGE::Flush;

        group = new Group( file->openGroup( "/PartType0/ElementAbundance" ) );
        DataSet dataset = group->openDataSet("Hydrogen");
        float *hydrogenMassFraction = new float[ gadgetHeader.npart[0] ];
        dataset.read( hydrogenMassFraction, PredType::NATIVE_FLOAT );
        delete group;
        
        group = new Group( file2->openGroup( "/PartType0" ) );
        DataSet datasetMolecular = group->openDataSet("MolecularHydrogenMassFraction");
        DataSet datasetHydrogen = group->openDataSet("HydrogenOneFraction");
        
        float *molecularMassFraction = new float[ gadgetHeader.npart[0] ];
        datasetMolecular.read( molecularMassFraction, PredType::NATIVE_FLOAT );
        float *hydrogenOneFraction   = new float[ gadgetHeader.npart[0] ];
        datasetHydrogen.read( hydrogenOneFraction, PredType::NATIVE_FLOAT );
        delete group;


        // HI mass = gas mass * hydrogen fraction * atomic (non-molecular) fraction * HI fraction
        float *weights = readData->weight();
        size_t dataOffset = (*numberParticlesRead);
        for (size_t i=0; i<size_t(gadgetHeader.npart[0]); ++i)
        {
            size_t index = dataOffset + i;
            weights[index] *= hydrogenMassFraction[i] * (float(1.)-molecularMassFraction[i]) * hydrogenOneFraction[i];
            weights[index] = weights[index]<float(0.) ? float(0.) : weights[index];
        }
        
        delete[] hydrogenMassFraction;
        delete[] molecularMassFraction;
        delete[] hydrogenOneFraction;
        message << MESSAGE::cGreen() << "Done" << MESSAGE::cReset() << "\n";
    }
    
    
    // read the velocities
    if ( userOptions.readParticleData[2] )
    {
        float *velocities = readData->velocity();
        size_t dataOffset = (*numberParticlesRead) * NO_DIM;
        message << "\t reading the particles velocities ... " << MESSAGE::Flush;
        for(int type=0; type<6; type++)
        {
            if ( gadgetHeader.npart[type]<=0 ) continue;
            char buf[500];
            snprintf( buf, sizeof(buf), "/PartType%d", type );
            group = new Group( file->openGroup(buf) );

            DataSet dataset = group->openDataSet("Velocities");

            dataset.read( &(velocities[dataOffset]), PredType::NATIVE_FLOAT );
            delete group;

            dataOffset += gadgetHeader.npart[type] * NO_DIM;
        }
        message << MESSAGE::cGreen() << "Done" << MESSAGE::cReset() << "\n";
    }


    int noScalarsRead = 0;
    // read the gas temperature (gas particles only)
    if ( userOptions.readParticleData[3] and gadgetHeader.npart[0]>0 )
    {
        message << "\t reading the gas temperature ..." << MESSAGE::Flush;
        group = new Group( file->openGroup( "/PartType0" ) );
        DataSet dataset = group->openDataSet("Temperature");
        float *tempData = new float[ gadgetHeader.npart[0] ];
        dataset.read( tempData, PredType::NATIVE_FLOAT );
        delete group;

        // store the mass-weighted temperature as a scalar field
        float *scalar = readData->scalar();
        float *weights = readData->weight();
        size_t dataOffset = (*numberParticlesRead);
        for (size_t i=0; i<size_t(gadgetHeader.npart[0]); ++i)
        {
            size_t index1 = dataOffset + i;
            size_t index2 = index1 * NO_SCALARS + noScalarsRead;
            scalar[index2] = weights[index1] * tempData[i];
        }

        delete[] tempData;
        noScalarsRead += 1;
        message << MESSAGE::cGreen() << "Done" << MESSAGE::cReset() << "\n";
    }


    delete file;
    delete file2;
    for (int i=0; i<6; ++i)
        (*numberParticlesRead) += gadgetHeader.npart[i];
}



// Reads HI data from single or multiple HDF5 files; reads the HI mass, not the gas mass.
void HDF5_readGadgetFile_HI(std::string filename,
                            Read_data<float> *readData,
                            User_options *userOptions)
{
    MESSAGE::Message message( userOptions->verboseLevel );
    message << "Reading the HI mass fraction from an HDF5 file ...\n" << MESSAGE::Flush;

    Gadget_header gadgetHeader;
    size_t noParticles = 0;
    HDF5_initializeGadget( filename, readData, userOptions, &gadgetHeader, &noParticles );

    // HI content is defined only for gas, so reject any other requested species
    bool otherSpecies = false;
    for (int i=1; i<6; ++i)
        if ( userOptions->readParticleSpecies[1] )
            otherSpecies = true;
    if ( otherSpecies )
        throwError( "You asked for the function that reads in the HI data yet you also asked for reading other particle species. This does not make sense." );
    if ( userOptions->additionalOptions.empty() )
        throwError( "You need to give the name of the HDF5 files giving the Urchin HI fraction if you want to compute the HI mass distribution. This is inserted using the '--options' program option." );


    size_t numberParticlesRead = 0;
    std::string fileName, h1FractionFile;
    for (int i=0; i<gadgetHeader.num_files; ++i)
    {
        fileName = gadgetHeader.filename( filename, i );
        h1FractionFile = gadgetHeader.filename( userOptions->additionalOptions[0], i );
        message << "Reading GADGET snapshot file '" << fileName << "' and the HI fraction file '" << h1FractionFile << "' which are files " << i+1 << " of " << gadgetHeader.num_files << " files ...\n" << MESSAGE::Flush;
        
        HDF5_readGadgetData_HI( fileName, h1FractionFile, readData, *userOptions, i, &numberParticlesRead );
    }
}






#endif





