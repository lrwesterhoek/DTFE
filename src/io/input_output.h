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



/* Defines the classes used for reading data from an input file, plus 4 functions to open
binary/text files for input/output. */

#include <algorithm>


// Owning pointer + allocated flag for one Read_data array; frees its buffer on destruction.
template <typename T>
struct PairPtrBool
{
    T *_ptr;
    bool _assigned;

    PairPtrBool()
    {
        _ptr = NULL;
        _assigned = false;
    }
    ~PairPtrBool()  // frees the buffer if it was allocated
    {
        if ( _assigned ) delete[] _ptr;
    }

    // Allocates 'size' items of 'dimensions' components each; enforces (via *expectedSize) that all
    // Read_data arrays share the same item count.
    T* assignMemory(size_t const size,
                    size_t *expectedSize,
                    size_t dimensions,
                    std::string name)
    {
        if ( _assigned ) throwError( "When allocating memory for the variable 'Read_data::", name, "'. The memory was already allocated for the given variable." );
        if ( size!=(*expectedSize) and (*expectedSize)!=size_t(0) )
        {
            MESSAGE::Error error;
            error << "When allocating memory for the variable 'Read_data::" << name << "'. The size of the variable (which is " << size << ") is different from the expected size of " << *expectedSize << " (all variables: position, velocity, weight and scalar must have the same size)." << MESSAGE::EndError;
        }
        *expectedSize = size;
        _ptr = new T[dimensions*size];
        _assigned = true;
        return _ptr;
    }

    // Returns the data pointer; throws if no memory was allocated.
    T* returnPointer(std::string name)
    {
        if ( not _assigned ) throwError( "When returning pointer for the variable 'Read_data::", name, "'. There was no memory allocation for the given variable." );
        return _ptr;
    }

    // Frees the buffer early (transferData releases each flat array right after repacking it).
    void release()
    {
        if ( _assigned )
        {
            delete[] _ptr;
            _ptr = NULL;
            _assigned = false;
        }
    }
};



// Holds the raw particle and sampling-grid arrays read from disk, then hands them to the solver.
template <typename T>
struct Read_data
{
    size_t _noParticles; // number of particles
    PairPtrBool<T>  _position;   // position data
    PairPtrBool<T>  _velocity;   // velocity data
    PairPtrBool<T>  _weight;     // weight data
    PairPtrBool<T>  _scalar;     // scalar data
#ifdef PHASE_SPACE
    PairPtrBool<T>  _lagrangianPosition; // Lagrangian (initial) position data
    std::vector<uint64_t> _particleIDs;  // particle IDs for ID-based Lagrangian matching
    bool _lagrangianPositionPopulated;   // true if lag positions were actually read (not just allocated)
#endif

    size_t _noSamples;   // number of user-given sample points (if any)
    PairPtrBool<T>  _sampling;   // user-given sampling grid points
    PairPtrBool<T>  _delta;      // cell sizes of the user-given sampling grid


    Read_data()
    {
        _noParticles = 0;
        _noSamples = 0;
#ifdef PHASE_SPACE
        _lagrangianPositionPopulated = false;
#endif
    }
    
    
    size_t noParticles() { return _noParticles;}
    size_t noSamples() { return _noSamples;}

    // no-arg overloads return the pointer (memory must already be allocated); the sized overloads
    // allocate the memory and return the pointer
    T * position()
    { return _position.returnPointer("position"); }
    T * position( size_t const noParticles)
    { return _position.assignMemory( noParticles, &_noParticles, NO_DIM, "position" ); }

    T * velocity()
    { return _velocity.returnPointer("velocity"); }
    T * velocity( size_t const noParticles)
    { return _velocity.assignMemory( noParticles, &_noParticles, noVelComp, "velocity" ); }

    T * weight()
    { return _weight.returnPointer("weight"); }
    T * weight( size_t const noParticles)
    { return _weight.assignMemory( noParticles, &_noParticles, 1, "weight" ); }

    T * scalar()
    { return _scalar.returnPointer("scalar"); }
    T * scalar( size_t const noParticles)
    { return _scalar.assignMemory( noParticles, &_noParticles, noScalarComp, "scalar" ); }

#ifdef PHASE_SPACE
    T * lagrangianPosition()
    { return _lagrangianPosition.returnPointer("lagrangianPosition"); }
    T * lagrangianPosition( size_t const noParticles)
    { return _lagrangianPosition.assignMemory( noParticles, &_noParticles, NO_DIM, "lagrangianPosition" ); }
#endif

    T * sampling()
    { return _sampling.returnPointer("sampling"); }
    T * sampling( size_t const noSamples)
    { return _sampling.assignMemory( noSamples, &_noSamples, NO_DIM, "sampling" ); }

    T * delta()
    { return _delta.returnPointer("delta"); }
    T * delta( size_t const noSamples)
    { return _delta.assignMemory( noSamples, &_noSamples, NO_DIM, "delta" ); }


    // Repacks the flat arrays into 'Particle_data' / 'Sample_point' vectors (only arrays that were allocated).
    void transferData(std::vector<Particle_data> *p,
                      std::vector<Sample_point>  *s)
    {
        // particle data: repack per ARRAY and release each flat array as soon as it is copied,
        // so the load-phase transient is the Particle_data vector plus the arrays not yet
        // copied instead of the vector plus ALL of them (this transient bounds the largest
        // simulation that fits in RAM; see the memory model in auto_tune.h)
        if ( _noParticles>size_t(0) )
        {
#ifdef PHASE_SPACE
            // particle IDs were only needed for the Lagrangian matching, which already ran
            std::vector<uint64_t>().swap( _particleIDs );
#endif
            p->clear();
            p->resize( _noParticles );
            if ( _position._assigned )
            {
                for (size_t i=0; i<_noParticles; ++i)
                    for (size_t j=0; j<NO_DIM; ++j)
                        (*p)[i].position(j) = _position._ptr[NO_DIM*i+j];
                _position.release();
            }
            if ( _velocity._assigned )
            {
                for (size_t i=0; i<_noParticles; ++i)
                    for (size_t j=0; j<noVelComp; ++j)
                        (*p)[i].velocity(j) = _velocity._ptr[noVelComp*i+j];
                _velocity.release();
            }
            if ( _weight._assigned )
            {
                for (size_t i=0; i<_noParticles; ++i)
                    (*p)[i].weight() = _weight._ptr[i];
                _weight.release();
            }
            if ( _scalar._assigned )
            {
                for (size_t i=0; i<_noParticles; ++i)
                    for (size_t j=0; j<noScalarComp; ++j)
                        (*p)[i].scalar(j) = _scalar._ptr[noScalarComp*i+j];
                _scalar.release();
            }
#ifdef PHASE_SPACE
            if ( _lagrangianPosition._assigned )
            {
                for (size_t i=0; i<_noParticles; ++i)
                    for (size_t j=0; j<NO_DIM; ++j)
                        (*p)[i].lagrangianPosition(j) = _lagrangianPosition._ptr[NO_DIM*i+j];
                _lagrangianPosition.release();
            }
#endif
        }
        // sample points: copy only the arrays that were actually allocated
        if ( _noSamples>size_t(0) )
        {
            s->clear();
            s->reserve( _noSamples );
            for (size_t i=0; i<_noSamples; ++i)
            {
                Sample_point temp;
                if ( _sampling._assigned )
                    for (int j=0; j<NO_DIM; ++j)
                        temp.position(j) = _sampling._ptr[NO_DIM*i+j];
                if ( _delta._assigned )
                    for (int j=0; j<NO_DIM; ++j)
                        temp.delta(j) = _delta._ptr[NO_DIM*i+j];
                s->push_back( temp );
            }
        }
    }
};




// Opens a binary file for input, exiting on failure.
void openInputBinaryFile(std::fstream & inputFile,
                         std::string & fileName)
{
    char const *temp = fileName.c_str();
    inputFile.open( temp, std::ios::in | std::ios::binary );
    
    if ( not inputFile.is_open() )
    {
        std::cout << "~~~ ERROR ~~~ The file '" << fileName << "' could not be opened for reading! The program ended unsucessfully!\n";
        exit( EXIT_FAILURE );
    }
}
// Opens a binary file for output, exiting on failure.
void openOutputBinaryFile(std::fstream & outputFile,
                          std::string & fileName)
{
    char const *temp = fileName.c_str();
    outputFile.open( temp, std::ios::out | std::ios::binary );
    
    if ( not outputFile.is_open() )
    {
        std::cout << "~~~ ERROR ~~~ The file '" << fileName << "' could not be opened for writing! The program ended unsucessfully!\n";
        exit( EXIT_FAILURE );
    }
}
// Opens a text file for input, exiting on failure.
void openInputTextFile(std::fstream & inputFile,
                       std::string & fileName)
{
    char const *temp = fileName.c_str();
    inputFile.open( temp, std::ios::in );
    
    if ( not inputFile.is_open() )
    {
        std::cout << "~~~ ERROR ~~~ The file '" << fileName << "' could not be opened for reading! The program ended unsucessfully!\n";
        exit( EXIT_FAILURE );
    }
}
// Opens a text file for output, exiting on failure.
void openOutputTextFile(std::fstream & outputFile,
                        std::string & fileName)
{
    char const *temp = fileName.c_str();
    outputFile.open( temp, std::ios::out );
    
    if ( not outputFile.is_open() )
    {
        std::cout << "~~~ ERROR ~~~ The file '" << fileName << "' could not be opened for writing! The program ended unsucessfully!\n";
        exit( EXIT_FAILURE );
    }
}


// Checks that the last file read/write succeeded (eof alone is not treated as an error).
void checkFileOperations(std::fstream & file,
                         std::string operationName)
{
    MESSAGE::Error error;
    if ( file.bad() )
        error << "The " << operationName << " file operation has failed. The function ios::bad() has return an error that can be due to the file not being open or if the device has no more writing space left." << MESSAGE::EndError;
    else if ( file.fail() )
        error << "The " << operationName << " file operation has failed. A format error has happened. There are two possible causes for the error: you were trying to read a number, but an alphabetical character was found or you got to the end of the file before being able to read all the data." << MESSAGE::EndError;
}


// Reverses 'n' bytes in place to convert a single value between endiannesses.
inline void ByteSwap(unsigned char * b, int n)
{
	int i = 0;
	int j = n-1;
	while (i<j)
	{
		std::swap(b[i], b[j]);
		i++, j--;
	}
}
// Swaps the endianness of each of the 'elements' values in array 'x'.
template <typename T>
void ByteSwapArray(T *x, size_t const elements)
{
	int size = sizeof(x[0]);
	for (size_t i=0; i<elements; ++i)
		ByteSwap( (unsigned char *) &(x[i]), size );
}

#define BYTESWAP(x) ByteSwap( (unsigned char *) &x, sizeof(x) )

