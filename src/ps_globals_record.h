/* The particle-derived GLOBALS of a snapshot, recorded in the tessellation cache folder (PS-DTFE).

   A periodic, partitioned phase-space run derives a few quantities from ALL its particles before the
   partition loop: their number (which sets the padding), the mean density (a sum of their masses) and
   the bounding box of their initial positions (which places every partition). All of them enter the
   identity of every cached partition (tessellation_cache.h makeDescriptor), and none follows from the
   file header alone. Any such run with '--tessellation-cache' writes them here, EXACTLY (hexadecimal
   floats), keyed by everything that decides which particles are read: the input files' paths and
   size:mtime stamps, the file type, the length unit, the periodicity, the species, a user box, the
   sub-sampling options.

   A later '--ps-window' run (the launcher's exact zoom) whose inputs match then starts WITHOUT reading
   the particles -- 16 s of a TNG100-3 exact zoom -- taking these globals from the record. The particles
   are read only if a partition the window needs turns out not to be cached (DTFE.cpp, at the skip
   set): the run is then the ordinary one from that point on. The record can only shorten a run, never
   change it: the partitions it loads are still checked against their full descriptors, and a late read
   verifies that the snapshot still gives the recorded box and particle count. */

#ifndef PS_GLOBALS_RECORD_HEADER
#define PS_GLOBALS_RECORD_HEADER

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <sys/stat.h>

#include "canonical_path.h"
#include "user_options.h"

namespace PSGlobals {

// "<size>:<mtime>" -- the same stamp as tessellation_cache.h fileStamp
inline std::string stamp(std::string const &path)
{
    if ( path.empty() ) return "absent";
    struct stat st;
    if ( ::stat( path.c_str(), &st ) != 0 ) return "missing";
    std::ostringstream os;
    os << (long long)st.st_size << ':' << (long long)st.st_mtime;
    return os.str();
}

inline std::string hexReal(double const v)
{
    char b[64];
    std::snprintf( b, sizeof b, "%a", v );
    return b;
}

// Everything that decides WHICH particles are read and with WHICH coordinates -- not the particles.
inline std::string key(User_options const &u)
{
    std::ostringstream d;
    d << "globals=1\n";
    d << "real=" << sizeof(Real) << " dim=" << NO_DIM << '\n';
    d << "input=" << canonicalPath( u.inputFilename ) << ' ' << stamp( u.inputFilename ) << '\n';
    d << "type=" << u.inputFileType << '\n';
#ifdef PHASE_SPACE
    d << "lagrange=" << canonicalPath( u.lagrangianInputFilename ) << ' ' << stamp( u.lagrangianInputFilename ) << '\n';
#endif
    d << "MpcUnit=" << hexReal( double(u.MpcValue) ) << '\n';
    d << "periodic=" << (u.periodicInput ? 1 : 0) << '\n';
    d << "species=";
    for (size_t i = 0; i < u.readParticleSpecies.size(); ++i) d << u.readParticleSpecies[i];
    d << '\n';
    d << "box=";
    if ( u.userGivenBoxCoordinates )
        for (size_t i = 0; i < u.boxCoordinates.coords.size(); ++i) d << hexReal( double(u.boxCoordinates.coords[i]) ) << ',';
    else
        d << "header";
    d << '\n';
    if ( not u.scalarDataset.empty() ) d << "scalarDataset=" << u.scalarDataset << '\n';
    d << "randomSample=" << hexReal( double(u.randomSample) ) << " poisson=" << u.poisson << '\n';
    d << "redshiftSpace=" << (u.transformToRedshiftSpaceOn ? 1 : 0) << ' ';
    for (size_t i = 0; i < u.transformToRedshiftSpace.size(); ++i) d << hexReal( double(u.transformToRedshiftSpace[i]) ) << ',';
    d << '\n';
    return d.str();
}

// the key main() computed before reading (a user --box is then still in the file's units), else now
inline std::string keyOf(User_options const &u) { return u.psGlobalsKey.empty() ? key( u ) : u.psGlobalsKey; }

inline std::string path(User_options const &u)
{
    std::string const k = keyOf( u );
    uint64_t h = 1469598103934665603ull;                  // FNV-1a, as the tessellation cache
    for (char const c : k) { h ^= static_cast<unsigned char>( c ); h *= 1099511628211ull; }
    char hex[17];
    std::snprintf( hex, sizeof hex, "%016llx", (unsigned long long)h );
    return u.tessellationCacheDir + "/globals_" + hex + ".txt";
}

struct Record
{
    size_t n = 0;                       // the particle count (the padding is derived from it)
    double averageDensity = 0.;         // exactly the run's mean density (Real bits)
    double box[2*NO_DIM] = {};          // the box after the header and the length unit
    double lagBox[2*NO_DIM] = {};       // the bounding box of the initial positions
    double hubble = -1., scaleFactor = -1.;   // the header's, as the reader sets them
};

// Written once the globals exist (DTFE.cpp, periodic PS block): aside, then renamed.
inline void write(User_options const &u, Record const &r)
{
    if ( u.tessellationCacheDir.empty() ) return;
    std::string const p = path( u ), tmp = p + ".part";
    std::ostringstream text;
    text << keyOf( u ) << "values\n";
    text << "N=" << r.n << '\n';
    text << "averageDensity=" << hexReal( r.averageDensity ) << '\n';
    text << "box=";
    for (int i = 0; i < 2*NO_DIM; ++i) text << hexReal( r.box[i] ) << ( i + 1 < 2*NO_DIM ? "," : "\n" );
    text << "lagBox=";
    for (int i = 0; i < 2*NO_DIM; ++i) text << hexReal( r.lagBox[i] ) << ( i + 1 < 2*NO_DIM ? "," : "\n" );
    text << "hubble=" << hexReal( r.hubble ) << '\n';
    text << "scaleFactor=" << hexReal( r.scaleFactor ) << '\n';
    text << "end\n";
    {   // the same record already there: leave it (a cache folder should not churn)
        std::ifstream old( p.c_str() );
        if ( old )
        {
            std::stringstream ss;
            ss << old.rdbuf();
            if ( ss.str() == text.str() ) return;
        }
    }
    {
        std::ofstream f( tmp.c_str(), std::ios::trunc );
        if ( not f ) return;
        f << text.str();
        if ( not f ) { f.close(); std::remove( tmp.c_str() ); return; }
    }
    if ( std::rename( tmp.c_str(), p.c_str() ) != 0 ) std::remove( tmp.c_str() );
}

// True with the values when a record for exactly these inputs exists.
inline bool read(User_options const &u, Record &r)
{
    if ( u.tessellationCacheDir.empty() ) return false;
    std::ifstream f( path( u ).c_str() );
    if ( not f ) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    std::string const text = ss.str(), k = keyOf( u );
    if ( text.compare( 0, k.size(), k ) != 0 ) return false;          // another snapshot (or a stale one)
    std::istringstream in( text.substr( k.size() ) );
    std::string line;
    if ( not std::getline( in, line ) or line != "values" ) return false;
    auto parseList = [](std::string const &s, double *out, int n) -> bool
    {
        std::istringstream ls( s );
        std::string tok;
        for (int i = 0; i < n; ++i)
        {
            if ( not std::getline( ls, tok, ',' ) ) return false;
            out[i] = std::strtod( tok.c_str(), nullptr );
        }
        return true;
    };
    int got = 0;
    while ( std::getline( in, line ) )
    {
        size_t const eq = line.find( '=' );
        if ( line == "end" ) { got |= 64; break; }
        if ( eq == std::string::npos ) return false;
        std::string const name = line.substr( 0, eq ), val = line.substr( eq + 1 );
        if      ( name == "N" )              { r.n = size_t( std::strtoull( val.c_str(), nullptr, 10 ) ); got |= 1; }
        else if ( name == "averageDensity" ) { r.averageDensity = std::strtod( val.c_str(), nullptr ); got |= 2; }
        else if ( name == "box" )            { if ( not parseList( val, r.box, 2*NO_DIM ) ) return false; got |= 4; }
        else if ( name == "lagBox" )         { if ( not parseList( val, r.lagBox, 2*NO_DIM ) ) return false; got |= 8; }
        else if ( name == "hubble" )         { r.hubble = std::strtod( val.c_str(), nullptr ); got |= 16; }
        else if ( name == "scaleFactor" )    { r.scaleFactor = std::strtod( val.c_str(), nullptr ); got |= 32; }
    }
    return got == 127 and r.n > 0 and r.averageDensity > 0.;
}

// A run that may start without its particles: a phase-space '--ps-window' run over a periodic box
// with a tessellation cache and an explicit partition split, reading a Gadget snapshot, with nothing
// that selects or generates particles differently (a region, sub-sampling, Poisson particles, a
// redshift-space shift, sample points, interlacing) and the mean density taken from the particles.
inline bool eligible(User_options const &u)
{
#if defined(PHASE_SPACE) && NO_DIM==3
    return u.psWindowOn and not u.tessellationCacheDir.empty() and u.periodic and u.partitionOn
        and u.partNo < 0 and not u.regionOn and u.poisson == 0 and u.randomSample < Real(0.)
        and not u.transformToRedshiftSpaceOn and u.psSamplePointsFile.empty() and not u.psServe
        and not u.interlace and not u.redshiftConeOn and not ( u.averageDensity > Real(0.) )
        and ( u.inputFileType == 101 or u.inputFileType == 105 );
#else
    (void)u;
    return false;
#endif
}

}   // namespace PSGlobals

#endif
