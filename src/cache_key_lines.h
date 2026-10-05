/* Pieces of the tessellation-cache key (tessellation_cache.h makeDescriptor) that a cheap reader needs
   too: the build-configuration lines and the input files' stamps. The server's tuner (auto_tune.h
   autoTuneServe) scans the cache folder for the partition splits of the SAME snapshot and build that
   are already cached, and must compare exactly the text the cache writes -- so both build it here. */

#ifndef DTFE_CACHE_KEY_LINES_HEADER
#define DTFE_CACHE_KEY_LINES_HEADER

#include <sstream>
#include <string>
#include <sys/stat.h>

#include "define.h"
#include "canonical_path.h"

// "<size>:<mtime>", or "absent" / "missing"
inline std::string cacheFileStamp(std::string const &path)
{
    if ( path.empty() ) return "absent";
    struct stat st;
    if ( ::stat( path.c_str(), &st ) != 0 ) return "missing";
    std::ostringstream os;
    os << (long long)st.st_size << ':' << (long long)st.st_mtime;
    return os.str();
}

// the compile-time configuration: it changes the payload layout and/or the triangulated coordinates
inline std::string cacheBuildLines()
{
    std::ostringstream d;
    d << "dim="      << NO_DIM << " real=" << sizeof(Real) << " velComp=" << noVelComp
      << " scalarComp=" << noScalarComp << '\n';
    d << "build="
#ifdef PHASE_SPACE
      << "PS"
#else
      << "std"
#endif
#ifdef VELOCITY
      << "+vel"
#endif
#ifdef SCALAR
      << "+scalar"
#endif
#ifdef TEST_PADDING
      << "+testpad"
#endif
      << '\n';
    return d.str();
}

// an input file's key line: its canonical path and stamp ("input=..." / "lagrange=...")
inline std::string cacheFileLine(char const *key, std::string const &path)
{
    return std::string( key ) + "=" + canonicalPath( path ) + ' ' + cacheFileStamp( path );
}

#endif
