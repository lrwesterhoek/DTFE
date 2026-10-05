/* The canonical form of an input path, for the identity keys of the tessellation cache
   (tessellation_cache.h makeDescriptor) and of the globals record (ps_globals_record.h).

   Those keys hold the input files' paths, so the same snapshot named two ways -- a relative path, a
   path through a symlink ('/tmp' is '/private/tmp' on macOS), or other letter case on a
   case-insensitive volume -- used to be two different keys: a cache written through one name was
   rebuilt through the other. realpath() gives every name of a file the same string. A path that
   does not resolve (a missing file) is kept as given; that run fails on reading it anyway. Paths
   that were already canonical keep their keys, so existing caches stay valid. */

#ifndef DTFE_CANONICAL_PATH_HEADER
#define DTFE_CANONICAL_PATH_HEADER

#include <cstdlib>
#include <string>

inline std::string canonicalPath(std::string const &path)
{
    if ( path.empty() ) return path;
    char *r = ::realpath( path.c_str(), nullptr );
    if ( r == nullptr ) return path;
    std::string const out( r );
    std::free( r );
    return out;
}

#endif
