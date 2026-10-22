/* The environment variables the binaries read (README.md 'Environment variables of the binaries' and
   metal/README.md 'Knobs' list each one). Every read goes through these helpers, so one rule holds everywhere:
   - an EMPTY value is the same as an unset one (for every kind of value);
   - a flag is off when set to 0, false, no or off (in any case) and on when set to anything else (flag(n, true): a
     switch that is on by default, which only those values turn off);
   - a tristate (a switch with an automatic default) is -1 while unset or set to auto, else the flag;
   - a number is the value read by std::atol / std::atof; the caller checks its range and keeps its default otherwise;
   - a Name may carry an ALIAS, read only when the name itself is unset: each GPU knob's Metal spelling is an alias
     of its backend-neutral name (Name{ "PS_GPU_CHUNK", "PS_METAL_CHUNK" }).
   Plain C++17 with no other header of the project: the CUDA/HIP sources (nvcc, hipcc, the CPU emulation) include it
   too. */
#ifndef DTFE_ENV_KNOBS_H
#define DTFE_ENV_KNOBS_H

#include <cstdlib>
#include <initializer_list>
#include <string>

namespace Env
{

// a variable's name and the alias read when the name is unset (nullptr: none)
struct Name
{
    char const *name;
    char const *alias;
    Name(char const *n, char const *a = nullptr) : name( n ), alias( a ) {}
};

// the value: of the name, else of the alias; nullptr when neither is set to a non-empty value
inline char const *value(Name const n)
{
    for (char const *k : { n.name, n.alias })
        if ( k != nullptr )
            if ( char const *v = std::getenv( k ) )
                if ( v[0] != '\0' ) return v;
    return nullptr;
}

inline bool isSet(Name const n) { return value( n ) != nullptr; }

// the value in lower case (ASCII), for the words below
inline std::string lower(char const *v)
{
    std::string s( v );
    for (char &c : s)
        if ( c >= 'A' and c <= 'Z' ) c = char( c - 'A' + 'a' );
    return s;
}

// true for the values that turn a flag off: 0, false, no, off
inline bool offValue(char const *v)
{
    std::string const s = lower( v );
    return s == "0" or s == "false" or s == "no" or s == "off";
}

inline bool flag(Name const n, bool const def = false)
{
    char const *v = value( n );
    return v == nullptr ? def : not offValue( v );
}

// -1 unset or auto (the switch's automatic choice), 0 an off value, 1 anything else
inline int tristate(Name const n)
{
    char const *v = value( n );
    if ( v == nullptr or lower( v ) == "auto" ) return -1;
    return offValue( v ) ? 0 : 1;
}

inline long integer(Name const n, long const def)
{
    char const *v = value( n );
    return v == nullptr ? def : std::atol( v );
}

inline double real(Name const n, double const def)
{
    char const *v = value( n );
    return v == nullptr ? def : std::atof( v );
}

inline std::string text(Name const n, std::string const &def = std::string())
{
    char const *v = value( n );
    return v == nullptr ? def : std::string( v );
}

}   // namespace Env

#endif
