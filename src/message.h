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


/*
    Runtime user messages, filtered by 'verboseLevel':
        3 - all messages, including progress
        2 - user messages, warnings and errors
        1 - warnings and errors only
        0 - errors only
    Messages are streamed with operator<<, e.g. Message m(level); m << "x = " << 10 << "\n";
*/


#ifndef MESSAGE_HEADER
#define MESSAGE_HEADER

#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <cstdlib>
#include <unistd.h>   // isatty -> only colour a real terminal


#define OUT std::cout


namespace MESSAGE
{

// ANSI colours for the runtime log; emitted only when stdout is a real terminal and NO_COLOR is unset, so piped output stays clean.
inline bool colorEnabled()
{
    static bool const enabled = ( isatty(STDOUT_FILENO)==1 ) && ( std::getenv("NO_COLOR")==nullptr );
    return enabled;
}
inline char const* cRed()     { return colorEnabled() ? "\033[1;31m" : ""; }  // errors
inline char const* cYellow()  { return colorEnabled() ? "\033[1;33m" : ""; }  // warnings
inline char const* cGreen()   { return colorEnabled() ? "\033[1;32m" : ""; }  // timings / success / progress
inline char const* cCyan()    { return colorEnabled() ? "\033[1;36m" : ""; }  // section banners / headers
inline char const* cBlue()    { return colorEnabled() ? "\033[1;34m" : ""; }  // file paths / I/O
inline char const* cMagenta() { return colorEnabled() ? "\033[1;35m" : ""; }  // physics results (density, streams, ...)
inline char const* cYellowD() { return colorEnabled() ? "\033[33m"   : ""; }  // labels / keys (non-bold)
inline char const* cDim()     { return colorEnabled() ? "\033[2m"    : ""; }  // separators / secondary
inline char const* cBold()    { return colorEnabled() ? "\033[1m"    : ""; }
inline char const* cReset()   { return colorEnabled() ? "\033[0m"    : ""; }

// Horizontal rule of repeated character 'c'.
inline std::string hrule( char c = '=', int width = 74 )
{ return std::string( size_t(width<0?0:width), c ); }

// Section banner: title centred between '=' rules (cyan+bold).
inline std::string banner( std::string const &title )
{
    size_t const W = 74;
    std::string t = "  " + title + "  ";
    size_t const pad  = ( t.size() < W ) ? ( W - t.size() ) : size_t(0);
    size_t const left = pad / 2;
    std::ostringstream o;
    o << "\n" << cCyan() << std::string(left,'=') << t << std::string(pad-left,'=')
      << cReset() << "\n";
    return o.str();
}

// Format a duration in seconds as "Hh Mm Ss" / "Mm Ss" / "Ss".
inline std::string formatDuration( double seconds )
{
    if ( seconds < 0. ) seconds = 0.;
    long s = long( seconds + 0.5 );
    long const h = s/3600; s %= 3600;
    long const m = s/60;   s %= 60;
    std::ostringstream o;
    if      ( h>0 ) o << h << "h " << m << "m " << s << "s";
    else if ( m>0 ) o << m << "m " << s << "s";
    else            o << s << "s";
    return o.str();
}


// Sentinel ending an error message; printing it terminates the program (must be sent).
struct EndErrorMessage
{
    inline friend std::ostream& operator << (std::ostream& out, EndErrorMessage &end)
    {
        out << " The program ended unsucessfully!\n\n";
        exit( EXIT_FAILURE );
        return out;
    }
};


// Sentinel ending a warning message.
struct EndWarningMessage
{
    inline friend std::ostream& operator << (std::ostream& out, EndWarningMessage &end)
    {
        out << "\n\n" << std::flush;
        return out;
    }
};


// Flushes a message from the buffer.
struct FlushMessage
{
    inline friend std::ostream& operator << (std::ostream& out, FlushMessage &end)
    {
        out << std::flush;
        return out;
    }
};



// Error message; always printed. Must end with 'EndError' to terminate the program.
struct Error
{
    int _verboseLevel;  // unused; kept for consistency with the other message classes
    int iterationNo;

    Error(void)
    {
        _verboseLevel = 0;
        iterationNo = 0;
    }
    Error(int const verboseLevel)
    {
        if ( verboseLevel<0 )
            OUT << "\n\n~~~ ERROR ~~~ " << "The constructor of the 'Error' class must receive an integer value >= " << 0 << ". This represents the verbose level of the program - check 'message.h' for additional details." << " The program ended unsucessfully!\n";
        _verboseLevel = verboseLevel;
        iterationNo = 0;
    }
    
    template <typename T>
    inline Error& operator << (T right)
    {
        if ( (iterationNo++)==0 )
            OUT << "\n\n" << cRed() << "~~~ ERROR ~~~ " << cReset();
        OUT << right;
        return *this;
    }
};


// Warning message; printed only if verbose level >= 1. Must end with 'EndWarning'.
struct Warning
{
    int _verboseLevel;
    int iterationNo;

    Warning(int const verboseLevel)
    {
        if ( verboseLevel<0 )
        {
            Error error;
            error << "The constructor of the 'Warning' class must receive an integer value >= " << 0 << ". This represents the verbose level of the program - check 'message.h' for additional details." << EndErrorMessage();
        }
        _verboseLevel = verboseLevel;
        iterationNo = 0;
    }
    
    template <typename T>
    inline Warning& operator << (T right)
    {
        if ( _verboseLevel<1 )
            return *this;
        if ( (iterationNo++)==0 )
            OUT << "\n\n" << cYellow() << "~~~ WARNING ~~~ " << cReset();
        OUT << right;
        return *this;
    }
};


// Sends a message to the user; printed only if verbose level >= 2.
struct Message
{
    int _verboseLevel;
    int _percentangeDone;
    
    
    Message(int const verboseLevel)
    {
        if ( verboseLevel<0 )
        {
            Error error;
            error << "The constructor of the 'Message' class must receive an integer value >= " << 0 << ". This represents the verbose level of the program - check 'message.h' for additional details." << EndErrorMessage();
        }
        _verboseLevel = verboseLevel;
        _percentangeDone = 0;
    }

    template <typename T>
    inline Message& operator << (T right)
    {
        if ( _verboseLevel>=2 )
            OUT << right;
        return *this;
    }
    
    inline void updateProgress(int const percentageDone)
    {
        if ( _verboseLevel<3 or _percentangeDone==percentageDone )
            return;
        _percentangeDone = percentageDone;
        char const del[4] = "\b\b\b";
        OUT << std::setw(2) << _percentangeDone << "\%" << std::flush << del;
    }
};



// Prints array/vector elements to a string, joined by 'separation'.
template <typename T>
std::string printElements(T *ptr, size_t const size, std::string separation=" ")
{
    if ( size<1 ) return "";
    
    std::ostringstream buffer;
    buffer << ptr[0];
    for (size_t i=1; i<size; ++i)
        buffer << separation << ptr[i];
    return buffer.str();
}
template <typename T>
std::string printElements(T &vec, std::string separation=" ")
{
    return printElements( &(vec[0]), vec.size(), separation );
}


static const EndErrorMessage   EndError   = EndErrorMessage();
static const EndWarningMessage EndWarning = EndWarningMessage();
static const FlushMessage      Flush      = FlushMessage();
} // end namespace MESSAGE




// Throw an error message (1 to 5 parts) and stop the program; for more parts use MESSAGE::Error directly.
template <typename T1, typename T2, typename T3, typename T4, typename T5>
void throwError(T1 message_1,
                T2 message_2,
                T3 message_3,
                T4 message_4,
                T5 message_5)
{
    MESSAGE::Error error;
    error << message_1 << message_2 << message_3 << message_4 <<  message_5 << MESSAGE::EndError;
}
template <typename T1, typename T2, typename T3, typename T4>
void throwError(T1 m1, T2 m2, T3 m3, T4 m4)
{
    throwError( m1, m2, m3, m4, "" );
}
template <typename T1, typename T2, typename T3>
void throwError(T1 m1, T2 m2, T3 m3)
{
    throwError( m1, m2, m3, "", "" );
}
template <typename T1, typename T2>
void throwError(T1 m1, T2 m2)
{
    throwError( m1, m2, "", "", "" );
}
template <typename T1>
void throwError(T1 m1)
{
    throwError( m1, "", "", "", "" );
}


#endif
