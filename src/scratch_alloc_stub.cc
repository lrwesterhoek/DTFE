/* libDTFE stand-in for scratch_alloc.cc: the library must never replace its host program's
 * global operator new/delete, so it gets this refusing stub instead of the real hook. The
 * caller (DTFE_setup) surfaces the false return as "scratch not available in this build". */

#include "scratch_alloc.h"

namespace ScratchAlloc
{
    bool scratchArm(char const *, std::size_t) { return false; }
    long scratchLiveBytes()   { return 0; }
    long scratchTotalAllocs() { return 0; }
}
