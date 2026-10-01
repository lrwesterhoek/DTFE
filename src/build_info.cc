/* Build provenance of the DTFE / PS-DTFE binaries (not the library).

   The Makefile recompiles this file on EVERY link with the current git revision
   ('git describe --always --dirty') and the UTC link time, so every run's log names the exact
   code that produced its outputs. That is what lets a results browser tell a grid made before
   an output-changing fix from one made after it (the run date alone cannot: a stale binary can
   run long after the fix landed). Outside a git checkout, or when the defines are missing, the
   fields read 'unknown'. */

#ifndef DTFE_BUILD_REV
#define DTFE_BUILD_REV "unknown"
#endif
#ifndef DTFE_BUILD_TIME
#define DTFE_BUILD_TIME "unknown"
#endif

char const *dtfeBuildRevision() { return DTFE_BUILD_REV; }
char const *dtfeBuildTime()     { return DTFE_BUILD_TIME; }
