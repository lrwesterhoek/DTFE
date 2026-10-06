#!/usr/bin/env bash
# Helpers for install.sh (sourced): keep the binaries a user already has until a rebuild has LINKED new
# ones, and say something useful when a build fails. Testable on their own (tests/run_scripts_check.sh K):
#   backup_binaries DIR          copy every built binary (DTFE, PS-DTFE, their -double and -2d variants) into DIR
#   restore_binaries DIR         put them back (a failed 'make' after 'make clean' used to leave NOTHING)
#   build_failed_hint LOG [N]    the last N lines of the build log plus the usual causes
# Survey rank 24, 2026-10-05.

INSTALL_BINARIES=(DTFE PS-DTFE DTFE-double PS-DTFE-double DTFE-2d PS-DTFE-2d)

backup_binaries() {
    local dir="$1" b n=0
    mkdir -p "$dir"
    for b in "${INSTALL_BINARIES[@]}"; do
        if [ -f "$b" ]; then cp -p "$b" "$dir/$b" && n=$((n + 1)); fi
    done
    echo "$n"
}

restore_binaries() {
    local dir="$1" b n=0
    for b in "${INSTALL_BINARIES[@]}"; do
        if [ -f "$dir/$b" ]; then cp -p "$dir/$b" "$b" && n=$((n + 1)); fi
    done
    echo "$n"
}

build_failed_hint() {
    local log="$1" n="${2:-20}"
    echo "---- the last ${n} lines of the build log (${log}):"
    tail -n "$n" "$log" 2>/dev/null || true
    echo "---- the usual causes: a header missing (make deps-check names it and the install command), a compiler"
    echo "     the Makefile does not know (make test-platform), a GPU toolkit without its compiler (--cpu builds"
    echo "     without it), or a half-installed Xcode Command Line Tools (finish it, then rerun)."
}
