# Sourced by the shell suites: the precision they run at.
#
#   DTFE_TEST_PRECISION=single (default)  ./PS-DTFE and ./DTFE, grid outputs float32
#   DTFE_TEST_PRECISION=double            ./PS-DTFE-double and ./DTFE-double ('make DTFE PS-DTFE DOUBLE=1
#                                         [METAL=1]'), grid outputs float64
#
# Every GRID output (.den, .a_den, .streams, .hidden_streams, .caustic, .causticClass, .tetTouch, ...)
# is written in the build's Real type, the masks and counts included; the point-evaluation outputs
# (.pts_*) have the same fixed types in both builds. So a suite reads grids with the dtype exported
# here (DTFE_TEST_REAL, numpy's name) and leaves its .pts_* reads alone. In double mode a suite never
# rebuilds (the pair is built by 'make ... DOUBLE=1'), and skips when the pair is missing.
case "${DTFE_TEST_PRECISION:-single}" in
    double)
        PS_BIN=./PS-DTFE-double; DTFE_BIN=./DTFE-double; REAL_BYTES=8; export DTFE_TEST_REAL=float64 ;;
    single)
        PS_BIN=./PS-DTFE; DTFE_BIN=./DTFE; REAL_BYTES=4; export DTFE_TEST_REAL=float32 ;;
    *)
        echo "DTFE_TEST_PRECISION must be 'single' or 'double' (got '${DTFE_TEST_PRECISION}')" >&2; exit 2 ;;
esac
export DTFE_TEST_PRECISION="${DTFE_TEST_PRECISION:-single}"
# Where the suites put their grids: OUTSIDE the repository by default (tests/tmp sat inside iCloud Drive and
# held 8.2 GB of throwaway grids that iCloud uploaded; survey rank 22, 2026-10-05). DTFE_TEST_TMP overrides.
export DTFE_TEST_TMP="${DTFE_TEST_TMP:-${TMPDIR:-/tmp}/dtfe-tests}"
mkdir -p "${DTFE_TEST_TMP}"
# A suite that cannot run (a binary or a feature missing) exits 0 with a SKIP line -- unless the stage says the
# thing MUST be there (DTFE_TEST_REQUIRED=1, as ci_suite.sh's double and 2d stages do after building it), in
# which case the skip is a FAILURE: a green CI that ran no assertion is worth nothing (survey rank 20).
skip_or_fail() {
    if [ "${DTFE_TEST_REQUIRED:-0}" = "1" ]; then echo "FAIL (required): $*" >&2; exit 1; fi
    echo "SKIP: $*"; exit 0
}
echo ">> precision: ${DTFE_TEST_PRECISION} (${PS_BIN}, ${DTFE_BIN}; grid outputs ${DTFE_TEST_REAL})"
# true in double mode: the suites skip their own build step
precision_double() { [ "${DTFE_TEST_PRECISION}" = "double" ]; }
# exit 0 with a SKIP line when a binary this suite needs is not built in double mode
precision_require() {
    local b
    for b in "$@"; do
        if precision_double && [ ! -x "${b}" ]; then
            skip_or_fail "${b} is not built (make DTFE PS-DTFE DOUBLE=1 builds the double pair)"
        fi
    done
}
