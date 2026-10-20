#!/usr/bin/env bash
# The suites' own skip / fail contract. A suite that cannot run says SKIP and exits 0, unless
# DTFE_TEST_REQUIRED=1 (ci_suite.sh's double and 2d stages, after building what they test) makes that a FAILURE;
# a binary that is broken is never a skip; and a check with nothing to compare against says so. Copies of the
# suites run in scratch trees beside stand-in binaries, so every case is deterministic and needs no build. The
# stand-in ./DTFE hands everything but --full_help to the real ./DTFE: the other tests of run_tests.py still
# pass, so an exit 1 can only come from the case under test.
#   R1  run_tests.py, a ./DTFE whose --full_help exits 3: both exact-average tests FAIL ('--full_help exited
#       with 3', not a skip: a crash is not a missing flag) and the run exits 1, with or without
#       DTFE_TEST_REQUIRED=1
#   R2  run_tests.py, a ./DTFE whose --full_help lacks --exact-average: both tests SKIP and the run exits 0;
#       under DTFE_TEST_REQUIRED=1 both FAIL ('DTFE_TEST_REQUIRED=1, but ...') and it exits 1
#   R3  run_tests.py, a fresh DTFE_TEST_REFERENCE_DIR: the regression is a SKIP (', 1 skipped' in the Results
#       line) and the run exits 0 EVEN under DTFE_TEST_REQUIRED=1 (no build can supply a reference; ci_suite.sh
#       compares with the stored one: R5); the next run compares against the reference written
#   R5  run_tests.py against the STORED reference (tests/reference/regression_periodic.txt, which CI compares
#       with): it PASSES as stored, and with every value moved by 5e-5 relative (rounding-level, under
#       the 3e-4 tolerance); ONE value moved by 1e-3 relative (the size of moving one of the 1000 particles by
#       0.05% of the box) makes it FAIL and the run exit 1
#   R4  run_tests.py, DTFE_TEST_PRECISION=double with no ./DTFE-double: 'SKIP:' and exit 0, or 'FAIL
#       (required):' and exit 1 under DTFE_TEST_REQUIRED=1
#   D1  ps_double_check.sh in a tree with only ./DTFE and ./PS-DTFE: 'SKIP:' and exit 0, or 'FAIL (required):'
#       on stderr and exit 1 under DTFE_TEST_REQUIRED=1
# Usage: tests/run_tests_check.sh [--no-build]   (--no-build is accepted and ignored: nothing is built)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1
PY="${PYTHON:-python3}"
command -v /opt/homebrew/bin/python3.14 >/dev/null 2>&1 && PY=/opt/homebrew/bin/python3.14
# (no precision.sh: the cases set the precision themselves) a missing ./DTFE is a skip, or a failure when required
skip_or_fail() { if [ "${DTFE_TEST_REQUIRED:-0}" = "1" ]; then echo "FAIL (required): $*" >&2; exit 1; fi; echo "SKIP: $*"; exit 0; }
[ -x ./DTFE ] || skip_or_fail "./DTFE is not built (make DTFE)"
TMP="${DTFE_TEST_TMP:-${TMPDIR:-/tmp}/dtfe-tests}/run_tests_check"
rm -rf "$TMP"; mkdir -p "$TMP/rt/tests" "$TMP/dc/tests"

fails=0
check() { if [ "$1" = "0" ]; then printf '   PASS  %s\n' "$2"; else printf '   FAIL  %s\n' "$2"; fails=$((fails+1)); fi; }
has()   { grep -aqE -e "$2" "$1" && echo 0 || echo 1; }
count() { grep -acE -e "$2" "$1" || true; }

# the scratch tree of run_tests.py: the current script (copied now, never a stale one) beside the stand-in ./DTFE
cp tests/run_tests.py "$TMP/rt/tests/"
cat > "$TMP/rt/DTFE" <<'EOF'
#!/usr/bin/env bash
# stand-in ./DTFE: STUB_HELP=fail -> --full_help exits 3; STUB_HELP=noflag -> the real --full_help without the
# --exact-average lines; everything else is the real binary (REAL_DTFE)
if [ "${1:-}" = "--full_help" ]; then
    case "${STUB_HELP:-}" in
        fail)   echo "stand-in: --full_help is broken" >&2; exit 3 ;;
        noflag) help="$("${REAL_DTFE}" --full_help)" || exit $?
                printf '%s\n' "$help" | grep -v -e '--exact-average'; exit 0 ;;
    esac
fi
exec "${REAL_DTFE}" "$@"
EOF
chmod +x "$TMP/rt/DTFE"
# rt <log> <reference dir> [VAR=value ...]: the copied run_tests.py; echoes its exit status. The precision and the
# requirement are cleared first, so each case sets exactly what it tests wherever ci_suite.sh runs this suite.
rt() {
    local log="$1" ref="$2"; shift 2
    local rc=0
    env -u DTFE_TEST_PRECISION -u DTFE_TEST_REQUIRED -u STUB_HELP REAL_DTFE="$ROOT/DTFE" \
        DTFE_TEST_REFERENCE_DIR="$ref" "$@" "$PY" "$TMP/rt/tests/run_tests.py" > "$log" 2>&1 || rc=$?
    echo "$rc"
}
fresh() { mktemp -d "$TMP/ref.XXXXXX"; }

echo ">> R1: --full_help exits 3"
for req in 0 1; do
    rc=$(rt "$TMP/r1_$req.log" "$(fresh)" STUB_HELP=fail DTFE_TEST_REQUIRED=$req)
    check "$([ "$rc" = "1" ] && [ "$(count "$TMP/r1_$req.log" '^  FAIL  exact_average_(lattice|convergence) -- DTFE --full_help exited with 3$')" = "2" ] \
             && [ "$(count "$TMP/r1_$req.log" '^  FAIL ')" = "2" ] && echo 0 || echo 1)" \
          "DTFE_TEST_REQUIRED=$req: both exact-average tests FAIL (--full_help exited with 3), nothing else does, exit 1 ($rc)"
done
echo ">> R2: --full_help without --exact-average"
rc=$(rt "$TMP/r2_0.log" "$(fresh)" STUB_HELP=noflag)
check "$([ "$rc" = "0" ] && [ "$(count "$TMP/r2_0.log" '^  SKIP  exact_average_(lattice|convergence) -- DTFE has no --exact-average$')" = "2" ] \
         && [ "$(count "$TMP/r2_0.log" '^  FAIL ')" = "0" ] && echo 0 || echo 1)" \
      "both exact-average tests SKIP (DTFE has no --exact-average), no failure, exit 0 ($rc)"
check "$(has "$TMP/r2_0.log" '^Results: [0-9]+/[0-9]+ passed, 3 skipped$')" "... the Results line counts them (and the regression's) as skipped"
rc=$(rt "$TMP/r2_1.log" "$(fresh)" STUB_HELP=noflag DTFE_TEST_REQUIRED=1)
check "$([ "$rc" = "1" ] && [ "$(count "$TMP/r2_1.log" '^  FAIL  exact_average_(lattice|convergence) -- DTFE_TEST_REQUIRED=1, but DTFE has no --exact-average$')" = "2" ] \
         && echo 0 || echo 1)" "DTFE_TEST_REQUIRED=1: both FAIL instead, exit 1 ($rc)"
echo ">> R3: no regression reference (a fresh DTFE_TEST_REFERENCE_DIR)"
ref="$(fresh)"
rc=$(rt "$TMP/r3.log" "$ref" DTFE_TEST_REQUIRED=1)
check "$([ "$rc" = "0" ] && [ "$(has "$TMP/r3.log" "^  SKIP  regression -- no reference in ")" = "0" ] \
         && [ "$(count "$TMP/r3.log" '^  FAIL ')" = "0" ] && echo 0 || echo 1)" \
      "the regression is a SKIP and the run exits 0 even under DTFE_TEST_REQUIRED=1 ($rc)"
check "$(has "$TMP/r3.log" '^Results: ([0-9]+)/[0-9]+ passed, 1 skipped$')" "... the Results line says '1 skipped'"
rc=$(rt "$TMP/r3b.log" "$ref" DTFE_TEST_REQUIRED=1)
check "$([ "$rc" = "0" ] && [ -s "$ref/regression_periodic.txt" ] && [ "$(has "$TMP/r3b.log" '^  PASS  regression$')" = "0" ] \
         && [ "$(has "$TMP/r3b.log" 'skipped')" = "1" ] && echo 0 || echo 1)" \
      "... it wrote a reference, and the next run compares against it: PASS, nothing skipped ($rc)"
echo ">> R5: the stored reference, as stored and perturbed"
STORED="$ROOT/tests/reference/regression_periodic.txt"
if [ -s "$STORED" ]; then
    # r5 <log> <scale every value> <scale value 100 of the density column>: a copy of the stored reference, scaled
    r5() {
        local d; d="$(fresh)"
        awk -v all="$2" -v one="$3" '/^#/ { print; next } { n++; a = $1 * all; if (n == 100) a *= one; printf "%.9e %.9e\n", a, $2 * all }' \
            "$STORED" > "$d/regression_periodic.txt"
        rt "$1" "$d"
    }
    rc=$(r5 "$TMP/r5a.log" 1 1)
    check "$([ "$rc" = "0" ] && [ "$(has "$TMP/r5a.log" '^  PASS  regression$')" = "0" ] && echo 0 || echo 1)" \
          "this build matches the stored reference: PASS ($rc)"
    rc=$(r5 "$TMP/r5b.log" 1.00005 1)
    check "$([ "$rc" = "0" ] && [ "$(has "$TMP/r5b.log" '^  PASS  regression$')" = "0" ] && echo 0 || echo 1)" \
          "... every value moved by 5e-5 relative (rounding-level): still PASS ($rc)"
    rc=$(r5 "$TMP/r5c.log" 1 1.001)
    check "$([ "$rc" = "1" ] && [ "$(has "$TMP/r5c.log" '^  FAIL  regression -- .*density: 1/1728 values differ')" = "0" ] && echo 0 || echo 1)" \
          "... ONE value moved by 1e-3 relative: FAIL, exit 1 ($rc)"
else
    check 1 "the stored reference $STORED exists"
fi
echo ">> R4: DTFE_TEST_PRECISION=double without ./DTFE-double"
rc=$(rt "$TMP/r4_0.log" "$(fresh)" DTFE_TEST_PRECISION=double)
check "$([ "$rc" = "0" ] && [ "$(has "$TMP/r4_0.log" '^SKIP: .*DTFE-double is not built')" = "0" ] && echo 0 || echo 1)" "SKIP: and exit 0 ($rc)"
rc=$(rt "$TMP/r4_1.log" "$(fresh)" DTFE_TEST_PRECISION=double DTFE_TEST_REQUIRED=1)
check "$([ "$rc" = "1" ] && [ "$(has "$TMP/r4_1.log" '^FAIL \(required\): .*DTFE-double is not built')" = "0" ] && echo 0 || echo 1)" \
      "DTFE_TEST_REQUIRED=1: FAIL (required): and exit 1 ($rc)"

echo ">> D1: ps_double_check.sh beside ./DTFE and ./PS-DTFE only"
cp tests/ps_double_check.sh "$TMP/dc/tests/"
printf '#!/bin/sh\nexit 0\n' > "$TMP/dc/DTFE"; cp "$TMP/dc/DTFE" "$TMP/dc/PS-DTFE"; chmod +x "$TMP/dc/DTFE" "$TMP/dc/PS-DTFE"
for req in 0 1; do
    rc=0; env -u DTFE_TEST_PRECISION DTFE_TEST_REQUIRED=$req DTFE_TEST_TMP="$TMP/dc_tmp" bash "$TMP/dc/tests/ps_double_check.sh" --no-build \
        > "$TMP/d1_$req.out" 2> "$TMP/d1_$req.err" || rc=$?
    if [ "$req" = "0" ]; then
        check "$([ "$rc" = "0" ] && [ "$(has "$TMP/d1_0.out" '^SKIP: \./DTFE-double is not built')" = "0" ] && echo 0 || echo 1)" "SKIP: and exit 0 ($rc)"
    else
        check "$([ "$rc" = "1" ] && [ "$(has "$TMP/d1_1.err" '^FAIL \(required\): \./DTFE-double is not built')" = "0" ] \
                 && [ "$(has "$TMP/d1_1.out" 'SKIP')" = "1" ] && echo 0 || echo 1)" \
              "DTFE_TEST_REQUIRED=1: FAIL (required): on stderr, no SKIP, exit 1 ($rc)"
    fi
done
check "$([ ! -e "$TMP/dc_tmp/double_check/waves.hdf5" ] && echo 0 || echo 1)" "... it stops before generating any data"

echo ""
if [ "$fails" -eq 0 ]; then
    rm -rf "$TMP"
    echo "RESULT: PASS  (run_tests.py and ps_double_check.sh skip what they cannot run, fail it when it is required,"
    echo "               fail a broken binary, and say so when the regression had nothing to compare)"
    exit 0
fi
echo "RESULT: FAIL  (${fails} check(s); logs kept under ${TMP})"
exit 1
