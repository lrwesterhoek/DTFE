#!/usr/bin/env bash
# PS-DTFE --tessellation-cache correctness check.
#
# The flag reuses the Delaunay tessellation (and its vertex densities) across runs. Because a
# stale or mismatched cache would silently corrupt the science rather than fail, the contract is
# mostly about WHEN the cache must NOT be used. Asserted here:
#  A) A first run MISSES and writes a cache file; a second identical run HITS.
#  B) Every output of a HIT run is BIT-IDENTICAL to the same run without the cache -- including
#     the point-evaluation ('--sample-points') files, which read the tessellation independently
#     of the grid deposit.
#  C) A run with a DIFFERENT GRID still HITS: the grid does not enter the tessellation, and this
#     is the whole point of the flag (re-query one snapshot cheaply).
#  D) Staleness is detected: touching the input file (its mtime is part of the cache identity)
#     forces a MISS, as does changing '--density0' (which rescales the cached vertex densities).
#  E) A MULTI-PARTITION run caches every partition SEPARATELY: one file each, all restored on the
#     next run, and demonstrably different tessellations. 'partNo' stays -1 for all of them, so the
#     descriptor separates them by their unpadded Lagrangian region -- without that, partitions
#     2..N would silently load the first partition's tessellation.
#  F) A partitioned cache hit also reproduces the uncached run bit-for-bit.
#  A2) The body is DEFLATED while magic/version/descriptor stay plaintext, so a staleness check
#     needs no inflation. Compression takes 735 MB down to 225 MB per 1.5e6 vertices at zlib
#     level 1 -- level 6 saves a further 5% but costs 24 s more on every write.
#
# Usage: tests/ps_tessellation_cache_check.sh [--no-build]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
command -v /opt/homebrew/bin/python3.14 >/dev/null 2>&1 && PY=/opt/homebrew/bin/python3.14
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

N="${N:-24}"; GRID="${GRID:-48}"; BOX="${BOX:-100.0}"
BIN="./PS-DTFE"
TMP="${SCRIPT_DIR}/tmp"; mkdir -p "${TMP}"
SNAP="${TMP}/ptc_input.hdf5"
# The cache must live on a LOCAL volume -- the repo itself is in iCloud, which the binary rejects.
# ${TMPDIR:-/tmp} keeps this working on Linux CI, where /private/tmp does not exist and the binary
# would refuse the path for not being a directory.
CACHE="${TESS_CACHE_DIR:-${TMPDIR:-/tmp}/dtfe-tesscache-test}"

echo "============================================================"
echo " PS-DTFE --tessellation-cache check   N=${N}^3  grid=${GRID}^3"
echo "============================================================"

if [ "${1:-}" != "--no-build" ]; then
    echo ">> building PS-DTFE ..."
    BUILD_MODE="$(cat o_ps/.build_mode 2>/dev/null || true)"
    make PS-DTFE ${BUILD_MODE:+"$BUILD_MODE"} >/dev/null
fi
[ -x "${BIN}" ] || { echo "FAIL: ${BIN} not built"; exit 1; }

rm -rf "${CACHE}"; mkdir -p "${CACHE}"
rm -f "${TMP}"/ptc_out_*

echo ">> generating the test snapshot ..."
"${PY}" "${SCRIPT_DIR}/generate_ps_test_data.py" --out "${SNAP}" --n "${N}" --box "${BOX}" \
        --crossed-waves >/dev/null

# a small point set for the --sample-points half of the contract
POINTS="${TMP}/ptc_points.bin"
"${PY}" - "$POINTS" "$BOX" <<'EOF'
import numpy as np, sys
rng = np.random.default_rng(7)
L = float(sys.argv[2])
rng.uniform(0.05*L, 0.95*L, size=(20000, 3)).astype(np.float64).tofile(sys.argv[1])
EOF

COMMON=(--periodic --field density velocity --MpcUnit 1 --max-concurrent 1 --sample-points "${POINTS}")

fails=0
note() { echo "   $1"; }
check() { if [ "$1" = "1" ]; then echo "   PASS  $2"; else echo "   FAIL  $2"; fails=$((fails+1)); fi; }

# run <outroot> <grid> <extra args...>; captures stdout in RUN_OUT. A non-zero exit must not kill
# the script under 'set -e' -- the checks below report it as a failure instead.
run() {
    local out="$1" grid="$2"; shift 2
    RUN_OUT="$( "${BIN}" "${SNAP}" "${out}" --grid "${grid}" "${COMMON[@]}" "$@" --verbose 3 2>&1 || true )"
}
hits()    { echo "${RUN_OUT}" | grep -c "Reused the cached" || true; }
ignored() { echo "${RUN_OUT}" | grep -c "is ignored for this run" || true; }

echo ""
echo "(A) first run misses and writes, second run hits"
run "${TMP}/ptc_out_nocache" "${GRID}"                                   # reference, no cache
run "${TMP}/ptc_out_write" "${GRID}" --tessellation-cache "${CACHE}"
check "$([ "$(hits)" = "0" ] && echo 1 || echo 0)" "first cached run is a MISS"
nfiles=$(ls -1 "${CACHE}"/*.tess 2>/dev/null | wc -l | tr -d ' ')
check "$([ "${nfiles}" = "1" ] && echo 1 || echo 0)" "exactly one cache file written (got ${nfiles})"
run "${TMP}/ptc_out_hit" "${GRID}" --tessellation-cache "${CACHE}"
check "$([ "$(hits)" -ge "1" ] && echo 1 || echo 0)" "second run HITS (different output root)"

echo ""
echo "(A2) the file is compressed, with a PLAINTEXT header"
# The body is deflated (gzip wrapper) but magic + version + descriptor stay uncompressed, so a
# staleness check costs one small read and no inflation. Asserting the layout keeps a future change
# from silently reverting to an uncompressed body (3.3x the disk) or compressing the header.
tessfile=$(ls -1 "${CACHE}"/*.tess 2>/dev/null | head -1)
if [ -n "${tessfile}" ]; then
    head -c 8 "${tessfile}" | grep -q "DTFETESS" \
        && check 1 "header starts with the plaintext magic" \
        || check 0 "header starts with the plaintext magic"
    # the gzip magic (1f 8b) must appear AFTER the header, not at byte 0
    zoff=$("${PY}" -c "
import sys
d = open(sys.argv[1], 'rb').read(65536)
sys.stdout.write(str(d.find(b'\x1f\x8b')))
" "${tessfile}")
    check "$([ "${zoff}" -gt "8" ] && echo 1 || echo 0)" "deflated body begins after the header (gzip magic at byte ${zoff})"
    grep -aq "format=" "${tessfile}" \
        && check 1 "descriptor is readable without inflating" \
        || check 0 "descriptor is readable without inflating"
else
    check 0 "a cache file exists to inspect"
fi

echo ""
echo "(B) a hit reproduces every output bit-for-bit"
for f in "${TMP}"/ptc_out_nocache.*; do
    ext="${f##*ptc_out_nocache}"
    g="${TMP}/ptc_out_hit${ext}"
    if [ -f "${g}" ] && cmp -s "${f}" "${g}"; then
        check 1 "bit-identical ${ext}"
    else
        check 0 "bit-identical ${ext}"
    fi
done

echo ""
echo "(C) the grid is not part of the tessellation: a different grid still hits"
run "${TMP}/ptc_out_grid" "$((GRID*2))" --tessellation-cache "${CACHE}"
check "$([ "$(hits)" -ge "1" ] && echo 1 || echo 0)" "different --grid still HITS"

echo ""
echo "(D) staleness is detected"
run "${TMP}/ptc_out_dens" "${GRID}" --tessellation-cache "${CACHE}" --density0 2.5
check "$([ "$(hits)" = "0" ] && echo 1 || echo 0)" "changed --density0 MISSES"
touch "${SNAP}"
run "${TMP}/ptc_out_touched" "${GRID}" --tessellation-cache "${CACHE}"
check "$([ "$(hits)" = "0" ] && echo 1 || echo 0)" "touched input file MISSES"

echo ""
echo "(E) a multi-partition run caches EACH partition separately"
# 'partNo' is -1 for every partition of a Lagrangian split, so the descriptor must separate them by
# their unpadded Lagrangian region. If it did not, partitions 2..N would load the first partition's
# tessellation -- the same vertex count everywhere, and wrong results.
PCACHE="${CACHE}-part"; rm -rf "${PCACHE}"; mkdir -p "${PCACHE}"
run "${TMP}/ptc_part_cold" "${GRID}" --tessellation-cache "${PCACHE}" --partition 2 2 2
nfiles=$(ls -1 "${PCACHE}"/*.tess 2>/dev/null | wc -l | tr -d ' ')
check "$([ "$(hits)" = "0" ] && echo 1 || echo 0)" "cold partitioned run is all misses"
check "$([ "${nfiles}" = "8" ] && echo 1 || echo 0)" "one cache file per partition (got ${nfiles}, want 8)"
run "${TMP}/ptc_part_warm" "${GRID}" --tessellation-cache "${PCACHE}" --partition 2 2 2
check "$([ "$(hits)" = "8" ] && echo 1 || echo 0)" "warm partitioned run hits all 8 partitions"
# the partitions must NOT all be the same tessellation
ndistinct=$(echo "${RUN_OUT}" | grep -oE "Reused the cached tessellation \([0-9]+" | grep -oE "[0-9]+$" | sort -u | wc -l | tr -d ' ')
check "$([ "${ndistinct}" -ge "2" ] && echo 1 || echo 0)" "partitions restored DIFFERENT tessellations (${ndistinct} distinct vertex counts)"

echo ""
echo "(F) a partitioned cache hit reproduces the uncached run bit-for-bit"
run "${TMP}/ptc_part_ref" "${GRID}" --partition 2 2 2          # no cache at all
pf=0
for f in "${TMP}"/ptc_part_ref.*; do
    ext="${f##*ptc_part_ref}"
    g="${TMP}/ptc_part_warm${ext}"
    if [ -f "${g}" ] && cmp -s "${f}" "${g}"; then check 1 "bit-identical ${ext}"; else check 0 "bit-identical ${ext}"; pf=1; fi
done

echo ""
echo "(G) the STANDARD binary caches its processor-split sub-domains too"
# DTFE splits the box across threads and merges via copySubgridResultsToMain. Those sub-domains are
# NOT --partition: they are distinguished in the descriptor by their region, so each gets its own
# cache file. Only one 'Reused' line is printed (non-master threads run at verbosity 0), so the
# evidence that all of them hit is that the warm run adds no new files and is much faster.
SCACHE="${CACHE}-std"; rm -rf "${SCACHE}"; mkdir -p "${SCACHE}"
if [ -x ./DTFE ]; then
    ./DTFE "${SNAP}" "${TMP}/ptc_std_cold" --grid "${GRID}" --periodic --field density \
           --MpcUnit 1 --tessellation-cache "${SCACHE}" >/dev/null 2>&1 || true
    n_cold=$(ls -1 "${SCACHE}"/*.tess 2>/dev/null | wc -l | tr -d ' ')
    check "$([ "${n_cold}" -ge "2" ] && echo 1 || echo 0)" "standard binary wrote one file per sub-domain (${n_cold})"
    ./DTFE "${SNAP}" "${TMP}/ptc_std_warm" --grid "${GRID}" --periodic --field density \
           --MpcUnit 1 --tessellation-cache "${SCACHE}" >/dev/null 2>&1 || true
    n_warm=$(ls -1 "${SCACHE}"/*.tess 2>/dev/null | wc -l | tr -d ' ')
    check "$([ "${n_warm}" = "${n_cold}" ] && echo 1 || echo 0)" "warm run added no files (${n_warm} == ${n_cold}) => every sub-domain hit"
    # '.den' is the deterministic unaveraged deposit. ('.a_den' would NOT do here: its Monte-Carlo
    # seed is std::rand() unless --seed is given, so it differs between two UNCACHED runs as well.)
    if cmp -s "${TMP}/ptc_std_cold.den" "${TMP}/ptc_std_warm.den"; then
        check 1 "standard '.den' is bit-identical across the cache"
    else
        check 0 "standard '.den' is bit-identical across the cache"
    fi
    rm -rf "${SCACHE}"
else
    echo "   SKIP  (./DTFE not built)"
fi

echo ""
echo "------------------------------------------------------------"
if [ "${fails}" -eq 0 ]; then
    echo "ps_tessellation_cache_check: ALL PASS"
    rm -rf "${CACHE}" "${CACHE}-part"
    exit 0
fi
echo "ps_tessellation_cache_check: ${fails} FAILURE(S)"
exit 1
