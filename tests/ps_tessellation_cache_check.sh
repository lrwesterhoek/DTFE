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
#  G) The standard binary caches its thread sub-domains too.
#  H) New files carry the CHUNKED body; single-stream files still load; the converter rewrites them.
#  I) The block-wise triangulation reader builds exactly CGAL's triangulation (structure hashes),
#     and the block-wise writer writes exactly CGAL's bytes.
#  J) A symlinked or relative name of the input hits the cache its absolute name wrote (canonical keys).
#  K) libdeflate (when compiled in) and zlib read each other's chunked files.
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
source "${SCRIPT_DIR}/precision.sh"     # DTFE_TEST_PRECISION=double: the double pair

N="${N:-24}"; GRID="${GRID:-48}"; BOX="${BOX:-100.0}"
BIN="${PS_BIN}"
precision_require "${BIN}"
TMP="${SCRIPT_DIR}/tmp"; mkdir -p "${TMP}"
SNAP="${TMP}/ptc_input.hdf5"
# The cache must live on a LOCAL volume -- the repo itself is in iCloud, which the binary rejects.
# ${TMPDIR:-/tmp} keeps this working on Linux CI, where /private/tmp does not exist and the binary
# would refuse the path for not being a directory.
CACHE="${TESS_CACHE_DIR:-${TMPDIR:-/tmp}/dtfe-tesscache-test}"

echo "============================================================"
echo " PS-DTFE --tessellation-cache check   N=${N}^3  grid=${GRID}^3"
echo "============================================================"

if [ "${1:-}" != "--no-build" ] && ! precision_double; then
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
if [ -x "${DTFE_BIN}" ]; then
    "${DTFE_BIN}" "${SNAP}" "${TMP}/ptc_std_cold" --grid "${GRID}" --periodic --field density \
           --MpcUnit 1 --tessellation-cache "${SCACHE}" >/dev/null 2>&1 || true
    n_cold=$(ls -1 "${SCACHE}"/*.tess 2>/dev/null | wc -l | tr -d ' ')
    check "$([ "${n_cold}" -ge "2" ] && echo 1 || echo 0)" "standard binary wrote one file per sub-domain (${n_cold})"
    "${DTFE_BIN}" "${SNAP}" "${TMP}/ptc_std_warm" --grid "${GRID}" --periodic --field density \
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
    echo "   SKIP  (${DTFE_BIN} not built)"
fi

echo ""
echo "(H) the CHUNKED body (parallel inflate), the single-stream format before it, and the converter"
tessfile=$(ls -1 "${CACHE}"/*.tess 2>/dev/null | head -1)
chunked=$("${PY}" -c "
import struct, sys
f = open(sys.argv[1], 'rb'); f.read(8); (n,) = struct.unpack('<I', f.read(4)); f.read(n)
sys.stdout.write('1' if f.read(8) == b'DTFECHK1' else '0')
" "${tessfile}")
check "${chunked}" "a new cache file carries the chunked body (marker after the descriptor)"
LCACHE="${CACHE}-legacy"; rm -rf "${LCACHE}"; mkdir -p "${LCACHE}"
export DTFE_TESS_CACHE_LEGACY=1
run "${TMP}/ptc_legacy_write" "${GRID}" --tessellation-cache "${LCACHE}"
unset DTFE_TESS_CACHE_LEGACY
run "${TMP}/ptc_legacy_hit" "${GRID}" --tessellation-cache "${LCACHE}"
check "$([ "$(hits)" -ge 1 ] && echo 1 || echo 0)" "a single-stream (older) cache file still loads"
cmp -s "${TMP}/ptc_out_nocache.den" "${TMP}/ptc_legacy_hit.den" && check 1 "... bit-identically" || check 0 "... bit-identically"
"${PY}" "${ROOT}/python/tools/upgrade_tess_cache.py" "${LCACHE}" -j 2 > "${TMP}/ptc_upgrade.log" 2>&1
check "$([ $? -eq 0 ] && grep -q '^converted' "${TMP}/ptc_upgrade.log" && echo 1 || echo 0)" "the converter rewrites the older file as chunks"
run "${TMP}/ptc_conv_hit" "${GRID}" --tessellation-cache "${LCACHE}"
[ "$(hits)" -ge 1 ] && cmp -s "${TMP}/ptc_out_nocache.den" "${TMP}/ptc_conv_hit.den" \
    && check 1 "the converted file loads, bit-identically" || check 0 "the converted file loads, bit-identically"
rm -rf "${LCACHE}"

echo ""
echo "(I) the block-wise reader and writer: exactly CGAL's triangulation, exactly CGAL's bytes"
# tessellation_cache.h reads the triangulation in blocks (readTriangulationBlocks) instead of through
# CGAL's operator>> (3.4x faster loads). DTFE_TESS_CACHE_CGAL_IO=1 restores CGAL's reader, and
# DTFE_TESS_CACHE_FINGERPRINT=1 prints a hash of the loaded STRUCTURE (points, every vertex's stored
# cell, every cell's vertices and neighbours, in iteration order). Same files, both readers: the same
# hashes and bit-identical outputs, for one tessellation, for 8 partitions, and for the standard
# binary's sub-domains (its hierarchy triangulation shares the format).
fprints() { echo "${RUN_OUT}" | grep "^TESS_FINGERPRINT" | sort || true; }
export DTFE_TESS_CACHE_FINGERPRINT=1
for which in single part; do
    if [ "${which}" = "single" ]; then dir="${CACHE}"; extra=(); else dir="${PCACHE}"; extra=(--partition 2 2 2); fi
    export DTFE_TESS_CACHE_CGAL_IO=1
    run "${TMP}/ptc_rd_cgal_${which}" "${GRID}" --tessellation-cache "${dir}" ${extra[@]+"${extra[@]}"}
    unset DTFE_TESS_CACHE_CGAL_IO
    fp_cgal="$(fprints)"; h_cgal=$(hits)
    run "${TMP}/ptc_rd_block_${which}" "${GRID}" --tessellation-cache "${dir}" ${extra[@]+"${extra[@]}"}
    fp_block="$(fprints)"; h_block=$(hits)
    nfp=$(echo "${fp_block}" | grep -c . || true)
    check "$([ "${h_cgal}" -ge 1 ] && [ "${h_cgal}" = "${h_block}" ] && [ "${nfp}" = "${h_block}" ] && echo 1 || echo 0)" \
          "${which}: both readers hit (${h_cgal} / ${h_block}) and fingerprint every load (${nfp})"
    check "$([ -n "${fp_block}" ] && [ "${fp_cgal}" = "${fp_block}" ] && echo 1 || echo 0)" \
          "${which}: identical structure fingerprints"
    same=1
    for f in "${TMP}"/ptc_rd_cgal_${which}.*; do
        ext="${f##*ptc_rd_cgal_${which}}"
        cmp -s "${f}" "${TMP}/ptc_rd_block_${which}${ext}" || same=0
    done
    check "${same}" "${which}: bit-identical outputs, sample points included"
done
if [ -x "${DTFE_BIN}" ]; then
    SCACHE="${CACHE}-std"; rm -rf "${SCACHE}"; mkdir -p "${SCACHE}"
    "${DTFE_BIN}" "${SNAP}" "${TMP}/ptc_std_rd0" --grid "${GRID}" --periodic --field density --MpcUnit 1 \
           --tessellation-cache "${SCACHE}" >/dev/null 2>&1 || true
    fp_cgal="$(DTFE_TESS_CACHE_CGAL_IO=1 "${DTFE_BIN}" "${SNAP}" "${TMP}/ptc_std_rd1" --grid "${GRID}" --periodic \
               --field density --MpcUnit 1 --tessellation-cache "${SCACHE}" 2>&1 | grep "^TESS_FINGERPRINT" | sort || true)"
    fp_block="$("${DTFE_BIN}" "${SNAP}" "${TMP}/ptc_std_rd2" --grid "${GRID}" --periodic \
               --field density --MpcUnit 1 --tessellation-cache "${SCACHE}" 2>&1 | grep "^TESS_FINGERPRINT" | sort || true)"
    check "$([ -n "${fp_block}" ] && [ "${fp_cgal}" = "${fp_block}" ] && echo 1 || echo 0)" \
          "standard binary: identical structure fingerprints ($(echo "${fp_block}" | grep -c . || true) sub-domains)"
    cmp -s "${TMP}/ptc_std_rd1.den" "${TMP}/ptc_std_rd2.den" && check 1 "standard binary: bit-identical '.den'" \
                                                            || check 0 "standard binary: bit-identical '.den'"
    SCACHE2="${CACHE}-std2"; rm -rf "${SCACHE2}"; mkdir -p "${SCACHE2}"
    DTFE_TESS_CACHE_CGAL_IO=1 "${DTFE_BIN}" "${SNAP}" "${TMP}/ptc_std_wr" --grid "${GRID}" --periodic --field density \
           --MpcUnit 1 --tessellation-cache "${SCACHE2}" >/dev/null 2>&1 || true
    same=1; nw=0
    for f in "${SCACHE2}"/*.tess; do nw=$((nw+1)); cmp -s "${f}" "${SCACHE}/$(basename "${f}")" || same=0; done
    check "$([ "${nw}" -ge 2 ] && [ "${same}" = 1 ] && echo 1 || echo 0)" "standard binary: the block writer's ${nw} sub-domain files are byte-identical to CGAL's"
    rm -rf "${SCACHE}" "${SCACHE2}"
fi
unset DTFE_TESS_CACHE_FINGERPRINT
# ... and the block-wise WRITER writes CGAL's bytes: the same run cached by each writer gives
# byte-identical files (the chunks deflate independently, so equal content means equal files)
for which in single part; do
    if [ "${which}" = "single" ]; then extra=(); else extra=(--partition 2 2 2); fi
    WA="${CACHE}-wcgal"; WB="${CACHE}-wblock"; rm -rf "${WA}" "${WB}"; mkdir -p "${WA}" "${WB}"
    export DTFE_TESS_CACHE_CGAL_IO=1
    run "${TMP}/ptc_wr_cgal" "${GRID}" --tessellation-cache "${WA}" ${extra[@]+"${extra[@]}"}
    unset DTFE_TESS_CACHE_CGAL_IO
    run "${TMP}/ptc_wr_block" "${GRID}" --tessellation-cache "${WB}" ${extra[@]+"${extra[@]}"}
    same=1; nw=0
    for f in "${WA}"/*.tess; do
        nw=$((nw+1))
        cmp -s "${f}" "${WB}/$(basename "${f}")" || same=0
    done
    check "$([ "${nw}" -ge 1 ] && [ "${same}" = 1 ] && echo 1 || echo 0)" "${which}: the block writer's ${nw} cache file(s) are byte-identical to CGAL's"
    rm -rf "${WA}" "${WB}"
done

echo ""
echo "(J) one key per FILE: a symlinked or relative name of the input hits the cache its absolute name wrote"
# The descriptor stores the input's realpath (canonical_path.h). Before, '/tmp/x' and '/private/tmp/x' or
# 'x' run from its folder were three keys, i.e. two needless rebuilds of the same tessellation.
LINKDIR="${TMP}_link"; rm -f "${LINKDIR}"; ln -s "${TMP}" "${LINKDIR}"
nbefore=$(ls -1 "${CACHE}"/*.tess 2>/dev/null | wc -l | tr -d ' ')
RUN_OUT="$( "${BIN}" "${LINKDIR}/$(basename "${SNAP}")" "${TMP}/ptc_sym" --grid "${GRID}" "${COMMON[@]}" \
            --tessellation-cache "${CACHE}" --verbose 3 2>&1 || true )"
check "$([ "$(hits)" -ge 1 ] && echo 1 || echo 0)" "a path through a symlinked folder HITS"
ABSBIN="$(pwd)/${PS_BIN#./}"
RUN_OUT="$( cd "${TMP}" && "${ABSBIN}" "$(basename "${SNAP}")" "${TMP}/ptc_rel" --grid "${GRID}" "${COMMON[@]}" \
            --tessellation-cache "${CACHE}" --verbose 3 2>&1 || true )"
check "$([ "$(hits)" -ge 1 ] && echo 1 || echo 0)" "a relative path HITS"
same=1
for f in "${TMP}"/ptc_out_hit.*; do
    ext="${f##*ptc_out_hit}"
    for g in "${TMP}/ptc_sym${ext}" "${TMP}/ptc_rel${ext}"; do cmp -s "${f}" "${g}" || same=0; done
done
check "${same}" "... both bit-identical to the absolute-path hit"
nafter=$(ls -1 "${CACHE}"/*.tess 2>/dev/null | wc -l | tr -d ' ')
check "$([ "${nafter}" = "${nbefore}" ] && echo 1 || echo 0)" "and no new cache file was written (${nbefore} -> ${nafter})"
rm -f "${LINKDIR}"

echo ""
echo "(K) the chunk codec: libdeflate and zlib read each other's files (both write plain gzip members)"
run "${TMP}/ptc_codec_probe" "${GRID}" --tessellation-cache "${CACHE}"
if echo "${RUN_OUT}" | grep -q "read with libdeflate"; then
    for w in zlib libdeflate; do
        KC="${CACHE}-codec-${w}"; rm -rf "${KC}"; mkdir -p "${KC}"
        if [ "${w}" = zlib ]; then export DTFE_TESS_CACHE_ZLIB=1; fi
        run "${TMP}/ptc_codec_w_${w}" "${GRID}" --tessellation-cache "${KC}"
        unset DTFE_TESS_CACHE_ZLIB
        if [ "${w}" = libdeflate ]; then export DTFE_TESS_CACHE_ZLIB=1; fi
        run "${TMP}/ptc_codec_r_${w}" "${GRID}" --tessellation-cache "${KC}"
        reader=$(echo "${RUN_OUT}" | grep -o "read with [a-z]*" | head -1 | cut -d' ' -f3)
        unset DTFE_TESS_CACHE_ZLIB
        same=1
        for f in "${TMP}"/ptc_out_nocache.*; do
            ext="${f##*ptc_out_nocache}"; cmp -s "${f}" "${TMP}/ptc_codec_r_${w}${ext}" || same=0
        done
        check "$([ "$(hits)" -ge 1 ] && [ "${reader}" != "${w}" ] && [ "${same}" = 1 ] && echo 1 || echo 0)" \
              "written with ${w}, read with ${reader}: a hit, bit-identical to the uncached run"
        rm -rf "${KC}"
    done
else
    echo "   SKIP  (this build has no libdeflate: every chunk goes through zlib)"
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
