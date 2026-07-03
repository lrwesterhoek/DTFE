# PS-DTFE Metal GPU acceleration — deposit kernels

Offloads the PS-DTFE grid **deposit** (the dominant cost, ~28 min/partition at nSub=3) to the Apple
GPU via **metal-cpp**, validated against the CPU path and analytic references.

Two kernels in `ps_deposit.metal`:
- `depositDensity` — mass only (the original prototype kernel).
- `depositFields` — mass + velocity moment + dispersion second moment (upper-triangle σ_ij) +
  velocity-gradient moment + stream count, all mass-share weighted exactly like
  `interpolateGrid_phaseSpace` in `src/CGAL_triangulation/ps_interpolation.cc`.
  (Scalar fields not yet ported; T-web/V-web are host-side post-processing of the gradient grid.)

## Files

| file | role |
|---|---|
| `ps_deposit.metal` | Metal compute kernels: one thread per tetrahedron, mass-conserving deposit with `atomic_float` scatter. Mirrors `src/CGAL_triangulation/ps_interpolation.cc`. |
| `validate_deposit.cpp` | Correctness harness: 17 tests — analytic edge cases, nSub sweep, determinism, random stress CPU-vs-GPU on every grid, and a sheared Zel'dovich pancake against an EXACT piecewise-linear pushforward reference (see below). `DENSITY_ONLY=1` gates to the density kernel; `BENCH=1` adds a 1M-tet timing; `DUMP_PROFILES=1` prints per-bin profiles. |
| `deposit_prototype.cpp` | Original density-only host + timing comparison. |
| `build_prototype.sh` | Offline-compiles the kernels to `ps_deposit.metallib` and builds both hosts. |
| `ps_deposit.metallib` | Offline-compiled kernels; hosts load it if present, else runtime-compile the `.metal`. Git-ignored. |

Dependency: **metal-cpp** headers, vendored (git-ignored) at `third_party/metal-cpp` — copied from
the Game Porting Toolkit volume (`/Volumes/Game Porting Toolkit/metal-cpp`). Metal-cpp is header-only
and standalone; the GPTK volume is only where they were found.

## Run

```
bash metal/build_prototype.sh
./metal/deposit_prototype metal/ps_deposit.metal [nTet]
```

## Validation results (Apple M1 Max, Metal 3) — 19/19 tests pass

Analytic velocity plumbing (T9): a constant field is reproduced exactly (v̄ ≡ v0 to 5e-6, σ ≡ 0,
gradient ≡ 0) and a linear field v = Gx recovers grad/W == G to 2e-5 in every covered cell
(well-conditioned tets; near-degenerate conditioning is covered by CPU-parity instead).

An adversarial 4-lens / 2-skeptic review workflow (22 agents) audited the kernels and this harness:
4 confirmed findings, all fixed — 64-bit moment-grid indexing (32-bit `flat*9` wrapped silently at
grids ≥ ~782³), a T2 that never actually reached the centroid fallback it claimed to test (tet sat
exactly on a sub-sample point; now off-lattice and asserts streams==1), and the missing analytic
velocity checks (now T9). Notable refuted-by-design caveats: zero-mass tets are skipped (production
deposits them with weight 0 into stream counts — irrelevant for physical masses), and the float32
raw-second-moment σ cancellation is inherited from production by design.

Edge cases (analytic): tet-in-one-cell mass placement, tiny-tet centroid fallback, periodic-wrap
shift equivalence (bit-exact), degenerate-tet drop, nSub∈{1..4} mass conservation (ratio 1.0000000),
determinism (~1e-6 atomic-order ULP).

Random stress set (60% normal + 25% flat/caustic + 15% tiny tets): CPU-vs-GPU mean-rel agreement
mass 4e-6, momentum 3e-6, second moment 7e-7, gradient 9e-9; stream counts differ in 0.012% of
cells by ±1 (FMA-order borderline samples).

Sheared Zel'dovich pancake (546k tets, 3-stream caustic flow) against an **exact** reference — for a
z-only displacement the Freudenthal interpolant is exactly the 1D piecewise-linear map, and the added
affine shear (αx+βy) makes the expected profile its pushforward convolved with two box kernels:
- mass conservation exact (1e-8); full grid coverage; density & stream slab profiles within
  **3.1% / 4.1%** of the exact expectation across ALL bins including caustics (nSub=3);
- tensor invariants exact: σ_xx…σ_yz ≡ 0 for the z-only flow (vs σ_zz ~ 6e6 (km/s)²);
  single-stream cells at nSub=1 have σ_zz ≤ 1.3 (km/s)² (float ULP floor — the same
  floor the production CPU pipeline shows);
- normalized σ and v̄ agree CPU-vs-GPU to ~1e-7 mean-rel.

Test-construction notes (matter if you modify the harness): a perfectly grid-commensurate tet
lattice puts sub-samples exactly on shared tet faces (double-counted streams via the ±1e-6
tolerance) and repeats one quantization pattern per slab (correlated moiré in profiles). The
harness decommensurates periods (N=45), offsets phases per axis, and adds the affine shear —
all while keeping the reference exact. At nSub=1, tets smaller than the sample spacing make
per-cell profiles quantization-noisy by construction (mass still conserved; informational only).

- **Speedup:** density-only ~18× vs one CPU core (1M tets: 61 s → 3.4 s); **all-fields ~23×**
  (1M tets, nSub=3: 243.5 s → 10.7 s). Compare against the OpenMP CPU baseline (cores actually
  used) before concluding net gain.

## Why the deposit (and not the triangulation)

The Delaunay triangulation is CGAL, sequential, and cheap (~35 s/partition). The deposit is
data-parallel over ~10M tetrahedra and dominates the runtime — the right GPU target.

## Pipeline integration (DONE)

Build with `make PS-DTFE METAL=1`, run with `--ps-metal`. Pieces:

- `src/CGAL_triangulation/ps_metal_host.{h,cc}` — metal-cpp host (singleton device/pipeline; the
  kernel source is embedded at build time via the generated `o_ps/ps_deposit_msl.h` and compiled
  once per process). Compiled only when `METAL=1` (`-DPS_METAL`); links `-framework Metal
  -framework Foundation`.
- `interpolateGrid_phaseSpace` (ps_interpolation.cc) extracts flat per-tet arrays (min-image-wrapped
  Eulerian vertices, vertex velocities, tet mass ρ̄·V_lag) with **exactly the CPU loop's filters**
  (ownership, hull, degeneracy), dispatches `depositFields`, and copies the moment grids back; the
  shared normalization/statistics tail is unchanged. Any failure (no device, compile, allocation)
  falls back to the CPU deposit with a warning. Scalar fields fall back to CPU.
- The kernel honours the **partition sub-grid** (`subOrigin/subDims` in `DepositParams`, matching
  the production `inSub` guard), so `--partition`/`--max-concurrent`/deferred-normalization work
  identically (validated: T10 + a partitioned A/B where stream counts came out bit-identical).

A/B parity (Zel'dovich 64³, all fields, single-domain and 2×2×2 partitioned): means identical,
mean-rel diffs ~1e-6 of peak; a handful of cells (6 of 262k) differ >1% from borderline
near-degenerate tets classified differently by the CPU's double-precision filter chain vs the
kernel's float chain — noise-level physics.

## Operational notes

- **GPU watchdog**: macOS kills command buffers when GPU pressure starves the display
  (`kIOGPUCommandBufferCallbackErrorImpactingInteractivity`). Empirically this reacts to SUSTAINED
  queue saturation, not just per-buffer duration: a pipelined (two-in-flight) submission was killed
  even with ~0.1 s buffers under active display use, while serialized submission with small
  host-side gaps survives. The host therefore submits SERIALIZED chunks that adapt toward ~0.25 s
  each (start 25k tets, `PS_METAL_CHUNK=<n>` overrides) with an ~8 ms sleep between buffers. A
  killed buffer may have partially deposited, so a failed chunk is never retried alone: each retry
  re-zeros the grids and redoes the whole partition with 4× shorter buffers and wider gaps (logged
  to stderr), then the CPU deposit takes over. Running with the display idle/locked avoids the
  watchdog entirely.
- **No mixed builds**: the PS build mode (METAL on/off) is stamped in `o_ps/`; changing it wipes all
  PS objects, so an incremental rebuild can never mix `-DPS_METAL` and plain objects (which would
  produce a binary that half-believes it has Metal support).

- **Full GPU utilization**: a chunk finishes when its slowest thread finishes, so mixing one
  stretched void tet (thousands of cells) with cheap halo tets idles most cores at every chunk
  boundary (~5× waste). The extraction therefore **cost-sorts** tets by grid footprint before
  dispatch — every chunk gets uniformly-sized work.
- **CPU ∥ GPU pipelining**: the partition loop is OpenMP over partitions; `--max-concurrent 2`
  (now the `run_ps_dtfe.sh` default) lets one partition triangulate on the CPU while another runs
  its GPU deposit (the Metal host mutex serializes GPU access). Measured steady-state:
  ~4–4.5 min per 512³ nSub=3 all-fields partition, ~45 min per full 8-partition run
  (vs ~13 min/partition unsorted-serialized, ~28 min/partition CPU-only).

## Remaining (optional)

1. Port the **scalar** fields (same pattern, 2 more accumulator grids).
2. Skip allocating unused moment grids (saves GPU memory when only density is requested).
3. Persistent GPU grids across partitions (skip per-partition readback/re-upload).
4. Split giant tets across multiple GPU threads (removes the cost tail entirely).

## Caveats

- Single GPU; no multi-GPU. Unified memory (`StorageModeShared`) avoids host↔device copies on Apple Si.
- Atomic-add ordering is non-deterministic → ULP-level run-to-run variation (fine for science, not
  bit-reproducible). For tighter determinism, accumulate via `atomic_uint` fixed-point.
- Requires Metal 3 (Apple7+/M-series) for native `atomic_float`. Verified on M1 Max.
- Fields grids at 512³ are 20 floats/cell ≈ 10.7 GB (mass 1 + momentum 3 + σ second-moment 6 +
  gradient 9 + streams 1). Fits M1 Max 64 GB unified memory; use the partition sub-grid (integration
  step 3) to shrink it.
- The m2 accumulator sums w·v_i·v_j in float32 (v ~ 10³ km/s ⇒ terms ~10⁶·w); fine at current scales
  (validated to 1e-7 vs CPU float), but consider Kahan or fixed-point if grids get much hotter.
