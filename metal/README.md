# PS-DTFE Metal GPU acceleration — deposit kernels

Offloads the PS-DTFE grid **deposit** (the dominant cost, ~28 min/partition at nSub=3) to the Apple
GPU via **metal-cpp**, validated against the CPU path and analytic references.

One kernel in `ps_deposit.metal`, `depositFields`: mass + velocity moment + dispersion second
moment (upper-triangle σ_ij) + velocity-gradient moment + stream count, all mass-share weighted
exactly like `interpolateGrid_phaseSpace` in `src/CGAL_triangulation/ps_interpolation.cc`; a
density-only run switches the moment grids off. (Scalar fields not yet ported; T-web/V-web are
host-side post-processing of the gradient grid. The old density-only prototype kernel is gone.)

**Which samples a tetrahedron contains is decided exactly as on the CPU**
(`src/CGAL_triangulation/ps_exact_inside.h`). Metal has no double, so the kernel classifies each
sample in float with a rigorous error bound, from a per-tet record the host builds in the exact
test's canonical frame (vertex 0 as a cell index plus an in-cell fraction, the edges as exact
differences; `psFillTetRecord` in `ps_deposit_params.h`). Outside the bound the float decision
provably equals the exact one. A tet with a sample inside it (within float rounding of a face;
2291 of 14.7M at nSub=1, 57k at nSub=3) is **deferred**: the kernel negates its mass and deposits
nothing, and the host deposits it with the CPU's exact test. Stream counts and the sub-sample mass flag
(`.hidden_streams` bit 2) are then byte-identical CPU vs GPU, partitioned or not.

**The exact deposit (`--ps-exact-deposit`) runs on a second kernel, `depositExactItems`**, which
splits the work below the tetrahedron: one thread per *item* = (tet, block of 16 cells of its
bounding-box window). With one thread per tet the per-thread cost spans five orders of magnitude
once cells are much smaller than tets (a stretched void tet clips ~10⁴ cells, a halo tet one), each
buffer waited on its few giants and the dispatch controller shrank buffers to 50 tets: 0.26M
particles on 256³ ran for more than 17 minutes. Two dispatches over all items: phase 0 adds each
item's clipped weights into a per-tet total `sumW[tet]` (atomic), phase 1 deposits each cell with
the share m/sumW[tet], so every tet still deposits exactly its mass. Tets whose total stays 0
(float-degenerate, or a window that clips empty or lies outside the grid) go through
`depositFields`, which falls through to the sampled path and centroid fallback or defers them,
exactly as before. Both kernels skip a window cell wholly outside one of the tet's face planes
before clipping it. The host builds the items from an upper bound of each window (the kernel
re-derives the window in float; surplus items idle) and caps the list at 32M items (256 MB).
`PS_EXACT_ITEMS=0` keeps the one-thread path (A/B), `PS_EXACT_ITEM_CELLS` sets the block size
(16–32 run within 10% of each other), `PS_METAL_TIMING=1` prints one line per kernel run.
On a non-periodic grid a tetrahedron cut by the grid face keeps only its inside share: the exact path
normalizes it by its whole weight, the sampled path counts its samples beyond the grid too
(`count_beyond_grid`: one sample column at a time — the certified numerators are affine along a
column, so the samples clearing the bound on all four faces are one run found by a seeded binary
search, and only those within the bound of a face are classified one by one; the count, and the
deferral decisions, are the per-sample ones). Crossed waves, 0.26M particles on 256³: more than 17 min → 30 s; the 11.8M-particle TNG100-3 region
on 256³: more than 25 min → 133 s; 128³ unchanged (about 15 s). Against the one-thread path: every
grid within ~1e-7 (atomic order), tet-touch counts identical (T14).

## Files

| file | role |
|---|---|
| `ps_deposit.metal` | Metal compute kernels: `depositFields` (one thread per tetrahedron, mass-conserving deposit with `atomic_float` scatter; mirrors `src/CGAL_triangulation/ps_interpolation.cc`) and `depositExactItems` (the exact deposit, one thread per block of a tet's window cells). |
| `validate_deposit.cpp` | Correctness harness (T1–T14): analytic edge cases, nSub sweep, determinism, random stress CPU-vs-GPU on every grid, analytic velocity-field checks, partition sub-grid support, the flag paths, a sheared Zel'dovich pancake against an EXACT piecewise-linear pushforward reference (see below), and `depositExactItems` against the analytic cube and the one-thread path (T14: periodic, wrapped sub-grid, non-periodic). It builds the per-tet records like the host and deposits deferred tets with its CPU reference. `BENCH=1` adds a 1M-tet timing; `DUMP_PROFILES=1` prints per-bin profiles. |
| `build_prototype.sh` | Builds the harness; offline-compiles the kernel to `ps_deposit.metallib` when Xcode's Metal toolchain is installed, else deletes any stale one (the harness would load it) and the harness compiles the source at run time. |
| `ps_deposit.metallib` | Offline-compiled kernels; hosts load it if present, else runtime-compile the `.metal`. Git-ignored. |

Dependency: **metal-cpp** headers, vendored at `third_party/metal-cpp` — copied from
the Game Porting Toolkit volume (`/Volumes/Game Porting Toolkit/metal-cpp`). Metal-cpp is header-only
and standalone; the GPTK volume is only where they were found.

## Run

```
bash metal/build_prototype.sh
./metal/validate_deposit
```

## Validation results (Apple M1 Max, Metal 3) — all T1–T10 pass

Analytic velocity plumbing (T9): a constant field is reproduced exactly (v̄ ≡ v0 to 5e-6, σ ≡ 0,
gradient ≡ 0) and a linear field v = Gx recovers grad/W == G to 2e-5 in every covered cell
(well-conditioned tets; near-degenerate conditioning is covered by CPU-parity instead).

Edge cases (analytic): tet-in-one-cell mass placement, tiny-tet centroid fallback, periodic-wrap
shift equivalence (bit-exact), degenerate-tet drop, nSub∈{1..4} mass conservation (ratio 1.0000000),
determinism (~1e-6 atomic-order ULP).

Random stress set (60% normal + 25% flat/caustic + 15% tiny tets): CPU-vs-GPU mean-rel agreement
mass 4e-6, momentum 3e-6, second moment 7e-7, gradient 9e-9; stream counts differ in 0.003% of
cells by ±1 -- the harness's CPU REFERENCE still uses a float test with a ±1e-6 tolerance; the
production CPU deposit and the kernel agree exactly (tests/point_exact_serve_check.sh D3/D4).

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

Build with `make PS-DTFE METAL=1`, run with `--ps-gpu`. Pieces:

- `src/CGAL_triangulation/ps_metal_host.cc` — metal-cpp host implementing the backend-neutral
  `gpu_host.h` interface (singleton device/pipeline; the kernel source is embedded at build time
  via the generated `o_ps/ps_deposit_msl.h` and compiled once per process). Compiled only when
  `METAL=1` (which defines `-DPS_GPU`); links `-framework Metal -framework Foundation`.
- `interpolateGrid_phaseSpace` (ps_interpolation.cc) extracts flat per-tet arrays (the 16-float
  record above, vertex velocities, tet mass ρ̄·V_lag) with **exactly the CPU loop's filters**
  (ownership, hull, degeneracy), dispatches `depositFields` (the exact deposit: `depositExactItems`,
  then `depositFields` on its leftovers), copies the moment grids back, and then
  runs the CPU loop over just the deferred tets' cells (their finite-cell ordinals travel through
  the cost sort with the records); the shared normalization/statistics tail is unchanged. Any
  failure (no device, compile, allocation) falls back to the CPU deposit with a warning. Scalar
  fields fall back to CPU.
- The kernel honours the **partition sub-grid** (`subOrigin/subDims` in `DepositParams`, matching
  the production `inSub` guard), including a box that WRAPS a periodic axis, so
  `--partition`/`--max-concurrent`/deferred-normalization work identically. (The sampled path
  missed that wrap from 2026-07-16 to 2026-09-28: in a periodic `--partition` run on the GPU most
  of each partition's cells were skipped and their mass went to centroid fallbacks -- 91% of cells
  wrong on crossed waves at 2×2×2. `tests/point_exact_serve_check.sh` D4 now guards it.)

A/B parity (crossed waves 128³, single-domain and 2×2×2 partitioned, nSub 1 and 3): `.streams`
and `.hidden_streams` byte-identical, float fields within ~1e-6 of peak (atomic summation order). A
tet the kernel's float degeneracy check would reject but the host's double one kept is deferred,
not dropped, so no mass is lost there either.

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
- **No mixed builds**: the PS build GPU mode (metal/off) is stamped in `o_ps/`; changing
  it wipes all PS objects, so an incremental rebuild can never mix `-DPS_GPU` and plain objects
  (which would produce a binary that half-believes it has GPU support). `o_ps/.build_mode` records
  the mode's make argument so tests can rebuild without downgrading the binary.

- **Full GPU utilization**: a chunk finishes when its slowest thread finishes, so mixing one
  stretched void tet (thousands of cells) with cheap halo tets idles most cores at every chunk
  boundary (~5× waste). The extraction therefore **cost-sorts** tets by grid footprint before
  dispatch — every chunk gets uniformly-sized work.
- **CPU ∥ GPU pipelining**: the partition loop is OpenMP over partitions; `--max-concurrent 2`
  (what the binary's auto-tuner targets under `--ps-gpu` when memory allows; set
  `MAX_CONCURRENT` to override) lets one partition triangulate on the CPU while another runs
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
  (validated to 1e-7 vs CPU float), but consider Kahan or fixed-point if grids get much hotter. The
  raw-second-moment σ cancellation this implies is inherited from the production CPU path by design.
- The harness skips zero-mass tets (production deposits them with weight 0 into stream counts —
  irrelevant for physical masses).
