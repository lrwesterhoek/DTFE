# GPU acceleration — the deposit kernels (Metal; CUDA / HIP port)

Offloads the PS-DTFE grid **deposit** (the dominant cost, ~28 min/partition at nSub=3) to the Apple
GPU via **metal-cpp**, validated against the CPU path and analytic references.

One kernel in `ps_deposit.metal`, `depositFields`: mass + velocity moment + dispersion second
moment (upper-triangle σ_ij) + velocity-gradient moment + stream count, all mass-share weighted
exactly like `interpolateGrid_phaseSpace` in `src/CGAL_triangulation/ps_interpolation.cc`; a
density-only run switches the moment grids off. The scalar field (`scalar_a`, `scalarGradient_a`;
builds with one scalar component, `NO_SCALARS=1`, the default) is deposited like the velocity
moment -- the linear profile's value at the sample or at the exact piece's centroid, times the
moment weight -- into buffers 19/20 (the vertex scalars arrive in buffer 18); `tests/gpu_scalar_check.sh`
pins it for both binaries, sampled and exact. (T-web/V-web are host-side post-processing of the
gradient grid. The old density-only prototype kernel is gone.)

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
**The exact clip's frame** (2026-10-02). r3d's moments come from origin-apex simplices, so a clipped
piece far from the origin loses its volume to cancellation. Every GPU exact path (`depositExactItems`,
the one-thread exact section, `depositExactAverage`; Metal and CUDA) therefore clips with the origin at
the centre of the overlap of the tet's bounding box with the cell, where the piece lies, and shifts the
centroid back for the linear profiles (the covariance is taken about that origin, which also removes a
cancellation from the dispersion). In the tet's vertex-0 frame a zoom's fine cells were 1.6% off per cell
at 12 kpc and 20% at 3 kpc; in a cell-centre frame a small tet inside a large cell cancelled instead.
The overlap frame agrees with the CPU to about 1e-6 at every resolution (`tests/ps_window_check.sh`'s
fine-grid case, `tests/dtfe_metal_check.sh`'s fine region). The CPU keeps r3d in double in the vertex-0
frame (fine to about 1 kpc cells).

**`--ps-window`** (2026-10-02; the launcher's exact zoom on any snapshot) deposits only into a window of
the full grid. The sampled kernel visits every cell of a straddling tet's bounding box as before and
counts the samples outside the window for the tet's normalization (`window_flat` = 1: inside the grid,
outside the sub-box), depositing only the inside ones, so the window is the full run's cells to the
usual float parity. The exact items are built from the HULL of each tet's bounding box intersected with
the window's periodic images (`window_hull` in the kernel, `psWindowHull` in `gpu_host.h` for the host's
item bound -- keep them in sync) and the tet is normalized by its whole weight `wFull`, as a straddling
tet is: on the fine virtual grid of a zoom a void tet's bounding box holds millions of cells, nearly all
outside a one-slice window, and clipping them for the normalization would cost more than the whole full
run. A tet whose window pieces are all empty then means "no volume inside the window", not "degenerate":
`depositFields` returns without the sampled fallback under windowMode. The CPU loop does the same
(`tests/ps_window_check.sh`: the sampled CPU window identical bytes, the exact one to 1e-5 relative in
the moments and identical in the counts and masks, the GPU in its parity class; a 256x256x16 grid
windowed to one slice: 7 s full vs 0 s window, exact deposit).
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

## Standard DTFE (`make DTFE METAL=1`, `--gpu`): the volume-averaged `_a` interpolation

`metal/dtfe_deposit.metal` + `src/CGAL_triangulation/dtfe_metal_host.cc` port method 1 of
`averaged_interpolation_1.cc`: a tetrahedron inside one grid cell deposits its centroid value times
its volume; a larger one scatters value × V/n at n shared Sobol sample points (n = 100 per cell
volume, capped at 10 000). The CPU precomputes volumes, sample counts and the single-cell indices
with its own helpers, so both paths make the same decisions; the table of barycentric samples is
shared. `tests/dtfe_metal_check.sh` pins the parity (means to 1e-5, per-cell to 5e-2 outside a 1e-3
fraction -- float vs double sample placement flips a borderline sample now and then), a repeated GPU
run and the per-tet A/B kernel to 1e-4, and the shared grid against the per-call path on a coarse
grid where most tetrahedra take the single-cell fast path.

**The regime flip.** The original kernel, `depositAveraged1` (kept for A/B runs, `DTFE_GPU_ITEMS=0`),
is one thread per tetrahedron with one atomic per field per sample. That is fine while most
tetrahedra fit in one cell (7.1M particles on 256³: ~2.8 tetrahedra per cell), but at 512³ there are
0.35 per cell: nearly every tetrahedron takes the sampled path with hundreds of samples, the loops
reach 10 000 samples per thread and trip the GPU watchdog (whole sub-domain deposits restarted), and
the run was 10× SLOWER than the CPU's plain `+=` (397 s against 37 s, 2026-10-01).

**The item kernels** `depositAveraged1Items{2,4,8}` (2026-10-01; the default is the 2-slot one):
- one thread per **work item** = (tetrahedron, run of ≤ `itemSamples` samples, default 48). Items
  are implicit: `itemStart[t]` (a prefix sum) and a block table map a thread index to its tetrahedron
  with a short binary search, so there is no per-item buffer (180 M items of a 512³ sub-domain would
  be 1.4 GB). A tetrahedron's samples are dealt to its items round-robin, so the items of one
  tetrahedron -- adjacent lanes -- read consecutive entries of the (float4) sample table: one
  512-byte run per SIMD group instead of 32 scattered lines.
- a per-thread **cell cache** (`K` slots of flat index, Σden, Σvel, count, indexed only inside unrolled
  predicated loops so it stays in registers) sums the samples landing in the same cell; the weight
  V/n is applied once at the flush, the gradient (constant over the tetrahedron) as gradFlat × count
  × V/n.
- the flush is a **SIMD-group reduction by cell**: the host orders the tetrahedra along a Morton curve
  of their base cell (measured 1.0-1.3 distinct base cells per 32 consecutive items at 512³), each
  slot elects the smallest pending key with `simd_min`, `simd_sum`s the lanes holding it and lets one
  lane add -- one atomic per field per distinct cell per SIMD group.
- the kernels are **specialized per field set** with function constants (`FC_DEN/FC_VEL/FC_GRAD`): the
  density-only pipeline carries no velocity registers. The item kernels are register-bound (640 of
  1024 threads per threadgroup at K=4, 832 at K=2; the lean per-tet kernel 1024), which is why 2
  slots beat 4 and 8, and why shorter items once won: with the reduction and the coalesced table in
  place, 48-64 samples per item is the plateau.
- **evictions go to a threadgroup cell table.** In the void regime a tetrahedron spans hundreds of
  cells and its ~10 000 samples never repeat a cell within one item, so every sample evicts its
  cache slot -- and samples are proportional to volume, so most samples ARE void samples. Each
  cell is still hit ~20 times, but across the tetrahedron's ~200 items, which sit in the same
  threadgroup (consecutive items). A 32 KB threadgroup table (open addressing, linear probing,
  32-bit cell keys claimed by compare-exchange, threadgroup float atomics; 1024 entries for
  density + velocity, 512 with the nine gradient sums) collects the evictions, and the threadgroup
  flushes it once at its end with one device atomic per field per entry. A full probe sequence
  falls back to direct device atomics (`DTFE_GPU_TABLE=0` turns the table off). This was the
  decisive step for the gradient: with 13 device atomics per evicted sample the gradient
  pipeline ran at 320 M samples/s whatever the cache size or item length; the probe that found
  it skipped only those eviction atomics and jumped to 1.05 G.

What each step bought, on 2.1M particles at 512³ (GPU seconds of the largest sub-domain, 2.0 G
samples): per-tet kernel 45 s (with retries); items + cache 2.7 s (16 samples, K=4; atomics per
distinct cell per item); Morton order, SIMD reduction, coalesced table, specialization, K=2, 48
samples: 1.97 s (~1.05 G samples/s); + the threadgroup table: 0.7 s (2.4-2.9 G samples/s; the
gradient pipeline 3.1 s, 650-750 M). Lesson from the way there: the density-only run that first
pointed at "atomics" was confounded with the velocity registers, and the SIMD reduction, which
removed most FLUSH atomics, changed nothing because the EVICTION atomics were the cost -- measure
the branch you think is hot by skipping exactly that branch.

**Sub-domains: one grid or one each?** The OpenMP sub-domains of a standard-DTFE run each deposit
their own sub-grid, serialized on the one command queue (the host-side upload now runs outside the
mutex, so it overlaps the queue). The alternative is in place and tested -- `dtfeGpuShared*` in
gpu_host.h: one full grid on the device for the whole `DTFE_parallel` call, each thread's batch
streamed into it with its owner box (positions stay sub-domain-relative so the cell classification is
the CPU loop's bit for bit; the fast-path indices are shifted on the host), a warm chunk controller,
owner-box re-zeroing on a retry, one read-back added into the main grid (a sub-domain handed back to
the CPU has zeros there, so the add is exact) -- but it is **off by default** (`DTFE_GPU_SHARED=1`):
the same kernel runs at 450-780 M samples/s into the 2 GB full grid against 880-1100 M into the
sub-grids. The atomics' footprint on the big grid costs more in TLB and cache reach than the
per-sub-domain zeroing and read-back it saves; the Morton order recovered most of it (32.8 → 22.2 s
for the whole run) but not all (per-call 17-19 s). A brick-ordered full grid would be the next step
if sharing is ever needed. With the threadgroup table the gap narrowed but stayed: 2.1M at 512³
whole run 12.9 s shared against 10.6 s per-call.

**Double-precision binaries.** `make DTFE PS-DTFE DOUBLE=1 METAL=1` builds `DTFE-double` / `PS-DTFE-double`
with the same kernels: the extraction narrows the double tetrahedra to float for the GPU and the
float sums are added into the double grids (everything else in those binaries is double). Their
GPU deposits match their own CPU runs within the parity class above (`tests/ps_double_check.sh`, D5).

Knobs: `DTFE_GPU_ITEM_SAMPLES`, `DTFE_GPU_CACHE_K=2|4|8`, `DTFE_GPU_TABLE=0`, `DTFE_GPU_ITEMS=0`,
`DTFE_GPU_SHARED=1` (+ `DTFE_GPU_SHARED_GB` cap), `DTFE_METAL_TIMING=1` (per batch: items, samples,
table size, chunks, retries, M samples/s, pipeline occupancy, cells per SIMD group),
`DTFE_GPU_DEBUG_NOWRITE` / `DTFE_GPU_DEBUG_NOATOMIC=1|2|3` (timing experiments). Whole runs (CPU / GPU, 2026-10-01 evening): 7.1M particles at 512³ 39.1 / 17.3 s (with the velocity
gradient 48.2 / 32.0 s; the per-tet kernel of the morning: 396.7 s), 2.1M at 512³ 24.6 / 10.6 s
(gradient 30.9 / 26.3), 0.26M at 256³ 3.8 / 2.0 s (gradient 4.4 / 3.9), 7.1M at 256³ 14.2 / 14.4 s
(the floor: reading, triangulation, writing). The remaining gradient cost is the nine component
sums per table entry and per flush.

A retry (never seen with items) **drains the queue** before re-zeroing: the per-tet kernel's retried
512³ runs differed run to run by a whole sample in void cells (2e-2 of max|v|, against 3e-6 without
retries), consistent with atomics of a killed buffer landing after the memset; the parity test's
repeated GPU run pins the reproducibility at 1e-4.

## CUDA / HIP (Linux): `make <target> CUDA=1` or `HIP=1` — a port of these kernels

`src/CGAL_triangulation/ps_gpu_cuda.cu` and `dtfe_gpu_cuda.cu` are line-by-line ports of
`ps_deposit.metal` and `dtfe_deposit.metal` (every helper, the same float32 arithmetic, the same
deferral rule, the same two kernels per binary), with hosts that mirror `ps_metal_host.cc` and
`dtfe_metal_host.cc` (the chunked dispatch controller, the exact-deposit items + leftovers +
deferrals, the owner-box re-zero before a retry, the shared full-grid accumulator). One source
compiles under both `nvcc` (CUDA, NVIDIA) and `hipcc` (HIP, AMD ROCm) through the macro shim
`gpu_cuda_compat.h`, which also carries the warp collectives. The build is selected per target like
`METAL=1` (one backend per build; the mode stamp wipes the object directory on a switch; `.build_mode`
holds `CUDA=1` / `HIP=1`), `scripts/install.sh` picks it up when `nvcc` / `hipcc` is found, and the
launcher's Setup names the backend from the stamp. `--gpu` / `--ps-gpu` and the automatic CPU
fallback are the same.

**Status (2026-10-02): compiled, fallback-tested and validated by CPU emulation, not on hardware.**
The two objects compile cleanly with CUDA 12.6 (`nvidia/cuda:12.6.3-devel-ubuntu24.04`;
`depositFields` 128 registers / 1.8 KB stack, the K=2 item kernels 48–64 registers), the whole pair
links with `CUDA=1` in that image, and a run with the GPU flag in a container without a device falls
back to the CPU with the usual warning. The kernels and hosts, unchanged, pass every parity suite
through the CPU emulation described below (`CUDAEMU=1`), at warp width 32 and 64. No NVIDIA or AMD
machine was available, so the kernel numerics have NOT been checked on real hardware. **First thing
to run on a GPU machine:**
`GPU_BUILD=CUDA=1 tests/dtfe_metal_check.sh` (CPU/GPU parity of the standard DTFE, the repeated run,
the per-tet A/B kernel, the shared grid), then the PS suites that take `--ps-gpu` when the stamp is a
GPU one (`tests/ps_linear_deposit_check.sh`, `ps_vertex_mass_check.sh`, `ps_volume_weighted_check.sh`,
`ps_halo_release_check.sh`, `ps_hidden_streams_check.sh`, `ps_caustic_class_check.sh`,
`python3 tests/ps_nonperiodic_test.py`) and `tests/ps_double_check.sh` for the double pair.
`metal/validate_deposit.cpp` is Metal-only. The CI job `cuda-compile` keeps the port compiling.

What is different from the Metal backend, by design:

- **Warp width.** Metal's `simd_*` operations act on 32 lanes; the port uses `__shfl_xor_sync` /
  `__any_sync` (32 lanes) under CUDA and `__shfl_xor` / `__any` over `warpSize` (64 on CDNA/GCN,
  32 on RDNA) under HIP. The reduction by cell is the same algorithm; only the group over which the
  sums are formed changes, which only reorders the float additions.
- **Every lane reaches the reduction.** Metal lets inactive lanes drop out of a SIMD operation; a
  CUDA warp collective needs all 32. A thread beyond the chunk's last item therefore runs the item
  body with nothing pending (`valid = false`) instead of returning, and no `return` precedes the two
  `__syncthreads()` of the item kernel.
- **Field-set specialization** is a template (`depositItemsKernel<K, FD, FV, FG>`, all 24
  instantiations compiled ahead of time) instead of Metal function constants; the cell table is
  dynamic shared memory sized by the host (32 KB budget, as the threadgroup memory).
- **Memory.** Device memory and explicit copies instead of unified buffers read in place: the exact
  deposit's item windows are bounded from the host copy of the records before it is uploaded, the
  masses keep a 4-byte-per-tet host mirror (leftovers and deferrals are read from it; the compact
  leftovers are gathered on the device by a small kernel), and the shared full grid is copied back to
  host vectors at `Result`. The chunk controller is kept because a display-attached Linux GPU kills
  kernels after a few seconds; the idle gap between kernels (the Metal host's concession to the window
  server) is zero. A CUDA kernel timeout is a *sticky* error: the retries then fail as well and the
  caller's CPU fallback takes over for the rest of the process.
- **Knobs** keep their names (`DTFE_GPU_ITEMS`, `DTFE_GPU_ITEM_SAMPLES`, `DTFE_GPU_CACHE_K`,
  `DTFE_GPU_TABLE`, `DTFE_GPU_SHARED[_GB]`, `PS_EXACT_ITEMS`, `PS_EXACT_ITEM_CELLS`); the timing and
  chunk knobs accept both spellings (`DTFE_METAL_TIMING` or `DTFE_GPU_TIMING`, `PS_METAL_CHUNK` or
  `PS_GPU_CHUNK`, `DTFE_METAL_CHUNK` or `DTFE_GPU_CHUNK`).
- **Compiler flags.** `-O3` only: Metal compiles its kernels with fast math on, and nvcc's default
  fma contraction covers what the certified inside test assumes, but `--use_fast_math` would make
  `sqrt` and division approximate and break the degeneracy sentinel and the exact clipping. `GPU_ARCH`
  pins an architecture (`sm_86`; for HIP it is practically required, `gfx90a`, because hipcc has no
  portable intermediate code).

## Standard DTFE: `--exact-average` on the GPU (`depositExactAverage`)

The analytic volume average (r3d, `--exact-average`) runs on the GPU as of 2026-10-02: one thread per
item = (tetrahedron, block of 16 cells of its bounding-box window), the phase-space exact items'
split, each piece depositing the linear interpolant's centroid value times the piece volume (order-1
moments) and the gradients times the piece volume -- no per-tetrahedron normalization, one dispatch;
single-cell tetrahedra take the centroid fast path by their only item. The r3d port now lives in
`metal/exact_clip.metal.inc`, which the Makefile prepends to BOTH kernel sources when it embeds them
(and `validate_deposit.cpp` prepends when it loads the source); the CUDA twin is
`src/CGAL_triangulation/gpu_exact_clip.cuh`. `tests/dtfe_metal_check.sh` pins the CPU/GPU pair: with
no sampling involved it agrees to float rounding (density 5e-7, velocity 4e-6, gradient 8e-6 peak
relative on the 96³ test box). 0.26M particles on 256³: CPU 26.9 s, GPU 4.1 s (the sampled average
3.3 vs 1.9 s). `DTFE_GPU_EXACT_CELLS` sets the block size. The shared full grid stays sampled-only.

## Validation without hardware: `make <target> CUDAEMU=1`

The CUDA/HIP sources can be compiled as plain C++ against a CPU emulation of the device API
(`src/CGAL_triangulation/gpu_cuda_emu.h`, selected by `-DGPU_EMU`; a C++20 translation unit): a
launch runs the grid one block at a time, a block's warps are OS threads and each warp's lanes are
coroutines (a register-only stack switch; arm64 verified, the x86-64 switch written but not yet run) run in lockstep at the collectives -- a
lane that calls a warp collective or `__syncthreads()` yields, and the warp's scheduler resumes it
with the result once every lane has arrived there. Non-uniform control flow (lanes of a warp at
different collectives, a lane exited before one) aborts with a message instead of hanging silently;
`GPU_EMU_WARP=64` runs the reductions over 64 lanes, the HIP wavefront width. Atomics are real
atomics, device memory is host memory, every runtime call succeeds, the device reports "CPU emulation".
It validates the port's transliteration and host logic, NOT the memory model, scheduling or numerics
of real hardware. On 2026-10-02 the UNCHANGED kernels and hosts passed, through this emulation, every
suite the Metal backend passes -- `dtfe_metal_check` (parity, repeat, per-tet A/B, shared grid, exact
average), `gpu_scalar_check`, the PS linear-deposit / vertex-mass / volume-weighted / halo-release /
hidden-streams / caustic-class / non-periodic / point-exact-serve suites and the smoke test -- at
warp 32 and at warp 64, with the same parity figures as the Metal backend. A run on an NVIDIA or AMD
machine remains the final word.

## Remaining (optional)

1. Skip allocating unused moment grids (saves GPU memory when only density is requested).
2. Persistent GPU grids across partitions (skip per-partition readback/re-upload).
3. Split giant tets of the SAMPLED phase-space deposit across threads (the exact deposit and the
   exact average already split below the tetrahedron).

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
