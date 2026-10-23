# Command cheatsheet

Every script in the repo with its useful flag combinations. All paths are relative to the
repository root; quote paths that contain spaces. Scripts that need h5py should be
run with a python whose h5py works (the shell scripts and tests pick one: `PYTHON`, else the first of `python3`,
Homebrew's and `/usr/local`'s that imports numpy and h5py).
The binaries' `--full_help` and each script's `--help` are the reference for every option and default.

---

## Build (`Makefile`, `scripts/install.sh`)

```bash
./scripts/install.sh              # one-command clean rebuild, best backend auto-detected
                                  #   (Apple Silicon -> METAL=1; Linux with nvcc / hipcc -> CUDA=1 / HIP=1; else CPU-only);
                                  #   a double or 2D pair already built is rebuilt too ('make clean' deletes every set)
./scripts/install.sh --cpu        # force CPU-only binaries
./scripts/install.sh --no-deps    # skip dependency installation, just clean + rebuild
./scripts/install.sh --jobs 4     # limit parallel build jobs
./scripts/install.sh --docker     # containerized LINUX image instead (never Metal)
./scripts/install.sh --double     # ALSO the double-precision pair (DOUBLE=1, same GPU backend)
./scripts/install.sh --no-python  # skip the 'pip install -e python' step (dtfelib)
./scripts/install.sh --no-gui     # skip the launcher's PySide6 environment (e.g. on a cluster)
python3 -m pip install -e python  # dtfelib alone (editable: finds ./DTFE, ./PS-DTFE itself;
                                  #   a non-editable install needs DTFE_BIN_DIR or $PATH)

make DTFE METAL=1 -j10            # standard DTFE, Apple-GPU '_a' interpolation backend
make PS-DTFE METAL=1 -j10         # phase-space DTFE, Apple-GPU deposit backend
make DTFE PS-DTFE CUDA=1 -j10     # Linux/NVIDIA: the same kernels ported to CUDA (nvcc; validated through the
                                  #   CPU emulation below, not yet on a GPU)
make DTFE PS-DTFE HIP=1 GPU_ARCH=gfx90a   # Linux/AMD ROCm (hipcc; one backend per build)
make DTFE PS-DTFE CUDAEMU=1       # the CUDA/HIP sources as plain C++ on a CPU emulation of the device API:
                                  #   a validation build for machines without a GPU; tests/cudaemu_check.sh runs
                                  #   the GPU suites on it (GPU_EMU_WARP=64 at run time: the HIP wavefront width)
make DTFE PS-DTFE DOUBLE=1 METAL=1 -j10   # the double-precision pair as well (./DTFE-double, ./PS-DTFE-double,
                                          # objects in o_d / o_ps_d; float64 end to end, ~2x memory; the GPU
                                          # deposits float copies of the tetrahedra). Run scripts:
                                          # DTFE_PRECISION=double ./scripts/run_ps_dtfe.sh ...
make DTFE PS-DTFE DIM=2           # the 2D pair: ./DTFE-2d and ./PS-DTFE-2d (own object dirs o_2d / o_ps_2d,
                                  #   so they coexist with the 3D binaries; CPU-only). 2D input: HDF5 with
                                  #   two-column Coordinates/Velocities/InitialCoordinates, e.g.
                                  #   python3 tests/generate_ps_test_data.py --dim 2 --out snap2d.hdf5 --n 96
                                  #   ./PS-DTFE-2d snap2d.hdf5 out2d --grid 192 --periodic --field density tweb
make PS-DTFE TBB=1                # REQUIRE tbb's parallel CGAL triangulation (the default TBB=auto compiles it in
                                  #   when tbb is installed, TBB=0 leaves it out); the opt-in is at run time,
                                  #   --parallel-triangulation (or DTFE_TBB_THREADS=N): runs then differ at rounding
make DTFE PS-DTFE LIBDEFLATE=0    # the tessellation cache with zlib instead of libdeflate (default auto)
make DTFE BUILD_MODE=debug        # debug symbols, no optimization, sanitizers
make library                      # shared libDTFE
make clean                        # EVERY binary set (single, -double, -2d), libDTFE and the objects;
                                  #   the build-mode stamps stay
make deps-check                   # verify headers/libraries exist before a long compile
make test-platform                # show detected platform / compiler / paths
make DTFE $(cat o/.build_mode)    # rebuild WITHOUT downgrading the current GPU mode
make PS-DTFE $(cat o_ps/.build_mode)
```

The build mode is stamped in `o/.build_mode` / `o_ps/.build_mode`; switching modes wipes the
object directory automatically. Header edits rebuild all affected objects (auto-deps).

---

## Download (`scripts/download_snapshots.sh`)

API key: `-k KEY`, or `TNG_API_KEY` env, or `~/.tng_api_key` (recommended). The data root, the folder that
holds the simulations, is REQUIRED and has no default: `export DTFE_DATA_ROOT=~/IllustrisTNG` (or any
`/path/to/IllustrisTNG`) before any script below; without it every script stops with "no data root" (`-d` does
not replace it). Everything lands in the simulation's folder, `sim_dir <SIM>` in `scripts/config.sh`
(Python: `dtfelib.cli.sim_dir`); `-d` overrides. The simulations may sit in FAMILY folders --
`<root>/<family>/<sim>/`, e.g. `TNG100/TNG100-3-Dark`, family = the name up to its first `-` -- or
flat, `<root>/<sim>/`; an existing flat folder wins, and a new simulation goes beside its family's
folder when that exists. A data root that does not exist (a disk not mounted) stops the scripts rather
than letting them write elsewhere. Anything whose merged/converted product already exists is skipped automatically.

```bash
./scripts/download_snapshots.sh -s TNG50-3-Dark 0 4 17 33 50 99    # raw snapshot chunks (z=20..0)
./scripts/download_snapshots.sh -c -s TNG50-3-Dark 0 4 17 33 50 99 # FoF/Subfind group catalogs
./scripts/download_snapshots.sh -t -s TNG50-3-Dark                 # SubLink merger trees (whole sim)
./scripts/download_snapshots.sh -i -s TNG100-3-Dark                # initial conditions ics.hdf5
                                                           #   ('-Dark' stripped: ICs are
                                                           #    served under TNG100-3)
./scripts/download_snapshots.sh -s TNG50-3-Dark                    # default snapshot ladder (19 snaps)
./scripts/download_snapshots.sh -k 0123abcd -d /scratch/TNG300-3-Dark -s TNG300-3-Dark 99   # into another folder
```

## Merge / units (`python/tools/merge_HDF5.py`, `convert_ic_units.py`)

Every `combined_*` file is h-FREE (ckpc, 1e10 Msun, ckpc km/s) with self-describing
`HFreeUnits` / `divided_by_h` markers. Chunk counts are auto-detected.

```bash
SIMDIR="$DTFE_DATA_ROOT/TNG50/TNG50-3-Dark"   # a simulation folder (here in the per-family layout)
python3 python/tools/merge_HDF5.py -d "$SIMDIR" 0 4 17 33 50 99   # snapshots
python3 python/tools/merge_HDF5.py -d "$SIMDIR" --groupcats 0 99  # group catalogs
python3 python/tools/merge_HDF5.py -d "$SIMDIR" --trees           # SubLink trees
python3 python/tools/merge_HDF5.py -d "$SIMDIR" --ics             # ICs conversion

# everything for one sim, then delete the (verified) source chunks to save space:
python3 python/tools/merge_HDF5.py -d "$SIMDIR" \
    --snapshots-too --groupcats --trees --ics --delete-chunks 0 4 17 33 50 99

# upgrade PRE-unification combined files in place (idempotent, chunk-free):
python3 python/tools/merge_HDF5.py -d "$SIMDIR" --fix-units

# standalone ICs conversion (what --ics wraps):
python3 python/tools/convert_ic_units.py "$SIMDIR/ics.hdf5" "$SIMDIR/combined_ics.hdf5"
```

`--hubble 0.6774` is only a fallback when a file lacks `HubbleParam`; `-n N` forces the chunk
count instead of globbing. `-d` (the simulation's folder) is required -- there is no default simulation -- and
without snapshot numbers it merges 0 4 17 33 50.

---

## Run scripts (`scripts/run_ps_dtfe.sh`, `run_dtfe.sh`, `run_ps_pipeline.sh`)

Config comes from `scripts/config.sh` (`DTFE_DATA_ROOT`, `DTFE_SIM` env; the simulation folder is
resolved by `sim_dir`, flat or per-family -- see Download above) plus flags. The data root is REQUIRED (the lines
below assume `export DTFE_DATA_ROOT=~/IllustrisTNG`), and so is the simulation: `-s SIM` or `DTFE_SIM` (no default;
the lines below without `-s` assume `export DTFE_SIM=TNG100-3-Dark`). Positional
arguments override the snapshot list (default: 99). Partitioning is auto-tuned unless `PARTITION` /
`MAX_CONCURRENT` are set. Both default to the production settings (512^3, the GPU on a GPU build,
the T-web and V-web at `LAMBDA_TH` 0.3).

```bash
./scripts/run_ps_dtfe.sh                                  # PS-DTFE, snapshot 99 of $DTFE_SIM
./scripts/run_ps_dtfe.sh -s TNG300-3-Dark 99              # one sim, one snapshot
./scripts/run_ps_dtfe.sh -g 512 -n 2 -m 99                # grid 512^3, nSub=2, GPU deposit
./scripts/run_ps_dtfe.sh -e -m 99                         # exact conservative deposit on the GPU
DTFE_SIM=TNG100-3-Dark ./scripts/run_ps_dtfe.sh 0 50 99   # sim via env
PARTITION="5 5 5" MAX_CONCURRENT=2 ./scripts/run_ps_dtfe.sh -s TNG300-3-Dark 99   # manual memory plan
THREADS=4 AVG_SUBSAMPLES=1 ./scripts/run_ps_dtfe.sh 99    # cap threads; one sample per cell (the '_a' files
                                                          #   then hold the cell-centre values)
DRY_RUN=1 ./scripts/run_ps_dtfe.sh 99                     # print the expanded command, run nothing
AUTO_TUNE_REPORT=1 ./scripts/run_ps_dtfe.sh 99            # only the auto-tuner's split + predicted peak

./scripts/run_dtfe.sh -s TNG50-3-Dark -g 512 99           # standard DTFE (exact cell average, padding 15)
DTFE_METAL=0 ./scripts/run_dtfe.sh 99                     # the CPU on a GPU build (the GPU is the default there)
DTFE_EXACT_AVERAGE=0 ./scripts/run_dtfe.sh 99             # Monte-Carlo cell means instead of the exact average
OUTPUT_PREFIX=output_mc ./scripts/run_dtfe.sh 99          # a variant beside output.* (a ps_* prefix is refused)
```

`run_ps_dtfe.sh` environment toggles (each maps to the flag in brackets):

```bash
PS_VERTEX_MASS=1        # [--ps-vertex-mass]     DEFAULT 1: chart-independent tet masses.
                        #                        Set 0 only for true-lattice Lagrangian input.
PS_VOLUME_WEIGHTED=0    # [--ps-volume-weighted] DEFAULT 1: Eulerian-volume-weighted velocity moments
                        #                        (the dispersion stays mass-weighted)
PS_EXACT=1              # [--ps-exact-deposit]   exact r3d deposit; combine with -m (GPU)
PS_LINEAR_DEPOSIT=1     # [--ps-linear-deposit]  linear-profile, mass-conserving sampled deposit
                        #                        (the moments then mass-weighted: PS_VOLUME_WEIGHTED=0)
PS_METAL=0              # [--ps-gpu]             default auto: the GPU deposit on a GPU build (1 = -m); 0 = the CPU
AVG_SUBSAMPLES=2        # [--avg-subsamples]     nSub for the '_a' pass (cost ~ nSub^3)
SAMPLE_POINTS=plane.bin # [--sample-points]      also point-evaluate this point set
PTS_VEL_GRAD=1          # [--pts-vel-grad]       + per-point velocity gradient (.pts_velGrad),
                        #                        which is what makes velDiv/shear/vort maps possible
PTS_CONVENTIONS=legacy  # [--pts-conventions]    the plane's estimators; default grid (the grids' conventions)
SCRATCH_DIR=/tmp/dtfe   # [--scratch-dir]        back the full-grid accumulators with mmap'ed
                        #                        files (out-of-core; must be LOCAL, non-synced)
PS_CAUSTICS=0           # [--ps-caustics]        DEFAULT 1: '.caustic' fold flag + '.causticClass'
PS_CAUSTIC_CUSPS=1      # [--ps-caustic-cusps]   + bit7 A3 cusp AND bit8 A4 swallowtail
                        #                        indicators (needs PS_CAUSTICS=1; opt-in, bit7
                        #                        0.01% false positives, ~1.8x caustic cost)
PS_PARALLEL_TRI=1       # [--parallel-triangulation]  parallel Delaunay insert (runs differ at rounding)
PS_LAG_PADDING=0        # [--ps-lag-padding]     the Eulerian padding rule (default: the binary's 8 spacings)
TESS_CACHE=/tmp/dtfe-t  # [--tessellation-cache] reuse the tessellation across runs (created if
                        #                        missing; LOCAL dir, ~0.15 KB/vertex -- deflated)
LAMBDA_TH=0.3           # [--lambda_th]          T-web/V-web eigenvalue threshold
                        #                        (script default 0.3; the BINARY default is 0.0)
LAGRANGIAN_INPUT=ics.h5 # [--lagrangianInput]    override the auto-detected combined_ics.hdf5
OUTPUT_PREFIX=ps_mw     # writes <prefix>.* so an A/B grid set can sit beside the production one
FIELDS="density_a ..."  # [--field]              override the production field list (default: all nine, tweb_a vweb_a included)
PARTITION="4 4 4"       # [--partition]          explicit Lagrangian split (else auto-tuned; "1" = one triangulation)
MAX_CONCURRENT=2        # [--max-concurrent]     cap simultaneous partition pipelines
ORDERED_MERGE=0         # [--completion-order-merge]  merge as the partitions finish (default 1: index order)
FIXED_POINT_MERGE=1     # [--fixed-point-merge]  opt-in order-free integer merge
PS_RESUME=0             # start a killed partitioned run over (default 1: --resume from its checkpoint)
THREADS=4               # OMP_NUM_THREADS cap
```

`run_ps_pipeline.sh` is **compute-only** (no plotting): per snapshot it regenerates the
production grids *and* point-evaluates a full-box hi-res image plane on the **same**
triangulation, so the plane costs only the evaluation on top of the grid run. Resumable.

`SIMS` is required (simulation names, or `all`); the lines below without it assume `export SIMS=TNG100-3-Dark`.

```bash
SIMS=all ./scripts/run_ps_pipeline.sh              # all sims, all snapshots found on disk
SIMS="TNG100-3-Dark" ./scripts/run_ps_pipeline.sh  # one simulation
SNAPS="50 67 99" ./scripts/run_ps_pipeline.sh      # only these snapshots
DRY_RUN=1 ./scripts/run_ps_pipeline.sh             # print the plan, run nothing
NU=4096 AXIS=z ./scripts/run_ps_pipeline.sh        # plane resolution (default 8192) / orientation
CENTER=55.35 ./scripts/run_ps_pipeline.sh          # plane position (regenerates the plane)
PLANES=16 THICKNESS=2 ./scripts/run_ps_pipeline.sh # a slab of planes (render with --project slab)
FORCE=1 ./scripts/run_ps_pipeline.sh               # ignore all freshness checks
GRID_SIZE=512 PTS_VEL_GRAD=1 ./scripts/run_ps_pipeline.sh
# then render:  python3 python/plot/plot_pointeval.py --sims all
```


## The binaries directly (`./DTFE`, `./PS-DTFE`)

```bash
# quickstart on the demo file (binary Gadget auto-detected; velocity block of THIS file is junk).
# The box is periodic, and its voids are wider than the default padding (5 spacings): without
# --periodic up to a quarter of a thread's cells lie outside the tessellation, with it ~1%.
./DTFE demo/z0_64.gadget out --grid 64 --field density --periodic --padding 12

# standard DTFE, all main fields, GPU '_a' pass (run_dtfe.sh adds --exact-average --padding 15
# and the T-web / V-web at --lambda_th 0.3)
./DTFE combined_099.hdf5 out --grid 512 --periodic --input 105 --MpcUnit 1000 \
    --field density_a velocity_a gradient_a --gpu

# PS-DTFE with separate Lagrangian input and explicit partitioning
./PS-DTFE combined_099.hdf5 ps_out --grid 512 --periodic --input 105 --MpcUnit 1000 \
    --field density velocity dispersion density_a --lagrangianInput combined_ics.hdf5 \
    --partition 4 4 4 --max-concurrent 3 --ps-gpu

# the PRODUCTION physics configuration (what run_ps_dtfe.sh does by default)
./PS-DTFE combined_099.hdf5 ps_out --grid 512 --periodic --input 105 --MpcUnit 1000 \
    --field density_a velocity_a dispersion_a --lagrangianInput combined_ics.hdf5 \
    --ps-vertex-mass          `# chart-independent tet masses (removes the 1-D(z_ic)/D(z) bias)` \
    --ps-volume-weighted      `# volume-weighted velocity moments; dispersion stays mass-weighted` \
    --ps-caustics             `# the caustic classes (.caustic, .causticClass)` \
    --ps-gpu

# point evaluation at arbitrary positions (text 'x y z' per line, or raw float64 Nx3);
# the STANDARD binary supports the same flags/formats (streams = 0/1 coverage; no
# --per-stream/--per-stream-ids; with --partition each partition answers the points in the
# tetrahedra it owns -- the single triangulation's values to float rounding)
./PS-DTFE snap.hdf5 out --grid 32 --periodic --input 105 --MpcUnit 1 \
    --sample-points points.txt \
    --per-stream               `# + ragged per-stream density/velocity records` \
    --per-stream-ids           `# + stream identities (4 sorted vertex ParticleIDs)` \
    --pts-den-grad             `# + density gradient per point (and per stream)` \
    --pts-vel-grad             `# + velocity gradient per point -> div/shear/vorticity` \
    --ps-caustics              `# + '.pts_caustic': the caustic mask AT EACH POINT (same bits as` \
                               `#   '.causticClass'), so the fold skeleton can be drawn at the` \
                               `#   full point-evaluation resolution instead of on the grid` \
    --ps-stream-density geometric   # per-stream estimator: 'dtfe' (default) | 'geometric'
./DTFE snap.hdf5 out --grid 32 --periodic --input 105 --MpcUnit 1 \
    --sample-points points.txt --pts-den-grad

./PS-DTFE snap.hdf5 out ... --ps-linear-deposit    # linear-profile mass-conserving deposit (CPU or GPU;
                                                   #   not with --ps-volume-weighted)
./PS-DTFE snap.hdf5 out ... --avg-subsamples 2     # cheaper '_a' pass (8 sub-points)
./PS-DTFE snap.hdf5 out ... --ps-caustics          # + '.caustic' fold flag and '.causticClass' (CPU or GPU),
                                                   #   a bitmask per cell:
                                                   #     bit0/1  det(J)>0 / det(J)<0 tet present
                                                   #             (both = fold, i.e. '.caustic';
                                                   #             bit1 is the exact overlap test)
                                                   #     bit2-5  a tet with 0/1/2/3 COLLAPSED AXES
                                                   #             (1 wall, 2 filament, 3 node),
                                                   #             recorded at the deposited samples
                                                   #     bit6    two eigenvalues coincide AT
                                                   #             ZERO (umbilic/D4 INDICATOR only)
                                                   #   decode: dtfelib.io.caustic_collapse_max()
./PS-DTFE snap.hdf5 out ... --ps-caustics --ps-caustic-cusps   # + bit7 = A3 CUSP indicator,
                                                   #   bit8 = A4 SWALLOWTAIL indicator, both from a
                                                   #   least-squares fit over the vertex-incident
                                                   #   ring: candidate maps at the tessellation's
                                                   #   resolution, NOT classifications. On a 1-D
                                                   #   wave with no cusps bit7 flags 0.01% of the
                                                   #   fold cells; an exact A4 lights 74.5% of the
                                                   #   cusp cells against 0.5% for a pure A3.
                                                   #   bit8 is NOT a subset of bit7. ~1.8x the
                                                   #   caustic pass, so off by default.
./PS-DTFE snap.hdf5 out ... --ps-halo-release 300  # halo-interior tets -> centroid deposit
./PS-DTFE snap.hdf5 out ... --ps-exact-deposit --ps-gpu   # exact r3d deposit (use the GPU!)
./PS-DTFE snap.hdf5 out ... --scratch-dir /tmp/dtfe       # out-of-core full-grid accumulators
./PS-DTFE snap.hdf5 out ... --auto-tune-report            # the auto-tuner's plan as one line, nothing run
./PS-DTFE snap.hdf5 out ... --partition 4 4 4 --resume    # continue a killed partitioned run (both binaries)

# reuse the tessellation across runs: the first run writes it, later runs with the SAME inputs
# load it and skip both the triangulation and the vertex-density pass. The grid, the fields and
# the sample points may all change and it still hits; a changed input file, --density0 or
# geometry is a MISS (rebuild). Needs a LOCAL directory and ~0.15 KB per tessellation vertex.
# Partitioned runs cache every tessellation they build (PS: one file per partition; standard:
# one per thread sub-box, or per partition under --sample-points).
mkdir -p /private/tmp/dtfe-tess
./PS-DTFE snap.hdf5 out1 --grid 64  --periodic --field density --MpcUnit 1 \
    --sample-points a.bin --tessellation-cache /private/tmp/dtfe-tess     # builds + writes
./PS-DTFE snap.hdf5 out2 --grid 256 --periodic --field density --MpcUnit 1 \
    --sample-points b.bin --tessellation-cache /private/tmp/dtfe-tess     # reuses it
./PS-DTFE snap.hdf5 out ... --tessellation-cache DIR --verbose 3          # says why a lookup missed
./PS-DTFE snap.hdf5 out ... --field vweb --lambda_th 0.3 --scale-factor 0.25 --hubble 0.6774
./DTFE   snap.hdf5 out ... --exact-average         # exact r3d '_a' averaging (on the GPU with --gpu)
./PS-DTFE snap.hdf5 out --grid 4096 --periodic --MpcUnit 1 --ps-exact-deposit \
    --ps-window 20 30 20 30 50 50.1               # only the window's cells of the 4096^3 grid: the
                                                  # full run's cells there, in seconds (the launcher's exact zoom)
./PS-DTFE snap.hdf5 out --grid 4096 --periodic --MpcUnit 1 --ps-exact-deposit --ps-window 20 30 20 30 50 50.1 \
    --partition 4 4 4 --tessellation-cache /private/tmp/dtfe-tess   # the query server's split and cache:
                                                  # cached partitions are loaded, those whose occupancy maps
                                                  # never reach the window are skipped
./PS-DTFE --full_help                              # every option with full descriptions
```

Common to both: `--grid N [NY NZ]`, `--periodic`, `--input 101|105|106|111|112|121|122`,
`--MpcUnit <units per Mpc>` (default 1000), `--partition X Y Z`, `--max-concurrent N`, `--verbose 0..3`,
`--padding` (`--region` / `--regionMpc` only in the standard binary: PS-DTFE refuses them, use `--box`). Input 101 = binary Gadget, 105 = Gadget HDF5 (default), 106 = its HI data, 111 = text
`x y z w`, 112 = text positions-only, 121 / 122 = raw binary (count, box, pos, weights, vels / structured).

**Which flags reach the GPU** (`METAL=1`, `CUDA=1` or `HIP=1` build): the grid deposit, and
`--ps-linear-deposit`, `--ps-volume-weighted`, `--ps-vertex-mass`, `--ps-caustics`,
`--ps-exact-deposit` and `--ps-window` all compose with `--ps-gpu`; the standard binary's
`--exact-average` and the scalar fields (`NO_SCALARS=1` builds) with `--gpu`. CPU-only: point
evaluation (`--sample-points`/`--serve`: multi-threaded over the tessellation, exact point location).

### Interactive point evaluation (`--serve`, `dtfelib.Estimator`)

```python
from dtfelib import Estimator
with Estimator("snap.hdf5", mpc_unit=1) as est:     # PS-DTFE; builds the tessellation ONCE
    f = est(points)                                 # (N,3) Mpc -> .density .velocity .streams
                                                    #   .dispersion (+ optional outputs below)
    est.density(halo_centres); est.streams(p)       # shortcuts
    cube = est.grid(xs, ys, zs)                     # arrays shaped (nx, ny, nz[, k])
Estimator("snap.hdf5", phase_space=False)           # standard DTFE interpolant instead
Estimator("snap.hdf5", per_stream=True, per_stream_ids=True, density_gradient=True,
          velocity_gradient=True, caustics=True)    # every --sample-points output
Estimator("snap.hdf5", tessellation_cache="/private/tmp/dtfe-tess")  # instant restarts
Estimator("snap.hdf5", options=["--ps-vertex-mass"], verbose=1)      # any extra flags; log on stderr
Estimator("snap.hdf5", per_stream=True, lagrangian_positions=True,  # .stream_lagrangian: q(x) per
          scalar_dataset="Potential")               #   stream; .scalar/.stream_scalar: the dataset
Estimator("big.hdf5", partition=4, resident=3,      # COMPOSITE: 4^3 partition tessellations on disk,
          tessellation_cache="/scratch/dtfe-tess")  #   at most 3 in memory (default: the RAM budget)
Estimator.from_arrays(x, velocities=v, lagrangian=q, values=phi, box=100.0)   # numpy arrays in
```
The answers are bit-identical to a `--sample-points` run on the same points; a single point takes
~20 us from Python (a cell index built at start-up). Velocities come back in peculiar km/s
(`peculiar=False` for raw Gadget units). The binary side, for other clients:
`./PS-DTFE snap.hdf5 unused --serve --MpcUnit 1 ...` -- logs on stderr, a binary protocol on
stdin/stdout (protocol version 2, documented above psServe in `src/CGAL_triangulation/ps_point_eval.cc`).
With `--partition` (or when one tessellation would not fit and `--tessellation-cache` is given) the server
is a COMPOSITE of partition tessellations, each with a 64^3 occupancy map ('<cache file>.occ'): a request
visits only the partitions that can hold its points, at most `--serve-resident N` stay in memory (LRU, ~666
B/vertex; `DTFE_SERVE_BUDGET_GB` overrides the budget), and the answers equal `--sample-points --partition`
with the same split. `--serve-progress` writes progress lines to stderr (the launcher's progress bar);
`DTFE_SERVE_INDEX=0|1` forces the walk or the cell index.

```bash
python3 python/tools/upgrade_tess_cache.py "$DTFE_DATA_ROOT/.dtfe-tessellation-cache"  # old cache files -> chunked
```
Cache files are written in 16 MB chunks, compressed and decompressed on several cores
(`DTFE_TESS_CACHE_THREADS=N` pins the threads per file, `DTFE_TESS_CACHE_LEGACY=1` writes the old single
stream; `DTFE_TESS_CACHE_ZLIB=1` uses zlib where the build has libdeflate (`make ... LIBDEFLATE=0` builds
without it); `DTFE_TESS_CACHE_CGAL_IO=1` reads and writes the triangulation through CGAL's own stream
operators instead of the block reader/writer, and `DTFE_TESS_CACHE_FINGERPRINT=1` prints a hash of each
loaded structure -- both for comparisons). A partitioned run with a cache also records the snapshot's particle totals there
(`globals_<hash>.txt`): a later `--ps-window` run whose partitions are all cached then never reads the
snapshot. `DTFE_PTS_THREADS=N` pins the point-evaluation
thread count (default: all cores, or cores / concurrent partitions inside a partitioned PS run).

```bash
# per-stream Lagrangian origin and any per-particle dataset, in a batch run too
./PS-DTFE snap.hdf5 out --sample-points pts.bin --per-stream --pts-lagrangian \
    --pts-scalar --scalar-dataset Potential --MpcUnit 1 --periodic -g 16
#   -> out.pts_stream_lagpos (float64 x3 per stream), out.pts_scalar, out.pts_stream_scalar
./PS-DTFE --version           # the build stamp (git revision + time); every run logs 'Build: ...'
```

### Standard-DTFE extras

```bash
# comparison interpolation schemes (instead of DTFE), density only for NGP/Voronoi
./DTFE snap.hdf5 out --grid 256 --periodic --NGP            # nearest grid point
./DTFE snap.hdf5 out --grid 256 --periodic --CIC            # cloud in cell
./DTFE snap.hdf5 out --grid 256 --periodic --TSC            # triangular shape cloud
./DTFE snap.hdf5 out --grid 256 --periodic --PCS            # piecewise cubic spline
./DTFE snap.hdf5 out --grid 256 --periodic --SPH 32         # SPH, 32 nearest neighbours
./DTFE snap.hdf5 out --grid 256 --periodic --Voronoi --field density    # 1/Voronoi-volume
                                                            #   density, NGP-assigned. Density
                                                            #   ONLY (other fields rejected);
                                                            #   '--field density_a' -> '.a_den'
./DTFE snap.hdf5 out --grid 256 --periodic --CIC --interlace  # anti-aliasing (2D and PS-DTFE too; without
                                                               # --periodic it needs a --region half a cell inside
                                                               # the box). Only the density is interlaced: the T-web
                                                               # comes from the first, un-interlaced pass.

# approximate phase-space density f = rho * g: a SECOND Delaunay in velocity space.
# Standard binary only; the result lands in the scalar output field. It ASSUMES the local
# distribution factorises into a spatial and a velocity part -- precisely what fails where
# streams cross, which is what makes it worth benchmarking against PS-DTFE (below).
./DTFE snap.hdf5 out --grid 128 --periodic --approxPSD

# Benchmarking that approximation against the exact phase-space structure.
# PS-DTFE does not need the factorisation: it resolves the streams themselves, so the exact
# distribution function is the per-stream decomposition, and the standard scalar proxy
# Q = rho/sigma^3 follows from fields it already deposits.
./DTFE    snap.hdf5 ap  --grid 128 --periodic --approxPSD                        # f ~ rho*g
./PS-DTFE snap.hdf5 ex  --grid 128 --periodic --field density_a dispersion_a \
          --lagrangianInput ics.hdf5                                             # exact rho, sigma
./PS-DTFE snap.hdf5 exp --grid 32 --periodic --sample-points pts.bin --per-stream \
          --lagrangianInput ics.hdf5      # the exact f itself: (rho_s, v_s) for every stream
#   then, in python:
#     from dtfelib.fields import pseudo_phase_space_density
#     Q = pseudo_phase_space_density(fs.density(units="mean"), np.sqrt(fs.load("dispersion")))

# redshift cone (spherical) grid instead of a box
./DTFE survey.hdf5 out --grid 128 --redshiftCone 0 200 -30 30 -30 30 --origin 0 0 0
# (--redshiftSpace d1 d2 d3 is NOT implemented in this version: the run is refused.
#  Shift the positions before running instead.)

# zoom region, sub-sampling, synthetic points, options file
./DTFE snap.hdf5 out --grid 256 --region 0.4 0.6 0.3 0.7 0.45 0.55   # fractions of the box
./DTFE snap.hdf5 out --grid 256 --regionMpc 20 30 20 30 20 30        # Mpc coordinates
./DTFE snap.hdf5 out --grid 256 --randomSample 0.1                   # keep 10% of particles
./DTFE snap.hdf5 out --grid 256 --poisson 64                         # 64^3 random positions
./DTFE snap.hdf5 out --grid 256 --density0 2.66                       # explicit density scale
./DTFE snap.hdf5 out --grid 256 --extensive                          # scalars are extensive
./DTFE snap.hdf5 out --grid 256 --noTest                             # skip the padding test
./DTFE snap.hdf5 out --grid 256 --partition 3 3 3 --no-overlap       # no look-ahead triangulation
./DTFE --config my_options.cfg                                       # read options from a file
```

---

## Analysis (`python/analyze.py`, `python/plot/*.py`)

All plot scripts share the CLI from `dtfelib.cli`: `--data-root PATH` (else `DTFE_DATA_ROOT`: one of the two
is required, there is no default), `--sim NAME`,
`--snap N`, `--method auto|ps|dtfe`, `--raw` (unaveraged grids), `--smooth SIGMA_CELLS`,
`--prefix` (an alternate `OUTPUT_PREFIX` grid set). `--sim` is required unless `DTFE_SIM` is set (there is no
default simulation); `analyze.py` takes it from `DTFE_SIM` only. The lines below without either assume
`export DTFE_SIM=TNG100-3-Dark`.

```bash
python3 python/analyze.py                          # = all: compute (own process) + plot, for $DTFE_SIM
python3 python/analyze.py check                    # which fields/products exist on disk
python3 python/analyze.py compute 99               # derived products for snapshot 99
python3 python/analyze.py plot --only plot_void_population.py plot_cosmic_web.py
DTFE_SIM=TNG300-3-Dark python3 python/analyze.py all
python3 python/analyze.py export 99                # void catalogues as CSV + HDF5
python3 python/analyze.py voids --only plot_void_fate.py     # the void scripts for DTFE_SIM
python3 python/analyze.py cache --prune            # list unreachable cache namespaces (--yes deletes)
python3 python/analyze.py cache --move ~/Library/Caches/DTFE   # dry run; --yes moves the cache

python3 python/plot/plot_PS_DTFE.py --sim TNG50-3-Dark --snap 99      # PS field maps
python3 python/plot/plot_DTFE.py --method dtfe --raw                  # standard DTFE maps
python3 python/plot/plot_void_population.py --snap 99 --smooth 10
python3 python/plot/plot_void_tracking.py --sim TNG50-3-Dark
python3 python/plot/plot_halo_tracking.py --sim TNG50-3-Dark --method ps
python3 python/plot/plot_marked_correlation_BBKS.py --snap 99
python3 python/plot/plot_caustic_skeleton.py --sim TNG100-3-Dark --snap 99 --prefix ps_caustic --voids
DTFE_SIM=TNG50-3-Dark python3 python/plot/plot_void_fate.py --sim TNG50-3-Dark --exclusive
DTFE_VOID_GATE=none DTFE_SIM=TNG100-3-Dark python3 python/plot/plot_void_population.py --sim TNG100-3-Dark --size-fixed-r
python3 python/plot/plot_void_crossbox.py --sims TNG50-3-Dark TNG100-3-Dark TNG300-3-Dark --smooth-mpc 2.1625   # side by side
# also: plot_cosmic_web, plot_eigenvalues, plot_shear_triaxial, plot_velDiv_den, plot_PDF_CDF,
#       plot_phi_delta, plot_tidal_correlation, plot_shape_filter, plot_contour_filter,
#       plot_ellipses_contour_3D, plot_smoothing_comparison, plot_pk_compare, plot_web_streams,
#       plot_void_profiles, plot_void_profile_evolution, plot_void_environment, plot_void_convergence
# void scripts: set DTFE_SIM to the --sim (the cached catalogue's lengths are in that frame);
# DTFE_SIGMA_MPC=2.1625 smooths every box at one physical scale; DTFE_CACHE_DIR moves the cache

# point-evaluation figure pipeline (grid-free, publication resolution)
python3 python/tools/make_image_plane.py --combined combined_099.hdf5 --axis z --nu 8192 -o plane
python3 python/plot/plot_pointeval.py                     # one figure per field, all snapshots
python3 python/plot/plot_pointeval_panels.py              # 3-panel density/sheet/N_streams
python3 python/tools/compare_deposits.py --sim TNG100-3-Dark --snap 99 --a ps_sampled --b ps_output
                                                          # A/B a sampled against an exact-deposit grid set
python3 python/tools/void_outlines.py --sim TNG100-3-Dark --snap 99 --sigma-mpc 2.1625 --csv voids.csv   # void table

# decoding the caustic mask ('.causticClass' / '.pts_caustic') in python:
#   from dtfelib.io import (caustic_is_fold, caustic_collapse_max, caustic_is_cusp,
#                           caustic_is_swallowtail)
#   fold = caustic_is_fold(m)          # both parities -> the A2 fold set (same as '.caustic')
#   k    = caustic_collapse_max(m)     # 0 uncollapsed, 1 wall, 2 filament, 3 node (-1 = untouched)
#   cusp = caustic_is_cusp(m)          # bit7, only under --ps-caustic-cusps; 0.01% false positives
#   swal = caustic_is_swallowtail(m)   # bit8, same flag; proximity to an A4, ~146x contrast.
#                                      # NOT a subset of cusp -- OR them for the A3-and-above set
#   FieldSet.fold_mask()               # the exact fold flag, also for grids with the per-sample '.caustic'

# single-stream cells need BOTH grids: '.streams' counts sub-samples only, so a fold thinner than
# the sampling can cross a cell unseen -- bit 1 of '.hidden_streams' marks exactly those cells (a
# folded tetrahedron overlaps them); bit 2 marks mass from a sub-sample tet (usually the same stream).
#   single = (np.abs(fs.load("streams") - 1) < STREAM_TOL) & ((np.rint(fs.load("hidden_streams")).astype(int) & 1) == 0)
#   v1 = fs.velocity_single_stream()   # does exactly this (streams alone on grids without '.hidden_streams')
#   fs.load("divergence", mean_field=True)   # a PS grid's div of the MEAN velocity (standard DTFE's quantity)
```

---

## Graphical launcher (`scripts/gui.sh`, `python/gui/`)

```bash
scripts/gui.sh                 # the window (PySide6 from ~/.venvs/dtfe-gui; scripts/install.sh sets it up)
scripts/gui.sh --setup         # the first-run dialog again: data folder, build, TNG API key, demo
scripts/gui.sh --smoke         # build the window, print what its jobs would use, quit
```
Tabs: Data (download + merge), Grids (run_ps_dtfe.sh / run_dtfe.sh, presets), Custom snapshot (the
binary on any snapshot file), Pipeline, Plots, Explore (maps of any run + click-to-query server, zooms);
beside the log: Queue, Figures, Runs (every run log on disk, stale-output flags, load settings,
move to Trash) and Running now (every DTFE run on the machine, with Stop). Every job shows its commands;
Export saves them as a bash script or SLURM job. The modules: `app.py` + `app_*.py`, `explore.py` +
`explore_*.py`, `runspec.py` + `runspec_*.py` (README, Graphical launcher).

## Tests

`tests/ci_suite.sh build|core|double|2d|gui|oldpy|variants|all` runs the suites CI runs (the list with
the reasons is in its header; README, Tests). Single suites:

```bash
tests/ci_suite.sh core                        # every CPU suite against ./DTFE and ./PS-DTFE
DTFE_TEST_PRECISION=double tests/ps_window_check.sh --no-build    # any suite, against the double pair
python3 tests/run_tests.py                    # standard-DTFE suite + the stored regression reference
python3 tests/run_tests.py --update-ref       # regenerate the reference after an intended change
tests/ps_smoke_test.sh                        # build + Zel'dovich pancake sanity
tests/ps_smoke_test.sh --no-build             # reuse the existing binary (all .sh support this)
python3 tests/ps_regression_test.py --update-baseline   # regenerate the PS metric baseline
N=32 GRID=64 tests/ps_point_eval_check.sh --no-build    # bigger problem via env
python3 tests/docs_drift_check.py             # README / COMMANDS.md flags and knobs against the code
~/.venvs/dtfe-gui/bin/python tests/py_gui_app_test.py   # the window, offscreen, end to end
python3 python/selftest.py                    # sandboxed pipeline+plots selftest (~2 min)
tests/ps_scaling_benchmark.sh                 # strong-scaling table (MIN_EFFICIENCY=0.4); not in CI

# GPU builds only (by hand; CI runs them through the CPU emulation, tests/cudaemu_check.sh):
tests/dtfe_metal_check.sh                     # CPU vs GPU parity, standard DTFE (GPU_BUILD=CUDA=1 for CUDA)
tests/gpu_scalar_check.sh; tests/gpu_mass_check.sh; tests/gpu_retry_check.sh; tests/dtfe_gpu_retry_check.sh

# synthetic snapshot generator used by the tests (Gadget HDF5, input type 105):
python3 tests/generate_ps_test_data.py --out snap.hdf5 --n 32 --box 100 \
    --amplitude-factor 1.8      `# >1 = multi-stream pancake; 0 = uniform` \
    --jitter-frac 0.05 --seed 42 \
    --crossed-waves             `# 3D displacement: stream counts {1,3,9,27}` \
    --margin-frac 0.2           `# non-periodic clump in vacuum instead`
```

CI (`.github/workflows/ci.yml`) runs six jobs: build-and-test (`ci_suite.sh build core gui double 2d`,
the standard-DTFE regression against its stored reference and the auto-tune and `--scratch-dir` suites
included), old-python, compile-variants, cudaemu (warp width 32 and 64), cuda-compile and hip-compile.
The Metal GPU path and the GPU suites on real hardware are local-only.
