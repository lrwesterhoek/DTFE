# Command cheatsheet

Every script in the repo with its useful flag combinations. All paths are relative to the
repository root; quote the repo path (it contains spaces). Scripts that need h5py should be
run with a python whose h5py works (`/opt/homebrew/bin/python3` on the development Mac).

---

## Build (`Makefile`, `scripts/install.sh`)

```bash
./scripts/install.sh              # one-command clean rebuild, best backend auto-detected
                                  #   (Apple Silicon -> METAL=1, else CPU-only)
./scripts/install.sh --cpu        # force CPU-only binaries
./scripts/install.sh --no-deps    # skip dependency installation, just clean + rebuild
./scripts/install.sh --jobs 4     # limit parallel build jobs
./scripts/install.sh --docker     # containerized LINUX image instead (never Metal)
./scripts/install.sh --double     # double-precision binaries (DOUBLE=1, CPU-only)
./scripts/install.sh --no-python  # skip the 'pip install -e python' step (dtfelib)
python3 -m pip install -e python  # dtfelib alone (editable: finds ./DTFE, ./PS-DTFE itself;
                                  #   a non-editable install needs DTFE_BIN_DIR or $PATH)

make DTFE METAL=1 -j10            # standard DTFE, Apple-GPU '_a' interpolation backend
make PS-DTFE METAL=1 -j10         # phase-space DTFE, Apple-GPU deposit backend
make DTFE DIM=2                   # 2D standard DTFE (default DIM=3). PS-DTFE is 3D ONLY and
                                  #   refuses to build with DIM=2 (see the #error in
                                  #   ps_interpolation.cc: no 2D Lagrangian input exists).
                                  #   Switching DIM wipes the object dir, like a backend switch.
make PS-DTFE TBB=1                # opt-in parallel CGAL triangulation (slower at small N!)
make PS-DTFE DOUBLE=1             # double precision end to end (Real = double; float64 output
                                  #   grids, full-precision HDF5 reads). CPU-only: refused with
                                  #   METAL=1. Stamped like DIM: switching it rebuilds.
make library                      # shared libDTFE
make clean                        # binaries + objects + auto-dependency .d files
make deps-check                   # verify headers/libraries exist before a long compile
make test-platform                # show detected platform / compiler / paths
make DTFE $(cat o/.build_mode)    # rebuild WITHOUT downgrading the current GPU mode
make PS-DTFE $(cat o_ps/.build_mode)
```

The GPU mode is stamped in `o/.build_mode` / `o_ps/.build_mode`; switching modes wipes the
object directory automatically. Header edits rebuild all affected objects (auto-deps).

---

## Download (`scripts/download_snapshots.sh`)

API key: `-k KEY`, or `TNG_API_KEY` env, or `~/.tng_api_key` (recommended). Everything lands
in the simulation's folder, `sim_dir <SIM>` in `scripts/config.sh` (Python: `dtfelib.cli.sim_dir`);
`-d` overrides. The data root (`DTFE_DATA_ROOT`) defaults to the Samsung T7,
`/Volumes/Samsung T7/Illustris TNG`, whose simulations sit in FAMILY folders --
`<root>/<family>/<sim>/`, e.g. `TNG100/TNG100-3-Dark`, family = the name up to its first `-`.
A flat `<root>/<sim>/` root works too (`DTFE_DATA_ROOT=~/output`, the old internal copy); an
existing flat folder wins. With the T7 unplugged the scripts stop rather than write elsewhere.
Anything whose merged/converted product already exists is skipped automatically.

```bash
./scripts/download_snapshots.sh -s TNG50-3-Dark 0 4 17 33 50 99    # raw snapshot chunks (z=20..0)
./scripts/download_snapshots.sh -c -s TNG50-3-Dark 0 4 17 33 50 99 # FoF/Subfind group catalogs
./scripts/download_snapshots.sh -t -s TNG50-3-Dark                 # SubLink merger trees (whole sim)
./scripts/download_snapshots.sh -i -s TNG100-3-Dark                # initial conditions ics.hdf5
                                                           #   ('-Dark' stripped: ICs are
                                                           #    served under TNG100-3)
./scripts/download_snapshots.sh -s TNG50-3-Dark                    # default snapshot ladder (19 snaps)
./scripts/download_snapshots.sh -k 0123abcd -d /scratch/tng -s TNG300-3-Dark 99
```

## Merge / units (`python/tools/merge_HDF5.py`, `convert_ic_units.py`)

Every `combined_*` file is h-FREE (ckpc, 1e10 Msun, ckpc km/s) with self-describing
`HFreeUnits` / `divided_by_h` markers. Chunk counts are auto-detected.

```bash
SIMDIR="/Volumes/Samsung T7/Illustris TNG/TNG50/TNG50-3-Dark"   # a simulation folder on the T7
python3 python/tools/merge_HDF5.py -d "$SIMDIR" 0 4 17 33 50 99   # snapshots
python3 python/tools/merge_HDF5.py -d "$SIMDIR" 25 40 67          # the T7's raw-only snapshots
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
count instead of globbing. The default simulation directory follows `DTFE_DATA_ROOT`.

---

## Run scripts (`scripts/run_ps_dtfe.sh`, `run_dtfe.sh`, `run_ps_pipeline.sh`)

Config comes from `scripts/config.sh` (`DTFE_DATA_ROOT`, `DTFE_SIM` env; the simulation folder is
resolved by `sim_dir`, flat or per-family -- see Download above) plus flags; positional
arguments override the snapshot list. Partitioning is auto-tuned unless `PARTITION` /
`MAX_CONCURRENT` are set.

```bash
./scripts/run_ps_dtfe.sh                                  # PS-DTFE, default sim + snapshot ladder
./scripts/run_ps_dtfe.sh -s TNG300-3-Dark 99              # one sim, one snapshot
./scripts/run_ps_dtfe.sh -g 512 -n 2 -m 99                # grid 512^3, nSub=2, GPU deposit
./scripts/run_ps_dtfe.sh -e -m 99                         # exact conservative deposit on the GPU
DTFE_SIM=TNG100-3-Dark ./scripts/run_ps_dtfe.sh 0 50 99   # sim via env
PARTITION="5 5 5" MAX_CONCURRENT=2 ./scripts/run_ps_dtfe.sh -s TNG300-3-Dark 99   # manual memory plan
THREADS=4 AVG_SUBSAMPLES=1 ./scripts/run_ps_dtfe.sh 99    # cap threads; cell-centre only (fast, no '_a')

./scripts/run_dtfe.sh -s TNG50-3-Dark -g 512 99           # standard DTFE
./scripts/run_dtfe.sh -m 99                               # GPU '_a' interpolation
```

`run_ps_dtfe.sh` environment toggles (each maps to the flag in brackets):

```bash
PS_VERTEX_MASS=1        # [--ps-vertex-mass]     DEFAULT 1: chart-independent tet masses.
                        #                        Set 0 only for true-lattice Lagrangian input.
PS_VOLUME_WEIGHTED=1    # [--ps-volume-weighted] Eulerian-volume-weighted velocity moments
                        #                        (dispersion stays mass-weighted, bit-identical)
PS_EXACT=1              # [--ps-exact-deposit]   exact r3d deposit; combine with -m (GPU)
PS_METAL=1              # [--ps-gpu]             GPU deposit (same as -m)
AVG_SUBSAMPLES=2        # [--avg-subsamples]     nSub for the '_a' pass (cost ~ nSub^3)
SAMPLE_POINTS=plane.bin # [--sample-points]      also point-evaluate this point set
PTS_VEL_GRAD=1          # [--pts-vel-grad]       + per-point velocity gradient (.pts_velGrad),
                        #                        which is what makes velDiv/shear/vort maps possible
SCRATCH_DIR=/tmp/dtfe   # [--scratch-dir]        back the full-grid accumulators with mmap'ed
                        #                        files (out-of-core; must be LOCAL, non-synced)
PS_CAUSTICS=1           # [--ps-caustics]        '.caustic' fold flag + (CPU) '.causticClass'
PS_CAUSTIC_CUSPS=1      # [--ps-caustic-cusps]   + bit7 A3 cusp AND bit8 A4 swallowtail
                        #                        indicators (needs PS_CAUSTICS=1; opt-in, bit7
                        #                        0.01% false positives, ~1.8x caustic cost)
TESS_CACHE=/tmp/dtfe-t  # [--tessellation-cache] reuse the tessellation across runs (created if
                        #                        missing; LOCAL dir, ~0.15 KB/vertex -- deflated)
LAMBDA_TH=0.3           # [--lambda_th]          T-web/V-web eigenvalue threshold
                        #                        (script default 0.3; the BINARY default is 0.0)
LAGRANGIAN_INPUT=ics.h5 # [--lagrangianInput]    override the auto-detected combined_ics.hdf5
OUTPUT_PREFIX=ps_mw     # writes <prefix>.* so an A/B grid set can sit beside the production one
FIELDS="density_a ..."  # [--field]              override the production field list
PARTITION="4 4 4"       # [--partition]          explicit Lagrangian split (else auto-tuned; "1" = one triangulation)
MAX_CONCURRENT=2        # [--max-concurrent]     cap simultaneous partition pipelines
THREADS=4               # OMP_NUM_THREADS cap
```

`run_ps_pipeline.sh` is **compute-only** (no plotting): per snapshot it regenerates the
production grids *and* point-evaluates a full-box hi-res image plane on the **same**
triangulation, so the plane costs only the evaluation on top of the grid run. Resumable.

```bash
./scripts/run_ps_pipeline.sh                       # all sims, all snapshots found on disk
SIMS="TNG100-3-Dark" ./scripts/run_ps_pipeline.sh  # one simulation
DRY_RUN=1 ./scripts/run_ps_pipeline.sh             # print the plan, run nothing
NU=8192 AXIS=z ./scripts/run_ps_pipeline.sh        # plane resolution / orientation
CENTER=55.35 ./scripts/run_ps_pipeline.sh          # plane position (regenerates the plane)
FORCE=1 ./scripts/run_ps_pipeline.sh               # ignore all freshness checks
GRID_SIZE=512 PTS_VEL_GRAD=1 ./scripts/run_ps_pipeline.sh
# then render:  python3 python/plot/plot_pointeval.py
```

## The binaries directly (`./DTFE`, `./PS-DTFE`)

```bash
# quickstart on the demo file (binary Gadget auto-detected; velocity block of THIS file is junk)
./DTFE demo/z0_64.gadget out --grid 64 --field density

# standard DTFE, all main fields, GPU '_a' pass
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
    --ps-gpu

# point evaluation at arbitrary positions (text 'x y z' per line, or raw float64 Nx3);
# the STANDARD binary supports the same flags/formats (streams = 0/1 coverage; no
# --per-stream/--per-stream-ids, single triangulation so no --partition)
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

./PS-DTFE snap.hdf5 out ... --ps-linear-deposit    # linear-profile mass-conserving deposit
./PS-DTFE snap.hdf5 out ... --avg-subsamples 2     # cheaper '_a' pass (8 sub-points)
./PS-DTFE snap.hdf5 out ... --ps-caustics          # + '.caustic' fold-flag grid (CPU or GPU)
                                                   #   also writes (CPU *and* GPU; only
                                                   #   --ps-linear-deposit forces CPU, since the
                                                   #   mask rides in that buffer)
                                                   #   '.causticClass', a bitmask per cell:
                                                   #     bit0/1  det(J)>0 / det(J)<0 tet present
                                                   #             (both = fold, i.e. '.caustic')
                                                   #     bit2-5  a tet with 0/1/2/3 COLLAPSED AXES
                                                   #             overlaps (1 wall, 2 filament,
                                                   #             3 node) -- exact, from the
                                                   #             per-tet deformation tensor
                                                   #     bit6    two eigenvalues coincide AT
                                                   #             ZERO (umbilic/D4 INDICATOR only).
                                                   #             BOTH halves matter: merely-equal
                                                   #             eigenvalues fill caustic-free
                                                   #             volume. Every flagged cell lies
                                                   #             on a fold, as a D4 must.
                                                   #   decode: dtfelib.io.caustic_collapse_max()
./PS-DTFE snap.hdf5 out ... --ps-caustics --ps-caustic-cusps   # + bit7 = A3 CUSP indicator.
                                                   #   Estimates grad(lambda_c) from the face
                                                   #   neighbours, so it is a candidate map at the
                                                   #   tessellation's resolution, NOT a
                                                   #   classification. Fitted over the VERTEX-
                                                   #   incident ring (~20-40 tets): on a 1-D wave
                                                   #   with provably NO cusps it flags 0.01% of
                                                   #   fold cells (7% with face neighbours only).
                                                   #   Off by default: costs ~1.8x the caustic pass.
                                                   #   The SAME stencil also gives:
                                                   #     bit8  = A4 SWALLOWTAIL indicator. Needs the
                                                   #             SECOND derivative to vanish too;
                                                   #             fits a cubic along the null
                                                   #             direction and asks that the
                                                   #             expansion START at cubic order.
                                                   #             An exact A4 lights 74.5% of the
                                                   #             cusp cells vs 0.5% for a pure A3
                                                   #             and 0% for a generic fold --
                                                   #             factor ~146, but it still RANKS
                                                   #             rather than certifies.
                                                   #             A4 is codimension 3 (isolated
                                                   #             POINTS), so this marks proximity
                                                   #             along the cusp curve, over a band
                                                   #             as wide as the stencil.
                                                   #             NOT a subset of bit7: at a true A4
                                                   #             lambda_c is CUBIC and bit7's
                                                   #             linear fit reads a false slope.
./PS-DTFE snap.hdf5 out ... --ps-halo-release 300  # halo-interior tets -> centroid deposit
./PS-DTFE snap.hdf5 out ... --ps-exact-deposit --ps-gpu   # exact r3d deposit (use the GPU!)
./PS-DTFE snap.hdf5 out ... --scratch-dir /tmp/dtfe       # out-of-core full-grid accumulators

# reuse the tessellation across runs: the first run writes it, later runs with the SAME inputs
# load it and skip both the triangulation and the vertex-density pass (measured 10.9s -> 4.6s on
# 884736 particles). The grid, the fields and the sample points may all change and it still hits;
# a changed input file, --density0 or geometry is a MISS (rebuild). Needs a LOCAL directory and
# ~0.5 KB per tessellation vertex. A partitioned PS run caches each partition separately (one
# file per partition); only the STANDARD binary's partitioned runs skip the cache.
mkdir -p /private/tmp/dtfe-tess
./PS-DTFE snap.hdf5 out1 --grid 64  --periodic --field density --MpcUnit 1 \
    --sample-points a.bin --tessellation-cache /private/tmp/dtfe-tess     # builds + writes
./PS-DTFE snap.hdf5 out2 --grid 256 --periodic --field density --MpcUnit 1 \
    --sample-points b.bin --tessellation-cache /private/tmp/dtfe-tess     # reuses it
./PS-DTFE snap.hdf5 out ... --tessellation-cache DIR --verbose 3          # says why a lookup missed
./PS-DTFE snap.hdf5 out ... --field vweb --lambda_th 0.3 --scale-factor 0.25 --hubble 0.6774
./DTFE   snap.hdf5 out ... --exact-average         # exact r3d '_a' averaging (CPU-only, slow)
./PS-DTFE --full_help                              # every option with full descriptions
```

Common to both: `--grid N [NY NZ]`, `--periodic`, `--input 101|105|111|112|121|122`,
`--MpcUnit <units per Mpc>`, `--partition X Y Z`, `--max-concurrent N`, `--verbose 0..3`,
`--region`, `--padding`. Input 101 = binary Gadget, 105 = Gadget HDF5 (default), 111 = text
`x y z w`, 112 = text positions-only, 121 = raw binary (count, box, pos, weights, vels).

**Which flags reach the GPU** (`METAL=1` build): the grid deposit, and `--ps-linear-deposit`,
`--ps-volume-weighted`, `--ps-vertex-mass`, `--ps-caustics` and `--ps-exact-deposit` all compose
with `--ps-gpu`. CPU-only: point evaluation (`--sample-points`/`--serve`: double precision by
design, multi-threaded over the tessellation, exact point location), the standard binary's
`--exact-average`, and scalar fields (they fall back with a warning).

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
          tessellation_cache="/Volumes/T7/tess")    #   at most 3 in memory (default: the RAM budget)
Estimator.from_arrays(x, velocities=v, lagrangian=q, values=phi, box=100.0)   # numpy arrays in
```
The answers are bit-identical to a `--sample-points` run on the same points. A small request
only touches the cells near its points (a cell index built at start-up): ~20 us per single
point from Python, 262k-particle snapshot. Velocities come back in peculiar km/s (x sqrt(a),
dispersion x a; `peculiar=False` for raw Gadget units). The binary side, for other clients:
`./PS-DTFE snap.hdf5 unused --serve --MpcUnit 1 ...` -- logs on stderr, a binary protocol on
stdin/stdout (protocol version 2, documented above psServe in
`src/CGAL_triangulation/ps_point_eval.cc`). With `--partition` (or when the auto-tuner finds one
tessellation would not fit and `--tessellation-cache` is given) the server is a COMPOSITE of
partition tessellations: built once in parallel, each with a 64^3 occupancy map ('<cache
file>.occ'), a request visits only the partitions whose map holds one of its points, and at most
`--serve-resident N` of them are in memory (LRU; the rest reload from the cache). Answers are
bit-identical to `--sample-points --partition` with the same split; a later session registers
every partition from the cache without building. `DTFE_PTS_THREADS=N` pins the point-evaluation
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
./DTFE snap.hdf5 out --grid 256 --periodic --CIC --interlace  # anti-aliasing (needs --periodic)

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

# redshift cone (spherical) grid instead of a box, and redshift-space distortions
./DTFE survey.hdf5 out --grid 128 --redshiftCone 0 200 -30 30 -30 30 --origin 0 0 0
./DTFE snap.hdf5 out --grid 256 --periodic --redshiftSpace 0 0 1

# zoom region, sub-sampling, synthetic points, options file
./DTFE snap.hdf5 out --grid 256 --region 0.4 0.6 0.3 0.7 0.45 0.55   # fractions of the box
./DTFE snap.hdf5 out --grid 256 --regionMpc 20 30 20 30 20 30        # Mpc coordinates
./DTFE snap.hdf5 out --grid 256 --randomSample 0.1                   # keep 10% of particles
./DTFE snap.hdf5 out --grid 256 --poisson 64                         # 64^3 random positions
./DTFE snap.hdf5 out --grid 256 --density0 2.66                       # explicit density scale
./DTFE snap.hdf5 out --grid 256 --extensive                          # scalars are extensive
./DTFE snap.hdf5 out --grid 256 --noTest                             # skip the padding test
./DTFE --config my_options.cfg                                       # read options from a file
```

---

## Analysis (`python/analyze.py`, `python/plot/*.py`)

All plot scripts share the CLI from `dtfelib.cli`: `--data-root PATH`, `--sim NAME`,
`--snap N`, `--method auto|ps|dtfe`, `--raw` (unaveraged grids), `--smooth SIGMA_CELLS`,
`--prefix` (an alternate `OUTPUT_PREFIX` grid set).

```bash
python3 python/analyze.py                          # check + compute + plot, default sim
python3 python/analyze.py check                    # which fields/products exist on disk
python3 python/analyze.py compute 99               # derived products for snapshot 99
python3 python/analyze.py plot --only plot_void_population.py plot_cosmic_web.py
python3 python/analyze.py all --sim TNG300-3-Dark

python3 python/plot/plot_PS_DTFE.py --sim TNG50-3-Dark --snap 99      # PS field maps
python3 python/plot/plot_DTFE.py --method dtfe --raw                  # standard DTFE maps
python3 python/plot/plot_void_population.py --snap 99 --smooth 10
python3 python/plot/plot_void_tracking.py --sim TNG50-3-Dark
python3 python/plot/plot_halo_tracking.py --sim TNG50-3-Dark --method ps
python3 python/plot/plot_marked_correlation_BBKS.py --snap 99
# also: plot_cosmic_web, plot_eigenvalues, plot_shear_triaxial, plot_velDiv_den,
#       plot_PDF_CDF, plot_phi_delta, plot_tidal_correlation, plot_shape_filter,
#       plot_contour_filter, plot_ellipses_contour_3D, plot_smoothing_comparison,
#       plot_conclusions_synthesis

# point-evaluation figure pipeline (grid-free, publication resolution)
python3 python/tools/make_image_plane.py --combined combined_099.hdf5 --axis z --nu 8192 -o plane
python3 python/plot/plot_pointeval.py                     # one figure per field, all snapshots
python3 python/plot/plot_pointeval_panels.py              # 3-panel density/sheet/N_streams
python3 python/tools/compare_deposits.py A_prefix B_prefix   # A/B sampled vs exact deposit

# decoding the caustic mask ('.causticClass' / '.pts_caustic') in python:
#   from dtfelib.io import (caustic_is_fold, caustic_collapse_max, caustic_is_cusp,
#                           caustic_is_swallowtail)
#   fold = caustic_is_fold(m)          # both parities -> the A2 fold set (same as '.caustic')
#   k    = caustic_collapse_max(m)     # 0 uncollapsed, 1 wall, 2 filament, 3 node (-1 = untouched)
#   cusp = caustic_is_cusp(m)          # bit7, only under --ps-caustic-cusps; 0.01% false positives
#   swal = caustic_is_swallowtail(m)   # bit8, same flag; proximity to an A4, ~146x contrast.
#                                      # NOT a subset of cusp -- OR them for the A3-and-above set

# single-stream cells need BOTH grids: '.streams' counts sub-samples only, so a fold thinner than
# the sampling can cross a cell unseen -- bit 1 of '.hidden_streams' marks exactly those cells (a
# folded tetrahedron overlaps them); bit 2 marks mass from a sub-sample tet (usually the same stream).
#   single = (np.abs(fs.load("streams") - 1) < STREAM_TOL) & ((np.rint(fs.load("hidden_streams")).astype(int) & 1) == 0)
#   v1 = fs.velocity_single_stream()   # does exactly this (streams alone on pre-2026-09-28 grids)
```

---

## Graphical launcher (`scripts/gui.sh`, `python/gui/`)

```bash
scripts/gui.sh                 # the window (PySide6 from ~/.venvs/dtfe-gui; scripts/install.sh sets it up)
scripts/gui.sh --setup         # the first-run dialog again: data folder, build, TNG API key, demo
scripts/gui.sh --smoke         # build the window, print what its jobs would use, quit
```
Tabs: Data (download + merge), Grids (run_ps_dtfe.sh / run_dtfe.sh, presets), Custom snapshot (the
binary on any snapshot file), Pipeline, Plots, Explore (maps of any run + click-to-query server);
beside the log: Queue, Figures, Runs (every run log on disk, stale-output flags, load settings,
move to Trash). Every job shows its commands; Export saves them as a bash script or SLURM job.

## Tests

```bash
python3 tests/run_tests.py                    # standard-DTFE suite (15 tests; -v verbose)
python3 tests/run_tests.py --update-ref       # regenerate the tracked density reference
tests/ps_smoke_test.sh                        # build + Zel'dovich pancake sanity
tests/ps_smoke_test.sh --no-build             # reuse the existing binary (all .sh support this)
python3 tests/ps_regression_test.py           # analytic pancake + tracked metric baseline
python3 tests/ps_regression_test.py --update-baseline
tests/ps_parallel_check.sh --no-build         # serial vs multi-thread race detector
tests/ps_point_eval_check.sh --no-build       # --sample-points / ids / gradients battery
N=32 GRID=64 tests/ps_point_eval_check.sh --no-build    # bigger problem via env
tests/ps_linear_deposit_check.sh --no-build   # --ps-linear-deposit checks (+ GPU parity,
                                              #   + gated --ps-exact-deposit section)
tests/ps_halo_release_check.sh --no-build     # --ps-halo-release battery (pancake + crossed)
tests/ps_vertex_mass_check.sh --no-build      # --ps-vertex-mass mass conservation + 1-alpha slope
tests/ps_volume_weighted_check.sh --no-build  # --ps-volume-weighted per-field weighting contract
tests/dtfe_point_eval_check.sh --no-build     # STANDARD-binary --sample-points battery
tests/point_exact_serve_check.sh --no-build   # exact point location (no double counts, not
                                              #   even at particle positions; partition- and
                                              #   thread-invariant) + --serve/Estimator ==
                                              #   --sample-points bit for bit, both binaries;
                                              #   composite (partitioned, out-of-core) servers;
                                              #   per-stream q(x) + scalars; from_arrays; the
                                              #   tessellation cache keyed on --periodic
tests/dtfe_metal_check.sh                     # CPU vs GPU parity, standard DTFE
tests/ps_auto_tune_check.sh                   # auto-tuner decisions (local only: macOS-isms)
tests/ps_scratch_check.sh                     # --scratch-dir out-of-core (local only)
tests/ps_tessellation_cache_check.sh --no-build  # --tessellation-cache: bit-identity + staleness
tests/ps_caustic_class_check.sh --no-build     # .causticClass stratification invariants
tests/run_scripts_check.sh --no-build          # scripts/run_ps_dtfe.sh contract: env vs -g
                                               #   precedence, InitialCoordinates fallback,
                                               #   honest exit codes, no macOS-isms
python3 tests/py_dtfelib_test.py              # dtfelib suite (real TNG data)
python3 tests/py_gui_runspec_test.py          # GUI job specs, run logs, presets, Explore data (no Qt)
~/.venvs/dtfe-gui/bin/python tests/py_gui_app_test.py   # the window, offscreen: real runs, demo,
                                              #   Explore + query server, Runs, setup dialog
python3 python/selftest.py                    # sandboxed pipeline+plots selftest (~2 min)
python3 tests/ps_3d_test.py                   # crossed-waves 3D stream-count analytics
python3 tests/ps_convergence_test.py          # density profile -> analytic convergence
python3 tests/ps_nonperiodic_test.py          # isolated cloud: mass, --box region faces, --partition == one triangulation
python3 tests/ps_standard_cross_check.py      # PS vs standard DTFE where streams==1
tests/ps_scaling_benchmark.sh                 # strong-scaling table (MIN_EFFICIENCY=0.4)

# synthetic snapshot generator used by the tests (Gadget HDF5, input type 105):
python3 tests/generate_ps_test_data.py --out snap.hdf5 --n 32 --box 100 \
    --amplitude-factor 1.8      `# >1 = multi-stream pancake; 0 = uniform` \
    --jitter-frac 0.05 --seed 42 \
    --crossed-waves             `# 3D displacement: stream counts {1,3,9,27}` \
    --margin-frac 0.2           `# non-periodic clump in vacuum instead`
```

CI (`.github/workflows/ci.yml`) runs the Linux CPU build plus `run_tests.py`,
`ps_smoke_test.sh`, `ps_regression_test.py` and the four PS flag suites. The Metal GPU path
and the two macOS-specific suites (`ps_auto_tune_check`, `ps_scratch_check`) are local-only.
