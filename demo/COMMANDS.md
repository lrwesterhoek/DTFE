# Command cheatsheet

Every script in the repo with its useful flag combinations. All paths are relative to the
repository root; quote the repo path (it contains spaces). Scripts that need h5py should be
run with a python whose h5py works (`/opt/homebrew/bin/python3.14` on the development Mac).

---

## Build (`Makefile`, `install.sh`)

```bash
./scripts/install.sh                      # one-command clean rebuild, best backend auto-detected
                                  #   (Apple Silicon -> METAL=1, else CPU-only)
./scripts/install.sh --cpu                # force CPU-only binaries
./scripts/install.sh --no-deps            # skip dependency installation, just clean + rebuild
./scripts/install.sh --jobs 4             # limit parallel build jobs
./scripts/install.sh --docker             # containerized LINUX image instead (never Metal)

make DTFE METAL=1 -j10            # standard DTFE, Apple-GPU '_a' interpolation backend
make PS-DTFE METAL=1 -j10         # phase-space DTFE, Apple-GPU deposit backend
make PS-DTFE TBB=1                # opt-in parallel CGAL triangulation (slower at small N!)
make library                      # shared libDTFE
make clean                        # binaries + objects + auto-dependency .d files
make deps-check                   # verify headers/libraries exist before a long compile
make DTFE $(cat o/.build_mode)    # rebuild WITHOUT downgrading the current GPU mode
make PS-DTFE $(cat o_ps/.build_mode)
```

The GPU mode is stamped in `o/.build_mode` / `o_ps/.build_mode`; switching modes wipes the
object directory automatically. Header edits rebuild all affected objects (auto-deps).

---

## Download (`download_snapshots.sh`)

API key: `-k KEY`, or `TNG_API_KEY` env, or `~/.tng_api_key` (recommended). Everything lands
under `$DATA_ROOT/<SIM>/` (`config.sh`: `DTFE_DATA_ROOT`, default `~/output`); `-d` overrides.
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
python3 python/tools/merge_HDF5.py -d ~/output/TNG50-3-Dark 0 4 17 33 50 99   # snapshots
python3 python/tools/merge_HDF5.py -d ~/output/TNG50-3-Dark --groupcats 0 99  # group catalogs
python3 python/tools/merge_HDF5.py -d ~/output/TNG50-3-Dark --trees           # SubLink trees
python3 python/tools/merge_HDF5.py -d ~/output/TNG50-3-Dark --ics             # ICs conversion

# everything for one sim, then delete the (verified) source chunks to save space:
python3 python/tools/merge_HDF5.py -d ~/output/TNG50-3-Dark \
    --snapshots-too --groupcats --trees --ics --delete-chunks 0 4 17 33 50 99

# upgrade PRE-unification combined files in place (idempotent, chunk-free):
python3 python/tools/merge_HDF5.py -d ~/output/TNG50-3-Dark --fix-units

# standalone ICs conversion (what --ics wraps):
python3 python/tools/convert_ic_units.py ~/output/SIM/ics.hdf5 ~/output/SIM/combined_ics.hdf5
```

`--hubble 0.6774` is only a fallback when a file lacks `HubbleParam`; `-n N` forces the chunk
count instead of globbing.

---

## Run scripts (`run_ps_dtfe.sh`, `run_dtfe.sh`)

Config comes from `config.sh` (`DTFE_DATA_ROOT`, `DTFE_SIM` env) plus flags; positional
arguments override the snapshot list. Partitioning is auto-tuned unless `PARTITION` /
`MAX_CONCURRENT` are set.

```bash
./scripts/run_ps_dtfe.sh                                  # PS-DTFE, default sim + snapshot ladder
./scripts/run_ps_dtfe.sh -s TNG300-3-Dark 99              # one sim, one snapshot
./scripts/run_ps_dtfe.sh -g 512 -n 2 -m 99                # grid 512^3, nSub=2, Metal deposit
DTFE_SIM=TNG100-3-Dark ./scripts/run_ps_dtfe.sh 0 50 99   # sim via env
PARTITION="5 5 5" MAX_CONCURRENT=2 ./scripts/run_ps_dtfe.sh -s TNG300-3-Dark 99   # manual memory plan
THREADS=4 AVG_SUBSAMPLES=1 ./scripts/run_ps_dtfe.sh 99    # cap threads; cell-centre only (fast, no '_a')

./scripts/run_dtfe.sh -s TNG50-3-Dark -g 512 99           # standard DTFE
./scripts/run_dtfe.sh -m 99                               # Metal '_a' interpolation
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

# point evaluation at arbitrary positions (text 'x y z' per line, or raw float64 Nx3);
# the STANDARD binary supports the same flags/formats (streams = 0/1 coverage; no
# --per-stream/--per-stream-ids, single triangulation so no --partition)
./PS-DTFE snap.hdf5 out --grid 32 --periodic --input 105 --MpcUnit 1 \
    --sample-points points.txt \
    --per-stream               `# + ragged per-stream density/velocity records` \
    --per-stream-ids           `# + stream identities (4 sorted vertex ParticleIDs)` \
    --pts-den-grad             `# + density gradient per point (and per stream)` \
    --ps-stream-density geometric   # per-stream estimator: 'dtfe' (default) | 'geometric'
./DTFE snap.hdf5 out --grid 32 --periodic --input 105 --MpcUnit 1 \
    --sample-points points.txt --pts-den-grad

./PS-DTFE snap.hdf5 out ... --ps-linear-deposit    # linear-profile mass-conserving deposit
./PS-DTFE snap.hdf5 out ... --avg-subsamples 2     # cheaper '_a' pass (8 sub-points)
./PS-DTFE snap.hdf5 out ... --ps-caustics          # + '.caustic' fold-flag grid (CPU deposit)
./PS-DTFE snap.hdf5 out ... --ps-halo-release 300  # halo-interior tets -> centroid deposit
./PS-DTFE snap.hdf5 out ... --ps-exact-deposit     # exact r3d tet-cell deposit (CPU, slow)
./DTFE   snap.hdf5 out ... --exact-average         # exact r3d '_a' averaging (CPU, slow)
./PS-DTFE --full_help                              # every option with full descriptions
```

Common to both: `--grid N [NY NZ]`, `--periodic`, `--input 101|105|111|112|121|122`,
`--MpcUnit <units per Mpc>`, `--partition X Y Z`, `--max-concurrent N`, `--verbose 0..3`,
`--region`, `--padding`. Input 101 = binary Gadget, 105 = Gadget HDF5 (default), 111 = text
`x y z w`, 112 = text positions-only, 121 = raw binary (count, box, pos, weights, vels).

---

## Analysis (`python/analyze.py`, `python/plot/*.py`)

All plot scripts share the CLI from `dtfelib.cli`: `--data-root PATH`, `--sim NAME`,
`--snap N`, `--method auto|ps|dtfe`, `--raw` (unaveraged grids), `--smooth SIGMA_CELLS`.

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
```

---

## Tests

```bash
python3 tests/run_tests.py                    # standard-DTFE suite (13 tests; -v verbose)
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
tests/dtfe_point_eval_check.sh --no-build     # STANDARD-binary --sample-points battery
tests/dtfe_metal_check.sh                     # CPU vs GPU parity, standard DTFE
python3 tests/py_dtfelib_test.py              # dtfelib suite (12 checks; real TNG data)
python3 python/selftest.py                    # sandboxed pipeline+plots selftest (~2 min)
python3 tests/ps_3d_test.py                   # crossed-waves 3D stream-count analytics
python3 tests/ps_convergence_test.py          # density profile -> analytic convergence
python3 tests/ps_nonperiodic_test.py          # isolated-cloud mass conservation
python3 tests/ps_standard_cross_check.py      # PS vs standard DTFE where streams==1
tests/ps_scaling_benchmark.sh                 # strong-scaling table (MIN_EFFICIENCY=0.4)

# synthetic snapshot generator used by the tests (Gadget HDF5, input type 105):
python3 tests/generate_ps_test_data.py --out snap.hdf5 --n 32 --box 100 \
    --amplitude-factor 1.8      `# >1 = multi-stream pancake; 0 = uniform` \
    --jitter-frac 0.05 --seed 42 \
    --crossed-waves             `# 3D displacement: stream counts {1,3,9,27}` \
    --margin-frac 0.2           `# non-periodic clump in vacuum instead`
```
