# The DTFE public software

The DTFE public code is a C++ implementation of the **Delaunay Tessellation Field Interpolation (DTFE)** method. Its purpose is to interpolate quantities stored at the location of an unstructured set of points to a regular grid using the maximum of information contained in the input points set. In particular, the code can calculate the following cosmological quantities:
* the density field - this is calculated directly from the point distribution,
* the velocity field and derivatives (e.g. gradient, divergence, vorticity) - uses the velocity at each particle position, and
* general vector quantities and their derivatives - these quantities must be given as input for each point in the set.

The code was written with the purpose of analysing cosmological simulations and galaxy redshift survey. Even though the code was designed with astrophysics in mind, it can be used for problems in a wide range of fields where one needs to interpolate from a discrete set of points to a grid.

The code was designed using a modular philosophy and with a wide set of features that can easily be selected using the different program options. The DTFE code is also written using OpenMP directives which allow it to run in parallel on shared-memory architectures.

The code comes with a complete [documentation](docs/DTFE_user_guide.pdf) and with a multitude of examples that detail the program features. A test dataset and analysis of the code output is given in the [demo directory](demo): `./DTFE demo/z0_64.gadget out --grid 64 --field density --periodic --padding 12` works out of the box (the box is periodic, and its voids are wider than the default padding of 5 particle spacings: with it about 1% of the cells would lie outside the tessellation, which the run reports). [demo/COMMANDS.md](demo/COMMANDS.md) is a cheat sheet of every script and binary with its useful flag combinations.

The public release of the code is summarised in the arxiv publication [Cautun et al. (2011)](https://ui.adsabs.harvard.edu/abs/2011arXiv1105.0370C/abstract) and it is based on the method paper [Schaap and van de Weygaert (2000)](https://ui.adsabs.harvard.edu/abs/2000A%26A...363L..29S/abstract).


## The DTFE method
The Delaunay Tessellation Field Interpolation (DTFE) method represents the natural way of going from discrete samples/measurements to values on a periodic grid and it is especially suitable for astronomical data due to the following reasons:
* Preserves the multi-scale character of the point distribution. This is the case in numerical simulations of large scale structure where the density varies over more than 6 orders of magnitude.
* Preserves the local geometry of the point distribution. This is important in recovering sharp features like the different components of the cosmic web (i.e. clusters, filaments, walls and voids).
* The method does not depend on user defined parameters or choices.
* The interpolated fields are volume weighted (versus mass weighted quantities in most other interpolation schemes). This can have a significant effect especially when comparing with analytical predictions which are volume weighted.

For detailed information about the DTFE method see [Schaap and van de Weygaert (2000)](https://ui.adsabs.harvard.edu/abs/2000A%26A...363L..29S/abstract), [van de Weygaert and Schaap (2009)](https://ui.adsabs.harvard.edu/abs/2009LNP...665..291V/abstract), and [Cautun et al. (2011)](https://ui.adsabs.harvard.edu/abs/2011arXiv1105.0370C/abstract).

| <img src="figures/DTFE_filament.png" width="600" title="An illustration of the adaptive nature of the DTFE method."> |
|:------:|
| Figure 1: *An illustration of the 2D Delaunay tessellation of a set of particles from a cosmological simulation. Courtesy: Willem Schaap.* |

| <img src="figures/DTFE_paper_density.png" width="800" title="Examples of DTFE density fields."> |
|:------:|
| Figure 2: *An example of the DTFE density field form a cosmological simulation. The right panel shows the same result but now using the smoothed particle hydrodynamics (SPH) method.* |

| <img src="figures/DTFE_paper_velocity.png" width="800" title="Illustration of the DTFE velocity field."> |
|:------:|
| Figure 3: *A map of the DTFE computed velocity flow (left panel) and velocity divergence (right panel) corresponding to the density field shown in Figure 2.* |


## Summary of software features

* Works in both 2 and 3 spatial dimensions.
* Interpolates the fields to three different types of grids:
  + Regular rectangular and cuboid grid - useful for cosmological simulation.
  + Redshift cone (spherical coordinates) grid - useful for galaxy redshift survey or for mock observations.
  + User given sampling points - can describe any complex or non-regular sampling geometry
* Returns both the value at the centre of each cell of the interpolation grid as well as the value averaged over each cell.
* Interpolates the density, velocity, velocity gradient, divergence, shear and vorticity, any number of further fields and their gradients, and the T-web and V-web classes.
* Each sample point has a weight associated to it to represent multiple resolution N-body simulations and observational biases for galaxy redshift surveys.
* Periodic boundary conditions, zoom-in regions, and splitting the data into partitions processed in parallel on shared-memory machines.
* A phase-space (Lagrangian) variant, `PS-DTFE`, that recovers the multi-stream structure of the cosmic web: the multi-stream density, the stream count, the velocity dispersion and the caustic surfaces.
* Exact, sampling-free cell averages as an alternative to Monte-Carlo averaging, and evaluation at arbitrary points with exact point location.
* An interactive mode: the tessellation is built once and queried from Python (`dtfelib.Estimator`) at any points; boxes too large for one tessellation are served as a composite of partitions kept on disk.
* Automatic choice of the domain decomposition from the data and the machine's memory, crash-safe outputs and resumable runs, and an out-of-core mode for grids larger than the memory.
* A tessellation cache on disk that later runs on the same data reuse.
* Optional GPU deposit on Apple Silicon (Metal) and, as a port of the same kernels, on NVIDIA (CUDA) or AMD (HIP) GPUs; an optional double-precision build.
* For comparison: nearest grid point (NGP), cloud in cell (CIC), triangular shape cloud (TSC), piecewise cubic spline (PCS), SPH and Voronoi interpolation.
* Returns the Delaunay tessellation of the given point set, and can be used as an external library.
* Batch drivers for IllustrisTNG snapshots, a Python analysis stack (`python/dtfelib`, `python/plot`) and a graphical launcher.


## Installation and Building

### Supported Platforms
- **macOS** (Intel and Apple Silicon)
- **Linux** (Ubuntu, Debian, Fedora, CentOS, RHEL, and other distributions)

### Prerequisites

#### macOS
1. **Install Xcode Command Line Tools:**
   ```bash
   xcode-select --install
   ```

2. **Install Homebrew** (if not already installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

3. **Install dependencies:**
   ```bash
   brew install gsl boost cgal mpfr gmp hdf5 fftw llvm libomp
   ```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install build-essential
sudo apt-get install libgsl-dev libboost-all-dev libcgal-dev libmpfr-dev libhdf5-dev libgmp-dev libfftw3-dev
```

#### Linux (Fedora/RHEL/CentOS)
```bash
# For Fedora
sudo dnf groupinstall "Development Tools"
sudo dnf install gsl-devel boost-devel CGAL-devel mpfr-devel hdf5-devel gmp-devel fftw-devel

# For older RHEL/CentOS
sudo yum groupinstall "Development Tools"
sudo yum install gsl-devel boost-devel CGAL-devel mpfr-devel hdf5-devel gmp-devel fftw-devel
```

#### Linux (Arch/Manjaro)
```bash
sudo pacman -S base-devel gsl boost cgal mpfr hdf5 gmp fftw
```

Optional: `tbb` (the parallel triangulation of PS-DTFE) and `libdeflate` (faster tessellation-cache compression);
the Makefile uses them when it finds them.

### Quick start

`./scripts/install.sh` installs any missing dependencies, builds both binaries with the best backend for the machine
(Metal on Apple Silicon, CUDA/HIP when `nvcc`/`hipcc` is found, else the CPU), rebuilds any double or 2D pair already
built, installs the Python package and sets up the graphical launcher. Flags: `--cpu`, `--no-deps`, `--jobs N`,
`--double` (also the double-precision pair), `--no-python`, `--no-gui` (e.g. on a cluster), `--no-app`, `--docker`
(the Linux container image instead; never Metal); `./scripts/install.sh --help` explains each. A failed compile
leaves the previous binaries in place and prints the end of its build log.

### Building by hand

```bash
make test-platform        # the detected platform, compiler and paths
make deps-check           # every required header; prints the install command for anything missing
make DTFE PS-DTFE         # the standard and the phase-space binary (add -j8)
make library              # the shared library libDTFE (.dylib / .so)
make clean                # every binary set, libDTFE and the objects
python3 -m pip install -e python   # dtfelib (editable: it finds ./DTFE and ./PS-DTFE; else set DTFE_BIN_DIR)
```
`./DTFE --version` / `./PS-DTFE --version` print the git revision and build time, which every run also writes into
its log (`Build:`).

Build variables (combine them on one `make` line):

| variable | effect |
|---|---|
| `METAL=1` | Apple Silicon: the standard binary's `_a` interpolation (`--gpu`) and the PS-DTFE deposit (`--ps-gpu`) on the GPU, with a CPU fallback |
| `CUDA=1` | Linux/NVIDIA: the same kernels ported to CUDA (`nvcc`; `CUDA_PATH`, `GPU_ARCH=sm_86`) |
| `HIP=1` | Linux/AMD ROCm (`hipcc`; set `GPU_ARCH`, e.g. `gfx90a`) |
| `CUDAEMU=1` | the CUDA/HIP sources compiled as C++ against a CPU emulation of the device, for validating the port without a GPU (`tests/cudaemu_check.sh`) |
| `DOUBLE=1` | the double-precision pair `DTFE-double` / `PS-DTFE-double` (see Precision) |
| `DIM=2` | the 2D pair `DTFE-2d` / `PS-DTFE-2d` (`-2d-double` with `DOUBLE=1`); CPU only |
| `TBB=auto` / `1` / `0` | the parallel Delaunay insert of PS-DTFE (`--parallel-triangulation`): when found / required / left out |
| `LIBDEFLATE=auto` / `1` / `0` | libdeflate instead of zlib for the tessellation cache (both write plain gzip) |
| `BUILD_MODE=debug` | debug symbols, no optimization, sanitizers |
| `EXTRA_FLAGS`, `CC_OVERRIDE`, `<LIB>_PATH_OVERRIDE` | extra compiler flags, another compiler, a library elsewhere (see the Makefile's header) |

One GPU backend per build. The CUDA/HIP port compiles in CI and passes every GPU parity suite through the CPU
emulation, but has not been run on NVIDIA or AMD hardware: its first run on a GPU machine should be the parity checks in
[metal/README.md](metal/README.md). Each binary set has its own object directories (`o/`, `o_ps/`; `o_d/`, `o_ps_d/`;
`o_2d/`, `o_ps_2d/`), so the sets coexist. The build mode is stamped there (`o/.build_mode`); a mode switch wipes the
objects, and `make DTFE $(cat o/.build_mode)` rebuilds without changing the mode. The run scripts find the GPU mode
from the `.gpu_mode_metal` / `_cuda` / `_hip` stamp.

**Compile-time options.** The Makefile's `OPTIONS_COMMON` block sets the compiled-in features and defaults:
`-DVELOCITY`, `-DSCALAR` with `-DNO_SCALARS=1` (one scalar component; the GPU deposits only that case),
`-DINPUT_FILE_DEFAULT=105` (Gadget HDF5), `-DMPC_UNIT=1000` (input lengths in kpc), `-DOUTPUT_FILE_DEFAULT=101`
(binary output), `-DOPEN_MP`, `-DTEST_PADDING`, `-DTRIANGULATION`. Edit it there; see the Makefile for the full list.

**Precision.** `DOUBLE=1` keeps every position, density, field and accumulator in double from the input read on
(the Gadget-HDF5 reader then keeps TNG's double coordinates, which the default build rounds on read) and writes float64
grids (`dtfelib` and the launcher read either), at about twice the memory. The run scripts select the pair with
`DTFE_PRECISION=double`, the launcher with its Precision setting. The GPU kernels are single precision: a double build
hands the GPU float copies of its tetrahedra, so its GPU stream counts and caustic flags may differ from its CPU's in up
to 1e-4 of the cells (identical in the single build). The PS-DTFE grids are exact 64-bit fixed-point sums on the GPU
and the CPU (single build: identical from run to run at any thread count; `PS_METAL_FLOAT_MASS=1` /
`PS_GPU_FLOAT_MASS=1` and `PS_METAL_FLOAT_MOMENTS=1` / `PS_GPU_FLOAT_MOMENTS=1` bring back float sums for comparison).
A share without a 64-bit integer value (NaN, infinity, beyond 2^63 units) is left out and counted, and the run ends
with a warning naming the grid.

**2D.** `make DTFE PS-DTFE DIM=2` reads the same Gadget-HDF5 layout with two-column `Coordinates`, `Velocities` and
`InitialCoordinates` (`python3 tests/generate_ps_test_data.py --dim 2` writes such files). Every field, the T-web
(classes 0 void, 1 filament, 2 node), the sampled and exact deposits, `--exact-average`, `--ps-caustics` (no swallowtail
bit), `--ps-window`, `--partition`, the tessellation cache, `--sample-points` (`x y` points) and `--serve` work in 2D;
the GPU and `--parallel-triangulation` are 3D only. A snapshot whose coordinates have the wrong number of columns for
the binary is refused with the name of the binary to use.

### Tests

`tests/ci_suite.sh <stage>` is the one list of suites that CI (`.github/workflows/ci.yml`) and the Docker image run:

| stage | runs |
|---|---|
| `build` | builds the single, double and 2D pairs (CPU) |
| `core` | every CPU suite against `./DTFE` and `./PS-DTFE`, the auto-tune and `--scratch-dir` suites, the dtfelib and void suites, the docs check, `python/selftest.py` |
| `double` | `tests/ps_double_check.sh`, then the program suites against the double pair |
| `2d` | `tests/ps_2d_check.sh` (the 2D pair against analytic 2D flows) |
| `gui` | the launcher's job specifications; the end-to-end window test when `DTFE_GUI_PYTHON` names a Python with PySide6 |
| `oldpy` | every `.py` parses under Python 3.10 |
| `variants` | `TBB=0 LIBDEFLATE=0`, `make library` and `demo/examples` linked against it |
| `all` | `build core double 2d` |

CI runs six jobs: build-and-test (build, core, gui, double, 2d), old-python, compile-variants, cudaemu (the GPU suites
through the emulation at warp width 32 and 64), cuda-compile and hip-compile. The GPU suites need a GPU build and run
by hand: `tests/dtfe_metal_check.sh`, `tests/gpu_scalar_check.sh`, `tests/gpu_mass_check.sh`,
`tests/gpu_retry_check.sh`, `tests/dtfe_gpu_retry_check.sh` (and `tests/cudaemu_check.sh` on a `CUDAEMU=1` build).
`DTFE_TEST_PRECISION=double tests/<suite>` runs a suite against the double pair. The standard-DTFE regression
(`tests/run_tests.py`) compares a periodic 1000-particle run with `tests/reference/regression_periodic.txt` to 3e-4
relative; after an intended change of the estimator regenerate it with `python3 tests/run_tests.py --update-ref`.


## Running the binaries

```bash
./DTFE    <input file> <output prefix> [options]      # standard DTFE (Eulerian tessellation)
./PS-DTFE <input file> <output prefix> [options]      # phase-space DTFE (Lagrangian tessellation)
./PS-DTFE --help          # the options in short;  --full_help: every option with its default and caveats
```
The `--full_help` texts are the reference for every option; [demo/COMMANDS.md](demo/COMMANDS.md) collects typical
command lines. The most used: `--grid N [NY NZ]`, `--periodic`, `--field ...`, `--input 101|105|106|111|112|121|122`
(binary Gadget, Gadget HDF5 (default), its HI data, text `x y z w`, text positions, two raw binary layouts),
`--MpcUnit` (input length units per Mpc; default 1000, i.e. kpc), `--box`, `--region` / `--regionMpc` (a zoom),
`--padding` (default 5 mean particle spacings), `--partition X Y Z`, `--max-concurrent N`, `--verbose 0..3`,
`--config file` (options from a file).

**Outputs.** Each field is written as `<prefix>.<name>` (the cell-centre value) or `<prefix>.a_<name>` (the `_a`
cell average): raw float32 (float64 in a double build), N^3 cells in C order times the components, no header
(`--output 111` writes text). The density is rho/rho_bar.

| `--field` | file | components |
|---|---|---|
| `density`, `density_a` | `.den`, `.a_den` | 1 |
| `velocity` | `.vel` | 3 (the input's units; dtfelib returns peculiar km/s) |
| `gradient` / `divergence` / `shear` / `vorticity` | `.velGrad` / `.velDiv` / `.velShear` / `.velVort` | 9 / 1 / 5 / 3 |
| `tweb`, `vweb` | `.tweb` + `.twebEig`, `.velVweb` + `.velVwebEig` | class (0 void, 1 wall, 2 filament, 3 node) + 3 eigenvalues |
| `dispersion` (PS-DTFE) | `.velDisp` + `.velDispTensor` | trace + 6 (xx xy xz yy yz zz) |
| `scalar`, `scalarGradient` | `.scalar`, `.scalarGrad` | the scalar dataset and its gradient |
| every grid pass (PS-DTFE) | `.streams` / `.a_streams`, `.hidden_streams` / `.a_hidden_streams` | stream count; bitmask of streams the count misses |
| `--ps-exact-deposit` | `.tetTouch` | tetrahedra touching the cell (not a stream count) |
| `--ps-caustics` | `.caustic`, `.causticClass` | fold flag; caustic bitmask (one file, no `a_` twin) |
| `--sample-points` | `.pts_den`, `.pts_vel`, `.pts_streams`, ... | one record per point (`.pts_den` float64 rho/rho_bar) |

The T-web needs a periodic box; both web classes take `--lambda_th` (binary default 0; the run scripts pass 0.3).
Standard DTFE also writes `*.badGridPoints*` files when the padding left cells outside the tessellation. While a run
goes it keeps `<prefix>.runlog.progress` (a JSON file: phase, partitions done, ETA, memory, GPU retries;
`DTFE_NO_PROGRESS=1` switches it off). Every output is written as `<file>.part.<pid>` and the set is renamed into
place after the last one; if the output folder is missing at that point (an unmounted disk) the binary waits for it
(`DTFE_OUTPUT_WAIT` seconds, default 300), then writes to `$TMPDIR/dtfe-rescue/...`, prints the `mv` command and
exits with status 3.

**Interlacing, mass assignment.** `--interlace` grids the box a second time moved by half a cell and averages the two
density grids in Fourier space (any method, PS-DTFE included); without `--periodic` it needs a `--region` at least half
a cell inside the box. Only the density is interlaced. The CIC/TSC/PCS/NGP kernels get periodic images of their reach
and are clipped at a face or partition seam.


## Phase-Space DTFE

Standard DTFE builds the Delaunay tessellation in Eulerian (present-day) space and returns a single-valued density at
each point. **Phase-Space DTFE** builds it in Lagrangian (initial-condition) space and follows it as it is deformed to
the present day. The deformed tessellation folds over onto itself, so the method recovers the *multi-stream*
structure of the cosmic web: the density at a point is the sum over every stream (folded tetrahedron) covering it, and
the number of streams is a field of its own. It follows the phase-space tessellation approach of Abel, Hahn & Kaehler
(2012) and Shandarin, Habib & Heitmann (2012).

It needs **two positions per particle**, the Eulerian and the Lagrangian one, from a Gadget-HDF5 file: an
`InitialCoordinates` dataset in the same file, or a separate initial-conditions snapshot (`--lagrangianInput`):
```bash
make PS-DTFE METAL=1     # METAL=1 on Apple Silicon (CUDA=1 / HIP=1 on Linux) for --ps-gpu
./PS-DTFE snapshot.hdf5 output_root --grid 256 --periodic --field density_a --lagrangianInput ics.hdf5
```

The `_a` fields are cell averages over an nSub^3 sub-grid (`--avg-subsamples`, default 3); the cost grows as nSub^3.
The options that shape the result (the drivers' production settings are marked):

| option | what it does |
|---|---|
| `--ps-gpu` | the deposit on the GPU (needs a GPU build; otherwise ignored with a warning); every grid field, the sampled and the exact deposit |
| `--ps-vertex-mass` (production) | each particle's mass shared over its tetrahedra: chart-independent masses, removing the bias rho_bar V_lag carries when TNG's z = 127 positions are the Lagrangian input |
| `--ps-volume-weighted` (production) | velocity moments weighted by Eulerian volume, the standard-DTFE convention; the dispersion stays mass-weighted |
| `--ps-caustics` (production) | the fold flag `.caustic` and the bitmask `.causticClass` |
| `--ps-caustic-cusps` | with `--ps-caustics`: the cusp (bit 7) and swallowtail (bit 8) indicators; about 1.8x the caustic pass |
| `--ps-halo-release D` | tetrahedra with a stream density above D (rho/rho_bar) are deposited whole at their centroid cell: the mixed halo interiors (Stücker et al. 2021) |
| `--ps-exact-deposit`, `--ps-linear-deposit`, `--ps-window` | see Field evaluation methods |
| `--ps-alpha-shape` | non-periodic clouds: keep tetrahedra with Lagrangian circumradius up to this many mean spacings (default 3; 0 = the convex hull) |
| `--ps-lag-padding` | the padding of each Lagrangian partition in mean spacings (default 8; 0 = the Eulerian `--padding` rule, under which a TNG partition triangulates 7-9x the particles it owns) |
| `--parallel-triangulation` | parallel Delaunay insert (TBB builds; `DTFE_TBB_THREADS`): faster, but two identical runs then differ at float rounding |

**Streams and hidden streams.** The stream count is measured at the sub-samples of each cell, so it cannot see a fold
thinner than the sampling. `.hidden_streams` bit 1 (value 1) marks exactly the cells a folded tetrahedron overlaps
while the count misses it; bit 2 marks cells that received the mass of a tetrahedron too small to hold a sample (its
mass goes to the cell of its centroid, so mass is conserved). A single-stream mask needs both: `.streams` equal to 1
and bit 1 clear (`FieldSet.velocity_single_stream()` / `single_stream_mask()` in dtfelib). Which tetrahedra contain a
sample is decided exactly, so a `--partition` split gives the same counts as one run and the GPU the same as the CPU.

**Caustics.** `.caustic` is geometric: a cell overlapped by tetrahedra of both orientations, equal to `.streams` > 1
or `.hidden_streams` bit 1, independent of `--avg-subsamples`, the deposit, the split and the GPU. The other bits of
`.causticClass` (the collapse multiplicity k in bits 2-5: 1 wall, 2 filament, 3 node; the umbilic indicator in bit 6;
the cusp and swallowtail indicators in bits 7-8) are recorded at the deposited samples and depend on the sampling.
Grids whose `.caustic` holds the older per-sample flag (it marked 21.0% of the cells where folded tetrahedra overlap
33.6%, TNG100-3-Dark z = 1) need no rerun: `FieldSet.fold_mask()` rebuilds the exact flag from `.streams` and
`.hidden_streams`, and the caustic scripts and the launcher use it. The cusp and swallowtail bits are indicators
(an exact swallowtail lights 74.5% of the cusp cells, a pure cusp 0.5%), not classifications; decode them with
`dtfelib.io.caustic_collapse_max`, `caustic_is_fold`, `caustic_is_cusp`, `caustic_is_swallowtail`.

**Velocity gradients in multi-stream cells.** PS-DTFE's gradient, and the divergence, shear, vorticity and V-web
built on it, is ⟨∇v⟩, the average over the streams of each stream's own gradient; standard DTFE's is ∇⟨v⟩, the
gradient of the mean velocity. They agree in single-stream cells and differ in folds (an inverted stream inside a wall
diverges while the mean flow converges), so the box mean of the PS divergence is not zero and the PS V-web finds few
filaments and nodes. `FieldSet.load(name, mean_field=True)` returns ∇⟨v⟩ for a PS grid (FFT derivatives of `.a_vel`,
with the binary's V-web normalisation and the run's `--lambda_th`); plot_velDiv_den.py, plot_cosmic_web.py,
plot_web_streams.py, plot_shear_triaxial.py, plot_DTFE.py, plot_halo_tracking.py and the launcher's Explore tab use it
for PS grids by default (`--ps-gradient stored` restores the stored grids). At 512^3 the whole-cube computation takes
3-7.5 GB; `fields.mean_field_plane` computes one slice in ~100 MB. A plane's `.pts_velGrad` stays ⟨∇v⟩.

**T-web at the grid scale.** `tweb` / `tweb_a` classify the tidal tensor of the raw density grid, with no smoothing, so
the classes follow each estimator's cell-scale noise (TNG100-3-Dark z = 1: 37.6% void with standard DTFE, 33.8% with
PS-DTFE; 46.8% for both after a one-cell Gaussian). Quote web fractions at a stated smoothing, recomputed from `.a_den`.

**Periodic seam.** A periodic run tiles one Lagrangian period exactly (`tests/ps_periodic_seam_check.sh`). Grids made
before that fix deposit some tetrahedra twice or not at all within a few cells of the box faces (up to 394 cells per
TNG snapshot, a box-mass offset of 1-2e-7); a run whose padding exceeds about a quarter of the box keeps that seam and
says so.

**Boundaries and regions.** A `--box` region cut from a larger cloud keeps exactly the mass inside it (a tetrahedron
crossing the face deposits only its inside share). The flatness cut of degenerate tetrahedra is relative to each
tetrahedron's size, so it does not depend on the length unit.

**CPU deposit threads.** Each thread sums into its own copy of the accumulators; when those copies would not fit (a
fifth of the memory) the deposit switches to slabs along x, bit-identical to one thread at any thread count
(`DTFE_DEPOSIT_SLABS=1` forces, `0` forbids it; `DTFE_DEPOSIT_THREADS` sets the count). `PS_MASS_AUDIT=1` prints each
partition's mass balance.

**Phase-space density.** Standard DTFE's `--approxPSD` approximates the distribution function as f ≈ ρ(x)·g(v), a
second tessellation in velocity space, which breaks down where streams cross. PS-DTFE resolves the streams: the exact
distribution function is the per-stream decomposition of `--sample-points --per-stream`, and Q = ρ/σ³ follows from its
density and dispersion grids (`dtfelib.fields.pseudo_phase_space_density`).


## Field evaluation methods

A cell's value is either the field at the cell centre (`density`, ...) or its average over the cell (`density_a`, ...).

* **Monte Carlo average** (standard DTFE `--method 1`, the binary's default; `--samples`, default an average of 100
  per cell). A tetrahedron is sampled at the first n points of a shared Sobol table, centred on its centroid and in a
  vertex order hashed from its positions, so the average is unbiased (the demo's box mean 1.00002) and the
  realization does not depend on `--partition`, the thread count or the padding (cells agree to 4-6e-6). `DTFE_MC_LEGACY=1` reproduces the
  older uncentred average, which lost 0.17-0.29% of the mass at z <= 2. `--seed` seeds method 2 (`--method 2`, random
  points per cell, which also gives `velocityStd_a`), `--randomSample` and `--poisson`.
* **Exact average** (`--exact-average`, standard DTFE) integrates the linear interpolant over every cell/tetrahedron
  intersection (the vendored [r3d](third_party/r3d/README.md) library, Powell & Abel 2015): no sampling noise, box mean
  1 to 3e-8 on TNG100-3-Dark. With `--gpu` the clipping runs on the GPU (6.6x the CPU on 0.26M particles at 256^3).
  It costs 1.14-1.65x the sampled wall time on a 512^3 TNG run, and is the production setting (`run_dtfe.sh` passes it
  unless `DTFE_EXACT_AVERAGE=0`).
* **Sampled deposit** (PS-DTFE, the default): each stream's mass is spread over the sub-samples inside it.
  `--ps-linear-deposit` weights the sub-samples by the linear density profile instead (still mass-conserving; not
  with `--ps-volume-weighted`).
* **Exact deposit** (`--ps-exact-deposit`, PS-DTFE): each stream is clipped analytically against the grid; use the GPU
  (on TNG300-3-Dark z = 0 it took 2.7x the sampled wall time). PS production stays sampled.
* **Window** (`--ps-window x0 x1 y0 y1 z0 z1`, PS-DTFE, Mpc): only the cells of the full `--grid` inside that box are
  deposited and written, equal to the full run's cells (counts and masks bit for bit, sums to the last fixed-point
  unit). The cost follows the window, not the grid: the launcher's exact zoom is a window of a fine virtual grid. With
  `--partition` and `--tessellation-cache` it loads the cached partitions, skips those whose streams never reach the
  window, and reads no snapshot when every partition is cached. Not with the T-web/V-web, `--region`, `--partNo` or
  `--ps-linear-deposit`.

### Point evaluation

Both binaries evaluate their fields at arbitrary points (`--sample-points FILE`: text `x y z` lines or raw float64
triples), one record per point: `.pts_den`, `.pts_vel`, `.pts_streams` (PS-DTFE: the stream count; standard DTFE: 0/1
coverage), and with options `.pts_velGrad` (`--pts-vel-grad`), the density gradient (`--pts-den-grad`), the caustic bits
(`--ps-caustics`), and in PS-DTFE the per-stream records (`--per-stream`, `--per-stream-ids`), each stream's Lagrangian
origin (`--pts-lagrangian`) and a per-particle dataset along it (`--pts-scalar --scalar-dataset NAME`). Point location
is exact: a point on a shared face, edge or vertex is counted once per stream, also across the periodic boundary and
under any `--partition` split (each partition answers the points in the tetrahedra whose centroid it owns, so the
values equal one triangulation's to rounding). `--ps-stream-density dtfe|geometric` picks the per-stream estimator.
Point evaluation runs on the CPU (double precision, all cores; `DTFE_PTS_THREADS`).

**Image planes follow the grids' conventions** (`--pts-conventions grid`, the default): a PS plane takes the run's
`--ps-vertex-mass` and `--ps-volume-weighted`, so its density contrast matches the grid's (slope 0.97-0.99 against the
exact-deposit grid on a TNG-like synthetic box). `--pts-conventions legacy` keeps the reference implementation's
estimators; planes made with them hold 0.81-0.82 of the grid's density contrast at z = 20 and the launcher's Runs tab
marks them. The T-web and V-web of a plane are sampled from the grid at the points
(`dtfelib.io.PointPlane.sample_grid_field`).

### Interactive point evaluation

```python
from dtfelib import Estimator

with Estimator("snapshot.hdf5") as est:       # PS-DTFE; the tessellation is built here, once
    f = est(points)                           # (N, 3) points in Mpc
    f.density, f.velocity, f.streams, f.dispersion
    cube = est.grid(xs, ys, zs)               # the fields on the grid xs x ys x zs
est = Estimator.from_arrays(x, velocities=v, lagrangian=q, box=100.0)   # particles from numpy arrays
```
Behind it the binary runs with `--serve` (a binary protocol on stdin/stdout, documented above `psServe` in
`src/CGAL_triangulation/ps_point_eval.cc`). Every answer is bit-identical to `--sample-points` for the same points;
`phase_space=False` gives the standard estimator, padded with 15 spacings (`Estimator(padding=...)`). Velocities come
back in peculiar km/s. With `tessellation_cache=dir` a later session starts from the cached tessellation. A box too
large for one tessellation is served as a composite of partitions (`partition=4`, or chosen automatically with a cache
directory when one tessellation would not fit): built once in parallel into the cache, only the partitions a request
reaches are visited, and at most `resident` (`--serve-resident`; default: as many as the memory budget holds) are kept
in memory. Answers equal `--sample-points` with the same split bit for bit in PS-DTFE, to rounding in standard DTFE.
`--serve-progress` writes progress lines for the launcher.


## Memory, partitions and long runs

**Auto-tuning.** Without `--partition` / `--max-concurrent` the binary chooses both from the particle count, the
grid, the fields and the machine, and prints the split and the predicted peak. The budget is 60% of the RAM
(`DTFE_MEM_FRACTION`), never more than the other programs leave free less 4 GB; `DTFE_MEM_BUDGET_GB` sets it outright
(still capped by the free memory). Explicit options always win; `--partition 1` keeps one tessellation. Below 2e6
particles (`DTFE_AUTO_MINN`) a run stays in one domain unless a split into 8 or 27 partitions is predicted to be
clearly faster; such a split merges in a fixed order (reproducible), but differs from one domain at float rounding.
`--auto-tune-report` prints the decision and the prediction as one line and stops (the drivers' `AUTO_TUNE_REPORT=1`,
the launcher's Check memory). An auto-tuned run ends its SUMMARY with the predicted against the measured peak.
- In PS-DTFE `--max-concurrent` is the number of partitions in flight, each writing into the whole grid; while
  several run, a partition waits until the free memory would hold it.
- In standard DTFE the partitions run one after another and `--max-concurrent` is the number of threads in each, every
  thread triangulating its own padded sub-box. The plan replays the split on a count grid of the particles (19-90%
  above the measured peaks of 38 TNG runs). A `--region` run is planned as the region.
- `--scratch-dir DIR` backs every allocation of 1 GB or more (the full grids, the T-web's FFT arrays) with files on a
  local disk, trading disk for RAM with unchanged results. Grids beyond 2^32 cells work (the GPU splits until every
  sub-grid holds fewer).
- A point evaluation's memory grows with the stream count at its points, which the tuner estimates from the plane's
  own slab (42 B per stream at least, 114 with `--pts-vel-grad`).

**Padding.** Every partition (and in standard DTFE every thread's sub-box) carries its own padding. The binary's
default of 5 mean spacings is too short for the emptiest voids at z = 0: `run_dtfe.sh`, dtfelib's standard server and
the launcher pass 15, complete at z = 0 on TNG50/100/300-3-Dark (10 is not). An incomplete padding leaves
`*.badGridPoints*` files, never silently wrong cells (point values within a few spacings of a face can be off, though:
up to 4% at padding 5). PS-DTFE partitions are padded in Lagrangian space by `--ps-lag-padding` (default 8).

**Partition merge (PS-DTFE).** By default the partitions' grids are added in partition index order whatever order they
finish in (`--ordered-merge`, the staged merge), so two identical runs give bit-identical grids. A partition that
finishes early parks its sub-grids in memory, or, when the memory is short, spills them to a file in `--scratch-dir`
(else `$TMPDIR`) with identical results (`DTFE_MERGE_SPILL=0|1`, `DTFE_MERGE_STAGE_GB`, `DTFE_SPILL_MIN_GB`). The run
log's `ordered merge:` line reports it. `--completion-order-merge` adds them as they finish (not bit-reproducible).
`--fixed-point-merge` (opt-in) adds each partition as it finishes into exact 72-bit integer grids, 9 B per cell
instead of 4: order-free, nothing parked, about 1e-7 of each grid's peak from the ordered merge; not in the double
build nor with `--ps-window`, and an overflow stops the run. A partition the GPU gives up on is deposited by the CPU,
which rounds differently, so such a run is reproducible only where the fallback recurs.

**Resume.** A partitioned run of either binary saves its merged grids and the list of finished partitions in
`<prefix>.runlog.checkpoint` (or in `--scratch-dir`). After a crash, `--resume` computes only the missing partitions,
and refuses when the build, the inputs, the split or the command line changed (`--verbose`, `--max-concurrent`,
`--scratch-dir`, `--tessellation-cache` and the overlap and merge-order flags may change). A resumed PS run merged in
index order, and a resumed standard CPU or `--gpu` run at the same thread count, are byte-identical to an uninterrupted
one; `--sample-points` runs resume too. The drivers resume by themselves (`PS_RESUME=0` / `DTFE_RESUME=0` start
over); `DTFE_PS_CHECKPOINT=0` / `DTFE_CHECKPOINT=0` write no checkpoints; `--interlace` runs are not checkpointed.

**Look-ahead.** A partitioned standard `--gpu` run triangulates the next partition while the GPU works on the current
one (`--no-overlap` switches it off); PS-DTFE does the same at `--max-concurrent 1` only (`--ps-no-overlap`). The
outputs are byte-identical either way.

**GPU failures.** The GPU deposit runs as short paced command buffers (kernels on CUDA/HIP) with a snapshot of the
grids every ~10 s of GPU work (`DTFE_GPU_CHECKPOINT_CHUNKS`). A killed buffer (the display watchdog, another program's
GPU hang, out of memory) restores the last snapshot and retries by cause; a watchdog or memory error slows the pace,
which eases back after a clean stretch (`DTFE_GPU_PACE_RECOVER_SEC`). Three failures of one chunk, sixteen in a deposit
or a lost device send that partition (standard DTFE: that sub-domain) to the CPU. The run log ends with
`GPU deposit retries: N (innocent a, timeout b, oom c, other d)`. `DTFE_GPU_FAULT_AT=<k>[:innocent|timeout|oom|persist]`
is the test hook. Standard DTFE's GPU `_a` grids are exact 64-bit fixed-point sums, so two `--gpu` runs are
byte-identical (`DTFE_METAL_FLOAT_GRIDS=1` / `DTFE_GPU_FLOAT_GRIDS=1`: float atomics, for comparison).

**Tessellation cache.** Building the tessellation and its vertex densities is the dominant fixed cost (81% of a run on
884736 particles). `--tessellation-cache DIR` (a local, non-synced folder) writes it once and later runs on the same
inputs load it, whatever their grid, fields or sample points; any change that would alter it (input files, box,
region, padding, periodicity, partition, build) is a miss and a rebuild, never a wrong result (`--verbose 3` says
why). The files are compressed in 16 MB chunks (about 0.15 KB per vertex). Partitioned runs cache every tessellation
they build: a PS partition per file, a standard run one file per thread sub-box (one per partition under
`--sample-points`; its split follows the `--grid`). `python/tools/upgrade_tess_cache.py <folder>` converts files of
the older single-stream format.


## Standard DTFE and PS-DTFE compared

| | standard DTFE (`./DTFE`) | PS-DTFE (`./PS-DTFE`) |
|---|---|---|
| Density | DTFE density at each particle, interpolated linearly | sum over the streams of tetrahedron mass over Eulerian volume |
| Velocity | volume average of the single-valued interpolant | mean over the streams (by volume under `--ps-volume-weighted`); dispersion by mass |
| Velocity derivatives | ∇⟨v⟩ | ⟨∇v⟩ (dtfelib's `mean_field=True` gives ∇⟨v⟩) |
| Cell average | Monte Carlo or `--exact-average` (production) | sampled on nSub^3 (production nSub = 3) or `--ps-exact-deposit` |
| GPU | `--gpu`: the `_a` averages | `--ps-gpu`: every grid field, sampled and exact |
| Run-to-run identical | yes (exact GPU sums; CPU at a fixed thread count) | yes (exact sums, index-order merge) |
| `--max-concurrent` | threads per partition | partitions in flight |
| Padding | 15 spacings (drivers, dtfelib, launcher) | `--ps-lag-padding 8` |
| Points | the interpolant; `.pts_streams` = 0/1 coverage | every stream (`--per-stream`) |
| Streams, dispersion, caustics | none | yes |

In single-stream void interiors the standard density is 8-14% above PS-DTFE's: its tetrahedra bridge wall and void
particles.


## Running on TNG snapshots

The thesis workflow has three steps, run from the repository root and driven by `DTFE_DATA_ROOT` (where the
simulations live; default `/Volumes/Samsung T7/Illustris TNG`) and `DTFE_SIM` (which one). `scripts/config.sh` holds
the defaults and resolves either data layout: per family (`$DATA_ROOT/TNG100/TNG100-3-Dark/snapdir_099/`, the family
being the name up to its first `-`) or flat (`$DATA_ROOT/TNG100-3-Dark/snapdir_099/`); `dtfelib` and
`python/config.py` read the same variables. There is **no default simulation**: every run names it, with `-s` (the
drivers), `--sim` (the plot scripts), `SIMS` (`run_ps_pipeline.sh`, `SIMS=all` for every simulation on disk) or
`DTFE_SIM`; without one a script stops with "no simulation given". Each snapshot is one folder `snapdir_NNN` holding its merged, h-free
`combined_NNN.hdf5` and, after a run, `ps_output.*` (PS-DTFE) or `output.*` (standard DTFE) beside their `.runlog`.

1. **Download and merge** (a TNG API key in `~/.tng_api_key`, `TNG_API_KEY` or `-k`):
   `DTFE_SIM=TNG100-3-Dark scripts/download_snapshots.sh 50 99` fetches the raw chunks (`-c` the group catalogues,
   `-t` the merger trees, `-i` the initial conditions; no numbers: the 19-snapshot ladder) and merges them with
   `python/tools/merge_HDF5.py`; a combined file appears only when every chunk is there and the totals verify.
2. **Run an estimator**: `DTFE_SIM=TNG100-3-Dark scripts/run_ps_dtfe.sh 50 99` (phase-space) or
   `DTFE_SIM=TNG100-3-Dark scripts/run_dtfe.sh 50 99` (standard; with no numbers both run snapshot 99). Both default to the production
   settings, so `-g 512 -m` (and run_dtfe.sh's `-e`) are not needed; run_ps_dtfe.sh's `-e` selects the exact deposit.
   `DRY_RUN=1` prints each snapshot's command and runs nothing; `--help` lists every knob.
   `SIMS=all scripts/run_ps_pipeline.sh` runs every simulation on disk, adds a full-box image plane per snapshot, and resumes
   (a snapshot is skipped when its outputs are newer than its inputs and its run log shows the same settings).
3. **Analyse**: `DTFE_SIM=TNG100-3-Dark python3 python/analyze.py all` and the scripts in `python/plot/` (below).

**Run logs and safety.** A driver writes `<prefix>.runlog.partial` while it runs and renames it to `.runlog` on success
or `.runlog.failed` on failure, so `.runlog` always belongs to the grids on disk. After a successful rerun the files of
the earlier run that this one did not write move to `<prefix>.superseded/<date_time>/` (never deleted). A driver holds
`<prefix>.runlog.lock` while it runs a snapshot (a second driver is refused), stops with exit 2 when the data disk is
missing, and skips a snapshot whose folder is not writable or lacks room (`DTFE_SPACE_CHECK=0` skips the space check).

| knob (environment) | meaning | default |
|---|---|---|
| `DTFE_DATA_ROOT`, `DTFE_SIM` | the data root and the simulation | root: `/Volumes/Samsung T7/Illustris TNG`; simulation: none, required |
| `DTFE_FIGURES_ROOT` | where every figure goes (the plot scripts, `analyze.py`, the launcher) | `/Volumes/Samsung T7/Illustris TNG/figures` while that drive is mounted, else `python/figures` |
| `DTFE_SIGMA_MPC` | the void catalogue's smoothing in Mpc, the same in every box (for the void scripts and `analyze.py export`) | unset: 10 cells |
| `GRID_SIZE` (or `-g`) | cells per axis of the grids | 512 (`config.sh`; `DTFE_GRID_SIZE` too; the production grids of both estimators) |
| `PADDING` | the padding of each partition (and in standard DTFE of each thread's sub-box) in mean particle spacings; `DTFE_PADDING` too | standard DTFE 15 (`run_dtfe.sh`), PS-DTFE 50 (`config.sh`; it matters only for an unsplit tessellation or `PS_LAG_PADDING=0`) |
| `PS_METAL`/`DTFE_METAL` (`-m`) | the deposit / interpolation on the GPU (1) or the CPU (0) | `auto` (the GPU when the binary was built with one, else the CPU) |
| `PS_EXACT`/`DTFE_EXACT_AVERAGE` (`-e`) | the exact deposit / exact cell averages (`DTFE_EXACT_AVERAGE=0` is the Monte-Carlo cell mean) | 0, 1 |
| `AVG_SUBSAMPLES` (`-n`) | sub-samples per cell of the sampled deposit | 3 |
| `PARTITION`, `MAX_CONCURRENT` | the split of the box and how many parts run at once | auto-tuned |
| `SCRATCH_DIR` | the full grids as files outside RAM (large runs) | off |
| `FIELDS`, `OUTPUT_PREFIX`, `LAMBDA_TH` | the fields, the files' prefix, the web threshold | the production set (PS-DTFE nine fields; standard DTFE eight: the T-web and V-web included), `ps_output` (standard DTFE `output`; a `ps_*` prefix refused there), 0.3 |
| `PS_VERTEX_MASS`, `PS_VOLUME_WEIGHTED`, `PS_CAUSTICS` | PS-DTFE: vertex masses; volume-weighted velocity moments; the caustic classes | 1, 1, 1 |
| `SAMPLE_POINTS`, `PTS_VEL_GRAD` | a `--sample-points` file evaluated in the same run; with velocity gradients | off, 0 |
| `PTS_CONVENTIONS` | PS-DTFE planes: `grid` (the grids' conventions) or `legacy` (`--pts-conventions legacy`) | `grid` |
| `TESS_CACHE` | a `--tessellation-cache` folder | off |
| `DTFE_PRECISION` | `double` runs the double-precision pair | `single` |
| `ORDERED_MERGE` | 1 = merge the partitions in index order (bit-reproducible); 0 = as they finish (`--completion-order-merge`) | 1 |
| `FIXED_POINT_MERGE` | 1 = `--fixed-point-merge` (opt-in; not with `DTFE_PRECISION=double`) | 0 |
| `PS_RESUME`/`DTFE_RESUME` | 0 = start a killed partitioned run over instead of resuming it | 1 |
| `AUTO_TUNE_REPORT` | 1 = only the auto-tuner's report per snapshot; nothing computed or written | 0 |
| `DRY_RUN` | 1 = print the commands, run nothing | 0 |
| `DTFE_SPACE_CHECK` | 0 = skip the drivers' free-space check | 1 |

The rest (`THREADS`, `PS_LINEAR_DEPOSIT`, `PS_CAUSTIC_CUSPS`, `PS_PARALLEL_TRI`, `PS_LAG_PADDING`, `LAGRANGIAN_INPUT`,
and the pipeline's `SIMS`, `SNAPS`, `NU`, `AXIS`, `CENTER`, `PLANES`, `THICKNESS`, `SUPERSAMPLE`, `WINDOW`, `PS_GPU`,
`FORCE`) is listed in each script's `--help`.

### The thesis runs

`scripts/thesis_runs.sh STAGE ...` is the recipe of the grids on the data disk (`--help` gives each stage's settings,
`DRY_RUN=1` prints every command). Times on the 64 GB / 10-core M1 Max:

| stage | what | time, memory |
|---|---|---|
| `ps27` | PS-DTFE on the 27 snapshots (TNG100/300/50-3-Dark, z = 20 ... 0): 512^3, nine fields, GPU, a 52 GB budget, `SCRATCH_DIR` (default `~/DTFE-scratch`), the caustics and a 4096^2 image plane | 8.5 h; 34.5-44.9 GB resident |
| `dtfe27` | standard DTFE on the same 27: exact cell average, padding 15, TNG50 4x4x4 four at a time, TNG300 5x5x5 and TNG100 3x3x3 ten at a time | 7.6 h |
| `dtfe27planes` | standard-DTFE 4096^2 planes of the 27 under the prefix `output_plane` | ~9 h (an estimate), 15-25 GB |
| `caustics`, `caustics2` | the caustic classes of TNG100-3-Dark z = 0, and z = 1 and 2, under the prefix `ps_caustic` | caustics2: ~10 min, ~41 GB per snapshot |
| `voids` | every void figure of the three boxes (`SIMS=` limits them) | a few hours; the TNG300 fate census is the memory peak |
| `overnight` | `dtfe` (TNG100 z = 0, 1, 5), `caustics2`, `refresh` (figure repairs), `prune` (the cache) | about 1-1.5 h |

`all` runs ps27 dtfe27 dtfe27planes caustics caustics2 voids. Before each snapshot ps27, dtfe27, dtfe27planes and dtfe
move its old files to `<data root>/_backup/<prefix>_<date>/...` (back after a failed run); a snapshot whose run log
already shows the stage's settings is skipped (`FORCE=1` reruns it). Run one stage at a time with nothing else heavy
beside it: the recipe holds `SCRATCH_DIR/thesis_runs.lock`, and the heavy stages wait up to `HEAVY_WAIT_SEC` (30 min)
for another DTFE run to end.
The memory budget is a cap: the auto-tuner also caps it by the memory free at the start, so the split depends on what
else is open (each run log records it).

### Analysis

`python3 python/analyze.py [check|compute|plot|all|export|voids|cache]` (default `all`) is the standard-DTFE thesis
set: `compute` caches the smoothed density, Hessian and void catalogue, `plot` draws from the cache, `export` writes
the void catalogues as CSV + HDF5, `voids` runs the void scripts, `cache` lists the cache namespaces (`--prune`
lists what is unreachable, `--prune --yes` deletes it; `--move DIR [--yes]` moves the cache, e.g. out of iCloud
Drive, and records the folder in `~/.config/dtfe/cache_dir`; `DTFE_CACHE_DIR` overrides it). The thesis `Figures/` tree
takes the core figures only from `config.THESIS_SIMULATION` (TNG50-3-Dark) and standard DTFE; `--no-mirror` (or
`DTFE_NO_MIRROR`) copies nothing there.

The plot scripts share `--data-root`, `--sim`, `--snap`, `--method auto|ps|dtfe`, `--prefix` (another grid set),
`--raw` (cell-centre grids) and `--smooth` (cells). Field maps: plot_PS_DTFE.py, plot_DTFE.py, plot_cosmic_web.py,
plot_web_streams.py, plot_velDiv_den.py, plot_PDF_CDF.py, plot_shear_triaxial.py, plot_eigenvalues.py,
plot_pk_compare.py; point-evaluated planes: plot_pointeval.py, plot_pointeval_panels.py; caustics:
plot_caustic_skeleton.py (needs `.causticClass`; `--prefix ps_caustic` for the separate caustic runs); haloes:
plot_halo_tracking.py; voids: below. A non-default option adds a tag to the output names, so the default figures are
never overwritten.

**Voids.** The catalogue's voids are the minima of the density contrast smoothed with a Gaussian of
`SMOOTHING_SIGMA_CELLS` = 10 cells (`python/config.py`), a different physical scale in every box (1.0, 2.2 and 5.9 Mpc in
TNG50-3, TNG100-3 and TNG300-3): compare boxes at one scale with `--smooth-mpc 2.1625` (`config.CROSSBOX_SMOOTHING_MPC`)
or `DTFE_SIGMA_MPC=2.1625`, whose products get their own names (`analyze.py compute|plot|all` refuses to run under it).
Each void has two sizes: R_eff, the ellipsoid fit's (abc)^(1/3), which sets the resolution cut (largest semi-axis
within 10 smoothing lengths and half the box) and the shapes; and R_v, where the spherically averaged smoothed density
first rises back to the mean (about 0.57 R_eff, searched out to min(24 sigma, L/2 - sigma)). `catalog.R_V_DEFINITIONS`
also gives `delta08` (enclosed contrast -0.8) and `ridge` (the compensation ridge): `catalog.r_v_mpc(cat, 'delta08')`.
The scripts use R_v and the *distinct* voids by default (a void centred inside a deeper one is left out;
`--keep-overlaps` keeps them, plot_void_profiles.py `--radius r_eff` stacks in R_eff). `DTFE_VOID_GATE=none` drops the
ellipsoid cut from the sample (outputs tagged `gatenone`, no thesis mirror). The radius settings are part of a
catalogue's identity: a cache made with others is re-measured on first load (about 15 s per 512^3 snapshot).
Caveat: the -0.8 radius of a smoothed-field minimum follows the smoothing length, so a size function from it is not a
test of the void abundance theory; `--size-rv delta08raw` measures it on the raw density instead.

| script | what | main options |
|---|---|---|
| plot_void_population.py | abundance, sizes, shapes; the size function against Sheth & van de Weygaert (Jennings et al. 2013; `dtfelib/theory.py`) | `--size-rv`, `--size-smooth` (default 5 cells), `--size-fixed-r` (with `DTFE_VOID_GATE=none`) |
| plot_void_profiles.py | stacked density and velocity profiles in r/R_v, Hamaus et al. (2014) fits | `--edges`, `--radius` |
| plot_void_profile_evolution.py | the stacked profiles at every snapshot, v_r against linear theory | `--rv-definition`, `--replot` |
| plot_void_environment.py | void-in-cloud against void-in-void; the default criterion is net infall in the shell at 2-3 R_v | `--criterion velocity` / `density_smoothed` / `density_raw` |
| plot_void_fate.py | survive / merge / split / vanish census: the particles within R_v (at most 15 smoothing lengths, `lagrangian.FATE_MAX_SIGMA`) found again in the next snapshot; a void survives when half of them end in one descendant (`lagrangian.SURVIVE_FRAC`, `SPLIT_FRAC`) | `--exclusive` (each particle to one void), `--ref-snap` (default 33) |
| plot_void_tracking.py | each z = 0 void back in time | `--link particles`, `--exclusive`, `--n-voids` |
| plot_void_crossbox.py | the boxes side by side at one smoothing in Mpc | `--smooth-mpc` |
| plot_void_convergence.py | grid, particle-mass and estimator convergence of the void statistics | `--factors`, `--match-sigma` |
| plot_caustic_skeleton.py | wall metrics of the voids against random spheres | `--voids`, `--null-draws`, `--all-caustic` |

The fate census reads every combined snapshot once (TNG100-3-Dark: 16 min, 12.5 GB); its
links are cached, so a rerun takes seconds (`DTFE_FATE_WORKERS`, default up to 4; `DTFE_FATE_CHUNK`, default 8e6
particles per read). `python/tools/void_outlines.py` writes a snapshot's void table as CSV.


## Graphical launcher

`scripts/gui.sh` opens a PySide6 window for the things normally typed in a terminal (`--setup` reopens the first-run
dialog, `--smoke` builds the window, prints what its jobs would use and quits). `scripts/install.sh` sets it up in its
own environment (`~/.venvs/dtfe-gui`, about 1 GB, seeing the main Python's numpy and h5py; `DTFE_GUI_VENV` /
`DTFE_GUI_PYTHON` override) and makes it launchable as an app (macOS: **DTFE Launcher** in Applications, which needs
Full Disk Access once for iCloud Drive and external disks; Linux: a menu entry). By hand:
```bash
python3 -m venv ~/.venvs/dtfe-gui --system-site-packages
~/.venvs/dtfe-gui/bin/pip install PySide6
```

The first start walks through the data folder, building the programs, the TNG API key and a **demo** (a synthetic
crossed-wave simulation run in seconds and opened in Explore). The tabs:
- **Data**: what is on disk per snapshot; downloads (snapshots, group catalogues, merger trees, initial conditions) and
  merges them, with the size against the free disk.
- **Grids**: one run of either estimator (`run_ps_dtfe.sh` / `run_dtfe.sh`): deposit, grid, fields, phase-space
  options, a hi-res slice, memory and merge settings, precision, presets.
- **Custom snapshot**: either binary on any Gadget (HDF5 or binary) snapshot or text file, with its units, box,
  periodicity and initial positions stated; 2D files run on the 2D binaries.
- **Pipeline**: `run_ps_pipeline.sh` for the chosen simulations, with the plane's geometry and the render options.
- **Plots**: the scripts of `python/plot` and `analyze.py`, each with its own options.
- **Explore**: maps of any run's grids (a slice read from disk), field, component, axis, scale and smoothing; a query
  server on the snapshot behind the map (a click lists every stream there; dragging a rectangle zooms by point
  evaluation); an **exact** zoom (a real run on the region: `--regionMpc` for standard DTFE, `--ps-window` for
  PS-DTFE); void outlines from the analysis catalogue; *Compare with*, the figure grid across outputs and redshifts,
  and *Save image* with publication presets. PS velocity derivatives are shown as the mean field ∇⟨v⟩ by default
  (*PS derivatives* switches to the stored ⟨∇v⟩).
- Beside the log: **Queue** (jobs run one after another, kept across sessions), **Figures**, **Runs** (every run log on
  disk: settings, time, peak memory; outputs made by a build before a fix that changed them, legacy planes and
  resumable runs are marked; load the settings back, open in Explore, move to the Trash, build the void catalogue) and
  **Running now** (every DTFE run on the machine, from its progress file, with a Stop button).

Every tab shows the exact commands its choices stand for, flags combinations that would be refused or would not fit
(**Check memory** asks the auto-tuner per snapshot), and can export them as a bash script or a SLURM job. A fresh job
ticks the GPU only when the binary was built with one.

The code lives in `python/gui`: `app.py` (the window) with the mixins `app_tabs.py`, `app_specs.py`, `app_run.py`,
`app_menus.py`, `app_check.py` and `app_widgets.py`; `explore.py` with `explore_canvas.py`, `explore_grid.py`,
`explore_save.py`, `explore_server.py`, `explore_voids.py` and `explore_zoom.py`; `runspec.py` (the job specifications
and their commands, no Qt) with `runspec_pipeline.py`, `runspec_plot.py`, `runspec_data.py` and `runspec_custom.py`;
and `grids.py`, `presets.py`, `results.py`, `runs_view.py`, `setup_wizard.py`, `void_cache.py`, `figsave.py`,
`make_icon.py` (the app icon).
Import `app`, `explore` or `runspec` before any of their mixin modules.


## Contributors
* **Marius Cautun (Kapteyn Astronomical Institute, Durham University, Leiden University)** - *code and documentation.*
* **Rien van de Weygaert (Kapteyn Astronomical Institute)** - *various discussions about the method and implementation.*
* **Luuk Westerhoek (Kapteyn Astronomical Institute, University of Groningen)** - *phase-space DTFE, the exact / point evaluation methods, automatic memory management, the GPU backends (Metal, and its CUDA/HIP port) and the analysis pipeline (bachelor research project "Fate of Cosmic Voids").*


## License

This project is licensed under GNU GENERAL PUBLIC LICENSE Version 3 - see the [LICENSE.md](LICENSE.md) file for details.
