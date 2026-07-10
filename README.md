# The DTFE public software

The DTFE public code is a C++ implementation of the **Delaunay Tessellation Field Interpolation (DTFE)** method. Its purpose is to interpolate quantities stored at the location of an unstructured set of points to a regular grid using the maximum of information contained in the input points set. In particular, the code can calculate the following cosmological quantities:
* the density field - this is calculated directly from the point distribution,
* the velocity field and derivatives (e.g. gradient, divergence, vorticity) - uses the velocity at each particle position, and
* general vector quantities and their derivatives - these quantities must be given as input for each point in the set.

The code was written with the purpose of analysing cosmological simulations and galaxy redshift survey. Even though the code was designed with astrophysics in mind, it can be used for problems in a wide range of fields where one needs to interpolate from a discrete set of points to a grid.

The code was designed using a modular philosophy and with a wide set of features that can easily be selected using the different program options. The DTFE code is also written using OpenMP directives which allow it to run in parallel on shared-memory architectures.

The code comes with a complete [documentation](documentation/DTFE_user_guide.pdf) and with a multitude of examples that detail the program features. A test dataset and analysis of the code output is given in the [demo directory](demo).

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
* Uses the point distribution to compute the density and interpolates the result to grid.
* Each sample point has a weight associated to it to represent multiple resolution N-body simulations and observational biases for galaxy redshift surveys.
* Interpolates the velocity, velocity gradient, velocity divergence, velocity shear and velocity vorticity.
* Interpolates any additional number of fields and their gradients to grid.
* Periodic boundary conditions.
* Zoom in option for regions of interest.
* Splitting the full data in smaller computational chunks when dealing with limited CPU resources.
* The computation can be distributed in parallel on shared-memory architectures.
* For comparison purposes, the software comes also with three other simpler interpolation techniques: nearest grid point (NGP), triangular shape cloud (TSC) and smoothed particle hydrodynamics (SPH).
* Returns the Delaunay tessellation of the given point set.
* Easy change of input/output data format.
* Easy to use as an external library.
* Extensive documentation of each feature.


## Installation and Building

### Quick install

**One command, any platform** — installs missing dependencies, wipes the build, and recompiles both binaries with the best backend for the machine (macOS/Apple Silicon → Metal GPU; Linux with `nvcc`/`hipcc` → CUDA/HIP; otherwise CPU):

```bash
./install.sh          # add --cpu to force CPU-only, --no-deps to skip package installation
```

Per-platform equivalents (details in the sections below):

| Platform | Commands |
|----------|----------|
| **macOS** (Homebrew) | `brew install gsl boost cgal mpfr gmp hdf5 fftw llvm libomp && make DTFE METAL=1 && make PS-DTFE METAL=1` |
| **Ubuntu/Debian** | `sudo apt-get install build-essential libgsl-dev libboost-all-dev libcgal-dev libmpfr-dev libhdf5-dev libgmp-dev libfftw3-dev && make DTFE && make PS-DTFE` |
| **Docker** (Linux image) | `docker build -t dtfe .` — builds both binaries CPU-only and runs the fast test battery as a build sanity stage; `--build-arg GPU=cuda` additionally compile-checks the CUDA backend on an `nvidia/cuda` base image (running it still needs an NVIDIA GPU + `docker run --gpus all`) |

**Docker is not the macOS path**: a container is a Linux VM even on a Mac, so it can never build or run the Metal backend — use `./install.sh` (or the brew line) for native macOS binaries, and Docker only to produce Linux/cluster images.

Run `make deps-check` at any time to verify every required header is present — it prints the exact `brew install` / `apt-get install` command for anything missing, instead of a long compile dying on a cryptic missing-header error.

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
(The Makefile automatically picks up Debian/Ubuntu's HDF5 serial sub-directory, `/usr/include/hdf5/serial`.)

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

#### Docker
The repository root ships a [Dockerfile](Dockerfile) (Ubuntu 24.04) that installs the packages above, builds **both** binaries CPU-only, and runs `tests/run_tests.py` + `tests/ps_smoke_test.sh` as a build sanity stage:
```bash
docker build -t dtfe .                            # CPU-only
docker build -t dtfe-cuda --build-arg GPU=cuda .  # + CUDA backend (compile-check; nvidia/cuda devel base)
docker run --rm -v $PWD/data:/data dtfe /opt/dtfe/PS-DTFE /data/snap.hdf5 /data/out \
    --grid 128 --periodic --field density --MpcUnit 1
```

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd DTFE
   ```

2. **Check the toolchain and dependencies:**
   ```bash
   make deps-check       # verifies every required header; prints install commands for anything missing
   make test-platform    # shows the detected platform/compiler/paths
   ```

3. **Build the main executable:**
   ```bash
   make DTFE               # CPU interpolation
   make DTFE METAL=1       # + GPU '_a' interpolation, Apple Silicon (macOS; Metal)
   make DTFE CUDA=1        # + GPU '_a' interpolation, NVIDIA (Linux; needs nvcc, CUDA_PATH=/usr/local/cuda)
   make DTFE HIP=1         # + GPU '_a' interpolation, AMD (Linux; needs ROCm hipcc, ROCM_PATH=/opt/rocm)
   ```
   **HIP note:** when building on a machine that cannot see the target GPU (login nodes,
   containers), pass the target architecture explicitly — HIP binaries are arch-specific:
   `make DTFE HIP=1 GPU_ARCH=gfx90a` (find it with `rocminfo | grep gfx`; several targets can
   be given as `GPU_ARCH="gfx90a gfx942"`). For CUDA, `GPU_ARCH=sm_86` is optional (nvcc's
   default PTX is forward-portable; setting it just skips the first-launch JIT).
   Any GPU build enables the `--gpu` flag (`--metal` is a legacy synonym), which moves the
   method-1 volume-averaged (`_a`) grid interpolation to the GPU with automatic CPU
   fallback. All backends implement the same kernel (Metal: `metal/dtfe_deposit.metal`,
   runtime-compiled from headers vendored in `third_party/metal-cpp/`; CUDA/HIP: the
   single-source `src/CGAL_triangulation/dtfe_gpu_cuda.cu`, compiled ahead of time), and
   results match the CPU interpolation to float rounding (validated by
   `tests/dtfe_metal_check.sh` — on Linux: `GPU_BUILD=CUDA=1 tests/dtfe_metal_check.sh`).
   Unaveraged fields and methods 2/3 always run on the CPU. Switching build modes wipes
   `o/` automatically so objects from different modes are never mixed; the current mode is
   recorded in `o/.build_mode`.

4. **Build the shared library:**
   ```bash
   make library
   ```

5. **Clean build files:**
   ```bash
   make clean
   ```

### Configuration Options

The software behavior can be configured by editing the `OPTIONS` section in the Makefile:

#### Spatial Dimensions
```makefile
OPTIONS += -DNO_DIM=3    # 3D (default)
# OPTIONS += -DNO_DIM=2  # 2D
```

#### Variable Precision
```makefile
# OPTIONS += -DDOUBLE    # Use double precision (uncomment for double)
```

#### Computed Quantities
```makefile
OPTIONS += -DVELOCITY          # Enable velocity computations
OPTIONS += -DSCALAR            # Enable scalar field interpolation
OPTIONS += -DNO_SCALARS=1      # Number of scalar components
```

#### Input/Output Defaults
```makefile
OPTIONS += -DINPUT_FILE_DEFAULT=105   # HDF5 gadget format
OPTIONS += -DMPC_UNIT=1000            # Data units (kpc in this example)
OPTIONS += -DOUTPUT_FILE_DEFAULT=101  # Binary output
```

#### Additional Features
```makefile
OPTIONS += -DOPEN_MP              # Enable OpenMP parallelization
OPTIONS += -DTEST_PADDING         # Validate Delaunay tesselation padding
OPTIONS += -DREDSHIFT_SPACE       # Enable redshift space computations
OPTIONS += -DTRIANGULATION        # Enable triangulation access
```

See the Makefile for the complete list of available options.


## Phase-Space DTFE (PS-DTFE)

Standard DTFE builds the Delaunay tessellation in Eulerian (present-day) space and returns a single-valued density at each point. **Phase-Space DTFE** instead builds the tessellation in **Lagrangian** (initial-condition) space and follows it as it is deformed into Eulerian space. Because the deformed tessellation can fold over onto itself, PS-DTFE recovers the **multi-stream** structure of the cosmic web: the density at a grid point is the sum of the contributions of every stream (Eulerian simplex) that overlaps it, and the number of overlapping streams is returned as a separate field. This makes the method well suited to caustics and to the multi-stream interior of voids, where the standard single-stream estimate breaks down. The approach follows the phase-space tessellation method of Abel, Hahn & Kaehler (2012) and Shandarin, Habib & Heitmann (2012).

PS-DTFE is selected by the `-DPHASE_SPACE` compile flag, which switches the triangulation to Lagrangian coordinates and the density estimate to the ratio of Lagrangian to Eulerian simplex volume. It is built as a **separate executable** so that it can coexist with the standard `DTFE` build.

### Building
```bash
make PS-DTFE            # CPU deposit
make PS-DTFE METAL=1    # + GPU deposit, Apple Silicon (macOS; Metal)
make PS-DTFE CUDA=1     # + GPU deposit, NVIDIA (Linux; needs nvcc)
make PS-DTFE HIP=1      # + GPU deposit, AMD (Linux; needs ROCm hipcc)
```
This produces a `PS-DTFE` executable (object files are placed in `o_ps/` to avoid clashing with the standard `o/` build). It needs the same dependencies as `make DTFE`, plus **HDF5** — PS-DTFE input is HDF5-only (see below). The `-DPHASE_SPACE` flag is set automatically by this target; do not add it to the standard `DTFE` build. Any GPU build enables `--ps-gpu` (`--ps-metal` is a legacy synonym), which moves the grid deposit to the GPU with automatic CPU fallback. The Metal backend embeds and runtime-compiles `metal/ps_deposit.metal` (headers vendored in `third_party/metal-cpp/`); CUDA/HIP compile the single-source `src/CGAL_triangulation/ps_gpu_cuda.cu` ahead of time — same kernel algorithm, same CPU-parity contract on every backend.

### Input data
PS-DTFE needs **two positions per particle**: the Eulerian (present-day) position and the Lagrangian (initial-condition) position. Both are read from Gadget-HDF5 files (input type `105`):
* Eulerian positions from the usual `Coordinates` dataset.
* Lagrangian positions from an `InitialCoordinates` dataset in the *same* file, **or** from a separate initial-conditions snapshot passed with `--lagrangianInput <ics.hdf5>` and matched to the main file by `ParticleIDs`.

The text and raw-binary readers do not carry Lagrangian positions, so PS-DTFE currently requires HDF5 input.

### Running
With the Lagrangian positions stored in the snapshot itself:
```bash
./PS-DTFE snapshot.hdf5 output_root --grid 256 --periodic --field density --MpcUnit 1
```
With the Lagrangian positions in a separate initial-conditions file:
```bash
./PS-DTFE snapshot.hdf5 output_root --grid 256 --periodic --field density --lagrangianInput ics.hdf5
```
The outputs are raw binary, one value per grid cell (row-major, single precision unless built with `-DDOUBLE`). The `--field` choices and the files they write:

| `--field` | output file(s) | comps (3D) | meaning |
|-----------|----------------|:----------:|---------|
| `density` | `.den` | 1 | multi-stream density Σ_s ρ_s (sum over streams) |
| `velocity` | `.vel` | 3 | mass-weighted multi-stream velocity ⟨v⟩ = Σ ρ_s v_s / Σ ρ_s |
| `gradient` | `.velGrad` | 9 | mass-weighted velocity gradient ∂v_i/∂x_j |
| `dispersion` | `.velDisp`, `.velDispTensor` | 1, 6 | velocity dispersion: trace σ² ("temperature") and full symmetric tensor σ_ij |
| `scalar` | `.scalar` | `NO_SCALARS` | mass-weighted scalar field(s) |
| `tweb` | `.velTweb`, `.velTwebEig` | 1, 3 | T-web cosmic-web label (0=void, 1=wall, 2=filament, 3=node) from the tidal tensor (FFT Poisson solve of the density grid) and its eigenvalues |
| `vweb` | `.velVweb`, `.velVwebEig` | 1, 3 | V-web label from the velocity shear tensor and its eigenvalues (threshold `--lambda_th`, default 0.3) |
| _(always)_ | `.streams` | 1 | number of streams per cell (1 single-stream, ≥3 in a caustic) |

Every field also has a volume-averaged _a form (density_a, velocity_a, dispersion_a, …) that sub-samples an nSub³ grid (nSub=3, `--avg-subsamples`) inside each cell and writes the matching .a_* file (.a_den, .a_velDisp, …); .a_streams is the per-cell average stream count. The dispersion is the density-weighted covariance of the per-stream velocities: for each pair of spatial directions you weight every stream's velocity deviation from the local mean flow by that stream's density, sum over all streams, and normalise by the total stream density — equivalently, the density-weighted mean of the velocity-component products minus the product of the mean velocity components. It is approximately zero in cold, single-stream void interiors and large in multi-stream walls and filaments. In single-stream regions the velocity, gradient and dispersion reproduce the standard DTFE result, with the dispersion going to zero.

### Point evaluation (`--sample-points`)

Besides the regular grid, PS-DTFE can evaluate the multi-stream fields at **arbitrary Eulerian points** (halo centres, void centres, random sight-lines, …):

```bash
./PS-DTFE snapshot.hdf5 output_root --grid 32 --periodic --field density \
    --sample-points points.txt [--per-stream] [--ps-stream-density dtfe|geometric]
```

* **Input**: a text file with one `x y z` per line, or a raw binary file of float64 triplets (N×3, row-major, native/little-endian, no header) — auto-detected. Coordinates are in the box coordinate system (same units as `--box`, i.e. Mpc after `--MpcUnit`); under `--periodic` they are wrapped into the box. Point evaluation runs **alongside** the grid interpolation, so pass a small `--grid` (e.g. `-g 32`) when only point values are wanted.
* **Outputs** (raw binary, native little-endian, in the input point order):

  | file | type × comps | meaning |
  |------|--------------|---------|
  | `.pts_den` | float64 × 1 | total density Σ_s ρ_s over the streams at the point, in ρ/ρ̄ units (same normalization as `.den`) |
  | `.pts_vel` | float64 × 3 | density-weighted mean velocity ⟨v⟩ = Σ ρ_s v_s / Σ ρ_s |
  | `.pts_velDisp` | float64 × 6 | velocity dispersion tensor σ_ij (upper triangle row-major: xx xy xz yy yz zz); exactly 0 for single-stream points |
  | `.pts_streams` | int32 × 1 | stream count (number of Eulerian tetrahedra containing the point) |

* **`--per-stream`** additionally writes every stream's own density and velocity in a ragged layout: `.pts_stream_offsets` (uint64, N+1 entries; point *i* owns records `[off[i], off[i+1])`) and `.pts_stream_records` (float64 × 4 per record: density, vx, vy, vz), with each point's streams sorted by density, descending.
* **`--ps-stream-density`** selects the per-stream density estimator (it also weights the mean velocity and dispersion):
  * `dtfe` (default) — linear interpolation of the Lagrangian-vertex DTFE densities inside each stream's tetrahedron. This matches the `PhaseSpaceDTFE` class of the method authors' reference implementation ([github.com/jfeldbrugge/PS-DTFE](https://github.com/jfeldbrugge/PS-DTFE), `Python/density.py`); the field is continuous within each stream but not exactly mass-conserving.
  * `geometric` — the constant per-tetrahedron density m_tet/V_eul = ρ̄·V_lag/V_eul (the reference code's `PhaseSpace` class). This is the density whose (cell-averaged, mass-conserving) rendering the PS-DTFE grid deposit produces; it is discontinuous across tetrahedra but integrates to the exact mass.

  In smooth single-stream regions the two agree to a few percent; they differ most near caustics.
* The evaluation is CPU-only and double precision, uses the same cell filters and ±1e-6 barycentric tolerance as the grid deposit, and works under any `--partition` split (auto-tuned or explicit): stream contributions accumulate across the Lagrangian partitions and are reduced once, in deterministic (sorted) order, so the outputs are independent of the partition/thread schedule. Validated by `tests/ps_point_eval_check.sh`.

### Linear-within-stream deposit (`--ps-linear-deposit`)

By default the mass-conserving grid deposit spreads each tetrahedron's mass **uniformly** over its interior sub-samples. With `--ps-linear-deposit` the samples are instead weighted by the DTFE-interpolated **linear density profile** inside the tetrahedron, and the weights are **renormalized per tetrahedron** so the deposited total still equals the tetrahedron's mass exactly — mass conservation (the fix for the historical caustic "square" artefacts) is untouched, only the sub-tetrahedron mass distribution changes. The two variants produce identical `.streams`, identical grid means, and differ smoothly (percent-level in single-stream regions, most visibly where tetrahedra span several cells). Works with the CPU deposit and all GPU backends (`--ps-gpu`); validated by `tests/ps_linear_deposit_check.sh`.

### Memory: `--partition` and `--max-concurrent` (auto-tuned by default)
When neither flag is given, **the binary picks both automatically** once the input is read, from the particle count, grid size, requested fields and the machine's physical RAM and cores (it prints an `AUTO-TUNE:` line with the chosen values and the predicted peak RSS; `DTFE_RAM_GB` overrides the detected RAM). Explicit flags always win, and small inputs (default: below 2×10⁶ particles, `DTFE_AUTO_MINN`) keep the historical single-domain behavior. Both run scripts defer to auto mode unless `PARTITION` / `MAX_CONCURRENT` env vars are set.

The model behind the tuner: large runs are bounded by the per-partition CGAL triangulation (~0.65 KB per padded vertex; the Lagrangian padding multiplies particle counts by ~4x) plus the global field grids. PS-DTFE partitions **Lagrangian** space with `--partition X Y Z` and caps concurrent partition pipelines with `--max-concurrent N`; peak RSS ≈ globals + N × (per-partition triangulation + sub-grids). Manual reference points (what auto should land near): a 512³ grid with the default field set fits TNG50-4 (2e7 particles) at `--partition 2 2 2 --max-concurrent 2` in ≲30 GB, and TNG300-3 (2.4e8 particles) at `--partition 5 5 5 --max-concurrent 2` in ~40 GB. The `run_ps_dtfe.sh` runlog reports the measured peak RSS after every run, closing the loop on the prediction.

## Batch scripts and configuration

Shared defaults (data root, simulation, snapshot list, grid size, padding) live in [config.sh](config.sh), sourced by all three workflow scripts and overridable per run via flags or environment variables (`DTFE_DATA_ROOT`, `DTFE_SIM` — also honoured by the Python side):

```bash
./download_snapshots.sh -s TNG50-4-Dark 99      # fetch TNG snapshot chunks
./download_snapshots.sh -c -s TNG50-4-Dark 99   # FoF/Subfind group catalogs
./download_snapshots.sh -t -s TNG50-4-Dark      # SubLink merger trees (whole simulation)
python3 python/tools/merge_HDF5.py ...                # merge chunks into combined_NNN.hdf5
./run_dtfe.sh [-d DATA_DIR] [-g GRID] [-m] [snap ...]    # standard DTFE batch (-m = GPU '_a' interpolation)
./run_ps_dtfe.sh [-d DATA_DIR] [-g GRID] [-n NSUB] [-m] [snap ...]   # PS-DTFE batch (-m = GPU deposit)
```

The TNG API key is read from `-k`, the `TNG_API_KEY` env var, or `~/.tng_api_key` (recommended:
`echo "<key>" > ~/.tng_api_key && chmod 600 ~/.tng_api_key` — outside the repo, never committed).

`run_ps_dtfe.sh` logs each run to `<snapdir>/ps_output.runlog` (ANSI-stripped) and prints wall time and peak memory. Terminal colours survive the log pipe via `CLICOLOR_FORCE` (respecting `NO_COLOR`).


## Python analysis pipeline

`python/` contains the thesis analysis stack, built around the `dtfelib` package:
* `dtfelib.io.FieldSet` — uniform loader for DTFE and PS-DTFE outputs (units, naming, averaged/raw).
* `dtfelib.cli` — shared CLI (`--sim`, `--snap`, `--method`, `--smooth`, …) used by every script in `python/plot/`.
* `python/plot/*.py` — one figure family per script (density/web maps, eigenvalues, correlations, void shapes, …).
* `dtfelib.trees` / `dtfelib.groupcat` / `dtfelib.environment` — SubLink merger trees (main branches, descendants, max-past-mass merger ratios), FoF/Subfind particle membership + shape tensors, and field sampling at halo positions. `python/plot/plot_halo_tracking.py` combines them: size/shape/orientation/mergers plus the DTFE **or** PS-DTFE density and web-class environment along each halo's history.
* `dtfelib.voids` + `python/plot/plot_void_tracking.py` — VOID evolution: voids found as minima of the smoothed density contrast at the reference snapshot are given an identity across time by their merger-tree tracer subhalos (median periodic displacement moves the centre), and their size (semi-axes, R_eff), BBKS shape (e, p) and major-axis orientation are re-measured from the DTFE or PS-DTFE field at every snapshot with grids on disk.
* `python/plot/plot_void_population.py` — the population view: counts, R_eff / e / p / central-δ medians, bands and distributions for ALL voids per snapshot (same catalogs and sample cuts as the correlation analyses), for either estimator.
* `dtfelib.pipeline` — the derived-product engine (smoothed delta, Hessian, void catalogs, critical points, per-parameter caches); `dtfelib.fields` / `dtfelib.figures` — grid-field utilities and figure styling/output.
* `python/analyze.py` — the batch driver: `analyze.py check | compute | plot | all`; `python/config.py` holds simulation constants and adapts to the active simulation (`DTFE_SIM` / `--sim`; box sizes pinned for TNG50-3/50-4/300-3, read from headers for new simulations); `python/selftest.py` smoke-tests the whole stack on synthetic data in a sandbox.


## Tests

```bash
tests/ps_smoke_test.sh          # build + Zel'dovich pancake sanity check (~1 min)
python3 tests/ps_regression_test.py   # physics metrics vs tests/reference baseline
tests/ps_parallel_check.sh      # serial vs partitioned/parallel agreement
tests/ps_point_eval_check.sh    # --sample-points point evaluation vs deposit/analytics
tests/ps_linear_deposit_check.sh # --ps-linear-deposit mass conservation + A/B smoothness
python3 tests/run_tests.py      # standard-DTFE regression
tests/dtfe_metal_check.sh       # standard-DTFE CPU vs --gpu parity (builds the current GPU mode; GPU_BUILD=CUDA=1 on Linux/NVIDIA)
python3 tests/py_dtfelib_test.py  # python stack: SubLink invariants, sampling, void tracking
python3 tests/ps_3d_test.py     # 3D crossed Zel'dovich waves: stream counts in {1,3,9,27}
python3 tests/ps_convergence_test.py  # single-stream density -> analytic profile as N grows
python3 tests/ps_nonperiodic_test.py  # finite cloud in vacuum: mass conservation without --periodic
python3 tests/ps_standard_cross_check.py  # PS-DTFE vs standard DTFE where streams == 1
tests/ps_scaling_benchmark.sh   # strong-scaling guard (fails if the partition loop serializes)
```

A minimal GitHub Actions workflow (Linux CPU build + `run_tests.py` + `ps_smoke_test.sh`) ships **disabled** as [.github/workflows/ci.yml.disabled](.github/workflows/ci.yml.disabled) — rename it to `ci.yml` to enable. It is deliberately a single fast job (an earlier, broader cross-platform workflow was removed on purpose).


## Contributors
* **Marius Cautun (Kapteyn Astronomical Institute, Durham University, Leiden University)** - *code and documentation.*
* **Rien van de Weygaert (Kapteyn Astronomical Institute)** - *various discussions about the method and implementation.*


## License

This project is licensed under GNU GENERAL PUBLIC LICENSE Version 3 - see the [LICENSE.md](LICENSE.md) file for details.