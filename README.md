# The DTFE public software

The DTFE public code is a C++ implementation of the **Delaunay Tessellation Field Interpolation (DTFE)** method. Its purpose is to interpolate quantities stored at the location of an unstructured set of points to a regular grid using the maximum of information contained in the input points set. In particular, the code can calculate the following cosmological quantities:
* the density field - this is calculated directly from the point distribution,
* the velocity field and derivatives (e.g. gradient, divergence, vorticity) - uses the velocity at each particle position, and
* general vector quantities and their derivatives - these quantities must be given as input for each point in the set.

The code was written with the purpose of analysing cosmological simulations and galaxy redshift survey. Even though the code was designed with astrophysics in mind, it can be used for problems in a wide range of fields where one needs to interpolate from a discrete set of points to a grid.

The code was designed using a modular philosophy and with a wide set of features that can easily be selected using the different program options. The DTFE code is also written using OpenMP directives which allow it to run in parallel on shared-memory architectures.

The code comes with a complete [documentation](docs/DTFE_user_guide.pdf) and with a multitude of examples that detail the program features. A test dataset and analysis of the code output is given in the [demo directory](demo): `./DTFE demo/z0_64.gadget out --grid 64 --field density --periodic --padding 12` works out of the box (the box is periodic, and its voids are wider than the default padding of 5 particle spacings: with it about 1% of the cells would lie outside the tessellation, which the run reports).

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
* A phase-space (Lagrangian) variant, `PS-DTFE`, that recovers the multi-stream structure of the cosmic web - the multi-stream density, the velocity dispersion, and the caustic surfaces.
* Exact, sampling-free field estimation as an alternative to Monte-Carlo cell averaging, and evaluation of the fields at arbitrary user-supplied points, with exact point location (a point on a shared face or vertex is counted once per stream).
* An interactive mode: the tessellation is built once and then queried from Python at any points, as often as needed (`dtfelib.Estimator`), on a snapshot file or on particles held in numpy arrays; boxes too large for one tessellation are served as a composite of partitions kept on disk.
* For every stream at a point, the Lagrangian position its matter started from, and any per-particle quantity interpolated along it.
* Automatic choice of the domain decomposition and memory footprint from the data and the machine, with an optional out-of-core mode for grids larger than the available memory.
* The tessellation can be cached on disk and reused by later runs, which skips its construction entirely when the same data is analysed again.
* Optional GPU acceleration of the field deposit on Apple Silicon (Metal) and, as a port of the same kernels, on Linux with an NVIDIA (CUDA) or AMD (HIP) GPU, covering every grid field of both estimators (density, velocity, dispersion, gradients, a scalar field and the exact deposits); and an optional double-precision build.
* For comparison purposes, the software comes also with three other simpler interpolation techniques: nearest grid point (NGP), triangular shape cloud (TSC) and smoothed particle hydrodynamics (SPH).
* Returns the Delaunay tessellation of the given point set.
* Easy change of input/output data format.
* Easy to use as an external library.
* Extensive documentation of each feature.


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

### Quick Start

The one-command way, after cloning: `./scripts/install.sh` installs any missing dependencies, builds
both binaries with the best backend for the machine (Metal on Apple Silicon, CUDA/HIP on a Linux GPU
box, else the CPU), installs the Python package and sets up the graphical launcher (`--no-gui` on a
cluster, `--cpu` to skip the GPU, `--double` for the double-precision pair); a double or 2D pair already built
is rebuilt with them. A failed compile leaves the previous binaries in place and prints the end of its build log. The steps below are what it does, by hand.

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd DTFE
   ```

2. **Check the platform and dependencies:**
   ```bash
   make test-platform    # shows the detected platform, compiler and paths
   make deps-check       # verifies every required header and prints the install command for anything missing
   ```

3. **Build the executables:**
   ```bash
   make DTFE             # standard DTFE
   make PS-DTFE          # phase-space DTFE (see below)
   ```
   `./PS-DTFE --version` prints the git revision and time of the build, which every run also writes into its log.

   On Apple Silicon, add `METAL=1` (e.g. `make DTFE METAL=1`) to run the volume-averaged interpolation / phase-space deposit on the GPU, with automatic fall-back to the CPU. On Linux, `CUDA=1` (NVIDIA, needs the CUDA toolkit's `nvcc`) or `HIP=1` (AMD, needs ROCm's `hipcc`; set `GPU_ARCH` to the card's architecture) builds the same deposit from a port of the Metal kernels; that port compiles (with `nvcc` and with ROCm's `hipcc`, both checked in CI) and falls back to the CPU correctly on Linux, but has not yet been validated on NVIDIA or AMD hardware, so its first run on a GPU machine should be the parity checks named in `metal/README.md`. As a shortcut, `./scripts/install.sh` installs any missing dependencies, builds both binaries with the best backend for the machine, installs the Python package (next step), and sets up the graphical launcher (see below; `--no-gui` leaves it out, e.g. on a cluster).

4. **Install the Python package** (`dtfelib`: loaders, analysis and the interactive `Estimator`):
   ```bash
   python3 -m pip install -e python
   ```
   Install it editable, so that it finds `./DTFE` and `./PS-DTFE` in the repository; otherwise point `DTFE_BIN_DIR` at the directory holding them.

5. **Build the shared library:**
   ```bash
   make library
   ```

6. **Clean build files:**
   ```bash
   make clean
   ```

7. **Run the tests** (optional):
   ```bash
   tests/ci_suite.sh core      # every CPU suite against ./DTFE and ./PS-DTFE
   ```
   `tests/ci_suite.sh` is the one list of suites that both the GitHub workflow (`.github/workflows/ci.yml`) and the Docker image run: `build` builds the single, double and 2D pairs, `double` repeats the suites against the double pair, `2d` runs the 2D suite and `gui` the launcher's tests. The GPU suites (`tests/dtfe_metal_check.sh`, `tests/gpu_scalar_check.sh`, `tests/gpu_mass_check.sh`) need a GPU build and are run by hand.

   **Standard-DTFE regression.** `tests/run_tests.py` compares a periodic 1000-particle run (point density and the
   `--exact-average` cell average on a 12^3 grid) with `tests/reference/regression_periodic.txt` to 3e-4 relative,
   on macOS and in Linux CI alike: the set has no Delaunay ties, so every CGAL builds the same tessellation, and the
   tolerance sits 11x above the measured rounding spread (double build, thread count, Linux vs Mac) and below the
   smallest real change tried. After an intended change of the estimator, regenerate it with
   `python3 tests/run_tests.py --update-ref` and commit the file. `tests/ps_auto_tune_check.sh` and
   `tests/ps_scratch_check.sh` now run on Linux too (part of `tests/ci_suite.sh core`); their GPU parts skip on a
   CPU build.

### Configuration Options

The software behavior can be configured by editing the `OPTIONS` section in the Makefile:

#### Spatial Dimensions
The number of dimensions is chosen per build, without editing anything:
```bash
make DTFE          # 3D (default)
make DTFE DIM=2    # 2D
```
A 2D build is a set of its own, `DTFE-2d` and `PS-DTFE-2d` (`-2d-double` with `DOUBLE=1`), built from its
own object directories, so the 2D and 3D binaries coexist; it is CPU-only. Phase-space DTFE works in
2D since 2 October 2026 (`make PS-DTFE DIM=2`): it reads the same Gadget-HDF5 layout with two-column
`Coordinates`, `Velocities` and `InitialCoordinates` (`tests/generate_ps_test_data.py --dim 2` writes
such files), and supports the sampled and exact deposits (the exact one clips each triangle against
the cells itself), the stream counts and hidden-stream flags, the velocity, dispersion and gradient
fields, `--ps-vertex-mass`, `--ps-volume-weighted`, `--ps-halo-release`, `--ps-linear-deposit`,
`--ps-caustics` (the parity bits and the collapse multiplicity, which runs to 2 in 2D), `--partition`,
periodic and non-periodic boxes, the tessellation cache and point evaluation (`--sample-points`, with
the points as `x y` text lines or raw float64 pairs; the `.pts_*` vectors then carry two components,
the dispersion three, the velocity gradient four), `--ps-window`, and `--ps-caustic-cusps`, which flags
the cusps where a fold line turns tangent to its null direction (there is no swallowtail bit in 2D: a
swallowtail is a moment of a 2D flow, not a point of it). The standard `DTFE-2d` averages exactly too
(`--exact-average`, clipping its triangles against the cells itself), and the interactive server works
in 2D too (`--serve`; `dtfelib.Estimator(..., dim=2)` or `Estimator.from_arrays` on (N, 2) arrays, and
`grid(xs, ys)`), its handshake telling the client the dimension. `tests/ps_2d_check.sh` checks it against analytic 2D Zel'dovich flows: the stream count in every
column of a pancake, the separable stream fractions of crossed waves, the convergence of the
single-stream density and velocity, the sampled deposit converging to the exact one, point
evaluation, whose stream counts at the cell centres equal the grid's exactly, the window against the
full run, the cusp flag against the two analytic cusps of a pancake modulated along y, and the exact
average against a perfect lattice. Until then a
2D build refused to compile: its input could not be read (a 3D file overran the 2D reader) and its
deposit had reported 62 streams where at most 3 exist. The T-web works in 2D too since 3 October 2026
(the same FFT Poisson solve; two eigenvalues, so the classes are 0 void, 1 filament, 2 node), checked by
`tests/tweb_check.sh` as in 3D plus an independent numpy solve of crossed waves for the cross term; the
GPU and the parallel triangulation stay 3D only. The graphical launcher runs 2D snapshots as well (see
below).

#### Variable Precision
The precision is chosen per build as well:
```bash
make DTFE PS-DTFE                   # the single-precision pair (default): ./DTFE, ./PS-DTFE
make DTFE PS-DTFE DOUBLE=1 METAL=1  # the double-precision pair as well: ./DTFE-double, ./PS-DTFE-double
```
`DOUBLE=1` keeps every position, density, field and accumulator in double precision, from the
input read onward -- the Gadget-HDF5 reader then keeps the full precision of snapshots that store
their coordinates in double (IllustrisTNG does), which the default build rounds on read -- and
writes the output grids as float64 (`dtfelib` and the launcher read either). It costs about twice
the memory. The double pair is a separate set of binaries built from its own object directories
(`o_d/`, `o_ps_d/`), so both precisions coexist: the run scripts pick the pair with
`DTFE_PRECISION=double`, the launcher with its precision setting, and `scripts/install.sh --double`
builds it next to the single one. The GPU is available to it: the GPU kernels (Metal, CUDA, HIP) are single
precision, so a double build hands the GPU float copies of its tetrahedra and adds the float sums
into its double grids, while reading, triangulation, densities, point evaluation and the CPU
deposits stay double. The one exception is the Metal mass grid (the density), which is an exact 64-bit
fixed-point sum in both builds, so a partition's mass comes out the same run after run whatever order the GPU's
threads add in (`PS_METAL_FLOAT_MASS=1` brings back the old float grid for comparison); the CUDA/HIP port has the same
fixed-point grid since 2026-10-11 (`PS_GPU_FLOAT_MASS=1` for its float grid), validated through its CPU emulation. A split run that merges several partitions at once (`--max-concurrent` above 1) still differs
from run to run in the last bits, because the merge adds the partitions in the order they finish. `tests/ps_double_check.sh` compares the two pairs on the same snapshot: the
grids agree the way two single-precision paths do (a few borderline samples change cell), stream
counts and caustic flags are identical, and the double pair's GPU deposit matches its own CPU run
like the single build's. Every suite that runs the programs also runs against the double pair:
`DTFE_TEST_PRECISION=double tests/<suite>` (or `tests/ci_suite.sh double`) points it at
`./DTFE-double` / `./PS-DTFE-double` and reads their float64 grids, with the same checks as the
single pair; only the comparisons of the GPU's stream counts with the CPU's exact ones allow one
cell in ten thousand to differ there, since the GPU works on float copies of the double tetrahedra.

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


## Phase-Space DTFE

Standard DTFE builds the Delaunay tessellation in Eulerian (present-day) space and returns a single-valued density at each point. **Phase-Space DTFE** builds the tessellation in Lagrangian (initial-condition) space and follows it as it is deformed to the present day. Because the deformed tessellation can fold over onto itself, the method recovers the *multi-stream* structure of the cosmic web: the density at a point is the sum over every stream (folded simplex) that covers it, and the number of streams is returned as a field of its own. This makes it especially suited to caustics and to the multi-stream interiors of voids, where the single-stream estimate breaks down. The method follows the phase-space tessellation approach of Abel, Hahn & Kaehler (2012) and Shandarin, Habib & Heitmann (2012).

Phase-Space DTFE is built as a separate `PS-DTFE` executable (the `-DPHASE_SPACE` compile option, set automatically by the target so that it can coexist with the standard build):
```bash
make PS-DTFE            # add METAL=1 on Apple Silicon (CUDA=1 / HIP=1 on Linux) to run the deposit on the GPU
```

It needs **two positions per particle** — the present-day (Eulerian) position and the initial-condition (Lagrangian) position — read from a Gadget-HDF5 file, either from an `InitialCoordinates` dataset in the same file or from a separate initial-conditions snapshot given with `--lagrangianInput`:
```bash
./PS-DTFE snapshot.hdf5 output_root --grid 256 --periodic --field density --lagrangianInput ics.hdf5
```

In addition to the standard fields, `PS-DTFE` returns the number of streams per cell, the velocity dispersion (its trace and the full symmetric tensor), and, optionally, the caustic structure of the map (`--ps-caustics`). Because the map of each tetrahedron is linear, its deformation tensor is constant inside it and known exactly, which gives both the fold surfaces — where the orientation of the map flips — and, alongside them, the number of principal axes along which the sheet has already collapsed: one for a wall, two for a filament, three for a node. Cells where two of the deformation eigenvalues coincide, the neighbourhood of an umbilic point, are flagged as well. The higher singularities of the caustic skeleton proper — the cusps — require in addition the derivative of the critical eigenvalue along its own eigenvector, which does not exist inside a tetrahedron where the deformation is constant; `--ps-caustic-cusps` estimates it from the neighbouring tetrahedra and reports the result as a further flag. The estimate is made over all tetrahedra sharing a vertex with the one in question, which matters: fitted from the four face neighbours alone it marked several per cent of the fold cells of a one-dimensional wave that has no cusps at all, and over the wider ring that falls to one cell in ten thousand. It is left switched off unless asked for, since the wider stencil is not free. The same neighbourhood answers the next question in the hierarchy at almost no extra cost, and the flag reports that too: where a cusp is a point at which the critical eigenvalue stops varying along its null direction, a swallowtail is one at which its curvature stops as well, so the estimate fits a cubic along that direction and asks whether the expansion begins only at third order. The separation this achieves is honest but modest — on waves whose singularities are known exactly, a true swallowtail lights some seventeen per cent of the cusp cells, a cusp that is genuinely not one lights one and a half, and an ordinary fold lights none — and it cannot be improved by tightening the test, because a swallowtail is a point rather than a surface and what is really being marked is the neighbourhood of one, over a band as wide as the stencil that measures it. Two refinements of the mass assignment keep the estimate accurate on evolved initial conditions: chart-independent tetrahedron masses (`--ps-vertex-mass`, which distributes each particle's mass over its incident tetrahedra) and volume-weighted velocity moments (`--ps-volume-weighted`, the volume-weighted convention used by standard DTFE). Inside virialised haloes, where the stream count is no longer converged, the thoroughly mixed sheet can be released from the deposit with `--ps-halo-release` following Stücker et al. (2021).


The stream count on the grid is measured the way the fields are, at the sub-samples of each cell (`--avg-subsamples`): it is the number of streams covering a sample, averaged over the cell, and it converges to the volume-weighted stream count as the sampling is refined. A tetrahedron smaller than the sample spacing can contain no sample at all. Its mass is still deposited, in the cell holding its centroid, so mass is conserved exactly, but it adds nothing to the count, which is what keeps the count unbiased; the stream count on its own cannot see a fold thinner than the sampling, so a separate output, `.hidden_streams`, marks the cells that hold multi-stream volume their count misses (exactly: a folded tetrahedron overlaps the cell), and, as a second bit, the cells that received such a sub-sample tetrahedron's mass. A single-stream mask should use both, `.streams` equal to one and bit 1 of `.hidden_streams` clear; `velocity_single_stream()` in `dtfelib` does this. The flag is conservative: it also marks cells in which the unsampled tetrahedron belongs to the one stream that is there, which happens throughout the voids when the tessellation is finer than the sampling. Which tetrahedra contain a sample is decided exactly, by the test described under point evaluation below, so with one sample per cell the grid count is the exact number of streams at the cell centre, in every cell, and a `--partition` split gives the same count as a serial run. When every unsampled tetrahedron was counted as a stream, 11.6% of the cells of a crossed-wave test were too high, by up to 839, and the floating-point inside test that preceded the exact one miscounted a few cells per million. The GPU deposit (`--ps-gpu`) reaches the same decisions: it has no double precision, so it classifies each sample in single precision against a rigorous bound on its rounding error and hands the few tetrahedra it cannot decide that way, those with a sample within rounding distance of a face, to the CPU's exact test; its stream counts are identical to the CPU's. Its mass grid is exact as well (2026-10-08): each deposit is added as a 64-bit fixed-point integer, so the sum no longer loses the small shares that float32 drops in cells receiving millions of them (the TNG50-3-Dark z = 0 grids made before this have a box mean of 0.99941 instead of 1) and is the same from run to run whatever the order of the GPU's atomic adds; `PS_METAL_FLOAT_MASS=1` brings back the float grid for comparison. The velocity moments are still summed in float32, and the CUDA/HIP port still sums the mass in float.

Whether a tetrahedron is too flat to be inverted is decided relative to its own size, not against a fixed volume, so the result does not depend on the length unit or on the size of the box: the same snapshot scaled into a 1 Mpc box used to lose a fifth of its mass to that test, and now conserves it exactly.

A `--box` region cut from a larger particle cloud keeps exactly the mass that lies inside it. A tetrahedron crossing the region's face deposits only its share inside: its samples beyond the face are counted but not deposited, and in the exact deposit its pieces inside the region are weighed against the whole tetrahedron. It used to give its entire mass to the part inside, so the cells along the faces of a region cut from a TNG100 box held three times the density of their neighbours, and where tetrahedra span many cells the excess reached well into the region. A region now reproduces the same cells of a larger grid to the last digit.

The Lagrangian domain of a non-periodic cloud is its alpha shape rather than its convex hull: only tetrahedra whose Lagrangian circumradius is at most three mean particle spacings are kept (`--ps-alpha-shape`, 0 keeps the whole hull). Interior tetrahedra of lattice-like initial conditions never exceed about 0.9 spacings, in jittered lattices and in TNG100-3 alike, so the cut touches only the boundary. What it removes are the flat slivers that fill the space between the nearly planar faces of the cloud and its hull, joining particles up to a whole face apart, together with the tetrahedra that bridge concavities, such as particles missing from a cut region. Neither is a flow element: their Eulerian images stretch across the cloud and deposit spurious mass and streams wherever they pass. They are also why a non-periodic run split with `--partition` used to differ from a single triangulation along the partition seams, since the hull of a face is a global object that each partition built differently. A tetrahedron within the cut, by contrast, belongs to a partition's triangulation exactly when it belongs to the global one, and the partitions are padded accordingly, so split and single runs now agree to the rounding of their sums. The boundary layer beyond the cut is the price: about half a percent of the mass of an isolated 24³ clump.

Two costs that dominate small runs were cut. The vertex densities are now summed over the cells, each adding its volume to its four vertices, instead of walking every vertex's incident cells and recomputing each cell's volume four times: 0.6 s → 0.14 s for a 0.26M-particle box in `PS-DTFE`, 0.25 s → 0.02 s in `DTFE` (the sums run in double and in a different order, so a density can move in its last float bit). And `PS-DTFE` can build its Delaunay triangulation in parallel (`--parallel-triangulation`, when the `tbb` library was present at build time; `make` detects it, `TBB=0` leaves it out): it uses the performance cores, at most six (`DTFE_TBB_THREADS` overrides: the lock-coupled insert gains nothing from more, and a thread on an efficiency core stalls the rest), shared among the tessellations being built at the same time, so a partitioned run, which already keeps every core busy, inserts as before and only its tail gains. For a small set — below two million particles — the tuner also predicts whether splitting the set into eight or twenty-seven partitions, each triangulated on its own core, beats one triangulation, and splits only for a clear gain: a 0.26M-particle slice takes 1.4 s instead of 2.4 s, its 16.8M point values 4.2 s instead of 5.3 s, while grids, whose deposit already uses every core (and on the GPU serialises anyway), stay in one domain. Such a split merges its partitions in a fixed order, so two identical runs still agree bit for bit; `--partition 1 1 1` asks for one tessellation when the bits of a single-domain run are wanted, for instance to compare with the query server. The parallel insert is off by default because its vertex and cell order differ from run to run, which moves every floating-point output between two identical runs at the level of rounding — the stream counts and the tessellation itself do not change — whereas the sequential insert reproduces bit for bit, which the query server, the tessellation cache and the test suites rely on. With the flag a 0.26M-particle slice takes 1.4 s instead of 2.4 s, and the 27-partition TNG region's GPU grid 110 s instead of 135 s; a partitioned run with more than one concurrent partition was never bit-reproducible, so there the flag costs nothing. The standard `DTFE` binary keeps the sequential insert: its clustered Eulerian points built four times slower in parallel, where the near-lattice Lagrangian points of `PS-DTFE` build about twice as fast.

The CPU deposit of `PS-DTFE` runs on every core by giving each thread its own copy of the accumulators, summed at the end; those copies are capped at a fifth of the memory, so on a fine grid, where one copy of the density and velocity accumulators is 24 GB at 1024³, the cap left a single thread. When the copies would leave fewer than half the threads, the deposit now divides the grid into slabs along x instead: each thread takes whole slabs as it comes free and writes only their cells, walking each slab's tetrahedra in the order of the single-threaded loop, so every cell receives its contributions in exactly that order and the result is identical, bit for bit, to a single-threaded deposit at any thread count (`tests/ps_slab_deposit_check.sh` compares every output file for every deposit variant). The slabs are cut where the estimated work, not the width, is equal, so a clustered box does not leave one thread with its halos. A 0.26M-particle box on 1024³ deposited in 65 s instead of 245 s, faster than the GPU's 160 s; on 768³ in 28 s instead of 64. `DTFE_DEPOSIT_SLABS=1` forces the slabs and `0` forbids them, `DTFE_DEPOSIT_THREADS` sets the thread count.

## Phase-space density

The distribution function of a collisionless sheet cannot be tabulated: it is a sum of delta
functions, one per stream. Standard DTFE therefore offers an *approximation*, `--approxPSD`, which
builds a second Delaunay tessellation in velocity space and multiplies the spatial and velocity
densities, f ≈ ρ(x)·g(v) — a factorisation that is reasonable in a cold, single-stream flow and
breaks down exactly where streams cross. Phase-Space DTFE needs no such assumption: it resolves
the streams themselves, so the exact distribution function is the per-stream decomposition it
already reports (`--sample-points --per-stream`), and the usual scalar proxy Q = ρ/σ³ follows from
the multi-stream density and dispersion it deposits (`dtfelib.fields.pseudo_phase_space_density`).
Running the two against each other measures how far the factorised estimate drifts in the
multi-stream regions where the phase-space structure actually matters.

## Field evaluation methods

The value assigned to a grid cell can be either the field at the cell centre or the field averaged over the cell volume; the volume average is normally estimated by Monte-Carlo sampling (`--method`, `--samples`). Two alternatives remove the sampling noise entirely:

* **Exact volume averaging** (`--exact-average`, standard DTFE) integrates the linear DTFE interpolant analytically over every grid-cell/tetrahedron intersection using the vendored [r3d](third_party/r3d/README.md) library (Powell & Abel 2015); with `--gpu` the clipping runs on the GPU (a float32 port of r3d that agrees with the CPU to float rounding, 6.6 times faster than the CPU on a 0.26M-particle box at 256³). For a linear field the integral over each intersection is its centroid value times its volume, so the averaged field carries no sampling noise at any resolution.
* **Exact conservative deposit** (`--ps-exact-deposit`, PS-DTFE) does the same for the phase-space deposit: each stream is clipped analytically against the grid and deposits its mass exactly, with the velocity and dispersion following from the linear profile over each intersection. An intermediate option, `--ps-linear-deposit`, keeps the sampled deposit but weights each sub-sample by the linear density profile inside its stream.
* **A window of the grid** (`--ps-window x0 x1 y0 y1 z0 z1`, PS-DTFE, coordinates in Mpc) deposits, allocates and writes only the cells of the full `--grid` inside that box. The tessellation is the full run's (periodic or not, with or without `--partition`) and so is every stream's normalization, so the window's cells are the full run's: bit for bit for the sampled deposit (a stream straddling the window edge counts all its sub-samples and deposits the part inside), to float rounding for the exact deposit (only the window's cells are clipped and each stream is normalized by its whole weight, which equals the sum over all its pieces to double rounding). The cost scales with the window, not the grid: a 4096³ grid over a 100 Mpc box windowed to one 512² slice holds a quarter of a million cells and runs in seconds after the tessellation. A box beyond the periodic edge wraps. With `--tessellation-cache` and `--partition`, a window run loads the cached partition tessellations instead of rebuilding them and skips every partition whose occupancy map (written beside its tessellation by the query server or by an earlier partitioned run) shows that its streams never reach the window: on TNG100-3-Dark (94 million particles, 4×4×4 partitions cached by the query server) a 512² exact window processes 12 of the 64 partitions and takes about 15 s, where it took about an hour before. Three things make that possible beyond the skipping. A partitioned run with a cache records the totals it derives from all particles (their number, the mean density and the bounding box of their initial positions) beside the partitions, so a later window run whose partitions are all cached does not read the snapshot at all (16 s and 12 GB of memory on TNG100-3-Dark), and reads it after all only if a partition it needs turns out to be missing. On the GPU, the vertex masses of `--ps-vertex-mass` are counted only for the tetrahedra that reach the window, which halves each partition's deposit. And the cached tessellations are compressed in independent chunks, written and decompressed on several cores. Not combinable with the T-web/V-web classes (they need the whole periodic grid), `--region`, `--partNo` or `--ps-linear-deposit`.

The caustic structure is reported at the sampling points as well, so the fold skeleton can be drawn at the full resolution of a point-evaluated image plane rather than at the resolution of a grid. The two cosmic-web classifications are a different matter: the T-web follows from the tidal tensor, a global solve over the whole density grid, and the V-web from the velocity shear under the normalisation the binary applies, so neither is a local quantity that point evaluation could produce by itself. They are instead sampled from the grid at the very positions the plane was evaluated at (`dtfelib.io.PointPlane.sample_grid_field`), which keeps them identical to the classification the grid reports.

Besides the regular grid, **both binaries** can evaluate their fields at **arbitrary points** supplied by the user (`--sample-points`) — halo or void centres, sight-lines, or the pixels of a high-resolution image plane. The points are read from a text file (one `x y z` per line) or a raw-binary file, and one record is written per point. For `PS-DTFE` this returns the full per-stream decomposition at each point (`--per-stream`); for the standard binary it is the ordinary single-valued DTFE interpolant.

Point location is exact. Which tetrahedra contain a point is decided with exact orientation predicates, and a point lying exactly on a face, an edge or a vertex -- a query at a particle position lies on the vertex shared by some twenty-five tetrahedra -- is assigned by a symbolic perturbation that every tetrahedron resolves the same way, so it is counted once per stream, never twice and never not at all. Periodic copies of a particle, whose coordinates differ from the original's by the rounding of the box shift, are mapped to one canonical position first, so the same holds across the periodic boundary and under any `--partition` split. A plain floating-point test with a tolerance does not have this property: at particle positions it reports the whole vertex star, some twenty-five spurious streams, and near shared faces it double-counts a small fraction of all points. The exact decision is only invoked for points within rounding distance of a face; everywhere else a floating-point test decides, with a margin that provably agrees with it, so it costs nothing measurable. The evaluation itself runs on all cores. The CPU grid deposit uses the same test for its sub-samples; with 27 samples per cell it is even some 20% faster than the tolerance test it replaced, which computed every sample's position in full before rejecting most of them.


## Interactive point evaluation

A `--sample-points` run evaluates one fixed set of points and ends. For exploratory work the tessellation can instead be kept alive and queried from Python whenever needed, at any points, without rebuilding it:
```python
from dtfelib import Estimator

with Estimator("snapshot.hdf5") as est:       # PS-DTFE; the tessellation is built here, once
    f = est(points)                           # (N, 3) points in Mpc
    f.density, f.velocity, f.streams, f.dispersion
    rho = est.density(halo_centres)
    cube = est.grid(xs, ys, zs)               # the fields on the grid xs x ys x zs
```
Behind it the binary runs with `--serve`: it builds the tessellation and the vertex densities exactly as a normal run does, and then answers point requests over its standard input and output until the Python object is closed. Every answer is bit-identical to what `--sample-points` would write for the same points, the per-stream records, gradients and caustic flags included (`Estimator(per_stream=True, density_gradient=True, ...)`), and `phase_space=False` gives the standard estimator. A small request only visits the tetrahedra near its points, through a cell index built once when the server starts, so a single point is answered in some tens of microseconds, the round trip from Python included; a large one runs on all cores. Combined with `--tessellation-cache` (`Estimator(..., tessellation_cache=dir)`), even starting a session on a snapshot seen before is immediate. Velocities are returned in peculiar km/s, the convention of the rest of `dtfelib`.

Particles that are not in a snapshot file can be handed over as arrays, the way CosmoDTFE's estimators take them; they are written to a temporary file that the server reads and that is removed again when the session closes:
```python
est = Estimator.from_arrays(x, velocities=v, lagrangian=q, values=phi, box=100.0,
                            per_stream=True, lagrangian_positions=True)
f = est(points)
f.stream_lagrangian        # (R, 3): for each stream at each point, where its matter started
f.stream_scalar            # (R,): 'values' (e.g. the potential) along that stream
f.scalar                   # (N,): its density-weighted mean over the streams at the point
```
Without `lagrangian` the arrays give the standard estimator, and without `box` the particles' own bounding box is used and the data is taken to be non-periodic. The same two outputs exist for any run: `--pts-lagrangian` writes each stream's Lagrangian coordinate q(x), the inverse of the folded map from initial to present positions (`.pts_stream_lagpos`), and `--pts-scalar` interpolates a per-particle quantity read with `--scalar-dataset NAME` from the snapshot (`.pts_scalar`, and `.pts_stream_scalar` per stream). Both are interpolated with the same weights as the velocity; on crossed Zel'dovich waves x - q(x) equals v/100 on every stream to the last floating-point digit.

A box too large for one tessellation is served as a composite of partition tessellations, the out-of-core counterpart of CosmoDTFE's `CompositeEstimator` (`Estimator(..., partition=4, tessellation_cache=dir)`, or `--serve --partition 4`; with a cache directory the code also switches to it by itself when one tessellation would not fit in memory, and then picks the split itself: one already in the cache for that snapshot, the finest complete one, so nothing is built again, and otherwise the split that holds about eight partitions in memory, the number a zoom touches -- on TNG100-3-Dark a cold 512² zoom took 9.8 s with 4³ partitions, ten of them held, against 19.7 s with 3³, four held). The partitions are built once, in parallel, and written to the cache; for each partition a coarse map of the region its tetrahedra cover is kept, so a request only visits the partitions that can hold one of its points, and only a few partitions are held in memory at once, the least recently used being dropped and reloaded on demand: by default as many as fit in the memory budget, counting each partition's tessellation (about 0.67 KB per vertex, measured) and the cell indexes that exist, or a fixed number with `resident` (`--serve-resident`). Until 2 October 2026 every partition was counted with an index it rarely has, which held TNG100-3-Dark's 14-million-vertex partitions to three where four fit, so a zoom needing four reloaded one each time. A later session with the same inputs starts from the cache without building anything, and a batch run with `--tessellation-cache` and the same `--partition` fills the same cache. The partitions a request needs that are not in memory are read back from the cache in parallel (up to eight at a time, within the resident limit) while the ones already in memory are evaluated, so the first request in a new region costs about one partition load rather than one per partition: on a 2-million-particle box split into 27 partitions, 3.9 s instead of 16 s. With `--serve-progress` (`Estimator(..., progress=True)`) the server writes a plain progress line to its log for every partition it builds or a request visits, saying whether that partition was in memory or loaded from disk, and a summary per request; the launcher's progress bar reads them. A partition's cell index, which speeds up small requests, is built only once that partition has answered eight requests while in memory; until then it is walked (on a TNG100-3-Dark partition of 14 million vertices the index takes about 20 s to build and a walk about 1.2 s, so building it at once made the first zoom or click in a region several times slower, and a region needing more partitions than fit in memory rebuilt it on every request). An index is also built only when it fits in the memory budget next to the partitions in memory, since evicting a partition for it would cost a reload; otherwise the request walks, and the progress log says so. The answers are bit-identical to `--sample-points` with the same partition split; against a single tessellation the stream counts are identical and the values agree to floating-point rounding.


## Automatic memory management

Large runs are split into smaller computational chunks (`--partition`) that are processed a few at a time (`--max-concurrent`; 0 means as many as there are threads, never more), so that the peak memory stays within the machine. When these options are not supplied the code **chooses them automatically** from the particle count, grid size, requested fields and the machine's memory and core count, printing the chosen decomposition and the predicted peak memory before it starts; any explicit option always takes precedence, and `--partition 1` asks for a single, unsplit computation. The memory it plans for is at most 60% of the machine's RAM, and never more than what the programs already running leave free (less a few GB of headroom), so the computer stays usable during a run; `DTFE_MEM_FRACTION` changes the share and `DTFE_MEM_BUDGET_GB` sets the budget outright, though never above what the other programs leave free at the start (less 4 GB). While a phase-space grid run has several partitions in flight, a partition does not start when the memory then free would not hold it: it waits until one in flight has finished (and says so), or starts alone when none is, so programs that take memory during the run slow it down instead of getting it killed by the system, as the TNG50-3-Dark run at z = 0 (6×6×6 partitions, three at a time) was after 82 minutes on 7 October 2026. This gate works from the auto-tuner's estimate of what one partition adds, so it acts only when the auto-tuner chose the split or the concurrency and more than one partition runs at a time. It does nothing when the partitions run one at a time (`--max-concurrent 1`, given or auto-tuned, as for the 5×5×5 rerun of that snapshot), when both `--partition` and `--max-concurrent` are given by hand, for a point evaluation (`--sample-points`), or for the small partitions the auto-tuner picks for speed when a run fits in memory easily; such a run has only the budget it was planned with. Very fine grids work beyond the 32-bit cell counts: a 1300³ grid (2.2 billion cells) on the CPU and a 1700³ grid (4.9 billion) on the GPU, both with the grids on `--scratch-dir`, reproduce the 2×2×2 block means of a run at half the resolution with twice the sub-samples (the same sample points) to float rounding. The GPU kernels number a partition's sub-grid cells with 32 bits, so with the GPU on the code now splits until every sub-grid holds fewer than 2³² cells (a 1700³ grid: 2×2×2) instead of falling back to the CPU, and a small particle set on such a grid splits for memory instead of staying in one domain over budget. The T-web's FFT is planned with FFTW's 64-bit interface, so it is no longer refused above 2³¹ cells; at that size its double-precision work arrays need on the order of 100 GB, and since they are ordinary allocations `--scratch-dir` backs them with disk like the grids: a 1300³ T-web (2.2 billion cells) ran on a 64 GB machine in 19 minutes, its classification checked in every cell. The memory of a phase-space point evaluation (`--sample-points`) grows with the number of streams crossing each point, which rises steeply towards low redshift; the code estimates it from the particles before it starts (the Eulerian over the Lagrangian volume of the tessellation, measured on a few hundred small patches) and plans with a generous margin on top, since a single plane can cross more collapsed structure than the box average. Each stream is stored with only the quantities that were asked for (42 bytes per stream at the least, 114 with `--pts-vel-grad`, instead of 176), which leaves every output unchanged. `--auto-tune-report` reads the input, prints the decision and the predicted peak memory as one line, and stops, so a configuration can be checked before it is run. When even a single chunk of the full-resolution grids would not fit in memory, the grids can be backed by memory-mapped files on a local disk with `--scratch-dir`, trading disk space for RAM while leaving the results unchanged.

### Long runs: ordered merge, look-ahead, crash-safe outputs, progress

**Reproducible partition merge (`--ordered-merge`).** A partitioned PS-DTFE run normally adds each partition's
grids into the result as soon as it finishes, so with several partitions in flight the float sums happen in a
different order every run and two identical runs differ in the last bits. `--ordered-merge` adds them in
partition index order instead, so the grids are bit-identical run to run (on the GPU only the exactly summed mass
grids are; the float velocity moments vary at rounding anyway). A finished partition waits for the
lower-numbered ones, which can cost time when partitions differ much in cost; the memory gate keeps working.

**Look-ahead tessellation (GPU runs at `--max-concurrent 1`).** While one partition deposits on the GPU, the next
partition's particles are selected and its Delaunay tessellation and vertex densities are built on a CPU thread,
one partition ahead. The outputs are byte-identical to a run without it; on a synthetic 2.1M-particle GPU run it
cut the wall time by ~20%. It holds one more partition's tessellation in memory, which the auto-tuner counts and
which is checked against the free memory before each look-ahead. Switch it off with `--ps-no-overlap`.

**Crash-safe outputs.** Every output file is first written as `<file>.part.<pid>` and the whole set is renamed
into place only after the last one, so an output folder never holds a mix of an old and a new run. If the
output folder is missing when the results are written (an unmounted disk), the program waits for it
(`DTFE_OUTPUT_WAIT` seconds, default 300) and otherwise writes everything to
`$TMPDIR/dtfe-rescue/<folder>_<name>`, prints the `mv` command to put it back, and exits with status 3.
Leftover `.part.<pid>` files of a killed earlier run are removed at the next start.

**Progress sidecar.** While a run goes, the binary keeps `<output>.runlog.progress` up to date: a small JSON file
with the phase, partitions done/total, elapsed time, ETA, current memory, GPU retries and the update time. It is
removed when the run finishes. The ETA weighs each partition by a predicted cost (how much of its mass sits in
dense regions) instead of counting partitions; on the TNG50 z=0 run that brings the ETA from about 2x too long to
within about 20%. `DTFE_NO_PROGRESS=1` switches the file off.

**Resuming a killed run.** A partitioned PS-DTFE run saves its merged grids, and the list of partitions
already in them, in `<output>.runlog.checkpoint` (with `--scratch-dir`: in the scratch folder) after a
partition merges. It skips a save while the previous one took more than 5% of the time since it, and
removes the folder once the outputs are written. If the run is killed (a restart, an unplugged disk,
out of memory), run it again with `--resume`: the binary loads the checkpoint and computes only the
missing partitions. It refuses, naming the differing settings, if the build or the binary itself (a
rebuild: its revision and link time), the input files, the partition split or the command line changed. `--verbose`, `--max-concurrent`, `--scratch-dir`,
`--tessellation-cache`, `--ps-no-overlap` and `--ordered-merge` may change. `--resume` implies
`--ordered-merge`: when the killed run also merged in index order, the grids are byte-identical to an
uninterrupted run at the same `--max-concurrent`. `scripts/run_ps_dtfe.sh` resumes on its own when it
finds the checkpoint (`PS_RESUME=0` starts over). `DTFE_PS_CHECKPOINT=0` writes no checkpoints, and
`--sample-points` and `--interlace` runs are not checkpointed. A save needs twice the grids' size plus 2 GB free; with
less, it is skipped with a warning.

**ORDERED_MERGE=1** (run_ps_dtfe.sh) passes `--ordered-merge`: the partitions' grids are added in
partition order, so two identical runs write bit-identical grids. It costs time when partitions differ
much in cost and MAX_CONCURRENT is above 1.

**Failed GPU buffers resume from a checkpoint.** The Metal deposit runs each partition as a sequence of short,
paced command buffers. When macOS kills one (the display watchdog, another app's GPU hang, a memory error), the
deposit no longer starts the partition over: it keeps a snapshot of its grids at the start of each pass and,
by default, every ~10 s of GPU work (`DTFE_GPU_CHECKPOINT_CHUNKS=<n>` sets a checkpoint every n buffers; 0 keeps
only the pass-start snapshot), restores the last one and continues from there, so a failure costs at most ~10 s
of work instead of up to ~30 min on a TNG50 partition. The retry depends on the cause: an "innocent victim"
error (another process hung the GPU) retries at once at the same pace; a watchdog or memory error shortens the
buffers and widens the gaps for the rest of the run. Only three failures of the same chunk (or a lost device)
send the partition to the CPU deposit. The snapshot costs one extra copy of the partition's grids in memory
while its deposit runs (~0.2 % of the time measured at 256^3). Each failed buffer counts into the progress
sidecar's `gpu_retries`. For testing, `DTFE_GPU_FAULT_AT=<k>[:innocent|timeout|oom|persist]` makes the k-th
buffer fail after it ran; tests/gpu_retry_check.sh checks that the resumed run's mass grid is byte-identical to
an undisturbed one.

**CUDA/HIP: exact mass grid, and the port checked in CI without a GPU.** The CUDA/HIP phase-space deposit now
sums the mass grid exactly, like the Metal one since 2026-10-08: every mass share is added as a 64-bit
fixed-point integer (two 32-bit atomic words per cell), so no small share is lost in a busy halo cell and the
density grid is bit-identical from run to run (`PS_GPU_FLOAT_MASS=1` brings back the float sum for
comparison). Without an NVIDIA or AMD card the port can still be run: `make DTFE PS-DTFE CUDAEMU=1` compiles it
against a CPU emulation of the device, and `tests/cudaemu_check.sh` runs every GPU parity suite through it at
warp width 32 (CUDA) and 64 (AMD's wavefront) -- also as a CI job on Linux. That validates the port's code and
host logic, not the numerics of real hardware; a run on a GPU machine
(`GPU_BUILD=CUDA=1 tests/dtfe_metal_check.sh` and the PS suites) remains the final word.

## Reusing the tessellation

Constructing the Delaunay tessellation and the densities at its vertices is the dominant fixed cost of a run — measured at 81% of the total for a snapshot of 884736 particles — and it is repeated in full every time, even when the same snapshot is analysed again with a different grid or a different set of sampling points. `--tessellation-cache <local dir>` writes the finished tessellation to disk once and reuses it in later runs, which skips both stages (the same example run takes 10.9 s without the cache and 4.6 s with it). For phase-space runs the interpolation grid, the fields and the sampling points may all change and the cached tessellation is still valid, since it is built in Lagrangian space and does not depend on them — which is what makes repeated point evaluation on one snapshot cheap. Anything that would change the tessellation itself or the densities stored in it — a modified input file, a different box, region, padding, periodicity or density normalisation, or a different build of the code — is detected and the tessellation is simply rebuilt, so a stale cache can never produce a wrong result. (Periodicity was missing from that list until 30 September 2026: a standard-DTFE run could load the tessellation a run with the other setting had cached. A periodic phase-space run in one piece, without `--partition`, rebuilds a tessellation cached before 2 October 2026 once: its ownership box, part of the identity, moved that day to fix a mass loss on lattice initial conditions.) The input files are identified by their resolved path, so a relative name, a name through a symbolic link, or other letter case of the same file finds the same cached tessellation; before 2 October 2026 each such name was a separate key and a rebuild (a cache written through such a name then is rebuilt once). The cached file is compressed and costs about 0.15 KB per tessellation vertex. Since 2 October 2026 it is compressed in independent 16 MB chunks, written and decompressed on several cores, which made writing a partition 3.6 times and loading it 1.7 times faster; files in the older single-stream format still load, and `python/tools/upgrade_tess_cache.py <cache folder>` converts them, checking each file's content before replacing it (an older binary cannot read converted files). After that, most of a load was CGAL's own reader taking every vertex and cell index through a separate stream call; the cache now reads and writes the same bytes in blocks and builds the triangulation through the same calls in the same order, so the result is CGAL's exactly (the tests compare a hash of the loaded structure and the written files byte for byte). For 3.65 million vertices a load went from 3.2 s to 0.9 s and a save from 3.8 s to 2.2 s; with eight loads at once, decompression on the shared cores is now the limit (2.5 s each instead of 5.6 s). The 512² exact zoom on TNG100-3-Dark, three partitions at a time, went from 27 to 15 s. When libdeflate is installed (`brew install libdeflate`, `apt-get install libdeflate-dev`; the Makefile finds it, `LIBDEFLATE=0` builds without it) the chunks are compressed and decompressed with it instead of zlib: on this Mac compression is 2.4 times faster and 3% smaller, which made a save 12% faster, while a load gains little (1.13 times). On Linux (Ubuntu 24.04) it is much the same: a 2.1-million-vertex tessellation saved in 2.0 s instead of 2.3 s and loaded in 1.15 s instead of 1.28 s. The chunks are already decompressed on all cores, so the codec is only a small part of either. Both codecs write plain gzip, so every build reads every cache, and a run that loads one names the codec it used.

## Running on TNG snapshots

The thesis workflow is three entry points in `scripts/`, run from the repository root, all driven by the
same two variables: `DTFE_DATA_ROOT` (where the simulations live) and `DTFE_SIM` (which one). `config.sh`
holds the defaults and resolves either data layout -- per family (`$DATA_ROOT/TNG100/TNG100-3-Dark/snapdir_099/`,
the family being the name up to its first `-`) or flat (`$DATA_ROOT/TNG100-3-Dark/snapdir_099/`); the Python
side (`dtfelib`, `python/config.py`) reads the same variables. Every snapshot is one folder `snapdir_NNN`
holding its merged, h-free `combined_NNN.hdf5` and, after a run, the grids `ps_output.a_den` ... (phase-space)
or `output.a_den` ... (standard DTFE) beside their `.runlog`. The `.runlog` always belongs to the grids on disk: a
run writes `<prefix>.runlog.partial` while it goes, renames it onto `.runlog` when it succeeds, and to
`.runlog.failed` when it fails or is stopped, leaving the old grids' log alone (a `.partial` left behind is a run
that was killed outright). The launcher's Runs tab shows such an attempt beside the run it did not replace.

1. **Download and merge** (needs a TNG API key in `~/.tng_api_key`):
   `DTFE_SIM=TNG100-3-Dark scripts/download_snapshots.sh 50 99` fetches the raw chunks of snapshots 50 and 99
   (`-i` the initial conditions, `-t` the merger trees, `-c` the group catalogues) and merges them with
   `python/tools/merge_HDF5.py`; a combined file appears only when every chunk is there and the totals verify.
2. **Run an estimator** over the snapshots on disk:
   `DTFE_SIM=TNG100-3-Dark scripts/run_ps_dtfe.sh -g 512 -m 50 99` (phase-space, the GPU deposit) or
   `scripts/run_dtfe.sh -g 512 -m -e 50 99` (standard, exact cell averages). `DRY_RUN=1` prints every
   snapshot's expanded command and runs nothing; `--help` prints the knobs. `scripts/run_ps_pipeline.sh`
   does the same for every simulation found, plus a full-box image plane per snapshot, and is resumable.
3. **Analyse**: `DTFE_SIM=TNG100-3-Dark python3 python/analyze.py all` (the standard-DTFE thesis set:
   `compute` caches the smoothed density, Hessian and void catalogue, `plot` draws from the cache, `export`
   writes the void catalogues as CSV + HDF5), and the phase-space scripts in `python/plot/` take `--sim`:
   `plot_PS_DTFE.py --snap 99`, `plot_web_streams.py --snaps 99 67`, `plot_void_profiles.py`, `plot_pk_compare.py`,
   `plot_void_population.py`, `plot_void_tracking.py`, `plot_caustic_skeleton.py` (the last needs a run with
   `PS_CAUSTICS=1`: TNG100-3-Dark at z = 0 has one under its own prefix, `--snap 99 --prefix ps_caustic`);
   `plot_void_crossbox.py` sets the boxes side by side at one smoothing length in Mpc (each in
   its own process with its own `DTFE_SIM`): resolved fraction, distinct voids per volume and R_v against
   redshift, and dn/dlnR_v and the stacked profiles in r/R_v at z = 0, 0.5 and 1.
   Set `DTFE_SIM` for every analysis run: the cached void catalogue's lengths are in that simulation's frame.

**Voids.** The catalogue's voids are the minima of the density contrast smoothed with a Gaussian of
`SMOOTHING_SIGMA_CELLS` = 10 cells (`python/config.py`). Each has two sizes:
- **R_eff**, the ellipsoid fit's (a b c)^(1/3), a curvature length from the Hessian at the minimum. It sets the
  resolution cut (`config.ELLIPSOID_CUT_RULES`: largest semi-axis within 10 smoothing lengths and half the box) and
  the shapes.
- **R_v**, the *measured* radius, where the spherically averaged smoothed density around the minimum first rises back
  to the mean: about 0.57 R_eff. R_v is defined by that crossing of the smoothed profile, so in units of R_v the stacked (raw) density reaching the
  mean near 1.0 R_v checks the smoothing against the raw field rather than being a result; what is measured is that the
  outflow turns to infall just beyond it.

The profile, abundance and skeleton scripts use R_v and the *distinct* voids by default: deepest first, a void whose
centre lies inside the region of a deeper one (min(R_v, R_eff)) is left out. `--radius r_eff` and
`--keep-overlaps` give the R_eff-based, overlapping sample instead.

The smoothing in cells is a different physical scale in every box: 1.0, 2.2 and 5.9 Mpc in TNG50-3, TNG100-3 and
TNG300-3. To compare boxes, smooth them all at the same physical scale with `--smooth-mpc 2.1625` on the void
scripts (or `DTFE_SIGMA_MPC=2.1625`; the minima finder's window follows). At that common scale TNG100 and TNG300
(in that order) have the same number density of distinct voids, 58.9 and 59.1 per (100 Mpc)³ at z = 0, and resolve
20.8 and 22.8% of their voids; their median R_v agrees to 3% at z = 0 (6.92 and 6.71 Mpc), but TNG300's is the
smaller by 7% at z = 0.5 (7.4 and 6.9 Mpc) and by 11% at z = 1 (7.6 and 6.8 Mpc).

Products made at a non-default smoothing get their own names (e.g. `void_profiles_ps_resolved_s2.1625Mpc_f5_z0.00`)
and never go into the thesis `Figures/` tree. `analyze.py compute|plot|all` refuses to run under `DTFE_SIGMA_MPC`,
because those scripts write under the production names. The thesis `Figures/` tree takes the core `analyze.py`
figures only from the thesis simulation and estimator (`config.THESIS_SIMULATION` = TNG50-3-Dark,
`THESIS_METHOD` = standard DTFE); another simulation's or estimator's run says so and copies nothing there, and
`analyze.py --no-mirror` keeps any run out of it. `analyze.py check` accepts either estimator's grids and names
the one it found; `compute` and `export` without snapshot numbers take the configured ladder where it is on disk.
Other non-default options add a tag to the output names instead of overwriting the defaults: `plot_web_streams`
(`_<prefix>`, `_raw`, `_snaps50-99` for a subset), `plot_caustic_skeleton` (`_<prefix>`, `_c6`, and with `--voids`
`_reff`, `_shells...`, `_overlaps`, `_cut<N>`), `plot_void_population` (`_dtfe` or `_<prefix>`, `_overlaps`),
`plot_void_tracking` (`_dtfe` or `_<prefix>`, `_snap<N>`) and the point-evaluation maps (`_slab`, `_smooth<s>`,
`_fixed`). So `--snap 99 --prefix ps_caustic` writes `caustic_skeleton_z0.00_ps_caustic.*`.

### Void analyses added 2026-10-11

**Three void radii.** Every void catalogue now measures R_v three ways in one pass over spherical shells of the smoothed
density contrast around each minimum (`catalog.R_V_DEFINITIONS`): `crossing` (the default R_v: where the shell mean first
rises to the cosmic mean), `delta08` (where the enclosed contrast Delta(<R) first reaches -0.8, the shell-crossing value
of the spherical model) and `ridge` (the radius of the compensation ridge, the maximum of the shell mean within 15 sigma).
Read them with `catalog.r_v_mpc(cat, 'delta08')`; the export has `r_v_delta08_mpc` and `r_v_ridge_mpc` columns. A void
whose crossing radius shrinks while its delta08 and ridge radii hold is latching onto an inner wall.

**R_v settings are part of the cache.** Each catalogue records how its radii were measured (step, reach, directions,
algorithm version; `pipeline.r_v_settings()`), and the CSV/HDF5 provenance repeats it. A cache made with other settings,
or before they were recorded, has its radii re-measured on first load (about 15 s per 512^3 snapshot) and says so once.

**DTFE_CACHE_DIR.** Set `DTFE_CACHE_DIR=/path` to keep the derived-product caches (void catalogues, slices, limits)
outside python/cache, e.g. off iCloud Drive. Unset, nothing changes.

**DTFE_VOID_GATE.** `DTFE_VOID_GATE=none` drops the ellipsoid fit's max-axis cut (10 sigma) from the void sample, so
the R_v abundance and profiles are defined on R_v and depth alone. It changes only what a run reads, never the shared
cache, and every output made under it carries the tag `gatenone` (no thesis mirror). Default `ellipsoid` = today's sample.

**Void profiles across cosmic time** (`python/plot/plot_void_profile_evolution.py --sim TNG100-3-Dark`, with
`DTFE_SIM` set): one figure of the stacked density rho/rho_bar against r/R_v of the distinct resolved voids at every
snapshot on disk (colour = redshift), their radial velocity in units of the Hubble flow at R_v against the linear-theory
prediction -(1/3) f(z) x Delta(<x), the interior single-stream volume fraction (r < 0.5 R_v and r < R_v, beside the whole
box) and v_r/v_lin at 0.5 and 1 R_v against 1+z. Every R_v bin and redshift is also fitted with the universal profile of
Hamaus, Sutter & Wandelt (2014), delta(r) = delta_c (1 - (r/r_s)^alpha) / (1 + (r/R_v)^beta), with the stack's scatter as
errors: profile_fits.png/.json/.txt (fits with a parameter on its bound are flagged and not drawn). Per-snapshot profiles
are cached under per_snapshot/, so `--replot` redraws with other bins in seconds. `--rv-definition ridge|delta08` uses
the other R_v definitions. plot_void_profiles.py now draws the same fit on each snapshot's density panel and saves the
parameters (hsw_params, hsw_errors) in its npz.

**Void size function.** `plot_void_population.py` also writes `population_size_function.png`: the measured
dn/dlnR_v and n(>R_v) of the distinct well-resolved voids at the panel snapshots against the Sheth & van de Weygaert
(2004) size function in the volume-conserving form of Jennings, Li & Hu (2013) (delta_v = -2.71, i.e. -0.8 nonlinear;
delta_c = 1.686; EH98 sigma(R) at TNG's Planck 2015 sigma_8 and the exact LambdaCDM growth), and their ratio. R_v is
the radius where the enclosed contrast reaches -0.8 (`--size-rv delta08`, the model's radius; `crossing` or `ridge`
give tagged files). The theory lives in `dtfelib/theory.py`. The -0.8 radius of a minimum of the smoothed field
follows the smoothing length (see 'Size function smoothing' below), so this comparison is not yet a test of the
theory.

**Void shapes against a Gaussian field.** `plot_void_crossbox.py` now carries the thesis's BBKS ellipticity e and
prolateness p of the minima for each box at the common sigma (medians with bootstrap 68% intervals), with the BBKS
(1986) expectation for minima of a Gaussian random field with the same smoothed linear spectrum, at the voids' own
heights and over all minima, and the e distribution at each z against it. At sigma = 2.16 Mpc TNG100 and TNG300 start
on the Gaussian expectation at z = 20 (e 0.18-0.19, p 0.015) and end at e 0.23-0.24, p 0.055-0.066 at z = 0.

**Void scripts from analyze.py.** `python3 analyze.py voids` runs the void-population scripts (population, profiles,
profile evolution, environment, fate) for `DTFE_SIM`'s simulation; `--only` picks some; a script not written yet is
skipped with a note.

**Void environment (void-in-void / void-in-cloud).** `python3 python/plot/plot_void_environment.py --sim TNG100-3-Dark`
classifies every distinct well-resolved void at every snapshot by the raw density contrast enclosed within 2 R_v
(dtfelib.classify; Ceccarelli et al. 2013). A void with Delta(<2 R_v) > 0 is void-in-cloud (S-type) and is expected to
shrink; any other void is void-in-void (R-type). The script draws the void-in-cloud fraction against redshift, the R_v,
depth and shape distributions of the two types per snapshot, and, where plot_void_tracking.py has followed voids,
their tracks coloured by type. Results go under figures/void_environment/<sim>/. `--factor` changes the 2 R_v and
`--r-v-definition` picks the R_v. At z = 0, 67% of TNG100-3-Dark's voids are void-in-cloud (60% at z = 1), and they are
smaller than the void-in-void ones.

**Caustic skeleton: random-sphere control.** With `--voids`, plot_caustic_skeleton.py also measures the same wall
metrics in 50 sets of random spheres that carry the voids' own radii (`--null-draws`, `--null-seed`). It reports each
void median against the null's mean +- std as a z-score, in the JSON `null` block, the text table and bands on the
void panel. `--sheet-min-density RHO` keeps only the sheet cells denser than RHO x the mean, and
`--min-component N` / `--min-component-mass F` keep only the sheet components of at least N cells, or that fraction
of the mass, for the void metrics. `--all-caustic` runs every snapshot that has a caustic_class grid.
On TNG100-3-Dark at z = 0 (`--prefix ps_caustic`), the fold fraction of the voids' R_v shells is 0.283 against
0.243 +- 0.011 in the random spheres (z = +3.8).

**Void fate census** (`python3 python/plot/plot_void_fate.py --sim TNG50-3-Dark`). Links the distinct well-resolved
voids of every snapshot to those of the next by their matter: the particles within R_v of a void's centre are found
again by ParticleID in the next combined snapshot (`dtfelib/lagrangian.py`, read in chunks). A void SURVIVES when
at least half of them end in one descendant, MERGES when two or more voids pass half to the same one, SPLITS when two
or more descendants get a quarter each, and VANISHES when no void at the next snapshot receives that much
(thresholds: `lagrangian.SURVIVE_FRAC`, `SPLIT_FRAC`). The descendants are every minimum with a usable radius
(min(R_v, R_eff)), not only the followed sample: a void whose matter ends in a minimum the sample does not follow is
UNRESOLVED, not vanished. How many really vanish depends on how members are counted: with overlapping spheres
(the default) 2-3% per late interval on TNG100-3-Dark, with `--exclusive` members 15% (z = 1 -> 0.5) and 29%
(z = 0.5 -> 0); see below. Writes `figures/void_fate/<SIM>/fate_census.png` (class counts per interval, the fraction of each
snapshot's voids still alive at z = 0, fate against R_v and depth), `fate_table.csv` (one row per void and interval)
and `fate_census.npz`. `--ref-snap` (default 33, z = 2) also links that snapshot straight to z = 0. The shared-particle
tables are cached in python/cache, so a re-run (or new thresholds) takes seconds; the first run reads every combined
snapshot once (TNG100-3-Dark: ~33 min, 31 GB with the whole descendant pool; TNG50-3-Dark, with 1.6e8 particles,
may need ~50 GB: lower `DTFE_FATE_CHUNK` or watch the memory). Like every void script it needs DTFE_SIM=<SIM> (the catalogue frame guard): `DTFE_SIM=TNG50-3-Dark python3
python/plot/plot_void_fate.py --sim TNG50-3-Dark`.

**Void tracking by particles** (`plot_void_tracking.py --link particles --n-voids 30`). Instead of merger-tree
tracers, follows each z = 0 void back through the void that gave it most of its member particles at each earlier
snapshot (the same cached links as the census; distinct resolved voids only). Files get `_particles` in their names;
the per-void figure shows that share as the identity quality. `plot_void_fate.py --exclusive` gives each particle
to one void only (the nearest centre in units of R_v), so overlapping high-z voids cannot both count the same matter;
its files are tagged `_exclusive`. From six voids on, the summary draws the median and
16-84% band of R_v, e and p per snapshot. All tracking figures now use a logarithmic 1 + z axis.

**Fate census: overlapping or partitioned members.** `plot_void_fate.py --exclusive` gives each particle to one void
only. In TNG100-3-Dark it turns most high-redshift mergers into survivals (z = 10 -> 5: 88 merges become 16), and at
late times it turns most 'unresolved' voids into vanished or split ones (z = 0.5 -> 0: 3 vanish and 50 unresolved
become 34 and 13 of 119), because without the overlap no single unfollowed minimum receives half of a dying void's
matter. The fraction of z = 2 voids with a descendant at z = 0 hardly changes (15% vs 17%). The partitioned census
needs about 10 GB instead of 31 GB. `plot_void_tracking.py --link particles --exclusive` follows voids back through
the partitioned links; with overlapping members the walk prefers the largest progenitor spheres (median progenitor
R_v at z = 1: 24 Mpc against 15 Mpc partitioned), while the shape evolution (median BBKS e) is the same either way.

**Void convergence.** `plot_void_convergence.py --sim <SIM>` (needs `DTFE_SIM`) tests whether the void statistics
are properties of the voids or of the measurement, at one physical smoothing length. (a) Grid: the density grid is
block-averaged to 256^3 and 128^3 (`--factors 2 4`; exactly the cell average a coarser run gives from the same
particles), voids are found at the same sigma in Mpc and matched one by one with the native catalogue
(`--match-sigma`, in smoothing lengths): R_v ratio, centre offset, voids unmatched both ways, abundance and the stacked
density. (b) Particle mass: TNG50-3 / TNG100-3 / TNG300-3 side by side from plot_void_crossbox.py's per-box products
(`--crossbox-dir`, `--crossbox-smooth-mpc`). (c) Estimator: DTFE against PS-DTFE void by void, with a per-void table,
for any snapshot that has both grids. A row that cannot be drawn says why. TNG100-3-Dark: the matched voids keep R_v to
0.3% (median; 16-84% within 2-4%) at 256^3 and 128^3, at z = 0 and 1, and the stacked interior is unchanged; only the
height of the compensation ridge falls with the grid. Coarse catalogues are cached as `..._voids_block<f>.npz` under
their own frame's cache namespace.

**Size function smoothing.** `plot_void_population.py` draws the size function from its own, less-smoothed catalogue
(`--size-smooth CELLS`, default 5; 0 = the catalogue of the other figures), written as
`population_size_function_ss<cells>.png` (a non-default value also tags `population_stats` `_ss<cells>`), because at the production 10 cells the delta08 radius lies below sigma. The figure and the log give the
smoothing and the fraction of the distinct voids whose delta08 radius exists. Measured on TNG100: that radius follows
the smoothing at every sigma tried (10, 5, 3, 2 cells: 0.7-1.06 sigma), and measured/Vdn stays at 1e-3-1e-2, so
the -0.8 radius of a minimum of the smoothed field is a smoothing scale, not a void size.

**Cleaning the cache.** `python3 python/analyze.py cache` lists the namespaces of `python/cache` (one folder per
grid size, smoothing, footprint, criterion and cache format) with their sizes and whether today's code can still
reach them. A namespace of an older format (`_v1`) or of a smoothing that no current setting produces (for example
`r512_s21.4043_f23`, TNG50-3-Dark smoothed at 2.16 Mpc, next to the current 2.1625 Mpc's `r512_s21.4291_f23`) is
unreachable, and so are, inside a kept namespace, fate links of an older link version and the temporary files of an
interrupted save. `cache --prune` only lists what would be deleted and its total size; `cache --prune --yes` deletes
it. `--keep-mpc` and `--keep-cells` count further smoothings as current. On 11 October the dry run listed 872 MB,
almost all of it the June `_v1` caches.

**Power-spectrum and profile file names.** `plot_pk_compare.py` adds `_nb<N>` to its file names for a `--nbins`
other than 40 and `_ref<S>` (or `_noref` for `--ref-snap -1`) for another reference snapshot; `plot_void_profiles.py`
adds `_edges<e0>-<e1>-...` for `--edges` other than the default bins. The default names are unchanged, so a run with
other bins no longer overwrites the default figure and npz.

`plot_pk_compare.py` draws:
- the chi^2 cosmic-variance band;
- a panel divided by the same box's early spectrum grown linearly (`--ref-snap`, default snapshot 0), where a single
  realisation's low-k scatter cancels.

**Thesis runs.** `scripts/thesis_runs.sh production|tng50z0|caustics|all` is the recipe that produced the thesis
grids, image planes and run logs: every snapshot of TNG100-3-Dark, TNG50-3-Dark and TNG300-3-Dark at 512^3 with
the T-web and V-web, image planes of 4096^2 (TNG50: 2048^2) pixels, a 52 GB memory budget and the grids'
accumulators in `SCRATCH_DIR` (default `~/DTFE-scratch`), then the TNG50 z = 0 redo at 40 GB and the TNG100 z = 0
caustic run under its own prefix `ps_caustic`, each followed by its figures. `DRY_RUN=1` prints every command and
runs nothing. Run one stage at a time with nothing else heavy beside it. The budget is an upper cap: the
auto-tuner also caps it by the memory free when a run starts, so the partition plan depends on what else is open
(each run log records it).

**One run per snapshot; preflight.** While `run_ps_dtfe.sh` or `run_dtfe.sh` runs a snapshot it holds
`<prefix>.runlog.lock` (a folder with the driver's pid): a second driver on the same snapshot is refused, and a
lock left by a killed driver is removed with a note. Before running, the drivers stop with exit 2 when the data
disk is not mounted, and skip a snapshot (exit 1) whose output folder is not writable or lacks room for its
grids (`DTFE_SPACE_CHECK=0` skips the space estimate).

**Resuming the pipeline.** `run_ps_pipeline.sh` skips a snapshot only when its outputs are newer than its
inputs AND its `.runlog` shows the same settings (grid, sub-samples, deposit options, precision; the grids may
carry more fields than asked for). A changed setting is printed as `was:`/`now:` lines and recomputed. Grids
without a run log are judged by their file times. `FORCE=1` still recomputes everything.

| knob (environment) | meaning | default |
|---|---|---|
| `DTFE_DATA_ROOT`, `DTFE_SIM` | the data root and the simulation | `config.sh` |
| `DTFE_FIGURES_ROOT` | where every figure goes (the plot scripts, `analyze.py`, the launcher) | `/Volumes/Samsung T7/Illustris TNG/figures` while that drive is mounted, else `python/figures` |
| `DTFE_SIGMA_MPC` | the void catalogue's smoothing in Mpc, the same in every box (for the void scripts and `analyze.py export`) | unset: 10 cells |
| `GRID_SIZE` (or `-g`) | cells per axis of the grids | 1024 (`config.sh`; `DTFE_GRID_SIZE` too) |
| `PS_METAL`/`DTFE_METAL` (`-m`) | the deposit / interpolation on the GPU | 0 |
| `PS_EXACT`/`DTFE_EXACT_AVERAGE` (`-e`) | the exact deposit / exact cell averages | 0 |
| `AVG_SUBSAMPLES` (`-n`) | sub-samples per cell of the sampled deposit | 3 |
| `PARTITION`, `MAX_CONCURRENT` | the split of the box and how many parts run at once | auto-tuned |
| `SCRATCH_DIR` | the full grids as files outside RAM (large runs) | off |
| `FIELDS`, `OUTPUT_PREFIX`, `LAMBDA_TH` | the fields, the files' prefix, the web threshold | the production set, `ps_output`, 0.3 |
| `DTFE_PRECISION` | `double` runs the double-precision pair | `single` |
| `DRY_RUN` | 1 = print the commands, run nothing | 0 |
| `DTFE_SPACE_CHECK` | 0 = skip the drivers' free-space check | 1 |
| `ORDERED_MERGE` | 1 = merge the partitions in index order (bit-reproducible) | 0 |
| `PS_RESUME` | 0 = start a killed run over instead of resuming it | 1 |

## Graphical launcher

`scripts/gui.sh` opens a window (PySide6, a prototype) for the things normally typed in a terminal. The **Data** tab gets a simulation onto disk: it shows, for every snapshot of the redshift ladder, whether its particles and group catalogue are there as raw chunks or merged, downloads the snapshots, group catalogues, merger trees and initial conditions from the TNG API (`download_snapshots.sh`, with the API key read from `~/.tng_api_key` and never shown), and merges them into the h-free combined files the rest of the code reads (`merge_HDF5.py`), leaving out whatever is merged already and estimating the download size against the free disk space. For the other tabs the simulation and snapshots are chosen from what is found under the data root, with their redshifts and which have been processed already. 

The **Grids** tab puts together one run of either estimator: the deposit and grid, the fields, the phase-space options, a high-resolution slice and the memory settings. 

The **Pipeline** tab runs `run_ps_pipeline.sh` for the chosen simulations, which computes the grids and a full-box image plane for every snapshot that is missing or out of date, and then renders the point-evaluated figures; it takes the same run options as the Grids tab (the deposit and its sub-samples, the GPU, vertex masses, volume-weighted velocities, caustics and cusps, the parallel triangulation), the plane's geometry (several planes across a slab for a projected view, sub-samples per pixel for pixel-area means, a window of the plane) and the figure options of the render step (fields, projection, smoothing, the fixed density range), and a plane already on disk is reused only when its geometry is the one asked for. 

The **Plots** tab runs the figure scripts of `python/plot` (field slice maps, cosmic-web classes, the point-evaluated maps and publication panels, void population and tracking, halo tracking, and the analysis set of `analyze.py`), each with its own options. 

In every tab the window shows the exact commands these choices stand for, which can be copied and run in a terminal just the same; it flags combinations that would be refused or would not fit in memory (including another DTFE run already going; a query server, the Explore tab's or any other, is not one), and then runs the commands in order with their progress, log and a stop button, either at once or as a job in a **queue**. 

Jobs from any tab can be queued, reordered and removed; the queue runs them one after another, stops at a failed job unless told otherwise, waits before a grid or pipeline job while another DTFE run is using the machine, and is kept when the window is closed. A queued job keeps the settings it was queued with and works out its commands when it starts, so it sees what the jobs before it wrote. 

Before a large run, **Check memory** (Grids, Custom snapshot and Pipeline tabs) asks the code itself, snapshot by snapshot, how much memory the chosen settings will need on this machine and how it will split the work, and warns about every snapshot that would not fit (which would make the computer swap and slow down); for the pipeline it also tries smaller image planes and names the largest that fits, and with **Adaptive** the pipeline then uses, per snapshot, the largest plane that fits. 

The GPU box follows where the deposit pays on this machine: ticked for the exact deposit and from about a million particles, left off for small files, and on for standard DTFE as well (its GPU interpolation is 1.5 to 2.3 times faster than the CPU on 512³ grids, a tie on 256³), with a warning whenever the chosen setting is the slower one. The deposit choice (sampled or exact) applies to standard DTFE as well, where exact means the exact cell average of the interpolated field (`--exact-average`, `run_dtfe.sh -e`), 6.6 times faster on the GPU (4.1 s against 26.9 s at 256³ with 0.26M particles), so picking it ticks the GPU; the parallel triangulation of small runs is a box of its own, off by default because two identical runs then differ at float rounding. 

A **Precision** choice on the Grids, Custom snapshot and Pipeline tabs runs the double-precision pair instead of the default single one (the estimate doubles, and the choice says so when the pair is not built; Setup builds it). 

The figures a run writes are listed and previewed beside the log. 

`scripts/install.sh` installs PySide6 for it in an environment of its own (`~/.venvs/dtfe-gui`, about 1 GB, which sees the numpy and h5py of the main Python), checks that Qt loads, and makes the window launchable like any other application (`scripts/install_gui_app.sh`): on macOS a **DTFE Launcher** app in the Applications folder with a shortcut on the Desktop, on Linux an entry in the applications menu. The app starts the GUI with the same environment a Terminal window has (the Python, `wget` and data root of the login shell). macOS protects iCloud Drive and external drives, and refuses such an app rather than asking, so on a Mac it needs **Full Disk Access** once (System Settings, Privacy & Security); it says so, and opens that page, when it is refused. To set the environment up by hand instead:
```bash
python3 -m venv ~/.venvs/dtfe-gui --system-site-packages
~/.venvs/dtfe-gui/bin/pip install PySide6
```

The first time it opens, the launcher walks through what a new user needs (the **Setup** button brings it back): a folder for the data, whether the programs are built (with a button that builds them and shows the `make` command it runs), the TNG API key for downloads if one is wanted (saved to `~/.tng_api_key`, readable only by its owner), and a **demo**: a small synthetic simulation of crossed Zel'dovich waves, folded into 1, 3, 9 or 27 streams, that is generated and run in a few seconds and then opened in the Explore tab, so a first result appears without downloading anything. Data that is not IllustrisTNG goes through the **Custom snapshot** tab, which runs either estimator directly on any Gadget HDF5 or binary snapshot or text file, with the length unit, the box, whether it is periodic (and, for a non-periodic cloud, the alpha shape that bounds its Lagrangian domain) and where the initial positions are stated rather than assumed, and writes its outputs and run log to a folder of its choice. A snapshot in a plane (its coordinates have two columns) runs on the 2D programs: the tab's **Dimensions** setting follows the file when one is chosen (Gadget HDF5 and text files say which), the box then takes four numbers, and the GPU and the parallel triangulation, which are 3D only, are switched off; Setup builds the 2D programs with a tick of its own (`make DTFE PS-DTFE DIM=2`), and Explore shows a 2D run as its one plane, with the query server, the zoom and the exact zoom working on it as in 3D. Explore knows a run's dimension from the settings it saved, and otherwise from the grid files themselves: vectors and tensors have fewer components in 2D, a phase-space run's stream counts are whole numbers only when read in the right precision, and when only scalar grids remain whose size fits a plane and a cube alike (512² cells are as many as 64³), the reading in which neighbouring cells are alike wins. 

The **Explore** tab shows any run's grids as maps, whatever its size (a slice is read from disk without loading the cube). The output is chosen in two steps, the simulation and then the snapshot by redshift (earliest first); the custom-snapshot runs and any grid files opened by hand have entries of their own in the first menu, listed by name in the second, and a third menu appears only when a snapshot has several outputs, such as a phase-space and a standard run. Then come the field and component, the slice along any axis, the colour scale, logarithmic or linear, and smoothing, with the position and value under the mouse. The vorticity map is the curl of the velocity, the same quantity the zoom and `dtfelib`'s point maps show: the grids store the antisymmetric part of the velocity gradient, half the curl with its components in another order, which the map converts (until 3 October 2026 it showed those stored values under the labels x, y, z, so a zoom read twice the map). It can also start a query server on the snapshot behind the map (the interactive mode above, as a composite of partitions on disk for a large box, in which case it first says how much memory one tessellation would need and how much disk the partitions will take, and asks); a click then lists every stream at that point, with its density, velocity and the position its matter came from, and whether the point lies on a caustic. With the server running, dragging a rectangle on the map zooms into it by re-evaluation: the server evaluates a dense plane of points inside the rectangle (grid-free point evaluation, at a chosen resolution, 512² by default) and the map shows that region at full resolution for the same field and component, with the position and value under the mouse as before; a further drag goes deeper, and Zoom Out (in the Explore menu) returns to the grid slice. This works for the outputs of both estimators (a standard-DTFE server answers with the plain interpolant). Ticking **voids** beside the component outlines the voids on the map: a circle wherever a distinct, well-resolved void's sphere of radius R_v cuts the slice (or the zoomed region), across the periodic box edges too, from the analysis pipeline's own catalogue of that snapshot at its production smoothing, found once per snapshot in a separate process (about a minute, then cached; for the production grids it is the analysis scripts' own cache) while the window stays usable. A catalogue that is not cached yet and would not fit in the free memory (about 80 bytes per cell to build: 10 GB at 512³) is not built, and the status line says so. Ticking **exact** beside the resolution runs the program itself on the dragged region instead, with the exact deposit, as a grid of that many cells across and one slice cell thick: mass-conserving cell averages rather than point values, a whole run (seconds to minutes, on the GPU when available), no server needed. A standard-DTFE run cuts the region from the box (`--regionMpc`); a phase-space run tessellates the whole box exactly as the original run did, periodic or not, and deposits only into the region as a window of a virtual grid whose cell is the region's size over the resolution (`--ps-window`), so the zoom shows what a full run at that resolution would show there, for any snapshot. It reuses the tessellations the query server keeps on disk (the same cache folder and the same split) and skips the partitions that never reach the region, so on a large snapshot it takes about a minute once the server has been started there once. While it runs, the status line shows what it is doing (reading the snapshot, partition k of n, how many were skipped) and a Cancel button stops it; a zoom estimated to take more than a minute asks first and says why (how many partitions reach the region, how many are cached, how many would have to be built). Every zoom, point-evaluated or exact, shows a progress bar and a running time while it works and its duration when it is done; on a large snapshot's server the bar follows the partitions the request visits, and the status line says how many were in memory and how many had to be loaded from disk. **Back** returns to the previous zoom level from memory, and dragging a region evaluated a moment ago brings it back at once (the last zooms are kept, up to 1.5 GB). 

The **Runs** tab beside the log lists every run found on disk, from its run log: the settings, when it finished, how long it took, its peak memory and the size of its outputs. It marks outputs made before a fix that changed them (for instance grids from GPU runs split into partitions before 29 September 2026, or stream counts made before the fix of 28 September), judged by the build of the program that made them, which every run now records in its log (`Build:`, also printed by `--version`). A run's settings can be loaded back into the Grids tab to repeat it, and its outputs opened in Explore or moved to the Trash. The time a run will take is estimated from comparable earlier runs; the options can be set from **presets** (a quick look, the standard production settings, figure quality, the exact deposit) or saved as new ones, and their descriptions, shown on hovering, are the program's own help text. When a job fails, the window says what probably went wrong (not enough memory, the data disk missing, an old build, a refused iCloud path) with a button that fixes it where it can; a notification says when a long job or the queue has finished. The commands behind a job can also be saved as a bash script, or as a SLURM batch job to run the same thing on a cluster.


**Voids on the Explore map.** Tick *voids* to outline the distinct, well-resolved voids of the snapshot shown (the
analysis pipeline's catalogue, cached). Point at an outline to read its R_v, R_eff and central density contrast;
click it for the whole entry (every R_v definition the catalogue has, R_eff, central δ, the resolved and distinct
flags, the centre) and *Go to its centre* for the slice through it. *R_eff too* adds the effective radius as dashed
circles. The σ box chooses the smoothing the voids are found at: *production* (the analysis scripts' 10 cells) or a
common physical σ in Mpc for every box, each its own cached catalogue. *Save image* writes the outlines into the PNG
and a CSV of the voids shown beside it (`<name>_voids.csv`: centre, R_v, R_eff, δ, flags); the figure grid, the
redshift series included, outlines each panel with its own snapshot's voids. `python/tools/void_outlines.py ...
--sigma-mpc S --csv voids.csv` gives the same table for a whole snapshot. While a DTFE run is going, Explore asks
before a query server, an exact zoom or a catalogue build, saying the run's memory and what is free.

**Memory strip and Running now.** The launcher's status bar shows the free memory, swap in use and every DTFE /
PS-DTFE program running on the Mac with its resident memory, refreshed every five seconds. The 'Running now' tab
lists every run going, also those started from a terminal or with nohup: simulation, snapshot, phase, partitions
done, ETA and memory from the binary's progress file beside its outputs, and a Stop button. A run whose process is
gone but left its progress file reads 'died'. A queue waiting for another run shows that run's ETA.

**Void catalogues in the Runs tab.** The Runs tab's Voids column says whether each run's void catalogue is cached,
stale (the grids were rewritten after it was built) or absent. 'Build void catalogue' queues a job that builds it
into the analysis cache (about a minute and 10 GB at 512 cubed; refused when the memory is not free); the same job
is the Plots tab's 'Void catalogue build' set.

**Void figures in the Plots tab.** Besides the existing void sets, the Plots tab runs the cross-box comparison
(plot_void_crossbox.py), the void profile evolution, environment and fate figures, and void tracking with the radius
and smoothing in Mpc.

**Keep-awake.** Custom-snapshot runs, analysis and plot steps and downloads started from the launcher now hold a
power assertion (caffeinate) for as long as they run, as the run scripts already did.

**Ordered merge.** The Grids, Pipeline and Custom tabs have a tick "Merge the partitions in index order"
under Resources. A partitioned PS-DTFE run then adds the partitions' grids in partition order whatever order they
finish in, so two identical runs give bit-identical grids (Grids and Pipeline jobs set ORDERED_MERGE=1 for
run_ps_dtfe.sh, the Custom tab passes the program's ordered-merge option). It can cost time when the partitions
differ much in cost. The Runs tab shows "ordered merge" in the Options column, and Load settings restores it.

**Resumable runs.** A partitioned PS-DTFE run that was killed leaves a checkpoint folder
'<prefix>.runlog.checkpoint' beside its outputs. The Runs tab shows such a run as "resumable"; running the same
job again continues from the partitions that had finished (run_ps_dtfe.sh resumes by itself when the folder is
there). The checkpoint is never listed as an output.

**Custom runs in the memory strip.** A run started from the Custom tab now shows in the memory strip, the Running
now panel and the "already running" warnings, like a run started from a terminal; only the Explore tab's own
exact zoom is left out. While it runs, the other tabs therefore say a run is going (it is: it occupies the
machine).

**Exported scripts keep the run log safe.** An exported job script (bash or SLURM) writes each run's log as
'<prefix>.runlog.partial' and puts it in place when the step ends: '<prefix>.runlog' on success,
'<prefix>.runlog.failed' on failure, so a failed re-run never overwrites the log of the grids already on disk
(the same rule the launcher and the run scripts follow).

**Void convergence figure.** The Plots tab lists "Void convergence" (plot_void_convergence.py): the resolution
and estimator convergence of the void statistics for the selected simulation.

## Contributors
* **Marius Cautun (Kapteyn Astronomical Institute, Durham University, Leiden University)** - *code and documentation.*
* **Rien van de Weygaert (Kapteyn Astronomical Institute)** - *various discussions about the method and implementation.*
* **Luuk Westerhoek (Kapteyn Astronomical Institute, University of Groningen)** - *phase-space DTFE, the exact / point evaluation methods, automatic memory management, the GPU backends (Metal, and its CUDA/HIP port) and the analysis pipeline (bachelor research project "Fate of Cosmic Voids").*


## License

This project is licensed under GNU GENERAL PUBLIC LICENSE Version 3 - see the [LICENSE.md](LICENSE.md) file for details.
