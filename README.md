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
   brew install gsl boost cgal mpfr hdf5 gmp
   ```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install build-essential
sudo apt-get install libgsl-dev libboost-all-dev libcgal-dev libmpfr-dev libhdf5-dev libgmp-dev
```

#### Linux (Fedora/RHEL/CentOS)
```bash
# For Fedora
sudo dnf groupinstall "Development Tools"
sudo dnf install gsl-devel boost-devel CGAL-devel mpfr-devel hdf5-devel gmp-devel

# For older RHEL/CentOS
sudo yum groupinstall "Development Tools"
sudo yum install gsl-devel boost-devel CGAL-devel mpfr-devel hdf5-devel gmp-devel
```

#### Linux (Arch/Manjaro)
```bash
sudo pacman -S base-devel gsl boost cgal mpfr hdf5 gmp
```

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd DTFE
   ```

2. **Test platform detection:**
   ```bash
   make test-platform
   ```

3. **Build the main executable:**
   ```bash
   make DTFE
   ```

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
make PS-DTFE
```
This produces a `PS-DTFE` executable (object files are placed in `o_ps/` to avoid clashing with the standard `o/` build). It needs the same dependencies as `make DTFE`, plus **HDF5** — PS-DTFE input is HDF5-only (see below). The `-DPHASE_SPACE` flag is set automatically by this target; do not add it to the standard `DTFE` build.

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
| _(always)_ | `.streams` | 1 | number of streams per cell (1 single-stream, ≥3 in a caustic) |

Every field also has a **volume-averaged `_a` form** (`density_a`, `velocity_a`, `dispersion_a`, …) that sub-samples an `nSub³` grid (nSub=3) inside each cell and writes the matching `.a_*` file (`.a_den`, `.a_velDisp`, …); `.a_streams` is the per-cell average stream count. The dispersion is σ_ij = Σ ρ_s (v_s−⟨v⟩)_i (v_s−⟨v⟩)_j / Σ ρ_s = ⟨v_i v_j⟩ − ⟨v_i⟩⟨v_j⟩ — ≈0 in cold single-stream void interiors, large in multi-stream walls/filaments. In single-stream regions the velocity, gradient and dispersion reproduce the standard DTFE result (dispersion → 0), validated by [`tests/ps_standard_cross_check.py`](tests/ps_standard_cross_check.py).

### Parallelization, scaling, and memory
PS-DTFE parallelizes **coarse-grained over Lagrangian partitions**, selected with `--partition nx ny nz`. Each OpenMP thread builds an independent Delaunay triangulation for one partition, interpolates it onto the full Eulerian grid, and the per-partition grids are summed into the shared output under a critical section (the per-cell scatter inside a partition is serial). Speedup therefore comes from `--partition`, and parallelism scales up to the number of partitions:
```bash
./PS-DTFE snapshot.hdf5 out --grid 256 --periodic --partition 2 2 2 --field density velocity --MpcUnit 1
```
Strong scaling measured on a 10-core Apple Silicon laptop (`N=48³`, `grid=96³`, `--partition 2 2 2`, `--field density velocity density_a`):

| cores | wall time | speedup | efficiency |
|------:|----------:|--------:|-----------:|
| 1     | 24.0 s    | 1.00×   | 100 %      |
| 2     | 12.3 s    | 1.95×   | 98 %       |
| 4     | 6.37 s    | 3.78×   | 94 %       |
| 8     | 3.63 s    | 6.63×   | 83 %       |

Reproduce (and regression-check the scaling) with [`tests/ps_scaling_benchmark.sh`](tests/ps_scaling_benchmark.sh).

> **Experimental: TBB-parallel triangulation.** `make PS-DTFE TBB=1` (needs `brew install tbb`) builds the *single global* tessellation — the no-`--partition` path — with CGAL's parallel Delaunay insertion. It is correct and genuinely multi-threaded, but on these point sets it benchmarks **~2× slower** than the default build: the parallel path cannot use CGAL's `Fast_location` (a sequential point-location hierarchy) and its lock-grid overhead outweighs the parallelism on the grid-structured + periodic-copy distribution (CGAL parallel Delaunay is tuned for much larger, uniform point sets). **Prefer `--partition`** — it parallelizes the triangulation *and* the interpolation. The `TBB=1` switch is provided for experimentation; the default builds are unaffected.

> **Memory.** Each concurrently-running Lagrangian partition holds its own output grid, so peak memory ≈ (threads running at once) × per-partition grid × number of fields — bounded by the **thread** count, not the total partition count (`OMP_NUM_THREADS` caps it). To keep that affordable, each partition allocates only the **Eulerian bounding box of the cells it actually touches**, not the whole grid: on the pancake test a half-box Lagrangian partition occupies ~15 % of the grid, cutting measured peak RSS ~2.6× (4.16 → 1.58 GB at grid 192³, `--partition 2 2 2`, 8 threads) with bit-identical output. A partition whose Eulerian footprint folds or wraps across an axis keeps the full extent on that axis. Still, choose the partition/thread count from your core budget *and* available RAM.

### Recommended settings and limitations
**Grid resolution.** Choose `--grid` comparable to the particle resolution (≈ the cube root of the particle number). The unaveraged density is point-sampled at cell centres, so on caustic-rich fields the mass it integrates to depends on how the grid aligns with the (near-singular) caustics — on the Zel'dovich pancake the grid mass varies by ~±10 % between grid 64 / 96 / 128, and the measured multi-stream *fraction* is set by the grid (how many cells fall in the slab), not the particle count. The volume-averaged `density_a` (sub-sampled) is steadier and is the better choice when mass conservation matters; for a smooth field both converge.

**Performance / averaging cost.** Runtime is dominated by the per-cell scatter, which grows with the number of grid points each Delaunay cell covers — i.e. with `--grid`. The volume-averaged `_a` fields sub-sample an `nSub³` grid inside every cell (`nSub=3`, so 27× the per-cell work), so request `_a` fields only when you need the volume average. Use `--partition` for parallelism (it scales with the partition count); keep the grid-to-particle ratio near 1 to avoid both empty cells (grid too fine) and washed-out caustics (grid too coarse).

**Periodicity.** Use `--periodic` for periodic cosmological boxes. A genuinely **non-periodic finite cloud** (particles in vacuum) is also supported and mass-conserving: the density is normalized to the cloud's *Lagrangian* mean (`N·m/V_lagBox`, not the box volume) and the convex-hull surface cells are kept (density from their own Lagrangian/Eulerian volume ratio). On a centred clump this conserves mass to ~0.96–1.02 across grids (`tests/ps_nonperiodic_test.py`); before it recovered only ~20%. Two caveats: it assumes a roughly **convex, regular** Lagrangian particle distribution (the bounding box stands in for the convex hull, exact for a grid; hull cells across a concavity would bridge empty space — the standard DTFE convex-hull assumption), and you must not feed *periodic* data without `--periodic` (the wrapped positions then mishandle the boundary).

**Dimensionality.** Validated in 3D (`NO_DIM=3`). The interpolation is written dimension-generically (it carries 2D branches) but 2D has not been exercised — treat 2D as untested.

**Unsupported options.** Phase-space mode supports **regular-grid** interpolation only; redshift-cone grids and user-defined sampling points exit with an explanatory message.

**Multi-stream semantics.** velocity, scalar and their gradients are the **mass-weighted mean over streams**, ⟨f⟩ = Σ ρ_s f_s / Σ ρ_s, which reduces to the single-stream value where `streams = 1`. Quantities derived from a mass-weighted multi-stream velocity gradient (divergence, shear, vorticity) are rigorous only in single-stream regions; for multi-stream kinematics use the stream count and the velocity dispersion.

### Verifying the build (tests)
Five self-contained tests exercise PS-DTFE on a synthetic Zel'dovich-pancake snapshot with genuine shell crossing. Each builds `PS-DTFE` (pass `--no-build` to reuse an existing build), generates the snapshot via [`tests/generate_ps_test_data.py`](tests/generate_ps_test_data.py), runs the executable, and checks the outputs; they require `python3` with `numpy` and `h5py`.

* **Smoke test** — [`tests/ps_smoke_test.sh`](tests/ps_smoke_test.sh): the build runs and the density/stream outputs are finite, non-negative, roughly mass-conserving and multi-stream.
* **Regression test** — [`tests/ps_regression_test.py`](tests/ps_regression_test.py): compares the output against the *analytic* 1-D pancake — three-stream volume fraction, Eulerian caustic positions, stream-count parity, hard mass conservation, and the **velocity dispersion** against the analytic three-stream value — and tracks behavioural metrics against a baseline in `tests/reference/` (`--update-baseline` to rewrite it).
* **Parallel check** — [`tests/ps_parallel_check.sh`](tests/ps_parallel_check.sh): runs the same `--partition` problem at 1 thread and all cores and verifies the outputs agree (integer stream counts exactly; float fields within rounding), catching parallel data races.
* **Scaling benchmark** — [`tests/ps_scaling_benchmark.sh`](tests/ps_scaling_benchmark.sh): a 1/2/4/8-thread speedup table that also fails if strong scaling regresses.
* **Standard-DTFE cross-check** — [`tests/ps_standard_cross_check.py`](tests/ps_standard_cross_check.py): in single-stream regions PS-DTFE must agree with standard DTFE — velocity matches tightly, density to estimator accuracy.

```bash
tests/ps_regression_test.py            # the strictest physics check
tests/ps_parallel_check.sh --no-build  # parallel == serial
```


## Contributors
* **Marius Cautun (Kapteyn Astronomical Institute, Durham University, Leiden University)** - *code and documentation.*
* **Rien van de Weygaert (Kapteyn Astronomical Institute)** - *various discussions about the method and implementation.*


## License

This project is licensed under GNU GENERAL PUBLIC LICENSE Version 3 - see the [LICENSE.md](LICENSE.md) file for details.