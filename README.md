# The DTFE public software

The DTFE public code is a C++ implementation of the **Delaunay Tessellation Field Interpolation (DTFE)** method. Its purpose is to interpolate quantities stored at the location of an unstructured set of points to a regular grid using the maximum of information contained in the input points set. In particular, the code can calculate the following cosmological quantities:
* the density field - this is calculated directly from the point distribution,
* the velocity field and derivatives (e.g. gradient, divergence, vorticity) - uses the velocity at each particle position, and
* general vector quantities and their derivatives - these quantities must be given as input for each point in the set.

The code was written with the purpose of analysing cosmological simulations and galaxy redshift survey. Even though the code was designed with astrophysics in mind, it can be used for problems in a wide range of fields where one needs to interpolate from a discrete set of points to a grid.

The code was designed using a modular philosophy and with a wide set of features that can easily be selected using the different program options. The DTFE code is also written using OpenMP directives which allow it to run in parallel on shared-memory architectures.

The code comes with a complete [documentation](docs/DTFE_user_guide.pdf) and with a multitude of examples that detail the program features. A test dataset and analysis of the code output is given in the [demo directory](demo): `./DTFE demo/z0_64.gadget out --grid 64 --field density` works out of the box.

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
* Exact, sampling-free field estimation as an alternative to Monte-Carlo cell averaging, and evaluation of the fields at arbitrary user-supplied points.
* Automatic choice of the domain decomposition and memory footprint from the data and the machine, with an optional out-of-core mode for grids larger than the available memory.
* Optional GPU acceleration of the field deposit on Apple Silicon (Metal).
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
   On Apple Silicon, add `METAL=1` (e.g. `make DTFE METAL=1`) to run the volume-averaged interpolation / phase-space deposit on the GPU, with automatic fall-back to the CPU. As a shortcut, `./scripts/install.sh` installs any missing dependencies and builds both binaries with the best backend for the machine.

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


## Phase-Space DTFE

Standard DTFE builds the Delaunay tessellation in Eulerian (present-day) space and returns a single-valued density at each point. **Phase-Space DTFE** builds the tessellation in Lagrangian (initial-condition) space and follows it as it is deformed to the present day. Because the deformed tessellation can fold over onto itself, the method recovers the *multi-stream* structure of the cosmic web: the density at a point is the sum over every stream (folded simplex) that covers it, and the number of streams is returned as a field of its own. This makes it especially suited to caustics and to the multi-stream interiors of voids, where the single-stream estimate breaks down. The method follows the phase-space tessellation approach of Abel, Hahn & Kaehler (2012) and Shandarin, Habib & Heitmann (2012).

Phase-Space DTFE is built as a separate `PS-DTFE` executable (the `-DPHASE_SPACE` compile option, set automatically by the target so that it can coexist with the standard build):
```bash
make PS-DTFE            # add METAL=1 on Apple Silicon to run the deposit on the GPU
```

It needs **two positions per particle** — the present-day (Eulerian) position and the initial-condition (Lagrangian) position — read from a Gadget-HDF5 file, either from an `InitialCoordinates` dataset in the same file or from a separate initial-conditions snapshot given with `--lagrangianInput`:
```bash
./PS-DTFE snapshot.hdf5 output_root --grid 256 --periodic --field density --lagrangianInput ics.hdf5
```

In addition to the standard fields, `PS-DTFE` returns the number of streams per cell, the velocity dispersion (its trace and the full symmetric tensor), and, optionally, a caustic flag that marks the fold surfaces of the cosmic web (`--ps-caustics`). Two refinements of the mass assignment keep the estimate accurate on evolved initial conditions: chart-independent tetrahedron masses (`--ps-vertex-mass`, which distributes each particle's mass over its incident tetrahedra) and volume-weighted velocity moments (`--ps-volume-weighted`, the volume-weighted convention used by standard DTFE). Inside virialised haloes, where the stream count is no longer converged, the thoroughly mixed sheet can be released from the deposit with `--ps-halo-release` following Stücker et al. (2021).


## Field evaluation methods

The value assigned to a grid cell can be either the field at the cell centre or the field averaged over the cell volume; the volume average is normally estimated by Monte-Carlo sampling (`--method`, `--samples`). Two alternatives remove the sampling noise entirely:

* **Exact volume averaging** (`--exact-average`, standard DTFE) integrates the linear DTFE interpolant analytically over every grid-cell/tetrahedron intersection using the vendored [r3d](third_party/r3d/README.md) library (Powell & Abel 2015). For a linear field the integral over each intersection is its centroid value times its volume, so the averaged field carries no sampling noise at any resolution.
* **Exact conservative deposit** (`--ps-exact-deposit`, PS-DTFE) does the same for the phase-space deposit: each stream is clipped analytically against the grid and deposits its mass exactly, with the velocity and dispersion following from the linear profile over each intersection. An intermediate option, `--ps-linear-deposit`, keeps the sampled deposit but weights each sub-sample by the linear density profile inside its stream.

Besides the regular grid, **both binaries** can evaluate their fields at **arbitrary points** supplied by the user (`--sample-points`) — halo or void centres, sight-lines, or the pixels of a high-resolution image plane. The points are read from a text file (one `x y z` per line) or a raw-binary file, and one record is written per point. For `PS-DTFE` this returns the full per-stream decomposition at each point (`--per-stream`); for the standard binary it is the ordinary single-valued DTFE interpolant.


## Automatic memory management

Large runs are split into smaller computational chunks (`--partition`) that are processed a few at a time (`--max-concurrent`), so that the peak memory stays within the machine. When these options are not supplied the code **chooses them automatically** from the particle count, grid size, requested fields and the machine's memory and core count, printing the chosen decomposition and the predicted peak memory before it starts; any explicit option always takes precedence. When even a single chunk of the full-resolution grids would not fit in memory, the grids can be backed by memory-mapped files on a local disk with `--scratch-dir`, trading disk space for RAM while leaving the results unchanged.


## Contributors
* **Marius Cautun (Kapteyn Astronomical Institute, Durham University, Leiden University)** - *code and documentation.*
* **Rien van de Weygaert (Kapteyn Astronomical Institute)** - *various discussions about the method and implementation.*
* **Luuk Westerhoek (Kapteyn Astronomical Institute, University of Groningen)** - *phase-space DTFE, the exact / point evaluation methods, automatic memory management, the Metal GPU backend and the analysis pipeline (bachelor research project "Fate of Cosmic Voids").*


## License

This project is licensed under GNU GENERAL PUBLIC LICENSE Version 3 - see the [LICENSE.md](LICENSE.md) file for details.
