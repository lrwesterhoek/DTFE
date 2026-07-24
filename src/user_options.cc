/*
 *  Copyright (c) 2011       Marius Cautun
 *
 *                           Kapteyn Astronomical Institute
 *                           University of Groningen, the Netherlands
 *
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

/* User_options implementation: defaults, command-line/config parsing, validation, and the run-configuration printout. */


#include <fstream>
#include <cstdio>
#include <sys/stat.h>
#include "user_options.h"

using namespace std;


#ifndef INPUT_FILE_DEFAULT
#define INPUT_FILE_DEFAULT 101
#endif
#ifndef OUTPUT_FILE_DEFAULT
#define OUTPUT_FILE_DEFAULT 101
#endif
#ifndef MPC_UNIT
#define MPC_UNIT 1.
#endif

// Initializes every option to its default before the command line/config overrides it.
User_options::User_options()
{
    periodic = false;
    boxCoordinates.assign(0.);
    userGivenBoxCoordinates = false;
    inputFileType = INPUT_FILE_DEFAULT;     // default input file type (see Makefile)
    outputFileType = OUTPUT_FILE_DEFAULT;   // default output file type (see Makefile)
    readParticleData.assign( 10, 0 );
    readParticleSpecies.assign( 10, 0 );
    readParticleData[0] = 1; readParticleData[1] = 1; readParticleData[2] = 1;
    readParticleSpecies[1] = 1;
    
    regionOn = false;
    regionMpcOn = false;
    
    partitionOn = false;
    partNo = -1;
    maxConcurrent = 0;    // 0 = build as many concurrent triangulations as threads (no memory cap)
    maxConcurrentOn = false;  // when neither --partition nor --max-concurrent is given, the auto-tuner picks them (auto_tune.h)

    paddingOn = false;
    paddingMpcOn = false;
    paddingParticles = Real(5.);
    testPaddedBoundaries = true;
    
    // density options
    method = 1;
    noPoints = 100;
    noPointsOn = false;
    useMetal = false;     // GPU interpolation off unless --gpu is given (requires a Metal build)
    gpuAlias = false;
    exactAverage = false; // sampled '_a' averaging unless --exact-average (standard DTFE only)
    psSamplePointsFile = "";          // point-evaluation mode off unless --sample-points is given (both binaries)
    psPerStream = false;
    psPerStreamIds = false;
    psPtsDenGrad = false;
    psPtsVelGrad = false;
    psStreamDensityGeometric = false; // default per-stream estimator: 'dtfe'
#ifdef PHASE_SPACE
    psAvgSubsamples = 3;  // nSub for PS-DTFE '_a' fields (27 sub-points in 3D)
    psUseMetal = false;   // GPU deposit off unless --ps-gpu is given (requires a Metal build)
    psLinearDeposit = false;          // uniform (equal-share) deposit unless --ps-linear-deposit
    psCaustics = false;               // no '.caustic' fold-flag grid unless --ps-caustics
    psHaloRelease = Real(0.);         // 0 = deposit every tetrahedron by bbox rasterization (no release)
    psExactDeposit = false;           // nSub^3 sub-sampled deposit unless --ps-exact-deposit
    psVertexMass = false;             // rho_bar * V_lag tet masses unless --ps-vertex-mass
    psVolumeWeighted = false;         // mass-weighted (momentum-like) velocity moments unless --ps-volume-weighted
#endif
    averageDensity = Real(-1.);

    redshiftConeOn = false;

    // additional options
    DTFE = true;
    NGP = false;
    CIC = false;
    TSC = false;
    PCS = false;
    SPH = false;
    Voronoi = false;
    interlace = false;
    MpcValue = Real(MPC_UNIT);
    extensive = false;
    approxPSD = false;
    lambda_th = Real(0.);
    hubbleParam = Real(-1.);
    scaleFactor = Real(-1.);
    verboseLevel = 3;
    randomSample = Real(-1.);
    poisson = 0;
    
    
    // internal variables
    paddedBox.assign(0.);
#ifdef PHASE_SPACE
    // these three are only set true for per-partition sub-calls in the PS-DTFE partition path
    psSuppressGridStats = false;
    psDeferNormalization = false;
    psUseSubgrid = false;
    psMayClearDT = false;   // set only by the DTFE_interpolation overload that owns its DT
#endif
    userDefinedSampling = false;
    noProcessors = 1;
    totalTime = Real(0.);
    programOptions = "";
}



// Register the available user options. allOptions = every option; visibleOptions = those shown in help.
void User_options::addOptions(po::options_description &allOptions,
                              po::options_description &visibleOptions,
                              po::positional_options_description &p)
{
    po::options_description mainOptions("Main options");
    mainOptions.add_options()
            ("help,h", "produce summary of help message. For more detailed help use the '--full_help' option.")
            ("full_help", "produce detailed help message. For more detailed help information consult the documentation.")
            ("grid,g", po::value< std::vector<size_t> >(&(this->gridSize))->multitoken(), "choose grid size along each direction (e.g. '-g 256' for a 256^3 grid; '-g 256 128 512' for different gridsize along each direction).")
            ("box", po::value< std::vector<Real> >()->multitoken(), "the coordinates of the box encompasing all the particles. It needs 6 arguments for 3D (4 for 2D) which give 'x_left', 'x_right', 'y_left', 'y_right', etc... (where 'x_left' is the left box coordinates along x-direction). For example '--box 0 1 0.5 1.5 0 10'.")
            ("input,i", po::value< std::vector<int> >()->multitoken(), "give the type of the input file (101=gadget multiple file, 102=gadget single file, 105=gadget HDF5 file, see documenation for more options). If present, a 2nd argument gives the data to be read from file (1=positions, 2=weights, 4=masses, ..., 2^n=the n+1 data) - e.g. to read positions, masses and velocities insert 1+2+4=7. If present, a 3rd argument gives the particle species to be read from file (1=1st species, 2=2nd species, ..., 2^n=the n+1 species) - e.g. to read the data of species 2,3 and 4 insert 2+4+8=14.")
            ("output,o", po::value< int >(&(this->outputFileType)), "give the type of the output file (101=binary file, 111=text file, see documenation for more options).")
            ("periodic,p", "particle data is in a periodic box with box coordinates given by option '--box' or read from input file.")
            ;
    
    
    po::options_description fieldOptions("Field choices");
    fieldOptions.add_options()
            ("field,f", po::value<std::vector<std::string> >()->multitoken(), "specify which field is to be interpolated to grid - e.g. '-f density'. You can specify multiple fields at a time. You can choose to output the field value at the sampling point or the volume averaged field value inside the grid cell associated to the sampling point (field names end with '_a' with 'a' standing for averaged). Available options are:\n"
                    "  density = \tcompute the density at the sampling point position.\n"
                    "  density_a = \tcompute the volume averaged density inside the sampling cell associated to the sampling point [DEFAULT choice if none provided]. Each of the options below have also a '*_a' version which will be left out to minimize the help messages.\n"
#ifdef VELOCITY
                    "  velocity = \tcompute the velocity at the sampling point position (use 'velocity_a' to get the averaged velocity inside the sampling cell).\n"
                    "  gradient = \tcompute the velocity gradient at the sampling point position (use 'gradient_a' to get the averaged velocity gradient inside the sampling cell).\n"
                    "  divergence = \tcompute velocity divergence at the sampling point position (use 'divergence_a' to get the averaged velocity divergence inside the sampling cell).\n"
                    "  shear = \tcompute velocity shear at the sampling point position (use 'shear_a' to get the averaged velocity shear inside the sampling cell).\n"
                    "  vorticity = \tcompute velocity vorticity at the sampling point position (use 'vorticity_a' to get the averaged velocity vortivity inside the sampling cell).\n"
                    "  velocityStd_a = \tcompute velocity standard deviation inside the sampling cell (NOTE: there is no 'velocityStd' of this option and this option works only with averaging method 2 '--method 2').\n"
#ifdef PHASE_SPACE
                    "  dispersion = \t[PS-DTFE only] mass-weighted multi-stream velocity dispersion. Writes the trace sigma^2 = sum(rho_s |v_s-<v>|^2)/sum(rho_s) to '.velDisp' and the full symmetric tensor sigma_ij to '.velDispTensor'. ~0 in single-stream regions (cold void interiors), large in multi-stream regions (walls/filaments). Use 'dispersion_a' for the volume-averaged version.\n"
#endif
#endif
#ifdef SCALAR
                    "  scalar = \tcompute scalar quantities at the sampling point position (use 'scalar_a' to get the averaged field components inside the sampling cell).\n"
                    "  scalarGradient = \tcompute the gradient of the scalar quantities at the sampling point position (use 'scalarGradient_a' to get the averaged field gradient inside the sampling cell).\n"
#endif
#ifdef VELOCITY
                    "  tweb = \tT-web cosmic web classification (0=void, 1=wall, 2=filament, 3=node) from the eigenvalues of the tidal tensor, FFT-Poisson-solved from the DENSITY grid (use 'tweb_a' for the volume-averaged density). Also outputs the eigenvalues. Requires the density field (enabled automatically), a periodic box, and 3D. Computed from the RAW density grid -- any smoothing is a plot-time choice in python (plot_PS_DTFE.py). See also --lambda_th.\n"
                    "  vweb = \tV-web cosmic web classification (0=void, 1=wall, 2=filament, 3=node) from velocity shear tensor eigenvalues (use 'vweb_a' for volume averaged). Also outputs eigenvalues.\n"
#endif
            );


    po::options_description regionOptions("Region options");
    regionOptions.add_options()
            ("region", po::value< std::vector<Real> >()->multitoken(), "choose this option to compute the field interpolation only in a given part of the full box. Use this option to specify the region of interest in terms of fractions of box length (i.e. '--region 0.4 0.6 0.3 0.7 0.45 0.55' computes the density for the box that extends from 0.4 to 0.6 of the box length along direction x, 0.3 to 0.7 along direction y and 0.45 to 0.55 along direction z).")
            ("regionMpc", po::value< std::vector<Real> >()->multitoken(), "choose this option to compute the field interpolation only in a given part of the box. Specify the limits of that region in Mpc units; see previous option for additional information.")
            ;
    
    
    po::options_description partitionOptions("Partition options");
    partitionOptions.add_options()
            ("partition", po::value< std::vector<size_t> >(&(this->partition))->multitoken(), "choose this option if the particle data is too large to compute the Delaunay triangulation for the full data at once. Specify here in how many parts to split the box along each direction (e.g. '--partition 3 3 3' splits the data in 27 chuncks). If NOT given, the program AUTO-SELECTS the split from the particle count, grid, requested fields and the machine's RAM/cores once the input is read (DTFE_RAM_GB env overrides the detected RAM)."
#ifdef PHASE_SPACE
             " In PS-DTFE this also drives the OpenMP parallelism: partitions run in parallel (one triangulation per thread), so speedup scales up to the number of partitions. NOTE on memory: each Lagrangian partition writes the WHOLE Eulerian grid, so peak memory ~ (partitions running concurrently) x full grid x number of fields -- choose the partition count from your core budget AND available RAM."
#endif
             )
            ("partNo", po::value<int>(&(this->partNo)), "choose to compute the density only for this partition number (from 0 to 'maximum partitions'-1). This options is usefull if you would like to compute the density for a large particle number on several different machines at the same time - need to run the program on each machine independently.")
            ("max-concurrent", po::value<int>(&(this->maxConcurrent))->default_value(0), "caps how many Delaunay triangulations are built AT ONCE, to bound peak memory on machines with less RAM (trades CPU cores for memory). Each concurrently-built triangulation is the dominant memory cost, so peak RAM ~ (concurrent triangulations) x per-triangulation size. '0' uses all available threads. If NOT given, the program AUTO-SELECTS the cap together with the partition split (see '--partition')."
#ifdef PHASE_SPACE
             " In PS-DTFE this limits how many Lagrangian partitions run in parallel (e.g. '--partition 4 4 4 --max-concurrent 4' builds at most 4 of the 64 partition triangulations simultaneously). Combine a FINER --partition (smaller per-triangulation size) with a SMALLER --max-concurrent to fit large grids in limited RAM."
#else
             " In standard DTFE this limits the concurrent spatial sub-triangulations inside the parallel interpolation; combine with --partition for serial, low-memory processing of very large data."
#endif
             )
            ;
    
    
    po::options_description paddingOptions("Padding options");
    paddingOptions.add_options()
            ("padding", po::value< std::vector<Real> >()->multitoken(), "give the size of the padding need to make sure that the Delaunay triangulation fully covers the region of interest. There are two ways to give the padding size:\n"
                    "  1) \t by giving one value which is the average number of particles that will be copied along each face of the region of interest. The actual computation uses all the particle that are within 'padding number' * 'particle grid spacing' distance from the region of interest. For example '--padding 5' will add an average of 5 particles on both the left and right sides for each dimension.\n"
                    "  2) \t by giving the size of the padding for each face of the box of interest. This size is given with respect to the box length along each coordinate (i.e. '-padding 0.1 0.2 0.5 0.5 0.1 0.1' means that box will be padded with '0.1*x box length' on the left of the x-coordiante and by '0.2*x box length' on the right of the x-coordinate, similar for the y and z dimensions).")
            ("paddingMpc", po::value< std::vector<Real> >()->multitoken(), "give the size of the padding need to make sure that the Delaunay triangulation fully covers the region of interest. Similar to option 'padding' choice '2)' with the difference that the padding size is given in Mpc and not box lengths.")
#ifdef TEST_PADDING
            ("noTest", "do not test for the efficiency of the padding when computing the Delaunay tesselation. The default is to use dummy particles positioned at the boundary of the extended padded box to test if the Delaunay tesselation fully covers the region of interest (i.e. the unpadded box).")
#endif
            ;
    
    
    po::options_description averagingOptions("Averaging options");
    averagingOptions.add_options()
            ("method,m", po::value<int>(&(this->method))->default_value(this->method), "choose volume averaging method (only for fields inserted using the '_a' ending):\n"
                    "  1 = \tvolume average the fields using a Monte Carlo method with pseudo-random numbers inside the Delaunay cell.\n"
                    "  2 = \tvolume average the fields using the Monte Carlo method inside the grid cell.\n"
                    "  3 = \tvolume average the fields using uniformly distributed volume sampling points in each grid cell (recommended only for testing purposes).")
            ("samples,s", po::value<int>(&(this->noPoints)), "specify the number of sampling points when volume averaging the fields inside each grid cell (e.g. '-s 20'). DEFAULT values if none specified:\n"
                    "  1st method: \tan average of 100 sample points per grid cell.\n"
                    "  2nd method: \t20 random points per grid cell.\n"
                    "  3rd method: \t27 random points per grid cell.")
            ("density0", po::value<Real>(&averageDensity), "supply a value to be used to scale the density. If none is supplied, the average density will be used for this task.")
            ("seed", po::value<size_t>(&(this->randomSeed)), "integer value to be used for the random seed generator when interpolating to the grid using Monte Carlo methods. Generated randomly if not supplied by the user.")
#ifndef PHASE_SPACE
            ("gpu", po::bool_switch(&(this->gpuAlias)), "standard DTFE only: run the volume-averaged ('_a', method 1) grid interpolation on the GPU. Requires a GPU build ('make DTFE METAL=1', macOS/Apple Silicon); otherwise the option is ignored with a warning and the CPU interpolation is used. Results match the CPU interpolation to float rounding (atomic summation order). Unaveraged fields and methods 2/3 always use the CPU.")
            ("exact-average", po::bool_switch(&(this->exactAverage)), "standard DTFE only, 3D only, CPU only: compute the volume-averaged ('_a') fields by integrating the LINEAR DTFE interpolant EXACTLY over every grid-cell/tetrahedron intersection (vendored r3d library, Powell & Abel 2015) instead of Monte-Carlo sampling. For a linear field the integral over each intersection is its centroid value times its volume, so order-1 moments suffice -- no sampling noise, and on a perfect particle lattice the averaged density is exactly 1 to rounding. Runs on the method-1 per-tetrahedron scatter topology with the same cell classification (the single-grid-cell fast path was already exact); '--samples' and '--method' are ignored (pass neither, or '-m 1'). Combined with '--gpu' the interpolation falls back to the CPU with a warning. An accuracy option, not a speed option.")
#endif
#ifdef PHASE_SPACE
            ("ps-gpu", po::bool_switch(&(this->gpuAlias)), "PS-DTFE only: run the grid deposit (the dominant cost) on the GPU. Requires a GPU build ('make PS-DTFE METAL=1', macOS/Apple Silicon); otherwise the option is ignored with a warning and the CPU deposit is used. Results match the CPU deposit to float rounding (atomic summation order).")
            ("avg-subsamples", po::value<int>(&(this->psAvgSubsamples))->default_value(3), "PS-DTFE only: linear sub-sample count nSub for the volume-averaged ('_a') fields. Each grid cell is volume-averaged over an nSub^3 regular sub-grid, so the '_a' interpolation cost scales as nSub^3 -- it is the dominant runtime cost. 3 (default) = 27 sub-points; 2 = 8 (~3.4x faster '_a' pass, slightly coarser average); 1 = cell-centre only (= the unaveraged value, no extra cost but no averaging benefit). Lower this to speed up runs dominated by the averaged-field pass.")
#endif
            ("sample-points", po::value<std::string>(&(this->psSamplePointsFile)), "evaluate the fields at arbitrary Eulerian points read from the given file, in ADDITION to the regular grid interpolation (pass a small '--grid', e.g. '-g 16', when only point values are wanted). PS-DTFE: the multi-stream phase-space fields, one stream per folded tetrahedron containing the point, deterministic under any '--partition' split. Standard DTFE: the plain Eulerian DTFE interpolant -- exactly one containing tetrahedron, '.pts_streams' becomes the 0/1 coverage flag, '.pts_velDisp' is identically 0, and the run uses a single triangulation (not combinable with '--partition'). Point coordinates are in the box coordinate system (the same units as '--box', i.e. Mpc after the '--MpcUnit' conversion). Accepted formats, auto-detected: a text file with one 'x y z' triplet per line, or a raw binary file of float64 triplets (N x 3, row-major, native/little-endian, no header). Writes '<output>.pts_den' (float64, rho/rho_bar), '.pts_vel' (float64 x3, density-weighted mean over streams), '.pts_velDisp' (float64 x6 dispersion tensor, xx xy xz yy yz zz), '.pts_streams' (int32); all CPU-only, double precision.")
            ("per-stream", po::bool_switch(&(this->psPerStream)), "PS-DTFE only: with '--sample-points', additionally write each stream's own density and velocity per point in a ragged layout: '<output>.pts_stream_offsets' (uint64, N+1; point i owns records [off[i], off[i+1])) and '<output>.pts_stream_records' (float64 x4 per record: density, vx, vy, vz; sorted by density, descending). The standard binary rejects this flag (its points have no stream decomposition).")
            ("per-stream-ids", po::bool_switch(&(this->psPerStreamIds)), "PS-DTFE only: with '--sample-points', also write each stream's identity -- the ParticleIDs of the 4 Lagrangian vertices of its tetrahedron -- to '<output>.pts_stream_ids' (uint64 x4 per record, same ragged order as '.pts_stream_records'; each quadruple sorted ascending, so it is orientation-independent and identical under any '--partition' split). Implies '--per-stream'. Needs an input reader that provides ParticleIDs (Gadget HDF5); otherwise the IDs are written as 0. The standard binary rejects this flag.")
            ("pts-den-grad", po::bool_switch(&(this->psPtsDenGrad)), "with '--sample-points', also write the spatial gradient of the total point density to '<output>.pts_denGrad' (float64 x3 per point, d(rho/rho_bar)/dx_i in 1/Mpc). PS-DTFE: the sum over the point's streams of each stream's constant linear 'dtfe' density-profile gradient; with '--per-stream', additionally writes '<output>.pts_stream_dengrad' (float64 x3 per record, same ragged order as '.pts_stream_records'). The gradient always belongs to the 'dtfe' estimator: under '--ps-stream-density geometric' the per-stream density is piecewise constant (zero gradient), so the dtfe-profile gradient (well-defined from the vertex densities) is still what is written; hull cells with the constant volume-ratio density contribute 0. Standard DTFE: the containing tetrahedron's constant DTFE density gradient.")
            ("pts-vel-grad", po::bool_switch(&(this->psPtsVelGrad)), "with '--sample-points', also write the velocity gradient of the point's multi-stream flow to '<output>.pts_velGrad' (float64 x9 per point, row-major [d*3+j] = d v_j / d x_d in velocity units per Mpc -- the SAME layout as the '.velGrad' grid output). PS-DTFE: the density-weighted mean over the point's streams of each stream's constant linear-velocity-profile gradient (the pointwise analogue of the grid velocity-gradient deposit; the weight is the same per-stream density that weights '.pts_vel'). Divergence, shear and vorticity follow algebraically from this tensor, so they need no files of their own. Standard DTFE: the containing tetrahedron's constant velocity gradient.")
#ifdef PHASE_SPACE
            ("ps-stream-density", po::value<std::string>()->default_value("dtfe"), "PS-DTFE only: per-stream density estimator for '--sample-points'. 'dtfe' (default) = linear interpolation of the Lagrangian-vertex DTFE densities inside each stream's tetrahedron (matches the PhaseSpaceDTFE class of the reference implementation, github.com/jfeldbrugge/PS-DTFE); 'geometric' = constant per-tetrahedron density m_tet/V_eul = rho_bar*V_lag/V_eul (the density whose mass-conserving rendering the grid deposit produces). The selected estimator also weights the per-point mean velocity and dispersion.")
            ("ps-linear-deposit", po::bool_switch(&(this->psLinearDeposit)), "PS-DTFE only: weight each tetrahedron's grid deposit by the DTFE-interpolated LINEAR density profile inside the tetrahedron instead of spreading its mass uniformly over the interior sub-samples. The per-sample weights are RENORMALIZED per tetrahedron so the deposited total still equals the tetrahedron's mass exactly -- mass conservation is unchanged (it is the fix for the historical caustic 'square' artefacts). Smoother sub-tetrahedron structure at the cost of reading the vertex densities during the deposit; works with the CPU and all GPU deposits.")
            ("ps-caustics", po::bool_switch(&(this->psCaustics)), "PS-DTFE only, 3D only: flag fold-caustic grid cells during the grid deposit and write '<output>.caustic' (float32, same N^3 layout as '.streams'; 1 = the cell is overlapped by tetrahedra of BOTH orientations, 0 otherwise). The orientation of a stream is the sign of det(Ax), the parity of the Lagrangian->Eulerian map, which flips at every fold caustic -- so a cell containing both parities is crossed by a fold surface (the caustic skeleton of Feldbrugge & van de Weygaert). The two per-cell orientation bits are merged with a bitwise OR across '--partition' splits, making the output partition- and thread-order invariant (byte-identical for any split, per backend). Computed by both the CPU and GPU deposits (the GPU ORs the bits atomically, so its output is equally order-invariant; CPU-vs-GPU grids can differ only for cells whose float-vs-double determinant sign straddles the degeneracy cut). Point evaluation ('--sample-points') is unaffected by this flag.")
            ("ps-exact-deposit", po::bool_switch(&(this->psExactDeposit)), "PS-DTFE only, 3D only: replace the nSub^3 sub-sampled grid deposit by the EXACT conservative voxelization of Powell & Abel 2015 (vendored r3d library, third_party/r3d): every kept tetrahedron is analytically clipped against each grid cell in its bounding box and deposits mass * V_intersection/V_tetrahedron per cell -- no sampling noise, mass conservation to double precision. The kept-cell classification, periodic minimum-image convention and per-tet renormalization are IDENTICAL to the standard deposit, so '.streams'-defining overlaps, '--ps-halo-release' and '--ps-caustics' compose unchanged. Velocity moments integrate the LINEAR velocity profile exactly (order-1 moments); the dispersion additionally carries each cell-intersection's exact velocity covariance (order-2 moments). With '--ps-linear-deposit' the per-cell mass shares integrate the linear density profile exactly (order-1), renormalized per tetrahedron exactly like the sampled path. nSub ('--avg-subsamples') is ignored: the exact deposit is the nSub->infinity limit, so the unaveraged and '_a' grids coincide. Runs on the CPU (double precision, the reference) and on the GPU with '--ps-gpu' (a float32 port of the same r3d clipping and moment recursion, agreeing with the CPU to float rounding like every other GPU field) -- the GPU is strongly recommended here, since the analytic clipping is by far the most expensive deposit. Point evaluation ('--sample-points') is unaffected. An accuracy option first: even on the GPU it is slower than the sampled deposit.")
            ("ps-volume-weighted", po::bool_switch(&(this->psVolumeWeighted)), "PS-DTFE only: weight the VELOCITY MOMENTS (velocity, gradient and its derived divergence/shear/vorticity/vweb, dispersion, scalars) by EULERIAN-VOLUME shares instead of mass shares during the grid deposit. The default mass weighting gives the momentum-like multi-stream mean sum(m_s f_s)/sum(m_s), whose sub-cell density-velocity covariance steepens the velocity-divergence--density relation by a few percent even in single-stream regions; volume weighting gives the volume average of the field over the cell -- the standard-DTFE '_a' convention and the quantity linear theory's -aHf*delta refers to. Each sampled point carries V_eul/n_samples (the exact deposit carries the analytic V_intersection), so single-stream regions reproduce the standard DTFE volume average. The VELOCITY DISPERSION is deliberately EXCLUDED and stays mass-weighted: sigma_ij is a second moment of the phase-space distribution f (rho*sigma_ij = integral f (v-vbar)_i (v-vbar)_j d3v), so it is f- i.e. mass-weighted BY DEFINITION -- that is what makes it the quantity entering the Jeans equations. It therefore carries its own mass-weighted mean and normalizer internally, and comes out bit-identical to a default (mass-weighted) run. So this flag gives the literature-standard estimator for EVERY field at once: volume-weighted velocity statistics (Bernardeau & van de Weygaert's motivation for the Delaunay estimator) plus mass-weighted phase-space moments. The density, stream-count and caustic outputs are unchanged. Works with the CPU, GPU and '--ps-exact-deposit' paths (the GPU kernel derives the volume shares from the same per-tet determinant as the CPU, so the float-parity contract carries over); cannot be combined with '--ps-linear-deposit' (which is density weighting inside each tetrahedron, the opposite convention).")
            ("ps-vertex-mass", po::bool_switch(&(this->psVertexMass)), "PS-DTFE only: assign each tetrahedron its CHART-INDEPENDENT mass m_tet = sum over its 4 vertices of (particle mass)/(number of finite tetrahedra incident to that vertex), instead of the default m_tet = rho_bar * V_lag. The default is exact when the Lagrangian positions are the UNPERTURBED lattice, but when '--lagrangianInput' holds evolved IC positions (e.g. TNG's z=127 snapshot) the IC configuration already carries the density contrast delta_ic = D(z_ic)/D(z) * delta(z): tetrahedra in future-overdense regions are pre-compressed and rho_bar*V_lag under-weights them, filtering EVERY density mode by 1 - D(z_ic)/D(z) (measured -16.5\% at z=20, -3\% at z=2 for TNG's z_ic=127; velocity fields are unaffected, but the divergence-density slope steepens by the inverse factor). Splitting each particle's mass equally among its incident tetrahedra makes the mass follow the particles, removing the bias at any z_ic; total mass is conserved exactly, at the cost of small-scale degree noise (per-tet masses vary with the local Delaunay connectivity). Grid deposit only (CPU, GPU and '--ps-exact-deposit'); the '--sample-points' per-stream density estimators are unchanged. The '--ps-halo-release' kept/released classification stays geometric (V_lag/V_eul), so it is identical with and without this flag.")
            ("ps-halo-release", po::value<Real>(&(this->psHaloRelease)), "PS-DTFE only: release halo-interior streams from the grid deposit's bbox rasterization. A tetrahedron compressed beyond the given threshold D -- geometric stream density rho_geo = rho_bar*V_lag/V_eul > D, in rho/rho_bar units -- is deposited MONOLITHICALLY at its Eulerian centroid cell (the same mass-conserving fallback used for sub-sample-spacing tetrahedra), with its velocity/dispersion moments carrying the centroid-evaluated velocity. Rationale (Stuecker et al. 2021, arXiv:2109.09760): inside virialized halos the dark-matter sheet is mixed, fold counts explode (millions of streams per cell) and the sheet estimate is not converged anyway -- releasing those tetrahedra cuts the dominant tet-cell overlap cost while conserving mass EXACTLY and leaving single-stream regions (void interiors) bit-identical. Applies to the CPU and GPU deposits (identical kept/released classification, double-precision |det Lag| > D*|det Ax| in both). Point evaluation ('--sample-points') is NOT affected by this flag. Suggested range 100-1000; see the README for measured speed/accuracy trade-offs.")
#endif
            ;
    
    
    po::options_description redshiftConeOptions("Redshift cone options");
    redshiftConeOptions.add_options()
            ("redshiftCone", po::value< std::vector<Real> >()->multitoken(), "specify to interpolate the fields on a redshift cone grid (i.e. on spherical coordinates). Must give 6 arguments in 3D (4 in 2D) which give 'r_min', 'r_max', 'theta_min', 'theta_max', 'psi_min' and 'psi_max' (where 'r' is the distance in Mpc and 'theta' and 'psi' the two angles expressed in degrees). For example '--redshiftCone 1 10 0 90 0 360' gives you a half a sphere shell." )
            ("origin", po::value< std::vector<Real> >( &(this->originPosition) )->multitoken(), "specify the origin of the spherical coordinate system used in option '--redshiftCone'. It gives the x, y and z values of the origin point.")
            ;
    
    
    po::options_description additionalOptions("Additional options");
    additionalOptions.add_options()
            ("config,c", po::value<std::string>(&(this->configFilename)), "supply all/part of the program options in a configuration file. The option syntax is the same as at the command line. Can insert comments using the '#' symbol (everything after this symbol until the end of the line will be considered a comment)." )
            ("NGP", "choose the NGP (nearest grid point) as the grid interpolation method instead of DTFE. This method is available only for: density and velocity fields.")
            ("CIC", "choose the CIC (Cloud In Cell) as the grid interpolation method instead of DTFE. This method is available only for: density and velocity fields.")
            ("TSC", "choose the TSC (Triangular Shape Cloud) as the grid interpolation method instead of DTFE. This method is available only for: density and velocity fields.")
            ("PCS", "choose the PCS (Piecewise Cubic Spline) as the grid interpolation method instead of DTFE. This method is available only for: density and velocity fields.")
            ("SPH", po::value<int>(&(this->SPH_neighbors)), "choose the SPH (Smoothed Particle Hydrodynamics) as the grid interpolation method instead of DTFE. This method is available only for: density, velocity and scalar fields.")
            ("Voronoi", "use Voronoi volume density estimation with NGP grid assignment. Builds the Delaunay triangulation, computes vertex densities (1/Voronoi volume), and assigns them to the grid using nearest grid point. Gives a piecewise-constant density field. Only outputs density.")
            ("interlace", "enable interlacing to reduce aliasing in the density field. Runs the grid interpolation twice with a half-cell offset and averages in Fourier space. Works with any interpolation method. Requires periodic boundary conditions.")
            ("MpcUnit", po::value<Real>(&(this->MpcValue)), "specify the value of 1Mpc in units of the input particle position data. [DEFAULT value is the one given in the 'DMPC_UNIT' variable in the Makefile.]")
            ("extensive", "specify that all the fields under 'scalar fields' are extensive quantities. If this option is missing than the code treats the variables as intensive fields. This option is important only when using the TSC or SPH interpolation methods applied to the scalar variable.")
            ("verbose,v", po::value<int>(&(this->verboseLevel))->default_value(verboseLevel), "choose the verbosity level of the program (a value from 0 to 3). See the documentation for additional help.")
            ("randomSample", po::value<Real>(&(this->randomSample)), "generates a random subsample of the input data. The size of the subsample is given by value supplied to the option (with values from 0. to 1.). Only this random subsample of the full data set will be used in any further computations. For example '--randomSample 0.1' will keep only 10\% of the data set for further computations.")
            ("poisson", po::value<size_t>(&(this->poisson)), "generate the particle positions randomly. The argument gives the root 3 in 3D (and root 2 in 2D) of the random number of particles. The particles have the same weight and are in a box of size unity. For example '-poisson 256' will generate 256^3 particles in 3D, while only 256^2 particles in 2D.")
#ifdef REDSHIFT_SPACE
            ("redshiftSpace", po::value< std::vector<Real> >()->multitoken(), "specify this option to transform the particle positions from position-space to redhsift-space. This option takes 3 arguments that specifies the direction (d1,d2,d3) along which to tranform to redshift-space. For example '--redshiftSpace d1 d2 d3' specifies to transform to 'redshift-space = position-space + (d1,d2,d3)*velocity / H', with H=100 h km/s /Mpc and (d1,d2,d3) normalized to a unit vector." )
#endif
            ("options", po::value< std::vector<std::string> >( &(this->additionalOptions) )->multitoken(), "variable used to supply additional options to the program in a very simple way. Each additional option will be stored as a string in 'User_options.additionalOptions' (this variable is a vector of strings).")
            ("scratch-dir", po::value<std::string>(&(this->scratchDir)), "back every allocation of >= 1 GB -- in practice, the full-resolution output grids, which '--partition' can NOT reduce (each partition deposits into the same shared grid) -- with memory-mapped files in this directory instead of RAM. Lets a 64 GB machine run e.g. a 1024^3 grid with the full field set (~146 GB of accumulators) against local disk; results are bit-identical, the kernel pages the grids in and out as needed, and the scratch files are unlinked at creation so they vanish automatically even if the run crashes. The directory must exist on a LOCAL, non-synced volume (e.g. /private/tmp/dtfe-scratch) with enough free space for the grids; iCloud paths are rejected. The 1 GB threshold can be changed with the DTFE_SCRATCH_MIN_GB environment variable.")
#ifdef PHASE_SPACE
            ("lagrangianInput", po::value<std::string>(&(this->lagrangianInputFilename)), "specify the HDF5 file containing the Lagrangian (initial condition) positions for PS-DTFE. The file should contain particle coordinates in the same order as the main input file. If not specified, the Lagrangian positions are read from the 'InitialCoordinates' dataset in the main input file.")
#endif
#ifndef PHASE_SPACE
            ("approxPSD", "compute approximate phase-space density f = rho * g, where rho is the spatial DTFE density and g is the velocity-space DTFE density. Builds a second Delaunay tessellation in velocity space. The result is stored in the scalar output field.")
#endif
            ("lambda_th", po::value<Real>(&(this->lambda_th))->default_value(Real(0.)), "eigenvalue threshold for T-web/V-web cosmic web classification. Grid cells with eigenvalues above this threshold are classified as collapsing along that axis. [DEFAULT: 0.0]")
            ("hubble", po::value<Real>(&(this->hubbleParam)), "Hubble parameter h for T-web/V-web normalization (H=100 h km/s/Mpc). If not specified, the value is read from the simulation file header.")
            ("scale-factor", po::value<Real>(&(this->scaleFactor)), "scale factor a of the snapshot, used to convert Gadget u-velocities (u = v_pec/sqrt(a)) to peculiar km/s in the V-web normalization. If not specified, the value is read from the simulation file header ('Time'); pass it explicitly for inputs whose header lacks it.")
            ;
    
    
    po::options_description hidden("Hidden options");
    hidden.add_options()
            ("inputFile", po::value<std::string>(&(this->inputFilename)), "name of the input position file")
            ("outputFile",po::value<std::string>(&(this->outputFilename)), "root name of the output file/files")
            ;
    
    
    // add the options to 'visibleOptions' and to 'allOptions'
    visibleOptions.add(mainOptions);
#ifdef FIELD_OPTIONS
    visibleOptions.add(fieldOptions);
#endif
#ifdef REGION_OPTIONS
    visibleOptions.add(regionOptions);
#endif
#ifdef PARTITION_OPTIONS
    visibleOptions.add(partitionOptions);
#endif
#ifdef PADDING_OPTIONS
    visibleOptions.add(paddingOptions);
#endif
#ifdef AVERAGING_OPTIONS
    visibleOptions.add(averagingOptions);
#endif
#ifdef REDSHIFT_CONE_OPTIONS
    visibleOptions.add(redshiftConeOptions);
#endif
#ifdef ADDITIONAL_OPTIONS
    visibleOptions.add(additionalOptions);
#endif
    
    
    allOptions.add(mainOptions);
    allOptions.add(fieldOptions);
    allOptions.add(regionOptions);
    allOptions.add(partitionOptions);
    allOptions.add(paddingOptions);
    allOptions.add(averagingOptions);
    allOptions.add(redshiftConeOptions);
    allOptions.add(additionalOptions);
    allOptions.add(hidden);
    
    // hidden options as positional arguments
    p.add( "inputFile", 1 );
    p.add( "outputFile", 1 );
}


// Prints the detailed help (full option descriptions) and exits.
void User_options::helpInformation( po::options_description &visibleOptions, char *progName )
{
    MESSAGE::Message message( 3 );
    message << "Use this program to interpolate fields on a grid using the DTFE method - there are multiple options which allow for increased flexibility of the program. The computations are done using the Delaunay Triangulation module from the CGAL library.\n";
    message << "Usage:    " << progName << "  name_position_file  output(root)_file  'options - see below' \n";
    message << "On top of the above, the user can add any of the following options:\n";
    message << visibleOptions << "\n" << MESSAGE::Flush;
    exit( EXIT_SUCCESS );
}
// Prints the summary help (short one-line option descriptions) and exits.
void User_options::shortHelp( char *progName )
{
    Real temp;  // dummy bind target: shortHelp only displays option names/blurbs, never reads values
    po::options_description mainOptions("Main options");
    mainOptions.add_options()
            ("help,h", "produce this help message.")
            ("full_help", "produce detailed help message.")
            ("grid,g", po::value< std::vector<size_t> >(&(this->gridSize))->multitoken(), "specify grid size along each direction.")
            ("box", po::value< Real >(&temp), "specify the coordinates of the box encompasing all the particles.")
            ("input,i", po::value< std::vector<int> >()->multitoken(), "give the type of the input file, which data to read and for which particle species. See full help for details.")
            ("output,o", po::value< std::vector<int> >(), "give the type of the output file. See full help for details.")
            ("periodic,p", "specify the data is in a periodic box.")
            ;
    po::options_description fieldOptions("Field choices");
    fieldOptions.add_options()
            ("field,f", po::value< Real >(&temp), "specify which field to interpolate to grid. Available options are:\n"
                    "  density = \tdensity at the sampling point position.\n"
                    "  density_a = \tvolume averaged density inside the sampling cell[DEFAULT]. Each of the options below also have a '*_a' version which is left out to minimize the help messages.\n"
#ifdef VELOCITY
                    "  velocity = \tnon-averaged velocity (use 'velocity_a' to get the volume averaged value).\n"
                    "  gradient = \tnon-averaged velocity gradient (use 'gradient_a' to get the volume averaged value).\n"
                    "  divergence = \tnon-averaged velocity divergence (use 'divergence_a' to get the volume averaged value).\n"
                    "  shear = \tnon-averaged velocity shear (use 'shear_a' to get the volume averaged value).\n"
                    "  vorticity = \tnon-averaged velocity vorticity (use 'vorticity_a' to get the volume averaged value).\n"
#endif
#ifdef SCALAR
                    "  scalar = \tnon-averaged scalar quantities (use 'scalar_a' to get the volume averaged value).\n"
                    "  scalarGradient = \tnon-averaged gradient of the scalar quantities (use 'scalarGradient_a' to get the volume averaged value).\n"
#endif
#ifdef VELOCITY
                    "  tweb = \tT-web classification from the tidal tensor of the density grid (use 'tweb_a' for volume averaged).\n"
                    "  vweb = \tV-web cosmic web classification from velocity shear tensor eigenvalues (use 'vweb_a' for volume averaged).\n"
#endif
            );
    po::options_description regionOptions("Region options");
    regionOptions.add_options()
            ("region", po::value< Real >(&temp), "specify to interpolate the fields only in a region of the box given in terms of fractions of box length.")
            ("regionMpc", po::value< Real >(&temp), "specify to interpolate the fields only in a region given in Mpc coordinates.")
            ;
    po::options_description partitionOptions("Partition options");
    partitionOptions.add_options()
            ("partition", po::value< Real >(&temp), "specify in how many parts to split the box along each direction.")
            ("partNo", po::value< Real >(&temp), "choose to compute the interpolation only for this partition number.")
            ("max-concurrent", po::value< Real >(&temp), "cap how many Delaunay triangulations build at once to bound peak memory (0 = all threads).")
            ;
    po::options_description paddingOptions("Padding options");
    paddingOptions.add_options()
            ("padding", po::value< Real >(&temp), "give the size of the buffer zone to make sure that the triangulation fully covers the region of interest.")
            ("paddingMpc", po::value< Real >(&temp), "give the size of the buffer zone in Mpc units.")
#ifdef TEST_PADDING
            ("noTest", "do NOT test for the efficiency of the padding.")
#endif
            ;
    po::options_description averagingOptions("Averaging options");
    averagingOptions.add_options()
            ("method,m", po::value< Real >(&temp), "choose the MC averaging method: 1 = volume average inside the Delaunay cell OR 2 = volume average inside the grid cell.")
            ("samples,s", po::value< Real >(&temp), "specify the number of MC sampling points for volume averaged interpolation.")
            ("density0", po::value< Real >(&temp), "value to scale the density [DEFAULT: use average density].")
            ("seed", po::value< Real >(&temp), "integer value used as seed for the random generator.")
#ifdef PHASE_SPACE
            ("avg-subsamples", po::value< Real >(&temp), "PS-DTFE: nSub for '_a' volume averaging (cost ~nSub^3; 3 default, 2 ~3.4x faster, 1 = no averaging).")
#endif
            ;
    po::options_description redshiftConeOptions("Redshift cone options");
    redshiftConeOptions.add_options()
            ("redshiftCone", po::value< Real >(&temp), "specify to interpolate the fields on a redshift cone grid and give the coordinates  ('r_min', 'r_max', 'theta_min', 'theta_max', 'psi_min' and 'psi_max')." )
            ("origin", po::value< Real >(&temp), "specify the origin of the spherical coordinate system.")
            ;
    po::options_description additionalOptions("Additional options");
    additionalOptions.add_options()
            ("config,c", po::value< Real >(&temp), "name the configuration file from which to read the program options." )
            ("NGP", "choose NGP (Nearest Grid Point) for grid interpolation.")
            ("CIC", "choose CIC (Cloud In Cell) for grid interpolation.")
            ("TSC", "choose TSC (Triangular Shape Cloud) for grid interpolation.")
            ("PCS", "choose PCS (Piecewise Cubic Spline) for grid interpolation.")
            ("SPH", po::value< Real >(&temp), "choose SPH for grid interpolation (argument = number nearest neighbors).")
            ("Voronoi", "use Voronoi volume density with NGP grid assignment (density only).")
            ("interlace", "enable interlacing to reduce aliasing (Fourier-space averaging of half-cell offset grids).")
            ("MpcUnit", po::value< Real >(&temp), "value of 1Mpc in units of the input position data.")
            ("extensive", "the fields under 'scalar fields' are extensive quantities [DEFAULT: intensive variables].")
            ("verbose,v", po::value< Real >(&temp), "choose the verbosity level (from 0 to 3).")
            ("randomSample", po::value< Real >(&temp), "generates a random subsample of the input data (argument = from 0 to 1, gives fraction of particles).")
            ("poisson", po::value< Real >(&temp), "generate the particle positions randomly.")
#ifdef REDSHIFT_SPACE
            ("redshiftSpace", po::value< std::vector<Real> >()->multitoken(), "transform the particle positions from position-space to redhsift-space -- need to give 3 values that give the direction for the shift." )
#endif
            ("options", po::value< Real >(&temp), "variable used to supply additional options.")
            ("scratch-dir", po::value<std::string>(&(this->scratchDir)), "back the full-grid accumulators with mmap'ed files in this LOCAL directory (out-of-core; >= 1 GB allocations).")
#ifdef PHASE_SPACE
            ("lagrangianInput", po::value<std::string>(&(this->lagrangianInputFilename)), "HDF5 file with Lagrangian (initial condition) positions for PS-DTFE.")
#endif
            ("lambda_th", po::value< Real >(&temp), "eigenvalue threshold for T-web/V-web classification [DEFAULT: 0.0].")
            ("hubble", po::value< Real >(&temp), "Hubble parameter h for T-web/V-web normalization [DEFAULT: from file header].")
            ;
    
    po::options_description visibleOptions;
    visibleOptions.add(mainOptions);
#ifdef FIELD_OPTIONS
    visibleOptions.add(fieldOptions);
#endif
#ifdef REGION_OPTIONS
    visibleOptions.add(regionOptions);
#endif
#ifdef PARTITION_OPTIONS
    visibleOptions.add(partitionOptions);
#endif
#ifdef PADDING_OPTIONS
    visibleOptions.add(paddingOptions);
#endif
#ifdef AVERAGING_OPTIONS
    visibleOptions.add(averagingOptions);
#endif
#ifdef REDSHIFT_CONE_OPTIONS
    visibleOptions.add(redshiftConeOptions);
#endif
#ifdef ADDITIONAL_OPTIONS
    visibleOptions.add(additionalOptions);
#endif
    
    
    MESSAGE::Message message( 3 );
    message << "Use this program to interpolate fields on a grid using the DTFE method - there are multiple options which allow for increased flexibility of the program. The computations are done using the Delaunay Triangulation module from the CGAL library.\n";
    message << "Usage:    " << progName << "  name_position_file  output(root)_file  'options - see below' \n";
    message << "On top of the above, the user can add any of the following options:\n";
    message << visibleOptions << "\n";
    message << "\nUse '--full_help' for a more detailed help information.\n" << MESSAGE::Flush;
    exit( EXIT_SUCCESS );
}




// Print the options the program will use.
void User_options::printOptions()
{
    std::string uField;
    if ( this->uField.density ) uField = " density,";
    if ( this->uField.velocity ) uField += " velocity,";
    if ( this->uField.velocity_gradient ) uField += " velocity gradient,";
    if ( this->uField.velocity_divergence ) uField += " velocity divergence,";
    if ( this->uField.velocity_shear ) uField += " velocity shear,";
    if ( this->uField.velocity_vorticity ) uField += " velocity vorticity,";
    if ( this->uField.scalar ) uField += " scalar,";
    if ( this->uField.scalar_gradient ) uField += " scalar gradient,";
    if ( this->uField.triangulation ) uField += " triangulation,";
    if ( not this->uField.selected() and this->NGP ) uField = "none since NGP interpolation";
    else if ( not this->uField.selected() and this->CIC ) uField = "none since CIC interpolation";
    else if ( not this->uField.selected() and this->TSC ) uField = "none since TSC interpolation";
    else if ( not this->uField.selected() and this->SPH ) uField = "none since SPH interpolation";
    else if ( not this->uField.selected() ) uField = "none";
    
    std::string aField;
    if ( this->aField.density ) aField = " density,";
    if ( this->aField.velocity ) aField += " velocity,";
    if ( this->aField.velocity_gradient ) aField += " velocity gradient,";
    if ( this->aField.velocity_divergence ) aField += " velocity divergence,";
    if ( this->aField.velocity_shear ) aField += " velocity shear,";
    if ( this->aField.velocity_vorticity ) aField += " velocity vorticity,";
    if ( this->aField.velocity_std ) aField += " velocity standard deviation,";
    if ( this->aField.scalar ) aField += " scalar,";
    if ( this->aField.scalar_gradient ) aField += " scalar gradient,";
    if ( not this->aField.selected() ) aField = "none";
    
    std::string interpolationMethod = "DTFE";
    if ( this->CIC ) interpolationMethod = "CIC";
    else if ( this->NGP ) interpolationMethod = "NGP";
    else if ( this->TSC ) interpolationMethod = "TSC";
    else if ( this->PCS ) interpolationMethod = "PCS";
    else if ( this->Voronoi ) interpolationMethod = "Voronoi (NGP)";
    else if ( this->SPH )
    {
        char temp[100];
        snprintf( temp, sizeof(temp), "%d neighbors", this->SPH_neighbors );
        interpolationMethod = "SPH ";
        interpolationMethod += temp;
    }
    
    std::string inFileType = "unknown";
    if ( this->inputFileType==101 ) inFileType = "Gadget multiple files";
    else if ( this->inputFileType==102 ) inFileType = "Gadget single file";
    else if ( this->inputFileType==105 ) inFileType = "Gadget HDF5 file/files";
    else if ( this->inputFileType==111 ) inFileType = "text file with positions (first 3 columns and weights in 4th column)";
    
    std::string outFileType = "unknown";
    if ( this->outputFileType==101 ) outFileType = "binary file";
    else if ( this->outputFileType==100 ) outFileType = "density binary file";
    else if ( this->outputFileType==110 ) outFileType = "text file";
    
    
    MESSAGE::Message message( verboseLevel );
    message << MESSAGE::banner("RUN CONFIGURATION");
    message << MESSAGE::cBold() << "RUNNING: " << this->programOptions << MESSAGE::cReset() << "\n\n";
    message << MESSAGE::cCyan() << "The program will interpolate to grid the chosen field using the DTFE method with the following input parameters:" << MESSAGE::cReset() << "\n"
            << "\t unaveraged field(s)    : " << uField << "\n"
            << "\t averaged field(s)      : " << aField << "\n"
            << "\t interpolation method   : " << interpolationMethod << "\n"
            << "\t input data file        : " << this->inputFilename << "\n"
            << "\t input data file type   : " << this->inputFileType << " - " << inFileType << "\n"
            << "\t input data blocks      : " << MESSAGE::printElements( this->readParticleData, "  " ) << "\n"
            << "\t input particle species : " << MESSAGE::printElements( this->readParticleSpecies, "  " ) << "\n"
            << "\t output file            : " << this->outputFilename << "\n"
            << "\t output file type       : " << this->outputFileType << " - " << outFileType << "\n";
    if ( not this->gridSize.empty() )
        message << "\t grid size              : " << MESSAGE::printElements( this->gridSize, "  " ) << (this->regionOn ? "   for the box region selected by the user\n" : "   for the full particle box\n" );
    else
        message << "\t grid size              : none specifed at the moment";
    if ( not boxCoordinates.isNullBox() )
        message << "\t box coordinates        : [" << MESSAGE::printElements( boxCoordinates, ", " ) << "]\n";
    if ( this->periodic )
        message << "\t computing the grid interpolation in a PERIODIC box\n";
    
    
    if ( this->regionOn )
    {
        char const axisName[] = "xyz";
        message << "\t computing the grid interpolation only for the user specified region of coordinates:\n";
        for (int d=0; d<NO_DIM; ++d)
            message << "\t\t " << axisName[d] << " extension : " << region[2*d] << "   " << region[2*d+1] << (regionMpcOn?"  Mpc":"  box length") << "\n";
    }
    
    
    if ( this->partitionOn )
    {
        message << "\t splitting the data in  : " << MESSAGE::printElements( partition, "  ") << "   separate data sets on which to apply the Delaunay triangulation.\n";
        if ( this->partNo>=0 )
            message << "\t computing grid interpolation ONLY for data set " << partNo << " of the partitioned data\n";
    }
    if ( this->maxConcurrent>0 )
        message << "\t max concurrent triangs : " << this->maxConcurrent << "   (capping concurrent Delaunay triangulations to bound peak memory)\n";
    if ( not this->partitionOn and not this->maxConcurrentOn and this->partNo<0 and not this->redshiftConeOn )
        message << "\t partition/concurrency  : auto (chosen from the data size, grid, fields and available RAM once the input is read; pass --partition / --max-concurrent to override)\n";
    
    
    if ( not this->paddingLength.isNullBox() )
    {
        char const padAxisName[] = "xyz";
        message << "\t padding length:\n";
        for (int d=0; d<NO_DIM; ++d)
            message << "\t\t " << padAxisName[d] << " axis : " << paddingLength[2*d] << "   " << paddingLength[2*d+1] << (paddingMpcOn?"  Mpc":"  box length") << "\n";
    }
    else
        message << "\t number padding particles: " << this->paddingParticles << "\n";
#ifndef PHASE_SPACE
    // suppressed in PS-DTFE: no dummy-boundary-test particles, and it uses nSub^NO_DIM
    // sub-grid averaging instead of the Monte-Carlo grid-cell averaging reported below
    if ( this->DTFE and this->testPaddedBoundaries )
        message << "\t the DTFE computation will add DUMMY TEST particles to test the efficiency of the padding\n";


    if ( this->aField.selected() and this->DTFE )
    {
        std::string averagingMethod;
        if ( this->method==1 ) averagingMethod = "Monte Carlo sampling using quasi-random points inside the Delaunay cells";
        else if ( this->method==2 ) averagingMethod = "Monte Carlo sampling using random points inside the sampling cells";
        else if ( this->method==3 ) averagingMethod = "equidistant sampling points inside the sampling cells";
        else averagingMethod = "unknown";
        message << "\t volume averaging method: " << this->method << " - " << averagingMethod << "\n"
                << "\t number sampling points : " << this->noPoints << "\n";
        if ( this->method==2 )
            message << "\t random generator seed  : " << this->randomSeed << "\n";
        if ( averageDensity>Real(0.) )
            message << "\t density scaling value  : " << averageDensity << "\n";
        if ( this->useMetal )
#ifdef DTFE_GPU
            message << "\t '_a' interpolation     : " GPU_BACKEND_NAME " GPU (--gpu)\n";
#else
            message << "\t '_a' interpolation     : CPU (--gpu given but this binary was built without GPU support; falling back)\n";
#endif
    }
#endif
    
    
    if ( redshiftConeOn )
        message << "\t Computing the grid interpolation to a redshift cone grid of coordinates " << (NO_DIM==2? "[r_min, r_max, psi_min, psi_max]" : "[r_min, r_max, theta_min, theta_max, psi_min, psi_max]") << " = [" << MESSAGE::printElements( redshiftCone, ", " ) << "]\n"
            << "\t The origin of the light cone is (x,y,z) : (" << MESSAGE::printElements( originPosition, ", " ) << ")\n";
    
#ifdef PHASE_SPACE
    message << "\t PS-DTFE mode           : enabled (triangulation in Lagrangian space)\n";
    if ( this->aField.selected() )
    {
        int nSubPts = 1;
        for (int d=0; d<NO_DIM; ++d) nSubPts *= (this->psAvgSubsamples<1 ? 1 : this->psAvgSubsamples);
        message << "\t avg sub-samples (nSub) : " << this->psAvgSubsamples << "   (each '_a' cell volume-averaged over nSub^" << NO_DIM << " = " << nSubPts << " points; this sets the dominant '_a' interpolation cost)\n";
        if ( this->psUseMetal )
#ifdef PS_GPU
            message << "\t PS deposit             : " GPU_BACKEND_NAME " GPU (--ps-gpu)\n";
#else
            message << "\t PS deposit             : CPU (--ps-gpu given but this binary was built without GPU support; falling back)\n";
#endif
    }
    if ( not this->lagrangianInputFilename.empty() )
        message << "\t Lagrangian input file  : " << this->lagrangianInputFilename << "\n";
    else
        message << "\t Lagrangian positions   : read from 'InitialCoordinates' dataset in main input file\n";
    if ( not this->psSamplePointsFile.empty() )
        message << "\t point evaluation       : at the sample points in '" << this->psSamplePointsFile
                << "' (per-stream density: " << (this->psStreamDensityGeometric ? "geometric" : "dtfe")
                << (this->psPerStream ? ", writing per-stream records" : "")
                << (this->psPerStreamIds ? " + stream ids" : "")
                << (this->psPtsDenGrad ? ", writing density gradients" : "")
                << (this->psPtsVelGrad ? ", writing velocity gradients" : "") << ")\n";
    if ( this->psLinearDeposit )
        message << "\t grid deposit profile   : linear within each tetrahedron (--ps-linear-deposit; renormalized per tet, mass-conserving)\n";
    if ( this->psCaustics )
        message << "\t caustic flagging       : writing the fold-caustic cell flag to '.caustic' (--ps-caustics; CPU deposit)\n";
    if ( this->psHaloRelease > Real(0.) )
        message << "\t halo-interior release  : tetrahedra with rho_geo/rho_bar > " << this->psHaloRelease
                << " deposit monolithically at their centroid cell (--ps-halo-release; mass-conserving)\n";
    if ( this->psExactDeposit )
        message << "\t grid deposit           : EXACT tetrahedron-cell intersection moments (--ps-exact-deposit; r3d, CPU; nSub ignored)\n";
#else
    if ( not this->psSamplePointsFile.empty() )
        message << "\t point evaluation       : at the sample points in '" << this->psSamplePointsFile
                << "' (standard DTFE interpolant, 0/1 coverage"
                << (this->psPtsDenGrad ? ", writing density gradients" : "")
                << (this->psPtsVelGrad ? ", writing velocity gradients" : "") << ")\n";
    if ( this->exactAverage )
        message << "\t volume averaging       : EXACT cell-tetrahedron integration of the linear interpolant (--exact-average; r3d, CPU; --samples ignored)\n";
#endif
    message << "\t 1Mpc = " << MpcValue << " in units of input data.\n";
    if ( poisson>0 )
        message << "\t Particle positions will be generated randomly. Particle number = " << (NO_DIM==2? poisson*poisson : poisson*poisson*poisson) << ".\n";

#ifdef REDSHIFT_SPACE
    if ( transformToRedshiftSpaceOn )
        message << "\t Transforming from position-space to redshift space using the velocity along the direction :  ( " << MESSAGE::printElements( transformToRedshiftSpace, ", " ) << " )\n";
#endif
    
    
    if ( this->gridSize.empty() )
        message << "\n\n~~~WARNING~~~ No grid size specified. Unless the input data file specifies the grid size for the output result, the program will end with and error message!\n\n";
    
    message << "\n" << MESSAGE::Flush;
}




// Read the user-supplied options and check they satisfy the restrictions.
void User_options::readOptions(int argc, char *argv[], bool getFileNames, bool showOptions)
{
    po::options_description visibleOptions("Allowed options"), allOptions("All options");
    po::positional_options_description p;

    this->addOptions( allOptions, visibleOptions, p );
    
    po::variables_map vm;
    MESSAGE::Message showMessage(3);
    try
    {
        po::store( po::command_line_parser(argc, argv).options(allOptions).positional(p).run(), vm );
        po::notify(vm);
    }
    catch (exception& e)
    {
        throwError( "When reading the command line program options:\n\t\"", e.what(), "\"\n" );
    }
    catch (...)
    {
        throwError( "Unknown error when reading the command line program options!" );
    }
    
    
    
    if ( vm.count("config") )
    {
        ifstream ifs( configFilename.c_str() );
        if (!ifs)
            showMessage << "~~~ERROR~~~ Can not open the configuration file '" << configFilename << "'.\n";
        else
        {
            try
            {
                store( parse_config_file(ifs, allOptions), vm );
                notify(vm);
            }
            catch (exception& e)
            {
                throwError( "When reading the configuration file program options:\n\t\"", e.what(), "\"\n" );
            }
            catch (...)
            {
                throwError( "Unknown error when reading the configuration file program options!" );
            }
        }
    }
    
    
    if ( not (vm.count("help") or vm.count("full_help")) and not vm.count("inputFile") and getFileNames )
        showMessage << "~~~ERROR~~~ No input file detected.\n";
    if ( not (vm.count("help") or vm.count("full_help")) and not vm.count("outputFile") and getFileNames )
        showMessage << "~~~ERROR~~~ No output file detected.\n";
    if ( not (vm.count("help") or vm.count("full_help")) and (not vm.count("inputFile") or not vm.count("outputFile")) and getFileNames )
        exit( EXIT_SUCCESS );
    
    if ( vm.count("help") )
        this->shortHelp( argv[0] );
    else if ( vm.count("full_help") )
        this->helpInformation( visibleOptions, argv[0] );
    
    
    conflicting_options(vm, "SPH", "TSC");
    conflicting_options(vm, "SPH", "CIC");
    conflicting_options(vm, "SPH", "PCS");
    conflicting_options(vm, "CIC", "TSC");
    conflicting_options(vm, "CIC", "PCS");
    conflicting_options(vm, "TSC", "PCS");
    conflicting_options(vm, "NGP", "CIC");
    conflicting_options(vm, "NGP", "TSC");
    conflicting_options(vm, "NGP", "PCS");
    conflicting_options(vm, "NGP", "SPH");
    conflicting_options(vm, "Voronoi", "NGP");
    conflicting_options(vm, "Voronoi", "CIC");
    conflicting_options(vm, "Voronoi", "TSC");
    conflicting_options(vm, "Voronoi", "PCS");
    conflicting_options(vm, "Voronoi", "SPH");
    conflicting_options(vm, "region", "regionMpc");
    conflicting_options(vm, "padding", "paddingMpc");
    
    option_dependency(vm, "partNo", "partition");
    option_dependency(vm, "redshiftCone", "origin");
    


    // Read the options supplied to the program
    if ( vm.count("grid") )
    {
        int const temp = this->gridSize.size();
        if ( temp==1 ) {
            this->gridSize = std::vector<size_t>(NO_DIM, this->gridSize.at(0));
        } else if ( temp!=NO_DIM ) {
            throwError( "You can only insert 1 or ", NO_DIM, " values for the '-g [ --grid ]' option (e.g. '-g 256' or '-g 128 256 256')." );
        }
        for (size_t i=0; i<this->gridSize.size(); ++i) {
            lowerBoundCheck( this->gridSize[i], size_t(1), "the values of option '-g [ --grid ]'" );
        }
    }
    if ( vm.count("box") )      // read the box coordinates
    {
        int const temp = vm["box"].as< std::vector<Real> >().size();
        if ( temp!=2*NO_DIM ) throwError( "You need to specify ", 2*NO_DIM, " arguments with the option '--box'." );
        
        boxCoordinates.coords = vm["box"].as< std::vector<Real> >();
        for (int i=0; i<NO_DIM; ++i)
            if( boxCoordinates[2*i+1]<boxCoordinates[2*i]) throwError( "The right coordinate of the box encompasing the data along axis " , i+1, " (1=x, 2=y, 3=z) must be larger than the left coordinate of the box. This is not the case in the arguments supplied to '--box' option." );
        userGivenBoxCoordinates = true;
    }
    if ( vm.count("input") )    // file type, data blocks to read, and particle species to read
    {
        std::vector<int> temp = vm["input"].as< std::vector<int> >();
        inputFileType = temp[0];
        if ( temp.size()>=2 )   // data blocks
            for (size_t i=0; i<readParticleData.size(); ++i)
            {
                readParticleData[i] = temp[1] % 2;
                temp[1] /= 2;
            }
        if ( temp.size()>=3 )   // particle species
            for (size_t i=0; i<readParticleSpecies.size(); ++i)
            {
                readParticleSpecies[i] = temp[2] % 2;
                temp[2] /= 2;
            }
    }
    if ( vm.count("periodic") )
        this->periodic = true;
    
    
    // read which field to compute using the DTFE method
    if ( vm.count("field") )
    {
        for (size_t i=0; i<vm["field"].as< std::vector<std::string> >().size(); ++i)
        {
            std::string field = vm["field"].as< std::vector<std::string> >().at(i);
            bool uFieldOption = this->uField.updateChoices( field, "triangulation", "density", "velocity", "gradient", "divergence", "shear", "vorticity", "", "scalar", "scalarGradient", "tweb", "vweb", "dispersion" );
            bool aFieldOption = this->aField.updateChoices( field, "", "density_a", "velocity_a", "gradient_a", "divergence_a", "shear_a", "vorticity_a",  "velocityStd_a", "scalar_a", "scalarGradient_a", "tweb_a", "vweb_a", "dispersion_a" );
            if ( not(uFieldOption or aFieldOption) ) throwError( "Unknown value '" + field + "' for the option '--field'." );
        }
    }
    else
        this->aField.density = true;
    
    
    // read the user defined region options
    if ( vm.count("regionMpc") )
        this->regionMpcOn = true;
    if ( vm.count("regionMpc") or vm.count("region") )
    {
        this->regionOn = true;
        size_t temp = 0;
        if (this->regionMpcOn)
        {
            temp = vm["regionMpc"].as< std::vector<Real> >().size();
            this->region.coords = vm["regionMpc"].as< std::vector<Real> >();
        }
        else
        {
            temp = vm["region"].as< std::vector<Real> >().size();
            this->region.coords = vm["region"].as< std::vector<Real> >();
        }
        if ( temp!=2*NO_DIM ) throwError( "You have to insert ", 2*NO_DIM, " values for the '--region' or '--regionMpc' option (e.g. '--region 0.4 0.6 0.3 0.7 0.45 0.55')." );
        for (int i=0; i<NO_DIM; ++i)
            if( region[2*i+1]<region[2*i]) throwError( "The right coordinate of the region of interest along axis ", i+1, " (1=x, 2=y, 3=z) must be larger than the left coordinate. This is not the case in the arguments supplied to '--region' option." );
    }
    
    
    // fold the '--gpu' / '--ps-gpu' switch into the internal member (the Metal path reads
    // useMetal / psUseMetal)
    if ( this->gpuAlias )
    {
#ifdef PHASE_SPACE
        this->psUseMetal = true;
#else
        this->useMetal = true;
#endif
    }

#ifndef PHASE_SPACE
    // exact '_a' averaging (--exact-average) options, standard binary only
    if ( this->exactAverage )
    {
#if NO_DIM!=3
        throwError( "'--exact-average' is only available in 3D builds (NO_DIM==3; r3d is a 3D clipper)." );
#endif
#ifdef MY_SCALAR
        throwError( "'--exact-average' cannot integrate the user-defined MY_SCALAR function (it is not linear inside a tetrahedron); rebuild without MY_SCALAR or use the sampled averaging." );
#endif
        // .defaulted(): '--method' carries a default_value, so vm.count() alone is always 1
        if ( vm.count("method") and not vm["method"].defaulted() and this->method != 1 )
            throwError( "'--exact-average' replaces the sampled averaging and runs on the method-1 scatter topology; drop '--method' or pass '-m 1'." );
        if ( vm.count("NGP") or vm.count("CIC") or vm.count("TSC") or vm.count("PCS") or vm.count("SPH") or vm.count("Voronoi") )
            throwError( "'--exact-average' needs the DTFE interpolation method; it is not available with NGP/CIC/TSC/PCS/SPH/Voronoi." );
    }
#endif

    // --scratch-dir (both binaries): out-of-core backing for the full-grid accumulators.
    // Validate the path HERE, before any big allocation could route through it.
    if ( not this->scratchDir.empty() )
    {
        // iCloud/cloud-synced volumes are refused outright: the scratch files hold up to
        // hundreds of GB and, although unlinked immediately, a sync daemon that raced the
        // unlink would try to upload them. This repo itself lives in iCloud Drive, so the
        // "obvious" choice of a subdirectory next to the outputs is exactly the wrong one.
        if ( this->scratchDir.find("Mobile Documents") != std::string::npos
             or this->scratchDir.find("com~apple~CloudDocs") != std::string::npos )
            throwError( "'--scratch-dir ", this->scratchDir, "' points into iCloud Drive. The scratch files hold up to hundreds of GB of grid data and must live on a LOCAL, non-synced volume -- use e.g. '/private/tmp/dtfe-scratch' (create it first)." );
        struct stat st;
        if ( ::stat(this->scratchDir.c_str(), &st) != 0 or not S_ISDIR(st.st_mode) )
            throwError( "'--scratch-dir ", this->scratchDir, "' is not an existing directory. Create it first (it must be on a local, non-synced volume with enough free space for the full-resolution grids)." );
    }

    // point evaluation (--sample-points) options, both binaries
#ifdef PHASE_SPACE
    {
        std::string const variant = vm["ps-stream-density"].as<std::string>();
        if ( variant == "geometric" )
            this->psStreamDensityGeometric = true;
        else if ( variant != "dtfe" )
            throwError( "Unknown value '", variant, "' for '--ps-stream-density'; use 'dtfe' or 'geometric'." );
    }

    // '--partNo' and '--region' both crop the PARTICLES to an Eulerian box (findParticlesInBox ->
    // Box::isParticleInBox reads p.pos). That is sound for standard DTFE, whose tessellation IS
    // Eulerian -- it is what they were written for. PS-DTFE instead triangulates in LAGRANGIAN
    // space, where an Eulerian-selected point set is full of holes: the tessellation of the subset
    // is NOT the subset of the tessellation. Measured on a 32^3 pancake, stitching
    // '--partition 2 2 2 --partNo 0..7' against a full-box run gave 626% max error with 3.5% of
    // cells entirely uncovered, and '--region' gave 99.7% of cells wrong -- both at exit code 0.
    // The CORRECT PS path is the Lagrangian partition loop in DTFE.cpp, which is gated on
    // 'partNo<0', so --partNo silently bypasses it entirely.
    // NOTE: this is not a memory route either. A Lagrangian sub-box maps to a displaced, distorted
    // Eulerian region, so partitions overlap and must be SUMMED into the shared full grid --
    // cropping the OUTPUT window without cropping the PARTICLES would be a new feature.
    if ( vm.count("partNo") )
        throwError( "'--partNo' is not supported by PS-DTFE. It selects this partition's particles by their EULERIAN position, but PS-DTFE builds the tessellation in LAGRANGIAN space, so the result is silently wrong (measured: 626% error, 3.5% of cells uncovered) rather than merely partitioned. Use '--partition' WITHOUT '--partNo': PS-DTFE partitions in Lagrangian space internally and sums the partitions itself." );
    if ( this->regionOn or this->regionMpcOn )
        throwError( "'--region'/'--regionMpc' are not supported by PS-DTFE: they crop the particles to an EULERIAN box, but the tessellation is built in LAGRANGIAN space, so the interpolated field is silently wrong (measured: 99.7% of cells). Run the full box; to bound memory use '--partition' (Lagrangian, correct) instead." );
#else
    // the flags parse in both binaries (shared help), but only PS-DTFE has streams to write
    if ( this->psPerStream or this->psPerStreamIds )
        throwError( "'--per-stream'/'--per-stream-ids' need the phase-space stream decomposition; they are PS-DTFE only (this standard DTFE binary reports a 0/1 point coverage in '.pts_streams')." );
    // vm.count, not partitionOn: the partition options are folded into members further down
    if ( not this->psSamplePointsFile.empty() and (vm.count("partition") or vm.count("partNo")) )
        throwError( "In the standard DTFE binary '--sample-points' evaluates a single triangulation; it cannot be combined with '--partition' (PS-DTFE supports partitioned point evaluation)." );
#endif
    if ( this->psPerStreamIds )
        this->psPerStream = true;   // the ids file indexes into the per-stream ragged layout
    if ( not this->psSamplePointsFile.empty() )
    {
        if ( vm.count("interlace") )
            throwError( "'--sample-points' cannot be combined with '--interlace' (interlacing runs the interpolation twice, which would double-count the point streams)." );
        if ( vm.count("NGP") or vm.count("CIC") or vm.count("TSC") or vm.count("PCS") or vm.count("SPH") or vm.count("Voronoi") )
            throwError( "'--sample-points' needs the (PS-)DTFE interpolation method; it is not available with NGP/CIC/TSC/PCS/SPH/Voronoi." );
    }
    else if ( this->psPerStreamIds )
        throwError( "'--per-stream-ids' requires '--sample-points'." );
    else if ( this->psPerStream )
        throwError( "'--per-stream' requires '--sample-points'." );
    else if ( this->psPtsDenGrad )
        throwError( "'--pts-den-grad' requires '--sample-points'." );
    else if ( this->psPtsVelGrad )
        throwError( "'--pts-vel-grad' requires '--sample-points'." );

#ifdef PHASE_SPACE
    // caustic flagging (--ps-caustics) options
    if ( this->psCaustics )
    {
#if NO_DIM!=3
        throwError( "'--ps-caustics' is only available in 3D builds (NO_DIM==3)." );
#endif
        if ( vm.count("interlace") )
            throwError( "'--ps-caustics' cannot be combined with '--interlace' (interlacing runs the interpolation twice, which would corrupt the per-cell orientation bits)." );
        if ( vm.count("NGP") or vm.count("CIC") or vm.count("TSC") or vm.count("PCS") or vm.count("SPH") or vm.count("Voronoi") )
            throwError( "'--ps-caustics' needs the PS-DTFE interpolation method; it is not available with NGP/CIC/TSC/PCS/SPH/Voronoi." );
    }

    // halo-interior release (--ps-halo-release) options
    if ( vm.count("ps-halo-release") and this->psHaloRelease <= Real(0.) )
        throwError( "'--ps-halo-release' needs a positive threshold D (geometric stream density in rho/rho_bar units, e.g. '--ps-halo-release 300')." );

    // volume-weighted velocity moments (--ps-volume-weighted) options
    if ( this->psVolumeWeighted and this->psLinearDeposit )
        throwError( "'--ps-volume-weighted' (equal volume shares for the velocity moments) cannot be combined with '--ps-linear-deposit' (density-weighted shares inside each tetrahedron); the two prescribe opposite sample weightings." );

    // exact conservative deposit (--ps-exact-deposit) options
    if ( this->psExactDeposit )
    {
#if NO_DIM!=3
        throwError( "'--ps-exact-deposit' is only available in 3D builds (NO_DIM==3; r3d is a 3D clipper)." );
#endif
        if ( vm.count("NGP") or vm.count("CIC") or vm.count("TSC") or vm.count("PCS") or vm.count("SPH") or vm.count("Voronoi") )
            throwError( "'--ps-exact-deposit' needs the PS-DTFE interpolation method; it is not available with NGP/CIC/TSC/PCS/SPH/Voronoi." );
    }
#endif

    // read the partition options
    if ( vm.count("max-concurrent") and not vm["max-concurrent"].defaulted() )
        this->maxConcurrentOn = true;   // user gave it (possibly 0 = uncapped); auto-tuner must not touch it
    if ( vm.count("partition") )
    {
        this->partitionOn = true;
        int const temp = this->partition.size();
        // build a new vector rather than self-assign: assign(n, partition.at(0)) reads the fill
        // value through a reference into the vector being reallocated (UB -> garbage partition
        // counts -> the partition loop silently runs zero times and writes all-zero grids)
        if ( temp==1 ) this->partition = std::vector<size_t>( NO_DIM, this->partition.at(0) );
        else if ( temp!=3 ) throwError( "You can only insert 1 or ", NO_DIM, " values for the '--partition' option (e.g. '--partition 3' or '--partition 2 3 3')." );
        
        for (size_t i=0; i<this->partition.size(); ++i)
            lowerBoundCheck( this->partition[i], size_t(1), "the values for the option '--partition'" );
        
        size_t temp2 = 1;
        for (size_t i=0; i<this->partition.size(); ++i)
            temp2 *= this->partition[i];
        if ( vm.count("partNo") ) intervalCheck( this->partNo, 0, int(temp2-1), "'--partNo' program option" );
        
        if ( temp2==1 ) // no actual partition
            this->partitionOn = false;
    }
    
    
    // read the padding options
    if ( vm.count("paddingMpc") )
        this->paddingMpcOn = true;
    if ( vm.count("paddingMpc") or vm.count("padding") )
    {
        this->paddingOn = true;
        size_t temp = 0;
        if (this->paddingMpcOn)
        {
            temp = vm["paddingMpc"].as< std::vector<Real> >().size();
            this->paddingLength.coords = vm["paddingMpc"].as< std::vector<Real> >();
        }
        else
        {
            temp = vm["padding"].as< std::vector<Real> >().size();
            if ( temp==1 )
            {
                paddingParticles = vm["padding"].as< std::vector<Real> >().at(0);
                lowerBoundCheck( paddingParticles, Real(0.), "'--padding' program option" );
                temp = 2*NO_DIM;
            }
            else
                paddingLength.coords = vm["padding"].as< std::vector<Real> >();
        }
        if ( temp!=2*NO_DIM )
            throwError( "You have to insert 1 or ", 2*NO_DIM, " values for the '--padding' or '--paddingMpc' option (e.g. '--padding 0.1 0.2 0.1 0.1 0.3 0.5')." );
    }
    if ( vm.count("noTest") ) this->testPaddedBoundaries = false;
    
    
    // read the averaging options
    intervalCheck( this->method, 1, 3, "'--method' can have only 3 values (from 1 to 3) since there are implemented only 3 methods for field averaging inside the sampling cell (see '--help' for additional details)" );
    if ( this->useMetal and this->method!=1 )
    {
        MESSAGE::Warning warning( this->verboseLevel );
        warning << "--gpu accelerates only the method-1 ('--method 1') volume-averaged interpolation; method " << this->method << " will run on the CPU.\n" << MESSAGE::EndWarning;
    }
    if ( vm.count("samples") )
    {
        this->noPointsOn = true;
        lowerBoundCheck( this->noPoints, 1, "value of the '-s' ['--samples'] option in the program command line options" );
    }
    else if ( this->method==2 )
        this->noPoints = 20;  // default for method 2
    else if ( this->method==3 )
        this->noPoints = 27;  // default for method 3
    if ( vm.count("density0") )
        lowerBoundCheck( averageDensity, Real(0.), "the value of '--density0'" );
    if ( vm.count("seed") )
        lowerBoundCheck( this->randomSeed, size_t(0), "value suplied with program option '--seed'" );
    else
    {
        std::srand( (unsigned)time(0) );
        this->randomSeed = std::rand();
    }
    
    
    // read the redshift cone options
    if ( vm.count("redshiftCone") )
    {
        redshiftConeOn = true;
        size_t temp = vm["redshiftCone"].as< std::vector<Real> >().size();
        if ( temp!=2*NO_DIM ) throwError( "The '--redshiftCone' option must be followed by ", 2*NO_DIM, " values which give the extension ", (NO_DIM==2 ? "(r_min, r_max, psi_min, psi_max)" : "(r_min, r_max, theta_min, theta_max, psi_min, psi_max)" ), " for the spherical coordinates region used to interpolate to grid."  );
        redshiftCone.coords = vm["redshiftCone"].as< std::vector<Real> >();
        
        for (int i=0; i<NO_DIM; ++i)
            if ( redshiftCone[2*i]>=redshiftCone[2*i+1] ) throwError( "When inserting the lower and upper values for option '--redshiftCone'. A lower value is higher than an upper value. If error was due to an angular interval value, increase the upper bound by 360 degrees." );
        if (NO_DIM==3)
        {
            intervalCheck( redshiftCone[2], Real(0.), Real(180.), "3rd value of option '--redshiftCone'" );
            intervalCheck( redshiftCone[3], Real(0.), Real(180.), "4th value of option '--redshiftCone'" );
        }
        Real tempRes = redshiftCone[2*(NO_DIM-1)+1] - redshiftCone[2*(NO_DIM-1)]; // psi_max - psi_min
        if ( tempRes>=Real(360.) ) throwError( "The 'psi' angle interval can strech at most 360 degrees." );
    }
    if ( vm.count("origin") )
        if ( originPosition.size()!=NO_DIM ) throwError( "The option '--origin' must be followed by ", NO_DIM, " values." );
    
    
    // read the additional options
    if ( vm.count("NGP") )
    {
        this->NGP = true;
        this->DTFE = false;
    }
    if ( vm.count("CIC") )
    {
        this->CIC = true;
        this->DTFE = false;
    }
    if ( vm.count("TSC") )
    {
        this->TSC = true;
        this->DTFE = false;
    }
    if ( vm.count("PCS") )
    {
        this->PCS = true;
        this->DTFE = false;
    }
    if ( vm.count("SPH") )
    {
        this->SPH = true;
        this->DTFE = false;
    }
    if ( vm.count("Voronoi") )
    {
        this->Voronoi = true;
        // Voronoi uses the DTFE pipeline (needs triangulation) so keep DTFE=true, density-only output
        this->aField.density = true;
    }
    if ( vm.count("interlace") )
        this->interlace = true;
    if ( vm.count("MpcUnit") )
        lowerBoundCheck( MpcValue, Real(0.), "value of option '--MpcUnit'" );
    if ( vm.count("extensive") )
        this->extensive = true;
#ifndef PHASE_SPACE
    if ( vm.count("approxPSD") )
    {
        this->approxPSD = true;
        // auto-enable scalar field output
        if ( not this->uField.scalar )
            this->uField.scalar = true;
    }
#endif
    if ( vm.count("verbose") )
        intervalCheck( verboseLevel, 0, 3, "value of option '--verbose'" );
    if ( vm.count("randomSample") )
        intervalCheck( randomSample, Real(0.), Real(1.), "value of option '--randomSample'" );
    if ( vm.count("poisson") )
        lowerBoundCheck( poisson, size_t(1), "value of option '--poisson'" );
#ifdef REDSHIFT_SPACE
    if ( vm.count("redshiftSpace") )
    {
        transformToRedshiftSpaceOn = true;
        transformToRedshiftSpace = vm["redshiftSpace"].as< std::vector<Real> >();
        size_t temp = vm["redshiftSpace"].as< std::vector<Real> >().size();
        if ( temp!=NO_DIM ) throwError( "The '--redshiftSpace' option must be followed by ", NO_DIM, " values which give the direction ", (NO_DIM==2 ? "(d1,d2)" : "(d1,d2,d3)" ), " of the vector used to transform from position-space to redshift-space."  );
        
        // normalize the direction to a unit vector
        Real length = transformToRedshiftSpace[0]*transformToRedshiftSpace[0] + transformToRedshiftSpace[1]*transformToRedshiftSpace[1];
        length += ( NO_DIM==2 ? 0 : transformToRedshiftSpace[2]*transformToRedshiftSpace[2] );
        length = std::sqrt( length );
        for (int i=0; i<NO_DIM; ++i)
            transformToRedshiftSpace[i] /= length;
    }
    else
#endif
    {
        transformToRedshiftSpaceOn = false;
        transformToRedshiftSpace.assign( NO_DIM, Real(0.) );
    }
    
    
    // set values to userOptions->programOptions
    for (int i=0; i<argc; ++i)
        this->programOptions += string( argv[i] ) + " ";
    
    
    // disable options that were turned off at compile time
#ifndef TEST_PADDING
    this->testPaddedBoundaries = false;
#endif
#ifndef VELOCITY
    this->uField.deselectVelocity();
    this->aField.deselectVelocity();
#endif
#ifndef SCALAR
    this->uField.deselectScalar();
    this->aField.deselectScalar();
#endif
    
    // special settings for the NGP/CIC/TSC/PCS/SPH methods
    if ( this->NGP or this->CIC or this->TSC or this->PCS or this->SPH )
    {
        // these methods produce only cell-averaged fields, so promote each unaveraged request to its '_a' form
        if ( this->uField.density ) this->aField.density = true;
        if ( this->uField.velocity ) this->aField.velocity = true;
        if ( this->uField.velocity_gradient ) this->aField.velocity_gradient = true;
        if ( this->uField.velocity_divergence ) this->aField.velocity_divergence = true;
        if ( this->uField.velocity_shear ) this->aField.velocity_shear = true;
        if ( this->uField.velocity_vorticity ) this->aField.velocity_vorticity = true;
        if ( this->uField.scalar ) this->aField.scalar = true;
        if ( this->uField.scalar_gradient ) this->aField.scalar_gradient = true;
        this->uField = Field();
        
        // these methods cannot compute the velocity gradient
        if ( this->aField.velocity_gradient or this->aField.selectedVelocityDerivatives() )
        {
            MESSAGE::Warning warning( verboseLevel );
            warning << "The NGP, CIC, TSC, PCS or SPH grid interpolation methods do not have implemented a method for computing the velocity gradient. The velocity gradient will not be computed!" << MESSAGE::EndWarning;
            this->aField.velocity_gradient = false;
            this->aField.deselectVelocityDerivatives(); // also turns off divergence, shear, vorticity
        }
        if ( this->aField.scalar_gradient )
        {
            MESSAGE::Warning warning( verboseLevel );
            warning << "The NGP, CIC, TSC, PCS or SPH grid interpolation methods do not have implemented a method for computing the velocity gradient. The velocity gradient will not be computed!" << MESSAGE::EndWarning;
            this->aField.scalar_gradient = false;
        }
        if ( (this->NGP or this->CIC or this->TSC or this->PCS) and this->aField.scalar )
        {
            MESSAGE::Warning warning( verboseLevel );
            warning << "The CIC and TSC grid interpolation method does not have implemented a method for computing the scalar fields on the grid. The scalar field cannot be computed!" << MESSAGE::EndWarning;
            this->aField.scalar = false;
        }
    }
    
    
    // check to see that there is at least one task selected
    if ( not (this->uField.selected() or this->uField.triangulation or this->aField.selected() or this->aField.triangulation) )
        throwError( "No valid field interpolation quantities were selected for the computation. The program cannot continue since it does not compute anything." );
    
    
    if (showOptions)
        this->printOptions();
}


// Sets 'boxCoordinates' to newBox and recomputes the per-axis 'fullBoxOffset' and 'fullBoxLength'.
void User_options::updateFullBox(Box &newBox)
{
    boxCoordinates = newBox;
    fullBoxOffset.clear();
    fullBoxOffset.reserve(NO_DIM);
    fullBoxLength.clear();
    fullBoxLength.reserve(NO_DIM);
    for (size_t i=0; i<NO_DIM; ++i)
    {
        fullBoxOffset.push_back( boxCoordinates[2*i] );
        fullBoxLength.push_back( boxCoordinates[2*i+1] - boxCoordinates[2*i] );
    }
}




// After input is read: resolves box/grid/region/padding, applies method overrides, and error-checks the options.
void User_options::updateEntries(size_t const noTotalParticles,
                                 bool userSampling)
{
    // check full box size and grid values
    if ( this->boxCoordinates.size()!=2*NO_DIM )
        throwError( "Failed a consistency check. The box encompasing the data should have ", 2*NO_DIM, " coordinates, but it has ", this->boxCoordinates.size(), " coordinates. Check again the values supplied as the coordinates of the full data box." );
    if ( this->boxCoordinates.volume()==0. )
        throwError( "Failed a consistency check. The box encompasing the data has 0. volume. Probably you forget to initialize the coordinates of the full particle data box." );
    
    if ( (this->gridSize.empty() and not userSampling) or this->gridSize.size()!=NO_DIM )
        throwError( "Failed a consistency check. The array storing the interpolation grid should have ", NO_DIM, " values, but it has ", this->gridSize.empty()?0:this->gridSize.size(), " values. Check again the values supplied as the size of the interpolation grid." );
    if ( not gridSize.empty() )
        for (size_t i=0; i<this->gridSize.size(); ++i)
            lowerBoundCheck( this->gridSize[i], size_t(1), "the values of option '--grid'" );
    
    
    // update paddedBox, fullBoxOffset, fullBoxLength
    paddedBox = boxCoordinates;
    this->updateFullBox( boxCoordinates );  // computes 'fullBoxOffset' and 'fullBoxLength'

    updatePadding( noTotalParticles );


    // set the 'region' variable
    if ( regionOn and not regionMpcOn )
    {
        for (size_t i=0; i<region.size(); ++i )
            region[i] = fullBoxOffset[i%2] + region[i] * fullBoxLength[i%2];
        regionMpcOn = true;
    }
    else if ( not regionOn )
        region = boxCoordinates;
    
    
    // update the user sampling points
    userDefinedSampling = userSampling;
    if ( userSampling )
    {
        redshiftConeOn = false;
        if ( method!=2 and aField.selected() )
        {
            method = 2;
            if ( not noPointsOn ) noPoints = 20;
            MESSAGE::Warning warning( verboseLevel );
            warning << "When computing the volume average of the fields on a user defined grid only volume averaging method 2 is available. The program will use volume averaging method '2 - Monte Carlo sampling using random points inside the grid cells' using '"<< noPoints <<"' random samples in each user given grid cell." << MESSAGE::EndWarning;
        }
    }
    else if ( redshiftConeOn )
    {
        if ( method!=2 and aField.selected() )
        {
            method = 2;
            if ( not noPointsOn ) noPoints = 20;
            MESSAGE::Warning warning( verboseLevel );
            warning << "When computing the volume average of the fields on a redshift cone grid only volume averaging method 2 is available. The program will use volume averaging method '2 - Monte Carlo sampling using random points inside the grid cells' using '"<< noPoints <<"' random samples in each grid cell." << MESSAGE::EndWarning;
        }
    }
    
    
    // check that the option "--partition" is disabled when dealing with redshift cone or used defined sampling coordinates
    if ( partitionOn and (userSampling or redshiftConeOn) )
    {
        MESSAGE::Warning warning( verboseLevel );
        warning << "The option '--partition' is not available when interpolating the fields to a redshift cone grid or to user defined sampling points. This '--partition' option will be disabled in the rest of the program. If the computation is too large for the available RAm we advise that you mannually split the computation in manageable data portions." << MESSAGE::EndWarning;
        partitionOn = false;
        for (int i=0; i<NO_DIM; ++i)
            partition[i] = 1;
        partNo = -1;
    }
    
    
    // additional error checking
    if ( redshiftConeOn )
    {
        if (NO_DIM==3)
        {
            intervalCheck( redshiftCone[2], Real(0.), Real(180.), "3rd value of option '--redshiftCone'" );
            intervalCheck( redshiftCone[3], Real(0.), Real(180.), "4th value of option '--redshiftCone'" );
        }
        Real tempRes = redshiftCone[2*(NO_DIM-1)+1] - redshiftCone[2*(NO_DIM-1)]; // psi_max - psi_min
        if ( tempRes>=Real(360.) ) throwError( "The 'psi' angle interval can strech at most 360 degrees." );
        if ( originPosition.empty() or originPosition.size()!=NO_DIM ) throwError( "The vector 'User_options::originPosition' must have ", NO_DIM, " entries." );
    }
    if ( randomSample>=Real(0.) )
        intervalCheck( randomSample, Real(0.), Real(1.), "value of '--randomSample'" );
    
    
    // for equidistant points, the sample count must be a perfect square/cube
    if ( method==3 )
        rootN( noPoints, NO_DIM );
    
    // check the values of the program constants defined in "define.h"
    if ( noVelComp!=NO_DIM )
        throwError( "The program constant 'noVelComp' must have the same value as the number of spatial dimensions. Please check the value of the constant in file 'define.h'." );
    if ( noGradComp != NO_DIM*NO_DIM )
        throwError( "The number of components of the velocity gradient must have the value (number of spatial dimensions)^2. Please check the value of the constant in file 'define.h'." );
    if ( noShearComp != (NO_DIM*(NO_DIM+1))/2-1 )
        throwError( "The number of components of the velocity shear must be 2 in 2D and 5 in 3D. Please check the value of the constant in file 'define.h'." );
    if ( noVortComp != ((NO_DIM-1)*NO_DIM)/2 )
        throwError( "The number of components of the velocity vorticity must be 1 in 2D and 3 in 3D. Please check the value of the constant in file 'define.h'." );
    if ( noScalarGradComp != noScalarComp*NO_DIM )
        throwError( "The number of components of the scalar values gradient must be the number of dimensions times the number of scalar components (which is ", noScalarComp*NO_DIM, " for the given parameters). Please check the value of the constant in file 'define.h'." );
    
    
    // these methods cannot compute gradients or scalar fields
    if ( NGP or CIC or TSC or PCS or SPH )
    {
        if ( aField.velocity_gradient or aField.selectedVelocityDerivatives() )
            throwError( "The NGP, CIC, TSC, PCS and SPH grid interpolation methods do not have implemented a method for computing the velocity gradient. The velocity gradient cannot be computed!");
        if ( aField.scalar_gradient )
            throwError( "The NGP, CIC, TSC, PCS and SPH grid interpolation methods do not have implemented a method for computing the scalar fields gradients. The scalar gradient cannot be computed!");
        if ( (CIC or TSC or PCS) and aField.scalar )
            throwError( "The NGP, CIC, TSC and PCS grid interpolation methods do not have implemented a method for computing the scalar fields on the grid. The scalar field cannot be computed!");
    }
    
    
    // check that the options are disabled in the absence of the VELOCITY and SCALAR compiler directives
#ifndef VELOCITY
    if ( uField.selectedVelocity() or aField.selectedVelocity() )
        throwError( "Compiler directive 'VELOCITY' not detected. You cannot interpolate the velocity and/or velocity related quantities if the compiler directive 'VELOCITY' is not activated." );
#endif
#ifndef SCALAR
    if ( uField.selectedScalar() or aField.selectedScalar() )
        throwError( "Compiler directive 'SCALAR' not detected. You cannot interpolate the scalar data and/or scalar gradient if the compiler directive 'SCALAR' is not activated." );
#endif
#ifndef TEST_PADDING
    if ( testPaddedBoundaries )
        throwError( "Compiler directive 'TEST_PADDING' not detected. You cannot test the padding efficiency if the compiler directive 'TEST_PADDING' is not activated." );
#endif
}



// Resolves the per-axis padding length in Mpc from the particle count or a box-relative input.
void User_options::updatePadding(size_t const noTotalParticles)
{
    if ( paddingLength.isNullBox() )
    {
        if ( paddingParticles<=Real(0.) )   // no padding requested
            paddingLength.assign( Real(0.) );
        else        // derive the padding from the requested particle-count per face
        {
            // DTFE/SPH: 'paddingParticles' grid spacings, scaled by the mean inter-particle distance
            if ( this->DTFE or this->SPH )
                for (int i=0; i<2*NO_DIM; ++i)
                    paddingLength[i] = paddingParticles / Real( pow(noTotalParticles,1./NO_DIM) ) * fullBoxLength[i%2];
            else if ( this->TSC or this->NGP )  // one grid cell suffices for these stencils
                for (int i=0; i<2*NO_DIM; ++i)
                    paddingLength[i] = fullBoxLength[i%2] / gridSize[i%2];
            else if ( this->CIC )               // CIC needs no padding
                for (int i=0; i<2*NO_DIM; ++i)
                    paddingLength[i] = 0.;
        }
        paddingOn = true;
        paddingMpcOn = true;
    }
    else if ( paddingOn and not paddingMpcOn )  // convert box-relative padding to Mpc
    {
        for (int i=0; i<2*NO_DIM; ++i)
            paddingLength[i] *= fullBoxLength[i%2];
        paddingMpcOn = true;
    }
}


