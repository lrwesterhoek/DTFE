# Cross-platform Makefile for compiling the DTFE code on Mac and Linux systems
#
# SUPPORTED ARCHITECTURES:
# ========================
# - x86_64 (Intel/AMD 64-bit) on both macOS and Linux
# - ARM64 (aarch64/Apple Silicon) on both macOS and Linux
# The Makefile automatically detects the architecture and sets appropriate paths.
#
# USAGE:
# ======
# This Makefile automatically detects your operating system and sets appropriate
# defaults for library paths and compiler settings.
#
# macOS (Homebrew):
#   Default paths assume Homebrew installation. Just run 'make DTFE'
#
# Linux:
#   Uses standard system paths (/usr). Install development packages:
#   sudo apt-get install libgsl-dev libboost-all-dev libcgal-dev libmpfr-dev libhdf5-dev libgmp-dev
#   or equivalent for your distribution, then run 'make DTFE'
#
# CUSTOMIZATION:
# ==============
# Override library paths by setting environment variables or make variables:
#   make DTFE GSL_PATH_OVERRIDE=/custom/path/to/gsl
#   make DTFE BOOST_PATH_OVERRIDE=/custom/path/to/boost
#   make DTFE CC_OVERRIDE=g++-11
#
# Or set them as environment variables:
#   export GSL_PATH_OVERRIDE=/custom/path/to/gsl
#   make DTFE
#
# Build in debug mode with sanitizers:
#   make DTFE BUILD_MODE=debug
#   make library BUILD_MODE=debug
#
# Add or override compiler flags:
#   make DTFE EXTRA_FLAGS="-Wno-unused -march=native"
#
# Note: Library headers are included with -I to suppress warnings from
# external code you cannot modify. Your own code warnings will still be shown.
#
# TARGETS:
# ========
# make DTFE                      - Build standard DTFE executable (Eulerian triangulation)
# make PS-DTFE                   - Build PS-DTFE executable (phase-space, Lagrangian triangulation)
# make DTFE BUILD_MODE=debug     - Build with debug symbols, no optimization, and sanitizers
# make library                   - Build the shared library (libDTFE.so/.dylib)
# make library BUILD_MODE=debug  - Build library in debug mode
# make clean                     - Clean object files and executables
#

# Detect operating system
UNAME_S := $(shell uname -s)

# Set platform-specific variables
ifeq ($(UNAME_S),Darwin)  # macOS
    PLATFORM := macos
    SHARED_EXT := .dylib
    EXE_EXT :=
    OBJ_EXT := .o
    LIB_EXT := .a
else ifeq ($(UNAME_S),Linux)
    PLATFORM := linux
    SHARED_EXT := .so
    EXE_EXT :=
    OBJ_EXT := .o
    LIB_EXT := .a
else
    $(error Unsupported operating system: $(UNAME_S). Only macOS and Linux are supported.)
endif

# Platform-specific library paths and compiler settings
ifeq ($(PLATFORM),macos)
    # macOS with Homebrew - detect architecture for correct paths
    ARCH := $(shell uname -m)
    ifeq ($(ARCH),arm64)
        # Apple Silicon (ARM64) - Homebrew installs to /opt/homebrew
        BREW_PREFIX = /opt/homebrew
    else
        # Intel (x86_64) - Homebrew installs to /usr/local
        BREW_PREFIX = /usr/local
    endif

    GSL_PATH   = $(BREW_PREFIX)/opt/gsl
    BOOST_PATH = $(BREW_PREFIX)/opt/boost
    CGAL_PATH  = $(BREW_PREFIX)/opt/cgal
    MPFR_PATH  = $(BREW_PREFIX)/opt/mpfr
    HDF5_PATH  = $(BREW_PREFIX)/opt/hdf5
    GMP_PATH   = $(BREW_PREFIX)/opt/gmp
    FFTW_PATH  = $(BREW_PREFIX)/opt/fftw

    # Try different compiler locations
    CC := $(shell which $(BREW_PREFIX)/opt/llvm/bin/clang++ 2>/dev/null || which clang++ 2>/dev/null || which g++ 2>/dev/null || echo "clang++")
else ifeq ($(PLATFORM),linux)
    # Linux - try to auto-detect common package manager installations
    GSL_PATH   = $(shell pkg-config --variable=prefix gsl 2>/dev/null || echo "/usr")
    BOOST_PATH = /usr
    CGAL_PATH  = /usr
    MPFR_PATH  = /usr
    HDF5_PATH  = /usr
    GMP_PATH = /usr
    FFTW_PATH = /usr
    # Debian/Ubuntu put the serial HDF5 headers and libraries in a sub-directory that is NOT
    # on the default search path (libhdf5-dev installs to /usr/include/hdf5/serial), so a
    # plain -I/usr/include build dies on '#include <H5Cpp.h>'. Pick the sub-directory up
    # automatically when it exists (harmless elsewhere).
    HDF5_EXTRA_INC := $(firstword $(wildcard /usr/include/hdf5/serial))
    HDF5_EXTRA_LIB := $(firstword $(wildcard /usr/lib/*/hdf5/serial))
    CC := $(shell which g++ 2>/dev/null || which clang++ 2>/dev/null || echo "g++")
endif

# Common utilities
MKDIR_P = mkdir -p
RM_RF = rm -rf

# Allow user override of library paths
GSL_PATH   := $(or $(GSL_PATH_OVERRIDE),$(GSL_PATH))
BOOST_PATH := $(or $(BOOST_PATH_OVERRIDE),$(BOOST_PATH))
CGAL_PATH  := $(or $(CGAL_PATH_OVERRIDE),$(CGAL_PATH))
MPFR_PATH  := $(or $(MPFR_PATH_OVERRIDE),$(MPFR_PATH))
HDF5_PATH  := $(or $(HDF5_PATH_OVERRIDE),$(HDF5_PATH))
GMP_PATH   := $(or $(GMP_PATH_OVERRIDE),$(GMP_PATH))
FFTW_PATH  := $(or $(FFTW_PATH_OVERRIDE),$(FFTW_PATH))
CC         := $(or $(CC_OVERRIDE),$(CC))


# paths to where to put the object files and the executables files. If you build the DTFE library than you also need to specify the directory where to put the library and the directory where to copy the header files needed by the library (choose an empty directory for the header files).
OBJ_DIR = ./o
OBJ_DIR_PS = ./o_ps
BIN_DIR = ./
LIB_DIR = ./
INC_DIR = ./DTFE_include

############################# Choose the compiler directives ##################################

############################# Overall options ##################################
# Common options shared by both standard DTFE and PS-DTFE builds
OPTIONS_COMMON =
#------------------------ set the number of spatial dimensions (2 or 3 dimensions)
OPTIONS_COMMON += -DNO_DIM=3
#------------------------ set type of variables - float (comment the next line) or double (uncomment the next line)
# OPTIONS_COMMON += -DDOUBLE

############################# Quantities to be computed ##################################
#------------------------ set which quantities can be computed (can save memory by leaving some out)
# Comment this line if you don't need to compute velocity and velocity related components
OPTIONS_COMMON += -DVELOCITY
# Comment this line if you don't need to interpolate additional fields stored in the scalar variable
OPTIONS_COMMON += -DSCALAR
# number of components of the scalar variable
OPTIONS_COMMON += -DNO_SCALARS=1

############################# Input and output operations default settings ##################################
#------------------------ set which are the default input and output functions for doing data io
# default function to read the input data (101-multiple gadget file, 102-single gadget file, 105-HDF5 gadget file, 111-text file, ... see documentation for more options). The input file type can be set during runtime using the option '--input'. This makefile option only sets a default input file in the case none is given via the program options.
OPTIONS_COMMON += -DINPUT_FILE_DEFAULT=105
# default value for the units of the input data (value=what is 1 Mpc in the units of the data - in this example the data is in kpc). You can change this also during runtime using the program option '--MpcUnit'.
OPTIONS_COMMON += -DMPC_UNIT=1000
# default function to write the output data (101-binary file, 111-text file, ... see documentation for more options). The output file type can be set during runtime using the option '--output'. This makefile option only sets a default output file in the case none is given via the program options.
OPTIONS_COMMON += -DOUTPUT_FILE_DEFAULT=101
#101 for binary file, 100 my density file

############################# additional compiler options ##################################
# enable this option if to use OpenMP (share the workload between CPU cores sharing the same RAM)
OPTIONS_COMMON += -DOPEN_MP
# enable to check if the padding gives a complete Delaunay Tesselation of the region of interest
OPTIONS_COMMON += -DTEST_PADDING
# enable this option to shift from position space to redshift space; You also need to activate this option during run-time using '--redshift-space arguments'
OPTIONS_COMMON += -DREDSHIFT_SPACE

# Standard DTFE build: no PHASE_SPACE flag (triangulates in Eulerian space, standard density)
OPTIONS = $(OPTIONS_COMMON)

# PS-DTFE build: includes PHASE_SPACE flag (triangulates in Lagrangian space, multi-stream regions)
OPTIONS_PS = $(OPTIONS_COMMON) -DPHASE_SPACE

# ---- GPU backend selection (choose at most one): METAL=1 (macOS/Apple Silicon),
# CUDA=1 (Linux/NVIDIA, needs nvcc), HIP=1 (Linux/AMD ROCm, needs hipcc).
# Any backend defines DTFE_GPU / PS_GPU (the backend-neutral guards used by the callers)
# plus GPU_BACKEND_NAME, and provides the host object + link libraries per build.
# Run with '--gpu' (standard DTFE) / '--ps-gpu' (PS-DTFE); --metal/--ps-metal are legacy
# synonyms. Metal embeds+runtime-compiles the kernels in metal/*.metal; CUDA/HIP compile
# the single-source hosts src/CGAL_triangulation/{ps,dtfe}_gpu_cuda.cu ahead of time.
GPU_MODE = off
ifeq ($(METAL),1)
GPU_MODE = metal
endif
ifeq ($(CUDA),1)
ifneq ($(GPU_MODE),off)
$(error choose exactly one GPU backend: METAL=1, CUDA=1 or HIP=1)
endif
GPU_MODE = cuda
endif
ifeq ($(HIP),1)
ifneq ($(GPU_MODE),off)
$(error choose exactly one GPU backend: METAL=1, CUDA=1 or HIP=1)
endif
GPU_MODE = hip
endif

DTFE_GPU_OBJS =
DTFE_GPU_L_OBJS =
DTFE_GPU_LIBS =
PS_GPU_OBJS =
PS_GPU_LIBS =
GPU_BUILD_ARG =

ifeq ($(GPU_MODE),metal)
OPTIONS    += -DDTFE_GPU -DGPU_BACKEND_NAME=\"Metal\"
OPTIONS_PS += -DPS_GPU -DGPU_BACKEND_NAME=\"Metal\"
DTFE_GPU_OBJS = $(OBJ_DIR)/dtfe_metal_host$(OBJ_EXT)
DTFE_GPU_L_OBJS = $(OBJ_DIR)/dtfe_metal_host_l$(OBJ_EXT)
DTFE_GPU_LIBS = -framework Metal -framework Foundation
PS_GPU_OBJS = $(OBJ_DIR_PS)/ps_metal_host$(OBJ_EXT)
PS_GPU_LIBS = -framework Metal -framework Foundation
GPU_BUILD_ARG = METAL=1
endif

# GPU_ARCH: optional target architecture. CUDA: nvcc's default (PTX) is forward-portable,
# set e.g. GPU_ARCH=sm_86 only to skip the first-launch JIT. HIP: STRONGLY recommended when
# building on a machine that cannot see the target GPU (login nodes, containers) -- HIP has
# no portable IR, hipcc silently compiles for the build machine's arch (or clang's default),
# and an arch-mismatched binary skips the GPU or can abort at startup on some ROCm versions.
# Find your arch with 'rocminfo | grep gfx' (e.g. GPU_ARCH=gfx90a; multiple: "gfx90a gfx942").
GPU_ARCH ?=

ifeq ($(GPU_MODE),cuda)
NVCC ?= nvcc
CUDA_PATH ?= /usr/local/cuda
GPUXX = $(NVCC)
GPUXX_FLAGS = -O3 -std=c++17 -Xcompiler -fPIC $(if $(GPU_ARCH),-arch=$(GPU_ARCH))
OPTIONS    += -DDTFE_GPU -DGPU_BACKEND_NAME=\"CUDA\"
OPTIONS_PS += -DPS_GPU -DGPU_BACKEND_NAME=\"CUDA\"
DTFE_GPU_OBJS = $(OBJ_DIR)/dtfe_gpu_cuda$(OBJ_EXT)
DTFE_GPU_L_OBJS = $(DTFE_GPU_OBJS)
DTFE_GPU_LIBS = -L$(CUDA_PATH)/lib64 -lcudart
PS_GPU_OBJS = $(OBJ_DIR_PS)/ps_gpu_cuda$(OBJ_EXT)
PS_GPU_LIBS = -L$(CUDA_PATH)/lib64 -lcudart
GPU_BUILD_ARG = CUDA=1
endif

ifeq ($(GPU_MODE),hip)
HIPCC ?= hipcc
ROCM_PATH ?= /opt/rocm
GPUXX = $(HIPCC)
GPUXX_FLAGS = -O3 -std=c++17 -fPIC -x hip $(foreach a,$(GPU_ARCH),--offload-arch=$(a))
OPTIONS    += -DDTFE_GPU -DGPU_BACKEND_NAME=\"HIP\"
OPTIONS_PS += -DPS_GPU -DGPU_BACKEND_NAME=\"HIP\"
DTFE_GPU_OBJS = $(OBJ_DIR)/dtfe_gpu_cuda$(OBJ_EXT)
DTFE_GPU_L_OBJS = $(DTFE_GPU_OBJS)
DTFE_GPU_LIBS = -L$(ROCM_PATH)/lib -lamdhip64
PS_GPU_OBJS = $(OBJ_DIR_PS)/ps_gpu_cuda$(OBJ_EXT)
PS_GPU_LIBS = -L$(ROCM_PATH)/lib -lamdhip64
GPU_BUILD_ARG = HIP=1
endif

# A build must never mix GPU-mode and plain objects (an incremental 'make PS-DTFE' after a
# METAL=1/CUDA=1/HIP=1 build -- or any backend switch -- would silently produce a binary whose
# components disagree about PS_GPU). The mode is stamped in the object dir; when it changes,
# all its objects are wiped first. '.build_mode' records the make argument ("METAL=1" etc.,
# empty for CPU-only) so tests can rebuild in the same mode: make PS-DTFE $$(cat o_ps/.build_mode)
.PHONY: ps_gpu_mode_check
ps_gpu_mode_check:
	@$(MKDIR_P) $(OBJ_DIR_PS)
	@if [ ! -f $(OBJ_DIR_PS)/.gpu_mode_$(GPU_MODE) ]; then \
		echo ">> PS build GPU mode is now '$(GPU_MODE)'; wiping $(OBJ_DIR_PS) to avoid mixed objects"; \
		rm -f $(OBJ_DIR_PS)/*$(OBJ_EXT) $(OBJ_DIR_PS)/*.d $(OBJ_DIR_PS)/ps_deposit_msl.h $(OBJ_DIR_PS)/.gpu_mode_* $(OBJ_DIR_PS)/.metal_mode_* $(OBJ_DIR_PS)/.build_mode; \
		touch $(OBJ_DIR_PS)/.gpu_mode_$(GPU_MODE); \
		printf '%s' "$(GPU_BUILD_ARG)" > $(OBJ_DIR_PS)/.build_mode; \
	fi

# Same guard for the standard build: the mode is stamped in o/ (which also holds the *_l library
# objects); when it changes, all standard objects are wiped first.
.PHONY: dtfe_gpu_mode_check
dtfe_gpu_mode_check:
	@$(MKDIR_P) $(OBJ_DIR)
	@if [ ! -f $(OBJ_DIR)/.gpu_mode_$(GPU_MODE) ]; then \
		echo ">> DTFE build GPU mode is now '$(GPU_MODE)'; wiping $(OBJ_DIR) to avoid mixed objects"; \
		rm -f $(OBJ_DIR)/*$(OBJ_EXT) $(OBJ_DIR)/*.d $(OBJ_DIR)/dtfe_deposit_msl.h $(OBJ_DIR)/.gpu_mode_* $(OBJ_DIR)/.metal_mode_* $(OBJ_DIR)/.build_mode; \
		touch $(OBJ_DIR)/.gpu_mode_$(GPU_MODE); \
		printf '%s' "$(GPU_BUILD_ARG)" > $(OBJ_DIR)/.build_mode; \
	fi

#------------------------ options usefull when using DTFE as a library
# uncomment the line to get access to a function that returns the Delaunay triangulation of the point set
OPTIONS_COMMON += -DTRIANGULATION


############################# Help menu messages options ##################################
#------------------------ compiler directive that affect only the help messages when using the '-h / --help' option (it does not affect the program in any other way)- if the option is uncommented, than it will show that set of options in the help menu
OPTIONS_COMMON += -DFIELD_OPTIONS
OPTIONS_COMMON += -DREGION_OPTIONS
OPTIONS_COMMON += -DPARTITION_OPTIONS
OPTIONS_COMMON += -DPADDING_OPTIONS
OPTIONS_COMMON += -DAVERAGING_OPTIONS
OPTIONS_COMMON += -DREDSHIFT_CONE_OPTIONS
OPTIONS_COMMON += -DADDITIONAL_OPTIONS



OPTIONS_COMMON += -DBOOST_TIMER_ENABLE_DEPRECATED
OPTIONS_COMMON += -DBOOST_ALLOW_DEPRECATED_HEADERS




###############  DO NOT MODIFY BELOW THIS LINE  ###########################
# do not modify below this line
SRC = ./src
INCLUDES =
LIBRARIES =

# Library path setup
ifneq ($(strip $(GSL_PATH)),)
    INCLUDES += -I $(strip $(GSL_PATH))/include
    LIBRARIES += -L$(strip $(GSL_PATH))/lib
endif
ifneq ($(strip $(BOOST_PATH)),)
    INCLUDES += -I $(strip $(BOOST_PATH))/include
    LIBRARIES += -L$(strip $(BOOST_PATH))/lib
endif
ifneq ($(strip $(CGAL_PATH)),)
    INCLUDES += -I $(strip $(CGAL_PATH))/include
    LIBRARIES += -L$(strip $(CGAL_PATH))/lib
endif
ifneq ($(strip $(GMP_PATH)),)
    INCLUDES += -I $(strip $(GMP_PATH))/include
    LIBRARIES += -L$(strip $(GMP_PATH))/lib
endif
ifneq ($(strip $(MPFR_PATH)),)
    INCLUDES += -I $(strip $(MPFR_PATH))/include
    LIBRARIES += -L$(strip $(MPFR_PATH))/lib
endif
ifneq ($(strip $(HDF5_PATH)),)
    INCLUDES += -I $(strip $(HDF5_PATH))/include
    LIBRARIES += -L$(strip $(HDF5_PATH))/lib   # -lhdf5/-lhdf5_cpp added once via HDF5_LIBS (avoid duplicate-library linker warning)
    ifneq ($(strip $(HDF5_EXTRA_INC)),)
        INCLUDES += -I $(strip $(HDF5_EXTRA_INC))
    endif
    ifneq ($(strip $(HDF5_EXTRA_LIB)),)
        LIBRARIES += -L$(strip $(HDF5_EXTRA_LIB))
    endif
    OPTIONS += -DHDF5
    OPTIONS_PS += -DHDF5
endif
ifneq ($(strip $(FFTW_PATH)),)
    INCLUDES += -I $(strip $(FFTW_PATH))/include
    LIBRARIES += -L$(strip $(FFTW_PATH))/lib
endif

# Compiler flags (same for both platforms)
# Build mode can be set with: make DTFE BUILD_MODE=debug
BUILD_MODE ?= release

ifeq ($(BUILD_MODE),debug)
    # Debug build: no optimization, with debug symbols and sanitizers
    BASE_CFLAGS = -O0 -g3 -DDEBUG $(OPTIONS)
    BASE_CFLAGS_PS = -O0 -g3 -DDEBUG $(OPTIONS_PS)
    # Add sanitizers for debug builds (catch memory errors, undefined behavior, etc.)
    SANITIZER_FLAGS = -fsanitize=address -fsanitize=undefined -fsanitize=leak
    DEBUG_FLAGS = $(SANITIZER_FLAGS) -fno-omit-frame-pointer
else
    # Release build: full optimization
    BASE_CFLAGS = -O3 -DNDEBUG $(OPTIONS)
    BASE_CFLAGS_PS = -O3 -DNDEBUG $(OPTIONS_PS)
    DEBUG_FLAGS =
endif

# macOS SDK pinning: after an OS / Command-Line-Tools update the Homebrew clang
# can derive a non-existent SDK (e.g. .../MacOSX26.sdk) and then fail on the
# C/C++ standard headers (mbstate_t, wcschr, FP_NAN, ...). Detect an SDK that
# actually exists and pass it explicitly as -isysroot (a command-line -isysroot
# overrides clang's bad guess). Empty on Linux. Override with MACOS_SDK_OVERRIDE.
MACOS_ISYSROOT =
ifeq ($(PLATFORM),macos)
    MACOS_SDK := $(or $(MACOS_SDK_OVERRIDE),$(shell s=$$(xcrun --show-sdk-path 2>/dev/null); \
        if [ -z "$$s" ] || [ ! -d "$$s" ]; then \
            s=$$(ls -d /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX*.sdk /Library/Developer/CommandLineTools/SDKs/MacOSX*.sdk 2>/dev/null | sort -V | tail -1); \
        fi; \
        [ -n "$$s" ] && [ -d "$$s" ] && printf '%s' "$$s"))
    ifneq ($(strip $(MACOS_SDK)),)
        MACOS_ISYSROOT := -isysroot $(MACOS_SDK)
    endif
endif

# Minimal compiler flags (from original Makefile)
# Additional warnings and quality flags can be enabled in the EXTRA_FLAGS section above
# -Wno-deprecated-declarations: silences Boost.MultiArray's internal use of the
# deprecated boost::array::assign() (third-party headers, not our code).
# -MMD -MP: every compile also writes a .d makefile next to its object with the REAL include
# graph (-MP adds phony targets so a deleted header cannot strand a stale .d). The .d files are
# pulled in by the -include lines below the object-list macros; they supplement the hand-written
# prerequisite lists (which are correct but incomplete), so header edits rebuild every affected
# object without a clean.
COMPILE_FLAGS = $(BASE_CFLAGS) -std=c++17 -Wno-psabi -Wno-cpp -Wno-deprecated-declarations -frounding-math -MMD -MP $(DEBUG_FLAGS) $(EXTRA_FLAGS) $(MACOS_ISYSROOT)
COMPILE_FLAGS_PS = $(BASE_CFLAGS_PS) -std=c++17 -Wno-psabi -Wno-cpp -Wno-deprecated-declarations -frounding-math -MMD -MP $(DEBUG_FLAGS) $(EXTRA_FLAGS) $(MACOS_ISYSROOT)
LINK_FLAGS =
# NOTE: -lboost_system was dropped: Boost.System is header-only since Boost 1.69,
# so recent Homebrew/Boost no longer ship libboost_system ("library not found").
BASE_LIBS = -lboost_thread -lboost_filesystem -lboost_program_options -lgsl -lgslcblas -lm -lgmp -lmpfr -lfftw3f -lfftw3
HDF5_LIBS = -lhdf5 -lhdf5_cpp

# Optional TBB-parallel Delaunay triangulation: `make PS-DTFE TBB=1` (or DTFE TBB=1).
# Enables CGAL's parallel insertion (Parallel_tag data structure + lock grid) so the
# single global tessellation (the no-`--partition` path) is built across cores.
# Requires the 'tbb' package (macOS: `brew install tbb`). Toggling TBB on/off changes
# the DT type, so do a clean rebuild (`make clean`) when switching. Override the TBB
# location with TBB_PATH_OVERRIDE=/path.
ifeq ($(TBB),1)
    ifeq ($(PLATFORM),macos)
        TBB_PATH := $(or $(TBB_PATH_OVERRIDE),$(BREW_PREFIX)/opt/tbb)
    else
        TBB_PATH := $(or $(TBB_PATH_OVERRIDE),/usr)
    endif
    OPTIONS    += -DPARALLEL_TRIANGULATION -DCGAL_LINKED_WITH_TBB
    OPTIONS_PS += -DPARALLEL_TRIANGULATION -DCGAL_LINKED_WITH_TBB
    INCLUDES   += -I $(TBB_PATH)/include
    LIBRARIES  += -L$(TBB_PATH)/lib
    BASE_LIBS  += -ltbb -ltbbmalloc
endif

# Platform-specific OpenMP settings only
ifeq ($(PLATFORM),macos)
    COMPILE_FLAGS += -fopenmp=libomp
    COMPILE_FLAGS_PS += -fopenmp=libomp
    OPENMP_LIB = -lomp
else ifeq ($(PLATFORM),linux)
    COMPILE_FLAGS += -fopenmp
    COMPILE_FLAGS_PS += -fopenmp
    OPENMP_LIB = -lgomp
endif

# Native-CPU tuning: let the compiler vectorise/schedule the interpolation hot loop for
# this exact chip. Pure tuning -- no -ffast-math, so FP results stay bit-identical to a
# generic build (safe for comparing runs). Override with ARCH_FLAGS=..., disable with ARCH_FLAGS=
ifeq ($(PLATFORM),macos)
    ifeq ($(ARCH),arm64)
        ARCH_FLAGS ?= -mcpu=native
    else
        ARCH_FLAGS ?= -march=native
    endif
else
    ARCH_FLAGS ?= -march=native
endif
COMPILE_FLAGS    += $(ARCH_FLAGS)
COMPILE_FLAGS_PS += $(ARCH_FLAGS)

DTFE_INC = $(INCLUDES)

# Linking
ifeq ($(findstring -DHDF5,$(OPTIONS)),-DHDF5)
    DTFE_LIB = $(LIBRARIES) $(BASE_LIBS) $(HDF5_LIBS)   # OpenMP runtime auto-linked by -fopenmp; $(OPENMP_LIB) omitted to avoid duplicate-library warning
else
    DTFE_LIB = $(LIBRARIES) $(BASE_LIBS)   # OpenMP runtime auto-linked by -fopenmp; $(OPENMP_LIB) omitted to avoid duplicate-library warning
endif


IO_SOURCES = $(addprefix io/, input_output.h gadget_reader_header.cc gadget_reader_binary.cc gadget_reader_HDF5.cc text_io.cc binary_io.cc)
MAIN_SOURCES = main.cpp DTFE.h message.h user_options.h io/io.h interlacing.h
IO_CC_SOURCES = input_output.cc $(IO_SOURCES) user_options.h define.h quantities.h message.h particle_data.h box.h ps_point_eval.h
DTFE_SOURCES = DTFE.cpp define.h particle_data.h user_options.h box.h quantities.h subpartition.h interpolations.h kdtree/kdtree2.hpp Pvector.h message.h miscellaneous.h auto_tune.h ps_point_eval.h
DTFE_CC_SOURCES = user_options.cc quantities.cc NGP_interpolation.cc CIC_interpolation.cc TSC_interpolation.cc PCS_interpolation.cc SPH_interpolation.cc interlacing.cc random.cc
TRIANG_HEADERS = $(addprefix CGAL_triangulation/, triangulation_common.h triangulation_miscellaneous.h field_computation.h padding_test.h my_function.h CGAL_include_2D.h CGAL_include_3D.h vertexData.h particle_data_traits.h) define.h particle_data.h user_options.h box.h quantities.h Pvector.h message.h math_functions.h miscellaneous.h ps_point_eval.h
TRIANG_SOURCES = $(addprefix CGAL_triangulation/, triangulation.cpp) $(TRIANG_HEADERS)
TRIANG_CC_SOURCES = $(addprefix CGAL_triangulation/, unaveraged_interpolation.cc averaged_interpolation_1.cc averaged_interpolation_2.cc ps_interpolation.cc ps_point_eval.cc)

ALL_FILES = $(DTFE_SOURCES) $(DTFE_CC_SOURCES) $(TRIANG_SOURCES) $(MAIN_SOURCES) kdtree/kdtree2.hpp kdtree/kdtree2.cpp
LIB_FILES = $(DTFE_SOURCES) $(DTFE_CC_SOURCES) $(TRIANG_SOURCES)

HEADERS_1 = DTFE.h define.h user_options.h particle_data.h quantities.h Pvector.h math_functions.h  message.h box.h miscellaneous.h interpolations.h
HEADERS_2 = $(addprefix CGAL_triangulation/, CGAL_include_2D.h CGAL_include_3D.h vertexData.h particle_data_traits.h)

# compiler-generated header dependencies (see -MMD -MP at COMPILE_FLAGS); wildcard expands to
# nothing on a fresh checkout / after clean, so missing .d files never break the build
-include $(wildcard $(OBJ_DIR)/*.d)
-include $(wildcard $(OBJ_DIR_PS)/*.d)

# Declare phony targets
.PHONY: DTFE PS-DTFE library DTFE-build PS-DTFE-build library-build clean test-platform copy_headers set_directories set_directories_ps set_directories_2


############################# Standard DTFE build (no PHASE_SPACE) ##################################
# Produces DTFE binary that uses Eulerian-space triangulation (standard DTFE)

DTFE_CC_OBJS = $(OBJ_DIR)/user_options$(OBJ_EXT) $(OBJ_DIR)/quantities$(OBJ_EXT) $(OBJ_DIR)/NGP_interpolation$(OBJ_EXT) $(OBJ_DIR)/CIC_interpolation$(OBJ_EXT) $(OBJ_DIR)/TSC_interpolation$(OBJ_EXT) $(OBJ_DIR)/PCS_interpolation$(OBJ_EXT) $(OBJ_DIR)/SPH_interpolation$(OBJ_EXT) $(OBJ_DIR)/interlacing$(OBJ_EXT) $(OBJ_DIR)/random$(OBJ_EXT) $(OBJ_DIR)/scratch_alloc$(OBJ_EXT)
IO_CC_OBJS = $(OBJ_DIR)/input_output$(OBJ_EXT)
TRIANG_CC_OBJS = $(OBJ_DIR)/unaveraged_interpolation$(OBJ_EXT) $(OBJ_DIR)/averaged_interpolation_1$(OBJ_EXT) $(OBJ_DIR)/averaged_interpolation_2$(OBJ_EXT) $(OBJ_DIR)/ps_interpolation$(OBJ_EXT) $(OBJ_DIR)/ps_point_eval$(OBJ_EXT)

# The mode check must FINISH before any object/header rule starts: under -j its wipe runs
# concurrently with sibling prerequisites and can delete files mid-recipe (observed: the
# generated *_msl.h lost its opening raw-string line). The recursive $(MAKE) starts only
# after the check's recipe completes, so the real build can never race the wipe.
DTFE: dtfe_gpu_mode_check
	@$(MAKE) DTFE-build

DTFE-build: set_directories $(OBJ_DIR)/DTFE$(OBJ_EXT) $(OBJ_DIR)/triangulation$(OBJ_EXT) $(OBJ_DIR)/main$(OBJ_EXT) $(OBJ_DIR)/kdtree2$(OBJ_EXT) $(OBJ_DIR)/r3d$(OBJ_EXT) $(DTFE_CC_OBJS) $(IO_CC_OBJS) $(TRIANG_CC_OBJS) $(DTFE_GPU_OBJS) Makefile
	$(CC) $(COMPILE_FLAGS) $(OBJ_DIR)/DTFE$(OBJ_EXT) $(OBJ_DIR)/triangulation$(OBJ_EXT) $(OBJ_DIR)/main$(OBJ_EXT) $(OBJ_DIR)/kdtree2$(OBJ_EXT) $(OBJ_DIR)/r3d$(OBJ_EXT) $(DTFE_CC_OBJS) $(IO_CC_OBJS) $(TRIANG_CC_OBJS) $(DTFE_GPU_OBJS) $(DTFE_LIB) $(DTFE_GPU_LIBS) -o $(BIN_DIR)/DTFE$(EXE_EXT)


$(OBJ_DIR)/main$(OBJ_EXT): $(addprefix $(SRC)/, $(MAIN_SOURCES)) Makefile
	$(CC) $(COMPILE_FLAGS) $(DTFE_INC) -o $(OBJ_DIR)/main$(OBJ_EXT) -c $(SRC)/main.cpp

$(OBJ_DIR)/input_output$(OBJ_EXT): $(addprefix $(SRC)/, $(IO_CC_SOURCES)) Makefile
	$(CC) $(COMPILE_FLAGS) $(DTFE_INC) -o $@ -c $(SRC)/input_output.cc

$(OBJ_DIR)/DTFE$(OBJ_EXT): $(addprefix $(SRC)/, $(DTFE_SOURCES)) Makefile
	$(CC) $(COMPILE_FLAGS) $(DTFE_INC) -o $(OBJ_DIR)/DTFE$(OBJ_EXT) -c $(SRC)/DTFE.cpp

$(OBJ_DIR)/user_options$(OBJ_EXT): $(SRC)/user_options.cc $(SRC)/user_options.h $(SRC)/define.h Makefile
	$(CC) $(COMPILE_FLAGS) $(DTFE_INC) -o $@ -c $(SRC)/user_options.cc

$(OBJ_DIR)/quantities$(OBJ_EXT): $(SRC)/quantities.cc $(SRC)/quantities.h $(SRC)/define.h Makefile
	$(CC) $(COMPILE_FLAGS) $(DTFE_INC) -o $@ -c $(SRC)/quantities.cc

$(OBJ_DIR)/NGP_interpolation$(OBJ_EXT): $(SRC)/NGP_interpolation.cc $(SRC)/interpolations.h $(SRC)/define.h Makefile
	$(CC) $(COMPILE_FLAGS) $(DTFE_INC) -o $@ -c $(SRC)/NGP_interpolation.cc

$(OBJ_DIR)/CIC_interpolation$(OBJ_EXT): $(SRC)/CIC_interpolation.cc $(SRC)/interpolations.h $(SRC)/define.h $(SRC)/grid_deposit_engine.h Makefile
	$(CC) $(COMPILE_FLAGS) $(DTFE_INC) -o $@ -c $(SRC)/CIC_interpolation.cc

$(OBJ_DIR)/TSC_interpolation$(OBJ_EXT): $(SRC)/TSC_interpolation.cc $(SRC)/interpolations.h $(SRC)/define.h $(SRC)/grid_deposit_engine.h Makefile
	$(CC) $(COMPILE_FLAGS) $(DTFE_INC) -o $@ -c $(SRC)/TSC_interpolation.cc

$(OBJ_DIR)/PCS_interpolation$(OBJ_EXT): $(SRC)/PCS_interpolation.cc $(SRC)/interpolations.h $(SRC)/define.h $(SRC)/grid_deposit_engine.h Makefile
	$(CC) $(COMPILE_FLAGS) $(DTFE_INC) -o $@ -c $(SRC)/PCS_interpolation.cc

$(OBJ_DIR)/interlacing$(OBJ_EXT): $(SRC)/interlacing.cc $(SRC)/interlacing.h $(SRC)/define.h Makefile
	$(CC) $(COMPILE_FLAGS) $(DTFE_INC) -o $@ -c $(SRC)/interlacing.cc

$(OBJ_DIR)/SPH_interpolation$(OBJ_EXT): $(SRC)/SPH_interpolation.cc $(SRC)/interpolations.h $(SRC)/define.h Makefile
	$(CC) $(COMPILE_FLAGS) $(DTFE_INC) -o $@ -c $(SRC)/SPH_interpolation.cc

$(OBJ_DIR)/random$(OBJ_EXT): $(SRC)/random.cc $(SRC)/define.h $(SRC)/user_options.h Makefile
	$(CC) $(COMPILE_FLAGS) $(DTFE_INC) -o $@ -c $(SRC)/random.cc

# --scratch-dir out-of-core backing: replaces the global operator new/delete, so it goes into
# the BINARIES only -- the library links scratch_alloc_stub.cc instead (see library-build).
$(OBJ_DIR)/scratch_alloc$(OBJ_EXT): $(SRC)/scratch_alloc.cc $(SRC)/scratch_alloc.h Makefile
	$(CC) $(COMPILE_FLAGS) $(DTFE_INC) -o $@ -c $(SRC)/scratch_alloc.cc

$(OBJ_DIR)/kdtree2$(OBJ_EXT): $(SRC)/kdtree/kdtree2.hpp $(SRC)/kdtree/kdtree2.cpp Makefile
	$(CC) -O3 -ffast-math -fomit-frame-pointer -Wno-deprecated-declarations -MMD -MP $(MACOS_ISYSROOT) $(DTFE_INC) -o $(OBJ_DIR)/kdtree2$(OBJ_EXT) -c $(SRC)/kdtree/kdtree2.cpp

# vendored r3d (third_party/r3d, plain C99; see its README for license/citation): exact
# tetrahedron-cell intersection moments for --ps-exact-deposit / --exact-average. One
# object per dir; -fPIC so the o/ object also links into libDTFE (the CUDA/HIP pattern,
# no _l twin). -x c: $(CC) is a C++ driver.
R3D_SRC = third_party/r3d/r3d.c
R3D_FLAGS = -x c -std=c99 -O3 -fPIC -fomit-frame-pointer -MMD -MP $(MACOS_ISYSROOT)

$(OBJ_DIR)/r3d$(OBJ_EXT): $(R3D_SRC) third_party/r3d/r3d.h third_party/r3d/r3d-config.h Makefile
	$(CC) $(R3D_FLAGS) -o $@ -c $(R3D_SRC)

$(OBJ_DIR_PS)/r3d$(OBJ_EXT): $(R3D_SRC) third_party/r3d/r3d.h third_party/r3d/r3d-config.h Makefile
	$(CC) $(R3D_FLAGS) -o $@ -c $(R3D_SRC)

$(OBJ_DIR)/triangulation$(OBJ_EXT): $(addprefix $(SRC)/, $(TRIANG_SOURCES)) Makefile
	$(CC) $(COMPILE_FLAGS) $(DTFE_INC) -o $(OBJ_DIR)/triangulation$(OBJ_EXT) -c $(SRC)/CGAL_triangulation/triangulation.cpp

$(OBJ_DIR)/unaveraged_interpolation$(OBJ_EXT): $(SRC)/CGAL_triangulation/unaveraged_interpolation.cc $(addprefix $(SRC)/, $(TRIANG_HEADERS)) Makefile
	$(CC) $(COMPILE_FLAGS) $(DTFE_INC) -o $@ -c $(SRC)/CGAL_triangulation/unaveraged_interpolation.cc

$(OBJ_DIR)/averaged_interpolation_1$(OBJ_EXT): $(SRC)/CGAL_triangulation/averaged_interpolation_1.cc $(SRC)/CGAL_triangulation/gpu_host.h $(addprefix $(SRC)/, $(TRIANG_HEADERS)) Makefile
	$(CC) $(COMPILE_FLAGS) $(DTFE_INC) -o $@ -c $(SRC)/CGAL_triangulation/averaged_interpolation_1.cc

$(OBJ_DIR)/averaged_interpolation_2$(OBJ_EXT): $(SRC)/CGAL_triangulation/averaged_interpolation_2.cc $(addprefix $(SRC)/, $(TRIANG_HEADERS)) Makefile
	$(CC) $(COMPILE_FLAGS) $(DTFE_INC) -o $@ -c $(SRC)/CGAL_triangulation/averaged_interpolation_2.cc

$(OBJ_DIR)/ps_interpolation$(OBJ_EXT): $(SRC)/CGAL_triangulation/ps_interpolation.cc $(addprefix $(SRC)/, $(TRIANG_HEADERS)) Makefile
	$(CC) $(COMPILE_FLAGS) $(DTFE_INC) -o $@ -c $(SRC)/CGAL_triangulation/ps_interpolation.cc

$(OBJ_DIR)/ps_point_eval$(OBJ_EXT): $(SRC)/CGAL_triangulation/ps_point_eval.cc $(addprefix $(SRC)/, $(TRIANG_HEADERS)) Makefile
	$(CC) $(COMPILE_FLAGS) $(DTFE_INC) -o $@ -c $(SRC)/CGAL_triangulation/ps_point_eval.cc


############################# PS-DTFE build (with PHASE_SPACE) ##################################
# Produces PS-DTFE binary that uses Lagrangian-space triangulation (phase-space DTFE)
# Object files are placed in $(OBJ_DIR_PS) to avoid conflicts with the standard DTFE build

PS_DTFE_CC_OBJS = $(OBJ_DIR_PS)/user_options$(OBJ_EXT) $(OBJ_DIR_PS)/quantities$(OBJ_EXT) $(OBJ_DIR_PS)/NGP_interpolation$(OBJ_EXT) $(OBJ_DIR_PS)/CIC_interpolation$(OBJ_EXT) $(OBJ_DIR_PS)/TSC_interpolation$(OBJ_EXT) $(OBJ_DIR_PS)/PCS_interpolation$(OBJ_EXT) $(OBJ_DIR_PS)/SPH_interpolation$(OBJ_EXT) $(OBJ_DIR_PS)/interlacing$(OBJ_EXT) $(OBJ_DIR_PS)/random$(OBJ_EXT) $(OBJ_DIR_PS)/scratch_alloc$(OBJ_EXT)
PS_DTFE_IO_OBJS = $(OBJ_DIR_PS)/input_output$(OBJ_EXT)
PS_DTFE_TRIANG_OBJS = $(OBJ_DIR_PS)/unaveraged_interpolation$(OBJ_EXT) $(OBJ_DIR_PS)/averaged_interpolation_1$(OBJ_EXT) $(OBJ_DIR_PS)/averaged_interpolation_2$(OBJ_EXT) $(OBJ_DIR_PS)/ps_interpolation$(OBJ_EXT) $(OBJ_DIR_PS)/ps_point_eval$(OBJ_EXT)

# Same wipe-vs-build serialization as the DTFE target above.
PS-DTFE: ps_gpu_mode_check
	@$(MAKE) PS-DTFE-build

PS-DTFE-build: set_directories_ps $(OBJ_DIR_PS)/DTFE$(OBJ_EXT) $(OBJ_DIR_PS)/triangulation$(OBJ_EXT) $(OBJ_DIR_PS)/main$(OBJ_EXT) $(OBJ_DIR_PS)/kdtree2$(OBJ_EXT) $(OBJ_DIR_PS)/r3d$(OBJ_EXT) $(PS_DTFE_CC_OBJS) $(PS_DTFE_IO_OBJS) $(PS_DTFE_TRIANG_OBJS) $(PS_GPU_OBJS) Makefile
	$(CC) $(COMPILE_FLAGS_PS) $(OBJ_DIR_PS)/DTFE$(OBJ_EXT) $(OBJ_DIR_PS)/triangulation$(OBJ_EXT) $(OBJ_DIR_PS)/main$(OBJ_EXT) $(OBJ_DIR_PS)/kdtree2$(OBJ_EXT) $(OBJ_DIR_PS)/r3d$(OBJ_EXT) $(PS_DTFE_CC_OBJS) $(PS_DTFE_IO_OBJS) $(PS_DTFE_TRIANG_OBJS) $(PS_GPU_OBJS) $(DTFE_LIB) $(PS_GPU_LIBS) -o $(BIN_DIR)/PS-DTFE$(EXE_EXT)

set_directories_ps:
	@$(MKDIR_P) $(OBJ_DIR_PS)

$(OBJ_DIR_PS)/main$(OBJ_EXT): $(addprefix $(SRC)/, $(MAIN_SOURCES)) Makefile
	$(CC) $(COMPILE_FLAGS_PS) $(DTFE_INC) -o $@ -c $(SRC)/main.cpp

$(OBJ_DIR_PS)/input_output$(OBJ_EXT): $(addprefix $(SRC)/, $(IO_CC_SOURCES)) Makefile
	$(CC) $(COMPILE_FLAGS_PS) $(DTFE_INC) -o $@ -c $(SRC)/input_output.cc

$(OBJ_DIR_PS)/DTFE$(OBJ_EXT): $(addprefix $(SRC)/, $(DTFE_SOURCES)) Makefile
	$(CC) $(COMPILE_FLAGS_PS) $(DTFE_INC) -o $@ -c $(SRC)/DTFE.cpp

$(OBJ_DIR_PS)/user_options$(OBJ_EXT): $(SRC)/user_options.cc $(SRC)/user_options.h $(SRC)/define.h Makefile
	$(CC) $(COMPILE_FLAGS_PS) $(DTFE_INC) -o $@ -c $(SRC)/user_options.cc

$(OBJ_DIR_PS)/quantities$(OBJ_EXT): $(SRC)/quantities.cc $(SRC)/quantities.h $(SRC)/define.h Makefile
	$(CC) $(COMPILE_FLAGS_PS) $(DTFE_INC) -o $@ -c $(SRC)/quantities.cc

$(OBJ_DIR_PS)/NGP_interpolation$(OBJ_EXT): $(SRC)/NGP_interpolation.cc $(SRC)/interpolations.h $(SRC)/define.h Makefile
	$(CC) $(COMPILE_FLAGS_PS) $(DTFE_INC) -o $@ -c $(SRC)/NGP_interpolation.cc

$(OBJ_DIR_PS)/CIC_interpolation$(OBJ_EXT): $(SRC)/CIC_interpolation.cc $(SRC)/interpolations.h $(SRC)/define.h $(SRC)/grid_deposit_engine.h Makefile
	$(CC) $(COMPILE_FLAGS_PS) $(DTFE_INC) -o $@ -c $(SRC)/CIC_interpolation.cc

$(OBJ_DIR_PS)/TSC_interpolation$(OBJ_EXT): $(SRC)/TSC_interpolation.cc $(SRC)/interpolations.h $(SRC)/define.h $(SRC)/grid_deposit_engine.h Makefile
	$(CC) $(COMPILE_FLAGS_PS) $(DTFE_INC) -o $@ -c $(SRC)/TSC_interpolation.cc

$(OBJ_DIR_PS)/PCS_interpolation$(OBJ_EXT): $(SRC)/PCS_interpolation.cc $(SRC)/interpolations.h $(SRC)/define.h $(SRC)/grid_deposit_engine.h Makefile
	$(CC) $(COMPILE_FLAGS_PS) $(DTFE_INC) -o $@ -c $(SRC)/PCS_interpolation.cc

$(OBJ_DIR_PS)/interlacing$(OBJ_EXT): $(SRC)/interlacing.cc $(SRC)/interlacing.h $(SRC)/define.h Makefile
	$(CC) $(COMPILE_FLAGS_PS) $(DTFE_INC) -o $@ -c $(SRC)/interlacing.cc

$(OBJ_DIR_PS)/SPH_interpolation$(OBJ_EXT): $(SRC)/SPH_interpolation.cc $(SRC)/interpolations.h $(SRC)/define.h Makefile
	$(CC) $(COMPILE_FLAGS_PS) $(DTFE_INC) -o $@ -c $(SRC)/SPH_interpolation.cc

$(OBJ_DIR_PS)/random$(OBJ_EXT): $(SRC)/random.cc $(SRC)/define.h $(SRC)/user_options.h Makefile
	$(CC) $(COMPILE_FLAGS_PS) $(DTFE_INC) -o $@ -c $(SRC)/random.cc

$(OBJ_DIR_PS)/scratch_alloc$(OBJ_EXT): $(SRC)/scratch_alloc.cc $(SRC)/scratch_alloc.h Makefile
	$(CC) $(COMPILE_FLAGS_PS) $(DTFE_INC) -o $@ -c $(SRC)/scratch_alloc.cc

$(OBJ_DIR_PS)/kdtree2$(OBJ_EXT): $(SRC)/kdtree/kdtree2.hpp $(SRC)/kdtree/kdtree2.cpp Makefile
	$(CC) -O3 -ffast-math -fomit-frame-pointer -Wno-deprecated-declarations -MMD -MP $(MACOS_ISYSROOT) $(DTFE_INC) -o $@ -c $(SRC)/kdtree/kdtree2.cpp

$(OBJ_DIR_PS)/triangulation$(OBJ_EXT): $(addprefix $(SRC)/, $(TRIANG_SOURCES)) Makefile
	$(CC) $(COMPILE_FLAGS_PS) $(DTFE_INC) -o $@ -c $(SRC)/CGAL_triangulation/triangulation.cpp

$(OBJ_DIR_PS)/unaveraged_interpolation$(OBJ_EXT): $(SRC)/CGAL_triangulation/unaveraged_interpolation.cc $(addprefix $(SRC)/, $(TRIANG_HEADERS)) Makefile
	$(CC) $(COMPILE_FLAGS_PS) $(DTFE_INC) -o $@ -c $(SRC)/CGAL_triangulation/unaveraged_interpolation.cc

$(OBJ_DIR_PS)/averaged_interpolation_1$(OBJ_EXT): $(SRC)/CGAL_triangulation/averaged_interpolation_1.cc $(addprefix $(SRC)/, $(TRIANG_HEADERS)) Makefile
	$(CC) $(COMPILE_FLAGS_PS) $(DTFE_INC) -o $@ -c $(SRC)/CGAL_triangulation/averaged_interpolation_1.cc

$(OBJ_DIR_PS)/averaged_interpolation_2$(OBJ_EXT): $(SRC)/CGAL_triangulation/averaged_interpolation_2.cc $(addprefix $(SRC)/, $(TRIANG_HEADERS)) Makefile
	$(CC) $(COMPILE_FLAGS_PS) $(DTFE_INC) -o $@ -c $(SRC)/CGAL_triangulation/averaged_interpolation_2.cc

$(OBJ_DIR_PS)/ps_interpolation$(OBJ_EXT): $(SRC)/CGAL_triangulation/ps_interpolation.cc $(SRC)/CGAL_triangulation/gpu_host.h $(addprefix $(SRC)/, $(TRIANG_HEADERS)) Makefile
	$(CC) $(COMPILE_FLAGS_PS) $(DTFE_INC) -o $@ -c $(SRC)/CGAL_triangulation/ps_interpolation.cc

$(OBJ_DIR_PS)/ps_point_eval$(OBJ_EXT): $(SRC)/CGAL_triangulation/ps_point_eval.cc $(addprefix $(SRC)/, $(TRIANG_HEADERS)) Makefile
	$(CC) $(COMPILE_FLAGS_PS) $(DTFE_INC) -o $@ -c $(SRC)/CGAL_triangulation/ps_point_eval.cc

# Metal GPU deposit host (METAL=1 only). The kernel source is embedded via a generated header so
# the binary needs no runtime file lookup; Metal compiles it once per process.
$(OBJ_DIR_PS)/ps_deposit_msl.h: metal/ps_deposit.metal Makefile
	@$(MKDIR_P) $(OBJ_DIR_PS)
	{ printf 'static const char PS_DEPOSIT_MSL[] = R"MSL(\n'; cat metal/ps_deposit.metal; printf '\n)MSL";\n'; } > $@.tmp
	mv $@.tmp $@

$(OBJ_DIR_PS)/ps_metal_host$(OBJ_EXT): $(SRC)/CGAL_triangulation/ps_metal_host.cc $(SRC)/CGAL_triangulation/gpu_host.h $(SRC)/CGAL_triangulation/ps_deposit_params.h $(OBJ_DIR_PS)/ps_deposit_msl.h Makefile
	$(CC) $(COMPILE_FLAGS_PS) -I third_party/metal-cpp -I $(OBJ_DIR_PS) -o $@ -c $(SRC)/CGAL_triangulation/ps_metal_host.cc

# Standard-DTFE Metal deposit host (METAL=1 only), same embedding scheme in $(OBJ_DIR).
$(OBJ_DIR)/dtfe_deposit_msl.h: metal/dtfe_deposit.metal Makefile
	@$(MKDIR_P) $(OBJ_DIR)
	{ printf 'static const char DTFE_DEPOSIT_MSL[] = R"MSL(\n'; cat metal/dtfe_deposit.metal; printf '\n)MSL";\n'; } > $@.tmp
	mv $@.tmp $@

$(OBJ_DIR)/dtfe_metal_host$(OBJ_EXT): $(SRC)/CGAL_triangulation/dtfe_metal_host.cc $(SRC)/CGAL_triangulation/gpu_host.h $(OBJ_DIR)/dtfe_deposit_msl.h Makefile
	$(CC) $(COMPILE_FLAGS) -I third_party/metal-cpp -I $(OBJ_DIR) -o $@ -c $(SRC)/CGAL_triangulation/dtfe_metal_host.cc

$(OBJ_DIR)/dtfe_metal_host_l$(OBJ_EXT): $(SRC)/CGAL_triangulation/dtfe_metal_host.cc $(SRC)/CGAL_triangulation/gpu_host.h $(OBJ_DIR)/dtfe_deposit_msl.h Makefile
	$(CC) $(COMPILE_FLAGS) -fPIC -I third_party/metal-cpp -I $(OBJ_DIR) -o $@ -c $(SRC)/CGAL_triangulation/dtfe_metal_host.cc

# CUDA/HIP GPU deposit hosts (CUDA=1 via nvcc, HIP=1 via hipcc): single-source .cu files,
# kernels compiled ahead of time (no vendored SDK -- the CUDA toolkit / ROCm provide headers).
$(OBJ_DIR_PS)/ps_gpu_cuda$(OBJ_EXT): $(SRC)/CGAL_triangulation/ps_gpu_cuda.cu $(SRC)/CGAL_triangulation/gpu_host.h $(SRC)/CGAL_triangulation/gpu_cuda_compat.h $(SRC)/CGAL_triangulation/ps_deposit_params.h Makefile
	$(GPUXX) $(GPUXX_FLAGS) -o $@ -c $(SRC)/CGAL_triangulation/ps_gpu_cuda.cu

$(OBJ_DIR)/dtfe_gpu_cuda$(OBJ_EXT): $(SRC)/CGAL_triangulation/dtfe_gpu_cuda.cu $(SRC)/CGAL_triangulation/gpu_host.h $(SRC)/CGAL_triangulation/gpu_cuda_compat.h Makefile
	$(GPUXX) $(GPUXX_FLAGS) -o $@ -c $(SRC)/CGAL_triangulation/dtfe_gpu_cuda.cu


############################# Shared library build ##################################

DTFE_CC_LIB_OBJS = $(OBJ_DIR)/user_options_l$(OBJ_EXT) $(OBJ_DIR)/quantities_l$(OBJ_EXT) $(OBJ_DIR)/NGP_interpolation_l$(OBJ_EXT) $(OBJ_DIR)/CIC_interpolation_l$(OBJ_EXT) $(OBJ_DIR)/TSC_interpolation_l$(OBJ_EXT) $(OBJ_DIR)/PCS_interpolation_l$(OBJ_EXT) $(OBJ_DIR)/SPH_interpolation_l$(OBJ_EXT) $(OBJ_DIR)/interlacing_l$(OBJ_EXT) $(OBJ_DIR)/random_l$(OBJ_EXT)
IO_CC_LIB_OBJS = $(OBJ_DIR)/input_output_l$(OBJ_EXT)
TRIANG_CC_LIB_OBJS = $(OBJ_DIR)/unaveraged_interpolation_l$(OBJ_EXT) $(OBJ_DIR)/averaged_interpolation_1_l$(OBJ_EXT) $(OBJ_DIR)/averaged_interpolation_2_l$(OBJ_EXT) $(OBJ_DIR)/ps_interpolation_l$(OBJ_EXT) $(OBJ_DIR)/ps_point_eval_l$(OBJ_EXT)

# Same wipe-vs-build serialization as the DTFE target (the library shares $(OBJ_DIR)).
library: dtfe_gpu_mode_check
	@$(MAKE) library-build

library-build: set_directories set_directories_2 $(addprefix $(SRC)/, $(LIB_FILES) ) copy_headers $(OBJ_DIR)/r3d$(OBJ_EXT) $(DTFE_GPU_L_OBJS) Makefile
	$(CC) $(COMPILE_FLAGS) -fPIC $(DTFE_INC) -o $(OBJ_DIR)/DTFE_l$(OBJ_EXT) -c $(SRC)/DTFE.cpp
	$(CC) -O3 -ffast-math -fomit-frame-pointer -fPIC -Wno-deprecated-declarations -MMD -MP $(MACOS_ISYSROOT) $(DTFE_INC) -o $(OBJ_DIR)/kdtree2_l$(OBJ_EXT) -c $(SRC)/kdtree/kdtree2.cpp
	$(CC) $(COMPILE_FLAGS) -fPIC $(DTFE_INC) -o $(OBJ_DIR)/triangulation_l$(OBJ_EXT) -c $(SRC)/CGAL_triangulation/triangulation.cpp
	$(CC) $(COMPILE_FLAGS) -fPIC $(DTFE_INC) -o $(OBJ_DIR)/unaveraged_interpolation_l$(OBJ_EXT) -c $(SRC)/CGAL_triangulation/unaveraged_interpolation.cc
	$(CC) $(COMPILE_FLAGS) -fPIC $(DTFE_INC) -o $(OBJ_DIR)/averaged_interpolation_1_l$(OBJ_EXT) -c $(SRC)/CGAL_triangulation/averaged_interpolation_1.cc
	$(CC) $(COMPILE_FLAGS) -fPIC $(DTFE_INC) -o $(OBJ_DIR)/averaged_interpolation_2_l$(OBJ_EXT) -c $(SRC)/CGAL_triangulation/averaged_interpolation_2.cc
	$(CC) $(COMPILE_FLAGS) -fPIC $(DTFE_INC) -o $(OBJ_DIR)/ps_interpolation_l$(OBJ_EXT) -c $(SRC)/CGAL_triangulation/ps_interpolation.cc
	$(CC) $(COMPILE_FLAGS) -fPIC $(DTFE_INC) -o $(OBJ_DIR)/ps_point_eval_l$(OBJ_EXT) -c $(SRC)/CGAL_triangulation/ps_point_eval.cc
	$(CC) $(COMPILE_FLAGS) -fPIC $(DTFE_INC) -o $(OBJ_DIR)/input_output_l$(OBJ_EXT) -c $(SRC)/input_output.cc
	$(CC) $(COMPILE_FLAGS) -fPIC $(DTFE_INC) -o $(OBJ_DIR)/user_options_l$(OBJ_EXT) -c $(SRC)/user_options.cc
	$(CC) $(COMPILE_FLAGS) -fPIC $(DTFE_INC) -o $(OBJ_DIR)/quantities_l$(OBJ_EXT) -c $(SRC)/quantities.cc
	$(CC) $(COMPILE_FLAGS) -fPIC $(DTFE_INC) -o $(OBJ_DIR)/NGP_interpolation_l$(OBJ_EXT) -c $(SRC)/NGP_interpolation.cc
	$(CC) $(COMPILE_FLAGS) -fPIC $(DTFE_INC) -o $(OBJ_DIR)/CIC_interpolation_l$(OBJ_EXT) -c $(SRC)/CIC_interpolation.cc
	$(CC) $(COMPILE_FLAGS) -fPIC $(DTFE_INC) -o $(OBJ_DIR)/TSC_interpolation_l$(OBJ_EXT) -c $(SRC)/TSC_interpolation.cc
	$(CC) $(COMPILE_FLAGS) -fPIC $(DTFE_INC) -o $(OBJ_DIR)/PCS_interpolation_l$(OBJ_EXT) -c $(SRC)/PCS_interpolation.cc
	$(CC) $(COMPILE_FLAGS) -fPIC $(DTFE_INC) -o $(OBJ_DIR)/SPH_interpolation_l$(OBJ_EXT) -c $(SRC)/SPH_interpolation.cc
	$(CC) $(COMPILE_FLAGS) -fPIC $(DTFE_INC) -o $(OBJ_DIR)/interlacing_l$(OBJ_EXT) -c $(SRC)/interlacing.cc
	$(CC) $(COMPILE_FLAGS) -fPIC $(DTFE_INC) -o $(OBJ_DIR)/random_l$(OBJ_EXT) -c $(SRC)/random.cc
	$(CC) $(COMPILE_FLAGS) -fPIC $(DTFE_INC) -o $(OBJ_DIR)/scratch_alloc_stub_l$(OBJ_EXT) -c $(SRC)/scratch_alloc_stub.cc
	$(CC) $(COMPILE_FLAGS) -shared $(OBJ_DIR)/DTFE_l$(OBJ_EXT) $(OBJ_DIR)/triangulation_l$(OBJ_EXT) $(OBJ_DIR)/kdtree2_l$(OBJ_EXT) $(OBJ_DIR)/r3d$(OBJ_EXT) $(OBJ_DIR)/scratch_alloc_stub_l$(OBJ_EXT) $(DTFE_CC_LIB_OBJS) $(IO_CC_LIB_OBJS) $(TRIANG_CC_LIB_OBJS) $(DTFE_GPU_L_OBJS) $(DTFE_LIB) $(DTFE_GPU_LIBS) -o $(LIB_DIR)/libDTFE$(SHARED_EXT)


clean:
	$(RM_RF) $(BIN_DIR)/DTFE$(EXE_EXT) $(BIN_DIR)/PS-DTFE$(EXE_EXT) $(OBJ_DIR)/*$(OBJ_EXT) $(OBJ_DIR_PS)/*$(OBJ_EXT) $(OBJ_DIR)/*.d $(OBJ_DIR_PS)/*.d $(LIB_DIR)/*DTFE$(SHARED_EXT)

# ---- deps-check: verify the required headers/libraries exist BEFORE a long compile dies
# mid-build with a cryptic missing-header error, and print the per-platform install command
# for anything missing. Each spec is header:name:brew-package:apt-package; the header is
# searched in every -I directory this Makefile will pass to the compiler.
DEPS_SPECS = \
	gsl/gsl_math.h:GSL:gsl:libgsl-dev \
	boost/program_options.hpp:Boost:boost:libboost-all-dev \
	CGAL/version.h:CGAL:cgal:libcgal-dev \
	gmp.h:GMP:gmp:libgmp-dev \
	mpfr.h:MPFR:mpfr:libmpfr-dev \
	H5Cpp.h:HDF5-C++:hdf5:libhdf5-dev \
	fftw3.h:FFTW3:fftw:libfftw3-dev

.PHONY: deps-check
deps-check:
	@echo ">> checking build dependencies (compiler: $(CC))"
	@command -v $(firstword $(CC)) >/dev/null 2>&1 \
		|| { echo "  MISSING  C++ compiler '$(CC)' (macOS: xcode-select --install && brew install llvm libomp; Linux: install g++)"; exit 1; }
	@missing_brew=""; missing_apt=""; fail=0; \
	incdirs="$(subst -I ,,$(INCLUDES)) /usr/include /usr/local/include /usr/include/*-linux-gnu"; \
	for spec in $(DEPS_SPECS); do \
		hdr=$$(echo "$$spec" | cut -d':' -f1); \
		name=$$(echo "$$spec" | cut -d':' -f2); \
		brewpkg=$$(echo "$$spec" | cut -d':' -f3); \
		aptpkg=$$(echo "$$spec" | cut -d':' -f4); \
		found=""; \
		for inc in $$incdirs; do \
			if [ -e "$$inc/$$hdr" ]; then found="$$inc/$$hdr"; break; fi; \
		done; \
		if [ -n "$$found" ]; then \
			printf '   OK       %-12s %s\n' "$$name" "$$found"; \
		else \
			printf '   MISSING  %-12s (header %s not found on the include path)\n' "$$name" "$$hdr"; \
			fail=1; missing_brew="$$missing_brew $$brewpkg"; missing_apt="$$missing_apt $$aptpkg"; \
		fi; \
	done; \
	if [ "$$fail" -eq 1 ]; then \
		echo ""; \
		echo ">> missing dependencies -- install them with:"; \
		echo "   macOS (Homebrew):  brew install$$missing_brew"; \
		echo "   Ubuntu/Debian:     sudo apt-get install$$missing_apt"; \
		echo "   Fedora/RHEL:       see the README 'Prerequisites' section"; \
		exit 1; \
	fi; \
	echo ">> all build dependencies found"

# Platform detection test (useful for debugging)
test-platform:
	@echo "Detected platform: $(PLATFORM)"
	@echo "Operating system: $(UNAME_S)"
	@echo "Architecture: $(shell uname -m)"
ifeq ($(PLATFORM),macos)
	@echo "Homebrew prefix: $(BREW_PREFIX)"
endif
	@echo "Build mode: $(BUILD_MODE)"
	@echo "Compiler: $(CC)"
	@echo "Executable extension: '$(EXE_EXT)'"
	@echo "Shared library extension: '$(SHARED_EXT)'"
	@echo "Object file extension: '$(OBJ_EXT)'"
	@echo "GSL path: $(GSL_PATH)"
	@echo "Boost path: $(BOOST_PATH)"
	@echo "CGAL path: $(CGAL_PATH)"
	@echo "Compile flags: $(COMPILE_FLAGS)"
	@echo "Include flags: $(INCLUDES)"
	@echo "Libraries: $(DTFE_LIB)"

copy_headers:
	cp $(addprefix $(SRC)/, $(HEADERS_1)) $(INC_DIR)
	cp $(addprefix $(SRC)/, $(HEADERS_2)) $(INC_DIR)/CGAL_triangulation

set_directories:
	@$(MKDIR_P) $(OBJ_DIR)
	@$(MKDIR_P) $(BIN_DIR)

set_directories_2:
	@$(MKDIR_P) $(LIB_DIR)
	@$(MKDIR_P) $(INC_DIR)
	@$(MKDIR_P) $(INC_DIR)/CGAL_triangulation
