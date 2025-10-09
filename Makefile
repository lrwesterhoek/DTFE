# Cross-platform Makefile for compiling the DTFE code on Mac, Linux and Windows systems
#
# SUPPORTED ARCHITECTURES:
# ========================
# - x86_64 (Intel/AMD 64-bit) on macOS, Linux, and Windows
# - ARM64 (aarch64/Apple Silicon) on macOS, Linux, and Windows
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
# Windows (MSYS2/MinGW):
#   Uses MSYS2 paths. Install MSYS2 and packages:
#   pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-gsl mingw-w64-x86_64-boost
#   mingw-w64-x86_64-cgal mingw-w64-x86_64-mpfr mingw-w64-x86_64-hdf5 mingw-w64-x86_64-gmp
#   For ARM64: Replace x86_64 with clangarm64, then run 'make DTFE'
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
# make DTFE                      - Build the main executable (optimized)
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
else ifneq (,$(findstring MINGW,$(UNAME_S)))  # Windows (MINGW32/MINGW64)
    PLATFORM := windows
    SHARED_EXT := .dll
    EXE_EXT := .exe
    OBJ_EXT := .o
    LIB_EXT := .a
else ifneq (,$(findstring MSYS,$(UNAME_S)))  # Windows (MSYS2)
    PLATFORM := windows
    SHARED_EXT := .dll
    EXE_EXT := .exe
    OBJ_EXT := .o
    LIB_EXT := .a
else ifneq (,$(findstring CYGWIN,$(UNAME_S)))  # Windows (Cygwin)
    PLATFORM := windows
    SHARED_EXT := .dll
    EXE_EXT := .exe
    OBJ_EXT := .o
    LIB_EXT := .a
else
    $(error Unsupported operating system: $(UNAME_S). Only macOS, Linux, and Windows are supported.)
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
    CC := $(shell which g++ 2>/dev/null || which clang++ 2>/dev/null || echo "g++")
else ifeq ($(PLATFORM),windows)
    # Windows with MSYS2/MinGW - detect architecture for correct paths
    ARCH := $(shell uname -m)
    ifeq ($(ARCH),x86_64)
        # x86_64 architecture - use mingw64 prefix
        MINGW_PREFIX = /mingw64
    else ifeq ($(ARCH),aarch64)
        # ARM64 architecture - use clangarm64 prefix
        MINGW_PREFIX = /clangarm64
    else ifeq ($(ARCH),i686)
        # 32-bit x86 architecture - use mingw32 prefix
        MINGW_PREFIX = /mingw32
    else
        # Default to mingw64 if architecture detection fails
        MINGW_PREFIX = /mingw64
    endif

    GSL_PATH   = $(MINGW_PREFIX)
    BOOST_PATH = $(MINGW_PREFIX)
    CGAL_PATH  = $(MINGW_PREFIX)
    MPFR_PATH  = $(MINGW_PREFIX)
    HDF5_PATH  = $(MINGW_PREFIX)
    GMP_PATH   = $(MINGW_PREFIX)

    # Try different compiler locations for Windows
    CC := $(shell which $(MINGW_PREFIX)/bin/g++ 2>/dev/null || which g++ 2>/dev/null || which clang++ 2>/dev/null || echo "g++")
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
CC         := $(or $(CC_OVERRIDE),$(CC))


# paths to where to put the object files and the executables files. If you build the DTFE library than you also need to specify the directory where to put the library and the directory where to copy the header files needed by the library (choose an empty directory for the header files).
OBJ_DIR = ./o
BIN_DIR = ./
LIB_DIR = ./
INC_DIR = ./DTFE_include

############################# Choose the compiler directives ##################################

############################# Overall options ##################################
OPTIONS = 
#------------------------ set the number of spatial dimensions (2 or 3 dimensions)
OPTIONS += -DNO_DIM=3 
#------------------------ set type of variables - float (comment the next line) or double (uncomment the next line)
# OPTIONS += -DDOUBLE 

############################# Quantities to be computed ##################################
#------------------------ set which quantities can be computed (can save memory by leaving some out)
# Comment this line if you don't need to compute velocity and velocity related components 
OPTIONS += -DVELOCITY 
# Comment this line if you don't need to interpolate additional fields stored in the scalar variable
OPTIONS += -DSCALAR 
# number of components of the scalar variable
OPTIONS += -DNO_SCALARS=1 

############################# Input and output operations default settings ##################################
#------------------------ set which are the default input and output functions for doing data io
# default function to read the input data (101-multiple gadget file, 102-single gadget file, 105-HDF5 gadget file, 111-text file, ... see documentation for more options). The input file type can be set during runtime using the option '--input'. This makefile option only sets a default input file in the case none is given via the program options.
OPTIONS += -DINPUT_FILE_DEFAULT=105
# default value for the units of the input data (value=what is 1 Mpc in the units of the data - in this example the data is in kpc). You can change this also during runtime using the program option '--MpcUnit'.
OPTIONS += -DMPC_UNIT=1000 
# default function to write the output data (101-binary file, 111-text file, ... see documentation for more options). The output file type can be set during runtime using the option '--output'. This makefile option only sets a default output file in the case none is given via the program options.
OPTIONS += -DOUTPUT_FILE_DEFAULT=101
#101 for binary file, 100 my density file

############################# additional compiler options ##################################
# enable this option if to use OpenMP (share the workload between CPU cores sharing the same RAM)
OPTIONS += -DOPEN_MP
# enable to check if the padding gives a complete Delaunay Tesselation of the region of interest
OPTIONS += -DTEST_PADDING
# enable this option to shift from position space to redshift space; You also need to activate this option during run-time using '--redshift-space arguments'
OPTIONS += -DREDSHIFT_SPACE

#------------------------ options usefull when using DTFE as a library
# uncomment the line to get access to a function that returns the Delaunay triangulation of the point set
OPTIONS += -DTRIANGULATION


############################# Help menu messages options ##################################
#------------------------ compiler directive that affect only the help messages when using the '-h / --help' option (it does not affect the program in any other way)- if the option is uncommented, than it will show that set of options in the help menu
OPTIONS += -DFIELD_OPTIONS 
OPTIONS += -DREGION_OPTIONS 
OPTIONS += -DPARTITION_OPTIONS 
OPTIONS += -DPADDING_OPTIONS 
OPTIONS += -DAVERAGING_OPTIONS
OPTIONS += -DREDSHIFT_CONE_OPTIONS 
OPTIONS += -DADDITIONAL_OPTIONS 



OPTIONS += -DBOOST_TIMER_ENABLE_DEPRECATED
OPTIONS += -DBOOST_ALLOW_DEPRECATED_HEADERS




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
ifeq ($(PLATFORM),linux)
    # On Linux (Ubuntu/Debian), HDF5 headers are in /usr/include/hdf5/serial
    INCLUDES += -I $(strip $(HDF5_PATH))/include/hdf5/serial
    # Detect architecture for library path
    ARCH := $(shell uname -m)
    ifeq ($(ARCH),x86_64)
        LIBRARIES += -L$(strip $(HDF5_PATH))/lib/x86_64-linux-gnu/hdf5/serial
    else ifeq ($(ARCH),aarch64)
        LIBRARIES += -L$(strip $(HDF5_PATH))/lib/aarch64-linux-gnu/hdf5/serial
    endif
    LIBRARIES += -L$(strip $(HDF5_PATH))/lib -lhdf5 -lhdf5_cpp
else
    LIBRARIES += -L$(strip $(HDF5_PATH))/lib -lhdf5 -lhdf5_cpp
endif
    OPTIONS += -DHDF5
endif

# Compiler flags (same for both platforms)
# Build mode can be set with: make DTFE BUILD_MODE=debug
BUILD_MODE ?= release

ifeq ($(BUILD_MODE),debug)
    # Debug build: no optimization, with debug symbols and sanitizers
    BASE_CFLAGS = -O0 -g3 -DDEBUG $(OPTIONS)
    # Add sanitizers for debug builds (catch memory errors, undefined behavior, etc.)
    SANITIZER_FLAGS = -fsanitize=address -fsanitize=undefined -fsanitize=leak
    DEBUG_FLAGS = $(SANITIZER_FLAGS) -fno-omit-frame-pointer
else
    # Release build: full optimization
    BASE_CFLAGS = -O3 -DNDEBUG $(OPTIONS)
    DEBUG_FLAGS =
endif

# Minimal compiler flags (from original Makefile)
# Additional warnings and quality flags can be enabled in the EXTRA_FLAGS section above
COMPILE_FLAGS = $(BASE_CFLAGS) -std=c++17 -Wno-psabi -Wno-cpp -frounding-math $(DEBUG_FLAGS) $(EXTRA_FLAGS)
LINK_FLAGS =
BASE_LIBS = -lboost_thread -lboost_filesystem -lboost_program_options -lgsl -lgslcblas -lm -lgmp -lmpfr -lboost_system
HDF5_LIBS = -lhdf5 -lhdf5_cpp

# Platform-specific OpenMP settings only
ifeq ($(PLATFORM),macos)
    COMPILE_FLAGS += -fopenmp=libomp
    OPENMP_LIB = -lomp
else ifeq ($(PLATFORM),linux)
    COMPILE_FLAGS += -fopenmp
    OPENMP_LIB = -lgomp
else ifeq ($(PLATFORM),windows)
    COMPILE_FLAGS += -fopenmp
    OPENMP_LIB = -lgomp
    # Windows-specific linking flags
    LINK_FLAGS += -static-libgcc -static-libstdc++
endif

DTFE_INC = $(INCLUDES)

# Linking
ifeq ($(findstring -DHDF5,$(OPTIONS)),-DHDF5)
    DTFE_LIB = $(LIBRARIES) $(BASE_LIBS) $(HDF5_LIBS) $(OPENMP_LIB)
else
    DTFE_LIB = $(LIBRARIES) $(BASE_LIBS) $(OPENMP_LIB)
endif


IO_SOURCES = $(addprefix io/, input_output.h gadget_reader_header.cc gadget_reader_binary.cc gadget_reader_HDF5.cc gadget_reader_HDF5_Cristian.cc gadget_reader_MOG.cc hdf5_input_my_DESI.cc text_io.cc binary_io.cc my_io.cc)
MAIN_SOURCES = main.cpp DTFE.h message.h user_options.h input_output.cc $(IO_SOURCES)
DTFE_SOURCES = DTFE.cpp define.h particle_data.h user_options.h box.h quantities.h user_options.cc quantities.cc subpartition.h random.cc CIC_interpolation.cc TSC_interpolation.cc SPH_interpolation.cc kdtree/kdtree2.hpp Pvector.h message.h miscellaneous.h
TRIANG_SOURCES = $(addprefix CGAL_triangulation/, triangulation.cpp triangulation_miscellaneous.cc unaveraged_interpolation.cc averaged_interpolation_1.cc averaged_interpolation_2.cc padding_test.cc CGAL_include_2D.h CGAL_include_3D.h vertexData.h particle_data_traits.h) define.h particle_data.h user_options.h box.h quantities.h Pvector.h message.h math_functions.h

ALL_FILES = $(DTFE_SOURCES) $(TRIANG_SOURCES) $(MAIN_SOURCES) kdtree/kdtree2.hpp kdtree/kdtree2.cpp
LIB_FILES = $(DTFE_SOURCES) $(TRIANG_SOURCES)

HEADERS_1 = DTFE.h define.h user_options.h particle_data.h quantities.h Pvector.h math_functions.h  message.h box.h miscellaneous.h
HEADERS_2 = $(addprefix CGAL_triangulation/, CGAL_include_2D.h CGAL_include_3D.h vertexData.h particle_data_traits.h)

# Declare phony targets
.PHONY: DTFE library clean test-platform copy_headers set_directories set_directories_2


DTFE: set_directories $(OBJ_DIR)/DTFE$(OBJ_EXT) $(OBJ_DIR)/triangulation$(OBJ_EXT) $(OBJ_DIR)/main$(OBJ_EXT) $(OBJ_DIR)/kdtree2$(OBJ_EXT) Makefile
	$(CC) $(COMPILE_FLAGS) $(LINK_FLAGS) $(OBJ_DIR)/DTFE$(OBJ_EXT) $(OBJ_DIR)/triangulation$(OBJ_EXT) $(OBJ_DIR)/main$(OBJ_EXT) $(OBJ_DIR)/kdtree2$(OBJ_EXT) $(DTFE_LIB) -o $(BIN_DIR)/DTFE$(EXE_EXT)


$(OBJ_DIR)/main$(OBJ_EXT): $(addprefix $(SRC)/, $(MAIN_SOURCES)) Makefile
	$(CC) $(COMPILE_FLAGS) $(DTFE_INC) -o $(OBJ_DIR)/main$(OBJ_EXT) -c $(SRC)/main.cpp

$(OBJ_DIR)/DTFE$(OBJ_EXT): $(addprefix $(SRC)/, $(DTFE_SOURCES)) Makefile
	$(CC) $(COMPILE_FLAGS) $(DTFE_INC) -o $(OBJ_DIR)/DTFE$(OBJ_EXT) -c $(SRC)/DTFE.cpp

$(OBJ_DIR)/kdtree2$(OBJ_EXT): $(SRC)/kdtree/kdtree2.hpp $(SRC)/kdtree/kdtree2.cpp Makefile
	$(CC) -O3 -ffast-math -fomit-frame-pointer $(DTFE_INC) -o $(OBJ_DIR)/kdtree2$(OBJ_EXT) -c $(SRC)/kdtree/kdtree2.cpp

$(OBJ_DIR)/triangulation$(OBJ_EXT): $(addprefix $(SRC)/, $(TRIANG_SOURCES)) Makefile
	$(CC) $(COMPILE_FLAGS) $(DTFE_INC) -o $(OBJ_DIR)/triangulation$(OBJ_EXT) -c $(SRC)/CGAL_triangulation/triangulation.cpp


library: set_directories set_directories_2 $(addprefix $(SRC)/, $(LIB_FILES) ) copy_headers Makefile
	$(CC) $(COMPILE_FLAGS) -fPIC $(DTFE_INC) -o $(OBJ_DIR)/DTFE_l$(OBJ_EXT) -c $(SRC)/DTFE.cpp
	$(CC) -O3 -ffast-math -fomit-frame-pointer -fPIC $(DTFE_INC) -o $(OBJ_DIR)/kdtree2_l$(OBJ_EXT) -c $(SRC)/kdtree/kdtree2.cpp
	$(CC) $(COMPILE_FLAGS) -fPIC $(DTFE_INC) -o $(OBJ_DIR)/triangulation_l$(OBJ_EXT) -c $(SRC)/CGAL_triangulation/triangulation.cpp
	$(CC) $(COMPILE_FLAGS) $(LINK_FLAGS) -shared $(OBJ_DIR)/DTFE_l$(OBJ_EXT) $(OBJ_DIR)/triangulation_l$(OBJ_EXT) $(OBJ_DIR)/kdtree2_l$(OBJ_EXT) $(DTFE_LIB) -o $(LIB_DIR)/libDTFE$(SHARED_EXT)


clean:
	$(RM_RF) $(BIN_DIR)/DTFE$(EXE_EXT) $(OBJ_DIR)/*$(OBJ_EXT) $(LIB_DIR)/*DTFE$(SHARED_EXT)

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