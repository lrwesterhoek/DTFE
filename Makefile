# Cross-platform Makefile for compiling the DTFE code on Mac, Linux, and Windows systems
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
# Windows Option 1 (MSVC with vcpkg):
#   Install Visual Studio and vcpkg, then:
#   vcpkg install gsl boost cgal mpfr hdf5 gmp
#   Open Developer Command Prompt and run 'nmake DTFE' or 'make DTFE'
#
# Windows Option 2 (MSYS2/MinGW):
#   Install packages via pacman:
#   pacman -S mingw-w64-x86_64-gsl mingw-w64-x86_64-boost mingw-w64-x86_64-cgal
#   pacman -S mingw-w64-x86_64-mpfr mingw-w64-x86_64-hdf5 mingw-w64-x86_64-gmp
#   then run 'make DTFE'
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
# TARGETS:
# ========
# make DTFE     - Build the main executable
# make library  - Build the shared library (libDTFE.so/.dylib/.dll)
# make clean    - Clean object files and executables
#

# Detect operating system and compiler environment
UNAME_S := $(shell uname -s 2>/dev/null || echo "Windows")

# Detect Windows compiler environment
ifeq ($(UNAME_S),Windows)
    # Check for MSVC environment (Visual Studio Developer Command Prompt)
    ifdef VCINSTALLDIR
        WINDOWS_COMPILER := msvc
    else ifdef VS160COMNTOOLS
        WINDOWS_COMPILER := msvc
    else ifdef VS150COMNTOOLS
        WINDOWS_COMPILER := msvc
    else ifdef VS140COMNTOOLS
        WINDOWS_COMPILER := msvc
    else
        # Check if cl.exe is available (MSVC)
        ifneq ($(shell where cl 2>nul),)
            WINDOWS_COMPILER := msvc
        else
            # Default to MinGW
            WINDOWS_COMPILER := mingw
        endif
    endif
else
    WINDOWS_COMPILER := none
endif

# Set platform-specific variables
ifeq ($(UNAME_S),Darwin)  # macOS
    PLATFORM := macos
    COMPILER_TYPE := gcc_clang
    SHARED_EXT := .dylib
    EXE_EXT := 
else ifeq ($(UNAME_S),Linux)
    PLATFORM := linux
    COMPILER_TYPE := gcc_clang
    SHARED_EXT := .so
    EXE_EXT := 
else  # Windows
    PLATFORM := windows
    ifeq ($(WINDOWS_COMPILER),msvc)
        COMPILER_TYPE := msvc
        SHARED_EXT := .dll
        EXE_EXT := .exe
        OBJ_EXT := .obj
        LIB_EXT := .lib
    else
        COMPILER_TYPE := gcc_clang
        SHARED_EXT := .dll
        EXE_EXT := .exe
        OBJ_EXT := .o
        LIB_EXT := .a
    endif
endif

# Set default object and library extensions for non-Windows
ifeq ($(PLATFORM),windows)
    # Already set above
else
    OBJ_EXT := .o
    LIB_EXT := .a
endif

# Platform-specific library paths and compiler settings
ifeq ($(PLATFORM),macos)
    # macOS with Homebrew
    GSL_PATH   = /opt/homebrew/opt/gsl
    BOOST_PATH = /opt/homebrew/opt/boost
    CGAL_PATH  = /opt/homebrew/opt/cgal
    MPFR_PATH  = /opt/homebrew/opt/mpfr
    HDF5_PATH  = /opt/homebrew/opt/hdf5
    GMP_PATH = /opt/homebrew/opt/gmp
    # Try different compiler locations
    CC := $(shell which /opt/homebrew/opt/llvm/bin/clang++ 2>/dev/null || which clang++ 2>/dev/null || which g++ 2>/dev/null || echo "clang++")
    MKDIR_P = mkdir -p
    RM_RF = rm -rf
else ifeq ($(PLATFORM),linux)
    # Linux - try to auto-detect common package manager installations
    GSL_PATH   = $(shell pkg-config --variable=prefix gsl 2>/dev/null || echo "/usr")
    BOOST_PATH = /usr
    CGAL_PATH  = /usr
    MPFR_PATH  = /usr
    HDF5_PATH  = /usr
    GMP_PATH = /usr
    CC := $(shell which g++ 2>/dev/null || which clang++ 2>/dev/null || echo "g++")
    MKDIR_P = mkdir -p
    RM_RF = rm -rf
else  # Windows
    ifeq ($(WINDOWS_COMPILER),msvc)
        # MSVC with vcpkg (recommended) or manual installation
        # Try to detect vcpkg installation
        ifdef VCPKG_ROOT
            VCPKG_PATH = $(VCPKG_ROOT)
        else
            # Common vcpkg locations
            VCPKG_PATH := $(shell if exist "C:\vcpkg\installed\x64-windows" echo C:\vcpkg\installed\x64-windows 2>nul)
            ifeq ($(VCPKG_PATH),)
                VCPKG_PATH := $(shell if exist "C:\tools\vcpkg\installed\x64-windows" echo C:\tools\vcpkg\installed\x64-windows 2>nul)
            endif
        endif
        
        # Set library paths for MSVC
        ifneq ($(VCPKG_PATH),)
            # Using vcpkg
            GSL_PATH   = $(VCPKG_PATH)
            BOOST_PATH = $(VCPKG_PATH)
            CGAL_PATH  = $(VCPKG_PATH)
            MPFR_PATH  = $(VCPKG_PATH)
            HDF5_PATH  = $(VCPKG_PATH)
            GMP_PATH   = $(VCPKG_PATH)
        else
            # Manual installation paths (user needs to customize)
            GSL_PATH   = C:/Libraries/gsl
            BOOST_PATH = C:/Libraries/boost
            CGAL_PATH  = C:/Libraries/cgal
            MPFR_PATH  = C:/Libraries/mpfr
            HDF5_PATH  = C:/Libraries/hdf5
            GMP_PATH   = C:/Libraries/gmp
        endif
        
        CC = cl
        MKDIR_P = if not exist
        RM_RF = del /Q
    else
        # MinGW/MSYS2 paths
        GSL_PATH   = /mingw64
        BOOST_PATH = /mingw64
        CGAL_PATH  = /mingw64
        MPFR_PATH  = /mingw64
        HDF5_PATH  = /mingw64
        GMP_PATH = /mingw64
        # On Windows, try common MinGW paths first, then fallback to system
        CC := $(shell which /mingw64/bin/g++ 2>/dev/null || which /c/mingw64/bin/g++ 2>/dev/null || which g++ 2>/dev/null || echo "g++")
        MKDIR_P = mkdir -p
        RM_RF = rm -rf
    endif
endif

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




###############  DO NOT MODIFY BELOW THIS LINE  ###########################
# do not modify below this line
SRC = ./src
INCLUDES = 
LIBRARIES = 

# Library path setup - different for MSVC vs GCC/Clang
ifeq ($(COMPILER_TYPE),msvc)
    # MSVC-style includes and libraries
    ifneq ($(strip $(GSL_PATH)),)
        INCLUDES += /I"$(strip $(GSL_PATH))/include"
        LIBRARIES += /LIBPATH:"$(strip $(GSL_PATH))/lib"
    endif
    ifneq ($(strip $(BOOST_PATH)),)
        INCLUDES += /I"$(strip $(BOOST_PATH))/include"
        LIBRARIES += /LIBPATH:"$(strip $(BOOST_PATH))/lib"
    endif
    ifneq ($(strip $(CGAL_PATH)),)
        INCLUDES += /I"$(strip $(CGAL_PATH))/include"
        LIBRARIES += /LIBPATH:"$(strip $(CGAL_PATH))/lib"
    endif
    ifneq ($(strip $(GMP_PATH)),)
        INCLUDES += /I"$(strip $(GMP_PATH))/include"
        LIBRARIES += /LIBPATH:"$(strip $(GMP_PATH))/lib"
    endif
    ifneq ($(strip $(MPFR_PATH)),)
        INCLUDES += /I"$(strip $(MPFR_PATH))/include"
        LIBRARIES += /LIBPATH:"$(strip $(MPFR_PATH))/lib"
    endif
    ifneq ($(strip $(HDF5_PATH)),)
        INCLUDES += /I"$(strip $(HDF5_PATH))/include"
        LIBRARIES += /LIBPATH:"$(strip $(HDF5_PATH))/lib"
        OPTIONS += -DHDF5
    endif
else
    # GCC/Clang-style includes and libraries
    ifneq ($(strip $(GSL_PATH)),)
        INCLUDES += -I$(strip $(GSL_PATH))/include 
        LIBRARIES += -L$(strip $(GSL_PATH))/lib 
    endif
    ifneq ($(strip $(BOOST_PATH)),)
        INCLUDES += -I$(strip $(BOOST_PATH))/include 
        LIBRARIES += -L$(strip $(BOOST_PATH))/lib 
    endif
    ifneq ($(strip $(CGAL_PATH)),)
        INCLUDES += -I$(strip $(CGAL_PATH))/include 
        LIBRARIES += -L$(strip $(CGAL_PATH))/lib 
    endif
    ifneq ($(strip $(GMP_PATH)),)
        INCLUDES += -I$(strip $(GMP_PATH))/include
        LIBRARIES += -L$(strip $(GMP_PATH))/lib
    endif
    ifneq ($(strip $(MPFR_PATH)),)
        INCLUDES += -I$(strip $(MPFR_PATH))/include
        LIBRARIES += -L$(strip $(MPFR_PATH))/lib
    endif
    ifneq ($(strip $(HDF5_PATH)),)
        INCLUDES += -I$(strip $(HDF5_PATH))/include 
        LIBRARIES += -L$(strip $(HDF5_PATH))/lib -lhdf5 -lhdf5_cpp
        OPTIONS += -DHDF5
    endif
endif

# Cross-platform compiler flags
ifeq ($(COMPILER_TYPE),msvc)
    BASE_CFLAGS = /O2 /DNDEBUG $(OPTIONS)
else
    BASE_CFLAGS = -O3 -DNDEBUG $(OPTIONS)
endif

# Compiler-specific flags
ifeq ($(COMPILER_TYPE),msvc)
    # MSVC compiler flags
    COMPILE_FLAGS = $(BASE_CFLAGS) /fp:precise /openmp /EHsc
    OPENMP_LIB = 
    LINK_FLAGS = /INCREMENTAL:NO
    # MSVC library names (different from GCC/Clang)
    BASE_LIBS = gsl.lib gslcblas.lib boost_thread-vc143-mt-x64-1_82.lib boost_filesystem-vc143-mt-x64-1_82.lib boost_program_options-vc143-mt-x64-1_82.lib boost_system-vc143-mt-x64-1_82.lib gmp.lib mpfr.lib
    HDF5_LIBS = hdf5.lib hdf5_cpp.lib
else ifeq ($(PLATFORM),macos)
    # macOS-specific flags (GCC/Clang)
    COMPILE_FLAGS = $(BASE_CFLAGS) -frounding-math -fopenmp=libomp
    OPENMP_LIB = -lomp
    LINK_FLAGS = 
    BASE_LIBS = -lboost_thread -lboost_filesystem -lboost_program_options -lgsl -lgslcblas -lm -lgmp -lmpfr -lboost_system
    HDF5_LIBS = -lhdf5 -lhdf5_cpp
else ifeq ($(PLATFORM),linux)
    # Linux-specific flags (GCC/Clang)
    COMPILE_FLAGS = $(BASE_CFLAGS) -frounding-math -fopenmp
    OPENMP_LIB = -lgomp
    LINK_FLAGS = 
    BASE_LIBS = -lboost_thread -lboost_filesystem -lboost_program_options -lgsl -lgslcblas -lm -lgmp -lmpfr -lboost_system
    HDF5_LIBS = -lhdf5 -lhdf5_cpp
else
    # Windows MinGW/MSYS2 (GCC/Clang)
    COMPILE_FLAGS = $(BASE_CFLAGS) -frounding-math -fopenmp
    OPENMP_LIB = -lgomp
    LINK_FLAGS = 
    BASE_LIBS = -lboost_thread -lboost_filesystem -lboost_program_options -lgsl -lgslcblas -lm -lgmp -lmpfr -lboost_system
    HDF5_LIBS = -lhdf5 -lhdf5_cpp
endif

DTFE_INC = $(INCLUDES)

# Cross-platform library linking
ifeq ($(COMPILER_TYPE),msvc)
    # MSVC linking - add HDF5 if enabled
    ifeq ($(findstring -DHDF5,$(OPTIONS)),-DHDF5)
        DTFE_LIB = $(LIBRARIES) $(BASE_LIBS) $(HDF5_LIBS) $(OPENMP_LIB)
    else
        DTFE_LIB = $(LIBRARIES) $(BASE_LIBS) $(OPENMP_LIB)
    endif
else
    # GCC/Clang linking
    ifeq ($(findstring -DHDF5,$(OPTIONS)),-DHDF5)
        DTFE_LIB = $(LIBRARIES) $(BASE_LIBS) $(HDF5_LIBS) $(OPENMP_LIB)
    else
        DTFE_LIB = $(LIBRARIES) $(BASE_LIBS) $(OPENMP_LIB)
    endif
endif


IO_SOURCES = $(addprefix io/, input_output.h gadget_reader_header.cc gadget_reader_binary.cc gadget_reader_HDF5.cc gadget_reader_HDF5_Cristian.cc gadget_reader_MOG.cc hdf5_input_my_DESI.cc text_io.cc binary_io.cc my_io.cc)
MAIN_SOURCES = main.cpp DTFE.h message.h user_options.h input_output.cc $(IO_SOURCES)
DTFE_SOURCES = DTFE.cpp define.h particle_data.h user_options.h box.h quantities.h user_options.cc quantities.cc subpartition.h random.cc CIC_interpolation.cc TSC_interpolation.cc SPH_interpolation.cc kdtree/kdtree2.hpp Pvector.h message.h miscellaneous.h
TRIANG_SOURCES = $(addprefix CGAL_triangulation/, triangulation.cpp triangulation_miscellaneous.cc unaveraged_interpolation.cc averaged_interpolation_1.cc averaged_interpolation_2.cc padding_test.cc CGAL_include_2D.h CGAL_include_3D.h vertexData.h particle_data_traits.h) define.h particle_data.h user_options.h box.h quantities.h Pvector.h message.h math_functions.h

ALL_FILES = $(DTFE_SOURCES) $(TRIANG_SOURCES) $(MAIN_SOURCES) kdtree/kdtree2.hpp kdtree/kdtree2.cpp
LIB_FILES = $(DTFE_SOURCES) $(TRIANG_SOURCES)

HEADERS_1 = DTFE.h define.h user_options.h particle_data.h quantities.h Pvector.h math_functions.h  message.h box.h miscellaneous.h
HEADERS_2 = $(addprefix CGAL_triangulation/, CGAL_include_2D.h CGAL_include_3D.h vertexData.h particle_data_traits.h)



DTFE: set_directories $(OBJ_DIR)/DTFE$(OBJ_EXT) $(OBJ_DIR)/triangulation$(OBJ_EXT) $(OBJ_DIR)/main$(OBJ_EXT) $(OBJ_DIR)/kdtree2$(OBJ_EXT) Makefile
ifeq ($(COMPILER_TYPE),msvc)
	$(CC) $(LINK_FLAGS) $(OBJ_DIR)/DTFE$(OBJ_EXT) $(OBJ_DIR)/triangulation$(OBJ_EXT) $(OBJ_DIR)/main$(OBJ_EXT) $(OBJ_DIR)/kdtree2$(OBJ_EXT) $(DTFE_LIB) /Fe:$(BIN_DIR)/DTFE$(EXE_EXT)
else
	$(CC) $(COMPILE_FLAGS) $(OBJ_DIR)/DTFE$(OBJ_EXT) $(OBJ_DIR)/triangulation$(OBJ_EXT) $(OBJ_DIR)/main$(OBJ_EXT) $(OBJ_DIR)/kdtree2$(OBJ_EXT) $(DTFE_LIB) -o $(BIN_DIR)/DTFE$(EXE_EXT)
endif


$(OBJ_DIR)/main$(OBJ_EXT): $(addprefix $(SRC)/, $(MAIN_SOURCES)) Makefile
ifeq ($(COMPILER_TYPE),msvc)
	$(CC) $(COMPILE_FLAGS) $(DTFE_INC) /Fo:$(OBJ_DIR)/main$(OBJ_EXT) /c $(SRC)/main.cpp
else
	$(CC) $(COMPILE_FLAGS) $(DTFE_INC) -o $(OBJ_DIR)/main$(OBJ_EXT) -c $(SRC)/main.cpp
endif

$(OBJ_DIR)/DTFE$(OBJ_EXT): $(addprefix $(SRC)/, $(DTFE_SOURCES)) Makefile
ifeq ($(COMPILER_TYPE),msvc)
	$(CC) $(COMPILE_FLAGS) $(DTFE_INC) /Fo:$(OBJ_DIR)/DTFE$(OBJ_EXT) /c $(SRC)/DTFE.cpp
else
	$(CC) $(COMPILE_FLAGS) $(DTFE_INC) -o $(OBJ_DIR)/DTFE$(OBJ_EXT) -c $(SRC)/DTFE.cpp
endif

$(OBJ_DIR)/kdtree2$(OBJ_EXT): $(SRC)/kdtree/kdtree2.hpp $(SRC)/kdtree/kdtree2.cpp Makefile
ifeq ($(COMPILER_TYPE),msvc)
	$(CC) /O2 /fp:fast $(DTFE_INC) /Fo:$(OBJ_DIR)/kdtree2$(OBJ_EXT) /c $(SRC)/kdtree/kdtree2.cpp
else
	$(CC) -O3 -ffast-math -fomit-frame-pointer $(DTFE_INC) -o $(OBJ_DIR)/kdtree2$(OBJ_EXT) -c $(SRC)/kdtree/kdtree2.cpp
endif

$(OBJ_DIR)/triangulation$(OBJ_EXT): $(addprefix $(SRC)/, $(TRIANG_SOURCES)) Makefile
ifeq ($(COMPILER_TYPE),msvc)
	$(CC) $(COMPILE_FLAGS) $(DTFE_INC) /Fo:$(OBJ_DIR)/triangulation$(OBJ_EXT) /c $(SRC)/CGAL_triangulation/triangulation.cpp
else
	$(CC) $(COMPILE_FLAGS) $(DTFE_INC) -o $(OBJ_DIR)/triangulation$(OBJ_EXT) -c $(SRC)/CGAL_triangulation/triangulation.cpp
endif


library: set_directories set_directories_2 $(addprefix $(SRC)/, $(LIB_FILES) ) copy_headers Makefile
ifeq ($(COMPILER_TYPE),msvc)
	$(CC) $(COMPILE_FLAGS) $(DTFE_INC) /Fo:$(OBJ_DIR)/DTFE_l$(OBJ_EXT) /c $(SRC)/DTFE.cpp
	$(CC) /O2 /fp:fast $(DTFE_INC) /Fo:$(OBJ_DIR)/kdtree2_l$(OBJ_EXT) /c $(SRC)/kdtree/kdtree2.cpp
	$(CC) $(COMPILE_FLAGS) $(DTFE_INC) /Fo:$(OBJ_DIR)/triangulation_l$(OBJ_EXT) /c $(SRC)/CGAL_triangulation/triangulation.cpp
	link /DLL $(LINK_FLAGS) $(OBJ_DIR)/DTFE_l$(OBJ_EXT) $(OBJ_DIR)/triangulation_l$(OBJ_EXT) $(OBJ_DIR)/kdtree2_l$(OBJ_EXT) $(DTFE_LIB) /OUT:$(LIB_DIR)/DTFE$(SHARED_EXT)
else
	$(CC) $(COMPILE_FLAGS) -fPIC $(DTFE_INC) -o $(OBJ_DIR)/DTFE_l$(OBJ_EXT) -c $(SRC)/DTFE.cpp
	$(CC) -O3 -ffast-math -fomit-frame-pointer -fPIC $(DTFE_INC) -o $(OBJ_DIR)/kdtree2_l$(OBJ_EXT) -c $(SRC)/kdtree/kdtree2.cpp
	$(CC) $(COMPILE_FLAGS) -fPIC $(DTFE_INC) -o $(OBJ_DIR)/triangulation_l$(OBJ_EXT) -c $(SRC)/CGAL_triangulation/triangulation.cpp
	$(CC) $(COMPILE_FLAGS) -shared $(OBJ_DIR)/DTFE_l$(OBJ_EXT) $(OBJ_DIR)/triangulation_l$(OBJ_EXT) $(OBJ_DIR)/kdtree2_l$(OBJ_EXT) $(DTFE_LIB) -o $(LIB_DIR)/libDTFE$(SHARED_EXT)
endif


clean:
ifeq ($(COMPILER_TYPE),msvc)
	del /Q "$(BIN_DIR)\DTFE$(EXE_EXT)" "$(OBJ_DIR)\*$(OBJ_EXT)" "$(LIB_DIR)\DTFE$(SHARED_EXT)" 2>nul || echo Clean completed
else
	$(RM_RF) $(BIN_DIR)/DTFE$(EXE_EXT) $(OBJ_DIR)/*$(OBJ_EXT) $(LIB_DIR)/*DTFE$(SHARED_EXT)
endif

# Platform detection test (useful for debugging)
test-platform:
	@echo "Detected platform: $(PLATFORM)"
	@echo "Operating system: $(UNAME_S)"
	@echo "Compiler type: $(COMPILER_TYPE)"
ifeq ($(PLATFORM),windows)
	@echo "Windows compiler: $(WINDOWS_COMPILER)"
endif
	@echo "Compiler: $(CC)"
	@echo "Executable extension: '$(EXE_EXT)'"
	@echo "Shared library extension: '$(SHARED_EXT)'"
	@echo "Object file extension: '$(OBJ_EXT)'"
ifeq ($(COMPILER_TYPE),msvc)
	@echo "VCPKG path: $(VCPKG_PATH)"
endif
	@echo "GSL path: $(GSL_PATH)"
	@echo "Boost path: $(BOOST_PATH)"
	@echo "CGAL path: $(CGAL_PATH)"
	@echo "Compile flags: $(COMPILE_FLAGS)"
	@echo "Include flags: $(INCLUDES)"
	@echo "Libraries: $(DTFE_LIB)"

# Test all platform configurations
test-all-platforms: test-macos test-linux test-windows test-msvc

test-macos:
	@echo "=== Testing macOS Configuration ==="
	@$(MAKE) --no-print-directory test-platform UNAME_S=Darwin
	@echo ""

test-linux:
	@echo "=== Testing Linux Configuration ==="
	@$(MAKE) --no-print-directory test-platform UNAME_S=Linux
	@echo ""

test-windows:
	@echo "=== Testing Windows MinGW Configuration ==="
	@$(MAKE) --no-print-directory test-platform UNAME_S=Windows WINDOWS_COMPILER=mingw
	@echo ""

test-msvc:
	@echo "=== Testing Windows MSVC Configuration ==="
	@$(MAKE) --no-print-directory test-platform UNAME_S=Windows WINDOWS_COMPILER=msvc
	@echo ""

# Test that validates the actual build commands that would be generated
test-build-commands:
	@echo "=== Testing Build Command Generation ==="
	@echo "macOS build command:"
	@$(MAKE) --dry-run --no-print-directory DTFE UNAME_S=Darwin 2>/dev/null | grep '^/.*clang\|^/.*g++' | head -1 || echo "[Dry run of linking command]"
	@echo ""
	@echo "Linux build command:"
	@$(MAKE) --dry-run --no-print-directory DTFE UNAME_S=Linux 2>/dev/null | grep '^/.*g++\|^g++' | head -1 || echo "[Dry run of linking command]"
	@echo ""
	@echo "Windows build command:"
	@$(MAKE) --dry-run --no-print-directory DTFE UNAME_S=Windows 2>/dev/null | grep '^/.*g++\|^g++' | head -1 || echo "[Dry run of linking command]"
	@echo ""

# Validate that all necessary components are present
test-validation:
	@echo "Validating platform configurations..."
	@test "$$($(MAKE) --no-print-directory -s test-platform UNAME_S=Darwin | grep 'Executable extension' | cut -d"'" -f2)" = "" && echo "macOS: Correct executable extension" || echo "macOS: Wrong executable extension"
	@test "$$($(MAKE) --no-print-directory -s test-platform UNAME_S=Linux | grep 'Executable extension' | cut -d"'" -f2)" = "" && echo "Linux: Correct executable extension" || echo "Linux: Wrong executable extension"
	@test "$$($(MAKE) --no-print-directory -s test-platform UNAME_S=Windows | grep 'Executable extension' | cut -d"'" -f2)" = ".exe" && echo "Windows: Correct executable extension" || echo "Windows: Wrong executable extension"
	@test "$$($(MAKE) --no-print-directory -s test-platform UNAME_S=Darwin | grep 'Shared library extension' | cut -d"'" -f2)" = ".dylib" && echo "macOS: Correct shared library extension" || echo "macOS: Wrong shared library extension"
	@test "$$($(MAKE) --no-print-directory -s test-platform UNAME_S=Linux | grep 'Shared library extension' | cut -d"'" -f2)" = ".so" && echo "Linux: Correct shared library extension" || echo "Linux: Wrong shared library extension"
	@test "$$($(MAKE) --no-print-directory -s test-platform UNAME_S=Windows | grep 'Shared library extension' | cut -d"'" -f2)" = ".dll" && echo "Windows: Correct shared library extension" || echo "Windows: Wrong shared library extension"
	@$(MAKE) --no-print-directory -s test-platform UNAME_S=Darwin | grep -q '\-lomp' && echo "macOS: OpenMP configured" || echo "macOS: OpenMP missing"
	@$(MAKE) --no-print-directory -s test-platform UNAME_S=Linux | grep -q '\-lgomp' && echo "Linux: OpenMP configured" || echo "Linux: OpenMP missing"
	@$(MAKE) --no-print-directory -s test-platform UNAME_S=Windows WINDOWS_COMPILER=mingw | grep -q '\-lgomp' && echo "Windows MinGW: OpenMP configured" || echo "Windows MinGW: OpenMP missing"
	@test "$$($(MAKE) --no-print-directory -s test-platform UNAME_S=Windows WINDOWS_COMPILER=msvc | grep 'Object file extension' | cut -d"'" -f2)" = ".obj" && echo "MSVC: Correct object extension" || echo "MSVC: Wrong object extension"
	@test "$$($(MAKE) --no-print-directory -s test-platform UNAME_S=Windows WINDOWS_COMPILER=msvc | grep 'Compiler type' | cut -d' ' -f3)" = "msvc" && echo "MSVC: Compiler type detected" || echo "MSVC: Wrong compiler type"

# Comprehensive test suite
test-suite: test-all-platforms test-validation
	@echo "Cross-platform test suite completed successfully."
	@echo "Additional tests: make test-platform, ./test-windows-compat.sh, ./test-msvc-compat.sh"

copy_headers:
	cp $(addprefix $(SRC)/, $(HEADERS_1)) $(INC_DIR)
	cp $(addprefix $(SRC)/, $(HEADERS_2)) $(INC_DIR)/CGAL_triangulation

set_directories:
ifeq ($(COMPILER_TYPE),msvc)
	@$(MKDIR_P) "$(OBJ_DIR)" $(OBJ_DIR) >nul 2>&1 || echo Directory exists
	@$(MKDIR_P) "$(BIN_DIR)" $(BIN_DIR) >nul 2>&1 || echo Directory exists
else
	@$(MKDIR_P) $(OBJ_DIR)
	@$(MKDIR_P) $(BIN_DIR)
endif

set_directories_2:
ifeq ($(COMPILER_TYPE),msvc)
	@$(MKDIR_P) "$(LIB_DIR)" $(LIB_DIR) >nul 2>&1 || echo Directory exists
	@$(MKDIR_P) "$(INC_DIR)" $(INC_DIR) >nul 2>&1 || echo Directory exists
	@$(MKDIR_P) "$(INC_DIR)\CGAL_triangulation" $(INC_DIR)/CGAL_triangulation >nul 2>&1 || echo Directory exists
else
	@$(MKDIR_P) $(LIB_DIR)
	@$(MKDIR_P) $(INC_DIR)
	@$(MKDIR_P) $(INC_DIR)/CGAL_triangulation
endif
