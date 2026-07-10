# Standalone example: access the Delaunay tessellation via the DTFE() library call.
#
# STATUS: UNMAINTAINED. main_particle_density.cpp targets the 2011 upstream API
# (vertexPositionMatrix free function, User_options::field, the old DTFE()/
# writeOutputData signatures) and no longer compiles against the current sources.
# The Makefile below is updated to the current build layout, so only the example
# code itself needs porting to revive it.
#
# Builds against the shared library produced by `make library` at the repo root
# (run automatically as a prerequisite). The OPTIONS block below MUST match the
# root Makefile's OPTIONS_COMMON (they set the data-structure layout, so a
# mismatch gives silent ABI corruption, not a link error).
#
#   make            # builds ../../libDTFE and then DTFE_particle_density

ROOT = ../..

# Platform paths: reuse the same defaults as the root Makefile (Homebrew on macOS, /usr on Linux)
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    ARCH := $(shell uname -m)
    ifeq ($(ARCH),arm64)
        BREW_PREFIX = /opt/homebrew
    else
        BREW_PREFIX = /usr/local
    endif
    GSL_PATH   = $(BREW_PREFIX)/opt/gsl
    BOOST_PATH = $(BREW_PREFIX)/opt/boost
    CGAL_PATH  = $(BREW_PREFIX)/opt/cgal
    GMP_PATH   = $(BREW_PREFIX)/opt/gmp
    MPFR_PATH  = $(BREW_PREFIX)/opt/mpfr
    CC := $(shell which $(BREW_PREFIX)/opt/llvm/bin/clang++ 2>/dev/null || which clang++ 2>/dev/null || echo "clang++")
    OPENMP_FLAG = -fopenmp=libomp
else
    GSL_PATH   = /usr
    BOOST_PATH = /usr
    CGAL_PATH  = /usr
    GMP_PATH   = /usr
    MPFR_PATH  = /usr
    CC := $(shell which g++ 2>/dev/null || echo "g++")
    OPENMP_FLAG = -fopenmp
endif

# ---- MUST match the root Makefile's OPTIONS_COMMON ----
OPTIONS  = -DNO_DIM=3
OPTIONS += -DVELOCITY
OPTIONS += -DSCALAR
OPTIONS += -DNO_SCALARS=1
OPTIONS += -DINPUT_FILE_DEFAULT=105
OPTIONS += -DMPC_UNIT=1000
OPTIONS += -DOUTPUT_FILE_DEFAULT=101
OPTIONS += -DOPEN_MP
OPTIONS += -DTEST_PADDING
OPTIONS += -DREDSHIFT_SPACE
OPTIONS += -DTRIANGULATION
OPTIONS += -DBOOST_TIMER_ENABLE_DEPRECATED
OPTIONS += -DBOOST_ALLOW_DEPRECATED_HEADERS

INCLUDES  = -I $(GSL_PATH)/include -I $(BOOST_PATH)/include -I $(CGAL_PATH)/include \
            -I $(GMP_PATH)/include -I $(MPFR_PATH)/include
LIBRARIES = -L$(GSL_PATH)/lib -L$(BOOST_PATH)/lib -L$(CGAL_PATH)/lib \
            -L$(GMP_PATH)/lib -L$(MPFR_PATH)/lib

COMPILE_FLAGS = -O3 -std=c++17 -frounding-math -Wno-deprecated-declarations $(OPENMP_FLAG) -DNDEBUG $(OPTIONS)
LIBS = -L$(ROOT) -lDTFE $(LIBRARIES) -lboost_thread -lboost_filesystem -lboost_program_options \
       -lgsl -lgslcblas -lm -lgmp -lmpfr

DTFE_particle_density: main_particle_density.cpp
	$(MAKE) -C $(ROOT) library
	$(CC) $(COMPILE_FLAGS) $(INCLUDES) main_particle_density.cpp $(LIBS) -o DTFE_particle_density

clean:
	rm -f DTFE_particle_density

.PHONY: clean
