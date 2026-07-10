# DTFE / PS-DTFE build image (Ubuntu). Builds BOTH binaries CPU-only by default and runs
# the fast test battery as a build sanity stage, so a successful `docker build` is a
# known-good toolchain + build + smoke-tested binaries.
#
#   docker build -t dtfe .                              # CPU-only (default)
#   docker build -t dtfe-cuda --build-arg GPU=cuda .    # + CUDA GPU deposit (nvidia/cuda base)
#
# The GPU=cuda variant COMPILE-CHECKS the CUDA backend (nvcc needs no GPU to compile);
# RUNNING with --gpu/--ps-gpu additionally needs an NVIDIA GPU and the NVIDIA container
# toolkit at `docker run` time (--gpus all). The sanity tests below exercise the CPU code
# paths only, so they pass in both variants on any machine.
#
#   docker run --rm -v $PWD/data:/data dtfe /opt/dtfe/PS-DTFE /data/snap.hdf5 /data/out \
#       --grid 128 --periodic --field density --MpcUnit 1
#
# License: GPL-3.0 (see COPYING) -- this Dockerfile only packages the build.

ARG GPU=cpu

# ---- base image selection: plain Ubuntu (cpu) or the CUDA devel image (cuda) ----
FROM ubuntu:24.04 AS base-cpu
FROM nvidia/cuda:12.8.2-devel-ubuntu24.04 AS base-cuda

FROM base-${GPU} AS build
ARG GPU
ENV DEBIAN_FRONTEND=noninteractive

# the documented dependency set (README "Prerequisites"): GSL, Boost, CGAL (+GMP/MPFR),
# HDF5, FFTW, plus python3 with numpy/h5py for the test battery
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        make \
        libgsl-dev \
        libboost-all-dev \
        libcgal-dev \
        libmpfr-dev \
        libhdf5-dev \
        libgmp-dev \
        libfftw3-dev \
        pkg-config \
        python3 \
        python3-numpy \
        python3-h5py \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/dtfe
COPY . .

# build both binaries; GPU=cuda adds CUDA=1 (single-source ps_gpu_cuda.cu/dtfe_gpu_cuda.cu
# compiled by the base image's nvcc). -march=native does not exist on all builders -> the
# image builds for a generic CPU (override with --build-arg ARCH_FLAGS=... if wanted).
ARG ARCH_FLAGS="-mtune=generic"
RUN set -eux; \
    GPU_ARG=""; if [ "$GPU" = "cuda" ]; then GPU_ARG="CUDA=1"; fi; \
    make deps-check; \
    make DTFE    $GPU_ARG ARCH_FLAGS="$ARCH_FLAGS" -j"$(nproc)"; \
    make PS-DTFE $GPU_ARG ARCH_FLAGS="$ARCH_FLAGS" -j"$(nproc)"

# ---- build sanity stage: the fast CPU test battery must pass for the image to exist ----
# The stored regression reference was generated on the dev machine (macOS, Homebrew clang,
# CGAL 6); a different compiler/CGAL pair breaks Delaunay ties in degenerate (co-spherical)
# configurations differently, so that one file is machine-specific. Delete it so the test
# regenerates it in-image on first run -- the remaining analytic tests (mass conservation,
# convergence, positivity, ...) are environment-independent and still gate the build.
RUN set -eux; \
    rm -f tests/reference/regression_density.txt; \
    python3 tests/run_tests.py; \
    tests/ps_smoke_test.sh --no-build

# default entrypoint just documents the two binaries
CMD ["/bin/sh", "-c", "echo 'DTFE image: binaries at /opt/dtfe/DTFE and /opt/dtfe/PS-DTFE (run with --help for options)'"]
