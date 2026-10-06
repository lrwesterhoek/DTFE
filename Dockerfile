# DTFE / PS-DTFE build image (Ubuntu). Builds BOTH binaries (CPU-only; the Metal GPU path is
# macOS-only) and runs the fast test battery as a build sanity stage, so a successful
# `docker build` is a known-good toolchain + build + smoke-tested binaries.
#
#   docker build -t dtfe .
#   docker run --rm -v $PWD/data:/data dtfe /opt/dtfe/PS-DTFE /data/snap.hdf5 /data/out \
#       --grid 128 --periodic --field density --MpcUnit 1
#
# License: GPL-3.0 (see LICENSE.md) -- this Dockerfile only packages the build.

FROM ubuntu:24.04 AS build
ENV DEBIAN_FRONTEND=noninteractive

# the documented dependency set (README "Prerequisites"): GSL, Boost, CGAL (+GMP/MPFR),
# HDF5, FFTW, libdeflate (optional, the cache's codec), plus python3 with numpy/h5py for the test battery
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        make \
        libgsl-dev \
        libboost-all-dev \
        libcgal-dev \
        libtbb-dev \
        libmpfr-dev \
        libhdf5-dev \
        libgmp-dev \
        libfftw3-dev \
        libdeflate-dev \
        pkg-config \
        python3 \
        python3-numpy \
        python3-h5py \
        python3-scipy \
        python3-matplotlib \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/dtfe
COPY . .

# build the three binary pairs CPU-only (single, DOUBLE=1, DIM=2), through the same script CI uses
# (tests/ci_suite.sh: one suite list for both, so they cannot drift). -march=native does not exist
# on all builders -> the image builds for a generic CPU (override with --build-arg ARCH_FLAGS=...).
ARG ARCH_FLAGS="-mtune=generic"
RUN set -eux; ARCH_FLAGS="$ARCH_FLAGS" tests/ci_suite.sh build

# ---- the test battery must pass for the image to exist ----
# Every CPU suite against the single pair, again against the double pair, and the 2D suite. The
# standard-DTFE regression reference stored in the repo was made with macOS's CGAL, which breaks
# Delaunay ties in co-spherical configurations differently: ci_suite.sh gives that one test a fresh
# reference folder (DTFE_TEST_REFERENCE_DIR) instead of deleting the tracked file.
# 'rm -rf tests/tmp' must stay in THIS RUN: files written by a RUN are baked into that layer, and
# deleting them later only adds a whiteout.
RUN set -eux; \
    tests/ci_suite.sh core double 2d; \
    rm -rf tests/tmp

# default entrypoint just documents the two binaries
CMD ["/bin/sh", "-c", "echo 'DTFE image: binaries at /opt/dtfe/DTFE and /opt/dtfe/PS-DTFE (+ -double, -2d; run with --help for options)'"]
