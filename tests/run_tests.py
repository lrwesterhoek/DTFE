#!/usr/bin/env python3

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DTFE_BIN = os.path.join(PROJECT_DIR, "DTFE")
REFERENCE_DIR = os.path.join(SCRIPT_DIR, "reference")

VERBOSE = False
PASSED = 0
FAILED = 0


def log(msg):
    if VERBOSE:
        print(f"  {msg}")


def write_text_input(filepath, particles, box):
    with open(filepath, "w") as f:
        f.write(f"{len(particles)}\n")
        f.write("  ".join(str(b) for b in box) + "\n")
        for p in particles:
            f.write("  ".join(str(v) for v in p) + "\n")


write_positions_input = write_text_input    # identical serialization (positions-only rows)


def read_density_output(filepath):
    values = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                values.append(float(line))
    return values


def run_dtfe(input_file, output_file, grid_size, input_type=111,
             output_type=111, field="density", extra_args=None):
    cmd = [
        DTFE_BIN, input_file, output_file,
        "-i", str(input_type),
        "-o", str(output_type),
        "-g", str(grid_size),
        "-f", field,
        "--MpcUnit", "1",
    ]
    if extra_args:
        cmd.extend(extra_args)

    log(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    output = result.stdout + result.stderr
    if result.returncode != 0:
        log(f"DTFE failed with return code {result.returncode}")
        log(output)
    return result.returncode == 0, output


class AssertionError(Exception):
    pass


def assert_close(actual, expected, tol, msg=""):
    if expected != 0:
        err = abs(actual - expected) / abs(expected)
    else:
        err = abs(actual - expected)
    if err > tol:
        raise AssertionError(
            f"{msg}: expected {expected}, got {actual}, "
            f"error {err:.4e} > tolerance {tol:.4e}")


def report(name, success, msg=""):
    global PASSED, FAILED
    if success:
        PASSED += 1
        print(f"  PASS  {name}")
    else:
        FAILED += 1
        print(f"  FAIL  {name}" + (f" -- {msg}" if msg else ""))


def make_lattice_particles(n_per_dim, box_size, weight=1.0):
    particles = []
    spacing = box_size / n_per_dim
    offset = spacing / 2.0
    for i in range(n_per_dim):
        for j in range(n_per_dim):
            for k in range(n_per_dim):
                x = offset + i * spacing
                y = offset + j * spacing
                z = offset + k * spacing
                particles.append((x, y, z, weight))
    return particles


def make_seeded_random_particles(n, box_size, weight=1.0, seed=42):
    a, c, m = 1664525, 1013904223, 2**32
    state = seed
    particles = []
    for _ in range(n):
        coords = []
        for _ in range(3):
            state = (a * state + c) % m
            coords.append((state / m) * box_size)
        particles.append((*coords, weight))
    return particles


def test_uniform_lattice_density(tmpdir):
    name = "uniform_lattice_density"
    n_per_dim = 6
    box_size = 10.0

    particles = make_lattice_particles(n_per_dim, box_size)
    infile = os.path.join(tmpdir, f"{name}.txt")
    outfile = os.path.join(tmpdir, name)
    write_text_input(infile, particles, (0, box_size, 0, box_size, 0, box_size))

    ok, out = run_dtfe(infile, outfile, grid_size=12, input_type=111)
    if not ok:
        report(name, False, "DTFE execution failed")
        return

    density = read_density_output(outfile + ".den")
    if not density:
        report(name, False, "No density output")
        return

    mean_den = sum(density) / len(density)
    max_den = max(density)
    min_den = min(density)
    log(f"Mean normalized density: {mean_den:.6f} (expected ~1.0)")
    log(f"Min: {min_den:.6f}, Max: {max_den:.6f}")

    try:
        assert_close(mean_den, 1.0, tol=0.6, msg="Mean normalized density")
        variance = sum((d - mean_den)**2 for d in density) / len(density)
        cv = math.sqrt(variance) / mean_den if mean_den > 0 else 0
        log(f"Coefficient of variation: {cv:.4f}")
        if cv > 1.0:
            raise AssertionError(f"Density too non-uniform: CV = {cv:.4f}")
        report(name, True)
    except AssertionError as e:
        report(name, False, str(e))


def test_weight_scaling_invariance(tmpdir):
    name = "weight_scaling_invariance"
    box_size = 10.0
    particles_w1 = make_seeded_random_particles(200, box_size, weight=1.0)
    particles_w2 = [(x, y, z, 2.0 * w) for x, y, z, w in particles_w1]

    infile1 = os.path.join(tmpdir, f"{name}_w1.txt")
    infile2 = os.path.join(tmpdir, f"{name}_w2.txt")
    outfile1 = os.path.join(tmpdir, f"{name}_w1")
    outfile2 = os.path.join(tmpdir, f"{name}_w2")

    box = (0, box_size, 0, box_size, 0, box_size)
    write_text_input(infile1, particles_w1, box)
    write_text_input(infile2, particles_w2, box)

    ok1, _ = run_dtfe(infile1, outfile1, grid_size=8, input_type=111)
    ok2, _ = run_dtfe(infile2, outfile2, grid_size=8, input_type=111)
    if not ok1 or not ok2:
        report(name, False, "DTFE execution failed")
        return

    den1 = read_density_output(outfile1 + ".den")
    den2 = read_density_output(outfile2 + ".den")

    if len(den1) != len(den2):
        report(name, False, "Output size mismatch")
        return

    max_diff = max(abs(a - b) for a, b in zip(den1, den2))
    log(f"Max difference between w=1 and w=2: {max_diff:.2e}")
    if max_diff > 1e-4:
        report(name, False, f"Density should be weight-invariant: max diff = {max_diff:.2e}")
    else:
        report(name, True)


def test_nonuniform_weight_effect(tmpdir):
    name = "nonuniform_weight_effect"
    box_size = 10.0

    particles = make_lattice_particles(4, box_size, weight=1.0)
    heavy_idx = len(particles) // 2
    x, y, z, _ = particles[heavy_idx]
    particles[heavy_idx] = (x, y, z, 100.0)

    infile = os.path.join(tmpdir, f"{name}.txt")
    outfile = os.path.join(tmpdir, name)
    write_text_input(infile, particles, (0, box_size, 0, box_size, 0, box_size))

    ok, _ = run_dtfe(infile, outfile, grid_size=8, input_type=111)
    if not ok:
        report(name, False, "DTFE execution failed")
        return

    density = read_density_output(outfile + ".den")
    if not density:
        report(name, False, "No density output")
        return

    peak = max(density)
    mean = sum(density) / len(density)
    log(f"Peak density: {peak:.4f}, Mean: {mean:.4f}")

    if peak > mean * 1.5:
        report(name, True)
    else:
        report(name, False, f"Expected density peak from heavy particle: peak={peak:.4f} vs mean={mean:.4f}")


def test_reproducibility(tmpdir):
    name = "reproducibility"
    box_size = 10.0
    particles = make_seeded_random_particles(200, box_size)

    infile = os.path.join(tmpdir, f"{name}.txt")
    outfile1 = os.path.join(tmpdir, f"{name}_run1")
    outfile2 = os.path.join(tmpdir, f"{name}_run2")
    write_text_input(infile, particles, (0, box_size, 0, box_size, 0, box_size))

    ok1, _ = run_dtfe(infile, outfile1, grid_size=8, input_type=111)
    ok2, _ = run_dtfe(infile, outfile2, grid_size=8, input_type=111)
    if not ok1 or not ok2:
        report(name, False, "DTFE execution failed")
        return

    den1 = read_density_output(outfile1 + ".den")
    den2 = read_density_output(outfile2 + ".den")

    if len(den1) != len(den2):
        report(name, False, f"Output sizes differ: {len(den1)} vs {len(den2)}")
        return

    max_diff = max(abs(a - b) for a, b in zip(den1, den2))
    log(f"Max difference between runs: {max_diff:.2e}")
    if max_diff > 1e-6:
        report(name, False, f"Outputs differ: max diff = {max_diff:.2e}")
    else:
        report(name, True)


def test_grid_resolution_convergence(tmpdir):
    name = "grid_resolution_convergence"
    box_size = 10.0
    particles = make_seeded_random_particles(500, box_size)

    infile = os.path.join(tmpdir, f"{name}.txt")
    write_text_input(infile, particles, (0, box_size, 0, box_size, 0, box_size))

    means = {}
    for grid in [4, 8, 16, 32]:
        outfile = os.path.join(tmpdir, f"{name}_g{grid}")
        ok, _ = run_dtfe(infile, outfile, grid_size=grid, input_type=111)
        if not ok:
            report(name, False, f"DTFE failed for grid {grid}")
            return
        density = read_density_output(outfile + ".den")
        means[grid] = sum(density) / len(density)
        log(f"Grid {grid:2d}: mean density = {means[grid]:.6f}")

    diff_low = abs(means[4] - means[8])
    diff_high = abs(means[16] - means[32])
    log(f"diff(4,8) = {diff_low:.6f}, diff(16,32) = {diff_high:.6f}")

    if diff_high < diff_low:
        report(name, True)
    else:
        report(name, False,
               f"Not converging: diff(16,32)={diff_high:.6f} >= diff(4,8)={diff_low:.6f}")


def test_density_peak_location(tmpdir):
    name = "density_peak_location"
    box_size = 10.0
    grid_size = 10

    particles = []
    spacing = box_size / 3
    for i in range(3):
        for j in range(3):
            for k in range(3):
                particles.append((
                    spacing * (i + 0.5), spacing * (j + 0.5),
                    spacing * (k + 0.5), 1.0))

    cluster_center = (7.5, 7.5, 7.5)
    state = 123
    a, c, m = 1664525, 1013904223, 2**32
    for _ in range(50):
        coords = []
        for dim in range(3):
            state = (a * state + c) % m
            coords.append(cluster_center[dim] + (state / m - 0.5) * 1.5)
        particles.append((*coords, 1.0))

    infile = os.path.join(tmpdir, f"{name}.txt")
    outfile = os.path.join(tmpdir, name)
    write_text_input(infile, particles, (0, box_size, 0, box_size, 0, box_size))

    ok, _ = run_dtfe(infile, outfile, grid_size=grid_size, input_type=111)
    if not ok:
        report(name, False, "DTFE execution failed")
        return

    density = read_density_output(outfile + ".den")
    if not density:
        report(name, False, "No density output")
        return

    peak_idx = density.index(max(density))
    nz = ny = grid_size
    iz = peak_idx % nz
    iy = (peak_idx // nz) % ny
    ix = peak_idx // (ny * nz)
    dx = box_size / grid_size
    peak_pos = (dx * (ix + 0.5), dx * (iy + 0.5), dx * (iz + 0.5))

    log(f"Peak at grid ({ix},{iy},{iz}), position {peak_pos}")
    log(f"Expected near {cluster_center}")

    dist = math.sqrt(sum((a - b)**2 for a, b in zip(peak_pos, cluster_center)))
    if dist > 2 * dx:
        report(name, False,
               f"Peak at {peak_pos}, expected near {cluster_center}, dist={dist:.2f}")
    else:
        report(name, True)


def test_positions_only_input(tmpdir):
    name = "positions_only_input"
    box_size = 10.0
    n_per_dim = 4
    spacing = box_size / n_per_dim

    particles = []
    for i in range(n_per_dim):
        for j in range(n_per_dim):
            for k in range(n_per_dim):
                particles.append((
                    spacing * (i + 0.5), spacing * (j + 0.5),
                    spacing * (k + 0.5)))

    infile = os.path.join(tmpdir, f"{name}.txt")
    outfile = os.path.join(tmpdir, name)
    write_positions_input(infile, particles,
                          (0, box_size, 0, box_size, 0, box_size))

    ok, _ = run_dtfe(infile, outfile, grid_size=8, input_type=112)
    if not ok:
        report(name, False, "DTFE execution failed")
        return

    density = read_density_output(outfile + ".den")
    if not density:
        report(name, False, "No density output")
        return

    if len(density) != 512:
        report(name, False, f"Expected 512 values, got {len(density)}")
        return

    min_den = min(density)
    if min_den < 0:
        report(name, False, f"Negative density: {min_den}")
    else:
        log(f"Output has {len(density)} values, all non-negative")
        report(name, True)


def test_density_positivity(tmpdir):
    name = "density_positivity"
    box_size = 10.0
    particles = make_seeded_random_particles(300, box_size, seed=99)

    infile = os.path.join(tmpdir, f"{name}.txt")
    outfile = os.path.join(tmpdir, name)
    write_text_input(infile, particles, (0, box_size, 0, box_size, 0, box_size))

    ok, _ = run_dtfe(infile, outfile, grid_size=16, input_type=111)
    if not ok:
        report(name, False, "DTFE execution failed")
        return

    density = read_density_output(outfile + ".den")
    negatives = [d for d in density if d < 0]
    log(f"{len(density)} values, {len(negatives)} negative")
    if negatives:
        report(name, False,
               f"{len(negatives)} negative values, min={min(negatives):.6e}")
    else:
        report(name, True)


def test_output_size_matches_grid(tmpdir):
    name = "output_size_matches_grid"
    box_size = 10.0
    particles = make_seeded_random_particles(100, box_size)

    infile = os.path.join(tmpdir, f"{name}.txt")
    write_text_input(infile, particles, (0, box_size, 0, box_size, 0, box_size))

    for grid in [4, 8, 16]:
        outfile = os.path.join(tmpdir, f"{name}_g{grid}")
        ok, _ = run_dtfe(infile, outfile, grid_size=grid, input_type=111)
        if not ok:
            report(f"{name} (grid={grid})", False, "DTFE execution failed")
            return
        density = read_density_output(outfile + ".den")
        expected = grid ** 3
        if len(density) != expected:
            report(name, False,
                   f"Grid {grid}: expected {expected} values, got {len(density)}")
            return

    report(name, True)


def _has_flag(flag):
    """True when the built binary's --full_help mentions the given option (feature gate for
    tests of flags newer than the binary; captured, never piped -- see the .sh scripts)."""
    result = subprocess.run([DTFE_BIN, "--full_help"], capture_output=True, text=True)
    return flag in result.stdout


def test_exact_average_lattice(tmpdir):
    """--exact-average on a PERFECT periodic lattice: every vertex density equals the mean
    exactly, the linear interpolant is the constant 1, so the exact cell integration must
    return 1 in EVERY grid cell to float rounding (the sampled average is only ~1 on
    average). Gated on the binary supporting the flag."""
    name = "exact_average_lattice"
    if not _has_flag("--exact-average"):
        log("skipping: binary lacks --exact-average")
        return
    n_per_dim, box_size = 6, 10.0
    particles = make_lattice_particles(n_per_dim, box_size)
    infile = os.path.join(tmpdir, f"{name}.txt")
    outfile = os.path.join(tmpdir, name)
    write_text_input(infile, particles, (0, box_size, 0, box_size, 0, box_size))

    ok, out = run_dtfe(infile, outfile, grid_size=12, input_type=111, output_type=101,
                       field="density_a", extra_args=["--periodic", "--exact-average"])
    if not ok:
        report(name, False, "DTFE execution failed")
        return
    density = _read_float32(outfile + ".a_den")
    if len(density) != 12 ** 3:
        report(name, False, f"expected {12**3} cells, got {len(density)}")
        return
    worst = max(abs(d - 1.0) for d in density)
    log(f"max |density_a - 1| = {worst:.3e}")
    try:
        if worst > 1e-4:
            raise AssertionError(f"exact average on a perfect lattice must be 1 to rounding "
                                 f"(max |v-1| = {worst:.3e})")
        report(name, True)
    except AssertionError as e:
        report(name, False, str(e))


def test_exact_average_convergence(tmpdir):
    """Method-1 sampled '_a' density approaches the --exact-average grid MONOTONICALLY as
    --samples grows (Sobol quasi-random sampling: deterministic, no seed involved). Gated
    on the binary supporting the flag."""
    name = "exact_average_convergence"
    if not _has_flag("--exact-average"):
        log("skipping: binary lacks --exact-average")
        return
    box_size, grid = 10.0, 8
    particles = make_seeded_random_particles(300, box_size, seed=7)
    infile = os.path.join(tmpdir, f"{name}.txt")
    write_text_input(infile, particles, (0, box_size, 0, box_size, 0, box_size))

    out_exact = os.path.join(tmpdir, f"{name}_exact")
    ok, _ = run_dtfe(infile, out_exact, grid_size=grid, input_type=111, output_type=101,
                     field="density_a", extra_args=["--exact-average"])
    if not ok:
        report(name, False, "exact-average run failed")
        return
    exact = _read_float32(out_exact + ".a_den")

    l1 = []
    for samples in (100, 1000, 10000):
        out_s = os.path.join(tmpdir, f"{name}_s{samples}")
        ok, _ = run_dtfe(infile, out_s, grid_size=grid, input_type=111, output_type=101,
                         field="density_a", extra_args=["--samples", str(samples)])
        if not ok:
            report(name, False, f"sampled run (--samples {samples}) failed")
            return
        vals = _read_float32(out_s + ".a_den")
        l1.append(sum(abs(a - b) for a, b in zip(vals, exact)) / len(exact))
    log(f"L1 to exact: {l1[0]:.5e} (100) / {l1[1]:.5e} (1000) / {l1[2]:.5e} (10000)")
    try:
        if not (l1[0] > l1[1] > l1[2]):
            raise AssertionError(f"L1 distances not monotone: {l1[0]:.4e}, {l1[1]:.4e}, {l1[2]:.4e}")
        report(name, True)
    except AssertionError as e:
        report(name, False, str(e))


def test_regression(tmpdir, update_ref=False):
    name = "regression"
    box_size = 10.0
    particles = make_seeded_random_particles(150, box_size, weight=1.0, seed=12345)

    infile = os.path.join(tmpdir, f"{name}.txt")
    outfile = os.path.join(tmpdir, name)
    write_text_input(infile, particles, (0, box_size, 0, box_size, 0, box_size))

    ok, _ = run_dtfe(infile, outfile, grid_size=8, input_type=111)
    if not ok:
        report(name, False, "DTFE execution failed")
        return

    density = read_density_output(outfile + ".den")
    ref_file = os.path.join(REFERENCE_DIR, "regression_density.txt")

    if update_ref or not os.path.exists(ref_file):
        os.makedirs(REFERENCE_DIR, exist_ok=True)
        with open(ref_file, "w") as f:
            for d in density:
                f.write(f"{d:.8e}\n")
        if update_ref:
            print(f"  UPDATED  {name} (reference file written)")
        else:
            print(f"  CREATED  {name} (reference file generated on first run)")
        return

    ref_density = read_density_output(ref_file)
    if len(density) != len(ref_density):
        report(name, False,
               f"Size mismatch: {len(density)} vs ref {len(ref_density)}")
        return

    max_rel_err = 0
    max_abs_err = 0
    for d, r in zip(density, ref_density):
        abs_err = abs(d - r)
        max_abs_err = max(max_abs_err, abs_err)
        if abs(r) > 1e-10:
            max_rel_err = max(max_rel_err, abs_err / abs(r))

    log(f"Max absolute error vs reference: {max_abs_err:.2e}")
    log(f"Max relative error vs reference: {max_rel_err:.2e}")

    if max_rel_err > 1e-4 and max_abs_err > 1e-6:
        report(name, False,
               f"Output differs from reference: "
               f"max rel err={max_rel_err:.2e}, max abs err={max_abs_err:.2e}")
    else:
        report(name, True)


def _read_float32(filepath):
    import array
    values = array.array("f")
    with open(filepath, "rb") as f:
        values.frombytes(f.read())
    return values


def _run_raw(cmd_tail, input_file, output_file):
    cmd = [DTFE_BIN, input_file, output_file] + cmd_tail
    log(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return result.returncode == 0, result.stdout + result.stderr


def test_raw_binary_reader(tmpdir):
    """Type-121 raw binary reader: the box is 2*NO_DIM floats (a 4-byte read misaligned every
    later block) and weights/velocities must land in their own buffers (they were read into
    the positions array, leaving the assigned weight/velocity arrays uninitialized)."""
    name = "raw_binary_reader_121"
    import struct
    box = 10.0
    particles = make_lattice_particles(8, box)
    input_file = os.path.join(tmpdir, "raw121.bin")
    output_file = os.path.join(tmpdir, "out121")
    with open(input_file, "wb") as f:
        f.write(struct.pack("<i", len(particles)))
        f.write(struct.pack("<6f", 0.0, box, 0.0, box, 0.0, box))
        for p in particles:
            f.write(struct.pack("<3f", p[0], p[1], p[2]))
        for p in particles:
            f.write(struct.pack("<f", p[3]))
        for _ in particles:
            f.write(struct.pack("<3f", 10.0, 20.0, 30.0))
    ok, output = _run_raw(["-i", "121", "-o", "101", "-g", "8", "-f", "density", "velocity",
                           "--MpcUnit", "1", "--periodic"], input_file, output_file)
    try:
        if not ok:
            raise AssertionError("DTFE failed on the type-121 raw binary input")
        for v in _read_float32(output_file + ".den"):
            assert_close(v, 1.0, 1e-3, "lattice density")
        for i, v in enumerate(_read_float32(output_file + ".vel")):
            expected = (10.0, 20.0, 30.0)[i % 3]
            if v != expected:
                raise AssertionError(f"velocity component {i}: {v} != {expected}")
        report(name, True)
    except AssertionError as e:
        report(name, False, str(e))


def test_text_reader_zero_velocity(tmpdir):
    """Type-111 text reader: velocities are not part of the x y z w format, so they must come
    out exactly zero (the reader used to allocate-and-mark velocity/scalar arrays it never
    filled, copying uninitialized memory into every particle)."""
    name = "text_reader_zero_velocity_111"
    box = 10.0
    particles = make_lattice_particles(6, box)
    input_file = os.path.join(tmpdir, "t111.txt")
    output_file = os.path.join(tmpdir, "out111")
    write_text_input(input_file, particles, (0, box, 0, box, 0, box))
    ok, output = _run_raw(["-i", "111", "-o", "101", "-g", "6", "-f", "density", "velocity",
                           "--MpcUnit", "1", "--periodic"], input_file, output_file)
    try:
        if not ok:
            raise AssertionError("DTFE failed on the text input")
        vel = _read_float32(output_file + ".vel")
        nonzero = sum(1 for v in vel if v != 0.0)
        if nonzero:
            raise AssertionError(f"{nonzero}/{len(vel)} velocity components are non-zero")
        report(name, True)
    except AssertionError as e:
        report(name, False, str(e))


def test_gadget_mass_block_skip(tmpdir):
    """Binary Gadget reader: a per-particle mass block must be seeked past when masses are
    not requested ('--input 101 5'); it used to trip a false 'snapshot file is corrupt'."""
    name = "gadget_mass_block_skip"
    import struct
    box = 10.0
    particles = make_lattice_particles(8, box)
    n = len(particles)
    header = struct.pack("<6i", 0, n, 0, 0, 0, 0) + struct.pack("<6d", 0, 0, 0, 0, 0, 0)
    header += struct.pack("<2d", 1.0, 0.0)              # time, redshift
    header += struct.pack("<2i", 0, 0)                  # flag_sfr, flag_feedback
    header += struct.pack("<6i", 0, n, 0, 0, 0, 0)      # npartTotal
    header += struct.pack("<2i", 0, 1)                  # flag_cooling, num_files
    header += struct.pack("<d", box)                    # BoxSize
    header += b"\x00" * (256 - len(header))

    def block(payload):
        return struct.pack("<i", len(payload)) + payload + struct.pack("<i", len(payload))

    input_file = os.path.join(tmpdir, "massblock.gadget")
    with open(input_file, "wb") as f:
        f.write(block(header))
        f.write(block(b"".join(struct.pack("<3f", p[0], p[1], p[2]) for p in particles)))
        f.write(block(struct.pack(f"<{3 * n}f", *([0.0] * 3 * n))))          # velocities
        f.write(block(struct.pack(f"<{n}I", *range(1, n + 1))))              # ids
        f.write(block(struct.pack(f"<{n}f", *([2.5] * n))))                  # per-particle masses
    try:
        for tail, what in ((["-i", "101"], "masses requested (mass block read)"),
                           (["-i", "101", "5"], "masses unrequested (mass block skipped)")):
            output_file = os.path.join(tmpdir, "outmb")
            ok, output = _run_raw(tail + ["-o", "101", "-g", "8", "-f", "density",
                                          "--MpcUnit", "1", "--periodic"],
                                  input_file, output_file)
            if not ok:
                raise AssertionError(f"DTFE failed with {what}")
            for v in _read_float32(output_file + ".den"):
                assert_close(v, 1.0, 1e-3, what)
        report(name, True)
    except AssertionError as e:
        report(name, False, str(e))


def main():
    global VERBOSE, PASSED, FAILED

    parser = argparse.ArgumentParser(description="DTFE integration tests")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--update-ref", action="store_true",
                        help="Regenerate reference files")
    args = parser.parse_args()
    VERBOSE = args.verbose

    if not os.path.isfile(DTFE_BIN):
        print(f"ERROR: DTFE binary not found at {DTFE_BIN}")
        print("Build it first with: make DTFE")
        sys.exit(1)

    print(f"Running DTFE tests (binary: {DTFE_BIN})")
    print()

    tmpdir = tempfile.mkdtemp(prefix="dtfe_test_")
    try:
        test_uniform_lattice_density(tmpdir)
        test_reproducibility(tmpdir)
        test_grid_resolution_convergence(tmpdir)
        test_density_peak_location(tmpdir)
        test_positions_only_input(tmpdir)
        test_density_positivity(tmpdir)
        test_output_size_matches_grid(tmpdir)
        test_weight_scaling_invariance(tmpdir)
        test_nonuniform_weight_effect(tmpdir)
        test_raw_binary_reader(tmpdir)
        test_text_reader_zero_velocity(tmpdir)
        test_gadget_mass_block_skip(tmpdir)
        test_exact_average_lattice(tmpdir)
        test_exact_average_convergence(tmpdir)
        test_regression(tmpdir, update_ref=args.update_ref)
    finally:
        shutil.rmtree(tmpdir)

    print()
    total = PASSED + FAILED
    print(f"Results: {PASSED}/{total} passed", end="")
    if FAILED:
        print(f", {FAILED} FAILED")
    else:
        print()

    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
