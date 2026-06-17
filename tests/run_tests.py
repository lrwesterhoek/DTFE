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


def write_positions_input(filepath, particles, box):
    with open(filepath, "w") as f:
        f.write(f"{len(particles)}\n")
        f.write("  ".join(str(b) for b in box) + "\n")
        for p in particles:
            f.write("  ".join(str(v) for v in p) + "\n")


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
