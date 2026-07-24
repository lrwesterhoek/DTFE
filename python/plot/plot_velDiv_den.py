"""Velocity divergence vs density contrast analysis across snapshots.

Data access goes through dtfelib.FieldSet, so the script is method- (DTFE / PS-DTFE),
simulation- and snapshot-agnostic. Snapshots without fields on disk are skipped.

    python3 plot/plot_velDiv_den.py                    # all snapshots, TNG50-4-Dark, auto method
    python3 plot/plot_velDiv_den.py --method dtfe --snap 99
"""

import _bootstrap  # noqa: F401  (puts python/ on sys.path)
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.stats import pearsonr
from dtfelib import fields as dtfe
from dtfelib.figures import save_plot_to_multiple_paths
from dtfelib import FieldSet, make_parser
import config
from dtfelib import figures as style

style.apply()

OUTPUT_DIR = config.figures_path('veldiv')

AXIS_UNITS = config.AXIS_UNITS
VELOCITY_UNITS = config.VELOCITY_UNITS
LINEAR_REGIME_RANGE = (-0.5, 1.0)
SMOOTHING_SIGMA = config.RAW_MAPS_SIGMA

H0 = config.COSMOLOGY['H0']
OMEGA_M0 = config.COSMOLOGY['Omega_m']
OMEGA_LAMBDA0 = config.COSMOLOGY['Omega_Lambda']

SAMPLE_SIZE = 100000
THRESHOLD = 1e-3
RNG = np.random.default_rng(config.CORRELATION_SEED)

SLICE_PLANES_TO_PLOT = [0, 1, 2]

DPI = config.DPI

SNAPSHOT_TO_REDSHIFT = config.SNAPSHOT_TO_REDSHIFT

SLICE_PLANES = config.SLICE_PLANES


def process_fields(density_field, velDiv_field, sigma=0):
    # u-units -> peculiar km/s is applied centrally by FieldSet.load() now
    # (dtfelib.io._VELOCITY_SCALE_EXP) -- no sqrt(a) correction here.
    mean_density = np.mean(density_field)
    delta_field = (density_field - mean_density) / mean_density
    delta_field = dtfe.smooth_field(delta_field, sigma=sigma)
    velDiv_field = dtfe.smooth_field(velDiv_field, sigma=sigma)
    return delta_field, velDiv_field

def compute_regression(delta_field, velDiv_field, threshold=1e-3, delta_range=None):
    delta_flat = delta_field.flatten()
    velDiv_flat = velDiv_field.flatten()
    
    mask = np.abs(delta_flat) > threshold
    
    if delta_range is not None:
        delta_min, delta_max = delta_range
        range_mask = (delta_flat >= delta_min) & (delta_flat <= delta_max)
        mask = mask & range_mask
    
    delta_reg = delta_flat[mask]
    velDiv_reg = velDiv_flat[mask]
    
    slope_reg = np.sum(delta_reg * velDiv_reg) / np.sum(delta_reg**2)
    return slope_reg, delta_reg, velDiv_reg

def sample_scatter_data(delta_reg, velDiv_reg, sample_size=100000, delta_range=None):
    if delta_range is not None:
        delta_min, delta_max = delta_range
        mask = (delta_reg >= delta_min) & (delta_reg <= delta_max)
        delta_reg = delta_reg[mask]
        velDiv_reg = velDiv_reg[mask]
    
    n_points = delta_reg.size
    if n_points > sample_size:
        idx = RNG.choice(n_points, size=sample_size, replace=False)
        return delta_reg[idx], velDiv_reg[idx]

    return delta_reg, velDiv_reg

def setup_symmetric_normalizations(delta_slice, velDiv_slice, residual_slice):
    delta_norm = style.norm_signed_linear(data=delta_slice)
    velDiv_norm = style.norm_signed_linear(data=velDiv_slice)
    residual_norm = colors.Normalize(vmin=velDiv_norm.vmin, vmax=velDiv_norm.vmax)
    return delta_norm, velDiv_norm, residual_norm

def create_density_plot(delta_slice, slice_dim, box_size, delta_norm, redshift, output_path):
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    im = ax.imshow(delta_slice.T, origin='lower', cmap=style.CMAP['delta_raw'], norm=delta_norm,
                   extent=[0, box_size, 0, box_size])

    plane_info = SLICE_PLANES[slice_dim]
    title = "Density Contrast δ"
    if redshift is not None:
        title += f" (z={redshift:.2f})"
    style.set_title(ax, title, fontsize=14)
    ax.set_xlabel(f"{plane_info['axis_labels'][0]} [{AXIS_UNITS}]")
    ax.set_ylabel(f"{plane_info['axis_labels'][1]} [{AXIS_UNITS}]")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                        ticks=style.norm_ticks(delta_norm))
    cbar.set_label("δ")

    ax.set_aspect('equal')
    plt.tight_layout()

    plane_name = plane_info['name']
    filepath = output_path / f"density_contrast_{plane_name}_z{redshift:.2f}.png"
    save_plot_to_multiple_paths(fig, filepath, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def create_velocity_divergence_plot(velDiv_slice, slice_dim, box_size, velDiv_norm, redshift, output_path):
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    im = ax.imshow(velDiv_slice.T, origin='lower', cmap=style.CMAP['divergence'], norm=velDiv_norm,
                   extent=[0, box_size, 0, box_size])

    plane_info = SLICE_PLANES[slice_dim]
    title = "Velocity Divergence"
    if redshift is not None:
        title += f" (z={redshift:.2f})"
    style.set_title(ax, title, fontsize=14)
    ax.set_xlabel(f"{plane_info['axis_labels'][0]} [{AXIS_UNITS}]")
    ax.set_ylabel(f"{plane_info['axis_labels'][1]} [{AXIS_UNITS}]")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                        ticks=style.norm_ticks(velDiv_norm))
    cbar.set_label(f"{VELOCITY_UNITS}/{AXIS_UNITS}")

    ax.set_aspect('equal')
    plt.tight_layout()

    plane_name = plane_info['name']
    filepath = output_path / f"velocity_divergence_{plane_name}_z{redshift:.2f}.png"
    save_plot_to_multiple_paths(fig, filepath, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def create_linear_fit_plot(delta_sample, velDiv_sample, slope_reg, slope_theory, 
                          delta_range, redshift, output_path):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    ax.scatter(delta_sample, velDiv_sample, s=1, alpha=0.5, color='lightblue')
    
    delta_min, delta_max = delta_range
    x_fit = np.linspace(delta_min, delta_max, 100)
    
    ax.plot(x_fit, slope_reg * x_fit, color="red", linewidth=2, 
            label=f"Linear Fit: {slope_reg:.2f}")
    
    ax.plot(x_fit, slope_theory * x_fit, color="green", linestyle="--", linewidth=2, 
            label=f"Theory: {slope_theory:.2f}")
    
    ax.set_xlabel("Density Contrast δ")
    ax.set_ylabel("Velocity Divergence (km/s/Mpc)")
    
    title = "Velocity Divergence vs Density Contrast δ"
    if redshift is not None:
        title += f" (z={redshift:.2f})"
    style.set_title(ax, title, fontsize=14)
    
    ax.set_xlim([delta_min, delta_max])
    
    y_range = max(abs(slope_reg * delta_max), abs(slope_theory * delta_max)) * 1.5
    ax.set_ylim([-y_range, y_range])
    
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    filepath = output_path / f"linear_fit_z{redshift:.2f}.png"
    save_plot_to_multiple_paths(fig, filepath, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def create_residual_plot(residual_slice, slice_dim, box_size, residual_norm, redshift, output_path):
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    im = ax.imshow(residual_slice.T, origin='lower', cmap=style.CMAP['divergence'], norm=residual_norm,
                   extent=[0, box_size, 0, box_size])

    plane_info = SLICE_PLANES[slice_dim]
    title = "Residual Field (∇·v - slope×δ)"
    if redshift is not None:
        title += f" (z={redshift:.2f})"
    style.set_title(ax, title, fontsize=14)
    ax.set_xlabel(f"{plane_info['axis_labels'][0]} [{AXIS_UNITS}]")
    ax.set_ylabel(f"{plane_info['axis_labels'][1]} [{AXIS_UNITS}]")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"{VELOCITY_UNITS}/{AXIS_UNITS}")

    ax.set_aspect('equal')
    plt.tight_layout()

    plane_name = plane_info['name']
    filepath = output_path / f"residual_field_{plane_name}_z{redshift:.2f}.png"
    save_plot_to_multiple_paths(fig, filepath, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def process_snapshot(snapshot, redshift, args):
    print(f"\nProcessing snapshot {snapshot} (z={redshift:.2f})")

    snapdir = args.data_root / args.sim / f'snapdir_{snapshot}'

    try:
        try:
            fs = FieldSet(snapdir, method=args.method, averaged=not args.raw, prefix=args.prefix)
        except ValueError:
            if args.method != 'auto':
                raise
            # both ps_output.* and output.* on disk: keep this script's historical
            # default (standard DTFE) so a plain run reproduces previous behavior
            print("  both PS and DTFE fields present; using --method dtfe")
            fs = FieldSet(snapdir, method='dtfe', averaged=not args.raw)
        print("Loading fields")
        density_field = fs.density(units='mean')   # rho/rho_bar for BOTH methods
        velDiv_field = fs.load('divergence')
    except FileNotFoundError as e:
        print(f"skipping snapshot {snapshot} (no {args.method} fields): {e}")
        return

    box_size = fs.meta.box_mpc

    output_path = Path(OUTPUT_DIR) / f'snapshot_{snapshot}_z{redshift:.2f}'
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        print("Computing cosmological parameters")
        cosmo_params = dtfe.get_cosmology_params(redshift, H0, OMEGA_M0, OMEGA_LAMBDA0)
        a_scale = cosmo_params['a']
        H_z = cosmo_params['H_z']
        f_growth = cosmo_params['f_growth']
        slope_theory = cosmo_params['slope_theory']

        delta_field, velDiv_field = process_fields(density_field, velDiv_field, args.smooth)
        
        print("Computing regression slopes")
        full_slope_reg, full_delta_reg, full_velDiv_reg = compute_regression(
            delta_field, velDiv_field, THRESHOLD
        )
        
        slope_reg, delta_reg, velDiv_reg = compute_regression(
            delta_field, velDiv_field, THRESHOLD, delta_range=LINEAR_REGIME_RANGE
        )
        
        if len(delta_reg) > 2:
            pearson_r, _ = pearsonr(delta_reg, velDiv_reg)
            residual_rms = float(np.std(velDiv_reg - slope_reg * delta_reg))
        else:
            pearson_r, residual_rms = np.nan, np.nan

        print(f"  Theoretical slope: {slope_theory:.2f} km/s/Mpc")
        print(f"  Measured slope: {slope_reg:.2f} km/s/Mpc")
        print(f"  Difference: {(slope_reg - slope_theory) / slope_theory * 100:.1f}%")
        print(f"  Linear-regime Pearson r: {pearson_r:.4f}, "
              f"residual RMS: {residual_rms:.1f} km/s/Mpc")
        
        delta_sample, velDiv_sample = sample_scatter_data(
            delta_reg, velDiv_reg, SAMPLE_SIZE, delta_range=LINEAR_REGIME_RANGE
        )

        residual_field = velDiv_field - slope_reg * delta_field

        print("Creating plots for all planes")
        for slice_dim in SLICE_PLANES_TO_PLOT:
            plane_name = SLICE_PLANES[slice_dim]['name']
            plane_dir = output_path / plane_name

            print(f"  Creating {plane_name} visualizations...")

            delta_slice = dtfe.extract_2d_slice(delta_field, slice_dim)
            velDiv_slice = dtfe.extract_2d_slice(velDiv_field, slice_dim)
            residual_slice = dtfe.extract_2d_slice(residual_field, slice_dim)

            delta_norm, velDiv_norm, residual_norm = setup_symmetric_normalizations(
                delta_slice, velDiv_slice, residual_slice
            )

            create_density_plot(delta_slice, slice_dim, box_size, delta_norm, redshift, plane_dir)
            create_velocity_divergence_plot(velDiv_slice, slice_dim, box_size, velDiv_norm, redshift, plane_dir)
            create_residual_plot(residual_slice, slice_dim, box_size, residual_norm, redshift, plane_dir)

        create_linear_fit_plot(delta_sample, velDiv_sample, slope_reg, slope_theory,
                             LINEAR_REGIME_RANGE, redshift, output_path)

        delta_slice_stats = dtfe.extract_2d_slice(delta_field, 0)
        velDiv_slice_stats = dtfe.extract_2d_slice(velDiv_field, 0)
        residual_slice_stats = dtfe.extract_2d_slice(residual_field, 0)

        summary_file = output_path / f'analysis_summary_z{redshift:.2f}.txt'
        with open(summary_file, 'w') as f:
            f.write(f"Velocity Divergence-Density Analysis Summary\n")
            f.write(f"Snapshot: {snapshot}, Redshift: {redshift:.2f}\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Cosmological Parameters:\n")
            f.write(f"  Scale factor a = {a_scale:.4f}\n")
            f.write(f"  H(z) = {H_z:.2f} km/s/Mpc\n")
            f.write(f"  f(z) = {f_growth:.3f}\n")
            f.write(f"  Theoretical slope = {slope_theory:.2f} km/s/Mpc\n\n")
            f.write(f"Regression Results:\n")
            f.write(f"  Full-range slope = {full_slope_reg:.2f} km/s/Mpc\n")
            f.write(f"  Linear regime slope = {slope_reg:.2f} km/s/Mpc\n")
            f.write(f"  Difference from theory = {(slope_reg - slope_theory) / slope_theory * 100:.1f}%\n")
            f.write(f"  Linear-regime Pearson r = {pearson_r:.4f}\n")
            f.write(f"  Residual RMS (linear regime) = {residual_rms:.2f} km/s/Mpc\n\n")
            f.write(f"Field Statistics (from YZ plane slice):\n")
            f.write(f"  Delta range: [{np.min(delta_slice_stats):.3f}, {np.max(delta_slice_stats):.3f}]\n")
            f.write(f"  VelDiv range: [{np.min(velDiv_slice_stats):.1f}, {np.max(velDiv_slice_stats):.1f}] km/s/Mpc\n")
            f.write(f"  Residual range: [{np.min(residual_slice_stats):.1f}, {np.max(residual_slice_stats):.1f}] km/s/Mpc\n")
        
        print(f"Completed snapshot {snapshot}")
        
    except Exception as e:
        print(f"Error processing snapshot {snapshot}: {str(e)}")
        import traceback
        traceback.print_exc()

def create_evolution_summary():
    print("\nCreating evolution summary")
    
    redshifts = []
    theoretical_slopes = []
    measured_slopes = []
    differences = []
    pearson_rs = []
    residual_rmss = []

    output_path = Path(OUTPUT_DIR)

    import re
    for snapshot, redshift in SNAPSHOT_TO_REDSHIFT.items():
        summary_file = output_path / f'snapshot_{snapshot}_z{redshift:.2f}' / f'analysis_summary_z{redshift:.2f}.txt'

        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    content = f.read()

                theory_match = re.search(r'Theoretical slope = ([-\d.]+) km/s/Mpc', content)
                linear_match = re.search(r'Linear regime slope = ([-\d.]+) km/s/Mpc', content)
                diff_match = re.search(r'Difference from theory = ([-\d.]+)%', content)
                r_match = re.search(r'Linear-regime Pearson r = ([-\d.]+)', content)
                rms_match = re.search(r'Residual RMS \(linear regime\) = ([-\d.]+)', content)

                if theory_match and linear_match and diff_match:
                    redshifts.append(redshift)
                    theoretical_slopes.append(float(theory_match.group(1)))
                    measured_slopes.append(float(linear_match.group(1)))
                    differences.append(float(diff_match.group(1)))
                    pearson_rs.append(float(r_match.group(1)) if r_match else np.nan)
                    residual_rmss.append(float(rms_match.group(1)) if rms_match else np.nan)

            except Exception as e:
                print(f"  Warning: Could not read summary for z={redshift:.2f}: {e}")

    if len(redshifts) > 0:
        sorted_idx = np.argsort(redshifts)
        redshifts = np.array(redshifts)[sorted_idx]
        theoretical_slopes = np.array(theoretical_slopes)[sorted_idx]
        measured_slopes = np.array(measured_slopes)[sorted_idx]
        differences = np.array(differences)[sorted_idx]
        pearson_rs = np.array(pearson_rs)[sorted_idx]
        residual_rmss = np.array(residual_rmss)[sorted_idx]

        zp1 = 1.0 + redshifts

        def _zaxis(ax):
            ax.set_xscale('log')
            ax.set_xticks([1, 2, 3, 6, 11, 21])
            ax.set_xticklabels(['0', '1', '2', '5', '10', '20'])
            ax.invert_xaxis()
            ax.set_xlabel('Redshift $z$')

        fig, ax1 = plt.subplots(figsize=style.FIGSIZE['single'])
        ax1.plot(zp1, theoretical_slopes, 'g--', label='Theory', linewidth=2)
        ax1.plot(zp1, measured_slopes, 'ro-', label='Measured', markersize=4)
        _zaxis(ax1)
        ax1.set_ylabel('Slope (km/s/Mpc)')
        style.set_title(ax1, 'Evolution of Velocity Divergence-Density Slope')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        save_plot_to_multiple_paths(fig, output_path / 'slope_evolution.png',
                                    dpi=DPI, bbox_inches='tight')
        plt.close(fig)

        fig, ax2 = plt.subplots(figsize=style.FIGSIZE['single'])
        ax2.plot(zp1, differences, 'bo-', markersize=4)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        _zaxis(ax2)
        ax2.set_ylabel('Difference from Theory (%)')
        style.set_title(ax2, 'Relative Difference: (Measured - Theory) / Theory × 100%')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        save_plot_to_multiple_paths(fig, output_path / 'slope_deviation_evolution.png',
                                    dpi=DPI, bbox_inches='tight')
        plt.close(fig)

        evolution_data_file = output_path / 'evolution_data.txt'
        with open(evolution_data_file, 'w') as f:
            f.write("Redshift Evolution of Velocity Divergence-Density Relationship\n")
            f.write("=" * 72 + "\n")
            f.write(f"{'Redshift':>8} {'Theory':>10} {'Measured':>10} {'Diff(%)':>8} "
                    f"{'Pearson r':>10} {'RMS':>9}\n")
            f.write("-" * 60 + "\n")
            for z, th, meas, diff, r, rms in zip(redshifts, theoretical_slopes,
                                                 measured_slopes, differences,
                                                 pearson_rs, residual_rmss):
                f.write(f"{z:8.2f} {th:10.2f} {meas:10.2f} {diff:8.1f} "
                        f"{r:10.4f} {rms:9.2f}\n")

        print("Evolution summary created")
    else:
        print("  Warning: No valid summary data found")

def main():
    parser = make_parser("Velocity divergence vs density contrast analysis across snapshots.")
    # this script iterates ALL snapshots by default; --snap restricts to one.
    # --smooth defaults to the configured raw-maps sigma (behavior-preserving).
    parser.set_defaults(snap=None, smooth=SMOOTHING_SIGMA)
    args = parser.parse_args()

    snapshot_items = list(SNAPSHOT_TO_REDSHIFT.items())
    if args.snap is not None:
        key = f"{args.snap:03d}"
        if key not in SNAPSHOT_TO_REDSHIFT:
            parser.error(f"snapshot {key} not in config.SNAPSHOT_TO_REDSHIFT")
        snapshot_items = [(key, SNAPSHOT_TO_REDSHIFT[key])]

    print(f"Starting velocity divergence-density analysis")
    print(f"Processing {len(snapshot_items)} snapshots")

    processed_count = 0
    failed_snapshots = []

    for snapshot, redshift in snapshot_items:
        try:
            process_snapshot(snapshot, redshift, args)
            processed_count += 1
        except Exception as e:
            print(f"Failed to process snapshot {snapshot} (z={redshift:.2f}): {str(e)}")
            failed_snapshots.append((snapshot, redshift))

    if processed_count > 1:
        try:
            create_evolution_summary()
        except Exception as e:
            print(f"Failed to create evolution summary: {str(e)}")

    print(f"\nAnalysis complete")
    print(f"Successfully processed: {processed_count}/{len(snapshot_items)} snapshots")
    
    if failed_snapshots:
        print(f"Failed snapshots: {len(failed_snapshots)}")
        for snapshot, redshift in failed_snapshots:
            print(f"  Snapshot {snapshot} (z={redshift:.2f})")
    
    print(f"\nOutput saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
