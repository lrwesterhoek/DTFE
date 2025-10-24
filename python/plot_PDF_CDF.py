"""
Critical Point PDF/CDF Analysis

Identifies critical points in density fields using Morse theory, computes their
probability density functions and cumulative distribution functions, and analyzes
their statistical properties across redshift.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
from pathlib import Path
import warnings
import dtfe_functions as dtfe
from dtfe_shared import save_plot_to_multiple_paths
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration Section
# ============================================================================

BASE_DATA_DIR = "output/TNG50-3-Dark"
OUTPUT_DIR = "python/figures/critical_point_analysis"

FIELD_RESOLUTION = 512
BOX_SIZE = 51.7  # Mpc

# Critical point detection parameters
GRADIENT_THRESHOLD = 0.1
MIN_POINT_DISTANCE = 5
SMOOTHING_SIGMA = 10.0

# Snapshot to redshift mapping
SNAPSHOT_TO_REDSHIFT = dtfe.SNAPSHOT_TO_REDSHIFT

DPI = 300

# ============================================================================

def get_critical_point_densities(density_field, minima_mask, maxima_mask, saddle1_mask, saddle2_mask):
    """Get normalized density values at critical points."""
    mean_density = np.mean(density_field)
    std_density = np.std(density_field)
    
    def normalize_density(density):
        return (density - mean_density) / std_density
    
    nu_minima = normalize_density(density_field[minima_mask])
    nu_maxima = normalize_density(density_field[maxima_mask])
    nu_1saddle = normalize_density(density_field[saddle1_mask])
    nu_2saddle = normalize_density(density_field[saddle2_mask])
    
    return {
        'minima': nu_minima,
        'maxima': nu_maxima,
        '1saddle': nu_1saddle,
        '2saddle': nu_2saddle
    }

def compute_pdf(values, bins):
    """Compute probability density function."""
    if len(values) == 0:
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        return bin_centers, np.zeros_like(bin_centers)
    
    counts, bin_edges = np.histogram(values, bins=bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    counts = np.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0)
    
    return bin_centers, counts

def compute_cdf(values, bins):
    """Compute cumulative distribution function."""
    if len(values) == 0:
        return bins[1:], np.zeros(len(bins)-1)
    
    counts, bin_edges = np.histogram(values, bins=bins)
    cdf = np.cumsum(counts) / len(values)
    
    return bin_edges[1:], cdf

def gaussian(x, A, mu, sigma):
    """Gaussian function for fitting."""
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def fit_gaussian(x_vals, y_vals):
    """Fit Gaussian to data."""
    if len(x_vals) == 0 or len(y_vals) == 0:
        return [0.0, 0.0, 1.0], None
    
    valid_mask = np.isfinite(x_vals) & np.isfinite(y_vals) & (y_vals >= 0)
    if not np.any(valid_mask):
        return [0.0, 0.0, 1.0], None
    
    x_clean = x_vals[valid_mask]
    y_clean = y_vals[valid_mask]
    
    if len(x_clean) < 3 or np.max(y_clean) == 0:
        return [0.0, 0.0, 1.0], None
    
    A_guess = np.max(y_clean)
    mu_guess = x_clean[np.argmax(y_clean)]
    sigma_guess = 1.0
    p0 = [A_guess, mu_guess, sigma_guess]
    
    try:
        popt, pcov = curve_fit(gaussian, x_clean, y_clean, p0=p0)
        return popt, pcov
    except (RuntimeError, ValueError):
        return p0, None

def plot_pdfs(nu_values, redshift=None, output_path=None):
    """Plot probability density functions of critical points."""
    nbins = 200
    bins = np.linspace(-4, 4, nbins+1)
    
    bin_centers_minima, pdf_minima = compute_pdf(nu_values['minima'], bins)
    bin_centers_maxima, pdf_maxima = compute_pdf(nu_values['maxima'], bins)
    bin_centers_1saddle, pdf_1saddle = compute_pdf(nu_values['1saddle'], bins)
    bin_centers_2saddle, pdf_2saddle = compute_pdf(nu_values['2saddle'], bins)
    
    popt_min, _ = fit_gaussian(bin_centers_minima, pdf_minima)
    popt_max, _ = fit_gaussian(bin_centers_maxima, pdf_maxima)
    popt_1s, _ = fit_gaussian(bin_centers_1saddle, pdf_1saddle)
    popt_2s, _ = fit_gaussian(bin_centers_2saddle, pdf_2saddle)
    
    xfine = np.linspace(-4, 4, 400)
    fit_minima = gaussian(xfine, *popt_min)
    fit_maxima = gaussian(xfine, *popt_max)
    fit_1saddle = gaussian(xfine, *popt_1s)
    fit_2saddle = gaussian(xfine, *popt_2s)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if len(nu_values['minima']) > 0:
        ax.plot(bin_centers_minima, pdf_minima, marker="o", ls="", ms=3, color='blue', alpha=0.5)
        ax.plot(xfine, fit_minima, color="blue", lw=2.0, 
                label=f"Minima Fit (N={len(nu_values['minima'])}, μ={popt_min[1]:.3f}, σ={popt_min[2]:.3f})")
    
    if len(nu_values['1saddle']) > 0:
        ax.plot(bin_centers_1saddle, pdf_1saddle, marker="s", ls="", ms=3, color='green', alpha=0.5)
        ax.plot(xfine, fit_1saddle, color="green", lw=2.0, 
                label=f"1-saddle Fit (N={len(nu_values['1saddle'])}, μ={popt_1s[1]:.3f}, σ={popt_1s[2]:.3f})")
    
    if len(nu_values['2saddle']) > 0:
        ax.plot(bin_centers_2saddle, pdf_2saddle, marker="^", ls="", ms=3, color='purple', alpha=0.5)
        ax.plot(xfine, fit_2saddle, color="purple", lw=2.0, 
                label=f"2-saddle Fit (N={len(nu_values['2saddle'])}, μ={popt_2s[1]:.3f}, σ={popt_2s[2]:.3f})")
    
    if len(nu_values['maxima']) > 0:
        ax.plot(bin_centers_maxima, pdf_maxima, marker="d", ls="", ms=3, color='red', alpha=0.5)
        ax.plot(xfine, fit_maxima, color="red", lw=2.0, 
                label=f"Maxima Fit (N={len(nu_values['maxima'])}, μ={popt_max[1]:.3f}, σ={popt_max[2]:.3f})")
    
    ax.set_xlabel("Normalized Density Threshold ν", fontsize=14)
    ax.set_ylabel("Probability Density", fontsize=14)
    ax.set_xlim(-4, 4)
    ax.set_ylim(0, None)
    
    title = "Probability Density Functions of Critical Points"
    if redshift is not None:
        title += f' (z={redshift:.2f})'
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=10, loc='upper right', framealpha=0.7)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        save_plot_to_multiple_paths(plt.gcf(), output_path, dpi=DPI, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return {
        'minima': popt_min,
        '1saddle': popt_1s,
        '2saddle': popt_2s,
        'maxima': popt_max
    }

def plot_cdfs(nu_values, redshift=None, output_path=None):
    """Plot cumulative distribution functions of critical points."""
    nbins = 200
    bins = np.linspace(-4, 4, nbins+1)
    
    bin_edges_minima, cdf_minima = compute_cdf(nu_values['minima'], bins)
    bin_edges_maxima, cdf_maxima = compute_cdf(nu_values['maxima'], bins)
    bin_edges_1saddle, cdf_1saddle = compute_cdf(nu_values['1saddle'], bins)
    bin_edges_2saddle, cdf_2saddle = compute_cdf(nu_values['2saddle'], bins)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if len(nu_values['minima']) > 0:
        ax.plot(bin_edges_minima, cdf_minima, label=f"Minima (N={len(nu_values['minima'])})", color='blue', lw=2.0)
    if len(nu_values['1saddle']) > 0:
        ax.plot(bin_edges_1saddle, cdf_1saddle, label=f"1-saddle (N={len(nu_values['1saddle'])})", color='green', lw=2.0)
    if len(nu_values['2saddle']) > 0:
        ax.plot(bin_edges_2saddle, cdf_2saddle, label=f"2-saddle (N={len(nu_values['2saddle'])})", color='purple', lw=2.0)
    if len(nu_values['maxima']) > 0:
        ax.plot(bin_edges_maxima, cdf_maxima, label=f"Maxima (N={len(nu_values['maxima'])})", color='red', lw=2.0)
    
    xfine = np.linspace(-4, 4, 400)
    ax.plot(xfine, norm.cdf(xfine), label="Standard Normal", color='black', lw=1.5, ls='--')
    
    ax.set_xlabel("Normalized Density Threshold ν", fontsize=14)
    ax.set_ylabel("Cumulative Probability", fontsize=14)
    ax.set_xlim(-4, 4)
    ax.set_ylim(0, 1.05)
    
    title = "Cumulative Distribution Functions of Critical Points"
    if redshift is not None:
        title += f' (z={redshift:.2f})'
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=10, loc='lower right', framealpha=0.7)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.axhline(y=0.5, color='black', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    
    if output_path:
        save_plot_to_multiple_paths(plt.gcf(), output_path, dpi=DPI, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_number_density_vs_threshold(nu_values, box_size_mpc, redshift=None, output_path=None):
    """Plot number density of critical points vs threshold."""
    volume = box_size_mpc**3
    thresholds = np.linspace(-4, 4, 100)
    
    maxima_density = np.zeros_like(thresholds)
    minima_density = np.zeros_like(thresholds)
    saddle1_density = np.zeros_like(thresholds)
    saddle2_density = np.zeros_like(thresholds)
    
    for i, threshold in enumerate(thresholds):
        if len(nu_values['maxima']) > 0:
            maxima_density[i] = np.sum(nu_values['maxima'] > threshold) / volume
        if len(nu_values['minima']) > 0:
            minima_density[i] = np.sum(nu_values['minima'] > threshold) / volume
        if len(nu_values['1saddle']) > 0:
            saddle1_density[i] = np.sum(nu_values['1saddle'] > threshold) / volume
        if len(nu_values['2saddle']) > 0:
            saddle2_density[i] = np.sum(nu_values['2saddle'] > threshold) / volume
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if len(nu_values['maxima']) > 0:
        ax.plot(thresholds, maxima_density, color='red', lw=2.0, label=f"Maxima (N={len(nu_values['maxima'])})")
    if len(nu_values['1saddle']) > 0:
        ax.plot(thresholds, saddle1_density, color='green', lw=2.0, label=f"1-Saddle (N={len(nu_values['1saddle'])})")
    if len(nu_values['2saddle']) > 0:
        ax.plot(thresholds, saddle2_density, color='purple', lw=2.0, label=f"2-Saddle (N={len(nu_values['2saddle'])})")
    if len(nu_values['minima']) > 0:
        ax.plot(thresholds, minima_density, color='blue', lw=2.0, label=f"Minima (N={len(nu_values['minima'])})")
    
    total_density = maxima_density + minima_density + saddle1_density + saddle2_density
    total_count = sum(len(nu_values[k]) for k in nu_values)
    ax.plot(thresholds, total_density, color='black', lw=3.0, ls='--', label=f"Total (N={total_count})")
    
    ax.set_xlabel("Normalized Density Threshold ν", fontsize=14)
    ax.set_ylabel("Number Density (Mpc$^{-3}$)", fontsize=14)
    ax.set_xlim(-4, 4)
    
    title = "Number Density of Critical Points vs. Density Threshold"
    if redshift is not None:
        title += f' (z={redshift:.2f})'
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=10, loc='upper right', framealpha=0.7)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    
    plt.tight_layout()
    
    if output_path:
        save_plot_to_multiple_paths(plt.gcf(), output_path, dpi=DPI, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def process_snapshot(snapshot, redshift):
    """Process a single snapshot for critical point analysis."""
    
    print(f"\nProcessing snapshot {snapshot} (z={redshift:.2f})")
    
    snapshot_dir = Path(BASE_DATA_DIR) / f"snapdir_{snapshot}"
    output_dir = Path(OUTPUT_DIR) / f"snapshot_{snapshot}_z{redshift:.2f}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    density_file = snapshot_dir / 'output.a_den'
    
    if not density_file.exists():
        print(f"  Warning: Density file not found")
        return False
    
    field_shape = (FIELD_RESOLUTION, FIELD_RESOLUTION, FIELD_RESOLUTION)
    
    try:
        print(f"  Loading and processing density field")
        density_field = dtfe.load_binary_field(str(density_file), field_shape, num_components=1)
        delta_field = dtfe.calculate_density_contrast(density_field)
        delta_field = dtfe.smooth_field(delta_field, sigma=SMOOTHING_SIGMA)
        
        print(f"  Computing gradient and finding critical points")
        grad_x, grad_y, grad_z = dtfe.calculate_gradient_fft(delta_field)
        grad_magnitude = dtfe.calculate_gradient_magnitude(grad_x, grad_y, grad_z)

        critical_mask = dtfe.find_gradient_critical_points(
            grad_magnitude,
            gradient_threshold=GRADIENT_THRESHOLD,
            min_distance=MIN_POINT_DISTANCE
        )

        print(f"  Computing Hessian and eigenvalues")
        hessian_components = dtfe.calculate_hessian_fft(delta_field)

        # Get critical point indices
        critical_indices = np.where(critical_mask)
        num_critical = len(critical_indices[0])

        print(f"  Classifying {num_critical} critical points using eigenvalues")
        # Initialize classification masks
        maxima_mask = np.zeros_like(critical_mask, dtype=bool)
        minima_mask = np.zeros_like(critical_mask, dtype=bool)
        saddle1_mask = np.zeros_like(critical_mask, dtype=bool)
        saddle2_mask = np.zeros_like(critical_mask, dtype=bool)

        # Extract Hessian components
        hxx = hessian_components['hxx']
        hxy = hessian_components['hxy']
        hxz = hessian_components['hxz']
        hyy = hessian_components['hyy']
        hyz = hessian_components['hyz']
        hzz = hessian_components['hzz']

        # Build Hessian matrices for critical points and compute eigenvalues
        i_indices, j_indices, k_indices = critical_indices
        hessian_matrices = np.zeros((num_critical, 3, 3))
        hessian_matrices[:, 0, 0] = hxx[i_indices, j_indices, k_indices]
        hessian_matrices[:, 0, 1] = hxy[i_indices, j_indices, k_indices]
        hessian_matrices[:, 0, 2] = hxz[i_indices, j_indices, k_indices]
        hessian_matrices[:, 1, 0] = hxy[i_indices, j_indices, k_indices]
        hessian_matrices[:, 1, 1] = hyy[i_indices, j_indices, k_indices]
        hessian_matrices[:, 1, 2] = hyz[i_indices, j_indices, k_indices]
        hessian_matrices[:, 2, 0] = hxz[i_indices, j_indices, k_indices]
        hessian_matrices[:, 2, 1] = hyz[i_indices, j_indices, k_indices]
        hessian_matrices[:, 2, 2] = hzz[i_indices, j_indices, k_indices]

        # Compute eigenvalues using standard library
        eigenvals = np.linalg.eigvalsh(hessian_matrices)

        # Classify based on eigenvalue signs
        positive_count = np.sum(eigenvals > 0, axis=1)
        negative_count = np.sum(eigenvals < 0, axis=1)

        for idx in range(num_critical):
            i, j, k = i_indices[idx], j_indices[idx], k_indices[idx]
            pos, neg = positive_count[idx], negative_count[idx]

            if pos == 3 and neg == 0:
                minima_mask[i, j, k] = True
            elif pos == 0 and neg == 3:
                maxima_mask[i, j, k] = True
            elif pos == 2 and neg == 1:
                saddle1_mask[i, j, k] = True
            elif pos == 1 and neg == 2:
                saddle2_mask[i, j, k] = True

        critical_point_masks = (maxima_mask, minima_mask, saddle1_mask, saddle2_mask)
        
        nu_values = get_critical_point_densities(
            delta_field, 
            minima_mask, 
            maxima_mask, 
            saddle1_mask, 
            saddle2_mask
        )
        
        print(f"  Creating visualizations")
        
        pdf_params = plot_pdfs(
            nu_values, redshift=redshift,
            output_path=output_dir / f"critical_point_pdfs_z{redshift:.2f}.png"
        )
        
        plot_cdfs(
            nu_values, redshift=redshift,
            output_path=output_dir / f"critical_point_cdfs_z{redshift:.2f}.png"
        )

        plot_number_density_vs_threshold(
            nu_values, box_size_mpc=BOX_SIZE, redshift=redshift,
            output_path=output_dir / f"number_density_vs_threshold_z{redshift:.2f}.png"
        )
        
        summary_file = output_dir / f'critical_point_analysis_summary_z{redshift:.2f}.txt'
        
        total_counts = {
            'maxima': np.sum(maxima_mask),
            'minima': np.sum(minima_mask),
            '1saddle': np.sum(saddle1_mask),
            '2saddle': np.sum(saddle2_mask)
        }
        total_count = sum(total_counts.values())
        volume = BOX_SIZE**3
        
        with open(summary_file, 'w') as f:
            f.write(f"Critical Point Analysis Summary\n")
            f.write(f"Snapshot: {snapshot}, Redshift: {redshift:.2f}\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Parameters:\n")
            f.write(f"  Box size: {BOX_SIZE} Mpc\n")
            f.write(f"  Grid resolution: {FIELD_RESOLUTION}³\n")
            f.write(f"  Smoothing sigma: {SMOOTHING_SIGMA}\n")
            f.write(f"  Gradient threshold: {GRADIENT_THRESHOLD}\n")
            f.write(f"  Min point distance: {MIN_POINT_DISTANCE}\n\n")
            f.write(f"Critical Point Counts:\n")
            for cp_type, count in total_counts.items():
                percentage = 100.0 * count / total_count if total_count > 0 else 0.0
                density = count / volume
                f.write(f"  {cp_type.capitalize()}: {count} ({percentage:.1f}%, {density:.6e} Mpc⁻³)\n")
            f.write(f"  Total: {total_count} ({total_count/volume:.6e} Mpc⁻³)\n\n")
            f.write(f"Gaussian Fit Parameters (PDF):\n")
            for cp_type in ['minima', '1saddle', '2saddle', 'maxima']:
                if cp_type in pdf_params:
                    params = pdf_params[cp_type]
                    f.write(f"  {cp_type.capitalize()}: A={params[0]:.4f}, μ={params[1]:.4f}, σ={params[2]:.4f}\n")
        
        print(f"  Found {total_count} critical points")
        return True
        
    except Exception as e:
        print(f"  Error: {e}")
        return False

def create_evolution_summary():
    """Create summary plot of critical point evolution."""
    
    print("\nCreating evolution summary")
    
    redshifts = []
    total_densities = []
    maxima_fractions = []
    minima_fractions = []
    saddle1_fractions = []
    saddle2_fractions = []
    
    for snapshot, redshift in SNAPSHOT_TO_REDSHIFT.items():
        summary_file = Path(OUTPUT_DIR) / f'snapshot_{snapshot}_z{redshift:.2f}' / f'critical_point_analysis_summary_z{redshift:.2f}.txt'
        
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    content = f.read()
                
                import re
                maxima_match = re.search(r'Maxima: \d+ \(([\d.]+)%,', content)
                minima_match = re.search(r'Minima: \d+ \(([\d.]+)%,', content)
                saddle1_match = re.search(r'1saddle: \d+ \(([\d.]+)%,', content)
                saddle2_match = re.search(r'2saddle: \d+ \(([\d.]+)%,', content)
                total_match = re.search(r'Total: \d+ \(([\de.-]+) Mpc', content)
                
                if all([maxima_match, minima_match, saddle1_match, saddle2_match, total_match]):
                    redshifts.append(redshift)
                    maxima_fractions.append(float(maxima_match.group(1)))
                    minima_fractions.append(float(minima_match.group(1)))
                    saddle1_fractions.append(float(saddle1_match.group(1)))
                    saddle2_fractions.append(float(saddle2_match.group(1)))
                    total_densities.append(float(total_match.group(1)))
                    
            except Exception:
                continue
    
    if len(redshifts) > 0:
        sorted_indices = np.argsort(redshifts)
        redshifts = np.array(redshifts)[sorted_indices]
        total_densities = np.array(total_densities)[sorted_indices]
        maxima_fractions = np.array(maxima_fractions)[sorted_indices]
        minima_fractions = np.array(minima_fractions)[sorted_indices]
        saddle1_fractions = np.array(saddle1_fractions)[sorted_indices]
        saddle2_fractions = np.array(saddle2_fractions)[sorted_indices]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax1.semilogy(redshifts, total_densities, 'ko-', markersize=4)
        ax1.set_xlabel('Redshift z')
        ax1.set_ylabel('Total Number Density (Mpc$^{-3}$)')
        ax1.set_title('Evolution of Critical Point Number Density')
        ax1.grid(True, alpha=0.3)
        ax1.invert_xaxis()
        
        ax2.plot(redshifts, maxima_fractions, 'ro-', label='Maxima', markersize=4)
        ax2.plot(redshifts, minima_fractions, 'bo-', label='Minima', markersize=4)
        ax2.plot(redshifts, saddle1_fractions, 'go-', label='1-Saddle', markersize=4)
        ax2.plot(redshifts, saddle2_fractions, 'mo-', label='2-Saddle', markersize=4)
        ax2.set_xlabel('Redshift z')
        ax2.set_ylabel('Fraction (%)')
        ax2.set_title('Evolution of Critical Point Type Fractions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.invert_xaxis()
        
        plt.tight_layout()
        
        evolution_file = Path(OUTPUT_DIR) / 'critical_point_evolution_summary.png'
        save_plot_to_multiple_paths(fig, evolution_file, dpi=DPI, bbox_inches='tight')
        plt.close(fig)

def main():
    """Process all snapshots for critical point analysis."""
    
    print("Starting critical point PDF/CDF analysis")
    print(f"Data directory: {BASE_DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Processing {len(SNAPSHOT_TO_REDSHIFT)} snapshots")
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    processed_snapshots = 0
    failed_snapshots = []
    
    for snapshot, redshift in SNAPSHOT_TO_REDSHIFT.items():
        if process_snapshot(snapshot, redshift):
            processed_snapshots += 1
        else:
            failed_snapshots.append((snapshot, redshift))
    
    if processed_snapshots > 1:
        try:
            create_evolution_summary()
        except Exception as e:
            print(f"\nError creating evolution summary: {e}")
    
    print(f"\nCompleted: {processed_snapshots}/{len(SNAPSHOT_TO_REDSHIFT)} snapshots processed")
    
    if failed_snapshots:
        print(f"Failed snapshots: {', '.join(f'{s} (z={z:.2f})' for s, z in failed_snapshots)}")
    
    print(f"Output saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()