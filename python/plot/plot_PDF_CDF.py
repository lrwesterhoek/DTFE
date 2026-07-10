import _bootstrap  # noqa: F401  (puts python/ on sys.path)
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
import warnings
from dtfelib import fields as dtfe
from dtfelib.figures import save_plot_to_multiple_paths
from dtfelib import make_parser
import config
from dtfelib import figures as style
from dtfelib import pipeline

style.apply()
warnings.filterwarnings('ignore')

OUTPUT_DIR = config.figures_path('critical_points')

SMOOTHING_SIGMA = config.SMOOTHING_SIGMA_CELLS

SNAPSHOT_TO_REDSHIFT = config.SNAPSHOT_TO_REDSHIFT

DPI = config.DPI

def get_critical_point_densities(density_field, minima_mask, maxima_mask, saddle1_mask, saddle2_mask):
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
    if len(values) == 0:
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        return bin_centers, np.zeros_like(bin_centers)
    
    counts, bin_edges = np.histogram(values, bins=bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    counts = np.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0)
    
    return bin_centers, counts

def compute_cdf(values, bins):
    if len(values) == 0:
        return bins[1:], np.zeros(len(bins)-1)
    
    counts, bin_edges = np.histogram(values, bins=bins)
    cdf = np.cumsum(counts) / len(values)
    
    return bin_edges[1:], cdf

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def fit_gaussian(x_vals, y_vals):
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
    
    ax.set_xlabel("Normalized Density Threshold ν")
    ax.set_ylabel("Probability Density")
    ax.set_xlim(-4, 4)
    ax.set_ylim(0, None)
    
    title = "Probability Density Functions of Critical Points"
    if redshift is not None:
        title += f' (z={redshift:.2f})'
    style.set_title(ax, title, fontsize=16)
    ax.legend(loc='upper right', framealpha=0.7)
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
    
    ax.set_xlabel("Normalized Density Threshold ν")
    ax.set_ylabel("Cumulative Probability")
    ax.set_xlim(-4, 4)
    ax.set_ylim(0, 1.05)
    
    title = "Cumulative Distribution Functions of Critical Points"
    if redshift is not None:
        title += f' (z={redshift:.2f})'
    style.set_title(ax, title, fontsize=16)
    ax.legend(loc='lower right', framealpha=0.7)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.axhline(y=0.5, color='black', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    
    if output_path:
        save_plot_to_multiple_paths(plt.gcf(), output_path, dpi=DPI, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_number_density_vs_threshold(nu_values, box_size_mpc, redshift=None, output_path=None):
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
    
    ax.set_xlabel("Normalized Density Threshold ν")
    ax.set_ylabel("Number Density (Mpc$^{-3}$)")
    ax.set_xlim(-4, 4)
    ax.set_yscale('log')

    title = "Number Density of Critical Points vs. Density Threshold"
    if redshift is not None:
        title += f' (z={redshift:.2f})'
    style.set_title(ax, title, fontsize=16)
    ax.legend(loc='upper right', framealpha=0.7)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        save_plot_to_multiple_paths(plt.gcf(), output_path, dpi=DPI, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def process_snapshot(snapshot, redshift, args):

    print(f"\nProcessing snapshot {snapshot} (z={redshift:.2f})")

    output_dir = Path(OUTPUT_DIR) / f"snapshot_{snapshot}_z{redshift:.2f}"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        p = pipeline.products(snapshot, redshift, sim=args.sim, method=args.method)
        cp = p.critical_points()
        box_size = p.fs.meta.box_mpc
        grid_n = p.fs.grid_n
        p.release()

        mean, std = cp['field_mean'], cp['field_std']
        nu_values = {
            'minima': (cp['minima']['values'] - mean) / std,
            'maxima': (cp['maxima']['values'] - mean) / std,
            '1saddle': (cp['saddle1']['values'] - mean) / std,
            '2saddle': (cp['saddle2']['values'] - mean) / std,
        }

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
            nu_values, box_size_mpc=box_size, redshift=redshift,
            output_path=output_dir / f"number_density_vs_threshold_z{redshift:.2f}.png"
        )
        
        summary_file = output_dir / f'critical_point_analysis_summary_z{redshift:.2f}.txt'
        
        total_counts = {
            'maxima': len(nu_values['maxima']),
            'minima': len(nu_values['minima']),
            '1saddle': len(nu_values['1saddle']),
            '2saddle': len(nu_values['2saddle'])
        }
        total_count = sum(total_counts.values())
        volume = box_size**3

        with open(summary_file, 'w') as f:
            f.write(f"Critical Point Analysis Summary\n")
            f.write(f"Snapshot: {snapshot}, Redshift: {redshift:.2f}\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Parameters:\n")
            f.write(f"  Box size: {box_size:g} Mpc\n")
            f.write(f"  Grid resolution: {grid_n}³\n")
            f.write(f"  Smoothing sigma: {SMOOTHING_SIGMA}\n")
            f.write(f"  Gradient threshold (saddles): {config.GRADIENT_THRESHOLD}\n")
            f.write(f"  Footprint / dedup distance: {config.FOOTPRINT_SIZE}\n")
            f.write(f"  Detection: unified pipeline catalog (extrema via footprint filters,\n")
            f.write(f"             saddles via gradient candidates + eigenvalue signature)\n\n")
            f.write(f"Critical Point Counts:\n")
            for cp_type, count in total_counts.items():
                percentage = 100.0 * count / total_count if total_count > 0 else 0.0
                density = count / volume
                f.write(f"  {cp_type.capitalize()}: {count} ({percentage:.1f}%, {density:.6e} Mpc⁻³)\n")
            f.write(f"  Total: {total_count} ({total_count/volume:.6e} Mpc⁻³)\n\n")
            chi = (total_counts['minima'] - total_counts['1saddle']
                   + total_counts['2saddle'] - total_counts['maxima'])
            f.write(f"Morse alternating sum "
                    f"(minima - 1saddle + 2saddle - maxima):\n")
            f.write(f"  chi = {chi}   (expected ~ 0 on a periodic box; "
                    f"|chi|/total = {abs(chi)/total_count:.3f})\n\n" if total_count
                    else f"  chi = {chi}\n\n")
            f.write(f"Data statistics of nu per type (non-parametric):\n")
            for cp_type in ['minima', '1saddle', '2saddle', 'maxima']:
                v = nu_values[cp_type]
                if len(v):
                    f.write(f"  {cp_type.capitalize()}: mean={np.mean(v):.4f}, "
                            f"std={np.std(v):.4f}, median={np.median(v):.4f}\n")
            f.write(f"\nGaussian Fit Parameters (PDF):\n")
            for cp_type in ['minima', '1saddle', '2saddle', 'maxima']:
                if cp_type in pdf_params:
                    params = pdf_params[cp_type]
                    f.write(f"  {cp_type.capitalize()}: A={params[0]:.4f}, μ={params[1]:.4f}, σ={params[2]:.4f}\n")
        
        print(f"  Found {total_count} critical points")
        return True

    except FileNotFoundError:
        method_label = 'ps/dtfe' if args.method == 'auto' else args.method
        print(f"  skipping snapshot {snapshot} (no {method_label} fields)")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False

def create_evolution_summary():
    
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
        
        zp1 = 1.0 + redshifts

        def _zaxis(ax):
            ax.set_xscale('log')
            ax.set_xticks([1, 2, 3, 6, 11, 21])
            ax.set_xticklabels(['0', '1', '2', '5', '10', '20'])
            ax.invert_xaxis()
            ax.set_xlabel('Redshift $z$')

        fig, ax1 = plt.subplots(figsize=style.FIGSIZE['single'])
        ax1.semilogy(zp1, total_densities, 'ko-', markersize=4)
        _zaxis(ax1)
        ax1.set_ylabel('Total Number Density (Mpc$^{-3}$)')
        style.set_title(ax1, 'Evolution of Critical Point Number Density')
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        save_plot_to_multiple_paths(
            fig, Path(OUTPUT_DIR) / 'critical_point_density_evolution.png',
            dpi=DPI, bbox_inches='tight')
        plt.close(fig)

        fig, ax2 = plt.subplots(figsize=style.FIGSIZE['single'])
        ax2.plot(zp1, maxima_fractions, 'ro-', label='Maxima', markersize=4)
        ax2.plot(zp1, minima_fractions, 'bo-', label='Minima', markersize=4)
        ax2.plot(zp1, saddle1_fractions, 'go-', label='1-Saddle', markersize=4)
        ax2.plot(zp1, saddle2_fractions, 'mo-', label='2-Saddle', markersize=4)
        _zaxis(ax2)
        ax2.set_ylabel('Fraction (%)')
        style.set_title(ax2, 'Evolution of Critical Point Type Fractions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        save_plot_to_multiple_paths(
            fig, Path(OUTPUT_DIR) / 'critical_point_fractions_evolution.png',
            dpi=DPI, bbox_inches='tight')
        plt.close(fig)

def main():
    parser = make_parser("Critical-point PDF/CDF analysis across the snapshot series "
                         "(--snap is ignored: all snapshots in the config table are processed).")
    args = parser.parse_args()

    print("Starting critical point PDF/CDF analysis")
    print(f"Data directory: {args.data_root / args.sim}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Processing {len(SNAPSHOT_TO_REDSHIFT)} snapshots")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    processed_snapshots = 0
    failed_snapshots = []

    for snapshot, redshift in SNAPSHOT_TO_REDSHIFT.items():
        if process_snapshot(snapshot, redshift, args):
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
