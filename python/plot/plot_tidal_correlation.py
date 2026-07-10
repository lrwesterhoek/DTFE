import _bootstrap  # noqa: F401  (puts python/ on sys.path)
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from dtfelib.figures import save_plot_to_multiple_paths
from dtfelib import make_parser
import config
from dtfelib import figures as style
from dtfelib import pipeline

style.apply()

OUTPUT_DIR = config.figures_path('tidal')

DPI = config.DPI
N_PERM = config.CORRELATION_N_PERMUTATIONS
BAND_LO, BAND_HI = config.CORRELATION_BAND_PERCENTILES

COMBOS = {
    'major_t3': dict(
        label=r'$|\cos\theta|$ (void major axis, tidal stretching axis)',
        color='darkgreen'),
    'minor_t1': dict(
        label=r'$|\cos\theta|$ (void minor axis, strongest compression)',
        color='darkred'),
}


def void_axes(cat):
    evals = np.asarray(cat['eigenvalues'], dtype=float)
    evecs = np.asarray(cat['eigenvectors'], dtype=float)
    n = len(evals)
    idx_maj = np.argmin(np.abs(evals), axis=1)
    idx_min = np.argmax(np.abs(evals), axis=1)
    rows = np.arange(n)
    major = evecs[rows, :, idx_maj]
    minor = evecs[rows, :, idx_min]
    major /= np.linalg.norm(major, axis=1, keepdims=True)
    minor /= np.linalg.norm(minor, axis=1, keepdims=True)
    return major, minor


def tidal_frame_at(coords, phi):
    T = pipeline.hessian_from_field(phi)
    t_evals, t_evecs = pipeline.eigh_at_coords(T, coords)
    del T
    e_t1 = t_evecs[:, :, 2]
    e_t3 = t_evecs[:, :, 0]
    return t_evals, e_t1, e_t3


def alignment_with_band(axis_vec, tidal_vec, perm_rows):
    cos = np.abs(np.einsum('nk,nk->n', axis_vec, tidal_vec))
    cos = np.clip(cos, 0.0, 1.0)

    perm_cos = np.abs(np.einsum('bnk,nk->bn', axis_vec[perm_rows], tidal_vec))
    null_means = perm_cos.mean(axis=1)
    lo, hi = np.percentile(null_means, [BAND_LO, BAND_HI])
    return cos, float(lo), float(hi)


def plot_alignment_histogram(cos, combo, redshift, snapshot, lo, hi,
                             output_dir):
    info = COMBOS[combo]
    fig, ax = plt.subplots(figsize=style.FIGSIZE['single'])
    ax.hist(cos, bins=20, range=(0, 1), density=True, alpha=0.75,
            color=info['color'], edgecolor='black', linewidth=1.0)
    ax.axhline(1.0, color='black', linestyle='--', alpha=0.5, linewidth=1.5,
               label='Isotropic (flat)')
    mean_label = rf'$\langle|\cos\theta|\rangle$ = {np.mean(cos):.3f}'
    if config.SHOW_NULL_BANDS:
        mean_label += f' (null: {lo:.3f}-{hi:.3f})'
    ax.axvline(np.mean(cos), color='orange', linestyle='-', linewidth=2.5,
               label=mean_label)
    ax.set_xlabel(info['label'])
    ax.set_ylabel('Probability Density')
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.25, linestyle=':', linewidth=1.0)
    ax.legend(framealpha=0.9, loc='upper left')
    plt.tight_layout()
    save_plot_to_multiple_paths(
        fig, output_dir / f'alignment_{combo}_snap{snapshot}'
                          f'_z{redshift:.2f}.png',
        dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_alignment_evolution(rows, combo):
    info = COMBOS[combo]
    fig, ax = plt.subplots(figsize=style.FIGSIZE['single'])
    x = 1.0 + np.array([r['z'] for r in rows])
    mean = [r[combo]['mean'] for r in rows]
    lo = [r[combo]['lo'] for r in rows]
    hi = [r[combo]['hi'] for r in rows]

    if config.SHOW_NULL_BANDS:
        ax.fill_between(x, lo, hi, color='0.55', alpha=0.35, lw=0,
                        label='95% permutation null')
    ax.plot(x, mean, 'o-', lw=2.5, color=info['color'], label='Measured')
    ax.axhline(0.5, color='black', linestyle='--', alpha=0.4, linewidth=1.2)

    ax.set_xscale('log')
    ax.set_xticks([1, 2, 3, 6, 11, 21])
    ax.set_xticklabels(['0', '1', '2', '5', '10', '20'])
    ax.invert_xaxis()
    ax.set_xlabel('Redshift $z$')
    ax.set_ylabel(info['label'].replace(
        r'$|\cos\theta|$', r'$\langle|\cos\theta|\rangle$'))
    ax.grid(alpha=0.25, linestyle=':', linewidth=1.0)
    ax.legend(framealpha=0.9)
    plt.tight_layout()
    save_plot_to_multiple_paths(
        fig, Path(OUTPUT_DIR) / f'tidal_alignment_evolution_{combo}.png',
        dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def analyze_snapshot(snapshot, redshift, output_root, sim, method):
    print(f"\nProcessing snapshot {snapshot} (z={redshift:.2f})")
    output_dir = output_root / f"snapshot_{snapshot}_z{redshift:.2f}"
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        p = pipeline.products(snapshot, redshift, sim=sim, method=method)
        cat = p.voids()
        coords = np.asarray(cat['coords'], dtype=int)
        n = len(coords)
        if n < 10:
            print(f"  Only {n} voids; skipping")
            return None

        major, minor = void_axes(cat)
        print(f"  Voids: {n}; computing tidal tensor from cached phi")
        _, e_t1, e_t3 = tidal_frame_at(coords, p.phi)
        p.release()

        rng = np.random.default_rng(config.CORRELATION_SEED + int(snapshot))
        perm_rows = np.stack([rng.permutation(n) for _ in range(N_PERM)])

        row = {'snapshot': snapshot, 'z': redshift, 'n': n}
        for combo, axis_vec, tidal_vec in (('major_t3', major, e_t3),
                                           ('minor_t1', minor, e_t1)):
            cos, lo, hi = alignment_with_band(axis_vec, tidal_vec, perm_rows)
            row[combo] = {'mean': float(np.mean(cos)),
                          'std': float(np.std(cos)), 'lo': lo, 'hi': hi,
                          'significant': not (lo <= np.mean(cos) <= hi)}
            plot_alignment_histogram(cos, combo, redshift, snapshot, lo, hi,
                                     output_dir)
            print(f"  {combo}: <|cos|> = {row[combo]['mean']:.3f} "
                  f"(null {lo:.3f}-{hi:.3f})"
                  f"{'  SIGNIFICANT' if row[combo]['significant'] else ''}")
        return row
    except FileNotFoundError:
        print(f"  skipping snapshot {snapshot} (no {method} fields)")
        return None
    except Exception as e:
        print(f"  Error analyzing snapshot {snapshot}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = make_parser("Void-tidal field alignment (histograms, "
                         "permutation null bands, redshift evolution).")
    for action in parser._actions:
        if action.dest == 'snap':
            action.default = None
            action.help = ("process only this snapshot "
                           "(default: all snapshots in the config table)")
    args = parser.parse_args()

    print("Void-tidal field alignment "
          f"({N_PERM} pairing permutations for the null band)")
    print(f"  sim: {args.sim}; method: {args.method}")
    output_root = Path(OUTPUT_DIR)
    output_root.mkdir(parents=True, exist_ok=True)

    items = config.snapshot_items()
    if args.snap is not None:
        want = f"{args.snap:03d}"
        items = [(s, z) for s, z in items if s == want]
        if not items:
            print(f"snapshot {want} not in config.SNAPSHOT_TO_REDSHIFT; nothing to do")
            return

    rows = []
    for snapshot, redshift in items:
        r = analyze_snapshot(snapshot, redshift, output_root,
                             sim=args.sim, method=args.method)
        if r is not None:
            rows.append(r)
    if not rows:
        print("No results; aborting")
        return

    for combo in COMBOS:
        plot_alignment_evolution(rows, combo)

    lines = [f"Void-tidal alignment (isotropic null <|cos|> = 0.5; "
             f"band = central 95% of {N_PERM} pairing shuffles)",
             f"{'z':>7} {'N':>6} "
             f"{'maj-t3':>7} {'null95%':>15} {'sig':>4} "
             f"{'min-t1':>7} {'null95%':>15} {'sig':>4}"]
    lines.append('-' * len(lines[1]))
    for r in rows:
        a, b = r['major_t3'], r['minor_t1']
        lines.append(
            f"{r['z']:7.2f} {r['n']:6d} "
            f"{a['mean']:7.3f} {a['lo']:7.3f}-{a['hi']:7.3f} "
            f"{'yes' if a['significant'] else 'no':>4} "
            f"{b['mean']:7.3f} {b['lo']:7.3f}-{b['hi']:7.3f} "
            f"{'yes' if b['significant'] else 'no':>4}")
    table = output_root / 'tidal_alignment_stats.txt'
    table.write_text('\n'.join(lines) + '\n')
    print(f"\nTable written: {table}")
    print(f"Output saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
