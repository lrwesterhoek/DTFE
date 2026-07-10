"""Track (sub)halos along their SubLink main-progenitor branch: SIZE, SHAPE, ORIENTATION,
MERGERS and the DTFE / PS-DTFE FIELD ENVIRONMENT.

Per tracked subhalo, walking back in time from --snap:
  SIZE        : total subhalo mass, half-mass radius, R_200c of the host (centrals)
                -- straight from the extended tree nodes (no group catalogs needed).
  SHAPE       : axis ratios b/a, c/a and triaxiality T from the DM mass tensor of the
                subhalo's member particles. Needs BOTH the group catalog (groups_NNN, for
                particle membership offsets) and the raw snapshot chunks (snapdir_NNN) at
                that snapshot; snapshots missing either are skipped for shape with a note.
  ORIENTATION : drift of the major axis, |cos angle| w.r.t. the z=0 (--snap) major axis,
                and the same for the SubhaloSpin vector (catalog-level fallback that works
                even without particle data).
  MERGERS     : secondary progenitors joining the branch (SubLink NextProgenitor chains);
                events above --min-ratio are ticked on the mass panel.
  ENVIRONMENT : the density contrast rho/rho_bar and (where available) the T-web/V-web
                class at the halo position, sampled from the DTFE or PS-DTFE grids of
                every snapshot on disk (--method ps|dtfe|auto via dtfelib.FieldSet, so it
                works identically for both estimators). Grids are loaded once per snapshot
                for all tracked halos.

Selection: --subhalo ID [ID ...] (SubfindIDs at --snap), or --top N most massive centrals
(default 5) chosen from the trees themselves.

Examples:
  python3 plot/plot_halo_tracking.py --sim TNG50-3-Dark                 # top 5 centrals, z=0
  python3 plot/plot_halo_tracking.py --sim TNG50-3-Dark --subhalo 0 17 --method dtfe
  python3 plot/plot_halo_tracking.py --sim TNG300-3-Dark --top 10 --method ps
Outputs: figures/halo_tracking/<sim>/subhalo<ID>_evolution.png + branch data as .npz.
"""

import _bootstrap  # noqa: F401  (puts python/ on sys.path)
import config
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from dtfelib import make_parser
from dtfelib.trees import TreeSet
from dtfelib import groupcat as gc
from dtfelib import environment as env

FIGURE_ROOT = Path(config.LOCAL_FIGURES_ROOT) / "halo_tracking"

BRANCH_FIELDS = ["SubhaloMass", "SubhaloPos", "SubhaloHalfmassRad", "SubhaloSpin",
                 "Group_R_Crit200", "SubhaloID", "FirstSubhaloInFOFGroupID", "SubhaloGrNr"]

ENV_FIELDS = ("density", "tweb", "vweb")
WEB_NAMES = {0: "void", 1: "wall", 2: "filament", 3: "node"}


def unit_cos(v, w):
    n = np.linalg.norm(v) * np.linalg.norm(w)
    return np.abs(float(np.dot(v, w) / n)) if n > 0 else np.nan


def plot_finite(ax, x, y, *fmt, **kw):
    """Line+marker plot connected through FINITE samples only.

    Sparse quantities (shape/orientation/environment exist only at snapshots with data on
    disk) are NaN elsewhere; a plain plot() breaks the line at every NaN, leaving isolated
    unreadable markers. Connecting the finite points keeps the time evolution legible."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    return ax.plot(x[ok], y[ok], *fmt, **kw)


def track_one(ts: TreeSet, sim: str, snap0: int, subfind0: int, hubble: float,
              box_ckpc_h: float, ztab: dict) -> dict:
    br = ts.main_branch(snap0, subfind0, BRANCH_FIELDS)
    n = br["SnapNum"].size
    print(f"  subhalo {subfind0}: branch of {n} snapshots "
          f"({int(br['SnapNum'][-1])} .. {int(br['SnapNum'][0])})")

    z = np.array([ztab.get(int(s), np.nan) for s in br["SnapNum"]])
    central = br["FirstSubhaloInFOFGroupID"] == br["SubhaloID"]

    out = {
        "snap": br["SnapNum"].astype(int),
        "redshift": z,
        "subfind_id": br["SubfindID"].astype(int),
        "central": central,
        # SIZE (h-free physical-ish units: mass Msun, radii comoving kpc)
        "mass_msun": br["SubhaloMass"] * 1e10 / hubble,
        "r_half_ckpc": br["SubhaloHalfmassRad"] / hubble,
        "r200c_ckpc": np.where(central, br["Group_R_Crit200"], np.nan) / hubble,
    }

    # SHAPE + ORIENTATION from particles, where groupcat + snapshot chunks exist
    b_over_a = np.full(n, np.nan)
    c_over_a = np.full(n, np.nan)
    triax = np.full(n, np.nan)
    axes = np.full((n, 3, 3), np.nan)
    shape_snaps, missing = [], []
    for i in range(n):
        s, sid = int(br["SnapNum"][i]), int(br["SubfindID"][i])
        try:
            coords = gc.read_subhalo_coordinates(sim, s, sid)
            if coords.shape[0] < 50:      # too few particles for a meaningful tensor
                continue
            sh = gc.shape_from_coords(coords, br["SubhaloPos"][i], box_ckpc_h)
            b_over_a[i], c_over_a[i], triax[i] = sh["b_over_a"], sh["c_over_a"], sh["triaxiality"]
            axes[i] = sh["axes"]
            shape_snaps.append(s)
        except (FileNotFoundError, IOError):
            missing.append(s)
    if shape_snaps:
        print(f"    particle shapes at {len(shape_snaps)} snapshots: {sorted(shape_snaps)}")
    if missing:
        u = sorted(set(missing))
        print(f"    no particle data (groupcat and/or snapshot chunks) at {len(u)} snapshots -> "
              f"shape skipped there (e.g. {u[:6]}...)")

    out.update({"b_over_a": b_over_a, "c_over_a": c_over_a, "triaxiality": triax})

    # orientation drift vs the reference (latest) snapshot
    ref_axis = axes[0, 0] if np.isfinite(axes[0]).all() else None
    out["major_axis_cos"] = np.array([
        unit_cos(axes[i, 0], ref_axis) if ref_axis is not None and np.isfinite(axes[i]).all()
        else np.nan for i in range(n)])
    spin = br["SubhaloSpin"]
    out["spin_cos"] = np.array([unit_cos(spin[i], spin[0]) for i in range(n)])
    out["pos_ckpc_h"] = np.asarray(br["SubhaloPos"], dtype=np.float64)
    return out


def add_environment(tracks: dict, sim: str, method: str):
    """Sample the field environment for every tracked halo, one grid load per snapshot.

    tracks: {subfind0: track dict}. Adds env_density / env_tweb / env_vweb arrays in place
    (NaN where no field grids exist on disk) and returns the set of sampled snapshots.
    """
    for t in tracks.values():
        for name in ENV_FIELDS:
            t[f"env_{name}"] = np.full(t["snap"].size, np.nan)

    all_snaps = sorted({int(s) for t in tracks.values() for s in t["snap"]}, reverse=True)
    sampled, skipped = [], []
    for s in all_snaps:
        where = [(sid, i) for sid, t in tracks.items() for i in np.nonzero(t["snap"] == s)[0]]
        pos = np.array([tracks[sid]["pos_ckpc_h"][i] for sid, i in where])
        try:
            smp = env.sample_fields_at(sim, s, pos, ENV_FIELDS, method=method)
        except (FileNotFoundError, ValueError):
            skipped.append(s)
            continue
        sampled.append(s)
        for k, (sid, i) in enumerate(where):
            for name in ENV_FIELDS:
                tracks[sid][f"env_{name}"][i] = smp[name][k]
    if sampled:
        print(f"  environment sampled at {len(sampled)} snapshots ({min(sampled)}..{max(sampled)})")
    if skipped:
        print(f"  no field grids at {len(skipped)} snapshots -> environment NaN there "
              f"(e.g. {skipped[:6]}...)")
    return sampled


def plot_one(t: dict, sim: str, subfind0: int, snap0: int, outdir: Path):
    x = t["redshift"]
    xlabel = "redshift z"
    if not np.isfinite(x).any():
        x, xlabel = t["snap"], "snapshot"

    have_env = np.isfinite(t.get("env_density", np.array(np.nan))).any()
    npanel = 4 if have_env else 3
    fig, axes_ = plt.subplots(npanel, 1, figsize=(8, 4 * npanel), sharex=True)
    ax1, ax2, ax3 = axes_[0], axes_[1], axes_[2]

    ax1.plot(x, t["mass_msun"], "-", color="tab:blue", lw=1.6, label=r"$M_{\rm sub}$ [$M_\odot$]")
    ax1.set_yscale("log")
    # merger events: tick marks along the top edge (size ~ mass ratio) instead of full-height
    # vertical lines, which drowned the mass curve for merger-rich halos
    mz = np.asarray(t.get("merger_x", np.empty(0)), dtype=float)
    mr = np.asarray(t.get("merger_ratio", np.empty(0)), dtype=float)
    ok_m = np.isfinite(mz)
    if ok_m.any():
        ax1.scatter(mz[ok_m], np.full(int(ok_m.sum()), 0.97), transform=ax1.get_xaxis_transform(),
                    marker="v", s=18 + 60 * np.clip(mr[ok_m], 0, 1), color="crimson",
                    alpha=0.7, clip_on=False,
                    label=f"mergers $\\geq$1:10 (n={int(ok_m.sum())}, size~ratio)")
    ax1b = ax1.twinx()
    plot_finite(ax1b, x, t["r_half_ckpc"], "--", color="tab:orange", lw=1.2, label=r"$r_{1/2}$ [ckpc]")
    plot_finite(ax1b, x, t["r200c_ckpc"], "--", color="tab:green", lw=1.2, label=r"$R_{200c}$ [ckpc]")
    ax1b.set_yscale("log")
    ax1.set_ylabel("mass"); ax1b.set_ylabel("radius [ckpc]")
    ax1.set_title(f"{sim}  subhalo {subfind0} (snap {snap0}) — size / shape / orientation"
                  + (" / environment" if have_env else ""))
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=9, loc="lower right")

    plot_finite(ax2, x, t["b_over_a"], "o-", ms=4, label="b/a")
    plot_finite(ax2, x, t["c_over_a"], "s-", ms=4, label="c/a")
    plot_finite(ax2, x, t["triaxiality"], "^--", ms=4, alpha=0.5, label="triaxiality T")
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("axis ratios"); ax2.legend(fontsize=9)
    if not np.isfinite(t["b_over_a"]).any():
        ax2.text(0.5, 0.5, "no particle data on disk for shape\n"
                 "(download groupcats -c and snapshot chunks)",
                 transform=ax2.transAxes, ha="center", va="center", fontsize=10, color="crimson")

    plot_finite(ax3, x, t["major_axis_cos"], "o-", ms=4, label=r"major axis $|\cos\theta|$ vs latest")
    plot_finite(ax3, x, t["spin_cos"], "-", color="tab:orange", lw=0.8, alpha=0.55,
                label=r"spin $|\cos\theta|$ vs latest (per-snapshot, noisy)")
    ax3.set_ylim(0, 1.05); ax3.set_ylabel("orientation drift")
    ax3.legend(fontsize=9)

    if have_env:
        ax4 = axes_[3]
        d = t["env_density"]
        plot_finite(ax4, x, d, "o-", ms=4, color="tab:purple",
                    label=r"$\rho/\bar\rho$ at halo position (%s)" % t.get("env_method", "?"))
        ax4.set_yscale("log")
        ax4.axhline(1.0, color="gray", lw=0.8, ls=":")
        ax4.set_ylabel(r"$\rho/\bar\rho$")
        # web class markers along the top edge (axis-fraction y), where available
        web_colors = {0: "tab:blue", 1: "tab:green", 2: "tab:orange", 3: "tab:red"}
        for name, y in (("tweb", 0.96), ("vweb", 0.88)):
            lab = t.get(f"env_{name}")
            if lab is None or not np.isfinite(lab).any():
                continue
            ok = np.isfinite(lab)
            ax4.scatter(np.asarray(x)[ok], np.full(int(ok.sum()), y),
                        transform=ax4.get_xaxis_transform(),
                        c=[web_colors[int(v)] for v in lab[ok]], marker="s", s=18,
                        label=f"{name} class (blue=void green=wall orange=fil red=node)")
        ax4.legend(fontsize=8, loc="lower right")
        ax4.set_xlabel(xlabel)
    else:
        ax3.set_xlabel(xlabel)

    for ax in axes_:
        ax.grid(alpha=0.3)
    if xlabel.startswith("redshift"):
        axes_[-1].invert_xaxis()

    outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outdir / f"subhalo{subfind0}_evolution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    np.savez_compressed(outdir / f"subhalo{subfind0}_branch.npz",
                        **{k: v for k, v in t.items() if isinstance(v, np.ndarray)})


def main():
    def extra(p):
        p.add_argument("--subhalo", type=int, nargs="+",
                       help="SubfindID(s) at --snap to track (default: --top most massive centrals)")
        p.add_argument("--top", type=int, default=5,
                       help="if --subhalo not given: track the N most massive centrals (default 5)")
        p.add_argument("--min-ratio", type=float, default=0.1,
                       help="minimum secondary/main mass ratio for a merger tick (default 0.1)")
        p.add_argument("--no-environment", action="store_true",
                       help="skip sampling the DTFE/PS-DTFE field environment")
    parser = make_parser("Track subhalo size/shape/orientation/mergers/environment along "
                         "SubLink main-progenitor branches.")
    extra(parser)
    args = parser.parse_args()

    ts = TreeSet(args.sim)
    print(f"{args.sim}: {len(ts.paths)} SubLink chunks in {ts.dir}")

    ztab = gc.redshift_table(args.sim)
    hubble = box = None
    try:
        hd = gc.header(args.sim, args.snap)
        hubble, box = hd["hubble"], hd["box_ckpc_h"]
    except FileNotFoundError:
        snapdirs = sorted((gc.DATA_ROOT / args.sim).glob("snapdir_*/snap_*.hdf5")) or \
                   sorted((gc.DATA_ROOT / args.sim).glob("snapdir_*/combined_*.hdf5"))
        if not snapdirs:
            raise SystemExit(f"no header source (groupcat or snapshot) found for {args.sim}; "
                             "download groupcats first: ./download_snapshots.sh -c -s " + args.sim)
        import h5py
        with h5py.File(snapdirs[0], "r") as f:
            hubble = float(f["Header"].attrs["HubbleParam"])
            box = float(f["Header"].attrs["BoxSize"])
            if "combined" in snapdirs[0].name:
                box *= hubble    # combined files are h-free ckpc; convert back to ckpc/h

    subs = args.subhalo or ts.subhalos_at(args.snap, n=args.top).tolist()
    print(f"tracking {len(subs)} subhalo(s) at snap {args.snap}: {subs}")

    tracks = {}
    for sid in subs:
        try:
            t = track_one(ts, args.sim, args.snap, int(sid), hubble, box, ztab)
        except KeyError as e:
            print(f"  ! {e}")
            continue
        # merger events onto the branch, mapped to the plot x-axis (redshift or snapshot)
        ev = ts.mergers_along_branch(args.snap, int(sid), min_ratio=args.min_ratio)
        if ev["snap"].size:
            use_z = np.isfinite(t["redshift"]).any()
            t["merger_x"] = np.array([ztab.get(int(s), np.nan) for s in ev["snap"]]) \
                if use_z else ev["snap"].astype(float)
            t["merger_ratio"] = ev["ratio"]
            print(f"    {ev['snap'].size} merger(s) >= 1:{int(round(1/args.min_ratio))} "
                  f"at snaps {sorted(set(int(s) for s in ev['snap']))}")
        else:
            t["merger_x"], t["merger_ratio"] = np.empty(0), np.empty(0)
        tracks[int(sid)] = t

    if tracks and not args.no_environment:
        add_environment(tracks, args.sim, args.method)
        for t in tracks.values():
            t["env_method"] = args.method

    outdir = FIGURE_ROOT / args.sim
    for sid, t in tracks.items():
        plot_one(t, args.sim, sid, args.snap, outdir)
        print(f"    -> {outdir / f'subhalo{sid}_evolution.png'}")
    print("Done.")


if __name__ == "__main__":
    main()
