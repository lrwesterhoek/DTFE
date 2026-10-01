"""Score both codes against the exact crossed-wave solution and draw the comparison figures.
   score.py <cmpdir> <N> <npix> <G> <z0> <figdir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).parent))
from analytic import Wave1D, slice_truth  # noqa: E402

C, N, NPIX, G, Z0, FIG = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), Path(sys.argv[6])
FIG.mkdir(parents=True, exist_ok=True)
L, F = 100.0, 1.8
o, c = C / "out" / f"o{N}", C / "out" / f"c{N}"
VAMP = 100.0 * F / (2 * np.pi / L)            # velocity amplitude 100 A
S = {}                                        # the scores

# ------------------------------------------------------------------ slice (point values)
tr = slice_truth(L, F, NPIX, Z0)
shape2 = (NPIX, NPIX)
ours = {"rho": np.fromfile(f"{o}.ps_slice.pts_den").reshape(shape2),
        "streams": np.fromfile(f"{o}.ps_slice.pts_streams", dtype=np.int32).reshape(shape2),
        "v": np.fromfile(f"{o}.ps_slice.pts_vel").reshape(NPIX, NPIX, 3)}
ours_dtfe = np.fromfile(f"{o}.dtfe_slice.pts_den").reshape(shape2)
cos_rho = np.fromfile(f"{c}_dtfe.rho_slice.bin", dtype=np.float32).reshape(shape2).astype(float)
cos_n = np.fromfile(f"{c}_ps.streams_slice.bin", dtype=np.int32).reshape(shape2)
cos_vsum = np.fromfile(f"{c}_ps.vsum_slice.bin", dtype=np.float32).reshape(NPIX, NPIX, 3).astype(float)
cos_vavg = cos_vsum / np.maximum(cos_n, 1)[..., None]

single = tr["streams"] == 1
multi = ~single


def dens_scores(est, truth, mask=None):
    m = np.isfinite(est) & (est > 0) if mask is None else (mask & np.isfinite(est) & (est > 0))
    r = est[m] / truth[m]
    return {"median_abs_log10": float(np.median(np.abs(np.log10(r)))),
            "within_10pct": float(np.mean(np.abs(r - 1) < 0.1)),
            "within_2x": float(np.mean((r > 0.5) & (r < 2)))}


S["slice_density"] = {name: {"all": dens_scores(e, tr["rho"]), "single": dens_scores(e, tr["rho"], single),
                             "multi": dens_scores(e, tr["rho"], multi)}
                      for name, e in (("ours PS-DTFE", ours["rho"]), ("ours DTFE", ours_dtfe),
                                      ("CosmoDTFE DTFE", cos_rho))}
S["slice_streams"] = {"ours PS-DTFE": float(np.mean(ours["streams"] == tr["streams"])),
                      "CosmoDTFE PS": float(np.mean(cos_n == tr["streams"]))}


def vel_scores(vx, vy, mask):
    d = np.hypot(vx - tr["vx"], vy - tr["vy"])[mask] / VAMP
    return {"rms_over_amp": float(np.sqrt(np.mean(d ** 2))), "median_over_amp": float(np.median(d))}


S["slice_velocity"] = {name: {"single": vel_scores(vx, vy, single), "multi": vel_scores(vx, vy, multi)}
                       for name, vx, vy in (("ours PS-DTFE (mass-weighted mean)", ours["v"][..., 0], ours["v"][..., 1]),
                                            ("CosmoDTFE PS (sum / streams)", cos_vavg[..., 0], cos_vavg[..., 1]),
                                            ("CosmoDTFE PS (raw sum)", cos_vsum[..., 0], cos_vsum[..., 1]))}

# ------------------------------------------------------------------ grid (cell averages)
w = Wave1D(L, F)
n1, s1, v1 = w.cells(np.linspace(0, L, G + 1))
rho_c = n1[:, None, None] * n1[None, :, None] * n1[None, None, :]
str_c = s1[:, None, None] * s1[None, :, None] * s1[None, None, :]
vx_c = np.broadcast_to(v1[:, None, None], (G, G, G))
g3 = (G, G, G)


def rd(path, dtype=np.float32, comps=1):
    a = np.fromfile(path, dtype=dtype)
    return a.reshape(g3 + ((comps,) if comps > 1 else ()))


grids = {}
if Path(f"{o}.ps_grid_exact.den").exists():
    grids["ours PS-DTFE exact deposit"] = (rd(f"{o}.ps_grid_exact.den"), rd(f"{o}.ps_grid_exact.vel", comps=3)[..., 0],
                                          rd(f"{o}.ps_grid_exact.streams"))
grids["ours PS-DTFE sampled deposit (GPU)"] = (rd(f"{o}.ps_grid_gpu.a_den"), rd(f"{o}.ps_grid_gpu.a_vel", comps=3)[..., 0],
                                              rd(f"{o}.ps_grid_gpu.a_streams"))
if Path(f"{o}.ps_centres.pts_den").exists():
    grids["ours PS-DTFE point at centre"] = (np.fromfile(f"{o}.ps_centres.pts_den").reshape(g3),
                                            np.fromfile(f"{o}.ps_centres.pts_vel").reshape(g3 + (3,))[..., 0],
                                            np.fromfile(f"{o}.ps_centres.pts_streams", dtype=np.int32).reshape(g3).astype(float))
grids["ours DTFE point at centre"] = (rd(f"{o}.dtfe_grid.den"), None, None)
cn = np.fromfile(f"{c}_ps.streams_grid.bin", dtype=np.int32).reshape(g3).astype(float)
cv = np.fromfile(f"{c}_ps.vsum_grid.bin", dtype=np.float32).reshape(g3 + (3,))[..., 0].astype(float)
grids["CosmoDTFE DTFE point at centre"] = (np.fromfile(f"{c}_dtfe.rho_grid.bin", dtype=np.float32).reshape(g3).astype(float), None, None)
grids["CosmoDTFE PS point at centre"] = (None, cv / np.maximum(cn, 1), cn)

multi_c = str_c > 1 + 1e-9
S["grid"] = {}
for name, (rho, vx, st) in grids.items():
    d = {}
    if rho is not None:
        d["mean_rho"] = float(rho.mean())
        d["L1_rel"] = float(np.abs(rho - rho_c).sum() / rho_c.sum())
        d["L1_rel_multi"] = float(np.abs(rho - rho_c)[multi_c].sum() / rho_c[multi_c].sum())
        d["median_abs_log10"] = float(np.median(np.abs(np.log10(np.maximum(rho, 1e-30) / rho_c))))
    if vx is not None:
        d["v_rms_over_amp"] = float(np.sqrt(np.mean((vx - vx_c) ** 2)) / VAMP)
        d["v_rms_over_amp_multi"] = float(np.sqrt(np.mean((vx - vx_c)[multi_c] ** 2)) / VAMP)
    if st is not None:
        d["streams_rms"] = float(np.sqrt(np.mean((st - str_c) ** 2)))
    S["grid"][name] = d

(FIG / f"scores_{N}.json").write_text(json.dumps(S, indent=1))
print(json.dumps(S, indent=1))

# ------------------------------------------------------------------ figures
ext = [0, L, 0, L]


def show(ax, img, title, **kw):
    im = ax.imshow(img.T, origin="lower", extent=ext, interpolation="nearest", **kw)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    return im


# 1. density at points
vmin, vmax = 0.05, 50
fig, axs = plt.subplots(2, 4, figsize=(15, 7.6), constrained_layout=True)
panels = (("exact", tr["rho"]), ("ours: PS-DTFE", ours["rho"]), ("ours: standard DTFE", ours_dtfe),
          ("CosmoDTFE: standard DTFE", cos_rho))
for ax, (t, img) in zip(axs[0], panels):
    im = show(ax, np.clip(img, vmin, vmax), t, norm=LogNorm(vmin, vmax), cmap="magma")
fig.colorbar(im, ax=axs[0], shrink=0.8, label="ρ/ρ̄")
axs[1][0].axis("off")
axs[1][0].text(0.02, 0.5, f"crossed Zel'dovich waves\n{N}³ = {N**3/1e6:.1f}M particles\nslice z = {Z0:g} Mpc, {NPIX}² points\n\n"
               "bottom: log10(estimate / exact)", fontsize=10, va="center")
for ax, (t, img) in zip(axs[1][1:], panels[1:]):
    r = np.log10(np.maximum(img, 1e-6) / tr["rho"])
    im2 = show(ax, r, f"{t}\nmedian |log ratio| {np.median(np.abs(r)):.3f}", cmap="RdBu_r", norm=TwoSlopeNorm(0, -1, 1))
fig.colorbar(im2, ax=axs[1][1:], shrink=0.8, label="log10(estimate / exact)")
fig.savefig(FIG / f"density_slice_{N}.png", dpi=130)
plt.close(fig)

# 2. zoom on the caustic crossing
zoom = (slice(int(0.35 * NPIX), int(0.65 * NPIX)),) * 2
fig, axs = plt.subplots(1, 4, figsize=(15, 4.2), constrained_layout=True)
for ax, (t, img) in zip(axs, panels):
    sub = np.clip(img[zoom], vmin, vmax)
    im = ax.imshow(sub.T, origin="lower", extent=[35, 65, 35, 65], norm=LogNorm(vmin, vmax), cmap="magma",
                   interpolation="nearest")
    ax.set_title(t, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im, ax=axs, shrink=0.85, label="ρ/ρ̄")
fig.suptitle("zoom: where the folds cross (x, y = 35-65 Mpc)", fontsize=10)
fig.savefig(FIG / f"density_zoom_{N}.png", dpi=130)
plt.close(fig)

# 3. streams + velocity
fig, axs = plt.subplots(2, 3, figsize=(13, 8.4), constrained_layout=True)
kw = dict(cmap="viridis", vmin=1, vmax=9)
show(axs[0][0], tr["streams"], "exact stream count", **kw)
show(axs[0][1], ours["streams"], f"ours: {S['slice_streams']['ours PS-DTFE']:.4%} exact", **kw)
im = show(axs[0][2], cos_n, f"CosmoDTFE: {S['slice_streams']['CosmoDTFE PS']:.4%} exact", **kw)
fig.colorbar(im, ax=axs[0], shrink=0.8, label="streams")
vkw = dict(cmap="RdBu_r", vmin=-VAMP, vmax=VAMP)
show(axs[1][0], tr["vx"], "exact mass-weighted v_x", **vkw)
show(axs[1][1], ours["v"][..., 0], "ours: mass-weighted mean over streams", **vkw)
im = show(axs[1][2], cos_vavg[..., 0], "CosmoDTFE: sum over streams / stream count", **vkw)
fig.colorbar(im, ax=axs[1], shrink=0.8, label="v_x [km/s]")
fig.savefig(FIG / f"streams_velocity_{N}.png", dpi=130)
plt.close(fig)

# 4. cell averages: a plane of the 3-D grid at the slice height
k = int(Z0 / L * G)
fig, axs = plt.subplots(2, 4, figsize=(15, 7.6), constrained_layout=True)
gp = [("exact cell average", rho_c[:, :, k])]
for name in ("ours PS-DTFE exact deposit", "ours PS-DTFE sampled deposit (GPU)", "CosmoDTFE DTFE point at centre"):
    if name in grids:
        gp.append((name, grids[name][0][:, :, k]))
for ax, (t, img) in zip(axs[0], gp):
    im = show(ax, np.clip(img, vmin, vmax), t, norm=LogNorm(vmin, vmax), cmap="magma")
fig.colorbar(im, ax=axs[0], shrink=0.8, label="cell-mean ρ/ρ̄")
axs[1][0].axis("off")
axs[1][0].text(0.02, 0.5, f"{G}³ grid, plane k = {k}\n\nbottom: log10(estimate / exact cell mean)\n\n"
               "a point value at the centre is not\na cell average: it aliases thin walls", fontsize=10, va="center")
for ax, (t, img) in zip(axs[1][1:], gp[1:]):
    r = np.log10(np.maximum(img, 1e-6) / rho_c[:, :, k])
    im2 = show(ax, r, f"{t}\nL1 {S['grid'][t]['L1_rel']:.3f}, mean ρ {S['grid'][t]['mean_rho']:.4f}",
               cmap="RdBu_r", norm=TwoSlopeNorm(0, -1, 1))
fig.colorbar(im2, ax=axs[1][1:], shrink=0.8, label="log10(estimate / exact)")
fig.savefig(FIG / f"grid_cells_{N}.png", dpi=130)
plt.close(fig)
print("figures in", FIG)
