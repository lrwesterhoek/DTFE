"""TNG100-3-Dark z=0 Lagrangian sub-cube: figures + the checks that need no analytic truth.
   tng.py <cmpdir> <figdir>
Region [40, 71)^3 Mpc (12 Mpc inside the patch image: no particle from outside the patch gets there),
slice z = 55.5 Mpc at 2048^2, grid 256^3 over the region."""
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

C, FIG = Path(sys.argv[1]), Path(sys.argv[2])
FIG.mkdir(parents=True, exist_ok=True)
LO, HI, Z0, NPIX, G = 40.0, 71.0, 55.5, 2048, 256
L_TNG, N_TNG = 110.71744906997343, 94196375
o, c = C / "out" / "otng", C / "out" / "ctng"
S = {}

# the exact mass in the region, from the particles, and our binaries' normalisations:
#   PS-DTFE (non-periodic): rho_bar = N*m / V(Lagrangian bounding box)   (DTFE.cpp)
#   DTFE: rho_bar = N*m / V(--box region)                                 (subpartition.h averageDensity)
with h5py.File(C / "data" / "tng_sub.h5", "r") as f:
    x = f["PartType1/Coordinates"][:]
    q = f["PartType1/InitialCoordinates"][:]
    L_hdr = float(f["Header"].attrs["BoxSize"])
n_sub = len(x)
inside = np.all((x >= LO) & (x < HI), axis=1)
nbar = N_TNG / L_TNG ** 3
S["true_mean_rho"] = float(inside.sum() / (nbar * (HI - LO) ** 3))
v_lag = float(np.prod(q.max(0).astype(float) - q.min(0).astype(float)))
SC_PS = (n_sub / v_lag) / nbar
SC_DT = (n_sub / (HI - LO) ** 3) / nbar
del x, q


def scale_from_log(name):
    return SC_DT if name.startswith("dtfe") else SC_PS


sh2 = (NPIX, NPIX)
sc_ps, sc_dt = scale_from_log("ps_slice"), scale_from_log("dtfe_slice")
ours_rho = np.fromfile(f"{o}.ps_slice.pts_den").reshape(sh2) * sc_ps
ours_st = np.fromfile(f"{o}.ps_slice.pts_streams", dtype=np.int32).reshape(sh2)
ours_v = np.fromfile(f"{o}.ps_slice.pts_vel").reshape(NPIX, NPIX, 3)
ours_dt = np.fromfile(f"{o}.dtfe_slice.pts_den").reshape(sh2) * sc_dt
cos_dt = np.fromfile(f"{c}_dtfe.rho_slice.bin", dtype=np.float32).reshape(sh2).astype(float)
cos_st = np.fromfile(f"{c}_ps.streams_slice.bin", dtype=np.int32).reshape(sh2)
cos_vs = np.fromfile(f"{c}_ps.vsum_slice.bin", dtype=np.float32).reshape(NPIX, NPIX, 3).astype(float)
cos_va = cos_vs / np.maximum(cos_st, 1)[..., None]
S["density_scale"] = {"ps": SC_PS, "dtfe": SC_DT, "check: median CosmoDTFE/ours standard DTFE": float(np.median(cos_dt / np.maximum(ours_dt, 1e-12)))}
S["slice"] = {
    "streams_agree": float(np.mean(ours_st == cos_st)),
    "multistream_fraction_ours": float(np.mean(ours_st > 1)),
    "dtfe_median_abs_log10_ours_vs_cosmo": float(np.median(np.abs(np.log10(np.maximum(ours_dt, 1e-9) / np.maximum(cos_dt, 1e-9))))),
    "mean_rho_slice": {"ours PS": float(ours_rho.mean()), "ours DTFE": float(ours_dt.mean()), "CosmoDTFE DTFE": float(cos_dt.mean())},
    "velocity_single_stream_rms_diff_kms": float(np.sqrt(np.mean(((ours_v - cos_va)[ours_st == 1]) ** 2))),
    "velocity_multi_stream_rms_diff_kms": float(np.sqrt(np.mean(((ours_v - cos_va)[ours_st > 1]) ** 2))),
}
g3 = (G, G, G)
grids = {}
if Path(f"{o}.ps_grid_exact.den").exists():
    grids["ours PS-DTFE exact cell averages"] = np.fromfile(f"{o}.ps_grid_exact.den", dtype=np.float32).reshape(g3) * scale_from_log("ps_grid_exact")
if Path(f"{o}.ps_grid_gpu.a_den").exists():
    grids["ours PS-DTFE sampled cell averages"] = np.fromfile(f"{o}.ps_grid_gpu.a_den", dtype=np.float32).reshape(g3) * scale_from_log("ps_grid_gpu")
if Path(f"{o}.ps_centres.pts_den").exists():
    grids["ours PS-DTFE point at centre"] = np.fromfile(f"{o}.ps_centres.pts_den").reshape(g3) * scale_from_log("ps_centres")
grids["ours DTFE point at centre"] = np.fromfile(f"{o}.dtfe_grid.den", dtype=np.float32).reshape(g3) * scale_from_log("dtfe_grid")
grids["CosmoDTFE DTFE point at centre"] = np.fromfile(f"{c}_dtfe.rho_grid.bin", dtype=np.float32).reshape(g3).astype(float)
S["grid_mean_rho"] = {k: float(v.mean()) for k, v in grids.items()}
S["grid_mass_error"] = {k: float(v.mean() / S["true_mean_rho"] - 1) for k, v in grids.items()}
(FIG / "scores_tng.json").write_text(json.dumps(S, indent=1))
print(json.dumps(S, indent=1))

ext = [LO, HI, LO, HI]
vmin, vmax = 0.03, 300


def show(ax, img, title, **kw):
    im = ax.imshow(img.T, origin="lower", extent=ext, interpolation="nearest", **kw)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x [Mpc]", fontsize=8); ax.tick_params(labelsize=7)
    return im


# 1. density
fig, axs = plt.subplots(1, 3, figsize=(16, 5.6), constrained_layout=True)
for ax, (t, img) in zip(axs, (("ours: PS-DTFE (phase-space density)", ours_rho), ("ours: standard DTFE", ours_dt),
                              ("CosmoDTFE: standard DTFE", cos_dt))):
    im = show(ax, np.clip(img, vmin, vmax), t, norm=LogNorm(vmin, vmax), cmap="magma")
fig.colorbar(im, ax=axs, shrink=0.85, label="ρ/ρ̄")
fig.suptitle(f"TNG100-3-Dark z=0, 11.8M-particle Lagrangian sub-cube, slice z = {Z0} Mpc ({NPIX}² points, "
             f"{(HI-LO)/NPIX*1000:.0f} kpc pixels)", fontsize=10)
fig.savefig(FIG / "tng_density.png", dpi=130)
plt.close(fig)

# 2. zoom on the densest structure
iy, ix = np.unravel_index(np.argmax(ours_rho.T), ours_rho.T.shape)
w = NPIX // 8
sl = (slice(max(ix - w, 0), ix + w), slice(max(iy - w, 0), iy + w))
zext = [LO + sl[0].start * (HI - LO) / NPIX, LO + min(sl[0].stop, NPIX) * (HI - LO) / NPIX,
        LO + sl[1].start * (HI - LO) / NPIX, LO + min(sl[1].stop, NPIX) * (HI - LO) / NPIX]
fig, axs = plt.subplots(1, 3, figsize=(16, 5.6), constrained_layout=True)
for ax, (t, img) in zip(axs, (("ours: PS-DTFE", ours_rho), ("ours: standard DTFE", ours_dt), ("CosmoDTFE: standard DTFE", cos_dt))):
    im = ax.imshow(np.clip(img[sl], vmin, 3e3).T, origin="lower", extent=zext, norm=LogNorm(vmin, 3e3),
                   cmap="magma", interpolation="nearest")
    ax.set_title(t, fontsize=9); ax.tick_params(labelsize=7)
fig.colorbar(im, ax=axs, shrink=0.85, label="ρ/ρ̄")
fig.suptitle("zoom on the densest halo of the slice", fontsize=10)
fig.savefig(FIG / "tng_density_zoom.png", dpi=130)
plt.close(fig)

# 3. streams + velocity (a vector field over |v|)
fig, axs = plt.subplots(2, 2, figsize=(12.5, 11.5), constrained_layout=True)
kw = dict(cmap="viridis", norm=LogNorm(1, max(int(ours_st.max()), 3)))
show(axs[0][0], np.maximum(ours_st, 1), f"ours: stream count (max {ours_st.max()})", **kw)
im = show(axs[0][1], np.maximum(cos_st, 1), f"CosmoDTFE: stream count ({S['slice']['streams_agree']:.3%} identical)", **kw)
fig.colorbar(im, ax=axs[0], shrink=0.8, label="streams")


def arrows(ax, v):
    b = NPIX // 32
    vb = v[:, :, :2].reshape(32, b, 32, b, 2).mean(axis=(1, 3))
    cx = LO + (np.arange(32) + 0.5) * (HI - LO) / 32
    X, Y = np.meshgrid(cx, cx, indexing="ij")
    ax.quiver(X, Y, vb[..., 0], vb[..., 1], color="white", edgecolor="black", linewidth=0.3, width=0.003,
              pivot="middle", scale=np.abs(vb).max() * 40)


vmag = np.linalg.norm(ours_v, axis=-1)
top = np.quantile(vmag, 0.995)
show(axs[1][0], vmag, "ours: mass-weighted mean velocity", cmap="inferno", vmin=0, vmax=top)
arrows(axs[1][0], ours_v)
im = show(axs[1][1], np.linalg.norm(cos_va, axis=-1), "CosmoDTFE: sum over streams / streams", cmap="inferno", vmin=0, vmax=top)
arrows(axs[1][1], cos_va)
fig.colorbar(im, ax=axs[1], shrink=0.8, label="|v| [km/s]")
fig.savefig(FIG / "tng_streams_velocity.png", dpi=130)
plt.close(fig)

# 4. a plane of the grids
k = int((Z0 - LO) / (HI - LO) * G)
names = [n for n in ("ours PS-DTFE sampled cell averages", "ours DTFE point at centre", "CosmoDTFE DTFE point at centre") if n in grids]
fig, axs = plt.subplots(1, len(names), figsize=(5.3 * len(names), 5.4), constrained_layout=True)
for ax, n in zip(np.atleast_1d(axs), names):
    im = show(ax, np.clip(grids[n][:, :, k], vmin, vmax), n,
              norm=LogNorm(vmin, vmax), cmap="magma")
fig.colorbar(im, ax=axs, shrink=0.85, label="ρ/ρ̄")
fig.suptitle(f"{G}³ grid over the region, plane k = {k}", fontsize=10)
fig.savefig(FIG / "tng_grid.png", dpi=130)
plt.close(fig)
print("figures in", FIG)
