"""Performance figures + table from the timing logs.   perf.py <cmpdir> <figdir>
Ours: whole-process wall time (read + tessellate + evaluate + write) and peak memory footprint.
CosmoDTFE: build + that task's evaluation (JIT warm-up and file loading excluded -- in its favour),
peak RSS of the process."""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C, FIG = Path(sys.argv[1]), Path(sys.argv[2])
FIG.mkdir(parents=True, exist_ok=True)


def load(path):
    out = {}
    p = Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            d = json.loads(line)
            out[d["stage"]] = d
    return out


rows = []
for tag, n_part in (("64", 64 ** 3), ("128", 128 ** 3), ("192", 192 ** 3), ("tng", 11_788_753)):
    cd, cp, o = (load(C / "out" / f"c{tag}_dtfe.timing.jsonl"), load(C / "out" / f"c{tag}_ps.timing.jsonl"),
                 load(C / "out" / f"o{tag}.timing.jsonl"))
    if not (cd or cp or o):
        continue

    def cos(d, task):
        if "build" not in d or task not in d:
            return None
        return {"s": d["build"]["seconds"] + d[task]["seconds"], "gb": d.get("total", d[task])["maxrss_gb"]}

    def ours(stage):
        d = o.get(stage)
        return None if d is None or d.get("rc", 1) != 0 else {"s": d["seconds"], "gb": d["footprint_gb"]}

    rows.append({"data": tag, "particles": n_part, "tasks": {
        "slice 2048², standard DTFE density": {"ours": ours("dtfe_slice"), "CosmoDTFE": cos(cd, "slice")},
        "slice 2048², phase-space (ours: ρ, v, σ, streams; Cosmo: streams, Σv)": {"ours": ours("ps_slice"), "CosmoDTFE": cos(cp, "slice")},
        "256³ grid, standard DTFE at cell centres": {"ours": ours("dtfe_grid"), "CosmoDTFE": cos(cd, "grid")},
        "256³ grid, phase-space at cell centres": {"ours": ours("ps_centres"), "CosmoDTFE": cos(cp, "grid")},
        "256³ grid, phase-space CELL AVERAGES (GPU)": {"ours": ours("ps_grid_gpu"), "CosmoDTFE": None},
        "256³ grid, phase-space EXACT cell averages (GPU)": {"ours": ours("ps_grid_exact"), "CosmoDTFE": None},
        "256³ grid, phase-space cell averages (CPU)": {"ours": ours("ps_grid_cpu"), "CosmoDTFE": None},
    }})
(FIG / "performance.json").write_text(json.dumps(rows, indent=1))

# table to stdout
for r in rows:
    print(f"\n== {r['data']}  ({r['particles']/1e6:.1f}M particles)")
    for task, v in r["tasks"].items():
        f = lambda x: "      --       " if x is None else f"{x['s']:7.1f} s {x['gb']:5.1f} GB"
        speed = (f"  x{v['CosmoDTFE']['s'] / v['ours']['s']:.1f}" if v["ours"] and v["CosmoDTFE"] else "")
        print(f"  {task:62s} ours {f(v['ours'])}   CosmoDTFE {f(v['CosmoDTFE'])}{speed}")

# figure: time and memory for the largest synthetic set (top), scaling of the shared tasks (bottom)
SHORT = {"slice 2048², standard DTFE density": "slice 2048², standard DTFE",
         "slice 2048², phase-space (ours: ρ, v, σ, streams; Cosmo: streams, Σv)": "slice 2048², phase-space",
         "256³ grid, standard DTFE at cell centres": "256³, standard DTFE, point samples",
         "256³ grid, phase-space at cell centres": "256³, phase-space, point samples",
         "256³ grid, phase-space CELL AVERAGES (GPU)": "256³, phase-space cell averages, GPU",
         "256³ grid, phase-space EXACT cell averages (GPU)": "256³, EXACT cell averages, GPU",
         "256³ grid, phase-space cell averages (CPU)": "256³, phase-space cell averages, CPU"}
big = next((r for r in rows if r["data"] == "192"), rows[-1])
tasks = list(big["tasks"])
fig = plt.figure(figsize=(15, 10), constrained_layout=True)
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])
y = np.arange(len(tasks))
for col, (key, label) in enumerate((("s", "wall time [s], log scale"), ("gb", "peak memory [GB]"))):
    ax = fig.add_subplot(gs[0, col])
    ov = [big["tasks"][t]["ours"][key] if big["tasks"][t]["ours"] else np.nan for t in tasks]
    cv = [big["tasks"][t]["CosmoDTFE"][key] if big["tasks"][t]["CosmoDTFE"] else np.nan for t in tasks]
    ax.barh(y - 0.2, ov, 0.4, label="ours (C++/CGAL; GPU where marked)", color="#1f6fb4")
    ax.barh(y + 0.2, cv, 0.4, label="CosmoDTFE (Julia, TetGen + BVH, 10 threads)", color="#d4700f")
    for yi, c_ in zip(y, cv):
        if np.isnan(c_):
            ax.annotate("not available", (0, yi + 0.2), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8, color="#d4700f")
    ax.set_yticks(y, [SHORT.get(t, t) for t in tasks] if col == 0 else [""] * len(tasks), fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(label)
    if key == "s":
        ax.set_xscale("log")
    ax.grid(axis="x", alpha=0.3)
    if col == 0:
        ax.legend(fontsize=8, loc="lower right")
        ax.set_title(f"{big['particles']/1e6:.1f}M particles (crossed waves {big['data']}³): whole task, "
                     "reading and tessellating included", fontsize=10, loc="left")
ax = fig.add_subplot(gs[1, :])
for task, mk in (("slice 2048², standard DTFE density", "o"), ("256³ grid, standard DTFE at cell centres", "s"),
                 ("slice 2048², phase-space (ours: ρ, v, σ, streams; Cosmo: streams, Σv)", "D"),
                 ("256³ grid, phase-space at cell centres", "^")):
    for who, col in (("ours", "#1f6fb4"), ("CosmoDTFE", "#d4700f")):
        pts = [(r["particles"], r["tasks"][task][who]["s"]) for r in rows if r["data"] != "tng" and r["tasks"][task][who]]
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker=mk, color=col, ls="-" if who == "ours" else "--", label=f"{who}: {SHORT[task]}")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("particles"); ax.set_ylabel("wall time [s]")
ax.set_title("scaling with the number of particles", fontsize=10, loc="left")
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3, which="both")
fig.savefig(FIG / "performance.png", dpi=130)
print("\nwrote", FIG / "performance.png")
