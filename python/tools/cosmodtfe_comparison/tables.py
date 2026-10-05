"""HTML tables for the report, straight from the score/timing JSON files.   tables.py <figdir> -> <figdir>/tables.html"""
import json
import sys
from pathlib import Path

FIG = Path(sys.argv[1])
out = []


def pct(x):
    return f"{100 * x:.1f}%"


def row(cells, head=False):
    tag = "th" if head else "td"
    return "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"


s = json.loads((FIG / "scores_192.json").read_text())
# accuracy at points (slice)
t = ['<table class="num"><thead>', row(["density at points (4.2M)", "median |log₁₀ error|", "within 10%", "within 10%, multi-stream"], True), "</thead><tbody>"]
for name, v in s["slice_density"].items():
    t.append(row([name, f"{v['all']['median_abs_log10']:.4f}", pct(v["all"]["within_10pct"]), pct(v["multi"]["within_10pct"])]))
t.append("</tbody></table>")
out.append(("acc_points", "".join(t)))

t = ['<table class="num"><thead>', row(["mean velocity at points", "rms error, single-stream", "rms error, multi-stream"], True), "</thead><tbody>"]
for name, v in s["slice_velocity"].items():
    t.append(row([name, pct(v["single"]["rms_over_amp"]), pct(v["multi"]["rms_over_amp"])]))
t.append("</tbody></table>")
t.append(f'<p class="note">errors as a fraction of the wave\'s velocity amplitude. Stream count at the 4.2M points: '
         f'ours {pct(s["slice_streams"]["ours PS-DTFE"])} exact, CosmoDTFE {pct(s["slice_streams"]["CosmoDTFE PS"])} exact.</p>')
out.append(("acc_velocity", "".join(t)))

t = ['<table class="num"><thead>', row(["256³ grid vs exact cell averages", "mean ρ/ρ̄ (exact 1)", "L1 error", "L1, multi-stream", "velocity rms"], True), "</thead><tbody>"]
for name, v in s["grid"].items():
    t.append(row([name, f"{v['mean_rho']:.4f}" if "mean_rho" in v else "–", pct(v["L1_rel"]) if "L1_rel" in v else "–",
                  pct(v["L1_rel_multi"]) if "L1_rel_multi" in v else "–",
                  pct(v["v_rms_over_amp"]) if "v_rms_over_amp" in v else "–"]))
t.append("</tbody></table>")
out.append(("acc_grid", "".join(t)))

perf = json.loads((FIG / "performance.json").read_text())
for r in perf:
    t = ['<table class="num"><thead>', row(["task", "ours", "CosmoDTFE", "ratio"], True), "</thead><tbody>"]
    for task, v in r["tasks"].items():
        f = lambda x: "not available" if x is None else f"{x['s']:.1f} s · {x['gb']:.1f} GB"
        ratio = ""
        if v["ours"] and v["CosmoDTFE"]:
            q = v["CosmoDTFE"]["s"] / v["ours"]["s"]
            ratio = f"ours {q:.1f}× faster" if q >= 1 else f"CosmoDTFE {1 / q:.1f}× faster"
        t.append(row([task, f(v["ours"]), f(v["CosmoDTFE"]), ratio]))
    t.append("</tbody></table>")
    out.append((f"perf_{r['data']}", "".join(t)))

if (FIG / "scores_tng.json").exists():
    g = json.loads((FIG / "scores_tng.json").read_text())
    sl = g["slice"]
    t = ['<table class="num"><thead>', row(["agreement on 4.2M slice points (TNG region)", "result"], True), "</thead><tbody>",
         row(["stream count, ours vs CosmoDTFE", f"{pct(sl['streams_agree'])} identical"]),
         row(["points with more than one stream", pct(sl["multistream_fraction_ours"])]),
         row(["mean velocity, single-stream points (rms difference)", f"{sl['velocity_single_stream_rms_diff_kms']:.1e} km/s"]),
         row(["mean velocity, multi-stream points (rms difference)", f"{sl['velocity_multi_stream_rms_diff_kms']:.1f} km/s"]),
         row(["standard DTFE density, ours vs CosmoDTFE (median |log₁₀ ratio|)", f"{sl['dtfe_median_abs_log10_ours_vs_cosmo']:.4f}"])]
    me = g.get("grid_mass_error", {})
    ex, sa = me.get("ours PS-DTFE exact cell averages"), me.get("ours PS-DTFE sampled cell averages")
    if ex is not None and sa is not None:
        t.append(row(["256³ cell averages: region mass vs the particles inside it (exact / sampled deposit)",
                      f"{100 * ex:+.3f}% / {100 * sa:+.3f}%"]))
    t.append("</tbody></table>")
    out.append(("tng_agree", "".join(t)))


# before/after the three fixes (out/before_fix holds the timing logs as first measured)
bf = Path(sys.argv[1]).parent / "out" / "before_fix"
if bf.exists():
    def last(path):
        d = {}
        if path.exists():
            for line in path.read_text().splitlines():
                x = json.loads(line)
                d[x["stage"]] = x
        return d
    STAGES = (("ps_centres", "phase-space point values, 16.8M points"),
              ("ps_slice", "phase-space slice, 4.2M points"),
              ("ps_grid_cpu", "sampled cell averages, CPU"),
              ("ps_grid_exact", "exact cell averages, GPU"),
              ("dtfe_avg_gpu", "standard-DTFE cell averages 256³, GPU"),
              ("dtfe_avg512_gpu", "standard-DTFE cell averages 512³, GPU"))
    NAMES = {"64": "0.26M", "128": "2.1M", "192": "7.1M", "tng": "TNG 11.8M"}
    # the standard-DTFE GPU stages were first measured with this morning's kernel into out/before_dtfe_<tag>
    # (the other "before" logs predate those stages)
    BEFORE_FILE = {"dtfe_avg_gpu": "before_dtfe_{tag}.timing.jsonl", "dtfe_avg512_gpu": "before_dtfe_{tag}.timing.jsonl"}

    def cell(x):
        if x is None:
            return "–"
        if x.get("rc", 1) != 0:
            return f"stopped after {int(x['seconds'] // 60)} min"
        return f"{x['seconds']:.1f} s · {x['footprint_gb']:.1f} GB"
    t = ['<table class="num"><thead>', row(["task", "particles", "before", "after"], True), "</thead><tbody>"]
    for stage, label in STAGES:
        for tag, nm in NAMES.items():
            if stage in BEFORE_FILE:
                b = last(Path(sys.argv[1]).parent / "out" / BEFORE_FILE[stage].format(tag=tag)).get(stage)
            else:
                b = last(bf / f"o{tag}.timing.jsonl").get(stage)
            a = last(Path(sys.argv[1]).parent / "out" / f"o{tag}.timing.jsonl").get(stage)
            t.append(row([label, nm, cell(b), cell(a)]))
    t.append("</tbody></table>")
    out.append(("fixes", "".join(t)))

(FIG / "tables.html").write_text("\n".join(f"<!-- {k} -->\n{v}" for k, v in out))
print("\n\n".join(f"== {k}\n{v}" for k, v in out))
