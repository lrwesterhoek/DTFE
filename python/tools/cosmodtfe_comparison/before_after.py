"""print before/after for the fixed stages   before_after.py <cmpdir>"""
import json, sys
from pathlib import Path
C = Path(sys.argv[1])
def last(p):
    d = {}
    for l in Path(p).read_text().splitlines():
        x = json.loads(l); d[x["stage"]] = x
    return d
for t in ("64", "128", "192", "tng"):
    b, a = last(C/"out"/"before_fix"/f"o{t}.timing.jsonl"), last(C/"out"/f"o{t}.timing.jsonl")
    print(f"== {t}")
    bd = C/"out"/f"before_dtfe_{t}.timing.jsonl"
    bd = last(bd) if bd.exists() else {}
    for s in ("ps_slice", "ps_centres", "ps_grid_cpu", "ps_grid_gpu", "ps_grid_exact", "dtfe_slice", "dtfe_grid",
              "dtfe_avg_cpu", "dtfe_avg_gpu", "dtfe_avg512_cpu", "dtfe_avg512_gpu"):
        f = lambda x: "--" if x is None else f"rc={x['rc']:3d} {x['seconds']:8.1f} s {x['footprint_gb']:6.2f} GB"
        print(f"  {s:16s} before {f(bd.get(s) if s.startswith('dtfe_avg') else b.get(s))}   after {f(a.get(s))}")
