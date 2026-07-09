"""Compare operators per spectral band across audit dirs (cell-average metric).
Usage: python _fv_compare.py <dir1> <dir2> ...
Tabulates mean spectral-shell error per (split, band) for each operator + ratio to np2.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

dirs = sys.argv[1:]
frames = []
for d in dirs:
    p = Path(d) / "spectral_shells.csv"
    if p.exists():
        frames.append(pd.read_csv(p))
df = pd.concat(frames, ignore_index=True)

# preferred column order if present
order = ["np1", "np2", "v12_point", "fv_l2_relax05", "fv_l2_relax1", "fv_l3"]
ops = [o for o in order if o in df.operator.unique()] + \
      [o for o in df.operator.unique() if o not in order]

g = (df.groupby(["split", "shell_label", "operator"])["mean_area_rel_l2"]
       .mean().reset_index())
piv = g.pivot_table(index=["split", "shell_label"], columns="operator", values="mean_area_rel_l2")
piv = piv.reindex(columns=ops)

for split in ["train", "holdout", "audit"]:
    if split not in piv.index.get_level_values(0):
        continue
    sub = piv.xs(split, level=0).sort_index()
    print(f"\n===== split={split} : mean cell-average shell error =====")
    with pd.option_context("display.float_format", lambda x: f"{x:.3e}", "display.width", 200):
        print(sub.to_string())
    if "np2" in sub.columns:
        print(f"  -- ratio to np2 --")
        rat = sub.div(sub["np2"], axis=0)
        with pd.option_context("display.float_format", lambda x: f"{x:.2f}", "display.width", 200):
            print(rat.to_string())

# real-field + analytic means too
for d in dirs:
    fp = Path(d) / "field_metrics.csv"
    if not fp.exists():
        continue
fm = pd.concat([pd.read_csv(Path(d) / "field_metrics.csv") for d in dirs if (Path(d)/"field_metrics.csv").exists()], ignore_index=True)
for cat in ["analytic", "real"]:
    sub = fm[fm.category == cat]
    if sub.empty:
        continue
    t = (sub.groupby(["split", "operator"])["area_rel_l2"].mean().reset_index()
            .pivot_table(index="split", columns="operator", values="area_rel_l2").reindex(columns=ops))
    print(f"\n===== {cat} field mean (cell-average) =====")
    with pd.option_context("display.float_format", lambda x: f"{x:.3e}", "display.width", 200):
        print(t.to_string())
