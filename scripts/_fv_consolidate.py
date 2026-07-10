"""Consolidate FV-retrain cell-average audits vs the point model + np1/np2.

Reads analysis_medium_improv/audits/fv_l2_seed*_cellavg_f64_proj800/{field_metrics,spectral_shells}.csv,
aggregates the FV operator across seeds (mean+/-std), and compares to the frozen point model
(v12_point) and TempestRemap np1/np2, split by train-pairs vs held-out FAMILIES (ICO / MPAS /
HEALPix).  Usage: python fv_consolidate.py <seed_dir> [<seed_dir> ...]
"""
import sys, glob
from pathlib import Path
import numpy as np
import pandas as pd

def family_of(pair):
    t = pair.split("_to_")[-1]
    for fam in ["HP", "ICOD", "ICO", "MPAS", "CS", "RLL"]:
        if t.startswith(fam):
            return "HEALPix" if fam == "HP" else fam
    return t

def load(dirs, name):
    frames = []
    for d in dirs:
        p = Path(d) / name
        if p.exists():
            df = pd.read_csv(p)
            df["seed_dir"] = Path(d).name
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def is_fv(op): return op.startswith("fv_l2_seed")

def summarize(df, val_col, group_extra=None):
    """Return per-(split[,extra]) means for np1/np2/v12_point and fv(mean+/-std over seeds)."""
    df = df.copy()
    df["fam"] = df["pair"].map(family_of)
    rows = []
    def agg_group(sub, label):
        out = {"group": label, "n_pairs": sub["pair"].nunique()}
        # point + tempest: identical across seeds -> average over pairs/fields from seed 0 rows only
        for op in ["np1", "np2", "v12_point"]:
            m = sub[sub.operator == op][val_col]
            out[op] = float(m.mean()) if len(m) else np.nan
        # fv: per-seed = per fv_l2_seedN operator; mean+/-std ACROSS the seeds
        fv = sub[sub.operator.map(is_fv)]
        if len(fv):
            per_seed = fv.groupby("operator")[val_col].mean()
            out["fv_mean"] = float(per_seed.mean())
            out["fv_std"] = float(per_seed.std(ddof=0)) if len(per_seed) > 1 else 0.0
            out["n_seeds"] = int(per_seed.shape[0])
        else:
            out["fv_mean"] = out["fv_std"] = np.nan; out["n_seeds"] = 0
        out["fv/point"] = out["fv_mean"] / out["v12_point"] if out.get("v12_point") else np.nan
        out["fv/np2"] = out["fv_mean"] / out["np2"] if out.get("np2") else np.nan
        rows.append(out)
    # train aggregate
    agg_group(df[df.split == "train"], "TRAIN (all)")
    # holdout by family
    ho = df[df.split == "holdout"]
    for fam in sorted(ho["fam"].unique()):
        agg_group(ho[ho.fam == fam], f"HOLDOUT:{fam}")
    agg_group(ho, "HOLDOUT (all)")
    return pd.DataFrame(rows)

def main():
    dirs = sys.argv[1:]
    if not dirs:
        dirs = sorted(glob.glob("analysis_medium_improv/audits/fv_l2_seed*_cellavg_f64_proj800"))
    print("seed dirs:", [Path(d).name for d in dirs])
    fld = load(dirs, "field_metrics.csv")
    sh = load(dirs, "spectral_shells.csv")

    def show(title, table):
        print(f"\n===== {title} =====")
        cols = ["group", "n_pairs", "n_seeds", "np1", "np2", "v12_point", "fv_mean", "fv_std", "fv/point", "fv/np2"]
        with pd.option_context("display.float_format", lambda x: f"{x:.4e}" if (abs(x) < 1 and x != 0) else f"{x:.3f}"):
            print(table[cols].to_string(index=False))

    if not fld.empty:
        an = fld[fld.category == "analytic"]
        show("ANALYTIC (cell-average area_rel_l2)", summarize(an, "area_rel_l2"))
        rl = fld[fld.category == "real"]
        if not rl.empty:
            show("REAL FIELDS (cell-native)", summarize(rl, "area_rel_l2"))
    if not sh.empty:
        show("SPECTRAL SHELLS (mean over shells)", summarize(sh, "mean_area_rel_l2"))
        # spectral profile by shell (train + holdout combined, fv vs point)
        sh2 = sh.copy()
        prof = (sh2.assign(is_fv=sh2.operator.map(is_fv))
                   .groupby(["shell_label", "split"])
                   .apply(lambda g: pd.Series({
                       "np2": g[g.operator=="np2"]["mean_area_rel_l2"].mean(),
                       "point": g[g.operator=="v12_point"]["mean_area_rel_l2"].mean(),
                       "fv": g[g.is_fv]["mean_area_rel_l2"].mean(),
                   }), include_groups=False).reset_index())
        print("\n===== SPECTRAL PROFILE by shell (cell-average) =====")
        with pd.option_context("display.float_format", lambda x: f"{x:.3e}"):
            print(prof.to_string(index=False))

    # G2 verdict on train pairs
    if not fld.empty:
        an_t = summarize(fld[fld.category=="analytic"], "area_rel_l2")
        row = an_t[an_t.group=="TRAIN (all)"].iloc[0]
        print(f"\nG2 (train analytic, cellavg): fv_mean={row.fv_mean:.4e} vs point={row.v12_point:.4e} "
              f"-> fv/point={row['fv/point']:.3f}  {'PASS (FV better)' if row['fv/point']<1 else 'FAIL'}")

if __name__ == "__main__":
    main()
