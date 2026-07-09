"""Cross-condition consolidator for the mesh-family generalization study.

Reads the combined cell-average audit (packs named fv_gen_<COND>_e<E>_s<SEED>) and, PER PACK,
labels each audited pair as in-sample or ZERO-SHOT using the condition's training families
(not the audit's global split column).  A pair tests zero-shot family F iff F is an endpoint
AND F is not in the pack's training families.  Emits: zero-shot error + fv/np2 vs #training
families, per held-out family, mean+/-std over seeds.

Usage: python _gen_consolidate.py <combined_audit_dir>
"""
import sys, re
from pathlib import Path
import numpy as np, pandas as pd

# training families per condition (must match jobs_gen_train.pbs)
COND_FAMS = {
    "D2": {"CS", "ICOD"},
    "D3": {"CS", "ICOD", "RLL"},
    "D4": {"CS", "ICOD", "RLL", "ICO"},
    "D5": {"CS", "ICOD", "RLL", "ICO", "MPAS"},
    "LOFO_ICO": {"CS", "ICOD", "RLL", "MPAS"},
    "LOFO_MPAS": {"CS", "ICOD", "RLL", "ICO"},
}

def family_of(mesh):
    for fam in ["HP", "ICOD", "ICO", "MPAS", "CSRR", "CS", "RLL"]:  # order: HP/ICOD/ICO before CS
        if mesh.startswith(fam):
            return "HEALPix" if fam == "HP" else fam
    return mesh

def pair_endpoints(pair):
    a, b = pair.split("_to_")
    return family_of(a), family_of(b)

def parse_cond(label):
    m = re.match(r"fv_gen_(D2|D3|D4|D5|LOFO_ICO|LOFO_MPAS)_e(\d+)_s(\d+)", label)
    if not m:
        return None
    return {"cond": m.group(1), "epochs": int(m.group(2)), "seed": int(m.group(3))}

def load(d, name):
    p = Path(d) / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

def main():
    d = sys.argv[1]
    fld = load(d, "field_metrics.csv")
    sh = load(d, "spectral_shells.csv")

    # np2 reference per pair (from any operator row labelled np2), for fv/np2
    def np2_by_pair(df, val):
        r = df[df.operator == "np2"]
        return r.groupby("pair")[val].mean().to_dict()

    def rows_for(df, val):
        out = []
        np2map = np2_by_pair(df, val)
        for _, r in df.iterrows():
            info = parse_cond(str(r["operator"]))
            if info is None:
                continue
            tfams = COND_FAMS[info["cond"]]
            fa, fb = pair_endpoints(r["pair"])
            zs = [f for f in {fa, fb} if f not in tfams]   # zero-shot families this pair tests for this pack
            insample = len(zs) == 0
            n2 = np2map.get(r["pair"], np.nan)
            base = dict(cond=info["cond"], nfam=len(tfams), seed=info["seed"],
                        pair=r["pair"], err=float(r[val]),
                        ratio_np2=(float(r[val]) / n2 if n2 and n2 == n2 else np.nan),
                        insample=insample)
            if insample:
                out.append({**base, "heldout_family": "IN-SAMPLE"})
            else:
                for f in zs:
                    out.append({**base, "heldout_family": f})
        return pd.DataFrame(out)

    def summarize(df, val, title):
        R = rows_for(df, val)
        if R.empty:
            return
        print(f"\n===== {title} =====")
        # zero-shot per held-out family vs #families (mean over pairs+seeds; std over seeds)
        for fam in sorted(x for x in R.heldout_family.unique() if x != "IN-SAMPLE"):
            sub = R[R.heldout_family == fam]
            # per (cond,seed) mean over pairs, then mean/std over seeds
            per_seed = sub.groupby(["cond", "nfam", "seed"]).agg(err=("err", "mean"), r=("ratio_np2", "mean")).reset_index()
            g = per_seed.groupby(["cond", "nfam"]).agg(
                err_mean=("err", "mean"), err_std=("err", "std"),
                r_mean=("r", "mean"), nseed=("seed", "nunique")).reset_index().sort_values("nfam")
            print(f"\n  held-out family = {fam} (zero-shot):")
            with pd.option_context("display.float_format", lambda x: f"{x:.3e}" if abs(x) < 1 else f"{x:.3f}"):
                print(g.to_string(index=False))

    if not fld.empty:
        an = fld[fld.category == "analytic"]
        summarize(an, "area_rel_l2", "ANALYTIC zero-shot vs #training families")
        rl = fld[fld.category == "real"]
        if not rl.empty:
            summarize(rl, "area_rel_l2", "REAL-FIELD zero-shot vs #training families")
    if not sh.empty:
        summarize(sh, "mean_area_rel_l2", "SPECTRAL-SHELL zero-shot vs #training families")

if __name__ == "__main__":
    main()
