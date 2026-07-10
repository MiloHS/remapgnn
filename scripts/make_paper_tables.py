"""Generate presentable, VERIFIED paper CSV tables from the combined cell-average audit.

Source of truth (ONE file):
  analysis_medium_improv/audits/fv_gen_combined_cellavg_f64_proj800/spectral_shells.csv
  (produced by jobs_gen_audit.pbs / scripts/audit_remap_operator.py, --truth-mode cellavg, f64, n_cg 800)

Metric: mean_area_rel_l2 = area-weighted relative L2 error of the remapped field vs the analytic
cell-average truth, per spherical-harmonic shell band. Lower = better. Conservative & consistent by
construction. Classical baselines (np1/np2 = TempestRemap 1st/2nd order; esmf_* = ESMF) are
deterministic (no seed). Our learned operators are seed-averaged.

Outputs (paper_tables/):
  zeroshot_healpix_abs.csv        per-band absolute err: baselines + ours D2..D5 (means)
  zeroshot_healpix_ratio.csv      same, ratio to np2
  zeroshot_healpix_seedstats.csv  per-band mean/std over seeds for ours D2..D5
  indist_abs.csv                  per-band absolute err: baselines + ours (ALL max-coverage)
  indist_ratio.csv                same, ratio to np2
  indist_seedstats.csv            per-band mean/std over seeds for ours ALL
  diversity_ladder.csv            zero-shot err vs #families, per held-out family (the generalization result)
"""
import os, sys
import numpy as np, pandas as pd

D = "analysis_medium_improv/audits/fv_gen_combined_cellavg_f64_proj800"
SRC = f"{D}/spectral_shells.csv"
OUT = "paper_tables"; os.makedirs(OUT, exist_ok=True)
sh = pd.read_csv(SRC)
BANDS = ["l1-8", "l9-16", "l17-24", "l25-32", "l33-40", "l41-48"]  # low->high wavenumber
def order(df): return df.reindex(BANDS)

# ---- training-pair sets per diversity level (from jobs_gen_train.pbs) ----
TRAIN = {
 "D2": {"CS-r32_to_ICOD-r32","ICOD-r32_to_CS-r32","CS-r64_to_ICOD-r64"},
}
TRAIN["D3"] = TRAIN["D2"] | {"CS-r32_to_RLL-r90-180","RLL-r90-180_to_CS-r32","ICOD-r32_to_RLL-r90-180"}
TRAIN["D4"] = TRAIN["D3"] | {"ICO-r32_to_CS-r32","ICOD-r32_to_ICO-r32"}
TRAIN["D5"] = TRAIN["D4"] | {"MPAS-r4_to_CS-r32"}
TRAIN["ALL"] = TRAIN["D5"] | {"CS-r32_to_HP-n32","HP-n32_to_CS-r32","ICOD-r32_to_HP-n32","HP-n32_to_ICOD-r32"}
FAM_TOKENS = {"HEALPix":"HP-n32","ICO":"ICO-r32","MPAS":"MPAS","RLL":"RLL-r90-180"}
# levels at which each family is still HELD OUT (family not in TRAIN[level])
HELDOUT_LEVELS = {"HEALPix":["D2","D3","D4","D5"], "MPAS":["D2","D3","D4"], "ICO":["D2","D3"], "RLL":["D2"]}

def pairs_with(tok): return sh[sh.pair.str.contains(tok, regex=False)].pair.unique()
def band_mean(df, opprefix, seeds):
    r = df[df.operator.str.startswith(opprefix)] if seeds else df[df.operator == opprefix]
    return order(r.groupby("shell_label")["mean_area_rel_l2"].mean())

CLASS = [("np1","np1",False),("np2","np2",False),("esmf_bilinear","esmf_bil",False),
         ("esmf_conserve","esmf_cons",False),("esmf_conserve2nd","esmf_cons2",False)]

# =================== VERIFICATION ===================
checks = []
def chk(name, ok, detail=""): checks.append((name, ok, detail));
# 1) esmf_conserve must equal np1 (both 1st-order conservative on the same overlap) -> pipeline sanity
hp = sh[sh.pair.str.contains("HP-n32", regex=False)]
a = band_mean(hp,"np1",False); b = band_mean(hp,"esmf_conserve",False)
chk("esmf_conserve == np1 (HEALPix, all bands)", bool(np.allclose(a.values,b.values,rtol=1e-6,atol=0)),
    f"max|rel diff|={np.nanmax(np.abs(a.values/b.values-1)):.2e}")
# 2) seed counts: each ours condition should have exactly 3 seeds present in the audit
for c in ["D2","D3","D4","D5","ALL"]:
    ns = sh[sh.operator.str.startswith(f"fv_gen_{c}_")].operator.nunique()
    chk(f"{c}: seed count", ns==3, f"n_seeds={ns} ({sorted(sh[sh.operator.str.startswith(f'fv_gen_{c}_')].operator.unique())})")
# 3) leakage: for each held-out (family, level), no eval pair may be in that level's training set
for fam, levels in HELDOUT_LEVELS.items():
    ep = set(pairs_with(FAM_TOKENS[fam]))
    for lv in levels:
        leak = ep & TRAIN[lv]
        chk(f"no leakage {fam}@{lv}", len(leak)==0, f"eval_pairs={len(ep)} leak={leak}")
# 4) independent recompute of two sample cells straight from the raw rows
def raw_cell(tok, op_is_prefix, op, band, seeds):
    r = sh[sh.pair.str.contains(tok, regex=False) & (sh.shell_label==band)]
    r = r[r.operator.str.startswith(op)] if seeds else r[r.operator==op]
    return r["mean_area_rel_l2"].mean()
v1 = raw_cell("HP-n32","",  "np2", "l33-40", False)
v2 = raw_cell("HP-n32","p","fv_gen_D4_", "l33-40", True)
t1 = band_mean(hp,"np2",False)["l33-40"]; t2 = band_mean(hp,"fv_gen_D4_",True)["l33-40"]
chk("recompute np2 HP l33-40", np.isclose(v1,t1), f"{v1:.6e} vs {t1:.6e}")
chk("recompute oursD4 HP l33-40", np.isclose(v2,t2), f"{v2:.6e} vs {t2:.6e}")

print("========== VERIFICATION ==========")
allok = True
for name, ok, detail in checks:
    allok &= ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail or not ok else ""))
print(f"  OVERALL: {'ALL PASS' if allok else 'FAILURES PRESENT'}")
print("==================================\n")

# =================== ZERO-SHOT (HEALPix held-out) ===================
zs = {name: band_mean(hp, op, sd) for op,name,sd in CLASS}
for c in ["D2","D3","D4","D5"]: zs[f"ours_{c}"] = band_mean(hp, f"fv_gen_{c}_", True)
zt = pd.DataFrame(zs).reindex(BANDS); zt.index.name = "band"
zt.to_csv(f"{OUT}/zeroshot_healpix_abs.csv", float_format="%.6e")
(zt.div(zt["np2"],axis=0)).to_csv(f"{OUT}/zeroshot_healpix_ratio.csv", float_format="%.4f")
# seed stats for ours
rows=[]
for c in ["D2","D3","D4","D5"]:
    r = hp[hp.operator.str.startswith(f"fv_gen_{c}_")]
    perseed = r.groupby(["operator","shell_label"])["mean_area_rel_l2"].mean().reset_index()
    g = perseed.groupby("shell_label")["mean_area_rel_l2"].agg(["mean","std","count"])
    g = order(g); g["cond"]=c; rows.append(g.reset_index())
pd.concat(rows).to_csv(f"{OUT}/zeroshot_healpix_seedstats.csv", index=False, float_format="%.6e")

# =================== IN-DISTRIBUTION (ALL model on trained pairs) ===================
ind = sh[sh.pair.isin(TRAIN["ALL"])]
it = {name: band_mean(ind, op, sd) for op,name,sd in CLASS}
it["ours_ALL"] = band_mean(ind, "fv_gen_ALL_", True)
itab = pd.DataFrame(it).reindex(BANDS); itab.index.name="band"
itab.to_csv(f"{OUT}/indist_abs.csv", float_format="%.6e")
(itab.div(itab["np2"],axis=0)).to_csv(f"{OUT}/indist_ratio.csv", float_format="%.4f")
r = ind[ind.operator.str.startswith("fv_gen_ALL_")]
perseed = r.groupby(["operator","shell_label"])["mean_area_rel_l2"].mean().reset_index()
order(perseed.groupby("shell_label")["mean_area_rel_l2"].agg(["mean","std","count"])).to_csv(
    f"{OUT}/indist_seedstats.csv", float_format="%.6e")

# =================== DIVERSITY LADDER (zero-shot err vs #families) ===================
DLEV = {"D2":2,"D3":3,"D4":4,"D5":5}
lad=[]
for fam, levels in HELDOUT_LEVELS.items():
    fp = sh[sh.pair.str.contains(FAM_TOKENS[fam], regex=False)]
    for lv in levels:
        r = fp[fp.operator.str.startswith(f"fv_gen_{lv}_")]
        # per-seed overall (mean over bands & pairs), then mean/std over seeds
        perseed = r.groupby("operator")["mean_area_rel_l2"].mean()
        lad.append({"held_out_family":fam, "n_train_families":DLEV[lv], "level":lv,
                    "spectral_err_mean":perseed.mean(), "spectral_err_std":perseed.std(),
                    "n_seeds":int(perseed.shape[0]), "n_eval_pairs":fp.pair.nunique()})
pd.DataFrame(lad).to_csv(f"{OUT}/diversity_ladder.csv", index=False, float_format="%.6e")

print("SOURCE (verify against this):", SRC)
print("WROTE:")
for f in sorted(os.listdir(OUT)):
    print("  ", os.path.join(OUT,f))
print("\n--- diversity_ladder.csv ---")
print(pd.DataFrame(lad).to_string(index=False,
      float_format=lambda x:"%.4e"%x if isinstance(x,float) else x))
