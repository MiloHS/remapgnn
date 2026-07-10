"""Fit order-of-accuracy (convergence) slopes from a cell-average convergence audit.
Reads field_metrics.csv (one row per operator,pair,function), h = sqrt(4*pi/n_tgt),
fits log(area_rel_l2) vs log(h) per (operator, direction, function); reports slope + R2.
Ours seeds (ours_s0/s1/s2) averaged. Reproduces summarize_convergence_slopes.fit_slope."""
import numpy as np, pandas as pd, sys, os
D = sys.argv[1] if len(sys.argv) > 1 else "analysis_medium_improv/audits/convergence_CS_ICOD_cellavg"
d = pd.read_csv(f"{D}/field_metrics.csv")
d = d[d.pair.str.match(r"(CS|ICOD)-r\d+_to_(ICOD|CS)-r\d+")].copy()
d["dir"] = d.pair.str.replace(r"-r\d+", "", regex=True)          # CS_to_ICOD / ICOD_to_CS
d["h"] = np.sqrt(4*np.pi / d.n_tgt)
d = d[d.area_rel_l2 > 0]
# collapse ours seeds -> single 'ours' by mean error per (pair,function)
seeds = d[d.operator.str.startswith("ours_")]
if len(seeds):
    agg = (seeds.groupby(["dir","function","pair","h"], as_index=False)["area_rel_l2"].mean())
    agg["operator"] = "ours(ALL)"
    d = pd.concat([d[~d.operator.str.startswith("ours_")], agg], ignore_index=True)

def fit(g):
    g = g.sort_values("h")
    if g.h.nunique() < 2: return None
    p, a = np.polyfit(np.log(g.h.values), np.log(g.area_rel_l2.values), 1)
    yh = a + p*np.log(g.h.values)
    ss = ((np.log(g.area_rel_l2.values)-yh)**2).sum()
    tot = ((np.log(g.area_rel_l2.values)-np.log(g.area_rel_l2.values).mean())**2).sum()
    return p, (1-ss/tot if tot>0 else np.nan), len(g), g.area_rel_l2.iloc[-1], g.area_rel_l2.iloc[0]

rows=[]
for (op, dr, fn), g in d.groupby(["operator","dir","function"]):
    r = fit(g)
    if r: rows.append(dict(operator=op, direction=dr, function=fn,
                           order=r[0], r2=r[1], n_levels=r[2], E_coarse=r[3], E_fine=r[4]))
res = pd.DataFrame(rows)
os.makedirs("paper_tables", exist_ok=True)
res.to_csv("paper_tables/order_of_accuracy_slopes.csv", index=False, float_format="%.4f")

print("=== per-(operator,direction,function) fitted order (cell-average metric) ===")
print(res.sort_values(["direction","operator","function"]).to_string(index=False,
      float_format=lambda x:"%.3f"%x))
print("\n=== MEAN fitted order per operator (averaged over directions & smooth functions) ===")
summ = res.groupby("operator").agg(order_mean=("order","mean"), order_std=("order","std"),
                                   r2_mean=("r2","mean"), n=("order","size")).reset_index()
print(summ.to_string(index=False, float_format=lambda x:"%.3f"%x))
summ.to_csv("paper_tables/order_of_accuracy_summary.csv", index=False, float_format="%.4f")
print("\nWROTE paper_tables/order_of_accuracy_slopes.csv, order_of_accuracy_summary.csv")
print("SOURCE:", f"{D}/field_metrics.csv")
