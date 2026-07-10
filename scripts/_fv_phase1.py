"""Phase 1 (GATE G1): frozen-model FV-moment A/B across ALL v12 pairs.

For each pair: build the frozen v12 operator with POINT vs FV cell-average moments, and
score each against POINT-VALUE and CELL-AVERAGE truth, alongside np1/np2.  Confirms whether
the kill-test win (FV closes ~half the gap to np2 on the cell-average metric) generalizes
beyond CS-r32->RLL.  Uses remapgnn/fv_moments for general n-gon quadrature.
"""
import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from remapgnn.config import load_config
from remapgnn.data import load_pair_tensors
from remapgnn import fv_moments as fv
from train_config_balanced_harmonic import (
    _real_sph_unnorm, read_source_xyz_from_edges, read_target_xyz_from_edges)
from train_config_highorder import operator_from_model, quadratic_moment_coef
from evaluate_refinement_convergence import analytic_function
from audit_remap_operator import build_learned_operator, parse_pack_spec

CFG = "configs/v20b_base_a3p0_mink8_geom_v12.json"
PACK = "models_medium_improv/highorder_signed_v12_geom_localmom_l2_fieldfirst.pt"
N_CG, EPS_REL, DTYPE, M = 800, 1e-12, torch.float64, 8
CACHE = "analysis_medium_improv/fv_moment_cache"

FIELDS = [("x","d1"),("y","d1"),("z","d1"),("Y_2_0","d2"),("Y_2_2","d2"),
          ("Y_3_0","d3"),("Y_4_0","d4"),("smooth1","sm"),("smooth2","sm"),
          ("Y_8_0","ctrl"),("Y_16_0","ctrl")]

def field_fn(name):
    if name.startswith("Y_"):
        _, ls, ms = name.split("_"); l, m = int(ls), int(ms)
        return lambda xyz: _real_sph_unnorm(l, m, xyz).astype(np.float64)
    return lambda xyz: analytic_function(name, xyz).astype(np.float64)

def area_rel_l2(pred, truth, area):
    return float(np.sqrt(np.sum(area*(pred-truth)**2))/max(np.sqrt(np.sum(area*truth**2)),1e-300))

def load_map_op(path):
    import xarray as xr
    ds = xr.open_dataset(path)
    r = ds["row"].values.astype(np.int64)-1; c = ds["col"].values.astype(np.int64)-1
    S = ds["S"].values.astype(np.float64); ds.close()
    return r, c, S

def apply_learned(S, si, ti, n_tgt, f):
    y = np.zeros(n_tgt); np.add.at(y, ti, S*f[si]); return y
def apply_map(r, c, S, n_tgt, f):
    y = np.zeros(n_tgt); np.add.at(y, r, S*f[c]); return y

def build_v12(op, b, asrc, atgt, ns, nt, mc, mc2):
    p = op.pack
    S,_ = operator_from_model(op.model, b, asrc, atgt, ns, nt, float(p.get("scale",1.0)),
        signed=op.signed, n_cg=N_CG, solve_dtype=DTYPE, eps_rel=EPS_REL,
        moment_coef=mc, moment_mode="local_soft_l2",
        moment_ridge=float(p.get("moment_ridge",1e-4)), moment_relax=float(p.get("moment_relax",1.0)),
        moment_iters=int(p.get("moment_iters",1)), moment_coef2=mc2,
        moment2_ridge=float(p.get("moment2_ridge",1e-3)), moment2_relax=float(p.get("moment2_relax",0.5)),
        moment2_iters=int(p.get("moment2_iters",1)), implicit_projection=bool(p.get("implicit_projection",False)))
    return S.detach().cpu().numpy().astype(np.float64)

def main():
    os.chdir(ROOT)
    cfg = load_config(CFG)
    pairs = cfg.pairs
    op = build_learned_operator(parse_pack_spec(f"v12={PACK}@{CFG}"), Path(CFG), torch.device("cpu"))
    order = ["np1","np2","v12-point","v12-fv"]
    summary = []
    for pair in pairs:
        try:
            mp1 = str(ROOT/f"maps_medium_improv/map_{pair}_conserve.nc")
            mp2 = str(ROOT/f"maps_medium_improv/map_{pair}_conserve_np2.nc")
            if not Path(mp1).exists() or not Path(mp2).exists():
                print(f"[skip] {pair}: missing map(s)"); continue
            edf = pd.read_parquet(cfg.edge_path(pair), columns=["source_index","target_index"])
            si = edf.source_index.values; ti = edf.target_index.values
            ns = int(si.max())+1; nt = int(ti.max())+1
            sx = read_source_xyz_from_edges(cfg.edge_path(pair), ns)
            tx = read_target_xyz_from_edges(cfg.edge_path(pair), nt)
            # grid geometry + quadrature (general n-gon)
            Vs, nvs, area_a, cen_a = fv.load_corners_from_map(mp1, "a")
            Vt, nvt, area_b, cen_b = fv.load_corners_from_map(mp1, "b")
            assert np.abs(cen_a-sx).max()<1e-6 and np.abs(cen_b-tx).max()<1e-6, "ordering mismatch"
            ps,ws,ci,cAs = fv.build_quadrature(Vs, nvs, m=M)
            pt,wt,cit,cAt = fv.build_quadrature(Vt, nvt, m=M)
            # FV moments
            mom_s = fv.compute_grid_moments(Vs, nvs, m=M)
            mom_t = fv.compute_grid_moments(Vt, nvt, m=M)
            si_t = torch.as_tensor(si, dtype=torch.long); ti_t = torch.as_tensor(ti, dtype=torch.long)
            sx_t = torch.as_tensor(sx, dtype=torch.float32); tx_t = torch.as_tensor(tx, dtype=torch.float32)
            mc_pt = sx_t[si_t]-tx_t[ti_t]; mc2_pt = quadratic_moment_coef(sx_t, tx_t, si_t, ti_t)
            mc1_fv, mc2_fv = fv.moment_coefs(mom_s, mom_t, si, ti)
            mc_fv = torch.as_tensor(mc1_fv, dtype=torch.float32)
            mc2_fv = torch.as_tensor(mc2_fv, dtype=torch.float32)
            # operators
            b = load_pair_tensors(op.cfg, pair, op.pack["stats"], device="cpu")
            asrc = b["area_src"].to(DTYPE); atgt = b["area_tgt"].to(DTYPE)
            ops = {"v12-point": build_v12(op,b,asrc,atgt,ns,nt,mc_pt,mc2_pt),
                   "v12-fv":    build_v12(op,b,asrc,atgt,ns,nt,mc_fv,mc2_fv)}
            r1,c1,S1 = load_map_op(mp1); r2,c2,S2 = load_map_op(mp2)
            area_b_np = area_b
            # score
            agg = {o: {"pt":[], "ca":[]} for o in order}
            for name, grp in FIELDS:
                fn = field_fn(name)
                fsrc_pt = fn(sx); ftgt_pt = fn(tx)
                fsrc_ca = fv.cell_average(fn, ps, ws, ci, cAs); ftgt_ca = fv.cell_average(fn, pt, wt, cit, cAt)
                res = {}
                for o,S in ops.items():
                    res[o] = (area_rel_l2(apply_learned(S,si,ti,nt,fsrc_pt),ftgt_pt,area_b_np),
                              area_rel_l2(apply_learned(S,si,ti,nt,fsrc_ca),ftgt_ca,area_b_np))
                res["np1"] = (area_rel_l2(apply_map(r1,c1,S1,nt,fsrc_pt),ftgt_pt,area_b_np),
                              area_rel_l2(apply_map(r1,c1,S1,nt,fsrc_ca),ftgt_ca,area_b_np))
                res["np2"] = (area_rel_l2(apply_map(r2,c2,S2,nt,fsrc_pt),ftgt_pt,area_b_np),
                              area_rel_l2(apply_map(r2,c2,S2,nt,fsrc_ca),ftgt_ca,area_b_np))
                if grp != "ctrl":
                    for o in order:
                        agg[o]["pt"].append(res[o][0]); agg[o]["ca"].append(res[o][1])
            m_ca = {o: float(np.mean(agg[o]["ca"])) for o in order}
            m_pt = {o: float(np.mean(agg[o]["pt"])) for o in order}
            row = dict(pair=pair,
                       ca_np2=m_ca["np2"], ca_point=m_ca["v12-point"], ca_fv=m_ca["v12-fv"],
                       ratio_point=m_ca["v12-point"]/m_ca["np2"], ratio_fv=m_ca["v12-fv"]/m_ca["np2"],
                       pt_point=m_pt["v12-point"], pt_fv=m_pt["v12-fv"])
            summary.append(row)
            print(f"[ok] {pair:26s} CELLAVG np2={m_ca['np2']:.3e} point={m_ca['v12-point']:.3e} "
                  f"fv={m_ca['v12-fv']:.3e} | ratio2np2 point={row['ratio_point']:.2f} fv={row['ratio_fv']:.2f} "
                  f"| POINT point={m_pt['v12-point']:.3e} fv={m_pt['v12-fv']:.3e}", flush=True)
        except Exception as e:
            print(f"[ERR] {pair}: {e}", flush=True)
    df = pd.DataFrame(summary)
    if not df.empty:
        Path(CACHE).mkdir(parents=True, exist_ok=True)
        df.to_csv(ROOT/"analysis_medium_improv/fv_phase1_summary.csv", index=False)
        print("\n=== SUMMARY (cell-average metric, ratio to np2; <1 beats np2) ===")
        print(df[["pair","ratio_point","ratio_fv"]].to_string(index=False))
        n_improve = int((df.ca_fv < df.ca_point).sum())
        print(f"\nFV improves cell-average error on {n_improve}/{len(df)} pairs")
        print(f"mean ratio2np2  point={df.ratio_point.mean():.3f}  fv={df.ratio_fv.mean():.3f}")
        # point-value metric (expected: FV slightly hurts)
        n_pt_worse = int((df.pt_fv > df.pt_point).sum())
        print(f"point-value metric: FV worse on {n_pt_worse}/{len(df)} (expected)")
        print("G1:", "PASS" if n_improve > len(df)//2 else "FAIL")

if __name__ == "__main__":
    main()
