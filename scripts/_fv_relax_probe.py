"""Quick probe: is the low-l gap vs np2 because the degree-2 moment correction is SOFT?

Re-evaluate the existing FV model (trained with moment2_relax=0.5) with the degree-2 local-soft
correction cranked up (relax/iters), on the CELL-AVERAGE metric, per spectral band.  No retrain.
If low-l error drops as relax->1 / more iters -> "soft" is the bottleneck (cheap fix / config).
If it doesn't move -> l=3 is genuinely missing -> degree-3 is the right lever.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from remapgnn.config import load_config
from remapgnn.data import load_pair_tensors
from remapgnn import fv_moments as fv
from remapgnn.models import scatter_sum_torch
from train_config_balanced_harmonic import read_source_xyz_from_edges, read_target_xyz_from_edges
from train_config_highorder import operator_from_model
from audit_remap_operator import build_learned_operator, parse_pack_spec

CFG = "configs/v20b_base_a3p0_mink8_geom_v12.json"
PACK = "models_medium_improv/highorder_signed_v12_fv_l2_seed0.pt"
PAIRS = ["CS-r32_to_ICOD-r32", "ICOD-r32_to_CS-r32"]
N_CG, EPS, DT, M = 800, 1e-12, torch.float64, 8
BANDS = {"l<=8": [(4, 0), (4, 2), (8, 0), (8, 4)],
         "l9-16": [(12, 0), (16, 0), (16, 8)],
         "l17-24": [(20, 0), (24, 0), (24, 12)]}
CONFIGS = [(0.5, 1), (0.9, 1), (1.0, 1), (1.0, 3), (0.5, 3)]   # (moment2_relax, moment2_iters)

def area_rel_l2(pred, truth, area):
    return float(np.sqrt(np.sum(area*(pred-truth)**2))/max(np.sqrt(np.sum(area*truth**2)), 1e-300))

def apply_edge(S, si, ti, nt, f):
    y = np.zeros(nt); np.add.at(y, ti, S*f[si]); return y

def map_op(path):
    import xarray as xr
    ds = xr.open_dataset(path); r = ds["row"].values-1; c = ds["col"].values-1; S = ds["S"].values.astype(float); ds.close()
    return r, c, S

op = build_learned_operator(parse_pack_spec(f"fv={PACK}@{CFG}"), Path(CFG), torch.device("cpu"))
cfg = load_config(CFG)
p = op.pack
print(f"model moment2_relax(train)={p.get('moment2_relax')} moment2_iters={p.get('moment2_iters')} geom={p.get('moment_geometry')}")

for pair in PAIRS:
    edf = pd.read_parquet(cfg.edge_path(pair), columns=["source_index", "target_index"])
    si = edf.source_index.values; ti = edf.target_index.values
    ns = int(si.max())+1; nt = int(ti.max())+1
    sx = read_source_xyz_from_edges(cfg.edge_path(pair), ns); tx = read_target_xyz_from_edges(cfg.edge_path(pair), nt)
    mp = str(ROOT/f"maps_medium_improv/map_{pair}_conserve.nc")
    mp2 = str(ROOT/f"maps_medium_improv/map_{pair}_conserve_np2.nc")
    Vs, nvs, _, _ = fv.load_corners_from_map(mp, "a"); Vt, nvt, _, _ = fv.load_corners_from_map(mp, "b")
    ms = fv.compute_grid_moments(Vs, nvs, m=M); mt = fv.compute_grid_moments(Vt, nvt, m=M)
    qs = fv.grid_quadrature(mp, "a", m=M); qt = fv.grid_quadrature(mp, "b", m=M)
    area_b = qt["area"]
    sit = torch.as_tensor(si); tit = torch.as_tensor(ti)
    mc = torch.as_tensor(ms["coord"], dtype=torch.float32)[sit] - torch.as_tensor(mt["coord"], dtype=torch.float32)[tit]
    mc2 = torch.as_tensor(ms["quad"], dtype=torch.float32)[sit] - torch.as_tensor(mt["quad"], dtype=torch.float32)[tit]
    b = load_pair_tensors(op.cfg, pair, op.pack["stats"], device="cpu")
    asrc = b["area_src"].to(DT); atgt = b["area_tgt"].to(DT)
    # truth per band field (cell-average)
    truths = {}
    for band, modes in BANDS.items():
        for (l, m) in modes:
            fn = lambda xyz, l=l, m=m: fv.real_sph_unnorm_fast(l, m, xyz)
            truths[(l, m)] = (fv.grid_cell_average(fn, qs), fv.grid_cell_average(fn, qt))
    # np2 baseline
    r2, c2, S2 = map_op(mp2)
    print(f"\n=== {pair} ===")
    hdr = f"{'cfg(relax,iters)':18s} " + " ".join(f"{bd:>10s}" for bd in BANDS) + "   cons_resid"
    print(hdr)
    # np2 row
    np2_band = {}
    for band, modes in BANDS.items():
        errs = [area_rel_l2(apply_edge(S2, c2, r2, nt, truths[(l, m)][0]), truths[(l, m)][1], area_b) for (l, m) in modes]
        np2_band[band] = np.mean(errs)
    print(f"{'np2 (2nd-order)':18s} " + " ".join(f"{np2_band[bd]:10.3e}" for bd in BANDS) + "        --")
    for relax, iters in CONFIGS:
        with torch.no_grad():
            S, Mo = operator_from_model(op.model, b, asrc, atgt, ns, nt, float(p.get("scale", 1.0)),
                signed=op.signed, n_cg=N_CG, solve_dtype=DT, eps_rel=EPS,
                moment_coef=mc, moment_mode="local_soft_l2",
                moment_ridge=float(p.get("moment_ridge", 1e-4)), moment_relax=float(p.get("moment_relax", 1.0)),
                moment_iters=int(p.get("moment_iters", 1)), moment_coef2=mc2,
                moment2_ridge=float(p.get("moment2_ridge", 1e-3)), moment2_relax=relax, moment2_iters=iters,
                implicit_projection=bool(p.get("implicit_projection", False)))
        Sn = S.detach().cpu().numpy().astype(float)
        Mn = Mo.detach().cpu().numpy().astype(float)
        cons = float(np.linalg.norm(np.bincount(si, Mn, ns) - asrc.numpy())/np.linalg.norm(asrc.numpy()))
        line = {}
        for band, modes in BANDS.items():
            errs = [area_rel_l2(apply_edge(Sn, si, ti, nt, truths[(l, m)][0]), truths[(l, m)][1], area_b) for (l, m) in modes]
            line[band] = np.mean(errs)
        tag = f"relax={relax} it={iters}" + (" [train]" if (relax, iters) == (0.5, 1) else "")
        print(f"{tag:18s} " + " ".join(f"{line[bd]:10.3e}" for bd in BANDS) + f"   {cons:.2e}")
