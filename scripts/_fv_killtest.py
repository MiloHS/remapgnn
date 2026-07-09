"""FV-moment kill-test on CS-r32 -> RLL-r90-180.

Question: v12's degree-1/2 moment correction uses monomials evaluated at cell
CENTERS (point value).  A true finite-volume (FV) remap wants CELL-AVERAGE
monomials.  The audit's analytic/spectral metric is POINT-VALUE truth, so
point-moments are aligned with it; real fields are cell-native (~cell-average),
which is exactly where v12 loses to np2.

This isolates the effect by building the frozen v12 operator with two moment
targets (point vs FV cell-average) and scoring each against two truths
(point-value vs cell-average), alongside np1 / np2.  If FV moments reduce the
CELL-AVERAGE error (and close the gap to np2), the lever is real.

Read-only w.r.t. the model; no training.  CPU is fine for one pair.
"""
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from remapgnn.config import load_config
from remapgnn.data import load_pair_tensors
from train_config_balanced_harmonic import (
    _real_sph_unnorm,
    read_source_xyz_from_edges,
    read_target_xyz_from_edges,
)
from train_config_highorder import operator_from_model, quadratic_moment_coef
from evaluate_refinement_convergence import analytic_function
from audit_remap_operator import build_learned_operator, parse_pack_spec

PAIR = "CS-r32_to_RLL-r90-180"
CFG = "configs/v20b_base_a3p0_mink8_geom_v12.json"
PACK = "models_medium_improv/highorder_signed_v12_geom_localmom_l2_fieldfirst.pt"
MAP_NP1 = "maps_medium_improv/map_CS-r32_to_RLL-r90-180_conserve.nc"
MAP_NP2 = "maps_medium_improv/map_CS-r32_to_RLL-r90-180_conserve_np2.nc"
M_SUBDIV = 8          # barycentric subdivisions per fan-triangle
N_CG = 800
EPS_REL = 1e-12
DTYPE = torch.float64

torch.manual_seed(0)
np.random.seed(0)


# ---------------------------------------------------------------- geometry ---
def ll2xyz(lon_deg, lat_deg):
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    return np.stack([np.cos(lat) * np.cos(lon),
                     np.cos(lat) * np.sin(lon),
                     np.sin(lat)], axis=-1)


def _normalize(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, 1e-300, None)


def _solid_angle(a, b, c):
    """Van Oosterom-Strackee solid angle of spherical triangle(s); a,b,c [...,3]."""
    triple = np.abs(np.einsum("...i,...i->...", a, np.cross(b, c)))
    denom = (1.0
             + np.einsum("...i,...i->...", a, b)
             + np.einsum("...i,...i->...", a, c)
             + np.einsum("...i,...i->...", b, c))
    return 2.0 * np.arctan2(triple, denom)


def _ref_small_triangles(m):
    """Barycentric corner-triples for m^2 small triangles tiling the unit tri."""
    tris = []
    for i in range(m):
        for j in range(m - i):
            # up triangle
            tris.append(((i, j), (i + 1, j), (i, j + 1)))
            if j < m - i - 1:
                # down triangle
                tris.append(((i + 1, j), (i, j + 1), (i + 1, j + 1)))
    out = []
    for (p, q, r) in tris:
        def bary(node):
            a, b = node
            wa = a / m
            wb = b / m
            return (wa, wb, 1.0 - wa - wb)
        out.append((bary(p), bary(q), bary(r)))
    return out  # list of (bary3, bary3, bary3)


def build_quadrature(xv, yv, m=M_SUBDIV):
    """Return (points [P,3], weights [P] = solid angle, cell_idx [P], cellA [N])
    for a grid whose cells are quads with corners xv,yv (deg, [N,4])."""
    V = ll2xyz(xv, yv)                       # [N,4,3]
    N = V.shape[0]
    center = _normalize(V.mean(axis=1))      # [N,3]
    ref = _ref_small_triangles(m)
    pts = []
    wts = []
    idx = []
    cell_arange = np.arange(N)
    # 4 fan triangles per quad: (center, V_k, V_{k+1})
    for k in range(4):
        A = center
        B = V[:, k, :]
        C = V[:, (k + 1) % 4, :]
        for (ba, bb, bc) in ref:
            # small-triangle sphere corners
            P1 = _normalize(ba[0] * A + ba[1] * B + ba[2] * C)
            P2 = _normalize(bb[0] * A + bb[1] * B + bb[2] * C)
            P3 = _normalize(bc[0] * A + bc[1] * B + bc[2] * C)
            cent = _normalize(P1 + P2 + P3)
            area = _solid_angle(P1, P2, P3)          # [N]
            pts.append(cent)
            wts.append(area)
            idx.append(cell_arange)
    points = np.concatenate(pts, axis=0)             # [P,3]
    weights = np.concatenate(wts, axis=0)            # [P]
    cell_idx = np.concatenate(idx, axis=0)           # [P]
    cellA = np.bincount(cell_idx, weights=weights, minlength=N)
    return points, weights, cell_idx, cellA


def cell_average(field_fn, points, weights, cell_idx, cellA):
    fv = field_fn(points)                            # [P]
    num = np.bincount(cell_idx, weights=fv * weights, minlength=cellA.shape[0])
    return num / np.clip(cellA, 1e-300, None)


# ------------------------------------------------------------------ fields ---
def field_fn(name):
    if name.startswith("Y_"):
        _, ls, ms = name.split("_")
        l, m = int(ls), int(ms)
        return lambda xyz: _real_sph_unnorm(l, m, xyz).astype(np.float64)
    return lambda xyz: analytic_function(name, xyz).astype(np.float64)


FIELDS = [
    ("x", "deg1"), ("y", "deg1"), ("z", "deg1"),
    ("Y_2_0", "deg2"), ("Y_2_2", "deg2"),
    ("Y_3_0", "deg3"), ("Y_4_0", "deg4"),
    ("smooth1", "smooth"), ("smooth2", "smooth"),
    ("Y_8_0", "hi_ctrl"), ("Y_16_0", "hi_ctrl"),
]


# --------------------------------------------------------------- operators ---
def area_rel_l2(pred, truth, area):
    num = np.sqrt(np.sum(area * (pred - truth) ** 2))
    den = np.sqrt(np.sum(area * truth ** 2))
    return float(num / max(den, 1e-300))


def apply_learned(S, si, ti, n_tgt, src_field):
    y = np.zeros(n_tgt, dtype=np.float64)
    np.add.at(y, ti, S * src_field[si])
    return y


def load_map_operator(path):
    ds = xr.open_dataset(path)
    row = ds["row"].values.astype(np.int64) - 1     # target (1-indexed)
    col = ds["col"].values.astype(np.int64) - 1     # source
    S = ds["S"].values.astype(np.float64)
    n_a = int(ds.sizes["n_a"]); n_b = int(ds.sizes["n_b"])
    ds.close()
    return row, col, S, n_a, n_b


def apply_map(row, col, S, n_tgt, src_field):
    y = np.zeros(n_tgt, dtype=np.float64)
    np.add.at(y, row, S * src_field[col])
    return y


def build_v12_operator(op, b, area_src, area_tgt, n_src, n_tgt, mc, mc2):
    p = op.pack
    S_t, _ = operator_from_model(
        op.model, b, area_src, area_tgt, n_src, n_tgt,
        float(p.get("scale", 1.0)), signed=op.signed,
        n_cg=N_CG, solve_dtype=DTYPE, eps_rel=EPS_REL,
        moment_coef=mc, moment_mode="local_soft_l2",
        moment_ridge=float(p.get("moment_ridge", 1e-4)),
        moment_relax=float(p.get("moment_relax", 1.0)),
        moment_iters=int(p.get("moment_iters", 1)),
        moment_coef2=mc2,
        moment2_ridge=float(p.get("moment2_ridge", 1e-3)),
        moment2_relax=float(p.get("moment2_relax", 0.5)),
        moment2_iters=int(p.get("moment2_iters", 1)),
        implicit_projection=bool(p.get("implicit_projection", False)),
    )
    return S_t.detach().cpu().numpy().astype(np.float64)


def main():
    import os
    os.chdir(str(ROOT))
    device = torch.device("cpu")
    cfg = load_config(CFG)

    # ---- geometry: corners from the np1 map -------------------------------
    dm = xr.open_dataset(MAP_NP1)
    xv_a, yv_a = dm["xv_a"].values, dm["yv_a"].values
    xv_b, yv_b = dm["xv_b"].values, dm["yv_b"].values
    area_a_map = dm["area_a"].values.astype(np.float64)
    area_b_map = dm["area_b"].values.astype(np.float64)
    cen_a = ll2xyz(dm["xc_a"].values, dm["yc_a"].values)
    cen_b = ll2xyz(dm["xc_b"].values, dm["yc_b"].values)
    dm.close()

    # ---- operator geometry (edge parquet ordering) -----------------------
    edf = pd.read_parquet(cfg.edge_path(PAIR), columns=["source_index", "target_index"])
    n_src = int(edf.source_index.max()) + 1
    n_tgt = int(edf.target_index.max()) + 1
    sx = read_source_xyz_from_edges(cfg.edge_path(PAIR), n_src)   # cell centers, edge order
    tx = read_target_xyz_from_edges(cfg.edge_path(PAIR), n_tgt)

    # ---- ALIGNMENT ASSERTS (edge order == map order) ---------------------
    print("=== alignment ===")
    print(f"n_src edge/map = {n_src}/{area_a_map.shape[0]}, n_tgt edge/map = {n_tgt}/{area_b_map.shape[0]}")
    d_src = np.abs(sx - cen_a).max()
    d_tgt = np.abs(tx - cen_b).max()
    print(f"max|edge_center - map_center|  src={d_src:.3e}  tgt={d_tgt:.3e}")
    assert n_src == area_a_map.shape[0] and n_tgt == area_b_map.shape[0], "size mismatch"
    assert d_src < 1e-6 and d_tgt < 1e-6, "cell ordering mismatch -> ABORT"

    # ---- quadrature ------------------------------------------------------
    print("=== building quadrature (m={}) ===".format(M_SUBDIV))
    ps, ws, ci, cellA_s = build_quadrature(xv_a, yv_a)
    pt, wt, cit, cellA_t = build_quadrature(xv_b, yv_b)
    print(f"src quad pts={ps.shape[0]}  cellA vs map: rel err {np.abs(cellA_s-area_a_map).max()/area_a_map.max():.2e}, "
          f"sum {cellA_s.sum():.6f} (4pi={4*np.pi:.6f})")
    print(f"tgt quad pts={pt.shape[0]}  cellA vs map: rel err {np.abs(cellA_t-area_b_map).max()/area_b_map.max():.2e}, "
          f"sum {cellA_t.sum():.6f}")

    # ---- cell-average coordinate + quadratic moments ---------------------
    def coord_fn(d):
        return lambda xyz: xyz[:, d]

    def quad_fn(d1, d2):
        return lambda xyz: xyz[:, d1] * xyz[:, d2]

    coords = [(0,), (1,), (2,)]
    quads = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]

    cavg_coord_s = np.stack([cell_average(coord_fn(d), ps, ws, ci, cellA_s) for (d,) in coords], axis=1)  # [n_src,3]
    cavg_coord_t = np.stack([cell_average(coord_fn(d), pt, wt, cit, cellA_t) for (d,) in coords], axis=1)
    cavg_quad_s = np.stack([cell_average(quad_fn(a, b), ps, ws, ci, cellA_s) for (a, b) in quads], axis=1)  # [n_src,6]
    cavg_quad_t = np.stack([cell_average(quad_fn(a, b), pt, wt, cit, cellA_t) for (a, b) in quads], axis=1)

    print("centroid drift (|<x>-center|) src max:", np.abs(cavg_coord_s - sx).max())

    si = torch.as_tensor(edf.source_index.values, dtype=torch.long)
    ti = torch.as_tensor(edf.target_index.values, dtype=torch.long)

    # POINT moments (exactly as v12) -- from cell centers sx/tx
    sx_t = torch.as_tensor(sx, dtype=torch.float32)
    tx_t = torch.as_tensor(tx, dtype=torch.float32)
    mc_pt = sx_t[si] - tx_t[ti]
    mc2_pt = quadratic_moment_coef(sx_t, tx_t, si, ti)

    # FV moments -- from cell averages
    cs1 = torch.as_tensor(cavg_coord_s, dtype=torch.float32)
    ct1 = torch.as_tensor(cavg_coord_t, dtype=torch.float32)
    cs2 = torch.as_tensor(cavg_quad_s, dtype=torch.float32)
    ct2 = torch.as_tensor(cavg_quad_t, dtype=torch.float32)
    mc_fv = cs1[si] - ct1[ti]
    mc2_fv = cs2[si] - ct2[ti]

    # ---- build operators -------------------------------------------------
    print("=== building operators (n_cg={}, {}, eps={}) ===".format(N_CG, DTYPE, EPS_REL))
    op = build_learned_operator(parse_pack_spec(f"v12={PACK}@{CFG}"), Path(CFG), device)
    b = load_pair_tensors(op.cfg, PAIR, op.pack["stats"], device=device)
    area_src = b["area_src"].to(DTYPE)
    area_tgt = b["area_tgt"].to(DTYPE)
    # sanity: operator tgt area == map area
    print("op area_tgt vs map area_b rel:",
          float((area_tgt.cpu().numpy() - area_b_map).__abs__().max() / area_b_map.max()))

    ops = {}
    print("  v12-point ...");   ops["v12-point"]   = build_v12_operator(op, b, area_src, area_tgt, n_src, n_tgt, mc_pt, mc2_pt)
    print("  v12-fvL2  ...");   ops["v12-fvL2"]    = build_v12_operator(op, b, area_src, area_tgt, n_src, n_tgt, mc_pt, mc2_fv)   # FV deg2 only
    print("  v12-fvL12 ...");   ops["v12-fvL12"]   = build_v12_operator(op, b, area_src, area_tgt, n_src, n_tgt, mc_fv, mc2_fv)   # FV deg1+deg2

    r1, c1, S1, _, _ = load_map_operator(MAP_NP1)
    r2, c2, S2, _, _ = load_map_operator(MAP_NP2)

    si_np = edf.source_index.values
    ti_np = edf.target_index.values
    area_b = area_b_map

    # ---- score -----------------------------------------------------------
    def score_all(name):
        fn = field_fn(name)
        # point-value representation (at centers)
        f_src_pt = fn(sx)
        f_tgt_pt = fn(tx)
        # cell-average representation
        f_src_ca = cell_average(fn, ps, ws, ci, cellA_s)
        f_tgt_ca = cell_average(fn, pt, wt, cit, cellA_t)
        res = {}
        for opn, S in ops.items():
            pred_pt = apply_learned(S, si_np, ti_np, n_tgt, f_src_pt)
            pred_ca = apply_learned(S, si_np, ti_np, n_tgt, f_src_ca)
            res[opn] = (area_rel_l2(pred_pt, f_tgt_pt, area_b),
                        area_rel_l2(pred_ca, f_tgt_ca, area_b))
        # maps
        for opn, (r, c, S) in {"np1": (r1, c1, S1), "np2": (r2, c2, S2)}.items():
            pred_pt = apply_map(r, c, S, n_tgt, f_src_pt)
            pred_ca = apply_map(r, c, S, n_tgt, f_src_ca)
            res[opn] = (area_rel_l2(pred_pt, f_tgt_pt, area_b),
                        area_rel_l2(pred_ca, f_tgt_ca, area_b))
        return res

    order = ["np1", "np2", "v12-point", "v12-fvL2", "v12-fvL12"]
    print("\n=== POINT-VALUE truth (area_rel_l2) ===")
    hdr = f"{'field':10s} {'grp':8s} " + " ".join(f"{o:>11s}" for o in order)
    print(hdr)
    agg_pt = {o: [] for o in order}
    agg_ca = {o: [] for o in order}
    rows_pt = {}
    rows_ca = {}
    for name, grp in FIELDS:
        res = score_all(name)
        rows_pt[name] = (grp, {o: res[o][0] for o in order})
        rows_ca[name] = (grp, {o: res[o][1] for o in order})
        print(f"{name:10s} {grp:8s} " + " ".join(f"{res[o][0]:11.4e}" for o in order))
        if grp != "hi_ctrl":
            for o in order:
                agg_pt[o].append(res[o][0])
                agg_ca[o].append(res[o][1])
    print("\n=== CELL-AVERAGE truth (area_rel_l2)  [the FV / real-field-like metric] ===")
    print(hdr)
    for name, grp in FIELDS:
        grp2, d = rows_ca[name]
        print(f"{name:10s} {grp2:8s} " + " ".join(f"{d[o]:11.4e}" for o in order))

    print("\n=== MEAN over non-control fields (deg1..smooth) ===")
    print(f"{'metric':16s} " + " ".join(f"{o:>11s}" for o in order))
    print(f"{'point-truth':16s} " + " ".join(f"{np.mean(agg_pt[o]):11.4e}" for o in order))
    print(f"{'cellavg-truth':16s} " + " ".join(f"{np.mean(agg_ca[o]):11.4e}" for o in order))

    print("\n=== ratio to np2 (mean, cellavg-truth; <1 beats np2) ===")
    base = np.mean(agg_ca["np2"])
    for o in order:
        print(f"  {o:12s} {np.mean(agg_ca[o])/base:6.3f}")


if __name__ == "__main__":
    main()
