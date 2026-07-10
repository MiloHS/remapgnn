#!/usr/bin/env python
"""Clean audit suite for supermesh-free remap operators.

This script evaluates TempestRemap np1/np2 maps, signed high-order base packs,
and iterative high-order corrector packs with one common set of metrics:

  * analytic/real-field area-relative L2 errors
  * spectral shell errors through configurable lmax bands
  * Cartesian and local tangent-plane moment/geometry residuals
  * conservation + consistency residuals
  * rough operator-build runtime and candidate edge counts

Pack specs use:

    label=path/to/pack.pt@path/to/config.json

The config suffix matters: v10b should be audited on its a3 graph, while the
wide-stencil v10d run should be audited on its a4/min_k16 graph.  If @config is
omitted, the script uses checkpoint metadata when present, otherwise --config.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from remapgnn.config import load_config
from remapgnn.data import load_pair_tensors
from remapgnn.models import build_model, scatter_sum_torch
from train_config_balanced_harmonic import (
    _real_sph_unnorm,
    build_harmonic_fields_with_truth,
    read_source_xyz_from_edges,
    read_target_xyz_from_edges,
    choose_m_values,
    _stable_pair_seed,
)
from remapgnn import fv_moments as fv
from train_config_highorder import operator_from_model, quadratic_moment_coef
from train_config_highorder_corrector import run_corrector_steps
from train_config_irno_corrector import as_int, torch_load_pack
from evaluate_refinement_convergence import analytic_function


DEFAULT_FUNCTIONS = [
    "x",
    "y",
    "z",
    "smooth1",
    "smooth2",
    "Y_4_0",
    "Y_8_0",
    "Y_16_0",
    "Y_24_0",
    "Y_24_12",
    "Y_32_0",
    "Y_40_0",
    "Y_48_0",
    "Y_48_24",
]

DEFAULT_REAL_FIELDS = [
    "AnalyticalFun1",
    "AnalyticalFun2",
    "TotalPrecipWater",
    "CloudFraction",
    "Topography",
]


@dataclass
class PackSpec:
    label: str
    pack_path: Path
    config_path: Path | None


@dataclass
class LearnedOperator:
    label: str
    pack_path: Path
    cfg_path: Path
    cfg: object
    pack: dict
    base_pack: dict | None
    model: torch.nn.Module | None
    base_model: torch.nn.Module | None
    corrector: torch.nn.Module | None
    is_corrector: bool
    signed: bool


@dataclass
class SparseOperator:
    label: str
    family: str
    pair: str
    src_index: np.ndarray
    tgt_index: np.ndarray
    S: np.ndarray
    M: np.ndarray
    src_area: np.ndarray
    tgt_area: np.ndarray
    n_src: int
    n_tgt: int
    n_edges: int
    elapsed_s: float
    graph_suffix: str


def parse_pack_spec(spec: str) -> PackSpec:
    if "=" in spec:
        label, rest = spec.split("=", 1)
    else:
        p0 = spec.split("@", 1)[0]
        label, rest = Path(p0).stem, spec
    if "@" in rest:
        path_s, cfg_s = rest.rsplit("@", 1)
        cfg_path = Path(cfg_s)
    else:
        path_s, cfg_path = rest, None
    return PackSpec(label=label, pack_path=Path(path_s), config_path=cfg_path)


def resolve_cfg_path(spec_cfg: Path | None, pack: dict, default_cfg: Path) -> Path:
    if spec_cfg is not None:
        return spec_cfg
    meta = pack.get("config_path")
    if meta:
        p = Path(meta)
        if p.exists():
            return p
    return default_cfg


def load_model_from_pack(pack: dict, cfg, *, edge_dim_extra: int = 0) -> torch.nn.Module:
    model = build_model(
        architecture=pack.get("architecture", cfg.architecture),
        src_dim=len(pack["src_node_features"]),
        tgt_dim=len(pack["tgt_node_features"]),
        edge_dim=len(pack["edge_features"]) + edge_dim_extra,
        hidden=int(pack.get("hidden", 128)),
        decoder_chunk_size=int(pack.get("decoder_chunk_size", 10000)),
    )
    model.load_state_dict(pack["model_state_dict"])
    model.eval()
    model.num_rounds = int(pack.get("rounds", 1))
    return model


def build_learned_operator(spec: PackSpec, default_cfg_path: Path, device: torch.device) -> LearnedOperator:
    pack = torch_load_pack(spec.pack_path, map_location=device)
    cfg_path = resolve_cfg_path(spec.config_path, pack, default_cfg_path)
    cfg = load_config(cfg_path)
    is_corrector = pack.get("kind") == "highorder_corrector"

    if is_corrector:
        base_pack_path = Path(pack["base_pack"])
        base_pack = torch_load_pack(base_pack_path, map_location=device)
        base_model = load_model_from_pack(base_pack, cfg).to(device)
        corrector = load_model_from_pack(pack, cfg, edge_dim_extra=4).to(device)
        model = None
        signed = True
    else:
        base_pack = None
        base_model = None
        corrector = None
        model = load_model_from_pack(pack, cfg).to(device)
        signed = bool(pack.get("signed", False))

    print(
        f"loaded learned operator {spec.label}: pack={spec.pack_path} cfg={cfg_path} "
        f"graph={cfg.graph_suffix} corrector={is_corrector}"
    )
    return LearnedOperator(
        label=spec.label,
        pack_path=spec.pack_path,
        cfg_path=cfg_path,
        cfg=cfg,
        pack=pack,
        base_pack=base_pack,
        model=model,
        base_model=base_model,
        corrector=corrector,
        is_corrector=is_corrector,
        signed=signed,
    )


def load_map_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    v = Dataset(path).variables
    S = np.asarray(v["S"][:]).ravel().astype(np.float64)
    tgt_index = np.asarray(v["row"][:]).ravel().astype(np.int64) - 1
    src_index = np.asarray(v["col"][:]).ravel().astype(np.int64) - 1
    src_area = np.asarray(v["area_a"][:]).ravel().astype(np.float64)
    tgt_area = np.asarray(v["area_b"][:]).ravel().astype(np.float64)
    return S, src_index, tgt_index, src_area, tgt_area


def scatter_numpy(n: int, index: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.zeros(n, dtype=np.float64)
    np.add.at(out, index, values)
    return out


def area_rel_l2(pred: np.ndarray, ref: np.ndarray, area: np.ndarray, eps: float = 1.0e-30) -> float:
    num = float(np.sum(area * (pred - ref) ** 2))
    den = max(float(np.sum(area * ref * ref)), eps)
    return math.sqrt(num / den)


def rel_l2(pred: np.ndarray, ref: np.ndarray, eps: float = 1.0e-30) -> float:
    return float(np.linalg.norm(pred - ref) / max(np.linalg.norm(ref), eps))


def rel_integral_error(pred_int: float, ref_int: float) -> float:
    return abs(pred_int - ref_int) / max(abs(ref_int), 1.0e-30)


def parse_projection_dtype(name: str):
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"unknown projection dtype {name!r}; expected float32 or float64")


def residuals_from_mass(
    M: np.ndarray,
    src_index: np.ndarray,
    tgt_index: np.ndarray,
    src_area: np.ndarray,
    tgt_area: np.ndarray,
) -> tuple[float, float]:
    src_sum = scatter_numpy(len(src_area), src_index, M)
    tgt_sum = scatter_numpy(len(tgt_area), tgt_index, M)
    cons = float(np.linalg.norm(src_sum - src_area) / max(np.linalg.norm(src_area), 1.0e-30))
    row = float(np.linalg.norm(tgt_sum - tgt_area) / max(np.linalg.norm(tgt_area), 1.0e-30))
    return cons, row


def build_tempest_operator(cfg, pair: str, order: str) -> SparseOperator:
    suffix = "" if order == "np1" else "_np2"
    path = cfg.maps_dir / f"map_{pair}_conserve{suffix}.nc"
    t0 = time.perf_counter()
    S, si, ti, asrc, atgt = load_map_arrays(path)
    elapsed = time.perf_counter() - t0
    M = S * atgt[ti]
    return SparseOperator(
        label=order,
        family="tempest",
        pair=pair,
        src_index=si,
        tgt_index=ti,
        S=S,
        M=M,
        src_area=asrc,
        tgt_area=atgt,
        n_src=len(asrc),
        n_tgt=len(atgt),
        n_edges=len(S),
        elapsed_s=elapsed,
        graph_suffix="tempest_supermesh",
    )


def build_esmf_operator(cfg, pair: str, method: str) -> SparseOperator:
    """Load an ESMF baseline weight file (SCRIP format, same schema as TR maps).
    method in {bilinear, conserve, conserve2nd}. Path: map_<pair>_esmf_<method>.nc."""
    path = cfg.maps_dir / f"map_{pair}_esmf_{method}.nc"
    t0 = time.perf_counter()
    S, si, ti, _asrc, _atgt = load_map_arrays(path)
    elapsed = time.perf_counter() - t0
    # ESMF bilinear weight files carry zero/absent cell areas -> take the TRUE areas from the
    # conserve (np1) map (same cells/ordering) so the area-weighted metric is correct and
    # consistent across all operators (conserve/conserve2nd areas are identical anyway).
    _s, _si, _ti, asrc, atgt = load_map_arrays(cfg.maps_dir / f"map_{pair}_conserve.nc")
    M = S * atgt[ti]
    return SparseOperator(
        label=f"esmf_{method}",
        family="esmf",
        pair=pair,
        src_index=si,
        tgt_index=ti,
        S=S,
        M=M,
        src_area=asrc,
        tgt_area=atgt,
        n_src=len(asrc),
        n_tgt=len(atgt),
        n_edges=len(S),
        elapsed_s=elapsed,
        graph_suffix="esmf",
    )


def build_learned_sparse_operator(
    op: LearnedOperator,
    pair: str,
    device: torch.device,
    *,
    base_n_cg: int | None = None,
    corrector_n_cg: int | None = None,
    projection_dtype=torch.float32,
    projection_eps_rel: float = 1e-9,
) -> SparseOperator:
    t0 = time.perf_counter()
    b = load_pair_tensors(op.cfg, pair, op.pack["stats"], device=device)
    si_t = b["src_index"]
    ti_t = b["tgt_index"]
    n_src = as_int(b["n_src"])
    n_tgt = as_int(b["n_tgt"])
    asrc_t = b["area_src"].float()
    atgt_t = b["area_tgt"].float()

    with torch.no_grad():
        if op.is_corrector:
            p = op.pack
            bands = [int(x) for x in p["bands"]]
            alpha = [float(x) for x in p["step_alphas"]] if p.get("step_alphas") else float(p["alpha"])
            scale = float(p["scale"])
            n_cg = int(corrector_n_cg if corrector_n_cg is not None else p.get("n_cg", 400))
            lmax_denom = float(p.get("lmax_denom", 32.0))
            moment_coef = None
            if bool(p.get("moment_l1_hard", False)):
                sx = read_source_xyz_from_edges(op.cfg.edge_path(pair), n_src)
                tx = read_target_xyz_from_edges(op.cfg.edge_path(pair), n_tgt)
                moment_coef = (
                    torch.tensor(sx, dtype=torch.float32, device=device)[si_t]
                    - torch.tensor(tx, dtype=torch.float32, device=device)[ti_t]
                )
            _, steps, _ = run_corrector_steps(
                op.base_model,
                op.corrector,
                b,
                bands,
                alpha,
                scale,
                n_cg,
                lmax_denom,
                moment_coef=moment_coef,
                solve_dtype=projection_dtype,
                eps_rel=projection_eps_rel,
            )
            S_t, M_t = steps[-1][0], steps[-1][1]
        else:
            moment_mode = op.pack.get("moment_mode") or (
                "local_soft_l2" if bool(op.pack.get("moment_l2_local_soft", False))
                else "local_soft" if bool(op.pack.get("moment_l1_local_soft", False))
                else ("hard" if bool(op.pack.get("moment_l1_hard", False)) else "none")
            )
            moment_mode = str(moment_mode)
            moment_geometry = str(op.pack.get("moment_geometry", "center"))
            moment_coef = None
            moment_coef2 = None
            moment_coef3 = None
            if moment_mode != "none":
                sx = read_source_xyz_from_edges(op.cfg.edge_path(pair), n_src)
                tx = read_target_xyz_from_edges(op.cfg.edge_path(pair), n_tgt)
                if moment_geometry == "fv":
                    # inference must use the SAME finite-volume cell-average moments the model
                    # was trained with, else the projection targets the wrong (point) moments.
                    quad_m = int(op.pack.get("quad_m", 8))
                    mp = str(op.cfg.maps_dir / f"map_{pair}_conserve.nc")
                    Vs, nvs, _, cas = fv.load_corners_from_map(mp, "a")
                    Vt, nvt, _, cat = fv.load_corners_from_map(mp, "b")
                    if float(np.abs(cas - sx).max()) > 1e-6 or float(np.abs(cat - tx).max()) > 1e-6:
                        raise ValueError(f"FV cell-order mismatch for {pair} (map vs edge dataset)")
                    cubic = moment_mode == "local_soft_l3"
                    ms = fv.compute_grid_moments(Vs, nvs, m=quad_m, cubic=cubic)
                    mt = fv.compute_grid_moments(Vt, nvt, m=quad_m, cubic=cubic)
                    cs1 = torch.tensor(ms["coord"], dtype=torch.float32, device=device)
                    ct1 = torch.tensor(mt["coord"], dtype=torch.float32, device=device)
                    moment_coef = cs1[si_t] - ct1[ti_t]
                    if moment_mode in ("local_soft_l2", "local_soft_l3"):
                        cs2 = torch.tensor(ms["quad"], dtype=torch.float32, device=device)
                        ct2 = torch.tensor(mt["quad"], dtype=torch.float32, device=device)
                        moment_coef2 = cs2[si_t] - ct2[ti_t]
                    if moment_mode == "local_soft_l3":
                        cs3 = torch.tensor(ms["cubic"], dtype=torch.float32, device=device)
                        ct3 = torch.tensor(mt["cubic"], dtype=torch.float32, device=device)
                        moment_coef3 = cs3[si_t] - ct3[ti_t]
                else:
                    sxyz_t = torch.tensor(sx, dtype=torch.float32, device=device)
                    txyz_t = torch.tensor(tx, dtype=torch.float32, device=device)
                    moment_coef = (
                        sxyz_t[si_t]
                        - txyz_t[ti_t]
                    )
                    if moment_mode == "local_soft_l2":
                        moment_coef2 = quadratic_moment_coef(sxyz_t, txyz_t, si_t, ti_t)
            S_t, M_t = operator_from_model(
                op.model,
                b,
                asrc_t,
                atgt_t,
                n_src,
                n_tgt,
                float(op.pack.get("scale", 1.0)),
                signed=op.signed,
                n_cg=int(base_n_cg if base_n_cg is not None else op.pack.get("n_cg", 400)),
                solve_dtype=projection_dtype,
                eps_rel=projection_eps_rel,
                moment_coef=moment_coef,
                moment_mode=moment_mode,
                moment_ridge=float(op.pack.get("moment_ridge", 1.0e-4)),
                moment_relax=float(op.pack.get("moment_relax", 1.0)),
                moment_iters=int(op.pack.get("moment_iters", 1)),
                moment_coef2=moment_coef2,
                moment2_ridge=float(op.pack.get("moment2_ridge", 1.0e-3)),
                moment2_relax=float(op.pack.get("moment2_relax", 0.5)),
                moment2_iters=int(op.pack.get("moment2_iters", 1)),
                moment_coef3=moment_coef3,
                moment3_ridge=float(op.pack.get("moment3_ridge", 1.0e-2)),
                moment3_relax=float(op.pack.get("moment3_relax", 0.5)),
                moment3_iters=int(op.pack.get("moment3_iters", 0)),
                implicit_projection=bool(op.pack.get("implicit_projection", False)),
            )

    elapsed = time.perf_counter() - t0
    return SparseOperator(
        label=op.label,
        family="learned_corrector" if op.is_corrector else "learned_base",
        pair=pair,
        src_index=si_t.detach().cpu().numpy().astype(np.int64),
        tgt_index=ti_t.detach().cpu().numpy().astype(np.int64),
        S=S_t.detach().cpu().numpy().astype(np.float64),
        M=M_t.detach().cpu().numpy().astype(np.float64),
        src_area=asrc_t.detach().cpu().numpy().astype(np.float64),
        tgt_area=atgt_t.detach().cpu().numpy().astype(np.float64),
        n_src=n_src,
        n_tgt=n_tgt,
        n_edges=int(si_t.numel()),
        elapsed_s=elapsed,
        graph_suffix=op.cfg.graph_suffix,
    )


def apply_operator(op: SparseOperator, src_field: np.ndarray) -> np.ndarray:
    y = np.zeros(op.n_tgt, dtype=np.float64)
    np.add.at(y, op.tgt_index, op.S * src_field[op.src_index])
    return y


def analytic_src_truth(name: str, src_xyz: np.ndarray, tgt_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if name.startswith("Y_"):
        _, l_s, m_s = name.split("_")
        l, m = int(l_s), int(m_s)
        src = _real_sph_unnorm(l, m, src_xyz)
        tgt = _real_sph_unnorm(l, m, tgt_xyz)
        nrm = float(np.sqrt(np.mean(src * src)))
        if nrm > 0.0:
            src = src / nrm
            tgt = tgt / nrm
        return src.astype(np.float64), tgt.astype(np.float64)
    return analytic_function(name, src_xyz), analytic_function(name, tgt_xyz)


def _analytic_callable(name: str):
    """Return f(xyz[N,3])->[N] for an analytic-function or Y_l_m name."""
    if name.startswith("Y_"):
        _, l_s, m_s = name.split("_")
        l, m = int(l_s), int(m_s)
        return lambda xyz: fv.real_sph_unnorm_fast(l, m, xyz)   # fast; == scipy up to global const
    return lambda xyz: analytic_function(name, xyz)


def analytic_src_truth_cellavg(name, qsrc, qtgt):
    """Cell-average (finite-volume) analog of analytic_src_truth: source field and target
    truth are per-cell AVERAGES over the cell polygons (quadrature), not point values at
    centers.  Same source-RMS normalization for Y_ fields (scale-invariant for the metric)."""
    fn = _analytic_callable(name)
    src = fv.grid_cell_average(fn, qsrc)
    tgt = fv.grid_cell_average(fn, qtgt)
    if name.startswith("Y_"):
        nrm = float(np.sqrt(np.mean(src * src)))
        if nrm > 0.0:
            src = src / nrm
            tgt = tgt / nrm
    return src.astype(np.float64), tgt.astype(np.float64)


def cellavg_harmonic_shells(pair, degrees, modes_per_degree, seed, qsrc, qtgt):
    """Cell-average analog of build_harmonic_fields_with_truth: SAME (l,m) mode selection
    (same rng seed + choose_m_values) but per-cell averages, so point/cellavg shells cover
    identical modes.  Returns (src_fields[nf,n_src], tgt_fields[nf,n_tgt]) float64."""
    rng = np.random.default_rng(seed + _stable_pair_seed(pair) % 1000000)
    sfs, tfs = [], []
    for l in degrees:
        for m in choose_m_values(l, modes_per_degree, rng):
            fn = (lambda xyz, l=l, m=m: fv.real_sph_unnorm_fast(l, m, xyz))
            ys = fv.grid_cell_average(fn, qsrc)
            yt = fv.grid_cell_average(fn, qtgt)
            nrm = float(np.sqrt(np.mean(ys * ys)))
            if nrm > 0.0:
                ys = ys / nrm
                yt = yt / nrm
            sfs.append(ys)
            tfs.append(yt)
    return np.stack(sfs, axis=0).astype(np.float64), np.stack(tfs, axis=0).astype(np.float64)


def read_flat_field(path: Path, field: str) -> np.ndarray:
    ds = xr.open_dataset(path)
    try:
        if field not in ds:
            raise KeyError(f"{field} not found in {path}")
        return np.asarray(ds[field].values, dtype=np.float64).reshape(-1)
    finally:
        ds.close()


def load_real_field_pair(cfg, pair: str, field: str, n_src: int, n_tgt: int):
    src_path, tgt_path = cfg.source_target_files(pair)
    if not src_path.exists() or not tgt_path.exists():
        return None, f"missing source/target file for {field}: {src_path.exists()}/{tgt_path.exists()}"
    try:
        src = read_flat_field(src_path, field)
        tgt = read_flat_field(tgt_path, field)
    except Exception as e:
        return None, str(e)
    if src.size != n_src or tgt.size != n_tgt:
        return None, f"field sizes {src.size}->{tgt.size} do not match operator sizes {n_src}->{n_tgt}"
    return (src, tgt), ""


def make_degree_shells(bands: list[int]) -> list[list[int]]:
    shells = []
    lo = 1
    for hi in bands:
        shells.append(list(range(lo, int(hi) + 1)))
        lo = int(hi) + 1
    return shells


def field_error_stats(
    op: SparseOperator,
    src_fields: np.ndarray,
    tgt_fields: np.ndarray,
) -> tuple[float, float, float]:
    errs = []
    for i in range(src_fields.shape[0]):
        pred = apply_operator(op, src_fields[i])
        errs.append(area_rel_l2(pred, tgt_fields[i], op.tgt_area))
    arr = np.asarray(errs, dtype=np.float64)
    return float(arr.mean()), float(np.median(arr)), float(arr.max())


def area_rmse(values: np.ndarray, area: np.ndarray) -> float:
    return float(np.sqrt(np.sum(area * values * values) / max(float(np.sum(area)), 1.0e-30)))


def tangent_basis(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a stable east/north-like tangent basis for every unit-sphere point."""
    p = np.asarray(xyz, dtype=np.float64)
    p = p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1.0e-30)
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    east = np.cross(z[None, :], p)
    bad = np.linalg.norm(east, axis=1) < 1.0e-10
    if np.any(bad):
        east[bad] = np.cross(x[None, :], p[bad])
    east = east / np.maximum(np.linalg.norm(east, axis=1, keepdims=True), 1.0e-30)
    north = np.cross(p, east)
    north = north / np.maximum(np.linalg.norm(north, axis=1, keepdims=True), 1.0e-30)
    return east, north


def add_moment_geometry_rows(
    rows: list[dict],
    *,
    op: SparseOperator,
    split: str,
    src_xyz: np.ndarray,
    tgt_xyz: np.ndarray,
):
    """Moment-oriented geometry diagnostics.

    Cartesian reproduction rows are field-reproduction errors for first- and
    second-degree coordinate polynomials.  The local tangent rows are more
    geometric: for each target cell, a linearly exact point-value remap would
    have sum_j S_ij * (source_j - target_i)_tangent ~= 0.  The second local
    moments are the analogous quadratic cancellation diagnostic.  These are not
    hard physical laws for finite-volume cell averages, but they are a useful
    signed-stencil sanity check and expose whether high-order behavior is coming
    from plausible geometry.
    """
    if src_xyz.shape[0] != op.n_src or tgt_xyz.shape[0] != op.n_tgt:
        raise ValueError(
            f"moment geometry size mismatch for {op.label} {op.pair}: "
            f"xyz {src_xyz.shape[0]}->{tgt_xyz.shape[0]} vs op {op.n_src}->{op.n_tgt}"
        )

    cart_basis = [
        ("cartesian", 1, "x", src_xyz[:, 0], tgt_xyz[:, 0]),
        ("cartesian", 1, "y", src_xyz[:, 1], tgt_xyz[:, 1]),
        ("cartesian", 1, "z", src_xyz[:, 2], tgt_xyz[:, 2]),
        ("cartesian", 2, "xx", src_xyz[:, 0] * src_xyz[:, 0], tgt_xyz[:, 0] * tgt_xyz[:, 0]),
        ("cartesian", 2, "xy", src_xyz[:, 0] * src_xyz[:, 1], tgt_xyz[:, 0] * tgt_xyz[:, 1]),
        ("cartesian", 2, "xz", src_xyz[:, 0] * src_xyz[:, 2], tgt_xyz[:, 0] * tgt_xyz[:, 2]),
        ("cartesian", 2, "yy", src_xyz[:, 1] * src_xyz[:, 1], tgt_xyz[:, 1] * tgt_xyz[:, 1]),
        ("cartesian", 2, "yz", src_xyz[:, 1] * src_xyz[:, 2], tgt_xyz[:, 1] * tgt_xyz[:, 2]),
        ("cartesian", 2, "zz", src_xyz[:, 2] * src_xyz[:, 2], tgt_xyz[:, 2] * tgt_xyz[:, 2]),
    ]

    for family, order, basis, src_field, truth in cart_basis:
        pred = apply_operator(op, src_field)
        err = pred - truth
        rows.append(
            {
                "pair": op.pair,
                "split": split,
                "operator": op.label,
                "family": op.family,
                "graph_suffix": op.graph_suffix,
                "moment_family": family,
                "order": order,
                "basis": basis,
                "area_rel_l2": area_rel_l2(pred, truth, op.tgt_area),
                "area_rmse": area_rmse(err, op.tgt_area),
                "normalized_rms": np.nan,
                "max_abs": float(np.max(np.abs(err))),
                "target": "analytic_coordinate_value",
            }
        )

    row_sum = scatter_numpy(op.n_tgt, op.tgt_index, op.S)
    row_err = row_sum - 1.0
    rows.append(
        {
            "pair": op.pair,
            "split": split,
            "operator": op.label,
            "family": op.family,
            "graph_suffix": op.graph_suffix,
            "moment_family": "local_tangent",
            "order": 0,
            "basis": "row_sum_minus_1",
            "area_rel_l2": np.nan,
            "area_rmse": area_rmse(row_err, op.tgt_area),
            "normalized_rms": area_rmse(row_err, op.tgt_area),
            "max_abs": float(np.max(np.abs(row_err))),
            "target": "0",
        }
    )

    east, north = tangent_basis(tgt_xyz)
    d = src_xyz[op.src_index] - tgt_xyz[op.tgt_index]
    u = np.sum(d * east[op.tgt_index], axis=1)
    v = np.sum(d * north[op.tgt_index], axis=1)
    h = np.sqrt(np.maximum(op.tgt_area, 1.0e-30))

    tangent_moments = [
        (1, "east", scatter_numpy(op.n_tgt, op.tgt_index, op.S * u), h),
        (1, "north", scatter_numpy(op.n_tgt, op.tgt_index, op.S * v), h),
        (2, "east2", scatter_numpy(op.n_tgt, op.tgt_index, op.S * u * u), h * h),
        (2, "east_north", scatter_numpy(op.n_tgt, op.tgt_index, op.S * u * v), h * h),
        (2, "north2", scatter_numpy(op.n_tgt, op.tgt_index, op.S * v * v), h * h),
    ]
    for order, basis, moment, denom_scale in tangent_moments:
        denom = max(float(np.sum(op.tgt_area * denom_scale * denom_scale)), 1.0e-30)
        normalized = float(np.sqrt(np.sum(op.tgt_area * moment * moment) / denom))
        rows.append(
            {
                "pair": op.pair,
                "split": split,
                "operator": op.label,
                "family": op.family,
                "graph_suffix": op.graph_suffix,
                "moment_family": "local_tangent",
                "order": order,
                "basis": basis,
                "area_rel_l2": np.nan,
                "area_rmse": area_rmse(moment, op.tgt_area),
                "normalized_rms": normalized,
                "max_abs": float(np.max(np.abs(moment))),
                "target": "0",
            }
        )


def pair_split(pair: str, train: set[str], guardrail: set[str], holdout: set[str]) -> str:
    labels = []
    if pair in train:
        labels.append("train")
    if pair in guardrail:
        labels.append("guardrail")
    if pair in holdout:
        labels.append("holdout")
    return "+".join(labels) if labels else "audit"


def add_field_metric(
    rows: list[dict],
    *,
    op: SparseOperator,
    split: str,
    category: str,
    function: str,
    src_field: np.ndarray,
    truth: np.ndarray,
):
    pred = apply_operator(op, src_field)
    src_int = float(np.sum(op.src_area * src_field))
    pred_int = float(np.sum(op.tgt_area * pred))
    truth_int = float(np.sum(op.tgt_area * truth))
    rows.append(
        {
            "pair": op.pair,
            "split": split,
            "operator": op.label,
            "family": op.family,
            "graph_suffix": op.graph_suffix,
            "category": category,
            "function": function,
            "n_src": op.n_src,
            "n_tgt": op.n_tgt,
            "n_edges": op.n_edges,
            "area_rel_l2": area_rel_l2(pred, truth, op.tgt_area),
            "rel_l2": rel_l2(pred, truth),
            "linf": float(np.max(np.abs(pred - truth))),
            "source_integral": src_int,
            "target_pred_integral": pred_int,
            "target_truth_integral": truth_int,
            "source_integral_abs": abs(src_int),
            "target_truth_integral_abs": abs(truth_int),
            "absolute_conservation_error_vs_source": abs(pred_int - src_int),
            "absolute_integral_error_vs_truth": abs(pred_int - truth_int),
            "relative_conservation_error_vs_source": rel_integral_error(pred_int, src_int),
            "relative_integral_error_vs_truth": rel_integral_error(pred_int, truth_int),
        }
    )


def markdown_table(df: pd.DataFrame, float_fmt: str = ".4e") -> str:
    if df.empty:
        return "_No rows._\n"
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(format(v, float_fmt))
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def write_summary(
    out_dir: Path,
    *,
    args,
    field_df: pd.DataFrame,
    shell_df: pd.DataFrame,
    moment_df: pd.DataFrame,
    cons_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
    skips: list[dict],
):
    lines = []
    lines.append("# RemapGNN operator audit\n")
    lines.append(f"- config: `{args.config}`")
    lines.append(f"- pairs: `{', '.join(args.pairs)}`")
    lines.append(f"- packs: `{', '.join(args.packs or [])}`")
    lines.append(f"- shell bands: `{', '.join(map(str, args.shell_bands))}`")
    if args.base_n_cg is not None or args.corrector_n_cg is not None:
        lines.append(f"- projection overrides: base_n_cg=`{args.base_n_cg}`, corrector_n_cg=`{args.corrector_n_cg}`")
    lines.append(f"- projection dtype: `{args.projection_dtype}`")
    lines.append(f"- projection eps_rel: `{args.projection_eps_rel:.3e}`")
    lines.append("")

    if not field_df.empty:
        field_summary = (
            field_df.groupby(["operator", "category"], as_index=False)
            .agg(
                mean_area_rel_l2=("area_rel_l2", "mean"),
                worst_area_rel_l2=("area_rel_l2", "max"),
                mean_abs_cons_field_err=("absolute_conservation_error_vs_source", "mean"),
                rows=("area_rel_l2", "size"),
            )
            .sort_values(["category", "mean_area_rel_l2"])
        )
        lines.append("## Field-error summary\n")
        lines.append(markdown_table(field_summary))

        worst_fields = (
            field_df.sort_values("area_rel_l2", ascending=False)
            .head(12)[["operator", "pair", "split", "category", "function", "area_rel_l2", "absolute_conservation_error_vs_source"]]
        )
        lines.append("## Worst field cases\n")
        lines.append(markdown_table(worst_fields))

    if not shell_df.empty:
        shell_summary = (
            shell_df.groupby(["operator"], as_index=False)
            .agg(
                mean_shell_rel_l2=("mean_area_rel_l2", "mean"),
                worst_shell_rel_l2=("mean_area_rel_l2", "max"),
                worst_single_mode_rel_l2=("max_area_rel_l2", "max"),
                rows=("mean_area_rel_l2", "size"),
            )
            .sort_values("mean_shell_rel_l2")
        )
        lines.append("## Spectral-shell summary\n")
        lines.append(markdown_table(shell_summary))

        worst_shells = (
            shell_df.sort_values("mean_area_rel_l2", ascending=False)
            .head(12)[["operator", "pair", "split", "shell_label", "lmin", "lmax", "mean_area_rel_l2", "max_area_rel_l2"]]
        )
        lines.append("## Worst spectral shells\n")
        lines.append(markdown_table(worst_shells))

    if not moment_df.empty:
        cart = moment_df[moment_df["moment_family"] == "cartesian"]
        if not cart.empty:
            cart_summary = (
                cart.groupby(["operator", "order"], as_index=False)
                .agg(
                    mean_area_rel_l2=("area_rel_l2", "mean"),
                    worst_area_rel_l2=("area_rel_l2", "max"),
                    mean_area_rmse=("area_rmse", "mean"),
                    rows=("area_rel_l2", "size"),
                )
                .sort_values(["order", "mean_area_rel_l2"])
            )
            lines.append("## Cartesian moment reproduction\n")
            lines.append(markdown_table(cart_summary))

        tangent = moment_df[moment_df["moment_family"] == "local_tangent"]
        if not tangent.empty:
            tangent_summary = (
                tangent.groupby(["operator", "order"], as_index=False)
                .agg(
                    mean_normalized_rms=("normalized_rms", "mean"),
                    worst_normalized_rms=("normalized_rms", "max"),
                    mean_area_rmse=("area_rmse", "mean"),
                    worst_max_abs=("max_abs", "max"),
                    rows=("normalized_rms", "size"),
                )
                .sort_values(["order", "mean_normalized_rms"])
            )
            lines.append("## Local tangent moment residuals\n")
            lines.append(markdown_table(tangent_summary))

        worst_moments = (
            moment_df.assign(sort_metric=moment_df["area_rel_l2"].fillna(moment_df["normalized_rms"]))
            .sort_values("sort_metric", ascending=False)
            .head(12)[["operator", "pair", "split", "moment_family", "order", "basis", "area_rel_l2", "normalized_rms", "max_abs"]]
        )
        lines.append("## Worst moment/geometry cases\n")
        lines.append(markdown_table(worst_moments))

    if not cons_df.empty:
        cons_summary = (
            cons_df.groupby(["operator"], as_index=False)
            .agg(
                max_conservation_resid=("conservation_resid", "max"),
                max_consistency_resid=("consistency_resid", "max"),
                mean_edges=("n_edges", "mean"),
                max_edges=("n_edges", "max"),
            )
            .sort_values("max_conservation_resid")
        )
        lines.append("## Conservation / consistency summary\n")
        lines.append(markdown_table(cons_summary))

    if not runtime_df.empty:
        runtime_summary = (
            runtime_df.groupby(["operator"], as_index=False)
            .agg(mean_operator_build_s=("operator_build_s", "mean"), max_operator_build_s=("operator_build_s", "max"))
            .sort_values("mean_operator_build_s")
        )
        lines.append("## Runtime summary\n")
        lines.append(markdown_table(runtime_summary))

    if skips:
        lines.append("## Skipped real-field cases\n")
        lines.append(markdown_table(pd.DataFrame(skips)))

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="default audit config / map directory")
    ap.add_argument("--packs", nargs="*", default=[], help="learned operators as label=pack.pt[@config.json]")
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--train-pairs", nargs="*", default=[])
    ap.add_argument("--guardrail-pairs", nargs="*", default=[])
    ap.add_argument("--holdout-pairs", nargs="*", default=[])
    ap.add_argument("--functions", nargs="+", default=DEFAULT_FUNCTIONS)
    ap.add_argument("--real-fields", nargs="+", default=DEFAULT_REAL_FIELDS)
    ap.add_argument("--shell-bands", nargs="+", type=int, default=[8, 16, 24, 32, 40, 48])
    ap.add_argument("--modes-per-degree", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-tempest", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--include-esmf", action=argparse.BooleanOptionalAction, default=True,
                    help="load ESMF baseline maps map_<pair>_esmf_{bilinear,conserve,conserve2nd}.nc when present")
    ap.add_argument("--base-n-cg", type=int, default=None,
                    help="override projection CG iterations for learned base packs")
    ap.add_argument("--corrector-n-cg", type=int, default=None,
                    help="override projection CG iterations for learned corrector packs")
    ap.add_argument("--projection-dtype", choices=["float32", "float64"], default="float32",
                    help="dtype used inside learned-operator projection solves")
    ap.add_argument("--projection-eps-rel", type=float, default=1e-9,
                    help="relative ridge used inside learned-operator projection solves")
    ap.add_argument("--truth-mode", choices=["point", "cellavg"], default="point",
                    help="analytic/spectral truth: point value at cell centers (legacy) or "
                         "finite-volume cell averages via per-cell quadrature (correct for FV remap)")
    ap.add_argument("--quad-m", type=int, default=8,
                    help="barycentric subdivisions per fan triangle for cell-average quadrature")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    default_cfg_path = Path(args.config)
    default_cfg = load_config(default_cfg_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    projection_dtype = parse_projection_dtype(args.projection_dtype)
    print("device:", device)
    print("projection_dtype:", args.projection_dtype)
    print("projection_eps_rel:", args.projection_eps_rel)
    print("truth_mode:", args.truth_mode, "(quad_m=%d)" % args.quad_m if args.truth_mode == "cellavg" else "")
    print("out_dir:", out_dir)

    learned_ops = [
        build_learned_operator(parse_pack_spec(spec), default_cfg_path, device)
        for spec in args.packs
    ]

    train = set(args.train_pairs)
    guardrail = set(args.guardrail_pairs)
    holdout = set(args.holdout_pairs)

    field_rows: list[dict] = []
    shell_rows: list[dict] = []
    moment_rows: list[dict] = []
    cons_rows: list[dict] = []
    runtime_rows: list[dict] = []
    skips: list[dict] = []

    shells = make_degree_shells(args.shell_bands)

    for pair in args.pairs:
        split = pair_split(pair, train, guardrail, holdout)
        print(f"\n=== AUDIT PAIR {pair} split={split} ===")

        # Use the default config for geometry/truth fields.  If its graph file is
        # missing, fall back to the first learned operator config that has it.
        geom_cfg = default_cfg
        if not geom_cfg.edge_path(pair).exists():
            for lop in learned_ops:
                if lop.cfg.edge_path(pair).exists():
                    geom_cfg = lop.cfg
                    break
        n_src_geom = n_tgt_geom = None
        src_xyz = tgt_xyz = None
        try:
            edge_df = pd.read_parquet(geom_cfg.edge_path(pair), columns=["source_index", "target_index"])
            n_src_geom = int(edge_df["source_index"].max()) + 1
            n_tgt_geom = int(edge_df["target_index"].max()) + 1
            src_xyz = read_source_xyz_from_edges(geom_cfg.edge_path(pair), n_src_geom)
            tgt_xyz = read_target_xyz_from_edges(geom_cfg.edge_path(pair), n_tgt_geom)
        except Exception as e:
            print(f"  geometry warning: {e}")

        # Finite-volume cell-average truth: per-cell quadrature over the source/target cell
        # polygons (corners from the conserve map file), aligned to the edge-dataset ordering.
        qsrc = qtgt = None
        if args.truth_mode == "cellavg" and src_xyz is not None:
            try:
                map_path = default_cfg.maps_dir / f"map_{pair}_conserve.nc"
                qsrc = fv.grid_quadrature(map_path, "a", m=args.quad_m, expected_centers=src_xyz)
                qtgt = fv.grid_quadrature(map_path, "b", m=args.quad_m, expected_centers=tgt_xyz)
                print(f"  cellavg quadrature ready (m={args.quad_m})")
            except Exception as e:
                print(f"  cellavg geometry warning ({pair}): {e} -> falling back to point truth")
                qsrc = qtgt = None

        sparse_ops: list[SparseOperator] = []
        if args.include_tempest:
            for order in ["np1", "np2"]:
                try:
                    sparse_ops.append(build_tempest_operator(default_cfg, pair, order))
                except Exception as e:
                    print(f"  skip {order}: {e}")
                    skips.append({"pair": pair, "operator": order, "field": "*operator*", "reason": str(e)})

        if args.include_esmf:
            for method in ["bilinear", "conserve", "conserve2nd"]:
                mp = default_cfg.maps_dir / f"map_{pair}_esmf_{method}.nc"
                if not mp.exists():
                    continue
                try:
                    sparse_ops.append(build_esmf_operator(default_cfg, pair, method))
                except Exception as e:
                    print(f"  skip esmf_{method}: {e}")
                    skips.append({"pair": pair, "operator": f"esmf_{method}", "field": "*operator*", "reason": str(e)})

        for lop in learned_ops:
            try:
                sparse_ops.append(
                    build_learned_sparse_operator(
                        lop,
                        pair,
                        device,
                        base_n_cg=args.base_n_cg,
                        corrector_n_cg=args.corrector_n_cg,
                        projection_dtype=projection_dtype,
                        projection_eps_rel=float(args.projection_eps_rel),
                    )
                )
            except Exception as e:
                print(f"  skip learned {lop.label}: {e}")
                skips.append({"pair": pair, "operator": lop.label, "field": "*operator*", "reason": str(e)})
            finally:
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        for op in sparse_ops:
            cons, row = residuals_from_mass(op.M, op.src_index, op.tgt_index, op.src_area, op.tgt_area)
            cons_rows.append(
                {
                    "pair": pair,
                    "split": split,
                    "operator": op.label,
                    "family": op.family,
                    "graph_suffix": op.graph_suffix,
                    "n_src": op.n_src,
                    "n_tgt": op.n_tgt,
                    "n_edges": op.n_edges,
                    "conservation_resid": cons,
                    "consistency_resid": row,
                }
            )
            runtime_rows.append(
                {
                    "pair": pair,
                    "split": split,
                    "operator": op.label,
                    "family": op.family,
                    "graph_suffix": op.graph_suffix,
                    "n_edges": op.n_edges,
                    "operator_build_s": op.elapsed_s,
                }
            )

        if src_xyz is None or tgt_xyz is None:
            print("  no geometry available for field/spectral audit")
            continue

        for op in sparse_ops:
            try:
                add_moment_geometry_rows(
                    moment_rows,
                    op=op,
                    split=split,
                    src_xyz=src_xyz,
                    tgt_xyz=tgt_xyz,
                )
            except Exception as e:
                print(f"  skip moment geometry {op.label}: {e}")
                skips.append({"pair": pair, "operator": op.label, "field": "*moment_geometry*", "reason": str(e)})

        for fn in args.functions:
            try:
                if args.truth_mode == "cellavg" and qsrc is not None:
                    src_field, truth = analytic_src_truth_cellavg(fn, qsrc, qtgt)
                else:
                    src_field, truth = analytic_src_truth(fn, src_xyz, tgt_xyz)
            except Exception as e:
                print(f"  skip analytic {fn}: {e}")
                skips.append({"pair": pair, "operator": "*all*", "field": fn, "reason": str(e)})
                continue
            for op in sparse_ops:
                if src_field.size != op.n_src or truth.size != op.n_tgt:
                    skips.append({
                        "pair": pair,
                        "operator": op.label,
                        "field": fn,
                        "reason": f"geometry sizes {src_field.size}->{truth.size} != op sizes {op.n_src}->{op.n_tgt}",
                    })
                    continue
                add_field_metric(
                    field_rows,
                    op=op,
                    split=split,
                    category="analytic",
                    function=fn,
                    src_field=src_field,
                    truth=truth,
                )

        for field in args.real_fields:
            loaded, reason = load_real_field_pair(geom_cfg, pair, field, n_src_geom, n_tgt_geom)
            if loaded is None:
                skips.append({"pair": pair, "operator": "*all*", "field": field, "reason": reason})
                continue
            src_field, truth = loaded
            for op in sparse_ops:
                if src_field.size != op.n_src or truth.size != op.n_tgt:
                    skips.append({
                        "pair": pair,
                        "operator": op.label,
                        "field": field,
                        "reason": f"real-field sizes {src_field.size}->{truth.size} != op sizes {op.n_src}->{op.n_tgt}",
                    })
                    continue
                add_field_metric(
                    field_rows,
                    op=op,
                    split=split,
                    category="real",
                    function=field,
                    src_field=src_field,
                    truth=truth,
                )

        for shell_i, degs in enumerate(shells):
            if args.truth_mode == "cellavg" and qsrc is not None:
                sf, tf = cellavg_harmonic_shells(
                    pair, degs, args.modes_per_degree, args.seed + 97 * shell_i, qsrc, qtgt)
            else:
                sf_t, tf_t = build_harmonic_fields_with_truth(
                    geom_cfg,
                    pair,
                    n_src_geom,
                    n_tgt_geom,
                    degs,
                    args.modes_per_degree,
                    args.seed + 97 * shell_i,
                )
                sf = sf_t.numpy().astype(np.float64)
                tf = tf_t.numpy().astype(np.float64)
            label = f"l{degs[0]}-{degs[-1]}"
            for op in sparse_ops:
                if sf.shape[1] != op.n_src or tf.shape[1] != op.n_tgt:
                    skips.append({
                        "pair": pair,
                        "operator": op.label,
                        "field": label,
                        "reason": f"shell sizes {sf.shape[1]}->{tf.shape[1]} != op sizes {op.n_src}->{op.n_tgt}",
                    })
                    continue
                mean_e, median_e, max_e = field_error_stats(op, sf, tf)
                shell_rows.append(
                    {
                        "pair": pair,
                        "split": split,
                        "operator": op.label,
                        "family": op.family,
                        "graph_suffix": op.graph_suffix,
                        "shell": shell_i + 1,
                        "shell_label": label,
                        "lmin": degs[0],
                        "lmax": degs[-1],
                        "n_fields": sf.shape[0],
                        "mean_area_rel_l2": mean_e,
                        "median_area_rel_l2": median_e,
                        "max_area_rel_l2": max_e,
                    }
                )

    field_df = pd.DataFrame(field_rows)
    shell_df = pd.DataFrame(shell_rows)
    moment_df = pd.DataFrame(moment_rows)
    cons_df = pd.DataFrame(cons_rows)
    runtime_df = pd.DataFrame(runtime_rows)
    skips_df = pd.DataFrame(skips)

    field_df.to_csv(out_dir / "field_metrics.csv", index=False)
    shell_df.to_csv(out_dir / "spectral_shells.csv", index=False)
    moment_df.to_csv(out_dir / "moment_geometry.csv", index=False)
    cons_df.to_csv(out_dir / "conservation.csv", index=False)
    runtime_df.to_csv(out_dir / "runtime.csv", index=False)
    skips_df.to_csv(out_dir / "skips.csv", index=False)
    write_summary(
        out_dir,
        args=args,
        field_df=field_df,
        shell_df=shell_df,
        moment_df=moment_df,
        cons_df=cons_df,
        runtime_df=runtime_df,
        skips=skips,
    )

    print(f"\nwrote {out_dir / 'field_metrics.csv'}")
    print(f"wrote {out_dir / 'spectral_shells.csv'}")
    print(f"wrote {out_dir / 'moment_geometry.csv'}")
    print(f"wrote {out_dir / 'conservation.csv'}")
    print(f"wrote {out_dir / 'runtime.csv'}")
    print(f"wrote {out_dir / 'skips.csv'}")
    print(f"wrote {out_dir / 'summary.md'}")
    print("AUDIT_REMAP_OPERATOR_DONE")


if __name__ == "__main__":
    main()
