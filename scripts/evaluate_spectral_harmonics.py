from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from remapgnn.config import load_config
from remapgnn.data import load_pair_tensors
from remapgnn.models import build_model
from remapgnn.sinkhorn import sparse_sinkhorn_balance, sparse_operator_weights


def parse_degrees(s: str) -> list[int]:
    return [int(x) for x in s.replace(",", " ").split() if x.strip()]


def choose_m_values(l: int, modes_per_degree: int, rng: np.random.Generator) -> list[int]:
    all_m = list(range(-l, l + 1))
    if len(all_m) <= modes_per_degree:
        return all_m

    keep = {-l, 0, l}
    remaining = [m for m in all_m if m not in keep]
    n_extra = max(0, modes_per_degree - len(keep))
    if n_extra > 0:
        keep.update(rng.choice(remaining, size=n_extra, replace=False).tolist())
    return sorted(keep)


def get_sph_harm_func():
    try:
        from scipy.special import sph_harm_y

        def sph(l, m, theta, phi):
            # scipy newer API: sph_harm_y(n, m, theta, phi)
            return sph_harm_y(l, m, theta, phi)

        return sph
    except Exception:
        pass

    try:
        from scipy.special import sph_harm

        def sph(l, m, theta, phi):
            # scipy older API: sph_harm(m, n, phi, theta)
            return sph_harm(m, l, phi, theta)

        return sph
    except Exception as e:
        raise RuntimeError(
            "Could not import scipy.special.sph_harm or sph_harm_y. "
            "Install scipy or use an environment that has scipy."
        ) from e


SPH = get_sph_harm_func()


def xyz_to_angles(xyz: np.ndarray):
    xyz = np.asarray(xyz, dtype=np.float64)
    r = np.linalg.norm(xyz, axis=1)
    z = np.clip(xyz[:, 2] / np.maximum(r, 1.0e-30), -1.0, 1.0)

    theta = np.arccos(z)  # colatitude, [0, pi]
    phi = np.mod(np.arctan2(xyz[:, 1], xyz[:, 0]), 2.0 * np.pi)  # longitude, [0, 2pi)
    return theta, phi


def real_spherical_harmonic(l: int, m: int, xyz: np.ndarray) -> np.ndarray:
    theta, phi = xyz_to_angles(xyz)

    if m == 0:
        y = SPH(l, 0, theta, phi).real
    elif m > 0:
        y = np.sqrt(2.0) * ((-1.0) ** m) * SPH(l, m, theta, phi).real
    else:
        mp = abs(m)
        y = np.sqrt(2.0) * ((-1.0) ** mp) * SPH(l, mp, theta, phi).imag

    y = np.asarray(y, dtype=np.float64)

    # Normalize so degrees/modes are comparable. This does not affect relative errors.
    norm = np.sqrt(np.mean(y * y))
    if norm > 0:
        y = y / norm
    return y


def random_band_field(lmin: int, lmax: int, xyz: np.ndarray, n_terms: int, rng: np.random.Generator) -> np.ndarray:
    field = np.zeros(xyz.shape[0], dtype=np.float64)

    degrees = rng.integers(lmin, lmax + 1, size=n_terms)
    for l in degrees:
        m = int(rng.integers(-l, l + 1))
        coeff = rng.normal()
        field += coeff * real_spherical_harmonic(int(l), m, xyz)

    norm = np.sqrt(np.mean(field * field))
    if norm > 0:
        field = field / norm
    return field


def apply_sparse(row: np.ndarray, col: np.ndarray, weight: np.ndarray, x_src: np.ndarray, n_tgt: int) -> np.ndarray:
    y = np.zeros(n_tgt, dtype=np.float64)
    np.add.at(y, row, weight * x_src[col])
    return y


def rel_l2(a: np.ndarray, b: np.ndarray, eps: float = 1.0e-30) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), eps))


def rel_l2_area(a: np.ndarray, b: np.ndarray, area: np.ndarray, eps: float = 1.0e-30) -> float:
    num = np.sum(area * (a - b) ** 2)
    den = np.sum(area * b ** 2)
    return float(np.sqrt(num / max(den, eps)))


def read_tempest_map(map_path: Path):
    ds = xr.open_dataset(map_path)

    row_name = "row" if "row" in ds.variables else "dst_address"
    col_name = "col" if "col" in ds.variables else "src_address"
    w_name = "S" if "S" in ds.variables else "remap_matrix"

    row = np.asarray(ds[row_name].values).astype(np.int64).ravel()
    col = np.asarray(ds[col_name].values).astype(np.int64).ravel()
    weight = np.asarray(ds[w_name].values).astype(np.float64)

    if weight.ndim == 2:
        # Common SCRIP shape can be (num_links, num_wgts). Conservative maps usually use first weight.
        weight = weight[:, 0]
    weight = weight.ravel()

    # Tempest/SCRIP usually stores 1-based addresses.
    if row.min() == 1:
        row = row - 1
    if col.min() == 1:
        col = col - 1

    ds.close()
    return row, col, weight


def read_source_xyz_from_edges(edge_path: Path, n_src: int) -> np.ndarray:
    df = pd.read_parquet(edge_path, columns=["source_index", "src_x", "src_y", "src_z"])
    g = df.groupby("source_index", sort=False)[["src_x", "src_y", "src_z"]].first()

    xyz = np.full((n_src, 3), np.nan, dtype=np.float64)
    idx = g.index.to_numpy(dtype=np.int64)
    xyz[idx] = g.to_numpy(dtype=np.float64)

    if np.isnan(xyz).any():
        missing = np.where(np.isnan(xyz[:, 0]))[0][:10]
        raise RuntimeError(f"Missing source coordinates for {len(missing)}+ source cells, first few: {missing}")

    return xyz


def as_int(x):
    if hasattr(x, "item"):
        return int(x.item())
    return int(x)


def load_model_and_pack(cfg, device):
    pack = torch.load(cfg.model_path, map_location=device)

    edge_features = pack["edge_features"]
    src_features = pack["src_node_features"]
    tgt_features = pack["tgt_node_features"]

    model = build_model(
        architecture=pack.get("architecture", cfg.architecture),
        src_dim=len(src_features),
        tgt_dim=len(tgt_features),
        edge_dim=len(edge_features),
        hidden=int(pack.get("hidden", 128)),
        decoder_chunk_size=int(pack.get("decoder_chunk_size", 10000)),
    ).to(device)

    state = pack.get("model_state_dict", pack.get("state_dict"))
    if state is None:
        raise KeyError("Checkpoint has neither model_state_dict nor state_dict")

    model.load_state_dict(state)
    model.eval()

    return model, pack


def model_q_from_output(out):
    if isinstance(out, dict):
        if "q" in out:
            return out["q"]
        edge_logit = out.get("edge_logit", out.get("logit"))
        raw_weight = out.get("raw_weight", out.get("positive_weight"))
    elif isinstance(out, (tuple, list)):
        if len(out) == 3:
            edge_logit, raw_weight, q = out
            return q
        if len(out) == 2:
            edge_logit, raw_weight = out
        else:
            raise RuntimeError(f"Unsupported model output tuple length: {len(out)}")
    else:
        raise RuntimeError(f"Unsupported model output type: {type(out)}")

    if edge_logit is None or raw_weight is None:
        raise RuntimeError("Could not extract edge_logit/logit and raw_weight/positive_weight from model output")

    return torch.sqrt(torch.sigmoid(edge_logit).clamp_min(1.0e-12)) * F.softplus(raw_weight)


def build_gnn_sparse_operator(model, cfg, pair: str, stats, device, balance_iters: int):
    batch = load_pair_tensors(cfg, pair, stats, device=device)

    n_src = as_int(batch["n_src"])
    n_tgt = as_int(batch["n_tgt"])

    with torch.no_grad():
        out = model(
            batch["src_node_attr"],
            batch["tgt_node_attr"],
            batch["edge_attr"],
            batch["src_index"],
            batch["tgt_index"],
            n_src,
            n_tgt,
        )
        q = model_q_from_output(out).float()

        M = sparse_sinkhorn_balance(
            q=q,
            src_index=batch["src_index"],
            tgt_index=batch["tgt_index"],
            area_src=batch["area_src"],
            area_tgt=batch["area_tgt"],
            n_src=n_src,
            n_tgt=n_tgt,
            n_iter=balance_iters,
        )

        S = sparse_operator_weights(
            M=M,
            tgt_index=batch["tgt_index"],
            area_tgt=batch["area_tgt"],
        )

    row = batch["tgt_index"].detach().cpu().numpy().astype(np.int64)
    col = batch["src_index"].detach().cpu().numpy().astype(np.int64)
    weight = S.detach().cpu().numpy().astype(np.float64)
    area_tgt = batch["area_tgt"].detach().cpu().numpy().astype(np.float64)

    del batch
    return row, col, weight, area_tgt, n_src, n_tgt


def evaluate_pair(cfg, model, pack, pair: str, degrees: list[int], modes_per_degree: int, band_terms: int,
                  rng: np.random.Generator, device, balance_iters: int):
    print(f"\nPAIR {pair}")
    print(f"  edge: {cfg.edge_path(pair)}")
    print(f"  map:  {cfg.map_path(pair)}")

    stats = pack["stats"]

    gnn_row, gnn_col, gnn_w, area_tgt, n_src, n_tgt = build_gnn_sparse_operator(
        model=model,
        cfg=cfg,
        pair=pair,
        stats=stats,
        device=device,
        balance_iters=balance_iters,
    )

    tmp_row, tmp_col, tmp_w = read_tempest_map(cfg.map_path(pair))
    src_xyz = read_source_xyz_from_edges(cfg.edge_path(pair), n_src=n_src)

    rows = []

    # Exact real harmonic modes.
    for l in degrees:
        m_values = choose_m_values(l, modes_per_degree, rng)
        print(f"  degree l={l}, modes={m_values}")

        for m in m_values:
            x = real_spherical_harmonic(l, m, src_xyz)

            y_tmp = apply_sparse(tmp_row, tmp_col, tmp_w, x, n_tgt)
            y_gnn = apply_sparse(gnn_row, gnn_col, gnn_w, x, n_tgt)

            rows.append({
                "pair": pair,
                "kind": "mode",
                "degree": l,
                "m": m,
                "band_lmin": l,
                "band_lmax": l,
                "rel_l2_vs_tempest": rel_l2(y_gnn, y_tmp),
                "area_rel_l2_vs_tempest": rel_l2_area(y_gnn, y_tmp, area_tgt),
                "tempest_norm": float(np.linalg.norm(y_tmp)),
                "gnn_norm": float(np.linalg.norm(y_gnn)),
                "mean_abs_diff": float(np.mean(np.abs(y_gnn - y_tmp))),
            })

    # Random band-limited fields, useful because real fields are combinations of modes.
    degree_sorted = sorted(set(degrees))
    for a, b in zip(degree_sorted[:-1], degree_sorted[1:]):
        lmin = a + 1
        lmax = b
        if lmin > lmax:
            continue

        for sample_id in range(3):
            x = random_band_field(lmin, lmax, src_xyz, n_terms=band_terms, rng=rng)

            y_tmp = apply_sparse(tmp_row, tmp_col, tmp_w, x, n_tgt)
            y_gnn = apply_sparse(gnn_row, gnn_col, gnn_w, x, n_tgt)

            rows.append({
                "pair": pair,
                "kind": "random_band",
                "degree": lmax,
                "m": sample_id,
                "band_lmin": lmin,
                "band_lmax": lmax,
                "rel_l2_vs_tempest": rel_l2(y_gnn, y_tmp),
                "area_rel_l2_vs_tempest": rel_l2_area(y_gnn, y_tmp, area_tgt),
                "tempest_norm": float(np.linalg.norm(y_tmp)),
                "gnn_norm": float(np.linalg.norm(y_gnn)),
                "mean_abs_diff": float(np.mean(np.abs(y_gnn - y_tmp))),
            })

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pair", default=None)
    parser.add_argument("--all-pairs", action="store_true")
    parser.add_argument("--degrees", default="0 1 2 4 8 12 16 24 32")
    parser.add_argument("--modes-per-degree", type=int, default=9)
    parser.add_argument("--band-terms", type=int, default=16)
    parser.add_argument("--balance-iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    degrees = parse_degrees(args.degrees)

    if args.all_pairs:
        pairs = cfg.pairs
    elif args.pair:
        pairs = [args.pair]
    else:
        raise SystemExit("Use --pair PAIR or --all-pairs")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rng = np.random.default_rng(args.seed)

    print(f"Config:       {args.config}")
    print(f"Run name:     {cfg.run_name}")
    print(f"Model tag:    {cfg.model_tag}")
    print(f"Architecture: {cfg.architecture}")
    print(f"Model path:   {cfg.model_path}")
    print(f"Device:       {device}")
    print(f"Degrees:      {degrees}")
    print(f"Balance iters:{args.balance_iters}")

    model, pack = load_model_and_pack(cfg, device)

    all_rows = []
    for pair in pairs:
        all_rows.extend(
            evaluate_pair(
                cfg=cfg,
                model=model,
                pack=pack,
                pair=pair,
                degrees=degrees,
                modes_per_degree=args.modes_per_degree,
                band_terms=args.band_terms,
                rng=rng,
                device=device,
                balance_iters=args.balance_iters,
            )
        )

    df = pd.DataFrame(all_rows)

    if args.out:
        out = Path(args.out)
    else:
        out = cfg.analysis_dir / f"{cfg.model_tag}_spectral_harmonics.csv"

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    summary = (
        df.groupby(["pair", "kind", "degree", "band_lmin", "band_lmax"], as_index=False)
          .agg(
              mean_rel_l2_vs_tempest=("rel_l2_vs_tempest", "mean"),
              max_rel_l2_vs_tempest=("rel_l2_vs_tempest", "max"),
              mean_area_rel_l2_vs_tempest=("area_rel_l2_vs_tempest", "mean"),
              max_area_rel_l2_vs_tempest=("area_rel_l2_vs_tempest", "max"),
              n_fields=("rel_l2_vs_tempest", "size"),
          )
    )

    summary_out = out.with_name(out.stem + "_summary.csv")
    summary.to_csv(summary_out, index=False)

    print()
    print("Spectral summary:")
    print(summary.to_string(index=False))
    print()
    print(f"Wrote {out}")
    print(f"Wrote {summary_out}")


if __name__ == "__main__":
    main()
