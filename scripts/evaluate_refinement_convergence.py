from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from remapgnn.config import load_config
from remapgnn.data import load_pair_tensors

TRUE_WEIGHT_CANDIDATES = [
    "S_true",
    "weight",
    "s_true",
    "target_weight",
    "true_weight",
    "remap_weight",
    "edge_weight",
    "S",
    "s",
]


def get_sph_harm_func():
    try:
        from scipy.special import sph_harm_y

        def sph(l, m, theta, phi):
            return sph_harm_y(l, m, theta, phi)

        return sph
    except Exception:
        pass

    try:
        from scipy.special import sph_harm

        def sph(l, m, theta, phi):
            return sph_harm(m, l, phi, theta)

        return sph
    except Exception as e:
        raise RuntimeError("Need scipy.special.sph_harm_y or scipy.special.sph_harm") from e


SPH = get_sph_harm_func()


def xyz_to_lon_lat(xyz: np.ndarray):
    xyz = np.asarray(xyz, dtype=np.float64)
    r = np.linalg.norm(xyz, axis=1)
    x = xyz[:, 0] / np.maximum(r, 1.0e-30)
    y = xyz[:, 1] / np.maximum(r, 1.0e-30)
    z = np.clip(xyz[:, 2] / np.maximum(r, 1.0e-30), -1.0, 1.0)

    lon = np.mod(np.arctan2(y, x), 2.0 * np.pi)
    lat = np.arcsin(z)
    theta = np.arccos(z)
    phi = lon
    return lon, lat, theta, phi


def real_spherical_harmonic(l: int, m: int, xyz: np.ndarray) -> np.ndarray:
    _, _, theta, phi = xyz_to_lon_lat(xyz)

    if m == 0:
        y = SPH(l, 0, theta, phi).real
    elif m > 0:
        y = np.sqrt(2.0) * ((-1.0) ** m) * SPH(l, m, theta, phi).real
    else:
        mp = abs(m)
        y = np.sqrt(2.0) * ((-1.0) ** mp) * SPH(l, mp, theta, phi).imag

    y = np.asarray(y, dtype=np.float64)
    nrm = np.sqrt(np.mean(y * y))
    if nrm > 0:
        y = y / nrm
    return y


def analytic_function(name: str, xyz: np.ndarray) -> np.ndarray:
    name = name.strip()

    if name in {"const", "constant", "1"}:
        return np.ones(xyz.shape[0], dtype=np.float64)

    if name == "x":
        return xyz[:, 0].astype(np.float64)
    if name == "y":
        return xyz[:, 1].astype(np.float64)
    if name == "z":
        return xyz[:, 2].astype(np.float64)

    lon, lat, _, _ = xyz_to_lon_lat(xyz)

    if name == "smooth1":
        return (
            1.0
            + 0.25 * xyz[:, 0]
            - 0.15 * xyz[:, 1]
            + 0.10 * xyz[:, 2]
            + 0.20 * np.sin(2.0 * lon) * np.cos(lat)
        ).astype(np.float64)

    if name == "smooth2":
        return (
            np.exp(0.5 * xyz[:, 0] - 0.25 * xyz[:, 1])
            + 0.10 * np.cos(3.0 * lon) * np.cos(lat) ** 2
        ).astype(np.float64)

    # Accept Y_8_0 or Y:8:0
    if name.startswith("Y_"):
        _, l, m = name.split("_")
        return real_spherical_harmonic(int(l), int(m), xyz)

    if name.startswith("Y:"):
        _, l, m = name.split(":")
        return real_spherical_harmonic(int(l), int(m), xyz)

    raise ValueError(f"Unknown analytic function spec: {name}")


def get_true_weight_column(df: pd.DataFrame) -> str:
    for c in TRUE_WEIGHT_CANDIDATES:
        if c in df.columns:
            return c
    raise KeyError(f"Could not find true remap weight column. Available columns: {list(df.columns)}")


def unique_indexed_array(df: pd.DataFrame, index_col: str, cols: list[str], n: int) -> np.ndarray:
    g = df.groupby(index_col, sort=False)[cols].first()
    out = np.full((n, len(cols)), np.nan, dtype=np.float64)
    idx = g.index.to_numpy(dtype=np.int64)
    out[idx] = g.to_numpy(dtype=np.float64)

    if np.isnan(out).any():
        missing = np.where(np.isnan(out[:, 0]))[0][:10]
        raise RuntimeError(f"Missing values for {index_col}, first missing: {missing}")
    return out


def scatter_numpy(n: int, index: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.zeros(n, dtype=np.float64)
    np.add.at(out, index, values)
    return out


def area_rel_l2(pred: np.ndarray, ref: np.ndarray, area: np.ndarray, eps: float = 1.0e-30) -> float:
    diff = pred - ref
    num = np.sum(area * diff * diff)
    den = max(np.sum(area * ref * ref), eps)
    return float(np.sqrt(num / den))


def unweighted_rel_l2(pred: np.ndarray, ref: np.ndarray, eps: float = 1.0e-30) -> float:
    return float(np.linalg.norm(pred - ref) / max(np.linalg.norm(ref), eps))


def linf(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.max(np.abs(pred - ref)))


def load_pair_geometry_and_tempest(cfg, pair: str):
    edge_path = cfg.edge_path(pair)
    df = pd.read_parquet(edge_path)

    src_index = df["source_index"].to_numpy(dtype=np.int64)
    tgt_index = df["target_index"].to_numpy(dtype=np.int64)

    n_src = int(src_index.max()) + 1
    n_tgt = int(tgt_index.max()) + 1

    src_xyz = unique_indexed_array(df, "source_index", ["src_x", "src_y", "src_z"], n_src)
    tgt_xyz = unique_indexed_array(df, "target_index", ["tgt_x", "tgt_y", "tgt_z"], n_tgt)

    src_area = unique_indexed_array(df, "source_index", ["src_area"], n_src)[:, 0]
    tgt_area = unique_indexed_array(df, "target_index", ["tgt_area"], n_tgt)[:, 0]

    wcol = get_true_weight_column(df)
    S_true = df[wcol].to_numpy(dtype=np.float64)

    mask = np.isfinite(S_true)
    if "edge_exists" in df.columns:
        mask &= df["edge_exists"].to_numpy(dtype=np.float64) > 0.5
    else:
        mask &= np.abs(S_true) > 0.0

    return {
        "edge_path": edge_path,
        "df": df,
        "src_index": src_index,
        "tgt_index": tgt_index,
        "S_true": S_true,
        "mask_true": mask,
        "n_src": n_src,
        "n_tgt": n_tgt,
        "src_xyz": src_xyz,
        "tgt_xyz": tgt_xyz,
        "src_area": src_area,
        "tgt_area": tgt_area,
        "h_src": float(np.sqrt(np.mean(src_area))),
        "h_tgt": float(np.sqrt(np.mean(tgt_area))),
    }


def try_load_irno(cfg, device):
    from evaluate_irno_corrector import build_models_and_state

    base_cfg, base_pack, base_model, corrector_pack, corrector, stats = build_models_and_state(cfg, device)
    return {
        "base_cfg": base_cfg,
        "base_pack": base_pack,
        "base_model": base_model,
        "corrector_pack": corrector_pack,
        "corrector": corrector,
        "stats": stats,
    }


@torch.no_grad()
def get_irno_states(cfg, irno_state, pair: str, balance_iters: int, device):
    from evaluate_irno_corrector import operator_sequence

    batch = load_pair_tensors(cfg, pair, irno_state["stats"], device=device)

    states = operator_sequence(
        base_model=irno_state["base_model"],
        corrector_pack=irno_state["corrector_pack"],
        corrector=irno_state["corrector"],
        batch=batch,
        balance_iters=balance_iters,
    )

    return batch, states


def apply_torch_operator_to_numpy_source(
    S: torch.Tensor,
    src_index: torch.Tensor,
    tgt_index: torch.Tensor,
    n_tgt: int,
    x_src_np: np.ndarray,
    device,
) -> np.ndarray:
    x_src = torch.as_tensor(x_src_np, device=device, dtype=torch.double)
    y = torch.zeros(n_tgt, device=device, dtype=torch.double)
    y.index_add_(0, tgt_index.long(), S.double() * x_src[src_index.long()])
    return y.detach().cpu().numpy()


def add_error_row(
    rows: list[dict],
    *,
    pair: str,
    function_name: str,
    method: str,
    step: int,
    lmax: int,
    step_label: str,
    n_src: int,
    n_tgt: int,
    h_src: float,
    h_tgt: float,
    pred: np.ndarray,
    truth: np.ndarray,
    src_field: np.ndarray,
    src_area: np.ndarray,
    tgt_area: np.ndarray,
):
    source_integral = float(np.sum(src_area * src_field))
    target_pred_integral = float(np.sum(tgt_area * pred))
    target_truth_integral = float(np.sum(tgt_area * truth))

    rows.append(
        {
            "pair": pair,
            "function": function_name,
            "method": method,
            "step": step,
            "lmax": lmax,
            "step_label": step_label,
            "n_src": n_src,
            "n_tgt": n_tgt,
            "h_src": h_src,
            "h_tgt": h_tgt,
            "area_rel_l2": area_rel_l2(pred, truth, tgt_area),
            "rel_l2": unweighted_rel_l2(pred, truth),
            "linf": linf(pred, truth),
            "source_integral": source_integral,
            "target_pred_integral": target_pred_integral,
            "target_truth_integral": target_truth_integral,
            "conservation_abs_error_vs_source": abs(target_pred_integral - source_integral),
            "target_integral_abs_error_vs_truth": abs(target_pred_integral - target_truth_integral),
        }
    )


def compute_observed_orders(df: pd.DataFrame, error_col: str = "area_rel_l2", h_col: str = "h_tgt") -> pd.DataFrame:
    rows = []

    group_cols = ["method", "step", "lmax", "step_label", "function"]

    for key, g in df.groupby(group_cols):
        g = g.sort_values("h_tgt", ascending=False).reset_index(drop=True)

        if len(g) < 2:
            continue

        for i in range(len(g) - 1):
            coarse = g.iloc[i]
            fine = g.iloc[i + 1]

            Ec = float(coarse[error_col])
            Ef = float(fine[error_col])
            hc = float(coarse[h_col])
            hf = float(fine[h_col])

            if Ec > 0 and Ef > 0 and hc > hf:
                p = math.log(Ec / Ef) / math.log(hc / hf)
            else:
                p = np.nan

            row = {
                "method": key[0],
                "step": key[1],
                "lmax": key[2],
                "step_label": key[3],
                "function": key[4],
                "coarse_pair": coarse["pair"],
                "fine_pair": fine["pair"],
                "h_col": h_col,
                "h_coarse": hc,
                "h_fine": hf,
                "E_coarse": Ec,
                "E_fine": Ef,
                "observed_order": p,
            }
            rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pairs", nargs="+", required=True)
    parser.add_argument(
        "--functions",
        nargs="+",
        default=["const", "x", "y", "z", "smooth1", "smooth2", "Y_2_0", "Y_4_0", "Y_8_0"],
    )
    parser.add_argument("--include-irno", action="store_true")
    parser.add_argument("--include-irno-trajectory", action="store_true")
    parser.add_argument("--balance-iters", type=int, default=2000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-prefix", default=None)
    parser.add_argument(
        "--order-h",
        choices=["src", "tgt", "max", "min"],
        default="max",
        help="Mesh-size proxy used for observed-order calculation.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    out_prefix = args.out_prefix
    if out_prefix is None:
        out_prefix = f"analysis_medium_improv/convergence_refinement_{cfg.run_name}"

    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    irno = None
    if args.include_irno or args.include_irno_trajectory:
        print("Loading IRNO/corrector model...")
        irno = try_load_irno(cfg, device)

    rows = []

    for pair in args.pairs:
        print(f"\nEvaluating pair: {pair}")

        geom = load_pair_geometry_and_tempest(cfg, pair)

        print(
            f"  n_src={geom['n_src']} n_tgt={geom['n_tgt']} "
            f"h_src={geom['h_src']:.6e} h_tgt={geom['h_tgt']:.6e}"
        )

        batch = None
        states = None
        if irno is not None:
            batch, states = get_irno_states(cfg, irno, pair, args.balance_iters, device)
            if not args.include_irno_trajectory:
                states = [states[-1]]

        for fname in args.functions:
            x_src = analytic_function(fname, geom["src_xyz"])
            truth_tgt = analytic_function(fname, geom["tgt_xyz"])

            mask = geom["mask_true"]
            y_tempest = scatter_numpy(
                geom["n_tgt"],
                geom["tgt_index"][mask],
                geom["S_true"][mask] * x_src[geom["src_index"][mask]],
            )

            add_error_row(
                rows,
                pair=pair,
                function_name=fname,
                method="tempest",
                step=-1,
                lmax=-1,
                step_label="tempest",
                n_src=geom["n_src"],
                n_tgt=geom["n_tgt"],
                h_src=geom["h_src"],
                h_tgt=geom["h_tgt"],
                pred=y_tempest,
                truth=truth_tgt,
                src_field=x_src,
                src_area=geom["src_area"],
                tgt_area=geom["tgt_area"],
            )

            if states is not None:
                for state in states:
                    y_gnn = apply_torch_operator_to_numpy_source(
                        state["S"],
                        batch["src_index"],
                        batch["tgt_index"],
                        geom["n_tgt"],
                        x_src,
                        device,
                    )

                    add_error_row(
                        rows,
                        pair=pair,
                        function_name=fname,
                        method="irno",
                        step=int(state["step"]),
                        lmax=int(state["lmax"]),
                        step_label=str(state["label"]),
                        n_src=geom["n_src"],
                        n_tgt=geom["n_tgt"],
                        h_src=geom["h_src"],
                        h_tgt=geom["h_tgt"],
                        pred=y_gnn,
                        truth=truth_tgt,
                        src_field=x_src,
                        src_area=geom["src_area"],
                        tgt_area=geom["tgt_area"],
                    )

    df = pd.DataFrame(rows)
    df["h_max"] = np.maximum(df["h_src"], df["h_tgt"])
    df["h_min"] = np.minimum(df["h_src"], df["h_tgt"])

    h_col = {
        "src": "h_src",
        "tgt": "h_tgt",
        "max": "h_max",
        "min": "h_min",
    }[args.order_h]

    orders = compute_observed_orders(df, error_col="area_rel_l2", h_col=h_col)

    detail_path = out_prefix.with_suffix(".csv")
    order_path = out_prefix.with_name(out_prefix.name + "_orders.csv")

    df.to_csv(detail_path, index=False)
    orders.to_csv(order_path, index=False)

    print(f"\nWrote details: {detail_path}")
    print(f"Wrote orders:  {order_path}")

    print("\nErrors:")
    show = (
        df.groupby(["method", "step_label", "function", "pair"], as_index=False)
        .agg(area_rel_l2=("area_rel_l2", "mean"), h_tgt=("h_tgt", "mean"), n_tgt=("n_tgt", "first"))
        .sort_values(["method", "step_label", "function", "h_tgt"], ascending=[True, True, True, False])
    )
    print(show.to_string(index=False, float_format=lambda x: f"{x:.6e}"))

    if len(orders):
        print("\nObserved orders:")
        print(
            orders.sort_values(["method", "step_label", "function", "h_coarse"], ascending=[True, True, True, False])
            .to_string(index=False, float_format=lambda x: f"{x:.6e}")
        )
    else:
        print("\nNo observed orders computed. Need at least two refinement levels per method/function.")


if __name__ == "__main__":
    main()
