from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import sys
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

from evaluate_irno_corrector import (
    build_models_and_state,
    operator_sequence,
    get_batch_true_weight,
)


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
        raise RuntimeError("scipy.special.sph_harm_y or scipy.special.sph_harm is required.") from e


SPH = get_sph_harm_func()


def parse_degrees(s: str) -> list[int]:
    return [int(x) for x in s.replace(",", " ").split()]


def stable_int_hash(text: str) -> int:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


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


def xyz_to_angles(xyz: np.ndarray):
    xyz = np.asarray(xyz, dtype=np.float64)
    r = np.linalg.norm(xyz, axis=1)
    z = np.clip(xyz[:, 2] / np.maximum(r, 1.0e-30), -1.0, 1.0)
    theta = np.arccos(z)  # colatitude
    phi = np.mod(np.arctan2(xyz[:, 1], xyz[:, 0]), 2.0 * np.pi)
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
    norm = np.sqrt(np.mean(y * y))
    if norm > 0:
        y = y / norm
    return y.astype("float64")


def read_source_xyz_from_edges(edge_path: Path, n_src: int) -> np.ndarray:
    df = pd.read_parquet(edge_path, columns=["source_index", "src_x", "src_y", "src_z"])
    g = df.groupby("source_index", sort=False)[["src_x", "src_y", "src_z"]].first()

    xyz = np.full((n_src, 3), np.nan, dtype=np.float64)
    idx = g.index.to_numpy(dtype=np.int64)
    xyz[idx] = g.to_numpy(dtype=np.float64)

    if np.isnan(xyz).any():
        missing = np.where(np.isnan(xyz[:, 0]))[0][:10]
        raise RuntimeError(f"Missing source coordinates in {edge_path}, first few: {missing}")

    return xyz


def as_int(x):
    return int(x.item() if hasattr(x, "item") else x)


def scatter_to_target(edge_values: torch.Tensor, tgt_index: torch.Tensor, n_tgt: int) -> torch.Tensor:
    y = torch.zeros(n_tgt, device=edge_values.device, dtype=edge_values.dtype)
    y.index_add_(0, tgt_index, edge_values)
    return y


def rel_l2(pred: torch.Tensor, ref: torch.Tensor, eps: float = 1.0e-30) -> float:
    return float((torch.linalg.norm(pred - ref) / torch.clamp(torch.linalg.norm(ref), min=eps)).detach().cpu())


def area_rel_l2(pred: torch.Tensor, ref: torch.Tensor, area_tgt: torch.Tensor, eps: float = 1.0e-30) -> float:
    diff = pred - ref
    num = torch.sum(area_tgt * diff * diff)
    den = torch.clamp(torch.sum(area_tgt * ref * ref), min=eps)
    return float(torch.sqrt(num / den).detach().cpu())


@torch.no_grad()
def evaluate_pair_spectral_trajectory(
    cfg,
    base_model,
    corrector_pack,
    corrector,
    stats,
    pair: str,
    degrees: list[int],
    modes_per_degree: int,
    balance_iters: int,
    seed: int,
    device,
):
    batch = load_pair_tensors(cfg, pair, stats, device=device)

    src_index = batch["src_index"]
    tgt_index = batch["tgt_index"]
    edge_exists = batch["edge_exists"].float()
    pos = edge_exists > 0.5

    area_tgt = batch["area_tgt"].double()
    n_src = as_int(batch["n_src"])
    n_tgt = as_int(batch["n_tgt"])

    S_true = get_batch_true_weight(batch).double()

    xyz = read_source_xyz_from_edges(cfg.edge_path(pair), n_src=n_src)

    states = operator_sequence(
        base_model=base_model,
        corrector_pack=corrector_pack,
        corrector=corrector,
        batch=batch,
        balance_iters=balance_iters,
    )

    rng = np.random.default_rng(seed + stable_int_hash(pair))

    mode_list = []
    for l in degrees:
        for m in choose_m_values(l, modes_per_degree, rng):
            mode_list.append((l, m))

    rows = []

    for l, m in mode_list:
        x_np = real_spherical_harmonic(l, m, xyz)
        x_src = torch.as_tensor(x_np, device=device, dtype=torch.double)
        x_edge = x_src[src_index]

        y_tempest = scatter_to_target(
            S_true[pos] * x_edge[pos],
            tgt_index[pos],
            n_tgt,
        )

        tempest_norm = float(torch.linalg.norm(y_tempest).detach().cpu())

        for state in states:
            S = state["S"].double()

            y_pred = scatter_to_target(
                S * x_edge,
                tgt_index,
                n_tgt,
            )

            rows.append(
                {
                    "pair": pair,
                    "run_name": cfg.run_name,
                    "model_tag": cfg.model_tag,
                    "step": int(state["step"]),
                    "lmax": int(state["lmax"]),
                    "step_label": state["label"],
                    "kind": "mode",
                    "degree": int(l),
                    "m": int(m),
                    "rel_l2_vs_tempest": rel_l2(y_pred, y_tempest),
                    "area_rel_l2_vs_tempest": area_rel_l2(y_pred, y_tempest, area_tgt),
                    "tempest_norm": tempest_norm,
                    "gnn_norm": float(torch.linalg.norm(y_pred).detach().cpu()),
                    "mean_abs_diff": float(torch.mean(torch.abs(y_pred - y_tempest)).detach().cpu()),
                }
            )

    del batch
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pair", default=None)
    parser.add_argument("--all-pairs", action="store_true")
    parser.add_argument("--degrees", default="0 1 2 4 8 12 16 24 32")
    parser.add_argument("--modes-per-degree", type=int, default=9)
    parser.add_argument("--balance-iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    degrees = parse_degrees(args.degrees)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.all_pairs:
        pairs = list(cfg.pairs)
    elif args.pair:
        pairs = [args.pair]
    else:
        pairs = [cfg.raw.get("training", {}).get("test_pair", cfg.pairs[0])]

    print(f"Config:           {args.config}")
    print(f"Run name:         {cfg.run_name}")
    print(f"Model path:       {cfg.model_path}")
    print(f"Device:           {device}")
    print(f"Pairs:            {pairs}")
    print(f"Degrees:          {degrees}")
    print(f"Modes per degree: {args.modes_per_degree}")
    print(f"Balance iters:    {args.balance_iters}")

    base_cfg, base_pack, base_model, corrector_pack, corrector, stats = build_models_and_state(cfg, device)

    print()
    print("Loaded frozen base:")
    print(f"  {corrector_pack['base_model_path']}")
    print("Loaded corrector:")
    print(f"  bands={corrector_pack['bands']}, alpha={corrector_pack['alpha']}")

    all_rows = []

    for pair in pairs:
        print()
        print(f"Evaluating spectral trajectory: {pair}")
        rows = evaluate_pair_spectral_trajectory(
            cfg=cfg,
            base_model=base_model,
            corrector_pack=corrector_pack,
            corrector=corrector,
            stats=stats,
            pair=pair,
            degrees=degrees,
            modes_per_degree=args.modes_per_degree,
            balance_iters=args.balance_iters,
            seed=args.seed,
            device=device,
        )
        all_rows.extend(rows)

        pair_df = pd.DataFrame(rows)
        pair_summary = (
            pair_df.groupby(["step", "lmax", "step_label", "degree"], as_index=False)
            .agg(
                mean_rel_l2_vs_tempest=("rel_l2_vs_tempest", "mean"),
                max_rel_l2_vs_tempest=("rel_l2_vs_tempest", "max"),
                mean_area_rel_l2_vs_tempest=("area_rel_l2_vs_tempest", "mean"),
                max_area_rel_l2_vs_tempest=("area_rel_l2_vs_tempest", "max"),
                n_modes=("rel_l2_vs_tempest", "size"),
            )
        )

        print(
            pair_summary[
                [
                    "step",
                    "lmax",
                    "step_label",
                    "degree",
                    "mean_rel_l2_vs_tempest",
                    "max_rel_l2_vs_tempest",
                    "n_modes",
                ]
            ].to_string(index=False)
        )

    df = pd.DataFrame(all_rows)

    if args.out is None:
        out = Path("analysis_medium_improv") / f"{cfg.model_tag}_irno_spectral_trajectory.csv"
    else:
        out = Path(args.out)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    summary = (
        df.groupby(["pair", "step", "lmax", "step_label", "kind", "degree"], as_index=False)
        .agg(
            mean_rel_l2_vs_tempest=("rel_l2_vs_tempest", "mean"),
            max_rel_l2_vs_tempest=("rel_l2_vs_tempest", "max"),
            mean_area_rel_l2_vs_tempest=("area_rel_l2_vs_tempest", "mean"),
            max_area_rel_l2_vs_tempest=("area_rel_l2_vs_tempest", "max"),
            n_modes=("rel_l2_vs_tempest", "size"),
        )
    )

    summary_out = out.with_name(out.stem + "_summary.csv")
    summary.to_csv(summary_out, index=False)

    global_summary = (
        df.groupby(["step", "lmax", "step_label", "degree"], as_index=False)
        .agg(
            mean_rel_l2_vs_tempest=("rel_l2_vs_tempest", "mean"),
            max_rel_l2_vs_tempest=("rel_l2_vs_tempest", "max"),
            mean_area_rel_l2_vs_tempest=("area_rel_l2_vs_tempest", "mean"),
            max_area_rel_l2_vs_tempest=("area_rel_l2_vs_tempest", "max"),
            n_modes=("rel_l2_vs_tempest", "size"),
        )
    )

    global_out = out.with_name(out.stem + "_global_summary.csv")
    global_summary.to_csv(global_out, index=False)

    print()
    print(f"Wrote detailed trajectory: {out}")
    print(f"Wrote pair summary:        {summary_out}")
    print(f"Wrote global summary:      {global_out}")

    print()
    print("Global mean spectral error by degree and step:")
    print(
        global_summary.pivot_table(
            index="degree",
            columns="step_label",
            values="mean_rel_l2_vs_tempest",
        ).to_string(float_format=lambda x: f"{x:.6e}")
    )

    print()
    print("Global max spectral error by degree and step:")
    print(
        global_summary.pivot_table(
            index="degree",
            columns="step_label",
            values="max_rel_l2_vs_tempest",
        ).to_string(float_format=lambda x: f"{x:.6e}")
    )


if __name__ == "__main__":
    main()
