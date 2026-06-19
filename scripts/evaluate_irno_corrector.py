from __future__ import annotations

from pathlib import Path
import argparse
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
from remapgnn.data import load_pair_tensors, get_feature_lists
from remapgnn.models import build_model

from train_config_irno_corrector import (
    torch_load_pack,
    as_int,
    get_batch_true_weight,
    base_q_from_model,
    compute_operator_from_logq,
    make_augmented_edge_attr,
    corrector_delta_from_output,
    conservation_metrics,
)

DEFAULT_FIELDS = [
    "AnalyticalFun1",
    "AnalyticalFun2",
    "TotalPrecipWater",
    "CloudFraction",
    "Topography",
]


def safe_pair_name(pair: str) -> str:
    return pair.replace("-", "_").replace(".", "p").replace("/", "_").replace(":", "_")


def unique_keep_order(xs):
    out, seen = [], set()
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def source_target_files(cfg, pair):
    x = cfg.source_target_files(pair)

    if isinstance(x, dict):
        src = x.get("src") or x.get("src_file") or x.get("source") or x.get("source_file")
        tgt = x.get("tgt") or x.get("tgt_file") or x.get("target") or x.get("target_file")
        return Path(src), Path(tgt)

    if isinstance(x, (list, tuple)) and len(x) >= 2:
        return Path(x[0]), Path(x[1])

    raise RuntimeError(f"Could not interpret cfg.source_target_files({pair!r}) result: {x!r}")


def load_field_flat(path: Path, field: str, expected_n: int) -> np.ndarray:
    try:
        import xarray as xr
    except Exception as e:
        raise RuntimeError("xarray is required for field evaluation.") from e

    ds = xr.open_dataset(path)

    if field not in ds:
        available = list(ds.data_vars)
        raise KeyError(f"{field!r} not found in {path}. Available variables: {available}")

    arr = np.asarray(ds[field].values).astype("float64").ravel()

    if arr.size != expected_n:
        raise RuntimeError(
            f"{field} in {path} has size {arr.size}, expected {expected_n}. "
            f"Shape was {ds[field].shape}."
        )

    return arr


def scatter_to_target(edge_values: torch.Tensor, tgt_index: torch.Tensor, n_tgt: int) -> torch.Tensor:
    y = torch.zeros(n_tgt, device=edge_values.device, dtype=edge_values.dtype)
    y.index_add_(0, tgt_index, edge_values)
    return y


def rel_l2(a: torch.Tensor, b: torch.Tensor, eps: float = 1.0e-30) -> float:
    return float(
        (
            torch.linalg.norm(a - b)
            / torch.clamp(torch.linalg.norm(b), min=eps)
        ).detach().cpu()
    )


def mean_abs(a: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(a)).detach().cpu())


def build_models_and_state(cfg, device):
    corrector_pack = torch_load_pack(cfg.model_path, map_location=device)

    if corrector_pack.get("kind") != "irno_corrector":
        raise RuntimeError(
            f"Expected an IRNO corrector checkpoint, got kind={corrector_pack.get('kind')!r}"
        )

    base_cfg_path = Path(corrector_pack["base_config"])
    base_cfg = load_config(str(base_cfg_path))
    base_model_path = Path(corrector_pack["base_model_path"])
    base_pack = torch_load_pack(base_model_path, map_location=device)

    edge_features = list(corrector_pack.get("edge_features", base_pack.get("edge_features", get_feature_lists(base_cfg)[0])))
    src_features = list(corrector_pack.get("src_node_features", base_pack.get("src_node_features", get_feature_lists(base_cfg)[1])))
    tgt_features = list(corrector_pack.get("tgt_node_features", base_pack.get("tgt_node_features", get_feature_lists(base_cfg)[2])))

    stats = base_pack["stats"]

    base_model = build_model(
        architecture=base_pack.get("architecture", base_cfg.architecture),
        src_dim=len(src_features),
        tgt_dim=len(tgt_features),
        edge_dim=len(edge_features),
        hidden=int(base_pack.get("hidden", 128)),
        decoder_chunk_size=int(base_pack.get("decoder_chunk_size", 10000)),
    ).to(device)
    base_model.load_state_dict(base_pack["model_state_dict"])
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)

    corrector = build_model(
        architecture=corrector_pack["corrector_architecture"],
        src_dim=len(src_features),
        tgt_dim=len(tgt_features),
        edge_dim=int(corrector_pack["corrector_edge_dim"]),
        hidden=int(corrector_pack.get("hidden", 128)),
        decoder_chunk_size=int(corrector_pack.get("decoder_chunk_size", 10000)),
    ).to(device)
    corrector.load_state_dict(corrector_pack["model_state_dict"])
    corrector.eval()
    for p in corrector.parameters():
        p.requires_grad_(False)

    return base_cfg, base_pack, base_model, corrector_pack, corrector, stats


@torch.no_grad()

def compute_operator_from_logq_eval(
    logq,
    src_index,
    tgt_index,
    area_src,
    area_tgt,
    n_src,
    n_tgt,
    n_iter,
    tol=1.0e-6,
):
    from remapgnn.sinkhorn import sparse_sinkhorn_balance, sparse_operator_weights

    q = torch.exp(torch.clamp(logq.double(), min=-60.0, max=40.0))
    # Eval/inference: iterate Sinkhorn to convergence so the operator is simultaneously
    # conservative AND consistent (rows sum to 1). `n_iter` (e.g. --balance-iters) acts as the
    # floor for the max-iteration cap. Setting tol=None reverts to the old fixed-count behavior.
    max_iter = None if tol is None else max(int(n_iter), 50000)
    M = sparse_sinkhorn_balance(
        q=q,
        src_index=src_index,
        tgt_index=tgt_index,
        area_src=area_src.double(),
        area_tgt=area_tgt.double(),
        n_src=n_src,
        n_tgt=n_tgt,
        n_iter=n_iter,
        tol=tol,
        max_iter=max_iter,
    )
    S = sparse_operator_weights(
        M=M,
        tgt_index=tgt_index,
        area_tgt=area_tgt.double(),
    )
    return M, S

def operator_sequence(base_model, corrector_pack, corrector, batch, balance_iters: int):
    src_index = batch["src_index"]
    tgt_index = batch["tgt_index"]
    area_src = batch["area_src"].float()
    area_tgt = batch["area_tgt"].float()
    n_src = as_int(batch["n_src"])
    n_tgt = as_int(batch["n_tgt"])

    bands = [int(x) for x in corrector_pack["bands"]]
    alpha = float(corrector_pack["alpha"])
    lmax_denominator = float(corrector_pack.get("lmax_denominator", 32.0))

    q0 = base_q_from_model(base_model, batch)
    logq = torch.log(torch.clamp(q0.float(), min=1.0e-30))

    M0, S0 = compute_operator_from_logq_eval(
        logq,
        src_index,
        tgt_index,
        area_src,
        area_tgt,
        n_src,
        n_tgt,
        balance_iters,
    )

    out = [
        {
            "step": 0,
            "lmax": 0,
            "label": "base",
            "logq": logq.detach(),
            "M": M0.detach(),
            "S": S0.detach(),
        }
    ]

    S_current = S0.detach()

    K = len(bands)
    for k, lmax in enumerate(bands, start=1):
        step_frac = k / max(K, 1)
        lmax_frac = float(lmax) / float(lmax_denominator)

        aug_edge_attr = make_augmented_edge_attr(
            batch["edge_attr"],
            logq,
            S_current,
            step_frac=step_frac,
            lmax_frac=lmax_frac,
        )

        corr_out = corrector(
            batch["src_node_attr"],
            batch["tgt_node_attr"],
            aug_edge_attr,
            src_index,
            tgt_index,
            n_src,
            n_tgt,
        )

        delta = corrector_delta_from_output(corr_out)
        logq = logq + alpha * torch.tanh(delta.float())

        M, S = compute_operator_from_logq_eval(
            logq,
            src_index,
            tgt_index,
            area_src,
            area_tgt,
            n_src,
            n_tgt,
            balance_iters,
        )

        out.append(
            {
                "step": k,
                "lmax": int(lmax),
                "label": f"corrected_lmax{lmax}",
                "logq": logq.detach(),
                "M": M.detach(),
                "S": S.detach(),
            }
        )

        S_current = S.detach()

    return out


@torch.no_grad()
def evaluate_pair(cfg, base_model, corrector_pack, corrector, stats, pair, fields, device, balance_iters):
    batch = load_pair_tensors(cfg, pair, stats, device=device)

    src_index = batch["src_index"]
    tgt_index = batch["tgt_index"]
    edge_exists = batch["edge_exists"].float()
    pos = edge_exists > 0.5
    S_true = get_batch_true_weight(batch)

    area_src = batch["area_src"].float()
    area_tgt = batch["area_tgt"].float()
    n_src = as_int(batch["n_src"])
    n_tgt = as_int(batch["n_tgt"])

    src_file, _ = source_target_files(cfg, pair)

    seq = operator_sequence(
        base_model=base_model,
        corrector_pack=corrector_pack,
        corrector=corrector,
        batch=batch,
        balance_iters=balance_iters,
    )

    rows = []

    # Preload source fields.
    source_fields = {}
    for field in fields:
        try:
            arr = load_field_flat(src_file, field, expected_n=n_src)
        except Exception as e:
            print(f"WARNING: skipping {field} for {pair}: {e}")
            continue
        source_fields[field] = torch.as_tensor(arr, device=device, dtype=torch.float64)

    for state in seq:
        M = state["M"].float()
        S = state["S"].float()

        source_rel, row_rel = conservation_metrics(
            M,
            src_index,
            tgt_index,
            area_src,
            area_tgt,
            n_src,
            n_tgt,
        )

        src_mass = torch.zeros(n_src, device=device, dtype=M.dtype)
        src_mass.index_add_(0, src_index, M)
        mean_abs_source_conservation = mean_abs(src_mass - area_src)

        field_metrics = {}
        rels = []

        for field, x_src in source_fields.items():
            x_edge = x_src[src_index].float()

            y_pred = scatter_to_target(
                S * x_edge,
                tgt_index,
                n_tgt,
            )

            y_tempest = scatter_to_target(
                S_true[pos].float() * x_edge[pos],
                tgt_index[pos],
                n_tgt,
            )

            e = rel_l2(y_pred.float(), y_tempest.float())
            field_metrics[f"{field}_rel_l2"] = e
            rels.append(e)

        row = {
            "pair": pair,
            "run_name": cfg.run_name,
            "model_tag": cfg.model_tag,
            "architecture": "irno_corrector",
            "base_model_tag": Path(corrector_pack["base_model_path"]).stem,
            "graph_suffix": cfg.graph_suffix,
            "step": int(state["step"]),
            "lmax": int(state["lmax"]),
            "step_label": state["label"],
            "n_edges": int(src_index.numel()),
            "mean_rel_l2_vs_tempest": float(np.mean(rels)) if rels else np.nan,
            "max_rel_l2_vs_tempest": float(np.max(rels)) if rels else np.nan,
            "row_sum_rel_l2": float(row_rel.detach().cpu()),
            "target_mass_rel_l2": float(row_rel.detach().cpu()),
            "source_mass_rel_l2": float(source_rel.detach().cpu()),
            "mean_abs_conservation_error": mean_abs_source_conservation,
            **field_metrics,
        }
        rows.append(row)

    del batch
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pair", default=None)
    parser.add_argument("--all-pairs", action="store_true")
    parser.add_argument("--fields", nargs="*", default=DEFAULT_FIELDS)
    parser.add_argument("--balance-iters", type=int, default=300)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.all_pairs:
        pairs = list(cfg.pairs)
    elif args.pair:
        pairs = [args.pair]
    else:
        pairs = [cfg.raw.get("training", {}).get("test_pair", cfg.pairs[0])]

    print(f"Config:        {args.config}")
    print(f"Run name:      {cfg.run_name}")
    print(f"Model path:    {cfg.model_path}")
    print(f"Device:        {device}")
    print(f"Pairs:         {pairs}")
    print(f"Fields:        {args.fields}")
    print(f"Balance iters: {args.balance_iters}")

    base_cfg, base_pack, base_model, corrector_pack, corrector, stats = build_models_and_state(cfg, device)

    print()
    print("Loaded frozen base:")
    print(f"  {corrector_pack['base_model_path']}")
    print("Loaded corrector:")
    print(f"  bands={corrector_pack['bands']}, alpha={corrector_pack['alpha']}")

    all_rows = []
    for pair in pairs:
        print()
        print(f"Evaluating {pair}")
        rows = evaluate_pair(
            cfg=cfg,
            base_model=base_model,
            corrector_pack=corrector_pack,
            corrector=corrector,
            stats=stats,
            pair=pair,
            fields=args.fields,
            device=device,
            balance_iters=args.balance_iters,
        )
        all_rows.extend(rows)

        pair_df = pd.DataFrame(rows)
        show_cols = [
            "pair",
            "step",
            "lmax",
            "step_label",
            "mean_rel_l2_vs_tempest",
            "max_rel_l2_vs_tempest",
            "row_sum_rel_l2",
            "source_mass_rel_l2",
        ]
        print(pair_df[show_cols].to_string(index=False))

    df = pd.DataFrame(all_rows)

    if args.out is None:
        out = Path("analysis_medium_improv") / f"{cfg.model_tag}_irno_field_trajectory.csv"
    else:
        out = Path(args.out)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    final = df.sort_values(["pair", "step"]).groupby("pair", as_index=False).tail(1)
    final_out = out.with_name(out.stem.replace("_trajectory", "_final") + out.suffix)
    final.to_csv(final_out, index=False)

    print()
    print(f"Wrote trajectory: {out}")
    print(f"Wrote final:      {final_out}")

    print()
    print("Final-step average mean rel_l2:")
    print(final.groupby("run_name")["mean_rel_l2_vs_tempest"].mean().to_string())

    print()
    print("Final-step pair table:")
    cols = [
        "pair",
        "step",
        "lmax",
        "mean_rel_l2_vs_tempest",
        "max_rel_l2_vs_tempest",
        "row_sum_rel_l2",
        "source_mass_rel_l2",
    ]
    print(final[cols].to_string(index=False))


if __name__ == "__main__":
    main()
