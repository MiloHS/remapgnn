#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from evaluate_refinement_convergence import (
    load_config,
    load_pair_geometry_and_tempest,
    try_load_irno,
    get_irno_states,
    analytic_function,
    scatter_numpy,
)


STAGE_ALIASES = {
    "base": "base",
    "lmax8": "corrected_lmax8",
    "lmax16": "corrected_lmax16",
    "lmax24": "corrected_lmax24",
    "corrected_lmax8": "corrected_lmax8",
    "corrected_lmax16": "corrected_lmax16",
    "corrected_lmax24": "corrected_lmax24",
}


def select_state(states, wanted_label):
    label_to_index = {
        "base": 0,
        "corrected_lmax8": 1,
        "corrected_lmax16": 2,
        "corrected_lmax24": 3,
    }

    for state in states:
        if state.get("step_label") == wanted_label:
            return state

    if wanted_label in label_to_index:
        idx = label_to_index[wanted_label]
        if idx < len(states):
            return states[idx]

    raise ValueError(f"Could not find stage {wanted_label}; number of states={len(states)}")


def to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def get_state_weights(state):
    for key in ["S", "weights", "S_pred", "remap_weights", "edge_weights"]:
        if key in state:
            return to_numpy(state[key]).astype(np.float64)
    raise KeyError(f"Could not find remap weights. Available keys: {list(state.keys())}")


def scatter_sum(n, index, values):
    out = np.zeros(n, dtype=np.float64)
    np.add.at(out, index, values)
    return out


def infer_src_area(geom):
    for key in ["src_area", "area_src", "source_area"]:
        if key in geom:
            return np.asarray(geom[key], dtype=np.float64)
    raise KeyError(f"Could not find source area in geometry keys: {list(geom.keys())}")


def operator_residuals(geom, weights, mask=None):
    src_index = geom["src_index"]
    tgt_index = geom["tgt_index"]
    tgt_area = np.asarray(geom["tgt_area"], dtype=np.float64)
    src_area = infer_src_area(geom)

    if mask is not None:
        src_index = src_index[mask]
        tgt_index = tgt_index[mask]
        weights = weights[mask]

    row_sum = scatter_sum(geom["n_tgt"], tgt_index, weights)
    source_area_from_operator = scatter_sum(
        len(src_area),
        src_index,
        tgt_area[tgt_index] * weights,
    )

    row_err = row_sum - 1.0
    src_area_err = source_area_from_operator - src_area

    return {
        "row_sum_linf": float(np.max(np.abs(row_err))),
        "row_sum_l1_mean": float(np.mean(np.abs(row_err))),
        "src_area_linf_abs": float(np.max(np.abs(src_area_err))),
        "src_area_l1_rel": float(
            np.sum(np.abs(src_area_err)) / max(np.sum(np.abs(src_area)), 1e-300)
        ),
        "src_area_global_rel": float(
            abs(np.sum(source_area_from_operator) - np.sum(src_area))
            / max(abs(np.sum(src_area)), 1e-300)
        ),
    }


def field_integrals(geom, weights, function_name, mask=None):
    src_index = geom["src_index"]
    tgt_index = geom["tgt_index"]
    tgt_area = np.asarray(geom["tgt_area"], dtype=np.float64)
    src_area = infer_src_area(geom)

    if mask is not None:
        src_index_use = src_index[mask]
        tgt_index_use = tgt_index[mask]
        weights_use = weights[mask]
    else:
        src_index_use = src_index
        tgt_index_use = tgt_index
        weights_use = weights

    x_src = analytic_function(function_name, geom["src_xyz"])
    truth_tgt = analytic_function(function_name, geom["tgt_xyz"])

    y = scatter_numpy(
        geom["n_tgt"],
        tgt_index_use,
        weights_use * x_src[src_index_use],
    )

    source_integral = float(np.sum(src_area * x_src))
    remapped_integral = float(np.sum(tgt_area * y))
    target_truth_integral = float(np.sum(tgt_area * truth_tgt))

    denom = max(abs(source_integral), 1e-300)

    return {
        "source_integral": source_integral,
        "remapped_integral": remapped_integral,
        "target_truth_integral": target_truth_integral,
        "signed_integral_error": remapped_integral - source_integral,
        "relative_integral_error": abs(remapped_integral - source_integral) / denom,
        "truth_vs_source_relative": abs(target_truth_integral - source_integral) / denom,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--functions", nargs="+", default=["smooth1", "smooth2"])
    ap.add_argument("--stage", default="lmax24")
    ap.add_argument("--balance-iters", type=int, default=2000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    stage_label = STAGE_ALIASES.get(args.stage, args.stage)

    cfg = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    irno = None
    rows = []

    for pair in args.pairs:
        print("=" * 80)
        print(f"Pair: {pair}")
        print("=" * 80)

        geom = load_pair_geometry_and_tempest(cfg, pair)

        # Tempest operator.
        mask = geom["mask_true"]
        tempest_weights = np.asarray(geom["S_true"], dtype=np.float64)
        temp_op = operator_residuals(geom, tempest_weights, mask=mask)

        for fname in args.functions:
            vals = field_integrals(geom, tempest_weights, fname, mask=mask)
            rows.append({
                "pair": pair,
                "method": "tempest",
                "stage": "tempest",
                "function": fname,
                **vals,
                **temp_op,
            })

        # Learned operator.
        if irno is None:
            print("Loading IRNO/corrector model")
            irno = try_load_irno(cfg, device)

        print(f"Computing learned operator stage: {stage_label}")
        _, states = get_irno_states(cfg, irno, pair, args.balance_iters, device)
        state = select_state(states, stage_label)
        learned_weights = get_state_weights(state)

        learned_op = operator_residuals(geom, learned_weights, mask=None)

        for fname in args.functions:
            vals = field_integrals(geom, learned_weights, fname, mask=None)
            rows.append({
                "pair": pair,
                "method": "irno",
                "stage": stage_label,
                "function": fname,
                **vals,
                **learned_op,
            })

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")

    show_cols = [
        "pair", "method", "stage", "function",
        "relative_integral_error",
        "signed_integral_error",
        "row_sum_linf",
        "src_area_l1_rel",
        "src_area_global_rel",
    ]
    print(df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
