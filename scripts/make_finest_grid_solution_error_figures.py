#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    area_rel_l2,
    xyz_to_lon_lat,
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


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def get_state_weights(state):
    for key in ["S", "weights", "S_pred", "remap_weights", "edge_weights"]:
        if key in state:
            return to_numpy(state[key]).astype(np.float64)

    raise KeyError(f"Could not find remap weights in state. Available keys: {list(state.keys())}")


def select_state(states, wanted_label):
    # get_irno_states returns the trajectory in order, but the state dicts
    # do not necessarily carry step_label metadata.
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

    labels = [s.get("step_label") for s in states]
    raise ValueError(
        f"Could not find stage {wanted_label}. "
        f"Available labels: {labels}; number of states={len(states)}"
    )


def lonlat_deg(xyz):
    lon, lat, _, _ = xyz_to_lon_lat(xyz)
    lon_deg = np.rad2deg(lon)
    lat_deg = np.rad2deg(lat)

    # Some mesh utilities return lon in [0, 360).  Wrap to [-180, 180)
    # so global plots are not clipped by xlim=(-180, 180).
    lon_deg = ((lon_deg + 180.0) % 360.0) - 180.0

    return lon_deg, lat_deg


def scatter_panel(ax, lon, lat, values, title, vmin=None, vmax=None, cmap=None, point_size=0.6):
    sc = ax.scatter(
        lon,
        lat,
        c=values,
        s=point_size,
        linewidths=0,
        rasterized=True,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_title(title)
    return sc


def plot_one(pair, function_name, stage_label, geom, y_tempest, y_irno, truth_tgt, x_src, outdir, point_size):
    src_lon, src_lat = lonlat_deg(geom["src_xyz"])
    tgt_lon, tgt_lat = lonlat_deg(geom["tgt_xyz"])

    err_tempest = y_tempest - truth_tgt
    err_irno = y_irno - truth_tgt

    e_tempest = area_rel_l2(y_tempest, truth_tgt, geom["tgt_area"])
    e_irno = area_rel_l2(y_irno, truth_tgt, geom["tgt_area"])

    field_values = np.concatenate([x_src, truth_tgt, y_tempest, y_irno])
    field_vmin = np.nanpercentile(field_values, 1.0)
    field_vmax = np.nanpercentile(field_values, 99.0)

    err_values = np.concatenate([err_tempest, err_irno])
    err_abs = np.nanpercentile(np.abs(err_values), 99.5)
    if not np.isfinite(err_abs) or err_abs == 0:
        err_abs = np.max(np.abs(err_values)) if len(err_values) else 1.0
    if err_abs == 0:
        err_abs = 1.0

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.0), constrained_layout=True)

    sc = scatter_panel(
        axes[0, 0],
        src_lon,
        src_lat,
        x_src,
        "Source analytic field",
        vmin=field_vmin,
        vmax=field_vmax,
        point_size=point_size,
    )
    fig.colorbar(sc, ax=axes[0, 0], shrink=0.85)

    sc = scatter_panel(
        axes[0, 1],
        tgt_lon,
        tgt_lat,
        truth_tgt,
        "Target analytic truth",
        vmin=field_vmin,
        vmax=field_vmax,
        point_size=point_size,
    )
    fig.colorbar(sc, ax=axes[0, 1], shrink=0.85)

    sc = scatter_panel(
        axes[0, 2],
        tgt_lon,
        tgt_lat,
        y_irno,
        f"GNN prediction: {stage_label}",
        vmin=field_vmin,
        vmax=field_vmax,
        point_size=point_size,
    )
    fig.colorbar(sc, ax=axes[0, 2], shrink=0.85)

    sc = scatter_panel(
        axes[1, 0],
        tgt_lon,
        tgt_lat,
        y_tempest,
        "Tempest prediction",
        vmin=field_vmin,
        vmax=field_vmax,
        point_size=point_size,
    )
    fig.colorbar(sc, ax=axes[1, 0], shrink=0.85)

    sc = scatter_panel(
        axes[1, 1],
        tgt_lon,
        tgt_lat,
        err_tempest,
        f"Tempest error, rel L2={e_tempest:.3e}",
        vmin=-err_abs,
        vmax=err_abs,
        cmap="coolwarm",
        point_size=point_size,
    )
    fig.colorbar(sc, ax=axes[1, 1], shrink=0.85)

    sc = scatter_panel(
        axes[1, 2],
        tgt_lon,
        tgt_lat,
        err_irno,
        f"GNN error, rel L2={e_irno:.3e}",
        vmin=-err_abs,
        vmax=err_abs,
        cmap="coolwarm",
        point_size=point_size,
    )
    fig.colorbar(sc, ax=axes[1, 2], shrink=0.85)

    fig.suptitle(f"{pair} finest-grid solution and errors: {function_name}", fontsize=15)

    out = outdir / f"finest_{safe_name(pair)}_{safe_name(function_name)}_{safe_name(stage_label)}_solution_errors.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)

    print(f"Wrote {out}")
    print(f"  Tempest rel L2: {e_tempest:.6e}")
    print(f"  GNN rel L2:     {e_irno:.6e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--functions", nargs="+", default=["smooth1", "smooth2"])
    parser.add_argument("--stage", default="lmax24")
    parser.add_argument("--balance-iters", type=int, default=2000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--outdir", default="analysis_medium_improv/github_results")
    parser.add_argument("--point-size", type=float, default=0.6)
    args = parser.parse_args()

    stage_label = STAGE_ALIASES.get(args.stage, args.stage)

    cfg = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Loading geometry for {args.pair}")
    geom = load_pair_geometry_and_tempest(cfg, args.pair)

    print("Loading IRNO/corrector model")
    irno = try_load_irno(cfg, device)
    _, states = get_irno_states(cfg, irno, args.pair, args.balance_iters, device)
    state = select_state(states, stage_label)
    S_irno = get_state_weights(state)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    mask = geom["mask_true"]

    for fname in args.functions:
        print(f"\nFunction: {fname}")

        x_src = analytic_function(fname, geom["src_xyz"])
        truth_tgt = analytic_function(fname, geom["tgt_xyz"])

        y_tempest = scatter_numpy(
            geom["n_tgt"],
            geom["tgt_index"][mask],
            geom["S_true"][mask] * x_src[geom["src_index"][mask]],
        )

        y_irno = scatter_numpy(
            geom["n_tgt"],
            geom["tgt_index"],
            S_irno * x_src[geom["src_index"]],
        )

        plot_one(
            pair=args.pair,
            function_name=fname,
            stage_label=stage_label,
            geom=geom,
            y_tempest=y_tempest,
            y_irno=y_irno,
            truth_tgt=truth_tgt,
            x_src=x_src,
            outdir=outdir,
            point_size=args.point_size,
        )


if __name__ == "__main__":
    main()
