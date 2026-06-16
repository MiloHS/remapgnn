#!/usr/bin/env python3

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


STAGE_ORDER = ["tempest", "base", "corrected_lmax8", "corrected_lmax16", "corrected_lmax24"]
STAGE_LABEL = {
    "tempest": "Tempest",
    "base": "v16 base",
    "corrected_lmax8": "v18 lmax8",
    "corrected_lmax16": "v18 lmax16",
    "corrected_lmax24": "v18 lmax24",
}


def geometric_mean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.mean(np.log(x))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", required=True)
    ap.add_argument("--slopes", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--outdir", default="analysis_medium_improv/github_results")
    args = ap.parse_args()

    detail_path = Path(args.detail)
    slopes_path = Path(args.slopes)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    detail = pd.read_csv(detail_path)
    detail["h_max"] = np.maximum(detail["h_src"], detail["h_tgt"])

    agg = (
        detail.groupby(["step_label", "h_max", "pair"], as_index=False)
        .agg(geom_mean_error=("area_rel_l2", geometric_mean))
    )

    plt.figure(figsize=(7.2, 5.0))
    for stage in STAGE_ORDER:
        g = agg[agg["step_label"] == stage].sort_values("h_max", ascending=False)
        if len(g) == 0:
            continue
        plt.loglog(
            g["h_max"],
            g["geom_mean_error"],
            marker="o",
            linewidth=2,
            label=STAGE_LABEL.get(stage, stage),
        )

    plt.gca().invert_xaxis()
    plt.xlabel("Mesh size proxy h")
    plt.ylabel("Geometric mean area-weighted relative L2 error")
    plt.title(args.title)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = outdir / f"{args.tag}_loglog.png"
    plt.savefig(out, dpi=200)
    print(f"Wrote {out}")

    slopes = pd.read_csv(slopes_path)
    rows = []
    for stage in STAGE_ORDER:
        g = slopes[slopes["step_label"] == stage]
        if len(g) == 0:
            continue
        rows.append({
            "step_label": stage,
            "display": STAGE_LABEL.get(stage, stage),
            "mean_order": g["fit_order"].mean(),
            "min_order": g["fit_order"].min(),
            "max_order": g["fit_order"].max(),
            "mean_r2": g["fit_r2"].mean(),
        })

    summary = pd.DataFrame(rows)
    summary_path = outdir / f"{args.tag}_stage_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

    x = np.arange(len(summary))
    y = summary["mean_order"].to_numpy()
    yerr = np.vstack([
        y - summary["min_order"].to_numpy(),
        summary["max_order"].to_numpy() - y,
    ])

    plt.figure(figsize=(7.2, 4.4))
    plt.bar(x, y)
    plt.errorbar(x, y, yerr=yerr, fmt="none", capsize=5)
    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.xticks(x, summary["display"], rotation=20, ha="right")
    plt.ylabel("Fitted convergence order")
    plt.title("Mean fitted order across analytic fields")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = outdir / f"{args.tag}_fitted_orders.png"
    plt.savefig(out, dpi=200)
    print(f"Wrote {out}")

    print("\nStage summary:")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
