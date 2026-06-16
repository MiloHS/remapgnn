#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DETAIL = Path("analysis_medium_improv/convergence_CS_to_ICOD_4level_v18_trajectory_smooth.csv")
SLOPES = Path("analysis_medium_improv/convergence_CS_to_ICOD_4level_v18_trajectory_smooth_slopes.csv")
OUTDIR = Path("analysis_medium_improv/github_results")
OUTDIR.mkdir(parents=True, exist_ok=True)

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
    detail = pd.read_csv(DETAIL)
    detail["h_max"] = np.maximum(detail["h_src"], detail["h_tgt"])

    # Geometric-mean error over the smooth analytic suite.
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
    plt.xlabel("Mesh size proxy h = max(sqrt(mean source area), sqrt(mean target area))")
    plt.ylabel("Geometric mean area-weighted relative L2 error")
    plt.title("CS→ICOD four-level analytic convergence")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = OUTDIR / "convergence_CS_to_ICOD_4level_loglog.png"
    plt.savefig(out, dpi=200)
    print(f"Wrote {out}")

    slopes = pd.read_csv(SLOPES)

    rows = []
    for stage in STAGE_ORDER:
        g = slopes[slopes["step_label"] == stage]
        if len(g) == 0:
            continue
        rows.append(
            {
                "step_label": stage,
                "display": STAGE_LABEL.get(stage, stage),
                "mean_order": g["fit_order"].mean(),
                "min_order": g["fit_order"].min(),
                "max_order": g["fit_order"].max(),
                "mean_r2": g["fit_r2"].mean(),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTDIR / "convergence_CS_to_ICOD_4level_stage_summary.csv", index=False)

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
    out = OUTDIR / "convergence_CS_to_ICOD_4level_fitted_orders.png"
    plt.savefig(out, dpi=200)
    print(f"Wrote {out}")

    print("\nStage summary:")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
