#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd


def fit_slope(g, error_col, h_col):
    g = g.sort_values(h_col, ascending=False)
    h = g[h_col].to_numpy(float)
    e = g[error_col].to_numpy(float)

    mask = np.isfinite(h) & np.isfinite(e) & (h > 0) & (e > 0)
    h = h[mask]
    e = e[mask]

    if len(h) < 2:
        return np.nan, np.nan

    x = np.log(h)
    y = np.log(e)

    # y = a + p*x, so p is observed order.
    p, a = np.polyfit(x, y, 1)

    yhat = a + p * x
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return float(p), float(r2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--error-col", default="area_rel_l2")
    ap.add_argument("--h-col", default="h_max")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    if args.h_col not in df.columns:
        if args.h_col == "h_max":
            df["h_max"] = np.maximum(df["h_src"], df["h_tgt"])
        elif args.h_col == "h_min":
            df["h_min"] = np.minimum(df["h_src"], df["h_tgt"])
        else:
            raise ValueError(f"Missing h column: {args.h_col}")

    rows = []
    group_cols = ["method", "step_label", "function"]

    for key, g in df.groupby(group_cols):
        p, r2 = fit_slope(g, args.error_col, args.h_col)
        rows.append({
            "method": key[0],
            "step_label": key[1],
            "function": key[2],
            "n_levels": len(g),
            "fit_order": p,
            "fit_r2": r2,
            "error_coarsest": g.sort_values(args.h_col, ascending=False)[args.error_col].iloc[0],
            "error_finest": g.sort_values(args.h_col, ascending=False)[args.error_col].iloc[-1],
        })

    out = pd.DataFrame(rows).sort_values(["method", "step_label", "function"])
    print(out.to_string(index=False, float_format=lambda x: f"{x:.6g}"))

    if args.out:
        out.to_csv(args.out, index=False)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
