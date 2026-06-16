from __future__ import annotations

from pathlib import Path
import argparse
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from remapgnn.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--out", default="analysis_medium_improv/config_comparison.csv")
    args = parser.parse_args()

    rows = []

    for cfg_path in args.configs:
        cfg = load_config(cfg_path)

        for pair in cfg.pairs:
            eval_csv = cfg.eval_csv(pair)
            diag_csv = cfg.diagnostics_csv(pair)
            edge_path = cfg.edge_path(pair)

            if not eval_csv.exists() or not diag_csv.exists() or not edge_path.exists():
                print(f"missing result for {cfg.run_name} {pair}")
                continue

            df = pd.read_csv(eval_csv)
            diag = pd.read_csv(diag_csv).iloc[0]
            vs_tempest = df[df["method"].str.contains("vs_tempest", regex=False)]

            row = {
                "pair": pair,
                "run_name": cfg.run_name,
                "model_tag": cfg.model_tag,
                "architecture": cfg.architecture,
                "graph_suffix": cfg.graph_suffix,
                "edges": len(pd.read_parquet(edge_path, columns=["source_index"])),
                "mean_rel_l2_vs_tempest": vs_tempest["rel_l2"].mean(),
                "max_rel_l2_vs_tempest": vs_tempest["rel_l2"].max(),
                "row_sum_rel_l2": diag["row_sum_rel_l2"],
                "target_mass_rel_l2": diag["target_mass_rel_l2"],
                "source_mass_rel_l2": diag["source_mass_rel_l2"],
                "mean_abs_conservation_error": vs_tempest["abs_conservation_error_vs_source"].abs().mean(),
            }

            for _, r in vs_tempest.iterrows():
                row[f"{r['field']}_rel_l2"] = r["rel_l2"]

            rows.append(row)

    out = pd.DataFrame(rows)

    if out.empty:
        print("No rows found.")
        return

    out = out.sort_values(["pair", "run_name"])
    print(out.to_string(index=False))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print("\nwrote", out_path)


if __name__ == "__main__":
    main()
