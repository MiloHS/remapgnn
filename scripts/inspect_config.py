from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from remapgnn.config import load_config


def check_exists(label, path):
    status = "OK" if path.exists() else "MISSING"
    print(f"{status:8s} {label:18s} {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    print(f"Config:       {cfg.path}")
    print(f"Run name:     {cfg.run_name}")
    print(f"Model tag:    {cfg.model_tag}")
    print(f"Architecture: {cfg.architecture}")
    print(f"Graph suffix: {cfg.graph_suffix}")
    print(f"K:            {cfg.K}")
    print()
    check_exists("model", cfg.model_path)
    check_exists("history", cfg.history_path)
    print()

    for pair in cfg.pairs:
        print("=" * 100)
        print(pair)
        check_exists("edge", cfg.edge_path(pair))
        check_exists("map", cfg.map_path(pair))
        check_exists("eval csv", cfg.eval_csv(pair))
        check_exists("diagnostics", cfg.diagnostics_csv(pair))
        src_file, tgt_file = cfg.source_target_files(pair)
        check_exists("source field", src_file)
        check_exists("target field", tgt_file)


if __name__ == "__main__":
    main()
