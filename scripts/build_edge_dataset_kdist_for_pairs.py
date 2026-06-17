#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_convergence_edge_dataset_kdist import build_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--map-dir", default="maps_medium_improv")
    ap.add_argument("--out-dir", default="analysis_medium_improv")
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--min-k", type=int, default=8)
    ap.add_argument("--flush-rows", type=int, default=200000)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    map_dir = Path(args.map_dir)
    out_dir = Path(args.out_dir)

    for pair in args.pairs:
        map_path = map_dir / f"map_{pair}_conserve.nc"
        if not map_path.exists():
            print(f"MISSING map: {map_path}")
            continue

        print("=" * 80)
        print(f"Building edge dataset for {pair}")
        print("=" * 80)

        build_one(
            map_path=map_path,
            out_dir=out_dir,
            alpha=args.alpha,
            min_k=args.min_k,
            flush_rows=args.flush_rows,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
