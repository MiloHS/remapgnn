#!/usr/bin/env python3
import argparse
import torch

from remapgnn.config import load_config
from scripts.evaluate_refinement_convergence import try_load_irno


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cpu")
    model = try_load_irno(cfg, device)

    total = 0
    trainable = 0
    for p in model.__dict__.values():
        pass

    # try common module locations
    modules = []
    for name in ["base", "corrector", "model", "gnn"]:
        if hasattr(model, name):
            modules.append((name, getattr(model, name)))

    seen = set()
    rows = []
    for name, module in modules:
        if not hasattr(module, "parameters"):
            continue
        t = sum(p.numel() for p in module.parameters())
        tr = sum(p.numel() for p in module.parameters() if p.requires_grad)
        rows.append((name, t, tr))
        total += t
        trainable += tr

    print("Parameter counts:")
    for name, t, tr in rows:
        print(f"{name:20s} total={t:,} trainable={tr:,}")

    print(f"\nTotal across detected modules: total={total:,} trainable={trainable:,}")


if __name__ == "__main__":
    main()
