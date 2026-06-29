"""Comprehensive compute benchmark for the paper, reported as a hardware-tiered ladder:
  (1) iteration counts vanilla vs SOR  -- hardware-FREE (algorithmic)
  (2) kNN candidate-graph build time   -- CPU (inference prep, scipy cKDTree)
  (3) GNN forward + SOR-Sinkhorn        -- GPU (the deployment path), with cuda.synchronize+warmup
  (4) multi-pair throughput             -- GPU (amortization)
Run on a GPU node. TR-CPU times come separately from _bench_tempest.sh."""
import os, sys, time
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
import numpy as np
import torch
from scipy.spatial import cKDTree
from remapgnn.config import load_config
from remapgnn.data import load_pair_tensors
from remapgnn.models import build_model, scatter_sum_torch
from remapgnn.sinkhorn import converged_balance, sparse_operator_weights, DEFAULT_OMEGA
from train_config_irno_corrector import torch_load_pack, base_q_from_model, as_int
from train_config_balanced_harmonic import read_source_xyz_from_edges, read_target_xyz_from_edges

EPS = 1e-30
CFG = "configs/v20b_base_diverse_topologies_l24_a2p0_mink8.json"
PAIRS = sys.argv[1:] or ["CS-r32_to_ICOD-r32", "CS-r64_to_ICOD-r64", "CS-r128_to_ICOD-r128", "CS-r256_to_ICOD-r256"]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = dev.type == "cuda"

cfg = load_config(CFG); pack = torch_load_pack(cfg.model_path, map_location=dev)
sf = list(pack["src_node_features"]); tf = list(pack["tgt_node_features"]); ef = list(pack["edge_features"])
base = build_model(architecture=pack.get("architecture", cfg.architecture), src_dim=len(sf), tgt_dim=len(tf),
                   edge_dim=len(ef), hidden=int(pack.get("hidden", 128)),
                   decoder_chunk_size=int(pack.get("decoder_chunk_size", 10000))).to(dev)
base.load_state_dict(pack["model_state_dict"]); base.eval()
for p in base.parameters(): p.requires_grad_(False)
sync = (torch.cuda.synchronize if cuda else (lambda: None))
print("device:", torch.cuda.get_device_name(0) if cuda else "cpu", " threads:", torch.get_num_threads())


def iters_to_tol(q, si, ti, asrc, atgt, ns, nt, omega, tol=1e-6, cap=60000, check=50):
    M = torch.clamp(q, min=EPS); it = 0; last = float("inf"); w = omega; Mg = M
    asn = torch.linalg.norm(asrc); atn = torch.linalg.norm(atgt)
    while it < cap:
        for _ in range(check):
            tm = scatter_sum_torch(M, ti, nt); M = M * ((atgt/torch.clamp(tm,min=EPS))[ti] if w==1.0 else (atgt/torch.clamp(tm,min=EPS))[ti]**w)
            sm = scatter_sum_torch(M, si, ns); M = M * ((asrc/torch.clamp(sm,min=EPS))[si] if w==1.0 else (asrc/torch.clamp(sm,min=EPS))[si]**w)
            it += 1
        r = float(max(torch.linalg.norm(scatter_sum_torch(M,si,ns)-asrc)/asn, torch.linalg.norm(scatter_sum_torch(M,ti,nt)-atgt)/atn))
        if r < tol: return it
        if w > 1.0 and r > last: M = Mg; w = 1.0 + (w-1.0)*0.5
        else: Mg = M; last = r
    return it


def med(fn, n=5):
    ts = []
    for _ in range(n):
        sync(); t = time.time(); fn(); sync(); ts.append(time.time()-t)
    ts.sort(); return ts[len(ts)//2]


print("\n%-26s %9s %9s %10s %12s %9s %9s %9s" % ("pair", "cells", "edges", "knn(CPU)", "vanilla_it", "SOR_it", "fwd(GPU)", "SOR(GPU)"))
for pair in PAIRS:
    try:
        b = load_pair_tensors(cfg, pair, pack["stats"], device=dev)
    except Exception as e:
        print("%-26s SKIP %s" % (pair, str(e)[:50])); continue
    si, ti = b["src_index"], b["tgt_index"]; ns, nt = as_int(b["n_src"]), as_int(b["n_tgt"]); ne = si.numel()
    asrc, atgt = b["area_src"].float(), b["area_tgt"].float()
    ep = cfg.edge_path(pair)
    sx = read_source_xyz_from_edges(ep, ns); tx = read_target_xyz_from_edges(ep, nt)
    t = time.time(); tree = cKDTree(sx); tree.query(tx, k=12); t_knn = time.time() - t   # representative candidate build
    with torch.no_grad():
        q = base_q_from_model(base, b).float()
        kw = dict(src_index=si, tgt_index=ti, area_src=asrc, area_tgt=atgt, n_src=ns, n_tgt=nt)
        van = iters_to_tol(q.clone(), si, ti, asrc, atgt, ns, nt, 1.0)
        sor = iters_to_tol(q.clone(), si, ti, asrc, atgt, ns, nt, DEFAULT_OMEGA)
        for _ in range(3): base_q_from_model(base, b); converged_balance(q=q, tol=1e-6, max_iter=50000, **kw)  # warmup
        t_fwd = med(lambda: base_q_from_model(base, b))
        t_sor = med(lambda: converged_balance(q=q, tol=1e-6, max_iter=50000, omega=DEFAULT_OMEGA, **kw))
    print("%-26s %9d %9d %9.3fs %12d %9d %8.3fs %8.3fs" % (pair, ns, ne, t_knn, van, sor, t_fwd, t_sor))
    del b

# throughput: cycle through pairs on GPU back-to-back
print("\n=== throughput (GPU, forward+SOR, all pairs back-to-back) ===")
bs = [load_pair_tensors(cfg, p, pack["stats"], device=dev) for p in PAIRS[:3]]
with torch.no_grad():
    sync(); t = time.time()
    for b in bs:
        q = base_q_from_model(base, b).float()
        converged_balance(q=q, tol=1e-6, max_iter=50000, omega=DEFAULT_OMEGA,
                          src_index=b["src_index"], tgt_index=b["tgt_index"], area_src=b["area_src"].float(),
                          area_tgt=b["area_tgt"].float(), n_src=as_int(b["n_src"]), n_tgt=as_int(b["n_tgt"]))
    sync(); dt = time.time() - t
print("  %d pairs in %.3fs -> %.1f pairs/s" % (len(bs), dt, len(bs)/dt))
print("BENCH_FULL_DONE")
