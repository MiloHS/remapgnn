"""Time the GNN weight-generation path (forward + converged Sinkhorn) on real pairs, CPU.
Reports cell counts, edge count, forward time, Sinkhorn time + iters-to-tol, and a residual-vs-iters
probe (to show how many iters the converged operator actually needs at inference)."""
import os, sys, time
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
import numpy as np
import torch
from remapgnn.config import load_config
from remapgnn.data import load_pair_tensors
from remapgnn.models import build_model
from remapgnn.sinkhorn import sparse_sinkhorn_balance, converged_balance, sparse_operator_weights
from train_config_irno_corrector import torch_load_pack, base_q_from_model, as_int

torch.set_num_threads(8)
CFG = "configs/v20b_base_diverse_topologies_l24_a2p0_mink8.json"
PAIRS = sys.argv[1:] or ["CS-r32_to_ICOD-r32", "CS-r64_to_ICOD-r64"]
device = torch.device("cpu")

cfg = load_config(CFG)
pack = torch_load_pack(cfg.model_path, map_location=device)
sf = list(pack["src_node_features"]); tf = list(pack["tgt_node_features"]); ef = list(pack["edge_features"])
base = build_model(architecture=pack.get("architecture", cfg.architecture),
                   src_dim=len(sf), tgt_dim=len(tf), edge_dim=len(ef),
                   hidden=int(pack.get("hidden", 128)),
                   decoder_chunk_size=int(pack.get("decoder_chunk_size", 10000))).to(device)
base.load_state_dict(pack["model_state_dict"]); base.eval()
for p in base.parameters(): p.requires_grad_(False)


def resid(M, si, ti, asrc, atgt, ns, nt):
    sm = torch.zeros(ns).index_add_(0, si, M); tm = torch.zeros(nt).index_add_(0, ti, M)
    rs = float(torch.linalg.norm(sm - asrc) / torch.linalg.norm(asrc))
    rt = float(torch.linalg.norm(tm - atgt) / torch.linalg.norm(atgt))
    return max(rs, rt)


for pair in PAIRS:
    b = load_pair_tensors(cfg, pair, pack["stats"], device=device)
    si, ti = b["src_index"], b["tgt_index"]
    ns, nt = as_int(b["n_src"]), as_int(b["n_tgt"]); ne = si.numel()
    asrc, atgt = b["area_src"].float(), b["area_tgt"].float()
    print("\n=== %s ===  src_cells=%d  tgt_cells=%d  edges=%d" % (pair, ns, nt, ne))

    with torch.no_grad():
        for _ in range(2): base_q_from_model(base, b)  # warmup
        t = time.time(); q = base_q_from_model(base, b); t_fwd = time.time() - t

    kw = dict(src_index=si, tgt_index=ti, area_src=asrc, area_tgt=atgt, n_src=ns, n_tgt=nt)
    t = time.time(); M = converged_balance(q=q, tol=1e-6, max_iter=50000, **kw); t_sink = time.time() - t
    S = sparse_operator_weights(M=M, tgt_index=ti, area_tgt=atgt)
    print("  forward:        %.3f s" % t_fwd)
    print("  converged sink: %.3f s   final resid=%.2e" % (t_sink, resid(M, si, ti, asrc, atgt, ns, nt)))
    print("  TOTAL weight-gen: %.3f s" % (t_fwd + t_sink))

    # residual vs fixed iters (how many iters inference actually needs)
    qf = q.float()
    for K in (300, 1000, 3000, 10000, 30000):
        t = time.time(); Mk = sparse_sinkhorn_balance(q=qf, n_iter=K, **kw); tk = time.time() - t
        print("    n_iter=%6d  resid=%.2e  (%.3f s)" % (K, resid(Mk, si, ti, asrc, atgt, ns, nt), tk))
    del b
print("\nBENCH_DONE")
