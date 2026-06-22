"""GPU weight-generation timing: forward + SOR-accelerated converged Sinkhorn, on cuda, with proper
warmup + cuda.synchronize. Compares to TempestRemap CPU (overlap+map): r32 1.49s, r64 4.45s, r128 14.6s."""
import os, sys, time
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
import torch
from remapgnn.config import load_config
from remapgnn.data import load_pair_tensors
from remapgnn.models import build_model
from remapgnn.sinkhorn import converged_balance, sparse_operator_weights, DEFAULT_OMEGA
from train_config_irno_corrector import torch_load_pack, base_q_from_model, as_int

assert torch.cuda.is_available(), "no cuda"
dev = torch.device("cuda")
CFG = "configs/v20b_base_diverse_topologies_l24_a2p0_mink8.json"
PAIRS = ["CS-r32_to_ICOD-r32", "CS-r64_to_ICOD-r64", "CS-r128_to_ICOD-r128"]
TEMPEST_CPU = {"CS-r32_to_ICOD-r32": 1.49, "CS-r64_to_ICOD-r64": 4.45, "CS-r128_to_ICOD-r128": 14.6}

cfg = load_config(CFG); pack = torch_load_pack(cfg.model_path, map_location=dev)
sf = list(pack["src_node_features"]); tf = list(pack["tgt_node_features"]); ef = list(pack["edge_features"])
base = build_model(architecture=pack.get("architecture", cfg.architecture),
                   src_dim=len(sf), tgt_dim=len(tf), edge_dim=len(ef),
                   hidden=int(pack.get("hidden", 128)),
                   decoder_chunk_size=int(pack.get("decoder_chunk_size", 10000))).to(dev)
base.load_state_dict(pack["model_state_dict"]); base.eval()
for p in base.parameters(): p.requires_grad_(False)
sync = torch.cuda.synchronize
print("device:", torch.cuda.get_device_name(0))

def med(fn, n=5):
    ts = []
    for _ in range(n):
        sync(); t = time.time(); fn(); sync(); ts.append(time.time() - t)
    ts.sort(); return ts[len(ts)//2]

for pair in PAIRS:
    try:
        b = load_pair_tensors(cfg, pair, pack["stats"], device=dev)
    except Exception as e:
        print("\n=== %s ===  SKIP (%s)" % (pair, str(e)[:70])); continue
    si, ti = b["src_index"], b["tgt_index"]
    ns, nt = as_int(b["n_src"]), as_int(b["n_tgt"]); ne = si.numel()
    asrc, atgt = b["area_src"].float(), b["area_tgt"].float()
    kw = dict(src_index=si, tgt_index=ti, area_src=asrc, area_tgt=atgt, n_src=ns, n_tgt=nt)

    with torch.no_grad():
        for _ in range(3):  # warmup (cudnn autotune, kernel compile)
            q = base_q_from_model(base, b).float()
            converged_balance(q=q, tol=1e-6, max_iter=50000, omega=DEFAULT_OMEGA, **kw)
        t_fwd = med(lambda: base_q_from_model(base, b))
        q = base_q_from_model(base, b).float()
        t_sink = med(lambda: converged_balance(q=q, tol=1e-6, max_iter=50000, omega=DEFAULT_OMEGA, **kw))
    tot = t_fwd + t_sink
    tcpu = TEMPEST_CPU.get(pair)
    cmp = ("  vs Tempest %.2fs CPU -> %.1fx %s" % (tcpu, tcpu/tot if tot > 0 else 0,
           "FASTER" if tot < tcpu else "slower")) if tcpu else ""
    print("\n=== %s ===  edges=%d src=%d tgt=%d" % (pair, ne, ns, nt))
    print("  forward(GPU)=%.3fs  SOR-sink(GPU)=%.3fs  TOTAL=%.3fs%s" % (t_fwd, t_sink, tot, cmp))
    del b
print("\nGPU_BENCH_DONE")
