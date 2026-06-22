"""Validate over-relaxed (SOR) Sinkhorn on the STIFF directions (RLL->CS, ICOD->CS) that motivated
converged balancing, with a divergence safeguard. Reports iters-to-1e-6 + wall-clock vs vanilla,
fixed-point agreement, and how often/where SOR has to back off omega."""
import os, sys, time
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
import torch
from remapgnn.config import load_config
from remapgnn.data import load_pair_tensors
from remapgnn.models import build_model, scatter_sum_torch
from remapgnn.sinkhorn import sparse_operator_weights
from train_config_irno_corrector import torch_load_pack, base_q_from_model, as_int

torch.set_num_threads(8)
EPS = 1e-30
CFG = "configs/v20b_base_diverse_topologies_l24_a2p0_mink8.json"
# stiff reverse-to-CS directions + a same-family control; r32 and r64
PAIRS = sys.argv[1:] or ["ICOD-r32_to_CS-r32", "RLL-r30-60_to_CS-r16",
                         "ICOD-r64_to_CS-r64", "RLL-r90-180_to_CS-r32"]
dev = torch.device("cpu")

cfg = load_config(CFG); pack = torch_load_pack(cfg.model_path, map_location=dev)
sf = list(pack["src_node_features"]); tf = list(pack["tgt_node_features"]); ef = list(pack["edge_features"])
base = build_model(architecture=pack.get("architecture", cfg.architecture),
                   src_dim=len(sf), tgt_dim=len(tf), edge_dim=len(ef),
                   hidden=int(pack.get("hidden", 128)),
                   decoder_chunk_size=int(pack.get("decoder_chunk_size", 10000))).to(dev)
base.load_state_dict(pack["model_state_dict"]); base.eval()
for p in base.parameters(): p.requires_grad_(False)


for pair in PAIRS:
    try:
        b = load_pair_tensors(cfg, pair, pack["stats"], device=dev)
    except Exception as e:
        print("\n=== %s ===  SKIP (%s)" % (pair, str(e)[:60])); continue
    si, ti = b["src_index"], b["tgt_index"]
    ns, nt = as_int(b["n_src"]), as_int(b["n_tgt"]); ne = si.numel()
    asrc, atgt = b["area_src"].float(), b["area_tgt"].float()
    with torch.no_grad():
        q = base_q_from_model(base, b).float()
    print("\n=== %s ===  edges=%d  src=%d tgt=%d" % (pair, ne, ns, nt))

    def resid(M):
        sm = scatter_sum_torch(M, si, ns); tm = scatter_sum_torch(M, ti, nt)
        return float(max(torch.linalg.norm(sm - asrc) / torch.linalg.norm(asrc),
                         torch.linalg.norm(tm - atgt) / torch.linalg.norm(atgt)))

    def run(omega, tol, mx, check=25, safeguard=True):
        """SOR with omega; if a residual check rises vs the previous check, back off omega toward 1
        and restart from the last-good M. Returns (M, iters, n_backoffs, final_omega)."""
        M = torch.clamp(q, min=EPS); Mgood = M; it = 0; last_r = float("inf"); w = omega; backoffs = 0
        while it < mx:
            for _ in range(check):
                tm = scatter_sum_torch(M, ti, nt); M = M * ((atgt / torch.clamp(tm, min=EPS))[ti]) ** w
                sm = scatter_sum_torch(M, si, ns); M = M * ((asrc / torch.clamp(sm, min=EPS))[si]) ** w
                it += 1
            r = resid(M)
            if r < tol:
                return M, it, backoffs, w
            if safeguard and (r > last_r or not torch.isfinite(torch.tensor(r))):
                backoffs += 1
                w = 1.0 + (w - 1.0) * 0.5      # halve the over-relaxation
                M = Mgood                       # roll back to last good state
            else:
                Mgood = M; last_r = r
        return M, it, backoffs, w

    t = time.time(); Mv, itv, _, _ = run(1.0, 1e-6, 60000, safeguard=False); tv = time.time() - t
    Sv = sparse_operator_weights(M=Mv, tgt_index=ti, area_tgt=atgt)
    print("  vanilla (w=1.0):  iters=%6d  time=%.2fs  resid=%.1e" % (itv, tv, resid(Mv)))

    for om in (1.9, 1.95, 1.98):
        t = time.time(); Ms, its, bo, wf = run(om, 1e-6, 60000); ts = time.time() - t
        sd = float(torch.linalg.norm(sparse_operator_weights(M=Ms, tgt_index=ti, area_tgt=atgt) - Sv) / torch.linalg.norm(Sv))
        ok = "OK" if (resid(Ms) < 1e-6 and sd < 1e-3) else "**CHECK**"
        print("  SOR w0=%.2f:       iters=%6d  time=%.2fs  iter-sp=%4.1fx  time-sp=%4.1fx  backoffs=%d (w_final=%.2f)  S-diff=%.1e  %s"
              % (om, its, ts, itv/max(its,1), tv/max(ts,1e-9), bo, wf, sd, ok))
    del b
print("\nSTIFF_BENCH_DONE")
