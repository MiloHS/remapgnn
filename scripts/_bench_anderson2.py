"""Sinkhorn acceleration done right: Anderson on the DUAL potentials (dim n_tgt) + over-relaxation
(SOR) in M-space. Both vs vanilla, iters-to-1e-6 + wall-clock, fixed-point agreement checked."""
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
log = torch.log
CFG = "configs/v20b_base_diverse_topologies_l24_a2p0_mink8.json"
PAIRS = sys.argv[1:] or ["CS-r32_to_ICOD-r32", "CS-r64_to_ICOD-r64"]
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
    b = load_pair_tensors(cfg, pair, pack["stats"], device=dev)
    si, ti = b["src_index"], b["tgt_index"]
    ns, nt = as_int(b["n_src"]), as_int(b["n_tgt"]); ne = si.numel()
    asrc, atgt = b["area_src"].float(), b["area_tgt"].float()
    lasrc, latgt = log(torch.clamp(asrc, min=EPS)), log(torch.clamp(atgt, min=EPS))
    with torch.no_grad():
        q = base_q_from_model(base, b).float()
    print("\n=== %s ===  edges=%d  src=%d tgt=%d" % (pair, ne, ns, nt))

    def resid(M):
        sm = scatter_sum_torch(M, si, ns); tm = scatter_sum_torch(M, ti, nt)
        return float(max(torch.linalg.norm(sm - asrc) / torch.linalg.norm(asrc),
                         torch.linalg.norm(tm - atgt) / torch.linalg.norm(atgt)))

    # ---- vanilla M-space ----
    def vanilla(tol, mx, check=50):
        M = torch.clamp(q, min=EPS); it = 0
        while it < mx:
            for _ in range(check):
                tm = scatter_sum_torch(M, ti, nt); M = M * (atgt / torch.clamp(tm, min=EPS))[ti]
                sm = scatter_sum_torch(M, si, ns); M = M * (asrc / torch.clamp(sm, min=EPS))[si]
                it += 1
            if resid(M) < tol: break
        return M, it

    # ---- SOR (over-relaxed) M-space, no log ----
    def sor(omega, tol, mx, check=50):
        M = torch.clamp(q, min=EPS); it = 0
        while it < mx:
            for _ in range(check):
                tm = scatter_sum_torch(M, ti, nt); M = M * ((atgt / torch.clamp(tm, min=EPS))[ti]) ** omega
                sm = scatter_sum_torch(M, si, ns); M = M * ((asrc / torch.clamp(sm, min=EPS))[si]) ** omega
                it += 1
            if resid(M) < tol: break
        return M, it

    # ---- Anderson on dual potential b (dim nt) ----
    def a_from_b(bb): return torch.clamp(lasrc - log(torch.clamp(scatter_sum_torch(q * torch.exp(bb[ti]), si, ns), min=EPS)), -60, 60)
    def b_from_a(aa): return torch.clamp(latgt - log(torch.clamp(scatter_sum_torch(q * torch.exp(aa[si]), ti, nt), min=EPS)), -60, 60)
    def M_ab(aa, bb): return q * torch.exp(aa[si] + bb[ti])
    def anderson_dual(m, tol, mx, check=10, reg=1e-10):
        bb = torch.zeros(nt); it = 0; hB, hF = [], []; last_r = float("inf")
        while it < mx:
            G = b_from_a(a_from_b(bb)); F = G - bb; it += 1
            hB.append(G); hF.append(F)
            if len(hB) > m: hB.pop(0); hF.pop(0)
            k = len(hF)
            if k == 1:
                bb = G
            else:
                R = torch.stack(hF, dim=1); RtR = R.transpose(0, 1) @ R
                RtR = RtR + reg * (torch.trace(RtR) / k) * torch.eye(k)
                sol = torch.linalg.solve(RtR, torch.ones(k, 1)); alpha = (sol / sol.sum()).squeeze(1)
                bb = torch.stack(hB, dim=1) @ alpha
            if it % check == 0:
                r = resid(M_ab(a_from_b(bb), bb))
                if r < tol: break
                if r > last_r: hB, hF = [], []
                last_r = r
        return M_ab(a_from_b(bb), bb), it

    t = time.time(); Mv, itv = vanilla(1e-6, 50000); tv = time.time() - t
    Sv = sparse_operator_weights(M=Mv, tgt_index=ti, area_tgt=atgt)
    print("  vanilla:        iters=%6d  time=%.2fs" % (itv, tv))

    for om in (1.7, 1.9, 1.95):
        t = time.time(); Ms, its = sor(om, 1e-6, 50000); ts = time.time() - t
        sd = float(torch.linalg.norm(sparse_operator_weights(M=Ms, tgt_index=ti, area_tgt=atgt) - Sv) / torch.linalg.norm(Sv))
        print("  SOR w=%.2f:      iters=%6d  time=%.2fs  iter-sp=%.1fx  time-sp=%.1fx  S-diff=%.1e" % (om, its, ts, itv/max(its,1), tv/max(ts,1e-9), sd))

    for m in (6, 12):
        t = time.time(); Ma, ita = anderson_dual(m, 1e-6, 50000); ta = time.time() - t
        sd = float(torch.linalg.norm(sparse_operator_weights(M=Ma, tgt_index=ti, area_tgt=atgt) - Sv) / torch.linalg.norm(Sv))
        print("  Anderson-dual m=%d: iters=%5d  time=%.2fs  iter-sp=%.1fx  time-sp=%.1fx  S-diff=%.1e" % (m, ita, ta, itv/max(ita,1), tv/max(ta,1e-9), sd))
    del b
print("\nBENCH2_DONE")
