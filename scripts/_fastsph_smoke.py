"""Empirical equivalence + speedup: cell-average harmonic TRUTH via fast vs slow SH on a real pair."""
import sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from remapgnn import fv_moments as fv
from train_config_balanced_harmonic import _real_sph_unnorm

pair = "CS-r64_to_ICOD-r64"   # a large (slow) pair
mp = str(ROOT / f"maps_medium_improv/map_{pair}_conserve.nc")
q = fv.grid_quadrature(mp, "b", m=8)
print(f"pair={pair} tgt quad points={q['points'].shape[0]}")
modes = [(1, 0), (4, 2), (16, 8), (48, 24)]

def cellavg_norm(fn):
    y = fv.grid_cell_average(fn, q)
    n = np.sqrt(np.mean(y * y))
    return y / n if n > 0 else y

t0 = time.perf_counter()
slow = {lm: cellavg_norm(lambda xyz, l=lm[0], m=lm[1]: _real_sph_unnorm(l, m, xyz)) for lm in modes}
t_slow = time.perf_counter() - t0

t0 = time.perf_counter()
fast = {lm: cellavg_norm(lambda xyz, l=lm[0], m=lm[1]: fv.real_sph_unnorm_fast(l, m, xyz)) for lm in modes}
t_fast = time.perf_counter() - t0

worst = max(float(np.max(np.abs(slow[lm] - fast[lm]))) for lm in modes)
print(f"max |normalized_slow - normalized_fast| over modes: {worst:.3e}")
print(f"slow SH time: {t_slow:.2f}s   fast SH time: {t_fast:.2f}s   speedup: {t_slow/max(t_fast,1e-9):.1f}x")
print("SMOKE:", "PASS (identical + faster)" if worst < 1e-9 and t_fast < t_slow else "CHECK")
