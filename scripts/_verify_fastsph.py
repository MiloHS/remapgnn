"""Adversarial equivalence check: fv_moments.real_sph_unnorm_fast vs the scipy-based
_real_sph_unnorm, across ALL (l,m) modes used by the audit/training, on random sphere points.

Safety gate: fast must equal slow UP TO A PER-(l,m) CONSTANT (that is all the metric needs,
since every field is a single (l,m) normalized by its own RMS and area_rel_l2 is a ratio).
For each mode we fit c = <fast,slow>/<slow,slow> and require ||fast - c*slow|| / ||fast|| < 1e-10.
Also reports whether the constant c is the SAME across all modes (it should be ~1/sqrt(4pi)).
"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from remapgnn.fv_moments import real_sph_unnorm_fast
from train_config_balanced_harmonic import _real_sph_unnorm

rng = np.random.default_rng(0)
N = 20000
v = rng.normal(size=(N, 3))
v = v / np.linalg.norm(v, axis=1, keepdims=True)
# include near-pole points to stress the recurrence
v[:50, 0] = 1e-9; v[:50, 1] = 1e-9; v[:50, 2] = 1.0
v[50:100, 2] = -1.0; v[50:100, 0] = 1e-9; v[50:100, 1] = 1e-9
v = v / np.linalg.norm(v, axis=1, keepdims=True)

degrees = [0, 1, 2, 3, 4, 8, 12, 16, 24, 32, 40, 48]
worst = 0.0
consts = []
n_modes = 0
fail = []
for l in degrees:
    # sample m: 0, +/-1, +/-(l//2), +/-l
    ms = sorted(set([0, 1, -1, l, -l, l // 2, -(l // 2), (3 * l) // 4, -((3 * l) // 4)]))
    ms = [m for m in ms if abs(m) <= l]
    for m in ms:
        slow = _real_sph_unnorm(l, m, v)
        fast = real_sph_unnorm_fast(l, m, v)
        ss = float(np.dot(slow, slow))
        if ss < 1e-300:
            continue
        c = float(np.dot(fast, slow) / ss)
        resid = np.linalg.norm(fast - c * slow) / max(np.linalg.norm(fast), 1e-300)
        worst = max(worst, resid)
        consts.append(c)
        n_modes += 1
        if resid > 1e-10:
            fail.append((l, m, resid, c))

consts = np.array(consts)
print(f"modes checked: {n_modes}")
print(f"worst up-to-constant residual: {worst:.3e}  (gate < 1e-10)")
print(f"per-mode constant c: mean={consts.mean():.6f} std={consts.std():.3e} "
      f"(1/sqrt(4pi)={1.0/np.sqrt(4*np.pi):.6f})")
if fail:
    print("FAILURES:")
    for l, m, r, c in fail[:20]:
        print(f"  l={l} m={m} resid={r:.3e} c={c:.4f}")
print("VERIFY:", "PASS" if worst < 1e-10 and not fail else "FAIL")
