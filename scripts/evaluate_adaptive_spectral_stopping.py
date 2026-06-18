#!/usr/bin/env python
"""Adaptive spectral stopping -- Phase 1: field/mesh spectral-bandwidth diagnostic.

Goal
----
Decide how many spherical-harmonic orders ("modes") are needed to represent a field on a
mesh to a user tolerance, so an iterative spectral remap can stop early when a given
(mesh, field) does not need the highest bands. This Phase-1 tool measures the *field's*
spectral content only (no model, no GPU); it is the predictive prior / lower bound for how
many correction steps the learned corrector will need. The reactive, operator-true stop
(running the actual base+corrector at increasing levels) is Phase 2.

Method
------
Project the field onto orthonormal real spherical harmonics by area-weighted quadrature on
the mesh (works on any mesh with cell centroids + areas: CS, ICOD, RLL, TRI, quads, HEALPix):

    a_lm = sum_i  area_i * f(x_i) * Y_lm(x_i)          (areas are steradians, sum to 4*pi)
    C_l  = sum_{m=-l..l} a_lm^2                         (angular power spectrum)
    P    = sum_i area_i * f(x_i)^2                      (total power, Parseval reference)

From C_l we derive two stop criteria:

  * Tail (truncation) error, MONOTONE in L -- the actual error of stopping at order L:
        E(L) = sqrt( max(P - sum_{l<=L} C_l, 0) / P )
        L*   = smallest L with E(L) < tol            (robust by construction)

  * Cauchy increment between successive probe levels (the mentor's "error between levels"),
    NON-monotone, so we require it to hold for `--consecutive` levels (guards parity notches):
        incr(L_k) = sqrt( sum_{L_{k-1} < l <= L_k} C_l / P )

Robustness diagnostics (the parts naive versions get wrong)
-----------------------------------------------------------
  * Nyquist cap: a mesh with N cells cannot represent order >> sqrt(N); probe levels above
    ~sqrt(N) are flagged/clipped (projecting beyond Nyquist aliases high frequencies down).
  * Achievable-error floor: quadrature on a finite mesh has its own error, so E(L) plateaus
    instead of reaching 0. A requested tolerance below that floor CANNOT be certified on this
    mesh -- reported explicitly rather than silently "met".
  * Captured-energy fraction sum(C_l)/P: if < ~0.99 the field has content above Lmax (under-
    resolved / aliasing) and the estimate is not trustworthy.

Projection choice
-----------------
Area-weighted quadrature (simple, mesh-agnostic, matches how the rest of the codebase
integrates; its accuracy is the reported floor). A least-squares fit (better for highly
irregular/clustered sampling) is a natural future option, not implemented here.

CPU only. No model required.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Orthonormal real spherical harmonics (NOT the per-field RMS-normalized version
# used in training -- here we need int Y^2 dOmega = 1 so C_l is a true power spectrum).
# ---------------------------------------------------------------------------

def _get_sph_harm():
    try:
        from scipy.special import sph_harm_y  # scipy >= 1.15: sph_harm_y(l, m, theta, phi)

        def sph(l, m, theta, phi):
            return sph_harm_y(l, m, theta, phi)

        return sph
    except Exception:
        pass
    try:
        from scipy.special import sph_harm  # legacy: sph_harm(m, l, phi, theta)

        def sph(l, m, theta, phi):
            return sph_harm(m, l, phi, theta)

        return sph
    except Exception as e:  # pragma: no cover
        raise RuntimeError("scipy.special.sph_harm or sph_harm_y is required.") from e


_SPH = _get_sph_harm()


def xyz_to_angles(xyz: np.ndarray):
    xyz = np.asarray(xyz, dtype=np.float64)
    r = np.linalg.norm(xyz, axis=1)
    z = np.clip(xyz[:, 2] / np.maximum(r, 1.0e-30), -1.0, 1.0)
    theta = np.arccos(z)                                   # colatitude in [0, pi]
    phi = np.mod(np.arctan2(xyz[:, 1], xyz[:, 0]), 2.0 * np.pi)
    return theta, phi


def real_sph_harm(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Orthonormal real spherical harmonic Y_lm (int Y_lm^2 dOmega = 1)."""
    if m == 0:
        y = _SPH(l, 0, theta, phi).real
    elif m > 0:
        y = math.sqrt(2.0) * ((-1.0) ** m) * _SPH(l, m, theta, phi).real
    else:
        mp = -m
        y = math.sqrt(2.0) * ((-1.0) ** mp) * _SPH(l, mp, theta, phi).imag
    return np.asarray(y, dtype=np.float64)


# ---------------------------------------------------------------------------
# Meshes
# ---------------------------------------------------------------------------

def fibonacci_sphere(n: int):
    """Near-uniform points on the unit sphere with equal quadrature weights (4*pi/n)."""
    i = np.arange(n, dtype=np.float64)
    golden = math.pi * (3.0 - math.sqrt(5.0))
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    theta = golden * i
    xyz = np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)
    area = np.full(n, 4.0 * math.pi / n, dtype=np.float64)
    return xyz, area


def load_mesh_from_parquet(path: Path, side: str):
    """Read unique cell centroids + areas from an edge-dataset parquet (source or target side)."""
    if side == "source":
        idx_col, cols, area_col = "source_index", ["src_x", "src_y", "src_z"], "src_area"
    elif side == "target":
        idx_col, cols, area_col = "target_index", ["tgt_x", "tgt_y", "tgt_z"], "tgt_area"
    else:
        raise ValueError("side must be 'source' or 'target'")

    df = pd.read_parquet(path, columns=[idx_col, area_col] + cols)
    g = df.groupby(idx_col, sort=True)
    xyz = g[cols].first().to_numpy(dtype=np.float64)
    area = g[area_col].first().to_numpy(dtype=np.float64)
    # Normalize areas to sum to 4*pi (defensive: some maps store areas up to a scale).
    s = float(area.sum())
    if s > 0:
        area = area * (4.0 * math.pi / s)
    return xyz, area


# ---------------------------------------------------------------------------
# Analytic test fields (include the handoff set + validation modes)
# ---------------------------------------------------------------------------

def _lon_lat(xyz):
    lon = np.arctan2(xyz[:, 1], xyz[:, 0])
    lat = np.arcsin(np.clip(xyz[:, 2], -1.0, 1.0))
    return lon, lat


def analytic_field(name: str, xyz: np.ndarray) -> np.ndarray:
    name = name.strip()
    if name in {"const", "1"}:
        return np.ones(xyz.shape[0])
    if name in {"x", "y", "z"}:
        return xyz[:, {"x": 0, "y": 1, "z": 2}[name]].astype(np.float64)

    lon, lat = _lon_lat(xyz)
    if name == "smooth1":
        return 1.0 + 0.25 * xyz[:, 0] - 0.15 * xyz[:, 1] + 0.10 * xyz[:, 2] \
            + 0.20 * np.sin(2.0 * lon) * np.cos(lat)
    if name == "smooth2":
        return np.exp(0.5 * xyz[:, 0] - 0.25 * xyz[:, 1]) \
            + 0.10 * np.cos(3.0 * lon) * np.cos(lat) ** 2
    if name in {"highfreq", "hf"}:
        # A single high mode: spectrum concentrated at l=40 (validation: L* should be ~40).
        theta, phi = xyz_to_angles(xyz)
        return real_sph_harm(40, 13, theta, phi)
    if name == "bump":
        # Localized Gaussian cap -> broad spectrum (needs many orders). Width ~ 10 degrees.
        c = np.array([0.0, 0.0, 1.0])
        cosang = np.clip(xyz @ c, -1.0, 1.0)
        sigma = math.radians(10.0)
        ang = np.arccos(cosang)
        return np.exp(-0.5 * (ang / sigma) ** 2)
    if name.startswith("Y_") or name.startswith("Y:"):
        sep = "_" if name.startswith("Y_") else ":"
        _, l, m = name.split(sep)
        theta, phi = xyz_to_angles(xyz)
        return real_sph_harm(int(l), int(m), theta, phi)
    raise ValueError(f"Unknown field spec: {name!r}")


# ---------------------------------------------------------------------------
# Power spectrum + stopping criteria
# ---------------------------------------------------------------------------

def power_spectra_multi(xyz, area, fields: dict, lmax: int):
    """Angular power spectra C_l for many fields at once.

    Uses a fully-normalized associated-Legendre recurrence (Holmes & Featherstone 2002),
    vectorized over mesh points -- orders of magnitude faster than per-(l,m) scipy calls,
    and stable to high degree. The normalization is the orthonormal real-SH convention
    (int Y_lm^2 dOmega = 1), verified end-to-end by `--validate` (a pure Y_l field yields
    C_l/P ~ 1 at degree l).

    Returns {name: C_l array}, {name: total power P}, n_bad_modes.
    """
    theta, phi = xyz_to_angles(xyz)
    t = np.cos(theta)
    u = np.sin(theta)
    names = list(fields)
    af = {nm: area * fields[nm] for nm in names}
    P = {nm: float(np.sum(area * fields[nm] * fields[nm])) for nm in names}
    C = {nm: np.zeros(lmax + 1, dtype=np.float64) for nm in names}

    sqrt2 = math.sqrt(2.0)
    inv_sqrt_4pi = 1.0 / math.sqrt(4.0 * math.pi)

    def accumulate(l, Plm, cosm, sinm, is_sectoral_m0):
        if is_sectoral_m0:
            Y = Plm                                   # m = 0: single real term
            for nm in names:
                a = float(np.dot(af[nm], Y))
                C[nm][l] += a * a
        else:
            Yc = sqrt2 * Plm * cosm                   # +m term (cos)
            Ys = sqrt2 * Plm * sinm                   # -m term (sin)
            for nm in names:
                ac = float(np.dot(af[nm], Yc))
                as_ = float(np.dot(af[nm], Ys))
                C[nm][l] += ac * ac + as_ * as_

    pmm = np.full_like(t, inv_sqrt_4pi)               # normalized P~_0^0 = 1/sqrt(4 pi)
    for m in range(0, lmax + 1):
        if m > 0:
            pmm = math.sqrt((2.0 * m + 1.0) / (2.0 * m)) * u * pmm   # P~_m^m
            cosm = np.cos(m * phi)
            sinm = np.sin(m * phi)
        else:
            cosm = sinm = None

        # l = m
        accumulate(m, pmm, cosm, sinm, m == 0)

        if m + 1 > lmax:
            continue
        # l = m + 1
        Pl_1 = math.sqrt(2.0 * m + 3.0) * t * pmm     # P~_{m+1}^m
        accumulate(m + 1, Pl_1, cosm, sinm, m == 0)

        # l = m + 2 .. lmax  (two-term recurrence in l, fixed m)
        Pl_2 = pmm
        for l in range(m + 2, lmax + 1):
            a = math.sqrt((2.0 * l + 1.0) * (2.0 * l - 1.0) / ((l - m) * (l + m)))
            b = math.sqrt((2.0 * l + 1.0) * (l - m - 1.0) * (l + m - 1.0)
                          / ((2.0 * l - 3.0) * (l - m) * (l + m)))
            Pl = a * t * Pl_1 - b * Pl_2
            accumulate(l, Pl, cosm, sinm, m == 0)
            Pl_2, Pl_1 = Pl_1, Pl

    n_bad = 0  # recurrence is finite by construction; kept for output-schema compatibility
    return C, P, n_bad


def power_spectrum(xyz, area, f, lmax: int):
    """Single-field convenience wrapper around power_spectra_multi."""
    C, P, n_bad = power_spectra_multi(xyz, area, {"_": f}, lmax)
    return C["_"], P["_"], n_bad


def tail_error_curve(C, P):
    """E(L) = sqrt(max(P - sum_{l<=L} C_l, 0) / P) for L = 0..len(C)-1 (monotone non-increasing)."""
    if P <= 0:
        return np.zeros_like(C)
    captured = np.cumsum(C)
    tail = np.clip(P - captured, 0.0, None)
    return np.sqrt(tail / P)


def increment_curve(C, P, levels):
    """Cauchy increment sqrt(sum_{prev<l<=L} C_l / P) between successive probe levels."""
    out = {}
    prev = -1
    for L in levels:
        band = C[prev + 1: L + 1].sum() if L >= 0 else 0.0
        out[L] = math.sqrt(max(band, 0.0) / P) if P > 0 else 0.0
        prev = L
    return out


def recommend_tail(E, levels, tol):
    """Smallest probe level with tail error < tol (monotone -> no consecutive rule needed)."""
    for L in levels:
        if L < len(E) and E[L] < tol:
            return L
    return None


def recommend_increment(incr, levels, tol, consecutive):
    """Smallest probe level after which the increment stays < tol for `consecutive` levels."""
    vals = [(L, incr[L]) for L in levels]
    for i in range(len(vals)):
        window = vals[i: i + consecutive]
        if len(window) == consecutive and all(v < tol for _, v in window):
            return window[0][0]
    return None


WORKFLOW_TOL = {"viz": 1e-2, "ai": 1e-8, "simulation": 1e-12}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build_row(name, C, P, n_bad, lmax, levels, tols, consecutive):
    E = tail_error_curve(C, P)
    incr = increment_curve(C, P, levels)

    captured_frac = float(np.sum(C) / P) if P > 0 else float("nan")
    floor = float(E[min(lmax, len(E) - 1)])   # residual at the top trustworthy level

    row = {
        "field": name,
        "total_power": P,
        "captured_fraction_le_lmax": captured_frac,
        "achievable_floor": floor,
        "n_bad_modes": n_bad,
    }
    for L in levels:
        row[f"tail_E_L{L}"] = float(E[L]) if L < len(E) else float("nan")
        row[f"incr_L{L}"] = float(incr[L])
    for tol in tols:
        lt = recommend_tail(E, levels, tol)
        li = recommend_increment(incr, levels, tol, consecutive)
        # Honest reporting: if tol is below the achievable floor, say so.
        below_floor = tol < floor
        row[f"Lstar_tail_tol{tol:g}"] = (
            "below_floor" if (lt is None and below_floor) else (lt if lt is not None else ">lmax")
        )
        row[f"Lstar_incr_tol{tol:g}"] = (
            "below_floor" if (li is None and below_floor) else (li if li is not None else ">lmax")
        )
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--edge-parquet", help="edge-dataset parquet to read the mesh from")
    src.add_argument("--synthetic-mesh", type=int, help="N near-uniform Fibonacci points (equal-area)")
    ap.add_argument("--side", choices=["source", "target"], default="source")
    ap.add_argument("--fields", nargs="+",
                    default=["x", "y", "z", "smooth1", "smooth2", "highfreq", "bump"])
    ap.add_argument("--levels", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    ap.add_argument("--tolerances", type=float, nargs="+", default=[1e-2, 1e-4, 1e-8, 1e-12])
    ap.add_argument("--consecutive", type=int, default=2)
    ap.add_argument("--lmax", type=int, default=None, help="override max degree (else max(levels), Nyquist-capped)")
    ap.add_argument("--out", default=None, help="CSV path for the per-field summary")
    ap.add_argument("--spectrum-out", default=None, help="CSV path for the full C_l spectrum")
    ap.add_argument("--validate", action="store_true",
                    help="self-test: pure Y_l fields on a Fibonacci mesh must recover L*=l")
    args = ap.parse_args()

    if args.validate:
        return run_validation()

    if not args.edge_parquet and not args.synthetic_mesh:
        ap.error("one of --edge-parquet / --synthetic-mesh is required (or use --validate)")

    if args.edge_parquet:
        xyz, area = load_mesh_from_parquet(Path(args.edge_parquet), args.side)
        mesh_name = Path(args.edge_parquet).stem + f"[{args.side}]"
    else:
        xyz, area = fibonacci_sphere(args.synthetic_mesh)
        mesh_name = f"fibonacci_{args.synthetic_mesh}"

    n = xyz.shape[0]
    nyquist = max(1, int(math.sqrt(n)) - 1)
    requested = args.lmax if args.lmax is not None else max(args.levels)
    lmax = min(requested, nyquist)

    print(f"mesh:            {mesh_name}")
    print(f"cells:           {n:,}")
    print(f"Nyquist order:   ~{nyquist}  (cannot resolve orders >> this)")
    print(f"lmax used:       {lmax}" + ("" if lmax == requested else f"  (capped from {requested})"))
    levels = [L for L in args.levels if L <= lmax]
    dropped = [L for L in args.levels if L > lmax]
    if dropped:
        print(f"WARNING: probe levels above Nyquist dropped (mesh too coarse to resolve them): {dropped}")
    if not levels:
        print("ERROR: no probe levels are resolvable on this mesh; use a finer mesh or lower --levels.")
        sys.exit(2)
    print(f"probe levels:    {levels}")
    print(f"tolerances:      {args.tolerances}")
    print()

    field_arrays = {name: analytic_field(name, xyz) for name in args.fields}
    C_all, P_all, n_bad = power_spectra_multi(xyz, area, field_arrays, lmax)
    if n_bad:
        print(f"NOTE: {n_bad} (l,m) modes returned non-finite values at high degree and were skipped.")

    rows, spec_rows = [], []
    for name in args.fields:
        row = build_row(name, C_all[name], P_all[name], n_bad, lmax, levels, args.tolerances, args.consecutive)
        row = {"mesh": mesh_name, "n_cells": n, "nyquist": nyquist, **row}
        rows.append(row)
        for l, cl in enumerate(C_all[name]):
            spec_rows.append({"mesh": mesh_name, "field": name, "l": l, "C_l": float(cl)})
        cf = row["captured_fraction_le_lmax"]
        flag = "" if cf >= 0.99 else f"  [UNDER-RESOLVED: only {cf:.3f} of energy <= lmax]"
        print(f"  {name:10}  floor~{row['achievable_floor']:.2e}  "
              + "  ".join(f"E(L{L})={row[f'tail_E_L{L}']:.2e}" for L in levels) + flag)

    df = pd.DataFrame(rows)
    out = Path(args.out) if args.out else Path("analysis_medium_improv/github_results") / f"adaptive_spectral_{mesh_name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nWrote summary:  {out}")

    if args.spectrum_out:
        sp = Path(args.spectrum_out)
        sp.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(spec_rows).to_csv(sp, index=False)
        print(f"Wrote spectrum: {sp}")

    print("\nL* (smallest order meeting tolerance, tail criterion):")
    for name in args.fields:
        r = next(x for x in rows if x["field"] == name)
        cells = "  ".join(f"{t:g}:{r[f'Lstar_tail_tol{t:g}']}" for t in args.tolerances)
        print(f"  {name:10}  {cells}")


def run_validation():
    """Pure Y_l^m fields on a Fibonacci mesh: tail criterion at tol=1e-2 must recover L*=l."""
    print("=== validation: pure Y_l^m must read L* = l (Fibonacci mesh, area quadrature) ===")
    n = 20000
    xyz, area = fibonacci_sphere(n)
    nyquist = int(math.sqrt(n)) - 1
    cases = [(2, 1), (5, -3), (8, 0), (16, 7), (24, -11), (32, 5)]
    levels = list(range(0, 41))  # fine probe so L* is exact-ish
    tol = 1e-2
    npass = 0
    for (l, m) in cases:
        f = analytic_field(f"Y_{l}_{m}", xyz)
        lmax = min(max(l + 8, 40), nyquist)
        C, P, _ = power_spectrum(xyz, area, f, lmax)
        E = tail_error_curve(C, P)
        probe = [L for L in levels if L <= lmax]
        lt = recommend_tail(E, probe, tol)
        # Energy should be concentrated at degree l: C[l]/P ~ 1.
        conc = float(C[l] / P) if P > 0 else 0.0
        ok = (lt is not None) and (lt >= l - 1) and (lt <= l + 1) and (conc > 0.9)
        npass += int(ok)
        print(f"  Y_{l}_{m:<3}  L*(tol={tol:g})={lt}  C_l/P={conc:.4f}  floor={E[min(lmax,len(E)-1)]:.2e}  "
              + ("PASS" if ok else "FAIL"))
    print(f"\n{npass}/{len(cases)} cases passed.")
    sys.exit(0 if npass == len(cases) else 1)


if __name__ == "__main__":
    main()
