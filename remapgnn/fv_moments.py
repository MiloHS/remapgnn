"""Finite-volume (FV) cell-average geometric moments for supermesh-free remapping.

For a conservative finite-volume remap, the operator should reproduce the CELL-AVERAGE
of low-degree polynomials, not their point value at the cell center.  The audit/training
code historically used monomials evaluated at cell centers (point value); this module
computes the true cell averages

    <f>_cell = (1 / A_cell) * integral_cell f dOmega

by spherical quadrature over each cell polygon, using ONLY each cell's own vertices
(no overlap supermesh).  A validated kill-test (scripts/_fv_killtest.py) showed swapping
point -> FV degree-2 moments closes ~half the gap to TempestRemap np2 on the cell-average
metric.

Cell vertices come from the SCRIP TempestRemap map files (maps_medium_improv/
map_<pair>_conserve.nc), which carry per-grid corner arrays xv_a/yv_a (source) and
xv_b/yv_b (target) in DEGREES, plus areas in steradians.  Cell ordering in those files
matches the edge-dataset ordering (source_index=col-1, target_index=row-1), so the
resulting moment arrays index 1:1 with the operator's cells.

Handles general n-gons:
  * CS (nv=4), ICOD (nv=6), ICO (nv=3): clean polygons;
  * MPAS (nv=7): variable 5/6/7-gons padded with TRAILING literal (0,0) -- stripped;
  * RLL (nv=4): pole-cap cells have a zero-length pole edge -> that fan triangle is
    degenerate (zero solid angle) and drops out naturally.

Degree-2 is the current target (coord [N,3], quad [N,6]); a degree-3 hook (cubic [N,10])
is provided but off by default (deferred per plan).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import xarray as xr

# monomial layouts (kept explicit so callers/consumers agree on column order)
QUAD_TERMS = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]              # xx xy xz yy yz zz
CUBIC_TERMS = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1), (0, 1, 2),
               (0, 2, 2), (1, 1, 1), (1, 1, 2), (1, 2, 2), (2, 2, 2)]     # xxx xxy ... zzz

_DEFAULT_M = 8
_PAD_TOL = 1e-12          # trailing (0,0) padding detector (deg): exact zeros in real files
_DEGEN_AREA = 1e-14       # solid-angle threshold below which a fan triangle is dropped


# ----------------------------------------------------------------- geometry ---
def ll2xyz(lon_deg, lat_deg):
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    return np.stack([np.cos(lat) * np.cos(lon),
                     np.cos(lat) * np.sin(lon),
                     np.sin(lat)], axis=-1)


def _normalize(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, 1e-300, None)


def _solid_angle(a, b, c):
    """Van Oosterom-Strackee solid angle of spherical triangle(s); a,b,c [...,3] unit."""
    triple = np.abs(np.einsum("...i,...i->...", a, np.cross(b, c)))
    denom = (1.0
             + np.einsum("...i,...i->...", a, b)
             + np.einsum("...i,...i->...", a, c)
             + np.einsum("...i,...i->...", b, c))
    return 2.0 * np.arctan2(triple, denom)


def _ref_small_triangles(m):
    """Barycentric corner-triples for m^2 small triangles tiling the unit triangle."""
    out = []
    for i in range(m):
        for j in range(m - i):
            nodes = [((i, j), (i + 1, j), (i, j + 1))]
            if j < m - i - 1:
                nodes.append(((i + 1, j), (i, j + 1), (i + 1, j + 1)))
            for tri in nodes:
                bary = []
                for (a, b) in tri:
                    wa, wb = a / m, b / m
                    bary.append((wa, wb, 1.0 - wa - wb))
                out.append(tuple(bary))
    return out


def _fan_area(V, nvalid):
    """Per-cell spherical-polygon solid angle via centroid fan (coarse, for disambiguation)."""
    N, nv, _ = V.shape
    slot = np.arange(nv)[None, :]
    valid = slot < nvalid[:, None]
    center = _normalize((V * valid[..., None]).sum(1) / np.clip(valid.sum(1, keepdims=True), 1, None))
    area = np.zeros(N)
    for k in range(nv):
        knext = (k + 1) % nv
        C = np.where((knext >= nvalid)[:, None], V[:, 0, :], V[:, knext, :])
        a = _solid_angle(center, V[:, k, :], C)
        area += np.where(k < nvalid, a, 0.0)
    return area


def _strip_padding(xv, yv, centers, cell_area):
    """Return per-cell corners + valid-corner count, stripping TRAILING (0,0) padding.

    xv,yv are [N,nv] in degrees.  Both MPAS (mixed 5/6/7-gons in nv=7) and ICOD (12
    pentagons in nv=6) pad unused TRAILING slots with literal (0,0), which maps to the
    point (1,0,0) -- NOT the origin.  (0,0) is a legal corner, so a trailing (0,0) is
    disambiguated by the map cell AREA: strip it only if the stripped polygon's area
    matches the map area better than keeping it.  Robust for real padding (kept ->
    spurious huge fan area) AND a genuine (0,0) corner (kept -> correct area), including
    cells physically near (1,0,0).  Returns (corners_xyz [N,nv,3], nvalid [N]).
    """
    N, nv = xv.shape
    V = ll2xyz(xv, yv)                                  # [N,nv,3]
    is_zero = (np.abs(xv) <= _PAD_TOL) & (np.abs(yv) <= _PAD_TOL)   # [N,nv]
    rev = is_zero[:, ::-1]
    trailing = np.cumprod(rev.astype(np.int64), axis=1)    # trailing exact-(0,0) run
    nvalid_strip = np.clip(nv - trailing.sum(axis=1), 3, nv)
    nv_full = np.full(N, nv, dtype=np.int64)
    has_tz = nvalid_strip < nv
    if not has_tz.any():
        return V, nv_full
    a_strip = _fan_area(V, nvalid_strip)
    a_full = _fan_area(V, nv_full)
    use_strip = np.abs(a_strip - cell_area) <= np.abs(a_full - cell_area)
    nvalid = np.where(has_tz & use_strip, nvalid_strip, nv_full)
    return V, nvalid


def build_quadrature(V, nvalid, m=_DEFAULT_M):
    """Spherical quadrature points/weights for general n-gon cells.

    V [N,nv,3] unit corner vectors (trailing slots ignored per nvalid [N]).  Fan each
    cell from its centroid into `nvalid` triangles, subdivide each into m^2 small
    triangles (Van Oosterom-Strackee solid-angle weights, centroid rule).  Degenerate
    triangles (pole edges, padding) contribute ~0 and are dropped.

    Returns (points [P,3], weights [P], cell_idx [P], cellA [N]).
    """
    N, nv, _ = V.shape
    ref = _ref_small_triangles(m)
    # per-cell centroid over VALID corners only
    slot = np.arange(nv)[None, :]                           # [1,nv]
    valid = slot < nvalid[:, None]                          # [N,nv]
    Vw = V * valid[..., None]
    center = _normalize(Vw.sum(axis=1) / np.clip(valid.sum(1, keepdims=True), 1, None))
    cell_arange = np.arange(N)

    pts, wts, idx = [], [], []
    for k in range(nv):
        # fan triangle (center, V_k, V_{k+1 within valid ring})
        knext = (k + 1) % nv
        A = center
        B = V[:, k, :]
        C = V[:, knext, :]
        # this fan triangle is active only when both k and knext are valid corners;
        # when knext wraps past nvalid, close the ring to corner 0
        active_k = k < nvalid
        wrap = (knext >= nvalid)                            # need to close ring to 0
        C = np.where(wrap[:, None], V[:, 0, :], C)
        active = active_k                                   # k valid => triangle exists
        for (ba, bb, bc) in ref:
            P1 = _normalize(ba[0] * A + ba[1] * B + ba[2] * C)
            P2 = _normalize(bb[0] * A + bb[1] * B + bb[2] * C)
            P3 = _normalize(bc[0] * A + bc[1] * B + bc[2] * C)
            cent = _normalize(P1 + P2 + P3)
            area = _solid_angle(P1, P2, P3)                 # [N]
            area = np.where(active, area, 0.0)
            pts.append(cent)
            wts.append(area)
            idx.append(cell_arange)
    points = np.concatenate(pts, axis=0)
    weights = np.concatenate(wts, axis=0)
    cell_idx = np.concatenate(idx, axis=0)
    # drop exactly-zero-weight quadrature points (degenerate) to save field evals
    keep = weights > _DEGEN_AREA
    points, weights, cell_idx = points[keep], weights[keep], cell_idx[keep]
    cellA = np.bincount(cell_idx, weights=weights, minlength=N)
    return points, weights, cell_idx, cellA


def cell_average(field_fn, points, weights, cell_idx, cellA):
    """Area-weighted cell average of field_fn(points[P,3])->[P]. Returns [N]."""
    fv = np.asarray(field_fn(points), dtype=np.float64)
    num = np.bincount(cell_idx, weights=fv * weights, minlength=cellA.shape[0])
    return num / np.clip(cellA, 1e-300, None)


# ------------------------------------------------------------- moment arrays ---
def _monomial_fn(idxs):
    if len(idxs) == 1:
        d, = idxs
        return lambda xyz: xyz[:, d]
    if len(idxs) == 2:
        a, b = idxs
        return lambda xyz: xyz[:, a] * xyz[:, b]
    a, b, c = idxs
    return lambda xyz: xyz[:, a] * xyz[:, b] * xyz[:, c]


def compute_grid_moments(V, nvalid, m=_DEFAULT_M, cubic=False):
    """Cell-average coordinate/quadratic (/cubic) moments for one grid.

    Returns dict: coord [N,3], quad [N,6], cellA [N], (cubic [N,10] if cubic).
    """
    points, weights, cell_idx, cellA = build_quadrature(V, nvalid, m=m)
    coord = np.stack([cell_average(_monomial_fn((d,)), points, weights, cell_idx, cellA)
                      for d in range(3)], axis=1)
    quad = np.stack([cell_average(_monomial_fn(t), points, weights, cell_idx, cellA)
                     for t in QUAD_TERMS], axis=1)
    out = {"coord": coord, "quad": quad, "cellA": cellA}
    if cubic:
        out["cubic"] = np.stack(
            [cell_average(_monomial_fn(t), points, weights, cell_idx, cellA)
             for t in CUBIC_TERMS], axis=1)
    return out


# ------------------------------------------------------------ map file access ---
def load_corners_from_map(map_path, side):
    """side in {'a' (source), 'b' (target)}. Returns (V [N,nv,3], nvalid [N],
    area [N], centers [N,3]) from a SCRIP TempestRemap conserve map file."""
    ds = xr.open_dataset(map_path)
    xv = ds[f"xv_{side}"].values.astype(np.float64)      # [N,nv] deg
    yv = ds[f"yv_{side}"].values.astype(np.float64)
    area = ds[f"area_{side}"].values.astype(np.float64)
    xc = ds[f"xc_{side}"].values.astype(np.float64)
    yc = ds[f"yc_{side}"].values.astype(np.float64)
    ds.close()
    centers = ll2xyz(xc, yc)
    V, nvalid = _strip_padding(xv, yv, centers, area)
    return V, nvalid, area, centers


# ----------------------------------------------------------------- caching ----
def grid_moment_cache(map_path, side, grid_name, cache_dir, m=_DEFAULT_M,
                      cubic=False, expected_centers=None, verbose=True):
    """Load-or-compute cell-average moments for one grid, cached to an .npz sidecar
    keyed by grid+m.  If expected_centers [N,3] is given (from the edge dataset),
    asserts cell-ordering alignment before trusting the map corners.

    Returns dict with coord/quad/cellA (+cubic), plus 'area','centers'."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_cub" if cubic else ""
    path = cache_dir / f"fvmoments_{grid_name}_m{m}{suffix}.npz"

    V, nvalid, area, centers = load_corners_from_map(map_path, side)
    if expected_centers is not None:
        d = float(np.abs(centers - expected_centers).max())
        if d > 1e-6:
            raise ValueError(
                f"cell-ordering mismatch for grid {grid_name} (side {side}): "
                f"max|map_center - edge_center| = {d:.3e} -> ABORT")

    if path.exists():
        z = np.load(path)
        if int(z.get("m", -1)) == m and z["coord"].shape[0] == V.shape[0]:
            out = {k: z[k] for k in z.files if k not in ("m",)}
            out["area"] = area
            out["centers"] = centers
            if verbose:
                print(f"  [fv] loaded cache {path.name}")
            return out

    mom = compute_grid_moments(V, nvalid, m=m, cubic=cubic)
    save = {"coord": mom["coord"], "quad": mom["quad"], "cellA": mom["cellA"],
            "m": np.int64(m)}
    if cubic:
        save["cubic"] = mom["cubic"]
    np.savez(path, **save)
    if verbose:
        print(f"  [fv] wrote cache {path.name}")
    mom["area"] = area
    mom["centers"] = centers
    return mom


def real_sph_unnorm_fast(l, m, xyz):
    """Fast vectorized real spherical harmonic matching train_config_balanced_harmonic.
    _real_sph_unnorm UP TO A PER-(l,m) CONSTANT (verified in scripts/_verify_fastsph.py).

    Uses the fully-normalized associated-Legendre recurrence (Holmes & Featherstone 2002),
    vectorized over all points at once and free of factorials/overflow -- vs scipy sph_harm_y
    called per (l,m).  Because every consumer normalizes each single-(l,m) field by its own RMS
    and the audit metric (area_rel_l2) is a ratio, a constant per-mode scale is irrelevant, so
    this is a drop-in replacement for the cell-average TRUTH path with identical metric values.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    r = np.sqrt((xyz * xyz).sum(axis=1))
    x = np.clip(xyz[:, 2] / np.maximum(r, 1e-30), -1.0, 1.0)      # cos(theta)
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    am = abs(int(m))
    u = np.sqrt(np.clip(1.0 - x * x, 0.0, None))                  # sin(theta)
    # sectoral: \bar P_{k,k}
    pmm = np.ones_like(x)
    for k in range(1, am + 1):
        pmm = u * np.sqrt((2.0 * k + 1.0) / (2.0 * k)) * pmm
    if l == am:
        plm = pmm
    else:
        pmmp1 = x * np.sqrt(2.0 * am + 3.0) * pmm                 # \bar P_{am+1,am}
        if l == am + 1:
            plm = pmmp1
        else:
            p2, p1 = pmm, pmmp1
            for n in range(am + 2, l + 1):
                a = np.sqrt((2.0 * n - 1.0) * (2.0 * n + 1.0) / ((n - am) * (n + am)))
                b = np.sqrt((2.0 * n + 1.0) * (n + am - 1.0) * (n - am - 1.0)
                            / ((2.0 * n - 3.0) * (n - am) * (n + am)))
                p = x * a * p1 - b * p2
                p2, p1 = p1, p
            plm = p1
    if m == 0:
        y = plm
    elif m > 0:
        y = np.sqrt(2.0) * plm * np.cos(m * phi)
    else:
        y = np.sqrt(2.0) * plm * np.sin(am * phi)
    return y.astype(np.float64)


def grid_quadrature(map_path, side, m=_DEFAULT_M, expected_centers=None):
    """Quadrature arrays for one grid, for cell-averaging ARBITRARY fields (eval truth).
    Returns dict points/weights/cell_idx/cellA/area/centers.  If expected_centers [N,3]
    is given, asserts map-order == edge-order before trusting corners."""
    V, nvalid, area, centers = load_corners_from_map(map_path, side)
    if expected_centers is not None:
        d = float(np.abs(centers - expected_centers).max())
        if d > 1e-6:
            raise ValueError(f"cell-ordering mismatch (side {side}): max|Δcenter|={d:.3e}")
    pts, w, ci, cA = build_quadrature(V, nvalid, m=m)
    return {"points": pts, "weights": w, "cell_idx": ci, "cellA": cA,
            "area": area, "centers": centers}


def grid_cell_average(field_fn, q):
    """Cell-average of field_fn(points[P,3])->[P] over a grid_quadrature() dict."""
    return cell_average(field_fn, q["points"], q["weights"], q["cell_idx"], q["cellA"])


def moment_coefs(mom_src, mom_tgt, src_index, tgt_index, cubic=False):
    """Per-edge FV moment coefficients (source-cell avg minus target-cell avg),
    matching the convention of quadratic_moment_coef.  src_index/tgt_index are the
    edge-dataset cell indices.  Returns (mc1 [E,3], mc2 [E,6], (mc3 [E,10] if cubic))."""
    mc1 = mom_src["coord"][src_index] - mom_tgt["coord"][tgt_index]
    mc2 = mom_src["quad"][src_index] - mom_tgt["quad"][tgt_index]
    if cubic:
        mc3 = mom_src["cubic"][src_index] - mom_tgt["cubic"][tgt_index]
        return mc1, mc2, mc3
    return mc1, mc2


# ------------------------------------------------------------- validation -----
def validate_grid(map_path, side, name, m=_DEFAULT_M):
    """Correctness gates for one grid: area, <1>=1, <x>=centroid drift, degeneracy.
    Returns dict of diagnostics; raises on hard failures."""
    V, nvalid, area, centers = load_corners_from_map(map_path, side)
    points, weights, cell_idx, cellA = build_quadrature(V, nvalid, m=m)
    N = V.shape[0]
    const = cell_average(lambda p: np.ones(p.shape[0]), points, weights, cell_idx, cellA)
    coord = np.stack([cell_average(_monomial_fn((d,)), points, weights, cell_idx, cellA)
                      for d in range(3)], axis=1)
    coord_unit = _normalize(coord)                          # centroid direction
    area_rel = float(np.abs(cellA - area).max() / area.max())
    const_err = float(np.abs(const - 1.0).max())
    # centroid should lie ~on-sphere near cell center; drift = angle(centroid, center)
    cos = np.clip(np.einsum("ij,ij->i", coord_unit, centers), -1, 1)
    centroid_drift = float(np.max(np.arccos(cos)))          # radians
    nv_hist = {int(k): int(v) for k, v in zip(*np.unique(nvalid, return_counts=True))}
    diag = dict(name=name, N=N, area_rel=area_rel, const_err=const_err,
                centroid_drift_rad=centroid_drift, area_sum=float(cellA.sum()),
                nv_hist=nv_hist)
    return diag


if __name__ == "__main__":
    # Per-family validation gate (G0).  Run one map per family covering all nv types.
    import sys
    repo = Path(__file__).resolve().parent.parent
    os.chdir(repo)
    MAPS = repo / "maps_medium_improv"
    # (map file, side, family label) -- pick maps exercising each family/nv
    CASES = [
        (MAPS / "map_CS-r32_to_ICOD-r32_conserve.nc", "a", "CS(nv4)"),
        (MAPS / "map_CS-r32_to_ICOD-r32_conserve.nc", "b", "ICOD(nv6)"),
        (MAPS / "map_CS-r32_to_RLL-r90-180_conserve.nc", "b", "RLL(nv4,poles)"),
        (MAPS / "map_MPAS-r4_to_CS-r32_conserve.nc", "a", "MPAS(nv7,pad)"),
        (MAPS / "map_ICOD-r32_to_ICO-r32_conserve.nc", "b", "ICO(nv3)"),
    ]
    m = int(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_M
    print(f"=== FV geometry validation (m={m}) ===")
    ok = True
    for mp, side, name in CASES:
        if not mp.exists():
            print(f"  MISSING map: {mp.name} ({name}) -- SKIP")
            ok = False
            continue
        d = validate_grid(mp, side, name, m=m)
        area_ok = d["area_rel"] < 1e-6
        const_ok = d["const_err"] < 1e-9
        drift_ok = d["centroid_drift_rad"] < 5e-2          # cells are small; centroid near center
        flag = "OK " if (area_ok and const_ok and drift_ok) else "FAIL"
        if flag == "FAIL":
            ok = False
        print(f"  [{flag}] {name:16s} N={d['N']:6d} area_rel={d['area_rel']:.2e} "
              f"const_err={d['const_err']:.2e} centroid_drift={d['centroid_drift_rad']:.2e} "
              f"nv={d['nv_hist']}")
    print("G0:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
