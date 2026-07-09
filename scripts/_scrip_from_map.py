"""Extract a per-grid SCRIP file from a TempestRemap conserve map (for ESMF baselines).

Usage: python _scrip_from_map.py <map_conserve.nc> <side a|b> <out.scrip.nc>

Reads xc/yc (centers, deg), xv/yv (corners, deg), area from the map; writes a SCRIP grid
(grid_center_lat/lon, grid_corner_lat/lon, grid_imask, grid_dims).  Uses fv_moments to detect the
true valid-corner count per cell and pads unused corner slots by REPEATING the last valid corner
(not the literal (0,0), which would inject a spurious equator point) — so ESMF sees clean polygons
with only zero-length degenerate edges (handled by --ignore_degenerate).
"""
import sys
from pathlib import Path
import numpy as np
import xarray as xr
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from remapgnn import fv_moments as fv

map_path, side, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
ds = xr.open_dataset(map_path)
xv = np.asarray(ds[f"xv_{side}"].values, dtype=np.float64)   # [N, nv] deg
yv = np.asarray(ds[f"yv_{side}"].values, dtype=np.float64)
xc = np.asarray(ds[f"xc_{side}"].values, dtype=np.float64)   # [N] deg
yc = np.asarray(ds[f"yc_{side}"].values, dtype=np.float64)
ds.close()
N, nv = xv.shape

# valid-corner count per cell (robust area-based padding detection, shared with the FV path)
_, nvalid, _, _ = fv.load_corners_from_map(map_path, side)

# pad unused trailing corners by repeating the last valid corner (index = min(k, nvalid-1))
k = np.arange(nv)[None, :]
idx = np.minimum(k, (nvalid - 1)[:, None])                   # [N, nv]
rows = np.arange(N)[:, None]
xv_pad = xv[rows, idx]
yv_pad = yv[rows, idx]

nc = Dataset(out_path, "w", format="NETCDF3_CLASSIC")
nc.createDimension("grid_size", N)
nc.createDimension("grid_corners", nv)
nc.createDimension("grid_rank", 1)
v = nc.createVariable("grid_dims", "i4", ("grid_rank",)); v[:] = np.array([N], dtype=np.int32)
v = nc.createVariable("grid_imask", "i4", ("grid_size",)); v[:] = np.ones(N, dtype=np.int32); v.units = "unitless"
v = nc.createVariable("grid_center_lat", "f8", ("grid_size",)); v[:] = yc; v.units = "degrees"
v = nc.createVariable("grid_center_lon", "f8", ("grid_size",)); v[:] = xc; v.units = "degrees"
v = nc.createVariable("grid_corner_lat", "f8", ("grid_size", "grid_corners")); v[:] = yv_pad; v.units = "degrees"
v = nc.createVariable("grid_corner_lon", "f8", ("grid_size", "grid_corners")); v[:] = xv_pad; v.units = "degrees"
nc.title = f"SCRIP grid from {Path(map_path).name} side {side}"
nc.close()
print(f"wrote {out_path}  N={N} nv={nv} nvalid_hist={dict(zip(*[a.tolist() for a in np.unique(nvalid, return_counts=True)]))}")
