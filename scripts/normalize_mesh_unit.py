"""Project an Exodus/SCRIP mesh's node coordinates onto the unit sphere (the MIRA RegionallyRefined
MPAS / refined-CS meshes store coords in km ~6371). Preserves angular positions + connectivity so
TempestRemap can overlap them with the unit-sphere CS/ICOD/RLL meshes. Usage: src.nc dst.nc"""
import sys, shutil
import numpy as np
from netCDF4 import Dataset

src, dst = sys.argv[1], sys.argv[2]
shutil.copy(src, dst)
d = Dataset(dst, "r+")
if "coord" in d.variables:
    c = np.array(d.variables["coord"][:])
    if c.shape[0] == 3:                       # (3, N)
        n = np.sqrt((c ** 2).sum(0)); c = c / n[None, :]
    else:                                     # (N, 3)
        n = np.sqrt((c ** 2).sum(1)); c = c / n[:, None]
    d.variables["coord"][:] = c
    rad = float(n.mean())
elif all(k in d.variables for k in ("coordx", "coordy", "coordz")):
    cx = np.array(d.variables["coordx"][:]); cy = np.array(d.variables["coordy"][:]); cz = np.array(d.variables["coordz"][:])
    n = np.sqrt(cx ** 2 + cy ** 2 + cz ** 2)
    d.variables["coordx"][:] = cx / n; d.variables["coordy"][:] = cy / n; d.variables["coordz"][:] = cz / n
    rad = float(n.mean())
else:
    raise RuntimeError("no coord/coordx vars found: %s" % list(d.variables.keys())[:20])
d.close()
print("normalized %s -> %s  (mean radius before = %.4g)" % (src.split("/")[-1], dst, rad))
