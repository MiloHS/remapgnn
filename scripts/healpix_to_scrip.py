"""Generate a HEALPix grid as a SCRIP mesh file (unit sphere) that TempestRemap can overlap.
HEALPix = equal-area, isolatitude curvilinear-quad pixelization. Usage: <nside> <out.nc>
npix = 12*nside^2 (nside power of 2). Corners from hp.boundaries (4 per pixel)."""
import sys
import numpy as np
import healpy as hp
from netCDF4 import Dataset

nside = int(sys.argv[1]); out = sys.argv[2]
npix = hp.nside2npix(nside)
pix = np.arange(npix)

theta, phi = hp.pix2ang(nside, pix)          # colatitude [0,pi], longitude [0,2pi)
clat = np.pi / 2.0 - theta
clon = phi

b = hp.boundaries(nside, pix, step=1)         # (npix, 3, 4) cartesian corners, CCW
x, y, z = b[:, 0, :], b[:, 1, :], b[:, 2, :]  # each (npix, 4)
crlat = np.arcsin(np.clip(z, -1.0, 1.0))
crlon = np.mod(np.arctan2(y, x), 2.0 * np.pi)

ds = Dataset(out, "w", format="NETCDF3_CLASSIC")
ds.createDimension("grid_size", npix)
ds.createDimension("grid_corners", 4)
ds.createDimension("grid_rank", 1)
v = ds.createVariable("grid_dims", "i4", ("grid_rank",)); v[:] = [npix]
v = ds.createVariable("grid_imask", "i4", ("grid_size",)); v[:] = 1
v = ds.createVariable("grid_center_lat", "f8", ("grid_size",)); v.units = "radians"; v[:] = clat
v = ds.createVariable("grid_center_lon", "f8", ("grid_size",)); v.units = "radians"; v[:] = clon
v = ds.createVariable("grid_corner_lat", "f8", ("grid_size", "grid_corners")); v.units = "radians"; v[:] = crlat
v = ds.createVariable("grid_corner_lon", "f8", ("grid_size", "grid_corners")); v.units = "radians"; v[:] = crlon
ds.title = "HEALPix nside=%d (npix=%d)" % (nside, npix)
ds.close()
print("wrote %s  nside=%d npix=%d" % (out, nside, npix))
