#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def first_existing(ds, names):
    for name in names:
        if name in ds:
            return name
    return None


def to_rad_if_needed(x):
    x = np.asarray(x, dtype=np.float64)
    finite = x[np.isfinite(x)]
    if finite.size and np.nanmax(np.abs(finite)) > 2.0 * np.pi + 1e-6:
        return np.deg2rad(x)
    return x


def read_lon_lat(ds, mesh_path=None, n=None):
    lon_name = first_existing(ds, ["lon", "longitude", "lonCell", "xlon"])
    lat_name = first_existing(ds, ["lat", "latitude", "latCell", "ylat"])

    if lon_name is not None and lat_name is not None:
        lon = np.asarray(ds[lon_name].values).reshape(-1)
        lat = np.asarray(ds[lat_name].values).reshape(-1)
    elif mesh_path is not None:
        m = xr.open_dataset(mesh_path)
        lon, lat = read_lon_lat(m, None, n)
        m.close()
        return lon, lat
    else:
        raise KeyError("Could not find lon/lat in prediction file. Pass --target-mesh-nc.")

    lon = to_rad_if_needed(lon)
    lat = to_rad_if_needed(lat)

    lon_deg = np.rad2deg(lon)
    lat_deg = np.rad2deg(lat)
    lon_deg = ((lon_deg + 180.0) % 360.0) - 180.0

    if n is not None and lon_deg.size != n:
        raise ValueError(f"lon/lat size {lon_deg.size} does not match field size {n}")

    return lon_deg, lat_deg


def read_field(path, field):
    ds = xr.open_dataset(path)
    if field not in ds:
        raise KeyError(f"{field} not found in {path}. Available: {list(ds.data_vars)}")
    values = np.asarray(ds[field].values, dtype=np.float64).reshape(-1)
    return ds, values


def scatter_panel(ax, lon, lat, values, title, vmin=None, vmax=None, cmap=None):
    sc = ax.scatter(
        lon, lat, c=values, s=0.7, linewidths=0,
        rasterized=True, vmin=vmin, vmax=vmax, cmap=cmap
    )
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_title(title)
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-nc", required=True)
    ap.add_argument("--field", required=True)
    ap.add_argument("--target-mesh-nc", default=None)
    ap.add_argument("--truth-nc", default=None)
    ap.add_argument("--truth-field", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pred_ds, pred = read_field(args.pred_nc, args.field)
    lon, lat = read_lon_lat(pred_ds, args.target_mesh_nc, len(pred))

    if args.truth_nc:
        truth_field = args.truth_field or args.field
        truth_ds, truth = read_field(args.truth_nc, truth_field)
        truth_ds.close()

        if truth.size != pred.size:
            raise ValueError(f"truth size {truth.size} != pred size {pred.size}")

        err = pred - truth
        fvals = np.concatenate([pred, truth])
        vmin = np.nanpercentile(fvals, 1)
        vmax = np.nanpercentile(fvals, 99)
        emax = np.nanpercentile(np.abs(err), 99.5)
        if emax == 0 or not np.isfinite(emax):
            emax = 1.0

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
        sc = scatter_panel(axes[0], lon, lat, pred, "Prediction", vmin=vmin, vmax=vmax)
        fig.colorbar(sc, ax=axes[0], shrink=0.85)
        sc = scatter_panel(axes[1], lon, lat, truth, "Truth", vmin=vmin, vmax=vmax)
        fig.colorbar(sc, ax=axes[1], shrink=0.85)
        sc = scatter_panel(axes[2], lon, lat, err, "Prediction - truth", vmin=-emax, vmax=emax, cmap="coolwarm")
        fig.colorbar(sc, ax=axes[2], shrink=0.85)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4.5), constrained_layout=True)
        sc = scatter_panel(ax, lon, lat, pred, "Prediction")
        fig.colorbar(sc, ax=ax, shrink=0.85)

    fig.suptitle(f"{Path(args.pred_nc).name}: {args.field}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    pred_ds.close()

    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
