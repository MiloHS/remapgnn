#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTDIR = Path("analysis_medium_improv/github_results")
OUTDIR.mkdir(parents=True, exist_ok=True)


def lonlat_to_xyz(lon, lat):
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return x, y, z


def smooth1(lon, lat):
    x, y, z = lonlat_to_xyz(lon, lat)
    return (
        1.0
        + 0.25 * x
        - 0.15 * y
        + 0.10 * z
        + 0.20 * np.sin(2.0 * lon) * np.cos(lat)
    )


def smooth2(lon, lat):
    x, y, z = lonlat_to_xyz(lon, lat)
    return (
        np.exp(0.5 * x - 0.25 * y)
        + 0.10 * np.cos(3.0 * lon) * np.cos(lat) ** 2
    )


def make_global_grid(nlon=721, nlat=361):
    lon_deg = np.linspace(-180.0, 180.0, nlon)
    lat_deg = np.linspace(-90.0, 90.0, nlat)
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)
    LON, LAT = np.meshgrid(lon, lat)
    return lon_deg, lat_deg, LON, LAT


def plot_global(func, name):
    lon_deg, lat_deg, LON, LAT = make_global_grid()
    vals = func(LON, LAT)

    plt.figure(figsize=(9.0, 4.6))
    im = plt.imshow(
        vals,
        extent=[lon_deg.min(), lon_deg.max(), lat_deg.min(), lat_deg.max()],
        origin="lower",
        aspect="auto",
    )
    plt.colorbar(im, label="Field value")
    plt.xlabel("Longitude (degrees)")
    plt.ylabel("Latitude (degrees)")
    plt.title(f"{name}: global analytic field")
    plt.tight_layout()

    out = OUTDIR / f"{name}_global.png"
    plt.savefig(out, dpi=200)
    plt.close()

    print(f"{name}: min={vals.min():.6f} max={vals.max():.6f} mean={vals.mean():.6f} std={vals.std():.6f}")
    print(f"Wrote {out}")


def plot_lon_profiles(func, name):
    lon_deg = np.linspace(-180.0, 180.0, 721)
    lon = np.deg2rad(lon_deg)

    plt.figure(figsize=(8.0, 5.0))
    for lat_deg in [-60, -30, 0, 30, 60]:
        lat = np.deg2rad(lat_deg) * np.ones_like(lon)
        plt.plot(lon_deg, func(lon, lat), label=f"lat={lat_deg}°")

    plt.xlabel("Longitude (degrees)")
    plt.ylabel("Field value")
    plt.title(f"{name}: longitude profiles")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out = OUTDIR / f"{name}_longitude_profiles.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Wrote {out}")


def plot_lat_profiles(func, name):
    lat_deg = np.linspace(-90.0, 90.0, 721)
    lat = np.deg2rad(lat_deg)

    plt.figure(figsize=(8.0, 5.0))
    for lon_deg in [-120, -60, 0, 60, 120]:
        lon = np.deg2rad(lon_deg) * np.ones_like(lat)
        plt.plot(lat_deg, func(lon, lat), label=f"lon={lon_deg}°")

    plt.xlabel("Latitude (degrees)")
    plt.ylabel("Field value")
    plt.title(f"{name}: latitude profiles")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out = OUTDIR / f"{name}_latitude_profiles.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Wrote {out}")


def main():
    for name, func in [("smooth1", smooth1), ("smooth2", smooth2)]:
        plot_global(func, name)
        plot_lon_profiles(func, name)
        plot_lat_profiles(func, name)

    print("\nGenerated smooth analytic field figures:")
    for p in sorted(OUTDIR.glob("smooth*.png")):
        print(" ", p)


if __name__ == "__main__":
    main()
