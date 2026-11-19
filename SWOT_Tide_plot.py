# Zachary Katz
# zachary_katz@mines.edu
# 11 November 2026

"""
Programatically download n tracks of SWOT L3_LR_SSH_Expert data for all available
cycles, clean, and calculate tides in an harmonic inversion.
"""
# Imports
import altimetry_downloader_aviso
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
from cartopy import crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from utide import solve
import matplotlib.dates as mdates
from joblib import Parallel, delayed
import time
import pyTMD
import os

### CHECK AND EXIT IF NOT ON NODE ###
if "SLURM_JOB_ID" not in os.environ:
    print("Not running using Slurm. Exiting.")
    sys.exit(1)
else:
    print(f"Running Slurm Job ID: {os.environ['SLURM_JOB_ID']}")

swot_path = '/home/m10921061/scratch/data/SWOT_Global_Tides/'
os.makedirs(swot_path, exist_ok=True)
out_nc_path = '/home/m10921061/scratch/output/SWOT_Global_Tides/netcdfs'
os.makedirs(out_nc_path, exist_ok=True)
out_img_path = '/home/m10921061/scratch/output/SWOT_Global_Tides/images/'
os.makedirs(out_img_path, exist_ok=True)

model_dir = "/home/m10921061/scratch/background/GOT5.6"   
model_name = "GOT5.6"
constits = ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1"]

# Calc median and RMS of differences across all passes
summary_file = f"./swot_tide_fit_summary.txt"
with open(summary_file, "a") as f:
    for constituent in constits:
        k = constits.index(constituent)
        files = sorted(Path(f"{out_nc_path}").glob("swot_tide_fit_pass_*.nc"))
        dA_all = []
        dP_all = []
        vec_all = []

        for path in files:
            with xr.open_dataset(path) as ds:
                dA = ds["dA"].isel(constituent=k).values
                dP = ds["dP"].isel(constituent=k).values
                vec = ds["vec"].isel(constituent=k).values

                valid = np.isfinite(dA) & np.isfinite(dP) & np.isfinite(vec)
                dA_all.append(dA[valid])
                dP_all.append(dP[valid])
                vec_all.append(vec[valid])

        dA_all = np.concatenate(dA_all)
        dP_all = np.concatenate(dP_all)
        vec_all = np.concatenate(vec_all)

        median_dA = np.nanmedian(np.abs(dA_all))
        median_dP = np.nanmedian(np.abs(dP_all))
        rms_vec = np.sqrt(np.nanmean(vec_all**2))

        # Print summary statistics to text file
        f.write(f"{constituent}: median |dA| = {median_dA*100:.1f} cm | "
                f"median |dphase| = {median_dP:.1f}° | "
                f"RMS vector diff = {rms_vec*100:.1f} cm\n")


def get_spilhaus_mask(lon2d, lat2d, projection):
    # Transform geographic coordinates into Spilhaus coordinates
    xy = projection.transform_points(
        ccrs.PlateCarree(), lon2d, lat2d
    )
    x = xy[..., 0]
    y = xy[..., 1]

    bound = np.mean(np.abs(np.asarray(projection.bounds)))

    # Identify points near the projection boundary
    near_bound = (
        (np.abs(x) > 0.95 * bound) |
        (np.abs(y) > 0.95 * bound)
    )

    # Rotate coordinates to align with the projection boundary
    theta = np.deg2rad(45)
    rot_x = x * np.cos(theta) - y * np.sin(theta)
    rot_y = y * np.cos(theta) + x * np.sin(theta)

    # Detect sign changes across cell diagonals
    change_sign0 = (
        (rot_x[1:, 1:] * rot_x[:-1, :-1] < 0) |
        (rot_y[1:, 1:] * rot_y[:-1, :-1] < 0)
    )

    change_sign1 = (
        (rot_x[1:, :-1] * rot_x[:-1, 1:] < 0) |
        (rot_y[1:, :-1] * rot_y[:-1, 1:] < 0)
    )

    change_sign = change_sign0 | change_sign1

    # Expand the mask to all four corners of affected cells
    sign_change = np.zeros_like(x, dtype=bool)

    sign_change[1:, 1:] |= change_sign
    sign_change[:-1, :-1] |= change_sign
    sign_change[1:, :-1] |= change_sign
    sign_change[:-1, 1:] |= change_sign

    return near_bound & sign_change


projections = ["spilhaus", "robinson"]
for projection in projections:
    for constituent in constits[:]:
        print(f"Plotting {constituent}")
        k = constits.index(constituent)
        files = sorted(Path(f"{out_nc_path}").glob("swot_tide_fit_pass_*.nc"))
        percentiles = [[] for _ in range(3)]
        # Calculate 98th percentile for each variable across all passes
        for path in files:
            with xr.open_dataset(path) as ds:
                for i, var in enumerate(["amplitude", "amp_got", "vec"]):
                    v = ds[var].isel(constituent=k).values
                    percentiles[i].append(v[np.isfinite(v)])
        limits = [(0, np.nanpercentile(np.concatenate(v), 98)) for v in percentiles]

        if projection == "spilhaus":
            fig_swot, ax_swot = plt.subplots(1, 1, figsize=(16, 10), subplot_kw={"projection": ccrs.Spilhaus()})
            fig_got, ax_got = plt.subplots(1, 1, figsize=(16, 10), subplot_kw={"projection": ccrs.Spilhaus()})
            fig_vec, ax_vec = plt.subplots(1, 1, figsize=(16, 10), subplot_kw={"projection": ccrs.Spilhaus()})
        elif projection == "robinson":
            fig_swot, ax_swot = plt.subplots(1, 1, figsize=(16, 10), subplot_kw={"projection": ccrs.Robinson()})
            fig_got, ax_got = plt.subplots(1, 1, figsize=(16, 10), subplot_kw={"projection": ccrs.Robinson()})
            fig_vec, ax_vec = plt.subplots(1, 1, figsize=(16, 10), subplot_kw={"projection": ccrs.Robinson()})
        titles = [f"SWOT fit {constituent} [m]",f"GOT5.6 {constituent} [m]",f"Vector difference {constituent} [m]"]
        cmaps = ["viridis", "viridis", "magma"]
        save_name = ['SWOT', 'GOT', 'Difference']

        for index, (fig, ax, ttl, cm, lim, svnm) in enumerate(zip([fig_swot, fig_got, fig_vec], [ax_swot, ax_got, ax_vec], titles, cmaps, limits, save_name)):
            for path in files:
                with xr.open_dataset(path) as ds:
                    lon = ds["longitude"].values
                    lat = ds["latitude"].values

                    if index == 0:
                        v = ds["amplitude"].isel(constituent=k).values
                    elif index == 1:
                        v = ds["amp_got"].isel(constituent=k).values
                    else:
                        v = ds["vec"].isel(constituent=k).values

                    valid = (
                        np.isfinite(v)
                        & np.isfinite(lon)
                        & np.isfinite(lat)
                    )
                    v = np.where(valid, v, np.nan)
                    if projection == "spilhaus":
                        mask = get_spilhaus_mask(lon, lat, ccrs.Spilhaus())
                        v = np.ma.masked_where(mask, v)

                    sc = ax.pcolormesh(
                        lon,
                        lat,
                        v,
                        cmap=cm,
                        vmin=lim[0],
                        vmax=lim[1],
                        shading="auto",
                        transform=ccrs.PlateCarree(),
                        rasterized=True
                    )

            ax.coastlines()
            ax.add_feature(
            cfeature.LAND,
            facecolor="gray",
            edgecolor="none",
            zorder=2
            )
            ax.set_title(ttl, fontsize=28)
            sm = plt.cm.ScalarMappable(
                norm=plt.Normalize(*lim),
                cmap=cm
            )
            cbar = plt.colorbar(
                sm,
                ax=ax,
                orientation="horizontal",
                pad=0.05,
                fraction=0.05
            )
            cbar.ax.tick_params(labelsize=15)
            cbar.set_label(
                "Amplitude [m]" if ttl != titles[2] else "Difference [m]",
                fontsize=20,
            )

            if projection == "spilhaus":
                ax.set_global()
            elif projection == "robinson":
                ax.set_global()

            fig.tight_layout()
            fig.savefig(f"{out_img_path}/swot_tide_fit_{constituent}_{svnm}_{projection}.png", dpi=300)

for constituent in constits[:]:
    k = constits.index(constituent)
    files = sorted(Path(f"{out_nc_path}").glob("swot_tide_fit_pass_*.nc"))
    percentiles = [[] for _ in range(3)]
    # Calculate 98th percentile for each variable across all passes
    for path in files:
        with xr.open_dataset(path) as ds:
            v = ds["vec"].isel(constituent=k).values
            percentiles[2].append(v[np.isfinite(v)])
    limits = [(0, np.nanpercentile(np.concatenate(percentiles[2]), 98))]

    fig_hist, ax_hist = plt.subplots(1, 1, figsize=(10, 6))
    all_values = []

    for path in files:
        with xr.open_dataset(path) as ds:
            v = ds["vec"].isel(constituent=k).values
            v = v[np.isfinite(v)]
            all_values.append(v)

    all_values = np.concatenate(all_values)

    ax_hist.hist(
        all_values,
        bins=200,
        range=limits[0],
        alpha=0.7,
    )
    ax_hist.set_xlabel("Vector difference [m]", fontsize=14)
    ax_hist.set_ylabel("Frequency", fontsize=14)
    ax_hist.set_title(f"{constituent} Vector Differences", fontsize=16)
    ax_hist.legend(fontsize=10)
    fig_hist.tight_layout()
    fig_hist.savefig(f"{out_img_path}/swot_tide_fit_{constituent}_histogram.png", dpi=300)