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

# Download L3 LR SSH if necessary for specified passes and cycles.
#altimetry_downloader_aviso.get("SWOT_L3_LR_SSH_Expert", output_dir=swot_path, cycle_number=list(range(1,59)), pass_number=[201, 214, 479, 520])

keep = ["time", "ssha_filtered", "ocean_tide", "latitude", "longitude","quality_flag"]
constits = ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1"]
min_obs = 40 # 8 amp + 8 ph + 1 trend

m = pyTMD.io.model(model_dir).from_database(model_name)   
ds_got = m.open_dataset(group="z")

print(f"Loaded GOT5.6 from {model_dir}")
# Load and inspect data
swot_passes = [f"{i:03d}" for i in range(1, 585)]
for swot_pass in swot_passes:
    # Check if output already exists and skip if so
    out_file = Path(f"{out_nc_path}/swot_tide_fit_pass_{swot_pass}.nc")
    if out_file.exists():
        print(f"Pass {swot_pass} already processed. Skipping.")
        continue
        
    data_paths = Path(f"{swot_path}/SWOT_Pass{swot_pass}").glob(f"SWOT_L3_LR_SSH_Expert_???_{swot_pass}_*.nc") # Name pattern _CYCLE_PASS_
    data_df = [xr.open_dataset(path) for path in data_paths]
    print(f"Pass {swot_pass}: {len(data_df)} files loaded")
    for path in data_paths:
        print(f"{path}")

    dfs_no_nadir = []
    # Drop conflicting num_nadir dimenstion and i_num_line, i_num_pixel vals which depend on it
    for df in data_df:
        try:
            df_dropped = df.drop_vars(['i_num_line', 'i_num_pixel'])
        except ValueError:
            print(df)
        dfs_no_nadir.append(df_dropped)

    small = []
    for ds in dfs_no_nadir:
        d = ds[keep]#.isel(num_lines=line_slice)
        d["ssha_detided"] = (d["ssha_filtered"] + d["ocean_tide"]).astype("float32")
        d["ssha_detided"] = d["ssha_detided"].where(d["quality_flag"] < 3)
        d = d.drop_vars(["ssha_filtered", "ocean_tide"])
        small.append(d.load())      # load only the subset, then release the file

    lat = small[0]["latitude"].values
    lon = small[0]["longitude"].values

    data_df = xr.concat(
        [d[["time", "ssha_detided"]] for d in small],
        dim="cycle_number",
    )
    data_df = data_df.assign_coords(
        latitude=(("num_lines", "num_pixels"), lat),
        longitude=(("num_lines", "num_pixels"), lon),
    )
    del small

    # Fit tides for this pass
    print(f"Fitting tides for pass {swot_pass}")
    tv = data_df["time"].values                                   # (cycle, num_lines)
    tnum = tv.astype("datetime64[ns]").astype(np.int64) / 86400e9 + 719163   # python ordinal days
    tnum[np.isnat(tv)] = np.nan

    h = data_df["ssha_detided"].values.astype("float32")          # (cycle, num_lines, num_pixels)
    lat = np.asarray(data_df["latitude"].values)                  # (num_lines, num_pixels)
    lon = np.asarray(data_df["longitude"].values)

    nvalid = np.isfinite(h).sum(axis=0)
    fit_lines = np.where((nvalid >= min_obs).any(axis=1))[0]          # positions along num_lines
    print("lines to fit:", fit_lines.size, "| points:", int((nvalid >= min_obs).sum()))

    def fit_line(t_j, h_j, lat_j):
        """t_j: (cycle,), h_j: (cycle, num_pixels), lat_j: (num_pixels,)"""
        npx = h_j.shape[1]
        amp = np.full((len(constits), npx), np.nan, dtype="float32")
        pha = np.full((len(constits), npx), np.nan, dtype="float32")
        for i in range(npx):
            y = h_j[:, i].astype("float64")
            ok = np.isfinite(y) & np.isfinite(t_j)
            if ok.sum() < min_obs or not np.isfinite(lat_j[i]):
                continue
            try:
                c = solve(t_j[ok], y[ok], lat=float(lat_j[i]), constit=constits,
                        epoch="python", nodal=False, phase="Greenwich",
                        trend=False, method="ols", conf_int="none", verbose=False)
            except Exception:
                continue
            names = list(c.name)
            for k, name in enumerate(constits):
                if name in names:
                    idx = names.index(name)
                    amp[k, i], pha[k, i] = c.A[idx], c.g[idx]
        return amp, pha

    res = Parallel(n_jobs=3, batch_size=10, verbose=5)(
        delayed(fit_line)(tnum[:, j], h[:, j, :], lat[j]) for j in fit_lines
    )
    print(len(res))  

    amp = np.stack([r[0] for r in res], axis=1)    # (constituent, n_lines, num_pixels)
    pha = np.stack([r[1] for r in res], axis=1)

    tide_fit = xr.Dataset(
        {"amplitude": (("constituent", "num_lines", "num_pixels"), amp),
        "phase": (("constituent", "num_lines", "num_pixels"), pha)},
        coords={"constituent": constits,
                "num_lines": fit_lines,
                "latitude": (("num_lines", "num_pixels"), lat[fit_lines]),
                "longitude": (("num_lines", "num_pixels"), lon[fit_lines])},
    )
    tide_fit["amplitude"].attrs["units"] = "m"
    tide_fit["phase"].attrs["units"] = "degrees (Greenwich)"

    # Compare to GOT5.6
    amp_s = tide_fit["amplitude"].values          # (const, lines, pixels)
    pha_s = tide_fit["phase"].values
    lat_s = tide_fit["latitude"].values           # (lines, pixels)
    lon_s = tide_fit["longitude"].values

    valid = np.isfinite(amp_s).all(axis=0)
    jj, ii = np.where(valid)
    lon_p, lat_p = lon_s[jj, ii], lat_s[jj, ii]
    names = [c.lower() for c in constits]

    def got_constants(lon, lat, names, batch=20000):
        """Pointwise interpolation. Returns amp (m) and Greenwich phase (deg), shape (n_const, n_points)."""
        n = lon.size
        amp = np.full((len(names), n), np.nan)
        ph = np.full((len(names), n), np.nan)
        for s in range(0, n, batch):
            sl = slice(s, min(s + batch, n))
            x = xr.DataArray(lon[sl], dims="points")
            y = xr.DataArray(lat[sl], dims="points")
            local = ds_got.tmd.interp(x, y, extrapolate=True)
            hc = np.stack([np.asarray(local[c].values) for c in names])
            assert hc.shape == (len(names), sl.stop - sl.start), hc.shape
            amp[:, sl] = np.abs(hc)
            ph[:, sl] = np.degrees(np.arctan2(-hc.imag, hc.real)) % 360
        return amp, ph

    amp_f, pha_f = got_constants(lon_p[:], lat_p[:], names)

    A_s = amp_s[:, jj, ii]
    P_s = pha_s[:, jj, ii]

    dA = A_s - amp_f
    dP = (P_s - pha_f + 180) % 360 - 180                   
    vec = np.sqrt(A_s**2 + amp_f**2 - 2*A_s*amp_f*np.cos(np.radians(P_s - pha_f)))

    shape = amp_s.shape

    for name, values in {
        "amp_got": amp_f,
        "pha_got": pha_f,
        "dA": dA,
        "dP": dP,
        "vec": vec,
    }.items():
        out = np.full(shape, np.nan, dtype=np.float32)
        out[:, jj, ii] = values
        tide_fit[name] = (
            ("constituent", "num_lines", "num_pixels"),
            out
        )

    # Save to netcdf
    tide_fit.to_netcdf(f"{out_nc_path}/swot_tide_fit_pass_{swot_pass}.nc")