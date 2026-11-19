# Zachary Katz
# zachary_katz@mines.edu
# 11 November 2026

"""
Programatically download n tracks of SWOT L3_LR_SSH_Expert data for all available
cycles, clean, and calculate tides in an harmonic inversion.
"""
# Imports
import altimetry_downloader_aviso
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from netrc import netrc
from xml.etree import ElementTree as ET
import requests
import re

### CHECK AND EXIT IF NOT ON NODE ###
if "SLURM_JOB_ID" not in os.environ:
    print("Not running using Slurm. Exiting.")
    sys.exit(1)
else:
    print(f"Running Slurm Job ID: {os.environ['SLURM_JOB_ID']}")

swot_path = '/home/m10921061/scratch/data/SWOT_Global_Tides/'
os.makedirs(swot_path, exist_ok=True)
swot_path = Path(swot_path)
out_nc_path = '/home/m10921061/scratch/output/SWOT_Global_Tides/netcdfs'
os.makedirs(out_nc_path, exist_ok=True)
out_img_path = '/home/m10921061/scratch/output/SWOT_Global_Tides/images/'
os.makedirs(out_img_path, exist_ok=True)


# OLD WITH ALTIMETRY_DOWNLOADER_AVISO (SLOW)

# Download L3 LR SSH if necessary for specified passes and cycles.
# 1-10 downloaded successfully prior
#for i in range(11, 100):
#    print(f"Pass {i}")
#    altimetry_downloader_aviso.get("SWOT_L3_LR_SSH_Expert", output_dir=swot_path, cycle_number=list(range(1,59)), pass_number=[i])

host = "tds-odatis.aviso.altimetry.fr"
base_before = (
    "https://tds-odatis.aviso.altimetry.fr/thredds/catalog/"
    "dataset-l3-swot-karin-nadir-validated/l3_lr_ssh/v3_0/Expert/reproc/"
)

base = (
    "https://tds-odatis.aviso.altimetry.fr/thredds/catalog/"
    "dataset-l3-swot-karin-nadir-validated/l3_lr_ssh/v3_0/Expert/forward/"
)
tracks = [f"{i:03d}" for i in range(100,585)]

auth = netrc(
    str(Path.home() / ".altimetry/.netrc")
).authenticators(host)

if auth is None:
    raise RuntimeError("AVISO credentials not found")

username, _, password = auth
auth = (username, password)


def get_cycle_files(cycle):
    catalog_urls = [
        f"{base_before}cycle_{cycle:03d}/catalog.xml",
        f"{base}cycle_{cycle:03d}/catalog.xml",
    ]

    files = []

    for catalog_url in catalog_urls:
        try:
            r = requests.get(catalog_url, auth=auth, timeout=60)
            r.raise_for_status()

            root = ET.fromstring(r.content)

            for dataset in root.iter():
                url_path = dataset.attrib.get("urlPath")

                if url_path and url_path.endswith(".nc"):
                    name = Path(url_path).name

                    match = re.search(
                        rf"SWOT_L3_LR_SSH_Expert_{cycle:03d}_(\d{{3}})_",
                        name
                    )

                    if match and match.group(1) in tracks:
                        files.append((
                            name,
                            f"https://{host}/thredds/fileServer/{url_path}"
                        ))

        except requests.RequestException as e:
            print(f"Catalog unavailable: {catalog_url}: {e}")

    # Remove duplicate URLs, retaining distinct files with different names
    files = list({url: (name, url) for name, url in files}.values())
    return files


def download_file(item):
    name, url = item
    output = swot_path / name
    temp = swot_path / (name + ".part")

    if output.exists() and output.stat().st_size > 0:
        return f"SKIP: {name}"
    try:
        with requests.get(
            url,
            auth=auth,
            stream=True,
            timeout=(30, 180)
        ) as r:
            r.raise_for_status()

            with open(temp, "wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)

            expected = r.headers.get("Content-Length")

        if expected and temp.stat().st_size != int(expected):
            raise IOError("Downloaded size does not match Content-Length")

        temp.rename(output)
        return f"OK: {name}"

    except Exception as e:
        temp.unlink(missing_ok=True)
        return f"FAILED: {name}: {e}"


# Discover files across cycles
all_files = []

for cycle in range(1, 59):
    print(f"Discovering cycle {cycle:03d}")

    try:
        all_files.extend(get_cycle_files(cycle))
    except Exception as e:
        print(f"Catalog failed for cycle {cycle}: {e}")

# Only missing files
all_files = [
    item for item in all_files
    if not (swot_path / item[0]).exists()
]

print(f"\nFiles to download: {len(all_files)}")
for name, url in all_files:
    print(name)

# Parallel downloads
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [
        executor.submit(download_file, item)
        for item in all_files
    ]

    for future in as_completed(futures):
        print(future.result())