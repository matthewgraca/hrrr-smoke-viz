import sys
import os
BASE_PATH = '/home/mgraca/Workspace/hrrr-smoke-viz'
SAVE_PATH = '/mnt/wildfire/processed-data/2026-01-27/goes_processed.npz'
CACHE_PATH = '/mnt/wildfire/raw-data'
sys.path.append(BASE_PATH)
from libs.goesdata import GOESData

goes = GOESData(
    start_date="2023-08-02 00:00",
    end_date="2025-08-02 00:00",
    extent=(-118.615, -117.70, 33.60, 34.35),
    dim=84,
    save_dir=CACHE_PATH,
    cache_path=SAVE_PATH,
    pre_downloaded=True,
)
