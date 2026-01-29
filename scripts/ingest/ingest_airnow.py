import sys
import os
BASE_PATH = '/home/mgraca/Workspace/hrrr-smoke-viz'
SAVE_PATH = '/mnt/wildfire/processed-data/2026-01-27/airnow_processed.npz'
JSON_PATH = '/mnt/wildfire/raw-data/epa-airnow/airnow.json'
ELEVATION_PATH = '/mnt/wildfire/processed-data/2026-01-27/elevation.npy'
sys.path.append(BASE_PATH)
from libs.airnowdata import AirNowData

ad = AirNowData(
    extent=(-118.615, -117.70, 33.60, 34.35),
    save_dir=JSON_PATH,
    processed_cache_dir=SAVE_PATH,
    elevation_path=ELEVATION_PATH,
    force_reprocess=True,
    dim=84,
)
