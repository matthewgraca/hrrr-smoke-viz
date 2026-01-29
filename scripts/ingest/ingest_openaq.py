import sys
import os
BASE_PATH = '/home/mgraca/Workspace/hrrr-smoke-viz'
SAVE_PATH = '/mnt/wildfire/processed-data/2026-01-27'
CSV_PATH = '/mnt/wildfire/raw-data/openaq-clarity'
ELEVATION_PATH = '/mnt/wildfire/processed-data/2026-01-27/elevation.npy'
sys.path.append(BASE_PATH)
from libs.openaqdata import OpenAQData

od = OpenAQData(
    load_csv=True,
    save_dir=CSV_PATH,
    save_path=SAVE_PATH,
    use_interpolation=True,
    elevation_path=ELEVATION_PATH,
    extent=(-118.615, -117.70, 33.60, 34.35),
    dim=84,
    verbose=1
)
