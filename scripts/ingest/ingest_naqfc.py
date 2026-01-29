import sys
import os
GRIB_PATH = '/mnt/wildfire'
BASE_PATH = '/home/mgraca/Workspace/hrrr-smoke-viz'
SAVE_PATH = '/mnt/wildfire/2026-01-27'
sys.path.append(BASE_PATH)
from libs.naqfcdata import NAQFCData

naqfc = NAQFCData(
    start_date="2023-08-02 00:00",
    end_date="2025-08-02 00:00",
    extent=(-118.615, -117.70, 33.60, 34.35),
    dim=84,
    product='pm25',         # 'pm25' = pm2.5, 'o3' = ozone, 'dust', 'smoke'
    local_path=GRIB_PATH,   # where grib files should be saved to/live in
    save_path=SAVE_PATH,    # where the final numpy file should be saved to
    load_numpy=False,       # specifies the numpy file should be loaded from cache
    verbose=1,              # 0 = all msgs, 1 = prog bar + errors, 2 = only errors
)
print(f'Loading data: {naqfc.data.shape}')
