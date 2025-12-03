import os
BASE_PATH = '/home/mgraca/Workspace/hrrr-smoke-viz'
DATA_PATH = os.path.join(
    BASE_PATH,
    'pwwb-experiments/tensorflow/basic-cnn-archive/processing-scripts'
)
L1_SAVE_PATH = os.path.join(DATA_PATH, 'l1')
L2_SAVE_PATH = os.path.join(DATA_PATH, 'l2')

import sys
sys.path.append(BASE_PATH)

import numpy as np
from libs.timedata import TimeData

time_steps = 17544

# elevation: needs to be tiled
data = np.load(os.path.join(L1_SAVE_PATH, 'elevation.npy'))
print(f'Tiling elevation data from {data.shape} to ({time_steps}, {data.shape[0]}, {data.shape[1]})')
res = np.tile(data[np.newaxis, :, :], (time_steps, 1, 1))
print(f'Result: {res.shape}')
print(f'Quick validation: first and last frame match: {np.all(res[0] == data), np.all(res[-1] == data)}')
np.save(os.path.join(L2_SAVE_PATH, 'elevation.npy'), res)
print()

# time data: needs to be generated, then channels extracted
print('Generating time data')
td = TimeData(
    start_date="2023-08-02",
    end_date="2025-08-02",
    dim=40,
    cyclical=True,
    month=True,
    day_of_week=False,
    day_of_month=False,
    verbose=False,
)
channels = ['month_sin', 'month_cos', 'hour_sin', 'hour_cos']

data = td.data
print(f'Result: {data.shape}')
print('Extracting channels.')
for i in range(data.shape[-1]):
    np.save(os.path.join(L2_SAVE_PATH, f'temporal_encoding_{channels[i]}.npy'), data[..., i])
    print(data[..., i].shape)
print()

# openaq: pull out data from npz
print('Extracting data from openaq')
data = np.load(os.path.join(L1_SAVE_PATH, 'openaq_processed.npz'))['data']
print(data.shape)
np.save(os.path.join(L2_SAVE_PATH, 'openaq_pm25.npy'), data)
print()

# goes pull out data from npz
print('Extracting data from goes')
data = np.load(os.path.join(L1_SAVE_PATH, 'goes_processed.npz'))['data']
print(data.shape)
np.save(os.path.join(L2_SAVE_PATH, 'goes_aod.npy'), data)
print()

# tempo pull out data from npz
print('Extracting data from tempo')
data = np.load(os.path.join(L1_SAVE_PATH, 'tempo_l3_no2_20230802_20250802_hourly.npz'))['data']
print(data.shape)
np.save(os.path.join(L2_SAVE_PATH, 'tempo_no2.npy'), data)
print()

# naqfc pull out data from npz
print('Extracting data from naqfc')
data = np.load(os.path.join(L1_SAVE_PATH, 'naqfc_pm25_processed.npz'))['data']
print(data.shape)
np.save(os.path.join(L2_SAVE_PATH, 'naqfc_pm25.npy'), data)
print()

# ndvi as-is
print('Saving ndvi as-is')
data = np.load(os.path.join(L1_SAVE_PATH, 'ndvi_processed.npy'))
print(data.shape)
np.save(os.path.join(L2_SAVE_PATH, 'ndvi.npy'), data)
print()

# extracting hrrr products 
print('Extracting data and pulling apart channels from hrrr')
sfc = np.load(os.path.join(L1_SAVE_PATH, 'hrrr_surface_2years_40x40.npz'))
# all keys: ['u_wind', 'v_wind', 'wind_speed', 'temp_2m', 'pbl_height', 'precip_rate', 'timestamps', 'extent', 'grid_size', 'variables']
for k in ['u_wind', 'v_wind', 'wind_speed', 'temp_2m', 'pbl_height', 'precip_rate']:
    data = sfc[k]
    print(data.shape)
    np.save(os.path.join(L2_SAVE_PATH, f'hrrr_{k}.npy'), data)
print()

# extracting airnow
print('Extracting data from airnow')
data = np.load(os.path.join(L1_SAVE_PATH, 'airnow_processed.npz'))['data']
print(data.shape)
np.save(os.path.join(L2_SAVE_PATH, 'airnow_pm25.npy'), data)
print()
