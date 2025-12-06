import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.metrics import mean_absolute_error as mae 

BASE_PATH = '/home/mgraca/Workspace/hrrr-smoke-viz/pwwb-experiments/tensorflow/basic-cnn-archive/'
DATA_PATH = os.path.join(BASE_PATH, 'processing-scripts/l2')

airnow = np.load(os.path.join(DATA_PATH, 'airnow_pm25.npy'))

dates = pd.date_range('2023-08-02', '2025-08-02', inclusive='left', freq='h')
idx_of_date = {date : i for i, date in enumerate(dates)}
date_of_idx = {i : date for i, date in enumerate(dates)}

def thirty_day_lookback_avg_of(idx, frames):
    '''
    Caveat: produces empty list if there is no previous frame to lookback

    Results in the first 24 frames being truncated.
    '''
    i = 1
    prev_day_idx = idx - 24
    thirty_days_frames = []
    while i < 30 and prev_day_idx >= 0:
        thirty_days_frames.append(frames[prev_day_idx])
        prev_day_idx -= 24
        i += 1

    return (
        np.mean(np.array(thirty_days_frames), axis=0)
        if len(thirty_days_frames) != 0
        else np.array([])
    )

# generate 30-day avg frames
res = []
for i in range(airnow.shape[0]):
    arr = thirty_day_lookback_avg_of(i, airnow)
    if len(arr != 0):
        res.append(arr)
res = np.array(res)
print(res.shape)

# viz
realigned_airnow = airnow[24:]
print(realigned_airnow.shape)
idx = 100
fig, axes = plt.subplots(ncols=2, nrows=1)

vmin = np.nanmin([airnow[idx], res[idx]])
vmax = np.nanmax([airnow[idx], res[idx]])
axes[0].imshow(airnow[idx+24], vmin=vmin, vmax=vmax)
axes[0].set_title('Airnow')
axes[1].imshow(res[idx], vmin=vmin, vmax=vmax)
axes[1].set_title('30-day average')
plt.suptitle(f'Time: {date_of_idx[idx+24]}')

plt.show()
plt.clf()

# error analysis on sensor
SENSORS = {
    'Reseda' : (8, 3),
    'North Holywood' : (8, 11),
    'Los Angeles - N. Main Street' : (15, 16),
    'Compton' : (23, 17),
    'Long Beach Signal Hill' : (29, 19),
    'Anaheim' : (27, 29),
    'Glendora - Laurel' : (10, 33)
}
X, Y = list(zip(*SENSORS.values()))

mae_sensor = mae(realigned_airnow[:, X, Y], res[:, X, Y])
rmse_sensor = rmse(realigned_airnow[:, X, Y], res[:, X, Y])
nrmse_sensor = rmse_sensor / np.mean(realigned_airnow[:, X, Y]) * 100

print(f'MAE: {mae_sensor:.2f}')
print(f'RMSE: {rmse_sensor:.2f}')
print(f'NRMSE: {nrmse_sensor:.2f}%')

# viz (use matplotlib for the viz, zooming in really helps)
plt.plot(dates[24:], np.mean(realigned_airnow[:, X, Y], axis=-1))
plt.plot(dates[24:], np.mean(res[:, X, Y], axis=-1))
plt.show()
