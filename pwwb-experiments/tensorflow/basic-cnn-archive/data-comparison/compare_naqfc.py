import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.metrics import mean_absolute_error as mae 

BASE_PATH = '/home/mgraca/Workspace/hrrr-smoke-viz/pwwb-experiments/tensorflow/basic-cnn-archive'
DATA_PATH = os.path.join(BASE_PATH, 'processing-scripts/l2')
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

dates = pd.date_range('2023-08-02', '2025-08-02', inclusive='left', freq='h')
idx_of_date = {date : i for i, date in enumerate(dates)}
date_of_idx = {i : date for i, date in enumerate(dates)}

naqfc_raw = np.load(os.path.join(DATA_PATH, 'naqfc_pm25.npy'))
airnow = np.load(os.path.join(DATA_PATH, 'airnow_pm25.npy'))
openaq = np.load(os.path.join(DATA_PATH, 'openaq_pm25.npy'))

# NOTE COOKING
naqfc = np.where(naqfc_raw > 200, 0, naqfc_raw)

# compare naqfc with airnow and openaq
mae_sensor = mae(airnow[:, X, Y], naqfc[:, X, Y])
rmse_sensor = rmse(airnow[:, X, Y], naqfc[:, X, Y])
nrmse_sensor = rmse_sensor / np.mean(airnow[:, X, Y]) * 100
bias = np.mean(np.subtract(naqfc[:, X, Y], airnow[:, X, Y]))

print('With airnow (nowcast)')
print(f'MAE: {mae_sensor:.2f}')
print(f'RMSE: {rmse_sensor:.2f}')
print(f'NRMSE: {nrmse_sensor:.2f}%')
print(f'Bias: {bias:.2f}')
print()

mae_sensor = mae(openaq[:, X, Y], naqfc[:, X, Y])
rmse_sensor = rmse(openaq[:, X, Y], naqfc[:, X, Y])
nrmse_sensor = rmse_sensor / np.mean(openaq[:, X, Y]) * 100
bias = np.mean(np.subtract(naqfc[:, X, Y], openaq[:, X, Y]))

print('With openaq (raw)')
print(f'MAE: {mae_sensor:.2f}')
print(f'RMSE: {rmse_sensor:.2f}')
print(f'NRMSE: {nrmse_sensor:.2f}%')

# this disagrees with the study; for NA West, the naqfc bias term is -2.11 for 24 hour average, while here, the naqfc is overpredicting
# note: mean obs = 10.09, mean model = 7.04
print(f'Bias: {bias:.2f}')
print()
print(f'Mean obs sensor: {np.mean(openaq[:, X, Y])}')
print(f'Mean mod sensor: {np.mean(naqfc[:, X, Y])}')
print(f'Mean obs frame: {np.mean(openaq)}')
print(f'Mean mod frame: {np.mean(naqfc)}')

# imputed
plt.plot(dates, np.mean(openaq[:, X, Y], axis=-1), label='openaq')
plt.plot(dates, np.mean(naqfc[:, X, Y], axis=-1), label='naqfc')
plt.legend()
plt.show()

# raw 
plt.plot(dates, np.mean(openaq[:, X, Y], axis=-1), label='openaq')
plt.plot(dates, np.mean(naqfc_raw[:, X, Y], axis=-1), label='naqfc')
plt.legend()
plt.show()
