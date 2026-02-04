import os
import sys
import pickle
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Driver code for the results visualizer')
parser.add_argument(
    'experiment_results',
    help='the folder containing the results of the experiment, with metadata.pkl, scalers.pkl, and y_pred.npy'
)
parser.add_argument(
    'data',
    help='the folder containing the training data used in the experiment'
)
args = parser.parse_args()

# NOTE parameterize this
BASE_PATH = '/home/mgraca/Workspace/hrrr-smoke-viz'
# NOTE be mindful of where your data is!
DATA_PATH = args.data
RESULTS_PATH = args.experiment_results
sys.path.append(BASE_PATH)

if not os.path.exists(RESULTS_PATH):
    raise ValueError(f"results path on {RESULTS_PATH} not found")
if not os.path.exists(DATA_PATH):
    raise ValueError(f"data path on {DATA_PATH} not found")

from libs.results_visualizer import *

# things you need to use plot_training_history
#if os.path.exists(os.path.join(RESULTS_PATH, 'history.pkl')):
print('Loading objects', end=' ')
with open(os.path.join(RESULTS_PATH, 'history.pkl'), 'rb') as f:
    history = pickle.load(f)
plot_training_history(
    history, os.path.join(RESULTS_PATH, 'loss_curves.png'), RESULTS_PATH
)

# things you need to use plot_sample
with open(os.path.join(DATA_PATH, 'metadata.pkl'), 'rb') as f:
    metadata = pickle.load(f)

with open(os.path.join(DATA_PATH, 'scalers.pkl'), 'rb') as f:
    scalers = pickle.load(f)
airnow_scaler = scalers['AirNow_PM25']

airnow_channel_idx = metadata['channel_names'].index('AirNow_PM25')
print('complete.')

# NOTE fix later to support valid vs test set
print('Loading test/validation', end=' ')
'''
y_true = np.load(os.path.join(DATA_PATH, 'npy_files/Y_test.npy'))
'''
y_true = np.load(os.path.join(DATA_PATH, 'npy_files/Y_valid.npy'))
y_pred = np.load(os.path.join(RESULTS_PATH, 'y_pred.npy'))
print('complete.')

#### NOTE experimenting with dead sensor removal
'''
print('WARNING! WE ARE REMOVING SAMPLES THAT HAVE MULTIPLE DEAD SENSORS!!!')
sample_contains_static_frames = np.array([
    np.array([
        np.isclose(frame, frame[0, ...]).all()
        for frame in sample
    ]).any()
    for sample in y_true
])
idx, *_ = np.where(sample_contains_static_frames)

y_true = y_true[~sample_contains_static_frames]
y_pred = y_pred[~sample_contains_static_frames]
####
'''

# NOTE fix later to support valid vs test set
print('Unscaling airnow', end=' ')
'''
X = np.load(os.path.join(DATA_PATH, 'npy_files/X_test.npy'))
'''
X = np.load(os.path.join(DATA_PATH, 'npy_files/X_valid.npy'), mmap_mode='r')
X_airnow_scaled = X[..., airnow_channel_idx]
X_airnow = (airnow_scaler
            .inverse_transform(X_airnow_scaled.reshape(-1, 1))
            .reshape(X_airnow_scaled.shape))
del X
print('complete.')

# NOTE outlier removal
print('Removing outlier dates (July 4th).')
mask = np.ones(len(y_true), dtype=bool)
mask[1916 : 1952] = False
y_true = y_true[mask]
y_pred = y_pred[mask]
X_airnow = X_airnow[mask]

#### NOTE modifications for stateful batch misalignment
'''
print('Truncating incomplete tail batch')
print(y_true.shape, y_pred.shape, X_airnow.shape)
X_airnow = X_airnow[:len(y_pred)]
y_true = y_true[:len(y_pred)]
print(y_true.shape, y_pred.shape, X_airnow.shape)
'''
####
print('Plotting samples', end=' ')
# sample plots + time series
rng = np.random.default_rng(seed=42)
idx = rng.integers(low=0, high=len(y_true), size=1).item()

os.makedirs(os.path.join(RESULTS_PATH, 'samples'), exist_ok=True)
plot_sample(
    X_airnow[idx],
    y_pred[idx],
    y_true[idx],
    os.path.join(RESULTS_PATH, 'samples/sample.png'),
    f'sample {idx}'
)

save_best_worst_samples(y_pred, y_true, os.path.join(RESULTS_PATH, 'samples'))
print('complete.')

# error plots
print('Plotting error', end=' ')
save_nrmse_plots(
    y_pred,
    y_true,
    list(metadata['sensors'].values()),
    os.path.join(RESULTS_PATH, 'error'),
    dim=y_pred.shape[2] # height
)
print('complete.')

'''
# plotting sensor timeseries
os.makedirs(os.path.join(RESULTS_PATH, 'timeseries'), exist_ok=True)

plot_sensor_timeseries(y_pred, y_true, os.path.join(RESULTS_PATH, 'timeseries/sensor_timeseries.png'), metadata['sensors'], start_idx=0, n_samples=1)

save_sensor_timeseries(y_pred, y_true, os.path.join(RESULTS_PATH, 'timeseries'), metadata['sensors'])
'''
