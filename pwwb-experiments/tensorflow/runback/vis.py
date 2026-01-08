import os
import sys
import pickle
import numpy as np

# config name should be the folder (just the folder, not entire path!) that contains the results of your experiment
# in turn, it should contain metadata.pkl, scalers.pkl, y_pred.npy
# this is where we save the viz to, as well.
CONFIG_NAME = sys.argv[1] if len(sys.argv) == 2 else None
if CONFIG_NAME == None:
    raise ValueError('Pass in a config name')
valid_configs = set(['classic_model', 'two_path_model'])
if CONFIG_NAME not in valid_configs:
    raise ValueError(f'Pass in a valid config name, or add it to the set: {valid_configs}')

BASE_PATH = '/home/mgraca/Workspace/hrrr-smoke-viz'
EXPERIMENT_PATH = os.path.join(BASE_PATH, 'pwwb-experiments/tensorflow/runback')
RESULTS_PATH = os.path.join(EXPERIMENT_PATH, f'results/{CONFIG_NAME}') 
DATA_PATH = os.path.join(EXPERIMENT_PATH, 'preprocessed_cache')
sys.path.append(BASE_PATH)

from scripts.vizfunctions import *

# things you need to use plot_training_history
if os.path.exists(os.path.join(RESULTS_PATH, 'history.pkl')):
    with open(os.path.join(RESULTS_PATH, 'history.pkl'), 'rb') as f:
        history = pickle.load(f)

    plot_training_history(history, os.path.join(RESULTS_PATH, 'loss_curves.png'), CONFIG_NAME)

# things you need to use plot_sample
with open(os.path.join(DATA_PATH, 'metadata.pkl'), 'rb') as f:
    metadata = pickle.load(f)

with open(os.path.join(DATA_PATH, 'scalers.pkl'), 'rb') as f:
    scalers = pickle.load(f)
airnow_scaler = scalers['AirNow_PM25']

airnow_channel_idx = metadata['channel_names'].index('AirNow_PM25')

y_true = np.load(os.path.join(DATA_PATH, 'npy_files/Y_test.npy'))
y_pred = np.load(os.path.join(RESULTS_PATH, 'y_pred.npy'))

X = np.load(os.path.join(DATA_PATH, 'npy_files/X_test.npy'))
X_airnow_scaled = X[..., airnow_channel_idx]
X_airnow = airnow_scaler.inverse_transform(X_airnow_scaled.reshape(-1, 1)).reshape(X_airnow_scaled.shape)

rng = np.random.default_rng(seed=42)
idx = rng.integers(low=0, high=len(y_true), size=1).item()

os.makedirs(os.path.join(RESULTS_PATH, 'samples'), exist_ok=True)
plot_sample(X_airnow[idx], y_pred[idx], y_true[idx], os.path.join(RESULTS_PATH, 'samples/sample.png'), f'sample {idx}')

# plotting sensor timeseries
os.makedirs(os.path.join(RESULTS_PATH, 'timeseries'), exist_ok=True)

plot_sensor_timeseries(y_pred, y_true, os.path.join(RESULTS_PATH, 'timeseries/sensor_timeseries.png'), metadata['sensors'], start_idx=0, n_samples=1)

save_sensor_timeseries(y_pred, y_true, os.path.join(RESULTS_PATH, 'timeseries'), metadata['sensors'])

save_best_worst_samples(y_pred, y_true, os.path.join(RESULTS_PATH, 'samples'))

