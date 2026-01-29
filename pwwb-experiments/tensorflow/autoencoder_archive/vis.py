import os
import sys
import pickle
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Driver code for the results visualizer')
parser.add_argument('config_name', help='the folder containing the results of the experiment, with metadata.pkl, scalers.pkl, and y_pred.npy')
parser.add_argument('data', help='the folder containing the training data used in the experiment')
args = parser.parse_args()

BASE_PATH = '/home/mgraca/Workspace/hrrr-smoke-viz'
EXPERIMENT_PATH = os.path.join(BASE_PATH, 'pwwb-experiments/tensorflow/autoencoder_archive')
RESULTS_PATH = os.path.join(EXPERIMENT_PATH, f'results/{args.config_name}') 
# NOTE be mindful of where your data is!
DATA_PATH = os.path.join(EXPERIMENT_PATH, args.data)
sys.path.append(BASE_PATH)

if not os.path.exists(RESULTS_PATH):
    raise ValueError(f"results path on {RESULTS_PATH} not found")

from libs.results_visualizer import *

# things you need to use plot_training_history
#if os.path.exists(os.path.join(RESULTS_PATH, 'history.pkl')):
with open(os.path.join(RESULTS_PATH, 'history.pkl'), 'rb') as f:
    history = pickle.load(f)
plot_training_history(
    history, os.path.join(RESULTS_PATH, 'loss_curves.png'), args.config_name
)

# things you need to use plot_sample
with open(os.path.join(DATA_PATH, 'metadata.pkl'), 'rb') as f:
    metadata = pickle.load(f)

with open(os.path.join(DATA_PATH, 'scalers.pkl'), 'rb') as f:
    scalers = pickle.load(f)
airnow_scaler = scalers['AirNow_PM25']

airnow_channel_idx = metadata['channel_names'].index('AirNow_PM25')

y_true = np.load(os.path.join(DATA_PATH, 'npy_files/Y_test.npy'))
y_pred = np.load(os.path.join(RESULTS_PATH, 'y_pred.npy'))

#### NOTE experimenting with dead sensor removal
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

X = np.load(os.path.join(DATA_PATH, 'npy_files/X_test.npy'))
X_airnow_scaled = X[..., airnow_channel_idx]
X_airnow = (airnow_scaler
            .inverse_transform(X_airnow_scaled.reshape(-1, 1))
            .reshape(X_airnow_scaled.shape))

#### NOTE modifications for stateful batch misalignment
'''
print('Truncating incomplete tail batch')
print(y_true.shape, y_pred.shape, X_airnow.shape)
X_airnow = X_airnow[:len(y_pred)]
y_true = y_true[:len(y_pred)]
print(y_true.shape, y_pred.shape, X_airnow.shape)
'''
####

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

# error plots
save_nrmse_plots(
    y_pred,
    y_true,
    list(metadata['sensors'].values()),
    os.path.join(RESULTS_PATH, 'error')
)

'''
# plotting sensor timeseries
os.makedirs(os.path.join(RESULTS_PATH, 'timeseries'), exist_ok=True)

plot_sensor_timeseries(y_pred, y_true, os.path.join(RESULTS_PATH, 'timeseries/sensor_timeseries.png'), metadata['sensors'], start_idx=0, n_samples=1)

save_sensor_timeseries(y_pred, y_true, os.path.join(RESULTS_PATH, 'timeseries'), metadata['sensors'])
'''
