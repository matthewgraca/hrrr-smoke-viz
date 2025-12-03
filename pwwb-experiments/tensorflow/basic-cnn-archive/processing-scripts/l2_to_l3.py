# assumes that every channel is set up (frames, h, w)
# will convert these to the final input that can be piped into the model
# TODO not sure if it should be one massive file or if we should keep it split by channel
import os
BASE_PATH = '/home/mgraca/Workspace/hrrr-smoke-viz'
DATA_PATH = os.path.join(
    BASE_PATH,
    'pwwb-experiments/tensorflow/basic-cnn-archive/processing-scripts'
)
L2_SAVE_PATH = os.path.join(DATA_PATH, 'l2')
L3_SAVE_PATH = os.path.join(DATA_PATH, 'l3')

import sys
sys.path.append(BASE_PATH)
from libs.pwwb.utils.dataset import *

import numpy as np
import joblib
import json

def process(file, target_file):
    print(f'Processing {file}:')
    if file == target_file:
        data = np.load(file)

        X_train, X_valid, X_test = train_valid_test_split(data, train_size=0.7, valid_size=0.15, test_size=0.15)

        X_train, Y_train = sliding_window(X_train, frames=1, compute_targets=True)
        X_valid, Y_valid = sliding_window(X_valid, frames=1, compute_targets=True)
        X_test, Y_test = sliding_window(X_test, frames=1, compute_targets=True)
        if (
            np.all(X_train[1] != Y_train[0]) or
            np.all(X_valid[1] != Y_valid[0]) or 
            np.all(X_test[1] != Y_test[0])
        ):
            raise ValueError('Targets offset.')

        X_train, X_valid, X_test = std_scale(
            X_train, X_valid, X_test,
            save=True, save_path=os.path.join(DATA_PATH, 'std_scale.bin')
        )
        np.save(os.path.join(L3_SAVE_PATH, 'Y_train.npy'), Y_train)
        np.save(os.path.join(L3_SAVE_PATH, 'Y_valid.npy'), Y_valid)
        np.save(os.path.join(L3_SAVE_PATH, 'Y_test.npy'), Y_test)

        return X_train, X_valid, X_test
    else:
        X = np.load(file)
        X_train, X_valid, X_test = train_valid_test_split(X, train_size=0.7, valid_size=0.15, test_size=0.15)
        X_train, X_valid, X_test = std_scale(X_train, X_valid, X_test)

        X_train, _ = sliding_window(X_train, frames=1, compute_targets=False)
        X_valid, _ = sliding_window(X_valid, frames=1, compute_targets=False)
        X_test, _ = sliding_window(X_test, frames=1, compute_targets=False)

        return X_train, X_valid, X_test

X_train = []
X_valid = []
X_test = []
channels = {}
for i, f in enumerate(os.listdir(L2_SAVE_PATH)):
    train, valid, test = process(
        os.path.join(L2_SAVE_PATH, f),
        os.path.join(L2_SAVE_PATH, 'airnow_pm25.npy')
    )

    X_train.append(train)
    X_valid.append(valid)
    X_test.append(test)

    channel_name = os.path.splitext(os.path.basename(f))[0]
    channels[channel_name] = i

print('Saving channel indices:')
for k, v in channels.items():
    print(f'{k} : {v}')
print()

json_file = os.path.join(DATA_PATH, 'channels.json')
with open(json_file, 'w') as f:
    json.dump(channels, f)

print('Saving training/validation/test data:')
X_train = np.stack(X_train, axis=-1)
np.save(os.path.join(L3_SAVE_PATH, 'X_train.npy'), X_train)
print(X_train.shape)

X_valid = np.stack(X_valid, axis=-1)
np.save(os.path.join(L3_SAVE_PATH, 'X_valid.npy'), X_valid)
print(X_valid.shape)

X_test = np.stack(X_test, axis=-1)
np.save(os.path.join(L3_SAVE_PATH, 'X_test.npy'), X_test)
print(X_test.shape)
