import argparse

valid_models = set(['classic', 'two_path'])
valid_losses = set(['grid', 'nhood'])

parser = argparse.ArgumentParser(description='Training script (you will need to edit paths in the script!)')
parser.add_argument('model', help=f'the model to train: {valid_models}')
parser.add_argument('loss', help=f'the loss function to use: {valid_losses}')
parser.add_argument('-t', '--test', action='store_true', help='triggers a test instead of a full run (saved in test_experiment, epochs set to 1)')
args = parser.parse_args()

# training parameters
EPOCHS = 100 if not args.test else 1
BATCH_SIZE = 64

MODEL_NAME = args.model
if MODEL_NAME not in valid_models:
    raise ValueError(f'Invalid model. Pick from: {valid_models}')

LOSS_NAME = args.loss
if LOSS_NAME not in valid_losses:
    raise ValueError(f'Invalid loss. Pick from: {valid_losses}')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import keras
tf.keras.backend.set_image_data_format('channels_last')

import sys

EXPERIMENT_NAME = 'test_experiment' if args.test else MODEL_NAME + '_model' + LOSS_NAME + '_loss'

BASE_PATH = '/home/mgraca/Workspace/hrrr-smoke-viz'
EXPERIMENT_PATH = os.path.join(BASE_PATH, 'pwwb-experiments/tensorflow/runback')
DATA_PATH = os.path.join(EXPERIMENT_PATH, 'preprocessed_cache/npy_files')
RESULTS_PATH = os.path.join(EXPERIMENT_PATH, f'results/{EXPERIMENT_NAME}')
os.makedirs(RESULTS_PATH, exist_ok=True)

sys.path.append(BASE_PATH)
from libs.pwwb.utils.dataset import PWWBPyDataset

import matplotlib.pyplot as plt
import numpy as np
import pickle

class TextColor:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'  # Resets formatting
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# dataset parameters
WORKERS = 4
USE_MULTIPROCESSING = False
#MAX_QUEUE_SIZE = 10

# channel information
with open(os.path.join(EXPERIMENT_PATH, 'preprocessed_cache/metadata.pkl'), 'rb') as f:
    metadata = pickle.load(f)
channel_to_idx = {ch : i for i, ch in enumerate(metadata['channel_names'])}

# common args in each layer to reduce function call size
CONVLSTM2D_ARGS = {
    'kernel_size': (3, 3),
    'padding': 'same',
    'return_sequences': True
}

CONV3D_ARGS = {
    'kernel_size': (3, 3, 3),
    'activation': 'relu',
    'padding': 'same'    
}

# model definition
def classic_model(input_shape):
    from keras.models import Sequential
    from keras.layers import ConvLSTM2D, Conv3D, InputLayer
    from keras.optimizers import Adam

    # model definition
    model = Sequential()
    model.add(InputLayer(input_shape))
    model.add(ConvLSTM2D(filters=25, **CONVLSTM2D_ARGS))
    model.add(ConvLSTM2D(filters=50, **CONVLSTM2D_ARGS))
    model.add(Conv3D(filters=25, **CONV3D_ARGS))
    model.add(Conv3D(filters=1, **CONV3D_ARGS))

    return model

def two_path_model(input_shape, path1_channels, path2_channels):
    # pass in the indices of the channels, not their name!
    from keras.layers import Input, Conv3D, ConvLSTM2D
    from keras.layers import Concatenate
    from keras.models import Model
    from libs.layers import ChannelSplitLayer

    def path_block(x, name):
        x = ConvLSTM2D(filters=20, **CONVLSTM2D_ARGS, name=f'{name}_convlstm_2d_1')(x)
        x = ConvLSTM2D(filters=40, **CONVLSTM2D_ARGS, name=f'{name}_convlstm_2d_2')(x)
        x = Conv3D(filters=20, **CONV3D_ARGS, name=f'{name}_conv3d_1')(x)
        return x

    def trunk_block(x, name):
        x = Conv3D(filters=30, **CONV3D_ARGS, name=f'{name}_conv3d_1')(x)
        x = Conv3D(filters=20, **CONV3D_ARGS, name=f'{name}_conv3d_2')(x)
        x = Conv3D(filters=1, **CONV3D_ARGS, name=f'{name}_conv3d_3')(x)
        return x

    inputs = Input(shape=input_shape)

    p_1 = ChannelSplitLayer(path1_channels)(inputs)
    p_1 = path_block(p_1, name='pm25_path')

    p_2 = ChannelSplitLayer(path2_channels)(inputs)
    p_2 = path_block(p_2, name='other_path')

    x = Concatenate(axis=-1)([p_1, p_2])
    x = trunk_block(x, name='trunk')

    model = Model(inputs, x, name='two_path_model')

    return model

# load dataset
print(f'{TextColor.BLUE}Lazy loading datasets...{TextColor.ENDC}')
train_ds = PWWBPyDataset(
    x_path=os.path.join(DATA_PATH, 'X_train.npy'),
    y_path=os.path.join(DATA_PATH, 'Y_train.npy'),
    batch_size=BATCH_SIZE,
    shuffle=True,
    workers=WORKERS,
    use_multiprocessing=USE_MULTIPROCESSING
)
print('\tTraining dataset loaded.')

valid_ds = PWWBPyDataset(
    x_path=os.path.join(DATA_PATH, 'X_valid.npy'),
    y_path=os.path.join(DATA_PATH, 'Y_valid.npy'),
    batch_size=BATCH_SIZE,
    shuffle=False,
    workers=WORKERS,
    use_multiprocessing=USE_MULTIPROCESSING
)
print('\tValidation dataset loaded.')

test_ds = PWWBPyDataset(
    x_path=os.path.join(DATA_PATH, 'X_test.npy'),
    y_path=os.path.join(DATA_PATH, 'Y_test.npy'),
    batch_size=BATCH_SIZE,
    shuffle=False,
    workers=WORKERS,
    use_multiprocessing=USE_MULTIPROCESSING
)
print('\tTest dataset loaded.\n')

input_shape = train_ds.input_shape
f, h, w, c = input_shape
match MODEL_NAME:
    case 'classic':
        model = classic_model(input_shape)
    case 'two_path':
        path1_channel_names = [
            'AirNow_PM25', 'AirNow_Hourly_Clim', 'OpenAQ_PM25', 'NAQFC_PM25',
            'Temporal_Month_Sin', 'Temporal_Month_Cos', 'Temporal_Hour_Sin',
            'Temporal_Hour_Cos'
        ]
        path2_channel_names = [
            'HRRR_Wind_U', 'HRRR_Wind_V', 'HRRR_Wind_Speed', 'HRRR_Temp_2m',
            'HRRR_PBL_Height', 'HRRR_Precip_Rate', 'Elevation', 'NDVI', 'GOES', 'TEMPO'
        ]
        model = two_path_model(
            input_shape,
            path1_channels=[channel_to_idx[ch] for ch in path1_channel_names],
            path2_channels=[channel_to_idx[ch] for ch in path2_channel_names],
        )
    case _:
        raise ValueError(f'Invalid model choice; pick from : {valid_models}')

match LOSS_NAME:
    case 'grid':
        model.compile(loss='mean_absolute_error', optimizer='adam')
    case 'nhood':
        from libs.loss import NHoodMAE
        model.compile(loss=NHoodMAE(sensors=metadata['sensors'], dim=h), optimizer='adam')
    case _:
        raise ValueError(f'Invalid loss choice; pick from : {valid_losses}')

model.summary()
print()

print(f'{TextColor.BLUE}Beginning Training{TextColor.ENDC}')
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS
)
print()

print(f'{TextColor.BLUE}Saving history file... {TextColor.ENDC}', end='')
with open(os.path.join(RESULTS_PATH, 'history.pkl'), 'wb') as f:
    pickle.dump(history.history, f)
print(f'{TextColor.GREEN}complete.{TextColor.ENDC}\n')

print(f'{TextColor.BLUE}Saving model... {TextColor.ENDC}', end='')
model.save(os.path.join(RESULTS_PATH, 'model.keras'))
print(f'{TextColor.GREEN}complete.{TextColor.ENDC}\n')

print(f'{TextColor.BLUE}Generating and saving predictions.{TextColor.ENDC}')
y_pred = model.predict(test_ds)
np.save(os.path.join(RESULTS_PATH, 'y_pred.npy'), y_pred)
