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

import argparse
def argparser(valid_models, valid_losses):
    parser = argparse.ArgumentParser(
        description='training script'
    )
    parser.add_argument(
        'model',
        help=f'the model to train: {valid_models}'
    )
    parser.add_argument(
        'loss',
        help=f'the loss function to use: {valid_losses}'
    )
    parser.add_argument(
        'data',
        help='location of the data'
    )
    parser.add_argument(
        'results',
        help='location results will be saved to.'
    )
    parser.add_argument(
        'suffix',
        help='extra string to attach to end of experiment results folder'
    )
    parser.add_argument(
        '-t', '--test', action='store_true',
        help='only builds and tests the model; does not trigger training'
    )

    # validate
    args = parser.parse_args()

    if args.model not in valid_models:
        raise ValueError(f'Invalid model. Pick from: {valid_models}')

    if args.loss not in valid_losses:
        raise ValueError(f'Invalid loss. Pick from: {valid_losses}')

    return args

args = argparser(
    valid_models=set(['classic', 'two_path', 'dual_autoencoder', 'dual_ae_gated_skips']),
    valid_losses=set(['grid_mae', 'grid_mse', 'nhood'])
)

# training parameters
EPOCHS = 100
BATCH_SIZE = 16

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import keras
tf.keras.backend.set_image_data_format('channels_last')

import sys

EXPERIMENT_NAME = args.model + '_' + args.loss + '_' + args.suffix

BASE_PATH = '/home/mgraca/Workspace/hrrr-smoke-viz'
DATA_PATH = args.data
RESULTS_PATH = (
    os.path.join(args.results, 'test_experiment')
    if args.test
    else os.path.join(args.results, EXPERIMENT_NAME)
)
if not os.path.exists(DATA_PATH):
    raise ValueError(f"data path on {DATA_PATH} not found")
os.makedirs(RESULTS_PATH, exist_ok=True)
print(
    f'{TextColor.BLUE}Results will be saved to:{TextColor.ENDC}',
    f'{TextColor.BOLD}{RESULTS_PATH}{TextColor.ENDC}'
)

sys.path.append(BASE_PATH)
from libs.pwwb.utils.dataset import PWWBPyDataset
import libs.models as local_model

import matplotlib.pyplot as plt
import numpy as np
import pickle
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping

# dataset parameters
WORKERS = 4
USE_MULTIPROCESSING = False
#MAX_QUEUE_SIZE = 10

# channel information
with open(os.path.join(DATA_PATH, 'metadata.pkl'), 'rb') as f:
    metadata = pickle.load(f)
channel_to_idx = {ch : i for i, ch in enumerate(metadata['channel_names'])}

# load dataset
print(f'{TextColor.BLUE}Lazy loading datasets...{TextColor.ENDC}')
train_ds = PWWBPyDataset(
    x_path=os.path.join(DATA_PATH, 'npy_files/X_train.npy'),
    y_path=os.path.join(DATA_PATH, 'npy_files/Y_train.npy'),
    batch_size=BATCH_SIZE,
    shuffle=True,
    workers=WORKERS,
    use_multiprocessing=USE_MULTIPROCESSING
)
print(f'\tTraining dataset loaded: {len(train_ds)} batches loaded.')

valid_ds = PWWBPyDataset(
    x_path=os.path.join(DATA_PATH, 'npy_files/X_valid.npy'),
    y_path=os.path.join(DATA_PATH, 'npy_files/Y_valid.npy'),
    batch_size=BATCH_SIZE,
    shuffle=False,
    workers=WORKERS,
    use_multiprocessing=USE_MULTIPROCESSING
)
print(f'\tValidation dataset loaded: {len(valid_ds)} batches loaded.')

# NOTE fix later to properly support test vs valid
'''
test_ds = PWWBPyDataset(
    x_path=os.path.join(DATA_PATH, 'npy_files/X_test.npy'),
    y_path=os.path.join(DATA_PATH, 'npy_files/Y_test.npy'),
    batch_size=BATCH_SIZE,
    shuffle=False,
    workers=WORKERS,
    use_multiprocessing=USE_MULTIPROCESSING
)
print('\tTest dataset loaded.\n')
'''

input_shape = train_ds.input_shape
f, h, w, c = input_shape

ARCH_5x5 = {
    'temporal_filters': [16, 24, 32],
    'spatial_filters': [32, 48, 64],
    'bottleneck': 128,
    'latent_size': 5,
    'strides': [2, 2, 2]
}

ARCH_10x10 = {
    'temporal_filters': [16, 24, 32],
    'spatial_filters': [32, 48],
    'bottleneck': 256,
    'latent_size': 10,
    'strides': [2, 2]
}

# NOTE
ARCH_84 = {
    'temporal_filters': [16, 24, 32],
    'spatial_filters': [32, 48, 64],
    'bottleneck': 256,
    'latent_size': 7,
    'strides': [2, 2, 3]
}

convlstm_reg_config = {
    'kernel_regularizer': keras.regularizers.l2(3e-4),
    'recurrent_regularizer': keras.regularizers.l2(3e-4),
    'dropout': 0.15
}

dual_ae = local_model.DualAutoencoder(
    input_shape=input_shape,
    arch_config=ARCH_84,
    convlstm_reg_config=convlstm_reg_config,
    output_horizon=metadata['forecast_horizon'],
    observed_channels=[
        metadata['channel_names'].index(ch)
        for ch in metadata['observed_channels']
    ],
    forecast_channels=[
        metadata['channel_names'].index(ch)
        for ch in metadata['forecast_channels']
    ]
)

match args.model:
    case 'classic':
        model = local_model.classic(input_shape)
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
        model = local_model.two_path(
            input_shape,
            path1_channels=[channel_to_idx[ch] for ch in path1_channel_names],
            path2_channels=[channel_to_idx[ch] for ch in path2_channel_names],
        )
    case 'dual_autoencoder':
        model = dual_ae.dual_autoencoder()
    case 'dual_ae_gated_skips':
        model = dual_ae.gated_skips()
    case _:
        raise ValueError(f'Model not implemented.')

match args.loss:
    case 'grid_mae':
        model.compile(
            loss='mean_absolute_error',
            optimizer=keras.optimizers.AdamW(weight_decay=1e-3)
        )
    case 'grid_mse':
        model.compile(loss='mean_squared_error', optimizer='adam')
    case 'nhood':
        from libs.loss import NHoodMAE
        model.compile(loss=NHoodMAE(sensors=metadata['sensors'], dim=h), optimizer='adam')
    case _:
        raise ValueError(f'Invalid loss choice; pick from : {valid_losses}')

model.summary()
print()

plot_model(
    model, 
    to_file=os.path.join(RESULTS_PATH, 'model.png'),
    show_layer_names=True, 
    expand_nested=True,
    show_shapes=True,
    rankdir='TB' # TB=vertical, LR=horizontal
)
if args.test:
    print(f'{TextColor.BLUE}Build mode activated, skipping training.{TextColor.ENDC}')
    print(f'{TextColor.BLUE}Testing if a batch of data can pass through...{TextColor.ENDC}', end=' ')
    y = model(tf.random.uniform((1, *input_shape)))
    print(f'{TextColor.GREEN}complete.{TextColor.ENDC}\n')
    sys.exit()

print(f'{TextColor.BLUE}Beginning Training{TextColor.ENDC}')

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=15,
        verbose=1,
        mode='min',
        restore_best_weights=True
    )
]

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=callbacks
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

# NOTE fix later to properly support test vs valid
'''
y_pred = model.predict(test_ds)
'''
y_pred = model.predict(valid_ds)
np.save(os.path.join(RESULTS_PATH, 'y_pred.npy'), y_pred)
