import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import keras
tf.keras.backend.set_image_data_format('channels_last')

BASE_PATH = '/home/mgraca/Workspace/hrrr-smoke-viz'
EXPERIMENT_PATH = os.path.join(BASE_PATH, 'pwwb-experiments/tensorflow/runback')
DATA_PATH = os.path.join(EXPERIMENT_PATH, 'preprocessed_cache/npy_files')
RESULTS_PATH = os.path.join(EXPERIMENT_PATH, 'results')

import sys
sys.path.append(BASE_PATH)
from libs.pwwb.utils.dataset import PWWBPyDataset

import matplotlib.pyplot as plt
import numpy as np

# training parameters
EPOCHS = 100 
BATCH_SIZE = 64

# dataset parameters
WORKERS = 4
USE_MULTIPROCESSING = False
#MAX_QUEUE_SIZE = 10

# model definition
def classic_model(input_shape):
    from keras.models import Sequential
    from keras.layers import ConvLSTM2D, Conv3D, InputLayer
    from keras.optimizers import Adam

    # common args in each layer
    convlstm2d_args = {
        'kernel_size': (3, 3),
        'padding': 'same',
        'return_sequences': True
    }

    conv3d_args = {
        'kernel_size': (3, 3, 3),
        'activation': 'relu',
        'padding': 'same'    
    }

    # model definition
    model = Sequential()
    model.add(InputLayer(input_shape))
    model.add(ConvLSTM2D(filters=25, **convlstm2d_args))
    model.add(ConvLSTM2D(filters=50, **convlstm2d_args))
    model.add(Conv3D(filters=25, **conv3d_args))
    model.add(Conv3D(filters=1, **conv3d_args))

    model.compile(loss='mean_absolute_error', optimizer='adam')
    model.summary()
    print()

    return model

# load dataset
train_ds = PWWBPyDataset(
    x_path=os.path.join(DATA_PATH, 'X_train.npy'),
    y_path=os.path.join(DATA_PATH, 'Y_train.npy'),
    batch_size=BATCH_SIZE,
    shuffle=True,
    workers=WORKERS,
    use_multiprocessing=USE_MULTIPROCESSING
)

valid_ds = PWWBPyDataset(
    x_path=os.path.join(DATA_PATH, 'X_valid.npy'),
    y_path=os.path.join(DATA_PATH, 'Y_valid.npy'),
    batch_size=BATCH_SIZE,
    shuffle=False,
    workers=WORKERS,
    use_multiprocessing=USE_MULTIPROCESSING
)

test_ds = PWWBPyDataset(
    x_path=os.path.join(DATA_PATH, 'X_test.npy'),
    y_path=os.path.join(DATA_PATH, 'Y_test.npy'),
    batch_size=BATCH_SIZE,
    shuffle=False,
    workers=WORKERS,
    use_multiprocessing=USE_MULTIPROCESSING
)

model = classic_model(train_ds.input_shape)

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS
)

def plot_loss(history, save_dir):
    plt.figure(figsize=(10, 6))

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')

    plt.legend()
    plt.title(f'\nTraining Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MAE)')
    plt.grid(True, alpha=0.3)

    plt.savefig(os.path.join(save_dir, 'loss_curves.png'))

    return

plot_loss(history, RESULTS_PATH)

y_pred = model.predict(test_ds)
np.save(os.path.join(RESULTS_PATH, 'y_pred.npy'), y_pred)
