'''
File contents:
    - PyDataset class that allows lazy loading for datasets too large to fit 
        into RAM.
    - Sliding window method.
    - Train-valid-test split method.
    - Standard scalers method.
'''

from keras.utils import PyDataset
import math
from keras.utils import timeseries_dataset_from_array
import numpy as np
import pandas as pd
import math
import joblib
import os
from sklearn.preprocessing import StandardScaler

class PWWBPyDataset(PyDataset):
    def __init__(
        self, x_path: str, y_path: str, batch_size: int, shuffle: bool = False
    ):
        self.X, self.Y = self._validate_datasets(x_path, y_path)
        self.batch_size = self._validate_batch_size(batch_size)
        self.shuffle = shuffle
        self.indices = np.arange(len(self.X))
        self.on_epoch_end()

    def __len__(self):
        # gets number of batches in the dataset
        return math.ceil(len(self.X) / self.batch_size)

    def __getitem__(self, idx):
        # python silently clamps if stop_idx overflows; no need for bounds checking
        start_idx = idx * self.batch_size
        stop_idx = (idx + 1) * self.batch_size
        batch_idxs = self.indices[start_idx : stop_idx]
        
        X_batch = self.X[batch_idxs]
        Y_batch = self.Y[batch_idxs]

        return X_batch, Y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    # NOTE: validation methods
    def _validate_batch_size(self, batch_size):
        if batch_size <= 0:
            raise ValueError('batch_size must be > 0')
        return batch_size
    
    def _validate_datasets(self, x_path, y_path):
        X = np.load(x_path, mmap_mode='r')
        Y = np.load(y_path, mmap_mode='r')

        if len(X) != len(Y):
            raise ValueError(
                f'X and Y must share the same number of samples '
                f'({len(X)} != {len(Y)}'
            )

        return X, Y
        

# NOTE Utility functions for messing with datasets

def sliding_window(data, frames, sequence_stride=1, compute_targets=False):
    '''
    Returns a sliding window of the given data as a numpy array, where:
        - Each sample contains (frames) number of frames
        - Slides one frame at a time
        - Gives the user the option to compute the targets or not

    If you have something like (samples, dim, dim) and want 
        (samples, frames, dim, dim, channel), then simply call:
        `sliding_window(np.expand_dims(data, -1), ... )`
    '''
    def create_timeseries_dataset(data, frames, sequence_stride):
        return np.array(
            [
                val.numpy() 
                for val in timeseries_dataset_from_array(
                    data=data, 
                    targets=None, 
                    sequence_length=frames, 
                    sequence_stride=sequence_stride,
                    batch_size=None
                )
            ],
            dtype=data.dtype if isinstance(data, np.ndarray) else float 
        )

    if len(data) < (2 * frames):
        raise ValueError(
            "Insufficient data to run sliding window. "
            f"Number of frames in data ({len(data)}) must be greater than or "
            f"equal to {2 * frames} to create room for target samples."
        )

    X = create_timeseries_dataset(data[:-frames], frames, sequence_stride)
    Y = (
        create_timeseries_dataset(data[frames:], frames, sequence_stride)
        if compute_targets
        else None
    )

    return X, Y

def train_valid_test_split(
    X, 
    train_size=0.7, 
    valid_size=0.1,
    test_size=0.2, 
    verbose=True
):
    '''
    Performs a train/test/valid split on a given dataset.
        - No shuffle option
        - Performs this split on the first axis
    
    Raises a ValueError if the sizes don't add up to 1.0

    Example usage with X and targets:
        X_train, X_valid, X_test = train_valid_test_split(X, 0.7, 0.15, 0.15)
        Y_train, Y_valid, Y_test = train_valid_test_split(Y, 0.7, 0.15, 0.15)
    '''
    if not np.isclose(train_size + test_size + valid_size, 1.0):
        raise ValueError("All train/test/valid sizes must add up to 1.0.")

    train_split = int(len(X) * train_size)
    test_split = int(len(X) * (train_size + valid_size))

    # indices, [start, end)
    train_start, train_end = 0, train_split
    valid_start, valid_end = train_split, test_split
    test_start, test_end = test_split, len(X)
    
    X_train = X[train_start : train_end]
    X_valid = X[valid_start : valid_end]
    X_test = X[test_start : test_end]

    if verbose:
        print(
            f"🪓  Temporal split at indices {train_end} and {valid_end}:\n"
            f"\tTraining: samples {train_start}-{train_end-1} "
            f"({train_size*100:.0f}%), data shape = {X_train.shape}\n"
            f"\tValidation: samples {valid_start}-{valid_end-1} "
            f"({valid_size*100:.0f}%), data shape = {X_valid.shape}\n"
            f"\tTesting: samples {test_start}-{test_end-1} "
            f"({test_size*100:.0f}%), data shape = {X_test.shape}"
        )

    return X_train, X_valid, X_test

def std_scale(
    X_train, 
    X_valid=None, 
    X_test=None, 
    save=False, 
    save_path='data',
    verbose=True
):
    '''
    Performs standard scaling on the data passed.
        - User has the option to include test/valid sets.
        - User has the option to save scaler object in a valid directory.
    '''
    if verbose:
        if save:
            print(f"⚖️  Scaling data and saving to {save_path}...", end= " ")
        else:
            print("⚖️  Scaling data...", end= " ")

    scaler = StandardScaler()

    # Fancy reshaping because scaler expects a 2D input.
    # Flatten (2D) -> scale -> Inflate (original shape)
    scaled_train = (
        scaler
        .fit_transform(X_train.reshape(-1, 1))
        .reshape(X_train.shape)
    )
    scaled_valid = (
        scaler.transform(X_valid.reshape(-1, 1)).reshape(X_valid.shape)
        if X_valid is not None
        else None
    )
    scaled_test = (
        scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
        if X_test is not None
        else None
    )

    if save:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(scaler, save_path, compress=True)

    if verbose: print("✅ Complete!")

    return scaled_train, scaled_valid, scaled_test
