# ignore the info logs display when you import keras 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np

def sliding_window(data, frames, compute_targets=False):
    '''
    Returns a sliding window of the given data as a numpy array, where:
        - Each sample contains (frames) number of frames
        - Slides one frame at a time
        - Gives the user the option to compute the targets or not

    If you have something like (samples, dim, dim) and want 
        (samples, frames, dim, dim, channel), then simply call:
        `sliding_window(np.expand_dims(data, -1), ... )`
    '''
    def create_timeseries_dataset(data):
        return np.array([
            val.numpy() 
            for val in timeseries_dataset_from_array(
                data=data, 
                targets=None, 
                sequence_length=frames, 
                batch_size=None
            )
        ])

    if len(data) < (2 * frames):
        raise ValueError(
            "Insufficient data to run sliding window. "
            f"Number of frames in data ({len(data)}) must be greater than or "
            f"equal to {2 * frames} to create room for target samples."
        )

    X = create_timeseries_dataset(data[:-frames])
    Y = create_timeseries_dataset(data[frames:]) if compute_targets else None

    return X, Y
