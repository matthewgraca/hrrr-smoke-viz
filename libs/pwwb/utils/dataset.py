import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # ignore informational logs
from keras.utils import timeseries_dataset_from_array

def sliding_window(data, frames, compute_targets=False):
    '''
    Returns a sliding window of the given data as a numpy array, where:
        - Each sample contains (frames) number of frames
        - Slides one frame at a time
        - Gives the user the option to compute the targets or not
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

    X = create_timeseries_dataset(data[:-frames])
    Y = create_timeseries_dataset(data[frames:]) if compute_targets else None

    return X, Y
