import numpy as np
import os
BASE_PATH = '/home/mgraca/Workspace/hrrr-smoke-viz/pwwb-experiments/tensorflow/autoencoder_archive'
DATA_PATH = os.path.join(BASE_PATH, 'preprocessed_cache/npy_files')
RESULTS_PATH = os.path.join(BASE_PATH, 'results/dual_autoencoder_model_grid_loss_0')

y_true = np.load(os.path.join(DATA_PATH, 'Y_test.npy'))
print(y_true.shape)

y_pred = np.load(os.path.join(RESULTS_PATH, 'y_pred.npy'))
print(y_pred.shape)

'''
(2059, 24, 40, 40, 1)
(2059, 24, 40, 40, 1)
'''

'''
static_samples = []
for sample in y_true:
    static_frames = []
    for frame in sample:
        is_static = np.isclose(frame, frame[0, ...]).all()
        static_frames.append(is_static)
    static_samples.append(np.array(static_frames).any())
'''

sample_contains_static_frames = np.array([
    np.array([
        np.isclose(frame, frame[0, ...]).all()
        for frame in sample
    ]).any()
    for sample in y_true
])
idx, *_ = np.where(sample_contains_static_frames)
print(idx)

y_true_pruned = y_true[~sample_contains_static_frames]
y_pred_pruned = y_pred[~sample_contains_static_frames]
print(y_true_pruned.shape)
print(y_pred_pruned.shape)

