import unittest
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from libs.pwwb.utils.dataset import sliding_window 
from libs.pwwb.utils.dataset import PWWBPyDataset
import numpy as np
import math

class TestUtils(unittest.TestCase):
    def test_targets_are_n_samples_from_training(self):
        a = np.arange(10000).reshape(100, 10, 10, 1)

        frames = 5
        X, Y = sliding_window(a, frames, True)

        # pre-test invariants
        # make sure X and Y aren't equal, else everything will pass
        msg = "X and Y should not be equal."
        self.assertTrue(not np.array_equal(X, Y), msg)
        msg = "For every sample of X, the next sample should not be equivalent."
        self.assertTrue(not np.array_equal(X[1:], X[:-1]), msg)

        # targets are 5 samples away from the training data
        msg = "X and Y should the same when {frames} frames apart."
        self.assertTrue(np.array_equal(X[frames:], Y[:-frames]), msg)

    def test_each_samples_frame_is_offset_by_one_from_next_sample(self):
        a = np.arange(10000).reshape(100, 10, 10, 1)

        frames = 5
        X, Y = sliding_window(a, frames, True)

        # pre-test invariants
        # make sure X and Y aren't equal, else everything will pass
        msg = "X and Y should not be equal."
        self.assertTrue(not np.array_equal(X, Y), msg)
        msg = "For every sample of X, the next sample should not be equivalent."
        self.assertTrue(not np.array_equal(X[1:], X[:-1]), msg)

        # each frame is offset by one relative to each sample
        msg = "For every sample in X, the next sample should be the same, offset by 1."
        self.assertTrue(
            np.all(np.array(
                [
                    np.array_equal(X[:-1, i+1], X[1:, i])
                    for i in range(frames-1)
                ]
            )),
            msg
        )

class TestPWWBPyDataset(unittest.TestCase):
    def test_sequence_length_matches_data_batches(self):
        X_paths = [
            "tests/utils/data/dummy_channel_1.npy",
            "tests/utils/data/dummy_channel_2.npy"
        ]
        y_path = "tests/utils/data/dummy_label.npy"
        batch_size = 4
        generator = PWWBPyDataset(X_paths, y_path, batch_size)
        actual = len(generator)

        a = np.load("tests/utils/data/dummy_channel_1.npy")
        expected = math.ceil(len(a) / batch_size)

        msg = f"Sequence length doesn't match batched data size."
        self.assertEqual(expected, actual, msg)

    def test_assert_raised_when_channels_dont_match(self):
        X_paths = [
            "tests/utils/data/dummy_channel_1.npy",
            "tests/utils/data/dummy_channel_2.npy",
            "tests/utils/data/dummy_channel_3.npy"
        ]
        y_path = "tests/utils/data/dummy_label.npy"
        batch_size = 4

        with self.assertRaises(AssertionError):
            PWWBPyDataset(X_paths, y_path, batch_size)
