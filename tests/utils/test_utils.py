import unittest
from libs.pwwb.utils.dataset import sliding_window 
import numpy as np

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
