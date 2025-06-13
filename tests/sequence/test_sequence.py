import unittest
from libs.sequence import PWWBPyDataset 
import numpy as np
import math

class TestPWWBPyDataset(unittest.TestCase):
    def test_sequence_length_matches_data_batches(self):
        X_paths = [
            "tests/sequence/data/dummy_channel_1.npy",
            "tests/sequence/data/dummy_channel_2.npy"
        ]
        y_path = "tests/sequence/data/dummy_label.npy"
        batch_size = 4
        generator = PWWBPyDataset(X_paths, y_path, batch_size)
        actual = len(generator)

        a = np.load("tests/sequence/data/dummy_channel_1.npy")
        expected = math.ceil(len(a) / batch_size)

        msg = f"Sequence length doesn't match batched data size."
        self.assertEqual(expected, actual, msg)

    def test_assert_raised_when_channels_dont_match(self):
        X_paths = [
            "tests/sequence/data/dummy_channel_1.npy",
            "tests/sequence/data/dummy_channel_2.npy",
            "tests/sequence/data/dummy_channel_3.npy"
        ]
        y_path = "tests/sequence/data/dummy_label.npy"
        batch_size = 4

        with self.assertRaises(AssertionError):
            PWWBPyDataset(X_paths, y_path, batch_size)
