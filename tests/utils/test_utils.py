import unittest
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from libs.pwwb.utils.dataset import sliding_window 
from libs.pwwb.utils.dataset import PWWBPyDataset
from libs.pwwb.utils.idw import IDW 
from libs.pwwb.utils.residual_kriging import ResidualKriging
import numpy as np
import math

class TestUtilsSlidingWindow(unittest.TestCase):
    def test_targets_are_n_samples_from_training(self):
        a = np.arange(10000).reshape(100, 10, 10, 1)

        frames = 5
        X, Y = sliding_window(a, frames, compute_targets=True)

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
        X, Y = sliding_window(a, frames, compute_targets=True)

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

    def test_each_samples_frame_is_offset_by_five_from_next_sample_if_sequence_stride_is_5(self):
        a = np.arange(10000).reshape(100, 10, 10, 1)

        frames = 5
        X, Y = sliding_window(a, frames, frames, compute_targets=True)

        # pre-test invariants
        # make sure X and Y aren't equal, else everything will pass
        msg = "X and Y should not be equal."
        self.assertTrue(not np.array_equal(X, Y), msg)
        msg = "For every sample of X, the next sample should not be equivalent."
        self.assertTrue(not np.array_equal(X[1:], X[:-1]), msg)
        # each frame is offset by 5 relative to each sample
        msg = "For every sample in X, the next sample should be the same, offset by 5."
        self.assertTrue(
            np.all(np.array(
                [
                    np.array_equal(X[:-5, i+5], X[5:, i])
                    for i in range(frames-5)
                ]
            )),
            msg
        )

class TestUtilsIDW(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.idw = IDW(verbose=2)
        cls.air_sens_loc = {
            'Simi Valley - Cochran Street': (8, 1),
            'Reseda': (12, 4),
            'Santa Clarita': (4, 5),
            'North Holywood': (12, 8),
            'Los Angeles - N. Main Street': (17, 11),
            'Compton': (23, 12),
            'Long Beach Signal Hill': (28, 13),
            'Anaheim': (26, 18),
            'Glendora - Laurel': (14, 20),
            'Mira Loma - Van Buren': (20, 28),
            'Fontana - Arrow Highway': (15, 28),
            'Riverside - Rubidoux': (20, 30),
            'Lake Elsinore - W. Flint Street': (32, 32),
            'Crestline - Lake Gregory': (10, 33),
            'Temecula (Lake Skinner)': (36, 38)
        }

    @classmethod
    def tearDownClass(cls):
        del cls.idw

    def test_find_closest_sensors_to_la(self):
        la_sensor_loc = self.air_sens_loc['Los Angeles - N. Main Street']
        all_sensor_locs = list(self.air_sens_loc.values())
        x, y = la_sensor_loc
        actual = list(self.idw._find_closest_sensors_and_distances(
            x=x, y=y,
            sensor_coords=all_sensor_locs,
            elevation_grid=self.idw.elevation,
            ).keys())[:4]
        expected = [
            self.air_sens_loc['Los Angeles - N. Main Street'],
            self.air_sens_loc['North Holywood'],
            self.air_sens_loc['Compton'],
            self.air_sens_loc['Reseda'],
        ]

        self.assertEqual(actual, expected)

    def test_find_closest_sensors_to_glendora(self):
        la_sensor_loc = self.air_sens_loc['Glendora - Laurel']
        all_sensor_locs = list(self.air_sens_loc.values())
        x, y = la_sensor_loc
        actual = list(self.idw._find_closest_sensors_and_distances(
            x=x, y=y,
            sensor_coords=all_sensor_locs,
            elevation_grid=self.idw.elevation,
            ).keys())[:4]

        expected = [
            self.air_sens_loc['Glendora - Laurel'],
            self.air_sens_loc['Fontana - Arrow Highway'],
            self.air_sens_loc['Los Angeles - N. Main Street'],
            self.air_sens_loc['Mira Loma - Van Buren'],
        ]

        self.assertEqual(actual, expected)
        
    def test_find_closest_sensors_to_simi_valley(self):
        la_sensor_loc = self.air_sens_loc['Simi Valley - Cochran Street']
        all_sensor_locs = list(self.air_sens_loc.values())
        x, y = la_sensor_loc
        actual = list(self.idw._find_closest_sensors_and_distances(
            x=x, y=y,
            sensor_coords=all_sensor_locs,
            elevation_grid=self.idw.elevation,
            ).keys())[:4]

        expected = [
            self.air_sens_loc['Simi Valley - Cochran Street'],
            self.air_sens_loc['Reseda'],
            self.air_sens_loc['Santa Clarita'],
            self.air_sens_loc['North Holywood'],
        ]

        self.assertEqual(actual, expected)

    def test_no_neighbors_returns_nan(self):
        actual = self.idw._idw_interpolate(
            np.full((40, 40), np.nan), {(3, 4): 5}, 2.0, 5
        )
        self.assertTrue(np.isnan(actual))

class TestPWWBPyDataset(unittest.TestCase):
    def test_sequence_length_matches_data_batches(self):
        X_paths = "tests/utils/data/dummy_channel_1.npy"
        y_path = "tests/utils/data/dummy_label.npy"
        batch_size = 4
        generator = PWWBPyDataset(X_paths, y_path, batch_size)
        actual = len(generator)

        a = np.load("tests/utils/data/dummy_channel_1.npy")
        expected = math.ceil(len(a) / batch_size)

        msg = f"Sequence length doesn't match batched data size."
        self.assertEqual(expected, actual, msg)

class TestResidualKriging(unittest.TestCase):
    ''' Unfortunately you just have to vibe check by looking at the frames '''
    def test_constructor_runs(self):
        '''Just tests if the constructor runs'''
        rk = ResidualKriging()

    def test_kriging_runs(self):
        '''Just tests if the interpolation method runs'''
        rk = ResidualKriging(dim=40, verbose=2)

        sensor_frames = np.load(
            'tests/utils/data/residual_kriging/uninterpolated_grid.npy'
        )
        model_frames = np.load(
            'tests/naqfcdata/data/naqfc_aqm_20240514T07.npy'
        )

        inter = rk.interpolate_frames(
            # expects (n, dim, dim)
            np.expand_dims(sensor_frames, axis=0),
            np.expand_dims(model_frames, axis=0)
        )
