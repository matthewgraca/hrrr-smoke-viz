import unittest
from libs.airnowdata import AirNowData 
import numpy as np
import subprocess
from shutil import which
import sys

class TestAirNowData(unittest.TestCase):
    def setUp(self):
        n_frames = 5
        dim = 40
        cache_dir = 'tests/airnowdata/data/airnow_processed.npz'
        # silence noisy output
        with open('/dev/null', 'w') as f_null:
            sys.stdout = f_null
            sys.stderr = f_null

            ad = AirNowData(
                start_date="2024-01-01",
                end_date="2024-01-02",
                extent=(-118.75, -117.5, 33.5, 34.5),
                airnow_api_key=None,
                save_dir='tests/airnowdata/data/airnow.json',
                processed_cache_dir=cache_dir,
                frames_per_sample=n_frames,
                dim=dim,
                idw_power=2,
                elevation_path="libs/inputs/elevation.npy",
                mask_path=None,
                use_mask=False,
                sensor_whitelist=None,
                use_whitelist=False,
                force_reprocess=False,
                chunk_days=30
            )
        self.ad = ad
        self.n_frames = n_frames
        self.cache_dir = cache_dir

    def tearDown(self):
        # clean up cache
        p = subprocess.run(
            f"{which('rm')} {self.cache_dir}",
            shell=True,
            capture_output=True,
            encoding="utf-8",
            check=True,
        )

        self.ad = None
        self.cache_dir = None
        self.n_frames = None 

    # NOTE: Tests for data shape and sliding window
    def test_right_shape_for_n_frames(self):
        '''
        Check if the data shares the expected shape

        Preloaded data: 25 frames of airnow sensors
        '''
        ad = self.ad
        n_frames = self.n_frames
        expected = (ad.data.shape[0] - n_frames, n_frames, len(ad.air_sens_loc))
        actual = ad.target_stations.shape
        
        msg = f"Expected shape {expected}, returned {actual}"
        self.assertEqual(expected, actual, msg)

    def test_target_data_is_shifted_by_first_n_frames(self):
        '''
        Check if the first sample is properly offset.

        Preloaded data: 25 frames of airnow sensors
        '''
        ad = self.ad
        n_frames = self.n_frames
        sample = 0
        gridded_data = ad.ground_site_grids  
        sensor_locations =  ad.air_sens_loc
        expected = np.empty((n_frames, len(sensor_locations)))
        for frame in range(n_frames):
            for i, (loc, coords) in enumerate(sensor_locations.items()):
                x, y = coords
                offset = sample + n_frames + frame
                if offset < len(gridded_data):
                    expected[frame][i] = gridded_data[offset][x][y]
                else:
                    expected[frame][i] = gridded_data[-1][x][y]

        actual = ad.target_stations[sample]

        msg = f"Expected {expected}, returned {actual}"
        np.testing.assert_array_equal(actual, expected, msg)

    def test_last_target_data_is_shifted_by_first_n_frames(self):
        '''
        Check if the last sample is properly offset.

        Preloaded data: 25 frames of airnow sensors
        '''
        ad = self.ad
        n_frames = self.n_frames
        sample = len(ad.target_stations) - 1 
        gridded_data = ad.ground_site_grids  
        sensor_locations = ad.air_sens_loc
        expected = np.empty((n_frames, len(sensor_locations)))
        for frame in range(n_frames):
            for i, (loc, coords) in enumerate(sensor_locations.items()):
                x, y = coords
                offset = sample + n_frames + frame
                if offset < len(gridded_data):
                    expected[frame][i] = gridded_data[offset][x][y]
                else:
                    expected[frame][i] = gridded_data[-1][x][y]

        actual = ad.target_stations[sample]

        msg = f"Expected {expected}, returned {actual}"
        np.testing.assert_array_equal(actual, expected, msg)

    # NOTE tests for finding nearest sensors
    def test_find_closest_sensors_to_la(self):
        '''
        Check if the right sensors are picked.
        '''
        ad = self.ad
        sensor_name = 'Los Angeles - N. Main Street'
        x, y = ad.air_sens_loc[sensor_name]
        loc_to_sensor = {v : k for k, v in ad.air_sens_loc.items()}
        # note that the sensor itself will be counted;
        # i.e. top 3 = top 4, ignoring the first value 
        actual, _ = ad._find_closest_values(
            x=x, 
            y=y, 
            coordinates=list(ad.air_sens_loc.values()), 
            n=4
        )
        actual = actual[1:]
        actual_sensors = [loc_to_sensor[(x, y)] for x, y in actual]
        expected = [
            ad.air_sens_loc['Compton'],
            ad.air_sens_loc['North Holywood'],
            ad.air_sens_loc['Reseda'],
        ]

        expected_sensors = [loc_to_sensor[(x, y)] for x, y in expected]

        msg = f"Expected {expected_sensors}, returned {actual_sensors}"
        self.assertEqual(actual, expected, msg)

    def test_find_closest_sensors_to_glendora(self):
        '''
        Check if the right sensors are picked.
        '''
        ad = self.ad
        sensor_name = 'Glendora - Laurel'
        x, y = ad.air_sens_loc[sensor_name]
        loc_to_sensor = {v : k for k, v in ad.air_sens_loc.items()}
        actual, _ = ad._find_closest_values(
            x=x, 
            y=y, 
            coordinates=list(ad.air_sens_loc.values()), 
            n=4
        )
        actual = actual[1:]
        actual_sensors = [loc_to_sensor[(x, y)] for x, y in actual]
        expected = [
            ad.air_sens_loc['Anaheim'],
            ad.air_sens_loc['Los Angeles - N. Main Street'],
            ad.air_sens_loc['Compton'],
        ]

        expected_sensors = [loc_to_sensor[(x, y)] for x, y in expected]

        msg = f"Expected {expected_sensors}, returned {actual_sensors}"
        self.assertEqual(actual, expected, msg)

    def test_find_closest_sensors_to_simi_valley(self):
        '''
        Check if the right sensors are picked.
        '''
        ad = self.ad
        sensor_name = 'Simi Valley - Cochran Street'
        x, y = ad.air_sens_loc[sensor_name]
        loc_to_sensor = {v : k for k, v in ad.air_sens_loc.items()}
        actual, _ = ad._find_closest_values(
            x=x, 
            y=y, 
            coordinates=list(ad.air_sens_loc.values()), 
            n=4
        )
        actual = actual[1:]
        actual_sensors = [loc_to_sensor[(x, y)] for x, y in actual]
        expected = [
            ad.air_sens_loc['Reseda'],
            ad.air_sens_loc['Santa Clarita'],
            ad.air_sens_loc['North Holywood'],
        ]

        expected_sensors = [loc_to_sensor[(x, y)] for x, y in expected]

        msg = f"Expected {expected_sensors}, returned {actual_sensors}"
        self.assertEqual(actual, expected, msg)
