import unittest
from libs.airnowdata import AirNowData 
import numpy as np
import subprocess
from shutil import which
from contextlib import redirect_stdout
import io

class TestAirNowData(unittest.TestCase):
    def setUp(self):
        n_frames = 5
        dim = 40
        cache_dir = 'tests/airnowdata/data/airnow_processed.npz'
        # silence noisy output
        with redirect_stdout(io.StringIO()):
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
        expected = (ad.data.shape[0], ad.data.shape[1], len(ad.air_sens_loc))
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
        np.testing.assert_allclose(actual, expected, err_msg=msg)

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
        np.testing.assert_allclose(actual, expected, err_msg=msg)

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

    def test_getting_all_stations_closest_to_all_sensors(self):
        ad = self.ad
        loc_to_sensor = {v : k for k, v in ad.air_sens_loc.items()}

        actual = ad._get_closest_stations_to_each_sensor(
            ad.air_sens_loc
        )
        actual_sensors = {
            sensor : [loc_to_sensor[loc] for loc in locations]
            for sensor, locations in actual.items()
        }
        
        expected = {
            'Simi Valley - Cochran Street': [
                (12, 6), (4, 7), (12, 12), (17, 16),
                (23, 17), (28, 18), (14, 28), (26, 25)
            ], 
            'Reseda': [
                (8, 2), (12, 12), (4, 7), (17, 16), 
                (23, 17), (28, 18), (14, 28), (26, 25)
            ], 
            'Santa Clarita': [
                (8, 2), (12, 6), (12, 12), (17, 16), 
                (23, 17), (14, 28), (28, 18), (26, 25)
            ], 
            'North Holywood': [
                (12, 6), (17, 16), (4, 7), (8, 2), 
                (23, 17), (14, 28), (28, 18), (26, 25)
            ], 
            'Los Angeles - N. Main Street': [
                (23, 17), (12, 12), (12, 6), (28, 18), 
                (14, 28), (26, 25), (4, 7), (8, 2)
            ], 
            'Compton': [
                (28, 18), (17, 16), (26, 25), (12, 12), 
                (14, 28), (12, 6), (8, 2), (4, 7)
            ], 
            'Long Beach Signal Hill': [
                (23, 17), (26, 25), (17, 16), (12, 12), 
                (14, 28), (12, 6), (8, 2), (4, 7)
            ], 
            'Anaheim': [
                (28, 18), (23, 17), (14, 28), (17, 16), 
                (12, 12), (12, 6), (4, 7), (8, 2)
            ], 
            'Glendora - Laurel': [
                (26, 25), (17, 16), (23, 17), (12, 12), 
                (28, 18), (12, 6), (4, 7), (8, 2)
            ]
        }

        expected_sensors = {
            sensor : [loc_to_sensor[loc] for loc in locations]
            for sensor, locations in expected.items()
        }

        msg = f"Expected {expected_sensors}, returned {actual_sensors}"
        self.assertEqual(actual, expected, msg)

    def test_imputing_ground_sites(self):
        '''
        Check if ground sites will be imputed with average of 3 closest sensors
        '''
        ad = self.ad
        air_sens_loc = ad.air_sens_loc
        synthetic_data = np.zeros((40, 40))
        # closest locations to glendora (14, 28)
        locations = ad._find_closest_values(
            *air_sens_loc['Glendora - Laurel'],
            list(air_sens_loc.values()),
            len(air_sens_loc)
        )[0][1:]
        locations_data = [100] * len(locations)
        locations_data[0] = 2
        locations_data[1] = 3
        locations_data[2] = 4

        for loc, data in zip(locations, locations_data):
            synthetic_data[*loc] = data 

        actual_grid = ad._impute_ground_site_grids([synthetic_data], air_sens_loc)
        actual = actual_grid[0, *air_sens_loc['Glendora - Laurel']]
        expected = 3

        msg = f"Expected {expected}, returned {actual}"
        self.assertEqual(actual, expected, msg)

    def test_imputing_ground_sites_with_a_dead_neighbor(self):
        '''
        Check if ground sites will be imputed with average of 3 closest valid sensors
        '''
        ad = self.ad
        air_sens_loc = ad.air_sens_loc
        synthetic_data = np.zeros((40, 40))
        # closest locations to glendora (14, 28)
        locations = ad._find_closest_values(
            *air_sens_loc['Glendora - Laurel'],
            list(air_sens_loc.values()),
            len(air_sens_loc)
        )[0][1:]
        # be very careful here; because we're actually peforming TWO imputations
        # at the same time. chased a crazy bug here expecting only one impute
        # on Glendora, when it's neighbor we set to 0 was also getting imputed,
        # thus Glendora getting imputed based on that imputation AFTER its 
        # neighbor got imputed first.
        locations_data = [100] * len(locations)
        locations_data[0] = 0

        for loc, data in zip(locations, locations_data):
            synthetic_data[*loc] = data 

        actual_grid = ad._impute_ground_site_grids([synthetic_data], air_sens_loc)
        actual = actual_grid[0, *air_sens_loc['Glendora - Laurel']]
        expected = 100

        msg = f"Expected {expected}, returned {actual}"
        msg = f"{locations}, {locations_data}"
        msg = f"{actual_grid[actual_grid != 0]}"
        self.assertEqual(actual, expected, msg)
