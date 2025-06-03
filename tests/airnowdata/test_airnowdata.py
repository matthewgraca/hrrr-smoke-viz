import unittest
from libs.airnowdata import AirNowData 
import numpy as np
import subprocess
from shutil import which

class TestAirNowData(unittest.TestCase):
    def test_right_shape_for_n_frames(self):
        '''
        Check if the data shares the expected shape

        Preloaded data: 25 frames of airnow sensors
        '''
        n_frames = 5
        dim = 40
        cache_dir = 'tests/airnowdata/data/airnow_processed.npz'
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

        expected = (ad.data.shape[0] - n_frames, n_frames, len(ad.air_sens_loc))
        actual = ad.target_stations.shape
        
        # clean up cache
        p = subprocess.run(
            f"{which('rm')} {cache_dir}",
            shell=True,
            capture_output=True,
            encoding="utf-8",
            check=True,
        )

        msg = f"Expected shape {expected}, returned {actual}"
        self.assertEqual(expected, actual, msg)

    def test_target_data_is_shifted_by_first_n_frames(self):
        '''
        Check if the first sample is properly offset.

        Preloaded data: 25 frames of airnow sensors
        '''
        n_frames = 5
        dim = 40
        cache_dir = 'tests/airnowdata/data/airnow_processed.npz'
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

        # clean up cache
        p = subprocess.run(
            f"{which('rm')} {cache_dir}",
            shell=True,
            capture_output=True,
            encoding="utf-8",
            check=True,
        )

        msg = f"Expected shape {expected}, returned {actual}"
        np.testing.assert_array_equal(actual, expected)

    def test_last_target_data_is_shifted_by_first_n_frames(self):
        '''
        Check if the last sample is properly offset.

        Preloaded data: 25 frames of airnow sensors
        '''
        n_frames = 5
        dim = 40
        cache_dir = 'tests/airnowdata/data/airnow_processed.npz'
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

        # clean up cache
        p = subprocess.run(
            f"{which('rm')} {cache_dir}",
            shell=True,
            capture_output=True,
            encoding="utf-8",
            check=True,
        )

        msg = f"Expected shape {expected}, returned {actual}"
        np.testing.assert_array_equal(actual, expected)
