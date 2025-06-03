import unittest
from libs.airnowdata import AirNowData 
import numpy as np

class TestAirNowData(unittest.TestCase):
    def test_right_shape_for_n_frames(self):
        '''
        Check if the data shares the expected shape

        Preloaded data: 25 frames of airnow sensors
        '''
        n_frames = 5
        dim = 40
        ad = AirNowData(
            start_date="2024-01-01",
            end_date="2024-01-02",
            extent=(-118.75, -117.5, 33.5, 34.5),
            airnow_api_key=None,
            save_dir='tests/airnowdata/data/airnow.json',
            processed_cache_dir='tests/airnowdata/data/airnow_processed.npz',
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

        expected = (max(1, 25 - n_frames + 1), n_frames, dim , dim , 1)
        actual = ad.data.shape
        
        msg = f"Expected shape {expected}, returned {actual}"
        self.assertEqual(expected, actual, msg)

    def test_target_data_is_shifted_by_n_frames(self):
        '''
        Check if the first sample is properly offset.

        Preloaded data: 25 frames of airnow sensors
        '''
        n_frames = 5
        dim = 40
        ad = AirNowData(
            start_date="2024-01-01",
            end_date="2024-01-02",
            extent=(-118.75, -117.5, 33.5, 34.5),
            airnow_api_key=None,
            save_dir='tests/airnowdata/data/airnow.json',
            processed_cache_dir='tests/airnowdata/data/airnow_processed.npz',
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

        actual = ad.target_stations[0]

        msg = f"Expected shape {expected}, returned {actual}"
        np.testing.assert_array_equal(actual, expected)
