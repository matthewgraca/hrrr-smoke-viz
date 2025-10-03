import unittest
from libs.airnowdata import AirNowData 
import numpy as np
import subprocess
from shutil import which
from libs.pwwb.utils.dataset import sliding_window
import pandas as pd

class TestAirNowData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
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
            dim=40,
            use_interpolation=True,
            idw_power=2,
            neighbors=9,
            elevation_path=None,
            elevation_scale_factor=100,
            mask_path=None,
            use_mask=False,
            sensor_whitelist=None,
            use_whitelist=False,
            force_reprocess=False,
            use_variable_blur=False,# determines if variable blur is used after interpolation
            chunk_days=30,
            verbose=2,              # 0=allow all, 1=progress bar only, 2=silence all except warning
        )
        cls.ad = ad
        cls.n_frames = n_frames
        cls.cache_dir = cache_dir

    @classmethod
    def tearDown(cls):
        # clean up cache
        p = subprocess.run(
            f"{which('rm')} {cls.cache_dir}",
            shell=True,
            capture_output=False,
            encoding="utf-8",
            check=True,
        )

        del cls.ad

    def test_test(self):
        self.assertTrue(True)
