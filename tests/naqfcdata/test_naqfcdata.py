import unittest
from libs.naqfcdata import NAQFCData 
import pandas as pd
import os
import numpy as np

class TestNAQFCData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.local_path = 'tests/naqfcdata/data/'

        cls.models = {
            'pm25' : 'aqm',
            'o3' : 'aqm',
            'dust' : 'dust',
            'smoke' : 'smoke'
        }

        cls.model_init_times = {
            'aqm' : ['06', '12'],
            'dust' : ['06', '12'],
            'smoke' : ['03']
        }

        cls.nd = NAQFCData(
            start_date="2025-01-10 00:00",
            end_date="2025-01-17 00:59",
            local_path=cls.local_path,
            save_path='tests/naqfcdata/data/',
        )

    @classmethod
    def tearDownClass(cls):
        del cls.nd
        del cls.models
        del cls.model_init_times

    # NOTE: Tests for searching for models in the bucket
    def test_can_find_aqm_model(self):
        expected = 1
        actual = len(self.nd._find_model_directories(self.nd._s3, model='aqm'))
        self.assertTrue(actual >= expected)

    def test_can_find_dust_model(self):
        expected = 1
        actual = len(self.nd._find_model_directories(self.nd._s3, model='dust'))
        self.assertTrue(actual >= expected)

    def test_can_find_smoke_model(self):
        expected = 1
        actual = len(self.nd._find_model_directories(self.nd._s3, model='smoke'))
        self.assertTrue(actual >= expected)

    def test_random_not_found_in_files(self):
        with self.assertRaises(ValueError) as err:
            self.nd._find_model_directories(self.nd._s3, model='random')

   # NOTE: Test being able to find the files in the bucket 

    def test_find_files_in_one_week(self):
        prefix = 'noaa-nws-naqfc-pds/AQMv7/CS/'
        expected = [
            f'{prefix}20250109/12/aqm.t12z.ave_1hr_pm25_bc.20250109.227.grib2',
            f'{prefix}20250110/06/aqm.t06z.ave_1hr_pm25_bc.20250110.227.grib2',
            f'{prefix}20250110/12/aqm.t12z.ave_1hr_pm25_bc.20250110.227.grib2',
            f'{prefix}20250111/06/aqm.t06z.ave_1hr_pm25_bc.20250111.227.grib2',
            f'{prefix}20250111/12/aqm.t12z.ave_1hr_pm25_bc.20250111.227.grib2',
            f'{prefix}20250112/06/aqm.t06z.ave_1hr_pm25_bc.20250112.227.grib2',
            f'{prefix}20250112/12/aqm.t12z.ave_1hr_pm25_bc.20250112.227.grib2',
            f'{prefix}20250113/06/aqm.t06z.ave_1hr_pm25_bc.20250113.227.grib2',
            f'{prefix}20250113/12/aqm.t12z.ave_1hr_pm25_bc.20250113.227.grib2',
            f'{prefix}20250114/06/aqm.t06z.ave_1hr_pm25_bc.20250114.227.grib2',
            f'{prefix}20250114/12/aqm.t12z.ave_1hr_pm25_bc.20250114.227.grib2',
            f'{prefix}20250115/06/aqm.t06z.ave_1hr_pm25_bc.20250115.227.grib2',
            f'{prefix}20250115/12/aqm.t12z.ave_1hr_pm25_bc.20250115.227.grib2',
            f'{prefix}20250116/06/aqm.t06z.ave_1hr_pm25_bc.20250116.227.grib2',
            f'{prefix}20250116/12/aqm.t12z.ave_1hr_pm25_bc.20250116.227.grib2',
        ]
        actual = self.nd._get_file_paths(
            self.nd._s3,
            self.models,
            self.model_init_times,
            self.nd.product,
            pd.to_datetime(self.nd.start_date),
            pd.to_datetime(self.nd.end_date)
        )
        self.assertEqual(expected, actual)

    def test_find_files_across_models(self):
        nd = NAQFCData(
            start_date="2024-05-10 00:00",
            end_date="2024-05-17 00:59",
            local_path='tests/naqfcdata/data',
            save_path='tests/naqfcdata/data'
        )

        models = {
            'pm25' : 'aqm',
            'o3' : 'aqm',
            'dust' : 'dust',
            'smoke' : 'smoke'
        }
        prefix = 'noaa-nws-naqfc-pds/'
        expected = [
            f'{prefix}AQMv6/CS/20240509/12/aqm.t12z.ave_1hr_pm25_bc.20240509.227.grib2',
            f'{prefix}AQMv6/CS/20240510/06/aqm.t06z.ave_1hr_pm25_bc.20240510.227.grib2',
            f'{prefix}AQMv6/CS/20240510/12/aqm.t12z.ave_1hr_pm25_bc.20240510.227.grib2',
            f'{prefix}AQMv6/CS/20240511/06/aqm.t06z.ave_1hr_pm25_bc.20240511.227.grib2',
            f'{prefix}AQMv6/CS/20240511/12/aqm.t12z.ave_1hr_pm25_bc.20240511.227.grib2',
            f'{prefix}AQMv6/CS/20240512/06/aqm.t06z.ave_1hr_pm25_bc.20240512.227.grib2',
            f'{prefix}AQMv6/CS/20240512/12/aqm.t12z.ave_1hr_pm25_bc.20240512.227.grib2',
            f'{prefix}AQMv6/CS/20240513/06/aqm.t06z.ave_1hr_pm25_bc.20240513.227.grib2',
            f'{prefix}AQMv6/CS/20240513/12/aqm.t12z.ave_1hr_pm25_bc.20240513.227.grib2',
            f'{prefix}AQMv7/CS/20240514/06/aqm.t06z.ave_1hr_pm25_bc.20240514.227.grib2',
            f'{prefix}AQMv7/CS/20240514/12/aqm.t12z.ave_1hr_pm25_bc.20240514.227.grib2',
            f'{prefix}AQMv7/CS/20240515/06/aqm.t06z.ave_1hr_pm25_bc.20240515.227.grib2',
            f'{prefix}AQMv7/CS/20240515/12/aqm.t12z.ave_1hr_pm25_bc.20240515.227.grib2',
            f'{prefix}AQMv7/CS/20240516/06/aqm.t06z.ave_1hr_pm25_bc.20240516.227.grib2',
            f'{prefix}AQMv7/CS/20240516/12/aqm.t12z.ave_1hr_pm25_bc.20240516.227.grib2',
        ]

        actual = self.nd._get_file_paths(
            nd._s3,
            self.models,
            self.model_init_times,
            nd.product,
            pd.to_datetime(nd.start_date),
            pd.to_datetime(nd.end_date)
        )
        self.assertEqual(expected, actual)

    def test_download_worked(self):
        expected = set([
            'aqm.t06z.ave_1hr_pm25_bc.20250110.227.grib2',
            'aqm.t12z.ave_1hr_pm25_bc.20250110.227.grib2',
            'aqm.t06z.ave_1hr_pm25_bc.20250111.227.grib2',
            'aqm.t12z.ave_1hr_pm25_bc.20250111.227.grib2',
            'aqm.t06z.ave_1hr_pm25_bc.20250112.227.grib2',
            'aqm.t12z.ave_1hr_pm25_bc.20250112.227.grib2',
            'aqm.t06z.ave_1hr_pm25_bc.20250113.227.grib2',
            'aqm.t12z.ave_1hr_pm25_bc.20250113.227.grib2',
            'aqm.t06z.ave_1hr_pm25_bc.20250114.227.grib2',
            'aqm.t12z.ave_1hr_pm25_bc.20250114.227.grib2',
            'aqm.t06z.ave_1hr_pm25_bc.20250115.227.grib2',
            'aqm.t12z.ave_1hr_pm25_bc.20250115.227.grib2',
            'aqm.t06z.ave_1hr_pm25_bc.20250116.227.grib2',
            'aqm.t12z.ave_1hr_pm25_bc.20250116.227.grib2',
            'aqm.t06z.ave_1hr_pm25_bc.20250117.227.grib2',
            'aqm.t12z.ave_1hr_pm25_bc.20250117.227.grib2'
        ])
        actual = set(os.listdir(os.path.join(
            self.local_path, 
            'noaa-nws-naqfc-pds-pm25'
        )))
        self.assertTrue(expected.issubset(actual))

    # NOTE: test some downloading helpers

    def test_backfilling(self):
        expected = (
            pd.to_datetime('2025-01-09 00:00:00+00:00'),
            1,
            pd.to_datetime('2025-01-16 00:59:00+00:00'),
            0
        )
        actual = self.nd._backfill_order(
            self.models,
            self.nd.product,
            pd.to_datetime(self.nd.start_date, utc=True),
            pd.to_datetime(self.nd.end_date, utc=True)
        )
        self.assertEqual(expected, actual)

    # NOTE: test caching logic

    def test_loading_from_cache_without_save_path_raises_error(self):
        with self.assertRaises(ValueError) as err:
            nd = NAQFCData(
                start_date="2025-01-10 00:00",
                end_date="2025-01-17 00:59",
                local_path='tests/naqfcdata/data/',
                save_path=None,
                load_numpy=True
            )

    def test_saving_the_cached_data(self):
        path = 'tests/naqfcdata/data/naqfc_pm25_processed.npz'
        self.nd._load_numpy_cache(path)
        self.assertTrue(True)
