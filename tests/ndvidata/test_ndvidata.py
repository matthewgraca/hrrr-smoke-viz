import unittest
from libs.ndvidata import NDVIData

class TestNDVIData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        N = NDVIData(
            start_date='2023-08-02',
            end_date='2023-08-22',
            extent=(-118.615, -117.70, 33.60, 34.35),
            dim=84,
            raw_dir='tests/ndvidata/data',
            save_dir='tests/ndvidata/data'
        )
        cls.N = N

    @classmethod
    def tearDownClass(cls):
        del cls.N

    def test_loads_cache_without_raising_errors(self):
        ndata = NDVIData(cache_path='tests/ndvidata/data/ndvi_processed.npz', verbose=2)
        self.assertTrue(True)
