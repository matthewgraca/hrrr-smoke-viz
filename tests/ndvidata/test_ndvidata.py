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
    def tearDown(cls):
        del cls.N

    def test_dummy(self):
        self.assertTrue(True)
