import unittest
from libs.openaqdata import OpenAQData
import os
import pandas as pd
import requests

class TestOpenAQDataQueries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.aq = OpenAQData(
            test_mode=True
        )
    
    @classmethod
    def tearDownClass(cls):
        del cls.aq

    def test_multiple_queries_for_single_sensor(self):
        start = pd.to_datetime("2023-08-02")
        end = pd.to_datetime("2023-10-02")

        actual = len(self.aq._measurement_queries_for_a_sensor(
            api_key=os.getenv('OPENAQ_API_KEY'),
            sensor_id=2150,
            start=start-pd.Timedelta(hours=1),
            end=end-pd.Timedelta(hours=2),
            save_dir='tests/data',
            verbose=0
        ))
        expected = len(pd.date_range(start, end, freq='h', inclusive='left', tz='UTC'))
        self.assertEqual(expected, actual)

    def test_bad_locations_query_throws_value_error(self):
        with self.assertRaises(requests.exceptions.HTTPError):
            aq = OpenAQData(extent=(-1000, -1000, -1000, -1000), verbose=2) 
