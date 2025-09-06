import unittest
from libs.openaqdata import OpenAQData
import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

class TestOpenAQDataQueries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.aq = OpenAQData(
            api_key=os.getenv('OPENAQ_API_KEY'),
            start_date="2023-08-02",
            end_date="2023-08-09",
            extent=(-118.2, -118.0, 34.0, 34.15),
            dim=40,
            product=2,          
            save_dir='tests/openaqdata/data', 
            load_json=False,    
            load_csv=True,
            load_numpy=False,      
            verbose=2,          
        )

    @classmethod
    def tearDownClass(cls):
        del cls.aq

    def test_multiple_queries_for_single_sensor(self):
        start = pd.to_datetime("2023-08-02")
        end = pd.to_datetime("2023-10-02")
        dates = pd.date_range(start, end, freq='h', inclusive='left', tz='UTC')

        actual = len(self.aq._measurement_queries_for_a_sensor(
            api_key=os.getenv('OPENAQ_API_KEY'),
            sensor_id=2150,
            start_dt=start-pd.Timedelta(hours=1),
            end_dt=end-pd.Timedelta(hours=1),
            dates=dates,
            save_dir='tests/openaqdata/data',
        ))

        expected = len(dates)
        self.assertEqual(expected, actual)

        # we bundle the test to check json loading here since we can't 
        # guarantee that other test will run after this one here.
        # should not throw an error
        self.aq._check_datetimes_in_sensor_dir(
            'tests/openaqdata/data/measurements/2150',
            pd.to_datetime(start, utc=True),
            pd.to_datetime(end, utc=True) - pd.Timedelta(hours=1),
        )

    def test_bad_locations_query_throws_value_error(self):
        with self.assertRaises(ValueError):
            aq = OpenAQData(extent=(-1000, -1000, -1000, -1000), verbose=2) 
