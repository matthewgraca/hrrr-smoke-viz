import unittest
from libs.openaqdata import OpenAQData
import re
import pandas as pd
import os
import requests
from dotenv import load_dotenv
load_dotenv()
import numpy as np

class TestOpenAQData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        extent = (-118.2, -118.0, 34.0, 34.15)
        dim = 40
        save_dir = 'tests/openaqdata/data'
        cls.aq = OpenAQData(
            api_key=os.getenv('OPENAQ_API_KEY'),
            start_date="2023-08-02",
            end_date="2023-08-09",
            extent=extent,
            dim=dim,
            product=2,          
            save_dir=save_dir, 
            load_json=False,    
            load_csv=True,
            load_numpy=False,      
            verbose=0,          
        )
        cls.dim = dim
        cls.save_dir = save_dir
    
    @classmethod
    def tearDownClass(cls):
        del cls.aq

    def test_responses(self):
        '''
        Just check if the string printed has the corresponding http code 
        '''
        aq = self.aq 
        expected = [
            200, 401, 403, 404, 405, 408, 410, 
            422, 429, 500, 501, 502, 503, 504,
            600, 700, 800
        ]
        strings = [aq._get_response_msg(code) for code in expected]
        actual = [int(re.match(r"^(\d+)", text).group(1)) for text in strings]
        self.assertEqual(expected, actual)

    def test_sensor_locations_are_pulled(self):
        expected = {
            'Rowan ES (6425)': (32, 3),
            'Garfield HS/Monterey CHS (8677)': (32, 9),
            'Harrison ES (4438)': (25, 3),
            'Sierra Park ES (6753)': (18, 7),
            'Annandale ES (2151)': (5, 3),
            'Multnomah St ES (5425)': (21, 2),
            'Sierra Vista ES (6767)': (14, 9),
            'El Sereno ES (3562)': (15, 5),
            'San Pascual ES (6493)': (8, 6),
            'UltimateReality': (3, 20)
        }

        df_locations = pd.read_csv(
            f'{self.save_dir}/locations_summary.csv',
            index_col='Unnamed: 0'
        )

        actual = self.aq._get_sensor_locations_on_grid(
                df_locations=df_locations,
                dim=self.dim,
                extent=self.aq.extent
        )

        self.assertEqual(expected, actual)

    def test_preprocess_ground_sites_on_cache(self):
        '''
        Given locations and dummy sensor data, they should be plotted on a dim x dim grid
        '''
        expected = 10
        df_locations = pd.read_csv(
            f'{self.save_dir}/locations_summary.csv',
            index_col='Unnamed: 0'
        )
        values = [0] * len(df_locations)
        grid = self.aq._preprocess_ground_sites(
            data=values,
            dim=self.dim,
            locations_on_grid=self.aq._get_sensor_locations_on_grid(
                df_locations=df_locations,
                dim=self.dim,
                extent=self.aq.extent
            ).values()
        )
        actual = len(np.where(~np.isnan(grid))[0])

        self.assertEqual(expected, actual)
        self.assertEqual((self.dim , self.dim), grid.shape)

    def test_verbose_flags(self):
        with self.assertRaises(ValueError):
            OpenAQData(verbose="asdf")
    
        with self.assertRaises(ValueError):
            OpenAQData(verbose=-1)

    def test_sensor_values_loaded_from_csv_and_json_match(self):
        df_locations = self.aq._load_locations_from_json_cache(
            self.save_dir,
            pd.to_datetime(self.aq.start_date, utc=True) - pd.Timedelta(hours=1),
            pd.to_datetime(self.aq.end_date, utc=True) - pd.Timedelta(hours=1), 
        )
        dates = pd.date_range(
            start=self.aq.start_date, 
            end=self.aq.end_date,
            freq='h',
            inclusive='left',
            tz='UTC'
        )

        actual_1 = self.aq._load_sensor_values_from_csv_cache(self.save_dir)
        actual_2 = self.aq._load_sensor_values_from_json_cache(
            self.save_dir, 
            df_locations,
            self.aq.start_date,
            self.aq.end_date, 
            dates
        )

        np.testing.assert_allclose(np.array(actual_1), np.array(actual_2))
