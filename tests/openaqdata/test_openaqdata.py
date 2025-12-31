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
            verbose=2,          
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
            'Rowan ES (6425)': (32, 2),
            'Garfield HS/Monterey CHS (8677)': (32, 8),
            'Harrison ES (4438)': (25, 2),
            'Sierra Park ES (6753)': (18, 6),
            'Annandale ES (2151)': (5, 2),
            'Multnomah St ES (5425)': (21, 1),
            'Sierra Vista ES (6767)': (14, 8),
            'El Sereno ES (3562)': (15, 4),
            'San Pascual ES (6493)': (8, 5),
            'UltimateReality': (3, 19)
        }

        df_locations = pd.read_csv(
            f'{self.save_dir}/locations_summary.csv',
            index_col='Unnamed: 0'
        )

        actual = self.aq._get_sensor_locations_on_grid(
                df_locations=df_locations,
                dim=self.dim,
                extent=self.aq.extent
        ).set_index('locations')['x, y'].to_dict()

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
            locations_on_grid=list(self.aq._get_sensor_locations_on_grid(
                df_locations=df_locations,
                dim=self.dim,
                extent=self.aq.extent
            )['x, y'])
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
            pd.to_datetime(self.aq.start_date, utc=True),
            pd.to_datetime(self.aq.end_date, utc=True), 
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

    def test_sensor_values_get_merged_on_the_same_location(self):
        actual_loc_to_vals = self.aq._merge_values_in_the_same_location(
            data=[1, 3, 4, 5, 6, 7, 8],
            locations_on_grid=[(0, 0), (1, 0), (3, 5), (4, 5), (0, 8), (0, 1), (0, 0)],
        )
        actual_vals = list(actual_loc_to_vals.values())
        actual_locs = list(actual_loc_to_vals.keys())
        expected_vals = np.array([4.5, 3, 4, 5, 6, 7])
        expected_locs = [(0, 0), (1, 0), (3, 5), (4, 5), (0, 8), (0, 1)]

        np.testing.assert_allclose(actual_vals, expected_vals)
        np.testing.assert_allclose(actual_locs, expected_locs)

    def test_sensor_values_merged_on_the_same_location_ignores_nan(self):
        actual_loc_to_vals = self.aq._merge_values_in_the_same_location(
            data=[1, 3, 4, 5, 6, 7, np.nan],
            locations_on_grid=[(0, 0), (1, 0), (3, 5), (4, 5), (0, 8), (0, 1), (0, 0)],
        )
        actual_vals = list(actual_loc_to_vals.values())
        actual_locs = list(actual_loc_to_vals.keys())
        expected_vals = np.array([1, 3, 4, 5, 6, 7])
        expected_locs = [(0, 0), (1, 0), (3, 5), (4, 5), (0, 8), (0, 1)]

        np.testing.assert_allclose(actual_vals, expected_vals)
        np.testing.assert_allclose(actual_locs, expected_locs)

    def test_annual_date_split(self):
        actual_start, actual_end = self.aq._annually_split_dates(
            start_dt=pd.to_datetime("2023-08-02", utc=True),
            end_dt=pd.to_datetime("2025-08-02", utc=True),
        )
        actual = len(actual_start), len(actual_end)
        expected = 2, 2

        self.assertEqual(actual, expected)

    def test_nowcast(self):
        airnow_df = pd.read_csv(f'{self.save_dir}/airnow_la_sensor_subset.csv')
        raw_df = pd.read_csv(f'{self.save_dir}/openaq_la_sensor_subset.csv')
        nowcast_df = self.aq._compute_nowcast(raw_df) 

        actual = airnow_df['Value'].values[12:]
        expected = np.squeeze(nowcast_df.values[12:])

        np.testing.assert_allclose(actual, expected, atol=0.1)
