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
        cls.extent = extent
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
            df=pd.DataFrame({
                'lat' : df_locations['latitude'],
                'lon' : df_locations['longitude'],
                'val' : values
            }),
            dim=self.dim,
            extent=self.extent,
        )
        actual = len(np.where(~np.isnan(grid))[0])

        self.assertEqual(expected, actual)
        self.assertEqual((self.dim , self.dim), grid.shape)
