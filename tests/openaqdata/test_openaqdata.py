import unittest
from libs.openaqdata import OpenAQData
import re
import pandas as pd
import os
import requests
from dotenv import load_dotenv
load_dotenv()

class TestOpenAQData(unittest.TestCase):
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
            load_json=True,    
            load_numpy=False,      
            verbose=2,          
        )
    
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
