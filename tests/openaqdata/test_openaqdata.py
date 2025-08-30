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
            test_mode=True
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

    '''
    def test_loading_json_from_cache(self):
        aq = OpenAQData(
            save_dir='tests/openaqdata/data',
            load_json=True,
            verbose=0
        ) 
    '''
