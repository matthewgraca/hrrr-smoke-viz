import unittest
from libs.openaqdata import OpenAQData
import re

class TestOpenAQData(unittest.TestCase):
    def test_responses(self):
        '''
        Just check if the string printed has the corresponding http code 
        '''
        aq = OpenAQData(
            save_dir='tests/openaqdata/data',
            load_json=True,
            verbose=2
        ) 
        expected = [
            200, 401, 403, 404, 405, 408, 410, 
            422, 429, 500, 501, 502, 503, 504,
            600, 700, 800
        ]
        strings = [aq._get_response_msg(code) for code in expected]
        actual = [int(re.match(r"^(\d+)", text).group(1)) for text in strings]
        self.assertEqual(expected, actual)

    def test_loading_json_from_cache(self):
        aq = OpenAQData(
            save_dir='tests/openaqdata/data',
            load_json=True,
            verbose=2
        ) 

    def test_bad_locations_query_throws_value_error(self):
        with self.assertRaises(ValueError):
            aq = OpenAQData(extent=(-1000, -1000, -1000, -1000), verbose=0) 
