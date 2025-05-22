import unittest
from libs.hrrrdata import HRRRData
from herbie import Herbie
import numpy as np

class TestHRRRData(unittest.TestCase):
    '''
    We will often test without using the intializer because we don't want
    to trigger downloads, and just want to test the functions of the class.

    Specifically, we'll use a combo of Herbie with dummy data / data we know
    to work, and an uninitialized HRRRData to test HRRRData's functions.
    '''
    @classmethod
    def setUpClass(cls):
        '''
        Set up a Herbie object that will be tested against for 
        the rest of the test class
        '''
        cls.herbie = Herbie("2021-01-01", model='hrrr')
        cls.grib_found = bool(cls.herbie)
        cls.herbie.download("COLMD")

    def setUp(self):
        if not self.grib_found:
            self.fail("Grib file not found, failing all tests.")

    def test_thing(self):
        #print(f"{self.herbie.get_localFilePath()}")
        """ playing with herbie args
        from herbie import FastHerbie
        import pandas as pd
        d = pd.date_range("2021-01-01", "2021-01-02", freq="h")
        FH = FastHerbie(d, model="hrrr", fxx=[0])
        FH.download("COLMD", verbose=True)
        """
        self.assertTrue(True)
