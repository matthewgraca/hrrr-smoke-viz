import unittest
from libs.hrrrdata import HRRRData
from herbie import Herbie, wgrib2
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
        cls._herbie = Herbie("2021-01-01", model='hrrr')
        cls._grib_found = bool(cls._herbie)
        cls._herbie.download("COLMD")

    def setUp(self):
        if not self._grib_found:
            self.fail("Grib file not found, failing all tests.")

    def test_subregion_with_extent(self):
        '''
        Simple test that subregion with extent creates the same file as wgrib2.
        '''
        instance = HRRRData.__new__(HRRRData)  

        extent = (-118.75, -117.5, 33.5, 34.5)
        extent_name = "test_region"
        product = "COLMD"

        actual = instance._subregion_grib_files(
            [self._herbie], 
            extent,
            extent_name,
            product
        )[0]
        expected = wgrib2.region(
            path=self._herbie.get_localFilePath(product), 
            extent=extent, 
            name=extent_name
        )

        self.assertEqual(
            actual, 
            expected,
            f"Expected {expected}, returned {actual}."
        )

    def test_subregion_with_no_extent(self):
        '''
        Ensure that with no extent, subregion does nothing.
        Basically, subregion should just be the same as doing nothing;
        i.e. the real file and the subregioned file are one and the same.
        '''
        instance = HRRRData.__new__(HRRRData)  

        extent = None
        extent_name = "test_region"
        product = "COLMD"

        actual = instance._subregion_grib_files(
            [self._herbie], 
            extent,
            extent_name,
            product
        )[0]
        expected = self._herbie.get_localFilePath(product)
        
        self.assertEqual(
            actual, 
            expected,
            f"Expected {expected}, returned {actual}."
        )

    def test_grib_file_with_no_subregion_converts_to_np(self):
        '''
        Test a single grib file without subregion converting to numpy.
        I've had issues where a grib file without being subregioned 
        couldn't be converted to numpy.
        '''
        instance = HRRRData.__new__(HRRRData)  

        extent = None
        extent_name = "test_region"
        product = "COLMD"
        files = instance._subregion_grib_files(
            [self._herbie], 
            extent,
            extent_name,
            product
        )

        actual = instance._grib_to_np(files)
        expected = instance._grib_to_np(
            [self._herbie.get_localFilePath(product)]
        )

        np.testing.assert_array_equal(actual, expected) 

    def test_many_grib_files_with_no_subregion_converts_to_np(self):
        '''
        Test multiple grib files without subregion converting to numpy.
        I've had issues where a grib file without being subregioned 
        couldn't be converted to numpy.
        '''
        instance = HRRRData.__new__(HRRRData)  

        extent = None
        extent_name = "test_region"
        product = "COLMD"
        files = instance._subregion_grib_files(
            [self._herbie, self._herbie], 
            extent,
            extent_name,
            product
        )

        actual = instance._grib_to_np(files)
        expected = instance._grib_to_np(
            [
                self._herbie.get_localFilePath(product), 
                self._herbie.get_localFilePath(product)
            ]
        )

        np.testing.assert_array_equal(actual, expected) 
