import unittest
from libs.hrrrdata import HRRRData
from herbie import Herbie, wgrib2
import numpy as np
import pandas as pd
import time
from requests.exceptions import ConnectionError
import subprocess
from shutil import which
import os

# default test parameters we use a lot
PRODUCT = "COLMD"
LA_EXTENT = (-118.75, -117.5, 33.5, 34.5)
LA_EXTENT_NAME = "test_region"
DATE_RANGE = pd.date_range("2021-01-01", "2021-01-01-03", freq='h')

def _safe_herbie(date):
    '''
    But who will unit test the unit test helpers? 🤔

    Get Herbie object, handling connection reset error
    '''
    success = False
    retries = 1 
    max_retries = 5
    while retries < max_retries: 
        try:
            return Herbie(date)
        except ConnectionError as e:
            retries += 1
            print(f"Failed, attempt {retries}...")
            time.sleep(5)

    raise Exception("Retried too many times.")

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
        cls._herbie = _safe_herbie("2021-01-01")
        if not cls._herbie:
            unittest.SkipTest("Grib file not found, failing all tests.")
        cls._herbie.download("COLMD")

    # NOTE complete class test
    def test_hrrrdata_doesnt_explode(self):
        '''
        The one tap; tests if the entire pipeline works without exploding.
        This meant to be a sanity check, not a comprehensive check (that's what
        the unit tests are for). We check if the combination of all the functions
        work in the pipeline (i.e. no exceptions get thrown).
        '''
        try:
            print()
            hd = HRRRData(
                start_date="2024-01-01", 
                end_date="2024-01-01-03",
                extent=None,
                extent_name="buh",
                product="COLMD",
                frames_per_sample=1,
                dim=40,
                sample_setting=1,
                verbose=False,
                processed_cache_dir=None,
                force_reprocess=False
            )
        except Exception as e:
            self.fail(f"Exception got raised: {type(e).__name__}: {e}")

        self.assertTrue(True)

    # NOTE _subregion_grib_files() tests
    def test_subregion_with_extent(self):
        '''
        Simple test that subregion with extent creates the same file as wgrib2.
        '''
        instance = HRRRData.__new__(HRRRData)  

        actual = instance._subregion_grib_files(
            [self._herbie], 
            LA_EXTENT,
            LA_EXTENT_NAME,
            PRODUCT 
        )[0]
        expected = wgrib2.region(
            path=self._herbie.get_localFilePath(PRODUCT), 
            extent=LA_EXTENT, 
            name=LA_EXTENT_NAME
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

        actual = instance._subregion_grib_files(
            [self._herbie], 
            extent,
            LA_EXTENT_NAME,
            PRODUCT 
        )[0]
        expected = self._herbie.get_localFilePath(PRODUCT)
        
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
        files = instance._subregion_grib_files(
            [self._herbie], 
            extent,
            LA_EXTENT_NAME,
            PRODUCT 
        )

        actual = instance._grib_to_np(files)
        expected = instance._grib_to_np(
            [self._herbie.get_localFilePath(PRODUCT)]
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
        files = instance._subregion_grib_files(
            [self._herbie, self._herbie], 
            extent,
            LA_EXTENT_NAME, 
            PRODUCT 
        )

        actual = instance._grib_to_np(files)
        expected = instance._grib_to_np(
            [
                self._herbie.get_localFilePath(PRODUCT), 
                self._herbie.get_localFilePath(PRODUCT)
            ]
        )

        np.testing.assert_array_equal(actual, expected) 

    def test_grib_to_np_raises_when_grib_file_is_empty(self):
        instance = HRRRData.__new__(HRRRData)  

        file_path = "tests/hrrrdata/data/empty_file.grib2"
        with self.assertRaises(EOFError):
            actual = instance._grib_to_np([file_path])

    def test_verify_download_fails_with_fake_data(self):
        '''
        Will trigger the redownload, then throw an exception
        because it can't delete the offending file with a fake subset
        '''
        instance = HRRRData.__new__(HRRRData)  
        product = "asdfasflk"

        with self.assertRaises(Exception):
            instance._verify_downloads(self._herbie, product)

    # NOTE _attempt_download() tests
    def test_multithread_dl_matches_singlethread_dl(self):
        '''
        Checking if the multithread download works compared to the single
        '''
        instance = HRRRData.__new__(HRRRData)  

        print()
        actual = [
            H.get_localFilePath(PRODUCT)
            for H in instance._attempt_download(
                DATE_RANGE,
                PRODUCT 
            )
        ]

        expected = [
            H.get_localFilePath(PRODUCT) 
            for H in [_safe_herbie(date) for date in DATE_RANGE]
        ]

        self.assertListEqual(
            actual, 
            expected,
            f"Expected {expected}, returned {actual}."
        )

    # NOTE subregion_grib_files() tests
    def test_missing_grib_hits_the_exception(self):
        '''
        Subregion fails when file deleted before subregioning
        I don't know how to test `wgrib2` failing, but it's CalledProcessError,
        so I'm going to test by using `rm`. 

        I wouldn't consider subregion_grib_files() tested until we can actually
        catch this error red-handed.
        '''
        instance = HRRRData.__new__(HRRRData)  

        print()
        H = [_safe_herbie(pd.to_datetime("2021-02-01"))]
        with self.assertRaises(subprocess.CalledProcessError):
            p = subprocess.run(
                f"{which("rm")} {H[0].get_localFilePath(PRODUCT)}",
                shell=True,
                capture_output=True,
                encoding="utf-8",
                check=True,
            )

    # NOTE caching tests
    def test_saving_cache_works(self):
        '''
        Checks if saving data to the cache actually works.
        '''
        cache_dir = 'tests/hrrrdata/data/hrrr_processed.npz'
        print()
        hd = HRRRData(
            start_date="2024-03-01-00", 
            end_date="2024-03-01-01",
            extent=None,
            extent_name="buh",
            product=PRODUCT,
            frames_per_sample=1,
            dim=40,
            sample_setting=1,
            verbose=False,
            processed_cache_dir=cache_dir,
            force_reprocess=False
        )
        
        path_exists = os.path.exists(cache_dir)

        # clean up for subsequent tests
        if path_exists:
            os.remove(cache_dir)

        msg = f"{cache_dir} does not exist."
        self.assertTrue(path_exists, msg)

    def test_no_cache_means_no_saving(self):
        '''
        Checks that setting processed_cache_dir to None doesn't save the data.
        '''
        cache_dir = 'tests/hrrrdata/data/hrrr_processed.npz'
        print()
        hd = HRRRData(
            start_date="2024-03-01-00", 
            end_date="2024-03-01-01",
            extent=None,
            extent_name="buh",
            product=PRODUCT,
            frames_per_sample=1,
            dim=40,
            sample_setting=1,
            verbose=False,
            processed_cache_dir=None,
            force_reprocess=False
        )
        
        path_exists = os.path.exists(cache_dir)

        # clean up for subsequent tests
        if path_exists:
            os.remove(cache_dir)

        msg = f"{cache_dir} should not exist."
        self.assertFalse(path_exists, msg)

    def test_saving_and_loading_from_cache(self):
        '''
        Checks if saving to AND loading from cache actually works.
        '''
        # create cache
        cache_dir = 'tests/hrrrdata/data/hrrr_processed.npz'
        print()
        hd = HRRRData(
            start_date="2024-03-01-00", 
            end_date="2024-03-01-01",
            extent=None,
            extent_name="buh",
            product=PRODUCT,
            frames_per_sample=1,
            dim=40,
            sample_setting=1,
            verbose=False,
            processed_cache_dir=cache_dir,
            force_reprocess=False
        )

        # read cache
        hd2 = HRRRData(
            start_date="2024-03-01-00", 
            end_date="2024-03-01-01",
            extent=None,
            extent_name="buh",
            product=PRODUCT,
            frames_per_sample=1,
            dim=40,
            sample_setting=1,
            verbose=False,
            processed_cache_dir=cache_dir,
            force_reprocess=False
        )
        
        path_exists = os.path.exists(cache_dir)

        # clean up for subsequent tests
        if path_exists:
            os.remove(cache_dir)
            msg = f"No data found in cache."
            self.assertTrue(len(hd2.data) > 0, msg)
        else:
            msg = f"{cache_dir} does not exist."
            self.assertTrue(path_exists, msg)
