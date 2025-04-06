from herbie import Herbie, FastHerbie, wgrib2
from pathlib import Path
from datetime import timedelta
import pandas as pd
import numpy as np
import xarray as xr
import cv2

class HRRRData:
    '''
    Gets the HRRR data.
    Pipeline:
        - Use Herbie to download the data as grib files
        - Use wgrib2 to subregion the grib files
        - Convert the grib files to xarray, then into numpy as frames
            - (num_frames, row, col)
        - Interpolate and add a channel axis to the array of frames
            - (num_frames, row, col, channel)
        - Create samples from a sliding window of frames
            - (samples, frames, row, col, channel)

    Members:
        data: The complete processed HRRR data
        herbie: The inventory of Herbie objects of the HRRR data
    '''
    def __init__(
        self,
        start_date,
        end_date,
        extent=None,
        product='MASSDEN',
        frames_per_sample=1,
        dim=40,
        verbose=False
    ):
        herbie_ds = self.__get_hrrr_data(start_date, end_date, product, verbose)
        subregion_grib_ds = self.__subregion_grib_files(herbie_ds, extent, product)
        subregion_frames = self.__grib_to_np(subregion_grib_ds)
        preprocessed_frames = self.__interpolate_and_add_channel_axis(subregion_frames, dim)
        processed_ds = self.__sliding_window_of(preprocessed_frames, frames_per_sample)

        self.data = processed_ds
        self.herbie = herbie_ds

    '''
    Uses FastHerbie to grab the remote data, and download it.

    Arguments:
        start_date: The start date of the query, in the form "yyyy-mm-dd-hh"
        end_date: The end date of the query (inclusive)
        product: regex of the product to download 
        verbose: Determines if Herbie objects should be printed

    Retunrs:
        The list of Herbie objects of the downloaded data
    '''
    def __get_hrrr_data(self, start_date, end_date, product, verbose):
        dates = pd.date_range(start_date, end_date, freq="1h")
        FH = FastHerbie(dates, model="hrrr", fxx=[0])
        FH.download(product)

        if verbose:
            [print(H) for H in FH.objects]

        return FH.objects

    '''
    Takes a list of Herbie objects, and subregions the downloaded grib files

    Arguments:
        herbie_data: The list containing the Herbie objects
        extent: The bounding box with the shape: (a, b, c, d):
            - a: bottom longitude
            - b: top longitude
            - c: bottom latitude
            - d: top latitude
        product: Regex of the product that was downloaded

    Returns:
        A list of the subregioned grib files
    '''
    def __subregion_grib_files(self, herbie_data, extent, product):
        subregion_grib_files = []
        for H in herbie_data:
            # subregion grib files
            file = H.get_localFilePath(product)
            idx_file = wgrib2.create_inventory_file(file)
            subset_file = wgrib2.region(file, extent, name="la_region")
            subregion_grib_files.append(subset_file)

        return subregion_grib_files

    '''
    Converts a list of downloaded grib files into numpy

    Arguments:
        grib_files: The paths of the grib files being converted

    Returns:
        A numpy array of the grib files
    '''
    def __grib_to_np(self, grib_files):
        np_of_grib = []
        for file in grib_files:
            hrrr_xarr = xr.open_dataset(file, engine="cfgrib", decode_timedelta=False)
            # hrrr data comes in (y, x), so it will need to be flipped for (x, y)
            np_of_grib.append(np.flip(hrrr_xarr.mdens.to_numpy(), axis=0))

        return np.array(np_of_grib)

    ''' 
    Takes a dataset frames (x, y), interpolates/decimates according to
    the dimensions, and adds a channel axis.

    Arguments:
        subregion_ds: The numpy array containing a list of frames
        dim: The desired dimensions

    Returns:
        A numpy array of frames
    '''
    def __interpolate_and_add_channel_axis(self, subregion_ds, dim):
        n_frames = len(subregion_ds)
        channels = 1
        frames = np.empty(shape=(n_frames, dim, dim, channels))

        # interpolate and add channel axis
        for i, frame in enumerate(subregion_ds):
            new_frame = cv2.resize(frame, (dim, dim))
            new_frame = np.reshape(new_frame, (dim, dim, channels))
            frames[i] = new_frame

        return frames

    '''
    Uses a sliding window to bundle frames into samples. 
    For example, with 5 frames and a 3 frames per sample, frames: 
        - 1-3 
        - 2-4
        - 3-5 
    will make up a total of 3 samples.

    Arguments:
        frames: A numpy array of the shape (num_frames, row, col, channels)
        frames_per_sample: The desired number of frames for each sample

    Returns:
        A numpy array of the shape (num_samples, num_frames, row, col, channels)
    '''
    def __sliding_window_of(self, frames, frames_per_sample):
        n_frames, row, col, channels = frames.shape
        n_samples = n_frames - frames_per_sample 
        samples = np.empty((n_samples, frames_per_sample, row, col, channels))
        for i in range(n_samples):
            samples[i] = np.array([frames[j] for j in range(i, i + frames_per_sample)])
            
        return samples
