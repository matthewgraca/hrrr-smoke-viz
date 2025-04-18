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
        extent_name='subset_region',
        product='MASSDEN',
        frames_per_sample=1,
        dim=40,
        sample_setting=1,
        verbose=False
    ):
        # pipeline
        # frame-by-frame sampling
        if sample_setting == 1:
            herbie_ds = self.__get_hrrr_data_frame_by_frame(
                start_date, end_date, product, verbose
            )
            subregion_grib_ds = self.__subregion_grib_files(
                herbie_ds, extent, extent_name, product
            )
            subregion_frames = self.__grib_to_np(subregion_grib_ds)
            preprocessed_frames = self.__interpolate_and_add_channel_axis(
                subregion_frames, dim
            )
            processed_ds = self.__sliding_window_of(
                preprocessed_frames, frames_per_sample
            )

            # attributes
            self.data = processed_ds
            self.herbie = herbie_ds

        # offset-by-sample with forecast sampling
        elif sample_setting == 2:
            # generate a list of samples. each sample is n frames (forecasts)
            # no sliding window needed; n frames already generated
            herbie_ds = self.__get_hrrr_data_offset_by_forecast(
                start_date, end_date, frames_per_sample, product, verbose
            )
            subregion_grib_ds = self.__subregion_grib_files(
                herbie_ds, extent, extent_name, product
            )
            subregion_frames = self.__grib_to_np(subregion_grib_ds)
            preprocessed_frames = self.__interpolate_and_add_channel_axis(
                subregion_frames, dim
            )
            # sliding window needs to be offset by the number of frames
            processed_ds = self.__sliding_window_of(
                frames=preprocessed_frames, 
                window_size=frames_per_sample, 
                full_slide=True 
            )

            # attributes
            self.data = processed_ds
            self.herbie = herbie_ds

        else:
            msg = (
                "Argument \"sample_setting\" must be either:\n",
                "1 - frame-by-frame\n",
                "2 - offset-by-sample with forecasts\n"
            )
            raise ValueError(" ".join(msg))

    '''
    Uses FastHerbie to grab the remote data, and download it; frame by frame.

    Arguments:
        start_date: The start date of the query, in the form "yyyy-mm-dd-hh"
        end_date: The end date of the query (inclusive)
        product: regex of the product to download 
        verbose: Determines if Herbie objects should be printed

    Returns:
        The list of Herbie objects of the downloaded data
    '''
    def __get_hrrr_data_frame_by_frame(
        self, 
        start_date, 
        end_date, 
        product, 
        verbose
    ):
        dates = pd.date_range(start_date, end_date, freq="1h")
        FH = FastHerbie(dates, model="hrrr", fxx=[0])
        FH.download(product)

        if verbose:
            [print(repr(H)) for H in FH.objects]

        return FH.objects

    '''
    Uses FastHerbie to grab the remote data, and download it; offset by the
    number of frames per sample, and using forecasts.

    Arguments:
        start_date: The start date of the query, in the form "yyyy-mm-dd-hh"
        end_date: The end date of the query (inclusive)
        offset: The number of frames per sample we offset by
        product: regex of the product to download 
        verbose: Determines if Herbie objects should be printed

    Returns:
        The list of Herbie objects of the downloaded data

    '''
    def __get_hrrr_data_offset_by_forecast(
        self, 
        start_date,
        end_date,
        offset,
        product,
        verbose
    ):
        # if sample is t=0, pull init @ 00, fxx = 01 so we provide next-sample forecast
        offset_start_date = pd.to_datetime(start_date) + pd.Timedelta(hours=offset - 1)
        dates = pd.date_range(offset_start_date, end_date, freq="1h")
        FH = FastHerbie(dates, model="hrrr", fxx=[i for i in range(1, offset + 1)])
        FH.download(product)

        if verbose:
            [print(repr(H)) for H in FH.objects]

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
        extent_name: Desired name of the subregion
        product: Regex of the product that was downloaded

    Returns:
        A list of the subregioned grib files
    '''
    def __subregion_grib_files(self, herbie_data, extent, extent_name, product):
        subregion_grib_files = []
        for H in herbie_data:
            # subregion grib files
            file = H.get_localFilePath(product)
            idx_file = wgrib2.create_inventory_file(file)
            subset_file = wgrib2.region(file, extent, name=extent_name)
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
    Uses a sliding window to create samples from frames.

    For example, with a step size of 5, with 15 frames and a 5 frames per 
    sample, frames: 
        - 1-5
        - 6-10
        - 11-15
    will make up a total of 3 samples.

    For example, with a step size of 1, with 5 frames and a 3 frames per 
    sample, frames: 
        - 1-3 
        - 2-4
        - 3-5 
    will make up a total of 3 samples.

    Arguments:
        frames: A numpy array of the shape (num_frames, row, col, channels)
        window_size: The desired number of frames per sample
        full_slide: Whether the window should slide by 1 (false) or by the size
            of the window (true)

    Returns:
        A numpy array of the shape (num_samples, num_frames, row, col, channels)
    '''
    def __sliding_window_of(self, frames, window_size, full_slide=False):
        n_frames, row, col, channels = frames.shape
        n_samples = (
            n_frames - window_size + 1
            if full_slide == False
            else n_frames // window_size
        )

        samples = np.empty((n_samples, window_size, row, col, channels))

        for i in range(n_samples):
            start_idx = i if full_slide == False else i * window_size 
            end_idx = start_idx + window_size

            samples[i] = np.array(
                [frames[j] for j in range(start_idx, end_idx)]
            )

        return samples
