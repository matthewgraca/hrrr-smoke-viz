from herbie import Herbie, FastHerbie, wgrib2
from pathlib import Path
from datetime import timedelta
import pandas as pd
import numpy as np
import xarray as xr
import cv2

class HRRRData:
    def __init__(
        self,
        start_date,
        end_date,
        extent=None, #NOTE this fails, will attempt to subregion nothing
        extent_name='subset_region',
        product='MASSDEN',
        frames_per_sample=1,
        dim=40,
        sample_setting=1,
        verbose=False
    ):
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
        # pipeline
        # frame-by-frame sampling
        if sample_setting == 1:
            herbie_ds = self._get_hrrr_data_frame_by_frame(
                start_date, end_date, product, verbose
            )
            subregion_grib_ds = self._subregion_grib_files(
                herbie_ds, extent, extent_name, product
            )
            subregion_frames = self._grib_to_np(subregion_grib_ds)
            preprocessed_frames = self._interpolate_and_add_channel_axis(
                subregion_frames, dim
            )
            processed_ds = self._sliding_window_of(
                preprocessed_frames, frames_per_sample
            )

            # attributes
            self.data = processed_ds
            self.herbie = herbie_ds

        # offset-by-sample with forecast sampling
        elif sample_setting == 2:
            # generate a list of samples. each sample is n frames (forecasts)
            # no sliding window needed; n frames already generated
            herbie_ds = self._get_hrrr_data_offset_by_forecast(
                start_date, end_date, frames_per_sample, product, verbose
            )
            subregion_grib_ds = self._subregion_grib_files(
                herbie_ds, extent, extent_name, product
            )
            subregion_frames = self._grib_to_np(subregion_grib_ds)
            preprocessed_frames = self._interpolate_and_add_channel_axis(
                subregion_frames, dim
            )
            # sliding window needs to be offset by the number of frames
            processed_ds = self._sliding_window_of(
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

    def _attempt_download(self, date_range, product, forecast_range=[0]):
        '''
        Attempt to use FastHerbie to download HRRR data.
        - Will handle reset connection (104), goes for 5 max attempts
        - Any unhandled exception will just exit.
        Returns the FastHerbie object, which is pretty much a list of Herbie objects.
        '''
        downloaded = False 
        max_tries = 5
        tries = 1 
        backoff_seconds = 2 
        while not downloaded and tries < max_tries:
            try:
                tries += 1
                FH = FastHerbie(date_range, model="hrrr", fxx=forecast_range)
                FH.download(product)
                downloaded = True
            except ConnectionResetError as e:
                if tries == max_tries:
                    print(
                        f"❌ {tries}/{max_tries} attempts made.\n"
                        f"Exiting..."
                    )
                    sys.exit(1)
                print(
                    f"⚠️  Error while downloading; {e}\n"
                    f"Attempt number {tries}/{max_tries}, "
                    f"backing off by {backoff_seconds} seconds."
                )
                time.sleep(backoff_seconds)
                backoff_seconds *= 2
            except Exception as e:
                print(
                    f"❌ Something went wrong.\n"
                    f"Exception type: {type(e)}, and name:{type(e).__name__}\n"
                    f"Exception: {e}\n"
                    f"Exiting..."
                )
                sys.exit(1)

        return FH

    def _get_hrrr_data_frame_by_frame(
        self, 
        start_date, 
        end_date, 
        product, 
        verbose
    ):
        '''
        Uses FastHerbie to grab the remote data, and download it; frame by frame.

        Arguments:
            start_date: The start date of the query, in the form "yyyy-mm-dd-hh"
            end_date: The end date of the query (exclusive)
            product: regex of the product to download 
            verbose: Determines if Herbie objects should be printed

        Returns:
            The list of Herbie objects of the downloaded data
        '''
        end_date = pd.to_datetime(end_date) - pd.Timedelta(hours=1)
        dates = pd.date_range(start_date, end_date, freq="1h")

        FH = self._attempt_download(
            date_range=dates, 
            product=product, 
            forecast_range=[0]
        )

        if verbose:
            [print(repr(H)) for H in FH.objects]

        return FH.objects

    def _get_hrrr_data_offset_by_forecast(
        self, 
        start_date,
        end_date,
        offset,
        product,
        verbose
    ):
        '''
        Uses FastHerbie to grab the remote data, and download it; offset by the
        number of frames per sample, and using forecasts.

        Arguments:
            start_date: The start date of the query, in the form "yyyy-mm-dd-hh"
            end_date: The end date of the query (exclusive)
            offset: The number of frames per sample we offset by
            product: regex of the product to download 
            verbose: Determines if Herbie objects should be printed

        Returns:
            The list of Herbie objects of the downloaded data

        '''
        # if sample is t=0, pull init @ 00, fxx = 01 so we provide next-sample forecast
        offset_start_date = pd.to_datetime(start_date) + pd.Timedelta(hours=offset - 1)
        end_date = pd.to_datetime(end_date) - pd.Timedelta(hours=1)
        dates = pd.date_range(offset_start_date, end_date, freq="1h")

        FH = self._attempt_download(
            date_range=dates,
            product=product,
            forecast_range=[i for i in range(1, offset + 1)]
        )

        if verbose:
            [print(repr(H)) for H in FH.objects]

        return FH.objects 

    def _subregion_grib_files(self, herbie_data, extent, extent_name, product):
        '''
        Takes a list of Herbie objects, and subregions the downloaded grib 
        files. If there is no defined extent, no changes will be made.

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
        subregion_grib_files = []
        for H in herbie_data:
            # subregion grib files
            file = H.get_localFilePath(product)
            idx_file = wgrib2.create_inventory_file(file)
            subset_file = (
                file if extent is None
                else wgrib2.region(file, extent, name=extent_name)
            )
            subregion_grib_files.append(subset_file)

        return subregion_grib_files

    def _grib_to_np(self, grib_files):
        '''
        Converts a list of downloaded grib files into numpy. We assume that
        there is only one data variable in the grib file.

        Arguments:
            grib_files: The paths of the grib files being converted

        Returns:
            A numpy array of the grib files
        '''
        np_of_grib = []
        for file in grib_files:
            hrrr_xarr = xr.open_dataset(file, engine="cfgrib", decode_timedelta=False)

            # for now, we'll only accept Datasets with one variable
            data_vars = [variable for variable in hrrr_xarr.data_vars]
            if len(data_vars) != 1:
                err_msg = (
                    f"xarray Dataset should have only one data variable."
                    f"Expected: 'unknown' for 'COLMD' or 'mdens' for 'MASSDEN',"
                    f"Returned: {hrrr_xarr.data_vars.keys()}"
                )
                raise ValueError(" ".join(err_msg))
            variable = data_vars[0]

            # hrrr data comes in (y, x), so it will need to be flipped for (x, y)
            np_of_grib.append(np.flip(hrrr_xarr[variable].to_numpy(), axis=0))

        return np.array(np_of_grib)

    def _interpolate_and_add_channel_axis(self, subregion_ds, dim):
        ''' 
        Takes a dataset frames (x, y), interpolates/decimates according to
        the dimensions, and adds a channel axis.

        Arguments:
            subregion_ds: The numpy array containing a list of frames
            dim: The desired dimensions

        Returns:
            A numpy array of frames
        '''
        n_frames = len(subregion_ds)
        channels = 1
        frames = np.empty(shape=(n_frames, dim, dim, channels))

        # interpolate and add channel axis
        for i, frame in enumerate(subregion_ds):
            new_frame = cv2.resize(frame, (dim, dim))
            new_frame = np.reshape(new_frame, (dim, dim, channels))
            frames[i] = new_frame

        return frames

    def _sliding_window_of(self, frames, window_size, full_slide=False):
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
