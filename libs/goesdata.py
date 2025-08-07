from goes2go.data import goes_timerange
import numpy as np
import xarray as xr
import pandas as pd
import rioxarray
import cv2
import io
import os
from contextlib import redirect_stdout
from pyproj import Geod
from tqdm import tqdm 
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from libs.pwwb.utils.interpolation import interpolate_frame
from libs.pwwb.utils.dataset import sliding_window

class GOESData:
    def __init__(
        self,
        start_date="2025-01-10 00:00",
        end_date="2025-01-10 00:59",
        extent=(-118.75, -117.0, 33.5, 34.5),
        dim=40,
        frames_per_sample=5,
        save_dir=None,      # where nc4 files should be saved to
        cache_path=None,    # location where to save or load cache data
        load_cache=False,   # determines if data should be loaded from cache_dir
        save_cache=True,    # determines if data should be written to cache_dir
        verbose=False,
        pre_downloaded=False,   # set to True if user already dl'd from aws
    ):
        """
        Pipeline:
            1. Ingest data
            2. Compute mean AOD for the day
            3. Reproject data to Plate Carree
            4. Subregion data to exent
            5. Resize dimensions
            6. Interpolate data gaps
            7. Apply sliding window to frames
            8. Save to cache (if enabled)

        On save_dir, save_cache, load_cache, save_cache:
            - save_dir is just where the nc files will be saved, NOT the cache 
            - cache_path is where the numpy data file lives
            - save_cache/load_cache tells us if we should save/read the data
                from cache_path

        On pre_downloaded:
            - We split the download and the reading of the data into two 
                discrete steps. Both of these steps load data into memory;
                but if you've downloaded the data already, you don't need
                the first step.
            - It's worth setting this flag to True if you already have 
                downloaded the data; for reference, just verifying the 
                ingest takes 36 minutes of compute due to disk read/write 
        """
        # validate parameters
        if save_cache or load_cache:
            self._validate_cache_path(save_cache, load_cache, cache_path)

        if load_cache:
            cache_data = self._load_cache(cache_path)
            self.data = cache_data['data'] 
            return

        if not pre_downloaded:
            self._download_dataset(start_date, end_date, save_dir, verbose)

        # define for one-time calculation of reprojected spatial resolution
        res_x, res_y = None, None
        outages, errors = 0, 0
        self.data = []
        prog_bar = tqdm(self._realigned_date_range(start_date, end_date))
        for date in prog_bar:
            try:
                # ingest
                prog_bar.set_description(self._retrieve_data_str(date))
                start, end = date, date + pd.Timedelta(minutes=59, seconds=59)
                ds = self._ingest_dataset(
                    start, end, save_dir, verbose, load=True
                )
                # process dataset -> gridded data
                prog_bar.set_description(self._process_data_str(date))
                ds, res_x, res_y = self._process_ds(ds, extent, res_x, res_y)
                gridded_data = self._ds_to_gridded_data(
                    ds, extent, dim, date, verbose
                )
            except FileNotFoundError:
                # file not found in aws, i.e. satellite outage; use prev frame
                outages += 1
                gridded_data = self._use_prev_frame(self.data, dim, date, verbose)
            except Exception as e:
                # generic message, default to prev frame. usually corrupt data
                # unknown errors should print thru verbose level
                errors += 1
                tqdm.write(self._unhandled_error_msg(date, e))
                gridded_data = self._use_prev_frame(self.data, dim, date, verbose)

            prog_bar.set_description(self._complete_ingest_str(date))
            self.data.append(gridded_data)
        
        if outages > 0: print(f"🛰️ 🪦 {outages} outage(s) imputed.")
        if errors > 0: print(f"🤷⁉️  {errors} error(s) imputed.")

        self.data, _ = sliding_window(
            np.expand_dims(np.array(self.data), -1),
            frames_per_sample
        )

        if cache_path is not None and save_cache:
            self._save_to_cache(
                cache_path, self.data, start_date, end_date, extent
            )

    ### NOTE: Methods for handling the cache

    def _load_cache(self, cache_path):
        """
        Loads cache data.
        """
        print(f"📂 Loading data from {cache_path}...", end=" ")
        cached_data = np.load(cache_path)

        # ensure data can be loaded
        data = cached_data['data']
        start_date = cached_data['start_date']
        end_date = cached_data['end_date']
        extent = cached_data['extent']
        print(f"✅ Completed!")

        return cached_data
    
    def _save_to_cache(self, cache_path, data, start_date, end_date, extent):
        print(f"💾 Saving data to {cache_path}...", end=" ")
        np.savez_compressed(
            cache_path,
            data=self.data,
            start_date=start_date,
            end_date=end_date,
            extent=extent
        )
        print("✅ Complete!")

        return

    def _validate_cache_path(self, save_cache, load_cache, cache_path):
        """
        Raises ValueErrors if the params don't agree
        """
        msg = (
            "In order to load from or save to cache, a cache path must be "
            "provided. "
            "Either set `load_cache` and `save_cache` to False to "
            "prevent cache loading/saving, or provide a valid `cache_path`."
        )
        if cache_path is None:
            raise ValueError(f"Cache path is None. {msg}")
        # we won't make the save directories for you
        if save_cache and not os.path.isdir(os.path.dirname(cache_path)):
            raise ValueError(f"Directory does not exist.")
        # only check if cache exists loading, b/c it will be made when saving
        if load_cache and not os.path.exists(cache_path):
            raise ValueError(f"Cache path does not exist. {msg}")
        return True

    ### NOTE: Methods for ingesting and preprocessing the data

    def _ingest_dataset(
        self, 
        start_date, 
        end_date, 
        save_dir, 
        verbose, 
        load,    # load as dataset or not. toggle off if only downloading.
    ):
        """
        Ingests the GOES data; expects a date range, not one timestamp.
        Also performs a preliminary preprocessing step of converting coordinates
        from radians to meters.

        Raises FileNotFoundError if the aws s3 bucket doesn't have data for 
            the given time range.
        """
        default_kwargs = {
            'start' : start_date,
            'end' : end_date,
            'satellite': 'goes18',
            'product': 'ABI-L2-AODC',
            'return_as': 'xarray' if load else 'filelist',
            'max_cpus': None,
            'verbose' : False,
            'ignore_missing' : False,
        }
        if save_dir is not None:
            default_kwargs['save_dir'] = save_dir

        f = io.StringIO()
        try:
            with redirect_stdout(f):
                ds = goes_timerange(**default_kwargs)

            if verbose: tqdm.write(f.getvalue())
        except FileNotFoundError:
            # aws missing file; reraise and let pipeline handle it
            raise
        except ValueError as e:
            # blows up when opening with xarray, usually time fill value
            if verbose: tqdm.write(self._decode_times_error_msg(
                start_date, end_date, e)
            )

            with redirect_stdout(f):
                default_kwargs['return_as'] = 'filelist' 
                df = goes_timerange(**default_kwargs)
                ds = self._as_xarray(df, **default_kwargs)

            if verbose: tqdm.write(f"🍾  Reread success!")

        return ds

    def _ds_to_gridded_data(self, ds, extent, dim, date, verbose):
        """
        Aliases the pipeline of converting the processed dataset into gridded
            data.
        1. Subregions dataset
        2. Resizes to given dimensions
        3. Interpolation/imputation of bad data
        """
        gridded_data = self._subregion(ds, extent).data
        gridded_data = cv2.resize(gridded_data, (dim, dim))
        gridded_data = (
            interpolate_frame(gridded_data, dim, interp_flag=np.nan)
            if self._data_meets_nonnan_threshold(gridded_data, 0.20)
            else self._use_prev_frame(self.data, dim, date, verbose)
        )
        
        return gridded_data

    def _process_ds(self, ds, extent, res_x, res_y):
        """
        Aliases the pipeline of converting the raw dataset into a dataset 
            that can be directly converted to gridded data
        1. Compute high quality mean AOD across time
        2. Reproject the dataset to Plate Carree (equirectangular)

        Also returns the resolution (x, y) of each pixel. We do this because
            we only actually need to compute it once; its set so that if 
            res_x and res_y are not None, they pass through; else they 
            are computed and returned.
        """
        ds = self._compute_high_quality_mean_aod(ds)
        ds, res_x, res_y = self._reproject(ds, extent, res_x, res_y)
        return ds, res_x, res_y

    # NOTE df->xarray methods courtesy of Brian Blaylock.
    '''
    We copy and paste these functions because sometimes we'll need to load the
        xarray using decode_times=False. Unfortunately, it's not a kwarg.
    '''

    def _as_xarray_MP(self, src, save_dir, i=None, n=None, verbose=True):
        """Open a file as a xarray.Dataset -- a multiprocessing helper."""
        # File destination
        local_copy = Path(save_dir) / src

        if local_copy.is_file():
            if verbose:
                print(
                    f"\r📖💽 Reading ({i:,}/{n:,}) file from LOCAL COPY [{local_copy}].",
                    end=" ",
                )
            with open(local_copy, "rb") as f:
                ds = xr.load_dataset(f, decode_times=False)
        else:
            raise FileNotFoundError(f"Local copy of {local_copy} not found!")

        # Turn some attributes to coordinates so they will be preserved
        # when we concat multiple GOES DataSets together.
        attr2coord = [
            "dataset_name",
            "date_created",
            "time_coverage_start",
            "time_coverage_end",
        ]
        for i in attr2coord:
            if i in ds.attrs:
                ds.coords[i] = ds.attrs.pop(i)

        ds["filename"] = src

        return ds

    def _as_xarray(self, df, **params):
        """Download files in the list to the desired path.

        Use multiprocessing to speed up the download process.

        Parameters
        ----------
        df : pandas.DataFrame
            A list of files in the GOES s3 bucket.
            This DataFrame must have a column of "files"
        params : dict
            Parameters from `goes_*` function.
        """
        params.setdefault("max_cpus", None)
        params.setdefault("verbose", True)
        save_dir = params["save_dir"]
        max_cpus = params["max_cpus"]
        verbose = params["verbose"]

        n = len(df.file)
        if n == 0:
            print("🛸 No data....🌌")
        elif n == 1:
            # If we only have one file, we don't need multiprocessing
            ds = self._as_xarray_MP(df.iloc[0].file, save_dir, 1, 1, verbose)
        else:
            # Use Multiprocessing to read multiple files.
            if max_cpus is None:
                max_cpus = multiprocessing.cpu_count()
            cpus = np.minimum(multiprocessing.cpu_count(), max_cpus)
            cpus = np.minimum(cpus, n)

            inputs = [(src, save_dir, i, n) for i, src in enumerate(df.file, start=1)]

            with multiprocessing.Pool(cpus) as p:
                results = p.starmap(self._as_xarray_MP, inputs)
                p.close()
                p.join()

            # Need some work to concat the datasets
            if df.attrs["product"].startswith("ABI"):
                print("concatenate Datasets", end="")
                ds = xr.concat(results, dim="t")
            else:
                ds = results

        if verbose:
            print(f"\r{'':1000}\r📚 Finished reading [{n}] files into xarray.Dataset.")
        ds.attrs["path"] = df.file.to_list()

        return ds

    def _download_dataset(self, start_date, end_date, save_dir, verbose):
        """
        Strictly responsible for downloading the dataset
        This is different from ingest in that it tracks the ingest itself
            with a progress bar and error messages
        """
        outages = 0
        prog_bar = tqdm(self._realigned_date_range(start_date, end_date))
        for date in prog_bar:
            try:
                prog_bar.set_description(
                    f"Downloading data for {date.strftime('%m/%d/%Y %H:%M')} " 
                )
                ds = self._ingest_dataset(
                    start_date=date, 
                    end_date=date + pd.Timedelta(minutes=59, seconds=59), 
                    save_dir=save_dir,
                    verbose=verbose,
                    load=False
                )
            except FileNotFoundError:
                # file not found in aws; just ignore since this is just ingest 
                if verbose:
                    tqdm.write(
                        self._filenotfound_error_msg(start_date, end_date)
                    )
                outages += 1
            except Exception as e:
                # generic message, just skip by default
                tqdm.write(self._unhandled_error_msg(date, e))
            prog_bar.set_description(
                f"Completed ingest for {date.strftime('%m/%d/%Y %H:%M')} " 
            )

        if outages > 0: print(self._outages_msg(outages))

        return

    def _subregion(self, ds, extent):
        """
        Subregions the data based on lat/lon extent. Assumes that 
        the data has been reprojected with lat/lon.
        """
        lon_bottom, lon_top, lat_bottom, lat_top = extent
        subset = ds.sel(
            x=slice(lon_bottom, lon_top),
            y=slice(lat_top, lat_bottom)  # y is latitude, decreasing
        )

        return subset

    def _compute_high_quality_mean_aod(self, ds):
        """
        Calculates mean AOD, and returns a Dataset with the added mean data
        Expects dataset with time component (e.g. (t, x, y)).

        What is meant by "Quality AOD"?
            • High Quality AOD is most accurate but is missing part of the 
            smoke plume, also big gaps along coastlines 
            (very stringent screening)
            • High + Medium Quality AOD (“top 2 qualities”) fills in 
            most of smoke plume and some of the gaps along coastlines
            • High + Medium + Low Quality AOD (“all qualities”) fully 
            resolves the smoke plume, but at the expense of erroneous high AOD 
            values along coastlines and over inland shallow lakes
            • Bottom Line: Make sure you process AOD using the appropriate 
            data quality flags!
                • Avoid low quality AOD for most situations.
                • Use high + medium (“top 2”) qualities AOD for routine 
                operational applications!
        """
        high, medium, low, no_retrieval = 0, 1, 2, 3
        quality_aod = ds['AOD'].where(ds['DQF'] <= medium)
        temp_ds = ds.assign(
            AOD_mean=quality_aod.mean(dim='t', skipna=True)
        )

        return temp_ds

    ### NOTE: Methods for reprojections

    def _convert_radians_to_meters(self, ds):
        """
        Converts coordinates from radians to meters
        """
        temp_ds = ds.copy(deep=True)
        H = temp_ds['goes_imager_projection'].attrs['perspective_point_height']
        temp_ds = temp_ds.assign_coords({
            'x': temp_ds.x * H,
            'y': temp_ds.y * H
        })

        return temp_ds

    def _calculate_reprojection_resolution(self, ds, extent, x=None, y=None):
        """
        Calculates the resolution in degrees for the reprojection. 
        Assumes coordinates have already been converted from radians to meters.

        We *could* just use the native satellite resolution at nadir 
            (0.02 deg); but it's not *precisely* that. For example, for the 
            LA region, the resolution (x, y) is (0.02169, 0.01806)

        If you pass in your own x and y, it is assumed you've calculated the
            resolutions already, and don't want to recalculate them, and will 
            return x, y as-is.
        """
        # if x and y are defined, passthrough
        if x is not None and y is not None:
            return x, y

        # prepare geodetic conversions between degrees and meters from lat/lon
        geod = Geod(ellps="WGS84")
        lon_bottom, lon_top, lat_bottom, lat_top = extent
        lat_center = (lat_top + lat_bottom) / 2

        # calculate meters per degree, relative to the center latitude
        _, _, m_per_deg_lat = geod.inv(0, lat_center, 0, lat_center + 1)
        _, _, m_per_deg_lon = geod.inv(0, lat_center, 1, lat_center)

        # calculate average grid resolution from lat/lon gaps in meters
        sat_res_x_in_meters = np.diff(ds.x).mean()
        sat_res_y_in_meters = np.diff(ds.y).mean()

        # convert resolution from meters to degrees
        res_y = sat_res_x_in_meters / m_per_deg_lat
        res_x = sat_res_y_in_meters / m_per_deg_lon

        # in total: the resolution of the average pixel, in degrees yipee
        return res_x, res_y

    def _reproject(self, ds, extent, x, y):
        """
        Performs a reprojection on the Dataset to Plate Carree.
        Expects the main variable to be AOD_mean, to avoid reprojection 
            over multiple variables.

        x and y make up the dimensions of the spatial grid, in Plate Carree.
        """
        temp_ds = self._convert_radians_to_meters(ds)
        temp_ds = temp_ds['AOD_mean']
        temp_ds = temp_ds.rio.write_crs(ds.FOV.crs)
        res_x, res_y = self._calculate_reprojection_resolution(
            temp_ds, extent, x, y
        )
        reprojected_ds = temp_ds.rio.reproject(
            dst_crs="EPSG:4326", 
            resolution=(res_x, res_y)
        )

        return reprojected_ds, res_x, res_y

    ### NOTE: Utilities, helper methods

    def _realigned_date_range(self, start_date, end_date):
        """
        The "value" of a given hour is he average AOD of the previous hour
            e.g. AOD @ 1:00 is the average AOD from 0:00-0:59,
            thus the offset.
        """
        offset_start_date = pd.to_datetime(start_date) - pd.Timedelta(hours=1)
        offset_end_date = pd.to_datetime(end_date) - pd.Timedelta(hours=1)
        return pd.date_range(
            offset_start_date,
            offset_end_date,
            freq='h',
            inclusive='left'
        )

    def _data_meets_nonnan_threshold(self, data, threshold=0.0):
        # threshold assumed to be [0.0, 1.0]
        x, y = data.shape
        return np.count_nonzero(~np.isnan(data)) / (x * y) >= threshold

    def _use_prev_frame(self, data, dim, date, verbose):
        if verbose: tqdm.write(self._use_prev_frame_msg(date))
        return data[-1] if len(data) > 0 else np.zeros((dim, dim))

    ### NOTE: Error message strings/strings that are long

    def _unhandled_error_msg(self, date, e):
        return(
            f"🤷⁉️  Unhandled error occurred while ingesting "
            f"on {date.strftime('%m/%d/%Y %H:%M:%S')}.\n"
            f"Error raised: {e}"
        )

    def _filenotfound_error_msg(self, start, end):
        return(
            f"🛰️ 🪦 "
            f"Outage on {start.strftime('%m/%d/%Y %H:%M:%S')} to "
            f"{end.strftime('%m/%d/%Y %H:%M:%S')}, skipping and imputing data."
        )

    def _decode_times_error_msg(self, start_date, end_date, e):
        return(
            f"⌚  Error while processing files on "
            f"{start_date.strftime('%m/%d/%Y %H:%M')}-"
            f"{end_date.strftime('%H:%M')}, likely due to time decoding:\n"
            f"\t{e}\n"
            f"🤔  Attempting to reread with decode_times=False..."
        )

    def _use_prev_frame_msg(self, date):
        return(
            "🚮 Data either doesn't meet threshold for real values or is "
            f"night on {date.strftime('%m/%d/%Y %H:%M')}; "
            "imputing with zeroes/using previous frame."
        )

    def _outages_msg(self, outages):
        return(
            f"🛰️ 🪦 {outages} outage(s) reported; "
            f"skipping ingest on affected dates. "
            f"These dates will be imputed using previous frames."
        )

    def _retrieve_data_str(self, date):
        return f"Retrieving data for {date.strftime('%m/%d/%Y %H:%M')} " 

    def _process_data_str(self, date):
        return f"Processing data for {date.strftime('%m/%d/%Y %H:%M')} " 

    def _complete_ingest_str(self, date):
        return f"Completed data for {date.strftime('%m/%d/%Y %H:%M')} " 
