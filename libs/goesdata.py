from goes2go.data import goes_timerange
import numpy as np
import xarray as xr
import pandas as pd
import rioxarray
import cv2
import io
from contextlib import redirect_stdout
from pyproj import Geod
from tqdm import tqdm 

#TODO figure out how to import this util funciton without this
import sys
sys.path.append('/home/mgraca/Workspace/hrrr-smoke-viz/libs')
sys.path.append('/home/mgraca/Workspace/hrrr-smoke-viz')
from pwwb.utils.interpolation import interpolate_frame

class GOESData:
    def __init__(
        self,
        start_date="2025-01-10 00:00",
        end_date="2025-01-10 00:59",
        extent=(-118.75, -117.0, 33.5, 34.5),
        dim=40,
        save_dir=None,
        verbose=False,
    ):
        """
        Pipeline:
            1. Ingest data
            2. Compute mean AOD for the day
            3. Reproject data to Plate Carree
            4. Subregion data to exent
            5. Resize dimensions
            6. Interpolate data gaps
        """
        # define for one-time calculation of reprojected spatial resolution
        res_x, res_y = None, None

        self.data = []
        for date in tqdm(self._realigned_date_range(start_date, end_date)):
            try:
                ds = self._ingest_dataset(
                    start_date=date, 
                    end_date=date + pd.Timedelta(minutes=59, seconds=59), 
                    save_dir=save_dir,
                    verbose=verbose
                )
                ds = self._compute_high_quality_mean_aod(ds)
                ds, res_x, res_y = self._reproject(ds, extent, res_x, res_y)
                gridded_data = self._subregion(ds, extent).data
                gridded_data = cv2.resize(gridded_data, (dim, dim))
                gridded_data = (
                    interpolate_frame(gridded_data, dim, interp_flag=np.nan)
                    if self._data_meets_nonnan_threshold(gridded_data, 0.20)
                    else self._use_prev_frame(self.data, dim)
                )
            except FileNotFoundError:
                # file not found in aws, i.e. satellite outage; use prev frame
                gridded_data = self._use_prev_frame(self.data, dim)
            except Exception as e:
                # generic message, default to empty frame
                tqdm.write(self._unhandled_error_msg(start, end, e))
                gridded_data = np.zeros((dim, dim))

            self.data.append(gridded_data)

    ### NOTE: Methods for ingesting and preprocessing the data

    def _ingest_dataset(self, start_date, end_date, save_dir, verbose):
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
            'return_as': 'xarray',
            'max_cpus': 12,
            'verbose' : False,
            'ignore_missing' : False
        }
        if save_dir is not None:
            default_kwargs['save_dir'] = save_dir

        try:
            if verbose:
                ds = goes_timerange(**default_kwargs)
            else:
                with redirect_stdout(io.StringIO()):
                    ds = goes_timerange(**default_kwargs)
        except FileNotFoundError:
            tqdm.write(self._filenotfound_error_msg(start_date, end_date))
            raise

        return ds

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

    ### NOTE: Error message strings

    def _unhandled_error_msg(self, start, end, e):
        return(
            f"🤷⁉️  "
            f"Unhandled error occurred while ingesting on "
            f"{start.strftime('%m/%d/%Y %H:%M:%S')} to "
            f"{end.strftime('%m/%d/%Y %H:%M:%S')}, imputing data."
            f"\n\tError raised: {e}"
        )

    def _filenotfound_error_msg(self, start, end):
        return(
            f"🛰️ 🪦 "
            f"Outage on {start.strftime('%m/%d/%Y %H:%M:%S')} to "
            f"{end.strftime('%m/%d/%Y %H:%M:%S')}, imputing data."
        )

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

    def _use_prev_frame(self, data, dim):
        return data[-1] if len(data) > 0 else np.zeros((dim, dim))
