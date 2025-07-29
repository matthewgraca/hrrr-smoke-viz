from goes2go.data import goes_timerange
import numpy as np
import xarray as xr
import pandas as pd
import rioxarray
import cv2
from pyproj import Geod
from tqdm import tqdm 

class GOESData:
    def __init__(
        self,
        start_date="2025-01-10 00:00",
        end_date="2025-01-10 00:59",
        extent=(-118.75, -117.0, 33.5, 34.5),
        dim=40,
    ):
        """
        Pipeline:
            1. Ingest data
            2. Compute mean AOD for the day
            3. Reproject data to Plate Carree
            4. Subregion data to exent
            5. Resize dimensions
        """
        #TODO: realized that for data on 2022-12-01-00, we need to ingest
        # 2022-11-30-00 to 59; the value AT 12/1 is the previous hour's average!
        # TODO: Investiage -- Something interesting, it seems like during the night, the use the same image as captured during the day. Why? I'd rather do that on my end so I'm not ingesting so much data. Is there a way to check if the data is "real" and not just an imputation?
        # I don't think this is a bug; but I should definitely plot out the first 12 hours of the nc4 files to see what's happening; and if it is just copying, see if I can avoid downloading it.
        # My guess is this data is a bunch of nans, auto generated in some manner to maintain consistency
        self.data = []
        dates = pd.date_range(start_date, end_date, freq='h', inclusive='left')
        for date in tqdm(dates):
            start, end = date, date + pd.Timedelta(minutes=59, seconds=59)
            try:
                ds = self._ingest_dataset(start, end)
                ds = self._compute_high_quality_mean_aod(ds)
                ds = self._reproject(ds, extent)
                gridded_data = self._subregion(ds, extent).data
                self.data.append(cv2.resize(gridded_data, (dim, dim)))
            # TODO: errors impute with empty grid for now; plan to use prev frame/next frame 
            except FileNotFoundError:
                # file not found in aws s3 bucket, i.e. satellite outage
                print(
                    f"🛰️ ❓ "
                    f"Outage on {start.strftime('%m/%d/%Y %H:%M:%S')} to "
                    f"{end.strftime('%m/%d/%Y %H:%M:%S')}, imputing data."
                )
                self.data.append(np.zeros((dim, dim)))
            except Exception as e:
                print(
                    f"🤨 ⁉️  "
                    f"Unhandled error occurred while ingesting on "
                    f"{start.strftime('%m/%d/%Y %H:%M:%S')} to "
                    f"{end.strftime('%m/%d/%Y %H:%M:%S')}, imputing data."
                    f"\n\tError raised: {e}"
                )
                self.data.append(np.zeros((dim, dim)))

    ### NOTE: Methods for ingesting and preprocessing the data

    def _ingest_dataset(self, start_date, end_date):
        """
        Ingests the GOES data; expects a date range, not one timestamp.
        Also performs a preliminary preprocessing step of converting coordinates
        from radians to meters.
        """
        ds = goes_timerange(
            start=start_date,
            end=end_date,
            satellite='goes18',
            product="ABI-L2-AODC",
            return_as='xarray',
            max_cpus=12,
            verbose=False,
            ignore_missing=False
        )

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
        """
        temp_ds = ds.assign(
            AOD_mean=ds['AOD'].where(ds['DQF'] > 0).mean(dim='t', skipna=True)
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

    def _calculate_reprojection_resolution(self, ds, extent):
        """
        Calculates the resolution in degrees for the reprojection. 
        Assumes coordinates have already been converted from radians to meters.

        We *could* just use the native satellite resolution at nadir 
            (0.02 deg); but it's not *precisely* that. For example, for the 
            LA region, the resolution (x, y) is (0.02169, 0.01806)
        """
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

    def _reproject(self, ds, extent):
        """
        Performs a reprojection on the Dataset to Plate Carree.
        Expects the main variable to be AOD_mean, to avoid reprojection 
            over multiple variables.
        """
        temp_ds = self._convert_radians_to_meters(ds)
        temp_ds = temp_ds['AOD_mean']
        temp_ds = temp_ds.rio.write_crs(ds.FOV.crs)
        x_deg, y_deg = self._calculate_reprojection_resolution(temp_ds, extent)
        reprojected_ds = temp_ds.rio.reproject(
            dst_crs="EPSG:4326", 
            resolution=(x_deg, y_deg)
        )

        return reprojected_ds
