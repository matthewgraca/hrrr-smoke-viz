from datetime import datetime
import numpy as np
import cartopy.crs as ccrs
from goes2go.data import goes_timerange
import xarray as xr
import matplotlib.pyplot as plt
import rioxarray
import cv2

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
        ds = self._ingest_dataset(start_date, end_date)
        ds = self._compute_high_quality_mean_aod(ds)
        ds = self._reproject(ds)
        gridded_data = self._subregion(ds, extent).data
        self.data = cv2.resize(gridded_data, (dim, dim))

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

    def _calculate_reprojection_resolution(self, ds):
        """
        Calculates the resolution for the reprojection. 
        Assumes coordinates have already been converted from radians to meters.
        """
        # meters per pixel
        dx_m = np.diff(ds.x).mean()
        dy_m = np.diff(ds.y).mean()

        # 1 degree / (meridonial or equatorial circumference / 360 degrees) = degree per meter lon or lat
        # not 1000% precise, but it's sufficient
        # you could also just use goes' resolution of 0.02; it's not 1-1, but it's pretty close (0.018)
        deg_per_meter_lon = 1 / 111320
        deg_per_meter_lat = 1 / 110574

        # degrees of each pixel
        dx_deg = dx_m * deg_per_meter_lon
        dy_deg = dy_m * deg_per_meter_lat

        return dx_deg, dy_deg

    def _reproject(self, ds):
        """
        Performs a reprojection on the Dataset to Plate Carree.
        Expects the main variable to be AOD_mean, to avoid reprojection 
            over multiple variables.
        """
        temp_ds = self._convert_radians_to_meters(ds)
        temp_ds = temp_ds['AOD_mean']
        temp_ds = temp_ds.rio.write_crs(ds.FOV.crs)
        dx_deg, dy_deg = self._calculate_reprojection_resolution(temp_ds)
        reprojected_ds = temp_ds.rio.reproject(
            dst_crs="EPSG:4326", 
            resolution=(dx_deg, dy_deg)
        )

        return reprojected_ds
