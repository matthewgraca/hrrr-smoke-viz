from datetime import datetime
import numpy as np
import cartopy.crs as ccrs
from goes2go.data import goes_timerange
import xesmf as xe
import xarray as xr
import matplotlib.pyplot as plt

class GOESData:
    def __init__(
        self,
        start_date="2025-01-10 00:00",
        end_date="2025-01-10 00:59",
        extent=(-118.75, -117.0, 33.5, 34.5),
        dim=40,
    ):
        ds = self._ingest_dataset(start_date, end_date)
        ds = self._preprocess_dataset_before_gridding_data(ds, extent)
        self.data = self._ds_to_gridded_data(ds, extent, dim)

    ### NOTE: Methods for ingesting the data (as an xarray Dataset)

    def _ingest_dataset(self, start_date, end_date):
        """
        Ingests the GOES data; expects a date range, not one timestamp.
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

    ### NOTE: Methods for preprocessing the Dataset

    def _preprocess_dataset_before_gridding_data(self, ds, extent):
        """
        Pipeline to preprocess the dataset before converting it gridded data.
        1. Convert coordinates from radians to lat/lon
        2. Subregion the data by the extent
        3. Compute the mean over time using high quality AOD
        """
        processed_ds = self._convert_radians_to_latlon(ds)
        processed_ds = self._subregion_ds(processed_ds, extent)
        processed_ds = self._compute_high_quality_mean_aod(processed_ds)

        return processed_ds

    def _convert_radians_to_latlon(self, ds):
        """
        Converts coordinates from radians to lat/lon
        """
        temp_ds = ds.copy(deep=True)
        sat_height = temp_ds['goes_imager_projection'].attrs['perspective_point_height']
        temp_ds.coords['x'] = temp_ds.coords['x'] * sat_height 
        temp_ds.coords['y'] = temp_ds.coords['y'] * sat_height 

        X, Y = np.meshgrid(temp_ds.x, temp_ds.y)
        a = ccrs.PlateCarree().transform_points(temp_ds.FOV.crs, X, Y)
        lons, lats, _ = a[:, :, 0], a[:, :, 1], a[:, :, 2]
        temp_ds.coords["lon"] = (("y", "x"), lons)
        temp_ds.coords["lat"] = (("y", "x"), lats)

        return temp_ds

    def _subregion_ds(self, ds, extent):
        lon_bottom, lon_top, lat_bottom, lat_top = extent
        sub_ds = ds.where(
            (ds.coords['lon'] >= lon_bottom) &
            (ds.coords['lon'] <= lon_top) &
            (ds.coords['lat'] >= lat_bottom) &
            (ds.coords['lat'] <= lat_top),
            drop=True
        )

        return sub_ds

    def _compute_high_quality_mean_aod(self, ds):
        """
        Store mean AOD (with quality 2+) into sub_ds as 'AOD_mean'.
        Expects dataset with time component (e.g. (t, x, y)).
        """
        quality_AOD_ds = ds.where(ds['DQF'] >= 1)
        quality_AOD_ds['AOD_mean'] = quality_AOD_ds['AOD'].mean(dim='t', skipna=True)

        return quality_AOD_ds 

    ### NOTE: Methods for reprojections

    def _reproject_gridded_data(self, ds, extent, dim):
        """
        Reproject data to plate carree (equirectangular).
        Expects a Dataset with 'AOD_mean' attribute
        """
        # Remove a touch of padding, otherwise dummy values appear the edges
        epsilon = 0.05
        lon_bottom, lon_top, lat_bottom, lat_top = extent
        target_lon = np.linspace(lon_bottom + epsilon, lon_top - epsilon, dim)
        target_lat = np.linspace(lat_bottom + epsilon, lat_top - epsilon, dim)

        target_grid = xr.Dataset(
            {
                "lon": (["lon"], target_lon),
                "lat": (["lat"], target_lat),
            }
        )

        regridder = xe.Regridder(ds['AOD_mean'], target_grid, method="bilinear")
        regridded_ds = regridder(ds['AOD_mean'])
        regridded_data = regridded_ds.values

        return regridded_data
    
    def _ds_to_gridded_data(self, ds, extent, dim):
        """
        Grabs AOD_mean data from Dataset and converts it to a grid in numpy.
        """
        regridded_data = self._reproject_gridded_data(ds, extent, dim)
        reoriented_data = np.rot90((regridded_data.transpose())) 

        return reoriented_data
