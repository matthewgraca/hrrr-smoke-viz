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
        # ingest day of data
        ds = goes_timerange(
            start=start_date,
            end=end_date,
            satellite='goes18',
            product="ABI-L2-AODC",
            return_as='xarray',
            max_cpus=12,
            verbose=False,
        )

        '''
        Pipeline:
            1. Convert coordinates from radians to lat/lon
            2. Subregion the data by the extent
            3. Compute the mean over time using high quality AOD
            4. Reproject the data to equirectangular projection
            5. Flip and rotate image
        '''
        self._convert_radians_to_latlon(ds)
        sub_ds = self._subregion_ds(ds, extent)
        self._compute_high_quality_mean_aod(sub_ds)
        regridded_data = self._reproject_gridded_data(sub_ds, extent, dim)
        out = np.rot90((regridded_data.T)) 
        self.data = out

    def _convert_radians_to_latlon(self, ds):
        """
        Converts coordinates from radians to lat/lon, modifying the original
        xarray.
        """
        sat_height = ds['goes_imager_projection'].attrs['perspective_point_height']
        ds.coords['x'] = ds.coords['x'] * sat_height 
        ds.coords['y'] = ds.coords['y'] * sat_height 

        X, Y = np.meshgrid(ds.x, ds.y)
        a = ccrs.PlateCarree().transform_points(ds.FOV.crs, X, Y)
        lons, lats, _ = a[:, :, 0], a[:, :, 1], a[:, :, 2]
        ds.coords["lon"] = (("y", "x"), lons)
        ds.coords["lat"] = (("y", "x"), lats)

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

    def _compute_high_quality_mean_aod(self, sub_ds):
        """Store mean AOD (with quality 2+) into sub_ds as 'AOD_mean'"""
        quality_AOD = sub_ds['AOD'].where(sub_ds['DQF'] >= 1)
        sub_ds['AOD_mean'] = quality_AOD.mean(dim='t', skipna=True)

    def _reproject_gridded_data(self, sub_ds, extent, dim):
        """
        Reproject data to plate carree (equirectangular)
        Remove a touch of padding, otherwise dummy values appear on the grid on the edges
        """
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

        regridder = xe.Regridder(sub_ds['AOD_mean'], target_grid, method="bilinear")
        regridded_ds = regridder(sub_ds['AOD_mean'])
        regridded_data = regridded_ds.values

        return regridded_data
