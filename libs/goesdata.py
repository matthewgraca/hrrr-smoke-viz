from datetime import datetime
import numpy as np
import cartopy.crs as ccrs
from goes2go.data import goes_timerange
import xesmf as xe
import xarray as xr
import matplotlib.pyplot as plt

# args
lat_bottom, lat_top = 33.5, 34.5
lon_bottom, lon_top = -118.75, -117.0
dim = 40
start_date = "2025-01-10 00:00"
end_date = "2025-01-10 00:59"

def get_latlon(ds):
    """Get lat/lon of all points. Modifies original xarray to contain lat/lon"""
    X, Y = np.meshgrid(ds.x, ds.y)
    a = ccrs.PlateCarree().transform_points(ds.FOV.crs, X, Y)
    lons, lats, _ = a[:, :, 0], a[:, :, 1], a[:, :, 2]

    ds.coords["lon"] = (("y", "x"), lons)
    ds.coords["lat"] = (("y", "x"), lats)
    return ds["lat"], ds["lon"]

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

# convert coords -> lat/lon
h = ds['goes_imager_projection'].attrs['perspective_point_height']
ds.coords['x'] = ds.coords['x'] * h
ds.coords['y'] = ds.coords['y'] * h
get_latlon(ds)

# subregion data
sub_ds = ds.where(
    (ds.coords['lon'] >= lon_bottom) &
    (ds.coords['lon'] <= lon_top) &
    (ds.coords['lat'] >= lat_bottom) &
    (ds.coords['lat'] <= lat_top),
    drop=True
)

# compute mean AOD with quality 2+ data over time
sub_ds['AOD_mean'] = sub_ds['AOD'].where(sub_ds['DQF'] >= 1).mean(dim='t', skipna=True)

# reproject data to plate carree (equirectangular)
# remove a touch of padding, otherwise dummy values appear on the grid on the edges
epsilon = 0.05
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

out = np.rot90((regridded_data.T)) 
