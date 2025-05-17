import xarray as xr
import pandas as pd

# open files into one dataset
files = [
    'MERRA2_400.tavg1_2d_flx_Nx.20241201.nc4',
    'MERRA2_400.tavg1_2d_flx_Nx.20241202.nc4'
]
ds = xr.open_mfdataset(files)
#print(ds)

# select relevant variables
variables = [
    'PBLH', # planetary boundary layer height (m)
    'TLML', # surface air temperature (K)
    'CDH'   # surface exchange coefficient for heat (kg m^(-2) s^(-1))
]
print(ds[variables])

# print variables. uncomment to see data go brrr
'''
for time in ds.time:
    pblh = ds['PBLH'].sel(time=time).values
    tlml = ds['TLML'].sel(time=time).values
    cdh = ds['CDH'].sel(time=time).values
    print(
        f"🕛 Time: {time.values}\n"
        f"📏 PBLH: {pblh[0][0]} ... {pblh[-1][-1]}\n"
        f"🌡️  TLML: {tlml[0][0]} ... {tlml[-1][-1]}\n"
        f"🥵  CDH: {cdh[0][0]} ... {cdh[-1][-1]}\n"
    )
'''

# plotting action
import matplotlib.pyplot as plt
def harry_plottah(ds, variables, t_1, t_2):
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    for i, var_name in enumerate(variables):
        ds[var_name].isel(time=t_1).plot(ax=axes[i][0])
        ds[var_name].isel(time=t_2).plot(ax=axes[i][1])
    fig.tight_layout()
    plt.show()

harry_plottah(ds=ds, variables=variables, t_1=0, t_2=47)

'''
ok now to subregion and interpolate. gulp...
- for some reason, the bounding_box in search_data() didn't subregion the data.
- why not?
- if i'm reading this correctly, it's used to include files in an area of interest
meaning the point isn't to subregion a file; it's to exclude files that don't
support that region?
- yep. https://github.com/nsidc/earthaccess/issues/515
'''
# subregion and plot
# use .compute() to force compute the dask chunks. xarray.plot() does it 
# implicitly, along with some other functions
min_lon, max_lon, min_lat, max_lat = (-118.75, -117.5, 33.5, 34.5)
la_ds = ds.sel(
    lat=slice(min_lat, max_lat),
    lon=slice(min_lon, max_lon)
).compute()
print(la_ds)
harry_plottah(la_ds, variables, 0, 47)

'''
- 9 whole pixels is hilarious. the spatial resolution of MERRA-2 is 50km,
so each pixel is a 250 km area, or 22500 km area for 9 pixels, which is
expected for the LA region which is apparently ~12.5k km
'''

# interpolate
import numpy as np
from scipy.ndimage import zoom

dim = 200
merra2_data = np.zeros((len(ds.time), dim, dim, 3))
for t, time in enumerate(ds.time):
    # create subregions
    subregion_ds = ds.sel(
        time=time,
        lat=slice(min_lat, max_lat), 
        lon=slice(min_lon, max_lon)
    ).compute()

    tlml_np = subregion_ds['TLML'].to_numpy()
    cdh_np = subregion_ds['CDH'].to_numpy()
    pblh_np = subregion_ds['PBLH'].to_numpy()

    # interpolate
    zoom_y = dim / pblh_np.shape[0] 
    zoom_x = dim / pblh_np.shape[1] 

    pblh_grid = zoom(pblh_np, (zoom_y, zoom_x), order=1, mode='nearest')
    tlml_grid = zoom(tlml_np, (zoom_y, zoom_x), order=1, mode='nearest')
    cdh_grid = zoom(cdh_np, (zoom_y, zoom_x), order=1, mode='nearest')

    # store
    merra2_data[t, :, :, 0] = pblh_grid
    merra2_data[t, :, :, 1] = tlml_grid
    merra2_data[t, :, :, 2] = cdh_grid

# check results
print(merra2_data.shape)
rand_idx = np.random.choice(merra2_data.shape[0])
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
plt.suptitle(f"Picking from sample {rand_idx}")
axes[0].set_title("pblh")
axes[0].imshow(merra2_data[rand_idx,:,:,0])
axes[1].set_title("tlml")
axes[1].imshow(merra2_data[rand_idx,:,:,1])
axes[2].set_title("cdh")
axes[2].imshow(merra2_data[rand_idx,:,:,2])
fig.tight_layout()
plt.show()

