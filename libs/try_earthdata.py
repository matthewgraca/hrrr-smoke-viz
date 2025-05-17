import earthaccess
import xarray as xr

# This will work if Earthdata prerequisite files have already been generated
auth = earthaccess.login()

# To download multiple files, change the second temporal parameter
min_lon, max_lon, min_lat, max_lat = (-118.75, -117.5, 33.5, 34.5)
results = earthaccess.search_data(
    short_name="M2T1NXFLX",
    version='5.12.4',
    temporal=('2024-12-01', '2024-12-02'), 
    #bounding_box=(-180, 0, 180, 90)
    #(lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat)
    bounding_box=(min_lon, min_lat, max_lon, max_lat)
)
print(results)

# Download granules to local path
downloaded_files = earthaccess.download(
    results,
    local_path='.', # Change this string to download to a different path
)
print(downloaded_files)

# OPTIONAL: Open granules using Xarray
ds = xr.open_mfdataset(downloaded_files)
print(ds)

# checking variables
res = [i for i in ds.data_vars]
print(res)

variables = ["PBLH", "TLML", "CDH"]
print(f"checking variables {variables}")
print(ds[variables])
