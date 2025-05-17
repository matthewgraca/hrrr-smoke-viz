'''
READ THIS if you encounter issues, want to know what to expect from this script.

Install earthaccess: `conda install conda-forge::earthaccess`

1. To avoid typing your credentials every time you run the script, check this out:
- Create `.netrc` file containing your Earthdata credentials: 

https://lb.gesdisc.eosdis.nasa.gov/meditor/notebookviewer/?notebookUrl=https://github.com/nasa/gesdisc-tutorials/blob/main/notebooks/How_to_Generate_Earthdata_Prerequisite_Files.ipynb#create_python_netrc_earthaccess

2. I encountered an error with libsqlite3.so: 
First, check out which libsqlite3 you have: `find /home/mgraca/miniconda3/envs/tf-hrrrenv/lib/libsqlite3*`
-> /home/mgraca/miniconda3/envs/tf-hrrrenv/lib/libsqlite3.so.3.49.1

Script expects "generic" libsqlite3.so, but I had libsqlite3.so.3.49.1, not one with a specific version, so I symlinked the generic to point to the specific one:
`ln -s /home/mgraca/miniconda3/envs/tf-hrrrenv/lib/libsqlite3.so.3.49.1 /home/mgraca/miniconda3/envs/tf-hrrrenv/lib/libsqlite3.so` 

3. Encountered an error: 
`ValueError: unrecognized chunk manager dask - must be one of: []`, which means `dask` isn't installed. Ran: `conda install conda-forge::dask`

What to expect: `download()` creates this `.nc4` file: `MERRA2_100.tavg1_2d_slv_Nx.19800101.nc4`, 
which are the granules. You can also pop open the granules using `xarray` to peek inside.
'''
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
