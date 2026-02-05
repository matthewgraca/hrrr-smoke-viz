import rasterio
import numpy as np
import cv2
import os
import pandas as pd
import re
import bisect
from harmony import Client, Environment, Collection, Request, BBox
from osgeo import gdal

class NDVIData:
    def __init__(
        self,
        start_date,
        end_date,
        extent=(-118.615, -117.70, 33.60, 34.35),
        dim=84,
        raw_dir='data', # saved to data/modis-ndvi
        save_dir='data', # saved to data/ndvi_processed.npz
        verbose=0 # 0 = all, 1 = progress + errors, 2 = only errors
    ):
        self.start_date_dt = pd.to_datetime(start_date)
        self.end_date_dt = pd.to_datetime(end_date)
        self.extent = extent
        self.dim = dim
        self.raw_dir = self._validate_raw_dir(raw_dir)
        self.save_dir = self._validate_save_dir(save_dir)
        self.VERBOSE = self._validate_verbose(verbose)

        lon_min, lon_max, lat_min, lat_max = self.extent
        '''
        harmony_client = Client() # pulls from .netrc file
        request = Request(
            # MODIS/Terra Vegetation Indices 16-Day L3 Global 1km SIN Grid V061
            collection=Collection(id='C2565788905-LPCLOUD'),
            spatial=BBox(w=lon_min, e=lon_max, s=lat_min, n=lat_max),
            temporal={
                'start' : self.start_date_dt,
                'stop' : self.end_date_dt
            }
        )
        job_id = harmony_client.submit(request)
        print(harmony_client.status(job_id))
        results = harmony_client.download_all(job_id, directory=raw_dir, overwrite=True)
        for r in results:
            print(r)
        '''
        
        gdal.UseExceptions()
        hdf_files = sorted(os.listdir(self.raw_dir))

        # crack open the dataset and search through the subdatasets
        hdf_path = os.path.join(self.raw_dir, hdf_files[0])
        ds = gdal.Open(hdf_path)
        if ds is None:
            raise RuntimeError(f'Unable to open {hdf_path}.')
        sub_ds = ds.GetSubDatasets()
        if not sub_ds:
            raise RuntimeError(f'No subdatasets found.')

        # grab and crack open the ndvi subdataset
        keyword = 'NDVI'
        matches = [
            (name, desc)
            for name, desc in sub_ds
            if keyword in name or keyword in desc
        ]
        if not matches:
            raise RuntimeError(f'No matches for {keyword} found in subdatasets')
        sds_name = matches[0][0]

        ndvi_sds = gdal.Open(sds_name)
        if ndvi_sds is None:
            raise RuntimeError(f'Unable to open {keyword} subdataset: {sds_name}')

        # warp
        out = os.path.join(save_dir, 'out_la.tif')
        warp_options = gdal.WarpOptions(
            dstSRS='EPSG:4326',
            resampleAlg='bilinear',
            format='GTiff',
            outputBounds=(lon_min, lat_min, lon_max, lat_max),
            multithread=True
        )

        result = gdal.Warp(destNameOrDestDS=out, srcDSOrSrcDSTab=ndvi_sds, options=warp_options)
        if result is None:
            raise RuntimeError('gdal.Warp failed.')

        # clean up
        result.FlushCache()
        result = None
        ndvi_sds = None
        ds = None

        return

    
    ### NOTE: Parameter validation methods
    def _validate_raw_dir(self, raw_dir):
        ''' Validates raw_dir existence and creates modis-ndvi under it '''
        if not os.path.exists(raw_dir):
            raise ValueError(
                f'{raw_dir} does not exist. ' +
                'Please provide an existing folder to store the raw files. ' +
                'It will be stored under \'raw_dir/modis-ndvi\'.'
            )

        new_dir = os.path.join(raw_dir, 'modis-ndvi')
        os.makedirs(new_dir, exist_ok=True)

        return new_dir


    def _validate_save_dir(self, save_dir):
        if not os.path.exists(save_dir):
            raise ValueError(
                'Please provide a folder to store the ndvi_processed.npz file.'
            )
        return save_dir

    def _validate_verbose(self, verbose):
        v = verbose if verbose in [0, 1, 2] else None
        if v is None:
            raise ValueError('Verbose must be 0, 1, 2.')
        return v


'''
params = {
    'cache_path' : '/mnt/wildfire/raw-data/modis-ndvi',
    'dim' : 84,
    'start_date' : '2023-08-02',
    'end_date' : '2025-08-02',
    'save_path': '/mnt/wildfire/processed-data/2026-01-27'
}

print(f'These are the parameters. Press ENTER if these look good, else Ctrl+c out of here and change them in the script.')
print(params)
input()

NDVI_SCALE_FACTOR = 0.0001 # ndvi is supposed to be [-1, 1], but NASA wants to use int for better storage, so we fix it.

dates = dict.fromkeys(pd.date_range(params['start_date'], params['end_date'], freq='h', inclusive='left'))

# match the year + julian day to the processed frame
pattern = r'doy(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})'
doy_to_frame = {}
for file in sorted(os.listdir(params['cache_path'])):
    match = re.search(pattern, file)
    if match:
        year, doy, hour, minute, second = match.groups()
        file_date = year + doy

    with rasterio.open(os.path.join(params['cache_path'], file), 'r') as src:
        ndvi = src.read()
        ndvi = np.squeeze(ndvi)
        ndvi = cv2.resize(ndvi, (params['dim'], params['dim']))
        ndvi = ndvi * NDVI_SCALE_FACTOR

    doy_to_frame[file_date] = ndvi.copy()

# match the date's year + julian day to the processed frame
file_ydoy = list(doy_to_frame.keys())
i = 0
for date in dates.keys():
    y_doy = date.strftime('%Y') + date.strftime('%j')
    if i + 1 == len(file_ydoy):
        dates[date] = file_ydoy[-1]
    else:
        dates[date] = file_ydoy[i] 
        if y_doy < file_ydoy[i] or y_doy >= file_ydoy[i+1]:
            i += 1

res = [doy_to_frame[ydoy] for ydoy in dates.values()]
res = np.stack(res, axis=0)
print(res.shape)

save_path = os.path.join(params['save_path'], 'ndvi_processed.npy')
print(f'Saved to {save_path}')
np.save(save_path, res)
'''
