import rasterio
import numpy as np
import cv2
import os
import pandas as pd
import re
import bisect

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
