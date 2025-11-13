import s3fs
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import re
import pandas as pd
import os
import bisect
from tqdm import tqdm

class NAQFCData:
    def __init__(
        self,
        start_date="2025-01-10 00:00",
        end_date="2025-01-17 00:59",
        extent=(-118.75, -117.0, 33.5, 34.5),
        dim=40,
        product='pm25',         # 'pm25' = pm2.5, 'o3' = ozone, 'dust', 'smoke'
        local_path=None,        # where grib files should be saved to/live in
        save_path=None,         # where the final numpy file should be saved to
        load_numpy=False,       # specifies the numpy file should be loaded from cache
        verbose=0,              # 0 = all msgs, 1 = prog bar + errors, 2 = only errors
    ): 
        '''
        Pipeline:
            1. Find all files within start/end date
            2. Download
            3. Process
                - Reproject, subregion, resize
        '''
        models = {
            'pm25' : 'aqm',
            'o3' : 'aqm',
            'dust' : 'dust',
            'smoke' : 'smoke'
        }

        model_init_times = {
            'aqm' : ['06', '12'],
            'dust' : ['06', '12'],
            'smoke' : ['03']
        }

        # members and validation
        self.VERBOSE = verbose if verbose in {0, 1, 2} else 0
        self._s3 = s3fs.S3FileSystem(anon=True)
        self.product = self._validate_product(product, models)
        self.start_date, self.end_date = self._validate_dates(
            start_date, end_date
        )
        self.extent = self._validate_extent(extent)
        self.local_path = self._validate_save_path(local_path, self.product)

        start_dt = pd.to_datetime(start_date, utc=True)
        end_dt = pd.to_datetime(end_date, utc=True)
        dates = pd.date_range(
            start_date, end_date, freq='h', inclusive='left', tz='UTC'
        )

        # TODO load from cache

        # find all files 
        sorted_paths = self._get_file_paths(
            self._s3,
            models,
            model_init_times,
            self.product,
            start_dt,
            end_dt
        )
        #print('\n'.join(sorted_paths))

        # download
        self._download(s3=self._s3, sources=sorted_paths, destination=self.local_path)

        # TODO process

        # get local files
        sorted_local_files = [
            os.path.join(self.local_path, os.path.basename(path))
            for path in sorted_paths
        ]
        # crack the file open
        # read 6 hours for 06, 18 hours for 12
        # for each hour, process (reproject, resize)

        '''
        # does not like with open() for some reason, some io buffer subscripting error
        ds = xr.load_dataset('/home/mgraca/Workspace/hrrr-smoke-viz/tests/naqfcdata/data/aqm.t06z.ave_1hr_pm25_bc.20240514.227.grib2', engine='cfgrib')
        '''

        # TODO save to cache

    ### NOTE: Validation helpers

    def _validate_product(self, product, models):
        valid_products = set(models.keys())
        if product not in valid_products:
            raise ValueError('Product must be in {valid_products}')
        return product 

    def _validate_dates(self, start_date, end_date):
        try:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
        except:
            raise ValueError(
                f'Unable to read {start_date} or {end_date} as a '
                f'datetime object.'
            )
        else:
            return start_date, end_date

    def _validate_extent(self, extent):
        try:
            a = len(extent)
            a == 4
        except:
            raise ValueError('Extent must be a tuple of four.')
        
        try:
            lon_min, lon_max, lat_min, lat_max = extent
            if (lon_min >= lon_max):
                raise ValueError('Longitude minimum should be less than longitude maximum')
            elif (lon_min < -180 or lon_max > 180):
                raise ValueError('Longitudes are out of bounds.')
            elif (lat_min < -90 or lat_max > 90):
                raise ValueError('Latitudes are out of bounds.')
            else:
                pass
        except:
            raise ValueError('Longitude and/or latitude values are invalid.')

        return extent

    def _validate_save_path(self, save_path, product):
        '''
        Ensure it's a valid directory that exists. Force the user to define
        one so that there are no surprises on where the data ends up. We want
        to create as few files and directories under the hood as possible.
        '''
        if not os.path.isdir(save_path):
            raise ValueError(
                f'Invalid save directory. '
                f'Either correct it or create {save_path}.'
            )
        
        final_save_path = os.path.join(
            save_path, 
            f'noaa-nws-naqfc-pds-{product}'
        )
        os.makedirs(name=final_save_path, exist_ok=True)

        return final_save_path

    ### NOTE: Querying and Downloading helpers

    def _find_model_directories(self, s3, model='aqm'):
        '''
        Done dynamically instead of statically to survive any updates
        to new AQM versions
        '''
        model_patterns = {
            # match to AQMv 0 to 99
            'aqm' : 'AQMv([0-9]|([0-9][0-9]))',
            'dust' : 'HYSPLIT_Dust',
            'smoke' : 'RAP_Smoke',
        }
        valid_models = set(model_patterns.keys())
        if model not in valid_models:
            raise ValueError(f'Invalid model. Pick from {valid_models}')

        text = s3.ls('noaa-nws-naqfc-pds', refresh=True)
        pattern = fr'noaa-nws-naqfc-pds\/{model_patterns[model]}\b'

        matches = [re.search(pattern, dirname) for dirname in text]
        return [match.group() for match in matches if match]

    def _get_file_paths(
        self,
        s3,
        models,
        model_init_times,
        product,
        start_dt,
        end_dt
    ):
        '''
        Grabs the file paths of the files we want from the bucket.
        Pipeline:
            1. Gets the models of interest (e.g. for PM2.5, it's AQMv5-v7)
            2. Picks only continential US for simplicity
            3. Finds the correct dates in the bucket
            4. Walks through the model initialization times
            5. Grabs all the relevant files
                - pm25: bias corrected, 1-hour averages
                - o3: bias corrected, 1-hour averages
                - dust: not implemented (can pick b/t surface and column)
                - smoke: not implemented (can pick b/t surface and column)

        The neat thing about this is it returns the paths sorted, not just
        sorted by file name but by the actual paths too!
        '''
        # backfilling logic DO NOT PEER INTO THE ABYSS
        model_backfill_hours = {
            'aqm' : (6, 12),
            'dust' : (6, 12),
            'smoke' : (3) 
        }

        start_hr, end_hr = model_backfill_hours[models[product]]

        start_dt_hr = int(start_dt.strftime('%H'))
        backfill_start_steps = 0
        if start_dt_hr < start_hr:
            backfill_start_steps = 1
        elif start_hr < start_dt_hr <= end_hr:
            backfill_start_steps = 0
        else:
            backfill_start_steps = 1

        start_dt = (
            start_dt - pd.Timedelta(days=1)
            if start_dt_hr < start_hr
            else start_dt
        )

        end_dt_hr = int(end_dt.strftime('%H'))
        backfill_end_steps = 0
        if end_dt_hr < start_hr:
            backfill_end_steps = 0
        elif start_hr < end_dt_hr <= end_hr: 
            backfill_end_steps = -1
        else:
            backfill_end_steps = 0

        end_dt = (
            end_dt - pd.Timedelta(days=1)
            if end_dt_hr < start_hr
            else end_dt
        )

        filename_form = {
            'pm25' : 'ave_1hr_pm25_bc',
            'o3' : 'ave_1hr_o3_bc',
            # saving dev time here to get core pm2.5 product running
            'dust' : NotImplementedError('Dust product not fully supported yet.'),
            'smoke' : NotImplementedError('Smoke product not fully supported yet.')
        }

        # bucket/model/CS
        paths = [
            # only supports conus for simplicity
            s3.ls(path + '/CS')
            for path in self._find_model_directories(s3, model=models[product])
        ]

        # bucket/model/CS/dates
        flattened_paths = [item for sublist in paths for item in sublist]
        model_dates = [
            os.path.basename(os.path.normpath(model_date))
            for model_date in flattened_paths
        ]

        start_idx = bisect.bisect_left(model_dates, start_dt.strftime('%Y%m%d'))

        end_idx = bisect.bisect_right(model_dates, end_dt.strftime('%Y%m%d'))

        paths = flattened_paths[start_idx : end_idx]

        # bucket/model/CS/dates/init_hour
        paths = [
            f'{path}/' + init_hr
            for path in paths 
            for init_hr in model_init_times[models[product]]
        ]

        # bucket/model/CS/dates/init_hour/file.grib2
        paths = [
            f
            for path in paths
            for f in s3.ls(path)
            if filename_form[product] in os.path.basename(f)
        ]

        return paths[backfill_start_steps : len(paths) + backfill_end_steps]

    def _download(self, s3, sources, destination):
        if self.VERBOSE < 2:
            print('Downloading GRIB files from the NAQFC bucket...')

        for src in (tqdm(sources) if self.VERBOSE < 2 else sources):
            file = os.path.basename(src)
            dst = os.path.join(destination, file)
            if os.path.exists(dst):
                if self.VERBOSE < 1:
                    tqdm.write(
                        f'Local copy of {file} found, skipping download.'
                    )
            else:
                s3.get_file(src, dst)

        return
