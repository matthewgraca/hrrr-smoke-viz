import s3fs
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import re
import pandas as pd
import os
import bisect
from tqdm import tqdm
import numpy as np
import rioxarray
from pyproj import CRS, Transformer
import cartopy.crs as ccrs
import cv2

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
        self.VERBOSE = verbose if verbose in {0, 1, 2} else 0

        # load from cache
        if load_numpy:
            if save_path is None:
                raise ValueError(
                    'Provide a save path to pull the numpy file from.'
                )
            cache_data = self._load_numpy_cache(save_path)
            self.data = cache_data['data'] 
            self.start_date = cache_data['start_date']
            self.end_date = cache_data['end_date']
            self.extent = cache_data['extent']
            self.product = cache_data['product']
            return

        # mapping of the product to its model name in the bucket
        models = {
            'pm25' : 'aqm',
            'o3' : 'aqm',
            'dust' : 'dust',
            'smoke' : 'smoke'
        }

        # initialization times of the models, in UTC
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
        self.data = None
        local_path = self._validate_local_path(local_path, self.product)
        save_path = self._validate_save_path(save_path)

        # get the datetimes
        start_dt = pd.to_datetime(start_date, utc=True)
        end_dt = pd.to_datetime(end_date, utc=True)
        dates = pd.date_range(
            start_date, end_date, freq='h', inclusive='left', tz='UTC'
        )

        # find all files in the bucket
        sorted_remote_files = self._get_file_paths(
            self._s3, models, model_init_times, self.product, start_dt, end_dt
        )

        # download
        self._download(
            s3=self._s3, sources=sorted_remote_files, destination=local_path
        )

        # gather local files
        sorted_local_files = [
            os.path.join(local_path, os.path.basename(path))
            for path in sorted_remote_files
        ]

        # process the batches 
        date_to_data = self._process_files(
            sorted_local_files, dates, self.extent, dim, self.product
        )

        # peel frames from dates and save it
        self.data = np.array([
            date_to_data[date] for date in sorted(date_to_data.keys())
        ])

        self._save_numpy_to_cache(
            cache_path=save_path,
            data=self.data,
            start_date=self.start_date,
            end_date=self.end_date,
            extent=self.extent,
            product=self.product
        )

        return

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

    def _validate_local_path(self, save_path, product):
        '''
        Ensure it's a valid directory that exists. Force the user to define
        one so that there are no surprises on where the data ends up. We want
        to create as few files and directories under the hood as possible.
        '''
        if save_path is None:
            raise ValueError('Provide a save path for the grib data.')
        if not os.path.exists(save_path):
            raise ValueError(
                f'Invalid save directory. '
                f'Either correct it or create {save_path}.'
            )
        
        final_save_path = os.path.join(
            save_path, 
            f'noaa-nws-naqfc-pds-{product}'
        )
        os.makedirs(name=final_save_path, exist_ok=True)

        if self.VERBOSE == 0:
            print(f'➡️  GRIB files will be downloaded to {final_save_path}')

        return final_save_path

    def _validate_save_path(self, save_path):
        '''
        Ensure it's a valid directory that exists. Force the user to define
        one so that there are no surprises on where the data ends up. We want
        to create as few files and directories under the hood as possible.
        '''
        if save_path is None:
            raise ValueError('Provide a save path.')

        dirpath = os.path.dirname(save_path)
        if not os.path.exists(dirpath):
            raise ValueError(
                f'Invalid path to file. '
                f'Either correct it or create {dirpath}.'
            )
        return save_path

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
        start_dt, fwd_steps, end_dt, back_steps = self._backfill_order(
            models, product, start_dt, end_dt
        )

        filename_form = {
            'pm25' : 'ave_1hr_pm25_bc',
            'o3' : 'ave_1hr_o3_bc',
            # saving dev time here to get core pm2.5 product running
            'dust' : NotImplementedError('Dust product not fully supported yet.'),
            'smoke' : NotImplementedError('Smoke product not fully supported yet.')
        }

        if self.VERBOSE == 0:
            print('🔎 Searching for files in the bucket...')

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
            for path in (tqdm(paths) if self.VERBOSE < 2 else paths)
            for f in s3.ls(path)
            if filename_form[product] in os.path.basename(f)
        ]

        return paths[fwd_steps : len(paths) + back_steps]

    def _download(self, s3, sources, destination, size=1e+7):
        '''
        Downloads the data.

        Simple integrity checks to trigger redownload:
            - Is the downloaded file more than 10MB large?

        Currently just throws an error if it fails after redownload.
        '''
        if self.VERBOSE < 2:
            print('🪣 Downloading files from the NAQFC bucket...')

        for src in (tqdm(sources) if self.VERBOSE < 2 else sources):
            file = os.path.basename(src)
            dst = os.path.join(destination, file)
            if os.path.exists(dst):
                if self.VERBOSE < 1:
                    tqdm.write(
                        f'Local copy of {file} found, skipping download.'
                    )
            else:
                retries = 0
                success = False
                while retries < 3 and not success:
                    s3.get_file(src, dst)
                    if os.path.getsize(dst) < size:
                        tqdm.write(f'{dst} is empty/corrupted, retrying download.')
                        retries += 1
                    else:
                        success = True
                if retries == 3:
                    tqdm.write(f'{dst} failed after {retries} retries.')

        return

    def _backfill_order(self, models, product, start_dt, end_dt):
        '''
        Given a start and end time, we may need to use the model initialized
            at a prior date.

        For PM2.5:
            e.g. at 2025-01-01, 00UTC, we'd need to pull from the previous
                day's model, 2024-12-31 initialized at 12UTC

        Same with the end date; if the end date is 2025-01-01 07UTC, then
            we pull from the given date, but only the 06 initialized model,
            not the 12 init model.

        The returned values:
            start_dt, which is the start date of the model to pull from
            end_dt, which is the end date of the model to pull from
            backfill_start_steps, which hour's model to pull from to start
                - i.e. if starting at the same date and stepping forward 1,
                you're saying you want 12
            backfill_end_steps, which hour's model to pull from to end
                - i.e. if starting at the previous day and stepping back 0,
                you're saying you want 12
        '''
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

        return start_dt, backfill_start_steps, end_dt, backfill_end_steps

    ### NOTE: Cache methods
    def _load_numpy_cache(self, cache_path):
        '''
        Loads numpy cache data.
        '''
        if self.VERBOSE == 0:
            print(f'📂 Loading numpy data from {cache_path}...', end=' ')
        cached_data = np.load(cache_path, allow_pickle=True)

        # ensure data can be loaded
        try:
            data = cached_data['data']
            start_date = cached_data['start_date']
            end_date = cached_data['end_date']
            extent = cached_data['extent']
            product = cached_data['product']
        except:
            raise ValueError(
                'Cache data is missing one or more keys: '
                '(date, start_date, end_date, extent, product)'
            )

        if self.VERBOSE == 0: print(f'✅ Completed!\n')

        return cached_data

    def _save_numpy_to_cache(
        self, cache_path, data, start_date, end_date, extent, product
    ):
        '''
        Saves to cache. If the cache path is just directories, it will save 
            to the directories under the file naqfc_product_processed.npz;
            otherwise, use the cache path file.
        '''
        cache_path = (
            cache_path
            if os.path.isfile(cache_path)
            else os.path.join(cache_path, f'naqfc_{product}_processed.npz')
        )

        if self.VERBOSE == 0:
            print(f'💾 Saving data to {cache_path}...', end=' ')

        np.savez_compressed(
            file=cache_path,
            data=self.data,
            start_date=start_date,
            end_date=end_date,
            extent=extent,
            product=product
        )

        if self.VERBOSE == 0:
            print('✅ Complete!\n')

        return

    ### NOTE: Processing helpers
    def _get_dates(self, ds, available_dates):
        '''
        Combs through the Dataset to find the valid dates to pull
        Assumptions:
            - For models that initialize on 06 and 12, we'll pull
                07->12 and 13->06
            - For models that initialize on 03, we'll pull 24 hours

        Furthermore, we use the dates we want to get the intersection 
            between the dates in the Dataset and the dates we're looking
            for.

        Returns the dates we're searching for as a numpy array.
        '''
        init_hr = pd.Timestamp(ds.time.values).hour
        look_ahead = -1
        if init_hr == 6:
            look_ahead = 6
        elif init_hr == 12:
            look_ahead = 18
        elif init_hr == 3:
            look_ahead = 24
        else:
            raise ValueError('Initialization hour must be 06, 12, or 03')

        ds_dates = pd.to_datetime(ds.valid_time.values[:look_ahead])
        dates_to_pull = set(available_dates.tz_localize(None)) & set(ds_dates)

        return np.array(sorted(list(dates_to_pull)), dtype='datetime64[ns]')

    def _reproject(self, ds_product):
        '''
        Given an xarray Dataset in Lambert Conformal Conical projection, 
            convert to equirectangular projection and coordinates.

        Reprojects on a given data variable, e.g. pmtf.
        '''
        ds = ds_product
        ds = ds.rio.set_spatial_dims(x_dim='x', y_dim='y', inplace=False)
        
        # set lcc crs
        crs_lcc = CRS.from_dict({
            'proj' : 'lcc',
            'lat_1' : ds.GRIB_Latin1InDegrees,
            'lat_2' : ds.GRIB_Latin2InDegrees,
            'lat_0' : ds.GRIB_LaDInDegrees,
            'lon_0' : ds.GRIB_LoVInDegrees,
            'x_0' : 0.0,
            'y_0' : 0.0,
            'a' : 6371229.0,
            'b' : 6371229.0,
            'units' : 'm',
            'no_defs' : True
        })
        ds = ds.rio.write_crs(crs_lcc, inplace=False)
        
        # project first grid point
        lat_first = ds.GRIB_latitudeOfFirstGridPointInDegrees
        lon_first = ds.GRIB_longitudeOfFirstGridPointInDegrees
        to_lcc = Transformer.from_crs("EPSG:4326", crs_lcc, always_xy=True)
        x0_center, y0_center = to_lcc.transform(lon_first, lat_first)
        
        # build x/y coordinate arrays
        Dx, Dy = ds.GRIB_DxInMetres, ds.GRIB_DyInMetres
        Nx, Ny = ds.GRIB_Nx, ds.GRIB_Ny
        
        x = x0_center + np.arange(Nx) * Dx
        y = y0_center + np.arange(Ny) * Dy
        
        # attach coords
        ds = ds.assign_coords(x=("x", x), y=("y", y))
        
        # reproject
        m_per_degree_of_lat = 111320.0 # earth circumference (m) / 360 (deg)
        mid_lat_of_conus = np.deg2rad(35.0)
        lat_res = Dy / m_per_degree_of_lat
        lon_res = Dx / (m_per_degree_of_lat * np.cos(mid_lat_of_conus))
        
        ds_out = ds.rio.reproject(
            dst_crs='EPSG:4326',
            resolution=(lon_res, lat_res)
            
        ).rename({"y": "lat", "x": "lon"})
        
        return ds_out

    def _process_grib(self, file_path, dates, extent, dim, product):
        '''
        Processes a given grib file under the following pipeline:
            1. Open
            2. Extract valid times of the model + required dates
            3. Reproject
            4. Align extent
            5. Resize

        If a grib file is corrupted, then arrays of 0 will be returned.
            We choose this over nan because the assumption is that this
            will be ran with Residual Kriging in mind; having zeros would
            ensure the actual result is just regular kriging.

        Returns a list of numpy arrays, each array being a timestep.
        '''
        # name of the data variable in the grib file
        grib_name = {
            'pm25' : 'pmtf',
            'o3' : NotImplementedError(),
            'dust' : NotImplementedError(),
            'smoke' : NotImplementedError(),
        }

        lon_min, lon_max, lat_min, lat_max = extent

        try:
            ds = xr.load_dataset(file_path, engine='cfgrib', decode_timedelta=True)
        except (FileNotFoundError, EOFError) as e:
            tqdm.write(f'Empty/corrupted GRIB file on {file_path}: {e}.')
            return {}

        ds = ds.where(
            ds.valid_time.isin(self._get_dates(ds, dates)), drop=True
        )
        ds = self._reproject(ds[grib_name[product]])
        ds = ds.sel(
            lon=slice(lon_min, lon_max),
            lat=slice(lat_max, lat_min)
        )

        resized_data = {
            pd.to_datetime(ds.valid_time[i].item(), utc=True) : cv2.resize(ds.data[i], (dim, dim))
            for i in range(len(ds))
        }

        return resized_data

    def _process_files(self, files, dates, extent, dim, product):
        date_to_data = {d : None for d in dates}

        if self.VERBOSE < 2:
            print(f'👷 Processing batches; reprojecting and resizing data.')

        for file in (tqdm(files) if self.VERBOSE < 2 else files):
            date_to_data.update(self._process_grib(file, dates, extent, dim, product))

        # impute dates without a matching set of data with zeros
        for k in date_to_data.keys():
            if date_to_data[k] is None:
                tqdm.write(f'⚡ {k} found empty; imputed with zero frame.')
                date_to_data[k] = np.zeros((dim, dim))

        return date_to_data
