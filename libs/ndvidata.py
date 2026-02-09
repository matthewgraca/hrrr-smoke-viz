import cv2
import os
import numpy as np
import pandas as pd
import re
from harmony import Client, Environment, Collection, Request, BBox
from concurrent.futures import as_completed
from tqdm import tqdm 
from osgeo import gdal

class NDVIData:
    def __init__(
        self,
        start_date='2023-08-02',
        end_date='2023-08-22',
        extent=(-118.615, -117.70, 33.60, 34.35),
        dim=84,
        raw_dir='data',     # saved to data/modis-ndvi
        save_dir='data',    # saved to data/ndvi_processed.npz
        cache_path=None,    # if provided, loads data from given path
        verbose=0 # 0 = all, 1 = progress + errors, 2 = only errors
    ):
        self.VERBOSE = self._validate_verbose(verbose)
        if cache_path is not None:
            cache_data = self._load_numpy_cache(cache_path)
            self.data = cache_data['data'] 
            self.start_date = cache_data['start_date']
            self.end_date = cache_data['end_date']
            self.extent = cache_data['extent']
            self.dim = cache_data['data'][0].shape
            return

        self.start_date_dt = pd.to_datetime(start_date)
        self.end_date_dt = pd.to_datetime(end_date)
        self.extent = extent
        self.dim = dim
        self.raw_dir = self._validate_raw_dir(raw_dir)
        self.save_dir = self._validate_save_dir(save_dir)
        self.data = None

        NDVI_SCALE_FACTOR = 0.0001

        self._ingest_hdfs(
            self.extent, self.start_date_dt, self.end_date_dt, self.raw_dir
        )

        filled_dates = self._map_hdf_numpy_to_date_dict(
            self.raw_dir,
            self.extent,
            self.dim,
            NDVI_SCALE_FACTOR,
            self.start_date_dt,
            self.end_date_dt
        )

        self.data = self._fill_gaps_and_to_numpy(filled_dates)

        self._check_nan_frames(self.data)

        self._save_numpy_to_cache(
            save_path=os.path.join(save_dir, 'ndvi_processed.npz'),
            data=self.data,
            start_date=start_date,
            end_date=end_date,
            extent=extent
        )
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
    
    ### NOTE: Ingestion method
    def _ingest_hdfs(self, extent, start_date, end_date, raw_dir):
        lon_min, lon_max, lat_min, lat_max = extent
        if self.VERBOSE < 2:
            print('Submitting job to Harmony...', end= ' ')

        harmony_client = Client() # pulls from .netrc file
        request = Request(
            # MODIS/Terra Vegetation Indices 16-Day L3 Global 1km SIN Grid V061
            collection=Collection(id='C2565788905-LPCLOUD'),
            spatial=BBox(w=lon_min, e=lon_max, s=lat_min, n=lat_max),
            temporal={
                'start' : start_date,
                'stop' : end_date
            }
        )
        job_id = harmony_client.submit(request)
        job_desc = harmony_client.status(job_id)
        futures = harmony_client.download_all(
            job_id, directory=raw_dir, overwrite=False
        )

        if self.VERBOSE < 2:
            print('complete.')
            print('Downloading granules...')

        # silences when job completes.
        # why they look here and auto turn it on for you is... idk lol
        os.environ['VERBOSE'] = 'FALSE'         
        results = []
        for f in tqdm(
            as_completed(futures),
            total=job_desc['num_input_granules'],
            disable=self.VERBOSE == 2
        ):
            try:
                results.append(f.result())
            except Exception as e:
                # unsuccessful file dl for some reason
                tqdm.write(str(e))
                tqdm.write('Skipping corrupted file...')
                results.append(None)
        return

    ### NOTE: Frame processing methods
    def _get_subdataset(self, hdf_path, keyword='NDVI'):
        # opens dataset, exposes subdatasets
        ds = gdal.Open(hdf_path)
        if ds is None:
            raise RuntimeError(f'Unable to open {hdf_path}.')
        sub_ds = ds.GetSubDatasets()
        if not sub_ds:
            raise RuntimeError(f'No subdatasets found.')
        ds = None # clean up

        # searches for keyword in subdatasets
        matches = [
            (name, desc)
            for name, desc in sub_ds
            if keyword in name or keyword in desc
        ]
        if not matches:
            raise RuntimeError(f'No matches for {keyword} found in subdatasets')
        sds_name = matches[0][0]

        # cracks open subdataset
        found_sds = gdal.Open(sds_name)
        if found_sds is None:
            raise RuntimeError(f'Unable to open {keyword} subdataset: {sds_name}')

        return found_sds

    def _reproject(self, product_sds, extent):
        ''' 
        Reprojects a given product subdataset.
            - To equirectangular
            - Bilinear resampling
        Returns the result as a numpy array
        '''
        lon_min, lon_max, lat_min, lat_max = extent
        result = gdal.Warp(
            destNameOrDestDS='',
            srcDSOrSrcDSTab=product_sds,
            options=gdal.WarpOptions(
                dstSRS='EPSG:4326',
                resampleAlg='bilinear',
                format='MEM',
                outputBounds=(lon_min, lat_min, lon_max, lat_max),
                multithread=True
            )
        )
        if result is None:
            raise RuntimeError('gdal.Warp failed.')

        # store as array and clean up
        arr = result.ReadAsArray()
        result = None

        return arr

    def _map_hdf_numpy_to_date_dict(self, raw_dir, extent, dim, NDVI_SCALE_FACTOR, start_date, end_date):
        '''
        Loads the hdf file from the raw directory, then processes that file 
            into numpy using extent, dims, and the scale factor.

        That numpy file maps to its specific date using the start and end date
            as anchors.

        Returns the dictionary mapping date to numpy file. Any gaps will 
            simply map the date to a None object.
        '''
        gdal.UseExceptions()
        pattern = re.compile(r"""
            MOD13A2\.   # product short name
            A(\d{7})\.  # julian day of acquisition
            (\w{6})\.   # tile identifier
            (\d{3})\.   # collection version
            (\d{13})    # julian date of production
        """, re.VERBOSE)
        dates = pd.date_range(
            start_date, end_date, freq='h', inclusive='left'
        )
        filled_dates = {d : None for d in dates}
        hdf_files = sorted(os.listdir(raw_dir))
        if self.VERBOSE < 2:
            print('Processing hdf files into numpy frames...')
        for f in tqdm(hdf_files, disable=self.VERBOSE == 2):
            frame = self._process_subdataset_into_numpy_frame(
                raw_dir, f, extent, dim, NDVI_SCALE_FACTOR
            )

            # map frame to day the image was taken
            filled_dates = self._align_frame_to_date(
                dates_dict=filled_dates,
                julian_day=self._search_pattern(
                        pattern, os.path.basename(f)
                    )['julian_acquisition_day'],
                frame=frame,
                start_date=start_date
            )
            ndvi_sds = None # clean up

        return filled_dates
    
    ### NOTE: dataset building/alignment methods
    def _process_subdataset_into_numpy_frame(self, raw_dir, f, extent, dim, NDVI_SCALE_FACTOR):
        ndvi_sds = self._get_subdataset(
            os.path.join(raw_dir, f), keyword='NDVI'
        )
        frame = self._reproject(ndvi_sds, extent)
        frame = NDVI_SCALE_FACTOR * frame
        frame = cv2.resize(src=frame, dsize=(dim, dim))
        return frame

    def _search_pattern(self, pattern, string):
        '''
        Pattern moved out so that compiling the regex is only a one-time cost.

        Example: MOD13A2.A2023209.h08v05.061.2023226000837.hdf
        '''
        match = re.search(pattern, string)
        if match:
            matches = match.groups()
        return {
            'julian_acquisition_day' : matches[0],
            'tile_id' : matches[1],
            'collection_id' : matches[2],
            'julian_production_date' : matches[3]
        }

    def _align_frame_to_date(self, dates_dict, julian_day, frame, start_date):
        date = pd.to_datetime(julian_day, format='%Y%j')
        date = start_date if date < start_date else date
        if date not in dates_dict:
            raise RuntimeError(f'Date {date} not found in date range!')
        dates_dict[date] = frame
        return dates_dict

    def _fill_gaps_and_to_numpy(self, data_dict):
        '''
        Converts to dataframe to fill gaps, then converts dataframe
            to numpy
        '''
        # avoids inferring dataframe structure from first element
        df = pd.DataFrame({'frames' : data_dict.values()})
        df = df.ffill()

        frames = df.to_numpy()
        frames = np.squeeze(frames)
        frames = [
            np.full((self.dim, self.dim), np.nan) if x is None else x
            for x in frames
        ]
        frames = np.stack(frames, axis=0)

        return frames

    def _check_nan_frames(self, data):
        def count_samples_with_nan_frames(arr):
            ''' expects (samples, h, w) '''
            return np.count_nonzero(np.all(arr != arr, axis=(1, 2)))

        if self.VERBOSE < 2:
            print('Performing frame integrity check...', end=' ')
        nan_frames_count = count_samples_with_nan_frames(data)
        if self.VERBOSE < 2:
            if nan_frames_count == 0:
                print('no nan frames found.')
            else:
                print(f'warning: {nan_frames_count} nan frames found.')
                print(
                    '\tThis happens if the files covering the start date are '
                    'corrupted/missing, as you cannot forward fill without a '
                    'beginning.'
                )
        return

    ### NOTE caching methods
    def _save_numpy_to_cache(
        self, save_path, data, start_date, end_date, extent
    ):
        if self.VERBOSE == 0:
            print(f'Saving data to {save_path}...', end=' ', flush=True)

        np.savez_compressed(
            file=save_path,
            data=self.data,
            start_date=start_date,
            end_date=end_date,
            extent=extent
            # product='NDVI' if we want to extend this class for more modis
        )

        if self.VERBOSE == 0:
            print('complete!\n')

    def _load_numpy_cache(self, cache_path):
        '''
        Loads numpy cache data.
        '''
        if self.VERBOSE == 0:
            print(f'Loading numpy data from {cache_path}...', end=' ')
        cached_data = np.load(cache_path, allow_pickle=True)

        # ensure data can be loaded
        try:
            data = cached_data['data']
            start_date = cached_data['start_date']
            end_date = cached_data['end_date']
            extent = cached_data['extent']
        except:
            raise ValueError(
                'Cache data is missing one or more keys: '
                '(date, start_date, end_date, extent)'
            )

        if self.VERBOSE == 0: print(f'complete!\n')

        return cached_data
