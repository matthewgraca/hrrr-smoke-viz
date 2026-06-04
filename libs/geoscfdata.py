from urllib.parse import urljoin, unquote, urlparse
from bs4 import BeautifulSoup
from pathlib import Path 
import requests
from datetime import datetime
import pandas as pd
import re
import time
from tqdm import tqdm
import traceback
import xarray as xr
import os
import cv2
import numpy as np

# pattern to extract the two dates from the geos-cf filename
# extracts first date to 'init', and the second date to 'fcast'
GEOS_CF_DATE_PATTERN = re.compile(r'(?P<init>\d{8}_\d{2}z)\+(?P<fcast>\d{8}_\d{4}z)')

# start/end of geos-cf v1 and v2 model
V1_START = pd.to_datetime('2019-12-21')
V1_END = pd.to_datetime('2026-01-01')
V2_START = pd.to_datetime('2025-08-04')

class GEOSCFData:
    '''
    A conservative webscraper for NASA GEOS-CF model ingestion and preprocessing.
    As of June 4th, 2026, NASA's CFAPI 404s, forcing us to use a webscraping 
    method.

    Highly conservative ingestion:
        One file at a time, generous backoffs, three-second wait after each 
        file download.   

    Operational and forecast mode:
        Since the model intializes every day, operational mode ingests each 
        day's worth of data and stitches them together. The goal is to 
        have a continuous timeseries of the best data, so we only care 
        about the first 24 hours of forecasts for each model initialization
        period. Expect the data in the shape: (samples, 24, dim, dim), where 
        the gap between samples (stride) is 1 hour.

        In forecast mode, we explictly want to keep all 120 hours of forecasts. 
        Use this is you want to evaluate the GEOS-CF model as a whole.
        Expect the data to be in the form (samples, 120, dim, dim), where the 
        stride is 24 hours.

    Date range:
        For operational mode, the date range doesn't matter. It will be the 
        classic; hourly, right-exclusive [start_date, end_date) or 
        [start_date, end_date - 1 hour].

        For forecast mode, since the goal is to evaluate the model itself, 
        we have guardrails that require you to ingest data at the beginning
        of the model's forecast cycle. The start date should be the hour of 
        the first forecast. 
            For the v1 model, that would be YYYY-MM-DD 13:00 since the model 
            is initialized daily at 12:00.
            For the v2 model, that would be YYYY-MM-DD 10:00 since the model
            is initialized daily at 09:00.
        
        The forecast mode date range is also right-exclusive; but since 
        bundles have strides of 24 hours, it is right-exclusive by day. So 
        it's still [start_date, end_date), but is [start_date, end_date - 1 day]

        Here's the model history:
            [v1 start: 2019-12-21, v1 end: 2026-01-01]
            [v2 start: 2020-08-04, v2 end: ongoing]

        Be warned that the models may have missing files due to maintenance 
        and development. In such cases, you'll be asked for a new date range.
     

    Numpy preprocessing
        As the data is already in gridded in lat/lon, there's no necessary 
        reprojection needed. Since the grid is tiny (usually only a few pixels), 
        the resizing up to something like an (84 x 84) grid requires 
        interpolation. We just use nearest-neighbor here, so as to not mess 
        with the inherently coarse resolution.
    '''
    def __init__(
        self,
        start_date: str,
        end_date: str,
        raw_dir: str,
        processed_path='geos_cf_processed.npz',
        extent=(-118.65, -117.70, 33.60, 34.25),
        dim=84,
        mode='operational',
    ):
        self.start_dt = pd.to_datetime(start_date)
        self.end_dt = pd.to_datetime(end_date)
        self.mode = mode
        if self.mode == 'forecast':
            if not self._valid_start_time(self.start_dt):
                raise ValueError(
                    'In forecast mode, you\'ll need to start at either hour '
                    '13 (v1) or 10 (v2)'
                )
        elif self.mode == 'operational':
            pass
        else:
            raise ValueError('Mode must be \'operational\' or \'forecast\'')
        dled_files = self._run_downloads(self.start_dt, self.end_dt, raw_dir, mode)
        is_v1 = self.start_dt < V2_START
        var = 'PM25_RH35_GCC' if is_v1 else 'PM25_RH25'
        data = self._files_to_numpy(extent, dim, dled_files, var)
        self._save_data(data, processed_dir, metadata={
            'start_date': start_date,
            'end_date': end_date,
            'extent': extent
        })
        
    ### NOTE: helpers for performing the download

    def _run_downloads(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        raw_dir: str,
        mode='operational'
    ) -> list[Path]:
        '''
        Downloads the data.

        Args:
            start_date (str): The starting date to ingest the forecasts. In 
                operational mode, you can use any date. In forecast mode, you 
                will be forced to use a date that matches the beginning of the 
                GEOS-CF model run.
            end_date (str): The end date of the final forecast to ingest, 
                exclusive. So 2025-10-27 10:00 will be excluded, with the final 
                frames using either 2025-10-27 09:00 (operational), or 
                2025-10-26 09:00 (forecast)
            raw_dir (str): The directory to save the data in. Will be stored as 
                raw_dir/nasa-geos-cf/*.nc4
            mode (str): Choice between operational and forecast. 
                In operational mode, the data will be downloaded as fresh as 
                possible, i.e. 24 hours at a time. so if 2025-08-01 10:00 is 
                given, forecasts up to 2025-08-02 09:00 will be given before 
                using the next iteration of the model.
                In forecast mode, all forecasts will be downloaded for the given 
                dates. So if 2025-08-01 10:00 is given, all 120 hours will be 
                downloaded.

        Returns:
            list[Path]: A list of the paths of the local files. 
        '''
        with requests.Session() as session:
            session.headers.update({
                'User-Agent': 'research-download-script/0.1 contact: mgraca@calstatela.edu'
            })

            date_of_death = None
            dled_files = []
            if mode == 'operational':
                try:
                    dates = pd.date_range(start_date, end_date, freq='h', inclusive='left')
                    for date in (pbar := tqdm(dates)):
                        date_of_death = date
                        url = self._to_url(session, date)
                        pbar.set_description(f'Ingesting on {date} from {url}')
                        save_path = self._download_file(session, raw_dir, url)
                        dled_files.append(save_path)
                        self._validate_nc4_file(save_path, session, raw_dir, url)
                except:
                    tqdm.write(f'Failed to ingest for {date}.')
                    tqdm.write(traceback.format_exc())
            elif mode == 'forecast':
                try:
                    dates = pd.date_range(start_date, end_date, freq='d', inclusive='left')
                    for date in (pbar_outer := tqdm(dates)):
                        date_of_death = date
                        urls = self._to_urls(session, date)
                        pbar_outer.set_description(f'Ingesting on {date}')
                        for url in (pbar_inner := tqdm(urls)):
                            pbar_inner.set_description(f'Ingesting from {url}')
                            save_path = self._download_file(session, raw_dir, url)
                            dled_files.append(save_path)
                            self._validate_nc4_file(save_path, session, raw_dir, url)
                except:
                    if date_of_death:
                        tqdm.write(f'Failed to ingest for {date}.')
                    else:
                        tqdm.write(f'Failed to ingest.')
                    tqdm.write(traceback.format_exc())

            else:
                raise ValueError('Mode must be \'operational\' or \'forecast\'')

        return dled_files

    def _valid_start_time(self, start_time: pd.Timestamp) -> bool:
        is_v1 = start_time < V2_START
        v1_match = is_v1 and start_time.hour == 13
        v2_match = not is_v1 and start_time.hour == 10
        return v1_match or v2_match

    def _validate_nc4_file(self, save_path, session, raw_dir, url) -> None:
        '''
        Checks if the nc4 file can be opened by xarray. If not, will attempt to 
        redownload.
        '''
        try:
            xr.open_dataset(save_path)
        except:
            tqdm.write(f'{save_path} failed to open by xarray; attempting redownload...')
            _ = self._download_file(session, raw_dir, url, overwrite=True)

    def _download_file(
        self,
        session: requests.Session,
        raw_dir: str,
        url: str,
        overwrite=False
    ) -> Path:
        '''
        Downloads described product from NASA GEOS model.

        Args:
            sessions (requests.Session): A requests session.
            raw_dir (str): The directory to save the data. The data will be saved
                under raw_dir/nasa-geos-cf/.
            url (str): The url to download the file.
            overwrite (bool): Whether or not to overwrite and redownload the 
                local file.

        Returns:
            Path: The path of the file that was saved. 
        '''

        save_path = Path(raw_dir) / 'nasa-geos-cf' / self._basename_from_url(url)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        #tqdm.write(f'Saving to {str(save_path)}')
        if save_path.is_file() and not overwrite:
            tqdm.write(f'{self._basename_from_url(url)} found locally, skipping download.')
        else:
            #tqdm.write(f'Attempting download: {url}')
            attempt = 0
            complete = False
            while attempt < 5 and not complete:
                try:
                    with session.get(str(url), stream=True) as response:
                        with open(save_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=1024 * 1024):
                                if chunk:
                                    f.write(chunk)
                    time.sleep(3) # chill before moving to next file
                    complete = True
                except requests.HTTPError as e:
                    status = e.response.status_code if e.response is not None else None

                    if status in {429, 500, 502, 503, 504}:
                        wait = 30 * (attempt + 1)
                        tqdm.write(f"Server returned {status}. Waiting {wait}s before retrying.")
                        time.sleep(wait)
                    else:
                        raise ValueError(f'Unhandled status code {status}')
                except requests.RequestException as e:
                    wait = 15 * (attempt + 1)
                    tqdm.write(f"Network error: {e}. Waiting {wait}s before retrying.")
                    time.sleep(wait)
                except Exception as e:
                    raise
            #tqdm.write(f'Downloaded to: {save_path}')
            if not complete:
                raise ValueError('Unable to complete request.')

        return save_path

    def _to_url(self, session: requests.Session, timestamp: pd.Timestamp) -> str:
        '''
        Converts a timestamp to a potentially viable file url to use for download.

        The following parameters are kept constant:
            config: cf (no reanalysis; geos-cf vs geos-fp)
            mode: fcst (no analysis; fcst vs ana)
            collection: aqc_tavg_1hr_glo_L1440x721_slv (no other collection)

        Args:
            sessions (requests.Session): A requests session.
            timestamp (pd.Timestamp) : A timestamp used to infer the bucket url 
                structure.

        Returns:
            str: A string of a potentially viable file url to use for download.
        '''
        bucket_url = self._bucket_url_builder(timestamp)
        urls = self._ls_files(session, bucket_url)
        if len(urls) != 120:
            raise ValueError(
                f'Bucket for {timestamp} mismatch ({len(urls)/120}). '
                'Suggest using a different time period. '
                'This can happen due to NASA uploading replay/assimilation files, '
                'or a truncated forecast range due to maintenance.'
            )

        try:
            found_path = self._find_path_with_timestamp_in_urls(timestamp, urls)
        except Exception as e:
            raise ValueError(f'Failed on {bucket_url}')

        return found_path

    def _to_urls(self, session: requests.Session, timestamp: pd.Timestamp) -> list[str]:
        '''
        Same as to_url(), but returns all of the urls in the bucket.

        The following parameters are kept constant:
            config: cf (no reanalysis)
            mode: fcst (no analysis)
            collection: aqc_tavg_1hr_glo_L1440x721_slv (no other collection)

        Args:
            sessions (requests.Session): A requests session.
            timestamp (pd.Timestamp) : A timestamp used to infer the bucket url 
                structure.

        Returns:
            list[str]: A list of strings of a potentially viable file urls to use 
            for download.
        '''
        bucket_url = self._bucket_url_builder(timestamp)
        urls = self._ls_files(session, bucket_url)
        if len(urls) != 120:
            raise ValueError(
                f'Bucket for {timestamp} mismatch ({len(urls)/120}). '
                'Suggest using a different time period. '
                'This can happen due to NASA uploading replay/assimilation files, '
                'or a truncated forecast range due to maintenance.'
            )

        try:
            found_path = self._find_path_with_timestamp_in_urls(timestamp, urls)
        except Exception as e:
            raise ValueError(f'Failed on {bucket_url}')

        return urls

    def _find_path_with_timestamp_in_urls(self, timestamp, urls):
        '''
        Grabs the url containing the timestamp. This is determined by looking 
        at the forecast time (denominated in 1 hour and 30 increments), 
        rounding to the nearest hour.

        Args:
            timestamp (pd.Timestamp): The timestamp to search the urls for.
            urls (list[str]): A list of the urls to search through.

        Returns:
            str: A string representing the path containing the timestamp.
        '''
        filenames = [self._basename_from_url(f) for f in urls]
        times = [self._parse_geos_cf_datetimes(f) for f in filenames]
        fcast_to_file = {time[1].ceil('h') : path for path, time in zip(urls, times)}
        if timestamp not in fcast_to_file:
            raise ValueError(
                f'Timestamp {timestamp} not found in the set of urls: \n'
                f'{"\n".join(urls)}'
            )
        return fcast_to_file[timestamp]

    def _bucket_url_builder(self, timestamp: pd.Timestamp) -> str:
        '''
        Reads a timestamp to generate the bucket directory the .nc4 files will 
        be read from. Forecasts eventually get removed, but air quality forecasts 
        are kept indefinitely under https://portal.nccs.nasa.gov/datashare/gmao/geos-cf

        Source https://portal.nccs.nasa.gov/datashare/gmao/geos-cf/v2/GEOS-CF_FileSpecv2.0_Draft2.pdf

        Args:
            timestamp (pd.Timestamp) : A timestamp used to infer the bucket url 
                structure.

        Returns:
            str: A string of a potentially viable bucket directory to use for download.
        '''
        if timestamp < V1_START:
            raise ValueError(f'Timestamp cannot be earlier than {V1_START}.')
        is_v1 = timestamp < V2_START

        # if the model isn't initialized yet, grab it from yesterday's forecast
        init_hour = 12 if is_v1 else 9
        ts_is_before_model_init = timestamp.hour <= init_hour
        if ts_is_before_model_init:
            timestamp = timestamp - pd.Timedelta(days=1)
            if timestamp < V1_START:
                raise ValueError(
                    f'Timestamp (after correction) cannot '
                    'be earlier than {V1_START}.'
                )

        base_url = 'https://portal.nccs.nasa.gov/datashare/gmao/geos-cf'
        model_version = 'v1' if is_v1 else 'v2'
        product = 'forecast' if is_v1 else 'fcst'
        year = 'Y' + timestamp.strftime('%Y')
        month = 'M' + timestamp.strftime('%m')
        day = 'D' + timestamp.strftime('%d')
        base_bucket_url = f'{base_url}/{model_version}/{product}/{year}/{month}/{day}/'
        bucket_url = base_bucket_url + 'H12/' if is_v1 else base_bucket_url

        return bucket_url

    def _ls_files(self, session: requests.Session, directory_url: str) -> list[str]:
        '''
        Lists the files in a given directory.
        
        Args:
            sessions (requests.Session): A requests session.
            directory_url (str): The url of the directory to list the files of.

        Returns:
            list[str]: A list of the .nc4 files in the given directory.
        '''
        
        attempt = 0
        complete = False
        while attempt < 5 and not complete:
            try:
                resp = session.get(directory_url, timeout=60)
                complete = True
            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else None

                if status in {429, 500, 502, 503, 504}:
                    wait = 30 * (attempt + 1)
                    tqdm.write(f"Server returned {status}. Waiting {wait}s before retrying.")
                    time.sleep(wait)
                else:
                    raise ValueError(f'Unhandled status code {status}')
            except requests.RequestException as e:
                wait = 15 * (attempt + 1)
                tqdm.write(f"Network error: {e}. Waiting {wait}s before retrying.")
                time.sleep(wait)
            except Exception as e:
                raise
        if not complete:
            raise ValueError('Unable to complete request.')

        soup = BeautifulSoup(resp.text, 'html.parser')

        seen = set()
        files = []

        for link in soup.find_all('a'):
            href_encoded = str(link.get('href'))
            if not href_encoded:
                continue

            filename = unquote(href_encoded)  # turns %2B into +

            if not filename.endswith('.nc4'):
                continue

            if filename in seen:
                continue

            seen.add(filename)

            # keep remote URL encoded
            files.append(urljoin(directory_url, href_encoded))  
        return files

    def _basename_from_url(self, url: str) -> str:
        '''
        Grabs basename from url, and decode url encodings to string.

        For example, '%2B' is converted back to '+'.

        Args:
            url (str): The url to containing the basename to extract.

        Returns:
            str: The basename from the url.
        '''
        return unquote(Path(urlparse(url).path).name)

    def _parse_geos_cf_datetimes(self, filename: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        '''
        Parses the geos-cf filename for the initialization and forecast dates

        Args:
            filename (str): The file name to parse.

        Returns:
            tuple[pd.Timestamp, pd.Timestamp]: A tuple where the first element is
                the model initialization time, and the second element is the model 
                forecast time.
        '''
        match = GEOS_CF_DATE_PATTERN.search(filename)

        if not match:
            raise ValueError(f'Could not find GEOS-CF timestamp pair in: {filename}')

        init_str = match.group('init')
        fcast_str = match.group('fcast')

        init_time = datetime.strptime(init_str, '%Y%m%d_%Hz')
        fcast_time = datetime.strptime(fcast_str, '%Y%m%d_%H%Mz')

        return pd.to_datetime(init_time), pd.to_datetime(fcast_time)

    ### NOTE helpers for processing the xarrays into the final payload of numpys
    def _file_to_numpy(
        self,
        file: Path,
        extent: tuple[float, float, float, float],
        dim: int,
        variable='PM25_RH35'
    ) -> np.ndarray:
        '''
        Takes a .nc4 file, extracts the data, converts it to a numpy array.
        '''
        lon_min, lon_max, lat_min, lat_max = extent
        ds = xr.open_dataset(file)

        # grab variable and subregion
        data = ds[variable].isel(time=0, lev=0).sel(
            lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)
        ).data
        data = cv2.resize(data, (dim, dim), interpolation=cv2.INTER_NEAREST)
        return data

    def _files_to_numpy(
        self,
        extent: tuple[float, float, float, float],
        dim: int,
        dled_files: list[Path],
        var: str
    ) -> np.ndarray:
        '''
        Opens the list of files with xarray, and converts it to numpy.
        '''
        files = self._map_init_times_to_files(dled_files)
        data = []
        tqdm.write('Processing xarrays into numpy bundles...')
        for k in tqdm(files.keys()):
            #print(k, len(files[k]))
            bundle = []
            for f in sorted(files[k]):
                bundle.append(self._file_to_numpy(f, extent, dim, var))
            data.append(bundle)
        return np.array(data)

    def _map_init_times_to_files(self, dled_files: list[Path]) -> dict:
        '''
        Grabs the downloaded files, creates as dictionary mapping the 
        model's initialization time to the all its forecast files.

        Args:
            dled_files (list[Path]): A list of paths where the files are 
                located.
        Returns:
            dict: A dictionary containing:
                - 'init_time' (pd.Timestamp): The timestamp the model was 
                    initialized.
                - 'files' (list[Path]): The file paths of the forecast files 
                    for that model's initialized time.
        '''
        files = {}
        for f in dled_files:
            match = GEOS_CF_DATE_PATTERN.search(str(f))

            init_str = match.group('init')
            fcast_str = match.group('fcast')

            init_dt = datetime.strptime(init_str, '%Y%m%d_%Hz')
            fcast_dt = datetime.strptime(fcast_str, '%Y%m%d_%H%Mz')

            init_time = pd.to_datetime(init_dt)
            fcast_time = pd.to_datetime(fcast_dt).ceil('h')
            
            if init_time not in files:
                files[init_time] = [f]
            else:
                files[init_time].append(f)
        return files

    def _save_data(self, data: np.ndarray, processed_path: str, metadata: dict) -> None:
        '''
        Saves the data.

        Args:
            data (np.ndarray): The data to save.
            processed_path (str): The path to save it.
            metadata (dict): The metadata to save: start date, end date, extent
        Returns:
            None
        '''
        save_path = Path(processed_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_path, data=data, **metadata)
        tqdm.write(f'Data shape {data.shape}')
        tqdm.write(f'Data written to {save_path}')

