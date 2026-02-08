import json
import pandas as pd
import re
import time
import requests
import os
import math
import numpy as np
from tqdm import tqdm
from libs.pwwb.utils.idw import IDW 
import warnings

class OpenAQData:
    def __init__(
        self,
        api_key=os.getenv('OPENAQ_API_KEY'),
        start_date='2025-01-10 00:00',
        end_date='2025-01-10 00:59',
        extent=(-118.75, -117.0, 33.5, 34.5),
        dim=40,
        product=2,              # sensor data to ingest (2 is pm2.5)
        is_nowcast=False,       # determines if the values should be nowcast or raw
        save_dir=None,          # where json files should be saved to
        save_path=None,         # where the final numpy file should be saved to
        load_json=False,        # specifies that jsons should be loaded from cache
        load_csv=False,         # specifies that the csvs should be loaded from cache
        load_numpy=False,       # specifies the numpy file should be loaded from cache
        use_interpolation=True,
        elevation_path=None,
        use_variable_blur=False,
        power=2.0,
        neighbors=10,
        elevation_scale_factor=100,
        verbose=0,              # 0 = all msgs, 1 = prog bar + errors, 2 = only errors
    ):
        '''
        Pipeline:
            1. Query for list of sensors within start/end date and extent
            2. Query for data of those sensors
            3. Read json, plot data on grids of a given dimension
            4. Interpolate

        To run from scratch, just keep load_json and load_np off.

        On save_dir, load_json, load_numpy, and load_csv:
            - If a save_dir is provided, that's where we'll read/write from as 
                our cache.
            - load_json is intended for when you have the measurements or
                sensor jsons, but not the processed numpy file.
            - load_csv is when you have processed the json files into csvs
                already.
            - load_numpy is intended for when you have the completely processed 
                numpy file. 

            Think of it as:
                all false = ingest from scratch
                json = farm-to-table raw api reponses from openaq
                csv = stitched-together jsons (no imputation or processing yet)
                numpy = gridded and interpolated
        '''
        # members
        self.data = None
        self.sensor_locations = None
        self.VERBOSE = self._validate_verbose_flag(verbose)
        self.start_date, self.end_date = (
            (None, None)
            if load_numpy
            else self._validate_dates(start_date, end_date)
        ) 
        self.extent = None if load_numpy else self._validate_extent(extent)
        self._validate_save_dir(save_dir)
        self.is_nowcast = is_nowcast

        # datetimes to use for queries
        start_dt = pd.to_datetime(start_date, utc=True)
        end_dt = pd.to_datetime(end_date, utc=True)
        dates = pd.date_range(
            start_date, end_date, freq='h', inclusive='left', tz='UTC'
        )

        if load_numpy:
            cache_path = (
                save_path 
                if save_path is not None 
                else f'{save_dir}/openaq_processed.npz'
            )
            cache_data = self._load_numpy_cache(cache_path)
            self.data = cache_data['data'] 
            self.start_date = cache_data['start_date']
            self.end_date = cache_data['end_date']
            self.extent = cache_data['extent']
            self.sensor_locations = cache_data['sensor_locations'].item()
            return
        else:
            sensor_values, df_locations = self._load_sensor_values_and_locations_df(
                api_key, self.extent, product,
                load_csv, load_json, save_dir, 
                start_dt, end_dt, dates
            )
            df_measurements = self._load_measurements_from_csv_cache(save_dir)

        # init IDW
        idw = IDW(
            power,
            neighbors,
            dim,
            elevation_path,
            elevation_scale_factor,
            use_variable_blur=False,
            verbose=self.VERBOSE
        )

        # preprocess dataframes 
        df_measurements, df_locations = self._preprocess_dataframes(
            df_measurements,
            df_locations,
            dim,
            self.extent
        )

        self.sensor_locations = dict(
            zip(df_locations['locations'], df_locations['x, y'])
        )
 
        # process to numpys
        ground_site_grids = self._df_to_gridded_data(
            df_measurements, dim, self.sensor_locations 
        )

        if self.VERBOSE < 2:
            print(
                "🐻 Performing IDW interpolation..."
                if use_interpolation
                else "IDW interpolation disabled, returning ground site grids."
            )

        grids = (
            idw.interpolate_frames(ground_site_grids)
            if use_interpolation
            else ground_site_grids
        )

        self.data = grids 

        self._save_numpy_to_cache(
            cache_path=(
                save_path 
                if save_path is not None 
                else f'{save_dir}/openaq_processed.npz'
            ),
            data=self.data,
            start_date=self.start_date,
            end_date=self.end_date,
            extent=self.extent,
            sensor_locations=self.sensor_locations
        )

        return

    ### NOTE Simplifying methods that make init cleaner

    def _load_sensor_values_and_locations_df(
        self,
        api_key,
        extent,
        product,
        load_csv,
        load_json,
        save_dir,
        start_dt,
        end_dt,
        dates
    ):
        '''
        Gets the sensor values and the locations dataframe of those sensors,
            given the user's choice in cache loading.
        '''
        if load_csv:
            df_locations = self._load_locations_from_csv_cache(save_dir)
            sensor_values = self._load_sensor_values_from_csv_cache(save_dir)
        elif load_json:
            df_locations = self._load_locations_from_json_cache(
                save_dir, start_dt, end_dt
            )
            sensor_values = self._load_sensor_values_from_json_cache(
                save_dir, df_locations, start_dt, end_dt, dates
            )
        else:
            df_locations = self._ingest_locations_from_api(
                api_key, extent, product, save_dir, start_dt, end_dt 
            )
            sensor_values = self._ingest_sensor_values_from_api(
                api_key, extent, df_locations, save_dir, start_dt, end_dt, dates
            )

        return sensor_values, df_locations

    ### NOTE: Methods for handling the query

    def _location_query(self, api_key, product, extent, save_dir):
        '''
        Extent: bounds of your region, in the form of:
            min lon, max lon, min lat, max lat
        Product: The data that is reported by the sensor, e.g.:
            2 = PM2.5

        The goal of this query is to find the sensor ids of the sensors that:
            1. Are located within the given extent
            2. Report data for the given product

        The response also gives you dates for the sensor's uptime, so this can be
            further used to prune sensors outside your date range.
        '''
        if self.VERBOSE == 0:
            tqdm.write(
                f'🔎  Performing query for sensors in extent={extent} '
                f'and product={product}...'
            )

        min_lon, max_lon, min_lat, max_lat = extent
        url = 'https://api.openaq.org/v3/locations'
        params = {
            'bbox'          : f'{min_lon},{min_lat},{max_lon},{max_lat}',
            'parameters_id' : f'{product}',
            'limit'         : 1000
        }
        headers = {'X-API-Key': api_key}

        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()

        if save_dir is not None:
            self._save_locations_json(response.json(), save_dir)

        if self.VERBOSE == 0:
            tqdm.write(
                f'{self._get_response_msg(response.status_code)}\n'
                f'Query made: {response.url}\n'
                f'Number of sensors in extent: '
                f"{len(response.json()['results'])}\n"
            )

        return response

    def _measurement_query(
        self,
        api_key,
        sensor_id,  # make sure this is the sensor for the specific product!
        start_dt,
        end_dt,
        page=1
    ):
        '''
        Query for a specific sensor

        Note on sensor_id: openaq generally has two sensor id's; one tied to
            the sensor location (think of it as the "sensor's id", and another
            for the specific product it senses. This is the one you want to 
            query.
        '''
        if self.VERBOSE == 0:
            tqdm.write(
                f'🔎  Performing query for sensor id = {sensor_id} '
                f'from {start_dt} to {end_dt}...'
            )

        url = f'https://api.openaq.org/v3/sensors/{sensor_id}/measurements'
        params = {
            'datetime_from' : start_dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
            'datetime_to'   : end_dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
            'page'          : page,
            'limit'         : 1000
        }
        headers = {'X-API-Key': api_key}
        response = requests.get(url, params=params, headers=headers)

        # handle 500+ errors
        retries = 0
        if response.status_code in {500, 501, 502, 503, 504}:
            if self.VERBOSE == 0:
                tqdm.write(f'{response.status_code} returned, will back off and retry')
            time.sleep(5)
            response = requests.get(url, params=params, headers=headers)
            retries += 1

            if retries > 4:
                tqdm.write(
                    f'{response.status_code} returned, '
                    'retries hit max of {retries}.'
                )
                response.raise_for_status()

        response.raise_for_status()

        if self.VERBOSE == 0:
            tqdm.write(
                f'{self._get_response_msg(response.status_code)}\n'
                f'Query made: {response.url}\n'
            )

        return response

    def _measurement_queries_for_a_sensor(
        self,
        api_key,
        sensor_id,
        start_dt,
        end_dt,
        dates,
        save_dir
    ):
        '''
        Uses pagination to get all the sensor values from a single sensor,
            given the start and end date

        For two years worth of data, it's estimated to be around ~17 calls

        We split the calls to a year, since openaq throws a fit and
            gives you a 408 error for doing anything

        Gaps are imputed with nan.
        '''
        date_to_sensorval = {k : np.nan for k in dates}

        # if sensor data for the range exists, we skip the query
        sensor_dir = f'{save_dir}/measurements/{sensor_id}'
        if os.path.isdir(sensor_dir):
            try:
                if self.VERBOSE == 0:
                    tqdm.write(
                        'Sensor directory found, will attempt to read from json'
                    )
                sensor_values = self._load_sensor_values_from_json_cache(
                    save_dir=save_dir,
                    df_locations=pd.DataFrame({'pm2.5 sensor id' : [sensor_id]}),
                    start_date=start_dt, 
                    end_date=end_dt,   
                    dates=dates,
                    save=False
                )
                if self.VERBOSE == 0:
                    tqdm.write('Successfully loaded sensor values from json.\n')

                return sensor_values[0]
            except Exception as e:
                if self.VERBOSE == 0:
                    tqdm.write(
                        f'{e}\n'
                        f'Unable to load full sensor values from json, '
                        f'performing query.'
                    )

        start_dates, end_dates = self._annually_split_dates(start_dt, end_dt)
        for start, end in zip(start_dates, end_dates):
            page = 1
            while page != -1:
                # query
                response = self._measurement_query(
                    api_key=api_key, 
                    sensor_id=sensor_id, 
                    start_dt=start, 
                    end_dt=end,
                    page=page
                )
                self._manage_rate_limit(response.headers)

                # read response
                response_data = response.json()

                for res in response_data['results']:
                    date = pd.to_datetime(res['period']['datetimeFrom']['utc']).round('h')
                    date_to_sensorval[date] = res['value'] 

                # save response in json
                self._save_measurements_json(save_dir, sensor_id, response_data)

                # read 'found' to determine if we should keep querying
                cont = False
                try:
                    remaining_count = int(response_data['meta']['found'])
                except ValueError: # will be '>1000' usually; just continue
                    cont = True
                page = page + 1 if cont else -1

        return [v for k, v in sorted(date_to_sensorval.items())]
    
    def _annually_split_dates(self, start_dt, end_dt):
        start_dates, end_dates = [], [] 
        current, end = start_dt, end_dt
        while current < end:
            start_dates.append(current)
            current += pd.DateOffset(years=1)
            end_dates.append(current)

        # enforce the last date to be the end datetime
        end_dates[-1] = end_dt

        return start_dates, end_dates

    def _measurement_query_for_all_sensors(
        self,
        df,         # dataframe containing the sensor ids
        api_key,
        start_dt,
        end_dt,
        dates,
        save_dir
    ):
        '''
        Simply performs the measurement query for every sensor descirbed
        '''
        sensor_values = []
        sensor_ids = (
            tqdm(list(df['pm2.5 sensor id'])) 
            if self.VERBOSE < 2 
            else list(df['pm2.5 sensor id'])
        )
        for i, sensor_id in enumerate(sensor_ids):
            if self.VERBOSE == 0:
                tqdm.write(
                    f"Ingesting data from {df.iloc[i]['provider']} "
                    f"sensor at {df.iloc[i]['locations']}."
                )
            sensor_values.append(
                self._measurement_queries_for_a_sensor(
                    api_key,
                    sensor_id,
                    start_dt,
                    end_dt,
                    dates,
                    save_dir
                )
            )
        with open('/tmp/test_openaq/debug_lengths.txt', 'w') as f:
            for loc, vals in zip(df['locations'], sensor_values):
                f.write(f"{loc}: {len(vals)}\n")
        df_measurements = self._save_measurements_csv(save_dir, df, sensor_values)

        if self.VERBOSE == 0:
            print(f'Measurements loaded:\n{df_measurements}\n')

        return sensor_values

    def _get_response_msg(self, status_code):
        '''
        Various response messages given the status code.
        '''
        server_err_msg = (
            'Server error: Something has failed on the side of OpenAQ services.'
        )
        unknown_err_msg = f'{status_code} Unknown Status Code.'
        support_msg = (
            'Go to https://docs.openaq.org/errors/about for help resolving '
            'errors.'
        )
        response_txt = {
            200 : '200 OK: Successful request.',
            401 : '401 Unauthorized: Valid API key is missing.',
            403 : (
                '403 Forbidden: The requested resource may exist but the user '
                'is not granted access. This may be a sign that the user '
                'account has been blocked for non-compliance of the terms of '
                'use.'
            ),
            404 : '404 Not Found: The requested resource does not exist.',
            405 : (
                    '405 Method Not Allowed: The HTTP method is not supported. '
                    'The OpenAQ API currently only supports GET requests.'
            ),
            408 : (
                '408 Request Timeout: The request timed out, the query may '
                'be too complex causing it to run too long.'
            ),
            410 : '410 Gone: v1 and v2 endpoints are no longer accessible.',
            422 : (
                '422 Unprocessable Content: The query provided is incorrect and'
                ' does not follow the standards set by the API specification.'
            ),
            429 : (
                '429 Too Many Requests: The number of requests exceeded the '
                'rate limit for the given time period.'
            ),
            500 : f'500 {server_err_msg}',
            502 : f'502 {server_err_msg}',
            503 : f'503 {server_err_msg}',
            504 : f'504 {server_err_msg}',
        }

        return (
            response_txt.get(status_code, unknown_err_msg)
            if status_code == 200
            else ( 
                f'{response_txt.get(status_code, unknown_err_msg)}\n'
                f'{support_msg}'
            )
        )

    def _manage_rate_limit(self, headers): 
        '''
        Checks rate limit, and throttles if queries get close to surpassing 
            the limit.

        Currently set to:
            Throttles when 90% of ratelimit is reached
            Backs off until reset + 5 seconds
        '''
        used = int(headers['X-Ratelimit-Used'])
        remains = int(headers['X-Ratelimit-Remaining'])
        reset = int(headers['X-Ratelimit-Reset'])
        limit = int(headers['X-Ratelimit-Limit'])

        max_used = math.ceil(limit * 0.9)
        if used > max_used:
            if self.VERBOSE == 0:
                tqdm.write(
                    f'90% of ratelimit reached; backing off until reset period '
                    f'in {reset + 5} seconds...'
                )
            time.sleep(reset + 5)

        return

    def _prune_sensor_list_by_date(self, data, start_dt, end_dt, save_dir):
        '''
        Given the json of locations data, return a dataframe that prunes
            this data containing only sensors that report data in between the
            given date range
        '''
        if self.VERBOSE == 0:
            print(
                f'🗓️  Pruning sensors by date operational between '
                f'{start_dt} and {end_dt}...'
            )

        # to build the dataframe
        d = {
            'sensor id'         : [],
            'pm2.5 sensor id'   : [],
            'provider'          : [],
            'locations'         : [],
            'latitude'          : [],
            'longitude'         : [] 
        }

        # further prune list of sensors based on time reporting
        for res in data['results']:
            start = pd.to_datetime(
                res['datetimeFirst']['utc']  
                if res['datetimeFirst'] is not None
                else end_dt
            )
            end = pd.to_datetime(
                res['datetimeLast']['utc']
                if res['datetimeLast'] is not None
                else start_dt
            )
            if start <= start_dt:
                d['sensor id'].append(res['id'])
                d['provider'].append(res['provider']['name'])
                d['locations'].append(res['name'])
                d['latitude'].append(res['coordinates']['latitude'])
                d['longitude'].append(res['coordinates']['longitude'])
                for sensor in res['sensors']:
                    if sensor['parameter']['id'] == 2:
                        d['pm2.5 sensor id'].append(sensor['id'])

        df = pd.DataFrame(d)

        if self.VERBOSE == 0:
            print(
                f'Number of sensors that meet criteria: {len(df)}\n'
                f"Count: {df['provider'].value_counts()}\n"
                f'{df}\n'
            )

        if save_dir is not None:
            save_path = f'{save_dir}/locations_summary.csv' 
            if self.VERBOSE == 0:
                print(
                    f'💾 Saving summary of locations data to {save_path}...',
                    end=' '
                )
            df.to_csv(save_path)
            if self.VERBOSE == 0: print('✅ Saved!\n')

        return df

    def _save_locations_json(self, response_data, save_dir):
        '''
        Saves the locations query repsonse as a json in the given save dir
        '''
        json_save_dir = f'{save_dir}/locations'
        os.makedirs(json_save_dir, exist_ok=True)
        json_save_path = f'{json_save_dir}/sensors_metadata.json'

        if self.VERBOSE == 0:
            print(f'Writing query response to {json_save_path}...')

        with open(json_save_path, 'w') as f:
            f.write(json.dumps(response_data, indent=4))

        return

    def _save_measurements_json(self, save_dir, sensor_id, response_data):
        '''
        Saves the json file of a specific sensor's measurements. Assumes
            a valid save directory.

        Saves to: 
            {save_dir}/measurements/{sensor_id}/{start_date}-{end_date}.json
        '''
        try:
            json_save_dir = f'{save_dir}/measurements/{sensor_id}'
            os.makedirs(json_save_dir, exist_ok=True)
            first_date = response_data['results'][0]['period']['datetimeFrom']['utc']
            last_date = response_data['results'][-1]['period']['datetimeFrom']['utc']
            json_save_path= f'{json_save_dir}/{first_date}_{last_date}.json'

            if self.VERBOSE == 0:
                tqdm.write(
                    f'Writing measurements from sensor {sensor_id} from '
                    f'{first_date} to {last_date} to {json_save_path}'
                )

            with open(json_save_path, 'w') as f:
                json.dump(response_data, f, indent=4)
        except:
            if self.VERBOSE == 0:
                tqdm.write(
                    f'No measurements from sensor {sensor_id} found, '
                    f'skipping save.'
                )

        return

    def _ingest_locations_from_api(
        self,
        api_key,
        extent,
        product,
        save_dir,
        start_dt,
        end_dt,
    ):
        '''
        Performs the location query along with pruning the sensor list
            by date.
        '''
        # query for list of sensors
        response = self._location_query(api_key, product, extent, save_dir)

        df_locations = self._prune_sensor_list_by_date(
            response.json(), start_dt, end_dt, save_dir 
        )

        return df_locations

    def _ingest_sensor_values_from_api(
        self,
        api_key,
        extent,
        df_locations,
        save_dir,
        start_dt,
        end_dt,
        dates
    ):
        '''
        Alias for performing the measurement query on all sensors
        '''
        # query by sensor
        sensor_values = self._measurement_query_for_all_sensors(
            df_locations, api_key, start_dt, end_dt, dates, save_dir 
        )

        return sensor_values

    ### NOTE: Methods for handling the cache

    def _load_numpy_cache(self, cache_path):
        '''
        Loads numpy cache data.
        '''
        print(f'📂 Loading numpy data from {cache_path}...', end=' ')
        cached_data = np.load(cache_path, allow_pickle=True)

        # ensure data can be loaded
        try:
            data = cached_data['data']
            start_date = cached_data['start_date']
            end_date = cached_data['end_date']
            extent = cached_data['extent']
            sensor_locations = cached_data['sensor_locations']
        except:
            raise ValueError(
                'Cache data is missing keys '
                '(date, start_date, end_date, extent)'
            )
        print(f'✅ Completed!\n')

        return cached_data
    
    def _save_numpy_to_cache(self, cache_path, data, start_date, end_date, extent, sensor_locations):
        if self.VERBOSE == 0:
            print(f'💾 Saving data to {cache_path}...', end=' ')

        np.savez_compressed(
            cache_path,
            data=self.data,
            start_date=start_date,
            end_date=end_date,
            extent=extent,
            sensor_locations=sensor_locations
        )

        if self.VERBOSE == 0:
            print('✅ Complete!\n')

        return

    def _check_datetimes_in_sensor_dir(self, sensor_dir, start_dt, end_dt):
        '''
        Checks if the files in a sensor directory contain the given start 
            and end datetimes.

        Throws errors if there's no match, returns if matches are found. 
        '''
        def find_dates_in_dir(sensor_dir):
            '''
            bro i wish
            '''
            # pattern: day_day
            pattern = re.compile(
                r"""
                (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)
                _
                (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)
                """, 
                re.VERBOSE
            )
            dates = []
            for filename in os.listdir(sensor_dir):
                match = pattern.search(filename)
                if match:
                    dates.append(pd.to_datetime(match.group(1)))
                    dates.append(pd.to_datetime(match.group(2)))

            return sorted(dates)

        if self.VERBOSE == 0:
            tqdm.write(
                f'👀 Examining files in {sensor_dir} '
                'that match start and end date...',
                end=' '
            )

        dates = find_dates_in_dir(sensor_dir)

        if dates[0] != start_dt or dates[-1] != end_dt:
            raise ValueError(
                f'📅 Date range misaligned.'
                f'start date = {dates[0]}, expected {start_dt} and '
                f'end date = {dates[-1]}, expected {end_dt}'
            )

        if self.VERBOSE == 0:
            tqdm.write('✅ Complete!')

        return

    def _open_locations_json(self, save_dir):
        '''
        Checks and opens the locations json, expecting the given path:
            {save_dir}/locations/sensors_metadata.json
        '''
        locations_path = f'{save_dir}/locations/sensors_metadata.json'
        if not os.path.exists(locations_path):
            raise ValueError(
                f'🤷 Cannot load locations data; '
                f'expected path is {locations_path}'
            )
        with open(locations_path, 'r') as f:
            loc_data = json.load(f)

        return loc_data

    def _load_measurements_jsons_of_sensor(self, sensor_dir, dates):
        '''
        Loads the sensor values of a given sensor by opening each json in 
            the sensor directory and combining the values.

        Missing values are imputed with np.nan.
        '''
        date_to_sensorval = {k : np.nan for k in dates}
        for f in sorted(os.listdir(sensor_dir)):
            with open(f'{sensor_dir}/{f}', 'r') as j:
                response_data = json.load(j)
            for res in response_data['results']:
                date = pd.to_datetime(res['period']['datetimeFrom']['utc']).round('H')
                date_to_sensorval[date] = res['value'] 

        return [v for k, v in sorted(date_to_sensorval.items())]

    def _load_locations_from_json_cache(self, save_dir, start_dt, end_dt):
        '''
        Loads the locations jsons, then prunes them by date.
        '''
        if self.VERBOSE == 0:
            print(f'📂 Attempting to load locations data from json...')

        # use locations to check if the sensors in measurements are valid
        df_locations = self._prune_sensor_list_by_date(
            data=self._open_locations_json(save_dir),
            start_dt=start_dt,
            end_dt=end_dt,
            save_dir=save_dir 
        )

        if self.VERBOSE == 0:
            print(f'Locations data loaded:\n{df_locations}\n')

        return df_locations

    def _load_sensor_values_from_json_cache(
        self,
        save_dir,
        df_locations,
        start_date, # start date used for checking files
        end_date,   # end date used for checking files
        dates,
        save=True
    ):
        '''
        Loads the sensor values from the json files by checking the
            sensor directories, then loading them.

        Also saves the csv summary.
        '''
        if self.VERBOSE == 0:
            tqdm.write(
                '📂 Attempting to load sensor values from json files...'
            )

        sensor_values = []
        for sensor_id in list(df_locations['pm2.5 sensor id']):
            sensor_dir = f'{save_dir}/measurements/{sensor_id}'
            self._check_datetimes_in_sensor_dir(
                sensor_dir=sensor_dir,
                start_dt=pd.to_datetime(start_date, utc=True),
                end_dt=pd.to_datetime(end_date, utc=True) - pd.Timedelta(hours=1),
            ) 
            # if all that is good, we load the sensor measurements
            sensor_values.append(
                self._load_measurements_jsons_of_sensor(sensor_dir, dates)
            )

        if save:
            df_measurements = self._save_measurements_csv(
                save_dir, df_locations, sensor_values
            )
            if self.VERBOSE == 0:
                print(f'Measurements loaded:\n{df_measurements}\n')

        return sensor_values
    
    def _load_locations_from_csv_cache(self, save_dir):
        if self.VERBOSE == 0:
            print(f'📂 Attempting to load locations data from csv...', end=' ')

        df_locations = pd.read_csv(
            f'{save_dir}/locations_summary.csv',
            index_col='Unnamed: 0'
        )

        if self.VERBOSE == 0:
            print('✅ Complete!')
            print(f'Locations loaded:\n{df_locations}\n')

        return df_locations
    
    def _load_sensor_values_from_csv_cache(self, save_dir):
        if self.VERBOSE == 0:
            print('📂 Attempting to load sensor values from csv...', end=' ')

        df_measurements = pd.read_csv(
            f'{save_dir}/measurements_summary.csv',
            index_col='Unnamed: 0'
        )

        if self.VERBOSE == 0:
            print('✅ Complete!')
            print(f'Measurements loaded:\n{df_measurements}\n')

        sensor_values = [list(df_measurements[col]) for col in df_measurements]

        return sensor_values

    def _load_measurements_from_csv_cache(self, save_dir):
        '''
        No messages because if we're here, we've already loaded it once. You
            might wonder why we load it twice? That's because planning is hard.
        '''
        df_measurements = pd.read_csv(
            f'{save_dir}/measurements_summary.csv',
            index_col='Unnamed: 0'
        )

        return df_measurements
    
    def _save_measurements_csv(self, save_dir, df, sensor_values):
        if self.VERBOSE == 0:
            print(f'💾 Saving measurements summary... ', end=' ')

        df_measurements = pd.DataFrame({
            location : vals
            for location, vals in zip(df['locations'], sensor_values)
        })
        df_measurements.to_csv(f'{save_dir}/measurements_summary.csv')
        
        if self.VERBOSE == 0: print('✅ Complete!\n')

        return df_measurements

    ### NOTE: Argument validation methods

    def _validate_verbose_flag(self, verbose):
        valid_options = {0, 1, 2} 
        if verbose in valid_options:
            return verbose
        else:
            raise ValueError(
                "Verbose flag must be either 0 (all messages), "
                "1 (progress bar and errors), or 2 (errors only)"
            )

        return 0 

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
    
    def _validate_save_dir(self, save_dir):
        if not os.path.isdir(save_dir):
            raise ValueError(
                f'Invalid save directory. '
                f'Either correct it or create {save_dir}.'
            )

        return

    ### NOTE: Numpy processing methods

    def _get_sensor_locations_on_grid(self, df_locations, dim, extent):
        '''
        Creates a dataframe with three columns:
            - locations
            - x, y
        Where locations is the sensor location string and x, y are the 
            converted lat/lon to x/y coordinates.
        '''
        df = pd.DataFrame({
            'lat' : df_locations['latitude'],
            'lon' : df_locations['longitude']
        })
        data = np.array(df)
        lon_min, lon_max, lat_min, lat_max = extent 
        lat_dist, lon_dist = abs(lat_max - lat_min), abs(lon_max - lon_min)
        xy_locations = []

        for i in range(data.shape[0]):
            lat, lon = data[i, 0], data[i, 1] 

            x = int(((lat_max - lat) / lat_dist) * dim)
            x = max(0, min(x, dim - 1))

            y = int(((lon - lon_min) / lon_dist) * dim)
            y = max(0, min(y, dim - 1))

            xy_locations.append((x, y))

        return pd.DataFrame(
            data=[
                (loc, xy)
                for loc, xy in zip(df_locations['locations'], xy_locations)
            ],
            columns=['locations', 'x, y']
        )

    def _preprocess_ground_sites(
        self,
        data,               # list of sensor values for one frame
        dim,     
        locations_on_grid   # list of x,y pairs of each sensor location on grid
    ):
        '''
        Plots sensor values on the grid. Handles values that would occupy the
            same location.
        '''
        grid = np.full((dim, dim), np.nan)
        
        merged_loc_to_value = self._merge_values_in_the_same_location(
            data, locations_on_grid
        )

        for loc, val in merged_loc_to_value.items():
            x, y = loc
            grid[x, y] = val
        
        return grid

    def _merge_values_in_the_same_location(self, data, locations_on_grid):
        '''
        If sensors are in the same location, we take the mean. Any nans
            will be ignored in the mean calculation.
        '''
        d = {}
        for val, loc in zip(data, locations_on_grid):
            d.setdefault(loc, []).append(val)

        # allow mean of empty slice and return nan for that sensor
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', message='Mean of empty slice')
            for k in d.keys():
                d[k] = np.nanmean(d[k])

        return d

    def _df_to_gridded_data(self, df_measurements, dim, sensor_locations):
        '''
        Converts the dataframe of sensor values to gridded data.
        '''
        if self.VERBOSE == 0: print('📍 Processing ground sites...')

        sensor_values = df_measurements.to_numpy()
        sensor_coords = sensor_locations.values()

        ground_site_grids = [
            self._preprocess_ground_sites(vals, dim, sensor_coords)
            for vals in (
                tqdm(sensor_values)
                if self.VERBOSE < 2
                else sensor_values
            )
        ]

        if self.VERBOSE == 0:
            tqdm.write(
                f" - Merged {len(sensor_locations)} sensors into "
                f"{(~np.isnan(ground_site_grids[0])).sum()} pixels."
            )

        return np.array(ground_site_grids)

    ### NOTE: Dataframe preprocessing methods

    def _preprocess_dataframes(
        self,
        df_measurements,
        df_locations,
        dim,
        extent,
        min_uptime=0.75,
        max_zscore=3,
        max_pm25=300,
        whitelist=set(['AirNow', 'Clarity'])
    ):
        #### start helpers
        # filter out sensors that are not in the whitelist
        def filter_whitelisted_sensors(
            df_measurements,
            df_locations,
            whitelist=set(['AirNow', 'Clarity'])
        ):
            if self.VERBOSE == 0:
                print(
                    f'✂️  Pruning sensors that are not in the whitelist of: '
                    f'{whitelist}'
                )

            blacklist = (
                df_locations[~df_locations['provider']
                    .isin(whitelist)]['locations']
                    .to_list()
            )
            df1 = df_measurements.drop(labels=blacklist, axis='columns')
            df2 = (
                df_locations[df_locations['provider']
                    .isin(whitelist)]
                    .reset_index(drop=True)
            )

            if self.VERBOSE == 0: print(f'Sensors removed: {blacklist}\n')

            return df1, df2

        # remove sensors than report < 75% of the time
        def remove_underreporting_sensors(df, df_loc, min_uptime):
            sensors_below_threshold = df.count() / len(df) < min_uptime
            if self.VERBOSE == 0:
                print(
                    f"✂️  Sensors to be removed for not reaching threshold:\n"
                    f"{df.columns[sensors_below_threshold].tolist()}\n"
                )

            dropped_cols = df.columns[sensors_below_threshold]

            filtered_df = df.drop(columns=dropped_cols)
            filtered_loc = df_loc[~df_loc['locations'].isin(dropped_cols)]
            
            return filtered_df, filtered_loc 

        # replace outliers (zscore > 3) with nan
        # if values > max_pm25 survive, cut those out too
        def impute_outliers_with_nan(df, max_zscore, max_pm25):
            temp_df = df.copy()
            for col in temp_df.columns:
                zscore = (temp_df[col] - temp_df[col].mean()) / temp_df[col].std()
                temp_df[col] = temp_df[col].mask(np.abs(zscore) > max_zscore)

            temp_df = temp_df.mask(temp_df >= max_pm25)
            return temp_df

        # replace all nans with a forward and back fill
        def impute_nans_with_fbfill(df):
            if self.VERBOSE == 0:
                print(
                    f"Total values that will be imputed from dead sensors "
                    f"and outliers:\n"
                    f"{df.isna().sum().sort_values(ascending=False)}\n"
                )
            return df.ffill().bfill()

        def drop_sensors_colliding_with_reference_monitors(
            df_measurements,
            df_locations
        ):
            if self.VERBOSE == 0:
                print(
                    '✂️  Pruning sensors that collide with '
                    'regulatory grade sensors... '
                )

            xy = set(df_locations[df_locations['provider'] == 'AirNow']['x, y'])

            df = df_locations[
                (df_locations['x, y'].isin(xy)) &
                (df_locations['provider'] != 'AirNow')
            ]
            trimmed_locations = df_locations[
                ~df_locations['locations'].isin(df['locations'])
            ]
            trimmed_measurements = df_measurements.drop(df['locations'], axis=1)

            if self.VERBOSE == 0:
                print('Sensors dropped due to collision with AirNow:')
                print(list(df['locations']), '\n')

            return trimmed_measurements, trimmed_locations

        def drop_sensors_outside_extent(df_measurements, df_locations, extent):
            '''
            We don't intentionally ingest out-of-range data. This is meant 
                for the scenario when we subset our extent and we don't want 
                to perform a reingest.
            '''
            if self.VERBOSE == 0:
                print(
                    '✂️  Pruning sensors that sit outside the '
                    'defined extent...'
                )
            
            min_lon, max_lon, min_lat, max_lat = extent
            out_of_extent = (
                (df_locations['latitude'] > max_lat) | 
                (df_locations['latitude'] < min_lat) | 
                (df_locations['longitude'] > max_lon) | 
                (df_locations['longitude'] < min_lon)
            )

            trimmed_locations = df_locations[~out_of_extent]
            trimmed_measurements = df_measurements.drop(
                df_locations[out_of_extent]['locations'], axis=1
            )
            print(trimmed_locations)
            print(trimmed_measurements)

            if self.VERBOSE == 0:
                print('Sensors dropped due to exceeding defined extent:')
                print(list(df_locations[out_of_extent]['locations']), '\n')

            return trimmed_measurements, trimmed_locations

        #### end helpers

        pd.set_option('display.precision', 1)
        if self.VERBOSE == 0:
            nowcast_msg = (
                " - Converting values from raw concentration to nowcast\n"
                if self.is_nowcast
                else ""
            )
            print(
                f"🧼 Cleaning data...\n"
                f" - Filtering sensors not in the whitelist: {whitelist}\n"
                f" - Removing sensors with <{min_uptime * 100:.2f}% uptime\n"
                f"{nowcast_msg}"
                f" - Removing sensors outside of the defined extent\n"
                #f" - Imputing dead sensors and outliers (zscore={max_zscore}) "
                #f"with a forward + backward fill\n"
                f"Current statistics:\n{df_measurements.describe()}\n"
            )

        filtered_df = df_measurements.copy()
        filtered_df, filtered_loc = filter_whitelisted_sensors(
            filtered_df,
            df_locations,
            whitelist
        )
        filtered_df, filtered_loc = remove_underreporting_sensors(
            filtered_df, filtered_loc, min_uptime
        )
        filtered_df, filtered_loc = drop_sensors_outside_extent(
            filtered_df, filtered_loc, extent
        )

        filtered_loc = pd.merge(
            self._get_sensor_locations_on_grid(filtered_loc, dim, extent),
            filtered_loc,
            on='locations',
            how='left'
        )
        #FIXME TODO Currently we just set the first 12 hours to nan, since nowcast
        # requires 12 hours of previous observations. To fix this, we'd need to
        # ingest 12 hours of data before the start date, then shave
        # off the first 12 hours somewhere here
        # however, it maybe annoying with how it interacts with the cache
        # perhaps: ALWAYS ingest 12 extra hours, and chop it off regardless?
        filtered_df = (
            self._compute_nowcast(filtered_df) 
            if self.is_nowcast else filtered_df
        )
        '''
        filtered_df = (
            self._compute_nowcast(filtered_df).iloc[12:].reset_index(drop=True)
            if self.is_nowcast
            else filtered_df.iloc[12:].reset_index(drop=True)
        '''
        filtered_df = impute_outliers_with_nan(filtered_df, max_zscore, max_pm25)
        #filtered_df = impute_nans_with_fbfill(filtered_df)
        filtered_df, filtered_loc = drop_sensors_colliding_with_reference_monitors(
            filtered_df, filtered_loc
        )

        if self.VERBOSE == 0:
            print(
                f"✅ Complete! Final statistics:\n"
                f"{filtered_df.describe()}\n"
            )

        return filtered_df, filtered_loc

    def _compute_nowcast(self, df):
        '''
        Converts PM2.5 raw concentrations into NowCast values.

        Would be a part of the process_dataframe pipeline. After imputation but 
            before interpolation.

        Assumptions made about the algorithm:
            - If the max of 12 hours is 0, we return 0
            - The result is truncated, not rounded
        ''' 
        WINDOW_SIZE = 12
        def nowcast(pm_data):
            if len(pm_data) < WINDOW_SIZE:
                return np.nan
            if np.isnan(pm_data.tail(3)).sum() >= 2:
                return np.nan 
            if np.nanmax(pm_data) == 0:
                return 0

            hours_ago = np.array(list(range(WINDOW_SIZE - 1, -1, -1)), dtype=float)
            hours_ago[np.isnan(pm_data)] = np.nan

            diff = np.nanmax(pm_data) - np.nanmin(pm_data)
            scaled_rate_of_change = diff / np.nanmax(pm_data)
            weight_factor = max(0.5, 1 - scaled_rate_of_change)
            weighted_pm_data = np.array([
                val * (weight_factor ** power)
                for val, power in zip(pm_data.to_numpy(), hours_ago)
            ])
            weighted_weight_factors = [weight_factor ** hour for hour in hours_ago]
            
            res = np.nansum(weighted_pm_data) / np.nansum(weighted_weight_factors)
            truncated_res = int(res * 10) / 10

            return max(0, truncated_res)

        if self.VERBOSE == 0:
            print("Converting raw concentrations to nowcast...")

        tqdm.pandas(total=df.shape[0] * df.shape[1])
        nowcast_df = (
            df.rolling(window=WINDOW_SIZE, min_periods=0).progress_apply(nowcast)
            if self.VERBOSE < 2
            else df.rolling(window=WINDOW_SIZE, min_periods=0).apply(nowcast)
        )

        return nowcast_df