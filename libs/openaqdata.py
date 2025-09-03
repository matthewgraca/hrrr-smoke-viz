import json
import pandas as pd
import re
import time
import requests
import os
import math
import numpy as np
from tqdm import tqdm

class OpenAQData:
    def __init__(
        self,
        api_key=os.getenv('OPENAQ_API_KEY'),
        start_date='2025-01-10 00:00',
        end_date='2025-01-10 00:59',
        extent=(-118.75, -117.0, 33.5, 34.5),
        dim=40,
        product=2,          # sensor data to ingest (2 is pm2.5)
        save_dir=None,      # where json files should be saved to
        load_json=False,    # specifies that jsons should be loaded from cache
        load_csv=False,     # speficifies that the csvs should be loaded from cache
        load_numpy=False,   # specifies the numpy file should be loaded from cache
        verbose=0,          # 0 = all msgs, 1 = prog bar + errors, 2 = only errors
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
                our cache
                Note: for LOADING, the save_dir must be valid. for SAVING, it 
                    need not be valid because it will be created.
            - load_json is intended for when you have the measurements or
                sensor jsons, but not the processed numpy file.
                1. It will attempt to read from 'measurements', then 
                    from 'locations' if that fails.
                2. It will pick up processing from there on. So this can be used
                    as a 'force reprocessing' of a numpy you don't like.
            - load_csv is when you have processed the json files already.
            - load_numpy is intended for when you have the completely processed 
                numpy file. We load it and run it as self.data.

            Think of it as:
                json = raw api reponses
                csv = processed jsons
                numpy = gridded and interpolated
        '''
        # TODO I'm starting to come around abit on private members being ok as long as they are strictly constants, e.g. dates
        # TODO esp because not having them makes the function signatures massive. I am at the very least down to make verbose private
        # datetimes to use for queries
        # stagger by 1 hour (we want values from 00:00 -> 01:00 to be attributed to 1:00)
        # we also are right-exclusive, so we shave an hour off for end time
        start_dt = pd.to_datetime(start_date, utc=True) - pd.Timedelta(hours=1)
        end_dt = pd.to_datetime(end_date, utc=True) - pd.Timedelta(hours=1)
        dates = pd.date_range(
            start_date, end_date, freq='h', inclusive='left', tz='UTC'
        )
        if load_numpy:
            cache_data = self._load_numpy_cache(cache_path)
            self.data = cache_data['data'] 
            return
        elif load_csv:
            df_locations = pd.read_csv(
                f'{save_dir}/locations_summary.csv',
                index_col='Unnamed: 0'
            )
            df_measurements = pd.read_csv(
                f'{save_dir}/measurements_summary.csv',
                index_col='Unnamed: 0'
            )
            sensor_values = [list(df_measurements[col]) for col in df_measurements]
        elif load_json:
            # use locations to check if the sensors in measurements are valid
            response_data = self._open_locations_json(save_dir)

            # then for each sensor, check if the first date matches the last date
            df_locations = self._prune_sensor_list_by_date(
                response_data, start_dt, end_dt, save_dir, verbose
            )

            sensor_values = self._load_sensor_vals_from_json_cache(
                df_locations,
                save_dir,
                start_date,
                end_date,
                start_dt,
                end_dt,
                dates,
                verbose
            )
        else:
            # query for list of sensors
            response = self._location_query(
                api_key, extent, product, save_dir, verbose
            )

            df_locations = self._prune_sensor_list_by_date(
                response.json(), start_dt, end_dt, save_dir, verbose
            )

            # query by sensor
            sensor_values = self._measurement_query_for_all_sensors(
                df_locations, api_key, start_dt, end_dt, dates, save_dir, verbose
            )

        # TODO process np 
        # by now, we have 200 sensors, each with around 17k values (for 2 yrs)
        # imputatations for gaps are np.nan 
        # then we plot them on a 1. grid, 2. impute, and 3. interpolate.
        ground_site_grids = [
            self._preprocess_ground_sites(
                df=pd.DataFrame({
                    'lat' : df_locations['latitude'],
                    'lon' : df_locations['longitude'],
                    'val' : values
                }),
                dim=dim,
                extent=extent
            )
            for values in np.transpose(sensor_values)
        ]

        return

    ### NOTE: Methods for handling the query

    def _location_query(self, api_key, extent, product, save_dir, verbose):
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
        if verbose == 0:
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
            self._save_locations_json(response.json(), save_dir, verbose)

        if verbose == 0:
            tqdm.write(
                f'{self._get_response_msg(response.status_code)}\n'
                f'Query made: {response.url}\n'
                f'Number of sensors in extent: '
                f'{len(response.json()['results'])}\n'
            )

        return response

    def _measurement_query(
        self,
        api_key,
        sensor_id,  # make sure this is the sensor for the specific product!
        start_datetime,
        end_datetime,
        page=1,
        verbose=2
    ):
        '''
        Query for a specific sensor
        '''
        if verbose == 0:
            tqdm.write(
                f'🔎  Performing query for sensor id = {sensor_id} '
                f'from {start_datetime} to {end_datetime}...'
            )

        url = f'https://api.openaq.org/v3/sensors/{sensor_id}/hours'
        params = {
            'datetime_from' : start_datetime.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
            'datetime_to'   : end_datetime.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
            'page'          : page,
            'limit'         : 1000
        }
        headers = {'X-API-Key': api_key}
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()

        if verbose == 0:
            tqdm.write(
                f'{self._get_response_msg(response.status_code)}\n'
                f'Query made: {response.url}\n'
            )

        return response

    def _measurement_queries_for_a_sensor(self, api_key, sensor_id, start, end, dates, save_dir, verbose):
        '''
        Uses pagination to get all the sensor values from a single sensor,
            given the start and end date

        For two years worth of data, it's estimated to be around ~17 calls

        Gaps are imputed with 0.
        '''
        page = 1
        date_to_sensorval = {k : np.nan for k in dates}
        while page != -1:
            # query
            response = self._measurement_query(
                api_key=api_key, 
                sensor_id=sensor_id, 
                start_datetime=start, 
                end_datetime=end,
                page=page,
                verbose=verbose
            )
            self._manage_rate_limit(response, verbose)

            # read response
            response_data = response.json()
            for res in response_data['results']:
                date_to_sensorval[
                    pd.to_datetime(res['period']['datetimeTo']['utc'])
                ] = res['value'] 

            # save response in json
            if save_dir is not None:
                self._save_measurements_json(save_dir, sensor_id, response_data, verbose)

            # read 'found' to determine if we should keep querying
            cont = False
            try:
                remaining_count = int(response_data['meta']['found'])
            except ValueError: # will be '>1000' usually; just continue
                cont = True
            page = page + 1 if cont else -1

        return [v for k, v in sorted(date_to_sensorval.items())]

    def _measurement_query_for_all_sensors(
        self,
        df,         # dataframe containing the sensor ids
        api_key,
        start_dt,
        end_dt,
        dates,
        save_dir,
        verbose
    ):
        sensor_values = []
        sensor_ids = (
            tqdm(list(df['pm2.5 sensor id'])) 
            if verbose < 2 
            else list(df['pm2.5 sensor id'])
        )
        for i, sensor_id in enumerate(sensor_ids):
            if verbose == 0:
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
                    save_dir,
                    verbose
                )
            )

        self._save_measurements_csv(save_dir, df, sensor_values, verbose)

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

    def _manage_rate_limit(self, response, verbose): 
        '''
        Checks rate limit, and throttles if queries get close to surpassing 
            the limit.

        Currently set to:
            Throttles when 90% of ratelimit is reached
            Backs off until reset + 5 seconds
        '''
        used = int(response.headers['X-Ratelimit-Used'])
        remains = int(response.headers['X-Ratelimit-Remaining'])
        reset = int(response.headers['X-Ratelimit-Reset'])
        limit = int(response.headers['X-Ratelimit-Limit'])

        max_used = math.ceil(limit * 0.9)
        if used > max_used:
            if verbose == 0:
                tqdm.write(
                    f'90% of ratelimit reached; backing off until reset period '
                    'in {reset + 5} seconds...'
                )
            time.sleep(reset + 5)

        return

    def _prune_sensor_list_by_date(
        self,
        data,
        start_dt,
        end_dt,
        save_dir,
        verbose
    ):
        '''
        Given the json of locations data, return a dataframe that prunes
            this data containing only sensors that report data in between the
            given date range
        '''
        if verbose == 0:
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
            if start <= start_dt and end >= end_dt:
                d['sensor id'].append(res['id'])
                d['provider'].append(res['provider']['name'])
                d['locations'].append(res['name'])
                d['latitude'].append(res['coordinates']['latitude'])
                d['longitude'].append(res['coordinates']['longitude'])
                for sensor in res['sensors']:
                    if sensor['parameter']['id'] == 2:
                        d['pm2.5 sensor id'].append(sensor['id'])

        df = pd.DataFrame(d)

        if verbose == 0:
            print(
                f'Number of sensors that meet criteria: {len(df)}\n'
                f"Count: {df['provider'].value_counts()}\n"
            )
            pd.set_option('display.max_rows', None)
            print(df, '\n')

        if save_dir is not None:
            save_path = f'{save_dir}/locations_summary.csv' 
            if verbose == 0:
                print(
                    f'💾 Saving summary of locations data to {save_path}...',
                    end=' '
                )
            df.to_csv(save_path)
            if verbose == 0: print("✅ Saved!")

        return df

    def _save_locations_json(self, response_data, save_dir, verbose):
        '''
        Saves the locations query repsonse as a json in the given save dir
        '''
        json_save_dir = f'{save_dir}/locations'
        os.makedirs(json_save_dir, exist_ok=True)
        json_save_path = f'{json_save_dir}/sensors_metadata.json'

        if verbose == 0:
            print(f'Writing query response to {json_save_path}...')

        with open(json_save_path, 'w') as f:
            f.write(json.dumps(response_data, indent=4))

        return

    def _save_measurements_json(self, save_dir, sensor_id, response_data, verbose):
        '''
        Saves the json file of a specific sensor's measurements. Assumes
            a valid save directory.

        Saves to: 
            {save_dir}/measurements/{sensor_id}/{start_date}-{end_date}.json
        '''
        json_save_dir = f'{save_dir}/measurements/{sensor_id}'
        os.makedirs(json_save_dir, exist_ok=True)
        first_date = response_data['results'][0]['period']['datetimeTo']['utc']
        last_date = response_data['results'][-1]['period']['datetimeTo']['utc']
        json_save_path= f'{json_save_dir}/{first_date}_{last_date}.json'

        if verbose == 0:
            tqdm.write(
                f'Writing measurements from sensor {sensor_id} from '
                f'{first_date} to {last_date} to {json_save_path}'
            )

        with open(json_save_path, 'w') as f:
            json.dump(response_data, f, indent=4)

        return

    ### NOTE: Methods for handling the cache

    def _load_numpy_cache(self, cache_path):
        '''
        Loads numpy cache data.
        '''
        print(f'📂 Loading numpy data from {cache_path}...', end=' ')
        cached_data = np.load(cache_path)

        # ensure data can be loaded
        data = cached_data['data']
        start_date = cached_data['start_date']
        end_date = cached_data['end_date']
        extent = cached_data['extent']
        print(f'✅ Completed!')

        return cached_data
    
    def _save_numpy_to_cache(self, cache_path, data, start_date, end_date, extent):
        print(f'💾 Saving data to {cache_path}...', end=' ')
        np.savez_compressed(
            cache_path,
            data=self.data,
            start_date=start_date,
            end_date=end_date,
            extent=extent
        )
        print('✅ Complete!')

        return

    def _validate_cache_path(self, save_cache, load_cache, cache_path):
        '''
        Raises ValueErrors if the params don't agree
        '''
        msg = (
            'In order to load from or save to cache, a cache path must be '
            'provided. '
            'Either set `load_cache` and `save_cache` to False to '
            'prevent cache loading/saving, or provide a valid `cache_path`.'
        )
        if cache_path is None:
            raise ValueError(f'Cache path is None. {msg}')
        # only check if cache exists loading, b/c it will be made when saving
        if load_cache and not os.path.exists(cache_path):
            raise ValueError(f'Cache path does not exist. {msg}')
        return True

    def _check_datetimes_in_sensor_dir(self, sensor_dir, start_dt, end_dt, verbose):
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

        dates = find_dates_in_dir(sensor_dir)
        if verbose == 0:
            print(
                f'👀 Examining files in {sensor_dir} '
                'that match start and end date...'
            )

        if dates[0] != start_dt or dates[-1] != end_dt:
            raise ValueError(
                f'📅 Date range misaligned.'
                f'start date = {dates[0]}, expected {start_dt} and '
                f'end date = {dates[-1]}, expected {end_dt}'
            )

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

        Missing values are imputed with -1.
        '''
        vals = []
        date_to_sensorval = {k : -1 for k in dates}
        for f in sorted(os.listdir(sensor_dir)):
            with open(f'{sensor_dir}/{f}', 'r') as j:
                response_data = json.load(j)
            for res in response_data['results']:
                date = pd.to_datetime(res['period']['datetimeTo']['utc'])
                date_to_sensorval[date] = res['value'] 
            vals.extend([v for k, v in sorted(date_to_sensorval.items())])

        return vals 

    def _load_sensor_vals_from_json_cache(
        self,
        df,         # dataframe containing locations data 
        save_dir,
        start_date, # start date used for searching files
        end_date,   # end date used for searching files
        start_dt,   # start date used for queries
        end_dt,     # end date used for queries
        dates,
        verbose
    ):
        sensor_values = []
        for sensor_id in list(df['pm2.5 sensor id']):
            sensor_dir = f'{save_dir}/measurements/{sensor_id}'
            self._check_datetimes_in_sensor_dir(
                sensor_dir=sensor_dir,
                start_dt=pd.to_datetime(start_date, utc=True),
                end_dt=pd.to_datetime(end_date, utc=True) - pd.Timedelta(hours=1),
                verbose=verbose
            ) 
            # if all that is good, we load the sensor measurements
            sensor_values.append(
                self._load_measurements_jsons_of_sensor(sensor_dir, dates)
            )

        self._save_measurements_csv(save_dir, df, sensor_values, verbose)

        if verbose == 0:
            print('Date range aligned on all sensors, values loaded.')

        return sensor_values
    
    def _save_measurements_csv(self, save_dir, df, sensor_values, verbose):
        if verbose == 0:
            print(f'💾 Saving measurements summary... ', end=' ')

        df_measurements = pd.DataFrame({
            location : vals
            for location, vals in zip(df['locations'], sensor_values)
        })
        df_measurements.to_csv(f'{save_dir}/measurements_summary.csv')
        
        if verbose == 0: print('✅ Complete!')

    # NOTE Numpy processing methods
    def _preprocess_ground_sites(self, df, dim, extent):
        """
        Places ground station data onto a regular grid with bounds checking and missing value handling.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with lat (0), lon (1), and value (2) columns
        dim : int
            Output grid dimension (dim x dim)
        extent : int 4-tuple
            Bounding box in the form: (lon min, lon max, lat min, lat max)
        
        Returns:
        --------
        numpy.ndarray
            Grid (dim x dim) with station values at their geographic positions
        """
        # TODO this is definitely inefficient, we recompute locations for EVERY frame
        # better to do a one-time compute of indicies (preprocess() + place ground sites())
        # or preprocess = find_locations() + loop : place_ground_sites()
        lon_min, lon_max, lat_min, lat_max = extent
        lat_dist, lon_dist = abs(lat_max - lat_min), abs(lon_max - lon_min)

        grid = np.full((dim, dim), np.nan)
        data = np.array(df)
        
        for i in range(data.shape[0]):
            try:
                x = int(((lat_max - data[i, 0]) / lat_dist) * dim)
                y = dim - int(((lon_max + abs(data[i, 1])) / lon_dist) * dim)
                
                x = max(0, min(x, dim - 1))
                y = max(0, min(y, dim - 1))
                
                #grid[x, y] = value if (value >= 0 and not np.isnan(value)) else 0
                # for openaq pm2.5, it's either valid or our own imputation, np.nan
                grid[x, y] = data[i, 2]
                        
            except Exception as e:
                print(f"Error placing point on grid: {e}")
        
        return grid
