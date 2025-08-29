import json
import pandas as pd
import re
import time
import requests
import os
import math
from tqdm import tqdm

'''
pipeline: 
- query for sensors in extent reporting pm2.5 O(1)
- query for concentration data O(n) 
    - contains percent coverage, play around with that
'''

#TODO
'''
- read from cache
- write to cache
    - these will be different from goesdata because you sometimes want to read the json but ignore the numpy
- properly read data into something useful
- move current init to a "print statistics" or "location info" function
- create tests for reading response codes and validating output from a known json like airnowdata
- figure out what to do with percent coverage. impute or remove? leaning on remove depending on how many it kills.
'''

class OpenAQData:
    def __init__(
        self,
        api_key=os.getenv('OPENAQ_API_KEY'),
        start_date="2025-01-10 00:00",
        end_date="2025-01-10 00:59",
        extent=(-118.75, -117.0, 33.5, 34.5),
        dim=40,
        product=2,          # sensor data to ingest (2 is pm2.5)
        save_dir=None,      # where json files should be saved to
        load_json=False,
        verbose=0,          # 0 = all msgs, 1 = prog bar + errors, 2 = only errors
        test_mode=False,    # TODO currently being used to test class methods. will be removed once caching is implemented, and can use this class without querying
    ):
        """
        Pipeline:
            1. Query for list of sensors within start/end date and extent
            2. Query for data of those sensors
            3. Read json, plot data on grids of a given dimension
            4. Interpolate

        On save_dir, save_cache, load_cache, save_cache:
            - save_dir is just where the json files will be saved, NOT the cache 
            - cache_path is where the numpy data file lives
            - save_cache/load_cache tells us if we should save/read the data
                from cache_path
        """
        if test_mode:
            return
        if load_json:
            # TODO currently loads the sensor jsons, should eventually load the final ones
            save_path = f"{save_dir}/openaq_sensors.json"
            with open(save_path, 'r') as f:
                response_data = json.load(f)
        else:
            # query for list of sensors
            response = self._location_query(api_key, extent, product, verbose)
            response_data = response.json()
            if save_dir is not None:
                self._save_query_response(
                    save_path=f"{save_dir}/openaq_sensors.json",
                    verbose=verbose
                )

        # stagger by 1 hour (we want values from 00:00 -> 01:00 to be attributed to 1:00)
        # we also are right-exclusive, so we shave another hour off
        start_datetime = pd.to_datetime(start_date, utc=True) - pd.Timedelta(hours=1)
        end_datetime = pd.to_datetime(end_date, utc=True) - pd.Timedelta(hours=2)

        df = self._prune_sensor_list_by_date(
            response_data, 
            start_datetime, 
            end_datetime,
            verbose
        )

        dates = pd.date_range(start_datetime, end_datetime, freq='h', inclusive='left')

        # query by sensor
        vals_for_all_sensors = []
        for i, sensor_id in enumerate(tqdm(list(df['pm2.5 sensor id']))):
            sensor_values = []
            if verbose == 0:
                tqdm.write(
                    f"Ingesting data from {df.iloc[i]['provider']} "
                    f"sensor at {df.iloc[i]['location']}."
                )
            # TODO will need to test ingest and stitching all data of a sensor
            # TODO need two tests: one file for transformations, the other for queries so we 
            # don't slam into the rate limit every time we run the test suite.
            # TODO should probably save all the jsons; by sensor id and page #. save_dir/{sensor_id}/1_{datetimeFrom}_{datetimeTo}.json
            #### using pagination instead
            vals_for_all_sensors.append(
                self._measurement_queries_for_a_sensor(
                    api_key, sensor_id, start_datetime, end_datetime, verbose
                )
            )

        return

    ### NOTE: Methods for handling the query

    def _location_query(self, api_key, extent, product, verbose):
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
            print(
                f"🔎  Performing query for sensors in extent={extent} "
                f"and product={product}..."
            )

        min_lon, max_lon, min_lat, max_lat = extent
        url = "https://api.openaq.org/v3/locations"
        params = {
            "bbox"          : f"{min_lon},{min_lat},{max_lon},{max_lat}",
            "parameters_id" : f"{product}",
            "limit"         : 1000
        }
        headers = {"X-API-Key": api_key}

        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()

        if verbose == 0:
            print(
                f"{self._get_response_msg(response.status_code)}\n"
                f"Query made: {response.url}\n"
                f"Number of sensors in extent: "
                f"{len(response.json()['results'])}\n"
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
            print(
                f"🔎  Performing query for sensor id = {sensor_id} "
                f"from {start_datetime} to {end_datetime}..."
            )

        url = f"https://api.openaq.org/v3/sensors/{sensor_id}/hours"
        params = {
            "datetime_from" : start_datetime.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
            "datetime_to"   : end_datetime.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
            "page"          : page,
            "limit"         : 1000
        }
        headers = {"X-API-Key": api_key}
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()

        if verbose == 0:
            print(
                f"{self._get_response_msg(response.status_code)}\n"
                f"Query made: {response.url}\n"
            )

        return response

    def _measurement_queries_for_a_sensor(self, api_key, sensor_id, start, end, verbose):
        '''
        Uses pagination to get all the sensor values from a single sensor,
            given the start and end date

        For two years worth of data, it's estimated to be around ~17 calls
        '''
        page = 1
        dates = pd.date_range(start, end, freq='h', inclusive='left', tz='UTC')
        date_to_sensorval = {k : -1 for k in dates}
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

            # TODO the call needs to offset the starting datetime by 1 hour(datetimeTo should be the metric for that hour)
            # read 'found' to determine if we should keep querying
            cont = False
            try:
                remaining_count = int(response_data['meta']['found'])
            except ValueError: # will be ">1000" usually; just continue
                cont = True
            page = page + 1 if cont else -1

        return [v for k, v in sorted(date_to_sensorval.items())]

    def _get_response_msg(self, status_code):
        '''
        Various response messages given the status code.
        '''
        server_err_msg = (
            "Server error: Something has failed on the side of OpenAQ services."
        )
        unknown_err_msg = f"{status_code} Unknown Status Code."
        support_msg = (
            "Go to https://docs.openaq.org/errors/about for help resolving "
            "errors."
        )
        response_txt = {
            200 : "200 OK: Successful request.",
            401 : "401 Unauthorized: Valid API key is missing.",
            403 : (
                "403 Forbidden: The requested resource may exist but the user "
                "is not granted access. This may be a sign that the user "
                "account has been blocked for non-compliance of the terms of "
                "use."
            ),
            404 : "404 Not Found: The requested resource does not exist.",
            405 : (
                    "405 Method Not Allowed: The HTTP method is not supported. "
                    "The OpenAQ API currently only supports GET requests."
            ),
            408 : (
                "408 Request Timeout: The request timed out, the query may "
                "be too complex causing it to run too long."
            ),
            410 : "410 Gone: v1 and v2 endpoints are no longer accessible.",
            422 : (
                "422 Unprocessable Content: The query provided is incorrect and"
                " does not follow the standards set by the API specification."
            ),
            429 : (
                "429 Too Many Requests: The number of requests exceeded the "
                "rate limit for the given time period."
            ),
            500 : f"500 {server_err_msg}",
            502 : f"502 {server_err_msg}",
            503 : f"503 {server_err_msg}",
            504 : f"504 {server_err_msg}",
        }

        return (
            response_txt.get(status_code, unknown_err_msg)
            if status_code == 200
            else ( 
                f"{response_txt.get(status_code, unknown_err_msg)}\n"
                f"{support_msg}"
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
                    f"90% of ratelimit reached; backing off until reset period "
                    "in {reset + 5} seconds..."
                )
            sleep(reset + 5)

        return

    def _prune_sensor_list_by_date(
        self,
        data,
        start_datetime,
        end_datetime,
        verbose
    ):
        if verbose == 0:
            print(
                f"🗓️  Pruning sensors by date operational between "
                f"{start_datetime} and {end_datetime}..."
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
                else end_datetime 
            )
            end = pd.to_datetime(
                res['datetimeLast']['utc']
                if res['datetimeLast'] is not None
                else start_datetime 
            )
            if start <= start_datetime and end >= end_datetime:
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
                f"Number of sensors that meet criteria: {len(df)}\n"
                f"Count: {df['provider'].value_counts()}\n"
            )
            pd.set_option("display.max_rows", None)
            print(df, '\n')

        return df

    def _save_query_response(save_path, verbose):
        '''
        Assumes the save path is valid
        '''
        if verbose == 0:
            print(f"Writing query response to {save_path}...")
        with open(save_path, "w") as f:
            f.write(json.dumps(response.text, indent=4))

    ### NOTE: Methods for handling the cache

    def _load_cache(self, cache_path):
        """
        Loads cache data.
        """
        print(f"📂 Loading data from {cache_path}...", end=" ")
        cached_data = np.load(cache_path)

        # ensure data can be loaded
        data = cached_data['data']
        start_date = cached_data['start_date']
        end_date = cached_data['end_date']
        extent = cached_data['extent']
        print(f"✅ Completed!")

        return cached_data
    
    def _save_to_cache(self, cache_path, data, start_date, end_date, extent):
        print(f"💾 Saving data to {cache_path}...", end=" ")
        np.savez_compressed(
            cache_path,
            data=self.data,
            start_date=start_date,
            end_date=end_date,
            extent=extent
        )
        print("✅ Complete!")

        return

    def _validate_cache_path(self, save_cache, load_cache, cache_path):
        """
        Raises ValueErrors if the params don't agree
        """
        msg = (
            "In order to load from or save to cache, a cache path must be "
            "provided. "
            "Either set `load_cache` and `save_cache` to False to "
            "prevent cache loading/saving, or provide a valid `cache_path`."
        )
        if cache_path is None:
            raise ValueError(f"Cache path is None. {msg}")
        # only check if cache exists loading, b/c it will be made when saving
        if load_cache and not os.path.exists(cache_path):
            raise ValueError(f"Cache path does not exist. {msg}")
        return True

