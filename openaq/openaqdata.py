import json
import pandas as pd
import re
import time
import requests
import os
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
- properly read data into something useful
- move current init to a "print statistics" or "location info" function
- create tests for reading response codes and validating output from a known json like airnowdata
- figure out what to do with percent coverage. impute or remove? leaning on remove depending on how many it kills.
'''

class OpenAQData:
    def __init__(self):
        response = self._location_query()
        # Save headers response.headers
        print(response.headers)

        # Save response body to file
        with open("response.json", "w") as f:
            f.write(response.text)

        # assumes you run the query and stored it in current directory as 'reponse.json'
        response_path = 'response.json'
        with open(response_path, 'r') as rfile:
            data = json.load(rfile)
            # pretty up the json if we want to manually inspect it
            with open(f'pretty_{response_path}', 'w') as wfile:
                wfile.write(json.dumps(data, indent=4))

            # goal: get sensor IDs, because we use ids to get actual data
            # query for sensors in the extent collecting PM2.5 data
            print(f"Query made: {response.url}")
            print(f"Number of sensors in extent: {len(data['results'])}\n")

            # to build the dataframe
            ids = list()
            providers = list()
            locations = list()
            lats, lons = list(), list()

            # further prune list of sensors based on time reporting
            # start = pd.to_datetime(start_date).tz_localize('UTC')
            # end = pd.to_datetime(end_date).tz_localize('UTC')
            START_DATE = pd.to_datetime("2023-08-02T00:00:00Z")
            END_DATE =  pd.to_datetime("2025-08-02T00:00:00Z")
            provider_ct = dict()
            for res in data['results']:
                name = res['provider']['name']
                start = pd.to_datetime(
                    res['datetimeFirst']['utc']  
                    if res['datetimeFirst'] is not None
                    else END_DATE
                )
                end = pd.to_datetime(
                    res['datetimeLast']['utc']
                    if res['datetimeLast'] is not None
                    else START_DATE
                )
                if start <= START_DATE and end >= END_DATE:
                    provider_ct.setdefault(name, 0)
                    provider_ct[name] = provider_ct[name] + 1
                    ids.append(res['id'])
                    providers.append(name)
                    locations.append(res['name'])
                    lats.append(res['coordinates']['latitude'])
                    lons.append(res['coordinates']['longitude'])

            pd.set_option("display.max_rows", None)
            df = pd.DataFrame({
                'id' : ids,
                'provider' : providers,
                'locations' : locations,
                'latitude' : lats,
                'longitude' : lons
            })

            print(
                "Providers that meet the criteria:\n"
                "\t1. Within extent\n"
                "\t2. Within date range\n"
                "\t3. Tracks PM2.5\n"
                f"Number of sensors that meet criteria: {len(ids)}\n"
                f"Count: {provider_ct}\n"
                f"Ids: {ids[:5]} ... {ids[-5:]}"
            )

            print(df)

            # once you have the sensor ids, you can query? Seems like it'd hit the api limit fast, with 200+ sensors.
            return

    def _location_query(self, extent=(-118.75, 33.5, -117.0, 34.5), product=2):
        '''
        Extent: bounds of your region, in the form of:
            min lon, min lat, max lon, max lat
        Product: The data that is reported by the sensor, e.g.:
            2 = PM2.5

        The goal of this query is to find the sensor ids of the sensors that:
            1. Are located within the given extent
            2. Report data for the given product

        The response also gives you dates for the sensor's uptime, so this can be
            further used to prune sensors outside your date range.
        '''
        min_lon, min_lat, max_lon, max_lat = extent
        api_key = os.getenv("OPENAQ_API_KEY")
        url = "https://api.openaq.org/v3/locations"
        params = {
            "bbox"          : f"{min_lon},{min_lat},{max_lon},{max_lat}",
            "parameters_id" : f"{product}",
            "limit"         : 1000
        }
        headers = {"X-API-Key": api_key}
        response = requests.get(url, params=params, headers=headers)

        return response

    def _measurement_queries(self, sensor_ids, start_date, end_date):
        def measurement_query(sensor_id, start_date, end_date):
            '''
            Query for a specific sensor
            '''
            api_key = os.getenv("OPENAQ_API_KEY")
            url = f"https://api.openaq.org/v3/sensors/{sensor_id}/hours"
            params = {
                "sensors_id"    : sensor_id,
                "datetime_from" : start_date,
                "datetime_to"   : end_date,
                "limit"         : 1000
            }
            headers = {"X-API-Key": api_key}
            response = requests.get(url, params=params, headers=headers)

            return response

        responses = []
        for sensor_id in tqdm(sensor_ids):
            response = measurement_query(sensor_id)
            self._manage_rate_limit(response)
            tqdm.write(self._get_response_msg(response))
            responses.append(response)

        return responses

    def _get_response_msg(self, response):
        '''
        Various response messages
        '''
        server_err_msg = (
            "Server error: Something has failed on the side of OpenAQ services."
        )
        unknown_err_msg = f"{response.status_code} Unknown Status Code."
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
            response_txt.get(response.status_code, unknown_err_msg)
            if response.status_code == 200
            else ( 
                f"{response_txt.get(response.status_code, unknown_err_msg)}\n"
                "{support_msg}"
            )
        )

    def _manage_rate_limit(self, response, verbose=True): 
        '''
        Checks rate limit, and throttles if queries get close to surpassing 
            the limit.

        Currently set to:
            Throttles when 90% of ratelimit is reached
            Backs off until reset + 5 seconds
        '''
        used = response.headers['X-Ratelimit-Used']
        remains = response.headers['X-Ratelimit-Remaining']
        reset = response.headers['X-Ratelimit-Reset']
        limit = response.headers['X-Ratelimit-Limit']

        max_used = int(limit * 0.9)
        if used > max_used:
            if verbose:
                print(
                    f"90% of ratelimit reached; backing off until reset period "
                    "in {reset + 5} seconds..."
                )
            sleep(reset + 5)

        return

