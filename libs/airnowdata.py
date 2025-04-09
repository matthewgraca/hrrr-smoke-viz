import requests
import json
import os
import sys
import pandas as pd
import numpy as np
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator

class AirNowData:
    '''
    Gets the AirNow Data.
    Pipeline:
        - Uses AirNow API to download the data as a list of dataframes
        - Extracts the ground site data and converts it into a grid
        - Interpolates the grid using IDW
        - Converts grids into numpy and adds a channel axis
            - (frames, row, col, channel)
        - Creates samples from a sliding window of frames
            - (samples, frames, row, col, channel)

    Members:
        data: The complete processed AirNow data
        air_sens_loc: A dictionary of air sensor locations:
            - (x, y) : Location
    '''
    def __init__(
        self,
        start_date,
        end_date,
        extent,
        airnow_api_key=None,
        save_dir='data/airnow.json',
        #save_dir="~/data/airnow.json",
        frames_per_sample=1,
        dim=40
    ):
        self.air_sens_loc = {}
        list_df = self.__get_airnow_data(
            start_date, end_date, 
            extent, 
            save_dir,
            airnow_api_key
        )
        ground_site_grids = [
            self.__preprocess_ground_sites(df, dim, extent) for df in list_df
        ]
        interpolated_grids = [
            self.__interpolate_frame(frame, dim) for frame in ground_site_grids
        ]
        frames = np.expand_dims(np.array(interpolated_grids), axis=-1)
        processed_ds = self.__sliding_window_of(frames, frames_per_sample)

        self.data = processed_ds
        self.ground_site_grids = ground_site_grids
        #self.interpolated_grids

    '''
    Grabs the AirNow data.

    Arguments:
        start_date: The start date of the data, in the form "yyyy-mm-dd-hh"
        end_date: The end date of the data, inclusive
        extent: bounding box (a, b, c, d):
            a - minimum longitude
            b - maximum longitude
            c - minimum latitude
            d - maximum latitude
        save_dir: Path where the data should be saved
        airnow_api_key: An AirNow API key

    Returns:
        A list of dataframes containing air sensor station information.
    '''
    def __get_airnow_data(
        self, 
        start_date, end_date, 
        extent, 
        save_dir, 
        airnow_api_key
    ):
        lon_bottom, lon_top, lat_bottom, lat_top = extent
        # get airnow data from the EPA
        if os.path.exists(save_dir):
            print(f"'{save_dir}' already exists; skipping request...")
        else:
            # preprocess a few parameters
            date_start = pd.to_datetime(start_date).isoformat()[:13]
            date_end = pd.to_datetime(end_date).isoformat()[:13]
            bbox = f'{lon_bottom},{lat_bottom},{lon_top},{lat_top}'
            URL = "https://www.airnowapi.org/aq/data"

            # defining a params dict for the parameters to be sent to the API
            PARAMS = {
                'startDate':date_start,
                'endDate':date_end,
                'parameters':'PM25',
                'BBOX':bbox,
                'dataType':'B',
                'format':'application/json',
                'verbose':'1',
                'monitorType':'2',
                'includerawconcentrations':'1',
                'API_KEY':airnow_api_key
            }

            # sending get request and saving the response as response object
            response = requests.get(url = URL, params = PARAMS)

            # extracting data in json format, then download
            airnow_data = response.json()
            with open(save_dir, 'w') as file:
                json.dump(airnow_data, file)
                print("JSON data saved to '{save_dir}'")

        # open json file and convert to dataframe
        with open(save_dir, 'r') as file:
            airnow_data = json.load(file)
        airnow_df = pd.json_normalize(airnow_data)

        # group station data by time
        list_df = [group for name, group in airnow_df.groupby('UTC')]

        return list_df

    '''
    Finds the ground sites from the station dataframe, and places them
    on a grid. Also saves the location and site name into a dictionary.

    Arguments:
        df: The dataframe containing the stations
        dim: The desired dimensions of the grid
        extent: bounding box (a, b, c, d):
            a - minimum longitude
            b - maximum longitude
            c - minimum latitude
            d - maximum latitude

    Returns:
        Grid of the values at each station, as a numpy array. 
    '''
    def __preprocess_ground_sites(self, df, dim, extent):
        lonMin, lonMax, latMin, latMax = extent
        latDist, lonDist = abs(latMax - latMin), abs(lonMax - lonMin)
        unInter = np.zeros((dim,dim))
        dfArr = np.array(df[['Latitude','Longitude','Value','SiteName']])
        for i in range(dfArr.shape[0]):
            # Calculate x
            x = int(((latMax - dfArr[i,0]) / latDist) * dim)
            if x >= dim:
                x = dim - 1
            if x <= 0:
                x = 0
            # Calculate y
            y = dim - int(((lonMax + abs(dfArr[i,1])) / lonDist) * dim)
            if y >= dim:
                y = dim - 1
            if y <= 0:
                y = 0
            if dfArr[i,2] < 0:
                unInter[x,y] = 0
            else:
                unInter[x,y] = dfArr[i,2]
                self.air_sens_loc[(x, y)] = dfArr[i,3]
        return unInter

    '''
    Interpolates a frame using Inverse Distance Weighting.

    Arguments:
        f: Frame, (grid)
        dim: The desired dimensions of the new frame

    Returns:
        A new frame, interpolated.
    '''
    def __interpolate_frame(self, f, dim):
        i = 0
        interpolated = []
        count = 0
        idx = 0
        x_list = []
        y_list = []
        values = []
        for x in range(f.shape[0]):
            for y in range(f.shape[1]):
                if f[x,y] != 0:
                    x_list.append(x)
                    y_list.append(y)
                    values.append(f[x,y])
        coords = list(zip(x_list,y_list))
        try:
            interp = NearestNDInterpolator(coords, values)
            X = np.arange(0,dim)
            Y = np.arange(0,dim)
            X, Y = np.meshgrid(X, Y)
            Z = interp(X, Y)
        except ValueError:
            Z = np.zeros((dim,dim))
        interpolated = Z
        count += 1
        i += 1
        interpolated = np.array(interpolated)
        return interpolated

    '''
    Uses a sliding window to bundle frames into samples. 
    For example, with 5 frames and a 3 frames per sample, frames: 
        - 1-3 
        - 2-4
        - 3-5 
    will make up a total of 3 samples.

    Arguments:
        frames: A numpy array of the shape (num_frames, row, col, channels)
        frames_per_sample: The desired number of frames for each sample

    Returns:
        A numpy array of the shape (num_samples, num_frames, row, col, channels)
    '''
    def __sliding_window_of(self, frames, frames_per_sample):
        n_frames, row, col, channels = frames.shape
        n_samples = n_frames - frames_per_sample 
        samples = np.empty((n_samples, frames_per_sample, row, col, channels))
        for i in range(n_samples):
            samples[i] = np.array([frames[j] for j in range(i, i + frames_per_sample)])
            
        return samples
