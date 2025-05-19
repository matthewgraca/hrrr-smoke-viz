import pandas as pd
import numpy as np
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
import matplotlib.pyplot as plt
import sys

def interpolate_frame(f, dim):
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

# note: currently only processes tmpf. will need to call this func for each variable.
def preprocess_ground_sites(df, dim, latMax, lonMax, latDist, lonDist):
    unInter = np.zeros((dim,dim))
    dfArr = np.array(df)
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
    return unInter

def cleaned_station_df(df, station_name, full_range):
    '''
    Takes a dataframe and a station name, groups it by that station, 
    organizes by the desired time range, and imputes.

    Returns a dataframe of all the data of the given station, cleaned up.
    '''
    # group the stations
    station_df = (df
        .groupby("station")
        .get_group(station_name)
        .copy()
    )

    # round the time stamps
    station_df['timestep'] = (pd
        .to_datetime(station_df['valid'])
        .dt
        .ceil('h')
    )

    # remove duplicate timestamps
    station_df = station_df.drop_duplicates(subset=['timestep'], keep='last')

    # reindex by timestamp to generate samples
    station_df = station_df.set_index('timestep', drop=True)
    station_df = station_df.reindex(full_range)

    # we intentionally impute with values not possible so people looking at
    # it can tell it's not real data.
    nan_date = "1900-01-01 00:00"
    nan_val = -1.0
    cols_to_fill = ['station', 'lon', 'lat']
    
    # impute
    station_df['valid'] = station_df['valid'].fillna(nan_date)
    station_df[cols_to_fill] = (
        station_df[cols_to_fill]
        .bfill()
    )
    station_df = station_df.fillna(nan_val)

    return station_df

def main():
    # let's print the whole thing
    df = pd.read_csv('asos.csv')
    print(df)

    # now lets see if we can do this for every station!
    org_start_date = "2024-12-01 00:00"
    org_end_date = "2024-12-03"
    start_date = pd.to_datetime(org_start_date)
    end_date = pd.to_datetime(org_end_date) - pd.Timedelta(hours=1)
    full_range = pd.date_range(start=org_start_date, end=end_date, freq='h')
    station_names = list(df.groupby("station").groups.keys())

    # clean, impute, organize by time
    df_by_stations = [
        cleaned_station_df(df, name, full_range) 
        for name in station_names
    ]
    df_by_stations = pd.concat(df_by_stations)
    df_by_time = [df_by_stations.loc[str(date)] for date in full_range]
    print(df_by_time)
    input("Enter to continue")

    """
    Interpolation section. pretty much just a proof of concept, don't 
    try to copy this one to one.
    """

    # interpolation!
    dim = 200
    min_lon, max_lon, min_lat, max_lat = (-118.75, -117.5, 33.5, 34.5)
    latDist = abs(max_lat - min_lat)
    lonDist = abs(max_lon - min_lon)

    # produce a grid for each data variable at each timestep
    data_variables = ['tmpf', 'dwpf', 'relh', 'feel', 'drct', 'sped', 'mslp', 'p01i'] 
    shape_of_data = (len(df_by_time), dim, dim, len(data_variables))
    metar_data = []
    for df in df_by_time:
        channels = []
        for val in data_variables:
            subset_df = df[['lat', 'lon', val]].copy()
            grid = preprocess_ground_sites(
                subset_df, dim, max_lat, max_lon, latDist, lonDist
            ) 
            # 8 grids, 10 stations per grid, dim x dim resolution
            interpolated_grid = interpolate_frame(grid, dim)
            channels.append(interpolated_grid)
        metar_data.append(channels)
    metar_data = np.array(metar_data)
    print(metar_data.shape)
    input("Enter to continue")

    np.random.seed(42)
    while True:
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.flatten()
        # first sample metar_data[0 ...] is empty b/c of downloaded data dw about it :) 
        rand_idx = np.random.randint(1, len(metar_data))
        for idx, ax in enumerate(axes):
            ax.imshow(metar_data[rand_idx,idx,:,:])
            ax.axis('off')
            ax.set_title(f"{data_variables[idx]}")
        plt.suptitle(f"Sample {rand_idx}")
        plt.show()
        input("Enter to continue, Ctrl+c and enter to leave.")

if __name__ == '__main__':
    main()
