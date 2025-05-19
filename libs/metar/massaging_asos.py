import pandas as pd
import numpy as np
import sys

# let's print the whole thing
df = pd.read_csv('asos.csv')
print(df)

# now lets see if we can do this for every station!
start_date = "2024-12-01 00:00"
end_date = "2024-12-02 23:00"
full_range = pd.date_range(start=start_date, end=end_date, freq='h')
station_names = list(df.groupby("station").groups.keys())
df_by_stations = []
for name in station_names:
    # group the stations
    station_df = (df
        .groupby("station")
        .get_group(name)
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

    df_by_stations.append(station_df)

# merge them into one 
df_by_stations = pd.concat(df_by_stations)

# examine them all
'''
for date in full_range:
    print(df_by_stations.loc[str(date)])
'''

# partition the stations by time so each element is a frame
df_by_time = [df_by_stations.loc[str(date)] for date in full_range]
print(df_by_time)

"""
Interpolation section. pretty much just a proof of concept, don't 
try to copy this one to one.
"""

# interpolation!
dim = 200
min_lon, max_lon, min_lat, max_lat = (-118.75, -117.5, 33.5, 34.5)
latDist = abs(max_lat - min_lat)
lonDist = abs(max_lon - min_lon)
# note: currently only processes tmpf. will need to call this func for each variable.
def preprocess_ground_sites(df, dim, latMax, lonMax, latDist, lonDist):
    unInter = np.zeros((dim,dim))
    #values = ['lat', 'lon', 'tmpf', 'dwpf', 'relh', 'feel', 'drct', 'sped', 'mslp', 'p01i'] 
    values = ['lat', 'lon', 'tmpf'] 
    dfArr = np.array(df[values])
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

ground_site_grids = [
    preprocess_ground_sites(
        df_frame, dim, max_lat, max_lon, latDist, lonDist
    ) 
    for df_frame in df_by_time
]

print(len(ground_site_grids))
print(ground_site_grids[1])
print(np.nonzero(ground_site_grids[1]))

from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
import matplotlib.pyplot as plt

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

interpolated_grids = [interpolate_frame(g, dim) for g in ground_site_grids]
import matplotlib.pyplot as plt
plt.imshow(interpolated_grids[1])
plt.show()

