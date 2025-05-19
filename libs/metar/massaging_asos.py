import pandas as pd
import numpy as np

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
    # using nearest, may consider different ones (like -1.0 impute)
    station_df = station_df.set_index('timestep', drop=True)
    station_df = station_df.reindex(full_range, method='nearest')

    # impute NaNs with -1.0. may consider smarter methods down the line
    station_df = station_df.fillna(-1.0)
    df_by_stations.append(station_df)

# merge them back
final_df = pd.concat(df_by_stations)
print(final_df)

# examine them all
'''
for date in full_range:
    print(final_df.loc[str(date)])
'''

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

np.set_printoptions(threshold=np.inf)
ground_site_grids = []
for date in full_range:
    stations = final_df.loc[str(date)]
    unInter = preprocess_ground_sites(stations, dim, max_lat, max_lon, latDist, lonDist)
    ground_site_grids.append(unInter)

print(len(ground_site_grids))
print(ground_site_grids[0])
print(np.nonzero(ground_site_grids[0]))
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
import matplotlib.pyplot as plt

def interpolate_frame(f):
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
        X = np.arange(0,200)
        Y = np.arange(0,200)
        X, Y = np.meshgrid(X, Y)
        Z = interp(X, Y)
    except ValueError:
        Z = np.zeros((200,200))
    interpolated = Z
    count += 1
    i += 1
    interpolated = np.array(interpolated)
    return interpolated

interpolated_grids = [interpolate_frame(g) for g in ground_site_grids]
import matplotlib.pyplot as plt
plt.imshow(interpolated_grids[0])
plt.show()

