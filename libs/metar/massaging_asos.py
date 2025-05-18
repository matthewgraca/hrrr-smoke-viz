import pandas as pd

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
    station_df['timestep'] = pd.to_datetime(station_df['valid']).dt.round('h')

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
for date in full_range:
    print(final_df.loc[str(date)])
