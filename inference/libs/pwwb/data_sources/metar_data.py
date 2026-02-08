import os
import numpy as np
import pandas as pd
import requests
import io
import time
from urllib.parse import urlencode
import traceback
import cv2

from libs.pwwb.data_sources.base_data_source import BaseDataSource
from libs.pwwb.utils.interpolation import preprocess_ground_sites, interpolate_frame, elevation_aware_wind_interpolation


class MetarDataSource(BaseDataSource):
    """
    Fetches METAR weather station data with elevation-aware interpolation.
    
    Automatically derives wind U/V components from speed/direction when requested.
    Uses semantic missing values: NaN for variables that can be negative, -1.0 for positive-only variables.
    """
    
    def __init__(
        self,
        timestamps,
        extent,
        dim,
        cache_dir='data/pwwb_cache/',
        cache_prefix='',
        use_cached_data=True,
        verbose=False,
        channels=None,
        elevation_path=None
    ):
        """
        Initialize METAR data source.
        
        Parameters:
        -----------
        timestamps : pandas.DatetimeIndex
            Target timestamps for data collection
        extent : tuple
            Geographic bounds (min_lon, max_lon, min_lat, max_lat)
        dim : int
            Output grid resolution (dim x dim)
        cache_dir : str
            Cache directory path
        cache_prefix : str
            Cache filename prefix
        use_cached_data : bool
            Whether to use existing cache
        verbose : bool
            Enable detailed logging
        channels : list, optional
            Specific METAR channels to include (default: all)
        elevation_path : str, optional
            Path to elevation data file
        """
        super().__init__(timestamps, extent, dim, cache_dir, cache_prefix, use_cached_data, verbose)
        
        self.raw_channels = [
            'METAR_Wind_Speed', 'METAR_Wind_Direction', 'METAR_Precipitation', 
            'METAR_Humidity', 'METAR_Heat_Index', 'METAR_Air_Temp', 
            'METAR_Air_Pressure', 'METAR_Dew_Point'
        ]
        
        self.derived_channels = ['METAR_Wind_U', 'METAR_Wind_V']
        self.all_channels = self.raw_channels + self.derived_channels
        
        self.channel_mapping = {
            'METAR_Wind_Speed': 'sped',
            'METAR_Wind_Direction': 'drct',
            'METAR_Precipitation': 'p01i',
            'METAR_Humidity': 'relh',
            'METAR_Heat_Index': 'feel',
            'METAR_Air_Temp': 'tmpf',
            'METAR_Air_Pressure': 'mslp',
            'METAR_Dew_Point': 'dwpf',
            'METAR_Wind_U': 'u_component',
            'METAR_Wind_V': 'v_component'
        }
        
        self.variables_that_can_be_negative = {
            'u_component', 'v_component', 'tmpf', 'dwpf'
        }
        
        if channels is not None:
            self.channels = [c for c in channels if c in self.all_channels]
        else:
            self.channels = self.all_channels.copy()
        
        self.need_wind_components = 'METAR_Wind_U' in self.channels or 'METAR_Wind_V' in self.channels
        
        self.data_variables = []
        
        for channel in self.channels:
            if channel in self.raw_channels and channel in self.channel_mapping:
                self.data_variables.append(self.channel_mapping[channel])
        
        if self.need_wind_components:
            sped_var = self.channel_mapping['METAR_Wind_Speed']
            drct_var = self.channel_mapping['METAR_Wind_Direction']
            
            if sped_var not in self.data_variables:
                self.data_variables.append(sped_var)
            if drct_var not in self.data_variables:
                self.data_variables.append(drct_var)
        
        self.elevation_path = elevation_path if elevation_path else "inputs/elevation.npy"
        os.makedirs(os.path.dirname(self.elevation_path), exist_ok=True)
        
        if os.path.exists(self.elevation_path):
            self.elevation = np.load(self.elevation_path)
            if self.elevation.shape != (self.dim, self.dim):
                self.elevation = cv2.resize(self.elevation, (self.dim, self.dim))
            self.elevation = self._normalize_elevation(self.elevation)
            if self.verbose:
                print(f"Loaded elevation data from {self.elevation_path}")
                print(f"Elevation range: {np.min(self.elevation):.0f}m to {np.max(self.elevation):.0f}m")
        else:
            if self.verbose:
                print(f"Elevation data not found at {self.elevation_path}. Using flat elevation.")
            self.elevation = np.zeros((self.dim, self.dim), dtype=np.float32)
        
        self.metar_sensors = [
            {'id': 'LAX', 'name': 'Los Angeles Intl', 'lat': 33.9382, 'lon': -118.3865},
            {'id': 'BUR', 'name': 'Burbank/Glendale', 'lat': 34.2007, 'lon': -118.3587},
            {'id': 'LGB', 'name': 'Long Beach Airport', 'lat': 33.8118, 'lon': -118.1472},
            {'id': 'VNY', 'name': 'Van Nuys Airport', 'lat': 34.2097, 'lon': -118.4892},
            {'id': 'SMO', 'name': 'Santa Monica Muni', 'lat': 34.0210, 'lon': -118.4471},
            {'id': 'HHR', 'name': 'Hawthorne Municipal', 'lat': 33.9228, 'lon': -118.3352},
            {'id': 'EMT', 'name': 'El Monte', 'lat': 34.0860, 'lon': -118.0350},
            {'id': 'SNA', 'name': 'Santa Ana/John Wayne', 'lat': 33.6757, 'lon': -117.8682},
            {'id': 'ONT', 'name': 'Ontario Intl', 'lat': 34.0560, 'lon': -117.6012},
            {'id': 'TOA', 'name': 'Zamperini Field (Torrance)', 'lat': 33.8034, 'lon': -118.3396},
            {'id': 'WHP', 'name': 'Whiteman Airport', 'lat': 34.2593, 'lon': -118.4134},
            {'id': 'CCB', 'name': 'Cable Airport (Upland)', 'lat': 34.1115, 'lon': -117.6876},
            {'id': 'POC', 'name': 'Brackett Field (La Verne)', 'lat': 34.0917, 'lon': -117.7817},
            {'id': 'CNO', 'name': 'Chino Airport', 'lat': 33.9747, 'lon': -117.6389},
            {'id': 'FUL', 'name': 'Fullerton Municipal', 'lat': 33.8719, 'lon': -117.9798},
            {'id': 'AJO', 'name': 'Corona Municipal Airport', 'lat': 33.8975, 'lon': -117.6003},
        ]
        
        if self.verbose:
            print(f"Initialized MetarDataSource with {len(self.channels)} channels: {self.channels}")
            print(f"Will fetch these raw variables: {self.data_variables}")
            if self.need_wind_components:
                print("Will calculate wind U/V components from speed/direction")
    
    def _normalize_elevation(self, elevation_data):
        """Normalize elevation to 0-100 range to prevent calculation overflow."""
        min_val = np.min(elevation_data)
        max_val = np.max(elevation_data)
        
        if max_val == min_val:
            return np.zeros_like(elevation_data)
            
        normalized = (elevation_data - min_val) / (max_val - min_val) * 100
        return normalized.astype(np.float32)
    
    def get_data(self):
        """
        Fetch meteorological data from Iowa Environmental Mesonet ASOS/AWOS database.
        
        Returns:
        --------
        numpy.ndarray
            METAR data (n_timestamps, dim, dim, n_features) with elevation-aware interpolation
        """
        channel_id = '_'.join([c.split('_')[-1].lower() for c in self.channels])
        cache_file = os.path.join(self.cache_dir, f"{self.cache_prefix}metar_{channel_id}_data.npy")
        
        output_channel_count = len(self.channels)
        
        data = self.check_cache(cache_file, (self.n_timestamps, self.dim, self.dim, output_channel_count))
        if data is not None:
            return data
        
        metar_data = np.zeros((self.n_timestamps, self.dim, self.dim, output_channel_count))
        
        if self.verbose:
            print(f"Including METAR variables: {self.data_variables}")
            print(f"Preparing data for channels: {self.channels}")
        
        min_lon, max_lon, min_lat, max_lat = self.extent
        stations_in_bounds = [
            station for station in self.metar_sensors
            if min_lon <= station['lon'] <= max_lon and min_lat <= station['lat'] <= max_lat
        ]
        
        if not stations_in_bounds:
            if self.verbose:
                print("Warning: No METAR stations found within bounds, using all stations")
            stations_in_bounds = self.metar_sensors
        
        if self.verbose:
            print(f"Using {len(stations_in_bounds)} METAR stations:")
            for station in stations_in_bounds:
                print(f"  {station['id']} - {station['name']} ({station['lat']}, {station['lon']})")
        
        station_ids = [station['id'] for station in stations_in_bounds]
        
        chunk_size = pd.Timedelta(days=90)
        current_start = self.timestamps[0]
        current_end = self.timestamps[-1]
        
        all_station_data = {}
        
        while current_start <= current_end:
            chunk_end = min(current_start + chunk_size, current_end)
            
            if self.verbose:
                print(f"Fetching METAR data chunk: {current_start.date()} to {chunk_end.date()}")
            
            chunk_data = self._fetch_iem_metar_data(station_ids, current_start, chunk_end)
            
            for station_id, data in chunk_data.items():
                if station_id not in all_station_data:
                    all_station_data[station_id] = []
                all_station_data[station_id].extend(data)
            
            current_start = chunk_end + pd.Timedelta(seconds=1)
        
        if self.verbose:
            print("METAR data fetching complete")
            for station_id, data in all_station_data.items():
                print(f"  Station {station_id}: {len(data)} total records")
        
        metar_df = self._create_metar_dataframe(all_station_data, stations_in_bounds)
        full_range = pd.date_range(start=self.timestamps[0], end=self.timestamps[-1], freq='h')
        
        station_names = list(metar_df.groupby("station").groups.keys())
        
        df_by_stations = []
        for name in station_names:
            try:
                station_df = self._cleaned_station_df(metar_df, name, full_range)
                df_by_stations.append(station_df)
            except Exception as e:
                if self.verbose:
                    print(f"Error processing station {name}: {e}")
                    traceback.print_exc()
        
        if not df_by_stations:
            if self.verbose:
                print("No valid station data found. Returning empty METAR data.")
            self.save_to_cache(metar_data, cache_file)
            return metar_data
        
        df_by_stations = pd.concat(df_by_stations)
        
        sped_var = self.channel_mapping['METAR_Wind_Speed']
        drct_var = self.channel_mapping['METAR_Wind_Direction']
        
        if self.need_wind_components and sped_var in df_by_stations.columns and drct_var in df_by_stations.columns:
            wind_speed_mph = df_by_stations[sped_var]
            wind_direction = df_by_stations[drct_var]

            valid_wind_mask = (wind_speed_mph != -1.0) & (wind_direction != -1.0)

            u_component = np.full_like(wind_speed_mph, np.nan, dtype=float)
            v_component = np.full_like(wind_speed_mph, np.nan, dtype=float)

            wind_dir_rad = np.radians(wind_direction)
            u_component[valid_wind_mask] = -wind_speed_mph[valid_wind_mask] * np.sin(wind_dir_rad[valid_wind_mask])
            v_component[valid_wind_mask] = -wind_speed_mph[valid_wind_mask] * np.cos(wind_dir_rad[valid_wind_mask])

            df_by_stations['u_component'] = u_component
            df_by_stations['v_component'] = v_component
            
            if self.verbose:
                print("Computed wind U/V components from speed/direction")
                print(f"U component range: {np.nanmin(u_component):.2f} to {np.nanmax(u_component):.2f}")
                print(f"Valid wind component count: {np.sum(valid_wind_mask)} / {len(valid_wind_mask)}")
        
        df_by_time = []
        for date in full_range:
            try:
                df_by_time.append(df_by_stations.loc[str(date)])
            except KeyError:
                if self.verbose:
                    print(f"No data for timestamp {date}")
                df_by_time.append(pd.DataFrame())
        
        if self.verbose:
            print(f"Processing {len(df_by_time)} timestamps of METAR data")
        
        latDist = abs(max_lat - min_lat)
        lonDist = abs(max_lon - min_lon)
        
        for t_idx, df in enumerate(df_by_time):
            if df.empty:
                continue
            
            u_idx = -1
            v_idx = -1
            if 'METAR_Wind_U' in self.channels:
                u_idx = self.channels.index('METAR_Wind_U')
            if 'METAR_Wind_V' in self.channels:
                v_idx = self.channels.index('METAR_Wind_V')
            
            wind_components_processed = False
            
            if u_idx >= 0 and v_idx >= 0:
                try:
                    u_var = self.channel_mapping['METAR_Wind_U']
                    v_var = self.channel_mapping['METAR_Wind_V']
                    
                    wind_data = []
                    for _, row in df.iterrows():
                        if (not np.isnan(row[u_var]) and not np.isnan(row[v_var]) and
                            not np.isnan(row['lat']) and not np.isnan(row['lon'])):
                            wind_data.append({
                                'lon': row['lon'],
                                'lat': row['lat'],
                                'u': row[u_var],
                                'v': row[v_var]
                            })
                    
                    if len(wind_data) > 0:
                        stations = [[d['lon'], d['lat']] for d in wind_data]
                        u_values = [d['u'] for d in wind_data]
                        v_values = [d['v'] for d in wind_data]
                        
                        u_grid, v_grid = elevation_aware_wind_interpolation(
                            stations, u_values, v_values,
                            self.extent, self.dim, self.elevation,
                            power=0.2,
                            elevation_weight=0.10,
                            smoothing_sigma=0,
                            verbose=self.verbose and t_idx == 0
                        )
                        
                        metar_data[t_idx, :, :, u_idx] = u_grid
                        metar_data[t_idx, :, :, v_idx] = v_grid
                        wind_components_processed = True
                        
                        if self.verbose and t_idx == 0:
                            print(f"3D elevation-aware wind interpolation: {len(stations)} stations")
                    
                except Exception as e:
                    if self.verbose:
                        print(f"Error in wind vector interpolation for timestamp {t_idx}: {e}")
            
            for channel_idx, channel in enumerate(self.channels):
                if channel in ['METAR_Wind_U', 'METAR_Wind_V'] and wind_components_processed:
                    continue
                    
                if channel in self.channel_mapping:
                    var = self.channel_mapping[channel]
                    
                    try:
                        subset_df = df[['lat', 'lon', var]].copy()
                        is_negative_allowed = var in self.variables_that_can_be_negative
                        
                        grid = preprocess_ground_sites(
                            subset_df, self.dim, max_lat, max_lon, latDist, lonDist,
                            allow_negative=is_negative_allowed
                        )
                        
                        interpolated_grid = interpolate_frame(grid, self.dim)
                        metar_data[t_idx, :, :, channel_idx] = interpolated_grid
                        
                    except Exception as e:
                        if self.verbose:
                            print(f"Error processing channel {channel} for timestamp {t_idx}: {e}")
                            traceback.print_exc()
        
        if self.verbose:
            print(f"Created METAR data with shape {metar_data.shape}")
            print(f"Final channels: {self.channels}")
            
            for i, channel in enumerate(self.channels):
                data = metar_data[0, :, :, i]
                print(f"{channel} sample stats: min={np.nanmin(data):.2f}, max={np.nanmax(data):.2f}, mean={np.nanmean(data):.2f}")
        
        self.save_to_cache(metar_data, cache_file)
        return metar_data
    
    def _get_missing_value_for_variable(self, variable_name):
        """Return appropriate missing value marker: NaN for variables that can be negative, -1.0 otherwise."""
        if variable_name in self.variables_that_can_be_negative:
            return np.nan
        else:
            return -1.0
    
    def _create_metar_dataframe(self, all_station_data, stations_in_bounds):
        """Convert nested station data to flat DataFrame with proper missing value handling."""
        station_lookup = {station['id']: station for station in stations_in_bounds}
        all_records = []
        
        for station_id, records in all_station_data.items():
            station_info = station_lookup.get(station_id)
            if not station_info:
                continue
                
            for record in records:
                try:
                    flat_record = {
                        'station': station_id,
                        'name': station_info['name'],
                        'lat': station_info['lat'],
                        'lon': station_info['lon'],
                        'valid': record.get('valid', '')
                    }
                    
                    for field in self.data_variables:
                        missing_value = self._get_missing_value_for_variable(field)
                        flat_record[field] = record.get(field, missing_value)
                    
                    all_records.append(flat_record)
                except Exception as e:
                    if self.verbose:
                        print(f"Error flattening record from {station_id}: {e}")
        
        if all_records:
            df = pd.DataFrame(all_records)
            df['valid'] = pd.to_datetime(df['valid'])
            return df
        else:
            columns = ['station', 'name', 'lat', 'lon', 'valid'] + self.data_variables
            return pd.DataFrame(columns=columns)

    def _cleaned_station_df(self, df, station_name, full_range):
        """Clean and reindex station data with semantic missing value imputation."""
        try:
            station_df = df.groupby("station").get_group(station_name).copy()
            station_df['timestep'] = pd.to_datetime(station_df['valid']).dt.ceil('h')
            station_df = station_df.drop_duplicates(subset=['timestep'], keep='last')
            station_df = station_df.set_index('timestep', drop=True)
            station_df = station_df.reindex(full_range)
            
            nan_date = "1900-01-01 00:00"
            cols_to_fill = ['station', 'lon', 'lat']
            
            station_df['valid'] = station_df['valid'].fillna(nan_date)
            station_df[cols_to_fill] = station_df[cols_to_fill].bfill()
            
            for column in station_df.columns:
                if column not in ['station', 'name', 'lat', 'lon', 'valid']:
                    missing_value = self._get_missing_value_for_variable(column)
                    station_df[column] = station_df[column].fillna(missing_value)
            
            return station_df
        
        except Exception as e:
            if self.verbose:
                print(f"Error cleaning station {station_name}: {e}")
                traceback.print_exc()
            return pd.DataFrame(index=full_range)
    
    def _fetch_iem_metar_data(self, stations, start_date, end_date):
        """Fetch METAR data from Iowa Environmental Mesonet with retry logic and caching."""
        if not stations:
            if self.verbose:
                print("No stations specified for IEM METAR data fetch")
            return {}
        
        station_str = ",".join(stations)
        start_date_with_margin = start_date - pd.Timedelta(hours=1)
        end_date_with_margin = end_date + pd.Timedelta(hours=1)
        
        if self.verbose:
            print(f"Fetching METAR data from {start_date_with_margin} to {end_date_with_margin}")
        
        start_str = start_date_with_margin.strftime('%Y-%m-%d %H:%M')
        end_str = end_date_with_margin.strftime('%Y-%m-%d %H:%M')
        
        start_cache_str = start_date_with_margin.strftime('%Y%m%d_%H%M')
        end_cache_str = end_date_with_margin.strftime('%Y%m%d_%H%M')
        cache_file = os.path.join(self.cache_dir, f"{self.cache_prefix}metar_{start_cache_str}_to_{end_cache_str}_routine_only.csv")
        
        if os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
            if self.verbose:
                print(f"Using cached METAR data from {cache_file}")
            with open(cache_file, 'r') as f:
                csv_data = f.read()
        else:
            form_url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
            
            form_data = {
                'station': station_str,
                'data': self.data_variables,
                'year1': start_date_with_margin.year,
                'month1': start_date_with_margin.month,
                'day1': start_date_with_margin.day,
                'hour1': start_date_with_margin.hour,
                'minute1': start_date_with_margin.minute,
                'year2': end_date_with_margin.year,
                'month2': end_date_with_margin.month,
                'day2': end_date_with_margin.day,
                'hour2': end_date_with_margin.hour,
                'minute2': end_date_with_margin.minute,
                'tz': 'Etc/UTC',
                'format': 'comma',
                'latlon': 'yes',
                'report_type': '3'
            }
            
            if self.verbose:
                print(f"Requesting METAR data for {len(stations)} stations from {start_str} to {end_str}")
            
            max_attempts = 6
            attempt = 0
            
            while attempt < max_attempts:
                try:
                    if self.verbose and attempt > 0:
                        print(f"Attempt {attempt+1}/{max_attempts} to fetch METAR data")
                        
                    response = requests.post(form_url, data=form_data, timeout=300)
                    
                    if response.status_code == 200:
                        csv_data = response.text
                        
                        if csv_data.startswith("#ERROR"):
                            if self.verbose:
                                print(f"Error from IEM API: {csv_data}")
                            attempt += 1
                            time.sleep(5)
                            continue
                        
                        with open(cache_file, 'w', encoding='utf-8') as f:
                            f.write(csv_data)
                        
                        if self.verbose:
                            print(f"Raw METAR data saved to {cache_file}")
                        
                        break
                    else:
                        if self.verbose:
                            print(f"Error fetching IEM METAR data: HTTP {response.status_code}")
                        attempt += 1
                        time.sleep(5)
                        continue
                        
                except Exception as e:
                    if self.verbose:
                        print(f"Exception when fetching IEM METAR data: {e}")
                        traceback.print_exc()
                    attempt += 1
                    time.sleep(5)
                    continue
            
            if attempt >= max_attempts:
                if self.verbose:
                    print("Exhausted attempts to download METAR data, returning empty data")
                return {station: [] for station in stations}
        
        full_date_range = pd.date_range(start=start_date_with_margin, end=end_date_with_margin, freq='h')
        
        try:
            df = pd.read_csv(
                io.StringIO(csv_data),
                comment='#',
                skip_blank_lines=True,
                na_values=['M', 'NA', ''],
                keep_default_na=True,
                on_bad_lines='warn'
            )
            
            if self.verbose:
                print(f"Successfully parsed CSV data with {len(df)} records")
            
            if len(df) == 0:
                if self.verbose:
                    print("No data found in CSV")
                return {station: [] for station in stations}
            
            numeric_cols = ['tmpf', 'dwpf', 'relh', 'drct', 'sknt', 'sped', 'p01i', 'alti', 'mslp', 'feel', 'vsby', 'gust']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            if 'valid' in df.columns:
                df['valid'] = pd.to_datetime(df['valid'])
            
            station_data = {}
            
            for station_id in stations:
                station_df = df[df['station'] == station_id].copy()
                
                if len(station_df) > 0:
                    station_df['timestep'] = station_df['valid'].dt.ceil('h')
                    station_df = station_df.drop_duplicates(subset=['timestep'], keep='last')
                    station_df = station_df.set_index('timestep')
                    station_df = station_df.reindex(full_date_range, method='nearest')
                    
                    for column in station_df.columns:
                        if column in numeric_cols:
                            missing_value = self._get_missing_value_for_variable(column)
                            station_df[column] = station_df[column].fillna(missing_value)
                        else:
                            station_df[column] = station_df[column].fillna(-1.0)
                    
                    records = []
                    for timestamp, row in station_df.iterrows():
                        record_dict = row.to_dict()
                        record_dict['valid'] = timestamp
                        records.append(record_dict)
                    
                    station_data[station_id] = records
                    
                    if self.verbose:
                        print(f"  Processed {len(station_data[station_id])} records for station {station_id}")
                else:
                    station_data[station_id] = []
                    
                    if self.verbose:
                        print(f"  No data available for station {station_id}")
            
            return station_data
            
        except Exception as e:
            if self.verbose:
                print(f"Error processing CSV data: {e}")
                traceback.print_exc()
            
            return {station: [] for station in stations}