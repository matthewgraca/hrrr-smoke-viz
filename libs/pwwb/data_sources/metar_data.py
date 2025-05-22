import os
import numpy as np
import pandas as pd
import requests
import io
import time
from urllib.parse import urlencode
import traceback

from libs.pwwb.data_sources.base_data_source import BaseDataSource
from libs.pwwb.utils.interpolation import preprocess_ground_sites, interpolate_frame

class MetarDataSource(BaseDataSource):
    """
    Class to handle METAR meteorological data collection and processing.
    
    Supports both original wind measurements (speed/direction) and derived U/V components.
    When users request U/V components, automatically fetches speed/direction data,
    calculates the components, but only includes the requested channels in output.
    
    Uses semantic missing data handling:
    - NaN for variables that can be negative (wind components, temperature in Celsius)
    - -1.0 for variables that should be positive (humidity, pressure, wind speed)
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
        channels=None
    ):
        """
        Initialize the METAR data source.
        
        Parameters:
        -----------
        timestamps : pandas.DatetimeIndex
            Timestamps for which to collect data
        extent : tuple
            Geographic bounds in format (min_lon, max_lon, min_lat, max_lat)
        dim : int
            Spatial resolution of the output grid
        cache_dir : str
            Directory to store cache files
        cache_prefix : str
            Prefix for cache filenames
        use_cached_data : bool
            Whether to use cached data if available
        verbose : bool
            Whether to print verbose output
        channels : list, optional
            List of specific METAR channels to include. Default is all channels.
        """
        super().__init__(timestamps, extent, dim, cache_dir, cache_prefix, use_cached_data, verbose)
        
        # Define all available channels - both raw and derived
        self.raw_channels = [
            'METAR_Wind_Speed', 'METAR_Wind_Direction', 'METAR_Precipitation', 
            'METAR_Humidity', 'METAR_Heat_Index', 'METAR_Air_Temp', 
            'METAR_Air_Pressure', 'METAR_Dew_Point'
        ]
        
        # Add derived channels
        self.derived_channels = ['METAR_Wind_U', 'METAR_Wind_V']
        
        # All available channels
        self.all_channels = self.raw_channels + self.derived_channels
        
        # Mapping between channels and data variables
        self.channel_mapping = {
            'METAR_Wind_Speed': 'sped',  # Using sped (mph) instead of sknt (knots)
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
        
        # Define variables that can legitimately be negative
        # Default assumption: most meteorological variables are positive-only
        self.variables_that_can_be_negative = {
            'u_component',      # Wind components can be negative
            'v_component',      
            'tmpf',             # Temperature in Fahrenheit can be negative
            'dwpf'              # Dew point in Fahrenheit can be negative
        }
        
        # Set the channels to include
        if channels is not None:
            # Start with user-requested channels (validate against all available channels)
            self.channels = [c for c in channels if c in self.all_channels]
        else:
            # Include all channels by default
            self.channels = self.all_channels.copy()
        
        # Check if we need wind component calculation
        self.need_wind_components = 'METAR_Wind_U' in self.channels or 'METAR_Wind_V' in self.channels
        
        # Prepare the list of raw data variables we need to fetch
        self.data_variables = []
        
        # Add variables for raw channels that were requested
        for channel in self.channels:
            if channel in self.raw_channels and channel in self.channel_mapping:
                self.data_variables.append(self.channel_mapping[channel])
        
        # Add variables needed for wind component calculation if needed
        if self.need_wind_components:
            sped_var = self.channel_mapping['METAR_Wind_Speed']
            drct_var = self.channel_mapping['METAR_Wind_Direction']
            
            # Ensure we fetch speed and direction if components are needed
            if sped_var not in self.data_variables:
                self.data_variables.append(sped_var)
            if drct_var not in self.data_variables:
                self.data_variables.append(drct_var)
        
        # Define sensor locations from documentation
        self.metar_sensors = [
            # Los Angeles area METAR stations
            {'id': 'LAX', 'name': 'Los Angeles Intl', 'lat': 33.9382, 'lon': -118.3865},
            {'id': 'BUR', 'name': 'Burbank/Glendale', 'lat': 34.2007, 'lon': -118.3587},
            {'id': 'LGB', 'name': 'Long Beach Airport', 'lat': 33.8118, 'lon': -118.1472},
            {'id': 'VNY', 'name': 'Van Nuys Airport', 'lat': 34.2097, 'lon': -118.4892},
            {'id': 'SMO', 'name': 'Santa Monica Muni', 'lat': 34.0210, 'lon': -118.4471},
            {'id': 'HHR', 'name': 'Hawthorne Municipal', 'lat': 33.9228, 'lon': -118.3352},
            {'id': 'EMT', 'name': 'El Monte', 'lat': 34.0860, 'lon': -118.0350},
            {'id': 'SNA', 'name': 'Santa Ana/John Wayne', 'lat': 33.6757, 'lon': -117.8682},
            {'id': 'ONT', 'name': 'Ontario Intl', 'lat': 34.0560, 'lon': -117.6012},
            {'id': 'PMD', 'name': 'Palmdale', 'lat': 34.6294, 'lon': -118.0846},
            {'id': 'WJF', 'name': 'Lancaster/Fox Field', 'lat': 34.7411, 'lon': -118.2186},
            {'id': 'CQT', 'name': 'Los Angeles Downtown/USC', 'lat': 34.0235, 'lon': -118.2912}
        ]
        
        if self.verbose:
            print(f"Initialized MetarDataSource with {len(self.channels)} channels: {self.channels}")
            print(f"Will fetch these raw variables: {self.data_variables}")
            if self.need_wind_components:
                print("Will calculate wind U/V components from speed/direction")
    
    def get_data(self):
        """
        Get meteorological data from Iowa State University METAR ASOS Dataset.
        
        This function fetches meteorological data from the Iowa Environmental Mesonet (IEM)
        ASOS/AWOS/METAR database, which provides hourly weather observations from airports
        around the world. For Los Angeles County, we use several key stations.
        
        Returns:
        --------
        numpy.ndarray
            METAR meteorological data with shape (n_timestamps, dim, dim, n_features)
        """
        # Generate channel-specific cache identifier
        channel_id = '_'.join([c.split('_')[-1].lower() for c in self.channels])
        cache_file = os.path.join(self.cache_dir, f"{self.cache_prefix}metar_{channel_id}_data.npy")
        
        # Count output channels - will match the requested channels
        output_channel_count = len(self.channels)
        
        # Check if cache exists and has correct shape
        data = self.check_cache(cache_file, (self.n_timestamps, self.dim, self.dim, output_channel_count))
        if data is not None:
            return data
        
        # Initialize empty array for METAR data with requested channels
        metar_data = np.zeros((self.n_timestamps, self.dim, self.dim, output_channel_count))
        
        if self.verbose:
            print(f"Including METAR variables: {self.data_variables}")
            print(f"Preparing data for channels: {self.channels}")
        
        # Filter stations to only those within our geographic bounds
        min_lon, max_lon, min_lat, max_lat = self.extent
        stations_in_bounds = [
            station for station in self.metar_sensors
            if min_lon <= station['lon'] <= max_lon and min_lat <= station['lat'] <= max_lat
        ]
        
        if not stations_in_bounds:
            if self.verbose:
                print("Warning: No METAR stations found within the specified geographic bounds!")
                print("Using stations closest to the bounds instead.")
            # Use all stations if none are in bounds
            stations_in_bounds = self.metar_sensors
        
        if self.verbose:
            print(f"Using {len(stations_in_bounds)} METAR stations for meteorological data:")
            for station in stations_in_bounds:
                print(f"  {station['id']} - {station['name']} ({station['lat']}, {station['lon']})")
        
        # Extract station IDs for API request
        station_ids = [station['id'] for station in stations_in_bounds]
        
        # Break the full date range into chunks to avoid too large requests
        # IEM recommends not requesting more than a few months at a time
        chunk_size = pd.Timedelta(days=90)  # 3 months chunks
        current_start = self.timestamps[0]
        current_end = self.timestamps[-1]
        
        # Dictionary to store all fetched data
        all_station_data = {}
        
        # Fetch data in chunks
        while current_start <= current_end:
            chunk_end = min(current_start + chunk_size, current_end)
            
            if self.verbose:
                print(f"Fetching METAR data chunk: {current_start.date()} to {chunk_end.date()}")
            
            # Fetch data for this chunk
            chunk_data = self._fetch_iem_metar_data(station_ids, current_start, chunk_end)
            
            # Merge with overall data
            for station_id, data in chunk_data.items():
                if station_id not in all_station_data:
                    all_station_data[station_id] = []
                all_station_data[station_id].extend(data)
            
            # Move to next chunk
            current_start = chunk_end + pd.Timedelta(seconds=1)
        
        if self.verbose:
            print("METAR data fetching complete")
            for station_id, data in all_station_data.items():
                print(f"  Station {station_id}: {len(data)} total records")
        
        # Create a full pandas DataFrame to match the script approach
        metar_df = self._create_metar_dataframe(all_station_data, stations_in_bounds)
        
        # Create a full date range for timestamps
        full_range = pd.date_range(start=self.timestamps[0], end=self.timestamps[-1], freq='h')
        
        # Clean and organize data by station and time
        station_names = list(metar_df.groupby("station").groups.keys())
        
        # Clean, impute, organize by time - similar to the script's approach
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
        
        # Concatenate all station dataframes
        df_by_stations = pd.concat(df_by_stations)
        
        sped_var = self.channel_mapping['METAR_Wind_Speed']
        drct_var = self.channel_mapping['METAR_Wind_Direction']
        
        if self.need_wind_components and sped_var in df_by_stations.columns and drct_var in df_by_stations.columns:
            wind_speed_mph = df_by_stations[sped_var]
            wind_direction = df_by_stations[drct_var]

            # BEFORE calculating U/V components, filter out invalid data
            # For positive-only variables, we use -1.0 as missing data marker
            valid_wind_mask = (wind_speed_mph != -1.0) & (wind_direction != -1.0)

            u_component = np.full_like(wind_speed_mph, np.nan, dtype=float)
            v_component = np.full_like(wind_speed_mph, np.nan, dtype=float)

            # Calculate only for valid data
            wind_dir_rad = np.radians(wind_direction)
            u_component[valid_wind_mask] = -wind_speed_mph[valid_wind_mask] * np.sin(wind_dir_rad[valid_wind_mask])
            v_component[valid_wind_mask] = -wind_speed_mph[valid_wind_mask] * np.cos(wind_dir_rad[valid_wind_mask])

            df_by_stations['u_component'] = u_component
            df_by_stations['v_component'] = v_component
            
            if self.verbose:
                print("Computed wind U/V components from speed/direction")
                print(f"U component range: {np.nanmin(u_component):.2f} to {np.nanmax(u_component):.2f}")
                print(f"V component range: {np.nanmin(v_component):.2f} to {np.nanmax(v_component):.2f}")
                print(f"Valid wind component count: {np.sum(valid_wind_mask)} / {len(valid_wind_mask)}")
        
        # Organize by time - get a DataFrame for each timestamp
        df_by_time = []
        for date in full_range:
            try:
                df_by_time.append(df_by_stations.loc[str(date)])
            except KeyError:
                # If no data for this timestamp, add an empty placeholder
                if self.verbose:
                    print(f"No data for timestamp {date}")
                df_by_time.append(pd.DataFrame())
        
        if self.verbose:
            print(f"Processing {len(df_by_time)} timestamps of METAR data")
        
        # Calculate geographic distances for interpolation
        latDist = abs(max_lat - min_lat)
        lonDist = abs(max_lon - min_lon)
        
        # Process each timestamp
        for t_idx, df in enumerate(df_by_time):
            if df.empty:
                continue
            
            # Process each requested channel
            for channel_idx, channel in enumerate(self.channels):
                if channel in self.channel_mapping:
                    var = self.channel_mapping[channel]
                    
                    try:
                        # Extract subset for this variable
                        subset_df = df[['lat', 'lon', var]].copy()
                        
                        # Determine if this variable can be negative
                        is_negative_allowed = var in self.variables_that_can_be_negative
                        
                        # Preprocess to place points on the grid
                        grid = preprocess_ground_sites(
                            subset_df, self.dim, max_lat, max_lon, latDist, lonDist,
                            allow_negative=is_negative_allowed
                        )
                        
                        # Interpolate to fill the grid
                        interpolated_grid = interpolate_frame(grid, self.dim)
                        
                        # Store in the metar_data array
                        metar_data[t_idx, :, :, channel_idx] = interpolated_grid
                        
                    except Exception as e:
                        if self.verbose:
                            print(f"Error processing channel {channel} for timestamp {t_idx}: {e}")
                            traceback.print_exc()
        
        if self.verbose:
            print(f"Created METAR data with shape {metar_data.shape}")
            print(f"Final channels: {self.channels}")
            
            # Print sample statistics for the first timestamp
            for i, channel in enumerate(self.channels):
                data = metar_data[0, :, :, i]
                print(f"{channel} sample stats: min={np.nanmin(data):.2f}, max={np.nanmax(data):.2f}, mean={np.nanmean(data):.2f}")
        
        # Save to cache
        self.save_to_cache(metar_data, cache_file)
        return metar_data
    
    def _get_missing_value_for_variable(self, variable_name):
        """
        Get the appropriate missing value marker for a variable.
        
        Parameters:
        -----------
        variable_name : str
            Name of the variable
            
        Returns:
        --------
        float
            Missing value marker (NaN for variables that can be negative, -1.0 for positive-only)
        """
        if variable_name in self.variables_that_can_be_negative:
            return np.nan
        else:
            return -1.0
    
    def _create_metar_dataframe(self, all_station_data, stations_in_bounds):
        """
        Convert station data dictionaries to a pandas DataFrame for processing.
        
        Parameters:
        -----------
        all_station_data : dict
            Dictionary with station IDs as keys and lists of data records as values
        stations_in_bounds : list
            List of station information dictionaries
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing all METAR data records
        """
        # Create a lookup dictionary for station info
        station_lookup = {station['id']: station for station in stations_in_bounds}
        
        # Flatten all records into a list for DataFrame creation
        all_records = []
        
        for station_id, records in all_station_data.items():
            # Get station info
            station_info = station_lookup.get(station_id)
            if not station_info:
                continue
                
            for record in records:
                try:
                    # Create a flattened record with all needed fields
                    flat_record = {
                        'station': station_id,
                        'name': station_info['name'],
                        'lat': station_info['lat'],
                        'lon': station_info['lon'],
                        'valid': record.get('valid', '')
                    }
                    
                    # Add the data variables with appropriate missing value markers
                    for field in self.data_variables:
                        missing_value = self._get_missing_value_for_variable(field)
                        flat_record[field] = record.get(field, missing_value)
                    
                    all_records.append(flat_record)
                except Exception as e:
                    if self.verbose:
                        print(f"Error flattening record from {station_id}: {e}")
        
        # Create DataFrame from all records
        if all_records:
            df = pd.DataFrame(all_records)
            # Ensure 'valid' is a datetime column
            df['valid'] = pd.to_datetime(df['valid'])
            return df
        else:
            # Return empty DataFrame with correct columns
            columns = ['station', 'name', 'lat', 'lon', 'valid'] + self.data_variables
            return pd.DataFrame(columns=columns)

    def _cleaned_station_df(self, df, station_name, full_range):
        """
        Takes a dataframe and a station name, groups it by that station, 
        organizes by the desired time range, and imputes with semantic missing values.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing all station data
        station_name : str
            Name of the station to process
        full_range : pandas.DatetimeIndex
            Full range of timestamps to include
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame of all the data of the given station, cleaned up
        """
        try:
            # Group the stations
            station_df = df.groupby("station").get_group(station_name).copy()
            
            # Round the timestamps to the nearest hour (ceiling)
            station_df['timestep'] = pd.to_datetime(station_df['valid']).dt.ceil('h')
            
            # Remove duplicate timestamps (keep last)
            station_df = station_df.drop_duplicates(subset=['timestep'], keep='last')
            
            # Reindex by timestamp to generate samples
            station_df = station_df.set_index('timestep', drop=True)
            station_df = station_df.reindex(full_range)
            
            # Impute with semantic missing values
            nan_date = "1900-01-01 00:00"
            cols_to_fill = ['station', 'lon', 'lat']
            
            # Impute non-data columns
            station_df['valid'] = station_df['valid'].fillna(nan_date)
            station_df[cols_to_fill] = station_df[cols_to_fill].bfill()
            
            # Impute data columns with appropriate missing value markers
            for column in station_df.columns:
                if column not in ['station', 'name', 'lat', 'lon', 'valid']:
                    missing_value = self._get_missing_value_for_variable(column)
                    station_df[column] = station_df[column].fillna(missing_value)
            
            return station_df
        
        except Exception as e:
            if self.verbose:
                print(f"Error cleaning station {station_name}: {e}")
                traceback.print_exc()
            # Return empty DataFrame with same index
            return pd.DataFrame(index=full_range)
    
    def _fetch_iem_metar_data(self, stations, start_date, end_date):
        """
        Fetch METAR data from the Iowa Environmental Mesonet (IEM) API.
        
        Parameters:
        -----------
        stations : list of str
            List of station IDs to fetch data for
        start_date : datetime
            Start date for data collection
        end_date : datetime
            End date for data collection
        
        Returns:
        --------
        dict
            Dictionary with station IDs as keys and lists of data records as values
        """
        if not stations:
            if self.verbose:
                print("No stations specified for IEM METAR data fetch")
            return {}
        
        # Convert station list to comma-separated string
        station_str = ",".join(stations)
        
        # Add a small safety margin to start_date to ensure complete coverage
        start_date_with_margin = start_date - pd.Timedelta(hours=1)
        end_date_with_margin = end_date + pd.Timedelta(hours=1)
        
        if self.verbose:
            print(f"Fetching METAR data from {start_date_with_margin} to {end_date_with_margin}")
        
        # Format dates for API request
        start_str = start_date_with_margin.strftime('%Y-%m-%d %H:%M')
        end_str = end_date_with_margin.strftime('%Y-%m-%d %H:%M')
        
        # Create a unique cache filename that includes the hour
        start_cache_str = start_date_with_margin.strftime('%Y%m%d_%H%M')
        end_cache_str = end_date_with_margin.strftime('%Y%m%d_%H%M')
        cache_file = os.path.join(self.cache_dir, f"{self.cache_prefix}metar_{start_cache_str}_to_{end_cache_str}_routine_only.csv")
        
        if os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
            if self.verbose:
                print(f"Using cached METAR data from {cache_file}")
            with open(cache_file, 'r') as f:
                csv_data = f.read()
        else:
            # Build form data for POST request to IEM's ASOS/AWOS data service
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
                'report_type': '3'  # Specifically requesting routine hourly observations only
            }
            
            if self.verbose:
                print(f"Requesting METAR data for {len(stations)} stations from {start_str} to {end_str}")
            
            # Use the robust download approach with multiple attempts
            max_attempts = 6
            attempt = 0
            
            while attempt < max_attempts:
                try:
                    if self.verbose and attempt > 0:
                        print(f"Attempt {attempt+1}/{max_attempts} to fetch METAR data")
                        
                    response = requests.post(form_url, data=form_data, timeout=300)  # 5-minute timeout
                    
                    if response.status_code == 200:
                        csv_data = response.text
                        
                        # Check if we got an error message
                        if csv_data.startswith("#ERROR"):
                            if self.verbose:
                                print(f"Error from IEM API: {csv_data}")
                            attempt += 1
                            time.sleep(5)  # Wait before retry
                            continue
                        
                        # Save the data to cache
                        with open(cache_file, 'w', encoding='utf-8') as f:
                            f.write(csv_data)
                        
                        if self.verbose:
                            print(f"Raw METAR data saved to {cache_file}")
                        
                        break  # Success, exit the retry loop
                    else:
                        if self.verbose:
                            print(f"Error fetching IEM METAR data: HTTP {response.status_code}")
                            print(f"Response text: {response.text[:200]}..." if response.text else "No response text")
                        attempt += 1
                        time.sleep(5)  # Wait before retry
                        continue
                        
                except Exception as e:
                    if self.verbose:
                        print(f"Exception when fetching IEM METAR data: {e}")
                        traceback.print_exc()
                    attempt += 1
                    time.sleep(5)  # Wait before retry
                    continue
            
            if attempt >= max_attempts:
                if self.verbose:
                    print("Exhausted attempts to download METAR data, returning empty data")
                return {station: [] for station in stations}  # Return empty lists for all stations
        
        # Create full date range for consistent data
        full_date_range = pd.date_range(start=start_date_with_margin, end=end_date_with_margin, freq='h')
        
        # Process the CSV data
        try:
            # Parse CSV using pandas
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
            
            # Handle empty dataframe
            if len(df) == 0:
                if self.verbose:
                    print("No data found in CSV")
                return {station: [] for station in stations}
            
            # Convert to numeric values
            numeric_cols = ['tmpf', 'dwpf', 'relh', 'drct', 'sknt', 'sped', 'p01i', 'alti', 'mslp', 'feel', 'vsby', 'gust']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert 'valid' column to datetime
            if 'valid' in df.columns:
                df['valid'] = pd.to_datetime(df['valid'])
            
            # Initialize station data
            station_data = {}
            
            # Process each station
            for station_id in stations:
                station_df = df[df['station'] == station_id].copy()
                
                if len(station_df) > 0:
                    # Round timestamps to nearest hour (ceiling is consistent with script)
                    station_df['timestep'] = station_df['valid'].dt.ceil('h')
                    
                    # Remove duplicate timestamps (keep last)
                    station_df = station_df.drop_duplicates(subset=['timestep'], keep='last')
                    
                    # Set timestep as index
                    station_df = station_df.set_index('timestep')
                    
                    # Reindex to include all hours in the date range
                    station_df = station_df.reindex(full_date_range, method='nearest')
                    
                    # Fill NaN values with appropriate missing value markers
                    for column in station_df.columns:
                        if column in numeric_cols:
                            missing_value = self._get_missing_value_for_variable(column)
                            station_df[column] = station_df[column].fillna(missing_value)
                        else:
                            # Non-numeric columns get generic -1.0
                            station_df[column] = station_df[column].fillna(-1.0)
                    
                    # Convert back to records
                    records = []
                    for timestamp, row in station_df.iterrows():
                        record_dict = row.to_dict()
                        record_dict['valid'] = timestamp  # Add the timestamp back
                        records.append(record_dict)
                    
                    station_data[station_id] = records
                    
                    if self.verbose:
                        print(f"  Processed {len(station_data[station_id])} records for station {station_id}")
                else:
                    # No data for this station - return empty list
                    station_data[station_id] = []
                    
                    if self.verbose:
                        print(f"  No data available for station {station_id}")
            
            return station_data
            
        except Exception as e:
            # Report the error and let it propagate up
            if self.verbose:
                print(f"Error processing CSV data: {e}")
                traceback.print_exc()
            
            # Return empty data instead of creating artificial values
            return {station: [] for station in stations}