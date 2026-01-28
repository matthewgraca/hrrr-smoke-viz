import requests
import json
import os
import sys
import pandas as pd
import numpy as np
import cv2
import time
from datetime import datetime
from tqdm import tqdm 
from libs.pwwb.utils.idw import IDW


class AirNowData:
    '''
    Gets the AirNow Data and processes it with IDW interpolation.
    Pipeline:
        - Downloads data from AirNow API in chunks to avoid record limits
        - Optionally filters sensors based on mask (excludes sensors outside valid areas)
        - Optionally filters sensors based on whitelist (includes only specified sensors)
        - Converts ground site data into grids
        - Interpolates using 3D IDW (with elevation)
    '''
    def __init__(
        self,
        start_date="2023-08-02-00",
        end_date="2025-08-02-00",
        extent=(-118.615, -117.70, 33.60, 34.35),
        airnow_api_key=None,
        save_dir='data/airnow.json',
        processed_cache_dir='data/airnow_processed.npz',
        dim=84,
        use_interpolation=True,
        idw_power=2,
        neighbors=10,
        elevation_path=None,
        elevation_scale_factor=50,
        mask_path=None,
        use_mask=False,
        sensor_whitelist=None,
        use_whitelist=False,
        force_reprocess=False,
        use_variable_blur=False,
        chunk_days=30,
        verbose=0,
        zscore_threshold=3,
        hard_cap=300,
    ):
        self.air_sens_loc = {}
        self.start_date = start_date
        self.end_date = end_date
        self.extent = extent
        self.dim = dim
        self.idw_power = idw_power
        self.use_mask = use_mask
        self.chunk_days = chunk_days
        self.verbose = verbose
        self.zscore_threshold = zscore_threshold
        self.hard_cap = hard_cap

        idw = IDW(
            power=idw_power,
            neighbors=neighbors,
            dim=dim,
            elevation_path=elevation_path,
            elevation_scale_factor=elevation_scale_factor,
            use_variable_blur=use_variable_blur,
            verbose=verbose
        )
        
        if not force_reprocess and os.path.exists(processed_cache_dir):
            if verbose < 1:
                print(
                    f"Loading processed AirNow data from cache: "
                    f"{processed_cache_dir}"
                )
            try:
                cached_data = np.load(processed_cache_dir, allow_pickle=True)
                self.data = cached_data['data']
                self.ground_site_grids = cached_data['ground_site_grids']
                
                air_sens_loc_array = cached_data['air_sens_loc']
                if isinstance(air_sens_loc_array, np.ndarray):
                    self.air_sens_loc = air_sens_loc_array.item() if air_sens_loc_array.size == 1 else {}
                
                if 'sensor_names' in cached_data:
                    self.sensor_names = cached_data['sensor_names'].tolist() if len(cached_data['sensor_names']) > 0 else list(self.air_sens_loc.keys())
                else:
                    self.sensor_names = list(self.air_sens_loc.keys())
                
                if verbose < 1:
                    print(
                        f"✓ Successfully loaded processed data from cache\n"
                        f"  - Data shape: {self.data.shape}\n"
                        f"  - Found {len(self.air_sens_loc)} sensor locations"
                    )
                    print(f"\n--- AirNow Data Summary ---")
                    print(f"Timesteps: {len(self.data)}")
                    print(f"Grid size: {self.dim}x{self.dim}")
                    print(f"Sensors ({len(self.sensor_names)}): {', '.join(sorted(self.sensor_names))}")
                    print(f"Value range: [{np.nanmin(self.data):.1f}, {np.nanmax(self.data):.1f}]")
                    print(f"---------------------------\n")
                return
            except Exception as e:
                print(f"Error: Couldn't load from cache: {e}. Will reprocess data.")

        self.use_whitelist = use_whitelist
        self.sensor_whitelist = sensor_whitelist if sensor_whitelist else []
        
        if use_whitelist and sensor_whitelist:
            if verbose < 1:
                print(
                    f"Using sensor whitelist: {len(self.sensor_whitelist)} "
                    f"sensors\n"
                    f"Whitelisted sensors: {self.sensor_whitelist}"
                )
        
        self.mask_path = mask_path if mask_path else "inputs/mask.npy"
        
        if use_mask and mask_path:
            os.makedirs(os.path.dirname(self.mask_path), exist_ok=True)
        os.makedirs(os.path.dirname(processed_cache_dir), exist_ok=True)
        
        self.mask = None
        if use_mask:
            if mask_path and os.path.exists(self.mask_path):
                self.mask = np.load(self.mask_path)
                if self.mask.shape != (dim, dim):
                    self.mask = cv2.resize(self.mask, (dim, dim))
                if verbose < 1: print(f"Using mask from {self.mask_path}")
            else:
                print(
                    f"Warning: Mask requested but not found at "
                    f"{self.mask_path}. Creating default mask (all valid)."
                )
                self.mask = np.ones((dim, dim), dtype=np.float32)
        else:
            if verbose < 1:
                print(
                    "Mask usage disabled. All sensors within extent "
                    "will be included."
                )

        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        self.sensor_names = []
        
        airnow_df = self._get_airnow_data(start_date, end_date, extent, save_dir, airnow_api_key)        

        if verbose < 1:
            print(
                "Removing sensors with low uptime, imputing "
                "invalid sensor data, imputing non-reporting sensors, imputing "
                "outliers, and fillings all gaps with forward/backward fill"
            )
        list_df = self._process_dataframe(airnow_df, start_date, end_date)
 
        if verbose < 1:
            print("Plotting sensor data onto grid...")

        ground_site_grids = []
        last_valid_grid = None
        forward_fill_count = 0

        for df in (tqdm(list_df) if verbose < 2 else list_df):
            grid = self._preprocess_ground_sites(df, dim, extent)
            
            if np.all(np.isnan(grid)):
                if last_valid_grid is not None:
                    forward_fill_count += 1
                    grid = last_valid_grid.copy()
            else:
                last_valid_grid = grid.copy()
            
            ground_site_grids.append(grid)

        if ground_site_grids:
            first_valid_idx = None
            for i, grid in enumerate(ground_site_grids):
                if not np.all(np.isnan(grid)):
                    first_valid_idx = i
                    break
            
            if first_valid_idx is not None and first_valid_idx > 0:
                if verbose < 2:
                    print(f"Backward-filling {first_valid_idx} leading NaN grids")
                for i in range(first_valid_idx):
                    ground_site_grids[i] = ground_site_grids[first_valid_idx].copy()

        if forward_fill_count > 0 and verbose < 2:
            print(f"Warning: Forward-filled {forward_fill_count} grids where all sensors were NaN")
        
        if not use_interpolation:
            if verbose < 1:
                print(
                    "Interpolation disabled. "
                    "Grids will be returned as-is with sensor data."
                )
            interpolated_grids = ground_site_grids
        else:
            if verbose < 1:
                print(
                    f"Performing IDW interpolation on "
                    f"{len(ground_site_grids)} frames..."
                )
            interpolated_grids = idw.interpolate_frames(ground_site_grids)        

        self.data = np.array(interpolated_grids)
        self.ground_site_grids = np.array(ground_site_grids)
        
        if self.air_sens_loc:
            self.sensor_names = list(self.air_sens_loc.keys())
        else:
            print("Warning: No air sensor locations found in the data.")
        
        if verbose < 1:
            print(f"\n--- AirNow Data Summary ---")
            print(f"Timesteps: {len(self.data)}")
            print(f"Grid size: {self.dim}x{self.dim}")
            print(f"Sensors ({len(self.sensor_names)}): {', '.join(sorted(self.sensor_names))}")
            print(f"Value range: [{np.nanmin(self.data):.1f}, {np.nanmax(self.data):.1f}]")
            print(f"---------------------------\n")
        
        if verbose < 1:
            print(
                f"Saving processed AirNow data to cache: {processed_cache_dir}"
            )
        try:
            air_sens_loc_array = np.array([self.air_sens_loc])
            sensor_names_array = np.array(self.sensor_names)
            
            np.savez_compressed(
                processed_cache_dir,
                data=self.data,
                ground_site_grids=np.array(self.ground_site_grids, dtype=object),
                air_sens_loc=air_sens_loc_array,
                sensor_names=sensor_names_array,
            )
            if verbose < 1:
                print("✓ Successfully saved processed data to cache")
        except Exception as e:
            print(f"Warning: Could not save processed data to cache: {e}")
    
    def _is_sensor_whitelisted(self, sensor_name):
        """Check if a sensor is in the whitelist."""
        if not self.use_whitelist:
            return True
            
        if not self.sensor_whitelist:
            return True
            
        return sensor_name in self.sensor_whitelist
    
    def _get_airnow_data(self, start_date, end_date, extent, save_dir, airnow_api_key):
        """Download or load AirNow data from API with chunking to avoid record limits."""
        lon_bottom, lon_top, lat_bottom, lat_top = extent
        end_date_adj = pd.to_datetime(end_date) - pd.Timedelta(hours=1)
        
        if os.path.exists(save_dir):
            if self.verbose < 1:
                print(f"Found existing file '{save_dir}'. Checking if download is complete...")
            try:
                with open(save_dir, 'r') as file:
                    existing_data = json.load(file)
                
                if not existing_data:
                    print("Existing file is empty. Starting fresh download...")
                else:
                    latest_dates = []
                    for record in existing_data[-100:]:
                        if 'UTC' in record:
                            try:
                                latest_dates.append(pd.to_datetime(record['UTC']))
                            except:
                                continue
                    
                    if latest_dates:
                        latest_date = max(latest_dates)
                        target_end = end_date_adj
                        
                        if self.verbose < 1:
                            print(f"Latest data in file: {latest_date}")
                            print(f"Target end date: {target_end}")
                        
                        if latest_date >= target_end:
                            if self.verbose < 1:
                                print("Download appears complete. Using existing file.")
                        else:
                            print(f"Download incomplete. Resuming from {latest_date}")
                            start_date = (latest_date + pd.Timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
                            print(f"Resuming download from: {start_date}")
                    else:
                        print("Could not determine latest date in file. Starting fresh...")
                        existing_data = []
            except Exception as e:
                print(f"Error reading existing file: {e}. Starting fresh download...")
                existing_data = []
        else:
            print("No existing file found. Starting fresh download...")
            existing_data = []
        
        if not os.path.exists(save_dir) or 'latest_date' in locals() and latest_date < end_date_adj:
            print("Downloading AirNow data in chunks to avoid record limits...")
            
            bbox = f'{lon_bottom},{lat_bottom},{lon_top},{lat_top}'
            URL = "https://www.airnowapi.org/aq/data"
            
            all_data = existing_data if 'existing_data' in locals() else []
            start_dt = pd.to_datetime(start_date)
            end_dt = end_date_adj
            current_start = start_dt
            chunk_days = self.chunk_days
            chunk_num = 0
            max_retries = 3
            
            print(f"Date range: {start_date} to {end_date}")
            
            while current_start < end_dt:
                current_end = min(current_start + pd.Timedelta(days=chunk_days), end_dt)
                
                if current_start >= current_end:
                    print(f"Reached end of date range at {current_start.strftime('%Y-%m-%d %H:%M')}")
                    break
                
                chunk_num += 1
                
                date_start = pd.to_datetime(current_start).isoformat()[:13]
                date_end = pd.to_datetime(current_end).isoformat()[:13]
                
                print(f"Chunk {chunk_num}: {date_start} to {date_end} ({chunk_days} days)")
                
                PARAMS = {
                    'startDate': date_start,
                    'endDate': date_end,
                    'parameters': 'PM25',
                    'BBOX': bbox,
                    'dataType': 'B',
                    'format': 'application/json',
                    'verbose': '1',
                    'monitorType': '2',
                    'includerawconcentrations': '1',
                    'API_KEY': airnow_api_key
                }

                retry_count = 0
                chunk_success = False
                
                while retry_count < max_retries and not chunk_success:
                    try:
                        print(f"  Requesting data from AirNow API (attempt {retry_count + 1})...")
                        response = requests.get(url=URL, params=PARAMS)
                        print(f"  Response: {response.status_code}")
                        
                        if response.status_code == 429:
                            wait_time = 60 + (retry_count * 30)
                            print(f"  Rate limited. Waiting {wait_time} seconds...")
                            time.sleep(wait_time)
                            retry_count += 1
                            continue
                        
                        if response.status_code != 200:
                            print(f"  Error: HTTP {response.status_code}")
                            print(f"  Response text: {response.text[:200]}")
                            retry_count += 1
                            time.sleep(5)
                            continue
                        
                        try:
                            chunk_data = response.json()
                        except json.JSONDecodeError as e:
                            print(f"  JSON decode error: {e}")
                            retry_count += 1
                            time.sleep(5)
                            continue
                        
                        if isinstance(chunk_data, list) and len(chunk_data) > 0 and isinstance(chunk_data[0], dict):
                            if 'WebServiceError' in chunk_data[0]:
                                error_msg = chunk_data[0]['WebServiceError'][0]['Message']
                                print(f"  API Error: {error_msg}")
                                
                                if "record query limit" in error_msg.lower():
                                    if chunk_days > 1:
                                        new_chunk_days = max(1, chunk_days // 2)
                                        print(f"  Reducing chunk size from {chunk_days} to {new_chunk_days} days")
                                        chunk_days = new_chunk_days
                                        break
                                    else:
                                        print("  Chunk size already at minimum (1 day). This chunk cannot be processed.")
                                        retry_count = max_retries
                                        break
                                else:
                                    retry_count += 1
                                    time.sleep(5)
                                    continue
                        
                        if isinstance(chunk_data, list):
                            if len(chunk_data) > 0:
                                try:
                                    existing_data = []
                                    if os.path.exists(save_dir):
                                        with open(save_dir, 'r') as file:
                                            existing_data = json.load(file)
                                    
                                    existing_data.extend(chunk_data)
                                    
                                    with open(save_dir, 'w') as file:
                                        json.dump(existing_data, file, indent=2)
                                    
                                    print(f"  ✓ Retrieved {len(chunk_data)} records")
                                    print(f"  ✓ Total records in file: {len(existing_data)}")
                                    
                                    del chunk_data
                                    
                                except Exception as e:
                                    print(f"  Error saving chunk data: {e}")
                                    retry_count += 1
                                    time.sleep(5)
                                    continue
                            else:
                                print(f"  No data for this period")
                            chunk_success = True
                        else:
                            print(f"  Unexpected response format: {type(chunk_data)}")
                            retry_count += 1
                            time.sleep(5)
                        
                    except requests.exceptions.Timeout:
                        print(f"  Request timeout (attempt {retry_count + 1})")
                        retry_count += 1
                        time.sleep(10)
                    except requests.exceptions.RequestException as e:
                        print(f"  Request error: {e} (attempt {retry_count + 1})")
                        retry_count += 1
                        time.sleep(10)
                    except Exception as e:
                        print(f"  Unexpected error: {e} (attempt {retry_count + 1})")
                        retry_count += 1
                        time.sleep(10)
                
                if not chunk_success:
                    if retry_count >= max_retries:
                        print(f"  Failed to retrieve chunk after {max_retries} attempts.")
                        if chunk_days < self.chunk_days:
                            print(f"  Retrying same time period with reduced chunk size ({chunk_days} days)")
                            continue
                        else:
                            print(f"  Cannot retrieve this chunk. Stopping data collection.")
                            break
                    else:
                        continue
                
                current_start = current_end + pd.Timedelta(hours=1)
                time.sleep(2)
                
                if chunk_num > 1000:
                    print("Safety limit reached (1000 chunks). Stopping.")
                    break
            
            try:
                with open(save_dir, 'r') as file:
                    final_data = json.load(file)
                print(f"✓ Complete dataset verified in '{save_dir}' ({len(final_data)} total records)")
            except Exception as e:
                print(f"Error verifying final file: {e}")
                return []

        try:
            if self.verbose < 1:
                print(f"Loading AirNow data from {save_dir}...")
            with open(save_dir, 'r') as file:
                airnow_data = json.load(file)
            
            if isinstance(airnow_data, list) and len(airnow_data) > 0 and isinstance(airnow_data[0], dict):
                if 'WebServiceError' in airnow_data[0]:
                    print(f"Error from AirNow API: {airnow_data[0]['WebServiceError']}")
                    return []
            
            if not airnow_data:
                print("Empty dataset loaded from file")
                return []
            
            airnow_df = pd.json_normalize(airnow_data)
            
            if airnow_df.empty:
                print("No data after JSON normalization")
                return []
            
            if 'UTC' not in airnow_df.columns:
                print("Error: 'UTC' column not found in AirNow data.")
                if 'DateObserved' in airnow_df.columns and 'HourObserved' in airnow_df.columns:
                    print("Attempting to construct UTC from DateObserved and HourObserved...")
                    try:
                        def construct_utc(row):
                            try:
                                date_part = pd.to_datetime(row['DateObserved']).date()
                                hour_part = int(float(row['HourObserved']))
                                dt = datetime.combine(date_part, datetime.min.time().replace(hour=hour_part))
                                return dt.strftime('%Y-%m-%dT%H:%M')
                            except Exception as e:
                                print(f"Error constructing UTC for row: {e}")
                                return None
                        
                        airnow_df['UTC'] = airnow_df.apply(construct_utc, axis=1)
                        airnow_df = airnow_df.dropna(subset=['UTC'])
                    except Exception as e:
                        print(f"Failed to construct UTC: {e}")
                        return []
                else:
                    print("Required columns for UTC construction not found.")
                    return []
            
            return airnow_df
                
        except Exception as e:
            print(f"Error processing AirNow data: {e}")
            return []

    def _process_dataframe(self, airnow_df, start_date, end_date, min_uptime=0.25):
        '''
        Performs several data cleaning methods, then returns a list of dataframes grouped by UTC.
        
        Outlier removal strategy:
        1. Per-sensor z-score filtering (removes values > zscore_threshold std from sensor mean)
        2. Hard cap backup (removes any remaining values > hard_cap)
        '''
        def remove_underreporting_sensors(df, min_uptime=0.25):
            timesteps = len(df.groupby('UTC').count())
            return df.groupby('FullAQSCode').filter(lambda x : len(x) / timesteps > min_uptime).copy()

        def impute_invalid_values_with_nan(df):
            df.loc[df['Value'] < 0, 'Value'] = np.nan
            return df

        def floor_low_values(df, floor=1.0):
            df.loc[(df['Value'] > 0) & (df['Value'] < floor), 'Value'] = floor
            return df

        def generate_samples_from_time(df, start_date, end_date):
            dates_df = pd.DataFrame({
                'UTC': pd.date_range(start_date, end_date, inclusive='left', freq='h')
            })
            df['UTC'] = pd.to_datetime(df['UTC'])
            cols_to_interpolate = [
                'Latitude', 'Longitude', 'Parameter', 'Unit', 
                'SiteName', 'AgencyName', 'FullAQSCode', 'IntlAQSCode'
            ]
            sensor_dfs = []
            for col in df['FullAQSCode'].unique():
                a = pd.merge(dates_df, df.loc[df['FullAQSCode'] == col], on='UTC', how='left')
                a[cols_to_interpolate] = a[cols_to_interpolate].ffill().bfill()
                sensor_dfs.append(a)
            return pd.concat(sensor_dfs, ignore_index=True)

        def remove_zscore_outliers(df, zscore_threshold):
            """Remove outliers using per-sensor z-score method."""
            df = df.copy()
            sensor_group = df.groupby('FullAQSCode')['Value']
            sensor_mean = sensor_group.transform('mean')
            sensor_std = sensor_group.transform('std')
            
            zscore = (df['Value'] - sensor_mean) / sensor_std
            
            outlier_mask = np.abs(zscore) > zscore_threshold
            n_outliers = outlier_mask.sum()
            
            if n_outliers > 0:
                df.loc[outlier_mask, 'Value'] = np.nan
                if self.verbose < 1:
                    print(f"  Z-score filtering: removed {n_outliers} outliers (>{zscore_threshold} std)")
            
            return df

        def apply_hard_cap(df, hard_cap):
            """Apply hard cap as backup - remove any values still above threshold."""
            df = df.copy()
            above_cap = df['Value'] > hard_cap
            n_capped = above_cap.sum()
            
            if n_capped > 0:
                df.loc[above_cap, 'Value'] = np.nan
                if self.verbose < 1:
                    print(f"  Hard cap filtering: removed {n_capped} values above {hard_cap}")
            
            return df

        def remove_sensors_out_of_extent(df, extent):
            min_lon, max_lon, min_lat, max_lat = extent
            out_of_extent = (
                (df['Latitude'] > max_lat) | 
                (df['Latitude'] < min_lat) | 
                (df['Longitude'] > max_lon) | 
                (df['Longitude'] < min_lon)
            )
            return df[~out_of_extent].reset_index(drop=True)

        original_data = airnow_df.copy()
        filtered_data = remove_underreporting_sensors(original_data, min_uptime)
        filtered_data = remove_sensors_out_of_extent(filtered_data, self.extent)
        filtered_data = impute_invalid_values_with_nan(filtered_data)
        filtered_data = floor_low_values(filtered_data)
        filtered_data = generate_samples_from_time(filtered_data, start_date, end_date)
        filtered_data = remove_zscore_outliers(filtered_data, self.zscore_threshold)
        filtered_data = apply_hard_cap(filtered_data, self.hard_cap)

        return [group for name, group in filtered_data.groupby('UTC')]

    def _preprocess_ground_sites(self, df, dim, extent):
        """Convert ground sites data into grid format, with optional whitelist filtering."""
        lonMin, lonMax, latMin, latMax = extent
        latDist, lonDist = abs(latMax - latMin), abs(lonMax - lonMin)
        unInter = np.full((dim, dim), np.nan)
        
        value_column = None
        if 'Value' in df.columns:
            value_column = 'Value'
        elif 'RawConcentration' in df.columns:
            value_column = 'RawConcentration'
        else:
            print(f"Error: No value column found. Available columns: {df.columns}")
            return unInter
        
        required_columns = ['Latitude', 'Longitude', value_column, 'SiteName']
        if not all(col in df.columns for col in required_columns):
            print(f"Warning: Missing required columns in dataframe. Available columns: {df.columns}")
            print(f"Required columns: {required_columns}")
            return unInter
            
        dfArr = np.array(df[required_columns])
        
        for i in range(dfArr.shape[0]):
            sitename = dfArr[i,3]
            
            if self.use_whitelist and not self._is_sensor_whitelisted(sitename):
                continue
            
            x = int(((latMax - dfArr[i,0]) / latDist) * dim)
            x = max(0, min(x, dim - 1))
            
            y = int(((dfArr[i,1] - lonMin) / lonDist) * dim)
            y = max(0, min(y, dim - 1))
            
            if self.use_mask and self.mask is not None:
                if (x < 0 or x >= self.mask.shape[0] or 
                    y < 0 or y >= self.mask.shape[1]):
                    continue
                elif self.mask[x, y] == 0:
                    continue
            
            value = dfArr[i,2]
            if not (pd.isna(value) or value < 0):
                unInter[x, y] = value
            
            self.air_sens_loc[sitename] = (x, y)
        
        return unInter

    def _grid_to_latlon(self, x, y):
        """Convert grid coordinates to latitude/longitude."""
        lon_min, lon_max, lat_min, lat_max = self.extent
        lat_dist = lat_max - lat_min
        lon_dist = lon_max - lon_min
        
        lat = lat_max - (x / self.dim) * lat_dist
        lon = lon_min + (y / self.dim) * lon_dist
        return lat, lon
    
    def _latlon_to_grid(self, lat, lon):
        """Convert latitude/longitude to grid coordinates."""
        lon_min, lon_max, lat_min, lat_max = self.extent
        lat_dist = lat_max - lat_min
        lon_dist = lon_max - lon_min
        
        x = int(((lat_max - lat) / lat_dist) * self.dim)
        y = int(((lon - lon_min) / lon_dist) * self.dim)
        
        x = max(0, min(x, self.dim - 1))
        y = max(0, min(y, self.dim - 1))
        return x, y

    def get_sensor_mask_status(self):
        """Get detailed information about sensor inclusion/exclusion by mask."""
        if not self.use_mask:
            return {"status": "Mask usage disabled", "all_sensors_included": True}
            
        if self.mask is None:
            return {"error": "Mask usage enabled but no mask available"}
        
        included_sensors = []
        for name, (x, y) in self.air_sens_loc.items():
            lat, lon = self._grid_to_latlon(x, y)
            included_sensors.append({
                'name': name,
                'grid_coords': (x, y),
                'lat_lon': (lat, lon),
                'mask_value': self.mask[x, y]
            })
        
        return {
            'mask_enabled': True,
            'included_sensors': included_sensors,
            'total_included': len(included_sensors),
            'mask_shape': self.mask.shape,
            'mask_coverage': f"{np.sum(self.mask == 1) / self.mask.size * 100:.1f}% of area is valid",
            'grid_extent': f"Grid covers x:[0-{self.mask.shape[0]-1}], y:[0-{self.mask.shape[1]-1}]"
        }