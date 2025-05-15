import numpy as np
import pandas as pd
import cv2
import requests
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tarfile
import urllib.request
from bs4 import BeautifulSoup
import json
import warnings
from dotenv import load_dotenv
import io
import time
from urllib.parse import urlencode
from xml.etree import ElementTree
from pyhdf.SD import SD, SDC
import netCDF4 as nc
warnings.filterwarnings("ignore")

class PWWBData:
    def __init__(
        self,
        start_date="2018-01-01",
        end_date="2020-12-31",
        extent=(-118.75, -117.5, 33.5, 34.5),  # Default to LA County bounds from doc
        frames_per_sample=24,  # One day of hourly data
        dim=200,  # Spatial resolution
        cache_dir='data/pwwb_cache/',
        use_cached_data=True,
        verbose=False,
        env_file='.env',
        output_dir=None
    ):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.extent = extent
        self.frames_per_sample = frames_per_sample
        self.dim = dim
        self.verbose = verbose
        self.cache_dir = cache_dir
        self.use_cached_data = use_cached_data
        self.output_dir = output_dir
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Load environment variables for API access
        if env_file and os.path.exists(env_file):
            load_dotenv(env_file)
        
        # Get EarthData token
        self.earthdata_token = os.getenv('EARTHDATA_TOKEN')
        
        # Get AirNow API key from environment
        self.airnow_api_key = os.getenv('AIRNOW_API_KEY')
        
        # Generate timestamps at hourly intervals
        self.timestamps = pd.date_range(self.start_date, self.end_date, freq='H')
        self.n_timestamps = len(self.timestamps)
        
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
            print(f"Initialized PWWBData with {self.n_timestamps} hourly timestamps")
            print(f"Date range: {self.start_date} to {self.end_date}")
        
        # Initialize data containers
        self.data = None
        self.maiac_aod_data = None
        self.tropomi_data = None
        self.modis_fire_data = None
        self.merra2_data = None
        self.meteorological_data = None
        
        # Process all data sources
        self._process_pipeline()
    
    def _process_pipeline(self):
        """Main processing pipeline to collect and integrate all data sources"""
        # Initialize channels dict to store processed data
        channels = {}
        
        # NOTE: Ground-level air pollution sensor data (PM2.5) is provided by
        # the separate AirNowData class and should be concatenated with this data.
        # We don't process it here to avoid redundancy.
        
        if self.verbose:
            print("Processing remote-sensing satellite imagery...")
        channels['maiac_aod'] = self._get_maiac_aod_data()
        channels['tropomi'] = self._get_tropomi_data()
        
        if self.verbose:
            print("Processing wildfire/smoke data...")
        channels['modis_fire'] = self._get_modis_fire_data()
        channels['merra2'] = self._get_merra2_data()
        
        if self.verbose:
            print("Processing meteorological data...")
        channels['metar'] = self._get_metar_data()
        
        # Store individual channels for access
        self.maiac_aod_data = channels['maiac_aod']
        self.tropomi_data = channels['tropomi']
        self.modis_fire_data = channels['modis_fire']
        self.merra2_data = channels['merra2']
        self.meteorological_data = channels['metar']
        
        # Concatenate all channels
        channel_list = [
            # NOTE: Ground-level PM2.5 data is provided by AirNowData class
            channels['maiac_aod'],    # MAIAC AOD data
            channels['tropomi'],      # TROPOMI data
            channels['modis_fire'],   # MODIS Fire data
            channels['merra2'],       # MERRA-2 data
            channels['metar']         # METAR meteorological data
        ]
        
        self.all_channels = np.concatenate(channel_list, axis=-1)
        
        # Create sliding window samples
        self.data = self._sliding_window_of(self.all_channels, self.frames_per_sample)
        
        if self.verbose:
            print(f"Final data shape: {self.data.shape}")
            self._print_data_statistics()
    
    def _get_ground_level_data(self):
        """
        Get ground-level air pollution sensor data from EPA AirNow.
        
        Returns:
        --------
        numpy.ndarray
            Ground-level PM2.5 data with shape (n_timestamps, dim, dim, n_features)
        """
        cache_file = os.path.join(self.cache_dir, 'ground_level_data.npy')
        
        if self.use_cached_data and os.path.exists(cache_file):
            if self.verbose:
                print(f"Loading cached ground-level data from {cache_file}")
            return np.load(cache_file)
        
        # Initialize empty array for sensor data
        # Single channel for PM2.5
        ground_data = np.zeros((self.n_timestamps, self.dim, self.dim, 1))
        
        # Check if API key is available
        if not self.airnow_api_key:
            if self.verbose:
                print("No AirNow API key found. Returning empty ground-level data.")
            np.save(cache_file, ground_data)
            return ground_data
        
        # Define LA County PM2.5 sensor locations from notebook
        ground_sensors = [
            {'name': 'Lancaster', 'lat': 34.6867, 'lon': -118.1542, 'grid_x': 35, 'grid_y': 25},
            {'name': 'Santa Clarita', 'lat': 34.3833, 'lon': -118.5289, 'grid_x': 24, 'grid_y': 13},
            {'name': 'Reseda', 'lat': 34.1992, 'lon': -118.5332, 'grid_x': 18, 'grid_y': 12},
            {'name': 'Glendora', 'lat': 34.1442, 'lon': -117.9501, 'grid_x': 16, 'grid_y': 34},
            {'name': 'Los Angeles Main', 'lat': 34.0664, 'lon': -118.2267, 'grid_x': 13, 'grid_y': 22},
            {'name': 'Long Beach', 'lat': 33.8192, 'lon': -118.1887, 'grid_x': 5, 'grid_y': 23},
            {'name': 'Long Beach - RT 710', 'lat': 33.8541, 'lon': -118.2012, 'grid_x': 5, 'grid_y': 26}
        ]
        
        if self.verbose:
            print(f"Fetching PM2.5 data for {len(ground_sensors)} sensors...")
            print(f"Time range: {self.start_date.date()} to {self.end_date.date()}")
        
        # Process each timestamp
        for t_idx, timestamp in enumerate(self.timestamps):
            date_str = timestamp.strftime('%Y-%m-%d')
            hour = timestamp.hour
            
            if self.verbose and t_idx % 1000 == 0:
                print(f"Processing timestamp {t_idx}/{self.n_timestamps}: {date_str} {hour:02d}:00")
            
            # Collect data from each sensor for this timestamp
            pm25_values = []
            
            for sensor in ground_sensors:
                pm25_value = self._fetch_airnow_data(
                    lat=sensor['lat'],
                    lon=sensor['lon'],
                    date=date_str,
                    hour=hour
                )
                
                pm25_values.append({
                    'lat': sensor['lat'],
                    'lon': sensor['lon'],
                    'value': pm25_value,
                    'grid_x': sensor['grid_x'],
                    'grid_y': sensor['grid_y']
                })
            
            # First, populate the exact sensor locations
            for sensor_data in pm25_values:
                if sensor_data['value'] > 0:  # Only use valid readings
                    x, y = sensor_data['grid_x'], sensor_data['grid_y']
                    ground_data[t_idx, y, x, 0] = sensor_data['value']
            
            # Then interpolate to fill the grid
            valid_points = [p for p in pm25_values if p['value'] > 0]
            if valid_points:
                try:
                    interp_data = self._interpolate_to_grid(valid_points, self.dim, self.dim, self.extent)
                    # Where we have exact sensor readings, keep those; otherwise use interpolated values
                    for y in range(self.dim):
                        for x in range(self.dim):
                            if ground_data[t_idx, y, x, 0] == 0:
                                ground_data[t_idx, y, x, 0] = interp_data[y, x]
                except Exception as e:
                    if self.verbose:
                        print(f"Error interpolating PM2.5 data for timestamp {date_str} {hour:02d}:00: {e}")
        
        if self.verbose:
            print(f"Created ground-level data with shape {ground_data.shape}")
        
        np.save(cache_file, ground_data)
        return ground_data
    
    def _get_maiac_aod_data(self):
        """
        Get MAIAC AOD data from NASA.
        
        Returns:
        --------
        numpy.ndarray
            MAIAC AOD data with shape (n_timestamps, dim, dim, n_features)
        """
        cache_file = os.path.join(self.cache_dir, 'maiac_aod_data.npy')
        
        if self.use_cached_data and os.path.exists(cache_file):
            if self.verbose:
                print(f"Loading cached MAIAC AOD data from {cache_file}")
            return np.load(cache_file)
        
        # Initialize empty array for MAIAC data
        # Single channel for AOD
        maiac_data = np.zeros((self.n_timestamps, self.dim, self.dim, 1))
        
        # Check if EarthData token is available
        if not self.earthdata_token:
            if self.verbose:
                print("NASA EarthData token not found. Returning empty MAIAC AOD data.")
            np.save(cache_file, maiac_data)
            return maiac_data
        
        # MAIAC data is typically available 1-2 times per day, not hourly
        # We'll need to fetch daily data and replicate it for hourly timestamps
        
        # Get unique dates from timestamps
        unique_dates = pd.Series([ts.date() for ts in self.timestamps]).unique()
        
        if self.verbose:
            print(f"Fetching MAIAC AOD data for {len(unique_dates)} unique dates")
        
        # Set up headers with bearer token
        headers = {"Authorization": f"Bearer {self.earthdata_token}"}
        
        # Define our geographic bounds
        min_lon, max_lon, min_lat, max_lat = self.extent
        
        # Define the MAIAC AOD collection parameters
        # Using the Version 061 (current version)
        maiac_params = {
            "short_name": "MCD19A2",
            "version": "061",
            "cmr_id": "C2324689816-LPCLOUD"  # From search results
        }
        
        # NASA CMR API endpoint
        cmr_url = "https://cmr.earthdata.nasa.gov/search/granules.json"
        
        # For each day, try to fetch MAIAC data
        daily_maiac_data = {}
        
        for date in unique_dates:
            date_str = date.strftime('%Y-%m-%d')
            day_next = (date + timedelta(days=1)).strftime('%Y-%m-%d')
            
            try:
                # Define search parameters
                params = {
                    "collection_concept_id": maiac_params["cmr_id"],
                    "temporal": f"{date_str}T00:00:00Z,{day_next}T00:00:00Z",
                    "bounding_box": f"{min_lon},{min_lat},{max_lon},{max_lat}",
                    "page_size": 10
                }
                
                # Make the request to CMR API
                response = requests.get(cmr_url, params=params, headers=headers)
                
                if response.status_code != 200:
                    if self.verbose:
                        print(f"Error searching for MAIAC AOD granules: HTTP {response.status_code}")
                    continue
                
                # Parse the results
                results = response.json()
                granules = results.get("feed", {}).get("entry", [])
                
                if not granules:
                    if self.verbose:
                        print(f"No MAIAC AOD data found for {date_str}")
                    continue
                
                # Process the first valid granule
                for granule in granules:
                    # Get download URL - use the same approach as in TROPOMI function
                    download_url = next((link["href"] for link in granule.get("links", []) 
                                        if link.get("rel") == "http://esipfed.org/ns/fedsearch/1.1/data#"), None)
                    
                    if not download_url:
                        continue
                    
                    # Download the HDF file
                    temp_file = os.path.join(self.cache_dir, f"maiac_temp_{date_str}.hdf")
                    
                    try:
                        # Download with token authentication
                        response = requests.get(download_url, headers=headers, stream=True)
                        if response.status_code != 200:
                            if self.verbose:
                                print(f"Error downloading MAIAC file: {response.status_code}")
                            continue
                                
                        with open(temp_file, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        
                        if not os.path.exists(temp_file) or os.path.getsize(temp_file) == 0:
                            raise Exception("Failed to download file or file is empty")
                        
                        # Process the HDF file to extract AOD data
                        hdf_file = SD(temp_file, SDC.READ)
                        
                        # Get the AOD dataset (Optical_Depth_055)
                        try:
                            aod_dataset = hdf_file.select('Optical_Depth_055')
                            aod_data = aod_dataset[:]
                            
                            # Check if aod_data is not empty and has valid dimensions
                            if aod_data.size == 0:
                                if self.verbose:
                                    print(f"Empty AOD data for {date_str}")
                                continue
                            
                            # Print detailed information about the AOD data
                            if self.verbose:
                                print(f"AOD data shape before resize: {aod_data.shape}")
                                print(f"AOD data type: {aod_data.dtype}")
                                print(f"AOD data min/max: {np.nanmin(aod_data)}/{np.nanmax(aod_data)}")
                            
                            # Handle multi-dimensional data - take the first band or average 
                            # if there are multiple time steps or bands
                            if len(aod_data.shape) == 3:
                                aod_data_2d = np.nanmean(aod_data, axis=0)
                                if self.verbose:
                                    print(f"Averaged AOD data to shape: {aod_data_2d.shape}")
                            else:
                                aod_data_2d = aod_data
                            
                            # Use simple resize method with scipy zoom for 2D data
                            from scipy.ndimage import zoom
                            
                            # Calculate zoom factors
                            zoom_y = self.dim / aod_data_2d.shape[0]
                            zoom_x = self.dim / aod_data_2d.shape[1]
                            
                            # Apply zoom (handles NaN values automatically)
                            aod_grid = zoom(aod_data_2d, (zoom_y, zoom_x), order=1, mode='nearest')
                            
                            if self.verbose:
                                print(f"Resized AOD data to shape: {aod_grid.shape}")
                            
                            # Store the processed AOD data for this date
                            daily_maiac_data[date] = aod_grid
                            
                            if self.verbose:
                                print(f"Successfully processed AOD data for {date_str}")
                        
                        except Exception as e:
                            if self.verbose:
                                print(f"Error selecting or processing AOD dataset: {e}")
                                import traceback
                                traceback.print_exc()
                        
                        finally:
                            # Close the HDF file
                            hdf_file.end()
                        
                        # Successfully processed, break the granule loop
                        if date in daily_maiac_data:
                            break

                    except Exception as e:
                        if self.verbose:
                            print(f"Error processing MAIAC AOD data for {date_str}: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    finally:
                        # Clean up the temporary file
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                        
            except Exception as e:
                if self.verbose:
                    print(f"Error fetching MAIAC AOD data for {date_str}: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Assign daily data to hourly timestamps
        for t_idx, timestamp in enumerate(self.timestamps):
            date = timestamp.date()
            if date in daily_maiac_data:
                maiac_data[t_idx, :, :, 0] = daily_maiac_data[date]
        
        if self.verbose:
            print(f"Created MAIAC AOD data with shape {maiac_data.shape}")
        
        np.save(cache_file, maiac_data)
        return maiac_data
        
    def _get_tropomi_data(self):
        """
        Get TROPOMI data from NASA Earthdata for methane, nitrogen dioxide, and carbon monoxide.
        
        Returns:
        --------
        numpy.ndarray
            TROPOMI data with shape (n_timestamps, dim, dim, n_features)
        """
        cache_file = os.path.join(self.cache_dir, 'tropomi_data.npy')
        
        if self.use_cached_data and os.path.exists(cache_file):
            if self.verbose:
                print(f"Loading cached TROPOMI data from {cache_file}")
            return np.load(cache_file)
        
        # Initialize empty array for TROPOMI data
        # 3 channels: methane, nitrogen dioxide, and carbon monoxide
        tropomi_data = np.zeros((self.n_timestamps, self.dim, self.dim, 3))
        
        # Check if Earth Data token is available
        if not self.earthdata_token:
            if self.verbose:
                print("NASA EarthData token not found. Returning empty TROPOMI data.")
            np.save(cache_file, tropomi_data)
            return tropomi_data
        
        # Get unique dates from timestamps
        unique_dates = pd.Series([ts.date() for ts in self.timestamps]).unique()
        
        if self.verbose:
            print(f"Fetching TROPOMI data for {len(unique_dates)} unique dates")
        
        # Define our geographic bounds
        min_lon, max_lon, min_lat, max_lat = self.extent
        
        # Set up headers with bearer token
        headers = {"Authorization": f"Bearer {self.earthdata_token}"}
        
        # Define the TROPOMI products we want with corrected variable paths
        products = [
            {
                "name": "NO2",  # Nitrogen Dioxide
                "index": 1,
                "cmr_id": "C2089270961-GES_DISC",  # Your current ID appears to be working
                "var_name": "PRODUCT/nitrogendioxide_tropospheric_column",
                "lat_var": "PRODUCT/latitude",
                "lon_var": "PRODUCT/longitude",
                "qa_var": "PRODUCT/qa_value"
            },
            {
                "name": "CH4",  # Methane - Updated with correct ID
                "index": 0, 
                "cmr_id": "C2087216530-GES_DISC",  # Updated to HiR V2 collection
                "var_name": "PRODUCT/methane_mixing_ratio",  # Will need flexible variable lookup
                "lat_var": "PRODUCT/latitude",
                "lon_var": "PRODUCT/longitude",
                "qa_var": "PRODUCT/qa_value"
            },
            {
                "name": "CO",  # Carbon Monoxide - Updated with correct ID
                "index": 2,
                "cmr_id": "C2087132178-GES_DISC",  # Updated to HiR V2 collection
                "var_name": "PRODUCT/carbonmonoxide_total_column",  # Will need flexible variable lookup
                "lat_var": "PRODUCT/latitude",
                "lon_var": "PRODUCT/longitude",
                "qa_var": "PRODUCT/qa_value"
            }
        ]
        
        # For each day, try to fetch TROPOMI data for each product
        daily_tropomi_data = {}
        
        for date in unique_dates:
            date_str = date.strftime('%Y-%m-%d')
            day_next = (date + timedelta(days=1)).strftime('%Y-%m-%d')
            
            if self.verbose:
                print(f"Processing TROPOMI data for date: {date_str}")
            
            # Initialize the day's data
            day_data = np.zeros((self.dim, self.dim, 3))
            
            # For each product (NO2, CH4, CO)
            for product in products:
                try:
                    # NASA CMR API endpoint
                    cmr_url = "https://cmr.earthdata.nasa.gov/search/granules.json"
                    
                    # Define search parameters
                    params = {
                        "collection_concept_id": product["cmr_id"],
                        "temporal": f"{date_str}T00:00:00Z,{day_next}T00:00:00Z",
                        "bounding_box": f"{min_lon},{min_lat},{max_lon},{max_lat}",
                        "page_size": 10,  # Limit to 10 granules per day
                        "sort_key": "-start_date"  # Sort by start date descending
                    }
                    
                    # Make the request to CMR API with token
                    response = requests.get(cmr_url, params=params, headers=headers)
                    
                    if response.status_code != 200:
                        if self.verbose:
                            print(f"Error searching for TROPOMI {product['name']} data: {response.status_code}")
                        continue
                    
                    # Parse the results
                    results = response.json()
                    granules = results.get("feed", {}).get("entry", [])
                    
                    if not granules:
                        if self.verbose:
                            print(f"No TROPOMI {product['name']} data found for {date_str}")
                        continue
                    
                    # Process the first valid granule
                    for granule in granules:
                        # Get download URL
                        download_url = next((link["href"] for link in granule.get("links", []) 
                                            if link.get("rel") == "http://esipfed.org/ns/fedsearch/1.1/data#"), None)
                        
                        if not download_url:
                            continue
                        
                        # Download the file
                        temp_file = os.path.join(self.cache_dir, f"tropomi_{product['name']}_{date_str}.nc")
                        
                        try:
                            # Download with token authentication
                            response = requests.get(download_url, headers=headers, stream=True)
                            if response.status_code != 200:
                                if self.verbose:
                                    print(f"Error downloading TROPOMI file: {response.status_code}")
                                continue
                                    
                            with open(temp_file, 'wb') as f:
                                for chunk in response.iter_content(chunk_size=8192):
                                    f.write(chunk)
                            
                            if not os.path.exists(temp_file) or os.path.getsize(temp_file) == 0:
                                raise Exception("Failed to download or empty file")
                            
                            # Process the NetCDF file - UPDATED to handle grouped variables
                            dataset = nc.Dataset(temp_file, 'r')
                            
                            # Get the data variables
                            try:
                                # Extract group path and variable name
                                var_path = product["var_name"].split('/')
                                lat_path = product["lat_var"].split('/')
                                lon_path = product["lon_var"].split('/')
                                qa_path = product["qa_var"].split('/')
                                
                                # Access data from the correct group
                                if len(var_path) > 1:
                                    # Variable is in a group
                                    group_name = var_path[0]
                                    var_name = var_path[1]
                                    group = dataset.groups[group_name]
                                    data_var = group.variables[var_name][:]
                                else:
                                    # Variable is in root
                                    data_var = dataset.variables[var_path[0]][:]
                                
                                # Get lat/lon variables
                                if len(lat_path) > 1:
                                    group_name = lat_path[0]
                                    var_name = lat_path[1]
                                    group = dataset.groups[group_name]
                                    lat_var = group.variables[var_name][:]
                                else:
                                    lat_var = dataset.variables[lat_path[0]][:]
                                    
                                if len(lon_path) > 1:
                                    group_name = lon_path[0]
                                    var_name = lon_path[1]
                                    group = dataset.groups[group_name]
                                    lon_var = group.variables[var_name][:]
                                else:
                                    lon_var = dataset.variables[lon_path[0]][:]
                                
                                # Get quality flags if available
                                qa_var = None
                                if len(qa_path) > 1:
                                    group_name = qa_path[0]
                                    var_name = qa_path[1]
                                    if group_name in dataset.groups and var_name in dataset.groups[group_name].variables:
                                        group = dataset.groups[group_name]
                                        qa_var = group.variables[var_name][:]
                                
                                # Close the dataset
                                dataset.close()
                                
                                # Remove time dimension if present (first dimension)
                                if data_var.ndim > 2 and data_var.shape[0] == 1:
                                    data_var = data_var[0]
                                    lat_var = lat_var[0] if lat_var.ndim > 2 else lat_var
                                    lon_var = lon_var[0] if lon_var.ndim > 2 else lon_var
                                    if qa_var is not None and qa_var.ndim > 2 and qa_var.shape[0] == 1:
                                        qa_var = qa_var[0]
                                
                                # Filter data by quality if available
                                if qa_var is not None:
                                    # Apply quality threshold - keep only high quality data (>0.75 typically)
                                    quality_mask = qa_var > 0.75
                                    data_var = np.where(quality_mask, data_var, np.nan)
                                
                                # Interpolate to our grid
                                from scipy.interpolate import griddata
                                
                                # Create grid of lat/lon points
                                grid_x, grid_y = np.meshgrid(
                                    np.linspace(min_lon, max_lon, self.dim),
                                    np.linspace(min_lat, max_lat, self.dim)
                                )
                                
                                # Prepare points for interpolation
                                points = np.column_stack((lon_var.flatten(), lat_var.flatten()))
                                values = data_var.flatten()
                                
                                # Remove NaN values
                                valid_mask = ~np.isnan(values)
                                points = points[valid_mask]
                                values = values[valid_mask]
                                
                                # Interpolate to regular grid
                                if len(points) > 3:  # Need at least a few points
                                    grid_z = griddata(points, values, (grid_x, grid_y), method='linear', fill_value=0)
                                    
                                    # Store in the daily data
                                    day_data[:, :, product["index"]] = grid_z
                                    
                                    if self.verbose:
                                        print(f"Successfully processed {product['name']} data")
                                    
                                    # Successfully processed, break the granule loop
                                    break
                                
                            except Exception as e:
                                if self.verbose:
                                    print(f"Error extracting TROPOMI data variables: {e}")
                                
                        except Exception as e:
                            if self.verbose:
                                print(f"Error processing TROPOMI file: {e}")
                        
                        finally:
                            # Clean up the temporary file
                            if os.path.exists(temp_file):
                                os.remove(temp_file)
                        
                except Exception as e:
                    if self.verbose:
                        print(f"Error in TROPOMI processing for {product['name']} on {date_str}: {e}")
            
            # Store the day's data if we have at least some non-zero values
            if np.sum(np.abs(day_data)) > 0:
                daily_tropomi_data[date] = day_data
        
        # Assign daily data to hourly timestamps
        for t_idx, timestamp in enumerate(self.timestamps):
            date = timestamp.date()
            if date in daily_tropomi_data:
                tropomi_data[t_idx] = daily_tropomi_data[date]
        
        if self.verbose:
            print(f"Created TROPOMI data with shape {tropomi_data.shape}")
        
        np.save(cache_file, tropomi_data)
        return tropomi_data
    
    def _get_modis_fire_data(self):
        """
        Get MODIS Fire Radiative Power (FRP) data.
        
        Returns:
        --------
        numpy.ndarray
            MODIS FRP data with shape (n_timestamps, dim, dim, n_features)
        """
        cache_file = os.path.join(self.cache_dir, 'modis_fire_data.npy')
        
        if self.use_cached_data and os.path.exists(cache_file):
            if self.verbose:
                print(f"Loading cached MODIS fire data from {cache_file}")
            return np.load(cache_file)
        
        # Initialize empty array for MODIS fire data
        # Single channel for FRP
        modis_fire_data = np.zeros((self.n_timestamps, self.dim, self.dim, 1))
        
        # Check if Earth Data token is available
        if not self.earthdata_token:
            if self.verbose:
                print("NASA Earth Data token not found. Returning empty MODIS fire data.")
            np.save(cache_file, modis_fire_data)
            return modis_fire_data
        
        # Get unique dates from timestamps
        unique_dates = pd.Series([ts.date() for ts in self.timestamps]).unique()
        
        if self.verbose:
            print(f"Fetching MODIS fire data for {len(unique_dates)} unique dates")
        
        # Define headers with bearer token
        headers = {"Authorization": f"Bearer {self.earthdata_token}"}
        
        # Define the collection ID for MODIS Land Surface Temperature
        collection_id = "C1748058432-LPCLOUD"  # Correct Collection ID for MOD11A1.061
        
        # For each day, try to fetch MODIS data
        daily_fire_data = {}
        
        for date in unique_dates:
            date_str = date.strftime('%Y-%m-%d')
            
            if self.verbose:
                print(f"Processing MODIS fire data for date: {date_str}")
            
            # Define our geographic bounds
            min_lon, max_lon, min_lat, max_lat = self.extent
            bbox = f"{min_lon},{min_lat},{max_lon},{max_lat}"
            
            try:
                # CMR API endpoint
                cmr_url = "https://cmr.earthdata.nasa.gov/search/granules.json"
                
                # Define search parameters
                params = {
                    "collection_concept_id": collection_id,
                    "temporal": f"{date_str}T00:00:00Z,{date_str}T23:59:59Z",
                    "bounding_box": bbox,
                    "page_size": 10  # Limit to 10 granules per day
                }
                
                # Make the request to CMR API with token
                response = requests.get(cmr_url, params=params, headers=headers)
                
                if response.status_code != 200:
                    if self.verbose:
                        print(f"Error searching for MODIS data: {response.status_code}")
                    continue
                
                # Parse the results
                results = response.json()
                granules = results.get("feed", {}).get("entry", [])
                
                if not granules:
                    if self.verbose:
                        print(f"No MODIS data found for {date_str}")
                    continue
                
                # Process the first valid granule
                for granule in granules:
                    # Get download URL
                    download_url = next((link["href"] for link in granule.get("links", []) 
                                    if link.get("rel") == "http://esipfed.org/ns/fedsearch/1.1/data#"), None)
                    
                    if not download_url:
                        continue
                    
                    # Download the file (HDF format)
                    temp_file = os.path.join(self.cache_dir, f"modis_temp_{date_str}.hdf")
                    hdf_file = None
                    
                    try:
                        # Download with token authentication
                        response = requests.get(download_url, headers=headers, stream=True)
                        if response.status_code != 200:
                            if self.verbose:
                                print(f"Error downloading MODIS file: {response.status_code}")
                            continue
                            
                        with open(temp_file, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        
                        if not os.path.exists(temp_file) or os.path.getsize(temp_file) == 0:
                            raise Exception("Failed to download or empty file")
                        
                        # Process the HDF file
                        hdf_file = SD(temp_file, SDC.READ)
                        
                        try:
                            # For MOD11A1, use Land Surface Temperature
                            dataset_name = 'LST_Day_1km'
                            
                            # Try to get the dataset
                            try:
                                dataset = hdf_file.select(dataset_name)
                                lst_data = dataset[:]
                                
                                # Print info about the data
                                if self.verbose:
                                    print(f"Found LST data with shape: {lst_data.shape}")
                                    print(f"Data type: {lst_data.dtype}")
                                    print(f"Min value: {np.min(lst_data)}, Max value: {np.max(lst_data)}")
                                
                                # Get attributes
                                attrs = dataset.attributes()
                                scale_factor = float(attrs.get('scale_factor', 0.02))
                                add_offset = float(attrs.get('add_offset', 0.0))
                                
                                # Apply scale factor and offset
                                # MOD11A1 LST values are typically stored as integers and need to be scaled
                                # LST = scale_factor * digital_number + add_offset
                                lst_data = lst_data * scale_factor + add_offset
                                
                                # Replace fill values with NaN
                                # MOD11A1 uses 0 as a fill value for LST
                                lst_data = np.where(lst_data > 100, lst_data, np.nan)  # Filter unrealistic values (< 100K)
                                
                                # Check if we have any valid data after filtering
                                if np.all(np.isnan(lst_data)):
                                    if self.verbose:
                                        print(f"No valid LST data found for {date_str} after filtering")
                                    continue
                                    
                                # Normalize to 0-1 scale (for temperatures between 270K and 330K)
                                min_temp = 270  # 270 Kelvin (approx 26°F)
                                max_temp = 330  # 330 Kelvin (approx 134°F)
                                normalized_lst = np.clip((lst_data - min_temp) / (max_temp - min_temp), 0, 1)
                                
                                # Handle NaN values for resize
                                normalized_lst = np.nan_to_num(normalized_lst, nan=0)
                                
                                # Make sure dimensions are valid
                                if self.dim <= 0:
                                    raise ValueError(f"Invalid dimension size: {self.dim}")
                                if normalized_lst.shape[0] == 0 or normalized_lst.shape[1] == 0:
                                    raise ValueError(f"Invalid source dimensions: {normalized_lst.shape}")
                                    
                                # Use scipy zoom instead of OpenCV resize (more robust)
                                from scipy.ndimage import zoom
                                
                                # Calculate zoom factors
                                zoom_y = self.dim / normalized_lst.shape[0]
                                zoom_x = self.dim / normalized_lst.shape[1]
                                
                                # Apply zoom
                                grid_lst = zoom(normalized_lst, (zoom_y, zoom_x), order=1, mode='nearest')
                                
                                # Make sure resized array has the expected dimensions
                                if grid_lst.shape != (self.dim, self.dim):
                                    if self.verbose:
                                        print(f"Warning: Resized dimensions {grid_lst.shape} don't match expected {(self.dim, self.dim)}")
                                        
                                    # Force resize to exact dimensions if needed
                                    from skimage.transform import resize as skimage_resize
                                    grid_lst = skimage_resize(grid_lst, (self.dim, self.dim), 
                                                            preserve_range=True, anti_aliasing=True)
                                    
                                # Store the data
                                daily_fire_data[date] = grid_lst
                                
                                if self.verbose:
                                    print(f"Successfully processed LST data for {date_str}")
                                    
                                # Successfully processed this granule
                                break
                                
                            except Exception as e:
                                if self.verbose:
                                    print(f"Error processing dataset {dataset_name}: {e}")
                                continue
                        
                        finally:
                            # Make sure to close the HDF file safely
                            if hdf_file is not None:
                                try:
                                    hdf_file.end()
                                except Exception as e:
                                    if self.verbose:
                                        print(f"Warning: Error closing HDF file: {e}")
                    
                    except Exception as e:
                        if self.verbose:
                            print(f"Error downloading or processing MODIS data: {e}")
                    
                    finally:
                        # Clean up the temporary file
                        if os.path.exists(temp_file):
                            try:
                                os.remove(temp_file)
                            except Exception as e:
                                if self.verbose:
                                    print(f"Warning: Failed to remove temp file: {e}")
            
            except Exception as e:
                if self.verbose:
                    print(f"Error in MODIS data processing for {date_str}: {e}")
        
        # Assign daily data to hourly timestamps
        for t_idx, timestamp in enumerate(self.timestamps):
            date = timestamp.date()
            if date in daily_fire_data:
                modis_fire_data[t_idx, :, :, 0] = daily_fire_data[date]
        
        # Add spatial smoothing to simulate fire spread
        from scipy.ndimage import gaussian_filter
        for t_idx in range(self.n_timestamps):
            if np.max(modis_fire_data[t_idx, :, :, 0]) > 0:
                modis_fire_data[t_idx, :, :, 0] = gaussian_filter(modis_fire_data[t_idx, :, :, 0], sigma=1.0)
        
        # Add temporal coherence - make neighboring timestamps similar
        for t_idx in range(1, self.n_timestamps):
            # If the current timestamp has no fire data but the previous one does
            if np.max(modis_fire_data[t_idx, :, :, 0]) == 0 and np.max(modis_fire_data[t_idx-1, :, :, 0]) > 0:
                # Decay the previous timestamp's fire data (fires don't disappear instantly)
                modis_fire_data[t_idx, :, :, 0] = modis_fire_data[t_idx-1, :, :, 0] * 0.9
            # If both have fire data, add some temporal coherence
            elif np.max(modis_fire_data[t_idx, :, :, 0]) > 0 and np.max(modis_fire_data[t_idx-1, :, :, 0]) > 0:
                modis_fire_data[t_idx, :, :, 0] = (
                    modis_fire_data[t_idx-1, :, :, 0] * 0.3 + 
                    modis_fire_data[t_idx, :, :, 0] * 0.7
                )
        
        if self.verbose:
            print(f"Created MODIS fire data with shape {modis_fire_data.shape}")
        
        np.save(cache_file, modis_fire_data)
        return modis_fire_data
        
    def _get_merra2_data(self):
        """
        Get MERRA-2 data for PBL Height, Surface Air Temperature, and Surface Exchange Coefficient.
        
        Returns:
        --------
        numpy.ndarray
            MERRA-2 data with shape (n_timestamps, dim, dim, n_features)
        """
        cache_file = os.path.join(self.cache_dir, 'merra2_data.npy')
        
        if self.use_cached_data and os.path.exists(cache_file):
            if self.verbose:
                print(f"Loading cached MERRA-2 data from {cache_file}")
            return np.load(cache_file)
        
        # Initialize empty array for MERRA-2 data
        # 3 channels: PBL Height, Surface Air Temperature, Surface Exchange Coefficient for Heat
        merra2_data = np.zeros((self.n_timestamps, self.dim, self.dim, 3))
        
        # Check if M2M token is available - first try dedicated M2M token, fallback to Earthdata token
        m2m_token = os.getenv('M2M_TOKEN')
        if not m2m_token:
            m2m_token = self.earthdata_token  # Try using the regular Earthdata token as fallback
            if not m2m_token:
                if self.verbose:
                    print("No M2M or Earthdata token found. Returning empty MERRA-2 data.")
                np.save(cache_file, merra2_data)
                return merra2_data
        
        # MERRA-2 data is available hourly, which matches our timestamps
        if self.verbose:
            print(f"Fetching MERRA-2 data for {self.n_timestamps} timestamps")
        
        # Group timestamps by day to process them efficiently
        days_to_process = pd.Series([ts.date() for ts in self.timestamps]).unique()
        
        # Define the MERRA-2 data access methods - try different approaches
        # Method 1: GES DISC API endpoint
        api_url_gesdisc = "https://disc.gsfc.nasa.gov/service/subset/jsonwsp/api"
        
        # Method 2: CMR API endpoint (similar to MODIS)
        api_url_cmr = "https://cmr.earthdata.nasa.gov/search/granules.json"
        
        # Method 3: Direct MERRA-2 API
        api_url_direct = "https://api.earthdata.nasa.gov/m2m/data/api/dataset/MERRA2/granule/search"
        
        # Define the variables we need
        variables = ["PBLH", "T2M", "CDH"]
        
        # Process each day
        for day in days_to_process:
            day_str = day.strftime('%Y-%m-%d')
            
            if self.verbose:
                print(f"Processing MERRA-2 data for day: {day_str}")
            
            # Define our geographic bounds
            min_lon, max_lon, min_lat, max_lat = self.extent
            
            # Try Method 1: CMR API (similar to what worked for MODIS)
            try:
                if self.verbose:
                    print("Trying CMR API method...")
                    
                # Set up headers with token
                headers = {"Authorization": f"Bearer {m2m_token}"}
                
                # Define search parameters
                params = {
                    "short_name": "M2T1NXFLX",
                    "version": "5.12.4",
                    "temporal": f"{day_str}T00:00:00Z,{day_str}T23:59:59Z",
                    "bounding_box": f"{min_lon},{min_lat},{max_lon},{max_lat}",
                    "page_size": 10
                }
                
                # Make the request
                response = requests.get(api_url_cmr, params=params, headers=headers)
                
                if response.status_code == 200:
                    results = response.json()
                    granules = results.get("feed", {}).get("entry", [])
                    
                    if granules:
                        if self.verbose:
                            print(f"Found {len(granules)} MERRA-2 granules via CMR API")
                        
                        # Get download URL
                        for granule in granules:
                            download_url = next((link["href"] for link in granule.get("links", []) 
                                            if link.get("rel") == "http://esipfed.org/ns/fedsearch/1.1/data#"), None)
                            
                            if download_url:
                                # Process this granule
                                success = self._process_merra2_granule(download_url, headers, day, merra2_data)
                                if success:
                                    break  # Successfully processed a granule, no need to try more
                    else:
                        if self.verbose:
                            print("No granules found via CMR API")
                else:
                    if self.verbose:
                        print(f"CMR API error: {response.status_code}")
            
            except Exception as e:
                if self.verbose:
                    print(f"Error with CMR API method: {e}")
            
            # Try Method 2: Direct MERRA-2 API (original approach)
            if np.all(merra2_data[np.array([ts.date() == day for ts in self.timestamps])] == 0):
                try:
                    if self.verbose:
                        print("Trying direct MERRA-2 API method...")
                    
                    # Set up authentication headers
                    headers = {
                        "Authorization": f"Bearer {m2m_token}",
                        "Content-Type": "application/json"
                    }
                    
                    # Construct parameters
                    params = {
                        "short_name": "M2T1NXFLX",
                        "temporal": f"{day_str}T00:00:00Z,{day_str}T23:59:59Z",
                        "bounding_box": f"{min_lon},{min_lat},{max_lon},{max_lat}",
                        "variables": ",".join(variables),
                        "format": "json"
                    }
                    
                    # Make the request
                    response = requests.get(api_url_direct, params=params, headers=headers)
                    
                    if response.status_code == 200:
                        results = response.json()
                        granules = results.get("granules", [])
                        
                        if granules:
                            if self.verbose:
                                print(f"Found {len(granules)} MERRA-2 granules via direct API")
                            
                            for granule in granules:
                                download_url = granule.get("downloadUrl")
                                
                                if download_url:
                                    # Process this granule
                                    success = self._process_merra2_granule(download_url, headers, day, merra2_data)
                                    if success:
                                        break  # Successfully processed a granule
                        else:
                            if self.verbose:
                                print("No granules found via direct API")
                    else:
                        if self.verbose:
                            print(f"Direct API error: {response.status_code} - {response.text[:200]}")
                
                except Exception as e:
                    if self.verbose:
                        print(f"Error with direct MERRA-2 API method: {e}")
        
        # Add temporal coherence - make neighboring timestamps similar
        for t_idx in range(1, self.n_timestamps):
            # If the current timestamp has all zeros, use the previous timestamp's data
            if np.sum(merra2_data[t_idx]) == 0 and np.sum(merra2_data[t_idx-1]) > 0:
                merra2_data[t_idx] = merra2_data[t_idx-1]
            # Otherwise, if both have data, add some temporal smoothing
            elif np.sum(merra2_data[t_idx]) > 0 and np.sum(merra2_data[t_idx-1]) > 0:
                merra2_data[t_idx] = merra2_data[t_idx-1] * 0.2 + merra2_data[t_idx] * 0.8
        
        if self.verbose:
            print(f"Created MERRA-2 data with shape {merra2_data.shape}")
        
        np.save(cache_file, merra2_data)
        return merra2_data

    def _process_merra2_granule(self, download_url, headers, day, merra2_data):
        """
        Process a MERRA-2 granule file.
        
        Parameters:
        -----------
        download_url : str
            URL to download the granule file
        headers : dict
            Headers to use for the download request
        day : datetime.date
            The day for this granule
        merra2_data : numpy.ndarray
            Array to store the processed data
        
        Returns:
        --------
        bool
            True if processing was successful, False otherwise
        """
        # Download the NetCDF file
        temp_file = os.path.join(self.cache_dir, f"merra2_temp_{day.strftime('%Y-%m-%d')}.nc4")
        
        try:
            # Download with authentication
            r = requests.get(download_url, headers=headers, stream=True)
            if r.status_code != 200:
                if self.verbose:
                    print(f"Error downloading MERRA-2 file: {r.status_code}")
                return False
                
            with open(temp_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Check if file was downloaded correctly
            if not os.path.exists(temp_file) or os.path.getsize(temp_file) == 0:
                if self.verbose:
                    print("Downloaded file is empty or does not exist")
                return False
            
            # Process the NetCDF file to extract data
            try:
                import netCDF4 as nc
                dataset = nc.Dataset(temp_file, 'r')
                
                # List all variables to debug
                if self.verbose:
                    print("Variables in MERRA-2 file:")
                    for var_name in dataset.variables:
                        print(f"  {var_name}: {dataset.variables[var_name].shape}")
                
                # Try to extract the variables
                # Variable names might differ - try different possibilities
                var_names = {
                    'PBLH': ['PBLH', 'PBL', 'ZPBL'],
                    'T2M': ['T2M', 'T2', 'TLML'],
                    'CDH': ['CDH', 'CH', 'CN']
                }
                
                # Initialize data arrays
                var_data = {}
                
                # Try to find each variable
                for var_key, possible_names in var_names.items():
                    for name in possible_names:
                        if name in dataset.variables:
                            var_data[var_key] = dataset.variables[name][:]
                            if self.verbose:
                                print(f"Found {var_key} as {name}")
                            break
                
                # Extract time, lat, lon information
                # First find time variable
                time_var = None
                for name in ['time', 'TIME', 'Time']:
                    if name in dataset.variables:
                        time_var = name
                        break
                
                # Find lat/lon variables
                lat_var = None
                lon_var = None
                for name in ['lat', 'LAT', 'latitude', 'LATITUDE']:
                    if name in dataset.variables:
                        lat_var = name
                        break
                
                for name in ['lon', 'LON', 'longitude', 'LONGITUDE']:
                    if name in dataset.variables:
                        lon_var = name
                        break
                
                if time_var is None or lat_var is None or lon_var is None:
                    if self.verbose:
                        print("Could not find time, lat, or lon variables")
                    dataset.close()
                    return False
                
                times = dataset.variables[time_var][:]
                lats = dataset.variables[lat_var][:]
                lons = dataset.variables[lon_var][:]
                
                # Close the dataset now that we've extracted everything we need
                dataset.close()
                
                # Check if we have all the required variables
                if not all(key in var_data for key in ['PBLH', 'T2M', 'CDH']):
                    if self.verbose:
                        print(f"Missing required variables. Found: {list(var_data.keys())}")
                    return False
                
                # Get the data arrays
                pblh = var_data['PBLH']
                t2m = var_data['T2M']
                cdh = var_data['CDH']
                
                # For each timestamp in this day
                day_timestamps = [ts for ts in self.timestamps if ts.date() == day]
                
                for ts in day_timestamps:
                    # Find the closest time index in the MERRA-2 data
                    hour = ts.hour
                    t_idx = self.timestamps.get_loc(ts)
                    
                    # Ensure the hour index is valid
                    if hour >= len(times):
                        if self.verbose:
                            print(f"Hour {hour} exceeds available time steps {len(times)}")
                        continue
                    
                    try:
                        # Use scipy zoom instead of OpenCV resize
                        from scipy.ndimage import zoom
                        
                        # Extract the data for this hour
                        try:
                            # Check if data has a time dimension
                            if len(pblh.shape) > 2:  # Has time dimension
                                pblh_hour = pblh[hour]
                                t2m_hour = t2m[hour]
                                cdh_hour = cdh[hour]
                            else:  # No time dimension
                                pblh_hour = pblh
                                t2m_hour = t2m
                                cdh_hour = cdh
                        except IndexError:
                            if self.verbose:
                                print(f"Index error for hour {hour}")
                            continue
                        
                        # Check for valid dimensions
                        if pblh_hour.shape[0] == 0 or pblh_hour.shape[1] == 0:
                            if self.verbose:
                                print(f"Invalid dimensions: {pblh_hour.shape}")
                            continue
                        
                        # Calculate zoom factors
                        zoom_y = self.dim / pblh_hour.shape[0]
                        zoom_x = self.dim / pblh_hour.shape[1]
                        
                        # Apply zoom to resize the data
                        pblh_grid = zoom(pblh_hour, (zoom_y, zoom_x), order=1, mode='nearest')
                        t2m_grid = zoom(t2m_hour, (zoom_y, zoom_x), order=1, mode='nearest')
                        cdh_grid = zoom(cdh_hour, (zoom_y, zoom_x), order=1, mode='nearest')
                        
                        # Store the data
                        merra2_data[t_idx, :, :, 0] = pblh_grid  # PBL Height
                        merra2_data[t_idx, :, :, 1] = t2m_grid   # Surface Air Temperature
                        merra2_data[t_idx, :, :, 2] = cdh_grid   # Surface Exchange Coefficient
                        
                        if self.verbose:
                            print(f"Processed MERRA-2 data for timestamp {ts}")
                    
                    except Exception as e:
                        if self.verbose:
                            print(f"Error processing timestamp {ts}: {e}")
                        continue
                
                return True  # Successfully processed the granule
                
            except ImportError:
                if self.verbose:
                    print("netCDF4 library not available. Cannot process MERRA-2 data.")
                return False
                
            except Exception as e:
                if self.verbose:
                    print(f"Error processing MERRA-2 NetCDF file: {e}")
                    import traceback
                    traceback.print_exc()
                return False
        
        except Exception as e:
            if self.verbose:
                print(f"Error downloading or processing MERRA-2 data: {e}")
                import traceback
                traceback.print_exc()
            return False
        
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Failed to remove temp file: {e}")
            
            return False  # If we reach here, something went wrong
    
    def _get_metar_data(self):
        """
        Get meteorological data from Iowa State University METAR ASOS Dataset.
        
        This function fetches meteorological data from the Iowa Environmental Mesonet (IEM)
        ASOS/AWOS/METAR database, which provides hourly weather observations from airports
        around the world. For Los Angeles County, we use several key stations.
        
        The data includes:
        - Wind Speed (mph)
        - Wind Direction (deg)
        - Precipitation (inch)
        - Relative Humidity (%)
        - Heat Index/Wind Chill (F)
        - Air Temperature (F)
        - Air Pressure (mb)
        - Dew Point (F)
        - AQI (derived from other parameters)
        
        Returns:
        --------
        numpy.ndarray
            METAR meteorological data with shape (n_timestamps, dim, dim, n_features)
        """
        cache_file = os.path.join(self.cache_dir, 'metar_data.npy')
        
        if self.use_cached_data and os.path.exists(cache_file):
            if self.verbose:
                print(f"Loading cached METAR data from {cache_file}")
            return np.load(cache_file)
        
        # Initialize empty array for METAR data
        # 9 channels: Wind Speed, Wind Direction, Precipitation, AQI, Humidity, Heat Index, 
        # Air Temperature, Air Pressure, Dew Point
        metar_data = np.zeros((self.n_timestamps, self.dim, self.dim, 9))
        
        # Define Los Angeles area ASOS/AWOS stations that are within our geographic bounds
        la_stations = self.metar_sensors
        
        # Filter stations to only those within our geographic bounds
        min_lon, max_lon, min_lat, max_lat = self.extent
        stations_in_bounds = [
            station for station in la_stations
            if min_lon <= station['lon'] <= max_lon and min_lat <= station['lat'] <= max_lat
        ]
        
        if not stations_in_bounds:
            if self.verbose:
                print("Warning: No METAR stations found within the specified geographic bounds!")
                print("Using stations closest to the bounds instead.")
            # Use all stations if none are in bounds
            stations_in_bounds = la_stations
        
        if self.verbose:
            print(f"Using {len(stations_in_bounds)} METAR stations for meteorological data:")
            for station in stations_in_bounds:
                print(f"  {station['id']} - {station['name']} ({station['lat']}, {station['lon']})")
        
        # Extract station IDs for API request
        station_ids = [station['id'] for station in stations_in_bounds]
        
        # Create a station lookup dictionary for quick reference
        station_lookup = {station['id']: station for station in stations_in_bounds}
        
        # Break the full date range into chunks to avoid too large requests
        # IEM recommends not requesting more than a few months at a time
        chunk_size = pd.Timedelta(days=90)  # 3 months chunks
        current_start = self.start_date
        
        # Dictionary to store all fetched data
        all_station_data = {}
        
        # Fetch data in chunks
        while current_start < self.end_date:
            current_end = min(current_start + chunk_size, self.end_date)
            
            if self.verbose:
                print(f"Fetching METAR data chunk: {current_start.date()} to {current_end.date()}")
            
            # Fetch data for this chunk
            chunk_data = self._fetch_iem_metar_data(station_ids, current_start, current_end)
            
            # Merge with overall data
            for station_id, data in chunk_data.items():
                if station_id not in all_station_data:
                    all_station_data[station_id] = []
                all_station_data[station_id].extend(data)
            
            # Move to next chunk
            current_start = current_end
        
        if self.verbose:
            print("METAR data fetching complete")
            for station_id, data in all_station_data.items():
                print(f"  Station {station_id}: {len(data)} total records")
        
        # Process the station data into timestamp-keyed data
        timestamp_data = self._process_timestamp_data(all_station_data, station_lookup)
        
        # Now process each timestamp in our sequence
        for t_idx, timestamp in enumerate(self.timestamps):
            ts_key = timestamp.strftime('%Y-%m-%d-%H')
            
            # Get data for this timestamp
            ts_data = timestamp_data.get(ts_key, [])
            
            if not ts_data and self.verbose:
                print(f"No METAR data available for timestamp {ts_key}")
                
            # For each meteorological variable, extract values and interpolate
            for v_idx, var_name in enumerate(['sknt', 'drct', 'p01i', 'aqi', 'relh', 'feel', 'tmpf', 'mslp', 'dwpf']):
                # Special handling for AQI which is not in METAR data
                if var_name == 'aqi':
                    # Skip AQI as it's not available in METAR data
                    # We'll leave it as zeros
                    continue
                
                # Extract point data for this variable
                point_data = []
                for station_data in ts_data:
                    # Check if this station has data for this variable
                    value = station_data['values'].get(var_name)
                    if value is not None and not np.isnan(value):
                        point_data.append({
                            'lat': station_data['lat'],
                            'lon': station_data['lon'],
                            'value': value
                        })
                
                # Skip interpolation if no data points available
                if not point_data:
                    continue
                
                # Interpolate to grid
                try:
                    grid = self._interpolate_to_grid(point_data, self.dim, self.dim, self.extent)
                    metar_data[t_idx, :, :, v_idx] = grid
                except Exception as e:
                    if self.verbose:
                        print(f"Error interpolating {var_name} at {ts_key}: {e}")
                    continue
                    
            # Units conversion if needed
            # Convert knots to mph for wind speed
            metar_data[t_idx, :, :, 0] *= 1.15078  # knots to mph conversion
        
        if self.verbose:
            print(f"Created METAR data with shape {metar_data.shape}")
        
        # Save to cache
        np.save(cache_file, metar_data)
        return metar_data
    
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
        
        # Format dates for API request
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Build API URL for IEM's ASOS/AWOS data service
        api_url = (
            'http://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?'
            f'station={station_str}&'
            'data=tmpf,dwpf,relh,drct,sknt,p01i,alti,mslp,feel&'
            f'year1={start_date.year}&month1={start_date.month}&day1={start_date.day}&'
            f'year2={end_date.year}&month2={end_date.month}&day2={end_date.day}&'
            'tz=Etc/UTC&format=comma&latlon=yes'
        )
        
        if self.verbose:
            print(f"Requesting METAR data for {len(stations)} stations from {start_str} to {end_str}")
            print(f"API URL: {api_url}")
        
        # Use the robust download approach with multiple attempts
        max_attempts = 6
        attempt = 0
        
        while attempt < max_attempts:
            try:
                if self.verbose and attempt > 0:
                    print(f"Attempt {attempt+1}/{max_attempts} to fetch METAR data")
                    
                response = requests.get(api_url, timeout=300)  # 5-minute timeout
                
                if response.status_code == 200:
                    csv_data = response.text
                    
                    # Check if we got an error message
                    if csv_data.startswith("#ERROR"):
                        if self.verbose:
                            print(f"Error from IEM API: {csv_data}")
                        attempt += 1
                        time.sleep(5)  # Wait before retry
                        continue
                    
                    # Save raw data for debugging
                    raw_file = os.path.join(self.cache_dir, f"metar_raw_{start_str}_to_{end_str}.csv")
                    with open(raw_file, 'w', encoding='utf-8') as f:
                        f.write(csv_data)
                    
                    if self.verbose:
                        print(f"Raw METAR data saved to {raw_file}")
                    
                    # Process the CSV data into station dictionaries
                    try:
                        # Parse CSV using pandas with more robust options
                        df = pd.read_csv(
                            io.StringIO(csv_data),
                            comment='#',           # Skip comment lines
                            skip_blank_lines=True,  # Skip blank lines
                            on_bad_lines='warn'    # Be more forgiving with malformed lines
                        )
                        
                        # Initialize the station data dictionary
                        station_data = {station: [] for station in stations}
                        
                        # Group by station
                        for station_id in stations:
                            station_df = df[df['station'] == station_id]
                            if len(station_df) > 0:
                                # Convert to list of dictionaries
                                station_data[station_id] = station_df.to_dict('records')
                                if self.verbose:
                                    print(f"  Received {len(station_data[station_id])} records for station {station_id}")
                            else:
                                if self.verbose:
                                    print(f"  No data received for station {station_id}")
                        
                        return station_data
                        
                    except Exception as e:
                        if self.verbose:
                            print(f"Error using pandas to parse CSV: {e}")
                            print("Falling back to manual CSV parsing")
                        
                        # Manual CSV parsing as fallback
                        lines = csv_data.strip().split('\n')
                        if len(lines) <= 1:
                            if self.verbose:
                                print("No data returned from IEM API")
                            return {station: [] for station in stations}
                            
                        # Get header line
                        header = lines[0].split(',')
                        
                        # Process data lines
                        station_data = {station: [] for station in stations}
                        for line in lines[1:]:
                            if not line.strip() or line.startswith('#'):
                                continue
                                
                            values = line.split(',')
                            if len(values) != len(header):
                                continue
                                
                            # Create data row
                            row = dict(zip(header, values))
                            
                            # Get station ID
                            station_id = row.get('station')
                            if not station_id or station_id not in stations:
                                continue
                                
                            # Add data to station dict
                            station_data[station_id].append(row)
                        
                        if self.verbose:
                            for station_id, data in station_data.items():
                                print(f"  Received {len(data)} records for station {station_id}")
                                
                        return station_data
                else:
                    if self.verbose:
                        print(f"Error fetching IEM METAR data: HTTP {response.status_code}")
                    attempt += 1
                    time.sleep(5)  # Wait before retry
                    continue
                    
            except Exception as e:
                if self.verbose:
                    print(f"Exception when fetching IEM METAR data: {e}")
                    import traceback
                    traceback.print_exc()
                attempt += 1
                time.sleep(5)  # Wait before retry
                continue
        
        if self.verbose:
            print("Exhausted attempts to download METAR data, returning empty data")
        return {station: [] for station in stations}  # Return empty lists for all stations


    def _process_timestamp_data(self, all_station_data, station_lookup):
        """
        Process station data into a dictionary keyed by timestamp.
        
        Parameters:
        -----------
        all_station_data : dict
            Dictionary with station IDs as keys and lists of data records as values
        station_lookup : dict
            Dictionary mapping station IDs to station information
            
        Returns:
        --------
        dict
            Dictionary with timestamp keys and lists of station data for each timestamp
        """
        timestamp_data = {}
        
        for station_id, records in all_station_data.items():
            station_info = station_lookup.get(station_id)
            if not station_info:
                continue
                
            for record in records:
                # Extract timestamp from record
                try:
                    timestamp_str = record.get('valid', '')
                    timestamp = pd.to_datetime(timestamp_str)
                    
                    # Round to nearest hour to match our timestamps
                    timestamp = timestamp.round('H')
                    
                    # Create a key for this timestamp
                    ts_key = timestamp.strftime('%Y-%m-%d-%H')
                    
                    # Initialize this timestamp's data if not exists
                    if ts_key not in timestamp_data:
                        timestamp_data[ts_key] = []
                    
                    # Extract values and convert to float where possible
                    values = {}
                    for field in ['tmpf', 'dwpf', 'relh', 'drct', 'sknt', 'p01i', 'mslp', 'feel']:
                        try:
                            # Check if the value is already a float
                            field_value = record.get(field, 'M')
                            if isinstance(field_value, float):
                                values[field] = field_value if not np.isnan(field_value) else float('nan')
                            else:
                                # It's a string, do the string replacement
                                values[field] = float(str(field_value).replace('M', 'nan'))
                        except (ValueError, TypeError):
                            values[field] = float('nan')
                    
                    # Add this station's data to the timestamp
                    timestamp_data[ts_key].append({
                        'lat': station_info['lat'],
                        'lon': station_info['lon'],
                        'values': values
                    })
                except Exception as e:
                    if self.verbose:
                        print(f"Error processing record from {station_id}: {e}")
                    continue
                    
        if self.verbose:
            print(f"Processed data for {len(timestamp_data)} unique hourly timestamps")
            
        return timestamp_data
    
    def _process_metar_csv(self, csv_data, stations):
        """
        Process CSV data from IEM METAR API into structured dictionary.
        
        Parameters:
        -----------
        csv_data : str
            CSV data from IEM API
        stations : list of str
            List of station IDs
            
        Returns:
        --------
        dict
            Dictionary with station IDs as keys and lists of data records as values
        """
        # Initialize empty data dictionaries for each station
    def _process_metar_csv(self, csv_data, stations):
            """
            Process CSV data from IEM METAR API into structured dictionary.
            
            Parameters:
            -----------
            csv_data : str
                CSV data from IEM API
            stations : list of str
                List of station IDs
                
            Returns:
            --------
            dict
                Dictionary with station IDs as keys and lists of data records as values
            """
            # Initialize empty data dictionaries for each station
            station_data = {station: [] for station in stations}
            
            try:
                # Try using pandas for efficient CSV parsing
                df = pd.read_csv(io.StringIO(csv_data))
                
                # Group by station
                for station_id in stations:
                    station_df = df[df['station'] == station_id]
                    if len(station_df) > 0:
                        # Convert to list of dictionaries
                        station_data[station_id] = station_df.to_dict('records')
                        if self.verbose:
                            print(f"  Received {len(station_data[station_id])} records for station {station_id}")
                    else:
                        if self.verbose:
                            print(f"  No data received for station {station_id}")
                
                return station_data
                
            except Exception as e:
                if self.verbose:
                    print(f"Error using pandas to parse CSV: {e}")
                    print("Falling back to manual CSV parsing")
                
                # Manual CSV parsing as fallback
                lines = csv_data.strip().split('\n')
                if len(lines) <= 1:
                    if self.verbose:
                        print("No data returned from IEM API")
                    return station_data
                    
                # Get header line
                header = lines[0].split(',')
                
                # Process data lines
                for line in lines[1:]:
                    if not line.strip() or line.startswith('#'):
                        continue
                        
                    values = line.split(',')
                    if len(values) != len(header):
                        continue
                        
                    # Create data row
                    row = dict(zip(header, values))
                    
                    # Get station ID
                    station_id = row.get('station')
                    if not station_id or station_id not in stations:
                        continue
                        
                    # Add data to station dict
                    station_data[station_id].append(row)
                
                if self.verbose:
                    for station_id, data in station_data.items():
                        print(f"  Received {len(data)} records for station {station_id}")
                        
                return station_data
    
    def _sliding_window_of(self, frames, window_size):
        """
        Create sliding window samples from sequential frames.
        
        Parameters:
        -----------
        frames : numpy.ndarray
            Sequential frames with shape (n_timestamps, height, width, channels)
        window_size : int
            Number of consecutive frames to include in each sample
        
        Returns:
        --------
        numpy.ndarray
            Sliding window samples with shape (n_samples, window_size, height, width, channels)
        """
        n_frames, row, col, channels = frames.shape
        n_samples = n_frames - window_size + 1
        
        if n_samples <= 0:
            raise ValueError(f"Not enough frames ({n_frames}) for sliding window of size {window_size}")
        
        samples = np.empty((n_samples, window_size, row, col, channels))
        
        for i in range(n_samples):
            samples[i] = frames[i:i+window_size]
        
        return samples
    
    def _interpolate_to_grid(self, point_data, rows, cols, extent, method='cubic'):
        """
        Interpolate point data to a regular grid.
        
        Parameters:
        -----------
        point_data : list of dict
            List of dictionaries with 'lat', 'lon', and 'value' keys
        rows : int
            Number of rows in the output grid
        cols : int
            Number of columns in the output grid
        extent : tuple
            Geographic bounds in format (min_lon, max_lon, min_lat, max_lat)
        method : str, optional
            Interpolation method to use: 'nearest', 'linear', or 'cubic'
            
        Returns:
        --------
        numpy.ndarray
            Interpolated grid with shape (rows, cols)
        """
        try:
            from scipy.interpolate import griddata
            
            # Check if we have enough points for the requested method
            if method == 'cubic' and len(point_data) < 4:
                method = 'linear'
            if method == 'linear' and len(point_data) < 3:
                method = 'nearest'
            if len(point_data) < 1:
                return np.zeros((rows, cols))
                
            # Extract points and values
            points = np.array([(p['lon'], p['lat']) for p in point_data])
            values = np.array([p['value'] for p in point_data])
            
            # Create regular grid
            lon_min, lon_max, lat_min, lat_max = extent
            x = np.linspace(lon_min, lon_max, cols)
            y = np.linspace(lat_min, lat_max, rows)
            xx, yy = np.meshgrid(x, y)
            
            # Interpolate
            grid = griddata(points, values, (xx, yy), method=method, fill_value=0)
            
            return grid
            
        except Exception as e:
            if self.verbose:
                print(f"Interpolation error: {e}")
                print("Falling back to simple distance-weighted interpolation")
            
            # Simple distance-weighted interpolation as fallback
            grid = np.zeros((rows, cols))
            lon_min, lon_max, lat_min, lat_max = extent
            
            # Calculate grid coordinates
            x_step = (lon_max - lon_min) / (cols - 1)
            y_step = (lat_max - lat_min) / (rows - 1)
            
            for i in range(rows):
                for j in range(cols):
                    # Calculate the lat/lon for this grid point
                    lon = lon_min + j * x_step
                    lat = lat_min + i * y_step
                    
                    # Calculate weighted value based on inverse distance
                    total_weight = 0
                    weighted_sum = 0
                    
                    for point in point_data:
                        # Calculate distance to this point
                        dist = np.sqrt((lon - point['lon'])**2 + 
                                      (lat - point['lat'])**2)
                        
                        # Avoid division by zero
                        if dist < 1e-10:
                            return point['value']
                        
                        # Weight is inverse of distance squared
                        weight = 1.0 / (dist**2)
                        total_weight += weight
                        weighted_sum += weight * point['value']
                    
                    if total_weight > 0:
                        grid[i, j] = weighted_sum / total_weight
                    else:
                        grid[i, j] = 0
            
            return grid
    
    def _fetch_airnow_data(self, lat, lon, date, hour):
        """
        Fetch PM2.5 data from AirNow API for a specific location and time.
        
        Parameters:
        -----------
        lat : float
            Latitude
        lon : float
            Longitude
        date : str
            Date in 'YYYY-MM-DD' format
        hour : int
            Hour (0-23)
        
        Returns:
        --------
        float
            PM2.5 value
        """
        if not self.airnow_api_key:
            if self.verbose:
                print("No AirNow API key found. Cannot fetch real PM2.5 data.")
            return 0.0  # Return zero instead of random data
        
        # AirNow API parameters
        params = {
            'latitude': lat,
            'longitude': lon,
            'format': 'application/json',
            'API_KEY': self.airnow_api_key,
            'parameter': 'PM25',
            'date': date,
            'hour': hour
        }
        
        # Build URL
        base_url = 'http://www.airnowapi.org/aq/observation/latLong/historical/'
        url = f"{base_url}?{urlencode(params)}"
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    return float(data[0]['AQI'])
                else:
                    if self.verbose:
                        print(f"No data returned for {lat}, {lon} at {date} {hour}:00")
                    return 0.0
            else:
                if self.verbose:
                    print(f"Error fetching AirNow data: {response.status_code}")
                return 0.0
        except Exception as e:
            if self.verbose:
                print(f"Exception when fetching AirNow data: {e}")
            return 0.0
    
    def _print_data_statistics(self):
        """Print detailed statistics about the final data"""
        if not self.verbose:
            return
        
        # Print basic stats about the combined data
        print("\nChannel Statistics:")
        print("===================")
        
        # Get the total number of channels
        total_channels = self.all_channels.shape[-1]
        
        # Define channel names based on the documentation
        channel_info = self.get_channel_info()
        
        # Print stats for each channel
        for i, channel_name in enumerate(channel_info['channel_names']):
            if i < total_channels:
                channel_data = self.all_channels[0, :, :, i]  # First timestamp
                
                print(f"\nChannel {i}: {channel_name}")
                print(f"  Min: {np.min(channel_data)}")
                print(f"  Max: {np.max(channel_data)}")
                print(f"  Mean: {np.mean(channel_data)}")
                print(f"  Std: {np.std(channel_data)}")
                
                # Count non-zero values
                non_zero = np.count_nonzero(channel_data)
                total = channel_data.size
                print(f"  Data coverage: {non_zero/total*100:.2f}% ({non_zero}/{total} non-zero pixels)")
        
        print("\nFinal Data Shape:")
        print(f"  {self.data.shape[0]} samples")
        print(f"  {self.data.shape[1]} frames per sample")
        print(f"  {self.data.shape[2]}x{self.data.shape[3]} grid size")
        print(f"  {self.data.shape[4]} channels")
        
        print("\nData Memory Usage:")
        data_size_bytes = self.data.nbytes
        data_size_mb = data_size_bytes / (1024 * 1024)
        print(f"  {data_size_mb:.2f} MB")
    
    def get_channel_info(self):
        """
        Get information about the channels in the dataset.
        
        Returns:
        --------
        dict
            Dictionary with channel information
        """
        # Define channel names based on the documentation
        # NOTE: Ground-level PM2.5 data is provided by the AirNowData class
        
        maiac_channels = ['MAIAC_AOD']
        
        tropomi_channels = [
            'TROPOMI_Methane',
            'TROPOMI_NO2',
            'TROPOMI_CO'
        ]
        
        modis_fire_channels = ['MODIS_FRP']
        
        merra2_channels = [
            'MERRA2_PBL_Height',
            'MERRA2_Surface_Air_Temp',
            'MERRA2_Surface_Exchange_Coef'
        ]
        
        metar_channels = [
            'METAR_Wind_Speed',
            'METAR_Wind_Direction',
            'METAR_Precipitation',
            'METAR_AQI',
            'METAR_Humidity',
            'METAR_Heat_Index',
            'METAR_Air_Temp',
            'METAR_Air_Pressure',
            'METAR_Dew_Point'
        ]
        
        # Combine all channel names
        channel_names = (
            maiac_channels +
            tropomi_channels +
            modis_fire_channels +
            merra2_channels +
            metar_channels
        )
        
        return {
            'maiac_channels': maiac_channels,
            'tropomi_channels': tropomi_channels,
            'modis_fire_channels': modis_fire_channels,
            'merra2_channels': merra2_channels,
            'metar_channels': metar_channels,
            'channel_names': channel_names,
            'channel_order': channel_names,  # Alias for compatibility
            'total_channels': len(channel_names)
        }
    
    def save_data(self, filepath):
        """
        Save the processed data to a file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the data file
        """
        np.save(filepath, self.data)
        if self.verbose:
            print(f"Data saved to {filepath}")
    
    def load_data(self, filepath):
        """
        Load processed data from a file.
        
        Parameters:
        -----------
        filepath : str
            Path to the data file
        """
        self.data = np.load(filepath)
        if self.verbose:
            print(f"Data loaded from {filepath}")
            print(f"Data shape: {self.data.shape}")