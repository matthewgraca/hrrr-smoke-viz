import os
import numpy as np
import pandas as pd
from datetime import timedelta
import requests
import netCDF4 as nc
from scipy.interpolate import griddata
import traceback

class TropomiDataSource:
    """
    Class to handle TROPOMI data collection and processing.
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
        Initialize the TROPOMI data source.
        
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
            List of specific TROPOMI channels to include. Default is all channels.
        """
        self.timestamps = timestamps
        self.n_timestamps = len(timestamps)
        self.extent = extent
        self.dim = dim
        self.cache_dir = cache_dir
        self.cache_prefix = cache_prefix
        self.use_cached_data = use_cached_data
        self.verbose = verbose
        
        # Set channels to include
        self.full_channels = ['TROPOMI_Methane', 'TROPOMI_NO2', 'TROPOMI_CO']
        if channels is not None:
            # Filter to only valid channels
            self.channels = [c for c in channels if c in self.full_channels]
        else:
            self.channels = self.full_channels.copy()
        
        # Get EarthData token from environment
        self.earthdata_token = os.getenv('EARTHDATA_TOKEN')
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_data(self):
        """
        Get TROPOMI data from NASA Earthdata for methane, nitrogen dioxide, and carbon monoxide.
        
        Returns:
        --------
        numpy.ndarray
            TROPOMI data with shape (n_timestamps, dim, dim, n_features)
        """
        # Count how many channels are included
        num_channels = len(self.channels)
        
        if self.verbose:
            print(f"Including TROPOMI channels: {self.channels}")
        
        # Generate a channel-specific cache identifier
        channel_id = '_'.join(c.split('_')[1].lower() for c in self.channels)
        cache_file = os.path.join(self.cache_dir, f"{self.cache_prefix}tropomi_{channel_id}_data.npy")
        
        if self.use_cached_data and os.path.exists(cache_file):
            if self.verbose:
                print(f"Loading cached TROPOMI data from {cache_file}")
            tropomi_data = np.load(cache_file)
            # Verify the shape matches what we expect
            if tropomi_data.shape[0] == self.n_timestamps and tropomi_data.shape[3] == num_channels:
                return tropomi_data
            else:
                if self.verbose:
                    print(f"Cached TROPOMI data has shape {tropomi_data.shape}, " + 
                          f"but expected ({self.n_timestamps}, {self.dim}, {self.dim}, {num_channels})")
                    print("Creating a new array with the correct dimensions")
        
        # Initialize empty array for TROPOMI data
        # channels based on requested channels
        tropomi_data = np.zeros((self.n_timestamps, self.dim, self.dim, num_channels))
        
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
        all_products = [
            {
                "name": "NO2",  # Nitrogen Dioxide
                "channel_name": "TROPOMI_NO2",
                "index": 0,  # This will be adjusted based on included channels
                "cmr_id": "C2089270961-GES_DISC",
                "var_name": "PRODUCT/nitrogendioxide_tropospheric_column",
                "lat_var": "PRODUCT/latitude",
                "lon_var": "PRODUCT/longitude",
                "qa_var": "PRODUCT/qa_value"
            },
            {
                "name": "CH4",  # Methane
                "channel_name": "TROPOMI_Methane",
                "index": 0,  # Will be adjusted
                "cmr_id": "C2087216530-GES_DISC",
                "var_name": "PRODUCT/methane_mixing_ratio",
                "lat_var": "PRODUCT/latitude",
                "lon_var": "PRODUCT/longitude",
                "qa_var": "PRODUCT/qa_value"
            },
            {
                "name": "CO",  # Carbon Monoxide
                "channel_name": "TROPOMI_CO",
                "index": 0,  # Will be adjusted
                "cmr_id": "C2087132178-GES_DISC",
                "var_name": "PRODUCT/carbonmonoxide_total_column",
                "lat_var": "PRODUCT/latitude",
                "lon_var": "PRODUCT/longitude",
                "qa_var": "PRODUCT/qa_value"
            }
        ]
        
        # Filter products based on requested channels and assign proper indices
        products = []
        idx = 0
        for product in all_products:
            if product["channel_name"] in self.channels:
                product_copy = product.copy()
                product_copy["index"] = idx
                products.append(product_copy)
                idx += 1
        
        if self.verbose:
            print(f"Processing TROPOMI products: {[p['name'] for p in products]}")
        
        # For each day, try to fetch TROPOMI data for each product
        daily_tropomi_data = {}
        
        for date in unique_dates:
            date_str = date.strftime('%Y-%m-%d')
            day_next = (date + timedelta(days=1)).strftime('%Y-%m-%d')
            
            if self.verbose:
                print(f"Processing TROPOMI data for date: {date_str}")
            
            # Initialize the day's data
            day_data = np.zeros((self.dim, self.dim, num_channels))
            
            # For each selected product (NO2, CH4, CO)
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
                                    traceback.print_exc()
                                
                        except Exception as e:
                            if self.verbose:
                                print(f"Error processing TROPOMI file: {e}")
                                traceback.print_exc()
                        
                        finally:
                            # Clean up the temporary file
                            if os.path.exists(temp_file):
                                os.remove(temp_file)
                        
                except Exception as e:
                    if self.verbose:
                        print(f"Error in TROPOMI processing for {product['name']} on {date_str}: {e}")
                        traceback.print_exc()
            
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