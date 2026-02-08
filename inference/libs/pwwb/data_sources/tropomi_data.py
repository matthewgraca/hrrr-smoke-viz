import os
import numpy as np
import pandas as pd
from datetime import timedelta
import requests
import netCDF4 as nc
from scipy.interpolate import griddata
import traceback
import time
import random

from libs.pwwb.data_sources.base_data_source import BaseDataSource


class TropomiDataSource(BaseDataSource):
    """Fetches TROPOMI atmospheric trace gas data (methane, NO2, carbon monoxide) from Sentinel-5P."""
    
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
        max_retries=5,
        retry_delay=30,
        exponential_backoff=True
    ):
        """
        Initialize TROPOMI data source.
        
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
            Specific TROPOMI channels to include (default: all)
        max_retries : int
            Maximum number of retry attempts for failed dates
        retry_delay : int
            Base delay in seconds between retries
        exponential_backoff : bool
            Whether to use exponential backoff for retries
        """
        super().__init__(timestamps, extent, dim, cache_dir, cache_prefix, use_cached_data, verbose)
        
        self.full_channels = ['TROPOMI_Methane', 'TROPOMI_NO2', 'TROPOMI_CO']
        if channels is not None:
            self.channels = [c for c in channels if c in self.full_channels]
        else:
            self.channels = self.full_channels.copy()
        
        self.earthdata_token = os.getenv('EARTHDATA_TOKEN')
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.exponential_backoff = exponential_backoff
    
    def _calculate_retry_delay(self, attempt):
        """Calculate delay for retry with optional exponential backoff and jitter."""
        if self.exponential_backoff:
            delay = self.retry_delay * (2 ** attempt)
        else:
            delay = self.retry_delay
        
        # Add jitter to prevent thundering herd
        jitter = random.uniform(0.5, 1.5)
        return delay * jitter
    
    def _is_network_error(self, exception):
        """Check if exception is a network-related error that should trigger retry."""
        network_errors = (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.HTTPError,
            ConnectionError,
            OSError
        )
        
        if isinstance(exception, network_errors):
            return True
        
        # Check for specific error messages
        error_str = str(exception).lower()
        network_keywords = [
            'name resolution',
            'connection',
            'timeout',
            'network',
            'dns',
            'socket',
            'ssl',
            'certificate'
        ]
        
        return any(keyword in error_str for keyword in network_keywords)
    
    def _process_date_with_retry(self, date, products, min_lon, max_lon, min_lat, max_lat, headers, num_channels):
        """Process a single date with retry logic."""
        date_str = date.strftime('%Y-%m-%d')
        day_next = (date + timedelta(days=1)).strftime('%Y-%m-%d')
        
        for attempt in range(self.max_retries + 1):
            try:
                if self.verbose and attempt > 0:
                    print(f"Retry attempt {attempt} for date: {date_str}")
                elif self.verbose:
                    print(f"Processing TROPOMI data for date: {date_str}")
                
                day_data = np.zeros((self.dim, self.dim, num_channels))
                
                for product in products:
                    try:
                        cmr_url = "https://cmr.earthdata.nasa.gov/search/granules.json"
                        
                        params = {
                            "collection_concept_id": product["cmr_id"],
                            "temporal": f"{date_str}T00:00:00Z,{day_next}T00:00:00Z",
                            "bounding_box": f"{min_lon},{min_lat},{max_lon},{max_lat}",
                            "page_size": 10,
                            "sort_key": "-start_date"
                        }
                        
                        response = requests.get(cmr_url, params=params, headers=headers, timeout=60)
                        
                        if response.status_code != 200:
                            if self.verbose:
                                print(f"Error searching for TROPOMI {product['name']} data: {response.status_code}")
                            continue
                        
                        results = response.json()
                        granules = results.get("feed", {}).get("entry", [])
                        
                        if not granules:
                            if self.verbose:
                                print(f"No TROPOMI {product['name']} data found for {date_str}")
                            continue
                        
                        for granule in granules:
                            download_url = next((link["href"] for link in granule.get("links", []) 
                                                if link.get("rel") == "http://esipfed.org/ns/fedsearch/1.1/data#"), None)
                            
                            if not download_url:
                                continue
                            
                            temp_file = os.path.join(self.cache_dir, f"tropomi_{product['name']}_{date_str}.nc")
                            
                            try:
                                response = requests.get(download_url, headers=headers, stream=True, timeout=300)
                                if response.status_code != 200:
                                    if self.verbose:
                                        print(f"Error downloading TROPOMI file: {response.status_code}")
                                    continue
                                        
                                with open(temp_file, 'wb') as f:
                                    for chunk in response.iter_content(chunk_size=8192):
                                        f.write(chunk)
                                
                                if not os.path.exists(temp_file) or os.path.getsize(temp_file) == 0:
                                    raise Exception("Failed to download or empty file")
                                
                                dataset = nc.Dataset(temp_file, 'r')
                                
                                try:
                                    var_path = product["var_name"].split('/')
                                    lat_path = product["lat_var"].split('/')
                                    lon_path = product["lon_var"].split('/')
                                    qa_path = product["qa_var"].split('/')
                                    
                                    if len(var_path) > 1:
                                        group_name = var_path[0]
                                        var_name = var_path[1]
                                        group = dataset.groups[group_name]
                                        data_var = group.variables[var_name][:]
                                    else:
                                        data_var = dataset.variables[var_path[0]][:]
                                    
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
                                    
                                    qa_var = None
                                    if len(qa_path) > 1:
                                        group_name = qa_path[0]
                                        var_name = qa_path[1]
                                        if group_name in dataset.groups and var_name in dataset.groups[group_name].variables:
                                            group = dataset.groups[group_name]
                                            qa_var = group.variables[var_name][:]
                                    
                                    dataset.close()
                                    
                                    # Remove time dimension if present
                                    if data_var.ndim > 2 and data_var.shape[0] == 1:
                                        data_var = data_var[0]
                                        lat_var = lat_var[0] if lat_var.ndim > 2 else lat_var
                                        lon_var = lon_var[0] if lon_var.ndim > 2 else lon_var
                                        if qa_var is not None and qa_var.ndim > 2 and qa_var.shape[0] == 1:
                                            qa_var = qa_var[0]
                                    
                                    # Apply quality filtering (keep only high quality data >0.75)
                                    if qa_var is not None:
                                        quality_mask = qa_var > 0.75
                                        data_var = np.where(quality_mask, data_var, np.nan)
                                    
                                    grid_x, grid_y = np.meshgrid(
                                        np.linspace(min_lon, max_lon, self.dim),
                                        np.linspace(min_lat, max_lat, self.dim)
                                    )
                                    
                                    points = np.column_stack((lon_var.flatten(), lat_var.flatten()))
                                    values = data_var.flatten()
                                    
                                    valid_mask = ~np.isnan(values)
                                    points = points[valid_mask]
                                    values = values[valid_mask]
                                    
                                    if len(points) > 3:
                                        grid_z = griddata(points, values, (grid_x, grid_y), method='linear', fill_value=0)
                                        day_data[:, :, product["index"]] = grid_z
                                        
                                        if self.verbose:
                                            print(f"Successfully processed {product['name']} data")
                                        
                                        break
                                    
                                except Exception as e:
                                    if self.verbose:
                                        print(f"Error extracting TROPOMI data variables: {e}")
                                    raise
                                    
                            except Exception as e:
                                if self.verbose:
                                    print(f"Error processing TROPOMI file: {e}")
                                raise
                            
                            finally:
                                if os.path.exists(temp_file):
                                    os.remove(temp_file)
                            
                    except Exception as e:
                        if self.verbose:
                            print(f"Error in TROPOMI processing for {product['name']} on {date_str}: {e}")
                        if self._is_network_error(e):
                            raise  # Re-raise network errors to trigger retry
                        # For non-network errors, continue with next product
                        continue
                
                # If we reach here, processing succeeded (even if some products failed)
                if self.verbose:
                    print(f"Successfully processed NO2 data")
                return day_data
                
            except Exception as e:
                if attempt < self.max_retries and self._is_network_error(e):
                    delay = self._calculate_retry_delay(attempt)
                    if self.verbose:
                        print(f"Network error for {date_str} (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                        print(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    # Either max retries reached or non-network error
                    if self.verbose:
                        if attempt >= self.max_retries:
                            print(f"Max retries reached for {date_str}. Using interpolated data.")
                        else:
                            print(f"Non-network error for {date_str}: {e}")
                    break
        
        # If all retries failed, return zeros (will be interpolated later)
        return np.zeros((self.dim, self.dim, num_channels))
    
    def _interpolate_missing_dates(self, daily_tropomi_data, unique_dates, num_channels):
        """Interpolate data for missing dates using neighboring dates."""
        if self.verbose:
            print("Interpolating missing dates...")
        
        # Convert to sorted list for easier interpolation
        sorted_dates = sorted(unique_dates)
        
        for i, date in enumerate(sorted_dates):
            if date not in daily_tropomi_data:
                if self.verbose:
                    print(f"Interpolating data for missing date: {date}")
                
                # Find neighboring dates with data
                before_data = None
                after_data = None
                
                # Look backward
                for j in range(i-1, -1, -1):
                    if sorted_dates[j] in daily_tropomi_data:
                        before_data = daily_tropomi_data[sorted_dates[j]]
                        break
                
                # Look forward
                for j in range(i+1, len(sorted_dates)):
                    if sorted_dates[j] in daily_tropomi_data:
                        after_data = daily_tropomi_data[sorted_dates[j]]
                        break
                
                # Interpolate or use available data
                if before_data is not None and after_data is not None:
                    # Linear interpolation between two dates
                    interpolated = (before_data + after_data) / 2
                elif before_data is not None:
                    # Use previous date's data
                    interpolated = before_data.copy()
                elif after_data is not None:
                    # Use next date's data
                    interpolated = after_data.copy()
                else:
                    # No data available, use zeros
                    interpolated = np.zeros((self.dim, self.dim, num_channels))
                
                daily_tropomi_data[date] = interpolated
    
    def get_data(self):
        """
        Fetch TROPOMI atmospheric data with robust retry logic to ensure no gaps.
        
        Returns:
        --------
        numpy.ndarray
            TROPOMI data (n_timestamps, dim, dim, n_features) with quality-filtered measurements
        """
        num_channels = len(self.channels)
        
        if self.verbose:
            print(f"Including TROPOMI channels: {self.channels}")
        
        channel_id = '_'.join(c.split('_')[1].lower() for c in self.channels)
        cache_file = os.path.join(self.cache_dir, f"{self.cache_prefix}tropomi_{channel_id}_data.npy")
        
        data = self.check_cache(cache_file, (self.n_timestamps, self.dim, self.dim, num_channels))
        if data is not None:
            return data
        
        tropomi_data = np.zeros((self.n_timestamps, self.dim, self.dim, num_channels))
        
        if not self.earthdata_token:
            if self.verbose:
                print("NASA EarthData token not found. Returning empty TROPOMI data.")
            self.save_to_cache(tropomi_data, cache_file)
            return tropomi_data
        
        unique_dates = pd.Series([ts.date() for ts in self.timestamps]).unique()
        
        if self.verbose:
            print(f"Fetching TROPOMI data for {len(unique_dates)} unique dates")
            print(f"Retry configuration: max_retries={self.max_retries}, base_delay={self.retry_delay}s, exponential_backoff={self.exponential_backoff}")
        
        min_lon, max_lon, min_lat, max_lat = self.extent
        headers = {"Authorization": f"Bearer {self.earthdata_token}"}
        
        all_products = [
            {
                "name": "NO2",
                "channel_name": "TROPOMI_NO2",
                "index": 0,
                "cmr_id": "C2089270961-GES_DISC",
                "var_name": "PRODUCT/nitrogendioxide_tropospheric_column",
                "lat_var": "PRODUCT/latitude",
                "lon_var": "PRODUCT/longitude",
                "qa_var": "PRODUCT/qa_value"
            },
            {
                "name": "CH4",
                "channel_name": "TROPOMI_Methane",
                "index": 0,
                "cmr_id": "C2087216530-GES_DISC",
                "var_name": "PRODUCT/methane_mixing_ratio",
                "lat_var": "PRODUCT/latitude",
                "lon_var": "PRODUCT/longitude",
                "qa_var": "PRODUCT/qa_value"
            },
            {
                "name": "CO",
                "channel_name": "TROPOMI_CO",
                "index": 0,
                "cmr_id": "C2087132178-GES_DISC",
                "var_name": "PRODUCT/carbonmonoxide_total_column",
                "lat_var": "PRODUCT/latitude",
                "lon_var": "PRODUCT/longitude",
                "qa_var": "PRODUCT/qa_value"
            }
        ]
        
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
        
        daily_tropomi_data = {}
        
        for date in unique_dates:
            day_data = self._process_date_with_retry(
                date, products, min_lon, max_lon, min_lat, max_lat, headers, num_channels
            )
            
            if np.sum(np.abs(day_data)) > 0:
                daily_tropomi_data[date] = day_data
        
        self._interpolate_missing_dates(daily_tropomi_data, unique_dates, num_channels)
        
        # Fill final data array
        for t_idx, timestamp in enumerate(self.timestamps):
            date = timestamp.date()
            if date in daily_tropomi_data:
                tropomi_data[t_idx] = daily_tropomi_data[date]
        
        if self.verbose:
            print(f"Created TROPOMI data with shape {tropomi_data.shape}")
            missing_dates = [date for date in unique_dates if date not in daily_tropomi_data]
            if missing_dates:
                print(f"Interpolated data for {len(missing_dates)} dates")
            else:
                print("No gaps in data - all dates processed successfully")
        
        self.save_to_cache(tropomi_data, cache_file)
        return tropomi_data