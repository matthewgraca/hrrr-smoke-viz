import os
import numpy as np
import pandas as pd
from datetime import timedelta
import requests
from pyhdf.SD import SD, SDC
from scipy.ndimage import zoom
import traceback

class MaiacDataSource:
    """
    Class to handle MAIAC AOD data collection and processing.
    """
    
    def __init__(
        self,
        timestamps,
        extent,
        dim,
        cache_dir='data/pwwb_cache/',
        cache_prefix='',
        use_cached_data=True,
        verbose=False
    ):
        """
        Initialize the MAIAC AOD data source.
        
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
        """
        self.timestamps = timestamps
        self.n_timestamps = len(timestamps)
        self.extent = extent
        self.dim = dim
        self.cache_dir = cache_dir
        self.cache_prefix = cache_prefix
        self.use_cached_data = use_cached_data
        self.verbose = verbose
        
        # Get EarthData token from environment
        self.earthdata_token = os.getenv('EARTHDATA_TOKEN')
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_data(self):
        """
        Get MAIAC AOD data from NASA.
        
        Returns:
        --------
        numpy.ndarray
            MAIAC AOD data with shape (n_timestamps, dim, dim, n_features)
        """
        # Use date-specific cache filename
        cache_file = os.path.join(self.cache_dir, f"{self.cache_prefix}maiac_aod_data.npy")
        
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
        # Using Version 061 (current version)
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
                    # Get download URL
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
                            traceback.print_exc()
                    
                    finally:
                        # Clean up the temporary file
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                        
            except Exception as e:
                if self.verbose:
                    print(f"Error fetching MAIAC AOD data for {date_str}: {e}")
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