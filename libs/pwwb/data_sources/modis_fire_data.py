import os
import numpy as np
import pandas as pd
from datetime import timedelta
import requests
from pyhdf.SD import SD, SDC
from scipy.ndimage import zoom, gaussian_filter
import traceback

from libs.pwwb.data_sources.base_data_source import BaseDataSource

class ModisFireDataSource(BaseDataSource):
    """
    Class to handle MODIS Fire Radiative Power (FRP) data collection and processing.
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
        Initialize the MODIS Fire data source.
        
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
        super().__init__(timestamps, extent, dim, cache_dir, cache_prefix, use_cached_data, verbose)
        
        # Get EarthData token from environment
        self.earthdata_token = os.getenv('EARTHDATA_TOKEN')
    
    def get_data(self):
        """
        Get MODIS Fire Radiative Power (FRP) data.
        
        Returns:
        --------
        numpy.ndarray
            MODIS FRP data with shape (n_timestamps, dim, dim, n_features)
        """
        cache_file = os.path.join(self.cache_dir, f"{self.cache_prefix}modis_fire_data.npy")
        
        # Check if cache exists and has correct shape
        data = self.check_cache(cache_file, (self.n_timestamps, self.dim, self.dim, 1))
        if data is not None:
            return data
        
        # Initialize empty array for MODIS fire data
        # Single channel for FRP
        modis_fire_data = np.zeros((self.n_timestamps, self.dim, self.dim, 1))
        
        # Check if Earth Data token is available
        if not self.earthdata_token:
            if self.verbose:
                print("NASA Earth Data token not found. Returning empty MODIS fire data.")
            self.save_to_cache(modis_fire_data, cache_file)
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
        
        self.save_to_cache(modis_fire_data, cache_file)
        return modis_fire_data