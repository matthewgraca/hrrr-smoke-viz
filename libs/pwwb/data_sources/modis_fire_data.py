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
    """Fetches MODIS Land Surface Temperature data as a proxy for fire activity."""
    
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
        Initialize MODIS fire data source.
        
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
        """
        super().__init__(timestamps, extent, dim, cache_dir, cache_prefix, use_cached_data, verbose)
        self.earthdata_token = os.getenv('EARTHDATA_TOKEN')
    
    def get_data(self):
        """
        Fetch MODIS Land Surface Temperature data, returning cached version if available.
        
        Returns:
        --------
        numpy.ndarray
            Temperature data (n_timestamps, dim, dim, 1) normalized to 0-1 range
        """
        cache_file = os.path.join(self.cache_dir, f"{self.cache_prefix}modis_fire_data.npy")
        
        data = self.check_cache(cache_file, (self.n_timestamps, self.dim, self.dim, 1))
        if data is not None:
            return data
        
        modis_fire_data = np.zeros((self.n_timestamps, self.dim, self.dim, 1))
        
        if not self.earthdata_token:
            if self.verbose:
                print("NASA Earth Data token not found. Returning empty MODIS fire data.")
            self.save_to_cache(modis_fire_data, cache_file)
            return modis_fire_data
        
        unique_dates = pd.Series([ts.date() for ts in self.timestamps]).unique()
        
        if self.verbose:
            print(f"Fetching MODIS fire data for {len(unique_dates)} unique dates")
        
        headers = {"Authorization": f"Bearer {self.earthdata_token}"}
        collection_id = "C1748058432-LPCLOUD"  # MOD11A1.061 Land Surface Temperature
        daily_fire_data = {}
        
        for date in unique_dates:
            date_str = date.strftime('%Y-%m-%d')
            
            if self.verbose:
                print(f"Processing MODIS fire data for date: {date_str}")
            
            min_lon, max_lon, min_lat, max_lat = self.extent
            bbox = f"{min_lon},{min_lat},{max_lon},{max_lat}"
            
            try:
                cmr_url = "https://cmr.earthdata.nasa.gov/search/granules.json"
                
                params = {
                    "collection_concept_id": collection_id,
                    "temporal": f"{date_str}T00:00:00Z,{date_str}T23:59:59Z",
                    "bounding_box": bbox,
                    "page_size": 10
                }
                
                response = requests.get(cmr_url, params=params, headers=headers)
                
                if response.status_code != 200:
                    if self.verbose:
                        print(f"Error searching for MODIS data: {response.status_code}")
                    continue
                
                results = response.json()
                granules = results.get("feed", {}).get("entry", [])
                
                if not granules:
                    if self.verbose:
                        print(f"No MODIS data found for {date_str}")
                    continue
                
                for granule in granules:
                    download_url = next((link["href"] for link in granule.get("links", []) 
                                    if link.get("rel") == "http://esipfed.org/ns/fedsearch/1.1/data#"), None)
                    
                    if not download_url:
                        continue
                    
                    temp_file = os.path.join(self.cache_dir, f"modis_temp_{date_str}.hdf")
                    hdf_file = None
                    
                    try:
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
                        
                        hdf_file = SD(temp_file, SDC.READ)
                        
                        try:
                            dataset_name = 'LST_Day_1km'
                            
                            try:
                                dataset = hdf_file.select(dataset_name)
                                lst_data = dataset[:]
                                
                                if self.verbose:
                                    print(f"Found LST data with shape: {lst_data.shape}")
                                    print(f"Min/Max values: {np.min(lst_data)} to {np.max(lst_data)}")
                                
                                attrs = dataset.attributes()
                                scale_factor = float(attrs.get('scale_factor', 0.02))
                                add_offset = float(attrs.get('add_offset', 0.0))
                                
                                lst_data = lst_data * scale_factor + add_offset
                                
                                # Filter unrealistic values (< 100K)
                                lst_data = np.where(lst_data > 100, lst_data, np.nan)
                                
                                if np.all(np.isnan(lst_data)):
                                    if self.verbose:
                                        print(f"No valid LST data found for {date_str} after filtering")
                                    continue
                                    
                                # Normalize to 0-1 range (270K to 330K)
                                min_temp = 270
                                max_temp = 330
                                normalized_lst = np.clip((lst_data - min_temp) / (max_temp - min_temp), 0, 1)
                                normalized_lst = np.nan_to_num(normalized_lst, nan=0)
                                
                                if self.dim <= 0:
                                    raise ValueError(f"Invalid dimension size: {self.dim}")
                                if normalized_lst.shape[0] == 0 or normalized_lst.shape[1] == 0:
                                    raise ValueError(f"Invalid source dimensions: {normalized_lst.shape}")
                                
                                zoom_y = self.dim / normalized_lst.shape[0]
                                zoom_x = self.dim / normalized_lst.shape[1]
                                
                                grid_lst = zoom(normalized_lst, (zoom_y, zoom_x), order=1, mode='nearest')
                                
                                if grid_lst.shape != (self.dim, self.dim):
                                    if self.verbose:
                                        print(f"Warning: Resized dimensions {grid_lst.shape} don't match expected {(self.dim, self.dim)}")
                                        
                                    from skimage.transform import resize as skimage_resize
                                    grid_lst = skimage_resize(grid_lst, (self.dim, self.dim), 
                                                            preserve_range=True, anti_aliasing=True)
                                    
                                daily_fire_data[date] = grid_lst
                                
                                if self.verbose:
                                    print(f"Successfully processed LST data for {date_str}")
                                    
                                break
                                
                            except Exception as e:
                                if self.verbose:
                                    print(f"Error processing dataset {dataset_name}: {e}")
                                continue
                        
                        finally:
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
                        if os.path.exists(temp_file):
                            try:
                                os.remove(temp_file)
                            except Exception as e:
                                if self.verbose:
                                    print(f"Warning: Failed to remove temp file: {e}")
            
            except Exception as e:
                if self.verbose:
                    print(f"Error in MODIS data processing for {date_str}: {e}")
        
        for t_idx, timestamp in enumerate(self.timestamps):
            date = timestamp.date()
            if date in daily_fire_data:
                modis_fire_data[t_idx, :, :, 0] = daily_fire_data[date]
        
        # Spatial smoothing to simulate heat diffusion
        for t_idx in range(self.n_timestamps):
            if np.max(modis_fire_data[t_idx, :, :, 0]) > 0:
                modis_fire_data[t_idx, :, :, 0] = gaussian_filter(modis_fire_data[t_idx, :, :, 0], sigma=1.0)
        
        # Temporal smoothing to create realistic heat persistence and gradual cooling
        for t_idx in range(1, self.n_timestamps):
            if np.max(modis_fire_data[t_idx, :, :, 0]) == 0 and np.max(modis_fire_data[t_idx-1, :, :, 0]) > 0:
                modis_fire_data[t_idx, :, :, 0] = modis_fire_data[t_idx-1, :, :, 0] * 0.9
            elif np.max(modis_fire_data[t_idx, :, :, 0]) > 0 and np.max(modis_fire_data[t_idx-1, :, :, 0]) > 0:
                modis_fire_data[t_idx, :, :, 0] = (
                    modis_fire_data[t_idx-1, :, :, 0] * 0.3 + 
                    modis_fire_data[t_idx, :, :, 0] * 0.7
                )
        
        if self.verbose:
            print(f"Created MODIS fire data with shape {modis_fire_data.shape}")
        
        self.save_to_cache(modis_fire_data, cache_file)
        return modis_fire_data