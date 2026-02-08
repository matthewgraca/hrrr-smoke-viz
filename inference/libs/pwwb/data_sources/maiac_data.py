import os
import numpy as np
import pandas as pd
from datetime import timedelta
import requests
from pyhdf.SD import SD, SDC
from scipy.ndimage import zoom
import traceback

from libs.pwwb.data_sources.base_data_source import BaseDataSource


class MaiacDataSource(BaseDataSource):
    """Fetches and processes MAIAC Aerosol Optical Depth data from NASA EarthData."""
    
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
        Initialize MAIAC AOD data source.
        
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
        Fetch MAIAC AOD data, returning cached version if available.
        
        Returns:
        --------
        numpy.ndarray
            AOD data (n_timestamps, dim, dim, 1) with -1.0 for missing values
        """
        cache_file = os.path.join(self.cache_dir, f"{self.cache_prefix}maiac_aod_data.npy")
        
        data = self.check_cache(cache_file, (self.n_timestamps, self.dim, self.dim, 1))
        if data is not None:
            return data
        
        maiac_data = np.full((self.n_timestamps, self.dim, self.dim, 1), -1.0, dtype=np.float32)
        
        if not self.earthdata_token:
            if self.verbose:
                print("NASA EarthData token not found. Returning missing MAIAC AOD data.")
            self.save_to_cache(maiac_data, cache_file)
            return maiac_data
        
        unique_dates = pd.Series([ts.date() for ts in self.timestamps]).unique()
        
        if self.verbose:
            print(f"Fetching MAIAC AOD data for {len(unique_dates)} unique dates")
        
        headers = {"Authorization": f"Bearer {self.earthdata_token}"}
        min_lon, max_lon, min_lat, max_lat = self.extent
        
        maiac_params = {
            "short_name": "MCD19A2",
            "version": "061",
            "cmr_id": "C2324689816-LPCLOUD"
        }
        
        cmr_url = "https://cmr.earthdata.nasa.gov/search/granules.json"
        daily_maiac_data = {}
        
        for date in unique_dates:
            date_str = date.strftime('%Y-%m-%d')
            day_next = (date + timedelta(days=1)).strftime('%Y-%m-%d')
            
            try:
                params = {
                    "collection_concept_id": maiac_params["cmr_id"],
                    "temporal": f"{date_str}T00:00:00Z,{day_next}T00:00:00Z",
                    "bounding_box": f"{min_lon},{min_lat},{max_lon},{max_lat}",
                    "page_size": 10
                }
                
                response = requests.get(cmr_url, params=params, headers=headers)
                
                if response.status_code != 200:
                    if self.verbose:
                        print(f"Error searching for MAIAC AOD granules: HTTP {response.status_code}")
                    continue
                
                results = response.json()
                granules = results.get("feed", {}).get("entry", [])
                
                if not granules:
                    if self.verbose:
                        print(f"No MAIAC AOD data found for {date_str}")
                    continue
                
                for granule in granules:
                    download_url = next((link["href"] for link in granule.get("links", []) 
                                        if link.get("rel") == "http://esipfed.org/ns/fedsearch/1.1/data#"), None)
                    
                    if not download_url:
                        continue
                    
                    temp_file = os.path.join(self.cache_dir, f"maiac_temp_{date_str}.hdf")
                    
                    try:
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
                        
                        hdf_file = SD(temp_file, SDC.READ)
                        
                        try:
                            aod_dataset = hdf_file.select('Optical_Depth_047')
                            aod_data = aod_dataset[:]
                            
                            if self.verbose:
                                print(f"Successfully loaded Optical_Depth_047 for {date_str}")
                            
                            if aod_data.size == 0:
                                if self.verbose:
                                    print(f"Empty AOD data for {date_str}")
                                continue
                            
                            if self.verbose:
                                print(f"AOD data shape: {aod_data.shape}, range: {np.nanmin(aod_data):.6f} to {np.nanmax(aod_data):.6f}")
                            
                            aod_data = aod_data.astype(np.float32)
                            
                            if hasattr(aod_dataset, 'attributes'):
                                attrs = aod_dataset.attributes()
                                fill_value = attrs.get('_FillValue', None)
                                scale_factor = attrs.get('scale_factor', 1.0)
                                add_offset = attrs.get('add_offset', 0.0)
                                
                                if fill_value is not None:
                                    aod_data[aod_data == fill_value] = -1.0
                                    if self.verbose:
                                        fill_count = np.sum(aod_data == -1.0)
                                        total_count = aod_data.size
                                        print(f"Fill values: {fill_count}/{total_count} ({100*fill_count/total_count:.1f}%)")
                                
                                valid_mask = aod_data >= 0
                                if scale_factor != 1.0 or add_offset != 0.0:
                                    aod_data[valid_mask] = (aod_data[valid_mask] * scale_factor) + add_offset
                                    if self.verbose:
                                        print(f"Applied scaling: factor={scale_factor}, offset={add_offset}")
                            
                            if len(aod_data.shape) == 3:
                                valid_data = np.where(aod_data == -1.0, np.nan, aod_data)
                                aod_data_2d = np.nanmean(valid_data, axis=0)
                                aod_data_2d = np.where(np.isnan(aod_data_2d), -1.0, aod_data_2d)
                                if self.verbose:
                                    print(f"Averaged AOD data to shape: {aod_data_2d.shape}")
                            else:
                                aod_data_2d = aod_data
                            
                            # Quality control
                            noise_mask = (aod_data_2d >= -0.1) & (aod_data_2d < 0)
                            aod_data_2d[noise_mask] = 0.0
                            
                            unrealistic_mask = (aod_data_2d < -0.1) | (aod_data_2d > 5.0)
                            aod_data_2d[unrealistic_mask] = -1.0
                            
                            if self.verbose and np.any(unrealistic_mask):
                                unrealistic_count = np.sum(unrealistic_mask)
                                print(f"Set {unrealistic_count} unrealistic values to -1")
                            
                            zoom_y = self.dim / aod_data_2d.shape[0]
                            zoom_x = self.dim / aod_data_2d.shape[1]
                            
                            aod_for_zoom = np.where(aod_data_2d == -1.0, np.nan, aod_data_2d)
                            aod_grid = zoom(aod_for_zoom, (zoom_y, zoom_x), order=1, mode='nearest')
                            aod_grid = np.where(np.isnan(aod_grid), -1.0, aod_grid)
                            
                            if self.verbose:
                                print(f"Resized AOD data to shape: {aod_grid.shape}")
                                valid_pixels = np.sum(aod_grid >= 0)
                                total_pixels = aod_grid.size
                                print(f"Final coverage: {valid_pixels}/{total_pixels} ({100*valid_pixels/total_pixels:.1f}%)")
                                
                                if valid_pixels > 0:
                                    valid_data = aod_grid[aod_grid >= 0]
                                    print(f"Valid AOD range: {np.min(valid_data):.6f} to {np.max(valid_data):.6f}")
                            
                            daily_maiac_data[date] = aod_grid
                            
                            if self.verbose:
                                print(f"✅ Successfully processed AOD data for {date_str}")
                        
                        except Exception as e:
                            if self.verbose:
                                print(f"Error processing AOD dataset for {date_str}: {e}")
                                traceback.print_exc()
                        
                        finally:
                            if aod_dataset:
                                aod_dataset.end()
                            hdf_file.end()
                        
                        if date in daily_maiac_data:
                            break

                    except Exception as e:
                        if self.verbose:
                            print(f"Error processing MAIAC AOD data for {date_str}: {e}")
                            traceback.print_exc()
                    
                    finally:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                        
            except Exception as e:
                if self.verbose:
                    print(f"Error fetching MAIAC AOD data for {date_str}: {e}")
                    traceback.print_exc()
        
        for t_idx, timestamp in enumerate(self.timestamps):
            date = timestamp.date()
            if date in daily_maiac_data:
                maiac_data[t_idx, :, :, 0] = daily_maiac_data[date]
        
        if self.verbose:
            print(f"\nCreated final MAIAC AOD data with shape {maiac_data.shape}")
            
            all_data = maiac_data.flatten()
            valid_count = np.sum(all_data >= 0)
            missing_count = np.sum(all_data == -1.0)
            total_count = len(all_data)
            
            print(f"Final MAIAC statistics:")
            print(f"  Valid values: {valid_count:,} ({100*valid_count/total_count:.1f}%)")
            print(f"  Missing values (-1): {missing_count:,} ({100*missing_count/total_count:.1f}%)")
            
            if valid_count > 0:
                valid_data = all_data[all_data >= 0]
                print(f"  AOD range: {np.min(valid_data):.6f} to {np.max(valid_data):.6f}")
                print(f"  AOD mean: {np.mean(valid_data):.6f}")
                print(f"  AOD std: {np.std(valid_data):.6f}")
            
            print("✅ Data ready for ConvLSTM training (no NaN values)")
        
        self.save_to_cache(maiac_data, cache_file)
        return maiac_data