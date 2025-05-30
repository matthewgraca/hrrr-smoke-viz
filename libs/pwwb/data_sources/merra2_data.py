import os
import numpy as np
import pandas as pd
import earthaccess
import xarray as xr
from scipy.ndimage import zoom
import shutil
import traceback

from libs.pwwb.data_sources.base_data_source import BaseDataSource


class Merra2DataSource(BaseDataSource):
    """Fetches MERRA-2 atmospheric data (PBL height, surface temperature, heat exchange coefficient)."""
    
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
        Initialize MERRA-2 data source.
        
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
        
        self.native_data = None
        self.native_lats = None
        self.native_lons = None
    
    def get_data(self):
        """
        Fetch MERRA-2 data using EarthAccess, preserving both native and regridded resolution.
        
        Returns:
        --------
        numpy.ndarray
            MERRA-2 data (n_timestamps, dim, dim, 3) for PBL height, temperature, heat exchange
        """
        cache_file = os.path.join(self.cache_dir, f"{self.cache_prefix}merra2_data.npy")
        native_cache_file = os.path.join(self.cache_dir, f"{self.cache_prefix}merra2_native_data.npy")
        
        data = self.check_cache(cache_file, (self.n_timestamps, self.dim, self.dim, 3))
        if data is not None:
            if os.path.exists(native_cache_file):
                self.native_data = np.load(native_cache_file)
                
                lats_file = os.path.join(self.cache_dir, f"{self.cache_prefix}merra2_native_lats.npy")
                lons_file = os.path.join(self.cache_dir, f"{self.cache_prefix}merra2_native_lons.npy")
                
                if os.path.exists(lats_file) and os.path.exists(lons_file):
                    self.native_lats = np.load(lats_file)
                    self.native_lons = np.load(lons_file)
            
            return data
        
        merra2_data = np.zeros((self.n_timestamps, self.dim, self.dim, 3))
        merra2_native_data = np.zeros((self.n_timestamps, 4, 4, 3))
        merra2_native_lats = None
        merra2_native_lons = None

        if self.verbose:
            print(f"Fetching MERRA-2 data for period: {self.timestamps[0]} to {self.timestamps[-1]}")
        
        min_lon, max_lon, min_lat, max_lat = self.extent
        
        months_to_process = pd.DataFrame({'date': self.timestamps}).groupby(
            [self.timestamps.year, self.timestamps.month]
        ).groups.keys()
        
        var_mapping = {
            'PBLH': ['PBLH', 'PBL', 'ZPBL'],
            'T2M': ['T2M', 'T2', 'TLML'],
            'CDH': ['CDH', 'CH', 'CN']
        }
        
        for year, month in months_to_process:
            month_timestamps = [ts for ts in self.timestamps 
                                if ts.year == year and ts.month == month]
            
            if not month_timestamps:
                continue
                
            start_date = month_timestamps[0]
            end_date = month_timestamps[-1]
            
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            if self.verbose:
                print(f"Processing MERRA-2 data for period: {start_str} to {end_str}")
            
            try:
                auth = earthaccess.login()
                
                if not auth:
                    if self.verbose:
                        print("Failed to authenticate with Earth Data. Please check your credentials.")
                    continue
                
                results = earthaccess.search_data(
                    short_name="M2T1NXFLX",
                    version='5.12.4',       
                    temporal=(start_str, end_str),
                    bounding_box=(min_lon, min_lat, max_lon, max_lat)
                )
                
                if not results:
                    if self.verbose:
                        print(f"No MERRA-2 granules found for period: {start_str} to {end_str}")
                    continue
                
                if self.verbose:
                    print(f"Found {len(results)} MERRA-2 granules")
                
                temp_dir = os.path.join(self.cache_dir, f"merra2_temp_{year}_{month}")
                os.makedirs(temp_dir, exist_ok=True)
                
                downloaded_files = earthaccess.download(results, local_path=temp_dir)
                
                if not downloaded_files:
                    if self.verbose:
                        print("Failed to download MERRA-2 granules")
                    continue
                
                if self.verbose:
                    print(f"Downloaded {len(downloaded_files)} MERRA-2 files to {temp_dir}")
                
                try:
                    ds = xr.open_mfdataset(downloaded_files)
                    
                    if self.verbose:
                        print("MERRA-2 dataset opened successfully")
                        print("Available variables:", list(ds.data_vars))
                    
                    try:
                        lon_dim = next((dim for dim in ds.dims if 'lon' in dim.lower()), None)
                        lat_dim = next((dim for dim in ds.dims if 'lat' in dim.lower()), None)
                        
                        if lon_dim and lat_dim:
                            if self.verbose:
                                print(f"Found geographic dimensions: lon={lon_dim}, lat={lat_dim}")
                                print(f"Subsetting to extent: {min_lon} to {max_lon}, {min_lat} to {max_lat}")
                            
                            ds = ds.sel({lon_dim: slice(min_lon, max_lon), 
                                        lat_dim: slice(min_lat, max_lat)}).compute()
                            
                            if self.verbose:
                                print(f"Subset size: {ds.dims[lon_dim]} x {ds.dims[lat_dim]}")
                            
                            actual_lat_cells = ds.dims[lat_dim]
                            actual_lon_cells = ds.dims[lon_dim]
                            
                            if actual_lat_cells > merra2_native_data.shape[1] or actual_lon_cells > merra2_native_data.shape[2]:
                                if self.verbose:
                                    print(f"Resizing native data array to {actual_lat_cells} x {actual_lon_cells}")
                                merra2_native_data = np.zeros((self.n_timestamps, actual_lat_cells, actual_lon_cells, 3))
                        else:
                            if self.verbose:
                                print("Could not identify longitude and latitude dimensions.")
                    except Exception as e:
                        if self.verbose:
                            print(f"Error subsetting data: {e}")
                    
                    var_actual_names = {}
                    for var_key, possible_names in var_mapping.items():
                        for name in possible_names:
                            if name in ds.data_vars:
                                var_actual_names[var_key] = name
                                break
                    
                    if self.verbose:
                        print(f"Found variables: {var_actual_names}")
                    
                    missing_vars = set(['PBLH', 'T2M', 'CDH']) - set(var_actual_names.keys())
                    if missing_vars:
                        if self.verbose:
                            print(f"Missing required variables: {missing_vars}")
                    
                    times = ds.time.values
                    
                    for ts in month_timestamps:
                        t_idx = self.timestamps.get_loc(ts)
                        
                        np_ts = np.datetime64(ts)
                        time_diffs = np.abs(times - np_ts)
                        closest_time_idx = np.argmin(time_diffs)
                        closest_time = times[closest_time_idx]
                        
                        for var_idx, (var_key, var_name) in enumerate(var_actual_names.items()):
                            try:
                                data_slice = ds[var_name].sel(time=closest_time)
                                
                                if len(data_slice.shape) > 2:
                                    data_slice = data_slice.isel(lev=0) if 'lev' in data_slice.dims else data_slice[0]
                                
                                data_array = data_slice.values
                                
                                h, w = data_array.shape
                                h = min(h, merra2_native_data.shape[1])
                                w = min(w, merra2_native_data.shape[2])
                                merra2_native_data[t_idx, :h, :w, var_idx] = data_array[:h, :w]
                                
                                if t_idx == 0 and var_idx == 0 and lon_dim and lat_dim:
                                    merra2_native_lons = ds[lon_dim].values
                                    merra2_native_lats = ds[lat_dim].values
                                    
                                    if self.verbose:
                                        print(f"Stored native grid coordinates: {len(merra2_native_lons)} x {len(merra2_native_lats)}")
                                
                                if data_array.shape[0] != self.dim or data_array.shape[1] != self.dim:
                                    zoom_y = self.dim / data_array.shape[0]
                                    zoom_x = self.dim / data_array.shape[1]
                                    
                                    if self.verbose and t_idx == 0 and var_idx == 0:
                                        print(f"WARNING: Interpolating MERRA-2 data from {data_array.shape} to {self.dim}x{self.dim}")
                                        print(f"Upsampling factor: {zoom_y:.1f}x/{zoom_x:.1f}x")
                                        print("Consider using native resolution data for analysis")
                                    
                                    grid = zoom(data_array, (zoom_y, zoom_x), order=1, mode='nearest')
                                else:
                                    grid = data_array
                                
                                merra2_data[t_idx, :, :, var_idx] = grid
                                
                            except Exception as e:
                                if self.verbose:
                                    print(f"Error processing variable {var_name} for timestamp {ts}: {e}")
                                continue
                    
                    ds.close()
                    
                except Exception as e:
                    if self.verbose:
                        print(f"Error processing MERRA-2 data: {e}")
                        traceback.print_exc()
                
                finally:
                    if not self.use_cached_data:
                        try:
                            shutil.rmtree(temp_dir)
                            if self.verbose:
                                print(f"Cleaned up temporary directory: {temp_dir}")
                        except Exception as e:
                            if self.verbose:
                                print(f"Error cleaning up temp directory: {e}")
            
            except Exception as e:
                if self.verbose:
                    print(f"Error during MERRA-2 data fetch for {year}-{month}: {e}")
                    traceback.print_exc()
        
        # Temporal post-processing to improve data quality for time series analysis:
        # Fill missing timestamps to prevent gaps in the time series
        # Smooth consecutive measurements to reduce noise from satellite retrieval variations
        for t_idx in range(1, self.n_timestamps):
            if np.sum(np.abs(merra2_data[t_idx])) == 0 and np.sum(np.abs(merra2_data[t_idx-1])) > 0:
                merra2_data[t_idx] = merra2_data[t_idx-1]
                merra2_native_data[t_idx] = merra2_native_data[t_idx-1]
            elif np.sum(np.abs(merra2_data[t_idx])) > 0 and np.sum(np.abs(merra2_data[t_idx-1])) > 0:
                merra2_data[t_idx] = merra2_data[t_idx-1] * 0.2 + merra2_data[t_idx] * 0.8
                merra2_native_data[t_idx] = merra2_native_data[t_idx-1] * 0.2 + merra2_native_data[t_idx] * 0.8
        
        if self.verbose:
            print(f"Created MERRA-2 data with shape {merra2_data.shape}")
            print(f"Preserved native resolution data with shape {merra2_native_data.shape}")
        
        self.save_to_cache(merra2_data, cache_file)
        np.save(native_cache_file, merra2_native_data)
        
        if merra2_native_lats is not None and merra2_native_lons is not None:
            np.save(os.path.join(self.cache_dir, f"{self.cache_prefix}merra2_native_lats.npy"), merra2_native_lats)
            np.save(os.path.join(self.cache_dir, f"{self.cache_prefix}merra2_native_lons.npy"), merra2_native_lons)
        
        self.native_data = merra2_native_data
        self.native_lats = merra2_native_lats
        self.native_lons = merra2_native_lons
        
        return merra2_data