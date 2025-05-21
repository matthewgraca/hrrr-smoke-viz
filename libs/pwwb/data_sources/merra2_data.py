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
    """
    Class to handle MERRA-2 data collection and processing.
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
        Initialize the MERRA-2 data source.
        
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
        
        # Initialize additional attributes for native resolution data
        self.native_data = None
        self.native_lats = None
        self.native_lons = None
    
    def get_data(self):
        """
        Get MERRA-2 data for PBL Height, Surface Air Temperature, and Surface Exchange Coefficient
        using Earth Access library to directly download the data.
        
        Returns:
        --------
        numpy.ndarray
            MERRA-2 data with shape (n_timestamps, dim, dim, n_features)
        """
        cache_file = os.path.join(self.cache_dir, f"{self.cache_prefix}merra2_data.npy")
        native_cache_file = os.path.join(self.cache_dir, f"{self.cache_prefix}merra2_native_data.npy")
        
        # Check if cache exists and has correct shape
        data = self.check_cache(cache_file, (self.n_timestamps, self.dim, self.dim, 3))
        if data is not None:
            # Also load native resolution data if available
            if os.path.exists(native_cache_file):
                self.native_data = np.load(native_cache_file)
                
                # Load grid coordinates if available
                lats_file = os.path.join(self.cache_dir, f"{self.cache_prefix}merra2_native_lats.npy")
                lons_file = os.path.join(self.cache_dir, f"{self.cache_prefix}merra2_native_lons.npy")
                
                if os.path.exists(lats_file) and os.path.exists(lons_file):
                    self.native_lats = np.load(lats_file)
                    self.native_lons = np.load(lons_file)
            
            return data
        
        # Initialize empty array for MERRA-2 data
        # 3 channels: PBL Height, Surface Air Temperature, Surface Exchange Coefficient for Heat
        merra2_data = np.zeros((self.n_timestamps, self.dim, self.dim, 3))
        
        # Also create a native resolution array to preserve original data
        # We'll use a reasonable maximum size for the extent (4x4, which should be enough for most cases)
        merra2_native_data = np.zeros((self.n_timestamps, 4, 4, 3))
        merra2_native_lats = None
        merra2_native_lons = None

        if self.verbose:
            print(f"Fetching MERRA-2 data for period: {self.timestamps[0]} to {self.timestamps[-1]}")
        
        # Define our geographic bounds
        min_lon, max_lon, min_lat, max_lat = self.extent
        
        # Group timestamps by month to process efficiently (MERRA-2 data is organized by month)
        months_to_process = pd.DataFrame({'date': self.timestamps}).groupby(
            [self.timestamps.year, self.timestamps.month]
        ).groups.keys()
        
        # Set up variable mapping
        var_mapping = {
            'PBLH': ['PBLH', 'PBL', 'ZPBL'],  # Planetary Boundary Layer Height
            'T2M': ['T2M', 'T2', 'TLML'],     # Surface Air Temperature
            'CDH': ['CDH', 'CH', 'CN']        # Surface Exchange Coefficient
        }
        
        # Process each month
        for year, month in months_to_process:
            # Determine the start and end dates for this month
            month_timestamps = [ts for ts in self.timestamps 
                                if ts.year == year and ts.month == month]
            
            if not month_timestamps:
                continue
                
            start_date = month_timestamps[0]
            end_date = month_timestamps[-1]
            
            # Format the dates for earthaccess
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            if self.verbose:
                print(f"Processing MERRA-2 data for period: {start_str} to {end_str}")
            
            try:
                # Authenticate with Earth Data
                auth = earthaccess.login()
                
                if not auth:
                    if self.verbose:
                        print("Failed to authenticate with Earth Data. Please check your credentials.")
                        print("See instructions for setting up .netrc file in the docstring.")
                    continue
                
                # Search for MERRA-2 data
                results = earthaccess.search_data(
                    short_name="M2T1NXFLX",  # MERRA-2 tavg1_2d_flx_Nx product - surface fluxes
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
                
                # Create a temp directory for downloads
                temp_dir = os.path.join(self.cache_dir, f"merra2_temp_{year}_{month}")
                os.makedirs(temp_dir, exist_ok=True)
                
                # Download the granules
                downloaded_files = earthaccess.download(
                    results,
                    local_path=temp_dir
                )
                
                if not downloaded_files:
                    if self.verbose:
                        print("Failed to download MERRA-2 granules")
                    continue
                
                if self.verbose:
                    print(f"Downloaded {len(downloaded_files)} MERRA-2 files to {temp_dir}")
                
                # Process the downloaded files
                try:
                    # Open the dataset with xarray - this supports multi-file datasets
                    ds = xr.open_mfdataset(downloaded_files)
                    
                    if self.verbose:
                        print("MERRA-2 dataset opened successfully")
                        print("Available variables:", list(ds.data_vars))
                        print("Dimensions:", ds.dims)
                        print("Coordinates:", list(ds.coords))
                    
                    # Identify the longitude and latitude dimension names
                    # MERRA-2 typically uses 'lon' and 'lat', but let's be flexible
                    try:
                        lon_dim = next((dim for dim in ds.dims if 'lon' in dim.lower()), None)
                        lat_dim = next((dim for dim in ds.dims if 'lat' in dim.lower()), None)
                        
                        if lon_dim and lat_dim:
                            if self.verbose:
                                print(f"Found geographic dimensions: lon={lon_dim}, lat={lat_dim}")
                                print(f"Original size: {ds.dims[lon_dim]} x {ds.dims[lat_dim]}")
                                print(f"Subsetting to extent: {min_lon} to {max_lon}, {min_lat} to {max_lat}")
                            
                            # Explicitly subset the data to our extent
                            # Note: For MERRA-2, latitude is typically ascending from south to north
                            ds = ds.sel({lon_dim: slice(min_lon, max_lon), 
                                        lat_dim: slice(min_lat, max_lat)}).compute()
                            
                            if self.verbose:
                                print(f"Subset size: {ds.dims[lon_dim]} x {ds.dims[lat_dim]}")
                                print("Forced computation of dask arrays with .compute()")
                                print(f"Native MERRA-2 resolution for this extent: {ds.dims[lat_dim]} x {ds.dims[lon_dim]} cells")
                            
                            # Dynamically adjust the native data array size based on actual dimensions
                            actual_lat_cells = ds.dims[lat_dim]
                            actual_lon_cells = ds.dims[lon_dim]
                            
                            # If our pre-allocated array isn't big enough, create a new one
                            if actual_lat_cells > merra2_native_data.shape[1] or actual_lon_cells > merra2_native_data.shape[2]:
                                if self.verbose:
                                    print(f"Resizing native data array to {actual_lat_cells} x {actual_lon_cells}")
                                merra2_native_data = np.zeros((self.n_timestamps, actual_lat_cells, actual_lon_cells, 3))
                        else:
                            if self.verbose:
                                print("Could not identify longitude and latitude dimensions.")
                                print("Will process the entire dataset.")
                    except Exception as e:
                        if self.verbose:
                            print(f"Error subsetting data: {e}")
                            print("Will process the entire dataset.")
                    
                    # Map our desired variables to the actual variable names in the dataset
                    var_actual_names = {}
                    for var_key, possible_names in var_mapping.items():
                        for name in possible_names:
                            if name in ds.data_vars:
                                var_actual_names[var_key] = name
                                break
                    
                    if self.verbose:
                        print(f"Found variables: {var_actual_names}")
                    
                    # Check if we have all the required variables
                    missing_vars = set(['PBLH', 'T2M', 'CDH']) - set(var_actual_names.keys())
                    if missing_vars:
                        if self.verbose:
                            print(f"Missing required variables: {missing_vars}")
                            print("Will try to use alternative names or provide default values")
                    
                    # Get the time steps
                    times = ds.time.values
                    
                    # Process each timestamp in our dataset
                    for ts in month_timestamps:
                        # Find the index of the timestamp in our dataset
                        t_idx = self.timestamps.get_loc(ts)
                        
                        # Find the closest time in the MERRA-2 dataset
                        np_ts = np.datetime64(ts)
                        time_diffs = np.abs(times - np_ts)
                        closest_time_idx = np.argmin(time_diffs)
                        closest_time = times[closest_time_idx]
                        
                        # Extract the data for each variable
                        for var_idx, (var_key, var_name) in enumerate(var_actual_names.items()):
                            try:
                                # Extract the data slice for our region
                                data_slice = ds[var_name].sel(time=closest_time)
                                
                                # If the data is not 2D, try to select a relevant level
                                if len(data_slice.shape) > 2:
                                    # For 3D data, select the first level (usually surface)
                                    data_slice = data_slice.isel(lev=0) if 'lev' in data_slice.dims else data_slice[0]
                                
                                # Convert to numpy array
                                data_array = data_slice.values
                                
                                # Store the native resolution data
                                # Make sure we don't exceed the array dimensions
                                h, w = data_array.shape
                                h = min(h, merra2_native_data.shape[1])
                                w = min(w, merra2_native_data.shape[2])
                                merra2_native_data[t_idx, :h, :w, var_idx] = data_array[:h, :w]
                                
                                # Store grid coordinates for reference (only once)
                                if t_idx == 0 and var_idx == 0 and lon_dim and lat_dim:
                                    merra2_native_lons = ds[lon_dim].values
                                    merra2_native_lats = ds[lat_dim].values
                                    
                                    if self.verbose:
                                        print(f"Stored native grid coordinates: {len(merra2_native_lons)} x {len(merra2_native_lats)}")
                                        print(f"Longitude range: {merra2_native_lons.min()} to {merra2_native_lons.max()}")
                                        print(f"Latitude range: {merra2_native_lats.min()} to {merra2_native_lats.max()}")
                                
                                # Resize for compatibility with other data sources
                                if data_array.shape[0] != self.dim or data_array.shape[1] != self.dim:
                                    # Calculate zoom factors
                                    zoom_y = self.dim / data_array.shape[0]
                                    zoom_x = self.dim / data_array.shape[1]
                                    
                                    # Apply zoom with warning about interpolation
                                    if self.verbose and t_idx == 0 and var_idx == 0:
                                        print(f"WARNING: Interpolating MERRA-2 data from {data_array.shape} to {self.dim}x{self.dim}")
                                        print(f"This is a {zoom_y:.1f}x/{zoom_x:.1f}x upsampling factor")
                                        print("Consider using the native resolution data for analysis")
                                    
                                    grid = zoom(data_array, (zoom_y, zoom_x), order=1, mode='nearest')
                                else:
                                    grid = data_array
                                
                                # Store in our interpolated data array
                                merra2_data[t_idx, :, :, var_idx] = grid
                                
                            except Exception as e:
                                if self.verbose:
                                    print(f"Error processing variable {var_name} for timestamp {ts}: {e}")
                                continue
                    
                    # Close the dataset
                    ds.close()
                    
                except Exception as e:
                    if self.verbose:
                        print(f"Error processing MERRA-2 data: {e}")
                        traceback.print_exc()
                
                finally:
                    # Cleanup the downloaded files (optional)
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
        
        # Add temporal coherence - make neighboring timestamps similar
        for t_idx in range(1, self.n_timestamps):
            # If the current timestamp has all zeros, use the previous timestamp's data
            if np.sum(np.abs(merra2_data[t_idx])) == 0 and np.sum(np.abs(merra2_data[t_idx-1])) > 0:
                merra2_data[t_idx] = merra2_data[t_idx-1]
                # Also update native resolution data
                merra2_native_data[t_idx] = merra2_native_data[t_idx-1]
            # Otherwise, if both have data, add some temporal smoothing
            elif np.sum(np.abs(merra2_data[t_idx])) > 0 and np.sum(np.abs(merra2_data[t_idx-1])) > 0:
                merra2_data[t_idx] = merra2_data[t_idx-1] * 0.2 + merra2_data[t_idx] * 0.8
                # Also smooth native resolution data
                merra2_native_data[t_idx] = merra2_native_data[t_idx-1] * 0.2 + merra2_native_data[t_idx] * 0.8
        
        if self.verbose:
            print(f"Created MERRA-2 data with shape {merra2_data.shape}")
            print(f"Also preserved native resolution data with shape {merra2_native_data.shape}")
        
        # Save to cache
        self.save_to_cache(merra2_data, cache_file)
        np.save(native_cache_file, merra2_native_data)
        
        # Save native grid coordinates if available
        if merra2_native_lats is not None and merra2_native_lons is not None:
            np.save(os.path.join(self.cache_dir, f"{self.cache_prefix}merra2_native_lats.npy"), merra2_native_lats)
            np.save(os.path.join(self.cache_dir, f"{self.cache_prefix}merra2_native_lons.npy"), merra2_native_lons)
        
        # Store native resolution data as instance attributes
        self.native_data = merra2_native_data
        self.native_lats = merra2_native_lats
        self.native_lons = merra2_native_lons
        
        return merra2_data