import os
import numpy as np
import pandas as pd

class BaseDataSource:
    """
    Base class for all data sources with common functionality.
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
        Initialize the base data source.
        
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
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_data(self):
        """
        Get data for this source. Must be implemented by subclasses.
        
        Returns:
        --------
        numpy.ndarray
            Data with shape (n_timestamps, dim, dim, n_features)
        """
        raise NotImplementedError("Subclasses must implement get_data method")
    
    def check_cache(self, cache_file, expected_shape):
        """
        Check if a cache file exists and has the expected shape.
        
        Parameters:
        -----------
        cache_file : str
            Path to the cache file
        expected_shape : tuple
            Expected shape of the data array
            
        Returns:
        --------
        numpy.ndarray or None
            Data from cache if available and valid, None otherwise
        """
        if self.use_cached_data and os.path.exists(cache_file):
            if self.verbose:
                print(f"Loading cached data from {cache_file}")
            data = np.load(cache_file)
            
            # Verify the shape matches what we expect
            if data.shape[0] == expected_shape[0] and data.shape[3] == expected_shape[3]:
                return data
            else:
                if self.verbose:
                    print(f"Cached data has shape {data.shape}, but expected {expected_shape}")
                    print("Creating a new array with the correct dimensions")
                return None
        return None
    
    def save_to_cache(self, data, cache_file):
        """
        Save data to a cache file.
        
        Parameters:
        -----------
        data : numpy.ndarray
            Data to save
        cache_file : str
            Path to the cache file
        """
        if self.verbose:
            print(f"Saving data to cache: {cache_file}")
        np.save(cache_file, data)