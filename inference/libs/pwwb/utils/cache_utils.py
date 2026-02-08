import os
import numpy as np


def save_to_cache(data, cache_file):
    """
    Save numpy array to cache file.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Array data to cache
    cache_file : str
        Output file path
    """
    np.save(cache_file, data)


def load_from_cache(cache_file, expected_shape=None, verify_shape=True):
    """
    Load numpy array from cache with optional shape validation.
    
    Parameters:
    -----------
    cache_file : str
        Cache file path
    expected_shape : tuple, optional
        Required array shape for validation
    verify_shape : bool
        Whether to validate array shape
        
    Returns:
    --------
    numpy.ndarray or None
        Cached array if valid, None if missing or invalid
    """
    if not os.path.exists(cache_file):
        return None
        
    try:
        data = np.load(cache_file)
        
        if verify_shape and expected_shape is not None:
            if data.shape != expected_shape:
                return None
        
        return data
    except Exception as e:
        print(f"Error loading cached data: {e}")
        return None


def clear_cache(cache_dir, prefix=None):
    """
    Remove cache files from directory, optionally filtered by prefix.
    
    Parameters:
    -----------
    cache_dir : str
        Directory containing cache files
    prefix : str, optional
        Only remove files starting with this prefix
    """
    if not os.path.exists(cache_dir):
        return
        
    for filename in os.listdir(cache_dir):
        if prefix and not filename.startswith(prefix):
            continue
            
        filepath = os.path.join(cache_dir, filename)
        if os.path.isfile(filepath):
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"Error removing cache file {filepath}: {e}")


def get_cache_size(cache_dir, prefix=None):
    """
    Calculate total size of cache files in bytes.
    
    Parameters:
    -----------
    cache_dir : str
        Directory containing cache files
    prefix : str, optional
        Only include files starting with this prefix
        
    Returns:
    --------
    int
        Total size in bytes
    """
    if not os.path.exists(cache_dir):
        return 0
        
    total_size = 0
    for filename in os.listdir(cache_dir):
        if prefix and not filename.startswith(prefix):
            continue
            
        filepath = os.path.join(cache_dir, filename)
        if os.path.isfile(filepath):
            total_size += os.path.getsize(filepath)
            
    return total_size