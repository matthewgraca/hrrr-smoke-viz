import os
import numpy as np
import json
import hashlib

def generate_cache_key(params_dict):
    """
    Generate a unique cache key based on parameter values.
    
    Parameters:
    -----------
    params_dict : dict
        Dictionary of parameters to include in the cache key
        
    Returns:
    --------
    str
        Unique hash string to use as cache key
    """
    # Convert dictionary to a stable string representation
    param_str = json.dumps(params_dict, sort_keys=True)
    
    # Generate hash
    hash_obj = hashlib.md5(param_str.encode())
    return hash_obj.hexdigest()

def save_to_cache(data, cache_file, metadata=None):
    """
    Save data to a cache file with optional metadata.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Data to save
    cache_file : str
        Path to the cache file
    metadata : dict, optional
        Additional metadata to save with the data
    """
    # Save the numpy array
    np.save(cache_file, data)
    
    # If metadata is provided, save it to a separate JSON file
    if metadata:
        metadata_file = os.path.splitext(cache_file)[0] + '_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

def load_from_cache(cache_file, expected_shape=None, verify_shape=True):
    """
    Load data from a cache file with optional shape verification.
    
    Parameters:
    -----------
    cache_file : str
        Path to the cache file
    expected_shape : tuple, optional
        Expected shape of the data array
    verify_shape : bool, optional
        Whether to verify the shape of the loaded data
        
    Returns:
    --------
    numpy.ndarray or None
        Data from cache if available and valid, None otherwise
    """
    if not os.path.exists(cache_file):
        return None
        
    try:
        data = np.load(cache_file)
        
        # Verify shape if requested
        if verify_shape and expected_shape is not None:
            if data.shape != expected_shape:
                return None
        
        return data
    except Exception as e:
        print(f"Error loading cached data: {e}")
        return None

def load_metadata(cache_file):
    """
    Load metadata for a cached file.
    
    Parameters:
    -----------
    cache_file : str
        Path to the cache file
        
    Returns:
    --------
    dict or None
        Metadata dictionary if available, None otherwise
    """
    metadata_file = os.path.splitext(cache_file)[0] + '_metadata.json'
    if not os.path.exists(metadata_file):
        return None
        
    try:
        with open(metadata_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None

def clear_cache(cache_dir, prefix=None):
    """
    Clear all cache files in the specified directory.
    
    Parameters:
    -----------
    cache_dir : str
        Directory containing cache files
    prefix : str, optional
        Only clear files with this prefix
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
    Get the total size of all cache files in the specified directory.
    
    Parameters:
    -----------
    cache_dir : str
        Directory containing cache files
    prefix : str, optional
        Only include files with this prefix
        
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

def prune_cache(cache_dir, max_size_bytes, keep_prefix=None):
    """
    Remove oldest cache files to keep total size under limit.
    
    Parameters:
    -----------
    cache_dir : str
        Directory containing cache files
    max_size_bytes : int
        Maximum allowed total size in bytes
    keep_prefix : str, optional
        Don't remove files with this prefix
    """
    if not os.path.exists(cache_dir):
        return
        
    # Get list of all cache files with their modification times
    files = []
    for filename in os.listdir(cache_dir):
        if keep_prefix and filename.startswith(keep_prefix):
            continue
            
        filepath = os.path.join(cache_dir, filename)
        if os.path.isfile(filepath):
            mtime = os.path.getmtime(filepath)
            size = os.path.getsize(filepath)
            files.append((filepath, mtime, size))
    
    # Sort by modification time (oldest first)
    files.sort(key=lambda x: x[1])
    
    # Calculate current total size
    total_size = sum(size for _, _, size in files)
    
    # Remove oldest files until under size limit
    for filepath, _, size in files:
        if total_size <= max_size_bytes:
            break
            
        try:
            os.remove(filepath)
            total_size -= size
        except Exception as e:
            print(f"Error removing cache file {filepath}: {e}")