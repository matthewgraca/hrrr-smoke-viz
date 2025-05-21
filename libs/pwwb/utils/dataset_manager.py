"""
Dataset manager for PWWBData.

This module provides functions to help manage multiple PWWBData instances,
especially useful when working in Jupyter notebooks.
"""

import os
import json
from datetime import datetime
import pandas as pd

class PWWBDatasetManager:
    """
    Manager for PWWBData datasets with registry functionality.
    """
    
    def __init__(self, registry_file="pwwb_datasets.json", cache_dir=None):
        """
        Initialize the dataset manager.
        
        Parameters:
        -----------
        registry_file : str
            Path to the registry file
        cache_dir : str, optional
            Path to the cache directory
        """
        self.registry_file = registry_file
        self.cache_dir = cache_dir
        self.registry = self._load_registry()
    
    def _load_registry(self):
        """Load the registry from the JSON file."""
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading registry: {e}")
                return {}
        return {}
    
    def _save_registry(self):
        """Save the registry to the JSON file."""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_dataset(self, name, description, params):
        """
        Register a dataset in the registry.
        
        Parameters:
        -----------
        name : str
            Name of the dataset
        description : str
            Description of the dataset
        params : dict
            Parameters used to create the dataset
            
        Returns:
        --------
        str
            The name of the registered dataset
        """
        # Create entry with timestamp and details
        self.registry[name] = {
            'created': datetime.now().isoformat(),
            'description': description,
            'parameters': params,
            'prefix': name
        }
        
        self._save_registry()
        return name
    
    def get_dataset_info(self, name):
        """
        Get detailed information about a specific dataset.
        
        Parameters:
        -----------
        name : str
            Name of the dataset
            
        Returns:
        --------
        dict or None
            Dataset information if found, None otherwise
        """
        if name not in self.registry:
            print(f"Dataset '{name}' not found in registry.")
            return None
        
        return self.registry[name]
    
    def list_datasets(self):
        """
        List all registered datasets.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with dataset information
        """
        if not self.registry:
            print("No datasets registered.")
            return pd.DataFrame()
        
        # Create a list of dictionaries for each dataset
        data = []
        for name, details in self.registry.items():
            created = details.get('created', 'Unknown')
            description = details.get('description', 'No description')
            start_date = details.get('parameters', {}).get('start_date', 'Unknown')
            end_date = details.get('parameters', {}).get('end_date', 'Unknown')
            
            # Get channels included
            include_channels = details.get('parameters', {}).get('include_channels', {})
            if isinstance(include_channels, dict):
                channels = [k for k, v in include_channels.items() if v]
            elif isinstance(include_channels, list):
                channels = include_channels
            else:
                channels = []
            
            data.append({
                'name': name,
                'created': created,
                'description': description,
                'start_date': start_date,
                'end_date': end_date,
                'channels': ', '.join(channels) if channels else 'All'
            })
        
        # Convert to DataFrame and return
        df = pd.DataFrame(data)
        return df
    
    def delete_dataset(self, name, delete_files=False):
        """
        Delete a dataset from the registry.
        
        Parameters:
        -----------
        name : str
            Name of the dataset
        delete_files : bool, optional
            Whether to also delete the associated files
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        if name not in self.registry:
            print(f"Dataset '{name}' not found in registry.")
            return False
        
        # Check if we should delete the files
        if delete_files and self.cache_dir:
            prefix = name
            deleted_files = []
            
            # Loop through files in cache directory and delete those with matching prefix
            for filename in os.listdir(self.cache_dir):
                if filename.startswith(prefix):
                    filepath = os.path.join(self.cache_dir, filename)
                    try:
                        os.remove(filepath)
                        deleted_files.append(filename)
                    except Exception as e:
                        print(f"Error deleting file {filepath}: {e}")
            
            if deleted_files:
                print(f"Deleted {len(deleted_files)} files:")
                for filename in deleted_files:
                    print(f"  - {filename}")
        
        # Remove from registry
        del self.registry[name]
        self._save_registry()
        
        print(f"Dataset '{name}' deleted from registry.")
        return True
    
    def create_dataset(self, name, description, PWWBData_class, **kwargs):
        """
        Create a new PWWBData instance and register it.
        
        Parameters:
        -----------
        name : str
            Name of the dataset
        description : str
            Description of the dataset
        PWWBData_class : class
            The PWWBData class to instantiate
        **kwargs : dict
            Additional arguments to pass to PWWBData constructor
            
        Returns:
        --------
        object
            The created PWWBData instance
        """
        # Register the dataset
        self.register_dataset(name, description, kwargs)
        
        # Set up PWWBData constructor arguments
        if 'cache_dir' not in kwargs and self.cache_dir:
            kwargs['cache_dir'] = self.cache_dir
            
        # Create the PWWBData instance
        kwargs['cache_prefix'] = name
        
        # Create and return the instance
        return PWWBData_class(**kwargs)
    
    def load_dataset(self, name, PWWBData_class):
        """
        Load a dataset by name.
        
        Parameters:
        -----------
        name : str
            Name of the dataset
        PWWBData_class : class
            The PWWBData class to instantiate
            
        Returns:
        --------
        object or None
            The loaded PWWBData instance if successful, None otherwise
        """
        info = self.get_dataset_info(name)
        if not info:
            return None
        
        # Get parameters from registry
        params = info.get('parameters', {})
        
        # Add cache prefix
        params['cache_prefix'] = name
        
        # Add cache directory if available
        if 'cache_dir' not in params and self.cache_dir:
            params['cache_dir'] = self.cache_dir
        
        # Create PWWBData instance with saved parameters
        instance = PWWBData_class(**params)
        
        # Try to load the data
        success = instance.load_data()
        
        if not success:
            print(f"Warning: Created PWWBData instance but could not load cached data.")
            return instance
        
        return instance


# Convenience functions for direct use
def create_dataset_manager(registry_file="pwwb_datasets.json", cache_dir="data/pwwb_cache/"):
    """
    Create a new dataset manager.
    
    Parameters:
    -----------
    registry_file : str
        Path to the registry file
    cache_dir : str
        Path to the cache directory
        
    Returns:
    --------
    PWWBDatasetManager
        The created manager
    """
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    
    return PWWBDatasetManager(registry_file, cache_dir)