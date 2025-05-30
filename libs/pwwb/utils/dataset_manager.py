"""Dataset manager for PWWBData instances with registry functionality."""

import os
import json
from datetime import datetime
import pandas as pd


class PWWBDatasetManager:
    """Manages PWWBData datasets with persistent registry tracking."""
    
    def __init__(self, registry_file="pwwb_datasets.json", cache_dir=None):
        """
        Initialize dataset manager with registry and cache location.
        
        Parameters:
        -----------
        registry_file : str
            Path to JSON registry file
        cache_dir : str, optional
            Cache directory path
        """
        self.registry_file = registry_file
        self.cache_dir = cache_dir
        self.registry = self._load_registry()
    
    def _load_registry(self):
        """Load registry from JSON file, returning empty dict if file missing or corrupted."""
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading registry: {e}")
                return {}
        return {}
    
    def _save_registry(self):
        """Save current registry state to JSON file."""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_dataset(self, name, description, params):
        """
        Register a new dataset with creation timestamp and parameters.
        
        Parameters:
        -----------
        name : str
            Unique dataset identifier
        description : str
            Human-readable dataset description
        params : dict
            PWWBData constructor parameters
            
        Returns:
        --------
        str
            Dataset name (for chaining)
        """
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
        Retrieve complete dataset information by name.
        
        Parameters:
        -----------
        name : str
            Dataset name
            
        Returns:
        --------
        dict or None
            Dataset metadata and parameters, or None if not found
        """
        if name not in self.registry:
            print(f"Dataset '{name}' not found in registry.")
            return None
        
        return self.registry[name]
    
    def list_datasets(self):
        """
        Get summary of all registered datasets as a DataFrame.
        
        Returns:
        --------
        pandas.DataFrame
            Columns: name, created, description, start_date, end_date, channels
        """
        if not self.registry:
            print("No datasets registered.")
            return pd.DataFrame()
        
        data = []
        for name, details in self.registry.items():
            created = details.get('created', 'Unknown')
            description = details.get('description', 'No description')
            start_date = details.get('parameters', {}).get('start_date', 'Unknown')
            end_date = details.get('parameters', {}).get('end_date', 'Unknown')
            
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
        
        return pd.DataFrame(data)
    
    def delete_dataset(self, name, delete_files=False):
        """
        Remove dataset from registry and optionally delete associated cache files.
        
        Parameters:
        -----------
        name : str
            Dataset name
        delete_files : bool
            If True, delete cache files with matching prefix
            
        Returns:
        --------
        bool
            True if deletion successful
        """
        if name not in self.registry:
            print(f"Dataset '{name}' not found in registry.")
            return False
        
        if delete_files and self.cache_dir:
            deleted_files = []
            
            for filename in os.listdir(self.cache_dir):
                if filename.startswith(name):
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
        
        del self.registry[name]
        self._save_registry()
        
        print(f"Dataset '{name}' deleted from registry.")
        return True
    
    def create_dataset(self, name, description, PWWBData_class, **kwargs):
        """
        Create and register a new PWWBData instance.
        
        Parameters:
        -----------
        name : str
            Dataset name
        description : str
            Dataset description
        PWWBData_class : class
            PWWBData class to instantiate
        **kwargs
            Arguments passed to PWWBData constructor
            
        Returns:
        --------
        PWWBData
            Created and configured PWWBData instance
        """
        self.register_dataset(name, description, kwargs)
        
        if 'cache_dir' not in kwargs and self.cache_dir:
            kwargs['cache_dir'] = self.cache_dir
            
        kwargs['cache_prefix'] = name
        
        return PWWBData_class(**kwargs)
    
    def load_dataset(self, name, PWWBData_class):
        """
        Load existing dataset by recreating PWWBData instance from registry parameters.
        
        Parameters:
        -----------
        name : str
            Dataset name
        PWWBData_class : class
            PWWBData class to instantiate
            
        Returns:
        --------
        PWWBData or None
            Loaded PWWBData instance, or None if dataset not found
        """
        info = self.get_dataset_info(name)
        if not info:
            return None
        
        params = info.get('parameters', {})
        params['cache_prefix'] = name
        
        if 'cache_dir' not in params and self.cache_dir:
            params['cache_dir'] = self.cache_dir
        
        instance = PWWBData_class(**params)
        
        success = instance.load_data()
        
        if not success:
            print(f"Warning: Created PWWBData instance but could not load cached data.")
            return instance
        
        return instance


def create_dataset_manager(registry_file="pwwb_datasets.json", cache_dir="data/pwwb_cache/"):
    """
    Create a dataset manager with specified registry and cache locations.
    
    Parameters:
    -----------
    registry_file : str
        Registry JSON file path
    cache_dir : str
        Cache directory path
        
    Returns:
    --------
    PWWBDatasetManager
        Configured dataset manager
    """
    os.makedirs(cache_dir, exist_ok=True)
    return PWWBDatasetManager(registry_file, cache_dir)