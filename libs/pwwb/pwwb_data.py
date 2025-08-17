import os
import numpy as np
import pandas as pd
from datetime import datetime
import json

# Import the data source classes
from libs.pwwb.data_sources.maiac_data import MaiacDataSource
from libs.pwwb.data_sources.tropomi_data import TropomiDataSource
from libs.pwwb.data_sources.modis_fire_data import ModisFireDataSource
from libs.pwwb.data_sources.merra2_data import Merra2DataSource
from libs.pwwb.data_sources.metar_data import MetarDataSource

class PWWBData:
    def __init__(
        self,
        start_date="2018-01-01",
        end_date="2020-12-31",
        extent=(-118.75, -117.5, 33.5, 34.5),  # Default to LA County bounds
        dim=200,  # Spatial resolution
        cache_dir='data/pwwb_cache/',
        use_cached_data=True,
        verbose=False,
        env_file='.env',
        output_dir=None,
        include_channels=None,  # Parameter for channel selection
        cache_prefix=None  # Parameter for custom cache prefix
    ):
        self.start_date = pd.to_datetime(start_date)
        self.orig_end_date = pd.to_datetime(end_date)
        self.end_date = self.orig_end_date - pd.Timedelta(hours=1)  # Adjust to last hour
        self.extent = extent
        self.dim = dim
        self.verbose = verbose
        self.cache_dir = cache_dir
        self.use_cached_data = use_cached_data
        self.output_dir = output_dir
        
        # Generate a cache prefix based on start/end dates if not provided
        if cache_prefix is None:
            self.cache_prefix = f"{self.start_date.strftime('%Y%m%d')}-{self.end_date.strftime('%Y%m%d')}_"
        else:
            self.cache_prefix = f"{cache_prefix}_"
            
        if self.verbose:
            print(f"Using cache prefix: {self.cache_prefix}")
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Define all available channels and their groupings - including wind components
        self.available_channels = {
            'maiac': ['MAIAC_AOD'],
            'tropomi': ['TROPOMI_Methane', 'TROPOMI_NO2', 'TROPOMI_CO'],
            'modis_fire': ['MODIS_FRP'],
            'merra2': ['MERRA2_PBL_Height', 'MERRA2_Surface_Air_Temp', 'MERRA2_Surface_Exchange_Coef'],
            'metar': ['METAR_Wind_Speed', 'METAR_Wind_Direction', 'METAR_Wind_U', 'METAR_Wind_V',
                      'METAR_Precipitation', 'METAR_Humidity', 'METAR_Heat_Index', 'METAR_Air_Temp', 
                      'METAR_Air_Pressure', 'METAR_Dew_Point']
        }
        
        # Set up what channels to include
        if include_channels is None:
            # Default: include all channels
            self.include_channels = {
                'maiac': True,
                'tropomi': True, 
                'modis_fire': True,
                'merra2': True,
                'metar': True
            }
            self.tropomi_channels = ['TROPOMI_Methane', 'TROPOMI_NO2', 'TROPOMI_CO']
        else:
            # Initialize all to False
            self.include_channels = {
                'maiac': False,
                'tropomi': False, 
                'modis_fire': False,
                'merra2': False,
                'metar': False
            }
            
            # Parse the include_channels parameter
            if isinstance(include_channels, dict):
                # If a dictionary is provided, use it directly for fine-grained control
                # e.g. {'maiac': True, 'tropomi': ['TROPOMI_NO2']}
                for key, value in include_channels.items():
                    if key in self.include_channels:
                        if isinstance(value, bool):
                            # Boolean flag for entire group
                            self.include_channels[key] = value
                        elif isinstance(value, list):
                            # List of specific channels in the group
                            self.include_channels[key] = True  # Mark the group as included
                            # Store the specific channels to include
                            setattr(self, f"{key}_channels", value)
            elif isinstance(include_channels, list):
                # If a list is provided, assume it contains group names to include
                # e.g. ['maiac', 'tropomi']
                for group in include_channels:
                    if group in self.include_channels:
                        self.include_channels[group] = True
        
        # Set specific tropomi channels if provided
        if not hasattr(self, 'tropomi_channels') and self.include_channels['tropomi']:
            self.tropomi_channels = ['TROPOMI_Methane', 'TROPOMI_NO2', 'TROPOMI_CO']
        
        # Generate timestamps at hourly intervals
        self.timestamps = pd.date_range(self.start_date, self.end_date, freq='h')
        self.n_timestamps = len(self.timestamps)
        
        if self.verbose:
            print(f"Initialized PWWBData with {self.n_timestamps} hourly timestamps")
            print(f"Date range: {self.start_date} to {self.end_date}")
            print(f"Channels included: {[k for k, v in self.include_channels.items() if v]}")
            if hasattr(self, 'tropomi_channels'):
                print(f"TROPOMI channels: {self.tropomi_channels}")
            if hasattr(self, 'metar_channels'):
                print(f"METAR channels: {self.metar_channels}")
        
        # Initialize data containers
        self.data = None
        
        # Data source containers
        self.maiac_aod_data = None
        self.tropomi_data = None
        self.modis_fire_data = None
        self.merra2_data = None
        self.meteorological_data = None
        
        # Process all data sources
        self._process_pipeline()
    
    def _process_pipeline(self):
        """Main processing pipeline to collect and integrate all data sources"""
        # Initialize dict to store processed data
        channels = {}
        active_channels = {}
        
        # Process each data source if included
        if self.include_channels['maiac']:
            if self.verbose:
                print("Processing MAIAC AOD data...")
            maiac_source = MaiacDataSource(
                timestamps=self.timestamps,
                extent=self.extent,
                dim=self.dim,
                cache_dir=self.cache_dir,
                cache_prefix=self.cache_prefix,
                use_cached_data=self.use_cached_data,
                verbose=self.verbose
            )
            channels['maiac_aod'] = maiac_source.get_data()
            active_channels['maiac_aod'] = channels['maiac_aod']
            self.maiac_aod_data = channels['maiac_aod']
        
        if self.include_channels['tropomi']:
            if self.verbose:
                print("Processing TROPOMI data...")
            tropomi_source = TropomiDataSource(
                timestamps=self.timestamps,
                extent=self.extent,
                dim=self.dim,
                cache_dir=self.cache_dir,
                cache_prefix=self.cache_prefix,
                use_cached_data=self.use_cached_data,
                verbose=self.verbose,
                channels=self.tropomi_channels if hasattr(self, 'tropomi_channels') else None
            )
            channels['tropomi'] = tropomi_source.get_data()
            active_channels['tropomi'] = channels['tropomi']
            self.tropomi_data = channels['tropomi']
        
        if self.include_channels['modis_fire']:
            if self.verbose:
                print("Processing MODIS fire data...")
            modis_source = ModisFireDataSource(
                timestamps=self.timestamps,
                extent=self.extent,
                dim=self.dim,
                cache_dir=self.cache_dir,
                cache_prefix=self.cache_prefix,
                use_cached_data=self.use_cached_data,
                verbose=self.verbose
            )
            channels['modis_fire'] = modis_source.get_data()
            active_channels['modis_fire'] = channels['modis_fire']
            self.modis_fire_data = channels['modis_fire']
        
        if self.include_channels['merra2']:
            if self.verbose:
                print("Processing MERRA2 data...")
            merra2_source = Merra2DataSource(
                timestamps=self.timestamps,
                extent=self.extent,
                dim=self.dim,
                cache_dir=self.cache_dir,
                cache_prefix=self.cache_prefix,
                use_cached_data=self.use_cached_data,
                verbose=self.verbose
            )
            channels['merra2'] = merra2_source.get_data()
            active_channels['merra2'] = channels['merra2']
            self.merra2_data = channels['merra2']
            # Also get native resolution data if available
            self.merra2_native_data = getattr(merra2_source, 'native_data', None)
            self.merra2_native_lats = getattr(merra2_source, 'native_lats', None)
            self.merra2_native_lons = getattr(merra2_source, 'native_lons', None)
        
        if self.include_channels['metar']:
            if self.verbose:
                print("Processing METAR meteorological data...")
            metar_source = MetarDataSource(
                timestamps=self.timestamps,
                extent=self.extent,
                dim=self.dim,
                elevation_path="../../libs/inputs/elevation.npy",
                cache_dir=self.cache_dir,
                cache_prefix=self.cache_prefix,
                use_cached_data=self.use_cached_data,
                verbose=self.verbose,
                channels=self.metar_channels if hasattr(self, 'metar_channels') else None
            )
            channels['metar'] = metar_source.get_data()
            active_channels['metar'] = channels['metar']
            self.meteorological_data = channels['metar']
        
        # Concatenate all active channels
        channel_list = list(active_channels.values())
        
        if channel_list:
            self.data = np.concatenate(channel_list, axis=-1)
            
            if self.verbose:
                print(f"Final data shape: {self.data.shape}")
                self._print_data_statistics()
        else:
            if self.verbose:
                print("No channels were included. Data arrays are empty.")
            # Create empty arrays with proper dimensions
            self.data = np.zeros((self.n_timestamps, self.dim, self.dim, 0))

    def get_channel_info(self):
        """
        Get information about the channels in the dataset.
        
        Returns:
        --------
        dict
            Dictionary with channel information
        """
        # Define channel names based on the included channels
        # Initialize empty channel lists
        maiac_channels = []
        tropomi_channels = []
        modis_fire_channels = []
        merra2_channels = []
        metar_channels = []
        
        # Populate the channel lists based on what's included
        if self.include_channels['maiac']:
            maiac_channels = ['MAIAC_AOD']
        
        if self.include_channels['tropomi']:
            if hasattr(self, 'tropomi_channels'):
                tropomi_channels = self.tropomi_channels
            else:
                tropomi_channels = ['TROPOMI_Methane', 'TROPOMI_NO2', 'TROPOMI_CO']
        
        if self.include_channels['modis_fire']:
            modis_fire_channels = ['MODIS_FRP']
        
        if self.include_channels['merra2']:
            merra2_channels = [
                'MERRA2_PBL_Height',
                'MERRA2_Surface_Air_Temp',
                'MERRA2_Surface_Exchange_Coef'
            ]
        
        if self.include_channels['metar']:
            if hasattr(self, 'metar_channels'):
                metar_channels = self.metar_channels
            else:
                # Default to all METAR channels
                metar_channels = [
                    'METAR_Wind_Speed',
                    'METAR_Wind_Direction',
                    'METAR_Wind_U',
                    'METAR_Wind_V',
                    'METAR_Precipitation',
                    'METAR_Humidity',
                    'METAR_Heat_Index',
                    'METAR_Air_Temp',
                    'METAR_Air_Pressure',
                    'METAR_Dew_Point'
                ]
        
        # Combine all channel names
        channel_names = (
            maiac_channels +
            tropomi_channels +
            modis_fire_channels +
            merra2_channels +
            metar_channels
        )
        
        return {
            'maiac_channels': maiac_channels,
            'tropomi_channels': tropomi_channels,
            'modis_fire_channels': modis_fire_channels,
            'merra2_channels': merra2_channels,
            'metar_channels': metar_channels,
            'channel_names': channel_names,
            'channel_order': channel_names,  # Alias for compatibility
            'total_channels': len(channel_names)
        }
    
    def _print_data_statistics(self):
        """Print detailed statistics about the final data"""
        if not self.verbose:
            return
        
        # Print basic stats about the combined data
        print("\nChannel Statistics:")
        print("===================")
        
        # Get channel info based on the current configuration
        channel_info = self.get_channel_info()
        channel_names = channel_info['channel_names']
        
        # Get the total number of channels
        total_channels = self.data.shape[-1] if self.data.shape[-1] > 0 else 0
        
        if total_channels == 0:
            print("No channels were included in the dataset.")
            return
        
        # Print stats for each channel
        for i, channel_name in enumerate(channel_names):
            if i < total_channels:
                channel_data = self.data[0, :, :, i]  # First timestamp
                
                print(f"\nChannel {i}: {channel_name}")
                print(f"  Min: {np.nanmin(channel_data)}")  
                print(f"  Max: {np.nanmax(channel_data)}")    
                print(f"  Mean: {np.nanmean(channel_data)}")
                print(f"  Std: {np.nanstd(channel_data)}")  
                
                # Count non-zero values
                non_zero = np.count_nonzero(channel_data)
                total = channel_data.size
                print(f"  Data coverage: {non_zero/total*100:.2f}% ({non_zero}/{total} non-zero pixels)")
        
        print("\nFinal Data Shape:")
        print(f"  {self.data.shape[0]} samples")
        print(f"  {self.data.shape[1]}x{self.data.shape[2]} grid size")
        print(f"  {self.data.shape[3]} channels")
        
        print("\nData Memory Usage:")
        data_size_bytes = self.data.nbytes
        data_size_mb = data_size_bytes / (1024 * 1024)
        print(f"  {data_size_mb:.2f} MB")
    
    def save_data(self, filepath=None):
        """
        Save the processed data to a file.
        
        Parameters:
        -----------
        filepath : str, optional
            Path to save the data file. If None, uses the cache directory with the cache prefix.
        """
        if filepath is None:
            # Use default filepath in cache directory
            filepath = os.path.join(self.cache_dir, f"{self.cache_prefix}full_data.npy")
        
        np.save(filepath, self.data)
        
        # Also save channel info
        channel_info = self.get_channel_info()
        channel_info_file = os.path.splitext(filepath)[0] + "_channel_info.json"
        
        with open(channel_info_file, 'w') as f:
            json.dump({k: v for k, v in channel_info.items() if isinstance(v, (list, str, int))}, f, indent=2)
        
        if self.verbose:
            print(f"Data saved to {filepath}")
            print(f"Channel info saved to {channel_info_file}")
    
    def load_data(self, filepath=None):
        """
        Load processed data from a file.
        
        Parameters:
        -----------
        filepath : str, optional
            Path to the data file. If None, uses the cache directory with the cache prefix.
        
        Returns:
        --------
        bool
            True if data was successfully loaded, False otherwise
        """
        if filepath is None:
            # Use default filepath in cache directory
            filepath = os.path.join(self.cache_dir, f"{self.cache_prefix}full_data.npy")
        
        if not os.path.exists(filepath):
            if self.verbose:
                print(f"Data file {filepath} does not exist. Cannot load data.")
            return False
            
        self.data = np.load(filepath)
        
        # Also load channel info if available
        channel_info_file = os.path.splitext(filepath)[0] + "_channel_info.json"
        if os.path.exists(channel_info_file):
            with open(channel_info_file, 'r') as f:
                self._loaded_channel_info = json.load(f)
                
            if self.verbose:
                print(f"Loaded channel info from {channel_info_file}")
                print(f"Channels: {self._loaded_channel_info.get('channel_names', [])}")
        
        if self.verbose:
            print(f"Data loaded from {filepath}")
            print(f"Data shape: {self.data.shape}")
            
        return True
