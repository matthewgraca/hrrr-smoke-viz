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

# Import temporal aggregation utilities
from libs.pwwb.utils.temporal_utils import (
    aggregate_temporal_data, 
    create_aggregation_config,
    get_aggregation_method_for_variable
)

class PWWBData:
    def __init__(
        self,
        start_date="2018-01-01",
        end_date="2020-12-31",
        extent=(-118.75, -117.5, 33.5, 34.5),  # Default to LA County bounds
        frames_per_sample=5,  # Number of time steps per sample
        dim=200,  # Spatial resolution
        frequency='hourly',  # 'hourly' or 'daily'
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
        self.frames_per_sample = frames_per_sample
        self.dim = dim
        self.frequency = frequency  # Store frequency parameter
        self.verbose = verbose
        self.cache_dir = cache_dir
        self.use_cached_data = use_cached_data
        self.output_dir = output_dir
        
        # Generate a cache prefix based on start/end dates and frequency if not provided
        if cache_prefix is None:
            freq_suffix = f"_{frequency}" if frequency != 'hourly' else ""
            self.cache_prefix = f"{self.start_date.strftime('%Y%m%d')}-{self.end_date.strftime('%Y%m%d')}{freq_suffix}_"
        else:
            self.cache_prefix = f"{cache_prefix}_"
            
        if self.verbose:
            print(f"Using cache prefix: {self.cache_prefix}")
            print(f"Target frequency: {frequency}")
        
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
        
        # Generate timestamps at hourly intervals (always start hourly, then aggregate)
        self.hourly_timestamps = pd.date_range(self.start_date, self.end_date, freq='h')
        self.n_hourly_timestamps = len(self.hourly_timestamps)
        
        # These will be set after aggregation in _process_pipeline
        self.timestamps = None
        self.n_timestamps = None
        
        if self.verbose:
            print(f"Initialized PWWBData with {self.n_hourly_timestamps} hourly timestamps")
            print(f"Date range: {self.start_date} to {self.end_date}")
            print(f"Channels included: {[k for k, v in self.include_channels.items() if v]}")
            if hasattr(self, 'tropomi_channels'):
                print(f"TROPOMI channels: {self.tropomi_channels}")
            if hasattr(self, 'metar_channels'):
                print(f"METAR channels: {self.metar_channels}")
        
        # Initialize data containers
        self.data = None
        self.all_channels = None
        
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
        
        # Process each data source if included (always collect hourly first)
        if self.include_channels['maiac']:
            if self.verbose:
                print("Processing MAIAC AOD data...")
            maiac_source = MaiacDataSource(
                timestamps=self.hourly_timestamps,  # Always use hourly for collection
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
                timestamps=self.hourly_timestamps,  # Always use hourly for collection
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
                timestamps=self.hourly_timestamps,  # Always use hourly for collection
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
                timestamps=self.hourly_timestamps,  # Always use hourly for collection
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
                timestamps=self.hourly_timestamps,  # Always use hourly for collection
                extent=self.extent,
                dim=self.dim,
                elevation_path="inputs/elevation.npy",
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
            self.all_channels = np.concatenate(channel_list, axis=-1)
            
            # Apply temporal aggregation if needed
            if self.frequency == 'daily':
                if self.verbose:
                    print(f"Aggregating from hourly to daily frequency...")
                
                # Get channel information for proper aggregation
                channel_info = self.get_channel_info()
                channel_names = channel_info['channel_names']
                
                # Create aggregation configuration
                agg_config = create_aggregation_config(
                    channel_names, 
                    preserve_wind_vectors=True
                )
                
                # Aggregate the data
                self.all_channels, self.timestamps = aggregate_temporal_data(
                    data=self.all_channels,
                    timestamps=self.hourly_timestamps,
                    target_frequency=self.frequency,
                    aggregation_method='mean',  # Default method
                    preserve_wind_vectors=True,
                    wind_u_indices=agg_config['wind_u_indices'],
                    wind_v_indices=agg_config['wind_v_indices'],
                    verbose=self.verbose
                )
                
                # Update individual data source arrays if they exist
                self._update_individual_arrays_after_aggregation(agg_config)
                
            else:
                # Keep hourly frequency
                self.timestamps = self.hourly_timestamps
            
            self.n_timestamps = len(self.timestamps)
            
            # Create sliding window samples
            self.data = self._sliding_window_of(self.all_channels, self.frames_per_sample)
            
            if self.verbose:
                print(f"Final data shape: {self.data.shape}")
                print(f"Using {len(self.timestamps)} timestamps at {self.frequency} frequency")
                self._print_data_statistics()
        else:
            if self.verbose:
                print("No channels were included. Data arrays are empty.")
            # Create empty arrays with proper dimensions
            self.timestamps = self.hourly_timestamps if self.frequency == 'hourly' else pd.date_range(
                self.start_date, self.end_date, freq='D'
            )
            self.n_timestamps = len(self.timestamps)
            self.all_channels = np.zeros((self.n_timestamps, self.dim, self.dim, 0))
            self.data = np.zeros((self.n_timestamps - self.frames_per_sample + 1, 
                                 self.frames_per_sample, self.dim, self.dim, 0))

    def _update_individual_arrays_after_aggregation(self, agg_config):
        """Update individual data source arrays after temporal aggregation."""
        if self.frequency != 'daily':
            return
            
        channel_info = self.get_channel_info()
        
        # Update MAIAC data
        if self.include_channels['maiac'] and self.maiac_aod_data is not None:
            maiac_data, _ = aggregate_temporal_data(
                self.maiac_aod_data, self.hourly_timestamps, 'daily', 
                aggregation_method='mean', verbose=False
            )
            self.maiac_aod_data = maiac_data
            
        # Update TROPOMI data
        if self.include_channels['tropomi'] and self.tropomi_data is not None:
            tropomi_data, _ = aggregate_temporal_data(
                self.tropomi_data, self.hourly_timestamps, 'daily',
                aggregation_method='mean', verbose=False
            )
            self.tropomi_data = tropomi_data
            
        # Update MODIS fire data (use max for fire)
        if self.include_channels['modis_fire'] and self.modis_fire_data is not None:
            fire_data, _ = aggregate_temporal_data(
                self.modis_fire_data, self.hourly_timestamps, 'daily',
                aggregation_method='max', verbose=False
            )
            self.modis_fire_data = fire_data
            
        # Update MERRA2 data
        if self.include_channels['merra2'] and self.merra2_data is not None:
            merra2_data, _ = aggregate_temporal_data(
                self.merra2_data, self.hourly_timestamps, 'daily',
                aggregation_method='mean', verbose=False
            )
            self.merra2_data = merra2_data
            
        # Update METAR data (with wind vector preservation)
        if self.include_channels['metar'] and self.meteorological_data is not None:
            # Find wind indices within METAR data
            metar_wind_u_indices = []
            metar_wind_v_indices = []
            for i, channel_name in enumerate(channel_info['metar_channels']):
                if 'wind_u' in channel_name.lower():
                    metar_wind_u_indices.append(i)
                elif 'wind_v' in channel_name.lower():
                    metar_wind_v_indices.append(i)
            
            metar_data, _ = aggregate_temporal_data(
                self.meteorological_data, self.hourly_timestamps, 'daily',
                aggregation_method='mean', preserve_wind_vectors=True,
                wind_u_indices=metar_wind_u_indices,
                wind_v_indices=metar_wind_v_indices,
                verbose=False
            )
            self.meteorological_data = metar_data

    def _sliding_window_of(self, frames, window_size):
        """
        Create sliding window samples from sequential frames.
        
        Parameters:
        -----------
        frames : numpy.ndarray
            Sequential frames with shape (n_timestamps, height, width, channels)
        window_size : int
            Number of consecutive frames to include in each sample
        
        Returns:
        --------
        numpy.ndarray
            Sliding window samples with shape (n_samples, window_size, height, width, channels)
        """
        n_frames, row, col, channels = frames.shape
        n_samples = n_frames - window_size + 1
        
        if n_samples <= 0:
            raise ValueError(f"Not enough frames ({n_frames}) for sliding window of size {window_size}")
        
        samples = np.empty((n_samples, window_size, row, col, channels))
        
        for i in range(n_samples):
            samples[i] = frames[i:i+window_size]
        
        return samples
    
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
        total_channels = self.all_channels.shape[-1] if self.all_channels.shape[-1] > 0 else 0
        
        if total_channels == 0:
            print("No channels were included in the dataset.")
            return
        
        # Print stats for each channel
        for i, channel_name in enumerate(channel_names):
            if i < total_channels:
                channel_data = self.all_channels[0, :, :, i]  # First timestamp
                
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
        print(f"  {self.data.shape[1]} frames per sample")
        print(f"  {self.data.shape[2]}x{self.data.shape[3]} grid size")
        print(f"  {self.data.shape[4]} channels")
        
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
            # Use default filepath in cache directory with frequency suffix
            freq_suffix = f"_{self.frequency}" if self.frequency != 'hourly' else ""
            filepath = os.path.join(self.cache_dir, f"{self.cache_prefix}full_data{freq_suffix}.npy")
        
        np.save(filepath, self.data)
        
        # Also save channel info and metadata
        channel_info = self.get_channel_info()
        metadata = {
            'frequency': self.frequency,
            'frames_per_sample': self.frames_per_sample,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'extent': self.extent,
            'dim': self.dim,
            'n_timestamps': self.n_timestamps
        }
        
        channel_info.update(metadata)
        channel_info_file = os.path.splitext(filepath)[0] + "_metadata.json"
        
        with open(channel_info_file, 'w') as f:
            json.dump({k: v for k, v in channel_info.items() if isinstance(v, (list, str, int, float))}, f, indent=2)
        
        if self.verbose:
            print(f"Data saved to {filepath}")
            print(f"Metadata saved to {channel_info_file}")
    
    def load_data(self, filepath=None):
        """
        Load processed data from a file.
        
        Parameters:
        -----------
        filepath : str, optional
            Path to the data file. If None, tries to find based on frequency.
        
        Returns:
        --------
        bool
            True if data was successfully loaded, False otherwise
        """
        if filepath is None:
            # Try to find file based on frequency
            freq_suffix = f"_{self.frequency}" if self.frequency != 'hourly' else ""
            filepath = os.path.join(self.cache_dir, f"{self.cache_prefix}full_data{freq_suffix}.npy")
        
        if not os.path.exists(filepath):
            if self.verbose:
                print(f"Data file {filepath} does not exist. Cannot load data.")
            return False
            
        self.data = np.load(filepath)
        
        # Also load metadata if available
        metadata_file = os.path.splitext(filepath)[0] + "_metadata.json"
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self._loaded_metadata = json.load(f)
                
            # Verify frequency matches
            loaded_freq = self._loaded_metadata.get('frequency', 'hourly')
            if loaded_freq != self.frequency:
                if self.verbose:
                    print(f"Warning: Loaded data frequency ({loaded_freq}) differs from requested ({self.frequency})")
                
            if self.verbose:
                print(f"Loaded metadata from {metadata_file}")
                print(f"Data frequency: {loaded_freq}")
                print(f"Channels: {self._loaded_metadata.get('channel_names', [])}")
        
        if self.verbose:
            print(f"Data loaded from {filepath}")
            print(f"Data shape: {self.data.shape}")
            
        return True