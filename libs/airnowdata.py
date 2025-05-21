import requests
import json
import os
import sys
import pandas as pd
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

# Check if cartopy is available and import it
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False
    print("Warning: Cartopy not available. Map visualizations will be limited.")

class AirNowData:
    '''
    Gets the AirNow Data.
    Pipeline:
        - Uses AirNow API to download the data as a list of dataframes
        - Extracts the ground site data and converts it into a grid
        - Interpolates the grid using 3D IDW (with elevation)
        - Converts grids into numpy and adds a channel axis
            - (frames, row, col, channel)
        - Creates samples from a sliding window of frames
            - (samples, frames, row, col, channel)

    Members:
        data: The complete processed AirNow data
        ground_site_grids: The uninterpolated, ground-site gridded stations
        target_stations: The station values each sample wants to predict
        air_sens_loc: A dictionary of air sensor locations:
            - Location : (x, y)
    '''
    def __init__(
        self,
        start_date,
        end_date,
        extent,
        airnow_api_key=None,
        save_dir='data/airnow.json',
        processed_cache_dir='data/airnow_processed.npz',  # New parameter for processed data
        frames_per_sample=1,
        dim=200,
        idw_power=2,
        elevation_path=None,
        mask_path=None,
        force_reprocess=False  # Flag to force reprocessing even if cache exists
    ):
        self.air_sens_loc = {}
        self.start_date = start_date
        self.end_date = end_date
        self.extent = extent
        self.dim = dim
        self.frames_per_sample = frames_per_sample
        self.idw_power = idw_power
        
        # Set default paths if not provided
        self.elevation_path = elevation_path if elevation_path else "inputs/elevation.npy"
        self.mask_path = mask_path if mask_path else "inputs/mask.npy"
        
        # Create directories for elevation and mask if they don't exist
        os.makedirs(os.path.dirname(self.elevation_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.mask_path), exist_ok=True)
        os.makedirs(os.path.dirname(processed_cache_dir), exist_ok=True)
        
        # Load elevation data for 3D interpolation if available
        if os.path.exists(self.elevation_path):
            self.elevation = np.load(self.elevation_path)
            # Resize elevation to match dim if needed
            if self.elevation.shape != (dim, dim):
                self.elevation = cv2.resize(self.elevation, (dim, dim))
            # Normalize elevation to prevent overflow
            self.elevation = self._normalize_elevation(self.elevation)
        else:
            print(f"Elevation data not found at {self.elevation_path}. Using flat elevation.")
            # Create a flat elevation model (all zeros)
            self.elevation = np.zeros((dim, dim), dtype=np.float32)
            
        # Load mask data if available
        if os.path.exists(self.mask_path):
            self.mask = np.load(self.mask_path)
            # Resize mask to match dim if needed
            if self.mask.shape != (dim, dim):
                self.mask = cv2.resize(self.mask, (dim, dim))
            print(f"Using mask from {self.mask_path}")
        else:
            # Create a full mask (all ones) if no mask file exists
            print(f"Mask data not found at {self.mask_path}. Interpolating without mask.")
            self.mask = np.ones((dim, dim), dtype=np.float32)

        # Create necessary directories
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        
        # Store sensor names
        self.sensor_names = []
        
        # Check if processed cache exists and use it if available
        if not force_reprocess and os.path.exists(processed_cache_dir):
            print(f"Loading processed AirNow data from cache: {processed_cache_dir}")
            try:
                cached_data = np.load(processed_cache_dir, allow_pickle=True)
                self.data = cached_data['data']
                self.ground_site_grids = cached_data['ground_site_grids']
                
                # Load air_sens_loc dictionary from the cache
                air_sens_loc_array = cached_data['air_sens_loc']
                if isinstance(air_sens_loc_array, np.ndarray):
                    self.air_sens_loc = air_sens_loc_array.item() if air_sens_loc_array.size == 1 else {}
                
                # Load sensor_names if present
                if 'sensor_names' in cached_data:
                    self.sensor_names = cached_data['sensor_names'].tolist() if len(cached_data['sensor_names']) > 0 else list(self.air_sens_loc.keys())
                else:
                    self.sensor_names = list(self.air_sens_loc.keys())
                
                # Load target_stations if present
                if 'target_stations' in cached_data:
                    self.target_stations = cached_data['target_stations']
                else:
                    self.target_stations = None
                
                print(f"✓ Successfully loaded processed data from cache")
                print(f"  - Data shape: {self.data.shape}")
                print(f"  - Found {len(self.air_sens_loc)} sensor locations")
                
                # Return early as we've loaded everything from cache
                return
            except Exception as e:
                print(f"Error loading from cache: {e}. Will reprocess data.")
                # Continue with normal processing
        
        # Get AirNow data
        list_df = self._get_airnow_data(
            start_date, end_date, 
            extent, 
            save_dir,
            airnow_api_key
        )
        
        # Check if we have valid AirNow data
        if not list_df:
            raise ValueError("No valid AirNow data available.")
        
        # Process AirNow data
        print("Processing AirNow data with IDW interpolation (this may take time)...")    
        ground_site_grids = [
            self._preprocess_ground_sites(df, dim, extent) for df in list_df
        ]
        
        print(f"Interpolating {len(ground_site_grids)} frames...")
        interpolated_grids = [
            self._interpolate_frame(frame) for frame in ground_site_grids
        ]
        
        # Continue with the rest of the pipeline
        frames = np.expand_dims(np.array(interpolated_grids), axis=-1)
        processed_ds = self._sliding_window_of(frames, frames_per_sample)

        self.data = processed_ds
        self.ground_site_grids = ground_site_grids
        
        # Get target stations
        if self.air_sens_loc:
            self.target_stations = self._get_target_stations(
                self.data, self.ground_site_grids, self.air_sens_loc
            )
            # Store sensor names
            self.sensor_names = list(self.air_sens_loc.keys())
        else:
            self.target_stations = None
            print("Warning: No air sensor locations found in the data.")
        
        # Save processed data to cache
        print(f"Saving processed AirNow data to cache: {processed_cache_dir}")
        try:
            # Convert air_sens_loc dictionary to a numpy array of one object for storage
            air_sens_loc_array = np.array([self.air_sens_loc])
            
            # Convert sensor_names to numpy array
            sensor_names_array = np.array(self.sensor_names)
            
            # Create target_stations array (empty if None)
            target_stations_array = self.target_stations if self.target_stations is not None else np.array([])
            
            # Save arrays to npz file
            np.savez_compressed(
                processed_cache_dir,
                data=self.data,
                ground_site_grids=np.array(self.ground_site_grids, dtype=object),
                air_sens_loc=air_sens_loc_array,
                sensor_names=sensor_names_array,
                target_stations=target_stations_array
            )
            print("✓ Successfully saved processed data to cache")
        except Exception as e:
            print(f"Warning: Could not save processed data to cache: {e}")
    
    def _normalize_elevation(self, elevation_data):
        """
        Normalize elevation data to a reasonable range to prevent overflow issues.
        
        Parameters:
        -----------
        elevation_data : numpy array
            Raw elevation data
            
        Returns:
        --------
        normalized_elevation : numpy array
            Elevation data scaled to prevent overflow
        """
        # Scale elevation to range [0, 1] then multiply by 100 for reasonable differences
        min_val = np.min(elevation_data)
        max_val = np.max(elevation_data)
        
        # Avoid division by zero if all values are the same
        if max_val == min_val:
            return np.zeros_like(elevation_data)
            
        normalized = (elevation_data - min_val) / (max_val - min_val) * 100
        return normalized.astype(np.float32)  # Use float32 to save memory

    def _get_airnow_data(
        self, 
        start_date, end_date, 
        extent, 
        save_dir, 
        airnow_api_key
    ):
        """
        Grabs the AirNow data from the API or loads from existing file.
        Returns a list of dataframes grouped by time, or an empty list if data is invalid.
        """
        lon_bottom, lon_top, lat_bottom, lat_top = extent
        
        # Get airnow data from the EPA
        if os.path.exists(save_dir):
            print(f"'{save_dir}' already exists; skipping request...")
        else:
            # Preprocess parameters
            date_start = pd.to_datetime(start_date).isoformat()[:13]
            date_end = pd.to_datetime(end_date).isoformat()[:13]
            bbox = f'{lon_bottom},{lat_bottom},{lon_top},{lat_top}'
            URL = "https://www.airnowapi.org/aq/data"

            # Parameters for the API
            PARAMS = {
                'startDate': date_start,
                'endDate': date_end,
                'parameters': 'PM25',
                'BBOX': bbox,
                'dataType': 'B',
                'format': 'application/json',
                'verbose': '1',
                'monitorType': '2',
                'includerawconcentrations': '1',
                'API_KEY': airnow_api_key
            }

            # Send request and save response
            try:
                print("Requesting data from AirNow API...")
                response = requests.get(url=URL, params=PARAMS)
                airnow_data = response.json()
                with open(save_dir, 'w') as file:
                    json.dump(airnow_data, file)
                    print(f"JSON data saved to '{save_dir}'")
            except Exception as e:
                print(f"Error retrieving AirNow data: {e}")
                return []

        # Load and process the data
        try:
            print(f"Loading AirNow data from {save_dir}...")
            with open(save_dir, 'r') as file:
                airnow_data = json.load(file)
            
            # Check if data has error message
            if isinstance(airnow_data, list) and len(airnow_data) > 0 and isinstance(airnow_data[0], dict):
                if 'WebServiceError' in airnow_data[0]:
                    print(f"Error from AirNow API: {airnow_data[0]['WebServiceError']}")
                    return []
            
            # Continue with normal processing if data looks valid
            airnow_df = pd.json_normalize(airnow_data)
            
            # Check for UTC column
            if 'UTC' not in airnow_df.columns:
                print("Error: 'UTC' column not found in AirNow data.")
                
                # Try to construct UTC from other fields if possible
                if 'DateObserved' in airnow_df.columns and 'HourObserved' in airnow_df.columns:
                    print("Attempting to construct UTC from DateObserved and HourObserved...")
                    try:
                        airnow_df['UTC'] = airnow_df.apply(
                            lambda row: pd.Timestamp(row['DateObserved']).replace(
                                hour=int(row['HourObserved'])
                            ).strftime('%Y-%m-%dT%H:%M'),
                            axis=1
                        )
                    except Exception as e:
                        print(f"Failed to construct UTC: {e}")
                        return []
                else:
                    print("Required columns for UTC construction not found.")
                    return []
            
            # Group by UTC
            try:
                list_df = [group for name, group in airnow_df.groupby('UTC')]
                print(f"Grouped AirNow data into {len(list_df)} time frames")
                return list_df
            except Exception as e:
                print(f"Error grouping by UTC: {e}")
                return []
                
        except Exception as e:
            print(f"Error processing AirNow data: {e}")
            return []

    def _preprocess_ground_sites(self, df, dim, extent):
        """
        Preprocess ground sites data into a grid with proper scaling for higher resolution.
        """
        lonMin, lonMax, latMin, latMax = extent
        latDist, lonDist = abs(latMax - latMin), abs(lonMax - lonMin)
        unInter = np.zeros((dim, dim))
        
        # Check if the required columns exist
        required_columns = ['Latitude', 'Longitude', 'Value', 'SiteName']
        if not all(col in df.columns for col in required_columns):
            print(f"Warning: Missing required columns in dataframe. Available columns: {df.columns}")
            return unInter
            
        dfArr = np.array(df[required_columns])
        
        for i in range(dfArr.shape[0]):
            # Calculate x (latitude) - properly scaled for the new grid size
            x = int(((latMax - dfArr[i,0]) / latDist) * dim)
            # Ensure x is within bounds
            x = max(0, min(x, dim - 1))
            
            # Calculate y (longitude) - fixed calculation for proper mapping
            y = int(((dfArr[i,1] - lonMin) / lonDist) * dim)
            # Ensure y is within bounds
            y = max(0, min(y, dim - 1))
            
            # Optional: Debug print statement to verify coordinates
            # print(f"Station: {dfArr[i,3]}, Lat: {dfArr[i,0]}, Lon: {dfArr[i,1]}, Grid X: {x}, Grid Y: {y}")
            
            # Set the value in the grid
            if dfArr[i,2] < 0:
                unInter[x, y] = 0
            else:
                unInter[x, y] = dfArr[i,2]
                # save sensor site name and location
                sitename = dfArr[i,3]
                self.air_sens_loc[sitename] = (x, y)
        
        return unInter

    def _find_closest_values(self, x, y, coordinates, n=10):
        """
        Find closest values in the grid for interpolation.
        """
        if not coordinates:
            return [], np.array([])
            
        # Convert coordinates to numpy array if it's not already
        coords_array = np.array(coordinates)
        
        # Compute Euclidean distances
        diffs = coords_array - np.array([x, y])
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        
        # Get indices of n closest points
        closest_indices = np.argsort(distances)[:n]
        sorted_distances = distances[closest_indices]
        
        # Normalize distances
        magnitude = np.linalg.norm(sorted_distances)
        if magnitude > 0:
            normalized_distances = sorted_distances / magnitude
        else:
            normalized_distances = sorted_distances
            
        # Get the n closest coordinates
        closest_values = [coordinates[i] for i in closest_indices]
        
        return closest_values, normalized_distances

    def _find_elevations(self, x, y, coordinates):
        """
        Calculate elevation differences between points.
        """
        if not coordinates:
            return np.array([])
            
        # Get elevation at the target point
        stat = self.elevation[x, y]
        
        # Calculate elevation differences for each coordinate - use float32 to prevent overflow
        elevations = []
        for a, b in coordinates:
            if 0 <= a < self.elevation.shape[0] and 0 <= b < self.elevation.shape[1]:
                # Use np.float32 type to handle large elevation values
                diff = np.float32(stat) - np.float32(self.elevation[a, b])
                elevations.append(diff)
            else:
                elevations.append(0.0)
                
        # Convert to numpy array
        elevations = np.array(elevations, dtype=np.float32)
        
        # Normalize elevations
        magnitude = np.linalg.norm(elevations)
        if magnitude > 0:
            elevations = elevations / magnitude
            
        return elevations

    def _find_values(self, coordinates, unInter):
        """
        Get values at specified coordinates.
        """
        values = []
        for a, b in coordinates:
            if 0 <= a < unInter.shape[0] and 0 <= b < unInter.shape[1]:
                values.append(unInter[a, b])
            else:
                values.append(0)
        return values

    def _idw_interpolate(self, x, y, values, distance_list, elevation_list, p=2):
        """
        Perform 3D IDW interpolation.
        """
        if len(values) == 0:
            return 0
            
        # Combine horizontal distance and elevation difference
        difference_factor = distance_list + elevation_list**2
        
        # Avoid division by zero
        eps = np.finfo(float).eps
        difference_factor[difference_factor == 0] = eps
        
        # Calculate weights
        weights = 1 / difference_factor**p
        
        # Normalize weights
        weights /= np.sum(weights)
        
        # Compute weighted average
        estimated_value = np.sum(weights * np.array(values))
        
        return estimated_value

    def _variable_blur(self, data, kernel_size):
        """
        Apply variable blur to smooth the interpolation.
        """
        data_blurred = np.empty(data.shape)
        Ni, Nj = data.shape
        
        for i in range(Ni):
            for j in range(Nj):
                res = 0.0
                weight = 0
                sigma = kernel_size[i, j]
                
                for ii in range(i - sigma, i + sigma + 1):
                    for jj in range(j - sigma, j + sigma + 1):
                        if ii < 0 or ii >= Ni or jj < 0 or jj >= Nj:
                            continue
                        res += data[ii, jj]
                        weight += 1
                        
                if weight > 0:
                    data_blurred[i, j] = res / weight
                else:
                    data_blurred[i, j] = data[i, j]
                    
        return data_blurred

    def _interpolate_frame(self, unInter):
        """
        Interpolate a frame using 3D IDW method.
        """
        # Get coordinates of non-zero values
        nonzero_indices = np.nonzero(unInter)
        coordinates = list(zip(nonzero_indices[0], nonzero_indices[1]))
        
        if not coordinates:
            return unInter
            
        # Initialize output grid with background color value
        # Use a very low negative value for purple background
        interpolated = np.full((self.dim, self.dim), -10.0)  # Set a value that will render as purple
        
        # Interpolate each point
        for x in range(self.dim):
            for y in range(self.dim):
                # Skip if already a station point
                if unInter[x, y] > 0:
                    interpolated[x, y] = unInter[x, y]
                    continue
                    
                # Find closest values and their distances
                coords, distance_list = self._find_closest_values(x, y, coordinates)
                
                if not coords:
                    continue
                    
                # Get elevation differences
                elevation_list = self._find_elevations(x, y, coords)
                
                # Get values at closest points
                vals = self._find_values(coords, unInter)
                
                # Interpolate the value
                value = self._idw_interpolate(
                    x, y, vals, distance_list, elevation_list, self.idw_power
                )
                
                # Only set interpolated value if it's above threshold
                # This preserves purple background where values are low
                if value > 0.1:  # Adjust threshold as needed
                    interpolated[x, y] = value
        
        # Apply variable blur for smoothing
        kernel_size = np.random.randint(0, 5, (self.dim, self.dim))
        out = self._variable_blur(interpolated, kernel_size)
        
        # Apply Gaussian filter for final smoothing
        out = gaussian_filter(out, sigma=0.5)
        
        # Apply mask while preserving background color
        if hasattr(self, 'mask') and np.any(self.mask != 1):
            # Create a copy of the output
            out_masked = out.copy()
            
            # Where mask is 0, set to background color (-10.0 for purple)
            out_masked[self.mask == 0] = -10.0
            
            # Update output
            out = out_masked
        
        return out

    def _sliding_window_of(self, frames, frames_per_sample):
        """
        Uses a sliding window to bundle frames into samples.
        """
        n_frames, row, col, channels = frames.shape
        n_samples = max(1, n_frames - frames_per_sample + 1)
        samples = np.empty((n_samples, frames_per_sample, row, col, channels))
        
        for i in range(n_samples):
            end_idx = min(i + frames_per_sample, n_frames)
            if end_idx - i < frames_per_sample:
                # Not enough frames, repeat the last frame
                sample_frames = []
                for j in range(i, end_idx):
                    sample_frames.append(frames[j])
                # Fill the rest with the last frame
                for j in range(end_idx - i, frames_per_sample):
                    sample_frames.append(frames[end_idx-1] if end_idx > 0 else frames[0])
                samples[i] = np.array(sample_frames)
            else:
                samples[i] = np.array([frames[j] for j in range(i, i + frames_per_sample)])
            
        return samples

    def _get_target_stations(self, X, gridded_data, sensor_locations):
        """
        Gets the desired target stations to predict for a given list of samples.
        """
        n_samples, frames_per_sample = X.shape[0], X.shape[1] 
        n_sensors = len(sensor_locations)
        
        if n_sensors == 0:
            raise ValueError("No sensor locations available to generate target stations")
            
        Y = np.empty((n_samples, n_sensors))
        
        for sample in range(len(Y)):
            for i, (loc, coords) in enumerate(sensor_locations.items()):
                x, y = coords
                offset = sample + frames_per_sample
                
                if offset < len(gridded_data):
                    Y[sample][i] = gridded_data[offset][x][y]
                else:
                    # Use the last available data for out-of-bounds targets
                    Y[sample][i] = gridded_data[-1][x][y]

        return Y

    def _grid_to_latlon(self, x, y):
        """
        Convert grid coordinates to latitude/longitude.
        
        Parameters:
        -----------
        x, y : int
            Grid coordinates
            
        Returns:
        --------
        lat, lon : float
            Latitude and longitude
        """
        lon_min, lon_max, lat_min, lat_max = self.extent
        lat_dist = lat_max - lat_min
        lon_dist = lon_max - lon_min
        
        # Convert from grid coordinates to lat/lon
        lat = lat_max - (x / self.dim) * lat_dist
        lon = lon_min + (y / self.dim) * lon_dist
        
        return lat, lon
    
    def _latlon_to_grid(self, lat, lon):
        """
        Convert latitude/longitude to grid coordinates.
        
        Parameters:
        -----------
        lat, lon : float
            Latitude and longitude
            
        Returns:
        --------
        x, y : int
            Grid coordinates
        """
        lon_min, lon_max, lat_min, lat_max = self.extent
        lat_dist = lat_max - lat_min
        lon_dist = lon_max - lon_min
        
        # Convert from lat/lon to grid coordinates
        x = int(((lat_max - lat) / lat_dist) * self.dim)
        y = int(((lon - lon_min) / lon_dist) * self.dim)
        
        # Ensure coordinates are within bounds
        x = max(0, min(x, self.dim - 1))
        y = max(0, min(y, self.dim - 1))
        
        return x, y

    def visualize_sensor_locations(self, figsize=(12, 10), marker_size=120, show_names=True, 
                                  save_path=None, with_background=True, dpi=100):
        """
        Visualizes the AirNow sensor locations on a map before interpolation.
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size (width, height) in inches
        marker_size : int, optional
            Size of the markers representing sensors
        show_names : bool, optional
            If True, displays sensor names next to their locations
        save_path : str, optional
            If provided, saves the figure to this path
        with_background : bool, optional
            If True, displays a full map background with Cartopy (if available)
        dpi : int, optional
            Resolution for the output figure
            
        Returns:
        --------
        fig, ax : matplotlib figure and axis objects
        """
        # Check if we have sensor locations
        if not self.air_sens_loc:
            raise ValueError("No sensor locations found in the AirNowData object")
        
        # Extract the extent
        lon_min, lon_max, lat_min, lat_max = self.extent
        
        # For Cartopy-enabled visualizations with map background
        if with_background and CARTOPY_AVAILABLE:
            # Create a new figure with a GeoAxes
            fig = plt.figure(figsize=figsize, dpi=dpi)
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            
            # Set map extent
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            
            # Add map features
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.STATES, linestyle=':')
            ax.add_feature(cfeature.LAND, alpha=0.1)
            ax.add_feature(cfeature.OCEAN, alpha=0.1)
            ax.add_feature(cfeature.LAKES, alpha=0.1)
            ax.add_feature(cfeature.RIVERS, alpha=0.1)
            
            # Add gridlines
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            
            # Convert grid coordinates to lat/lon for plotting
            sensor_lons = []
            sensor_lats = []
            sensor_names_list = []
            
            for name, (x, y) in self.air_sens_loc.items():
                lat, lon = self._grid_to_latlon(x, y)
                sensor_lats.append(lat)
                sensor_lons.append(lon)
                sensor_names_list.append(name)
            
            # Plot sensors
            ax.scatter(sensor_lons, sensor_lats, s=marker_size, c='red', marker='^', 
                      edgecolor='black', linewidth=1, alpha=0.8, 
                      transform=ccrs.PlateCarree(), label='Air Quality Sensors')
            
            # Add sensor names if requested
            if show_names:
                for i, name in enumerate(sensor_names_list):
                    ax.annotate(name, (sensor_lons[i], sensor_lats[i]), 
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=9, color='black', weight='bold',
                              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
            
        else:
            # Simple visualization without map background
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            
            # Set up the axes for a pseudo-geographic plot
            ax.set_xlim(lon_min, lon_max)
            ax.set_ylim(lat_min, lat_max)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # Convert grid coordinates to lat/lon for plotting
            sensor_lons = []
            sensor_lats = []
            sensor_names_list = []
            
            for name, (x, y) in self.air_sens_loc.items():
                lat, lon = self._grid_to_latlon(x, y)
                sensor_lats.append(lat)
                sensor_lons.append(lon)
                sensor_names_list.append(name)
            
            # Plot sensors
            ax.scatter(sensor_lons, sensor_lats, s=marker_size, c='red', marker='^', 
                      edgecolor='black', linewidth=1, alpha=0.8, label='Air Quality Sensors')
            
            # Add sensor names if requested
            if show_names:
                for i, name in enumerate(sensor_names_list):
                    ax.annotate(name, (sensor_lons[i], sensor_lats[i]), 
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=9, color='black', weight='bold',
                              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
        
        # Add title and legend
        plt.title('AirNow Sensor Locations', fontsize=16)
        plt.legend(loc='upper right')
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        return fig, ax

    def visualize_interpolated_data(self, frame_index=0, sample_index=0, figsize=(14, 10), 
                                   show_sensors=True, marker_size=80, cmap='plasma_r', 
                                   save_path=None, vmin=None, vmax=None,
                                   alpha=0.7, with_background=True, dpi=100):
        """
        Visualizes the interpolated AirNow data on a map.
        
        Parameters:
        -----------
        frame_index : int, optional
            Index of the frame to visualize within the sample
        sample_index : int, optional
            Index of the sample to visualize
        figsize : tuple, optional
            Figure size (width, height) in inches
        show_sensors : bool, optional
            If True, shows sensor locations on the map
        marker_size : int, optional
            Size of the markers representing sensors
        cmap : str or matplotlib colormap, optional
            Colormap to use for the interpolated data
            Use 'aqi' for a custom AQI-like colormap
        save_path : str, optional
            If provided, saves the figure to this path
        vmin, vmax : float, optional
            Minimum and maximum values for the colormap; if None, auto-determined
        alpha : float, optional
            Transparency of the interpolated data overlay
        with_background : bool, optional
            If True, displays a full map background with Cartopy (if available)
        dpi : int, optional
            Resolution for the output figure
            
        Returns:
        --------
        fig, ax : matplotlib figure and axis objects
        """
        # Check interpolated data and extract it
        if not hasattr(self, 'data') or self.data is None:
            raise ValueError("No interpolated data available in the AirNowData object")
            
        try:
            interpolated_data = self.data[sample_index, frame_index, :, :, 0]
        except IndexError:
            raise ValueError(f"Sample index {sample_index} or frame index {frame_index} is out of range")
        
        # Get the raw ground site grid (before interpolation)
        if hasattr(self, 'ground_site_grids') and self.ground_site_grids is not None:
            if len(self.ground_site_grids) > sample_index:
                raw_grid = self.ground_site_grids[sample_index]
            else:
                raw_grid = None
                print(f"Warning: Sample index {sample_index} is out of range for ground_site_grids")
        else:
            raw_grid = None
            print(f"Warning: ground_site_grids not found in airnow_data")
            
        # Extract the extent
        lon_min, lon_max, lat_min, lat_max = self.extent
            
        # Create AQI-like colormap if requested
        if cmap == 'aqi':
            # AQI-like colormap: Green-Yellow-Orange-Red-Purple-Maroon
            colors = [(0, 1, 0),    # Green (Good)
                     (1, 1, 0),     # Yellow (Moderate)
                     (1, 0.5, 0),   # Orange (Unhealthy for Sensitive Groups)
                     (1, 0, 0),     # Red (Unhealthy)
                     (0.5, 0, 0.5), # Purple (Very Unhealthy)
                     (0.5, 0, 0)]   # Maroon (Hazardous)
            cmap = LinearSegmentedColormap.from_list('aqi_cmap', colors)
            
        # Get data limits if not provided
        if vmin is None:
            vmin = np.min(interpolated_data[interpolated_data > -10])  # Exclude background values
        if vmax is None:
            vmax = np.max(interpolated_data)
            
        # For Cartopy-enabled visualizations with map background
        if with_background and CARTOPY_AVAILABLE:
            # Create a new figure with a GeoAxes
            fig = plt.figure(figsize=figsize, dpi=dpi)
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            
            # Set map extent
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            
            # Add map features
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.STATES, linestyle=':')
            ax.add_feature(cfeature.LAND, alpha=0.1)
            ax.add_feature(cfeature.OCEAN, alpha=0.1)
            ax.add_feature(cfeature.LAKES, alpha=0.1)
            ax.add_feature(cfeature.RIVERS, alpha=0.1)
            
            # Create meshgrid for the plot
            lons = np.linspace(lon_min, lon_max, self.dim)
            lats = np.linspace(lat_max, lat_min, self.dim)  # Note: reversed to match the grid orientation
            lons_mesh, lats_mesh = np.meshgrid(lons, lats)
            
            # Plot the interpolated data
            # Mask values below 0 (likely background)
            masked_data = np.ma.masked_where(interpolated_data < 0, interpolated_data)
            c = ax.pcolormesh(lons_mesh, lats_mesh, masked_data, 
                            transform=ccrs.PlateCarree(), 
                            cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
            
            # Add gridlines
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            
        else:
            # Simple visualization without map background
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            
            # Plot the interpolated data
            masked_data = np.ma.masked_where(interpolated_data < 0, interpolated_data)
            c = ax.imshow(masked_data, cmap=cmap, vmin=vmin, vmax=vmax, 
                         extent=[lon_min, lon_max, lat_min, lat_max],
                         origin='upper', alpha=alpha)
            
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.grid(True, linestyle='--', alpha=0.5)
            
        # Add colorbar
        cbar = plt.colorbar(c, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label('PM2.5 Concentration (μg/m³)', fontsize=12)
            
        # Show sensor locations if requested
        if show_sensors and self.air_sens_loc:
            sensor_lons = []
            sensor_lats = []
            sensor_names_list = []
            sensor_values = []
            
            # Extract sensor locations and convert to lat/lon
            for name, (x, y) in self.air_sens_loc.items():
                lat, lon = self._grid_to_latlon(x, y)
                sensor_lats.append(lat)
                sensor_lons.append(lon)
                sensor_names_list.append(name)
                
                # Get the actual value from the grid if possible
                if raw_grid is not None:
                    try:
                        value = raw_grid[x, y]
                        sensor_values.append(value)
                    except IndexError:
                        sensor_values.append(None)
                else:
                    sensor_values.append(None)
            
            # Plot sensors
            sc = ax.scatter(sensor_lons, sensor_lats, s=marker_size, c='white', marker='^', 
                          edgecolor='black', linewidth=1, alpha=1.0,
                          label='Air Quality Sensors',
                          zorder=5)  # Ensure sensors appear on top
            
            # Add sensor names and values if data is available
            for i, name in enumerate(sensor_names_list):
                label = name
                if sensor_values[i] is not None and sensor_values[i] > 0:
                    label += f"\n({sensor_values[i]:.1f})"
                    
                ax.annotate(label, (sensor_lons[i], sensor_lats[i]), 
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=9, color='black', weight='bold',
                          bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                          zorder=6)  # Ensure labels appear on top
        
        # Add title
        title = f'Interpolated PM2.5 Concentration - Sample {sample_index}, Frame {frame_index}'
        plt.title(title, fontsize=16)
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        return fig, ax

    def compare_raw_vs_interpolated(self, sample_index=0, figsize=(18, 8), save_path=None,
                                  cmap='plasma_r', dpi=100):
        """
        Creates a side-by-side comparison of raw sensor data vs. interpolated data.
        
        Parameters:
        -----------
        sample_index : int, optional
            Index of the sample to visualize
        figsize : tuple, optional
            Figure size (width, height) in inches
        save_path : str, optional
            If provided, saves the figure to this path
        cmap : str or matplotlib colormap, optional
            Colormap to use for the visualization
        dpi : int, optional
            Resolution for the output figure
            
        Returns:
        --------
        fig, axes : matplotlib figure and axes objects
        """
        # Check if data is available
        if not hasattr(self, 'ground_site_grids') or self.ground_site_grids is None:
            raise ValueError("No raw ground site data available")
            
        if not hasattr(self, 'data') or self.data is None:
            raise ValueError("No interpolated data available")
            
        # Get the raw and interpolated data
        try:
            raw_grid = self.ground_site_grids[sample_index]
        except IndexError:
            raise ValueError(f"Sample index {sample_index} is out of range for ground_site_grids")
            
        try:
            interpolated_data = self.data[sample_index, 0, :, :, 0]  # First frame of the sample
        except IndexError:
            raise ValueError(f"Sample index {sample_index} is out of range for interpolated data")
            
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
        
        # Plot raw sensor data (with mask for zeros)
        masked_raw = np.ma.masked_where(raw_grid <= 0, raw_grid)
        im1 = axes[0].imshow(masked_raw, cmap=cmap)
        axes[0].set_title('Raw Sensor Data (Before Interpolation)', fontsize=14)
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], orientation='vertical', fraction=0.046, pad=0.04)
        
        # Add sensor locations to the raw data plot
        for name, (x, y) in self.air_sens_loc.items():
            value = raw_grid[x, y]
            if value > 0:
                axes[0].plot(y, x, 'w^', markersize=10, markeredgecolor='black')
                axes[0].annotate(f"{name} ({value:.1f})", (y, x), 
                                xytext=(5, 5), textcoords='offset points',
                                fontsize=8, color='black', weight='bold',
                                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))
        
        # Plot interpolated data (with mask for background)
        masked_interp = np.ma.masked_where(interpolated_data < 0, interpolated_data)
        im2 = axes[1].imshow(masked_interp, cmap=cmap)
        axes[1].set_title('Interpolated Data (IDW Method)', fontsize=14)
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)
        
        # Add sensor locations to the interpolated data plot
        for name, (x, y) in self.air_sens_loc.items():
            axes[1].plot(y, x, 'w^', markersize=10, markeredgecolor='black')
        
        plt.suptitle(f'Raw Sensor Data vs. Interpolated PM2.5 Concentration - Sample {sample_index}', 
                    fontsize=16)
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.tight_layout()
        return fig, axes