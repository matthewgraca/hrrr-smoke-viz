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

import cartopy.crs as ccrs
import cartopy.feature as cfeature

class AirNowData:
    '''
    Gets the AirNow Data and processes it with IDW interpolation.
    Pipeline:
        - Downloads data from AirNow API
        - Optionally filters sensors based on mask (excludes sensors outside valid areas)
        - Converts ground site data into grids
        - Interpolates using 3D IDW (with elevation)
        - Creates sliding window samples for time series
    '''
    def __init__(
        self,
        start_date,
        end_date,
        extent,
        airnow_api_key=None,
        save_dir='data/airnow.json',
        processed_cache_dir='data/airnow_processed.npz',
        frames_per_sample=1,
        dim=200,
        idw_power=2,
        elevation_path=None,
        mask_path=None,
        use_mask=True,  # NEW: Control whether to use mask filtering
        force_reprocess=False
    ):
        self.air_sens_loc = {}
        self.start_date = start_date
        self.end_date = end_date
        self.extent = extent
        self.dim = dim
        self.frames_per_sample = frames_per_sample
        self.idw_power = idw_power
        self.use_mask = use_mask  # Store the mask usage preference
        
        # Set default paths
        self.elevation_path = elevation_path if elevation_path else "inputs/elevation.npy"
        self.mask_path = mask_path if mask_path else "inputs/mask.npy"
        
        # Create directories
        os.makedirs(os.path.dirname(self.elevation_path), exist_ok=True)
        if use_mask and mask_path:
            os.makedirs(os.path.dirname(self.mask_path), exist_ok=True)
        os.makedirs(os.path.dirname(processed_cache_dir), exist_ok=True)
        
        # Load elevation data for 3D interpolation
        if os.path.exists(self.elevation_path):
            self.elevation = np.load(self.elevation_path)
            if self.elevation.shape != (dim, dim):
                self.elevation = cv2.resize(self.elevation, (dim, dim))
            self.elevation = self._normalize_elevation(self.elevation)
        else:
            print(f"Elevation data not found at {self.elevation_path}. Using flat elevation.")
            self.elevation = np.zeros((dim, dim), dtype=np.float32)
            
        # Load mask data only if use_mask is True
        self.mask = None
        if use_mask:
            if mask_path and os.path.exists(self.mask_path):
                self.mask = np.load(self.mask_path)
                if self.mask.shape != (dim, dim):
                    self.mask = cv2.resize(self.mask, (dim, dim))
                print(f"Using mask from {self.mask_path}")
            else:
                print(f"Mask requested but not found at {self.mask_path}. Creating default mask (all valid).")
                self.mask = np.ones((dim, dim), dtype=np.float32)
        else:
            print("Mask usage disabled. All sensors within extent will be included.")

        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        self.sensor_names = []
        
        # Try to load from cache first
        if not force_reprocess and os.path.exists(processed_cache_dir):
            print(f"Loading processed AirNow data from cache: {processed_cache_dir}")
            try:
                cached_data = np.load(processed_cache_dir, allow_pickle=True)
                self.data = cached_data['data']
                self.ground_site_grids = cached_data['ground_site_grids']
                
                air_sens_loc_array = cached_data['air_sens_loc']
                if isinstance(air_sens_loc_array, np.ndarray):
                    self.air_sens_loc = air_sens_loc_array.item() if air_sens_loc_array.size == 1 else {}
                
                if 'sensor_names' in cached_data:
                    self.sensor_names = cached_data['sensor_names'].tolist() if len(cached_data['sensor_names']) > 0 else list(self.air_sens_loc.keys())
                else:
                    self.sensor_names = list(self.air_sens_loc.keys())
                
                if 'target_stations' in cached_data:
                    self.target_stations = cached_data['target_stations']
                else:
                    self.target_stations = None
                
                print(f"✓ Successfully loaded processed data from cache")
                print(f"  - Data shape: {self.data.shape}")
                print(f"  - Found {len(self.air_sens_loc)} sensor locations")
                return
            except Exception as e:
                print(f"Error loading from cache: {e}. Will reprocess data.")
        
        # Process data from scratch
        list_df = self._get_airnow_data(start_date, end_date, extent, save_dir, airnow_api_key)
        
        if not list_df:
            raise ValueError("No valid AirNow data available.")
        
        print("Processing AirNow data with IDW interpolation (this may take time)...")    
        ground_site_grids = [self._preprocess_ground_sites(df, dim, extent) for df in list_df]
        
        print(f"Interpolating {len(ground_site_grids)} frames...")
        interpolated_grids = [self._interpolate_frame(frame) for frame in ground_site_grids]
        
        frames = np.expand_dims(np.array(interpolated_grids), axis=-1)
        processed_ds = self._sliding_window_of(frames, frames_per_sample)

        self.data = processed_ds
        self.ground_site_grids = ground_site_grids
        
        if self.air_sens_loc:
            self.target_stations = self._get_target_stations(self.data, self.ground_site_grids, self.air_sens_loc)
            self.sensor_names = list(self.air_sens_loc.keys())
        else:
            self.target_stations = None
            print("Warning: No air sensor locations found in the data.")
        
        # Save processed data to cache
        print(f"Saving processed AirNow data to cache: {processed_cache_dir}")
        try:
            air_sens_loc_array = np.array([self.air_sens_loc])
            sensor_names_array = np.array(self.sensor_names)
            target_stations_array = self.target_stations if self.target_stations is not None else np.array([])
            
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
        """Normalize elevation data to prevent overflow in calculations."""
        min_val = np.min(elevation_data)
        max_val = np.max(elevation_data)
        
        if max_val == min_val:
            return np.zeros_like(elevation_data)
            
        normalized = (elevation_data - min_val) / (max_val - min_val) * 100
        return normalized.astype(np.float32)

    def _get_airnow_data(self, start_date, end_date, extent, save_dir, airnow_api_key):
        """Download or load AirNow data from API."""
        lon_bottom, lon_top, lat_bottom, lat_top = extent
        
        if os.path.exists(save_dir):
            print(f"'{save_dir}' already exists; skipping request...")
        else:
            date_start = pd.to_datetime(start_date).isoformat()[:13]
            date_end = pd.to_datetime(end_date).isoformat()[:13]
            bbox = f'{lon_bottom},{lat_bottom},{lon_top},{lat_top}'
            URL = "https://www.airnowapi.org/aq/data"

            PARAMS = {
                'startDate': date_start,
                'endDate': date_end,
                'parameters': 'PM25',
                'BBOX': bbox,
                'dataType': 'A',
                'format': 'application/json',
                'verbose': '1',
                'monitorType': '2',
                'includerawconcentrations': '1',
                'API_KEY': airnow_api_key
            }

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
            
            # Check for API errors
            if isinstance(airnow_data, list) and len(airnow_data) > 0 and isinstance(airnow_data[0], dict):
                if 'WebServiceError' in airnow_data[0]:
                    print(f"Error from AirNow API: {airnow_data[0]['WebServiceError']}")
                    return []
            
            airnow_df = pd.json_normalize(airnow_data)
            
            # Handle UTC column
            if 'UTC' not in airnow_df.columns:
                print("Error: 'UTC' column not found in AirNow data.")
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
        """Convert ground sites data into grid format, optionally filtering by mask."""
        lonMin, lonMax, latMin, latMax = extent
        latDist, lonDist = abs(latMax - latMin), abs(lonMax - lonMin)
        unInter = np.zeros((dim, dim))
        
        required_columns = ['Latitude', 'Longitude', 'Value', 'SiteName']
        if not all(col in df.columns for col in required_columns):
            print(f"Warning: Missing required columns in dataframe. Available columns: {df.columns}")
            return unInter
            
        dfArr = np.array(df[required_columns])
        excluded_sensors = []
        included_sensors = []
        
        for i in range(dfArr.shape[0]):
            # Calculate grid coordinates
            x = int(((latMax - dfArr[i,0]) / latDist) * dim)
            x = max(0, min(x, dim - 1))
            
            y = int(((dfArr[i,1] - lonMin) / lonDist) * dim)
            y = max(0, min(y, dim - 1))
            
            # Apply mask filtering only if use_mask is True
            if self.use_mask and self.mask is not None:
                # First check if coordinates are within mask bounds
                if (x < 0 or x >= self.mask.shape[0] or 
                    y < 0 or y >= self.mask.shape[1]):
                    excluded_sensors.append({
                        'name': dfArr[i,3],
                        'lat': dfArr[i,0],
                        'lon': dfArr[i,1],
                        'value': dfArr[i,2],
                        'grid_x': x,
                        'grid_y': y,
                        'reason': 'out_of_bounds'
                    })
                    continue  # Skip sensors outside mask bounds
                
                # Then check if sensor is in a masked area (mask value == 0)
                elif self.mask[x, y] == 0:
                    excluded_sensors.append({
                        'name': dfArr[i,3],
                        'lat': dfArr[i,0],
                        'lon': dfArr[i,1],
                        'value': dfArr[i,2],
                        'grid_x': x,
                        'grid_y': y,
                        'reason': 'masked_area'
                    })
                    continue  # Skip sensors in masked areas
            
            # Include sensor in grid
            if dfArr[i,2] < 0:
                unInter[x, y] = 0  # Set negative values to 0 (valid PM2.5 baseline)
            else:
                unInter[x, y] = dfArr[i,2]
                sitename = dfArr[i,3]
                self.air_sens_loc[sitename] = (x, y)
                included_sensors.append({
                    'name': sitename,
                    'lat': dfArr[i,0],
                    'lon': dfArr[i,1],
                    'value': dfArr[i,2],
                    'grid_x': x,
                    'grid_y': y
                })
        
        # Report filtering results
        if self.use_mask and excluded_sensors:
            out_of_bounds = [s for s in excluded_sensors if s.get('reason') == 'out_of_bounds']
            masked_areas = [s for s in excluded_sensors if s.get('reason') == 'masked_area']
            
            print(f"Excluded {len(excluded_sensors)} sensors due to mask filtering:")
            if out_of_bounds:
                print(f"  - {len(out_of_bounds)} sensors out of mask bounds")
            if masked_areas:
                print(f"  - {len(masked_areas)} sensors in masked areas")
        
        print(f"Included {len(included_sensors)} sensors in interpolation")
        return unInter

    def _find_closest_values(self, x, y, coordinates, n=10):
        """Find n closest sensor locations for interpolation."""
        if not coordinates:
            return [], np.array([])
            
        coords_array = np.array(coordinates)
        diffs = coords_array - np.array([x, y])
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        
        closest_indices = np.argsort(distances)[:n]
        sorted_distances = distances[closest_indices]
        
        magnitude = np.linalg.norm(sorted_distances)
        if magnitude > 0:
            normalized_distances = sorted_distances / magnitude
        else:
            normalized_distances = sorted_distances
            
        closest_values = [coordinates[i] for i in closest_indices]
        return closest_values, normalized_distances

    def _find_elevations(self, x, y, coordinates):
        """Calculate elevation differences between points."""
        if not coordinates:
            return np.array([])
            
        stat = self.elevation[x, y]
        elevations = []
        for a, b in coordinates:
            if 0 <= a < self.elevation.shape[0] and 0 <= b < self.elevation.shape[1]:
                diff = np.float32(stat) - np.float32(self.elevation[a, b])
                elevations.append(diff)
            else:
                elevations.append(0.0)
                
        elevations = np.array(elevations, dtype=np.float32)
        magnitude = np.linalg.norm(elevations)
        if magnitude > 0:
            elevations = elevations / magnitude
            
        return elevations

    def _find_values(self, coordinates, unInter):
        """Get sensor values at specified coordinates."""
        values = []
        for a, b in coordinates:
            if 0 <= a < unInter.shape[0] and 0 <= b < unInter.shape[1]:
                values.append(unInter[a, b])
            else:
                values.append(0)
        return values

    def _idw_interpolate(self, x, y, values, distance_list, elevation_list, p=2):
        """Perform 3D IDW interpolation using distance and elevation."""
        if len(values) == 0:
            return 0
            
        difference_factor = distance_list + elevation_list**2
        eps = np.finfo(float).eps
        difference_factor[difference_factor == 0] = eps
        
        weights = 1 / difference_factor**p
        weights /= np.sum(weights)
        estimated_value = np.sum(weights * np.array(values))
        return estimated_value

    def _variable_blur(self, data, kernel_size):
        """Apply variable blur for smoothing."""
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
        """Interpolate a frame using 3D IDW method."""
        nonzero_indices = np.nonzero(unInter)
        coordinates = list(zip(nonzero_indices[0], nonzero_indices[1]))
        
        if not coordinates:
            return unInter
            
        # Initialize with zeros (valid PM2.5 background)
        interpolated = np.zeros((self.dim, self.dim), dtype=np.float32)
        
        for x in range(self.dim):
            for y in range(self.dim):
                if unInter[x, y] > 0:
                    # Use actual sensor measurement
                    interpolated[x, y] = unInter[x, y]
                    continue
                    
                coords, distance_list = self._find_closest_values(x, y, coordinates)
                if not coords:
                    continue
                    
                elevation_list = self._find_elevations(x, y, coords)
                vals = self._find_values(coords, unInter)
                
                # Always use IDW result - let the weights handle it naturally
                value = self._idw_interpolate(x, y, vals, distance_list, elevation_list, self.idw_power)
                interpolated[x, y] = max(0, value)  # PM2.5 can't be negative
        
        # Apply smoothing
        kernel_size = np.random.randint(0, 5, (self.dim, self.dim))
        out = self._variable_blur(interpolated, kernel_size)
        out = gaussian_filter(out, sigma=0.5)
        
        # Apply mask for geographic boundaries only if using mask
        if self.use_mask and self.mask is not None and np.any(self.mask != 1):
            out_masked = out.copy()
            out_masked[self.mask == 0] = 0  # Set masked areas to 0 (water bodies, etc.)
            out = out_masked
        
        return out

    def _sliding_window_of(self, frames, frames_per_sample):
        """Create sliding window samples from frames."""
        n_frames, row, col, channels = frames.shape
        n_samples = max(1, n_frames - frames_per_sample + 1)
        samples = np.empty((n_samples, frames_per_sample, row, col, channels))
        
        for i in range(n_samples):
            end_idx = min(i + frames_per_sample, n_frames)
            if end_idx - i < frames_per_sample:
                sample_frames = []
                for j in range(i, end_idx):
                    sample_frames.append(frames[j])
                for j in range(end_idx - i, frames_per_sample):
                    sample_frames.append(frames[end_idx-1] if end_idx > 0 else frames[0])
                samples[i] = np.array(sample_frames)
            else:
                samples[i] = np.array([frames[j] for j in range(i, i + frames_per_sample)])
            
        return samples

    def _get_target_stations(self, X, gridded_data, sensor_locations):
        """Generate target values for prediction at sensor locations."""
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
                    Y[sample][i] = gridded_data[-1][x][y]

        return Y

    def _grid_to_latlon(self, x, y):
        """Convert grid coordinates to latitude/longitude."""
        lon_min, lon_max, lat_min, lat_max = self.extent
        lat_dist = lat_max - lat_min
        lon_dist = lon_max - lon_min
        
        lat = lat_max - (x / self.dim) * lat_dist
        lon = lon_min + (y / self.dim) * lon_dist
        return lat, lon
    
    def _latlon_to_grid(self, lat, lon):
        """Convert latitude/longitude to grid coordinates."""
        lon_min, lon_max, lat_min, lat_max = self.extent
        lat_dist = lat_max - lat_min
        lon_dist = lon_max - lon_min
        
        x = int(((lat_max - lat) / lat_dist) * self.dim)
        y = int(((lon - lon_min) / lon_dist) * self.dim)
        
        x = max(0, min(x, self.dim - 1))
        y = max(0, min(y, self.dim - 1))
        return x, y

    def get_sensor_mask_status(self):
        """Get detailed information about sensor inclusion/exclusion by mask."""
        if not self.use_mask:
            return {"status": "Mask usage disabled", "all_sensors_included": True}
            
        if self.mask is None:
            return {"error": "Mask usage enabled but no mask available"}
        
        included_sensors = []
        for name, (x, y) in self.air_sens_loc.items():
            lat, lon = self._grid_to_latlon(x, y)
            included_sensors.append({
                'name': name,
                'grid_coords': (x, y),
                'lat_lon': (lat, lon),
                'mask_value': self.mask[x, y]
            })
        
        return {
            'mask_enabled': True,
            'included_sensors': included_sensors,
            'total_included': len(included_sensors),
            'mask_shape': self.mask.shape,
            'mask_coverage': f"{np.sum(self.mask == 1) / self.mask.size * 100:.1f}% of area is valid",
            'grid_extent': f"Grid covers x:[0-{self.mask.shape[0]-1}], y:[0-{self.mask.shape[1]-1}]"
        }

    def visualize_sensors_and_mask(self, figsize=(15, 10), save_path=None, dpi=100):
        """Visualize mask and sensor locations to show filtering results."""
        if not self.use_mask or self.mask is None:
            print("Cannot visualize mask: mask usage disabled or no mask available")
            return None, None
            
        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
        
        # Plot 1: Mask with sensor locations
        im1 = axes[0].imshow(self.mask, cmap='RdYlBu_r', alpha=0.7)
        axes[0].set_title('Mask with Sensor Locations', fontsize=14)
        
        # Add included sensors
        for name, (x, y) in self.air_sens_loc.items():
            axes[0].plot(y, x, '^', color='green', markersize=10, 
                        markeredgecolor='black', label='Included Sensors' if name == list(self.air_sens_loc.keys())[0] else "")
            axes[0].annotate(name, (y, x), xytext=(5, 5), textcoords='offset points',
                            fontsize=8, color='black', weight='bold',
                            bbox=dict(boxstyle="round,pad=0.2", fc="lightgreen", ec="black", alpha=0.8))
        
        axes[0].legend()
        plt.colorbar(im1, ax=axes[0], label='Mask Value (1=Valid, 0=Invalid)')
        
        # Plot 2: Geographic view
        ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
        lon_min, lon_max, lat_min, lat_max = self.extent
        ax2.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        
        ax2.add_feature(cfeature.COASTLINE)
        ax2.add_feature(cfeature.STATES, linestyle=':')
        ax2.add_feature(cfeature.LAND, alpha=0.1)
        ax2.add_feature(cfeature.OCEAN, alpha=0.3, color='blue')
        
        # Plot mask and sensors on map
        lons = np.linspace(lon_min, lon_max, self.dim)
        lats = np.linspace(lat_max, lat_min, self.dim)
        lons_mesh, lats_mesh = np.meshgrid(lons, lats)
        
        masked_mask = np.ma.masked_where(self.mask == 1, self.mask)
        ax2.pcolormesh(lons_mesh, lats_mesh, masked_mask, 
                      transform=ccrs.PlateCarree(), cmap='Reds', alpha=0.5)
        
        for name, (x, y) in self.air_sens_loc.items():
            lat, lon = self._grid_to_latlon(x, y)
            ax2.plot(lon, lat, '^', color='green', markersize=8, 
                    markeredgecolor='black', transform=ccrs.PlateCarree())
        
        ax2.set_title('Geographic View with Mask', fontsize=14)
        
        gl = ax2.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        
        plt.suptitle('Sensor Filtering by Mask', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, axes

    def visualize_sensor_locations(self, figsize=(12, 10), marker_size=120, show_names=True, 
                                  save_path=None, with_background=True, dpi=100):
        """Visualize sensor locations on a map."""
        if not self.air_sens_loc:
            raise ValueError("No sensor locations found in the AirNowData object")
        
        lon_min, lon_max, lat_min, lat_max = self.extent
        
        if with_background:
            fig = plt.figure(figsize=figsize, dpi=dpi)
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.STATES, linestyle=':')
            ax.add_feature(cfeature.LAND, alpha=0.1)
            ax.add_feature(cfeature.OCEAN, alpha=0.1)
            ax.add_feature(cfeature.LAKES, alpha=0.1)
            ax.add_feature(cfeature.RIVERS, alpha=0.1)
            
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            
            sensor_lons = []
            sensor_lats = []
            sensor_names_list = []
            
            for name, (x, y) in self.air_sens_loc.items():
                lat, lon = self._grid_to_latlon(x, y)
                sensor_lats.append(lat)
                sensor_lons.append(lon)
                sensor_names_list.append(name)
            
            ax.scatter(sensor_lons, sensor_lats, s=marker_size, c='red', marker='^', 
                      edgecolor='black', linewidth=1, alpha=0.8, 
                      transform=ccrs.PlateCarree(), label='Air Quality Sensors')
            
            if show_names:
                for i, name in enumerate(sensor_names_list):
                    ax.annotate(name, (sensor_lons[i], sensor_lats[i]), 
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=9, color='black', weight='bold',
                              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
            
        else:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            
            ax.set_xlim(lon_min, lon_max)
            ax.set_ylim(lat_min, lat_max)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.grid(True, linestyle='--', alpha=0.5)
            
            sensor_lons = []
            sensor_lats = []
            sensor_names_list = []
            
            for name, (x, y) in self.air_sens_loc.items():
                lat, lon = self._grid_to_latlon(x, y)
                sensor_lats.append(lat)
                sensor_lons.append(lon)
                sensor_names_list.append(name)
            
            ax.scatter(sensor_lons, sensor_lats, s=marker_size, c='red', marker='^', 
                      edgecolor='black', linewidth=1, alpha=0.8, label='Air Quality Sensors')
            
            if show_names:
                for i, name in enumerate(sensor_names_list):
                    ax.annotate(name, (sensor_lons[i], sensor_lats[i]), 
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=9, color='black', weight='bold',
                              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
        
        plt.title('AirNow Sensor Locations', fontsize=16)
        plt.legend(loc='upper right')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        return fig, ax

    def visualize_interpolated_data(self, frame_index=0, sample_index=0, figsize=(14, 10), 
                                   show_sensors=True, marker_size=80, cmap='plasma_r', 
                                   save_path=None, vmin=None, vmax=None,
                                   alpha=0.7, with_background=True, dpi=100):
        """Visualize interpolated AirNow data on a map."""
        if not hasattr(self, 'data') or self.data is None:
            raise ValueError("No interpolated data available in the AirNowData object")
            
        try:
            interpolated_data = self.data[sample_index, frame_index, :, :, 0]
        except IndexError:
            raise ValueError(f"Sample index {sample_index} or frame index {frame_index} is out of range")
        
        if hasattr(self, 'ground_site_grids') and self.ground_site_grids is not None:
            if len(self.ground_site_grids) > sample_index:
                raw_grid = self.ground_site_grids[sample_index]
            else:
                raw_grid = None
        else:
            raw_grid = None
            
        lon_min, lon_max, lat_min, lat_max = self.extent
            
        # Create AQI-like colormap if requested
        if cmap == 'aqi':
            colors = [(0, 1, 0), (1, 1, 0), (1, 0.5, 0), (1, 0, 0), (0.5, 0, 0.5), (0.5, 0, 0)]
            cmap = LinearSegmentedColormap.from_list('aqi_cmap', colors)
            
        if vmin is None:
            vmin = np.min(interpolated_data[interpolated_data > 0])
        if vmax is None:
            vmax = np.max(interpolated_data)
            
        if with_background:
            fig = plt.figure(figsize=figsize, dpi=dpi)
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.STATES, linestyle=':')
            ax.add_feature(cfeature.LAND, alpha=0.1)
            ax.add_feature(cfeature.OCEAN, alpha=0.1)
            ax.add_feature(cfeature.LAKES, alpha=0.1)
            ax.add_feature(cfeature.RIVERS, alpha=0.1)
            
            lons = np.linspace(lon_min, lon_max, self.dim)
            lats = np.linspace(lat_max, lat_min, self.dim)
            lons_mesh, lats_mesh = np.meshgrid(lons, lats)
            
            # Mask out zero values for better visualization
            masked_data = np.ma.masked_where(interpolated_data <= 0, interpolated_data)
            c = ax.pcolormesh(lons_mesh, lats_mesh, masked_data, 
                            transform=ccrs.PlateCarree(), 
                            cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
            
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            
        else:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            
            masked_data = np.ma.masked_where(interpolated_data <= 0, interpolated_data)
            c = ax.imshow(masked_data, cmap=cmap, vmin=vmin, vmax=vmax, 
                         extent=[lon_min, lon_max, lat_min, lat_max],
                         origin='upper', alpha=alpha)
            
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.grid(True, linestyle='--', alpha=0.5)
            
        cbar = plt.colorbar(c, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label('PM2.5 Concentration (μg/m³)', fontsize=12)
            
        if show_sensors and self.air_sens_loc:
            sensor_lons = []
            sensor_lats = []
            sensor_names_list = []
            sensor_values = []
            
            for name, (x, y) in self.air_sens_loc.items():
                lat, lon = self._grid_to_latlon(x, y)
                sensor_lats.append(lat)
                sensor_lons.append(lon)
                sensor_names_list.append(name)
                
                if raw_grid is not None:
                    try:
                        value = raw_grid[x, y]
                        sensor_values.append(value)
                    except IndexError:
                        sensor_values.append(None)
                else:
                    sensor_values.append(None)
            
            sc = ax.scatter(sensor_lons, sensor_lats, s=marker_size, c='white', marker='^', 
                          edgecolor='black', linewidth=1, alpha=1.0,
                          label='Air Quality Sensors', zorder=5)
            
            for i, name in enumerate(sensor_names_list):
                label = name
                if sensor_values[i] is not None and sensor_values[i] > 0:
                    label += f"\n({sensor_values[i]:.1f})"
                    
                ax.annotate(label, (sensor_lons[i], sensor_lats[i]), 
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=9, color='black', weight='bold',
                          bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                          zorder=6)
        
        title = f'Interpolated PM2.5 Concentration - Sample {sample_index}, Frame {frame_index}'
        plt.title(title, fontsize=16)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        return fig, ax

    def compare_raw_vs_interpolated(self, sample_index=0, figsize=(18, 8), save_path=None,
                                  cmap='plasma_r', dpi=100):
        """Create side-by-side comparison of raw vs interpolated data."""
        if not hasattr(self, 'ground_site_grids') or self.ground_site_grids is None:
            raise ValueError("No raw ground site data available")
            
        if not hasattr(self, 'data') or self.data is None:
            raise ValueError("No interpolated data available")
            
        try:
            raw_grid = self.ground_site_grids[sample_index]
        except IndexError:
            raise ValueError(f"Sample index {sample_index} is out of range for ground_site_grids")
            
        try:
            interpolated_data = self.data[sample_index, 0, :, :, 0]
        except IndexError:
            raise ValueError(f"Sample index {sample_index} is out of range for interpolated data")
            
        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
        
        # Plot raw data
        masked_raw = np.ma.masked_where(raw_grid <= 0, raw_grid)
        im1 = axes[0].imshow(masked_raw, cmap=cmap)
        axes[0].set_title('Raw Sensor Data (Before Interpolation)', fontsize=14)
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], orientation='vertical', fraction=0.046, pad=0.04)
        
        for name, (x, y) in self.air_sens_loc.items():
            value = raw_grid[x, y]
            if value > 0:
                axes[0].plot(y, x, 'w^', markersize=10, markeredgecolor='black')
                axes[0].annotate(f"{name} ({value:.1f})", (y, x), 
                                xytext=(5, 5), textcoords='offset points',
                                fontsize=8, color='black', weight='bold',
                                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))
        
        # Plot interpolated data
        masked_interp = np.ma.masked_where(interpolated_data <= 0, interpolated_data)
        im2 = axes[1].imshow(masked_interp, cmap=cmap)
        axes[1].set_title('Interpolated Data (IDW Method)', fontsize=14)
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)
        
        for name, (x, y) in self.air_sens_loc.items():
            axes[1].plot(y, x, 'w^', markersize=10, markeredgecolor='black')
        
        plt.suptitle(f'Raw Sensor Data vs. Interpolated PM2.5 Concentration - Sample {sample_index}', 
                    fontsize=16)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.tight_layout()
        return fig, axes