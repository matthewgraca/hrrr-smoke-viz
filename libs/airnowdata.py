import requests
import json
import os
import sys
import pandas as pd
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

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
        frames_per_sample=1,
        dim=200,
        idw_power=2,
        elevation_path=None,
        mask_path=None,
        create_elevation_mask=False
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
        
        # Load or create elevation data for 3D interpolation
        if os.path.exists(self.elevation_path):
            self.elevation = np.load(self.elevation_path)
            # Resize elevation to match dim if needed
            if self.elevation.shape != (dim, dim):
                self.elevation = cv2.resize(self.elevation, (dim, dim))
            # Normalize elevation to prevent overflow
            self.elevation = self._normalize_elevation(self.elevation)
        elif create_elevation_mask:
            print(f"Creating sample elevation data at {self.elevation_path}")
            self._create_sample_elevation()
        else:
            raise FileNotFoundError(f"Elevation data not found at {self.elevation_path}. "
                                   f"Set create_elevation_mask=True to create sample data.")
            
        # Load or create mask data for interpolation boundary handling
        if os.path.exists(self.mask_path):
            self.mask = np.load(self.mask_path)
            # Resize mask to match dim if needed
            if self.mask.shape != (dim, dim):
                self.mask = cv2.resize(self.mask, (dim, dim))
        elif create_elevation_mask:
            print(f"Creating sample mask data at {self.mask_path}")
            self._create_sample_mask()
        else:
            raise FileNotFoundError(f"Mask data not found at {self.mask_path}. "
                                   f"Set create_elevation_mask=True to create sample data.")

        # Create necessary directories
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        
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
        ground_site_grids = [
            self._preprocess_ground_sites(df, dim, extent) for df in list_df
        ]
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
        else:
            raise ValueError("No air sensor locations found in the data.")
    
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
    
    def _create_sample_elevation(self):
        """Create a simple elevation model with a central peak"""
        x, y = np.mgrid[0:self.dim, 0:self.dim]
        center_x, center_y = self.dim//2, self.dim//2
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Create a Gaussian peak with some random noise (smaller values)
        elevation = np.exp(-distance**2/(2*(self.dim/5)**2)) * 100  # Reduced from 1000 to 100
        elevation += np.random.normal(0, 2, (self.dim, self.dim))   # Reduced noise from 20 to 2
        
        # Normalize to prevent overflow
        elevation = self._normalize_elevation(elevation)
        
        # Save the elevation data
        np.save(self.elevation_path, elevation)
        self.elevation = elevation
        print(f"Created sample elevation data at {self.elevation_path}")
    
    def _create_sample_mask(self):
        """Create a circular mask with 1s inside the circle and 0s outside"""
        mask = np.ones((self.dim, self.dim))
        
        # Create coordinates grid
        x, y = np.mgrid[0:self.dim, 0:self.dim]
        center_x, center_y = self.dim//2, self.dim//2
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Create a circular mask
        radius = self.dim * 0.45
        mask[distance > radius] = 0
        
        # Save the mask data
        np.save(self.mask_path, mask)
        self.mask = mask
        print(f"Created sample mask data at {self.mask_path}")

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
            for sensor, loc in enumerate(sensor_locations):
                x, y = sensor_locations[loc]
                offset = sample + frames_per_sample
                
                if offset < len(gridded_data):
                    Y[sample][sensor] = gridded_data[offset][x][y]
                else:
                    # Use the last available data for out-of-bounds targets
                    Y[sample][sensor] = gridded_data[-1][x][y]

        return Y