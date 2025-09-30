import numpy as np
from scipy.ndimage import gaussian_filter
import cv2
from tqdm import tqdm
import os

class IDW:
    def __init__(
        self,
        power=2.0,
        neighbors=10,
        dim=40,
        elevation_path=None,
        use_variable_blur=False,
        verbose=0   # 0 = all, 1 = progress bar + errors, 2 = errors only
    ):
        """
        Performs IDW interpolation.

        Expects list of sparsely gridded data, where pixels to interpolate with
            are a valid number, and pixels to interpolate over are np.nan

        If elevation data is provided, it is expected to be of the same extent.

        Usage: 
            - Initialize the IDW parameters with IDW()
            - Call interpolate_frames(frames). All parameters should already
                be initialized from IDW().
        """
        # members
        self.power = power
        self.neighbors = neighbors
        self.dim = dim
        self.VERBOSE = verbose
        self.use_variable_blur = use_variable_blur
        self.elevation = self._get_elevation_data(elevation_path, dim)

        return

    ### NOTE: Public method
    def interpolate_frames(self, frames):
        closest_coords_and_dists = self._init_closest_coords_and_dists_per_pixel(
            unInter=frames[0],
            dim=self.dim,
            coordinates=self._init_sensor_coords_from_grid(frames[0]),
            neighbors=self.neighbors,
            elevation=self.elevation
        )
        # TODO remove prints
        print(closest_coords_and_dists)

        closest_elevation_diffs = self._init_elevation_dists_per_pixel(
            self.elevation,
            frames[0],
            self.dim,
            closest_coords_and_dists
        )
        print(closest_elevation_diffs)
        print()

        interpolated_grids = [
            self._interpolate_frame(
                frame,
                self.dim,
                closest_coords_and_dists,
                closest_elevation_diffs,
                self.power,
                self.use_variable_blur
            )
            for frame in (
                tqdm(frames) 
                if self.VERBOSE < 2 else frames 
            )
        ]

        return np.array(interpolated_grids)
        
    ### NOTE: Validation methods

    def _get_elevation_data(self, elevation_path, dim):
        def validate_path(elevation_path, dim):
            flat_elevation = np.zeros((dim, dim), dtype=np.float32)

            if elevation_path is None:
                if self.VERBOSE == 0:
                    print("Elevation path set to None. Using flat elevation.")
                elevation = flat_elevation
            elif os.path.exists(elevation_path):
                elevation = np.load(elevation_path)
            else:
                raise ValueError(
                    f"Invalid path {elevation_path}. "
                    "Pass in a valid path or None for flat elevation."
                )
            return elevation

        def validate_shape(elevation, dim):
            if elevation.shape != (dim, dim):
                if self.VERBOSE == 0:
                    print(
                        f"Elevation data {elevation.shape} does not match "
                        f"({dim}, {dim}), resizing"
                    ) 
                elevation = cv2.resize(elevation, (dim, dim))
            return elevation

        def min_max_scale(elevation):
            min_val = np.min(elevation)
            max_val = np.max(elevation)
            
            if max_val == min_val:
                return np.zeros_like(elevation)
                
            normalized = (elevation - min_val) / (max_val - min_val) * 100
            return normalized.astype(np.float32)

        elevation = validate_path(elevation_path, dim)
        elevation = validate_shape(elevation, dim)
        elevation = min_max_scale(elevation)

        if self.VERBOSE == 0:
            print(
                "⛰️  Elevation data statistics:\n"
                f"  mean:   {np.mean(elevation):.2f}\n"
                f"  std:    {np.std(elevation):.2f}\n"
                f"  median: {np.median(elevation):.2f}\n"
                f"  min:    {np.min(elevation):.2f}\n"
                f"  max:    {np.max(elevation):.2f}\n"
            )

        return elevation

    ### NOTE: Methods that perform one-time calculations regardless of number of frames

    def _init_sensor_coords_from_grid(self, unInter):
        """
        Initializes where the locations of the sensors are based on whether the
            pixel is NaN or not. This means that we expect the frame to contain
            ALL the sensor values (pre-imputed), and non-sensor locations to be
            NaN.
        """
        sensor_indices = np.where(~np.isnan(unInter))
        coordinates = list(zip(sensor_indices[0], sensor_indices[1]))
        if not coordinates:
            print(
                "No non-nan points found on grid, returning uninterpolated frame.\n"
                "Note: non-nan points are used to determine sensor locations."
            )

        return coordinates

    def _init_closest_coords_and_dists_per_pixel(
        self,
        unInter,
        dim,
        coordinates,
        neighbors,
        elevation
    ):
        """
        For every pixel, generates a pair:
            1. The coordinates of the closest sensors
                e.g. [(0,2), (3,5), ... ]
            2. The distances of those sensors
                e.g. [13, 24, ...]

        For the pixel that is itself a sensor location, it will be set to NaN
        """
        if neighbors > len(coordinates):
            print(
                f"Neighbors cannot exceed number of sensors; setting neighbors to "
                f"{len(coordinates)}."
            )
            neighbors = len(coordinates)

        closest_coords_and_dists = [[np.nan for _ in range(dim)] for _ in range(dim)] 
        # FIXME ELEVATION GOES HERE. PERHAPS COORDINATES = (X, Y, Z)?
        # then np.sqrt((x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2)^2)
        # z1 = elevation_grid[x, y], z2 = elevation_grid[a, b] where (a, b) in coordinates
        for x in range(dim):
            for y in range(dim):
                closest_coords_and_dists[x][y] = (
                    self._find_closest_values(x, y, coordinates, elevation, neighbors)
                    if np.isnan(unInter[x, y])
                    else np.nan
                )

        return closest_coords_and_dists

    def _init_elevation_dists_per_pixel(self, elevation, unInter, dim, closest_coords_and_dists):
        def extract_coords_from_coords_and_dists(closest_coords_and_dists):
            closest_coords = [[np.nan for _ in range(dim)] for _ in range(dim)] 
            for x in range(self.dim):
                for y in range(self.dim):
                    val = (
                        closest_coords_and_dists[x][y][0]
                        # checks nan or pair of lists
                        if type(closest_coords_and_dists[x][y]) != float
                        else np.nan
                    )
                    closest_coords[x][y] = val
            return closest_coords

        closest_coords = extract_coords_from_coords_and_dists(closest_coords_and_dists)
        closest_elevation_diffs = [[np.nan for _ in range(dim)] for _ in range(dim)] 
        for x in range(dim):
            for y in range(dim):
                closest_elevation_diffs[x][y] = (
                    self._find_elevation_diff(elevation, x, y, closest_coords[x][y])
                    if np.isnan(unInter[x, y])
                    else np.nan
                )

        return closest_elevation_diffs

    ### NOTE: Helpers

    def _find_closest_values(self, x, y, coordinates, elevation, n):
        """Find n closest sensor locations for interpolation."""
        if not coordinates:
            return [], np.array([])
            
        # TODO remove
        '''
        coords_array = np.array(coordinates)
        diffs = coords_array - np.array([x, y])
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        '''
        # euclidian dist with 3 dimensions 
        x1_arr = np.array([x for x, y in coordinates])
        y1_arr = np.array([y for x, y in coordinates])
        xs, ys = zip(*coordinates)
        z1_arr = np.array([z for z in elevation[xs, ys]])

        x2, y2 = x, y
        z2 = elevation[x, y]

        distances = np.sqrt((x1_arr - x2)**2 + (y1_arr - y2)**2 + (z1_arr - z2)**2)
        
        closest_indices = np.argsort(distances)[:n]
        sorted_distances = distances[closest_indices]
        
        # TODO remove?
        magnitude = np.linalg.norm(sorted_distances)
        if magnitude > 0:
            normalized_distances = sorted_distances / magnitude
        else:
            normalized_distances = sorted_distances
            
        closest_values = [coordinates[i] for i in closest_indices]
        return closest_values, sorted_distances
        return closest_values, normalized_distances

    def _find_values(self, coordinates, unInter):
        """Get sensor values at specified coordinates."""
        def validate_coordinates(coordinates, unInter):
            n, m = unInter.shape
            for a, b in coordinates:
                err_msg = f"Coordinates ({a},{b}) out of bound for shape ({n},{m})."
                if 0 > a >= n or 0 > b >= m:
                    raise ValueError(err_msg)

        validate_coordinates(coordinates, unInter)
        return [unInter[a, b] for a, b in coordinates]

    def _find_elevation_diff(self, elevation_grid, x, y, coordinates):
        """Calculate elevation differences between points."""
        if not coordinates:
            return np.array([])
            
        stat = elevation_grid[x, y]
        elevations = []
        for a, b in coordinates:
            if 0 <= a < elevation_grid.shape[0] and 0 <= b < elevation_grid.shape[1]:
                diff = np.float32(stat) - np.float32(elevation_grid[a, b])
                elevations.append(diff)
            else:
                elevations.append(0.0)
                
        elevations = np.array(elevations, dtype=np.float32)
        # TODO remove?
        magnitude = np.linalg.norm(elevations)
        if magnitude > 0:
            elevations = elevations / magnitude
            
        return elevations

    ### NOTE: Core interpolation methods

    def _idw_interpolate(self, value_list, distance_list, elevation_list, p):
        """Perform 3D IDW interpolation using x, y and z distance"""
        def validate_params(value_list, distance_list, elevation_list):
            if not (
                len(value_list) == 
                len(distance_list) == 
                len(elevation_list)
            ):
                raise ValueError(
                    f"The number of values ({len(values)}), distances "
                    f"({len(distance_list)}), and elevations "
                    f"({len(elevation_list)}) must match."
                )
            return

        validate_params(value_list, distance_list, elevation_list)
        distances = np.sqrt(distance_list**2 + 5*elevation_list**2)

        estimate = np.sum(value_list / distances**p) / np.sum(1 / distances**p)

        return estimate

    def _interpolate_frame(
        self,
        unInter,
        dim,
        closest_coords_and_dists,
        closest_elevation_diffs,
        power,
        use_variable_blur
    ):
        """Interpolate a frame using 3D IDW method."""
        interpolated = np.full((dim, dim), np.nan)
        
        for x in range(dim):
            for y in range(dim):
                if not np.isnan(unInter[x, y]):
                    interpolated[x, y] = unInter[x, y]
                else:
                    closest_coords, closest_dists = closest_coords_and_dists[x][y] 
                    closest_sensor_vals = self._find_values(closest_coords, unInter)
                    closest_elevation_diffs_on_xy = closest_elevation_diffs[x][y]
                    
                    interpolated[x, y] = self._idw_interpolate(
                        closest_sensor_vals,
                        closest_dists,
                        closest_elevation_diffs_on_xy,
                        power 
                    )
        
        # Apply smoothing
        if use_variable_blur:
            kernel_size = np.random.randint(0, 5, (dim, dim))
            interpolated = _variable_blur(interpolated, kernel_size)
            interpolated = gaussian_filter(interpolated, sigma=0.5)
        
        return interpolated 

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

