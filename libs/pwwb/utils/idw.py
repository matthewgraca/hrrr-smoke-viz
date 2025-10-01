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
        elevation_scale_factor=100,
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
        self.elevation = self._get_elevation_data(elevation_path, dim, elevation_scale_factor)

        return

    ### NOTE: Public method

    def interpolate_frames(self, frames):
        """
        Interpolates frames.

        To avoid recomputing distances, we compute the closest coordinates, 
            distances, and elevation differences for each pixel prior to 
            interpolation.
        """
        first_frame = frames[0]
        if not self._validate_grid_is_interpolatable(first_frame):
            return frames

        closest_coords_and_dists = self._get_closest_coords_and_dists_per_pixel(
            unInter=first_frame,
            dim=self.dim,
            sensor_coords=self._get_sensor_coords(first_frame),
            neighbors=self.neighbors,
            elevation_grid=self.elevation
        )

        interpolated_grids = [
            self._interpolate_frame(
                frame,
                self.dim,
                closest_coords_and_dists,
                self.power,
                self.use_variable_blur
            )
            for frame in (
                tqdm(frames) 
                if self.VERBOSE < 2 else frames 
            )
        ]

        return np.array(interpolated_grids)
        
    ### NOTE: Methods that perform one-time calculations regardless of number of frames

    def _get_elevation_data(self, elevation_path, dim, elevation_scale_factor):
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

        def min_max_scale(elevation, elevation_scale_factor):
            """
            Elevation scale factor controls how "powerful" elevation is.

            Note that distances are not equal. A pixel of the current default
                extent is about 4.86km, and the highest elevation is 2.5km.

            If we considered elevation equal to x/y dimension, (by dividing
                elevation by 4.86km), then elevation would have barely any 
                effect on IDW.
            
            We can increase the scale of elevation to improve its role in IDW.
                Intuitively, pollution (is complicated) but is affected more
                by elevation than by distance on the x/y plane, so it makes
                sense to boost its importance.

            But by how much? You decide. For us, 100-500 is something we played
                around with that looked reasonable.
            """
            min_val = np.min(elevation)
            max_val = np.max(elevation)
            
            if max_val == min_val:
                return np.zeros_like(elevation)
                
            normalized = (elevation - min_val) / (max_val - min_val)
            return normalized.astype(np.float32) * elevation_scale_factor

        elevation = validate_path(elevation_path, dim)
        elevation = validate_shape(elevation, dim)
        elevation = min_max_scale(elevation, elevation_scale_factor)

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

    def _validate_grid_is_interpolatable(self, unInter):
        """
        Checks for sensor values (numbers) and nan values to interpolate.
        
        If there are no nans, that means nothing can be interpolated.
        If there are only nans, then there are no real values to interpolate
            with.
        """
        sensor_indices = np.where(~np.isnan(unInter))
        if len(sensor_indices[0]) == 0:
            print(
                "No non-nan points found on grid, returning uninterpolated frame.\n"
                "Note: non-nan points are used to determine sensor locations."
            )
            return False

        x_dim, y_dim = unInter.shape

        if np.isnan(unInter).all():
            raise ValueError("Every value is nan; no value to interpolate.")

        return True
    
    def _get_sensor_coords(self, unInter):
        """
        Initializes where the locations of the sensors are based on whether the
            pixel is NaN or not. This means that we expect the frame to contain
            ALL the sensor values (pre-imputed), and non-sensor locations to be
            NaN.
        """
        sensor_indices = np.where(~np.isnan(unInter))
        x_idxs, y_idxs = sensor_indices[0], sensor_indices[1]
        coordinates = list(zip(x_idxs, y_idxs))

        return coordinates

    def _get_closest_coords_and_dists_per_pixel(
        self,
        unInter,
        dim,
        sensor_coords,
        neighbors,
        elevation_grid
    ):
        """
        For every pixel, generates a map:
            1. The coordinates of the closest sensors
            2. The distances of those sensors
            e.g At pixel (0, 0):
            {
                (0,2) : 13,
                (3,5) : 24,
                ...
            }
            
            The map is in sorted order, by distance.

        For the pixel that is itself a sensor location, it will have a map
            with its own coordinate mapped to a value of 0.
        """
        if neighbors > len(sensor_coords):
            print(
                f"Neighbors cannot exceed number of sensors; setting neighbors to "
                f"{len(coordinates)}."
            )
            neighbors = len(sensor_coords)

        closest_coords_and_dists = [[np.nan for _ in range(dim)] for _ in range(dim)] 
        for x in range(dim):
            for y in range(dim):
                closest_coords_and_dists[x][y] = (
                    self._find_closest_sensors_and_distances(
                        x, y, sensor_coords, elevation_grid, neighbors
                    )
                )

        return closest_coords_and_dists

    ### NOTE: Helpers

    def _find_closest_sensors_and_distances(
        self,
        x, y,           
        sensor_coords,  
        elevation_grid, 
        neighbors       
    ):
        """
        Find n closest sensor locations for interpolation.

        (x, y): source point
        sensor_coords: list of sensor locations (destinations)
        elevation_grid: gridded elevation data providing z-axis data
        neighbors: the closest n sensors and their distances to return

        Returns a dictionary mapping the closest sensor coordinates and 
            their distances. Python dictionaries should return dictionaries
            in the order the entries are added, so it should also be in
            sorted order when iterating.

        If the source point is a sensor coordinate, then we simply return
            the point mapping to 0.
        """
        if not sensor_coords:
            raise ValueError("No sensor locations given; aborting IDW.")
        if (x, y) in sensor_coords:
            return { (x, y) : 0 }
            
        # euclidian dist with 3 dimensions 
        def euclidian_dist(x, y, sensor_coords, elevation_grid):
            """
            Calculates a list of euclidian distances with three dimensions.
                x, y: The starting point
                sensor_coords: The list of pairs of points on the x and y axis
                elevation: The list of points on the z axis
            """
            x1 = np.array([x for x, y in sensor_coords])
            y1 = np.array([y for x, y in sensor_coords])
            x_coords, y_coords = zip(*sensor_coords)
            z1 = np.array([z for z in elevation_grid[x_coords, y_coords]])

            x2, y2 = x, y
            z2 = elevation_grid[x, y]

            return np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
        
        distances = euclidian_dist(x, y, sensor_coords, elevation_grid)
        closest_indices = np.argsort(distances)[:neighbors]
        closest_dists = distances[closest_indices]
        closest_sensor_coords = [sensor_coords[i] for i in closest_indices]

        # likely should keep this off
        #unit_distances = sorted_distances / np.linalg.norm(sorted_distances) 

        return dict(zip(closest_sensor_coords, closest_dists))

    ### NOTE: Core interpolation methods

    def _idw_interpolate(self, unInter, closest_coords_and_dists, p):
        """Perform 3D IDW interpolation using x, y and z distance"""
        sensor_coords, distance_list = zip(*closest_coords_and_dists.items())
        value_list = [unInter[x, y] for x, y in sensor_coords]
        values, distances = np.array(value_list), np.array(distance_list)

        estimate = np.sum(values / distances**p) / np.sum(1 / distances**p)

        return estimate

    def _interpolate_frame(
        self,
        unInter,
        dim,
        closest_coords_and_dists,
        power,
        use_variable_blur
    ):
        """Interpolate a frame using 3D IDW method."""
        interpolated = np.full((dim, dim), np.nan)
        
        for x in range(dim):
            for y in range(dim):
                interpolated[x, y] = (
                    self._idw_interpolate(
                        unInter, closest_coords_and_dists[x][y], power
                    )
                    if np.isnan(unInter[x, y])
                    else unInter[x, y]
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

