import numpy as np
from scipy.ndimage import gaussian_filter
import cv2
from tqdm import tqdm

def preprocess_ground_sites(df, dim, lat_max, lon_max, lat_dist, lon_dist, allow_negative=False):
    """
    Places ground station data onto a regular grid with bounds checking and missing value handling.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with lat, lon, and value columns
    dim : int
        Output grid dimension (dim x dim)
    lat_max, lon_max : float
        Maximum bounds of the geographic region
    lat_dist, lon_dist : float
        Total geographic distance spans
    allow_negative : bool
        If True, preserves negative values (for wind/temperature). If False, treats them as missing.
    
    Returns:
    --------
    numpy.ndarray
        Grid (dim x dim) with station values at their geographic positions
    """
    grid = np.zeros((dim, dim))
    data = np.array(df)
    
    for i in range(data.shape[0]):
        try:
            x = int(((lat_max - data[i, 0]) / lat_dist) * dim)
            y = dim - int(((lon_max + abs(data[i, 1])) / lon_dist) * dim)
            
            x = max(0, min(x, dim - 1))
            y = max(0, min(y, dim - 1))
            
            value = data[i, 2]
            
            if allow_negative:
                grid[x, y] = 0 if np.isnan(value) else value
            else:
                grid[x, y] = value if (value >= 0 and not np.isnan(value)) else 0
                    
        except Exception as e:
            print(f"Error placing point on grid: {e}")
    
    return grid

def interpolate_frames(
    frames,
    dim,
    power=2.0,
    neighbors=10,
    elevation_grid=None,
    use_variable_blur=False,
    use_progbar=True
):
    closest_coords_and_dists = _init_closest_coords_and_dists_per_pixel(
        unInter=frames[0],
        coordinates=_init_sensor_coords_from_grid(frames[0]),
        neighbors=10
    )
    # NOTE: elevation is disabled
    if elevation_grid is None:
        elevation_grid = np.zeros((dim, dim), dtype=np.float32)

    interpolated_grids = [
        interpolate_frame(frame, dim, closest_coords_and_dists, elevation_grid, power, use_variable_blur)
        for frame in (
            tqdm(frames) 
            if use_progbar else frames 
        )
    ]

    return np.array(interpolated_grids)

def _find_closest_values(x, y, coordinates, n=10):
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

def _init_sensor_coords_from_grid(unInter):
    sensor_indices = np.where(~np.isnan(unInter))
    coordinates = list(zip(sensor_indices[0], sensor_indices[1]))
    if not coordinates:
        print(
            "No non-nan points found on grid, returning uninterpolated frame.\n"
            "Note: non-nan points are used to determine sensor locations."
        )

    return coordinates

def _init_closest_coords_and_dists_per_pixel(unInter, coordinates, neighbors=10):
# generate a frame that has the closest sensor locations for each pixel that can be reused
# each pixel = two lists; one with the closest coordinates, the other with the closest distances
# if its an actual sensor location, make it np.nan
    if neighbors > len(coordinates):
        print(
            f"Neighbors cannot exceed number of sensors; setting neighbors to "
            f"{len(coordinates)}"
        )
        neighbors = len(coordinates)

    closest_coords_and_dists = [[np.nan for _ in range(40)] for _ in range(40)] 
    for x in range(40):
        for y in range(40):
            closest_coords_and_dists[x][y] = (
                _find_closest_values(x, y, coordinates, neighbors)
                if np.isnan(unInter[x, y])
                else np.nan
            )

    return closest_coords_and_dists

def interpolate_frame(
    unInter,
    dim,
    closest_coords_and_dists,
    elevation_grid,
    power=2.0,
    use_variable_blur=False
):
    """Interpolate a frame using 3D IDW method."""
    interpolated = np.full((dim, dim), np.nan)
    
    for x in range(dim):
        for y in range(dim):
            if not np.isnan(unInter[x, y]):
                interpolated[x, y] = unInter[x, y]
            else:
                closest_coords, closest_dists = closest_coords_and_dists[x][y] 
                closest_sensor_vals = _find_values(closest_coords, unInter)
                # TODO pull out for one-time calculation like with init_closest. Elevation doesn't change, so we shouldn't need to recompute
                # passed in as a parameter; closest_elevation_diffs[x][y]
                closest_elevation_diffs = _find_elevations(elevation_grid, x, y, closest_coords)
                
                interpolated[x, y] = _idw_interpolate(
                    closest_sensor_vals,
                    closest_dists,
                    closest_elevation_diffs,
                    power 
                )
    
    out = interpolated

    # Apply smoothing
    if use_variable_blur:
        kernel_size = np.random.randint(0, 5, (dim, dim))
        out = _variable_blur(interpolated, kernel_size)
        out = gaussian_filter(out, sigma=0.5)
    
    return out

def _find_values(coordinates, unInter):
    """Get sensor values at specified coordinates."""
    values = []
    for a, b in coordinates:
        if 0 <= a < unInter.shape[0] and 0 <= b < unInter.shape[1]:
            values.append(unInter[a, b])
        else:
            values.append(0)
    return values

def _variable_blur(data, kernel_size):
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

def _idw_interpolate(values, distance_list, elevation_list, p=2):
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

def _find_elevations(elevation_grid, x, y, coordinates):
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
    magnitude = np.linalg.norm(elevations)
    if magnitude > 0:
        elevations = elevations / magnitude
        
    return elevations

def elevation_aware_wind_interpolation(stations, u_values, v_values, extent, dim, 
                                     elevation_grid, power=1.0, elevation_weight=0.15, 
                                     smoothing_sigma=2.0, verbose=False):
    """
    Interpolates wind vectors using 3D IDW that considers both geographic distance and elevation differences.
    Applies identical weights to U and V components to preserve wind vector relationships.
    
    Parameters:
    -----------
    stations : list
        Station coordinates as [[lon, lat], ...]
    u_values, v_values : list
        Wind component values for each station
    extent : tuple
        Geographic bounds (min_lon, max_lon, min_lat, max_lat)
    dim : int
        Output grid dimension
    elevation_grid : numpy.ndarray
        Elevation data (dim x dim)
    power : float
        IDW power parameter (lower = smoother interpolation)
    elevation_weight : float
        Weight for elevation differences in distance calculation (0-1)
    smoothing_sigma : float
        Gaussian smoothing sigma applied after interpolation
    verbose : bool
        Print interpolation statistics
    
    Returns:
    --------
    tuple
        (u_grid, v_grid) - interpolated wind component grids
    """
    min_lon, max_lon, min_lat, max_lat = extent
    
    u_grid = np.zeros((dim, dim))
    v_grid = np.zeros((dim, dim))
    
    if len(stations) == 0:
        return u_grid, v_grid
    
    stations_array = np.array(stations)
    u_array = np.array(u_values)
    v_array = np.array(v_values)
    
    station_elevations = []
    for lon, lat in stations:
        x_idx = int((lon - min_lon) / (max_lon - min_lon) * (dim - 1))
        y_idx = int((lat - min_lat) / (max_lat - min_lat) * (dim - 1))
        x_idx = max(0, min(x_idx, dim - 1))
        y_idx = max(0, min(y_idx, dim - 1))
        
        station_elevations.append(elevation_grid[y_idx, x_idx])
    
    station_elevations = np.array(station_elevations)
    
    x_step = (max_lon - min_lon) / (dim - 1) if dim > 1 else 0
    y_step = (max_lat - min_lat) / (dim - 1) if dim > 1 else 0
    epsilon = 1e-6
    
    if verbose:
        print(f"3D Wind IDW: {len(stations)} stations")
        print(f"Elevation range: {np.min(station_elevations):.0f}-{np.max(station_elevations):.0f}m")
    
    for i in range(dim):
        for j in range(dim):
            lon = min_lon + j * x_step
            lat = min_lat + i * y_step
            grid_elevation = elevation_grid[i, j]
            
            horizontal_distances = np.sqrt((lon - stations_array[:, 0])**2 + 
                                         (lat - stations_array[:, 1])**2)
            
            elevation_diffs = np.abs(grid_elevation - station_elevations)
            max_elev_diff = np.max(elevation_diffs) if np.max(elevation_diffs) > 0 else 1
            normalized_elev_diffs = elevation_diffs / max_elev_diff
            
            exact_match = np.where(horizontal_distances < 1e-10)[0]
            if len(exact_match) > 0:
                u_grid[i, j] = u_array[exact_match[0]]
                v_grid[i, j] = v_array[exact_match[0]]
                continue
            
            combined_distances = horizontal_distances + elevation_weight * normalized_elev_diffs
            weights = 1.0 / (combined_distances**power + epsilon)
            normalized_weights = weights / np.sum(weights)
            
            u_grid[i, j] = np.sum(normalized_weights * u_array)
            v_grid[i, j] = np.sum(normalized_weights * v_array)
    
    if smoothing_sigma > 0:
        u_grid = gaussian_filter(u_grid, sigma=smoothing_sigma, mode='constant', cval=0)
        v_grid = gaussian_filter(v_grid, sigma=smoothing_sigma, mode='constant', cval=0)
    
    return u_grid, v_grid
