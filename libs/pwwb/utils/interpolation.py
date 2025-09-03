import numpy as np
from scipy.ndimage import gaussian_filter
import cv2


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


def interpolate_frame(
    f,                  
    dim,                
    apply_filter=False, 
    interp_flag=0,      
    power=2.0,           
    epsilon=1e-6
):
    """
    Interpolates sparse data across a grid using Inverse Distance Weighting (IDW).
    
    Parameters:
    -----------
    f : numpy.ndarray
        Sparse grid with values at known locations
    dim : int
        Grid dimension
    apply_filter : bool
        Determines if IDW should apply Gaussian filter
    interp_flag : any
        Value that IDW should interpolate over. Usually 0 or np.nan 
        Use case: if 0 is a valid value, then you would use np.nan as the
        sentinel value that determines what cell gets interpolated
    power : double
        Controls the significance of the known points. More = nearby points 
        have more influence
    
    Returns:
    --------
    numpy.ndarray
        Smoothly interpolated grid
    """
    x_list, y_list, values = [], [], []
    for x in range(f.shape[0]):
        for y in range(f.shape[1]):
            add_xy_to_list = False

            if np.isnan(interp_flag):
                if not np.isnan(f[x, y]):
                    add_xy_to_list = True
            else:
                if f[x, y] != interp_flag:
                    add_xy_to_list = True

            if add_xy_to_list:
                x_list.append(x)
                y_list.append(y)
                values.append(f[x, y])
    
    coords = list(zip(x_list, y_list))
    
    if not coords:
        return np.zeros((dim, dim))
    
    if len(coords) == 1:
        interpolated = np.zeros((dim, dim))
        x0, y0 = coords[0]
        value = values[0]
        radius = dim * 0.3
        
        for x in range(dim):
            for y in range(dim):
                dist = np.sqrt((x - x0)**2 + (y - y0)**2)
                if dist < radius:
                    influence = np.exp(-2 * (dist / radius)**2)
                    interpolated[x, y] = value * influence
        
        return interpolated
    
    try:
        interpolated = np.zeros((dim, dim))
        epsilon = 1e-6
        
        for i in range(dim):
            for j in range(dim):
                distances = np.sqrt([(i - x)**2 + (j - y)**2 for x, y in coords])
                distances = np.array(distances)
                
                exact_match = np.where(distances < 1e-10)[0]
                if len(exact_match) > 0:
                    interpolated[i, j] = values[exact_match[0]]
                    continue
                
                weights = 1.0 / (distances**power + epsilon)
                normalized_weights = weights / np.sum(weights)
                interpolated[i, j] = np.sum(normalized_weights * np.array(values))
        
        return (
            gaussian_filter(interpolated, sigma=1.5, mode='constant', cval=0)
            if apply_filter 
            else interpolated
        )
        
    except Exception as e:
        print(f"Error in IDW interpolation: {e}")
        return np.zeros((dim, dim))


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
