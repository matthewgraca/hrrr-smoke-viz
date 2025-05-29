import numpy as np
from scipy.interpolate import griddata

def interpolate_to_grid(point_data, rows, cols, extent, method='cubic'):
    """
    Interpolate point data to a regular grid.
    
    Parameters:
    -----------
    point_data : list of dict
        List of dictionaries with 'lat', 'lon', and 'value' keys
    rows : int
        Number of rows in the output grid
    cols : int
        Number of columns in the output grid
    extent : tuple
        Geographic bounds in format (min_lon, max_lon, min_lat, max_lat)
    method : str, optional
        Interpolation method to use: 'nearest', 'linear', or 'cubic'
        
    Returns:
    --------
    numpy.ndarray
        Interpolated grid with shape (rows, cols)
    """
    try:
        # Check if we have enough points for the requested method
        if method == 'cubic' and len(point_data) < 4:
            method = 'linear'
        if method == 'linear' and len(point_data) < 3:
            method = 'nearest'
        if len(point_data) < 1:
            return np.zeros((rows, cols))
            
        # Extract points and values
        points = np.array([(p['lon'], p['lat']) for p in point_data])
        values = np.array([p['value'] for p in point_data])
        
        # Create regular grid
        lon_min, lon_max, lat_min, lat_max = extent
        x = np.linspace(lon_min, lon_max, cols)
        y = np.linspace(lat_min, lat_max, rows)
        xx, yy = np.meshgrid(x, y)
        
        # Interpolate
        grid = griddata(points, values, (xx, yy), method=method, fill_value=0)
        
        return grid
        
    except Exception as e:
        print(f"Interpolation error: {e}")
        print("Falling back to simple distance-weighted interpolation")
        
        # Simple distance-weighted interpolation as fallback
        grid = np.zeros((rows, cols))
        lon_min, lon_max, lat_min, lat_max = extent
        
        # Calculate grid coordinates
        x_step = (lon_max - lon_min) / (cols - 1)
        y_step = (lat_max - lat_min) / (rows - 1)
        
        for i in range(rows):
            for j in range(cols):
                # Calculate the lat/lon for this grid point
                lon = lon_min + j * x_step
                lat = lat_min + i * y_step
                
                # Calculate weighted value based on inverse distance
                total_weight = 0
                weighted_sum = 0
                
                for point in point_data:
                    # Calculate distance to this point
                    dist = np.sqrt((lon - point['lon'])**2 + 
                                  (lat - point['lat'])**2)
                    
                    # Avoid division by zero
                    if dist < 1e-10:
                        return point['value']
                    
                    # Weight is inverse of distance squared
                    weight = 1.0 / (dist**2)
                    total_weight += weight
                    weighted_sum += weight * point['value']
                
                if total_weight > 0:
                    grid[i, j] = weighted_sum / total_weight
                else:
                    grid[i, j] = 0
        
        return grid

def preprocess_ground_sites(df, dim, lat_max, lon_max, lat_dist, lon_dist, allow_negative=False):
    """
    Preprocess ground site data to place on grid with semantic missing data handling.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with lat, lon, and value columns
    dim : int
        Grid dimension
    lat_max, lon_max : float
        Maximum latitude and longitude values
    lat_dist, lon_dist : float
        Latitude and longitude distances
    allow_negative : bool, optional
        Whether negative values are valid for this variable (default: False)
        True for wind components and temperature, False for humidity, wind speed, etc.
    
    Returns:
    --------
    numpy.ndarray
        Grid with values placed at station locations
    """
    unInter = np.zeros((dim, dim))
    dfArr = np.array(df)
    
    for i in range(dfArr.shape[0]):
        try:
            # Calculate x (latitude index)
            x = int(((lat_max - dfArr[i, 0]) / lat_dist) * dim)
            if x >= dim:
                x = dim - 1
            if x < 0:
                x = 0
                
            # Calculate y (longitude index)
            y = dim - int(((lon_max + abs(dfArr[i, 1])) / lon_dist) * dim)
            if y >= dim:
                y = dim - 1
            if y < 0:
                y = 0
                
            # Handle missing values based on variable type
            value = dfArr[i, 2]
            
            if allow_negative:
                # For variables that can be negative (wind components, temperature)
                # Only filter NaN values (missing data marker for these variables)
                if np.isnan(value):
                    unInter[x, y] = 0
                else:
                    unInter[x, y] = value
            else:
                # For variables that should be positive (humidity, wind speed, etc.)
                # Filter negative values (including -1.0 missing data marker)
                if value < 0 or np.isnan(value):
                    unInter[x, y] = 0
                else:
                    unInter[x, y] = value
                    
        except Exception as e:
            print(f"Error placing point on grid: {e}")
    
    return unInter

def interpolate_frame(f, dim):
    """
    Interpolate frame using Inverse Distance Weighting (IDW).
    Handles both positive and negative values correctly.
    
    Parameters:
    -----------
    f : numpy.ndarray
        Frame with values at station locations
    dim : int
        Grid dimension
    
    Returns:
    --------
    numpy.ndarray
        Interpolated grid
    """
    # Extract non-zero points (including negative values)
    x_list = []
    y_list = []
    values = []
    
    for x in range(f.shape[0]):
        for y in range(f.shape[1]):
            if f[x, y] != 0:  # Only exclude actual zeros, not negative values
                x_list.append(x)
                y_list.append(y)
                values.append(f[x, y])
    
    coords = list(zip(x_list, y_list))
    
    # If no valid points, return zeros
    if not coords:
        return np.zeros((dim, dim))
    
    # If only one point, create a circular influence region
    if len(coords) == 1:
        interpolated = np.zeros((dim, dim))
        x0, y0 = coords[0]
        value = values[0]
        radius = dim * 0.2  # 20% of grid size as influence radius
        
        for x in range(dim):
            for y in range(dim):
                dist = np.sqrt((x - x0)**2 + (y - y0)**2)
                if dist < radius:
                    # Linear falloff within the radius
                    interpolated[x, y] = value * (1 - dist/radius)
        
        return interpolated
    
    # For multiple points, use IDW interpolation
    try:
        interpolated = np.zeros((dim, dim))
        power = 2  # Power parameter for IDW - higher makes peaks sharper
        
        # Create grid coordinates
        X, Y = np.meshgrid(np.arange(dim), np.arange(dim))
        
        # For each grid point
        for i in range(dim):
            for j in range(dim):
                # Calculate distances to all known points
                distances = np.sqrt([(i - x)**2 + (j - y)**2 for x, y in coords])
                
                # Check for exact point match (avoid division by zero)
                exact_match = np.where(distances < 1e-10)[0]
                if len(exact_match) > 0:
                    # Use the exact value
                    interpolated[i, j] = values[exact_match[0]]
                    continue
                
                weights = 1.0 / (distances**power)
                normalized_weights = weights / np.sum(weights)
                interpolated[i, j] = np.sum(normalized_weights * np.array(values))
        
        return interpolated
        
    except Exception as e:
        print(f"Error in IDW interpolation: {e}")
        return np.zeros((dim, dim))