import numpy as np
import pandas as pd
from typing import Union, Literal


def aggregate_temporal_data(
    data: np.ndarray,
    timestamps: pd.DatetimeIndex,
    target_frequency: Literal['hourly', 'daily'] = 'daily',
    aggregation_method: str = 'mean',
    preserve_wind_vectors: bool = True,
    wind_u_indices: list = None,
    wind_v_indices: list = None,
    verbose: bool = False
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    if target_frequency == 'hourly':
        return data, timestamps
    
    if target_frequency != 'daily':
        raise ValueError(f"Unsupported frequency: {target_frequency}")
    
    if verbose:
        print(f"Aggregating from {len(timestamps)} hourly to daily using {aggregation_method}")
    
    dates = pd.Series(timestamps.date).unique()
    n_days = len(dates)
    
    if verbose:
        print(f"Found {n_days} unique dates")
    
    aggregated_shape = (n_days, data.shape[1], data.shape[2], data.shape[3])
    aggregated_data = np.zeros(aggregated_shape)
    
    daily_timestamps = pd.DatetimeIndex([
        pd.Timestamp(date) + pd.Timedelta(hours=12) for date in dates
    ])
    
    mixed_days_count = 0
    
    for day_idx, date in enumerate(dates):
        day_mask = pd.Series(timestamps.date) == date
        day_indices = np.where(day_mask)[0]
        
        if len(day_indices) == 0:
            continue
            
        day_data = data[day_indices]
        
        if preserve_wind_vectors and wind_u_indices and wind_v_indices:
            aggregated_data[day_idx], is_mixed = _aggregate_with_wind_vectors_smart(
                day_data, aggregation_method, wind_u_indices, wind_v_indices
            )
        else:
            aggregated_data[day_idx], is_mixed = _smart_daily_aggregation(
                day_data, aggregation_method
            )
        
        if is_mixed:
            mixed_days_count += 1
    
    if verbose:
        print(f"Aggregated data shape: {aggregated_data.shape}")
        print(f"Timestamps reduced from {len(timestamps)} to {len(daily_timestamps)}")
        if mixed_days_count > 0:
            print(f"Found {mixed_days_count} days with mixed values (data changed during day)")
    
    return aggregated_data, daily_timestamps


def _smart_daily_aggregation(day_data: np.ndarray, aggregation_method: str) -> tuple[np.ndarray, bool]:
    """
    Smart daily aggregation that detects value changes within a day.
    
    Returns:
    --------
    tuple
        (aggregated_day_data, is_mixed_day)
    """
    aggregated = np.zeros(day_data.shape[1:])
    is_mixed = False
    
    for ch_idx in range(day_data.shape[-1]):
        channel_data = day_data[:, :, :, ch_idx]
        
        # Check if all values in this channel are identical (repeated data)
        if _is_channel_uniform(channel_data):
            # All values are the same - just use that value
            aggregated[:, :, ch_idx] = channel_data[0]
        else:
            # Values changed during the day - this is a mixed day
            is_mixed = True
            
            # Handle mixed day based on aggregation method
            if aggregation_method == 'mean':
                # For mixed days, use median to be more robust
                aggregated[:, :, ch_idx] = np.median(channel_data, axis=0)
            elif aggregation_method == 'max':
                aggregated[:, :, ch_idx] = np.max(channel_data, axis=0)
            elif aggregation_method == 'min':
                aggregated[:, :, ch_idx] = np.min(channel_data, axis=0)
            elif aggregation_method == 'median':
                aggregated[:, :, ch_idx] = np.median(channel_data, axis=0)
            else:
                # Default to most recent value for unknown methods
                aggregated[:, :, ch_idx] = channel_data[-1]
    
    return aggregated, is_mixed


def _is_channel_uniform(channel_data: np.ndarray, tolerance: float = 1e-6) -> bool:
    """
    Check if all values in a channel are identical (within tolerance).
    
    Parameters:
    -----------
    channel_data : np.ndarray
        Shape (hours, dim, dim)
    tolerance : float
        Tolerance for considering values identical
        
    Returns:
    --------
    bool
        True if all values are uniform (repeated data)
    """
    first_frame = channel_data[0]
    
    for hour_idx in range(1, channel_data.shape[0]):
        if not np.allclose(channel_data[hour_idx], first_frame, atol=tolerance, rtol=tolerance):
            return False
    
    return True


def _aggregate_with_wind_vectors_smart(
    day_data: np.ndarray,
    aggregation_method: str,
    wind_u_indices: list,
    wind_v_indices: list
) -> tuple[np.ndarray, bool]:
    """
    Smart wind vector aggregation that detects changes within a day.
    """
    aggregated = np.zeros(day_data.shape[1:])
    is_mixed = False
    
    all_wind_indices = set(wind_u_indices + wind_v_indices)
    non_wind_indices = [i for i in range(day_data.shape[-1]) if i not in all_wind_indices]
    
    for ch_idx in non_wind_indices:
        channel_data = day_data[:, :, :, ch_idx]
        
        if _is_channel_uniform(channel_data):
            aggregated[:, :, ch_idx] = channel_data[0]
        else:
            is_mixed = True
            if aggregation_method == 'mean':
                aggregated[:, :, ch_idx] = np.median(channel_data, axis=0)
            elif aggregation_method == 'max':
                aggregated[:, :, ch_idx] = np.max(channel_data, axis=0)
            elif aggregation_method == 'min':
                aggregated[:, :, ch_idx] = np.min(channel_data, axis=0)
            elif aggregation_method == 'median':
                aggregated[:, :, ch_idx] = np.median(channel_data, axis=0)
            else:
                aggregated[:, :, ch_idx] = channel_data[-1]
    
    for u_idx, v_idx in zip(wind_u_indices, wind_v_indices):
        u_data = day_data[:, :, :, u_idx]
        v_data = day_data[:, :, :, v_idx]
        
        u_uniform = _is_channel_uniform(u_data)
        v_uniform = _is_channel_uniform(v_data)
        
        if u_uniform and v_uniform:
            aggregated[:, :, u_idx] = u_data[0]
            aggregated[:, :, v_idx] = v_data[0]
        else:
            is_mixed = True
            
            magnitude = np.sqrt(u_data**2 + v_data**2)
            direction = np.arctan2(v_data, u_data)
            
            if aggregation_method == 'mean':
                avg_magnitude = np.median(magnitude, axis=0)
            elif aggregation_method == 'max':
                avg_magnitude = np.max(magnitude, axis=0)
            elif aggregation_method == 'min':
                avg_magnitude = np.min(magnitude, axis=0)
            elif aggregation_method == 'median':
                avg_magnitude = np.median(magnitude, axis=0)
            else:
                # Default to most recent for unknown methods
                avg_magnitude = magnitude[-1]
            
            avg_direction = _circular_mean(direction, axis=0)
            
            aggregated[:, :, u_idx] = avg_magnitude * np.cos(avg_direction)
            aggregated[:, :, v_idx] = avg_magnitude * np.sin(avg_direction)
    
    return aggregated, is_mixed


def _circular_mean(angles: np.ndarray, axis: int = 0) -> np.ndarray:
    x = np.cos(angles)
    y = np.sin(angles)
    
    mean_x = np.mean(x, axis=axis)
    mean_y = np.mean(y, axis=axis)
    
    mean_angle = np.arctan2(mean_y, mean_x)
    
    return mean_angle


def get_aggregation_method_for_variable(variable_name: str) -> str:
    max_aggregation_vars = [
        'precipitation', 'rain', 'precip', 'gust', 'fire', 'frp'
    ]
    
    variable_lower = variable_name.lower()
    
    for var in max_aggregation_vars:
        if var in variable_lower:
            return 'max'
    
    return 'mean'


def create_aggregation_config(channel_names: list, preserve_wind_vectors: bool = True) -> dict:
    config = {
        'wind_u_indices': [],
        'wind_v_indices': [],
        'channel_methods': {}
    }
    
    for i, channel_name in enumerate(channel_names):
        if preserve_wind_vectors:
            if 'wind_u' in channel_name.lower() or 'u_component' in channel_name.lower():
                config['wind_u_indices'].append(i)
            elif 'wind_v' in channel_name.lower() or 'v_component' in channel_name.lower():
                config['wind_v_indices'].append(i)
        
        config['channel_methods'][channel_name] = get_aggregation_method_for_variable(channel_name)
    
    return config