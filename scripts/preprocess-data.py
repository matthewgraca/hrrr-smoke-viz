import os
import numpy as np
import gc
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from numpy.lib.format import open_memmap

SENSORS = {
    'Reseda': (8, 3),
    'North Holywood': (8, 11),
    'LA - N. Main Street': (15, 16),
    'Compton': (23, 17),
    'Long Beach Signal Hill': (29, 19),
    'Anaheim': (27, 29),
    'Glendora - Laurel': (10, 33),
}

# def sliding_window(data, frames, sequence_stride=1, compute_targets=False,
#                    forecast_horizon=24):
#     gc.collect()

#     max_samples_per_batch = 50 if forecast_horizon == 24 else 200

#     if compute_targets:
#         min_length = frames + forecast_horizon
#         if len(data) < min_length:
#             raise ValueError(f"Need at least {min_length} timesteps, got {len(data)}")

#         n_total_samples = (len(data) - frames - forecast_horizon) // sequence_stride + 1
#         if n_total_samples <= 0:
#             raise ValueError("Not enough timesteps to create any target windows.")

#         X_chunks, Y_chunks = [], []

#         for batch_start in range(0, n_total_samples, max_samples_per_batch):
#             batch_end = min(batch_start + max_samples_per_batch, n_total_samples)
#             batch_indices = range(batch_start * sequence_stride,
#                                   batch_end * sequence_stride,
#                                   sequence_stride)

#             X_batch = np.array([data[i:i+frames] for i in batch_indices])
#             Y_batch = np.array([data[i+frames:i+frames+forecast_horizon] for i in batch_indices])

#             X_chunks.append(X_batch)
#             Y_chunks.append(Y_batch)
#             gc.collect()

#         X = np.concatenate(X_chunks, axis=0)
#         Y = np.concatenate(Y_chunks, axis=0)
#         del X_chunks, Y_chunks
#         gc.collect()
#         return X, Y

#     else:
#         min_length = frames + forecast_horizon
#         if len(data) < min_length:
#             raise ValueError(f"Need at least {min_length} timesteps, got {len(data)}")

#         data_trimmed = data[:-forecast_horizon]
#         n_total_samples = (len(data_trimmed) - frames) // sequence_stride + 1
#         if n_total_samples <= 0:
#             raise ValueError("Not enough timesteps to create any input windows.")

#         X_chunks = []
#         for batch_start in range(0, n_total_samples, max_samples_per_batch):
#             batch_end = min(batch_start + max_samples_per_batch, n_total_samples)
#             batch_indices = range(batch_start * sequence_stride,
#                                   batch_end * sequence_stride,
#                                   sequence_stride)

#             X_batch = np.array([data_trimmed[i:i+frames] for i in batch_indices])
#             X_chunks.append(X_batch)
#             gc.collect()

#         X = np.concatenate(X_chunks, axis=0)
#         del X_chunks
#         gc.collect()
#         return X, None
def sliding_window(data, frames, sequence_stride=1, compute_targets=False, forecast_horizon=24):
    min_length = frames + forecast_horizon
    if len(data) < min_length:
        raise ValueError(f"Need at least {min_length} timesteps, got {len(data)}")
    
    if compute_targets:
        n_samples = (len(data) - frames - forecast_horizon) // sequence_stride + 1
        indices = range(0, n_samples * sequence_stride, sequence_stride)
        
        X = np.array([data[i:i+frames] for i in indices])
        Y = np.array([data[i+frames:i+frames+forecast_horizon] for i in indices])
        return X, Y
    else:
        data_trimmed = data[:-forecast_horizon]
        n_samples = (len(data_trimmed) - frames) // sequence_stride + 1
        indices = range(0, n_samples * sequence_stride, sequence_stride)
        
        X = np.array([data_trimmed[i:i+frames] for i in indices])
        return X, None

# def sliding_window_forecast(data, frames, forecast_horizon, sequence_stride=1):
#     """
#     Create windows for forecast channels - these are time-shifted to align with targets.
#     For each sample, extracts the forecast_horizon timesteps that correspond to the target window.
#     """
#     gc.collect()

#     max_samples_per_batch = 50 if forecast_horizon == 24 else 200

#     min_length = frames + forecast_horizon
#     if len(data) < min_length:
#         raise ValueError(f"Need at least {min_length} timesteps, got {len(data)}")

#     n_total_samples = (len(data) - frames - forecast_horizon) // sequence_stride + 1
#     if n_total_samples <= 0:
#         raise ValueError("Not enough timesteps to create any forecast windows.")

#     X_chunks = []

#     for batch_start in range(0, n_total_samples, max_samples_per_batch):
#         batch_end = min(batch_start + max_samples_per_batch, n_total_samples)
#         batch_indices = range(batch_start * sequence_stride,
#                               batch_end * sequence_stride,
#                               sequence_stride)

#         X_batch = np.array([data[i+frames:i+frames+forecast_horizon] for i in batch_indices])
#         X_chunks.append(X_batch)
#         gc.collect()

#     X = np.concatenate(X_chunks, axis=0)
#     del X_chunks
#     gc.collect()
#     return X
def sliding_window_forecast(data, frames, forecast_horizon, sequence_stride=1):
    min_length = frames + forecast_horizon
    if len(data) < min_length:
        raise ValueError(f"Need at least {min_length} timesteps, got {len(data)}")
    
    n_samples = (len(data) - frames - forecast_horizon) // sequence_stride + 1
    indices = range(0, n_samples * sequence_stride, sequence_stride)
    
    X = np.array([data[i+frames:i+frames+forecast_horizon] for i in indices])
    return X

def load_airnow_data(cache_dir):
    """Load AirNow PM2.5 data."""
    airnow_path = f"{cache_dir}/airnow_processed.npz"
    if os.path.exists(airnow_path):
        print(f"    Loading AirNow PM2.5 data from {airnow_path}...")
        loaded = np.load(airnow_path, allow_pickle=True)
        if 'data' in loaded:
            return loaded['data']
        elif 'arr_0' in loaded:
            return loaded['arr_0']
        else:
            return loaded[list(loaded.keys())[0]]
    else:
        raise FileNotFoundError(f"REQUIRED: AirNow data not found at {airnow_path}")


def load_openaq_data(cache_dir):
    """Load OpenAQ PM2.5 data."""
    openaq_path = f"{cache_dir}/openaq_processed.npz"
    if os.path.exists(openaq_path):
        print(f"    Loading OpenAQ PM2.5 data from {openaq_path}...")
        loaded = np.load(openaq_path, allow_pickle=True)
        if 'data' in loaded:
            return loaded['data']
        elif 'arr_0' in loaded:
            return loaded['arr_0']
        else:
            return loaded[list(loaded.keys())[0]]
    else:
        raise FileNotFoundError(f"REQUIRED: OpenAQ data not found at {openaq_path}")


def load_naqfc_data(cache_dir):
    """Load NAQFC PM2.5 data (used for forecast channel)."""
    naqfc_path = f"{cache_dir}/naqfc_pm25_processed.npz"
    if os.path.exists(naqfc_path):
        print(f"    Loading NAQFC PM2.5 data from {naqfc_path}...")
        loaded = np.load(naqfc_path, allow_pickle=True)
        if 'data' in loaded:
            return loaded['data']
        elif 'arr_0' in loaded:
            return loaded['arr_0']
        else:
            return loaded[list(loaded.keys())[0]]
    else:
        raise FileNotFoundError(f"REQUIRED: NAQFC data not found at {naqfc_path}")


def load_satellite_data(cache_dir, sat_type='goes'):
    if sat_type == 'goes':
        goes_path = f"{cache_dir}/goes_processed.npz"
        if os.path.exists(goes_path):
            print(f"    Loading GOES satellite data from {goes_path}...")
            loaded = np.load(goes_path, allow_pickle=True)
            if 'data' in loaded:
                return loaded['data']
            elif 'arr_0' in loaded:
                return loaded['arr_0']
            else:
                return loaded[list(loaded.keys())[0]]
        else:
            raise FileNotFoundError(f"REQUIRED: GOES data not found at {goes_path}")

    elif sat_type == 'tempo':
        tempo_path = f"{cache_dir}/tempo_l3_no2_20230802_20250802_hourly.npz"
        if os.path.exists(tempo_path):
            print(f"    Loading TEMPO satellite data from {tempo_path}...")
            loaded = np.load(tempo_path, allow_pickle=True)
            if 'data' in loaded:
                return loaded['data']
            elif 'arr_0' in loaded:
                return loaded['arr_0']
            else:
                return loaded[list(loaded.keys())[0]]
        else:
            raise FileNotFoundError(f"REQUIRED: TEMPO data not found at {tempo_path}")

    return None


def generate_temporal_features(n_timesteps, height=40, width=40, 
                                start_date="2023-08-02", end_date="2025-08-02"):
    """
    Generate temporal features (Month sin/cos, Hour sin/cos) for the given time range.
    """
    print(f"    Generating temporal features from {start_date} to {end_date}...")
    
    dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq='h',
        inclusive='left'
    )
    
    dates = dates[:n_timesteps]
    
    temporal_channels = []
    
    months = dates.month.values.astype(float)
    month_rad = 2 * np.pi * months / 12
    month_sin = np.sin(month_rad)
    month_cos = np.cos(month_rad)
    
    hours = dates.hour.values.astype(float)
    hour_rad = 2 * np.pi * hours / 23
    hour_sin = np.sin(hour_rad)
    hour_cos = np.cos(hour_rad)
    
    for arr in [month_sin, month_cos, hour_sin, hour_cos]:
        expanded = np.repeat(arr[:, np.newaxis, np.newaxis], height, axis=1)
        expanded = np.repeat(expanded, width, axis=2)
        expanded = np.expand_dims(expanded, axis=-1)
        temporal_channels.append(expanded)
    
    temporal_data = np.concatenate(temporal_channels, axis=-1)
    
    print(f"    Temporal features shape: {temporal_data.shape}")
    
    return temporal_data


def load_hrrr_data(cache_dir):
    """Load HRRR surface data (wind, temp, PBL, precip)."""
    hrrr_path = f"{cache_dir}/hrrr_surface_2years_40x40.npz"
    if os.path.exists(hrrr_path):
        print(f"    Loading HRRR surface data from {hrrr_path}...")
        hrrr_data = np.load(hrrr_path, allow_pickle=True)
        
        result = {
            'u_wind': hrrr_data['u_wind'],
            'v_wind': hrrr_data['v_wind'],
            'wind_speed': hrrr_data['wind_speed'],
            'temp_2m': hrrr_data['temp_2m'],
            'pbl_height': hrrr_data['pbl_height'],
            'precip_rate': hrrr_data['precip_rate']
        }
        
        print(f"    HRRR shape: {result['u_wind'].shape}")
        print(f"    Variables: u_wind, v_wind, wind_speed, temp_2m, pbl_height, precip_rate")
        
        return result
    else:
        print(f"    Warning: HRRR data not found at {hrrr_path}")
        return None


def load_static_data(data_key, n_timesteps, height, width, cache_dir):
    """Load NDVI or Elevation data and expand to all timesteps."""
    if data_key == 'elevation':
        elevation_path = f"{cache_dir}/elevation.npy"
        if os.path.exists(elevation_path):
            print(f"    Loading elevation from {elevation_path}...")
            elevation = np.load(elevation_path)
            if elevation.shape != (height, width):
                import cv2
                elevation = cv2.resize(elevation, (width, height))
            elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min() + 1e-8)
            return np.tile(elevation[np.newaxis, :, :], (n_timesteps, 1, 1))
        
        print("    Warning: Elevation file not found, using zeros")
        return np.zeros((n_timesteps, height, width))
    
    elif data_key == 'ndvi':
        ndvi_path = f"{cache_dir}/ndvi_processed.npy"
        if os.path.exists(ndvi_path):
            print(f"    Loading NDVI from {ndvi_path}...")
            ndvi = np.load(ndvi_path)
            if ndvi.ndim == 3:
                ndvi = ndvi[0]
            if ndvi.shape != (height, width):
                import cv2
                ndvi = cv2.resize(ndvi, (width, height))
            
            ndvi_min, ndvi_max = ndvi.min(), ndvi.max()
            print(f"    NDVI raw range: [{ndvi_min:.3f}, {ndvi_max:.3f}]")
            
            if ndvi_min >= -1 and ndvi_max <= 1:
                ndvi = (ndvi + 1) / 2
            else:
                ndvi = (ndvi - ndvi_min) / (ndvi_max - ndvi_min + 1e-8)
            
            print(f"    NDVI normalized range: [{ndvi.min():.3f}, {ndvi.max():.3f}]")
            return np.tile(ndvi[np.newaxis, :, :], (n_timesteps, 1, 1))
        
        print("    Warning: NDVI file not found, using zeros")
        return np.zeros((n_timesteps, height, width))
    
    return None


def generate_hourly_climatology(pm25_data, window_days=30):
    """
    Generate rolling 30-day hourly climatology for PM2.5.
    
    For each timestep t at hour h, compute the mean PM2.5 at hour h
    over the previous 30 days. This provides a "typical value for this 
    hour of day based on recent history" channel.
    
    Args:
        pm25_data: shape (n_timesteps, height, width)
        window_days: number of days to look back (default 30)
    
    Returns:
        climatology: shape (n_timesteps, height, width)
    """
    print(f"    Generating {window_days}-day rolling hourly climatology...")
    
    n_timesteps, height, width = pm25_data.shape
    hours_per_day = 24
    window_hours = window_days * hours_per_day
    
    climatology = np.zeros((n_timesteps, height, width), dtype=np.float32)
    
    for t in range(n_timesteps):
        hour_of_day = t % hours_per_day
        
        same_hour_indices = []
        for past_t in range(t - hours_per_day, max(-1, t - window_hours - 1), -hours_per_day):
            if past_t >= 0:
                same_hour_indices.append(past_t)
        
        if len(same_hour_indices) > 0:
            climatology[t] = np.mean(pm25_data[same_hour_indices], axis=0)
        else:
            climatology[t] = pm25_data[t]
        
        if t % 5000 == 0 and t > 0:
            print(f"      Processed {t}/{n_timesteps} timesteps...")
    
    print(f"    Climatology shape: {climatology.shape}")
    print(f"    Climatology range: [{climatology.min():.2f}, {climatology.max():.2f}]")
    
    return climatology


def preprocess_dataset_split(
    frames_per_sample=24, 
    forecast_horizon=24,
    target_source='airnow',
    train_pct=0.75,
    valid_pct=0.13
):
    """
    Preprocess data with a defined temporal split.
    """
    train_split = int(train_pct*100)
    valid_split = int(valid_pct*100)
    test_split = 100 - train_split - valid_split
    print("\n" + "="*80)
    print(
        f"{train_split}/{valid_split}/{test_split} "
        "SPLIT PREPROCESSING: {frames_per_sample}h → {forecast_horizon}h"
    )
    print(f"TARGET SOURCE: {target_source.upper()}")
    print("Split: {train_split}% Train / {valid_split}% Valid / {test_split}% Test")
    print("Using: AirNow, OpenAQ, HRRR (6 vars), Satellites")
    print("       + Forecast channels: NAQFC, HRRR (time-shifted to target window)")
    print("       + Hourly Climatology (30-day rolling average by hour)")
    print("="*80)

    channels = [
        ('airnow_pm25', 'AirNow_PM25', True, False, False),
        ('airnow_hourly_clim', 'AirNow_Hourly_Clim', True, False, False),
        ('openaq_pm25', 'OpenAQ_PM25', True, False, False),
        ('naqfc_pm25', 'NAQFC_PM25', True, False, False),
        ('hrrr_wind_u', 'HRRR_Wind_U', True, False, False),
        ('hrrr_wind_v', 'HRRR_Wind_V', True, False, False),
        ('hrrr_wind_speed', 'HRRR_Wind_Speed', True, False, False),
        ('hrrr_temp_2m', 'HRRR_Temp_2m', True, False, False),
        ('hrrr_pbl_height', 'HRRR_PBL_Height', True, False, False),
        ('hrrr_precip_rate', 'HRRR_Precip_Rate', True, False, False),
        ('elevation', 'Elevation', False, False, False),
        ('ndvi', 'NDVI', False, False, False),
        ('temporal_0', 'Temporal_Month_Sin', False, True, False),
        ('temporal_1', 'Temporal_Month_Cos', False, True, False),
        ('temporal_4', 'Temporal_Hour_Sin', False, True, False),
        ('temporal_5', 'Temporal_Hour_Cos', False, True, False),
        ('goes', 'GOES', True, False, False),
        ('tempo', 'TEMPO', True, False, False),
        # forecast channels
        ('naqfc_pm25', 'NAQFC_PM25_Forecast', True, False, True),
        ('hrrr_wind_u', 'HRRR_Wind_U_Forecast', True, False, True),
        ('hrrr_wind_v', 'HRRR_Wind_V_Forecast', True, False, True),
        ('hrrr_wind_speed', 'HRRR_Wind_Speed_Forecast', True, False, True),
        ('hrrr_temp_2m', 'HRRR_Temp_2m_Forecast', True, False, True),
        ('hrrr_pbl_height', 'HRRR_PBL_Height_Forecast', True, False, True),
        ('hrrr_precip_rate', 'HRRR_Precip_Rate_Forecast', True, False, True),
        ('airnow_hourly_clim', 'AirNow_Hourly_Clim_Forecast', True, False, True),
        ('temporal_0', 'Temporal_Month_Sin_Forecast', False, True, True),
        ('temporal_1', 'Temporal_Month_Cos_Forecast', False, True, True),
        ('temporal_4', 'Temporal_Hour_Sin_Forecast', False, True, True),
        ('temporal_5', 'Temporal_Hour_Cos_Forecast', False, True, True),
    ]

    n_channels = len(channels)
    channel_names_list = [ch[1] for ch in channels]
    
    # NOTE: change for your use
    ####
    base_path = '/home/mgraca/Workspace/hrrr-smoke-viz'
    experiment_path = os.path.join(base_path, 'pwwb-experiments/tensorflow/autoencoder_archive')
    cache_dir = os.path.join(experiment_path, "raw_data")
    output_cache_dir = os.path.join(experiment_path, "preprocessed_cache")
    ####
    os.makedirs(output_cache_dir, exist_ok=True)

    npy_dir = f"{output_cache_dir}/npy_files"
    scalers_file = f"{output_cache_dir}/scalers.pkl"
    metadata_file = f"{output_cache_dir}/metadata.pkl"

    if (os.path.exists(f"{npy_dir}/X_train.npy")
        and os.path.exists(f"{npy_dir}/X_valid.npy")
        and os.path.exists(f"{npy_dir}/X_test.npy")
        and os.path.exists(f"{npy_dir}/Y_train.npy")
        and os.path.exists(f"{npy_dir}/Y_valid.npy")
        and os.path.exists(f"{npy_dir}/Y_test.npy")):
        print(f"\n{target_source.upper()} target cache already exists at: {npy_dir}")
        print("Delete this folder if you want to reprocess.")
        return npy_dir, scalers_file, metadata_file

    print("\nLoading AirNow PM2.5 data...")
    X_airnow_pm25 = load_airnow_data(cache_dir)
    n_timesteps = X_airnow_pm25.shape[0]
    height, width = X_airnow_pm25.shape[1], X_airnow_pm25.shape[2]
    
    print(f"  AirNow shape: {X_airnow_pm25.shape}")
    print(f"  AirNow range: [{X_airnow_pm25.min():.2f}, {X_airnow_pm25.max():.2f}]")

    print("\nLoading OpenAQ PM2.5 data...")
    X_openaq_pm25 = load_openaq_data(cache_dir)
    n_timesteps = min(n_timesteps, X_openaq_pm25.shape[0])
    print(f"  OpenAQ shape: {X_openaq_pm25.shape}")
    print(f"  OpenAQ range: [{X_openaq_pm25.min():.2f}, {X_openaq_pm25.max():.2f}]")

    if target_source == 'airnow':
        X_target = X_airnow_pm25
        target_name = 'AirNow_PM25'
    elif target_source == 'openaq':
        X_target = X_openaq_pm25
        target_name = 'OpenAQ_PM25'
    else:
        raise ValueError(f"Unknown target_source: {target_source}. Use 'airnow' or 'openaq'")

    print(f"\n*** Using {target_name} as prediction target ***")
    print(f"  Shape: {X_target.shape}")
    print(f"  Range: [{X_target.min():.2f}, {X_target.max():.2f}]")

    print("\nLoading NAQFC PM2.5 data (for forecast channel)...")
    X_naqfc_pm25 = load_naqfc_data(cache_dir)
    n_timesteps = min(n_timesteps, X_naqfc_pm25.shape[0])
    print(f"  NAQFC shape: {X_naqfc_pm25.shape}")

    print("\nLoading HRRR surface data...")
    X_hrrr = load_hrrr_data(cache_dir)
    if X_hrrr is not None:
        n_timesteps = min(n_timesteps, X_hrrr['u_wind'].shape[0])
        print(f"  Aligning to HRRR timesteps: {X_hrrr['u_wind'].shape[0]}")

    X_goes, X_tempo = None, None
    need_goes = any(ch[0] == 'goes' for ch in channels)
    need_tempo = any(ch[0] == 'tempo' for ch in channels)
    
    if need_goes:
        print("\nLoading GOES satellite data...")
        X_goes = load_satellite_data(cache_dir, 'goes')
        n_timesteps = min(n_timesteps, X_goes.shape[0])
    
    if need_tempo:
        print("\nLoading TEMPO satellite data...")
        X_tempo = load_satellite_data(cache_dir, 'tempo')
        n_timesteps = min(n_timesteps, X_tempo.shape[0])

    print(f"\nAligning all data to {n_timesteps} timesteps")
    X_airnow_pm25 = X_airnow_pm25[:n_timesteps]
    X_openaq_pm25 = X_openaq_pm25[:n_timesteps]
    X_target = X_target[:n_timesteps]
    X_naqfc_pm25 = X_naqfc_pm25[:n_timesteps]
    if X_hrrr is not None:
        for key in X_hrrr:
            X_hrrr[key] = X_hrrr[key][:n_timesteps]
    if X_goes is not None:
        X_goes = X_goes[:n_timesteps]
    if X_tempo is not None:
        X_tempo = X_tempo[:n_timesteps]

    train_end_idx = int(n_timesteps * train_pct)
    valid_end_idx = int(n_timesteps * (train_pct + valid_pct))
    
    print(f"\nTimesteps: {n_timesteps}")
    print(f"Train: [0:{train_end_idx}] ({train_end_idx} hours, {train_end_idx/24:.1f} days)")
    print(f"Valid: [{train_end_idx}:{valid_end_idx}] ({valid_end_idx - train_end_idx} hours)")
    print(f"Test:  [{valid_end_idx}:{n_timesteps}] ({n_timesteps - valid_end_idx} hours)")

    def nonneg(n): return max(0, n)
    n_train_windows = nonneg((train_end_idx - frames_per_sample - forecast_horizon) + 1)
    n_valid_windows = nonneg(((valid_end_idx - train_end_idx) - frames_per_sample - forecast_horizon) + 1)
    n_test_windows = nonneg(((n_timesteps - valid_end_idx) - frames_per_sample - forecast_horizon) + 1)

    if min(n_train_windows, n_valid_windows, n_test_windows) <= 0:
        raise ValueError("Not enough timesteps to produce windows")

    print(f"\nExpected windows:")
    print(f"  Train: {n_train_windows}")
    print(f"  Valid: {n_valid_windows}")
    print(f"  Test:  {n_test_windows}")

    os.makedirs(npy_dir, exist_ok=True)

    X_train_mmap = open_memmap(f"{npy_dir}/X_train.npy", mode='w+',
                               dtype='float32',
                               shape=(n_train_windows, frames_per_sample, height, width, n_channels))
    X_valid_mmap = open_memmap(f"{npy_dir}/X_valid.npy", mode='w+',
                               dtype='float32',
                               shape=(n_valid_windows, frames_per_sample, height, width, n_channels))
    X_test_mmap = open_memmap(f"{npy_dir}/X_test.npy", mode='w+',
                              dtype='float32',
                              shape=(n_test_windows, frames_per_sample, height, width, n_channels))

    scalers = {}

    observed_channels = [ch[1] for ch in channels if not ch[4]]
    forecast_channels = [ch[1] for ch in channels if ch[4]]
    
    print(f"\nTotal channels: {n_channels}")
    print(f"Observed channels ({len(observed_channels)}): {', '.join(observed_channels)}")
    print(f"Forecast channels ({len(forecast_channels)}):  {', '.join(forecast_channels)}")

    def load_channel(data_key):
            if data_key == 'airnow_pm25':
                return X_airnow_pm25
            
            if data_key == 'openaq_pm25':
                return X_openaq_pm25
            
            if data_key == 'naqfc_pm25':
                return X_naqfc_pm25
            
            if data_key == 'hrrr_wind_u':
                return X_hrrr['u_wind'] if X_hrrr else None
            
            if data_key == 'hrrr_wind_v':
                return X_hrrr['v_wind'] if X_hrrr else None
            
            if data_key == 'hrrr_wind_speed':
                return X_hrrr['wind_speed'] if X_hrrr else None
            
            if data_key == 'hrrr_temp_2m':
                return X_hrrr['temp_2m'] if X_hrrr else None
            
            if data_key == 'hrrr_pbl_height':
                return X_hrrr['pbl_height'] if X_hrrr else None
            
            if data_key == 'hrrr_precip_rate':
                return X_hrrr['precip_rate'] if X_hrrr else None
            
            if data_key in ('elevation', 'ndvi'):
                return load_static_data(data_key, n_timesteps, height, width, cache_dir)

            if data_key == 'goes':
                return X_goes

            if data_key == 'tempo':
                return X_tempo

            if data_key.startswith('temporal_'):
                if not hasattr(load_channel, '_temporal_cache'):
                    load_channel._temporal_cache = generate_temporal_features(
                        n_timesteps, height, width,
                        start_date="2023-08-02", end_date="2025-08-02"
                    )
                
                X_temporal = load_channel._temporal_cache
                if data_key == 'temporal_0':
                    idx = 0
                elif data_key == 'temporal_1':
                    idx = 1
                elif data_key == 'temporal_4':
                    idx = 2
                elif data_key == 'temporal_5':
                    idx = 3
                else:
                    raise ValueError(f"Unknown temporal key: {data_key}")
                return X_temporal[:, :, :, idx]

            if data_key == 'airnow_hourly_clim':
                if not hasattr(load_channel, '_airnow_clim_cache'):
                    load_channel._airnow_clim_cache = generate_hourly_climatology(
                        X_airnow_pm25, window_days=30
                    )
                return load_channel._airnow_clim_cache

            raise ValueError(f"Unknown data_key: {data_key}")

    for ch_idx, (data_key, display_name, should_scale, is_temporal, is_forecast) in enumerate(channels):
        print(f"\nProcessing channel {ch_idx+1}/{n_channels}: {display_name}" + 
              (" [FORECAST]" if is_forecast else ""))

        data = load_channel(data_key)
        
        if data is None:
            print(f"  Skipping {display_name} - data not available")
            continue
            
        gc.collect()

        train_data = data[:train_end_idx]
        valid_data = data[train_end_idx:valid_end_idx]
        test_data = data[valid_end_idx:]

        if should_scale:
            scaler_key = display_name.replace('_Forecast', '')
            
            if scaler_key not in scalers:
                print(f"  Scaling {display_name} using training statistics...")
                scaler = StandardScaler()
                train_flat = train_data.reshape(-1, 1)
                scaler.fit(train_flat)
                scalers[scaler_key] = scaler
                print(f"    Train Mean: {scaler.mean_[0]:.4f}, Std: {scaler.scale_[0]:.4f}")
            else:
                print(f"  Scaling {display_name} using existing {scaler_key} scaler...")
                scaler = scalers[scaler_key]

            train_data = scaler.transform(train_data.reshape(-1, 1)).reshape(train_data.shape)
            valid_data = scaler.transform(valid_data.reshape(-1, 1)).reshape(valid_data.shape)
            test_data = scaler.transform(test_data.reshape(-1, 1)).reshape(test_data.shape)
        elif is_temporal:
            print(f"  {display_name} already normalized (sin/cos)")
        else:
            if data_key in ('elevation', 'ndvi'):
                print(f"  Using pre-normalized {display_name} (range 0-1)")
            else:
                print("  Using pre-scaled/normalized (or static)")

        def ensure_4d(a):
            return a if a.ndim == 4 else np.expand_dims(a, -1)

        if is_forecast:
            print(f"  Creating train forecast windows (t+1 to t+{forecast_horizon})...")
            train_w = sliding_window_forecast(ensure_4d(train_data), frames_per_sample, 
                                               forecast_horizon, 1)
            X_train_mmap[..., ch_idx] = train_w[..., 0]
            del train_w
            gc.collect()

            print(f"  Creating valid forecast windows...")
            valid_w = sliding_window_forecast(ensure_4d(valid_data), frames_per_sample,
                                               forecast_horizon, 1)
            X_valid_mmap[..., ch_idx] = valid_w[..., 0]
            del valid_w
            gc.collect()

            print(f"  Creating test forecast windows...")
            test_w = sliding_window_forecast(ensure_4d(test_data), frames_per_sample,
                                              forecast_horizon, 1)
            X_test_mmap[..., ch_idx] = test_w[..., 0]
            del test_w
            gc.collect()
        else:
            print("  Creating train windows...")
            train_w, _ = sliding_window(ensure_4d(train_data), frames_per_sample, 1, False, forecast_horizon)
            X_train_mmap[..., ch_idx] = train_w[..., 0]
            del train_w
            gc.collect()

            print("  Creating valid windows...")
            valid_w, _ = sliding_window(ensure_4d(valid_data), frames_per_sample, 1, False, forecast_horizon)
            X_valid_mmap[..., ch_idx] = valid_w[..., 0]
            del valid_w
            gc.collect()

            print("  Creating test windows...")
            test_w, _ = sliding_window(ensure_4d(test_data), frames_per_sample, 1, False, forecast_horizon)
            X_test_mmap[..., ch_idx] = test_w[..., 0]
            del test_w
            gc.collect()

        del train_data, valid_data, test_data
        gc.collect()

    if X_hrrr is not None:
        del X_hrrr
    if X_goes is not None:
        del X_goes
    if X_tempo is not None:
        del X_tempo
    del X_airnow_pm25, X_openaq_pm25, X_naqfc_pm25
    gc.collect()

    print(f"\nCreating targets from unscaled {target_name}...")

    _, Y_train = sliding_window(
        ensure_4d(X_target[:train_end_idx]),
        frames_per_sample, 1, True, forecast_horizon
    )
    _, Y_valid = sliding_window(
        ensure_4d(X_target[train_end_idx:valid_end_idx]),
        frames_per_sample, 1, True, forecast_horizon
    )
    _, Y_test = sliding_window(
        ensure_4d(X_target[valid_end_idx:]),
        frames_per_sample, 1, True, forecast_horizon
    )

    del X_target
    gc.collect()

    print("\nUsing verified sensor locations...")
    print(f"  Total sensors: {len(SENSORS)}")

    print("\nSaving to NPY cache...")

    X_train_mmap.flush()
    X_valid_mmap.flush()
    X_test_mmap.flush()
    del X_train_mmap, X_valid_mmap, X_test_mmap
    gc.collect()

    np.save(f"{npy_dir}/Y_train.npy", Y_train, allow_pickle=False)
    np.save(f"{npy_dir}/Y_valid.npy", Y_valid, allow_pickle=False)
    np.save(f"{npy_dir}/Y_test.npy", Y_test, allow_pickle=False)

    with open(scalers_file, 'wb') as f:
        pickle.dump(scalers, f)

    metadata_to_save = {
        'sensors': SENSORS,
        'channel_names': channel_names_list,
        'n_channels': n_channels,
        'observed_channels': [ch[1] for ch in channels if not ch[4]],
        'forecast_channels': [ch[1] for ch in channels if ch[4]],
        'target_source': target_source,
        'target_name': target_name,
        'train_timestep_start': 0,
        'train_timestep_end': train_end_idx,
        'valid_timestep_start': train_end_idx,
        'valid_timestep_end': valid_end_idx,
        'test_timestep_start': valid_end_idx,
        'test_timestep_end': n_timesteps,
        'frames_per_sample': frames_per_sample,
        'forecast_horizon': forecast_horizon,
        'split_type': f'{train_split}_{valid_split}_{test_split}_temporal',
        'data_source': target_name,
        'start_date': "2023-08-02-00",
        'end_date': "2025-08-02-00",
    }
    
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata_to_save, f)

    print(
        f"\n✓ SUCCESS! ({train_split}/{valid_split}/{test_split} split with "
        "{target_name} target + Forecast channels + Hourly Climatology)"
    )
    print(f"  X_train, X_valid, X_test saved under: {npy_dir}")
    print(f"  Y_train, Y_valid, Y_test saved under: {npy_dir}")
    print(f"  Scalers: {scalers_file}")
    print(f"  Metadata: {metadata_file}")

    print("\nFinal shapes:")
    print(f"  X_train: {np.load(f'{npy_dir}/X_train.npy', mmap_mode='r').shape}")
    print(f"  X_valid: {np.load(f'{npy_dir}/X_valid.npy', mmap_mode='r').shape}")
    print(f"  X_test:  {np.load(f'{npy_dir}/X_test.npy', mmap_mode='r').shape}")
    print(f"  Y_train: {Y_train.shape}")
    print(f"  Y_valid: {Y_valid.shape}")
    print(f"  Y_test:  {Y_test.shape}")
    print(f"  Target source: {target_name}")
    print(f"  Observed channels ({len(observed_channels)}): {', '.join(observed_channels)}")
    print(f"  Forecast channels ({len(forecast_channels)}):  {', '.join(forecast_channels)}")
    print(f"  Sensor locations: {len(SENSORS)}")
    print(f"\nCache saved to: {output_cache_dir}")

    return npy_dir, scalers_file, metadata_file


def main():
    print("="*80)
    print("GENERATING DATASET SPLIT WITH CONFIGURABLE TARGET SOURCE")
    print("="*80)

    print("\n" + "="*80)
    print("EXPERIMENT 1: AIRNOW PM2.5 AS TARGET")
    print("="*80)
    
    try:
        cache_airnow, scalers_airnow, metadata_airnow = preprocess_dataset_split(
            frames_per_sample=24,
            forecast_horizon=24,
            target_source='airnow',
            train_pct=0.75,
            valid_pct=0.13
        )
        print("\n✓ AirNow target preprocessing complete!")
        print(f"  Cache: {cache_airnow}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

  
    print("\n" + "="*80)
    print("✓ ALL PREPROCESSING COMPLETE!")

if __name__ == "__main__":
    main()
