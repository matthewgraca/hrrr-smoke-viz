import sys
sys.path.append('/home/moh/nasa/hrrr-smoke-viz')

import os
import time
import numpy as np
import pandas as pd
import pickle
from dotenv import load_dotenv

load_dotenv()

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(msg):
    print(f"{Colors.BOLD}{Colors.HEADER}{msg}{Colors.END}")

def print_step(msg):
    print(f"{Colors.CYAN}{msg}{Colors.END}")

def print_success(msg):
    print(f"{Colors.GREEN}    ✓ {msg}{Colors.END}")

def print_error(msg):
    print(f"{Colors.RED}    ✗ {msg}{Colors.END}")

def print_info(msg):
    print(f"{Colors.YELLOW}    {msg}{Colors.END}")

start_time = time.time()

EXTENT = (-118.615, -117.70, 33.60, 34.35)
DIM = 84
SCALERS_PATH = '/home/moh/nasa/hrrr-smoke-viz/pwwb-experiments/tensorflow/multipath_archive/data/24out_84x84_no_holidays/scalers.pkl'

now = pd.Timestamp.now(tz='UTC').floor('h').tz_localize(None)
input_start = now - pd.Timedelta(hours=23)
input_end = now + pd.Timedelta(hours=1)
forecast_start = now + pd.Timedelta(hours=1)
forecast_end = now + pd.Timedelta(hours=25)

SATELLITE_BUFFER_HOURS = 16
satellite_start = input_start - pd.Timedelta(hours=SATELLITE_BUFFER_HOURS)

START_DATE = input_start.strftime('%Y-%m-%d-%H')
END_DATE = input_end.strftime('%Y-%m-%d-%H')
START_DATE_SPACE = input_start.strftime('%Y-%m-%d %H:%M')
END_DATE_SPACE = input_end.strftime('%Y-%m-%d %H:%M')
elevation_path = "data/elevation.npy"
OUT_DIR = 'data/operational'
os.makedirs(OUT_DIR, exist_ok=True)

print_step("\n[0/19] Loading scalers...")
with open(SCALERS_PATH, 'rb') as f:
    scalers = pickle.load(f)
print_success(f"Loaded {len(scalers)} scalers: {list(scalers.keys())}")

def normalize(data, scaler_key):
    """Apply StandardScaler normalization."""
    if scaler_key not in scalers:
        print_error(f"Scaler '{scaler_key}' not found!")
        return data
    scaler = scalers[scaler_key]
    shape = data.shape
    return scaler.transform(data.reshape(-1, 1)).reshape(shape)

print_header("=" * 60)
print_header("OPERATIONAL DATA COLLECTION")
print_header("=" * 60)
print_info(f"Current time (UTC):  {now}")
print_info(f"Input window:        {input_start} to {input_end} (24h observations)")
print_info(f"Forecast window:     {forecast_start} to {forecast_end} (24h predictions)")
print_info(f"Satellite buffer:    {satellite_start} to {input_end} ({24 + SATELLITE_BUFFER_HOURS}h for GOES/TEMPO)")
print_info(f"Extent:              {EXTENT}")
print_info(f"Dimension:           {DIM}")
print_header("=" * 60)

print_step("\n[1/19] Fetching AirNow PM2.5...")
t1 = time.time()
from libs.airnowdata import AirNowData

airnow = AirNowData(
    start_date=START_DATE,
    end_date = END_DATE,
    extent=EXTENT,
    airnow_api_key=os.getenv('AIRNOW_API_KEY'),
    save_dir=f'{OUT_DIR}/airnow.json',
    processed_cache_dir=f'{OUT_DIR}/airnow_processed.npz',
    elevation_path=elevation_path,
    dim=DIM,
    force_reprocess=True,
    use_whitelist=True,
    sensor_whitelist=[
        'Anaheim',
        'Compton',
        'Glendora - Laurel',
        'Long Beach Signal Hill',
        'Los Angeles - N. Main Street',
        'North Holywood',
        'Reseda'
    ],
    verbose=0,
)
airnow_data = normalize(airnow.data[-24:], 'AirNow_PM25')
print_success(f"AirNow PM2.5: {airnow_data.shape} ({time.time() - t1:.1f}s)")

print_step("\n[2/19] Computing AirNow hourly climatology (30-day lookback)...")
t2 = time.time()

clim_start = input_start - pd.Timedelta(days=30)
clim_end = input_start

airnow_clim_raw = AirNowData(
    start_date=clim_start.strftime('%Y-%m-%d-%H'),
    end_date=clim_end.strftime('%Y-%m-%d-%H'),
    extent=EXTENT,
    airnow_api_key=os.getenv('AIRNOW_API_KEY'),
    save_dir=f'{OUT_DIR}/airnow_clim.json',
    processed_cache_dir=f'{OUT_DIR}/airnow_clim_processed.npz',
    elevation_path=elevation_path,
    dim=DIM,
    force_reprocess=True,
    verbose=0,
    use_whitelist=True,
    sensor_whitelist=[
        'Anaheim',
        'Compton',
        'Glendora - Laurel',
        'Long Beach Signal Hill',
        'Los Angeles - N. Main Street',
        'North Holywood',
        'Reseda'
    ],
)

clim_data = airnow_clim_raw.data
clim_data = clim_data.reshape(30, 24, DIM, DIM)
hourly_clim = clim_data.mean(axis=0)

airnow_hourly_clim = np.zeros((24, DIM, DIM))
for i, ts in enumerate(pd.date_range(input_start, input_end, freq='h', inclusive='left')[:24]):
    hour = ts.hour
    airnow_hourly_clim[i] = hourly_clim[hour]
airnow_hourly_clim = normalize(airnow_hourly_clim, 'AirNow_Hourly_Clim')
print_success(f"AirNow hourly clim: {airnow_hourly_clim.shape} ({time.time() - t2:.1f}s)")

print_step("\n[3/19] Fetching OpenAQ PM2.5...")
t3 = time.time()
from libs.openaqdata import OpenAQData

os.makedirs(f'{OUT_DIR}/openaq', exist_ok=True)

try:
    openaq = OpenAQData(
        api_key=os.getenv('OPENAQ_API_KEY'),
        start_date=START_DATE_SPACE,
        end_date=END_DATE_SPACE,
        extent=EXTENT,
        dim=DIM,
        elevation_path=elevation_path,
        save_dir=f'{OUT_DIR}/openaq',
        save_path=f'{OUT_DIR}/openaq/processed.npz',
        inference_mode=True,
        expected_hours=24,
        verbose=0,
    )
    openaq_data = normalize(openaq.data[-24:], 'OpenAQ_PM25')
except Exception as e:
    print_error(f"OpenAQ failed: {e}")
    openaq_data = np.zeros((24, DIM, DIM))
print_success(f"OpenAQ PM2.5: {openaq_data.shape} ({time.time() - t3:.1f}s)")

print_step("\n[4/19] Fetching NAQFC PM2.5...")
t4 = time.time()
from libs.naqfcdata import NAQFCData

os.makedirs(f'{OUT_DIR}/naqfc', exist_ok=True)

try:
    naqfc_full = NAQFCData(
        start_date=START_DATE_SPACE,
        end_date=forecast_end.strftime('%Y-%m-%d %H:%M'),
        extent=EXTENT,
        dim=DIM,
        product='pm25',
        local_path=f'{OUT_DIR}/naqfc',
        save_path=f'{OUT_DIR}/naqfc',
        realtime=True,
        verbose=0,
    )
    naqfc_data = normalize(naqfc_full.data[:24], 'NAQFC_PM25')
    naqfc_data_forecast = normalize(naqfc_full.data[24:48], 'NAQFC_PM25')
except Exception as e:
    print_error(f"NAQFC failed: {e}")
    naqfc_data = np.zeros((24, DIM, DIM))
    naqfc_data_forecast = np.zeros((24, DIM, DIM))
print_success(f"NAQFC PM2.5: {naqfc_data.shape} ({time.time() - t4:.1f}s)")

print_step("\n[5/19] Fetching HRRR (observed)...")
t5 = time.time()
from libs.hrrrdata import HRRRData

os.makedirs(f'{OUT_DIR}/hrrr', exist_ok=True)

try:
    hrrr = HRRRData(
        start_date=START_DATE,
        end_date=END_DATE,
        extent=EXTENT,
        grid_size=DIM,
        output_dir=f'{OUT_DIR}/hrrr',
        force_reprocess=True,
        verbose=False,
        max_threads=10,
        forecast=False,
    )
    hrrr_u = normalize(hrrr.data['u_wind'], 'HRRR_Wind_U')
    hrrr_v = normalize(hrrr.data['v_wind'], 'HRRR_Wind_V')
    hrrr_wind_speed = normalize(hrrr.data['wind_speed'], 'HRRR_Wind_Speed')
    hrrr_temp = normalize(hrrr.data['temp_2m'], 'HRRR_Temp_2m')
    hrrr_pbl = normalize(hrrr.data['pbl_height'], 'HRRR_PBL_Height')
    hrrr_precip = normalize(hrrr.data['precip_rate'], 'HRRR_Precip_Rate')
except Exception as e:
    print_error(f"HRRR failed: {e}")
    hrrr_u = np.zeros((24, DIM, DIM))
    hrrr_v = np.zeros((24, DIM, DIM))
    hrrr_wind_speed = np.zeros((24, DIM, DIM))
    hrrr_temp = np.zeros((24, DIM, DIM))
    hrrr_pbl = np.zeros((24, DIM, DIM))
    hrrr_precip = np.zeros((24, DIM, DIM))
print_success(f"HRRR u/v/wind/temp/pbl/precip: {hrrr_u.shape} ({time.time() - t5:.1f}s)")

print_step("\n[6/19] Fetching HRRR forecasts (24h ahead)...")
t6 = time.time()

try:
    hrrr_forecast = HRRRData(
        extent=EXTENT,
        grid_size=DIM,
        output_dir=f'{OUT_DIR}/hrrr_forecast',
        force_reprocess=True,
        verbose=False,
        max_threads=10,
        forecast=True,
    )
    hrrr_u_forecast = normalize(hrrr_forecast.data['u_wind'], 'HRRR_Wind_U')
    hrrr_v_forecast = normalize(hrrr_forecast.data['v_wind'], 'HRRR_Wind_V')
    hrrr_wind_speed_forecast = normalize(hrrr_forecast.data['wind_speed'], 'HRRR_Wind_Speed')
    hrrr_temp_forecast = normalize(hrrr_forecast.data['temp_2m'], 'HRRR_Temp_2m')
    hrrr_pbl_forecast = normalize(hrrr_forecast.data['pbl_height'], 'HRRR_PBL_Height')
    hrrr_precip_forecast = normalize(hrrr_forecast.data['precip_rate'], 'HRRR_Precip_Rate')
except Exception as e:
    print_error(f"HRRR forecast failed: {e}")
    hrrr_u_forecast = np.zeros((24, DIM, DIM))
    hrrr_v_forecast = np.zeros((24, DIM, DIM))
    hrrr_wind_speed_forecast = np.zeros((24, DIM, DIM))
    hrrr_temp_forecast = np.zeros((24, DIM, DIM))
    hrrr_pbl_forecast = np.zeros((24, DIM, DIM))
    hrrr_precip_forecast = np.zeros((24, DIM, DIM))
print_success(f"HRRR forecast: {hrrr_u_forecast.shape} ({time.time() - t6:.1f}s)")

print_step("\n[7/19] Loading elevation...")
t7 = time.time()

elevation_path = 'data/elevation.npy'
if os.path.exists(elevation_path):
    elevation = np.load(elevation_path)
    if elevation.shape != (DIM, DIM):
        import cv2
        elevation = cv2.resize(elevation, (DIM, DIM))
    elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min() + 1e-8)
    elevation_data = np.broadcast_to(elevation, (24, DIM, DIM)).copy()
else:
    print_info("Elevation not found, using zeros")
    elevation_data = np.zeros((24, DIM, DIM))
print_success(f"Elevation: {elevation_data.shape} ({time.time() - t7:.1f}s)")

print_step("\n[8/19] NDVI (placeholder)...")
t8 = time.time()

ndvi_data = np.zeros((24, DIM, DIM))
print_success(f"NDVI: {ndvi_data.shape} ({time.time() - t8:.1f}s)")

print_step("\n[9/19] Generating time encoding (input window)...")
t9 = time.time()
from libs.timedata import TimeData

time_enc = TimeData(
    start_date=input_start.strftime('%Y-%m-%d %H:%M'),
    end_date=input_end.strftime('%Y-%m-%d %H:%M'),
    dim=DIM,
    cyclical=True,
    month=True,
    day_of_week=False,
    day_of_month=False,
    verbose=False,
)
time_data = time_enc.data

temporal_month_sin = time_data[..., 0]
temporal_month_cos = time_data[..., 1]
temporal_hour_sin = time_data[..., 2]
temporal_hour_cos = time_data[..., 3]
print_success(f"Time encoding (input): {time_data.shape} ({time.time() - t9:.1f}s)")

print_step("\n[10/19] Generating time encoding (forecast window)...")
t10 = time.time()

time_enc_forecast = TimeData(
    start_date=forecast_start.strftime('%Y-%m-%d %H:%M'),
    end_date=forecast_end.strftime('%Y-%m-%d %H:%M'),
    dim=DIM,
    cyclical=True,
    month=True,
    day_of_week=False,
    day_of_month=False,
    verbose=False,
)
time_data_forecast = time_enc_forecast.data

temporal_month_sin_forecast = time_data_forecast[..., 0]
temporal_month_cos_forecast = time_data_forecast[..., 1]
temporal_hour_sin_forecast = time_data_forecast[..., 2]
temporal_hour_cos_forecast = time_data_forecast[..., 3]
print_success(f"Time encoding (forecast): {time_data_forecast.shape} ({time.time() - t10:.1f}s)")

print_step("\n[11/19] Fetching GOES AOD (with 16h lookback buffer)...")
t11 = time.time()

from libs.goesdata import GOESData

os.makedirs(f'{OUT_DIR}/goes', exist_ok=True)

try:
    goes_full = GOESData(
        start_date=satellite_start.strftime('%Y-%m-%d %H:%M'),
        end_date=input_end.strftime('%Y-%m-%d %H:%M'),
        extent=EXTENT,
        dim=DIM,
        hourly_mean=True,
        save_dir=f'{OUT_DIR}/goes/raw',
        cache_path=f'{OUT_DIR}/goes/cache.npz',
        load_cache=False,
        save_cache=True,
        verbose=False,
        pre_downloaded=False,
    )
    goes_data = normalize(goes_full.data[-24:], 'GOES')
    print_success(f"GOES AOD: fetched {goes_full.data.shape[0]}h, using last 24 -> {goes_data.shape}")
except Exception as e:
    print_error(f"GOES failed: {e}")
    goes_data = np.zeros((24, DIM, DIM))
print_success(f"GOES: {goes_data.shape} ({time.time() - t11:.1f}s)")

print_step("\n[12/19] Fetching TEMPO NO2 (with 16h lookback buffer)...")
t12 = time.time()

from libs.tempodata import TempoNO2Data

os.makedirs(f'{OUT_DIR}/tempo', exist_ok=True)

try:
    tempo_processed_path = f'{OUT_DIR}/tempo/operational.npz'
    
    if os.path.exists(tempo_processed_path):
        tempo_cache = np.load(tempo_processed_path, allow_pickle=True)
        cached_end = pd.to_datetime(tempo_cache['end_date'].item())
        
        if (now - cached_end) > pd.Timedelta(hours=6):
            raise ValueError("Cache stale, refreshing")
        
        tempo_data = normalize(tempo_cache['data'][-24:], 'TEMPO')
        print_success(f"TEMPO NO2: loaded from cache -> {tempo_data.shape}")
    else:
        raise FileNotFoundError("No cache")
        
except Exception as e:
    print_info(f"TEMPO cache miss ({e}), fetching fresh...")
    
    try:
        tempo_full = TempoNO2Data(
            start_date=satellite_start.strftime('%Y-%m-%d %H:%M'),
            end_date=input_end.strftime('%Y-%m-%d %H:%M'),
            extent=EXTENT,
            dim=DIM,
            raw_dir=f'{OUT_DIR}/tempo/raw/',
            processed_dir=f'{OUT_DIR}/tempo/',
            n_threads=4,
            cloud_threshold=0.5,
            test_mode=False,
        )
        tempo_full.run()
        
        tempo_files = [f for f in os.listdir(f'{OUT_DIR}/tempo/') if f.startswith('tempo_no2') and f.endswith('.npz')]
        if tempo_files:
            tempo_processed = np.load(f'{OUT_DIR}/tempo/{sorted(tempo_files)[-1]}')
            tempo_all = tempo_processed['data']
            tempo_data = normalize(tempo_all[-24:], 'TEMPO')
            
            np.savez_compressed(
                tempo_processed_path,
                data=tempo_all,
                end_date=str(input_end),
            )
            print_success(f"TEMPO NO2: fetched {tempo_all.shape[0]}h, using last 24 -> {tempo_data.shape}")
        else:
            raise FileNotFoundError("No processed TEMPO files found")
            
    except Exception as e2:
        print_error(f"TEMPO fetch failed: {e2}, using zeros")
        tempo_data = np.zeros((24, DIM, DIM))

print_success(f"TEMPO: {tempo_data.shape} ({time.time() - t12:.1f}s)")

print_step("\n[13/19] NAQFC PM2.5 Forecast (already fetched)...")
t13 = time.time()
print_success(f"NAQFC forecast: {naqfc_data_forecast.shape} ({time.time() - t13:.1f}s)")

print_step("\n[14/19] HRRR Forecast (already fetched)...")
t14 = time.time()
print_success(f"HRRR forecast: {hrrr_u_forecast.shape} ({time.time() - t14:.1f}s)")

print_step("\n[15/19] Computing AirNow hourly clim (forecast window)...")
t15 = time.time()

airnow_hourly_clim_forecast = np.zeros((24, DIM, DIM))
for i, ts in enumerate(pd.date_range(forecast_start, forecast_end, freq='h', inclusive='left')[:24]):
    hour = ts.hour
    airnow_hourly_clim_forecast[i] = hourly_clim[hour]
airnow_hourly_clim_forecast = normalize(airnow_hourly_clim_forecast, 'AirNow_Hourly_Clim')
print_success(f"AirNow hourly clim (forecast): {airnow_hourly_clim_forecast.shape} ({time.time() - t15:.1f}s)")

print_step("\n[16/19] Temporal encoding forecast (already fetched)...")
t16 = time.time()
print_success(f"Temporal forecast: {time_data_forecast.shape} ({time.time() - t16:.1f}s)")

print_step("\n[17/19] Stacking into input tensor (30 channels)...")

X_input = np.stack([
    airnow_data,
    airnow_hourly_clim,
    openaq_data,
    naqfc_data,
    hrrr_u,
    hrrr_v,
    hrrr_wind_speed,
    hrrr_temp,
    hrrr_pbl,
    hrrr_precip,
    elevation_data,
    ndvi_data,
    temporal_month_sin,
    temporal_month_cos,
    temporal_hour_sin,
    temporal_hour_cos,
    goes_data,
    tempo_data,
    naqfc_data_forecast,
    hrrr_u_forecast,
    hrrr_v_forecast,
    hrrr_wind_speed_forecast,
    hrrr_temp_forecast,
    hrrr_pbl_forecast,
    hrrr_precip_forecast,
    airnow_hourly_clim_forecast,
    temporal_month_sin_forecast,
    temporal_month_cos_forecast,
    temporal_hour_sin_forecast,
    temporal_hour_cos_forecast,
], axis=-1)

X_input = np.expand_dims(X_input, axis=0)

X_input = np.nan_to_num(X_input, nan=0.0)

print_success(f"Input tensor: {X_input.shape}")

print_step("\n[18/19] Saving payload...")
np.savez_compressed(
    f'{OUT_DIR}/payload.npz',
    X_input=X_input,
    input_start=str(input_start),
    input_end=str(input_end),
    forecast_start=str(forecast_start),
    forecast_end=str(forecast_end),
)
print_success(f"Saved to {OUT_DIR}/payload.npz")

total_time = time.time() - start_time

print_step("\n[19/19] Visualizing payload...")
from viz_payload import visualize_payload
visualize_payload()

print_header("\n" + "=" * 60)
print_header("SUMMARY")
print_header("=" * 60)
print_success(f"Input tensor: {X_input.shape}")
print_success(f"Channels: {X_input.shape[-1]}")
print_success(f"Input window: {input_start} to {input_end}")
print_success(f"Forecast window: {forecast_start} to {forecast_end}")
print_success(f"Satellite buffer: {satellite_start} (16h lookback for GOES/TEMPO)")
print_success(f"Saved to: {OUT_DIR}/payload.npz")
print_header("=" * 60)
print_header(f"TOTAL TIME: {total_time:.1f}s ({total_time/60:.1f} min)")
print_header("=" * 60)