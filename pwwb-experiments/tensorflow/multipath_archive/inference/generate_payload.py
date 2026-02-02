import sys
sys.path.append('/home/moh/nasa/hrrr-smoke-viz')

import os
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

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

# Current time floored to the hour
now = pd.Timestamp.now(tz='UTC').floor('h')

# Input window: last 24 hours of observations up to and including current hour
input_end = now + pd.Timedelta(hours=1)  # exclusive end, so +1 to include current hour
input_start = input_end - pd.Timedelta(hours=24)

forecast_start = input_end  # starts right after input window
forecast_end = forecast_start + pd.Timedelta(hours=24)  # exclusive end

START_DATE = input_start.strftime('%Y-%m-%d-%H')
END_DATE = input_end.strftime('%Y-%m-%d-%H')
START_DATE_SPACE = input_start.strftime('%Y-%m-%d %H:%M')
END_DATE_SPACE = input_end.strftime('%Y-%m-%d %H:%M')
START_DATE_DAY = input_start.strftime('%Y-%m-%d')
END_DATE_DAY = input_end.strftime('%Y-%m-%d')
elevation_path = "data/elevation.npy"
OUT_DIR = 'data/operational'
os.makedirs(OUT_DIR, exist_ok=True)

print_header("=" * 60)
print_header("OPERATIONAL DATA COLLECTION")
print_header("=" * 60)
print_info(f"Current time (UTC):  {now}")
print_info(f"Input window:        {input_start} to {input_end} (24h observations)")
print_info(f"Forecast window:     {forecast_start} to {forecast_end} (24h predictions)")
print_info(f"Extent:              {EXTENT}")
print_info(f"Dimension:           {DIM}")
print_header("=" * 60)

print_step("\n[1/12] Fetching AirNow PM2.5...")
t1 = time.time()
from libs.airnowdata import AirNowData

airnow = AirNowData(
    start_date=START_DATE,
    end_date=END_DATE,
    extent=EXTENT,
    airnow_api_key=os.getenv('AIRNOW_API_KEY'),
    save_dir=f'{OUT_DIR}/airnow.json',
    processed_cache_dir=f'{OUT_DIR}/airnow_processed.npz',
    elevation_path=elevation_path,
    dim=DIM,
    force_reprocess=True,
    verbose=2,
)
airnow_data = airnow.data[-24:]
print_success(f"AirNow PM2.5: {airnow_data.shape} ({time.time() - t1:.1f}s)")

print_step("\n[2/12] Computing AirNow hourly climatology (30-day lookback)...")
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
    verbose=2,
)

clim_data = airnow_clim_raw.data
clim_data = clim_data.reshape(30, 24, DIM, DIM)
hourly_clim = clim_data.mean(axis=0)

airnow_hourly_clim = np.zeros((24, DIM, DIM))
for i, ts in enumerate(pd.date_range(input_start, input_end, freq='h', inclusive='both')[:24]):
    hour = ts.hour
    airnow_hourly_clim[i] = hourly_clim[hour]
print_success(f"AirNow hourly clim: {airnow_hourly_clim.shape} ({time.time() - t2:.1f}s)")

print_step("\n[3/12] Fetching OpenAQ PM2.5...")
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
        verbose=0,
    )
    openaq_data = openaq.data[-24:]
except Exception as e:
    print_error(f"OpenAQ failed: {e}")
    openaq_data = np.zeros((24, DIM, DIM))
print_success(f"OpenAQ PM2.5: {openaq_data.shape} ({time.time() - t3:.1f}s)")

print_step("\n[4/12] Fetching NAQFC PM2.5...")
t4 = time.time()
from libs.naqfcdata import NAQFCData

os.makedirs(f'{OUT_DIR}/naqfc', exist_ok=True)

try:
    naqfc = NAQFCData(
        start_date=START_DATE_SPACE,
        end_date=END_DATE_SPACE,
        extent=EXTENT,
        dim=DIM,
        product='pm25',
        local_path=f'{OUT_DIR}/naqfc',
        save_path=f'{OUT_DIR}/naqfc',
        verbose=2,
    )
    naqfc_data = naqfc.data[-24:]
except Exception as e:
    print_error(f"NAQFC failed: {e}")
    naqfc_data = np.zeros((24, DIM, DIM))
print_success(f"NAQFC PM2.5: {naqfc_data.shape} ({time.time() - t4:.1f}s)")

print_step("\n[5/12] Fetching HRRR (observed)...")
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
    hrrr_u = hrrr.data['u_wind']
    hrrr_v = hrrr.data['v_wind']
    hrrr_wind_speed = hrrr.data['wind_speed']
    hrrr_temp = hrrr.data['temp_2m']
    hrrr_pbl = hrrr.data['pbl_height']
    hrrr_precip = hrrr.data['precip_rate']
except Exception as e:
    print_error(f"HRRR failed: {e}")
    hrrr_u = np.zeros((24, DIM, DIM))
    hrrr_v = np.zeros((24, DIM, DIM))
    hrrr_wind_speed = np.zeros((24, DIM, DIM))
    hrrr_temp = np.zeros((24, DIM, DIM))
    hrrr_pbl = np.zeros((24, DIM, DIM))
    hrrr_precip = np.zeros((24, DIM, DIM))
print_success(f"HRRR u/v/wind/temp/pbl/precip: {hrrr_u.shape} ({time.time() - t5:.1f}s)")

print_step("\n[6/12] Fetching HRRR forecasts (24h ahead)...")
t6 = time.time()

hrrr_forecast = HRRRData(
    extent=EXTENT,
    grid_size=DIM,
    output_dir=f'{OUT_DIR}/hrrr_forecast',
    force_reprocess=True,
    verbose=False,
    max_threads=10,
    forecast=True,
)
hrrr_u_forecast = hrrr_forecast.data['u_wind']
hrrr_v_forecast = hrrr_forecast.data['v_wind']
hrrr_wind_speed_forecast = hrrr_forecast.data['wind_speed']
hrrr_temp_forecast = hrrr_forecast.data['temp_2m']
hrrr_pbl_forecast = hrrr_forecast.data['pbl_height']
hrrr_precip_forecast = hrrr_forecast.data['precip_rate']

print_success(f"HRRR forecast: {hrrr_u_forecast.shape} ({time.time() - t6:.1f}s)")

print_step("\n[7/12] Loading elevation...")
t7 = time.time()

elevation_path = 'data/elevation.npy'
if os.path.exists(elevation_path):
    elevation = np.load(elevation_path)
    if elevation.shape != (DIM, DIM):
        import cv2
        elevation = cv2.resize(elevation, (DIM, DIM))
    elevation_data = np.broadcast_to(elevation, (24, DIM, DIM)).copy()
else:
    print_info("Elevation not found, using zeros")
    elevation_data = np.zeros((24, DIM, DIM))
print_success(f"Elevation: {elevation_data.shape} ({time.time() - t7:.1f}s)")

print_step("\n[8/12] Generating time encoding (input window)...")
t8 = time.time()
from libs.timedata import TimeData

time_enc = TimeData(
    start_date=START_DATE_DAY,
    end_date=END_DATE_DAY,
    dim=DIM,
    cyclical=True,
    month=True,
    day_of_week=False,
    day_of_month=False,
    verbose=False,
)
time_data = time_enc.data[-24:]

temporal_month_sin = time_data[..., 0]
temporal_month_cos = time_data[..., 1]
temporal_hour_sin = time_data[..., 2]
temporal_hour_cos = time_data[..., 3]
print_success(f"Time encoding (input): {time_data.shape} ({time.time() - t8:.1f}s)")

print_step("\n[9/12] Generating time encoding (forecast window)...")
t9 = time.time()

time_enc_forecast = TimeData(
    start_date=forecast_start.strftime('%Y-%m-%d'),
    end_date=forecast_end.strftime('%Y-%m-%d'),
    dim=DIM,
    cyclical=True,
    month=True,
    day_of_week=False,
    day_of_month=False,
    verbose=False,
)
time_data_forecast = time_enc_forecast.data[-24:]

temporal_month_sin_forecast = time_data_forecast[..., 0]
temporal_month_cos_forecast = time_data_forecast[..., 1]
temporal_hour_sin_forecast = time_data_forecast[..., 2]
temporal_hour_cos_forecast = time_data_forecast[..., 3]
print_success(f"Time encoding (forecast): {time_data_forecast.shape} ({time.time() - t9:.1f}s)")

print_step("\n[10/12] Fetching NAQFC forecast...")
t10 = time.time()

try:
    os.makedirs(f'{OUT_DIR}/naqfc_forecast', exist_ok=True)
    naqfc_forecast = NAQFCData(
        start_date=forecast_start.strftime('%Y-%m-%d %H:%M'),
        end_date=(forecast_end + pd.Timedelta(hours=1)).strftime('%Y-%m-%d %H:%M'),
        extent=EXTENT,
        dim=DIM,
        product='pm25',
        local_path=f'{OUT_DIR}/naqfc_forecast',
        save_path=f'{OUT_DIR}/naqfc_forecast',
        verbose=2,
    )
    naqfc_data_forecast = naqfc_forecast.data[-24:]
except Exception as e:
    print_error(f"NAQFC forecast failed: {e}")
    naqfc_data_forecast = np.zeros((24, DIM, DIM))
print_success(f"NAQFC forecast: {naqfc_data_forecast.shape} ({time.time() - t10:.1f}s)")

print_step("\n[11/12] Computing AirNow hourly clim (forecast window)...")
t11 = time.time()

airnow_hourly_clim_forecast = np.zeros((24, DIM, DIM))
for i, ts in enumerate(pd.date_range(forecast_start, forecast_end, freq='h', inclusive='both')[:24]):
    hour = ts.hour
    airnow_hourly_clim_forecast[i] = hourly_clim[hour]
print_success(f"AirNow hourly clim (forecast): {airnow_hourly_clim_forecast.shape} ({time.time() - t11:.1f}s)")

print_step("\n[12/12] GOES & TEMPO (skipped for now)...")
t12 = time.time()
goes_data = np.zeros((24, DIM, DIM))
tempo_data = np.zeros((24, DIM, DIM))
print_success(f"GOES: {goes_data.shape}, TEMPO: {tempo_data.shape} ({time.time() - t12:.1f}s)")

print_step("\nStacking into input tensor...")

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

print_success(f"Input tensor: {X_input.shape}")

print_step("\nSaving payload...")
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

print_header("\n" + "=" * 60)
print_header("SUMMARY")
print_header("=" * 60)
print_success(f"Input tensor: {X_input.shape}")
print_success(f"Channels: {X_input.shape[-1]}")
print_success(f"Input window: {input_start} to {input_end}")
print_success(f"Forecast window: {forecast_start} to {forecast_end}")
print_success(f"Saved to: {OUT_DIR}/payload.npz")
print_header("=" * 60)
print_header(f"TOTAL TIME: {total_time:.1f}s ({total_time/60:.1f} min)")
print_header("=" * 60)