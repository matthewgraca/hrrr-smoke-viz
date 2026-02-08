#!/usr/bin/env python3
"""
Memory-optimized operational payload generation.
Pre-allocates output tensor and frees each data source immediately after use.
"""

import sys
import os
import time
import gc
import numpy as np
import pandas as pd
import pickle
from dotenv import load_dotenv

load_dotenv()


def setup_netrc():
    username = os.environ.get('EARTHDATA_USERNAME')
    password = os.environ.get('EARTHDATA_PASSWORD')
    if username and password:
        netrc_path = '/tmp/.netrc'
        with open(netrc_path, 'w') as f:
            f.write(f"machine urs.earthdata.nasa.gov\n")
            f.write(f"    login {username}\n")
            f.write(f"    password {password}\n")
        os.environ['NETRC'] = netrc_path
        os.chmod(netrc_path, 0o600)


setup_netrc()

# ── Constants ──────────────────────────────────────────────────────────────────
EXTENT = (-118.615, -117.70, 33.60, 34.35)
DIM = 84
SCALERS_PATH = 'model/scalers.pkl'
N_CHANNELS = 30
SATELLITE_BUFFER_HOURS = 16
OUT_DIR = '/tmp/realtime'
ELEVATION_PATH = 'data/elevation.npy'

# ── Channel index mapping ─────────────────────────────────────────────────────
CH_AIRNOW_PM25 = 0
CH_AIRNOW_CLIM = 1
CH_OPENAQ_PM25 = 2
CH_NAQFC_PM25 = 3
CH_HRRR_U = 4
CH_HRRR_V = 5
CH_HRRR_WIND_SPEED = 6
CH_HRRR_TEMP = 7
CH_HRRR_PBL = 8
CH_HRRR_PRECIP = 9
CH_ELEVATION = 10
CH_NDVI = 11
CH_MONTH_SIN = 12
CH_MONTH_COS = 13
CH_HOUR_SIN = 14
CH_HOUR_COS = 15
CH_GOES = 16
CH_TEMPO = 17
CH_NAQFC_FORECAST = 18
CH_HRRR_U_FC = 19
CH_HRRR_V_FC = 20
CH_HRRR_WIND_SPEED_FC = 21
CH_HRRR_TEMP_FC = 22
CH_HRRR_PBL_FC = 23
CH_HRRR_PRECIP_FC = 24
CH_AIRNOW_CLIM_FC = 25
CH_MONTH_SIN_FC = 26
CH_MONTH_COS_FC = 27
CH_HOUR_SIN_FC = 28
CH_HOUR_COS_FC = 29


# ── Formatting helpers ─────────────────────────────────────────────────────────
class Colors:
    HEADER = '\033[95m'
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


# ── Core helpers ───────────────────────────────────────────────────────────────
def load_scalers():
    with open(SCALERS_PATH, 'rb') as f:
        scalers = pickle.load(f)
    print_success(f"Loaded {len(scalers)} scalers: {list(scalers.keys())}")
    return scalers


def normalize(data, scaler_key, scalers):
    """Apply StandardScaler normalization, return float32."""
    if scaler_key not in scalers:
        print_error(f"Scaler '{scaler_key}' not found!")
        return data.astype(np.float32)
    scaler = scalers[scaler_key]
    shape = data.shape
    return scaler.transform(data.reshape(-1, 1)).reshape(shape).astype(np.float32)


# ── Per-source fetch functions ─────────────────────────────────────────────────
# Each writes directly into X_input and frees intermediates before returning.

SENSOR_WHITELIST = [
    'Anaheim',
    'Compton',
    'Glendora - Laurel',
    'Long Beach Signal Hill',
    'Los Angeles - N. Main Street',
    'North Holywood',
    'Reseda',
]


def fetch_airnow(X_input, scalers, timestamps):
    """Fetch AirNow PM2.5 and 30-day hourly climatology."""
    print_step("\n[1/12] Fetching AirNow PM2.5...")
    t = time.time()
    from libs.airnowdata import AirNowData

    airnow = AirNowData(
        start_date=timestamps['start_date'],
        end_date=timestamps['end_date'],
        extent=EXTENT,
        airnow_api_key=os.getenv('AIRNOW_API_KEY'),
        save_dir=f'{OUT_DIR}/airnow.json',
        processed_cache_dir=f'{OUT_DIR}/airnow_processed.npz',
        elevation_path=ELEVATION_PATH,
        dim=DIM,
        force_reprocess=True,
        use_whitelist=True,
        sensor_whitelist=SENSOR_WHITELIST,
        verbose=0,
    )
    X_input[0, :, :, :, CH_AIRNOW_PM25] = normalize(airnow.data[-24:], 'AirNow_PM25', scalers)
    del airnow
    gc.collect()
    print_success(f"AirNow PM2.5 ({time.time() - t:.1f}s)")

    # ── 30-day hourly climatology ──
    print_step("\n[2/12] Computing AirNow hourly climatology (30-day lookback)...")
    t = time.time()

    clim_start = timestamps['input_start'] - pd.Timedelta(days=30)
    airnow_clim_raw = AirNowData(
        start_date=clim_start.strftime('%Y-%m-%d-%H'),
        end_date=timestamps['input_start'].strftime('%Y-%m-%d-%H'),
        extent=EXTENT,
        airnow_api_key=os.getenv('AIRNOW_API_KEY'),
        save_dir=f'{OUT_DIR}/airnow_clim.json',
        processed_cache_dir=f'{OUT_DIR}/airnow_clim_processed.npz',
        elevation_path=ELEVATION_PATH,
        dim=DIM,
        force_reprocess=True,
        verbose=0,
        use_whitelist=True,
        sensor_whitelist=SENSOR_WHITELIST,
    )
    hourly_clim = (
        airnow_clim_raw.data
        .reshape(30, 24, DIM, DIM)
        .mean(axis=0)
        .astype(np.float32)
    )
    del airnow_clim_raw
    gc.collect()

    # Input window clim
    clim_buf = np.zeros((24, DIM, DIM), dtype=np.float32)
    for i, ts in enumerate(pd.date_range(timestamps['input_start'], timestamps['input_end'], freq='h', inclusive='left')[:24]):
        clim_buf[i] = hourly_clim[ts.hour]
    X_input[0, :, :, :, CH_AIRNOW_CLIM] = normalize(clim_buf, 'AirNow_Hourly_Clim', scalers)

    # Forecast window clim
    for i, ts in enumerate(pd.date_range(timestamps['forecast_start'], timestamps['forecast_end'], freq='h', inclusive='left')[:24]):
        clim_buf[i] = hourly_clim[ts.hour]
    X_input[0, :, :, :, CH_AIRNOW_CLIM_FC] = normalize(clim_buf, 'AirNow_Hourly_Clim', scalers)

    del hourly_clim, clim_buf
    gc.collect()
    print_success(f"AirNow hourly clim ({time.time() - t:.1f}s)")


def fetch_openaq(X_input, scalers, timestamps):
    print_step("\n[3/12] Fetching OpenAQ PM2.5...")
    t = time.time()
    from libs.openaqdata import OpenAQData
    os.makedirs(f'{OUT_DIR}/openaq', exist_ok=True)
    try:
        openaq = OpenAQData(
            api_key=os.getenv('OPENAQ_API_KEY'),
            start_date=timestamps['start_date_space'],
            end_date=timestamps['end_date_space'],
            extent=EXTENT,
            dim=DIM,
            elevation_path=ELEVATION_PATH,
            save_dir=f'{OUT_DIR}/openaq',
            save_path=f'{OUT_DIR}/openaq/processed.npz',
            verbose=0,
        )
        X_input[0, :, :, :, CH_OPENAQ_PM25] = normalize(openaq.data[-24:], 'OpenAQ_PM25', scalers)
        del openaq
    except Exception as e:
        print_error(f"OpenAQ failed: {e}")
        # channel stays zeros from pre-allocation
    gc.collect()
    print_success(f"OpenAQ PM2.5 ({time.time() - t:.1f}s)")


def fetch_naqfc(X_input, scalers, timestamps):
    print_step("\n[4/12] Fetching NAQFC PM2.5...")
    t = time.time()
    from libs.naqfcdata import NAQFCData

    os.makedirs(f'{OUT_DIR}/naqfc', exist_ok=True)
    try:
        naqfc = NAQFCData(
            start_date=timestamps['start_date_space'],
            end_date=timestamps['forecast_end'].strftime('%Y-%m-%d %H:%M'),
            extent=EXTENT,
            dim=DIM,
            product='pm25',
            local_path=f'{OUT_DIR}/naqfc',
            save_path=f'{OUT_DIR}/naqfc',
            realtime=True,
            verbose=0,
        )
        X_input[0, :, :, :, CH_NAQFC_PM25] = normalize(naqfc.data[:24], 'NAQFC_PM25', scalers)
        X_input[0, :, :, :, CH_NAQFC_FORECAST] = normalize(naqfc.data[24:48], 'NAQFC_PM25', scalers)
        del naqfc
    except Exception as e:
        print_error(f"NAQFC failed: {e}")
    gc.collect()
    print_success(f"NAQFC PM2.5 ({time.time() - t:.1f}s)")


def fetch_hrrr(X_input, scalers, timestamps):
    print_step("\n[5/12] Fetching HRRR (observed)...")
    t = time.time()
    from libs.hrrrdata import HRRRData

    os.makedirs(f'{OUT_DIR}/hrrr', exist_ok=True)

    hrrr_fields = {
        'u_wind':     (CH_HRRR_U,          'HRRR_Wind_U'),
        'v_wind':     (CH_HRRR_V,          'HRRR_Wind_V'),
        'wind_speed': (CH_HRRR_WIND_SPEED, 'HRRR_Wind_Speed'),
        'temp_2m':    (CH_HRRR_TEMP,       'HRRR_Temp_2m'),
        'pbl_height': (CH_HRRR_PBL,        'HRRR_PBL_Height'),
        'precip_rate':(CH_HRRR_PRECIP,     'HRRR_Precip_Rate'),
    }

    try:
        hrrr = HRRRData(
            start_date=timestamps['start_date'],
            end_date=timestamps['end_date'],
            extent=EXTENT,
            grid_size=DIM,
            output_dir=f'{OUT_DIR}/hrrr',
            force_reprocess=True,
            verbose=False,
            max_threads=10,
            forecast=False,
        )
        for field, (ch_idx, scaler_key) in hrrr_fields.items():
            X_input[0, :, :, :, ch_idx] = normalize(hrrr.data[field], scaler_key, scalers)
        del hrrr
    except Exception as e:
        print_error(f"HRRR failed: {e}")
    gc.collect()
    print_success(f"HRRR observed ({time.time() - t:.1f}s)")

    # ── HRRR Forecast ──
    print_step("\n[6/12] Fetching HRRR forecasts (24h ahead)...")
    t = time.time()

    hrrr_fc_fields = {
        'u_wind':     (CH_HRRR_U_FC,          'HRRR_Wind_U'),
        'v_wind':     (CH_HRRR_V_FC,          'HRRR_Wind_V'),
        'wind_speed': (CH_HRRR_WIND_SPEED_FC, 'HRRR_Wind_Speed'),
        'temp_2m':    (CH_HRRR_TEMP_FC,       'HRRR_Temp_2m'),
        'pbl_height': (CH_HRRR_PBL_FC,        'HRRR_PBL_Height'),
        'precip_rate':(CH_HRRR_PRECIP_FC,     'HRRR_Precip_Rate'),
    }

    try:
        hrrr_fc = HRRRData(
            extent=EXTENT,
            grid_size=DIM,
            output_dir=f'{OUT_DIR}/hrrr_forecast',
            force_reprocess=True,
            verbose=False,
            max_threads=10,
            forecast=True,
        )
        for field, (ch_idx, scaler_key) in hrrr_fc_fields.items():
            X_input[0, :, :, :, ch_idx] = normalize(hrrr_fc.data[field], scaler_key, scalers)
        del hrrr_fc
    except Exception as e:
        print_error(f"HRRR forecast failed: {e}")
    gc.collect()
    print_success(f"HRRR forecast ({time.time() - t:.1f}s)")


def fetch_static(X_input):
    """Elevation and NDVI (placeholder)."""
    print_step("\n[7/12] Loading elevation + NDVI...")
    t = time.time()

    if os.path.exists(ELEVATION_PATH):
        elevation = np.load(ELEVATION_PATH).astype(np.float32)
        if elevation.shape != (DIM, DIM):
            import cv2
            elevation = cv2.resize(elevation, (DIM, DIM)).astype(np.float32)
        elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min() + 1e-8)
        X_input[0, :, :, :, CH_ELEVATION] = elevation[np.newaxis, :, :]  # broadcast across 24h
        del elevation
    else:
        print_info("Elevation not found, using zeros")

    # NDVI stays zeros from pre-allocation
    gc.collect()
    print_success(f"Elevation + NDVI ({time.time() - t:.1f}s)")


def fetch_time_encoding(X_input, timestamps):
    """Generate cyclical time encoding for input and forecast windows."""
    print_step("\n[8/12] Generating time encodings...")
    t = time.time()
    from libs.timedata import TimeData

    time_kwargs = dict(dim=DIM, cyclical=True, month=True, day_of_week=False, day_of_month=False, verbose=False)

    # Input window
    te = TimeData(
        start_date=timestamps['input_start'].strftime('%Y-%m-%d %H:%M'),
        end_date=timestamps['input_end'].strftime('%Y-%m-%d %H:%M'),
        **time_kwargs,
    )
    X_input[0, :, :, :, CH_MONTH_SIN] = te.data[..., 0]
    X_input[0, :, :, :, CH_MONTH_COS] = te.data[..., 1]
    X_input[0, :, :, :, CH_HOUR_SIN]  = te.data[..., 2]
    X_input[0, :, :, :, CH_HOUR_COS]  = te.data[..., 3]
    del te

    # Forecast window
    te_fc = TimeData(
        start_date=timestamps['forecast_start'].strftime('%Y-%m-%d %H:%M'),
        end_date=timestamps['forecast_end'].strftime('%Y-%m-%d %H:%M'),
        **time_kwargs,
    )
    X_input[0, :, :, :, CH_MONTH_SIN_FC] = te_fc.data[..., 0]
    X_input[0, :, :, :, CH_MONTH_COS_FC] = te_fc.data[..., 1]
    X_input[0, :, :, :, CH_HOUR_SIN_FC]  = te_fc.data[..., 2]
    X_input[0, :, :, :, CH_HOUR_COS_FC]  = te_fc.data[..., 3]
    del te_fc

    gc.collect()
    print_success(f"Time encodings ({time.time() - t:.1f}s)")


def fetch_goes(X_input, scalers, timestamps):
    print_step("\n[9/12] Fetching GOES AOD...")
    t = time.time()
    from libs.goesdata import GOESData

    os.makedirs(f'{OUT_DIR}/goes', exist_ok=True)
    try:
        goes = GOESData(
            start_date=timestamps['satellite_start'].strftime('%Y-%m-%d %H:%M'),
            end_date=timestamps['input_end'].strftime('%Y-%m-%d %H:%M'),
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
        X_input[0, :, :, :, CH_GOES] = normalize(goes.data[-24:], 'GOES', scalers)
        del goes
    except Exception as e:
        print_error(f"GOES failed: {e}")
    gc.collect()
    print_success(f"GOES AOD ({time.time() - t:.1f}s)")


def fetch_tempo(X_input, scalers, timestamps):
    print_step("\n[10/12] Fetching TEMPO NO2...")
    t = time.time()
    from libs.tempodata import TempoNO2Data

    os.makedirs(f'{OUT_DIR}/tempo', exist_ok=True)
    tempo_processed_path = f'{OUT_DIR}/tempo/operational.npz'

    try:
        if os.path.exists(tempo_processed_path):
            tempo_cache = np.load(tempo_processed_path, allow_pickle=True)
            cached_end = pd.to_datetime(tempo_cache['end_date'].item())
            if (timestamps['now'] - cached_end) > pd.Timedelta(hours=6):
                raise ValueError("Cache stale")
            X_input[0, :, :, :, CH_TEMPO] = normalize(tempo_cache['data'][-24:], 'TEMPO', scalers)
            del tempo_cache
            print_success("TEMPO NO2: loaded from cache")
        else:
            raise FileNotFoundError("No cache")

    except Exception as e:
        print_info(f"TEMPO cache miss ({e}), fetching fresh...")
        try:
            tempo = TempoNO2Data(
                start_date=timestamps['satellite_start'].strftime('%Y-%m-%d %H:%M'),
                end_date=timestamps['input_end'].strftime('%Y-%m-%d %H:%M'),
                extent=EXTENT,
                dim=DIM,
                raw_dir=f'{OUT_DIR}/tempo/raw/',
                processed_dir=f'{OUT_DIR}/tempo/',
                n_threads=4,
                cloud_threshold=0.5,
                test_mode=False,
            )
            tempo.run()
            del tempo
            gc.collect()

            tempo_files = sorted(
                f for f in os.listdir(f'{OUT_DIR}/tempo/')
                if f.startswith('tempo_no2') and f.endswith('.npz')
            )
            if tempo_files:
                tempo_proc = np.load(f'{OUT_DIR}/tempo/{tempo_files[-1]}')
                tempo_all = tempo_proc['data']
                X_input[0, :, :, :, CH_TEMPO] = normalize(tempo_all[-24:], 'TEMPO', scalers)
                np.savez_compressed(tempo_processed_path, data=tempo_all, end_date=str(timestamps['input_end']))
                del tempo_proc, tempo_all
            else:
                raise FileNotFoundError("No processed TEMPO files")

        except Exception as e2:
            print_error(f"TEMPO fetch failed: {e2}, using zeros")

    gc.collect()
    print_success(f"TEMPO NO2 ({time.time() - t:.1f}s)")


# ── Main ───────────────────────────────────────────────────────────────────────
def generate_payload():
    start_time = time.time()

    now = pd.Timestamp.now(tz='UTC').floor('h').tz_localize(None)
    input_start = now - pd.Timedelta(hours=23)
    input_end = now + pd.Timedelta(hours=1)
    forecast_start = now + pd.Timedelta(hours=1)
    forecast_end = now + pd.Timedelta(hours=25)
    satellite_start = input_start - pd.Timedelta(hours=SATELLITE_BUFFER_HOURS)

    timestamps = {
        'now': now,
        'input_start': input_start,
        'input_end': input_end,
        'forecast_start': forecast_start,
        'forecast_end': forecast_end,
        'satellite_start': satellite_start,
        'start_date': input_start.strftime('%Y-%m-%d-%H'),
        'end_date': input_end.strftime('%Y-%m-%d-%H'),
        'start_date_space': input_start.strftime('%Y-%m-%d %H:%M'),
        'end_date_space': input_end.strftime('%Y-%m-%d %H:%M'),
    }

    print_header("=" * 60)
    print_header("OPERATIONAL DATA COLLECTION (memory-optimized)")
    print_header("=" * 60)
    print_info(f"Current time (UTC):  {now}")
    print_info(f"Input window:        {input_start} to {input_end}")
    print_info(f"Forecast window:     {forecast_start} to {forecast_end}")
    print_info(f"Satellite buffer:    {satellite_start}")
    print_info(f"Extent:              {EXTENT}")
    print_info(f"Dimension:           {DIM}")
    print_header("=" * 60)

    # ── Load scalers ──
    print_step("\n[0/12] Loading scalers...")
    scalers = load_scalers()

    # ── Pre-allocate output tensor (float32) ──
    os.makedirs(OUT_DIR, exist_ok=True)
    X_input = np.zeros((1, 24, DIM, DIM, N_CHANNELS), dtype=np.float32)

    # ── Fetch each source, write to X_input, free immediately ──
    fetch_airnow(X_input, scalers, timestamps)
    fetch_openaq(X_input, scalers, timestamps)
    fetch_naqfc(X_input, scalers, timestamps)
    fetch_hrrr(X_input, scalers, timestamps)
    fetch_static(X_input)
    fetch_time_encoding(X_input, timestamps)
    fetch_goes(X_input, scalers, timestamps)
    fetch_tempo(X_input, scalers, timestamps)

    # ── Cleanup NaNs ──
    np.nan_to_num(X_input, copy=False, nan=0.0)

    # ── Save ──
    print_step("\n[11/12] Saving payload...")
    np.savez_compressed(
        f'{OUT_DIR}/payload.npz',
        X_input=X_input,
        input_start=str(input_start),
        input_end=str(input_end),
        forecast_start=str(forecast_start),
        forecast_end=str(forecast_end),
    )
    print_success(f"Saved to {OUT_DIR}/payload.npz")

    # ── Summary ──
    total_time = time.time() - start_time
    print_header(f"\n{'=' * 60}")
    print_header("SUMMARY")
    print_header("=" * 60)
    print_success(f"Input tensor: {X_input.shape} (dtype={X_input.dtype})")
    print_success(f"Channels: {N_CHANNELS}")
    print_success(f"Input window: {input_start} to {input_end}")
    print_success(f"Forecast window: {forecast_start} to {forecast_end}")
    print_success(f"Saved to: {OUT_DIR}/payload.npz")
    print_header(f"TOTAL TIME: {total_time:.1f}s ({total_time/60:.1f} min)")
    print_header("=" * 60)

    return X_input


if __name__ == "__main__":
    generate_payload()