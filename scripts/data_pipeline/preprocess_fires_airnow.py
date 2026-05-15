#!/usr/bin/env python3
"""Stage 1 — see README.md."""

import os
import json
import numpy as np
import gc
import pickle
import pandas as pd
import argparse

def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Collects channel data, metadata, and prepares dataset to be built out."
    )
    parser.add_argument(
        '--fires_dir',
        default="/home/moh/nasa/hrrr-smoke-viz/data/new_fire_data/fires",
        help='Directory where the fire samples are located'
    )
    parser.add_argument(
        '--output_dir',
        default="/home/moh/nasa/hrrr-smoke-viz/data/new_fire_data/processed_airnow",
        help="Directory where the processed data will live; same directory given to build_dataset_airnow --processed_dir"
    )
    return parser.parse_args()

args = parse_cli_args()

    # channel name,         long channel name,              scale?, temporal?, forecast?, branch name
CHANNEL_SPEC = [
    ('airnow_pm25',           'AirNow_PM25',                  True,  False,   False,  'observed'),
    ('hrrr_wind_u',           'HRRR_Wind_U',                  True,  False,   False,  'observed'),
    ('hrrr_wind_v',           'HRRR_Wind_V',                  True,  False,   False,  'observed'),
    ('hrrr_wind_speed',       'HRRR_Wind_Speed',              True,  False,   False,  'observed'),
    ('hrrr_temp_2m',          'HRRR_Temp_2m',                 True,  False,   False,  'observed'),
    ('hrrr_pbl_height',       'HRRR_PBL_Height',              True,  False,   False,  'observed'),
    ('goes',                  'GOES',                         True,  False,   False,  'observed'),
    ('tempo',                 'TEMPO',                        True,  False,   False,  'observed'),
    ('ndvi',                  'NDVI',                         False,  False,   False,  'observed'),
    ('elevation',             'Elevation',                    True,  False,   False,  'observed'),
    ('temporal_0',            'Temporal_Month_Sin',           False, True,    False,  'observed'),
    ('temporal_1',            'Temporal_Month_Cos',           False, True,    False,  'observed'),
    ('temporal_2',            'Temporal_Hour_Sin',            False, True,    False,  'observed'),
    ('temporal_3',            'Temporal_Hour_Cos',            False, True,    False,  'observed'),
    ('smoke_massden',         'HRRR_Smoke_MassDen',           True,  False,   False,  'fire'),
    ('goes_frp',              'GOES_FRP',                     True,  False,   False,  'fire'),
    ('goes_smoke_adp',        'GOES_Smoke_ADP',               True,  False,   False,  'fire'),
    ('frp_mask',              'GOES_FRP_Mask',                False, False,   False,  'fire'),
    ('frp_time_delta',        'GOES_FRP_TimeDelta',           True,  False,   False,  'fire'),
    ('time_since_ignition',   'Time_Since_Ignition',          False, False,   False,  'fire'),
    ('hrrr_fc_wind_u',        'HRRR_Wind_U_Forecast',         True,  False,   True,   'forecast'),
    ('hrrr_fc_wind_v',        'HRRR_Wind_V_Forecast',         True,  False,   True,   'forecast'),
    ('hrrr_fc_wind_speed',    'HRRR_Wind_Speed_Forecast',     True,  False,   True,   'forecast'),
    ('hrrr_fc_temp_2m',       'HRRR_Temp_2m_Forecast',        True,  False,   True,   'forecast'),
    ('hrrr_fc_pbl_height',    'HRRR_PBL_Height_Forecast',     True,  False,   True,   'forecast'),
    ('temporal_0',            'Temporal_Month_Sin_Forecast',  False, True,    True,   'forecast'),
    ('temporal_1',            'Temporal_Month_Cos_Forecast',  False, True,    True,   'forecast'),
]

FIRES_BASE = args.fires_dir
OUTPUT_DIR = args.output_dir

ALL_FIRES = [
    ("la",  "eldorado_bobcat"),
    ("la",  "line_bridge"),
    ("la",  "palisades_eaton"),
    ("sac", "august_complex_1"),
    ("sac", "august_complex_2"),
    ("sac", "glass"),
    ("sf",  "lightning_complex"),
    ("eu",  "cedar_creek"),
    ("eu",  "labor_day_eugene"),
    ("pl",  "labor_day_portland"),
]

FRAMES  = 24
HORIZON = 24
H, W    = 84, 84
N_CH    = len(CHANNEL_SPEC)


def derive_channel_metadata():
    SHARED_SCALER = {
        'HRRR_Temp_2m_Forecast':    'HRRR_Temp_2m',
        'HRRR_PBL_Height_Forecast': 'HRRR_PBL_Height',
    }
    channel_names = [ch[1] for ch in CHANNEL_SPEC]
    scalable, observed, forecast, wind_u, wind_v = {}, [], [], [], []
    for i, (_, name, scale, _temporal, _is_fc, branch) in enumerate(CHANNEL_SPEC):
        if scale:
            scalable[i] = SHARED_SCALER.get(name, name)
        (forecast if branch == 'forecast' else observed).append(i)
        if 'Wind_U' in name and 'Speed' not in name:
            wind_u.append(i)
        if 'Wind_V' in name and 'Speed' not in name:
            wind_v.append(i)
    return {
        'n_channels':         len(CHANNEL_SPEC),
        'channel_names':      channel_names,
        'scalable_channels':  scalable,
        'observed_channels':  observed,
        'forecast_channels':  forecast,
        'wind_u_channels':    wind_u,
        'wind_v_channels':    wind_v,
        'n_observed':         len(observed),
        'n_forecast':         len(forecast),
    }




def to_4d(a):
    return a if a.ndim == 4 else np.expand_dims(a, -1)


def sliding_window(data, frames, stride=1, compute_targets=False, forecast_horizon=24):
    min_len = frames + forecast_horizon
    if len(data) < min_len:
        raise ValueError(f"Need >= {min_len} timesteps, got {len(data)}")
    if compute_targets:
        n = (len(data) - frames - forecast_horizon) // stride + 1
        idx = range(0, n * stride, stride)
        X = np.array([data[i:i + frames] for i in idx])
        Y = np.array([data[i + frames:i + frames + forecast_horizon] for i in idx])
        return X, Y
    else:
        trimmed = data[:-forecast_horizon]
        n = (len(trimmed) - frames) // stride + 1
        idx = range(0, n * stride, stride)
        return np.array([trimmed[i:i + frames] for i in idx]), None


def sliding_window_forecast(data, frames, forecast_horizon, stride=1):
    min_len = frames + forecast_horizon
    if len(data) < min_len:
        raise ValueError(f"Need >= {min_len} timesteps, got {len(data)}")
    n = (len(data) - frames - forecast_horizon) // stride + 1
    idx = range(0, n * stride, stride)
    return np.array([data[i + frames:i + frames + forecast_horizon] for i in idx])


def compute_frp_mask(frp):
    return (frp.reshape(frp.shape[:3]) > 0).astype(np.float32)


def compute_frp_time_delta(frp, cap=168):
    n_t, h, w = frp.shape[:3]
    flat = frp.reshape(n_t, -1)
    td = np.full_like(flat, cap, dtype=np.float32)
    last = np.full(flat.shape[1], -cap, dtype=np.float32)
    for t in range(n_t):
        active = flat[t] > 0
        last[active] = t
        td[t] = np.clip(t - last, 0, cap)
    return np.log1p(td).reshape(n_t, h, w)


def make_temporal(start_date, n_timesteps, dim=84):
    dates = pd.date_range(start=start_date, periods=n_timesteps, freq='h')
    month_a = 2 * np.pi * (dates.month - 1).values.astype(float) / 12
    hour_a  = 2 * np.pi * dates.hour.values.astype(float) / 24
    out = np.zeros((n_timesteps, dim, dim, 4), dtype=np.float32)
    for i, arr in enumerate([np.sin(month_a), np.cos(month_a),
                             np.sin(hour_a),  np.cos(hour_a)]):
        out[:, :, :, i] = arr[:, None, None]
    return out



def load_fire(region, fire_name):
    """
    Load all available data for one fire. Uses AirNow PM2.5 (not OpenAQ).
    Returns: (data_dict, sensor_locations, start_date, n_timesteps)
    """
    fdir = f"{FIRES_BASE}/{region}/{fire_name}"
    print(f"  Loading {region}/{fire_name} ...")

    an = np.load(f"{fdir}/airnow_processed.npz", allow_pickle=True)
    sensor_locs = {
        'air_sens_loc': an['air_sens_loc'].tolist() if 'air_sens_loc' in an.files else None,
        'sensor_names': an['sensor_names'].tolist() if 'sensor_names' in an.files else None,
    }

    hrrr = np.load(f"{fdir}/hrrr_surface_84x84.npz", allow_pickle=True)
    start_date = pd.Timestamp(hrrr['timestamps'][0])
    u = hrrr['u_wind']
    v = hrrr['v_wind']

    frp_raw = np.load(f"{fdir}/goes_frp_processed.npz", allow_pickle=True)['data']
    n_t = min(an['data'].shape[0], u.shape[0], frp_raw.shape[0])

    tempo_files = [f for f in os.listdir(fdir) if f.startswith('tempo')]
    has_tempo = len(tempo_files) > 0

    data = {
        'airnow_pm25':     an['data'][:n_t].astype(np.float32),
        'hrrr_wind_u':     u[:n_t].astype(np.float32),
        'hrrr_wind_v':     v[:n_t].astype(np.float32),
        'hrrr_wind_speed': np.sqrt(u[:n_t]**2 + v[:n_t]**2).astype(np.float32),
        'hrrr_temp_2m':    hrrr['temp_2m'][:n_t].astype(np.float32),
        'hrrr_pbl_height': (hrrr['pbl_height'][:n_t].astype(np.float32)
                            if 'pbl_height' in hrrr else None),
        'goes':            np.load(f"{fdir}/goes_processed.npz",
                                   allow_pickle=True)['data'][:n_t].astype(np.float32),
        'tempo':           (np.load(f"{fdir}/{tempo_files[0]}",
                                    allow_pickle=True)['data'][:n_t].astype(np.float32)
                            if has_tempo else None),
        'ndvi':            np.load(f"{fdir}/ndvi_processed.npz",
                                   allow_pickle=True)['data'][:n_t].astype(np.float32),
        'elevation':       np.tile(np.load(f"{fdir}/elevation.npy"), (n_t, 1, 1)),
        'smoke_massden':   (hrrr['smoke_massden'][:n_t].astype(np.float32)
                            if 'smoke_massden' in hrrr else None),
        'goes_frp':        frp_raw[:n_t].astype(np.float32),
        'goes_smoke_adp':  np.load(f"{fdir}/goes_adp_processed.npz",
                                   allow_pickle=True)['data'][:n_t].astype(np.float32),
        'frp_mask':        compute_frp_mask(frp_raw[:n_t]),
        'frp_time_delta':  compute_frp_time_delta(frp_raw[:n_t]),
        'time_since_ignition': np.load(f"{fdir}/scaled_hours_since_ignition.npy"),
        'hrrr_fc_wind_u':     u[:n_t].astype(np.float32),
        'hrrr_fc_wind_v':     v[:n_t].astype(np.float32),
        'hrrr_fc_wind_speed': np.sqrt(u[:n_t]**2 + v[:n_t]**2).astype(np.float32),
        'hrrr_fc_temp_2m':    hrrr['temp_2m'][:n_t].astype(np.float32),
        'hrrr_fc_pbl_height': (hrrr['pbl_height'][:n_t].astype(np.float32)
                               if 'pbl_height' in hrrr else None),
    }

    miss = [k for k, v in data.items() if v is None]
    print(f"    T={n_t}h  start={start_date}  zeroed: {miss or 'none'}")

    return data, sensor_locs, start_date, n_t



def process_fire_unscaled(fire_data, start_date, n_t):
    """Build sliding-window X and Y for one fire. All channels UNSCALED."""
    n_samples = n_t - FRAMES - HORIZON + 1
    temporal = make_temporal(start_date, n_t, dim=H)

    X = np.zeros((n_samples, FRAMES, H, W, N_CH), dtype=np.float32)

    for ch_idx, (key, name, _, is_temporal, is_forecast, _) in enumerate(CHANNEL_SPEC):
        if is_temporal:
            tidx = int(key.split('_')[1])
            raw = temporal[:, :, :, tidx]
        elif fire_data.get(key) is not None:
            raw = fire_data[key]
        else:
            continue

        raw_4d = to_4d(raw)
        if is_forecast:
            w = sliding_window_forecast(raw_4d, FRAMES, HORIZON)[..., 0]
        else:
            w, _ = sliding_window(raw_4d, FRAMES, 1, False, HORIZON)
            w = w[..., 0]

        X[:, :, :, :, ch_idx] = w
        del w, raw_4d

    _, Y = sliding_window(
        to_4d(fire_data['airnow_pm25']), FRAMES, 1, True, HORIZON
    )

    return X, Y



def main():
    print("=" * 80)
    print(f"PER-FIRE PREPROCESSING — AIRNOW TARGET (UNSCALED)")
    print(f"Channels: {N_CH}  |  {FRAMES}h in -> {HORIZON}h out")
    print(f"Fires: {[f[1] for f in ALL_FIRES]}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 80)

    fires_dir = f"{OUTPUT_DIR}/fires"
    os.makedirs(fires_dir, exist_ok=True)

    channel_spec = derive_channel_metadata()
    with open(f"{fires_dir}/channel_spec.json", 'w') as f:
        json.dump(channel_spec, f, indent=2)

    all_sensors = {}
    fire_info = {}

    for region, fire_name in ALL_FIRES:
        print(f"\n{'─' * 60}")
        print(f"FIRE: {region}/{fire_name}")
        print(f"{'─' * 60}")

        raw_x_path = f"{fires_dir}/{fire_name}_X.npy"

        if os.path.exists(raw_x_path):
            print(f"  Already processed, skipping.")
            X = np.load(raw_x_path, mmap_mode='r')
            fire_info[fire_name] = {
                'region': region,
                'n_samples': len(X),
            }
            del X
            continue

        data, sensors, start_date, n_t = load_fire(region, fire_name)
        all_sensors[fire_name] = sensors

        print(f"  Building sliding windows...")
        X, Y = process_fire_unscaled(data, start_date, n_t)
        print(f"    X: {X.shape}  Y: {Y.shape}")
        del data
        gc.collect()

        print(f"  Saving raw...")
        np.save(f"{fires_dir}/{fire_name}_X.npy", X)
        np.save(f"{fires_dir}/{fire_name}_Y.npy", Y)

        fire_info[fire_name] = {
            'region': region,
            'n_samples': len(X),
            'sensors': sensors,
            'start_date': str(start_date),
            'n_timesteps': n_t,
        }

        del X, Y
        gc.collect()

    channel_names = [ch[1] for ch in CHANNEL_SPEC]
    metadata = {
        'channel_spec':      CHANNEL_SPEC,
        'channel_names':     channel_names,
        'observed_channels': [ch[1] for ch in CHANNEL_SPEC if ch[5] == 'observed'],
        'fire_channels':     [ch[1] for ch in CHANNEL_SPEC if ch[5] == 'fire'],
        'forecast_channels': [ch[1] for ch in CHANNEL_SPEC if ch[5] == 'forecast'],
        'n_channels':        N_CH,
        'n_observed':        sum(1 for ch in CHANNEL_SPEC if ch[5] == 'observed'),
        'n_fire':            sum(1 for ch in CHANNEL_SPEC if ch[5] == 'fire'),
        'n_forecast':        sum(1 for ch in CHANNEL_SPEC if ch[5] == 'forecast'),
        'target_name':       'AirNow_PM25',
        'frames_per_sample': FRAMES,
        'forecast_horizon':  HORIZON,
        'fires':             fire_info,
        'sensors':           all_sensors,
    }

    with open(f"{OUTPUT_DIR}/metadata.pkl", 'wb') as f:
        pickle.dump(metadata, f)

    print("\n" + "=" * 80)
    print("DONE! Per-fire files saved to:", fires_dir)
    for region, name in ALL_FIRES:
        info = fire_info.get(name, {})
        n = info.get('n_samples', '?')
        print(f"  {name:25s}  {n:>4} raw")
    print(f"\nNext: run build_dataset_airnow.py to pick train/valid split.")
    print("=" * 80)


if __name__ == "__main__":
    main()
