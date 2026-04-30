"""Stage 2 — see README.md."""

import os
import argparse
import json
import numpy as np
import gc
import pickle
import shutil
from sklearn.preprocessing import StandardScaler
from numpy.lib.format import open_memmap

def parse_cli_args():
    default_valid = ['palisades_eaton']
    default_processed_dir = "/home/moh/nasa/hrrr-smoke-viz/data/new_fire_data/processed_airnow" 

    parser = argparse.ArgumentParser(description="Build AirNow train/valid dataset from per-fire files")
    parser.add_argument('--processed_dir', default=default_processed_dir, 
                        help='Where are your processed data lives')
    parser.add_argument('--valid', nargs='+', default=default_valid,
                        help='Fire names for validation (rest go to training)')
    parser.add_argument('--downsample', type=int, default=None,
                        help='Downsample spatial dims by this factor (e.g. 4 for 84->21)')
    parser.add_argument('--log-target', action='store_true',
                        help='Apply log1p transform to Y target (compresses heavy tail). '
                             'Predictions need expm1 inverse-transform for raw-µg/m³ metrics.')
    parser.add_argument('--exclude', nargs='+', default=[],
                        help='Fire names to exclude from BOTH train and valid pools '
                             '(e.g. drop the new PNW fires to test their effect).')
    return parser.parse_args()

args = parse_cli_args()

PROCESSED_DIR = args.processed_dir
FIRES_DIR     = f"{PROCESSED_DIR}/fires"
OUTPUT_DIR    = f"{PROCESSED_DIR}/dataset"

ALL_FIRE_NAMES = [
    "eldorado_bobcat",
    "line_bridge",
    "palisades_eaton",
    "august_complex_1",
    "august_complex_2",
    "glass",
    "lightning_complex",
    "cedar_creek",
    "labor_day_eugene",
    "labor_day_portland",
]



def load_channel_spec():
    path = f"{FIRES_DIR}/channel_spec.json"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"channel_spec.json not found at {path}. Run stage 1 first.")
    with open(path) as f:
        spec = json.load(f)
    names = spec['channel_names']
    scalable = {int(idx): (names[int(idx)], sk)
                for idx, sk in spec['scalable_channels'].items()}
    return spec, scalable


def fit_scalers(train_fire_names, scalable_channels):
    print("\n[SCALERS] Fitting on training fires...")
    raw_pools = {}

    for name in train_fire_names:
        X = np.load(f"{FIRES_DIR}/{name}_X.npy", mmap_mode='r')
        for ch_idx, (ch_name, sk) in scalable_channels.items():
            if sk not in raw_pools:
                raw_pools[sk] = []
            ch_data = X[..., ch_idx]
            if np.any(ch_data != 0):
                raw_pools[sk].append(ch_data.flatten())
        del X

    scalers = {}
    for sk in sorted(raw_pools.keys()):
        arrays = raw_pools[sk]
        if arrays:
            concat = np.concatenate(arrays).reshape(-1, 1)
            scaler = StandardScaler()
            scaler.fit(concat)
            scalers[sk] = scaler
            print(f"    {sk:35s} mean={scaler.mean_[0]:10.4f}  "
                  f"std={scaler.scale_[0]:10.4f}  (n={len(concat)})")
            del concat
        else:
            scaler = StandardScaler()
            scaler.mean_ = np.array([0.0])
            scaler.scale_ = np.array([1.0])
            scaler.var_ = np.array([1.0])
            scaler.n_features_in_ = 1
            scalers[sk] = scaler
            print(f"    {sk:35s} identity (no data)")

    return scalers


def scale_array(X, scalers, scalable_channels):
    """Apply scalers to an X array (copy). Returns scaled copy."""
    X_scaled = X.copy() if not isinstance(X, np.memmap) else X.astype(np.float32)
    for ch_idx, (ch_name, sk) in scalable_channels.items():
        scaler = scalers[sk]
        ch = X_scaled[..., ch_idx]
        X_scaled[..., ch_idx] = (
            (ch - scaler.mean_[0]) / scaler.scale_[0]
        ).astype(np.float32)
    return X_scaled



def main():
    valid_fires = args.valid
    excluded = set(args.exclude)
    pool = [f for f in ALL_FIRE_NAMES if f not in excluded]
    train_fires = [f for f in pool if f not in valid_fires]
    if any(v in excluded for v in valid_fires):
        raise ValueError(f"Cannot --exclude a fire that is also --valid: "
                         f"valid={valid_fires} excluded={sorted(excluded)}")
    ds = args.downsample
    log_target = args.log_target

    print("=" * 80)
    print("BUILD DATASET — AIRNOW")
    print(f"  Train: {train_fires}")
    print(f"  Valid: {valid_fires}")
    if ds:
        print(f"  Downsample: {ds}x (84x84 -> {84//ds}x{84//ds})")
    if log_target:
        print(f"  Y transform: log1p (predictions need expm1 inverse for raw metrics)")
    if excluded:
        print(f"  Excluded: {sorted(excluded)}")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    channel_spec, scalable_channels = load_channel_spec()
    print(f"  Channels: {channel_spec['n_channels']}  "
          f"(observed={channel_spec['n_observed']}, "
          f"forecast={channel_spec['n_forecast']})")
    shutil.copy(f"{FIRES_DIR}/channel_spec.json",
                f"{OUTPUT_DIR}/channel_spec.json")

    scalers = fit_scalers(train_fires, scalable_channels)

    def avg_pool_np(arr, factor):
        N, T, H, W, C = arr.shape
        h_new, w_new = H // factor, W // factor
        arr = arr[:, :, :h_new*factor, :w_new*factor, :]
        arr = arr.reshape(N, T, h_new, factor, w_new, factor, C)
        return arr.mean(axis=(3, 5)).astype(np.float32)

    def stream_concat(fires, x_suffix, y_suffix, out_x, out_y,
                      label, chunk=8):
        import time
        print(f"\n[{label}] Streaming (chunk={chunk}) ...", flush=True)
        offset = 0
        for name in fires:
            X_src = np.load(f"{FIRES_DIR}/{name}{x_suffix}", mmap_mode='r')
            Y_src = np.load(f"{FIRES_DIR}/{name}{y_suffix}", mmap_mode='r')
            n = X_src.shape[0]
            n_chunks = (n + chunk - 1) // chunk
            print(f"    {name:25s}  X={X_src.shape}  "
                  f"-> offset {offset}  ({n_chunks} chunks)", flush=True)
            t0 = time.time()
            for ci, s in enumerate(range(0, n, chunk)):
                e = min(s + chunk, n)
                x_chunk = np.array(X_src[s:e], dtype=np.float32)
                y_chunk = np.array(Y_src[s:e], dtype=np.float32)
                if ds:
                    x_chunk = avg_pool_np(x_chunk, ds)
                    y_chunk = avg_pool_np(y_chunk, ds)
                if log_target:
                    y_chunk = np.log1p(y_chunk).astype(np.float32)
                out_x[offset + s:offset + e] = x_chunk
                out_y[offset + s:offset + e] = y_chunk
                del x_chunk, y_chunk

                step = max(1, n_chunks // 20)
                if (ci + 1) % step == 0 or ci == n_chunks - 1:
                    done = ci + 1
                    pct = 100 * done / n_chunks
                    elapsed = time.time() - t0
                    eta = elapsed * (n_chunks - done) / max(done, 1)
                    print(f"      [{done:>4}/{n_chunks}] {pct:5.1f}%  "
                          f"elapsed={elapsed:6.1f}s  eta={eta:6.1f}s", flush=True)
            offset += n
            del X_src, Y_src
            gc.collect()
        out_x.flush()
        out_y.flush()

    def total_n_and_shape(fires, x_suffix, y_suffix):
        total_n = 0
        x_shape = y_shape = None
        for name in fires:
            X = np.load(f"{FIRES_DIR}/{name}{x_suffix}", mmap_mode='r')
            Y = np.load(f"{FIRES_DIR}/{name}{y_suffix}", mmap_mode='r')
            total_n += X.shape[0]
            if x_shape is None:
                x_shape = X.shape[1:]
                y_shape = Y.shape[1:]
            del X, Y
        return total_n, x_shape, y_shape

    n_train, x_sample_shape, y_sample_shape = total_n_and_shape(
        train_fires, '_X.npy', '_Y.npy'
    )
    n_valid, _, _ = total_n_and_shape(valid_fires, '_X.npy', '_Y.npy')

    def out_shape(sample_shape):
        if not ds:
            return sample_shape
        T, H, W, C = sample_shape
        return (T, H // ds, W // ds, C)

    x_out_shape = out_shape(x_sample_shape)
    y_out_shape = out_shape(y_sample_shape)

    print(f"\n[PREALLOC] X_train shape=({n_train},) + {x_out_shape}")
    print(f"           X_valid shape=({n_valid},) + {x_out_shape}")

    from numpy.lib.format import open_memmap
    X_train_out = open_memmap(f"{OUTPUT_DIR}/X_train_raw.npy", mode='w+',
                              dtype=np.float32, shape=(n_train,) + x_out_shape)
    Y_train_out = open_memmap(f"{OUTPUT_DIR}/Y_train_raw.npy", mode='w+',
                              dtype=np.float32, shape=(n_train,) + y_out_shape)
    X_valid_out = open_memmap(f"{OUTPUT_DIR}/X_valid_raw.npy", mode='w+',
                              dtype=np.float32, shape=(n_valid,) + x_out_shape)
    Y_valid_out = open_memmap(f"{OUTPUT_DIR}/Y_valid_raw.npy", mode='w+',
                              dtype=np.float32, shape=(n_valid,) + y_out_shape)

    stream_concat(train_fires, '_X.npy', '_Y.npy',
                  X_train_out, Y_train_out, 'TRAIN')
    stream_concat(valid_fires, '_X.npy', '_Y.npy',
                  X_valid_out, Y_valid_out, 'VALID')

    X_train = X_train_out
    X_valid = X_valid_out
    del X_train_out, Y_train_out, X_valid_out, Y_valid_out
    gc.collect()
    print(f"\n  X_train_raw: ({n_train},) + {x_out_shape}")
    print(f"  X_valid_raw: ({n_valid},) + {x_out_shape}")

    with open(f"{OUTPUT_DIR}/scalers.pkl", 'wb') as f:
        pickle.dump(scalers, f)
    with open(os.path.join(os.path.dirname(OUTPUT_DIR), 'scalers.pkl'), 'wb') as f:
        pickle.dump(scalers, f)

    split_info = {
        'train_fires': train_fires,
        'valid_fires': valid_fires,
        'excluded_fires': sorted(excluded),
        'n_train':     len(X_train),
        'n_valid':     len(X_valid),
        'target':      'AirNow_PM25',
        'y_transform': 'log1p' if log_target else None,
        'scalers':     {sk: {'mean': s.mean_[0], 'std': s.scale_[0]}
                        for sk, s in scalers.items()},
    }
    with open(f"{OUTPUT_DIR}/split_info.pkl", 'wb') as f:
        pickle.dump(split_info, f)

    print("\n" + "=" * 80)
    print("DONE — raw arrays written. Run stage 3 to scale (and optionally augment).")
    print(f"  X_train_raw: ({n_train},) + {x_out_shape}")
    print(f"  Y_train_raw: ({n_train},) + {y_out_shape}")
    print(f"  X_valid_raw: ({n_valid},) + {x_out_shape}")
    print(f"  Y_valid_raw: ({n_valid},) + {y_out_shape}")
    print(f"  scalers.pkl  (fit on raw train, applied in stage 3)")
    print(f"  Target:  AirNow_PM25")
    print(f"  Train fires: {train_fires}")
    print(f"  Valid fires: {valid_fires}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
