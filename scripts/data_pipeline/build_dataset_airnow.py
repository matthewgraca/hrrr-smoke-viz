"""Stage 2 — see README.md."""

import os
import argparse
import json
import numpy as np
import pickle
import shutil
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def parse_cli_args():
    default_valid = ['palisades_eaton']
    default_processed_dir = "/home/moh/nasa/hrrr-smoke-viz/data/new_fire_data/processed_airnow"

    parser = argparse.ArgumentParser(description="Pick train/valid split and fit scalers from per-fire AirNow files")
    parser.add_argument('--processed_dir', default=default_processed_dir,
                        help='Where the processed data lives')
    parser.add_argument('--valid', nargs='+', default=default_valid,
                        help='Fire names for validation (rest go to training)')
    parser.add_argument('--exclude', nargs='+', default=[],
                        help='Fire names to exclude from BOTH train and valid pools')
    parser.add_argument('--minmax-scale', action='store_true',
                        help='Whether or not to minmax scale after standard scaling')
    return parser.parse_args()


args = parse_cli_args()

PROCESSED_DIR = args.processed_dir
FIRES_DIR     = f"{PROCESSED_DIR}/fires"
OUTPUT_DIR    = f"{PROCESSED_DIR}"

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


def fit_scalers(train_fire_names, scalable_channels, scaler_choices=['standard']):
    def std_scale(data, sk):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        print(
            f"{sk:<30}{'Standard before/after':<25}"
            f"{'mean':<8}{scaler.mean_[0]:>9.4f}, {np.mean(scaled_data):>9.4f} "
            f"{'std':<8}{scaler.scale_[0]:>9.4f}, {np.std(scaled_data):>9.4f} "
            f"n:{len(concat):>8}"
        )

        return scaled_data, scaler

    def minmax_scale(data, sk):
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        print(
            f"{sk:<30}{'MinMax before/after':<25}"
            f"{'min':<8}{scaler.data_min_[0]:>9.4f}, {np.min(scaled_data):>9.4f} "
            f"{'max':<8}{scaler.data_max_[0]:>9.4f}, {np.max(scaled_data):>9.4f} "
            f"n:{len(concat):>8}"
        )

        return scaled_data, scaler

    valid_scaler_choices = ['standard', 'minmax']
    for choice in scaler_choices:
        if choice not in valid_scaler_choices:
            raise ValueError(f'Invalid scaler choice {choice}, pick from {valid_scaler_chocies}')

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

    scalers = []
    std_scalers = {}
    minmax_scalers = {}
    for sk in sorted(raw_pools.keys()):
        arrays = raw_pools[sk]
        if arrays:
            concat = np.concatenate(arrays).reshape(-1, 1)
            # std scaler -> minmax scaler, if enabled
            for choice in scaler_choices:
                if choice == 'standard':
                    concat, std_scalers[sk] = std_scale(concat, sk)
                elif choice == 'minmax':
                    _, minmax_scalers[sk] = minmax_scale(concat, sk)
                else:
                    raise NotImplementedError()
            del concat
        else:
            print(f"    {sk:35s} identity (no data)")
            raise NotImplementedError()
    
    scalers.append(std_scalers)
    if minmax_scalers:
        scalers.append(minmax_scalers)

    return scalers


def fire_n_samples(fire_names):
    counts = {}
    sample_shape = None
    for name in fire_names:
        X = np.load(f"{FIRES_DIR}/{name}_X.npy", mmap_mode='r')
        counts[name] = int(X.shape[0])
        if sample_shape is None:
            sample_shape = tuple(X.shape[1:])
        del X
    return counts, sample_shape


def main():
    valid_fires = args.valid
    excluded = set(args.exclude)
    pool = [f for f in ALL_FIRE_NAMES if f not in excluded]
    train_fires = [f for f in pool if f not in valid_fires]
    if any(v in excluded for v in valid_fires):
        raise ValueError(f"Cannot --exclude a fire that is also --valid: "
                         f"valid={valid_fires} excluded={sorted(excluded)}")

    print("=" * 80)
    print("BUILD DATASET — AIRNOW (split + scalers, no concat)")
    print(f"  Train: {train_fires}")
    print(f"  Valid: {valid_fires}")
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

    scaler_choices = ['standard']
    if args.minmax_scale:
        scaler_choices.append('minmax')
    scalers = fit_scalers(train_fires, scalable_channels, scaler_choices)

    train_counts, sample_shape = fire_n_samples(train_fires)
    valid_counts, _            = fire_n_samples(valid_fires)
    n_train = sum(train_counts.values())
    n_valid = sum(valid_counts.values())

    with open(f"{OUTPUT_DIR}/scalers.pkl", 'wb') as f:
        pickle.dump(scalers, f)

    split_info = {
        'train_fires':     train_fires,
        'valid_fires':     valid_fires,
        'excluded_fires':  sorted(excluded),
        'train_counts':    train_counts,
        'valid_counts':    valid_counts,
        'n_train':         n_train,
        'n_valid':         n_valid,
        'sample_shape':    sample_shape,
        'target':          'AirNow_PM25',
        # NOTE make life easier and turn this off until we need it.
        #'scalers':         {sk: {'mean': s.mean_[0], 'std': s.scale_[0]}
        #                    for sk, s in scalers.items()},
    }
    with open(f"{OUTPUT_DIR}/split_info.pkl", 'wb') as f:
        pickle.dump(split_info, f)

    print("\n" + "=" * 80)
    print("DONE — split + scalers saved. Run stage 3 to build final arrays.")
    print(f"  Train fires: {train_fires}  ({n_train} samples)")
    for fn, c in train_counts.items():
        print(f"    {fn:25s} {c:>5}")
    print(f"  Valid fires: {valid_fires}  ({n_valid} samples)")
    for fn, c in valid_counts.items():
        print(f"    {fn:25s} {c:>5}")
    print(f"  Sample shape: {sample_shape}")
    print(f"  scalers.pkl, split_info.pkl, channel_spec.json -> {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
