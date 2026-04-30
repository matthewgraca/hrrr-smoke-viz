#!/usr/bin/env python3
"""Stage 3 — see README.md."""

import argparse
import gc
import json
import os
import pickle

import numpy as np
from numpy.lib.format import open_memmap
from scipy.ndimage import zoom as ndimage_zoom



def _hflip(x, u_ch, v_ch):
    out = np.flip(x, axis=3).copy()
    if u_ch:
        out[..., u_ch] *= -1
    return out


def _vflip(x, u_ch, v_ch):
    out = np.flip(x, axis=2).copy()
    if v_ch:
        out[..., v_ch] *= -1
    return out


def _rot90(x, u_ch, v_ch):
    out = np.rot90(x, k=1, axes=(2, 3)).copy()
    if u_ch:
        u = x[..., u_ch].copy()
        v = x[..., v_ch].copy()
        out[..., u_ch] = v
        out[..., v_ch] = -u
    return out


def _rot270(x, u_ch, v_ch):
    out = np.rot90(x, k=3, axes=(2, 3)).copy()
    if u_ch:
        u = x[..., u_ch].copy()
        v = x[..., v_ch].copy()
        out[..., u_ch] = -v
        out[..., v_ch] = u
    return out


def _transpose(x, u_ch, v_ch):
    out = np.swapaxes(x, 2, 3).copy()
    if u_ch:
        u = x[..., u_ch].copy()
        v = x[..., v_ch].copy()
        out[..., u_ch] = v
        out[..., v_ch] = u
    return out


def _y_hflip(y):     return np.flip(y, axis=3).copy()
def _y_vflip(y):     return np.flip(y, axis=2).copy()
def _y_rot90(y):     return np.rot90(y, k=1, axes=(2, 3)).copy()
def _y_rot270(y):    return np.rot90(y, k=3, axes=(2, 3)).copy()
def _y_transpose(y): return np.swapaxes(y, 2, 3).copy()


def _zoom_5d(data, factor):
    N, T, H, W, C = data.shape
    out = np.empty_like(data)
    for n in range(N):
        for t in range(T):
            for c in range(C):
                z = ndimage_zoom(data[n, t, :, :, c], factor,
                                 order=1, mode='reflect')
                zh, zw = z.shape
                if factor > 1.0:
                    y0 = (zh - H) // 2
                    x0 = (zw - W) // 2
                    out[n, t, :, :, c] = z[y0:y0 + H, x0:x0 + W]
                else:
                    ph = H - zh
                    pw = W - zw
                    out[n, t, :, :, c] = np.pad(
                        z,
                        ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)),
                        mode='reflect',
                    )
    return out


def build_aug_specs(zoom_levels=(1.5, 2.0)):
    base = [
        ('hflip',     _hflip,     _y_hflip),
        ('vflip',     _vflip,     _y_vflip),
        ('rot90',     _rot90,     _y_rot90),
        ('rot270',    _rot270,    _y_rot270),
        ('transpose', _transpose, _y_transpose),
    ]
    specs = []
    for name, x_fn, y_fn in base:
        specs.append((name, x_fn, y_fn, None))
        for z in zoom_levels:
            specs.append((f'{name}_z{z}x', x_fn, y_fn, z))
    return specs


def avg_pool_5d(arr, factor):
    N, T, H, W, C = arr.shape
    h_new, w_new = H // factor, W // factor
    arr = arr[:, :, :h_new * factor, :w_new * factor, :]
    arr = arr.reshape(N, T, h_new, factor, w_new, factor, C)
    return arr.mean(axis=(3, 5)).astype(np.float32)


def scale_inplace(X, scalers, scalable_channels):
    for ch_idx, sk in scalable_channels.items():
        m = np.float32(scalers[sk].mean_[0])
        sd = np.float32(scalers[sk].scale_[0])
        X[..., ch_idx] = (X[..., ch_idx] - m) / sd
    return X


def finalize_block(x_block, y_block, downsample, log_target,
                   scalers, scalable_channels):
    if downsample:
        x_block = avg_pool_5d(x_block, downsample)
        y_block = avg_pool_5d(y_block, downsample)
    scale_inplace(x_block, scalers, scalable_channels)
    if log_target:
        y_block = np.log1p(y_block).astype(np.float32)
    return x_block, y_block


def process_fire_into_memmap(fire_name, fires_dir,
                             X_out, Y_out, offset,
                             u_ch, v_ch,
                             scalers, scalable_channels,
                             downsample, log_target,
                             augment, zoom_levels=(1.5, 2.0)):
    X_src = np.load(f"{fires_dir}/{fire_name}_X.npy", mmap_mode='r')
    Y_src = np.load(f"{fires_dir}/{fire_name}_Y.npy", mmap_mode='r')
    n_raw = X_src.shape[0]

    x_block = np.array(X_src, dtype=np.float32)
    y_block = np.array(Y_src, dtype=np.float32)
    del X_src, Y_src

    xb, yb = finalize_block(x_block.copy(), y_block.copy(),
                            downsample, log_target, scalers, scalable_channels)
    X_out[offset:offset + n_raw] = xb
    Y_out[offset:offset + n_raw] = yb
    del xb, yb
    ptr = offset + n_raw

    if augment:
        specs = build_aug_specs(zoom_levels=zoom_levels)
        for i, (name, x_fn, y_fn, zoom) in enumerate(specs):
            print(f"    [{i + 1:2d}/{len(specs)}] {fire_name:25s} {name:18s}",
                  end=' ', flush=True)
            xa = x_fn(x_block, u_ch, v_ch)
            ya = y_fn(y_block)
            if zoom is not None:
                xa = _zoom_5d(xa, zoom)
                ya = _zoom_5d(ya, zoom)
            xa, ya = finalize_block(xa, ya, downsample, log_target,
                                    scalers, scalable_channels)
            X_out[ptr:ptr + n_raw] = xa
            Y_out[ptr:ptr + n_raw] = ya
            ptr += n_raw
            del xa, ya
            gc.collect()
            print("done")

    del x_block, y_block
    gc.collect()
    return ptr - offset


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-dir', required=True,
                   help='Stage 2 output dir (has split_info.pkl, scalers.pkl, channel_spec.json)')
    p.add_argument('--fires-dir', default=None,
                   help='Per-fire .npy dir (default: <dataset-dir>/../fires)')
    p.add_argument('--no-aug', action='store_true',
                   help='Skip augmentation; just scale (and optionally downsample/log) and write.')
    p.add_argument('--downsample', type=int, default=None,
                   help='Downsample spatial dims by this factor (e.g. 4 for 84->21)')
    p.add_argument('--log-target', action='store_true',
                   help='Apply log1p transform to Y target.')
    p.add_argument('--zoom-levels', type=float, nargs='*', default=[1.5, 2.0])
    args = p.parse_args()

    d = os.path.normpath(args.dataset_dir)
    fires_dir = args.fires_dir or os.path.join(os.path.dirname(d), 'fires')
    spec_path     = os.path.join(d, 'channel_spec.json')
    scalers_path  = os.path.join(d, 'scalers.pkl')
    split_path    = os.path.join(d, 'split_info.pkl')
    for path in (spec_path, scalers_path, split_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Run stage 2 first.")

    with open(spec_path) as f:
        spec = json.load(f)
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)
    with open(split_path, 'rb') as f:
        split_info = pickle.load(f)

    u_channels = spec['wind_u_channels']
    v_channels = spec['wind_v_channels']
    scalable_channels = {int(idx): sk
                         for idx, sk in spec['scalable_channels'].items()}

    train_fires    = split_info['train_fires']
    valid_fires    = split_info['valid_fires']
    train_counts   = split_info['train_counts']
    valid_counts   = split_info['valid_counts']
    sample_shape   = tuple(split_info['sample_shape'])  # (T, H, W, C)
    T, H, W, C     = sample_shape
    Y_C            = 1

    augment = not args.no_aug
    n_variants = (1 + 5 * (1 + len(args.zoom_levels))) if augment else 1

    n_train_total = sum(train_counts.values()) * n_variants
    n_valid_total = sum(valid_counts.values())

    if args.downsample:
        ds = args.downsample
        H_out, W_out = H // ds, W // ds
    else:
        H_out, W_out = H, W
    out_x_shape = (T, H_out, W_out, C)
    out_y_shape = (T, H_out, W_out, Y_C)

    print("=" * 80)
    print(f"STAGE 3 — per-fire augment + scale (option 2)")
    print(f"  Train fires:  {train_fires}")
    print(f"  Valid fires:  {valid_fires}")
    print(f"  Augment:      {'on (' + str(n_variants) + 'x)' if augment else 'off'}")
    print(f"  Downsample:   {args.downsample if args.downsample else 'no'}")
    print(f"  Log target:   {args.log_target}")
    print(f"  Output shape: X=({n_train_total},){out_x_shape}  "
          f"V=({n_valid_total},){out_x_shape}")
    print("=" * 80)

    x_train_path = os.path.join(d, 'X_train.npy')
    y_train_path = os.path.join(d, 'Y_train.npy')
    x_valid_path = os.path.join(d, 'X_valid.npy')
    y_valid_path = os.path.join(d, 'Y_valid.npy')

    print(f"\n[ALLOC] memmaps -> {d}")
    X_train_out = open_memmap(x_train_path, mode='w+', dtype='float32',
                              shape=(n_train_total,) + out_x_shape)
    Y_train_out = open_memmap(y_train_path, mode='w+', dtype='float32',
                              shape=(n_train_total,) + out_y_shape)
    X_valid_out = open_memmap(x_valid_path, mode='w+', dtype='float32',
                              shape=(n_valid_total,) + out_x_shape)
    Y_valid_out = open_memmap(y_valid_path, mode='w+', dtype='float32',
                              shape=(n_valid_total,) + out_y_shape)

    print(f"\n[TRAIN] processing per-fire ...")
    offset = 0
    for fire in train_fires:
        n = train_counts[fire] * n_variants
        print(f"  [{fire}] offset {offset} (+{n}) ...")
        written = process_fire_into_memmap(
            fire, fires_dir,
            X_train_out, Y_train_out, offset,
            u_channels, v_channels,
            scalers, scalable_channels,
            args.downsample, args.log_target,
            augment, tuple(args.zoom_levels),
        )
        assert written == n, f"{fire}: wrote {written} expected {n}"
        offset += n
    X_train_out.flush()
    Y_train_out.flush()

    print(f"\n[VALID] processing per-fire (no aug) ...")
    offset = 0
    for fire in valid_fires:
        n = valid_counts[fire]
        print(f"  [{fire}] offset {offset} (+{n}) ...")
        written = process_fire_into_memmap(
            fire, fires_dir,
            X_valid_out, Y_valid_out, offset,
            u_channels, v_channels,
            scalers, scalable_channels,
            args.downsample, args.log_target,
            augment=False,
        )
        assert written == n, f"{fire}: wrote {written} expected {n}"
        offset += n
    X_valid_out.flush()
    Y_valid_out.flush()

    print(f"\nDONE — final arrays in {d}")
    print(f"  X_train.npy  {X_train_out.shape}")
    print(f"  Y_train.npy  {Y_train_out.shape}")
    print(f"  X_valid.npy  {X_valid_out.shape}")
    print(f"  Y_valid.npy  {Y_valid_out.shape}")


if __name__ == '__main__':
    main()
