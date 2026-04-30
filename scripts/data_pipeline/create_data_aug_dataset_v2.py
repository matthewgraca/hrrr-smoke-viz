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
    """Horizontal flip (axis=W). U-component sign flips."""
    out = np.flip(x, axis=3).copy()
    if u_ch:
        out[..., u_ch] *= -1
    return out


def _vflip(x, u_ch, v_ch):
    """Vertical flip (axis=H). V-component sign flips."""
    out = np.flip(x, axis=2).copy()
    if v_ch:
        out[..., v_ch] *= -1
    return out


def _rot90(x, u_ch, v_ch):
    """90° CCW rotation in (H, W). (U, V) -> (V, -U)."""
    out = np.rot90(x, k=1, axes=(2, 3)).copy()
    if u_ch:
        u = x[..., u_ch].copy()
        v = x[..., v_ch].copy()
        out[..., u_ch] = v
        out[..., v_ch] = -u
    return out


def _rot270(x, u_ch, v_ch):
    """270° CCW rotation in (H, W). (U, V) -> (-V, U)."""
    out = np.rot90(x, k=3, axes=(2, 3)).copy()
    if u_ch:
        u = x[..., u_ch].copy()
        v = x[..., v_ch].copy()
        out[..., u_ch] = -v
        out[..., v_ch] = u
    return out


def _transpose(x, u_ch, v_ch):
    """Transpose H, W. (U, V) -> (V, U)."""
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
    """Zoom (H, W) of a (N, T, H, W, C) tensor by factor; center-crop or
    reflect-pad back to original (H, W)."""
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
    """Return list of (name, x_fn, y_fn, zoom_or_None) tuples — one per
    augmentation variant. Default 15 specs (5 base transforms × 3 zoom
    levels, where zoom=None counts as 1.0x). Combined with the original,
    augmentation factor = 16x.
    """
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


def augment_arrays(X, Y, u_channels=None, v_channels=None,
                   zoom_levels=(1.5, 2.0), include_original=True,
                   memmap_path=None):
    """Generate augmented copies of (X, Y) with consistent wind handling.

    Args:
        X: array of shape (N, T, H, W, C). Channel-last layout.
        Y: array of shape (N, T, H, W, C_y).
        u_channels: indices of wind-U channels in X. Pass None or [] to
            disable wind-aware adjustments.
        v_channels: matching wind-V channel indices.
        zoom_levels: extra zoom factors to add as variants.
        include_original: if True, originals are written first.
        memmap_path: optional (X_path, Y_path) — write to disk-backed memmaps.

    Returns:
        (X_aug, Y_aug) of shape (n_total, T, H, W, C[_y]).
    """
    if (u_channels is None) ^ (v_channels is None):
        raise ValueError("u_channels and v_channels must both be set or both None")
    if u_channels and len(u_channels) != len(v_channels):
        raise ValueError("u_channels and v_channels must have equal length")

    u_ch = u_channels or []
    v_ch = v_channels or []

    specs = build_aug_specs(zoom_levels=zoom_levels)
    n_raw = len(X)
    n_variants = len(specs)
    n_total = (n_raw if include_original else 0) + n_variants * n_raw

    if memmap_path is not None:
        x_path, y_path = memmap_path
        X_aug = open_memmap(x_path, mode='w+', dtype='float32',
                            shape=(n_total,) + X.shape[1:])
        Y_aug = open_memmap(y_path, mode='w+', dtype='float32',
                            shape=(n_total,) + Y.shape[1:])
    else:
        X_aug = np.empty((n_total,) + X.shape[1:], dtype=np.float32)
        Y_aug = np.empty((n_total,) + Y.shape[1:], dtype=np.float32)

    ptr = 0
    if include_original:
        X_aug[:n_raw] = X
        Y_aug[:n_raw] = Y
        ptr = n_raw

    for i, (name, x_fn, y_fn, zoom) in enumerate(specs):
        print(f"  [{i + 1:2d}/{n_variants}] {name:24s}", end=' ', flush=True)
        Xa = x_fn(X.copy(), u_ch, v_ch)
        Ya = y_fn(Y.copy())
        if zoom is not None:
            Xa = _zoom_5d(Xa, zoom)
            Ya = _zoom_5d(Ya, zoom)
        X_aug[ptr:ptr + n_raw] = Xa
        Y_aug[ptr:ptr + n_raw] = Ya
        ptr += n_raw
        del Xa, Ya
        gc.collect()
        print("done")

    if memmap_path is not None:
        X_aug.flush()
        Y_aug.flush()

    return X_aug, Y_aug


def scale_inplace(X, scalers, scalable_channels):
    for ch_idx, sk in scalable_channels.items():
        m = np.float32(scalers[sk].mean_[0])
        sd = np.float32(scalers[sk].scale_[0])
        X[..., ch_idx] = (X[..., ch_idx] - m) / sd
    return X


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument('--dataset-dir', required=True,
                   help='Path to dataset dir (stage 2 output) containing '
                        'X_train_raw.npy etc. and scalers.pkl, channel_spec.json')
    p.add_argument('--no-aug', action='store_true',
                   help='Skip augmentation; just scale and write final arrays.')
    p.add_argument('--zoom-levels', type=float, nargs='*', default=[1.5, 2.0])
    args = p.parse_args()

    d = args.dataset_dir
    spec_path    = os.path.join(d, 'channel_spec.json')
    scalers_path = os.path.join(d, 'scalers.pkl')
    for path in (spec_path, scalers_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Run stage 2 first.")

    with open(spec_path) as f:
        spec = json.load(f)
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)

    u_channels = spec['wind_u_channels']
    v_channels = spec['wind_v_channels']
    scalable_channels = {int(idx): sk
                         for idx, sk in spec['scalable_channels'].items()}
    print(f"  Wind: U={u_channels}  V={v_channels}")
    print(f"  Scalable channels: {len(scalable_channels)}")

    print(f"\n[LOAD] reading raw train + valid ...")
    X_train_raw = np.load(os.path.join(d, 'X_train_raw.npy')).astype(np.float32)
    Y_train_raw = np.load(os.path.join(d, 'Y_train_raw.npy')).astype(np.float32)
    X_valid_raw = np.load(os.path.join(d, 'X_valid_raw.npy')).astype(np.float32)
    Y_valid_raw = np.load(os.path.join(d, 'Y_valid_raw.npy')).astype(np.float32)
    print(f"  train: X={X_train_raw.shape}  Y={Y_train_raw.shape}")
    print(f"  valid: X={X_valid_raw.shape}  Y={Y_valid_raw.shape}")

    if args.no_aug:
        print(f"\n[SCALE-ONLY] --no-aug given; skipping augmentation.")
        X_train_out = scale_inplace(X_train_raw, scalers, scalable_channels)
        Y_train_out = Y_train_raw
    else:
        print(f"\n[AUG] Augmenting train (16x) on RAW values ...")
        X_aug, Y_aug = augment_arrays(
            X_train_raw, Y_train_raw,
            u_channels=u_channels, v_channels=v_channels,
            zoom_levels=tuple(args.zoom_levels),
        )
        del X_train_raw, Y_train_raw
        gc.collect()
        print(f"  augmented train: X={X_aug.shape}  Y={Y_aug.shape}")
        print(f"\n[SCALE] applying scalers (fit on raw train in stage 2) ...")
        X_train_out = scale_inplace(X_aug, scalers, scalable_channels)
        Y_train_out = Y_aug

    print(f"[SCALE] valid (no aug) ...")
    X_valid_out = scale_inplace(X_valid_raw, scalers, scalable_channels)
    Y_valid_out = Y_valid_raw

    print(f"\n[WRITE] final arrays ...")
    np.save(os.path.join(d, 'X_train.npy'), X_train_out)
    np.save(os.path.join(d, 'Y_train.npy'), Y_train_out)
    np.save(os.path.join(d, 'X_valid.npy'), X_valid_out)
    np.save(os.path.join(d, 'Y_valid.npy'), Y_valid_out)
    print(f"  X_train.npy  {X_train_out.shape}")
    print(f"  Y_train.npy  {Y_train_out.shape}")
    print(f"  X_valid.npy  {X_valid_out.shape}")
    print(f"  Y_valid.npy  {Y_valid_out.shape}")
    print(f"\nDONE — final scaled arrays written to {d}")


if __name__ == '__main__':
    main()
