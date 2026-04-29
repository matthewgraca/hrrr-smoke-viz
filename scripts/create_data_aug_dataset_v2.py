#!/usr/bin/env python3
"""
Wind-aware data augmentation for spatio-temporal forecasting datasets.

Generates 16 variants per sample (1 original + 15 augmented) by applying
geometric transforms (flips, rotations, transpose, zoom 1.5x/2.0x) to a 5D
input X = (N, T, H, W, C) and target Y = (N, T, H, W, C_y), correctly
handling wind-vector channels so the physics stays consistent under flips
and rotations.

CLI
───
    python augment_dataset.py \\
        --x path/to/X_train.npy --y path/to/Y_train.npy \\
        --u-channels 1 20 --v-channels 2 21 \\
        --out-dir augmented/

Writes:
    augmented/X_aug.npy   (16x augmented copies of X)
    augmented/Y_aug.npy   (16x augmented copies of Y)

Wind handling
─────────────
--u-channels and --v-channels are parallel lists of channel indices in X.
The pair (u_channels[i], v_channels[i]) tells the script "channel u is the
U-component of a wind vector whose V-component is in channel v." Under
flips and rotations the script adjusts those signs so the wind direction
stays physically correct relative to the new geometry.

Pass empty lists to skip wind correction entirely (fine for non-wind data).

Library use
───────────
    from augment_dataset import augment_arrays
    X_aug, Y_aug = augment_arrays(X, Y, u_channels=[1, 20], v_channels=[2, 21])
"""

import argparse
import gc
import os

import numpy as np
from numpy.lib.format import open_memmap
from scipy.ndimage import zoom as ndimage_zoom


# ═══════════════════════════════════════════════════════════════════════
# Core augmentation primitives
# ═══════════════════════════════════════════════════════════════════════

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


# Y has no wind components — these helpers ignore u_ch / v_ch.
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


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument('--x', required=True, help='Path to X.npy (N,T,H,W,C)')
    p.add_argument('--y', required=True, help='Path to Y.npy (N,T,H,W,C)')
    p.add_argument('--out-dir', required=True, help='Output directory')
    p.add_argument('--u-channels', type=int, nargs='+', required=True,
                   help='Wind-U channel indices in X (e.g. 1 20). Pair-wise '
                        'matched with --v-channels.')
    p.add_argument('--v-channels', type=int, nargs='+', required=True,
                   help='Wind-V channel indices in X (e.g. 2 21).')
    p.add_argument('--zoom-levels', type=float, nargs='*', default=[1.5, 2.0])
    args = p.parse_args()

    print(f"  X: {args.x}\n  Y: {args.y}")
    X = np.load(args.x).astype(np.float32)
    Y = np.load(args.y).astype(np.float32)
    print(f"  X shape: {X.shape}\n  Y shape: {Y.shape}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_x = os.path.join(args.out_dir, 'X_aug.npy')
    out_y = os.path.join(args.out_dir, 'Y_aug.npy')

    print(f"\nAugmenting (16x) ...")
    augment_arrays(
        X, Y,
        u_channels=args.u_channels, v_channels=args.v_channels,
        zoom_levels=tuple(args.zoom_levels),
        memmap_path=(out_x, out_y),
    )
    print(f"\nWrote {out_x}\nWrote {out_y}")


if __name__ == '__main__':
    main()
