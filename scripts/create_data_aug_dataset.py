import numpy as np
from numpy.lib.format import open_memmap
from datetime import datetime, timedelta
from scipy.ndimage import rotate as ndimage_rotate, zoom as ndimage_zoom

NPY_DIR = (
    "/home/moh/nasa/hrrr-smoke-viz/pwwb-experiments/tensorflow/"
    "multipath_archive/data/fire_cache_processed_84x84/npy_files"
)
CACHE_DIR = (
    "/home/moh/nasa/hrrr-smoke-viz/pwwb-experiments/tensorflow/"
    "multipath_archive/data/fire_cache"
)

START_DATE        = datetime(2023, 8, 2)
FRAMES_PER_SAMPLE = 24
TARGET_H, TARGET_W = 84, 84

U_CH = [1, 19]
V_CH = [2, 20]



def find_fire_window(search_start: datetime, search_end: datetime):
    frp_path = f"{CACHE_DIR}/goes_frp_processed.npz"
    print(f"  Loading FRP: {frp_path}")
    frp = np.load(frp_path, allow_pickle=True)['data']
    frp = frp[..., 0] if frp.ndim == 4 else frp

    s = int((search_start - START_DATE).total_seconds() // 3600)
    e = int((search_end   - START_DATE).total_seconds() // 3600)
    e = min(e, frp.shape[0])

    active = np.any(frp[s:e] > 0, axis=(1, 2))
    if not np.any(active):
        raise ValueError("No FRP activity in search window.")

    first = int(np.argmax(active))
    last  = int(len(active) - 1 - np.argmax(active[::-1]))

    abs_start = s + first
    abs_end   = s + last
    print(f"  FRP window : {START_DATE + timedelta(hours=abs_start)}"
          f"  ->  {START_DATE + timedelta(hours=abs_end)}"
          f"  ({abs_end - abs_start} h)")
    return abs_start, abs_end


def fire_sample_indices(fire_h_start, fire_h_end, n_train):
    fps = FRAMES_PER_SAMPLE
    first_idx = fire_h_start - fps
    last_idx = fire_h_end - fps + 1
    first_idx = max(0, first_idx)
    last_idx = min(n_train - 1, last_idx)
    idx = np.arange(first_idx, last_idx + 1, dtype=np.int64)
    return idx



def _rotate_wind(x, theta_deg):
    theta = np.radians(theta_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    old_u = x[..., U_CH].copy()
    old_v = x[..., V_CH].copy()
    x[..., U_CH] = old_u * cos_t - old_v * sin_t
    x[..., V_CH] = old_u * sin_t + old_v * cos_t
    return x


def _rotate_frames(data, angle_deg):
    N, T, H, W, C = data.shape
    out = np.empty((N, T, TARGET_H, TARGET_W, C), dtype=data.dtype)
    for n in range(N):
        for t in range(T):
            for c in range(C):
                rotated = ndimage_rotate(
                    data[n, t, :, :, c], angle_deg,
                    reshape=True, order=1, mode='reflect'
                )
                rh, rw = rotated.shape
                y0 = (rh - TARGET_H) // 2
                x0 = (rw - TARGET_W) // 2
                out[n, t, :, :, c] = rotated[y0:y0 + TARGET_H, x0:x0 + TARGET_W]
    return out


def _zoom_frames(data, zoom_factor):
    N, T, H, W, C = data.shape
    out = np.empty((N, T, TARGET_H, TARGET_W, C), dtype=data.dtype)
    for n in range(N):
        for t in range(T):
            for c in range(C):
                zoomed = ndimage_zoom(data[n, t, :, :, c], zoom_factor, order=1, mode='reflect')
                zh, zw = zoomed.shape
                if zoom_factor > 1.0:
                    y0 = (zh - TARGET_H) // 2
                    x0 = (zw - TARGET_W) // 2
                    out[n, t, :, :, c] = zoomed[y0:y0 + TARGET_H, x0:x0 + TARGET_W]
                else:
                    pad_h = TARGET_H - zh
                    pad_w = TARGET_W - zw
                    top = pad_h // 2
                    left = pad_w // 2
                    out[n, t, :, :, c] = np.pad(
                        zoomed,
                        ((top, pad_h - top), (left, pad_w - left)),
                        mode='reflect'
                    )
    return out


def _rotate_and_zoom_frames(data, angle_deg, zoom_factor):
    rotated = _rotate_frames(data, angle_deg)
    return _zoom_frames(rotated, zoom_factor)



def _rotate_y(y, angle_deg):
    return _rotate_frames(y, angle_deg)

def _zoom_y(y, zoom_factor):
    return _zoom_frames(y, zoom_factor)

def _rotate_and_zoom_y(y, angle_deg, zoom_factor):
    return _rotate_and_zoom_frames(y, angle_deg, zoom_factor)



def aug_hflip(x):
    out = np.flip(x, axis=3).copy()
    out[..., U_CH] *= -1
    return out

def aug_vflip(x):
    out = np.flip(x, axis=2).copy()
    out[..., V_CH] *= -1
    return out

def aug_hvflip(x):
    out = np.flip(np.flip(x, axis=3), axis=2).copy()
    out[..., U_CH] *= -1
    out[..., V_CH] *= -1
    return out

def aug_rot90(x):
    out = np.rot90(x, k=1, axes=(2, 3)).copy()
    old_u = x[..., U_CH].copy()
    old_v = x[..., V_CH].copy()
    out[..., U_CH] = old_v
    out[..., V_CH] = -old_u
    return out

def aug_rot270(x):
    out = np.rot90(x, k=3, axes=(2, 3)).copy()
    old_u = x[..., U_CH].copy()
    old_v = x[..., V_CH].copy()
    out[..., U_CH] = -old_v
    out[..., V_CH] = old_u
    return out

def aug_transpose(x):
    out = np.swapaxes(x, 2, 3).copy()
    old_u = x[..., U_CH].copy()
    old_v = x[..., V_CH].copy()
    out[..., U_CH] = old_v
    out[..., V_CH] = old_u
    return out

def aug_anti_transpose(x):
    out = np.flip(np.flip(np.swapaxes(x, 2, 3), axis=2), axis=3).copy()
    old_u = x[..., U_CH].copy()
    old_v = x[..., V_CH].copy()
    out[..., U_CH] = -old_v
    out[..., V_CH] = -old_u
    return out

def flip_y(y, axis):
    return np.flip(y, axis=axis).copy()

def rot_y(y, k):
    return np.rot90(y, k=k, axes=(2, 3)).copy()

def transpose_y(y):
    return np.swapaxes(y, 2, 3).copy()

def anti_transpose_y(y):
    return np.flip(np.flip(np.swapaxes(y, 2, 3), axis=2), axis=3).copy()



def aug_rot_angle(x, angle_deg):
    out = _rotate_frames(x, angle_deg)
    return _rotate_wind(out, angle_deg)



def aug_zoom(x, zoom_factor):
    return _zoom_frames(x, zoom_factor)



def aug_rot_zoom(x, angle_deg, zoom_factor):
    out = _rotate_and_zoom_frames(x, angle_deg, zoom_factor)
    return _rotate_wind(out, angle_deg)



def main():
    print("FIRE AUGMENTATION DATASET BUILDER")

    print("\n[1] Finding FRP fire window...")
    fire_h_start, fire_h_end = find_fire_window(
        search_start=datetime(2025, 1, 7),
        search_end  =datetime(2025, 1, 20),
    )

    print("\n[2] Opening X_train / Y_train (mmap, read-only)...")
    X_train = np.load(f"{NPY_DIR}/X_train.npy", mmap_mode='r')
    Y_train = np.load(f"{NPY_DIR}/Y_train.npy", mmap_mode='r')
    n_train, x_shape, y_shape = len(X_train), X_train.shape, Y_train.shape
    print(f"  X_train : {x_shape}")
    print(f"  Y_train : {y_shape}")

    print("\n[3] Finding fire sample indices...")
    fire_idx = fire_sample_indices(fire_h_start, fire_h_end, n_train)
    n_fire   = len(fire_idx)
    first_forecast = START_DATE + timedelta(hours=int(fire_idx[0] + FRAMES_PER_SAMPLE))
    last_input_end = START_DATE + timedelta(hours=int(fire_idx[-1] + FRAMES_PER_SAMPLE - 1))
    print(f"  Fire samples : {n_fire}  (rows {fire_idx[0]}..{fire_idx[-1]})")
    print(f"  First forecast starts: {first_forecast}")
    print(f"  Last input ends:       {last_input_end}")

    print("\n[4] Loading fire samples into RAM...")
    X_fire = X_train[fire_idx].copy()
    Y_fire = Y_train[fire_idx].copy()
    print(f"  X_fire : {X_fire.shape}   ~{X_fire.nbytes / 1e9:.2f} GB")

    print("\n  Saving fire eval set...")
    np.save(f"{NPY_DIR}/X_fire_eval.npy", X_fire)
    np.save(f"{NPY_DIR}/Y_fire_eval.npy", Y_fire)

    print("\n[5] Generating augmented variants...")

    print("\n  --- Flips & rotations ---")
    flip_rot_augmentations = [
        ('h-flip',         aug_hflip(X_fire),         flip_y(Y_fire, 3)),
        ('v-flip',         aug_vflip(X_fire),         flip_y(Y_fire, 2)),
        ('hv-flip',        aug_hvflip(X_fire),        flip_y(flip_y(Y_fire, 3), 2)),
        ('rot90',          aug_rot90(X_fire),          rot_y(Y_fire, 1)),
        ('rot270',         aug_rot270(X_fire),         rot_y(Y_fire, 3)),
        ('transpose',      aug_transpose(X_fire),      transpose_y(Y_fire)),
        ('anti-transpose', aug_anti_transpose(X_fire), anti_transpose_y(Y_fire)),
    ]
    for name, Xa, _ in flip_rot_augmentations:
        print(f"    {name:16s}: {Xa.shape}")

    print("\n  --- 45 degree rotations ---")
    rot_angles = [45, 135, 225, 315]
    rot_augmentations = []
    for angle in rot_angles:
        name = f'rot{angle}'
        Xa = aug_rot_angle(X_fire, angle)
        Ya = _rotate_y(Y_fire, angle)
        rot_augmentations.append((name, Xa, Ya))
        print(f"    {name:16s}: {Xa.shape}")

    print("\n  --- Zoom transforms ---")
    zoom_augmentations = [
        ('zoom-in-1.2x',  aug_zoom(X_fire, 1.2), _zoom_y(Y_fire, 1.2)),
        ('zoom-out-0.8x', aug_zoom(X_fire, 0.8), _zoom_y(Y_fire, 0.8)),
    ]
    for name, Xa, _ in zoom_augmentations:
        print(f"    {name:16s}: {Xa.shape}")

    print("\n  --- Zoom + 45 degree rotation combos ---")
    combo_augmentations = []
    zoom_factors = [('zin1.2', 1.2), ('zout0.8', 0.8)]
    for angle in rot_angles:
        for zname, zfactor in zoom_factors:
            name = f'rot{angle}_{zname}'
            Xa = aug_rot_zoom(X_fire, angle, zfactor)
            Ya = _rotate_and_zoom_y(Y_fire, angle, zfactor)
            combo_augmentations.append((name, Xa, Ya))
            print(f"    {name:16s}: {Xa.shape}")

    all_augmentations = flip_rot_augmentations + rot_augmentations + zoom_augmentations + combo_augmentations
    n_variants = len(all_augmentations)
    n_aug = n_variants * n_fire
    print(f"  Total variants: {n_variants}")

    print(f"\n[6] Saving X_fire_aug / Y_fire_aug  ({n_aug} samples = {n_variants} variants x {n_fire} fire)...")
    out_x = f"{NPY_DIR}/X_fire_aug.npy"
    out_y = f"{NPY_DIR}/Y_fire_aug.npy"
    X_out = open_memmap(out_x, mode='w+', dtype='float32', shape=(n_aug,) + x_shape[1:])
    Y_out = open_memmap(out_y, mode='w+', dtype='float32', shape=(n_aug,) + y_shape[1:])

    ptr = 0
    for name, Xa, Ya in all_augmentations:
        n = len(Xa)
        X_out[ptr:ptr + n] = Xa
        Y_out[ptr:ptr + n] = Ya
        print(f"  {name:16s}: {n} samples at offset {ptr}")
        ptr += n

    X_out.flush()
    Y_out.flush()

if __name__ == "__main__":
    main()
