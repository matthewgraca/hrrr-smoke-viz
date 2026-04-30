# Data Pipeline — AirNow PM2.5 Forecasting

Three-stage pipeline that turns raw per-fire data into scaled,
augmented `(N, 24, H, W, C)` arrays ready for training. Augmentation
runs on raw physical wind values (correct sign-flip / rotation
behavior); scaling is applied after.

`CHANNEL_SPEC` in `preprocess_fires_airnow.py` is the **single source of
truth** for channel layout. Stage 1 writes a `channel_spec.json`
alongside its outputs; stages 2 and 3 read it. To add/remove a channel,
edit `CHANNEL_SPEC` and re-run the pipeline — downstream picks it up.

---

## Flow

```
raw fire data
      │
      ▼  preprocess_fires_airnow.py
processed_airnow/fires/{fire}_X.npy, _Y.npy
processed_airnow/fires/channel_spec.json
      │
      ▼  build_dataset_airnow.py            (concat by split, fit scalers, NO scaling)
processed_airnow/dataset/X_train_raw.npy, Y_train_raw.npy
                         X_valid_raw.npy, Y_valid_raw.npy
                         scalers.pkl, split_info.pkl, channel_spec.json
      │
      ▼  create_data_aug_dataset_v2.py      (aug raw train, then scale all)
processed_airnow/dataset/X_train.npy, Y_train.npy   (final, scaled, optionally augmented)
                         X_valid.npy, Y_valid.npy
```

---

## Stage 1 — `preprocess_fires_airnow.py`

Loads each fire's raw data, builds 24-hour sliding windows
`(N, 24, 84, 84, C)`, writes `channel_spec.json`.

```bash
python preprocess_fires_airnow.py
python preprocess_fires_airnow.py --no-aug
```

**Outputs**: `processed_airnow/fires/{fire}_X.npy`, `_Y.npy`,
`channel_spec.json`.

---

## Stage 2 — `build_dataset_airnow.py`

Picks train/valid split by fire, fits scalers on raw train, concats
into raw arrays.

```bash
python build_dataset_airnow.py
python build_dataset_airnow.py --valid palisades_eaton line_bridge
python build_dataset_airnow.py --downsample 4
python build_dataset_airnow.py --exclude cedar_creek labor_day_eugene
python build_dataset_airnow.py --log-target
```

**Outputs**: `processed_airnow/dataset/X_train_raw.npy`, `Y_train_raw.npy`,
`X_valid_raw.npy`, `Y_valid_raw.npy`, `scalers.pkl`, `split_info.pkl`,
`channel_spec.json`.

---

## Stage 3 — `create_data_aug_dataset_v2.py`

Augments raw train (16 variants: h/v flips, rot90/270, transpose, zoom
1.5x/2x with wind-vector sign/swap handling), then applies scalers from
stage 2 to both train and valid.

```bash
python create_data_aug_dataset_v2.py --dataset-dir /path/to/processed_airnow/dataset/
python create_data_aug_dataset_v2.py --dataset-dir /path/to/processed_airnow/dataset/ --no-aug
```

**Outputs**: `X_train.npy`, `Y_train.npy`, `X_valid.npy`, `Y_valid.npy`
(scaled; train is 16x size if augmented).
