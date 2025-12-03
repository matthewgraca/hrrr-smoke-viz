Assumes the files you have on-hand are stored like so. The only folder that needs to be populated is `l1`. `l2` and `l3` need to be created.
```
├── l1
│   ├── airnow_processed.npz
│   ├── elevation.npy
│   ├── goes_processed.npz
│   ├── hrrr_wind_2years_new_extent.npz
│   ├── naqfc_pm25_processed.npz
│   ├── ndvi_processed.npy
│   ├── openaq_processed.npz
│   └── tempo_l3_no2_20230802_20250802_hourly.npz
├── l1_to_l2.py
├── l2
│   ├── airnow_pm25.npy
│   ├── elevation.npy
│   ├── goes_aod.npy
│   ├── hrrr_u_component.npy
│   ├── hrrr_v_component.npy
│   ├── hrrr_wind_speed.npy
│   ├── naqfc_pm25.npy
│   ├── ndvi.npy
│   ├── openaq_pm25.npy
│   ├── tempo_no2.npy
│   ├── temporal_encoding_hour_cos.npy
│   ├── temporal_encoding_hour_sin.npy
│   ├── temporal_encoding_month_cos.npy
│   └── temporal_encoding_month_sin.npy
├── l2_to_l3.py
├── l3
│   ├── X_test.npy
│   ├── X_train.npy
│   ├── X_valid.npy
│   ├── Y_test.npy
│   ├── Y_train.npy
│   └── Y_valid.npy
└── README.md
```

Contents:
- `l1/`: direct data from the drive.
- `l2/`: l1 data processed into (time, h, w) for each channel.
- `l3/`: l2 data split, scaled, windowed, ready for training, combining all channels.
- `channels.json`: mapping of channel name to its channel index. That way you can write `X_train[..., channel['airnow_pm25']]` to get the AirNow channel. 
- `std_scaler.bin`: the standard scaler used to scale the training data for the airnow channel.

Current limitations:
- Did very limited validation; which is why I keep two scripts so you can manually inspect output before commiting to the second level of processing.
- Lot of stuff is hardcoded because we don't really have a consistent npz format. Would love for something like:
    - `arr['data'].keys()` to contain the actual data across ALL files so we can dynamically read it without needing hard coding.
    - For single channel data, we have it stored directly in 'data', and for multi-channel it's a free for all.
