#!/usr/bin/env python3
"""
Standalone visualization for spatio-temporal PM2.5 forecast predictions.

Inputs:
  --preds, --trues : (N, T, H, W) or (N, T, 1, H, W) numpy arrays in ug/m3
  --inputs         : (N, T, H, W, C) numpy array of model inputs - used to
                     render the last 5 hours of context alongside each
                     forecast panel. The PM2.5 input channel is un-scaled
                     for display using the stats in scalers.pkl found next
                     to the --inputs file.

Optional inputs:
  --sensor-locations : path to airnow_processed.npz / openaq_processed.npz
                       containing a 0-d object array under 'air_sens_loc' or
                       'sensor_locations' that maps sensor name -> (row, col).
                       When given, every sample gets per-sensor time series
                       plus combined and sensor-mean summaries, and a single
                       sensor_map.png is written at the run root.
  --training-history : path to either a Keras-style history.pkl (with 'loss'
                       and 'val_loss' lists) or a train_*.log. Format is
                       auto-detected from the file extension. Renders
                       training_curve.png in the trainer's exact format
                       (best val / best epoch / final-train / final-val info box).
  --target-scaler : path to a target scaler, so that the target... gets scaled.

Output: a directory containing:
  full_timeseries.png                  - overlap-averaged across all samples
  nrmse_hourly.png                     - hourly NRMSE bar chart
  metrics.json                         - overall metrics
  sensor_map.png                       - (if sensors provided)
  training_curve.png                   - (if training log/curve provided)
  sample_NNN/
    sample_NNN_frames.png              - last-5-hour input + 24h pred + 24h truth
    sample_NNN_sensors_combined.png    - (if sensors) small-multiples per sensor
    sample_NNN_sensors_mean.png        - (if sensors) mean across sensors as a
                                         single pred-vs-truth time series
    sensor_timeseries/                 - (if sensors) one PNG per sensor
      <sensor>.png

Usage
-----
    python prediction_viz_and_metrics.py \
        --preds /path/to/preds.npy \
        --trues /path/to/trues.npy \
        --inputs /path/to/X_valid.npy \
        --out-dir reports/my_run/ \
        --fire-name palisades_eaton \
        --sensor-locations /path/to/airnow_processed.npz \
        --training-history /path/to/history.pkl
"""

import argparse
import json
import os
import pickle
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


PM25_CMAP = LinearSegmentedColormap.from_list('pm25_gradient', [
    (0/75,    '#228B22'),
    (9.0/75,  '#adff2f'),
    (20.0/75, '#ffff00'),
    (32.0/75, '#ff7e00'),
    (55.4/75, '#ff0000'),
    (65/75,   '#8f3f97'),
    (75/75,   '#7e0023'),
], N=512)
PM25_VMIN, PM25_VMAX = 0, 75


def _squeeze_channel(arr):
    """Accept (N, T, H, W), (N, T, 1, H, W), or (N, T, H, W, 1); return (N, T, H, W)."""
    err_msg = "Expected (N,T,H,W), (N,T,1,H,W), or (N,T,H,W,1); got {arr.shape}"
    if arr.ndim == 5:
        if arr.shape[2] == 1:
            return arr[:, :, 0, :, :]
        elif arr.shape[4] == 1:
            return arr[:, :, :, :, 0]
        else:
            raise ValueError(err_msg)
    if arr.ndim == 4:
        return arr
    raise ValueError(err_msg)


def _safe_filename(name):
    out = re.sub(r'[^\w\-]+', '_', name).strip('_')
    return out or 'sensor'


def _fire_label(fire_name):
    if not fire_name:
        return ''
    return fire_name.replace('_', ' ').title() + ' Fire'


def load_sensor_locations(path, grid_size, sensor_key=None):
    """Return dict {sensor_name: (row, col)} on the model grid. Scales raw 84x84
    coords to grid_size, drops 'N/A', and de-duplicates names that collide on
    the same downsampled cell. Returns (sensors, source_label) - the label is
    'AirNow' / 'OpenAQ' inferred from the npz contents, or '' if unknown."""
    if not path:
        return None, ''
    if not os.path.exists(path):
        print(f"  WARNING: --sensor-locations file not found at {path}")
        return None, ''
    d = np.load(path, allow_pickle=True)
    candidates = [sensor_key] if sensor_key else ['air_sens_loc', 'sensor_locations']
    raw = None
    used_key = None
    for k in candidates:
        if k and k in d.files:
            raw = d[k].item()
            used_key = k
            break
    if raw is None:
        print(f"  WARNING: no sensor key in {path} (tried {candidates}; "
              f"file has {list(d.files)})")
        return None, ''
    source = ''
    if used_key == 'air_sens_loc':
        source = 'AirNow'
    elif used_key == 'sensor_locations':
        source = 'OpenAQ'

    ds = max(1, 84 // grid_size)
    sensors = {}
    seen_cells = {}
    for name, (r, c) in raw.items():
        if name in ('N/A', '', None):
            continue
        rr, cc = int(r) // ds, int(c) // ds
        if not (0 <= rr < grid_size and 0 <= cc < grid_size):
            continue
        cell = (rr, cc)
        if cell in seen_cells:
            sensors[f"{name} (=={seen_cells[cell]})"] = cell
        else:
            seen_cells[cell] = name
            sensors[name] = cell
    return sensors, source


def save_frames(pred_all, truth_all, X_inputs, scalers, pm25_key,
                input_channel, sample_idx, fire_name, sample_dir):
    """Frame grid only: last-5-hours input, 24h PRED, 24h TRUTH."""
    fig = plt.figure(figsize=(26, 5.5))
    n_cols = 24
    gs = fig.add_gridspec(3, n_cols, hspace=0.25, wspace=0.03)
    axes = np.empty((3, n_cols), dtype=object)
    for r in range(3):
        for c in range(n_cols):
            axes[r, c] = fig.add_subplot(gs[r, c])

    truth_sample = truth_all[sample_idx]
    pred_sample = pred_all[sample_idx]
    truth_hour_mean = truth_sample.mean(axis=(1, 2))

    for k in range(5):
        ih = 19 + k
        x = X_inputs[sample_idx, ih, :, :, input_channel]
        org_shape = x.shape
        for scaler in reversed(scalers):
            x = (scaler[pm25_key]
                .inverse_transform(x.reshape(-1, 1))
                .reshape(org_shape)
            )
        axes[0, k].imshow(x, cmap=PM25_CMAP, vmin=PM25_VMIN, vmax=PM25_VMAX)
        axes[0, k].set_title(f'In {ih+1}\n{x.mean():.1f}', fontsize=6)
        axes[0, k].set_xticks([]); axes[0, k].set_yticks([])
    for k in range(5, n_cols):
        axes[0, k].axis('off')
    axes[0, 0].set_ylabel('Last 5 hours\nof input', fontsize=9, fontweight='bold',
                          rotation=0, labelpad=55, va='center', ha='right')

    for ri, (label, data) in enumerate(zip(['PRED', 'TRUTH'], [pred_sample, truth_sample])):
        r = ri + 1
        for h in range(24):
            axes[r, h].imshow(data[h], cmap=PM25_CMAP, vmin=PM25_VMIN, vmax=PM25_VMAX)
            if r == 1:
                axes[r, h].set_title(f'+{h+1}h\n{truth_hour_mean[h]:.1f}', fontsize=6)
            axes[r, h].set_xticks([]); axes[r, h].set_yticks([])
        axes[r, 0].set_ylabel(label, fontsize=9, fontweight='bold', rotation=0,
                              labelpad=55, va='center', ha='right')

    fig.suptitle(f'{_fire_label(fire_name) or "Forecast"} - Sample {sample_idx}',
                 fontsize=14, fontweight='bold', y=1.02)

    path = os.path.join(sample_dir, f'sample_{sample_idx:03d}_frames.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return path


def save_per_sensor_timeseries(pred_all, truth_all, sample_idx, sensors,
                               fire_name, sample_dir):
    """One time-series PNG per sensor inside sample_dir/sensor_timeseries/."""
    pred = pred_all[sample_idx]
    truth = truth_all[sample_idx]
    hours = np.arange(1, 25)
    fire_label = _fire_label(fire_name) or 'Forecast'
    sensors_dir = os.path.join(sample_dir, 'sensor_timeseries')
    os.makedirs(sensors_dir, exist_ok=True)
    metrics = {}
    for name, (r, c) in sensors.items():
        t_ts = truth[:, r, c]
        p_ts = pred[:, r, c]
        rmse = float(np.sqrt(np.mean((p_ts - t_ts) ** 2)))
        nrmse = float(rmse / (t_ts.mean() + 1e-8) * 100)
        mae = float(np.abs(p_ts - t_ts).mean())
        metrics[name] = {'row': int(r), 'col': int(c),
                         'nrmse': nrmse, 'rmse': rmse, 'mae': mae,
                         'truth_mean': float(t_ts.mean())}

        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.plot(hours, t_ts, 'g-o', markersize=4, linewidth=2, label='Truth')
        ax.plot(hours, p_ts, 'b-o', markersize=4, linewidth=2, label='Pred')
        ax.fill_between(hours, t_ts, p_ts, alpha=0.15, color='blue')
        ax.set_xlim(0.5, 24.5)
        ax.set_xlabel('Forecast Hour')
        ax.set_ylabel('PM2.5 (ug/m3)')
        ax.set_title(f'{fire_label} - Sample {sample_idx} - {name}  -  '
                     f'NRMSE={nrmse:.1f}%  MAE={mae:.2f}',
                     fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        fname = _safe_filename(name) + '.png'
        fig.savefig(os.path.join(sensors_dir, fname), dpi=130, bbox_inches='tight')
        plt.close(fig)
    return metrics


def save_combined_sensors(pred_all, truth_all, sample_idx, sensors,
                          fire_name, sample_dir):
    """All-sensor small-multiples figure at sample_dir level."""
    pred = pred_all[sample_idx]
    truth = truth_all[sample_idx]
    hours = np.arange(1, 25)
    fire_label = _fire_label(fire_name) or 'Forecast'

    n = len(sensors)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 2.8), squeeze=False)
    for i, (name, (r, c)) in enumerate(sensors.items()):
        ax = axes[i // cols, i % cols]
        t_ts = truth[:, r, c]
        p_ts = pred[:, r, c]
        rmse = float(np.sqrt(np.mean((p_ts - t_ts) ** 2)))
        nrmse = float(rmse / (t_ts.mean() + 1e-8) * 100)
        mae = float(np.abs(p_ts - t_ts).mean())
        ax.plot(hours, t_ts, 'g-o', markersize=3, linewidth=1.5, label='Truth')
        ax.plot(hours, p_ts, 'b-o', markersize=3, linewidth=1.5, label='Pred')
        ax.fill_between(hours, t_ts, p_ts, alpha=0.12, color='blue')
        ax.set_title(f'{name}\nNRMSE={nrmse:.1f}%  MAE={mae:.2f}', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, loc='upper right')
    for j in range(n, rows * cols):
        axes[j // cols, j % cols].axis('off')

    fig.suptitle(f'{fire_label} - Sample {sample_idx} - All sensors',
                 fontsize=13, fontweight='bold', y=1.0)
    fig.supxlabel('Forecast Hour', fontsize=10)
    fig.supylabel('PM2.5 (ug/m3)', fontsize=10)
    plt.tight_layout()

    path = os.path.join(sample_dir, f'sample_{sample_idx:03d}_sensors_combined.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return path


def save_sensor_mean_timeseries(pred_all, truth_all, sample_idx, sensors,
                                fire_name, sample_dir):
    """Single time series: at each forecast hour, average pred and truth across
    all sensor pixels. Returns the metrics dict."""
    pred = pred_all[sample_idx]
    truth = truth_all[sample_idx]
    hours = np.arange(1, 25)
    fire_label = _fire_label(fire_name) or 'Forecast'

    rows = np.array([rc[0] for rc in sensors.values()])
    cols = np.array([rc[1] for rc in sensors.values()])
    pred_per_sensor = pred[:, rows, cols]
    truth_per_sensor = truth[:, rows, cols]
    p_ts = pred_per_sensor.mean(axis=1)
    t_ts = truth_per_sensor.mean(axis=1)

    rmse = float(np.sqrt(np.mean((p_ts - t_ts) ** 2)))
    nrmse = float(rmse / (t_ts.mean() + 1e-8) * 100)
    mae = float(np.abs(p_ts - t_ts).mean())

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(hours, t_ts, 'g-o', markersize=4, linewidth=2, label='Truth')
    ax.plot(hours, p_ts, 'b-o', markersize=4, linewidth=2, label='Pred')
    ax.fill_between(hours, t_ts, p_ts, alpha=0.15, color='blue')
    ax.set_xlim(0.5, 24.5)
    ax.set_xlabel('Forecast Hour')
    ax.set_ylabel('PM2.5 (ug/m3)')
    ax.set_title(f'{fire_label} - Sample {sample_idx} - Mean across {len(sensors)} sensors  -  '
                 f'NRMSE={nrmse:.1f}%  MAE={mae:.2f}',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(sample_dir, f'sample_{sample_idx:03d}_sensors_mean.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return {'nrmse': nrmse, 'rmse': rmse, 'mae': mae}


def save_sensor_map(sensors, grid_size, fire_name, sensor_source, out_dir, bg=None):
    """Spatial map of sensor pixel locations on the model grid."""
    fig, ax = plt.subplots(figsize=(8, 8))
    if bg is not None:
        ax.imshow(bg, cmap=PM25_CMAP, vmin=PM25_VMIN, vmax=PM25_VMAX, alpha=0.55)
    else:
        ax.imshow(np.zeros((grid_size, grid_size)), cmap='Greys', vmin=0, vmax=1)

    for name, (r, c) in sensors.items():
        ax.scatter([c], [r], s=90, c='red', marker='o',
                   edgecolors='black', linewidths=1.5, zorder=3)
        ax.annotate(name, (c, r), xytext=(6, 4), textcoords='offset points',
                    fontsize=7, color='black',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              alpha=0.8, edgecolor='gray', lw=0.5),
                    zorder=4)

    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(grid_size - 0.5, -0.5)
    ax.set_xlabel('col')
    ax.set_ylabel('row')
    fire_label = _fire_label(fire_name) or 'Forecast'
    src = f' {sensor_source}' if sensor_source else ''
    ax.set_title(f'{fire_label}{src} Sensor Map  ({len(sensors)} sensors)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(out_dir, 'sensor_map.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return path


def save_full_timeseries(pred_all, truth_all, out_dir):
    """Overlap-averaged grid-mean time series across all samples."""
    pred_by_hour = defaultdict(list)
    truth_by_hour = defaultdict(list)
    n_valid = len(pred_all)

    for s in range(n_valid):
        for h in range(24):
            abs_h = s + 24 + h
            pred_by_hour[abs_h].append(pred_all[s, h].mean())
            truth_by_hour[abs_h].append(truth_all[s, h].mean())

    sorted_hours = sorted(pred_by_hour.keys())
    avg_pred = np.array([np.mean(pred_by_hour[t]) for t in sorted_hours])
    avg_truth = np.array([np.mean(truth_by_hour[t]) for t in sorted_hours])

    fig, ax = plt.subplots(figsize=(20, 5))
    ax.plot(sorted_hours, avg_truth, 'g-', linewidth=1.5, label='Truth (grid mean)')
    ax.plot(sorted_hours, avg_pred, 'b-', linewidth=1.5, label='Pred (grid mean)')
    ax.fill_between(sorted_hours, avg_truth, avg_pred, alpha=0.1, color='blue')

    err_curve = avg_pred - avg_truth
    mae = float(np.abs(err_curve).mean())
    rmse = float(np.sqrt((err_curve ** 2).mean()))
    bias = float(err_curve.mean())
    nrmse = float(rmse / (avg_truth.mean() + 1e-8) * 100)
    r = float(np.corrcoef(avg_pred, avg_truth)[0, 1])

    ax.set_xlabel('Hour offset from start', fontsize=12)
    ax.set_ylabel('PM2.5 (ug/m3)', fontsize=12)
    ax.set_title(f'Full Time Series ({n_valid} samples) | '
                 f'NRMSE={nrmse:.1f}%  MAE={mae:.2f}  RMSE={rmse:.2f}  '
                 f'bias={bias:.2f}  r={r:.3f}',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(out_dir, 'full_timeseries.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return {'full_ts_mae': mae, 'full_ts_rmse': rmse,
            'full_ts_nrmse': nrmse, 'full_ts_bias': bias, 'full_ts_r': r}


def save_nrmse_chart(pred_all, truth_all, out_dir):
    """Hourly NRMSE bar chart with overall metrics summary."""
    nrmse_hourly = []
    for h in range(24):
        p_h = pred_all[:, h].flatten()
        t_h = truth_all[:, h].flatten()
        rmse_h = np.sqrt(np.mean((p_h - t_h) ** 2))
        nrmse_hourly.append((rmse_h / (np.mean(t_h) + 1e-8)) * 100)

    hrs = list(range(1, 25))
    avg_nrmse = float(np.mean(nrmse_hourly))

    errors = pred_all - truth_all
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    overall_nrmse = float(rmse / truth_all.mean() * 100)

    pm25_per_hour = [float(truth_all[:, h].mean()) for h in range(24)]
    tick_labels = [f'{h}\n{pm25_per_hour[h-1]:.1f}' for h in hrs]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(hrs, nrmse_hourly, color='steelblue',
                  edgecolor='navy', lw=1.2, width=0.8)
    ax.axhline(y=avg_nrmse, color='red', ls='--', alpha=0.7, lw=1.5)
    ax.text(0.99, 0.95, f'Avg: {avg_nrmse:.2f}%', transform=ax.transAxes,
            fontsize=10, color='red', ha='right', va='top',
            bbox=dict(facecolor='white', edgecolor='red', alpha=0.85, pad=3))
    for bar, val in zip(bars, nrmse_hourly):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontsize=7)

    ax.set_xlabel('Forecast Hour (top)  /  Truth PM2.5 grid-mean ug/m3 (bottom)',
                  fontsize=11)
    ax.set_ylabel('NRMSE (%)', fontsize=11)
    ax.set_title(f'Hourly NRMSE ({len(pred_all)} samples) | '
                 f'MAE={mae:.2f}  RMSE={rmse:.2f}  Overall NRMSE={overall_nrmse:.1f}%',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(hrs)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_xlim(0, 25)
    ax.set_ylim(0, max(nrmse_hourly) * 1.15)
    plt.tight_layout()

    path = os.path.join(out_dir, 'nrmse_hourly.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return {'mae': mae, 'rmse': rmse, 'nrmse': overall_nrmse,
            'avg_hourly_nrmse': avg_nrmse}


def parse_training_log(log_path):
    epochs, train_losses, val_losses = [], [], []
    with open(log_path) as f:
        for line in f:
            m = re.search(r'Epoch:\s*(\d+).*Train Loss:\s*([\d.]+).*Vali Loss:\s*([\d.]+)', line)
            if m:
                epochs.append(int(m.group(1)))
                train_losses.append(float(m.group(2)))
                val_losses.append(float(m.group(3)))
    return epochs, train_losses, val_losses


def _load_history_pkl(path):
    """Load a Keras-style history dict with 'loss' and 'val_loss' lists.
    Returns (epochs, train_losses, val_losses) or (None, None, None) if the
    file doesn't have the expected keys."""
    with open(path, 'rb') as f:
        h = pickle.load(f)
    if not isinstance(h, dict):
        return None, None, None
    train = h.get('loss')
    val = h.get('val_loss')
    if not train or not val:
        return None, None, None
    epochs = list(range(1, len(train) + 1))
    return epochs, list(map(float, train)), list(map(float, val))


def save_training_curve(out_dir, training_history=None):
    """Render training_curve.png from either a Keras-style history.pkl or a
    train_*.log. The format is auto-detected from the file extension."""
    out_path = os.path.join(out_dir, 'training_curve.png')
    if not training_history or not os.path.exists(training_history):
        return None

    ext = os.path.splitext(training_history)[1].lower()
    if ext == '.pkl':
        epochs, train_losses, val_losses = _load_history_pkl(training_history)
    elif ext == '.log':
        epochs, train_losses, val_losses = parse_training_log(training_history)
    else:
        print(f"  WARNING: unrecognized extension {ext} for {training_history} "
              f"(expected .pkl or .log)")
        return None

    if not epochs:
        print(f"  Could not parse {training_history}")
        return None
    print(f"  Loaded {len(epochs)} epochs from {training_history}")

    best_val = min(val_losses)
    best_epoch = val_losses.index(best_val) + 1
    epochs_x = list(range(1, len(train_losses) + 1))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs_x, train_losses, label='Training Loss', lw=2, color='blue')
    ax.plot(epochs_x, val_losses,   label='Validation Loss', lw=2, color='orange')
    ax.axhline(y=best_val, color='r', ls='--', alpha=0.5)
    ax.axvline(x=best_epoch, color='g', ls='--', alpha=0.3)
    ax.set_xlim(1, len(train_losses))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training History', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    textstr = (f'Best Val: {best_val:.4f}\nBest Epoch: {best_epoch}\n'
               f'Final Train: {train_losses[-1]:.4f}\nFinal Val: {val_losses[-1]:.4f}')
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Wrote training_curve.png ({len(epochs)} epochs); "
          f"best val={best_val:.4f} @ epoch {best_epoch}")
    return out_path


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument('--preds', required=True, help='Path to preds.npy')
    p.add_argument('--trues', required=True, help='Path to trues.npy')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--title', default='forecast',
                   help='Title prefix for full_timeseries / nrmse_hourly. '
                        'Per-sample titles use --fire-name when given.')
    p.add_argument('--fire-name', default=None,
                   help='Snake-case fire name (e.g. palisades_eaton). Used to '
                        'render clean titles like "Palisades Eaton Fire - Sample N".')
    p.add_argument('--inputs', required=True,
                   help='Path to X_valid.npy for input panels (N,T,H,W,C). '
                        'scalers.pkl is auto-loaded from the same directory '
                        'to un-scale the PM2.5 input channel.')
    p.add_argument('--input-channel', type=int, default=0,
                   help='Channel index of PM2.5 in --inputs (default 0)')
    p.add_argument('--sensor-locations', default=None,
                   help='Path to airnow_processed.npz / openaq_processed.npz '
                        'with a sensor-location dict (key auto-detected).')
    p.add_argument('--sensor-key', default=None,
                   help='Override the sensor-location dict key in the npz '
                        '(default: auto-detect air_sens_loc / sensor_locations).')
    p.add_argument('--sensor-source', default=None,
                   help='Label for the sensor-map title (e.g. AirNow, OpenAQ). '
                        'Auto-detected from the npz key if not given.')
    p.add_argument('--training-history', default=None,
                   help='Path to a training history file - either a Keras-style '
                        'history.pkl (with "loss" and "val_loss" lists) or a '
                        'train_*.log. Format is auto-detected from the extension.')
    p.add_argument('--target-scaler', default=None,
                   help='Path to a target scaler pkl file.')
    args = p.parse_args()

    scalers_path = os.path.join(os.path.dirname(args.inputs), 'scalers.pkl')
    if not os.path.exists(scalers_path):
        sys.exit(f"scalers.pkl not found at {scalers_path}. The script needs "
                 f"it to un-scale the PM2.5 input channel for display.")
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)
    pm25_key = next((k for k in ('AirNow_PM25', 'OpenAQ_PM25') if k in scalers[0]), None)
    if pm25_key is None:
        sys.exit(f"No PM2.5 scaler in {scalers_path} (looked for "
                 f"AirNow_PM25 / OpenAQ_PM25). Keys present: {list(scalers[0].keys())}")
    print(f"{len(scalers)} scalers found.")

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading preds: {args.preds}")
    pred_all = _squeeze_channel(np.load(args.preds, mmap_mode='r'))
    print(f"  shape: {pred_all.shape}")
    print(f"Loading trues: {args.trues}")
    truth_all = _squeeze_channel(np.load(args.trues, mmap_mode='r'))
    print(f"  shape: {truth_all.shape}")

    if pred_all.shape != truth_all.shape:
        raise ValueError(f"shape mismatch: preds {pred_all.shape} vs "
                         f"trues {truth_all.shape}")

    pred_all = np.array(pred_all, dtype=np.float32)
    truth_all = np.array(truth_all, dtype=np.float32)

    if args.target_scaler:
        if not os.path.exists(args.target_scaler):
            raise ValueError(f'Target scaler file on {args.target_scaler} not found.')
        else:
            with open(args.target_scaler, 'rb') as f:
                scaler = pickle.load(f)
            org_shape = pred_all.shape
            pred_all = scaler.inverse_transform(pred_all.reshape(-1, 1)).reshape(org_shape)
            org_shape = truth_all.shape
            truth_all = scaler.inverse_transform(truth_all.reshape(-1, 1)).reshape(org_shape)

    print(f"Loading inputs: {args.inputs}")
    X_inputs = np.load(args.inputs, mmap_mode='r')
    print(f"  shape: {X_inputs.shape}")
    if X_inputs.ndim != 5:
        raise ValueError(f"--inputs must be (N,T,H,W,C); got {X_inputs.shape}")
    if len(X_inputs) != len(pred_all):
        raise ValueError(f"--inputs N={len(X_inputs)} != preds N={len(pred_all)}")
    if args.input_channel >= X_inputs.shape[-1]:
        raise ValueError(f"--input-channel {args.input_channel} out of range "
                         f"for C={X_inputs.shape[-1]}")

    grid_size = pred_all.shape[-1]
    sensors, auto_source = load_sensor_locations(args.sensor_locations, grid_size,
                                                 sensor_key=args.sensor_key)
    sensor_source = args.sensor_source or auto_source

    if sensors:
        print(f"Loaded {len(sensors)} sensor locations "
              f"(source={sensor_source or 'unknown'})")
        bg = truth_all[0, 0]
        save_sensor_map(sensors, grid_size, args.fire_name, sensor_source,
                        args.out_dir, bg=bg)
        print("  Saved sensor_map.png at run root")

    print(f"\nWriting per-sample frames + sensor plots for all {len(pred_all)} samples...")
    per_sample_metrics = {}
    for idx in range(len(pred_all)):
        sample_dir = os.path.join(args.out_dir, f'sample_{idx:03d}')
        os.makedirs(sample_dir, exist_ok=True)
        save_frames(pred_all, truth_all, X_inputs, scalers, pm25_key,
                    args.input_channel, idx, args.fire_name, sample_dir)
        if sensors:
            ps = save_per_sensor_timeseries(pred_all, truth_all, idx, sensors,
                                            args.fire_name, sample_dir)
            save_combined_sensors(pred_all, truth_all, idx, sensors,
                                  args.fire_name, sample_dir)
            mean_metrics = save_sensor_mean_timeseries(pred_all, truth_all, idx,
                                                       sensors, args.fire_name,
                                                       sample_dir)
            avg_nrmse = float(np.mean([v['nrmse'] for v in ps.values()]))
            per_sample_metrics[idx] = {'avg_sensor_nrmse': avg_nrmse,
                                       'sensors_mean': mean_metrics,
                                       'per_sensor': ps}

    print("Writing full time series...")
    full_metrics = save_full_timeseries(pred_all, truth_all, args.out_dir)

    print("Writing NRMSE chart...")
    overall = save_nrmse_chart(pred_all, truth_all, args.out_dir)

    print("Writing training curve...")
    save_training_curve(args.out_dir, training_history=args.training_history)

    metrics = {**overall, **full_metrics, 'n_samples': len(pred_all)}
    if per_sample_metrics:
        metrics['per_sample_sensor_metrics'] = {
            str(k): v for k, v in per_sample_metrics.items()
        }
    metrics_path = os.path.join(args.out_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics: MAE={overall['mae']:.2f}  RMSE={overall['rmse']:.2f}  "
          f"NRMSE={overall['nrmse']:.1f}%")
    print(f"All output written to {args.out_dir}")


if __name__ == '__main__':
    main()
