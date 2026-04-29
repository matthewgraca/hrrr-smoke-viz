#!/usr/bin/env python3
"""
Standalone visualization for spatio-temporal PM2.5 forecast predictions.

Inputs:
  --preds, --trues : (N, T, H, W) or (N, T, 1, H, W) numpy arrays in µg/m³
  --inputs         : (N, T, H, W, C) numpy array of model inputs — used to
                     render the last 5 hours of context alongside each
                     forecast panel. The PM2.5 input channel is un-scaled
                     for display using the stats in scalers.pkl found next
                     to the --inputs file.

Output: a directory of PNGs:
  sample_NNN_panel.png       — input + pred + truth grids per sample
  sample_NNN_combined.png    — same panels + grid-mean time series
  sample_NNN_timeseries.png  — grid-mean time series per sample
  full_timeseries.png        — overlap-averaged time series across all samples
  nrmse_hourly.png           — hourly NRMSE bar chart with overall metrics
  metrics.json               — overall metrics

Usage
-----
    python visualize_forecasts.py \
        --preds /path/to/preds.npy \
        --trues /path/to/trues.npy \
        --inputs /path/to/X_valid.npy \
        --out-dir reports/my_run/
"""

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
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
    """Accept (N, T, H, W) or (N, T, 1, H, W); return (N, T, H, W)."""
    if arr.ndim == 5 and arr.shape[2] == 1:
        return arr[:, :, 0, :, :]
    if arr.ndim == 4:
        return arr
    raise ValueError(f"Expected (N,T,H,W) or (N,T,1,H,W); got {arr.shape}")


def pick_sample_indices(truth_all, n_samples=5):
    """Pick top-by-mean and evenly-spaced sample indices."""
    n = len(truth_all)
    if n <= n_samples:
        return list(range(n))
    means = np.array([truth_all[i].mean() for i in range(n)])
    top_k = min(3, n_samples)
    top = set(means.argsort()[-top_k:][::-1].tolist())
    rest = n_samples - len(top)
    evenly = set(np.linspace(0, n - 1, rest + 4, dtype=int).tolist())
    candidates = sorted(top | evenly)
    if len(candidates) > n_samples:
        extras = sorted(evenly - top)
        keep = sorted(top | set(extras[:n_samples - len(top)]))
        candidates = keep
    return candidates[:n_samples]


def _grid_metrics(pred_ts, truth_ts):
    err = pred_ts - truth_ts
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    nrmse = rmse / (truth_ts.mean() + 1e-8) * 100
    r = (np.corrcoef(pred_ts, truth_ts)[0, 1]
         if np.std(truth_ts) > 0 else 0.0)
    return mae, rmse, nrmse, r



def save_sample_panel(pred_all, truth_all, X_inputs, input_mean, input_std,
                      input_channel, sample_idx, title, out_dir):
    """Spatial panel: 5 input hours + 24 pred / 24 truth hours."""
    pred_sample = pred_all[sample_idx]
    truth_sample = truth_all[sample_idx]

    n_cols = 29
    fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 1.5, 3.5))
    fig.suptitle(f'{title} — Sample {sample_idx}', fontsize=13,
                 fontweight='bold', y=1.02)

    for k in range(5):
        ih = 19 + k
        inp = np.array(X_inputs[sample_idx, ih, :, :, input_channel],
                       dtype=np.float32)
        inp = inp * input_std + input_mean
        axes[0, k].imshow(inp, cmap=PM25_CMAP, vmin=PM25_VMIN, vmax=PM25_VMAX)
        axes[0, k].set_title(f'In {ih+1}', fontsize=6)
        axes[0, k].axis('off')
        axes[1, k].axis('off')

    for h in range(24):
        col = 5 + h
        axes[0, col].imshow(pred_sample[h], cmap=PM25_CMAP,
                            vmin=PM25_VMIN, vmax=PM25_VMAX)
        axes[0, col].set_title(f'+{h+1}h\n{truth_sample[h].mean():.1f}',
                               fontsize=5, color='blue')
        axes[0, col].axis('off')
        axes[1, col].imshow(truth_sample[h], cmap=PM25_CMAP,
                            vmin=PM25_VMIN, vmax=PM25_VMAX)
        axes[1, col].axis('off')

    fig.subplots_adjust(hspace=0.08, wspace=0.05)
    path = os.path.join(out_dir, f'sample_{sample_idx:03d}_panel.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_sample_timeseries(pred_all, truth_all, sample_idx, title, out_dir):
    """Grid-mean time series for one sample."""
    pred_ts = pred_all[sample_idx].mean(axis=(1, 2))
    truth_ts = truth_all[sample_idx].mean(axis=(1, 2))
    hours = np.arange(1, 25)
    mae, rmse, nrmse, r = _grid_metrics(pred_ts, truth_ts)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hours, truth_ts, 'g-o', markersize=4, linewidth=2, label='Truth')
    ax.plot(hours, pred_ts, 'b-o', markersize=4, linewidth=2, label='Pred')
    ax.fill_between(hours, truth_ts, pred_ts, alpha=0.15, color='blue')
    ax.set_xlabel('Forecast Hour')
    ax.set_ylabel('PM2.5 (µg/m³)')
    ax.set_title(f'{title} — Sample {sample_idx} | '
                 f'NRMSE={nrmse:.1f}%  MAE={mae:.2f}  RMSE={rmse:.2f}  r={r:.3f}',
                 fontweight='bold')
    ax.legend()
    ax.set_xlim(1, 24)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(out_dir, f'sample_{sample_idx:03d}_timeseries.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_combined(pred_all, truth_all, X_inputs, input_mean, input_std,
                  input_channel, sample_idx, title, out_dir):
    """Combined panel: spatial frames + grid-mean time series in one figure."""
    fig = plt.figure(figsize=(26, 10))
    gs = GridSpec(2, 1, height_ratios=[1.0, 0.8], hspace=0.22, figure=fig)
    n_cols = 24
    inner = gs[0].subgridspec(3, n_cols, hspace=0.25, wspace=0.03)
    axes = np.empty((3, n_cols), dtype=object)
    for r in range(3):
        for c in range(n_cols):
            axes[r, c] = fig.add_subplot(inner[r, c])

    truth_sample = truth_all[sample_idx]
    pred_sample = pred_all[sample_idx]
    truth_hour_mean = truth_sample.mean(axis=(1, 2))

    for k in range(5):
        ih = 19 + k
        inp = np.array(X_inputs[sample_idx, ih, :, :, input_channel],
                       dtype=np.float32) * input_std + input_mean
        axes[0, k].imshow(inp, cmap=PM25_CMAP,
                          vmin=PM25_VMIN, vmax=PM25_VMAX)
        axes[0, k].set_title(f'In {ih+1}\n{inp.mean():.1f}', fontsize=6)
        axes[0, k].set_xticks([]); axes[0, k].set_yticks([])
    for k in range(5, n_cols):
        axes[0, k].axis('off')
    axes[0, 0].set_ylabel('Last 5 hours\nof input', fontsize=9,
                          fontweight='bold', rotation=0,
                          labelpad=55, va='center', ha='right')

    for ri, (label, data) in enumerate(
            zip(['PRED', 'TRUTH'], [pred_sample, truth_sample])):
        r = ri + 1
        for h in range(24):
            axes[r, h].imshow(data[h], cmap=PM25_CMAP,
                              vmin=PM25_VMIN, vmax=PM25_VMAX)
            if r == 1:
                axes[r, h].set_title(f'+{h+1}h\n{truth_hour_mean[h]:.1f}',
                                     fontsize=6)
            axes[r, h].set_xticks([]); axes[r, h].set_yticks([])
        axes[r, 0].set_ylabel(label, fontsize=9, fontweight='bold',
                              rotation=0, labelpad=55, va='center', ha='right')

    pred_ts = pred_sample.mean(axis=(1, 2))
    truth_ts = truth_sample.mean(axis=(1, 2))
    hours = np.arange(1, 25)
    mae, rmse, nrmse, r = _grid_metrics(pred_ts, truth_ts)

    all_vals = np.concatenate([truth_ts, pred_ts])
    pad = (all_vals.max() - all_vals.min()) * 0.12 + 0.5
    ylim = (all_vals.min() - pad, all_vals.max() + pad)

    ts_ax = fig.add_subplot(gs[1])
    ts_ax.plot(hours, truth_ts, 'g-o', markersize=4, linewidth=2, label='Truth')
    ts_ax.plot(hours, pred_ts, 'b-o', markersize=4, linewidth=2, label='Pred')
    ts_ax.fill_between(hours, truth_ts, pred_ts, alpha=0.15, color='blue')
    for h, (t, p) in enumerate(zip(truth_ts, pred_ts)):
        ts_ax.annotate(f'{t:.1f}', (hours[h], t), textcoords='offset points',
                       xytext=(0, -11), ha='center', fontsize=6.5,
                       color='darkgreen')
        ts_ax.annotate(f'{p:.1f}', (hours[h], p), textcoords='offset points',
                       xytext=(0, 6), ha='center', fontsize=6.5,
                       color='darkblue')
    ts_ax.set_xlim(0.5, 24.5); ts_ax.set_ylim(ylim)
    ts_ax.set_xlabel('Forecast Hour')
    ts_ax.set_ylabel('PM2.5 (µg/m³)')
    ts_ax.set_title(f'Grid-mean timeseries | NRMSE={nrmse:.1f}%  '
                    f'MAE={mae:.2f}  RMSE={rmse:.2f}  r={r:.3f}',
                    fontweight='bold')
    ts_ax.legend(loc='upper left')
    ts_ax.grid(True, alpha=0.3)

    errors = pred_sample - truth_sample
    px_nrmse = (float(np.sqrt((errors ** 2).mean())) /
                (truth_sample.mean() + 1e-8) * 100)
    fig.suptitle(f'{title} — Sample {sample_idx}  |  '
                 f'pixel NRMSE={px_nrmse:.1f}%',
                 fontsize=13, fontweight='bold', y=0.995)

    path = os.path.join(out_dir, f'sample_{sample_idx:03d}_combined.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def save_full_timeseries(pred_all, truth_all, title, out_dir):
    """Overlap-averaged grid-mean time series across all samples.

    Each sample contributes one value per forecast hour; samples i and i+1
    overlap in time, so the curves are averaged where they overlap.
    """
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
    nrmse = rmse / (avg_truth.mean() + 1e-8) * 100
    r = float(np.corrcoef(avg_pred, avg_truth)[0, 1])

    ax.set_xlabel('Hour offset from start', fontsize=12)
    ax.set_ylabel('PM2.5 (µg/m³)', fontsize=12)
    ax.set_title(f'{title} — Full Time Series ({n_valid} samples) | '
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


def save_nrmse_chart(pred_all, truth_all, title, out_dir):
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

    ax.set_xlabel('Forecast Hour (top)  /  Truth PM2.5 grid-mean µg/m³ (bottom)',
                  fontsize=11)
    ax.set_ylabel('NRMSE (%)', fontsize=11)
    ax.set_title(f'{title} — Hourly NRMSE ({len(pred_all)} samples) | '
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



def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument('--preds', required=True, help='Path to preds.npy')
    p.add_argument('--trues', required=True, help='Path to trues.npy')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--title', default='forecast',
                   help='Title prefix for plots (e.g. fire name or run id)')
    p.add_argument('--inputs', required=True,
                   help='Path to X_valid.npy for input panels (N,T,H,W,C). '
                        'scalers.pkl is auto-loaded from the same directory '
                        'to un-scale the PM2.5 input channel.')
    p.add_argument('--input-channel', type=int, default=0,
                   help='Channel index of PM2.5 in --inputs (default 0)')
    args = p.parse_args()

    # Auto-load scalers.pkl from the same directory as --inputs to recover
    # the un-scaling stats for the PM2.5 input channel.
    scalers_path = os.path.join(os.path.dirname(args.inputs), 'scalers.pkl')
    if not os.path.exists(scalers_path):
        sys.exit(f"scalers.pkl not found at {scalers_path}. The script needs "
                 f"it to un-scale the PM2.5 input channel for display.")
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)
    pm25_key = next((k for k in ('AirNow_PM25', 'OpenAQ_PM25') if k in scalers), None)
    if pm25_key is None:
        sys.exit(f"No PM2.5 scaler in {scalers_path} (looked for "
                 f"AirNow_PM25 / OpenAQ_PM25). Keys present: {list(scalers.keys())}")
    input_mean = float(scalers[pm25_key].mean_[0])
    input_std  = float(scalers[pm25_key].scale_[0])
    print(f"Auto-loaded scaler '{pm25_key}': mean={input_mean:.3f}, std={input_std:.3f}")

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

    print(f"\nWriting combined panel for all {len(pred_all)} samples...")
    for idx in range(len(pred_all)):
        save_combined(pred_all, truth_all, X_inputs,
                      input_mean, input_std,
                      args.input_channel,
                      idx, args.title, args.out_dir)

    print("Writing full time series...")
    full_metrics = save_full_timeseries(pred_all, truth_all,
                                        args.title, args.out_dir)

    print("Writing NRMSE chart...")
    overall = save_nrmse_chart(pred_all, truth_all, args.title, args.out_dir)

    metrics = {**overall, **full_metrics, 'n_samples': len(pred_all)}
    metrics_path = os.path.join(args.out_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics: {metrics}")
    print(f"\nAll output written to {args.out_dir}")


if __name__ == '__main__':
    main()
