import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from itertools import product

# Don't remove these for now. Metadata doesn't save the sensor names, so for now we will hardcode it in 
# and I'll update the preprocess script later.
LA_BASIN_SENSORS = {
    'Reseda': (8, 3),
    'North Hollywood': (8, 11),
    'LA - N. Main Street': (15, 16),
    'Compton': (23, 17),
    'Long Beach Signal Hill': (29, 19),
    'Anaheim': (27, 29),
    'Glendora - Laurel': (10, 33),
}


def plot_training_history(history, save_path, config_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    train_loss = history['loss']
    val_loss = history['val_loss']
    best_val = min(val_loss)
    best_epoch = val_loss.index(best_val) + 1
    
    ax.plot(train_loss, label='Training Loss', lw=2, color='blue')
    ax.plot(val_loss, label='Validation Loss', lw=2, color='orange')
    ax.axhline(y=best_val, color='r', ls='--', alpha=0.5)
    ax.axvline(x=best_epoch-1, color='g', ls='--', alpha=0.3)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Training History - {config_name}', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    textstr = f'Best Val: {best_val:.4f}\nBest Epoch: {best_epoch}\nFinal Train: {train_loss[-1]:.4f}\nFinal Val: {val_loss[-1]:.4f}'
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=9, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_sample(x, pred, true, save_path, title):
    if x is None:
        _plot_sample_without_x(pred, true, save_path, title)
    else:
        _plot_sample_with_x(x, pred, true, save_path, title)


def _plot_sample_without_x(pred, true, save_path, title):
    n_frames = pred.shape[0]
    
    fig = plt.figure(figsize=(1.5*n_frames, 5))
    gs = GridSpec(3, n_frames+1, width_ratios=[1]*n_frames + [0.05], hspace=0.15, wspace=0.1)

    pred = np.squeeze(pred)
    true = np.squeeze(true)
    
    error = pred - true
    vmin, vmax = min(np.min(true), np.min(pred)), max(np.max(true), np.max(pred))
    max_err = np.max(np.abs(error))
    
    for frame in range(n_frames):
        ax = fig.add_subplot(gs[0, frame])
        im = ax.imshow(pred[frame], cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f't+{frame+1}')
        ax.axis('off')
        if frame == 0:
            ax.text(-0.2, 0.5, 'Pred', transform=ax.transAxes, fontweight='bold', va='center', ha='right')
        
        ax = fig.add_subplot(gs[1, frame])
        ax.imshow(true[frame], cmap='viridis', vmin=vmin, vmax=vmax)
        ax.axis('off')
        if frame == 0:
            ax.text(-0.2, 0.5, 'Truth', transform=ax.transAxes, fontweight='bold', va='center', ha='right')
        
        ax = fig.add_subplot(gs[2, frame])
        im_err = ax.imshow(error[frame], cmap='RdBu_r', vmin=-max_err, vmax=max_err)
        ax.axis('off')
        if frame == 0:
            ax.text(-0.2, 0.5, 'Error', transform=ax.transAxes, fontweight='bold', va='center', ha='right')
    
    fig.colorbar(im, cax=fig.add_subplot(gs[0:2, -1]), label='PM2.5')
    fig.colorbar(im_err, cax=fig.add_subplot(gs[2, -1]), label='Error')
    
    fig.suptitle(title, fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def _plot_sample_with_x(x, pred, true, save_path, title):
    n_frames = pred.shape[0]
    
    fig = plt.figure(figsize=(1.5*n_frames, 6.5))
    gs = GridSpec(4, n_frames+1, width_ratios=[1]*n_frames + [0.05], hspace=0.35, wspace=0.1)

    x = np.squeeze(x)
    pred = np.squeeze(pred)
    true = np.squeeze(true)
    
    error = pred - true
    vmin, vmax = min(np.min(x), np.min(pred), np.min(true)), max(np.max(x), np.max(pred), np.max(true))
    max_err = np.max(np.abs(error))
    
    for frame in range(n_frames):
        ax = fig.add_subplot(gs[0, frame])
        im = ax.imshow(x[frame], cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f't-{n_frames-frame-1}' if n_frames - frame - 1 > 0 else 't')
        ax.axis('off')
        if frame == 0:
            ax.text(-0.2, 0.5, 'Input', transform=ax.transAxes, fontweight='bold', va='center', ha='right')

        ax = fig.add_subplot(gs[1, frame])
        ax.imshow(pred[frame], cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f't+{frame+1}')
        ax.axis('off')
        if frame == 0:
            ax.text(-0.2, 0.5, 'Pred', transform=ax.transAxes, fontweight='bold', va='center', ha='right')
        
        ax = fig.add_subplot(gs[2, frame])
        ax.imshow(true[frame], cmap='viridis', vmin=vmin, vmax=vmax)
        ax.axis('off')
        if frame == 0:
            ax.text(-0.2, 0.5, 'Truth', transform=ax.transAxes, fontweight='bold', va='center', ha='right')
        
        ax = fig.add_subplot(gs[3, frame])
        im_err = ax.imshow(error[frame], cmap='RdBu_r', vmin=-max_err, vmax=max_err)
        ax.axis('off')
        if frame == 0:
            ax.text(-0.2, 0.5, 'Error', transform=ax.transAxes, fontweight='bold', va='center', ha='right')
    
    fig.colorbar(im, cax=fig.add_subplot(gs[0:3, -1]), label='PM2.5')
    fig.colorbar(im_err, cax=fig.add_subplot(gs[3, -1]), label='Error')
    
    fig.suptitle(title, fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_single_horizon(Y_pred, Y_true, save_path, horizon=1, n_samples=24, start_idx=None, sensors=None):
    if sensors is None:
        sensors = LA_BASIN_SENSORS
    
    pred = Y_pred[..., 0] if Y_pred.ndim == 5 else Y_pred
    true = Y_true[..., 0] if Y_true.ndim == 5 else Y_true
    
    max_start = len(pred) - n_samples
    if start_idx is None:
        start_idx = np.random.randint(0, max_start) if max_start > 0 else 0
    else:
        start_idx = min(start_idx, max_start) if max_start > 0 else 0
    
    pred_h = pred[start_idx:start_idx + n_samples, horizon-1, :, :]
    true_h = true[start_idx:start_idx + n_samples, horizon-1, :, :]
    
    samples = np.arange(n_samples)
    n_sensors = len(sensors)
    fig, axes = plt.subplots(n_sensors, 1, figsize=(14, 2.5 * n_sensors), sharex=True)
    
    if n_sensors == 1:
        axes = [axes]
    
    for idx, (name, (r, c)) in enumerate(sensors.items()):
        ax = axes[idx]
        pred_vals, true_vals = pred_h[:, r, c], true_h[:, r, c]
        
        ax.plot(samples, true_vals, 'b-', lw=1.5, label='Truth', alpha=0.8)
        ax.plot(samples, pred_vals, 'orange', ls='--', lw=1.5, label='Predicted', alpha=0.8)
        
        mae = np.mean(np.abs(pred_vals - true_vals))
        ax.set_ylabel('PM2.5', fontsize=10)
        ax.set_title(f'{name} (MAE: {mae:.2f})', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
    
    axes[-1].set_xlabel('Hour', fontsize=11)
    duration = "24 hours" if n_samples <= 24 else f"{n_samples // 24} days"
    plt.suptitle(f't+{horizon} Horizon Performance - {duration}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_full_forecast(Y_pred, Y_true, save_path, start_idx=None, sensors=None):
    if sensors is None:
        sensors = LA_BASIN_SENSORS
    
    pred = Y_pred[..., 0] if Y_pred.ndim == 5 else Y_pred
    true = Y_true[..., 0] if Y_true.ndim == 5 else Y_true
    
    if start_idx is None:
        start_idx = np.random.randint(0, len(pred))
    
    pred_sample, true_sample = pred[start_idx], true[start_idx]
    n_hours = pred_sample.shape[0]
    hours = np.arange(1, n_hours + 1)
    
    n_sensors = len(sensors)
    fig, axes = plt.subplots(n_sensors, 1, figsize=(14, 2.5 * n_sensors), sharex=True)
    
    if n_sensors == 1:
        axes = [axes]
    
    for idx, (name, (r, c)) in enumerate(sensors.items()):
        ax = axes[idx]
        pred_vals, true_vals = pred_sample[:, r, c], true_sample[:, r, c]
        
        ax.plot(hours, true_vals, 'b-', lw=1.5, label='Truth', alpha=0.8)
        ax.plot(hours, pred_vals, 'orange', ls='--', lw=1.5, label='Predicted', alpha=0.8)
        
        mae = np.mean(np.abs(pred_vals - true_vals))
        ax.set_ylabel('PM2.5', fontsize=10)
        ax.set_title(f'{name} (MAE: {mae:.2f})', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
    
    axes[-1].set_xlabel('Forecast Hour (t+1 to t+24)', fontsize=11)
    plt.suptitle(f'Full 24-Hour Forecast (Sample {start_idx})', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_best_worst_samples(Y_pred, Y_true, save_dir, n_samples=10, horizons=[1, 6, 12, 24], sensors=None):
    if sensors is None:
        sensors = LA_BASIN_SENSORS
    
    os.makedirs(f"{save_dir}/best_samples", exist_ok=True)
    os.makedirs(f"{save_dir}/worst_samples", exist_ok=True)
    
    sample_rmse = np.array([np.sqrt(np.mean((p - t)**2)) for p, t in zip(Y_pred, Y_true)])
    sorted_idx = np.argsort(sample_rmse)
    
    for rank, idx in enumerate(sorted_idx[:n_samples], 1):
        _save_sample_plots(Y_pred, Y_true, idx, sample_rmse[idx], 
                          f"{save_dir}/best_samples", f"best_{rank:02d}", 
                          f"Best #{rank}", horizons, sensors)
    
    for rank, idx in enumerate(sorted_idx[-n_samples:][::-1], 1):
        _save_sample_plots(Y_pred, Y_true, idx, sample_rmse[idx],
                          f"{save_dir}/worst_samples", f"worst_{rank:02d}",
                          f"Worst #{rank}", horizons, sensors)


def _save_sample_plots(Y_pred, Y_true, idx, rmse, save_dir, prefix, label, horizons, sensors):
    plot_sample(None, Y_pred[idx], Y_true[idx],
               f"{save_dir}/{prefix}_grid.png",
               f"{label} (RMSE: {rmse:.2f})")
    
    plot_full_forecast(Y_pred, Y_true,
                      f"{save_dir}/{prefix}_full_ts.png",
                      start_idx=idx, sensors=sensors)
    
    for h in horizons:
        plot_single_horizon(Y_pred, Y_true,
                           f"{save_dir}/{prefix}_t{h}_ts.png",
                           horizon=h, n_samples=24, start_idx=idx, sensors=sensors)


def create_region_masks(sensor_locations, dim=40, nhood_radius=2):
    def find_neighbors(sources, radius):
        n_hood = set(product(range(-radius, radius + 1), repeat=2))
        n_hood.discard((0, 0))
        neighbors = set()
        for x, y in sources:
            for a, b in n_hood:
                f, g = x + a, y + b
                if 0 <= f < dim and 0 <= g < dim:
                    neighbors.add((f, g))
        return neighbors
    
    sensor_mask = np.zeros((dim, dim), dtype=np.float32)
    for x, y in sensor_locations:
        sensor_mask[x, y] = 1.0
    
    nhood_mask = np.zeros((dim, dim), dtype=np.float32)
    for x, y in find_neighbors(set(sensor_locations), nhood_radius):
        nhood_mask[x, y] = 1.0
    
    return {
        'sensor': sensor_mask,
        'nhood': np.clip(sensor_mask + nhood_mask, 0, 1),
        'full': np.ones((dim, dim))
    }


def compute_hourly_nrmse(Y_pred, Y_true, region_masks, dim=40):
    def extract_region(y_true, y_pred, mask):
        y_true_flat = y_true.reshape(-1, dim, dim)
        y_pred_flat = y_pred.reshape(-1, dim, dim)
        idx = np.where(mask > 0)
        return y_true_flat[:, idx[0], idx[1]].flatten(), y_pred_flat[:, idx[0], idx[1]].flatten()
    
    pred = Y_pred[..., 0] if Y_pred.ndim == 5 else Y_pred
    true = Y_true[..., 0] if Y_true.ndim == 5 else Y_true
    n_hours = pred.shape[1]
    
    results = {}
    for region_name, mask in region_masks.items():
        hourly_nrmse = []
        for h in range(n_hours):
            y_true_h, y_pred_h = extract_region(true[:, h:h+1], pred[:, h:h+1], mask)
            rmse = np.sqrt(np.mean((y_pred_h - y_true_h)**2))
            nrmse = (rmse / (np.mean(y_true_h) + 1e-8)) * 100
            hourly_nrmse.append(nrmse)
        results[region_name] = hourly_nrmse
    
    return results


def plot_hourly_nrmse(hourly_nrmse, save_path, region_name):
    fig, ax = plt.subplots(figsize=(14, 6))
    
    hours = list(range(1, len(hourly_nrmse) + 1))
    avg = np.mean(hourly_nrmse)
    
    bars = ax.bar(hours, hourly_nrmse, color='steelblue', edgecolor='navy', lw=1.2, width=0.8)
    ax.axhline(y=avg, color='red', ls='--', alpha=0.7, lw=1.5)
    ax.text(1.5, avg + 1, f'Average: {avg:.2f}%', fontsize=11, color='red')
    
    for bar, val in zip(bars, hourly_nrmse):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Forecast Hour', fontsize=12)
    ax.set_ylabel('NRMSE (%)', fontsize=12)
    ax.set_title(f'{region_name.capitalize()} Region - Hourly NRMSE', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(hours)
    ax.set_xlim(0, len(hours) + 1)
    ax.set_ylim(0, max(hourly_nrmse) * 1.15)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_hourly_nrmse_combined(results, save_path):
    fig, axes = plt.subplots(3, 1, figsize=(14, 15))
    
    for idx, region_name in enumerate(['sensor', 'nhood', 'full']):
        ax = axes[idx]
        hourly_nrmse = results[region_name]
        hours = list(range(1, len(hourly_nrmse) + 1))
        avg = np.mean(hourly_nrmse)
        
        bars = ax.bar(hours, hourly_nrmse, color='steelblue', edgecolor='navy', lw=1.2, width=0.8)
        ax.axhline(y=avg, color='red', ls='--', alpha=0.7, lw=1.5)
        ax.text(1.5, avg + 1, f'Average: {avg:.2f}%', fontsize=11, color='red')
        
        for bar, val in zip(bars, hourly_nrmse):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Forecast Hour', fontsize=12)
        ax.set_ylabel('NRMSE (%)', fontsize=12)
        ax.set_title(f'{region_name.capitalize()} Region', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks(hours)
        ax.set_xlim(0, len(hours) + 1)
        ax.set_ylim(0, max(hourly_nrmse) * 1.15)
    
    plt.suptitle('Hourly NRMSE by Region', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_nrmse_plots(Y_pred, Y_true, sensor_locations, save_dir, dim=40):
    os.makedirs(save_dir, exist_ok=True)
    
    masks = create_region_masks(sensor_locations, dim)
    results = compute_hourly_nrmse(Y_pred, Y_true, masks, dim)
    
    for region_name in ['sensor', 'nhood', 'full']:
        plot_hourly_nrmse(results[region_name], f"{save_dir}/nrmse_{region_name}.png", region_name)
    
    plot_hourly_nrmse_combined(results, f"{save_dir}/nrmse_combined.png")
    
    return results