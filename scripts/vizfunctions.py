import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


SENSORS = {
    'Reseda': (8, 3),
    'North Holywood': (8, 11),
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
    
    textstr = f'Training Loss\nValidation Loss\nBest Val: {best_val:.4f}\nBest Epoch: {best_epoch}\nFinal Train: {train_loss[-1]:.4f}\nFinal Val: {val_loss[-1]:.4f}'
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=9, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_sample(pred, true, save_path, title):
    n_frames = pred.shape[0]
    
    fig = plt.figure(figsize=(1.5*n_frames, 5))
    gs = GridSpec(3, n_frames+1, width_ratios=[1]*n_frames + [0.05], hspace=0.15, wspace=0.1)
    
    if pred.ndim == 4:
        pred, true = pred[..., 0], true[..., 0]
    
    error = pred - true
    vmin, vmax = min(np.min(pred), np.min(true)), max(np.max(pred), np.max(true))
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


def plot_sensor_timeseries(Y_pred, Y_true, save_path, start_idx=0, n_samples=1):
    if n_samples is None:
        n_samples = len(Y_pred) - start_idx
    
    end_idx = min(start_idx + n_samples, len(Y_pred))
    
    pred = Y_pred[start_idx:end_idx]
    true = Y_true[start_idx:end_idx]
    
    if pred.ndim == 5:
        pred, true = pred[..., 0], true[..., 0]
    
    horizon = pred.shape[1]
    pred = pred.reshape(-1, pred.shape[2], pred.shape[3])
    true = true.reshape(-1, true.shape[2], true.shape[3])
    
    n_hours = pred.shape[0]
    hours = np.arange(1, n_hours + 1)
    
    n_sensors = len(SENSORS)
    fig, axes = plt.subplots(n_sensors, 1, figsize=(max(14, n_hours * 0.15), 2.5 * n_sensors), sharex=True)
    
    for idx, (name, (r, c)) in enumerate(SENSORS.items()):
        ax = axes[idx]
        
        pred_vals = pred[:, r, c]
        true_vals = true[:, r, c]
        
        ax.plot(hours, true_vals, 'b-', lw=1.5, label='Truth', alpha=0.8)
        ax.plot(hours, pred_vals, 'orange', ls='--', lw=1.5, label='Predicted', alpha=0.8)
        
        mae = np.mean(np.abs(pred_vals - true_vals))
        ax.set_ylabel('PM2.5', fontsize=10)
        ax.set_title(f'{name} (MAE: {mae:.2f})', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
    
    axes[-1].set_xlabel('Hour', fontsize=11)
    
    duration = f"{n_hours} hours ({end_idx - start_idx} samples)"
    plt.suptitle(f'Sensor Time Series - {duration}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_sensor_timeseries(Y_pred, Y_true, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    n_samples = len(Y_pred)
    
    idx = np.random.randint(0, n_samples)
    plot_sensor_timeseries(Y_pred, Y_true, f"{save_dir}/ts_24h.png", start_idx=idx, n_samples=1)
    
    if n_samples >= 7:
        idx = np.random.randint(0, n_samples - 7)
        plot_sensor_timeseries(Y_pred, Y_true, f"{save_dir}/ts_1week.png", start_idx=idx, n_samples=7)
    
    if n_samples >= 30:
        idx = np.random.randint(0, n_samples - 30)
        plot_sensor_timeseries(Y_pred, Y_true, f"{save_dir}/ts_1month.png", start_idx=idx, n_samples=30)
    
    plot_sensor_timeseries(Y_pred, Y_true, f"{save_dir}/ts_full.png", start_idx=0, n_samples=None)


def save_best_worst_samples(Y_pred, Y_test, save_dir, n_samples=10):
    os.makedirs(f"{save_dir}/best_samples", exist_ok=True)
    os.makedirs(f"{save_dir}/worst_samples", exist_ok=True)
    
    sample_rmse = np.array([np.sqrt(np.mean((Y_pred[i] - Y_test[i])**2)) for i in range(len(Y_pred))])
    sorted_idx = np.argsort(sample_rmse)
    
    for rank, idx in enumerate(sorted_idx[:n_samples], 1):
        plot_sample(Y_pred[idx], Y_test[idx],
                   f"{save_dir}/best_samples/best_{rank:02d}.png",
                   f"Best #{rank} (RMSE: {sample_rmse[idx]:.2f})")
    
    for rank, idx in enumerate(sorted_idx[-n_samples:][::-1], 1):
        plot_sample(Y_pred[idx], Y_test[idx],
                   f"{save_dir}/worst_samples/worst_{rank:02d}.png",
                   f"Worst #{rank} (RMSE: {sample_rmse[idx]:.2f})")