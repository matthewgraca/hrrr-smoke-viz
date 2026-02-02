import os
import argparse
import numpy as np
import pickle
import tensorflow as tf
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gc
import json
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')



def tile_for_decoder(x):
    return tf.tile(tf.expand_dims(x, axis=1), [1, 24, 1, 1, 1])

def create_pos_encoding(x):
    batch_size = tf.shape(x)[0]
    h = tf.shape(x)[2]
    w = tf.shape(x)[3]
    horizon = 24
    pos = tf.reshape(tf.range(horizon, dtype=tf.float32) / horizon, [1, horizon, 1, 1, 1])
    return tf.tile(pos, [batch_size, 1, h, w, 1])

CUSTOM_OBJECTS = {
    'tile_for_decoder': tile_for_decoder,
    'create_pos_encoding': create_pos_encoding,
}



CONFIGS = {
    '40x40': {
        'extent': (-118.615, -117.70, 33.60, 34.35),
        'dim': 40,
        'model_path': 'results/mlp_fusion/sq2sq-5x5-wide-no-holidays_airnow/model.keras',
        'data_dir': 'data/24out_no_holidays/npy_files',
    },
    '84x84': {
        'extent': (-118.615, -117.70, 33.60, 34.35),
        'dim': 84,
        'model_path': 'results/mlp_fusion/arch_7x7_84x84_airnow/best_model.keras',
        'data_dir': 'data/24out_84x84_no_holidays/npy_files',
    },
}

CONFIG = CONFIGS['40x40']



def load_full_test_set(data_dir):
    """Load the full test set."""
    print(f"Loading full test set from {data_dir}...")
    
    X = np.load(f"{data_dir}/X_test.npy")
    Y = np.load(f"{data_dir}/Y_test.npy")
    
    print(f"  X shape: {X.shape}")
    print(f"  Y shape: {Y.shape}")
    print(f"  Y range: [{Y.min():.2f}, {Y.max():.2f}]")
    
    return X, Y


def load_model(model_path):
    """Load trained model with custom objects."""
    print(f"\nLoading model from {model_path}...")
    model = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS)
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
    return model



def run_batch_inference(model, X, batch_size=8):
    """Run model prediction in batches."""
    print(f"\nRunning inference on {X.shape[0]} samples (batch_size={batch_size})...")
    
    Y_pred = []
    n_samples = X.shape[0]
    
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        batch_pred = model.predict(X[i:end], verbose=0)
        Y_pred.append(batch_pred)
        if (i // batch_size + 1) % 10 == 0 or end == n_samples:
            print(f"  Batch {i//batch_size + 1}/{(n_samples + batch_size - 1)//batch_size} done")
    
    Y_pred = np.concatenate(Y_pred, axis=0)
    print(f"  Output shape: {Y_pred.shape}")
    print(f"  Predictions range: [{Y_pred.min():.2f}, {Y_pred.max():.2f}]")
    
    return Y_pred


def compute_hourly_nrmse(Y_pred, Y_true, dim):
    """Compute hourly NRMSE for full grid."""
    pred = Y_pred[..., 0] if Y_pred.ndim == 5 else Y_pred
    true = Y_true[..., 0] if Y_true.ndim == 5 else Y_true
    
    n_hours = pred.shape[1]
    hourly_nrmse = []
    
    for h in range(n_hours):
        y_true_h = true[:, h].flatten()
        y_pred_h = pred[:, h].flatten()
        
        rmse = np.sqrt(np.mean((y_pred_h - y_true_h)**2))
        nrmse = (rmse / (np.mean(y_true_h) + 1e-8)) * 100
        hourly_nrmse.append(nrmse)
    
    return hourly_nrmse


def compute_full_metrics(Y_pred, Y_true):
    """Compute overall metrics for full grid."""
    pred = Y_pred[..., 0] if Y_pred.ndim == 5 else Y_pred
    true = Y_true[..., 0] if Y_true.ndim == 5 else Y_true
    
    pred_flat = pred.flatten()
    true_flat = true.flatten()
    
    mae = np.mean(np.abs(pred_flat - true_flat))
    mse = np.mean((pred_flat - true_flat)**2)
    rmse = np.sqrt(mse)
    nrmse = (rmse / (np.mean(true_flat) + 1e-8)) * 100
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'nrmse': nrmse
    }


def find_best_worst_samples(Y_pred, Y_true, n=10):
    """Find best and worst samples by RMSE."""
    sample_rmse = np.array([
        np.sqrt(np.mean((p - t)**2)) 
        for p, t in zip(Y_pred, Y_true)
    ])
    
    sorted_idx = np.argsort(sample_rmse)
    best_indices = sorted_idx[:n]
    worst_indices = sorted_idx[-n:][::-1]
    
    return best_indices, worst_indices, sample_rmse



def generate_svg(prediction, hour_idx, output_path, reference_time=None, vmin=0, vmax=None):
    """Generate a single forecast SVG — just the grid, no labels."""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    if vmax is None:
        vmax = prediction.max()
    
    ax.imshow(
        prediction,
        origin='upper',
        cmap='viridis',
        vmin=vmin,
        vmax=vmax,
        interpolation='bilinear'
    )
    
    ax.axis('off')
    
    plt.savefig(output_path, format='svg', bbox_inches='tight', pad_inches=0)
    plt.close()


def generate_hourly_svgs(Y_pred, output_dir, sample_idx, reference_time=None):
    """Generate 24 SVGs for a single sample's predictions (memory-safe)."""
    os.makedirs(output_dir, exist_ok=True)
    
    predictions = Y_pred[sample_idx, :, :, :, 0].copy()
    
    vmin = float(predictions.min())
    vmax = float(predictions.max())
    
    paths = []
    for hour in range(1, 25):
        pred = predictions[hour - 1]
        path = f"{output_dir}/forecast_t{hour:02d}.svg"
        generate_svg(pred, hour, path, reference_time, vmin=vmin, vmax=vmax)
        paths.append(path)
        print(f"    ✓ t+{hour:02d}h")
        plt.close('all')
        gc.collect()
    
    del predictions
    gc.collect()
    
    return paths


def plot_comparison_grid_both_resolutions(Y_pred_40, Y_true_40, Y_pred_84, Y_true_84, save_path, sample_idx):
    """Generate comparison grid: 40x40 pred, 40x40 GT, 84x84 pred, 84x84 GT for all 24 hours."""
    
    pred_40 = Y_pred_40[sample_idx, :, :, :, 0]
    true_40 = Y_true_40[sample_idx, :, :, :, 0]
    pred_84 = Y_pred_84[sample_idx, :, :, :, 0]
    true_84 = Y_true_84[sample_idx, :, :, :, 0]
    
    n_hours = 24
    n_rows = 4
    
    fig, axes = plt.subplots(n_rows, n_hours, figsize=(1.5 * n_hours, 1.5 * n_rows))
    
    vmin = min(pred_40.min(), true_40.min(), pred_84.min(), true_84.min())
    vmax = max(pred_40.max(), true_40.max(), pred_84.max(), true_84.max())
    
    for h in range(n_hours):
        axes[0, h].imshow(pred_40[h], cmap='viridis', vmin=vmin, vmax=vmax, interpolation='bilinear')
        axes[0, h].axis('off')
        
        axes[1, h].imshow(true_40[h], cmap='viridis', vmin=vmin, vmax=vmax, interpolation='bilinear')
        axes[1, h].axis('off')
        
        axes[2, h].imshow(pred_84[h], cmap='viridis', vmin=vmin, vmax=vmax, interpolation='bilinear')
        axes[2, h].axis('off')
        
        axes[3, h].imshow(true_84[h], cmap='viridis', vmin=vmin, vmax=vmax, interpolation='bilinear')
        axes[3, h].axis('off')
    
    plt.subplots_adjust(wspace=0.02, hspace=0.05)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    return save_path


def plot_hourly_nrmse(hourly_nrmse, save_path, title_suffix=""):
    """Generate bar chart of hourly NRMSE %."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    hours = list(range(1, len(hourly_nrmse) + 1))
    avg = np.mean(hourly_nrmse)
    
    bars = ax.bar(hours, hourly_nrmse, color='steelblue', edgecolor='navy', lw=1.2, width=0.8)
    ax.axhline(y=avg, color='red', ls='--', alpha=0.7, lw=1.5)
    ax.text(1.5, avg + 1, f'Average: {avg:.2f}%', fontsize=11, color='red', fontweight='bold')
    
    for bar, val in zip(bars, hourly_nrmse):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Forecast Hour', fontsize=12)
    ax.set_ylabel('NRMSE (%)', fontsize=12)
    ax.set_title(f'Hourly NRMSE (Full Grid){title_suffix}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(hours)
    ax.set_xlim(0, len(hours) + 1)
    ax.set_ylim(0, max(hourly_nrmse) * 1.15)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparison_grid(Y_pred, Y_true, save_path, sample_idx, reference_time=None):
    """Generate side-by-side comparison of prediction vs ground truth for all 24 hours."""
    if reference_time is None:
        reference_time = datetime.utcnow()
    
    pred = Y_pred[sample_idx, :, :, :, 0]
    true = Y_true[sample_idx, :, :, :, 0]
    
    n_hours = 24
    n_cols = 8
    n_rows = 3 * (n_hours // n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows))
    
    extent = CONFIG['extent']
    lon_min, lon_max, lat_min, lat_max = extent
    
    vmin = min(pred.min(), true.min())
    vmax = max(pred.max(), true.max())
    
    all_errors = pred - true
    max_err = max(abs(all_errors.min()), abs(all_errors.max()), 1)
    
    for hour in range(n_hours):
        col = hour % n_cols
        row_block = (hour // n_cols) * 3
        
        ax = axes[row_block, col]
        im = ax.imshow(pred[hour], extent=[lon_min, lon_max, lat_min, lat_max],
                       origin='upper', cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f't+{hour+1}h', fontsize=8, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel('Pred', fontsize=9, fontweight='bold')
        
        ax = axes[row_block + 1, col]
        ax.imshow(true[hour], extent=[lon_min, lon_max, lat_min, lat_max],
                  origin='upper', cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel('True', fontsize=9, fontweight='bold')
        
        ax = axes[row_block + 2, col]
        error = pred[hour] - true[hour]
        im_err = ax.imshow(error, extent=[lon_min, lon_max, lat_min, lat_max],
                          origin='upper', cmap='RdBu_r', vmin=-max_err, vmax=max_err)
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel('Error', fontsize=9, fontweight='bold')
        
        mae = np.mean(np.abs(error))
        nrmse = np.sqrt(np.mean(error**2)) / (np.mean(true[hour]) + 1e-8) * 100
        ax.set_xlabel(f'{mae:.1f} / {nrmse:.0f}%', fontsize=7)
    
    fig.subplots_adjust(right=0.92)
    cbar_ax1 = fig.add_axes([0.93, 0.4, 0.015, 0.5])
    fig.colorbar(im, cax=cbar_ax1, label='PM2.5 (µg/m³)')
    cbar_ax2 = fig.add_axes([0.93, 0.08, 0.015, 0.25])
    fig.colorbar(im_err, cax=cbar_ax2, label='Error')
    
    fig.suptitle(f'Forecast vs Ground Truth - All 24 Hours (Sample {sample_idx})\n'
                 f'Reference: {reference_time.strftime("%Y-%m-%d %H:%M UTC")} | Labels: MAE / NRMSE%',
                 fontsize=12, fontweight='bold')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()



def evaluate(model, X_test, Y_test, output_dir, batch_size=8):
    """
    Run evaluation: inference on test data + compute metrics.
    
    No scaling/unscaling needed because:
    - Y_test is raw µg/m³ (never scaled in preprocessing)
    - Model was trained to output raw µg/m³ (loss computed against unscaled Y)
    - So Y_pred is already in µg/m³
    """
    print(f"\n{'='*70}")
    print("EVALUATION")
    print(f"{'='*70}")
    
    os.makedirs(output_dir, exist_ok=True)
    dim = CONFIG['dim']
    
    Y_pred = run_batch_inference(model, X_test, batch_size)
    
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    
    overall = compute_full_metrics(Y_pred, Y_test)
    hourly_nrmse = compute_hourly_nrmse(Y_pred, Y_test, dim)
    
    print(f"\n  Samples: {X_test.shape[0]}")
    print(f"  Full Grid Metrics:")
    print(f"    MAE:   {overall['mae']:.2f} µg/m³")
    print(f"    MSE:   {overall['mse']:.2f} (µg/m³)²")
    print(f"    RMSE:  {overall['rmse']:.2f} µg/m³")
    print(f"    NRMSE: {overall['nrmse']:.2f}%")
    
    print(f"\n  Hourly NRMSE (%):")
    for h, nrmse in enumerate(hourly_nrmse):
        print(f"    t+{h+1:02d}h: {nrmse:.2f}%")
    print(f"    Average: {np.mean(hourly_nrmse):.2f}%")
    
    best_indices, worst_indices, sample_rmse = find_best_worst_samples(Y_pred, Y_test, n=10)
    
    print(f"\n  Best sample:  idx={best_indices[0]}, RMSE={sample_rmse[best_indices[0]]:.2f} µg/m³")
    print(f"  Worst sample: idx={worst_indices[0]}, RMSE={sample_rmse[worst_indices[0]]:.2f} µg/m³")
    
    plot_hourly_nrmse(
        hourly_nrmse, 
        f"{output_dir}/nrmse_full_grid.png",
        title_suffix=f"\nFull Test Set (n={X_test.shape[0]})"
    )
    plt.close('all')
    gc.collect()
    print(f"\n  ✓ Saved NRMSE plot")
    
    reference_time = datetime.utcnow()
    plot_comparison_grid(Y_pred, Y_test, f"{output_dir}/best_sample_comparison.png", 
                        best_indices[0], reference_time)
    plt.close('all')
    gc.collect()
    print(f"  ✓ Saved best sample comparison grid")
    
    best_svg_dir = f"{output_dir}/best_sample_svgs"
    print(f"\n  Generating 24 SVGs for best sample (idx={best_indices[0]})...")
    generate_hourly_svgs(Y_pred, best_svg_dir, best_indices[0], reference_time)
    print(f"  ✓ Saved 24 SVGs to {best_svg_dir}")
    
    metrics = {
        'n_samples': int(X_test.shape[0]),
        'full_mae': float(overall['mae']),
        'full_mse': float(overall['mse']),
        'full_rmse': float(overall['rmse']),
        'full_nrmse': float(overall['nrmse']),
        'hourly_nrmse': [float(x) for x in hourly_nrmse],
        'avg_hourly_nrmse': float(np.mean(hourly_nrmse)),
        'best_idx': int(best_indices[0]),
        'best_rmse': float(sample_rmse[best_indices[0]]),
        'worst_idx': int(worst_indices[0]),
        'worst_rmse': float(sample_rmse[worst_indices[0]]),
    }
    
    with open(f"{output_dir}/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    with open(f"{output_dir}/metrics.pkl", 'wb') as f:
        pickle.dump(metrics, f)
    
    print(f"  ✓ Saved metrics to {output_dir}/metrics.json")
    
    return Y_pred, Y_test, metrics



def main():
    parser = argparse.ArgumentParser(description='Test inference and evaluation')
    parser.add_argument('--config', choices=['40x40', '84x84', 'both'], default='both',
                        help='Which config to run: 40x40, 84x84, or both')
    parser.add_argument('--model', default=None, help='Override model path')
    parser.add_argument('--data-dir', default=None, help='Override data directory')
    parser.add_argument('--output-dir', default='forecasts/eval/', help='Base output directory')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--sample-idx', type=int, default=None, 
                        help='Generate comparison grid for specific sample index')
    parser.add_argument('--compare-resolutions', action='store_true',
                        help='Generate side-by-side comparison of 40x40 and 84x84')
    args = parser.parse_args()
    
    if args.compare_resolutions:
        args.config = 'both'
    
    configs_to_run = ['40x40', '84x84'] if args.config == 'both' else [args.config]
    
    all_metrics = {}
    all_predictions = {}
    all_ground_truth = {}
    
    for config_name in configs_to_run:
        print("\n" + "="*70)
        print(f"EVALUATING: {config_name}")
        print("="*70)
        
        global CONFIG
        CONFIG = CONFIGS[config_name]
        
        model_path = args.model if args.model else CONFIG['model_path']
        data_dir = args.data_dir if args.data_dir else CONFIG['data_dir']
        output_dir = f"{args.output_dir}/{config_name}"
        
        X_test, Y_test = load_full_test_set(data_dir)
        model = load_model(model_path)
        
        Y_pred, Y_test, metrics = evaluate(
            model, X_test, Y_test,
            output_dir, args.batch_size
        )
        
        all_metrics[config_name] = metrics
        
        if args.compare_resolutions:
            all_predictions[config_name] = Y_pred
            all_ground_truth[config_name] = Y_test
        
        del model, X_test
        tf.keras.backend.clear_session()
        gc.collect()
        
        print(f"\n  ✓ {config_name} complete: Avg Hourly NRMSE = {metrics['avg_hourly_nrmse']:.2f}%")
    
    if args.compare_resolutions and args.sample_idx is not None:
        sample_idx = args.sample_idx
        print(f"\n  Generating resolution comparison for sample {sample_idx}...")
        
        os.makedirs(f"{args.output_dir}/comparison", exist_ok=True)
        
        plot_comparison_grid_both_resolutions(
            all_predictions['40x40'], all_ground_truth['40x40'],
            all_predictions['84x84'], all_ground_truth['84x84'],
            f"{args.output_dir}/comparison/sample_{sample_idx}_both_resolutions.png",
            sample_idx
        )
        print(f"  ✓ Saved to {args.output_dir}/comparison/sample_{sample_idx}_both_resolutions.png")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for config_name, metrics in all_metrics.items():
        print(f"  {config_name}: Avg Hourly NRMSE = {metrics['avg_hourly_nrmse']:.2f}%")
    print("="*70)
    
    return all_metrics


if __name__ == "__main__":
    main()