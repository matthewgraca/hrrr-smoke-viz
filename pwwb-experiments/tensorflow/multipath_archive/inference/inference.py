#!/usr/bin/env python3
"""
Operational inference pipeline: generates payload and runs prediction.
"""

import os
import argparse
import subprocess
import sys
import numpy as np
import tensorflow as tf
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gc
import json

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


CONFIG = {
    'extent': (-118.615, -117.70, 33.60, 34.35),
    'dim': 84,
    'model_path': 'best_model.keras',
}


def generate_payload(payload_script='generate_payload.py', verbose=True):
    """Run payload generation script and return the payload path."""
    if verbose:
        print("\n" + "="*60)
        print("GENERATING OPERATIONAL PAYLOAD")
        print("="*60)
    
    result = subprocess.run(
        [sys.executable, payload_script],
        capture_output=not verbose,
        text=True
    )
    
    if result.returncode != 0:
        if not verbose:
            print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"Payload generation failed with code {result.returncode}")
    
    payload_path = 'data/operational/payload.npz'
    if not os.path.exists(payload_path):
        raise FileNotFoundError(f"Expected payload at {payload_path}")
    
    return payload_path


def load_payload(payload_path):
    """Load operational payload."""
    print(f"\nLoading payload from {payload_path}...")
    
    payload = np.load(payload_path, allow_pickle=True)
    X_input = payload['X_input']
    
    metadata = {
        'input_start': str(payload['input_start']),
        'input_end': str(payload['input_end']),
        'forecast_start': str(payload['forecast_start']),
        'forecast_end': str(payload['forecast_end']),
    }
    
    print(f"  X shape: {X_input.shape}")
    print(f"  Input window: {metadata['input_start']} to {metadata['input_end']}")
    print(f"  Forecast window: {metadata['forecast_start']} to {metadata['forecast_end']}")
    
    return X_input, metadata


def load_model(model_path):
    """Load trained model with custom objects."""
    print(f"\nLoading model from {model_path}...")
    model = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS)
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
    return model


def run_inference(model, X_input):
    """Run model prediction."""
    print(f"\nRunning inference...")
    Y_pred = model.predict(X_input, verbose=0)
    print(f"  Output shape: {Y_pred.shape}")
    print(f"  Predictions range: [{Y_pred.min():.2f}, {Y_pred.max():.2f}] µg/m³")
    return Y_pred


def generate_forecast_svgs(Y_pred, output_dir, metadata, config):
    """Generate 24 SVGs for the forecast."""
    os.makedirs(output_dir, exist_ok=True)
    
    predictions = Y_pred[0, :, :, :, 0]
    vmin, vmax = float(predictions.min()), float(predictions.max())
    
    print(f"\nGenerating forecast SVGs...")
    for hour in range(24):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(predictions[hour], origin='upper', cmap='viridis',
                  vmin=vmin, vmax=vmax, interpolation='bilinear')
        ax.axis('off')
        
        path = f"{output_dir}/forecast_t{hour+1:02d}.svg"
        plt.savefig(path, format='svg', bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"    ✓ t+{hour+1:02d}h")
    
    return output_dir


def generate_summary_plot(Y_pred, output_dir, metadata, config):
    """Generate a summary grid of all 24 forecast hours."""
    os.makedirs(output_dir, exist_ok=True)
    
    predictions = Y_pred[0, :, :, :, 0]
    extent = config['extent']
    lon_min, lon_max, lat_min, lat_max = extent
    
    fig, axes = plt.subplots(4, 6, figsize=(18, 12))
    axes = axes.flatten()
    
    vmin, vmax = predictions.min(), predictions.max()
    
    for h in range(24):
        ax = axes[h]
        im = ax.imshow(predictions[h], extent=[lon_min, lon_max, lat_min, lat_max],
                       origin='upper', cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f't+{h+1}h', fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
    
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='PM2.5 (µg/m³)')
    
    fig.suptitle(
        f'24-Hour PM2.5 Forecast\n'
        f'Init: {metadata["input_end"]} | Valid: {metadata["forecast_start"]} to {metadata["forecast_end"]}',
        fontsize=14, fontweight='bold'
    )
    
    save_path = f"{output_dir}/forecast_summary.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n  ✓ Saved summary plot to {save_path}")
    return save_path


def save_forecast(Y_pred, output_dir, metadata):
    """Save forecast arrays and metadata."""
    os.makedirs(output_dir, exist_ok=True)
    
    np.savez_compressed(
        f"{output_dir}/forecast.npz",
        predictions=Y_pred,
        **metadata
    )
    
    with open(f"{output_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  ✓ Saved forecast to {output_dir}/forecast.npz")


def main():
    parser = argparse.ArgumentParser(description='Operational PM2.5 forecast inference')
    parser.add_argument('--model', default=None, help='Override model path')
    parser.add_argument('--payload', default=None, 
                        help='Use existing payload instead of generating new one')
    parser.add_argument('--payload-script', default='generate_payload.py',
                        help='Path to payload generation script')
    parser.add_argument('--output-dir', default='forecasts/operational',
                        help='Output directory for forecasts')
    parser.add_argument('--skip-svgs', action='store_true',
                        help='Skip generating individual SVG files')
    args = parser.parse_args()
    
    model_path = args.model if args.model else CONFIG['model_path']
    
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M')
    output_dir = f"{args.output_dir}/{timestamp}"
    
    if args.payload:
        payload_path = args.payload
    else:
        payload_path = generate_payload(args.payload_script)
    
    X_input, metadata = load_payload(payload_path)
    
    model = load_model(model_path)
    
    Y_pred = run_inference(model, X_input)
    
    save_forecast(Y_pred, output_dir, metadata)
    generate_summary_plot(Y_pred, output_dir, metadata, CONFIG)
    
    if not args.skip_svgs:
        generate_forecast_svgs(Y_pred, f"{output_dir}/svgs", metadata, CONFIG)
    
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    
    print("\n" + "="*60)
    print("FORECAST COMPLETE")
    print("="*60)
    print(f"  Init time: {metadata['input_end']}")
    print(f"  Forecast: {metadata['forecast_start']} to {metadata['forecast_end']}")
    print(f"  Output: {output_dir}")
    print("="*60)
    
    return Y_pred, metadata


if __name__ == "__main__":
    main()