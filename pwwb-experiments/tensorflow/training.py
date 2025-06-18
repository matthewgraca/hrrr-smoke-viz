# =============================================================================
# CONFIGURATION - EDIT THESE PATHS AS NEEDED
# =============================================================================
BASE_DATA_DIR = "final_input_data/two_years_pwwb_airnow_hrrr"
LIBS_PATH = "../.."
OUTPUT_BASE_DIR = "experiment_output"
TENSORBOARD_DIR = "my_logs"

CHANNEL_FILE_PATTERN = "{channel}_X_{split}.npy"
TARGET_FILE_PATTERN = "Y_{split}.npy"

DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 8

FORCE_CPU = False

import matplotlib
matplotlib.use('Agg')
# =============================================================================

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time

import tensorflow as tf

if FORCE_CPU:
    tf.config.set_visible_devices([], 'GPU')
    print("Using CPU (FORCE_CPU=True)")
else:
    print("Using TensorFlow default device selection (GPU if available)")

import keras
from keras.models import Sequential
from keras.layers import (Conv3D, ConvLSTM2D,
                          Flatten,
                         TimeDistributed, Dropout, Dense, InputLayer)
from keras.callbacks import EarlyStopping, TensorBoard

sys.path.append(LIBS_PATH)

class GradientLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, data_sample):
        super().__init__()
        self.writer = tf.summary.create_file_writer(log_dir)
        self.data_sample = data_sample

    def on_epoch_end(self, epoch, logs=None):
        x, y = self.data_sample
        with tf.GradientTape() as tape:
            loss = self.model.compute_loss(
                y=y, 
                y_pred=self.model(x, training=True)
            )
        gradients = tape.gradient(loss, self.model.trainable_weights)
        grad_norm = tf.linalg.global_norm(gradients)

        with self.writer.as_default():
            tf.summary.scalar("gradient_norm/global", grad_norm, step=epoch)
            for weight, grad in zip(self.model.trainable_weights, gradients):
                if grad is not None:
                    tf.summary.histogram(f"gradients/{weight.name}", grad, step=epoch)
            self.writer.flush()

def get_run_logdir(root_logdir, run_id):
    return os.path.join(root_logdir, run_id)

def create_classic_pwwb_model(input_shape, output_size):
    model = Sequential()
    model.add(InputLayer(shape=input_shape))
    
    model.add(ConvLSTM2D(
        filters=15, 
        kernel_size=(3, 3),
        padding='same', 
        return_sequences=True
    ))
    
    model.add(ConvLSTM2D(
        filters=30, 
        kernel_size=(3, 3),
        padding='same', 
        return_sequences=True
    ))
    
    model.add(Conv3D(
        filters=15, 
        kernel_size=(3, 3, 3),
        activation='relu',
        padding='same',
        data_format='channels_last'
    ))
    
    model.add(Conv3D(
        filters=1, 
        kernel_size=(3, 3, 3),
        activation='relu',
        padding='same',
        data_format='channels_last'
    ))
    
    model.add(TimeDistributed(Flatten()))
    model.add(Dense(output_size, activation='relu'))
    
    return model

def create_current_pwwb_model(input_shape, output_size):
    model = Sequential()
    model.add(InputLayer(shape=input_shape))
    
    model.add(ConvLSTM2D(
        filters=15, 
        kernel_size=(3, 3),
        padding='same', 
        return_sequences=True,
        dropout=0.6,
        recurrent_dropout=0.6,
        kernel_regularizer=keras.regularizers.l2(0.01)
    ))
    
    model.add(ConvLSTM2D(
        filters=30, 
        kernel_size=(3, 3),
        padding='same', 
        return_sequences=True,
        dropout=0.6,
        recurrent_dropout=0.6,
        kernel_regularizer=keras.regularizers.l2(0.01)
    ))
    
    model.add(Conv3D(
        filters=15, 
        kernel_size=(3, 3, 3),
        activation='relu',
        padding='same'
    ))
    
    model.add(Conv3D(
        filters=1, 
        kernel_size=(3, 3, 3),
        activation='relu',
        padding='same'
    ))
    
    model.add(TimeDistributed(Flatten()))
    model.add(Dropout(rate=0.6))
    model.add(Dense(output_size, activation='relu'))
    
    return model

def load_data_splits(channel_indices):
    """Load pre-split train/valid/test data"""
    all_channel_names = ['MAIAC_AOD', 'TROPOMI_NO2', 'METAR_Wind_U', 'METAR_Wind_V', 'AirNow_PM25', 'HRRR_COLMD']
    
    selected_channels = [all_channel_names[i] for i in channel_indices]
    print(f"Loading pre-split dataset for channels: {selected_channels}")
    
    splits = {}
    for split in ['train', 'valid', 'test']:
        channel_files = [CHANNEL_FILE_PATTERN.format(channel=all_channel_names[i], split=split) 
                        for i in channel_indices]
        X_data = np.stack([np.load(f"{BASE_DATA_DIR}/{file}") for file in channel_files], axis=-1)
        
        Y_data = np.load(f"{BASE_DATA_DIR}/{TARGET_FILE_PATTERN.format(split=split)}")
        
        splits[split] = (X_data, Y_data)
        print(f"  {split.capitalize()}: X={X_data.shape}, Y={Y_data.shape}")
    
    return splits['train'], splits['valid'], splits['test']

def train_model(channels, architecture, experiment_id, run_dir=None, epochs=None, batch_size=None):
    epochs = epochs or DEFAULT_EPOCHS
    batch_size = batch_size or DEFAULT_BATCH_SIZE
    
    all_channel_names = ['MAIAC_AOD', 'TROPOMI_NO2', 'METAR_Wind_U', 'METAR_Wind_V', 'AirNow_PM25', 'HRRR_COLMD']
    channel_indices = channels
    selected_channels = [all_channel_names[i] for i in channel_indices]
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {experiment_id}")
    print(f"Channels: {selected_channels}")
    print(f"Architecture: {architecture}")
    print(f"{'='*60}")
    
    (X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test) = load_data_splits(channel_indices)
    
    input_shape = X_test.shape[1:]
    _, n_frames, output_size = Y_test.shape
    
    print(f"Input shape: {input_shape}")
    print(f"Output shape: ({n_frames}, {output_size})")
    
    tf.keras.backend.set_image_data_format('channels_last')
    
    if architecture == 'classic':
        model = create_classic_pwwb_model(input_shape, output_size)
    elif architecture == 'current':
        model = create_current_pwwb_model(input_shape, output_size)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    if architecture == 'classic':
        model.compile(
            loss='mean_absolute_error', 
            optimizer='adam'
        )
    elif architecture == 'current':
        model.compile(
            loss='mean_absolute_error', 
            optimizer=keras.optimizers.Adam(learning_rate=0.0001, weight_decay=0.01)
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    if run_dir:
        output_dir = f"{run_dir}/{experiment_id}"
    else:
        output_dir = f"{OUTPUT_BASE_DIR}/{experiment_id}"
    
    os.makedirs(output_dir, exist_ok=True)

    if run_dir:
        run_logdir = get_run_logdir(
            root_logdir=os.path.join(run_dir, "tensorboard_logs"),
            run_id=experiment_id
        )
    else:
        run_logdir = get_run_logdir(
            root_logdir=os.path.join(os.curdir, TENSORBOARD_DIR),
            run_id=experiment_id
        )
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20),
        TensorBoard(run_logdir, histogram_freq=1),
        GradientLogger(run_logdir, data_sample=(X_test[:32], Y_test[:32]))
    ]
    
    print(f"Training with {X_train.shape[0]} samples")
    print("Starting training...")
    start_time = time.time()
    
    history = model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_valid, Y_valid),
        shuffle=True,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
    
    print("Evaluating model...")
    y_pred = model.predict(X_test, verbose=0)
    
    final_val_loss = min(history.history['val_loss'])
    
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    np.save(os.path.join(results_dir, "y_pred.npy"), y_pred)
    np.save(os.path.join(results_dir, "Y_test.npy"), Y_test)
    
    model.save(os.path.join(results_dir, "model.keras"))
    
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title(f'{experiment_id}\nTraining Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MAE)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_dir, "training_curves.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    sensor_names = [
        'Simi Valley - Cochran Street', 'Reseda', 'Santa Clarita', 'North Holywood', 
        'Los Angeles - N. Main Street', 'Compton', 'Long Beach Signal Hill', 'Glendora - Laurel'
    ]
    
    from libs.plotting import (
        plot_frame_by_frame_rmse,
        plot_avg_rmse_per_station, 
        plot_frame_heatmap,
        plot_frame_time_series,
        plot_frame_scatter,
        print_summary_table,
        print_detailed_frame_stats
    )
    
    plots_dir = os.path.join(results_dir, "plots")
    frame_plots_dir = os.path.join(plots_dir, "frame_analysis")
    scatter_plots_dir = os.path.join(plots_dir, "scatter_plots")
    time_series_dir = os.path.join(plots_dir, "time_series")
    
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(frame_plots_dir, exist_ok=True) 
    os.makedirs(scatter_plots_dir, exist_ok=True)
    os.makedirs(time_series_dir, exist_ok=True)
    
    current_plot_dir = plots_dir
    current_plot_name = "plot"
    
    def save_plot_with_name():
        plt.savefig(os.path.join(current_plot_dir, f"{current_plot_name}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    original_show = plt.show
    plt.show = save_plot_with_name
    
    try:
        print("Generating comprehensive analysis plots...")
        current_plot_dir = frame_plots_dir
        current_plot_name = "frame_by_frame_rmse"
        plot_frame_by_frame_rmse(y_pred, Y_test)
        
        current_plot_dir = plots_dir
        current_plot_name = "avg_rmse_per_station"
        plot_avg_rmse_per_station(y_pred, Y_test, sensor_names)
        
        current_plot_dir = plots_dir
        current_plot_name = "rmse_heatmap"
        plot_frame_heatmap(y_pred, Y_test, sensor_names)
        
        current_plot_dir = time_series_dir
        for frame_idx in range(Y_test.shape[1]):
            current_plot_name = f"time_series_frame_{frame_idx+1:02d}"
            plot_frame_time_series(y_pred, Y_test, sensor_names, frame_idx)
        
        current_plot_dir = scatter_plots_dir
        for frame_idx in range(Y_test.shape[1]):
            current_plot_name = f"scatter_frame_{frame_idx+1:02d}"
            plot_frame_scatter(y_pred, Y_test, frame_idx)
        
        print("✓ Generated organized analysis plots")
        
    finally:
        plt.show = original_show
    
    import io
    from contextlib import redirect_stdout
    
    analysis_output = io.StringIO()
    with redirect_stdout(analysis_output):
        print_summary_table(y_pred, Y_test, sensor_names)
        print_detailed_frame_stats(y_pred, Y_test, sensor_names)
    
    with open(os.path.join(results_dir, "comprehensive_analysis.txt"), 'w') as f:
        f.write(analysis_output.getvalue())
    
    metadata = {
        'experiment_id': experiment_id,
        'channels': channel_indices,
        'channel_names': selected_channels,
        'architecture': architecture,
        'final_validation_loss': float(final_val_loss),
        'training_time_seconds': float(training_time),
        'training_time_formatted': f"{int(training_time//60)}m {int(training_time%60)}s",
        'epochs_trained': len(history.history['loss']),
        'batch_size': batch_size,
        'timestamp': datetime.now().isoformat(),
        'framework': 'tensorflow'
    }
    
    with open(os.path.join(results_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Results saved to: {results_dir}")
    print(f"Final validation loss: {final_val_loss:.6f}")
    
    return final_val_loss, results_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train air quality prediction model')
    parser.add_argument('--channels', type=int, nargs='+', required=True,
                       help='Channel indices to use (0=AOD, 1=NO2, 2=Wind_U, 3=Wind_V, 4=AirNow, 5=HRRR)')
    parser.add_argument('--architecture', choices=['classic', 'current'], default='classic',
                       help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=None,
                       help=f'Number of training epochs (default: {DEFAULT_EPOCHS})')
    parser.add_argument('--batch-size', type=int, default=None,
                       help=f'Batch size for training (default: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--experiment-id', type=str, required=True,
                       help='Unique experiment identifier')
    parser.add_argument('--run-dir', type=str, default=None,
                       help='Parent directory for grouped experiments')
    
    args = parser.parse_args()
    
    print(f"Using data directory: {BASE_DATA_DIR}")
    print(f"Using libs path: {LIBS_PATH}")
    
    final_loss, output_dir = train_model(
        channels=args.channels,
        architecture=args.architecture,
        experiment_id=args.experiment_id,
        run_dir=args.run_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    print(f"\nExperiment {args.experiment_id} completed!")
    print(f"Final validation loss: {final_loss:.6f}")
    print(f"Results: {output_dir}")