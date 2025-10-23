import os
import numpy as np
import pandas as pd
import tensorflow as tf
import gc
import warnings
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys
sys.path.append('/home/mgraca/Workspace/hrrr-smoke-viz')
from libs.pwwb.utils.dataset import PWWBPyDataset
from keras.utils import plot_model

warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

class PM25TrainingPipeline:
    
    def __init__(self, experiment_config):
        self.config = experiment_config
        self.dim = 40
        self.sensor_locations = None
        self.base_hour = 0
        self.detailed_results = []
        
        self._configure_gpu()
        
    def _configure_gpu(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                print(f"✓ GPU configured: {len(gpus)} device(s)")
            except:
                print("GPU configuration failed")
    
    def load_data(self):
        if 'data_cache' in self.config:
            cache_path = self.config['data_cache']
            if cache_path.endswith('/npy_files'):
                npy_dir = cache_path
            else:
                npy_dir = f"{cache_path}/npy_files"
            print(f"Loading memory-mapped data from custom cache: {cache_path}")
        else:
            horizon = self.config['forecast_horizon']
            folder_name = f"{horizon}in_{horizon}out_metar_temporal4_satellites"
            npy_dir = f"data/shared/preprocessed_cache/{folder_name}/npy_files"
            print(f"Loading memory-mapped data from default cache: {folder_name}")
        
        if not os.path.exists(npy_dir):
            raise FileNotFoundError(f"Data directory not found: {npy_dir}")
        
        data = {
            'X_train': np.load(f"{npy_dir}/X_train.npy", mmap_mode='r'),
            'X_valid': np.load(f"{npy_dir}/X_valid.npy", mmap_mode='r'),
            'X_test': np.load(f"{npy_dir}/X_test.npy", mmap_mode='r'),
            'Y_train': np.load(f"{npy_dir}/Y_grid_train.npy", mmap_mode='r'),
            'Y_valid': np.load(f"{npy_dir}/Y_grid_valid.npy", mmap_mode='r'),
            'Y_test': np.load(f"{npy_dir}/Y_grid_test.npy", mmap_mode='r')
        }
        
        # this negates the use of mmap_mode='r'
        '''
        data['Y_train_orig'] = data['Y_train'].copy()
        data['Y_valid_orig'] = data['Y_valid'].copy()
        data['Y_test_orig'] = data['Y_test'].copy()
        '''
        
        import pickle
        metadata_path = f"{npy_dir}/../metadata.pkl"
        if not os.path.exists(metadata_path):
            parent_dir = os.path.dirname(npy_dir)
            metadata_path = f"{parent_dir}/metadata.pkl"
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.sensor_locations = metadata['sensor_locations']
        self.base_hour = metadata.get('base_hour', 0)
        data['channel_names'] = metadata['channel_names']
        data['metadata'] = metadata
        
        print(f"Data loaded: {data['X_train'].shape}")
        print(f"Input horizon: {data['X_train'].shape[1]} hours")
        print(f"Output horizon: {data['Y_train'].shape[1]} hours")
        print(f"Channels: {', '.join(data['channel_names'])}")
        return data
    
    def visualize_input_samples(self, data, save_dir, sample_indices=[800, 1000, 1232]):
        os.makedirs(f"{save_dir}/inputs", exist_ok=True)
        
        max_idx = len(data['X_train']) - 1
        adjusted_indices = [min(idx, max_idx) for idx in sample_indices]
        
        for sample_idx in adjusted_indices:
            self._visualize_single_input(data, sample_idx, f"{save_dir}/inputs")
    
    def _visualize_single_input(self, data, sample_idx, save_dir):
        X_data = data['X_train']
        Y_data = data['Y_train']
        
        if sample_idx >= len(X_data):
            sample_idx = len(X_data) - 1
        
        input_sample = X_data[sample_idx]
        target_sample = Y_data[sample_idx]
        
        n_frames = min(6, input_sample.shape[0])
        channel_names = data['channel_names']
        
        channel_mapping = {
            'AirNow PM2.5': 0,
            'OpenAQ PM2.5': 1,
            'METAR Wind U': 2,
            'METAR Wind V': 3,
            'METAR Wind Speed': 4,
            'NDVI': 5,
            'Elevation': 6,
            'Month Sin': 7,
            'Month Cos': 8,
            'Hour Sin': 9,
            'Hour Cos': 10,
            'GOES': 11,
            'TEMPO': 12
        }
        
        n_channels = len(channel_mapping)
        
        fig = plt.figure(figsize=(4*n_frames, 3*(n_channels+1)))
        gs = GridSpec(n_channels+1, n_frames+1, 
                     width_ratios=[1]*n_frames + [0.05],
                     hspace=0.3, wspace=0.15)
        
        for row_idx, (ch_name, ch_idx) in enumerate(channel_mapping.items()):
            if 'Wind U' in ch_name or 'Wind V' in ch_name:
                cmap = 'viridis'
            elif 'Sin' in ch_name or 'Cos' in ch_name:
                cmap = 'viridis'
            elif 'Elevation' in ch_name:
                cmap = 'viridis'
            elif 'NDVI' in ch_name:
                cmap = 'viridis'
            else:
                cmap = 'viridis'
            
            channel_data = input_sample[:, :, :, ch_idx]
            vmin, vmax = np.percentile(channel_data, [1, 99])
            
            if 'Wind U' in ch_name or 'Wind V' in ch_name:
                vmax = max(abs(vmin), abs(vmax))
                vmin = -vmax
            
            for col_idx in range(n_frames):
                ax = fig.add_subplot(gs[row_idx, col_idx])
                im = ax.imshow(input_sample[col_idx, :, :, ch_idx],
                              cmap=cmap, vmin=vmin, vmax=vmax)
                
                if row_idx == 0:
                    ax.set_title(f't={col_idx+1}', fontsize=10)
                if col_idx == 0:
                    ax.set_ylabel(ch_name, fontsize=9, fontweight='bold')
                
                ax.set_xticks([])
                ax.set_yticks([])
                
                if col_idx == n_frames-1:
                    cbar_ax = fig.add_subplot(gs[row_idx, -1])
                    plt.colorbar(im, cax=cbar_ax)
        
        target_vmin, target_vmax = np.percentile(target_sample, [1, 99])
        for col_idx in range(target_sample.shape[0]):
            ax = fig.add_subplot(gs[n_channels, col_idx])
            im = ax.imshow(target_sample[col_idx, :, :, 0],
                          cmap='viridis', vmin=target_vmin, vmax=target_vmax)
            if col_idx == 0:
                ax.set_ylabel('Target PM2.5', fontsize=9, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            
            if col_idx == n_frames-1:
                cbar_ax = fig.add_subplot(gs[n_channels, -1])
                plt.colorbar(im, cax=cbar_ax, label='μg/m³')
        
        fig.suptitle(f'Sample {sample_idx} - All {len(channel_mapping)} Input Channels + Target',
                    fontsize=14, fontweight='bold')
        
        plt.savefig(f"{save_dir}/sample_{sample_idx}_all_channels.png", 
                   dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Input visualization saved for sample {sample_idx}")
    
    def save_best_worst_samples(self, Y_pred, Y_test, save_dir, n_samples=10):
        os.makedirs(f"{save_dir}/best_samples", exist_ok=True)
        os.makedirs(f"{save_dir}/worst_samples", exist_ok=True)
        
        n_total = Y_pred.shape[0]
        sample_rmse = np.zeros(n_total)
        
        for i in range(n_total):
            sample_rmse[i] = np.sqrt(np.mean((Y_pred[i] - Y_test[i])**2))
        
        sorted_idx = np.argsort(sample_rmse)
        best_idx = sorted_idx[:n_samples]
        worst_idx = sorted_idx[-n_samples:][::-1]
        
        for rank, idx in enumerate(best_idx, 1):
            self._plot_sample(Y_pred[idx], Y_test[idx], sample_rmse[idx],
                            f"{save_dir}/best_samples", "Best", rank, idx)
        
        for rank, idx in enumerate(worst_idx, 1):
            self._plot_sample(Y_pred[idx], Y_test[idx], sample_rmse[idx],
                            f"{save_dir}/worst_samples", "Worst", rank, idx)
        
        print(f"  Saved {n_samples} best and {n_samples} worst samples")
    
    def _plot_sample(self, pred, true, rmse, save_dir, sample_type, rank, idx):
        n_frames = min(pred.shape[0], 8)
        
        fig = plt.figure(figsize=(4*n_frames, 12))
        gs = GridSpec(3, n_frames+1, width_ratios=[1]*n_frames + [0.05],
                     hspace=0.15, wspace=0.1)
        
        if pred.ndim == 4:
            pred = pred[..., 0]
        if true.ndim == 4:
            true = true[..., 0]
        
        error = pred - true
        vmin = min(np.min(pred), np.min(true))
        vmax = max(np.max(pred), np.max(true))
        max_error = np.max(np.abs(error))
        
        for frame in range(n_frames):
            ax = fig.add_subplot(gs[0, frame])
            im = ax.imshow(pred[frame], cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_title(f't+{frame+1}', fontsize=10)
            ax.axis('off')
            if frame == 0:
                ax.text(-0.2, 0.5, 'Prediction', transform=ax.transAxes,
                       fontsize=12, fontweight='bold', va='center', ha='right')
            
            ax = fig.add_subplot(gs[1, frame])
            ax.imshow(true[frame], cmap='viridis', vmin=vmin, vmax=vmax)
            ax.axis('off')
            if frame == 0:
                ax.text(-0.2, 0.5, 'Truth', transform=ax.transAxes,
                       fontsize=12, fontweight='bold', va='center', ha='right')
            
            ax = fig.add_subplot(gs[2, frame])
            im_err = ax.imshow(error[frame], cmap='viridis', 
                              vmin=-max_error, vmax=max_error)
            mae = np.mean(np.abs(error[frame]))
            ax.set_title(f'MAE: {mae:.2f}', fontsize=9)
            ax.axis('off')
            if frame == 0:
                ax.text(-0.2, 0.5, 'Error', transform=ax.transAxes,
                       fontsize=12, fontweight='bold', va='center', ha='right')
        
        cbar_ax1 = fig.add_subplot(gs[0:2, -1])
        plt.colorbar(im, cax=cbar_ax1, label='PM2.5 (μg/m³)')
        
        cbar_ax2 = fig.add_subplot(gs[2, -1])
        plt.colorbar(im_err, cax=cbar_ax2, label='Error (μg/m³)')
        
        mae_total = np.mean(np.abs(error))
        fig.suptitle(f'{sample_type} Sample #{rank} (Index {idx}) - '
                    f'RMSE: {rmse:.2f}, MAE: {mae_total:.2f}',
                    fontsize=14, fontweight='bold')
        
        plt.savefig(f"{save_dir}/{sample_type.lower()}_{rank:02d}_idx{idx}.png",
                   dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def build_model(self, input_shape):
        if self.config['model_type'] == 'seq2seq':
            return self._build_seq2seq(input_shape)
        elif self.config['model_type'] == 'two_path':
            return self._build_two_path(input_shape)
        elif self.config['model_type'] == 'two_path_res':
            return self._build_two_path_res(input_shape)
        elif self.config['model_type'] == 'multi_path':
            return self._build_multi_path(input_shape)
        else:
            raise ValueError(f"Unknown model type: {self.config['model_type']}")
    
    def _build_seq2seq(self, input_shape):
        from keras.layers import Input, Conv3D, ConvLSTM2D, Lambda, Add
        from keras.models import Model
        
        inputs = Input(shape=input_shape)
        horizon = self.config['forecast_horizon']
        
        pm25 = Lambda(lambda x: x[..., 0:2])(inputs)
        others = Lambda(lambda x: x[..., 2:])(inputs)
        
        enc_pm = ConvLSTM2D(32, (3,3), padding='same', return_sequences=True)(pm25)
        enc_pm, h_pm, c_pm = ConvLSTM2D(64, (3,3), padding='same', return_state=True)(enc_pm)
        
        enc_ot = ConvLSTM2D(32, (3,3), padding='same', return_sequences=True)(others)
        enc_ot, h_ot, c_ot = ConvLSTM2D(64, (3,3), padding='same', return_state=True)(enc_ot)
        
        last_pm25 = Lambda(lambda x: x[:, -1:, :, :, 0:1])(inputs)
        dec_seed = Lambda(lambda x: tf.repeat(x, repeats=horizon, axis=1))(last_pm25)
        
        h0 = Add()([h_pm, h_ot])
        c0 = Add()([c_pm, c_ot])
        
        dec = ConvLSTM2D(64, (3,3), padding='same', return_sequences=True)
        dec_seq = dec(dec_seed, initial_state=[h0, c0])
        dec_seq = Conv3D(32, (3,3,3), padding='same', activation='relu')(dec_seq)
        
        rep_last = Conv3D(32, (3,3,3), padding='same', activation='relu')(dec_seed)
        fused = Add()([dec_seq, rep_last])
        
        output = Conv3D(1, (3,3,3), padding='same', activation='relu')(fused)
        
        model = Model(inputs, output)
        return model, self._create_loss_fn()
    
    def _build_two_path(self, input_shape):
        from keras.layers import Input, Conv2D, Conv3D, ConvLSTM2D, Lambda, concatenate, Add, Reshape
        from keras.models import Model
        tf.keras.backend.set_image_data_format('channels_last')

        inputs = Input(shape=input_shape)
        pm25_temporal = Lambda(lambda x: tf.concat([x[..., 0:2], x[..., 7:11]], axis=-1))(inputs)
        path1 = ConvLSTM2D(20, (3,3), padding='same', return_sequences=True)(pm25_temporal)
        path1 = ConvLSTM2D(40, (3,3), padding='same', return_sequences=True)(path1)
        path1 = Conv3D(20, (3,3,3), activation='relu', padding='same')(path1)
        
        others = Lambda(lambda x: x[..., 2:])(inputs)
        path2 = ConvLSTM2D(20, (3,3), padding='same', return_sequences=True)(others)
        path2 = ConvLSTM2D(40, (3,3), padding='same', return_sequences=True)(path2)
        path2 = Conv3D(20, (3,3,3), activation='relu', padding='same')(path2)
        
        combined = concatenate([path1, path2], axis=-1)
        combined = Conv3D(30, (3,3,3), activation='relu', padding='same')(combined)
        combined = Conv3D(20, (3,3,3), activation='relu', padding='same')(combined)
        
        '''
        weighted = Add()([path1, path2])
        final = concatenate([combined, weighted], axis=-1)
        '''

        output = Conv3D(1, (3,3,3), activation='relu', padding='same')(combined)
        
        model = Model(inputs, output)
        return model, self._create_loss_fn()

    def _build_two_path_res(self, input_shape):
        from keras.layers import Input, Conv2D, Conv3D, ConvLSTM2D, Lambda, concatenate, Add, ReLU
        from keras.models import Model
        tf.keras.backend.set_image_data_format('channels_last')

        inputs = Input(shape=input_shape)

        # path 1
        path1_in = Lambda(lambda x: tf.concat([x[..., 0:2], x[..., 7:11]], axis=-1))(inputs)
        path1 = ConvLSTM2D(20, (3,3), padding='same', return_sequences=True)(path1_in)
        path1 = ConvLSTM2D(40, (3,3), padding='same', return_sequences=True)(path1)
        path1 = Conv3D(20, (3,3,3), padding='same')(path1)

        res_con1 = Conv3D(20, (1,1,1), padding='same')(path1_in)
        path1 = Add()([res_con1, path1])
        path1 = ReLU()(path1)
        
        # path 2
        path2_in = Lambda(lambda x: x[..., 2:])(inputs)
        path2 = ConvLSTM2D(20, (3,3), padding='same', return_sequences=True)(path2_in)
        path2 = ConvLSTM2D(40, (3,3), padding='same', return_sequences=True)(path2)
        path2 = Conv3D(20, (3,3,3), activation='relu', padding='same')(path2)

        res_con2 = Conv3D(20, (1,1,1), padding='same')(path2_in)
        path2 = Add()([res_con2, path2])
        path2 = ReLU()(path2)
        
        # merged
        combined_in = concatenate([path1, path2], axis=-1)
        combined = Conv3D(30, (3,3,3), activation='relu', padding='same')(combined_in)
        combined = Conv3D(20, (3,3,3), padding='same')(combined)
        
        output = Conv3D(1, (3,3,3), activation='relu', padding='same')(combined)
        
        model = Model(inputs, output)
        return model, self._create_loss_fn()

    def _build_multi_path(self, input_shape):
        from keras.layers import Input, Conv3D, ConvLSTM2D, Lambda, concatenate, Add
        from keras.models import Model
        
        inputs = Input(shape=input_shape)
        paths = []
        
        channel_configs = [
            (0, 1, "AirNow PM2.5"),
            (1, 2, "OpenAQ PM2.5"),
            (2, 5, "METAR Wind"),
            (5, 7, "NDVI/Elevation"),
            (7, 11, "Temporal"),
            (11, 12, "GOES"),
            (12, 13, "TEMPO")
        ]
        
        for start_idx, end_idx, name in channel_configs:
            channel = Lambda(lambda x, s=start_idx, e=end_idx: x[..., s:e])(inputs)
            path = ConvLSTM2D(10, (3,3), padding='same', return_sequences=True)(channel)
            path = ConvLSTM2D(20, (3,3), padding='same', return_sequences=True)(path)
            path = Conv3D(10, (3,3,3), activation='relu', padding='same')(path)
            paths.append(path)
        
        combined = concatenate(paths, axis=-1)
        combined = Conv3D(40, (3,3,3), activation='relu', padding='same')(combined)
        combined = Conv3D(20, (3,3,3), activation='relu', padding='same')(combined)
        
        weighted = Add()(paths)
        final = concatenate([combined, weighted], axis=-1)
        
        output = Conv3D(1, (3,3,3), activation='relu', padding='same')(final)
        
        model = Model(inputs, output)
        return model, self._create_loss_fn()
    
    def _create_loss_fn(self):
        sensor_coords = []
        for loc in self.sensor_locations:
            if isinstance(loc, (tuple, list)) and len(loc) >= 2:
                x, y = int(loc[0]), int(loc[1])
                if 0 <= x < self.dim and 0 <= y < self.dim:
                    sensor_coords.append((x, y))
        
        '''
        def masked_mae(y_true, y_pred):
            errors = []
            for x, y in sensor_coords:
                true_at_sensor = y_true[:, :, x, y]
                pred_at_sensor = y_pred[:, :, x, y]
                errors.append(tf.abs(true_at_sensor - pred_at_sensor))
            return tf.reduce_mean(tf.stack(errors, axis=-1))
        
        '''

        def masked_mae(y_true, y_pred):
            # grab vals at sensors: (None, 5, 15)
            truth_sensors = tf.concat(
                [y_true[:, :, x, y] for x, y in sensor_coords],
                axis=-1
            )
            pred_sensors = tf.concat(
                [y_pred[:, :, x, y] for x, y in sensor_coords],
                axis=-1
            )

            # avg per frame: (None, 5)
            avg_truth = tf.reduce_mean(truth_sensors, axis=-1)
            avg_pred = tf.reduce_mean(pred_sensors, axis=-1)

            # error of avg across frames
            return tf.reduce_mean(tf.abs(avg_truth - avg_pred))

        return masked_mae
    
    def train(self, data, data_dir):
        print(f"\nTraining {self.config['model_type']} model...")
        
        input_shape = data['X_train'].shape[1:]
        model, loss_fn = self.build_model(input_shape)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['mae'])
        
        print(f"Model parameters: {model.count_params():,}")
        
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
            )
        ]
        
        batch_size = self.config.get('batch_size', 32)
        
        train_gen = PWWBPyDataset(
            X_paths=[f'{data_dir}/npy_files/X_train.npy'],
            y_path=f'{data_dir}/npy_files/Y_grid_train.npy',
            batch_size=batch_size
        )
        valid_gen = PWWBPyDataset(
            X_paths=[f'{data_dir}/npy_files/X_valid.npy'],
            y_path=f'{data_dir}/npy_files/Y_grid_valid.npy',
            batch_size=batch_size
        )
        history = model.fit(
            train_gen,
            validation_data=valid_gen,
            epochs=self.config.get('epochs', 100),
            callbacks=callbacks,
            verbose=1
        )
        
        best_idx = np.argmin(history.history['val_loss'])
        best_train_loss = history.history['loss'][best_idx]
        best_val_loss = history.history['val_loss'][best_idx]
        
        print(f"Best epoch: {best_idx+1}")
        print(f"Best Train Loss: {best_train_loss:.4f}, Best Val Loss: {best_val_loss:.4f}")
        
        return model, history, {'train_loss': best_train_loss, 'val_loss': best_val_loss}
    
    def evaluate(self, model, data):
        print("\nEvaluating on test set...")
        
        Y_pred = model.predict(data['X_test'], batch_size=32, verbose=1)

        Y_test = data['Y_test']

        Y_test_sensors = self._extract_sensors(Y_test)
        Y_pred_sensors = self._extract_sensors(Y_pred)
        
        frame_metrics = {}
        horizon = self.config['forecast_horizon']
        for hour in range(horizon):
            y_true_h = Y_test_sensors[:, hour, :].flatten()
            y_pred_h = Y_pred_sensors[:, hour, :].flatten()
            
            hour_rmse = np.sqrt(mean_squared_error(y_true_h, y_pred_h))
            hour_nrmse = (hour_rmse / np.mean(y_true_h)) * 100
            
            frame_metrics[hour+1] = {
                'rmse': hour_rmse,
                'nrmse': hour_nrmse
            }
        
        avg_nrmse = np.mean([frame_metrics[h]['nrmse'] for h in frame_metrics])
        
        print(f"\nTest Set Metrics:")
        print(f"  Average NRMSE: {avg_nrmse:.2f}%")
        
        print(f"\nHourly NRMSE:")
        for h in range(1, horizon + 1):
            print(f"  Hour {h:2d}: {frame_metrics[h]['nrmse']:.2f}%")
        
        return {
            'avg_nrmse': avg_nrmse,
            'predictions': Y_pred,
            'frame_metrics': frame_metrics
        }
    
    def save_incremental_results(self, metrics, train_metrics, save_dir):
        os.makedirs(f"{save_dir}/detailed_results", exist_ok=True)
        
        horizon = self.config['forecast_horizon']
        
        entry = {
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Model': self.config['name'],
            'Strategy': self.config.get('strategy', ''),
            'Horizon': horizon,
            'Train_Loss': train_metrics['train_loss'],
            'Val_Loss': train_metrics['val_loss'],
            'Avg_NRMSE': metrics['avg_nrmse']
        }
        
        for hour in range(1, horizon + 1):
            if hour in metrics['frame_metrics']:
                entry[f'H{hour}_NRMSE'] = metrics['frame_metrics'][hour]['nrmse']
        
        self.detailed_results.append(entry)
        
        df = pd.DataFrame(self.detailed_results)
        df.to_csv(f"{save_dir}/detailed_results/frame_by_frame_metrics.csv", index=False)
        print(f"Detailed metrics saved")
    
    def create_performance_table(self, results_df, save_dir):
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.axis('tight')
        ax.axis('off')
        
        display_columns = ['Experiment', 'Model', 'Horizon', 'Train_Loss', 'Val_Loss', 'Avg_NRMSE']
        
        horizon_cols = [col for col in results_df.columns if col.startswith('H') and col.endswith('_NRMSE')]
        display_columns.extend(sorted(horizon_cols, key=lambda x: int(x[1:].split('_')[0])))
        
        display_columns = [col for col in display_columns if col in results_df.columns]
        display_df = results_df[display_columns].round(2)
        
        table = ax.table(cellText=display_df.values,
                        colLabels=display_df.columns,
                        cellLoc='center',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.8)
        
        for i in range(len(display_df.columns)):
            table[(0, i)].set_facecolor('#2c3e50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        highlight_cols = ['Train_Loss', 'Val_Loss', 'Avg_NRMSE']
        for col in highlight_cols:
            if col in display_df.columns:
                col_idx = display_df.columns.get_loc(col)
                best_val = display_df[col].min()
                for row_idx in range(len(display_df)):
                    if display_df.iloc[row_idx][col] == best_val:
                        table[(row_idx+1, col_idx)].set_facecolor('#66d966')
        
        plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.savefig(f"{save_dir}/performance_table.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Performance table saved")
    
    def _extract_sensors(self, grid_data):
        n_samples, n_frames = grid_data.shape[:2]
        n_sensors = len(self.sensor_locations)
        
        values = np.zeros((n_samples, n_frames, n_sensors))
        for i, loc in enumerate(self.sensor_locations):
            if isinstance(loc, (tuple, list)) and len(loc) >= 2:
                x, y = int(loc[0]), int(loc[1])
                if 0 <= x < self.dim and 0 <= y < self.dim:
                    values[:, :, i] = grid_data[:, :, x, y, 0]
        
        return values
    
    def visualize_results(self, predictions, test_data, history, save_dir):
        print("\nCreating visualizations...")
        os.makedirs(save_dir, exist_ok=True)
        
        self._plot_time_series_comparison(predictions, test_data, save_dir)
        self._plot_frame_metrics(save_dir)
        
        try:
            from visualization_lib import ModelVisualizer
            visualizer = ModelVisualizer(
                experiment_type='grid',
                forecast_horizon=self.config['forecast_horizon']
            )
            visualizer.create_summary_report(
                y_pred=predictions,
                Y_test=test_data,
                history=history,
                model_name=self.config['name'],
                save_path=save_dir
            )
            print("ModelVisualizer report created (including training history)")
        except ImportError:
            print("  ModelVisualizer not available")
    
    def _plot_frame_metrics(self, save_dir):
        if not self.detailed_results:
            return
        
        latest_result = self.detailed_results[-1]
        horizon = self.config['forecast_horizon']
        
        hours = []
        nrmse_values = []
        
        for hour in range(1, horizon + 1):
            col_name = f'H{hour}_NRMSE'
            if col_name in latest_result:
                hours.append(hour)
                nrmse_values.append(latest_result[col_name])
        
        if not hours:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.bar(hours, nrmse_values, color='steelblue', edgecolor='navy', linewidth=1.5)
        
        avg_nrmse = np.mean(nrmse_values)
        for bar, value in zip(bars, nrmse_values):
            if value < avg_nrmse * 0.9:
                bar.set_facecolor('green')
            elif value > avg_nrmse * 1.1:
                bar.set_facecolor('red')
        
        ax.set_xlabel('Forecast Hour', fontsize=12)
        ax.set_ylabel('NRMSE (%)', fontsize=12)
        ax.set_title(f'{self.config["name"]} - Hourly NRMSE Performance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        ax.axhline(y=avg_nrmse, color='red', linestyle='--', alpha=0.7, 
                  label=f'Average: {avg_nrmse:.2f}%')
        
        for bar, value in zip(bars, nrmse_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.1f}', ha='center', va='bottom', fontsize=9)
        
        ax.legend()
        ax.set_xticks(hours)
        ax.set_xlim(0.5, horizon + 0.5)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/hourly_nrmse_performance.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print("Hourly NRMSE bar plot saved")
    
    def _plot_time_series_comparison(self, predictions, test_data, save_dir):
        os.makedirs(f"{save_dir}/time_series", exist_ok=True)
        
        Y_test_sensors = self._extract_sensors(test_data)
        Y_pred_sensors = self._extract_sensors(predictions)
        
        n_sensors = Y_test_sensors.shape[2]
        n_plots = min(6, n_sensors)
        sensor_indices = np.random.choice(n_sensors, n_plots, replace=False)
        
        fig, axes = plt.subplots(n_plots, 1, figsize=(15, 3*n_plots))
        if n_plots == 1:
            axes = [axes]
        
        n_samples_to_plot = min(100, Y_test_sensors.shape[0])
        horizon = self.config['forecast_horizon']
        
        for idx, sensor_id in enumerate(sensor_indices):
            ax = axes[idx]
            
            y_true = Y_test_sensors[:n_samples_to_plot, :, sensor_id].flatten()
            y_pred = Y_pred_sensors[:n_samples_to_plot, :, sensor_id].flatten()
            
            time_steps = np.arange(len(y_true))
            
            ax.plot(time_steps, y_true, label='Ground Truth', alpha=0.7, linewidth=1)
            ax.plot(time_steps, y_pred, label='Prediction', alpha=0.7, linewidth=1)
            
            for i in range(1, n_samples_to_plot):
                ax.axvline(x=i*horizon, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
            
            ax.set_title(f'Sensor {sensor_id} - Time Series Comparison')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('PM2.5 (μg/m³)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/time_series/sensor_comparisons.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print("Time series comparison plots saved")

def run_experiments():
    data_cache_base_dir = '/home/mgraca/Workspace/hrrr-smoke-viz/pwwb-experiments/tensorflow/final_input_data/multipath_data' 
    experiments = [
#        {
#            'name': 'Two-Path-test',
#            'model_type': 'two_path_res',
#            'data_cache': f'{data_cache_base_dir}/25_scale',
#            'forecast_horizon': 5,
#            'epochs': 3,
#            'batch_size': 32 
#        },
#        {
#            'name' : 'error-of-avg',
#            'model_type': 'two_path',
#            'data_cache': f'{data_cache_base_dir}/25_scale',
#            'forecast_horizon': 5,
#            'epochs': 100,
#            'batch_size': 32
#        },
        {
            'name' : 'error-of-avg-rescon-with-1-kernel',
            'model_type': 'two_path_res',
            'data_cache': f'{data_cache_base_dir}/25_scale',
            'forecast_horizon': 5,
            'epochs': 100,
            'batch_size': 32
        },
    ]
    
    results = []
    
    for exp_config in experiments:
        print("\n" + "="*60)
        print(f"Running: {exp_config['name']}")
        print(f"Forecast Horizon: {exp_config['forecast_horizon']} hours")
        print("="*60)
        
        pipeline = PM25TrainingPipeline(exp_config)
        
        data = pipeline.load_data()
        
        save_dir = f"results/{exp_config['name'].replace(' ', '_')}"
        pipeline.visualize_input_samples(data, save_dir)
        
        model, history, train_metrics = pipeline.train(data, exp_config['data_cache'])
        
        metrics = pipeline.evaluate(model, data)

        np.save(f'{save_dir}/Y_pred.npy', metrics['predictions'])
        model.save(f'{save_dir}/model.keras')

        plot_model(
            model, 
            to_file=f'{save_dir}/model_architecture.png',
            show_shapes=True,
            show_layer_activations=True
        )
        
        pipeline.save_best_worst_samples(
            metrics['predictions'], 
            data['Y_test'],
            save_dir
        )
        
        pipeline.save_incremental_results(metrics, train_metrics, save_dir)
        
        pipeline.visualize_results(
            metrics['predictions'],
            data['Y_test'],
            history,
            save_dir
        )
        
        result_entry = {
            'Experiment': exp_config['name'],
            'Model': exp_config['model_type'],
            'Horizon': exp_config['forecast_horizon'],
            'Train_Loss': train_metrics['train_loss'],
            'Val_Loss': train_metrics['val_loss'],
            'Avg_NRMSE': metrics['avg_nrmse']
        }
        
        for hour in range(1, exp_config['forecast_horizon'] + 1):
            if hour in metrics['frame_metrics']:
                result_entry[f'H{hour}_NRMSE'] = metrics['frame_metrics'][hour]['nrmse']
        
        results.append(result_entry)
        
        del model
        tf.keras.backend.clear_session()
        gc.collect()
    
    if results:
        df = pd.DataFrame(results)
        print("\n" + "="*60)
        print("EXPERIMENT RESULTS")
        print("="*60)
        
        summary_cols = ['Experiment', 'Model', 'Horizon', 'Train_Loss', 'Val_Loss', 'Avg_NRMSE']
        print(df[summary_cols].to_string(index=False))
        
        pipeline = PM25TrainingPipeline(experiments[0])
        pipeline.create_performance_table(df, "results")
        
        df.to_csv("results/comparison_full.csv", index=False)
        print(f"\nFull results saved to results/comparison_full.csv")

if __name__ == "__main__":
    run_experiments()
