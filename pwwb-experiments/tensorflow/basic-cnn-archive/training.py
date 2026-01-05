import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import keras
tf.keras.backend.set_image_data_format('channels_last')

import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from sklearn.metrics import root_mean_squared_error
import json
from tqdm import tqdm
import joblib
import sys

BASE_PATH = '/home/mgraca/Workspace/hrrr-smoke-viz'
EXPERIMENT_PATH = os.path.join(BASE_PATH, 'pwwb-experiments/tensorflow/basic-cnn-archive')
DATA_PATH = os.path.join(EXPERIMENT_PATH, 'processing-scripts/l3')
RESULTS_PATH = os.path.join(EXPERIMENT_PATH, 'results')
# hardcoded since we already processed AirNowData
SENSORS = {
    'Reseda' : (8, 3),
    'North Holywood' : (8, 11),
    'Los Angeles - N. Main Street' : (15, 16),
    'Compton' : (23, 17),
    'Long Beach Signal Hill' : (29, 19),
    'Anaheim' : (27, 29),
    'Glendora - Laurel' : (10, 33)
}
CHANNELS = None

json_file = os.path.join(EXPERIMENT_PATH, 'processing-scripts/channels.json')
with open(json_file, 'r') as f:
    CHANNELS = json.load(f) 

class TextColor:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'  # Resets formatting
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class TrainingPipeline():
    def __init__(
        self,
        epochs=100,
        batch_size=64,
    ):
        print('Loading default parameters:')
        print(f'\tPath of the training data: {DATA_PATH}')
        print(f'\tPath of the results: {RESULTS_PATH}\n')

        print(f'Airnow sensor locations in use ({len(SENSORS)}):')
        for k, v in SENSORS.items():
            print(f'\t{k} : {v}')
        print()

        print(f'Channels in use: ({len(CHANNELS)}):')
        print('Channel name : Channel index')
        for k, v in CHANNELS.items():
            print(f'\t{k} : {v}')
        print()

        X_train, X_valid, X_test, Y_train, Y_valid, Y_test = self._load_data(DATA_PATH)

        # NOTE toggle for quick test
        X_test = X_test[0:25]
        Y_test = Y_test[0:25]

        dim = X_train.shape[1]
        input_shape = X_train.shape[1:]

        #model = self._base_model(input_shape)
        model = self._aux_loss_model(input_shape, passthru_channels=['naqfc_pm25', '30_day_lookback'])
        # TODO use raw instead of nowcast
        # TODO use forecast instead of current for naqfc
        #sys.exit(0)

        print(f'{TextColor.BLUE}Beginning Training{TextColor.ENDC}')
        history = model.fit(
            X_train,
            Y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_valid, Y_valid)
        )
        print()

        print(f'{TextColor.BLUE}Beginning Autoregressive Inference{TextColor.ENDC}')
        self._autoregress(model, X_test, Y_test, horizon=24)

        print(f'{TextColor.BLUE}Beginning Inference{TextColor.ENDC}')
        y_pred = model.predict(X_test)
        res = self._evaluate(SENSORS, dim, Y_test, y_pred)
        print(res)

        print(f'{TextColor.BLUE}Saving results{TextColor.ENDC}')
        self._plot_loss_curves(history)
        self._plot_predictions(X_test, Y_test, y_pred)
        np.save(os.path.join(RESULTS_PATH, 'y_pred.npy'), y_pred)

        return

    def _load_data(self, DATA_PATH):
        print('Loading training data (eager loading)...', end=' ')
        X_train = np.load(os.path.join(DATA_PATH, 'X_train.npy'))
        X_valid = np.load(os.path.join(DATA_PATH, 'X_valid.npy'))
        X_test = np.load(os.path.join(DATA_PATH, 'X_test.npy'))
        Y_train = np.load(os.path.join(DATA_PATH, 'Y_train.npy'))
        Y_valid = np.load(os.path.join(DATA_PATH, 'Y_valid.npy'))
        Y_test = np.load(os.path.join(DATA_PATH, 'Y_test.npy'))
        print('complete!\n')

        # initially fit data for conv3d, not conv2d; requires us to squeeze out the frame dimension
        # (sample, frames, h, w, c) -> (sample, h, w, c)
        X_train = np.squeeze(X_train)
        X_valid = np.squeeze(X_valid)
        X_test = np.squeeze(X_test)
        Y_train = np.squeeze(Y_train)
        Y_valid = np.squeeze(Y_valid)
        Y_test = np.squeeze(Y_test)

        train_usage = X_train.nbytes + Y_train.nbytes
        valid_usage = X_valid.nbytes + Y_valid.nbytes
        test_usage = X_test.nbytes + Y_test.nbytes

        print('Datasets shape')
        print(f'\tTrain: X={X_train.shape}, Y={Y_train.shape}, Space usage: {train_usage/1e9:.2f}GB')
        print(f'\tValid: X={X_valid.shape}, Y={Y_valid.shape}, Space usage: {valid_usage/1e9:.2f}GB')
        print(f'\tTest : X={X_test.shape}, Y={Y_test.shape}, Space usage: {test_usage/1e9:.2f}GB')
        print(f'Total usage: {(train_usage + valid_usage + test_usage) / 1e9:.2f}GB\n')

        return X_train, X_valid, X_test, Y_train, Y_valid, Y_test

    def _evaluate(self, sensors, dim, Y_test, y_pred):
        # note that this is the same nhood loss as the training. so sensor location + nhood + background 
        # are accounted for; this is not just a raw nhood loss only.
        y_pred = np.squeeze(y_pred) if len(np.squeeze(y_pred).shape) == 2 else y_pred[..., 0]
        nhood_loss = self.NHoodLoss(sensors, dim)(Y_test, y_pred).numpy()
        sensor_loss = self.SensorLoss(sensors)(Y_test, y_pred).numpy()
        grid_loss = np.mean(np.abs(Y_test - y_pred))
        return {'nhood_loss' : nhood_loss, 'sensor_loss' : sensor_loss, 'frame_loss' : grid_loss}
        
        '''
        frame_metrics = {}
        for hour in range(horizon):
            y_true_h = Y_test_sensors[:, hour, :].flatten()
            y_pred_h = Y_pred_sensors[:, hour, :].flatten()
            
            hour_rmse = root_mean_squared_error(y_true_h, y_pred_h))
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
        '''

    def _autoregress(self, model, X_test, Y_test, horizon=24):
        def create_next_sample(X_prev, X_next, y_pred):
            '''
            Forecast/ffill channels are hardcoded!

            X_prev = previous sample (40, 40, 17)
            X_next = future sample (40, 40, 17)
            y_pred = prediction made (40, 40)
                - unless it's aux loss, which is (40, 40, n)
            '''
            ffill_channels = [
                'goes_aod',
                'openaq_pm25',
                'tempo_no2',
                'elevation',
                'ndvi'
            ]
            fcast_channels = [
                'hrrr_pbl_height',
                'hrrr_precip_rate',
                'temporal_encoding_hour_sin',
                'hrrr_wind_speed',
                'hrrr_temp_2m',
                'naqfc_pm25',
                'hrrr_v_wind',
                'temporal_encoding_month_sin',
                'hrrr_u_wind',
                'temporal_encoding_month_cos',
                'temporal_encoding_hour_cos',
            ]
            y_pred = y_pred if len(y_pred.shape) == 2 else y_pred[..., 0]
            X = X_prev.copy()
            for c in fcast_channels:
                X[..., CHANNELS[c]] = X_next[..., CHANNELS[c]]

            std_scaler = joblib.load(
                os.path.join(EXPERIMENT_PATH, 'processing-scripts/std_scale.bin')
            )
            X[..., CHANNELS['airnow_pm25']] = (
                std_scaler
                .transform(y_pred.reshape(-1, 1))
                .reshape(y_pred.shape)
            )

            return X

        sample_nhood_loss = []
        sample_sensor_loss = []
        sample_grid_loss = []
        for i in tqdm(range(X_test.shape[0] - horizon)):
            X = X_test[i]
            nhood_loss = []
            sensor_loss = []
            grid_loss = []
            sample_y = []
            sample_x = []
            sample_actual = []
            for j in range(horizon):
                y_pred = model.predict(np.expand_dims(X, axis=0), batch_size=1, verbose=0)
                y_pred = y_pred if len(np.squeeze(y_pred).shape) == 2 else y_pred[..., 0]
                X = create_next_sample(X, X_test[i+j+1], y_pred)
                res = self._evaluate(SENSORS, 40, Y_test[i+j], y_pred)

                sample_x.append(Y_test[i+j-1])
                sample_y.append(np.squeeze(y_pred))
                sample_actual.append(Y_test[i+j])

                nhood_loss.append(res['nhood_loss'])
                sensor_loss.append(res['sensor_loss'])
                grid_loss.append(res['frame_loss'])

            sample_nhood_loss.append(nhood_loss)
            sample_sensor_loss.append(sensor_loss)
            sample_grid_loss.append(grid_loss)

        # plot losses over frames
        sensor_loss_by_frame = np.mean(np.array(sample_sensor_loss), axis=0)
        nhood_loss_by_frame = np.mean(np.array(sample_nhood_loss), axis=0)
        grid_loss_by_frame = np.mean(np.array(sample_grid_loss), axis=0)

        plt.bar(np.arange(0, len(sensor_loss_by_frame)), sensor_loss_by_frame)
        plt.title('Sensor loss by frame')
        plt.xlabel('Frame')
        plt.ylabel('MAE')
        plt.savefig(os.path.join(RESULTS_PATH, 'sensor_loss_bar.png'))
        plt.close()

        plt.bar(np.arange(0, len(nhood_loss_by_frame)), nhood_loss_by_frame)
        plt.title('Neighborhood loss by frame')
        plt.xlabel('Frame')
        plt.ylabel('MAE')
        plt.savefig(os.path.join(RESULTS_PATH, 'nhood_loss_bar.png'))
        plt.close()

        plt.bar(np.arange(0, len(grid_loss_by_frame)), grid_loss_by_frame)
        plt.title('Grid loss by frame')
        plt.xlabel('Frame')
        plt.ylabel('MAE')
        plt.savefig(os.path.join(RESULTS_PATH, 'grid_loss_bar.png'))
        plt.close()

        # sample plotting (plots the last sample)
        sample_y = np.array(sample_y)
        sample_x = np.array(sample_x)
        sample_actual = np.array(sample_actual)

        # need to load unscaled data
        naqfc = np.load(os.path.join(EXPERIMENT_PATH, 'processing-scripts/l2/naqfc_pm25.npy'))
        lookback = np.load(os.path.join(EXPERIMENT_PATH, 'processing-scripts/l2/30_day_lookback.npy'))
        # mystical indexing allowed since X_test makes up the last porition of all the data
        naqfc_sample = naqfc[-len(X_test) + i - horizon : -len(X_test) + i]
        lookback_sample = lookback[-len(X_test) + i - horizon : -len(X_test) + i]

        # horizontal plot
        ncols = horizon 
        all_samples = [sample_x, sample_y, sample_actual, naqfc_sample, lookback_sample] 
        vmax = np.nanmax(all_samples)
        vmin = np.nanmin(all_samples)
        fig, axes = plt.subplots(nrows=len(all_samples), ncols=ncols, figsize=(56, 14))
        for c in range(ncols):
            # autoregression only has one real frame: the first one. the other inputs are last frame's output
            # input
            axes[0, c].imshow(sample_x[c] if c == 0 else sample_y[c-1], vmin=vmin, vmax=vmax)
            axes[0, c].set_xticks([])
            axes[0, c].set_yticks([])

            # target
            axes[1, c].imshow(sample_actual[c], vmin=vmin, vmax=vmax)
            axes[1, c].set_xticks([])
            axes[1, c].set_yticks([])

            # prediction
            im = axes[2, c].imshow(sample_y[c], vmin=vmin, vmax=vmax)
            axes[2, c].set_xticks([])
            axes[2, c].set_yticks([])

            # naqfc
            im = axes[3, c].imshow(naqfc_sample[c], vmin=vmin, vmax=vmax)
            axes[3, c].set_xticks([])
            axes[3, c].set_yticks([])

            # 30-day average
            im = axes[4, c].imshow(lookback_sample[c], vmin=vmin, vmax=vmax)
            axes[4, c].set_xticks([])
            axes[4, c].set_yticks([])

            fig.colorbar(im, ax=axes[:, c], orientation='horizontal', fraction=0.046, pad=0.04)

        '''
        axes[0, 0].set_title(f'frame {0}')
        axes[1, 0].set_title(f'frame {0}')
        axes[2, 0].set_title(f'frame {0}')
        '''

        fig.text(0.12, 0.8125, 'Test Input', rotation=90, va='center', ha='center', fontsize=14)
        fig.text(0.12, 0.6625, 'Test Target', rotation=90, va='center', ha='center', fontsize=14)
        fig.text(0.12, 0.525, 'Prediction', rotation=90, va='center', ha='center', fontsize=14)
        fig.text(0.12, 0.375, 'NAQFC \n(pred-aligned)', rotation=90, va='center', ha='center', fontsize=14)
        fig.text(0.12, 0.225, '30-day Lookback \n(pred-aligned)', rotation=90, va='center', ha='center', fontsize=14)

        plt.savefig(os.path.join(RESULTS_PATH, 'horizontal_sample.png'))

        # vertical plot
        '''
        nrows = 24
        fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(8, 50))
        for r in range(nrows):
            vmax = np.nanmax([sample_x, sample_y, sample_actual])
            vmin = np.nanmin([sample_x, sample_y, sample_actual])

            axes[r, 0].imshow(sample_x[r], vmin=vmin, vmax=vmax)
            axes[r, 0].set_xticks([])
            axes[r, 0].set_yticks([])
            axes[r, 0].set_title(f'frame {r}')

            axes[r, 1].imshow(sample_actual[r], vmin=vmin, vmax=vmax)
            axes[r, 1].set_xticks([])
            axes[r, 1].set_yticks([])
            axes[r, 1].set_title(f'frame {r}')

            im = axes[r, 2].imshow(sample_y[r], vmin=vmin, vmax=vmax)
            axes[r, 2].set_xticks([])
            axes[r, 2].set_yticks([])
            axes[r, 2].set_title(f'frame {r}')

            fig.colorbar(im, ax=axes[r, :], orientation='vertical', fraction=0.015, pad=0.04)
            
        axes[0, 0].set_title(f'Test Input \nframe {0}')
        axes[0, 1].set_title(f'Test Target \nframe {0}')
        axes[0, 2].set_title(f'Prediction \nframe {0}')
        plt.savefig(os.path.join(RESULTS_PATH, 'vertical_sample.png'))
        '''

        return

    ### NOTE visualizations (things that will be saved to results/)

    def _plot_predictions(self, X_test, Y_test, y_pred, save_path=RESULTS_PATH):
        np.random.seed(42)
        y_pred = y_pred if len(y_pred.shape) == 3 else y_pred[..., 0]
        fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(12, 8))

        # not directly plotting X_test since scaling affects visualization
        for c in range(axes.shape[1]):
            idx = np.random.randint(1, len(Y_test))
            vmin = np.nanmin([Y_test[idx], y_pred[idx], Y_test[idx-1]])
            vmax = np.nanmax([Y_test[idx], y_pred[idx], Y_test[idx-1]])
            
            axes[0, c].imshow(Y_test[idx-1], vmin=vmin, vmax=vmax)
            axes[0, c].set_title(f'index: {idx}')
            axes[0, c].set_xticks([])
            axes[0, c].set_yticks([])
            
            axes[1, c].imshow(Y_test[idx], vmin=vmin, vmax=vmax)
            axes[1, c].set_xticks([])
            axes[1, c].set_yticks([])

            im = axes[2, c].imshow(y_pred[idx], vmin=vmin, vmax=vmax)
            axes[2, c].set_xticks([])
            axes[2, c].set_yticks([])

            fig.colorbar(im, ax=axes[:, c], orientation='horizontal', fraction=0.046, pad=0.04)
            
        fig.text(0.11, 0.775, 'Test Input', rotation=90, va='center', ha='center', fontsize=14)
        fig.text(0.11, 0.525, 'Test Target', rotation=90, va='center', ha='center', fontsize=14)
        fig.text(0.11, 0.275, 'Prediction', rotation=90, va='center', ha='center', fontsize=14)
        plt.savefig(os.path.join(RESULTS_PATH, 'pred_vs_test_samples.png'))

        return

    def _plot_loss_curves(self, history, save_path=RESULTS_PATH):
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.title(f'\nTraining Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MAE)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_path, 'loss_curve.png'))

        return

    ### NOTE models

    def _base_model(self, input_shape):
        from keras.models import Sequential
        from keras.layers import Conv2D, Input, Reshape
        from keras.optimizers import Adam

        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(Conv2D(filters=15, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(Conv2D(filters=30, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(Conv2D(filters=15, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(Conv2D(filters=1, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(Reshape((input_shape[0], input_shape[1])))

        model.compile(
            loss=self.NHoodLoss(
                sensors=SENSORS,
                dim=input_shape[1],
                #source_weight,
                #nhood_weight,
                #r
            ),
            optimizer=Adam()
        )

        model.summary()
        print()

        return model

    def _aux_loss_model(self, input_shape, passthru_channels=['naqfc_pm25', '30_day_lookback']):
        from keras.models import Model
        from keras import Sequential
        from keras.layers import Conv2D, Input, Reshape, Concatenate, Lambda
        from keras.optimizers import Adam
        from keras.utils import plot_model

        def conv_block(name):
            model = Sequential(name=name)
            conv2d_kwargs = {
                'kernel_size': (3, 3),
                'activation': 'relu',
                'padding':'same'
            }
            model.add(Conv2D(filters=15, **conv2d_kwargs))
            model.add(Conv2D(filters=30, **conv2d_kwargs))
            model.add(Conv2D(filters=15, **conv2d_kwargs))
            model.add(Conv2D(filters=1, **conv2d_kwargs))
            return model
        
        class PassthruLayer(keras.Layer):
            def call(self, x):
                return tf.concat(
                    [tf.expand_dims(x[..., CHANNELS[c]], axis=-1) for c in passthru_channels],
                    axis=-1
                )

        input_layer = Input(shape=input_shape)
        passthru = PassthruLayer()(input_layer)
        conv_layers = conv_block(name='conv_block')(input_layer)
        output = Concatenate(axis=-1, name='output_layer')([conv_layers, passthru])

        model = Model(input_layer, output)
        model.compile(
            loss=self.AuxLoss(),
            optimizer=Adam()
        )
        model.summary()

        plot_model_kwargs = {
            'model': model,
            'show_shapes': True,
            'show_layer_names': True
        }
        plot_model(**plot_model_kwargs, to_file=os.path.join(RESULTS_PATH, 'model.png'))
        plot_model(**plot_model_kwargs, expand_nested=True, to_file=os.path.join(RESULTS_PATH, 'model_nested.png'))
        print()

        return model
        

    class SensorLoss(tf.keras.losses.Loss):
        def __init__(self, sensors):
            super().__init__()
            self.sensors = list(sensors.values())

        def call(self, y_true, y_pred):
            errors = [
                tf.abs(tf.subtract(y_true[..., x, y], y_pred[..., x, y]))
                for x, y in self.sensors
            ]
            return tf.reduce_mean(tf.stack(errors, axis=-1))

        # for saving and loading models using this loss
        def get_config(self):
            config = super().get_config()
            config.update({'sensors': self.sensors})
            return config

    class NHoodLoss(tf.keras.losses.Loss):
        def __init__(self, sensors, dim, source_weight=25, nhood_weight=5, bg_weight=1, r=2):
            super().__init__()
            self.weights = self._get_weights(
                list(sensors.values()),
                dim,
                source_multiplier=source_weight,
                n_hood_multiplier=nhood_weight,
                bg_multiplier=bg_weight,
                radius=r
            )

        def call(self, y_true, y_pred):
            return tf.reduce_mean(tf.abs(y_true - y_pred) * self.weights)

        #### NOTE long block of funcs that find weights
        def _in_bounds(self, x, y, bound):
            '''
            ensures that neighborhood pair that is outside the dimensions doesn't
                get counted
            '''
            x_in_bound = x >= 0 and x < bound
            y_in_bound = y >= 0 and y < bound

            return x_in_bound and y_in_bound
        
        def _find_neighbors(self, sources, radius, dim):
            '''
            Finds the (x, y) pairs that serve as the neighbors of the sources
            '''
            n_hood = set(product(range(-radius, radius + 1), repeat=2))
            n_hood.remove((0, 0))
            neighbors = set()
            for x, y in sources:
                for a, b in n_hood:
                    f, g = x + a, y + b
                    if self._in_bounds(f, g, dim):
                        neighbors.add((f, g))

            return neighbors

        def _determine_weights(
            self,
            sources,
            n_hood,
            dim,
            source_multiplier,
            n_hood_multiplier,
            bg_multiplier
        ):
            '''
            applies the proper weights to:
                - the background
                - the neighborhoods
                - the sources
            based on the list of pairs in sources and nhood
            '''
            weights = np.full((dim, dim), bg_multiplier)
            for (x, y) in n_hood:
                weights[x, y] = n_hood_multiplier
            for (x, y) in sources:
                weights[x, y] = source_multiplier 

            return weights
        
        def _get_weights(
            self,
            sensor_locations,
            dim,
            source_multiplier=25,
            n_hood_multiplier=5,
            bg_multiplier=1,
            radius=1
        ):
            sensor_coords = set(sensor_locations)
            neighbors = self._find_neighbors(sensor_coords, radius, dim)
            weights = self._determine_weights(
                sensor_coords,
                neighbors,
                dim,
                source_multiplier,
                n_hood_multiplier,
                bg_multiplier
            )
            return weights

        # for saving and loading models using this loss
        def get_config(self):
            config = super().get_config()
            config.update({'sensors': self.sensors})
            return config

    # channel order is order of concatenation
    class AuxLoss(tf.keras.losses.Loss):
        def __init__(self):
            super().__init__()

        def call(self, y_true, y_pred):
            airnow_true = y_true
            airnow_pred = y_pred[..., 0]
            # y_pred contains the passthru channels, so the pred actually contains some true channels
            naqfc_true = y_pred[..., 1]
            lookback_true = y_pred[..., 2]
            w = 0.3
            return tf.reduce_mean(
                tf.abs(airnow_true - airnow_pred) + 
                w * tf.abs(naqfc_true - airnow_pred) +
                w * tf.abs(lookback_true - airnow_pred)
            )

        def get_config(self):
            config = super().get_config()
            return config

def main():
    TrainingPipeline(
        epochs=5,
        batch_size=64,
    )

if __name__ == "__main__":
    main()
