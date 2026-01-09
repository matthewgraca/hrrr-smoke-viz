import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, Conv3D, ConvLSTM2D, Dense, Flatten, 
    Reshape, UpSampling2D, Lambda, concatenate
)
from tensorflow.keras.models import Model


ARCH_5x5 = {
    'temporal_filters': [16, 24, 32],
    'spatial_filters': [32, 48, 64],
    'bottleneck': 128,
    'latent_size': 5,
    'strides': [2, 2, 2]
}

ARCH_10x10 = {
    'temporal_filters': [16, 24, 32],
    'spatial_filters': [32, 48],
    'bottleneck': 256,
    'latent_size': 10,
    'strides': [2, 2]
}


def build_model(input_shape, arch_config, output_horizon, n_observed):
    inputs = Input(shape=input_shape)
    height, width = input_shape[1], input_shape[2]
    
    print(f"Input: {input_shape} -> Output: ({output_horizon}, {height}, {width}, 1)")
    
    temporal_filters = arch_config['temporal_filters']
    spatial_filters = arch_config['spatial_filters']
    bottleneck = arch_config['bottleneck']
    latent_size = arch_config['latent_size']
    strides = arch_config['strides']
    
    def tile_for_decoder(x):
        return tf.tile(tf.expand_dims(x, axis=1), [1, output_horizon, 1, 1, 1])
    
    def create_pos_encoding(x):
        batch_size = tf.shape(x)[0]
        pos = tf.reshape(tf.range(output_horizon, dtype=tf.float32) / output_horizon, [1, output_horizon, 1, 1, 1])
        return tf.tile(pos, [batch_size, 1, height, width, 1])
    
    def spatial_encoder(x, prefix):
        encs = []
        for i, filters in enumerate(spatial_filters):
            x = Conv2D(filters, (3,3), strides=strides[i], padding='same', activation='relu', name=f'{prefix}_e{i+1}')(x)
            encs.append(x)
        return encs
    
    def spatial_decoder(x, past_encs, fcast_encs):
        skips = list(zip(reversed(past_encs[:-1]), reversed(fcast_encs[:-1])))
        for i, (past_skip, fcast_skip) in enumerate(skips):
            x = UpSampling2D(size=(2,2), interpolation='bilinear', name=f'dec_up{i+1}')(x)
            x = Conv2D(spatial_filters[-(i+2)], (3,3), padding='same', activation='relu', name=f'dec_d{i+1}')(x)
            x = x + past_skip + fcast_skip
        x = UpSampling2D(size=(2,2), interpolation='bilinear', name=f'dec_up{len(spatial_filters)}')(x)
        x = Conv2D(temporal_filters[2], (3,3), padding='same', activation='relu', name=f'dec_d{len(spatial_filters)}')(x)
        return x
    
    observed = inputs[:, :, :, :, :n_observed]
    forecast = inputs[:, :, :, :, n_observed:]
    
    past_proj = Conv3D(temporal_filters[0], (1,1,1), padding='same', name='past_proj')(observed)
    past_lstm1 = ConvLSTM2D(temporal_filters[0], (3,3), padding='same', return_sequences=True, name='past_lstm1')(past_proj)
    past_lstm1 = past_lstm1 + past_proj
    past_lstm2 = ConvLSTM2D(temporal_filters[1], (3,3), padding='same', return_sequences=True, name='past_lstm2')(past_lstm1)
    past_lstm1_proj = Conv3D(temporal_filters[1], (1,1,1), padding='same', name='past_res1_proj')(past_lstm1)
    past_lstm2 = past_lstm2 + past_lstm1_proj
    past_lstm3, h, c = ConvLSTM2D(temporal_filters[2], (3,3), padding='same', return_state=True, name='past_lstm3')(past_lstm2)
    past_lstm2_proj = Conv2D(temporal_filters[2], (1,1), padding='same', name='past_res2_proj')(past_lstm2[:, -1])
    past_state = past_lstm3 + past_lstm2_proj
    
    past_encs = spatial_encoder(past_state, 'past')
    
    fcast_proj = Conv3D(temporal_filters[0], (1,1,1), padding='same', name='fcast_proj')(forecast)
    fcast_lstm1 = ConvLSTM2D(temporal_filters[0], (3,3), padding='same', return_sequences=True, name='fcast_lstm1')(fcast_proj)
    fcast_lstm1 = fcast_lstm1 + fcast_proj
    fcast_lstm2 = ConvLSTM2D(temporal_filters[1], (3,3), padding='same', return_sequences=True, name='fcast_lstm2')(fcast_lstm1)
    fcast_lstm1_proj = Conv3D(temporal_filters[1], (1,1,1), padding='same', name='fcast_res1_proj')(fcast_lstm1)
    fcast_lstm2 = fcast_lstm2 + fcast_lstm1_proj
    fcast_state = fcast_lstm2[:, -1]
    fcast_state = Conv2D(temporal_filters[2], (1,1), padding='same', activation='relu', name='fcast_out_proj')(fcast_state)
    
    fcast_encs = spatial_encoder(fcast_state, 'fcast')
    
    fused = concatenate([past_encs[-1], fcast_encs[-1]], axis=-1, name='bn_concat')
    fused = Flatten(name='bn_flat')(fused)
    fused = Dense(bottleneck, activation='relu', name='bn_dense')(fused)
    
    dec = Dense(latent_size * latent_size * spatial_filters[-1], activation='relu', name='dec_proj')(fused)
    dec = Reshape((latent_size, latent_size, spatial_filters[-1]), name='dec_reshape')(dec)
    
    dec = spatial_decoder(dec, past_encs, fcast_encs)
    
    dec_tiled = Lambda(tile_for_decoder, name='dec_tile')(dec)
    pos_enc = Lambda(create_pos_encoding, name='pos_enc')(dec_tiled)
    dec_input = concatenate([dec_tiled, pos_enc], axis=-1, name='dec_input')
    
    dec_lstm1 = ConvLSTM2D(temporal_filters[2], (3,3), padding='same', return_sequences=True, name='dec_lstm1')(
        dec_input, initial_state=[h, c]
    )
    enc_tiled = Lambda(tile_for_decoder, name='enc_tile')(dec)
    dec_lstm1 = dec_lstm1 + enc_tiled
    
    dec_lstm2 = ConvLSTM2D(temporal_filters[2], (3,3), padding='same', return_sequences=True, name='dec_lstm2')(dec_lstm1)
    dec_lstm2 = dec_lstm2 + dec_lstm1
    
    output = Conv3D(1, (1,1,1), padding='same', activation='relu', name='output')(dec_lstm2)
    
    model = Model(inputs, output)
    print(f"Architecture: {latent_size}x{latent_size} | Parameters: {model.count_params():,}")
    return model

# This was for me testing to make sure the model builds correctly with the hyperparamter config setup.
if __name__ == "__main__":
    import pickle
    
    cache_path = 'data/shared/new_extent_cache/preprocessed_cache/24in_24out_no_holidays'
    
    with open(f"{cache_path}/metadata.pkl", 'rb') as f:
        metadata = pickle.load(f)
    
    n_observed = len(metadata.get('observed_channels', []))
    n_forecast = len(metadata.get('forecast_channels', []))
    n_channels = n_observed + n_forecast
    input_horizon = metadata.get('input_horizon', 24)
    output_horizon = metadata.get('output_horizon', 24)
    
    input_shape = (input_horizon, 40, 40, n_channels)
    
    print(f"Observed: {n_observed}, Forecast: {n_forecast}, Total: {n_channels}")
    
    model = build_model(input_shape, ARCH_5x5, output_horizon, n_observed)
    model.summary()