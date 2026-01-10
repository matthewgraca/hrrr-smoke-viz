import sys
sys.path.append('/home/mgraca/Workspace/hrrr-smoke-viz')

def classic(input_shape):
    from keras.models import Sequential
    from keras.layers import ConvLSTM2D, Conv3D, InputLayer
    from keras.optimizers import Adam

    # common args in each layer to reduce function call length
    CONVLSTM2D_ARGS = {
        'kernel_size': (3, 3),
        'padding': 'same',
        'return_sequences': True
    }

    CONV3D_ARGS = {
        'kernel_size': (3, 3, 3),
        'activation': 'relu',
        'padding': 'same'    
    }

    # model definition
    model = Sequential()
    model.add(InputLayer(input_shape))
    model.add(ConvLSTM2D(filters=25, **CONVLSTM2D_ARGS))
    model.add(ConvLSTM2D(filters=50, **CONVLSTM2D_ARGS))
    model.add(Conv3D(filters=25, **CONV3D_ARGS))
    model.add(Conv3D(filters=1, **CONV3D_ARGS))

    return model

def two_path(input_shape, path1_channels, path2_channels):
    # pass in the indices of the channels, not their name!
    from keras.layers import Input, Conv3D, ConvLSTM2D
    from keras.layers import Concatenate
    from keras.models import Model
    from libs.layers import ChannelSplitLayer

    # common args in each layer to reduce function call length
    CONVLSTM2D_ARGS = {
        'kernel_size': (3, 3),
        'padding': 'same',
        'return_sequences': True
    }

    CONV3D_ARGS = {
        'kernel_size': (3, 3, 3),
        'activation': 'relu',
        'padding': 'same'    
    }

    def path_block(x, name):
        x = ConvLSTM2D(filters=20, **CONVLSTM2D_ARGS, name=f'{name}_convlstm_2d_1')(x)
        x = ConvLSTM2D(filters=40, **CONVLSTM2D_ARGS, name=f'{name}_convlstm_2d_2')(x)
        x = Conv3D(filters=20, **CONV3D_ARGS, name=f'{name}_conv3d_1')(x)
        return x

    def trunk_block(x, name):
        x = Conv3D(filters=30, **CONV3D_ARGS, name=f'{name}_conv3d_1')(x)
        x = Conv3D(filters=20, **CONV3D_ARGS, name=f'{name}_conv3d_2')(x)
        x = Conv3D(filters=1, **CONV3D_ARGS, name=f'{name}_conv3d_3')(x)
        return x

    inputs = Input(shape=input_shape)

    p_1 = ChannelSplitLayer(path1_channels)(inputs)
    p_1 = path_block(p_1, name='pm25_path')

    p_2 = ChannelSplitLayer(path2_channels)(inputs)
    p_2 = path_block(p_2, name='other_path')

    x = Concatenate(axis=-1)([p_1, p_2])
    x = trunk_block(x, name='trunk')

    model = Model(inputs, x, name='two_path_model')

    return model

def dual_autoencoder(
    input_shape, 
    arch_config, 
    output_horizon, 
    observed_channels, 
    forecast_channels
):
    from tensorflow.keras.layers import (
        Input, Conv2D, Conv3D, ConvLSTM2D, Dense, Flatten, 
        Reshape, UpSampling2D, Lambda, concatenate,
        Add
    )
    from tensorflow.keras.models import Model
    from libs.layers import ChannelSplitLayer

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
        return tf.tile(pos, [batch_size, 1, in_h, in_w, 1])
    
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
            x = Add()([x ,past_skip, fcast_skip])
        x = UpSampling2D(size=(2,2), interpolation='bilinear', name=f'dec_up{len(spatial_filters)}')(x)
        x = Conv2D(temporal_filters[2], (3,3), padding='same', activation='relu', name=f'dec_d{len(spatial_filters)}')(x)
        return x

    def encoder_arm_block(input_shape, temporal_filters, name='encoder_arm_block'):
        _b, t, h, w, c = input_shape
        filters_1, filters_2, filters_3 = temporal_filters

        inputs = Input(shape=(t, h, w, c))
        x_1 = Conv3D(filters=filters_1, kernel_size=(1,1,1), padding='same', name='past_proj')(inputs)
        x_2 = ConvLSTM2D(filters=filters_1, kernel_size=(3,3), padding='same', return_sequences=True, name='past_lstm1')(x_1)
        x_out = Add()([x_2, x_1])

        y_1 = ConvLSTM2D(temporal_filters[1], (3,3), padding='same', return_sequences=True, name='past_lstm2')(x_out)
        y_2 = Conv3D(temporal_filters[1], (1,1,1), padding='same', name='past_res1_proj')(x_out)
        y_out = Add()([y_1, y_2])

        z_1, hidden_state, cell_state = ConvLSTM2D(temporal_filters[2], (3,3), padding='same', return_state=True, name='past_lstm3')(y_out)
        z_2 = Conv2D(temporal_filters[2], (1,1), padding='same', name='past_res2_proj')(
            Lambda(lambda t: tf.gather(t, indices=tf.shape(t)[1] - 1, axis=1), output_shape=(h, w, c), name='gather_last_frame')(y_out)
        )
        out = Add()([z_1, z_2])

        return Model(inputs, outputs=[out, hidden_state, cell_state], name=name)

    inputs = Input(shape=input_shape)
    in_t, in_h, in_w, in_c = input_shape
    
    print(f"Input: {input_shape} -> Output: ({output_horizon}, {in_h}, {in_w}, 1)")
    
    observed_in = ChannelSplitLayer(observed_channels)(inputs) 

    past_state, h, c = encoder_arm_block(
        input_shape=observed_in.shape,
        temporal_filters=temporal_filters,
        name='current_encoder_arm_block'
    )(observed_in)
    
    past_encs = spatial_encoder(past_state, 'past')
    
    forecast_in = ChannelSplitLayer(forecast_channels)(inputs)
    fcast_proj = Conv3D(temporal_filters[0], (1,1,1), padding='same', name='fcast_proj')(forecast_in)
    fcast_lstm1 = ConvLSTM2D(temporal_filters[0], (3,3), padding='same', return_sequences=True, name='fcast_lstm1')(fcast_proj)
    fcast_lstm1 = Add()([fcast_lstm1, fcast_proj])
    fcast_lstm2 = ConvLSTM2D(temporal_filters[1], (3,3), padding='same', return_sequences=True, name='fcast_lstm2')(fcast_lstm1)
    fcast_lstm1_proj = Conv3D(temporal_filters[1], (1,1,1), padding='same', name='fcast_res1_proj')(fcast_lstm1)
    fcast_lstm2 = Add()([fcast_lstm2, fcast_lstm1_proj])
    fcast_state = fcast_lstm2[:, -1]
    fcast_state = Conv2D(temporal_filters[2], (1,1), padding='same', activation='relu', name='fcast_out_proj')(fcast_state)
    
    fcast_encs = spatial_encoder(fcast_state, 'fcast')
    
    fused = concatenate([past_encs[-1], fcast_encs[-1]], axis=-1, name='bn_concat')
    fused = Flatten(name='bn_flat')(fused)
    fused = Dense(bottleneck, activation='relu', name='bn_dense')(fused)
    
    dec = Dense(latent_size * latent_size * spatial_filters[-1], activation='relu', name='dec_proj')(fused)
    dec = Reshape((latent_size, latent_size, spatial_filters[-1]), name='dec_reshape')(dec)
    
    dec = spatial_decoder(dec, past_encs, fcast_encs)
    
    dec_tiled = Lambda(tile_for_decoder, output_shape=(in_t, in_w, in_h, spatial_filters[0]), name='dec_tile')(dec)
    pos_enc = Lambda(create_pos_encoding, output_shape=(in_t, in_w, in_h, 1), name='pos_enc')(dec_tiled)
    dec_input = concatenate([dec_tiled, pos_enc], axis=-1, name='dec_input')
    
    dec_lstm1 = ConvLSTM2D(temporal_filters[2], (3,3), padding='same', return_sequences=True, name='dec_lstm1')(
        dec_input, initial_state=[h, c]
    )
    enc_tiled = Lambda(tile_for_decoder, output_shape=(in_t, in_w, in_h, spatial_filters[0]), name='enc_tile')(dec)
    dec_lstm1 = Add()([dec_lstm1, enc_tiled])
    
    dec_lstm2 = ConvLSTM2D(temporal_filters[2], (3,3), padding='same', return_sequences=True, name='dec_lstm2')(dec_lstm1)
    dec_lstm2 = Add()([dec_lstm2, dec_lstm1])
    
    output = Conv3D(1, (1,1,1), padding='same', activation='relu', name='output')(dec_lstm2)
    
    model = Model(inputs, output)
    print(f"Architecture: {latent_size}x{latent_size} | Parameters: {model.count_params():,}")
    return model
