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

    model = Sequential()
    model.add(InputLayer(input_shape))
    model.add(ConvLSTM2D(filters=25, **CONVLSTM2D_ARGS))
    model.add(ConvLSTM2D(filters=50, **CONVLSTM2D_ARGS))
    model.add(Conv3D(filters=25, **CONV3D_ARGS))
    model.add(Conv3D(filters=1, **CONV3D_ARGS))

    return model

def stateful_classic(input_shape, batch_size):
    from keras.models import Sequential
    from keras.layers import ConvLSTM2D, Conv3D, InputLayer
    from keras.optimizers import Adam

    # common args in each layer to reduce function call length
    CONVLSTM2D_ARGS = {
        'kernel_size': (3, 3),
        'padding': 'same',
        'return_sequences': True,
        'stateful': True
    }

    CONV3D_ARGS = {
        'kernel_size': (3, 3, 3),
        'activation': 'relu',
        'padding': 'same'    
    }

    model = Sequential()
    model.add(InputLayer(shape=input_shape, batch_size=batch_size))
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
    convlstm_reg_config,
    output_horizon, 
    observed_channels, 
    forecast_channels,
    stateful=False,
    batch_size=None
):
    import tensorflow as tf
    from tensorflow.keras.layers import (
        Input, Conv2D, Conv3D, ConvLSTM2D, Dense, Flatten, 
        Reshape, UpSampling2D, Lambda, concatenate,
        Add
    )
    from tensorflow.keras.models import Model
    from libs.layers import ChannelSplitLayer

    def spatial_decoder(x, past_encs, fcast_encs, spatial_filters, temporal_filters):
        '''
        Super tough to wrap this in a Model since tf really doesn't like the passing of 
        past_encs and fcast_encs, they'll need to be their own Input, etc.

        If the goal is a cleaner plot_model(), the juice is not worth the squeeze
        '''
        skips = list(zip(reversed(past_encs[:-1]), reversed(fcast_encs[:-1])))
        for i, (past_skip, fcast_skip) in enumerate(skips):
            x = UpSampling2D(
                size=(2,2), 
                interpolation='bilinear', 
                name=f'dec_up{i+1}'
            )(x)
            x = Conv2D(
                filters=spatial_filters[-(i+2)],
                kernel_size=(3,3), 
                padding='same', 
                activation='relu', 
                name=f'dec_d{i+1}'
            )(x)
            x = Add(name=f'e_{2-i}_skip')([x, past_skip, fcast_skip])
        x = UpSampling2D(size=(2,2), interpolation='bilinear', name=f'dec_up{len(spatial_filters)}')(x)
        x = Conv2D(temporal_filters[2], (3,3), padding='same', activation='relu', name=f'dec_d{len(spatial_filters)}')(x)
        return x

    def arm_block(
        input_shape, 
        filters, 
        forecast_arm,
        stateful_args
    ):
        '''
        Two return options:
            - forecast_arm false: uses past arm architecture,
                returns hidden and cell state for skip connection
            - forecast_arm true: uses future arm architecture, 
                no hidden/cell state returned
        '''
        b, t, h, w, c = input_shape
        prefix = 'fcast' if forecast_arm else 'past'

        # block 1
        inputs = Input(shape=(t, h, w, c), batch_size=b, name=f'{prefix}_inputs')
        x_1 = Conv3D(
            filters=filters[0], 
            kernel_size=(1,1,1),
            padding='same', 
            name=f'{prefix}_proj'
        )(inputs)
        x_2 = ConvLSTM2D(
            filters=filters[0], 
            kernel_size=(3,3),
            padding='same', 
            return_sequences=True,
            name=f'{prefix}_lstm1',
            **stateful_args
        )(x_1)
        x_out = Add(name=f'skip_{prefix}_lstm1')([x_2, x_1])

        # block 2
        y_1 = ConvLSTM2D(
            filters=filters[1],
            kernel_size=(3,3),
            padding='same',
            return_sequences=True,
            name=f'{prefix}_lstm2',
            **stateful_args
        )(x_out)
        y_2 = Conv3D(
            filters=filters[1],
            kernel_size=(1,1,1),
            padding='same', 
            name=f'{prefix}_res1_proj'
        )(x_out)
        y_out = Add(name=f'skip_{prefix}_res1_proj')([y_1, y_2])

        # block 3
        # past arm supports a 3rd convlstm layer that returns its hidden and cell state
        # also expands filters
        out, hidden_state, cell_state = (
            (y_out, None, None)
            if forecast_arm
            else ConvLSTM2D(
                filters=filters[2],
                kernel_size=(3,3),
                padding='same', 
                return_sequences=True,
                return_state=True, 
                name=f'{prefix}_lstm3',
                **stateful_args
            )(y_out)
        )

        return Model(
            inputs, 
            outputs=out if forecast_arm else [out, hidden_state, cell_state],
            name=f'{prefix}_arm'
        )

    def encoder_block(input_shape, t_filters, s_filters, strides, prefix):
        b, t, h, w, c = input_shape
        inputs = Input(shape=(t, h, w, c), batch_size=b)

        # temporal compression
        x = Lambda(
            function=lambda tensor: tf.gather(tensor, indices=tf.shape(tensor)[1] - 1, axis=1),
            output_shape=(h, w, c),
            name=f'{prefix}_gather_last_frame'
        )(inputs)
        x = Conv2D(
            filters=t_filters[2],
            kernel_size=(1,1), 
            padding='same', 
            name=f'{prefix}_res2_proj'
        )(x)

        # spatial compression
        encs = []
        for i, filters in enumerate(s_filters):
            x = Conv2D(
                filters=filters, 
                kernel_size=(3,3),
                strides=strides[i], 
                padding='same', 
                activation='relu', 
                name=f'{prefix}_e{i+1}'
            )(x)
            encs.append(x)
        return Model(inputs, outputs=[x, *encs], name=f'{prefix}_encoder')

    def bottleneck_block(input_shape, latent_size, bottleneck, spatial_filters):
        # aka latent space fusion
        b, h, w, c = input_shape
        inputs = Input(shape=(h, w, c), batch_size=b, name='fused_encodings')
        
        x = Flatten(name='bn_flat')(inputs)
        x = Dense(bottleneck, activation='relu', name='bn_dense')(x)
        x = Dense(
            units=latent_size * latent_size * spatial_filters[-1],
            activation='relu',
            name='bn_proj'
        )(x)
        x = Reshape(
            target_shape=(latent_size, latent_size, spatial_filters[-1]),
            name='bn_reshape'
        )(x)
    
        return Model(inputs, outputs=x, name='bottleneck')

    def temporal_decoder(input_shapes, in_t, output_horizon, spatial_filters, temporal_filters, stateful_args):
        def create_pos_encoding(x, output_horizon):
            pos = tf.reshape(
                tensor=tf.range(output_horizon, dtype=tf.float32) / output_horizon,
                shape=[1, output_horizon, 1, 1, 1]
            )
            return tf.tile(
                input=pos,
                multiples=[tf.shape(x)[0], 1, tf.shape(x)[2], tf.shape(x)[3], 1]
            )
        def tile(x, output_horizon):
            return tf.tile(
                input=tf.expand_dims(x, axis=1),
                multiples=[1, output_horizon, 1, 1, 1]
            )
    
        b, h, w, _c = input_shapes[0]
        inputs = Input(shape=input_shapes[0][1:], batch_size=b)
        hidden_state = Input(shape=input_shapes[1][1:], batch_size=b, name='past_hidden_skip')
        cell_state = Input(shape=input_shapes[2][1:], batch_size=b, name='past_cell_skip')

        x_1 = Lambda(
            function=lambda t : tile(t, output_horizon),
            output_shape=(in_t, w, h, spatial_filters[0]),
            name='dec_tile'
        )(inputs)
        x_2 = Lambda(
            function=lambda t: create_pos_encoding(t, output_horizon),
            output_shape=(in_t, w, h, 1),
            name='pos_enc'
        )(x_1)
        y = concatenate([x_1, x_2], axis=-1, name='dec_input')
        
        y = ConvLSTM2D(
            filters=temporal_filters[2],
            kernel_size=(3,3), 
            padding='same', 
            return_sequences=True, 
            name='dec_lstm1',
            **stateful_args
        )(
            y, initial_state=[hidden_state, cell_state]
        )
        y = Add(name='skip_pos_enc_and_dec_lstm1')([y, x_1])
        
        z = ConvLSTM2D(
            filters=temporal_filters[2], 
            kernel_size=(3,3),
            padding='same', 
            return_sequences=True, 
            name='dec_lstm2',
            **stateful_args
        )(y)
        z = Add(name='skip_dec_lstm2')([z, y])

        return Model(
            inputs=[inputs, hidden_state, cell_state],
            outputs=z, 
            name='temporal_decoder'
        )
    
    temporal_filters = arch_config['temporal_filters']
    spatial_filters = arch_config['spatial_filters']
    bottleneck = arch_config['bottleneck']
    latent_size = arch_config['latent_size']
    strides = arch_config['strides']

    in_t, in_h, in_w, in_c = input_shape 
    inputs = Input(shape=input_shape, batch_size=batch_size)

    stateful_args = {
        'stateful': True,
        #'batch_input_shape': (batch_size, output_horizon, in_h, in_w, in_c)
    } if stateful else {}
    
    print(f"Input: {input_shape} -> Output: ({output_horizon}, {in_h}, {in_w}, 1)")
    
    # past encoder arm
    past_in = ChannelSplitLayer(observed_channels, name='past_channels')(inputs)
    past_state, h, c = arm_block(
        input_shape=past_in.shape,
        filters=temporal_filters,
        forecast_arm=False,
        stateful_args=stateful_args
    )(past_in)
    past_enc, *past_encs = encoder_block(
        past_state.shape, temporal_filters, spatial_filters, strides, 'past'
    )(past_state)

    # fcast encoder arm
    fcast_in = ChannelSplitLayer(forecast_channels, name='fcast_channels')(inputs)
    fcast_state = arm_block(
        input_shape=fcast_in.shape,
        filters=temporal_filters,
        forecast_arm=True,
        stateful_args=stateful_args
    )(fcast_in)
    fcast_enc, *fcast_encs = encoder_block(
        fcast_state.shape, temporal_filters, spatial_filters, strides, 'fcast'
    )(fcast_state)
    
    # bottleneck/latent space fusion
    fused = concatenate([past_encs[-1], fcast_encs[-1]], axis=-1, name='bn_concat')
    fused = bottleneck_block(fused.shape, latent_size, bottleneck, spatial_filters)(fused)
    
    dec = spatial_decoder(fused, past_encs, fcast_encs, spatial_filters, temporal_filters)
    x = temporal_decoder([dec.shape, h.shape, c.shape], in_t, output_horizon, spatial_filters, temporal_filters, stateful_args)([dec, h, c])

    output = Conv3D(1, (1,1,1), padding='same', activation='relu', name='output')(x)
    
    model = Model(inputs, output)
    print(f"Architecture: {latent_size}x{latent_size} | Parameters: {model.count_params():,}")
    return model
