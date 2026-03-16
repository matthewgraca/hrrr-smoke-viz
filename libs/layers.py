import tensorflow as tf
from keras.saving import register_keras_serializable
from keras import layers

@register_keras_serializable()
class ChannelSplitLayer(layers.Layer):
    # uses channel indices to select particular channels
    def __init__(self, channel_idxs, **kwargs):
        super().__init__(**kwargs)
        self.idxs = channel_idxs

    def call(self, x):
        return tf.gather(x, self.idxs, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({'channel_idxs' : self.idxs})
        return config

@register_keras_serializable()
class TemporalCompressionLayer(layers.Layer):
    # selects the data in the final time channel
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return x[:, -1, ...]

    def get_config(self):
        config = super().get_config()
        return config

@register_keras_serializable()
class TileAcrossHorizonLayer(layers.Layer):
    # tiles a single tensor across a given horizon
    def __init__(self, output_horizon, **kwargs):
        super().__init__(**kwargs)
        self.output_horizon = output_horizon

    def call(self, x):
        return tf.tile(
            tf.expand_dims(x, axis=1),
            multiples=[1, self.output_horizon, 1, 1, 1]
        )

    def get_config(self):
        config = super().get_config()
        config.update({"output_horizon": self.output_horizon})
        return config

@register_keras_serializable()
class PositionalEncoding3DLayer(layers.Layer):
    # generate positional encodings, per frame, for each timestep
    def __init__(self, output_horizon, **kwargs):
        super().__init__(**kwargs)
        self.output_horizon = output_horizon

    def call(self, x):
        return tf.tile(
            input=tf.reshape(
                tensor=(
                    tf.range(self.output_horizon, dtype=tf.float32) / 
                    tf.cast(self.output_horizon, tf.float32)
                ),
                shape=[1, self.output_horizon, 1, 1, 1]
            ),
            multiples=[tf.shape(x)[0], 1, tf.shape(x)[2], tf.shape(x)[3], 1]
        )

    def get_config(self):
        config = super().get_config()
        config.update({"output_horizon": self.output_horizon})
        return config
