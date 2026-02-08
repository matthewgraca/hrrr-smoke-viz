import keras
import tensorflow as tf

@keras.utils.register_keras_serializable(package="Custom")
class ChannelSplitLayer(keras.layers.Layer):
    def __init__(self, channel_idxs, **kwargs):
        super().__init__(**kwargs)
        self.idxs = channel_idxs

    def call(self, inputs):
        return tf.gather(inputs, self.idxs, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({'channel_idxs' : self.idxs})
        return config
