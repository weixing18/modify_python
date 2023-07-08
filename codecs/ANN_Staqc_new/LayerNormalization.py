from tensorflow.keras.layers import Layer
import tensorflow as tf


class LayerNormalization(Layer):
    def __init__(self, epsilon=1e-8, **kwargs):
        self.epsilon = epsilon
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.beta = self.add_weight(
            shape=(input_shape[-1],),
            initializer='zero',
            name='beta'
        )
        self.gamma = self.add_weight(
            shape=(input_shape[-1],),
            initializer='one',
            name='gamma'
        )
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
        normalized = (inputs - mean) / ((variance + self.epsilon) ** 0.5)
        outputs = self.gamma * normalized + self.beta
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape
