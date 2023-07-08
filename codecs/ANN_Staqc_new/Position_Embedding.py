import tensorflow as tf
import numpy as np

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

class PositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, size=None, mode='sum', **kwargs):
        """
        Position Embedding layer.

        Args:
            size (int): Length of the position embedding vector. Must be even.
                        If not provided, it will be inferred from the input shape.
            mode (str): Mode of combining the position embedding with the input.
                        'sum' or 'concat'. Default is 'sum'.
            **kwargs: Additional keyword arguments for the base class.
        """
        self.size = size
        self.mode = mode
        super(PositionEmbedding, self).__init__(**kwargs)

    def call(self, x):
        if (self.size is None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])
        batch_size, seq_len = tf.shape(x)[0], tf.shape(x)[1]
        position_j = 1. / tf.pow(10000., 2 * tf.range(self.size / 2, dtype='float32') / self.size)
        position_j = tf.expand_dims(position_j, 0)
        position_i = tf.cumsum(tf.ones_like(x[:, :, 0]), 1) - 1
        position_i = tf.expand_dims(position_i, 2)
        position_ij = tf.matmul(position_i, position_j)
        position_ij_2i = tf.sin(position_ij)[..., tf.newaxis]
        position_ij_2i_1 = tf.cos(position_ij)[..., tf.newaxis]
        position_ij = tf.concat([position_ij_2i, position_ij_2i_1], axis=-1)
        position_ij = tf.reshape(position_ij, (batch_size, seq_len, self.size))
        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return tf.concat([position_ij, x], axis=-1)

    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)



'''
query = tf.random.truncated_normal([100, 50, 150])
w = Position_Embedding(150,'concat')(query)
print(w.shape)
'''