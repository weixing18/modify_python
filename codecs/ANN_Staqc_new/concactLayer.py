import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.layers import Layer

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)

class ConcatLayer(Layer):
    """
    自定义的连接层。
    """

    def __init__(self, **kwargs):
        super(ConcatLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        构建连接层。

        参数:
            input_shape: 输入张量的形状。
        """
        super(ConcatLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """
        定义连接层的前向传播。

        参数:
            inputs: 输入张量。
            kwargs: 关键字参数。
        """
        block_level_code_output = tf.split(inputs, inputs.shape[1], axis=1)
        block_level_code_output = tf.concat(block_level_code_output, axis=2)
        block_level_code_output = tf.squeeze(block_level_code_output, axis=1)
        print(block_level_code_output)
        return block_level_code_output

    def compute_output_shape(self, input_shape):
        print("===========================", input_shape)
        return (input_shape[0], input_shape[1] * input_shape[2])
