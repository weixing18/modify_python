import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

class AttentionLayer(Layer):
    """
    自定义的注意力层，用于增强模型的注意力机制。
    """

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        构建注意力层，通过初始化核心权重来实现。

        参数:
            input_shape: 输入张量的形状，应为一个包含两个输入形状的列表。
        """
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('An attention layer should be called '
                             'on a list of 2 inputs.')
        if input_shape[0][2] != input_shape[1][2]:
            raise ValueError('Embedding sizes should be of the same size')

        # 初始化核心权重
        self.kernel = self.add_weight(shape=(input_shape[0][2], input_shape[0][2]),         
                                      initializer='glorot_uniform',
                                      name='kernel',
                                      trainable=True)

        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        """
        定义注意力层的前向传播逻辑。

        参数:
            inputs: 一个包含两个输入张量的列表。

        返回:
            经过注意力计算后的张量。
        """

        a = K.dot(inputs[0], self.kernel)  # 执行输入张量和核心权重的矩阵相乘
        y_trans = K.permute_dimensions(inputs[1], (0, 2, 1))  # 调整第二个输入张量的维度顺序
        b = K.batch_dot(a, y_trans, axes=[2, 1])  # 执行批次矩阵相乘
        return K.tanh(b)

    def compute_output_shape(self, input_shape):
        """
        计算输出张量的形状。

        参数:
            input_shape: 输入张量的形状，应为一个包含两个输入形状的列表。

        返回:
            输出张量的形状。
        """
        return (None, input_shape[0][1], input_shape[1][1])
