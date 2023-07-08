import tensorflow as tf
import numpy as np

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)


class ScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(self, return_attention=False, history_only=False, **kwargs):
        """
        Scaled Dot-Product Attention layer.

        Args:
            return_attention (bool): Whether to return attention weights.
            history_only (bool): Whether to only use history data.
            **kwargs: Additional keyword arguments for the base class.
        """
        super(ScaledDotProductAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.return_attention = return_attention
        self.history_only = history_only
        self.intensity = None
        self.attention = None

    def get_config(self):
        config = {
            'return_attention': self.return_attention,
            'history_only': self.history_only,
        }
        base_config = super(ScaledDotProductAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            query_shape, key_shape, value_shape = input_shape
        else:
            query_shape = key_shape = value_shape = input_shape
        output_shape = query_shape[:-1] + value_shape[-1:]
        if self.return_attention:
            attention_shape = query_shape[:2] + (key_shape[1],)
            return [output_shape, attention_shape]
        return output_shape

    def compute_mask(self, inputs, mask=None):
        if isinstance(mask, list):
            mask = mask[0]
        if self.return_attention:
            return [mask, None]
        return mask

    def call(self, inputs, mask=None, **kwargs):
        if isinstance(inputs, list):
            query, key, value = inputs
        else:
            query = key = value = inputs
        if isinstance(mask, list):
            mask = mask[1]
        feature_dim = tf.shape(query)[-1]
        e = tf.matmul(query, tf.transpose(key, perm=[0, 2, 1])) / tf.sqrt(tf.cast(feature_dim, dtype=tf.float32))
        if self.history_only:
            query_len, key_len = tf.shape(query)[1], tf.shape(key)[1]
            indices = tf.expand_dims(tf.range(0, key_len), axis=0)
            upper = tf.expand_dims(tf.range(0, query_len), axis=-1)
            e -= 10000.0 * tf.expand_dims(tf.cast(indices > upper, tf.float32), axis=0)
        if mask is not None:
            e -= 10000.0 * (1.0 - tf.cast(tf.expand_dims(mask, axis=-2), tf.float32))
        self.intensity = e
        e = tf.exp(e - tf.reduce_max(e, axis=-1, keepdims=True))
        self.attention = e / tf.reduce_sum(e, axis=-1, keepdims=True)
        v = tf.matmul(self.attention, value)
        if self.return_attention:
            return [v, self.attention]
        return v


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self,
                 head_num,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 history_only=False,
                 **kwargs):
        """
        Multi-Head Attention layer.

        Args:
            head_num (int): Number of attention heads.
            activation (str): Activation function for linear mappings.
            use_bias (bool): Whether to use bias terms.
            kernel_initializer (str or callable): Initializer for linear mappings.
            bias_initializer (str or callable): Initializer for linear mappings.
            kernel_regularizer (str or callable): Regularizer for linear mappings.
            bias_regularizer (str or callable): Regularizer for linear mappings.
            kernel_constraint (str or callable): Constraints for linear mappings.
            bias_constraint (str or callable): Constraints for linear mappings.
            history_only (bool): Whether to only use history in attention layer.
            **kwargs: Additional keyword arguments for the base class.
        """
        self.supports_masking = True
        self.head_num = head_num
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.history_only = history_only

        self.Wq = self.Wk = self.Wv = self.Wo = None
        self.bq = self.bk = self.bv = self.bo = None
        self.intensity = self.attention = None

        super(MultiHeadAttention, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'head_num': self.head_num,
            'activation': tf.keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': tf.keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': tf.keras.constraints.serialize(self.bias_constraint),
            'history_only': self.history_only,
        }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            q, k, v = input_shape
            return q[:-1] + (v[-1],)
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        if isinstance(input_mask, list):
            return input_mask[0]
        return input_mask

    def build(self, input_shape):
        if isinstance(input_shape, list):
            q, k, v = input_shape
        else:
            q = k = v = input_shape
        feature_dim = int(v[-1])
        if feature_dim % self.head_num != 0:
            raise IndexError('Invalid head number %d with the given input dim %d' % (self.head_num, feature_dim))
        self.Wq = self.add_weight(
            shape=(int(q[-1]), feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wq' % self.name,
        )
        if self.use_bias:
            self.bq = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bq' % self.name,
            )
        self.Wk = self.add_weight(
            shape=(int(k[-1]), feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wk' % self.name,
        )
        if self.use_bias:
            self.bk = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bk' % self.name,
            )
        self.Wv = self.add_weight(
            shape=(int(v[-1]), feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wv' % self.name,
        )
        if self.use_bias:
            self.bv = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bv' % self.name,
            )
        self.Wo = self.add_weight(
            shape=(feature_dim, feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wo' % self.name,
        )
        if self.use_bias:
            self.bo = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bo' % self.name,
            )
        super(MultiHeadAttention, self).build(input_shape)

    @staticmethod
    def _reshape_to_batches(x, head_num):
        input_shape = tf.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        head_dim = feature_dim // head_num
        x = tf.reshape(x, (batch_size, seq_len, head_num, head_dim))
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(x, (batch_size * head_num, seq_len, head_dim))

    @staticmethod
    def _reshape_attention_from_batches(x, head_num):
        input_shape = tf.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        x = tf.reshape(x, (batch_size // head_num, head_num, seq_len, feature_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    @staticmethod
    def _reshape_from_batches(x, head_num):
        input_shape = tf.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        x = tf.reshape(x, (batch_size // head_num, head_num, seq_len, feature_dim))
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(x, (batch_size // head_num, seq_len, feature_dim * head_num))

    @staticmethod
    def _reshape_mask(mask, head_num):
        if mask is None:
            return mask
        seq_len = tf.shape(mask)[1]
        mask = tf.expand_dims(mask, axis=1)
        mask = tf.tile(mask, [1, head_num, 1])
        return tf.reshape(mask, (-1, seq_len))

    def call(self, inputs, mask=None):
        if isinstance(inputs, list):
            q, k, v = inputs
        else:
            q = k = v = inputs
        if isinstance(mask, list):
            q_mask, k_mask, v_mask = mask
        else:
            q_mask = k_mask = v_mask = mask
        q = tf.matmul(q, self.Wq)
        k = tf.matmul(k, self.Wk)
        v = tf.matmul(v, self.Wv)
        if self.use_bias:
            q += self.bq
            k += self.bk
            v += self.bv
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)
        scaled_dot_product_attention = ScaledDotProductAttention(
            history_only=self.history_only,
            name='%s-Attention' % self.name,
        )
        y = scaled_dot_product_attention(
            inputs=[
                self._reshape_to_batches(q, self.head_num),
                self._reshape_to_batches(k, self.head_num),
                self._reshape_to_batches(v, self.head_num),
            ],
            mask=[
                self._reshape_mask(q_mask, self.head_num),
                self._reshape_mask(k_mask, self.head_num),
                self._reshape_mask(v_mask, self.head_num),
            ],
        )
        y = self._reshape_from_batches(y, self.head_num)
        y = tf.matmul(y, self.Wo)
        if self.use_bias:
            y += self.bo
        if self.activation is not None:
            y = self.activation(y)

        input_shape = [tf.keras.backend.int_shape(q), tf.keras.backend.int_shape(k), tf.keras.backend.int_shape(v)]
        output_shape = self.compute_output_shape(input_shape)
        if output_shape[1] is not None:
            output_shape = (-1,) + output_shape[1:]
            y = tf.reshape(y, output_shape)
        return y
