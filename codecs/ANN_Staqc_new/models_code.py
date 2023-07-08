import os
import logging
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
import pickle
from concactLayer import concatLayer
from mediumlayer import MediumLayer
from attention_layer import AttentionLayer
from MultiHeadAttention import MultiHeadAttention_
from LayerNormalization import LayerNormalization
from Position_Embedding import Position_Embedding
from PositionWiseFeedForward import PositionWiseFeedForward

tf.compat.v1.disable_eager_execution()
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
logger = logging.getLogger(__name__)

'''
变体：模型输入只要代码和查询描述
'''
class CodeMF:
    def __init__(self, config):
        self.config = config
        self.text_length = 100
        self.queries_length = 25
        self.code_length = 350
        self.class_model = None
        self.train_model = None
        self.text_s1 = Input(shape=(self.text_length,), dtype='int32', name='input_s1')
        self.text_s2 = Input(shape=(self.text_length,), dtype='int32', name='input_s2')
        self.code = Input(shape=(self.code_length,), dtype='int32', name='input_code')
        self.queries = Input(shape=(self.queries_length,), dtype='int32', name='input_queries')
        self.labels = Input(shape=(1,), dtype='int32', name='input_labels')
        self.nb_classes = 2
        self.dropout = None

        self.model_params = config.get('model_params', dict())
        self.data_params = config.get('data_params', dict())
        self.text_embedding = pickle.load(open(self.data_params['text_pretrain_emb_path'], "rb"), encoding='iso-8859-1')
        self.code_embedding = pickle.load(open(self.data_params['code_pretrain_emb_path'], "rb"), encoding='iso-8859-1')
        # Create a model path to store model info
        model_dir = self.config['workdir'] + 'models/' + self.model_params['model_name'] + '/'
        os.makedirs(model_dir, exist_ok=True)

        self.nb_classes = 2
        self.dropout1 = None
        self.dropout2 = None
        self.dropout3 = None
        self.dropout4 = None
        self.dropout5 = None
        self.regularizer = None
        self.random_seed = None
        self.num = None

    def params_adjust(self, dropout1=0.5, dropout2=0.5, dropout3=0.5, dropout4=0.5, dropout5=0.5, regularizer=0.01, num=100, seed=42):
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.dropout3 = dropout3
        self.dropout4 = dropout4
        self.dropout5 = dropout5
        self.regularizer = regularizer
        self.random_seed = seed
        self.num = num

    def build(self):
        '''
        1. Build Code Representation Model
        '''
        logger.debug('Building Code Representation Model')

        '''
        2. Embedding
        '''
        embedding_layer = Embedding(self.text_embedding.shape[0], self.text_embedding.shape[1],
                                    weights=[self.text_embedding], input_length=self.text_length,
                                    trainable=False, mask_zero=True)

        text_s1_embedding = embedding_layer(self.text_s1)
        text_s2_embedding = embedding_layer(self.text_s2)

        '''
        3. Position Embedding
        '''
        position_embedding = Position_Embedding(10, 'concat')
        text_s1_embedding_p = position_embedding(text_s1_embedding)
        text_s2_embedding_p = position_embedding(text_s2_embedding)

        '''
        4. Dropout
        '''
        dropout_embedding = Dropout(self.dropout1, name='dropout_embed', seed=self.random_seed)
        text_s1_embedding_d = dropout_embedding(text_s1_embedding_p)
        text_s2_embedding_d = dropout_embedding(text_s2_embedding_p)

        '''
        5. Transformer
        '''
        multihead_attention = MultiHeadAttention_(10)
        t1 = multihead_attention([text_s1_embedding_d, text_s1_embedding_d, text_s1_embedding_d])
        t2 = multihead_attention([text_s2_embedding_d, text_s2_embedding_d, text_s2_embedding_d])

        add_out = Lambda(lambda x: x[0] + x[1])
        t1 = add_out([t1, text_s1_embedding_d])
        t2 = add_out([t2, text_s2_embedding_d])

        t1_l = LayerNormalization()(t1)
        t2_l = LayerNormalization()(t2)

        positionwise_feedforward = PositionWiseFeedForward(310, 2048)
        ff_t1 = positionwise_feedforward(t1_l)
        ff_t2 = positionwise_feedforward(t2_l)

        dropout_ffn = Dropout(self.dropout2, name='dropout_ffn', seed=self.random_seed)
        ff_t1 = dropout_ffn(ff_t1)
        ff_t2 = dropout_ffn(ff_t2)

        ff_t1 = add_out([ff_t1, t1_l])
        ff_t2 = add_out([ff_t2, t2_l])

        t1 = LayerNormalization()(ff_t1)
        t2 = LayerNormalization()(ff_t2)

        '''
        6. Fuse Code and Context Semantics
        '''
        dropout_qc = Dropout(self.dropout3, name='dropout_qc', seed=self.random_seed)
        leaky_relu = Lambda(lambda x: tf.nn.leaky_relu(x))
        text_s1_semantic = GlobalAveragePooling1D(name='globaltext_1')(t1)
        text_s1_semantic = leaky_relu(text_s1_semantic)
        text_s2_semantic = GlobalAveragePooling1D(name='globaltext_2')(t2)
        text_s2_semantic = leaky_relu(text_s2_semantic)

        '''
        7. Fuse Semantic Representations
        '''
        sentence_token_level_outputs = MediumLayer()([text_s1_semantic, text_s2_semantic])
        bidirectional_gru = Bidirectional(GRU(units=128, dropout=self.dropout4))
        f1 = bidirectional_gru(sentence_token_level_outputs)
        dropout_classification = Dropout(self.dropout5, name='dropout_classification', seed=self.random_seed)
        f1 = dropout_classification(f1)

        '''
        8. Classification
        '''
        classf = Dense(2, activation='softmax', name="final_class", kernel_regularizer=regularizers.l2(self.regularizer))(f1)

        class_model = Model(
            inputs=[self.text_s1, self.text_s2, self.code, self.queries],
            outputs=[classf],
            name='class_model'
        )
        self.class_model = class_model

        print("\nSummary of the class model:")
        self.class_model.summary()
        fname = self.config['workdir'] + 'models/' + self.model_params['model_name'] + '/_class_model.png'
        P1, P2, Pc, Pq = None,None,None,None
        myloss = self.dice_loss(P1, P2, Pc, Pq)
        optimizer = Adam(learning_rate=0.001, clipnorm=0.001)
        self.class_model.compile(loss=myloss, optimizer=optimizer)

    def compile(self, optimizer, **kwargs):
        logger.info('compiling models')
        '''
        model_dice = self.dice_loss(smooth=1e-5, thresh=0.5)
        model.compile(loss=model_dice)
        '''
        self.class_model.compile(loss=self.example_loss, optimizer=optimizer, **kwargs)

    def fit(self, x, y, **kwargs):
        assert self.class_model is not None, 'Must compile the model before fitting data'
        return self.class_model.fit(x, to_categorical(y), **kwargs)


    def predict(self, x, **kwargs):
        return self.class_model.predict(x, **kwargs)

    def save(self, class_model_file, **kwargs):
        assert self.class_model is not None, 'Must compile the model before saving weights'
        self.class_model.save_weights(class_model_file, **kwargs)

    def load(self, class_model_file, **kwargs):
        assert self.class_model is not None, 'Must compile the model loading weights'
        self.class_model.load_weights(class_model_file, **kwargs)

    def concat(self, inputs):
        block_level_code_output = tf.split(inputs, inputs.shape[1], axis=1)
        block_level_code_output = tf.concat(block_level_code_output, axis=2)
        # (bs,600)
        block_level_code_output = tf.squeeze(block_level_code_output, axis=1)
        return block_level_code_output

    def my_crossentropy(self, y_true, y_pred, e=0.1):
        loss1 = K.categorical_crossentropy(y_true, y_pred)
        loss2 = K.categorical_crossentropy(K.ones_like(y_pred) / self.nb_classes, y_pred)
        return (1 - e) * loss1 + e * loss2

    def example_loss(self, y_true, y_pred):
        crossent = tf.compat.v1.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
        # crossent = K.categorical_crossentropy(y_true, y_pred)
        loss = tf.reduce_sum(crossent) / tf.cast(100, tf.float32)
        print("========", loss.shape)
        return loss

    def dice_coef(self, y_true, y_pred, p1, p2, p3, p4, e=0.1):
        #P_loss = (p1 + p2 + p3 + p4) / 4

        loss1 = K.categorical_crossentropy(y_true, y_pred)
        loss2 = K.categorical_crossentropy(K.ones_like(y_pred) / self.nb_classes, y_pred)
        return (1 - e) * loss1 + e * loss2 #+ 0.001 * P_loss

    def dice_loss(self, p1, p2, p3, p4):
        def dice(y_true, y_pred):
            return self.dice_coef(y_true, y_pred, p1, p2, p3, p4)

        return dice











