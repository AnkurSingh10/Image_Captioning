import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    Dense,
    Embedding,
    Dropout,
    LayerNormalization,
    MultiHeadAttention,
    Layer
)
from keras.saving import register_keras_serializable



@register_keras_serializable()
class PositionalEmbedding(keras.layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim,
                                   embeddings_regularizer=keras.regularizers.l2(1e-6))
        self.pos_emb = Embedding(input_dim=sequence_length, output_dim=embed_dim)
    
    def call(self, x):
        seq_length = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_length, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


@register_keras_serializable()
class Transformer_decoder(keras.layers.Layer):
    def __init__(self, embed_dimension, ff_dimension, num_heads, vocab_size, max_len, num_layers=4, rate=0.1,**kwargs ):
        super().__init__(**kwargs)
        self.embed = PositionalEmbedding(max_len, vocab_size, embed_dimension)
        self.dec_layers = []
        for _ in range(num_layers):
            self.dec_layers.append({
                'MHA1': MultiHeadAttention(num_heads=num_heads, key_dim=embed_dimension//num_heads),
                'LN1': LayerNormalization(epsilon=1e-6),
                'drop1': Dropout(rate),
                'FFN': keras.Sequential([Dense(ff_dimension, activation="relu"), Dense(embed_dimension)]),
                'LN2': LayerNormalization(epsilon=1e-6),
                'drop2': Dropout(rate),
                'MHA2': MultiHeadAttention(num_heads=num_heads, key_dim=embed_dimension//num_heads),
                'LN3': LayerNormalization(epsilon=1e-6),
                'drop3': Dropout(rate)
            })
        self.final_layer = Dense(vocab_size)
    
    def call(self, x, context, training=False):
        x = self.embed(x)  # (batch, seq_length, embed_dim)
        for layer in self.dec_layers:
            # Self-Attention
            seq_length = tf.shape(x)[1]
            mask = tf.linalg.band_part(tf.ones((seq_length, seq_length)), -1, 0)
            attention1 = layer['MHA1'](x, x, x, attention_mask=mask)
            attention1 = layer['drop1'](attention1, training=training)
            output1 = layer['LN1'](x + attention1)
            # Cross-Attention 
            attention2 = layer['MHA2'](output1, context, context)
            attention2 = layer['drop3'](attention2, training=training)
            output2 = layer['LN3'](output1 + attention2)
            # Feed-Forward-Network
            ffn_output = layer['FFN'](output2)
            ffn_output = layer['drop2'](ffn_output, training=training)
            x = layer['LN2'](output2 + ffn_output)
        return self.final_layer(x)  # (batch, seq_length, vocab_size)


@register_keras_serializable()
class Expand_Dimension(layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, x):
        return tf.expand_dims(x, axis=self.axis)

    def compute_output_shape(self, input_shape):
        return input_shape[:self.axis] + (1,) + input_shape[self.axis:]

@register_keras_serializable()
def Masked_Loss(y_true, y_pred):
    loss_func = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = loss_func(y_true, y_pred)
    loss *= mask
    return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + 1e-9)
