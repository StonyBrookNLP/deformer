#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from common import layers
from common import tf


class AnswerClassifier(layers.Dense):

    def __init__(self, units, **kwargs):
        super().__init__(units, **kwargs)

    def call(self, hidden_states, **kwargs):
        seq_length = kwargs.get('max_seq_length', None)
        state_shape = tf.shape(hidden_states)
        batch_size = state_shape[0]
        seq_length = seq_length if seq_length else state_shape[1]
        hidden_matrix = tf.reshape(hidden_states, [batch_size * seq_length, -1])
        logits = tf.matmul(hidden_matrix, self.kernel)
        logits = tf.nn.bias_add(logits, self.bias)
        logits = tf.reshape(logits, [batch_size, seq_length, 2])
        logits = tf.transpose(logits, [2, 0, 1], name='logits')
        return logits


class LabelClassifier(layers.Dense):
    def __init__(self, units, dropout_rate=0, **kwargs):
        super().__init__(units, **kwargs)
        self.dropout_layer = layers.Dropout(dropout_rate)

    def call(self, inputs, **kwargs):
        training = kwargs.get('training', False)
        dropout_inputs = self.dropout_layer(inputs, training=training)
        logits = tf.matmul(dropout_inputs, self.kernel)
        logits = tf.nn.bias_add(logits, self.bias)
        return logits
