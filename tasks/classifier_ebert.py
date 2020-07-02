#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import OrderedDict

from common import layers
from common import tf
from common.tf_util import get_initializer
from features.feature_ebert_classifier import EbertClassifierDataBuilder
from models.ebert import Ebert
from tasks.classifier import ClassifierModel


class EbertClassifier(ClassifierModel):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.ebert_encoder = Ebert(config)

        initializer = get_initializer(config)
        self.num_classes = config.num_classes
        self.pooler = layers.Dense(config.hidden_size,
                                   kernel_initializer=initializer,
                                   name='ebert/pooler/dense', activation='tanh')
        self.cls_dropout_layer = layers.Dropout(config.hidden_dropout_prob)
        self.num_choices = config.get('num_choices', 0)
        num_classes = 1 if self.num_choices else self.num_classes
        self.cls_layer = layers.Dense(num_classes,
                                      kernel_initializer=initializer,
                                      name='classifier/dense')

        self.max_first_length = config.max_first_length + 2
        self.max_second_length = config.max_seq_length - self.max_first_length
        self.num_choices = config.get('num_choices', 0)

        self.pooled_output = None
        self.encoded_output = None
        self.embeddings = None
        self.logits = None
        self.first_embeddings = None
        self.second_embeddings = None

        task = config.task
        replace_map = OrderedDict(
            {'LayerNorm': 'layer_norm',
             'bert/pooler': 'ebert_{}/ebert/pooler'.format(task),
             'bert/embeddings': 'ebert_{}/ebert/embeddings'.format(task)})
        # upper layers must be replaced first (i.e., longest match)
        layer_key = 'bert/encoder/layer_{}'
        layer_val = 'ebert_{}/ebert/{}_encoder/layer_{}'
        for layer_idx in range(config.sep_layers, config.num_hidden_layers):
            k = layer_key.format(layer_idx)
            replace_map[k] = layer_val.format(task, 'upper', layer_idx)
        for layer_idx in range(config.sep_layers):
            k = layer_key.format(layer_idx)
            replace_map[k] = layer_val.format(task, 'lower', layer_idx)
        if config.use_replace_map:
            self.replace_map = replace_map
        else:
            self.replace_map = {}
        self.data_builder = EbertClassifierDataBuilder(config)

    def call(self, inputs, **kwargs):
        first_ids, second_ids = inputs
        if self.num_choices:
            first_ids = tf.reshape(first_ids, [-1, self.max_first_length])
            second_ids = tf.reshape(second_ids, [-1, self.max_second_length])

        encoder_inputs = [first_ids, second_ids]
        self.encoded_output = self.ebert_encoder(encoder_inputs, **kwargs)
        self.first_embeddings = self.ebert_encoder.first_embeddings
        self.second_embeddings = self.ebert_encoder.second_embeddings
        # We "pool" the first token hidden state
        # We assume that this has been pre-trained
        first_token_tensor = tf.squeeze(self.encoded_output[-1][:, 0:1, :],
                                        axis=1)
        self.pooled_output = self.pooler(first_token_tensor)
        training = kwargs.get('training', False)
        sequence_output = self.cls_dropout_layer(self.pooled_output,
                                                 training=training)
        self.logits = self.cls_layer(sequence_output)
        if self.num_choices:
            self.logits = tf.reshape(self.logits, [-1, self.num_choices])
        return self.logits

    def warm_up(self, config):
        first_ids = tf.zeros([1, self.max_first_length], tf.int32)
        second_ids = tf.zeros([1, self.max_second_length], tf.int32)
        self([first_ids, second_ids])

    def infer(self, inputs, **kwargs):
        logits = self(inputs, **kwargs)
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        probabilities = tf.nn.softmax(logits, axis=-1)
        return predictions, probabilities

    def infer_graph(self, config):
        pass

    def get_logits(self, features, **kwargs):
        first_ids = features["seq1_ids"]
        second_ids = features["seq2_ids"]
        return self([first_ids, second_ids], **kwargs)
