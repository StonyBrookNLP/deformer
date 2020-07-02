#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import OrderedDict

from common import layers
from common import tf
from common.tf_util import get_initializer
from features.feature_ebert_qa import EbertQaDataBuilder
from .qa import QaModel
from models.ebert import Ebert


class EbertQa(QaModel):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.ebert_encoder = Ebert(config)

        initializer = get_initializer(config)
        num_classes = config.num_classes
        if num_classes:
            self.pooler = layers.Dense(config.hidden_size,
                                       kernel_initializer=initializer,
                                       name='pooler/dense', activation='tanh')
            self.cls_layer = layers.Dense(config.num_classes,
                                          kernel_initializer=initializer,
                                          name='answer_class/dense')

        self.span_layer = layers.Dense(2, kernel_initializer=initializer,
                                       name='answer_classifier/dense')
        self.max_first_length = self.ebert_encoder.max_first_length
        self.max_c_length = self.ebert_encoder.max_second_length

        task = config.task
        replace_map = OrderedDict(
            {'LayerNorm': 'layer_norm',
             'bert/answer_classifier':
                 'ebert_{}/ebert/answer_classifier'.format(task),
             'bert/embeddings':
                 'ebert_{}/ebert/embeddings'.format(task)})
        # upper layers must be replaced first (i.e., longest match)
        layer_key = 'bert/encoder/layer_{}'
        layer_val = 'ebert_{}/ebert/{}_encoder/layer_{}'
        for layer_idx in range(config.sep_layers, config.num_hidden_layers):
            k = layer_key.format(layer_idx)
            replace_map[k] = layer_val.format(task, 'upper', layer_idx)
        for layer_idx in range(config.sep_layers):
            k = layer_key.format(layer_idx)
            replace_map[k] = layer_val.format(task, 'lower', layer_idx)

        self.replace_map = replace_map if config.use_replace_map else {}
        self.encoded_output = None
        self.q_embeddings = None
        self.c_embeddings = None
        self.logits = None
        self.data_builder = EbertQaDataBuilder(config)

    def call(self, inputs, **kwargs):
        self.encoded_output = self.ebert_encoder(inputs, **kwargs)
        self.q_embeddings = self.ebert_encoder.first_embeddings
        self.c_embeddings = self.ebert_encoder.second_embeddings
        self.logits = self.span_layer(self.encoded_output[-1])
        if kwargs.get('logits', False):
            return self.logits

        start_logits, end_logits = tf.split(self.logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)
        return start_logits, end_logits

    def infer(self, input_features):
        q_ids = [f.question_ids for f in input_features]
        c_ids = [f.context_ids for f in input_features]
        return self([q_ids, c_ids])

    def infer_graph(self, config, **kwargs):

        q_ids_ph = tf.placeholder(shape=[None, self.max_first_length],
                                  dtype=tf.int32, name='input/q_ids')
        c_ids_ph = tf.placeholder(shape=[None, self.max_c_length],
                                  dtype=tf.int32, name='input/c_ids')

        def feed_fn(input_features):
            q_ids = [f.question_ids for f in input_features]
            c_ids = [f.context_ids for f in input_features]
            return {q_ids_ph: q_ids, c_ids_ph: c_ids}

        return feed_fn, self([q_ids_ph, c_ids_ph], **kwargs)

    def get_logits(self, features, **kwargs):
        question_ids = features["question_ids"]
        context_ids = features["context_ids"]
        return self([question_ids, context_ids], **kwargs)

    def get_context_start(self, item):
        return self.max_first_length

    def warm_up(self):
        q_ids = tf.zeros([1, self.max_first_length], tf.int32)
        c_ids = tf.zeros([1, self.max_c_length], tf.int32)
        self([q_ids, c_ids])

    # def export_graph(self, config, **kwargs):
    #     import numpy as np
    #     if hasattr(config, 'batch_size'):
    #         batch_size = config.batch_size
    #     else:
    #         batch_size = None
    #     q_ids_ph = tf.placeholder(shape=[batch_size, self.max_first_length],
    #                               dtype=tf.int32, name='input/q_ids')
    #
    #     c_ids_ph = tf.placeholder(shape=[batch_size, self.max_c_length],
    #                               dtype=tf.int32, name='input/c_ids')
    #     export = kwargs.get('export', False)
    #     if export:
    #         q_embeddings_ph = tf.placeholder(
    #             shape=[batch_size, self.max_first_length, config.hidden_size],
    #             dtype=tf.float32, name='input/q_embeddings')
    #         c_embeddings_ph = tf.placeholder(
    #             shape=[batch_size, self.max_c_length, config.hidden_size],
    #             dtype=tf.float32, name='input/c_embeddings')
    #         logits = self([q_ids_ph, c_ids_ph],
    #                       q_embeddings=q_embeddings_ph,
    #                       c_embeddings=c_embeddings_ph, **kwargs)
    #         inputs_dict = {'q_ids': q_ids_ph, 'q_embeddings': q_embeddings_ph,
    #                        'c_ids': c_ids_ph, 'c_embeddings': c_embeddings_ph}
    #     else:
    #         logits = self([q_ids_ph, c_ids_ph], **kwargs)
    #
    #         inputs_dict = {
    #             q_ids_ph: np.zeros([batch_size, self.max_first_length],
    #                                dtype=np.int32),
    #             c_ids_ph: np.zeros([batch_size, self.max_c_length],
    #                                dtype=np.int32),
    #         }
    #     return inputs_dict, logits
