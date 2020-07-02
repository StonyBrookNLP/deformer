#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from common import layers
from common import tf
from common.tf_util import get_initializer
from features.feature_bert_qa import BertQaDataBuilder
from models.bert import Bert
from .qa import QaModel


class BertQa(QaModel):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.bert_encoder = Bert(config)

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
        self.max_seq_length = config.max_seq_length
        self.max_answer_span = config.max_answer_span

        self.attentions = None
        self.encoded_output = None
        self.embeddings = None
        self.logits = None
        if config.use_replace_map:
            self.replace_map = {'LayerNorm': 'layer_norm',
                                'bert/': 'bert_' + config.task + '/bert/'}
        else:
            self.replace_map = {}
        self.data_builder = BertQaDataBuilder(config)

    def call(self, inputs, **kwargs):
        self.encoded_output = self.bert_encoder(inputs, **kwargs)
        self.embeddings = self.bert_encoder.embeddings
        self.attentions = self.bert_encoder.attentions
        self.logits = self.span_layer(self.encoded_output[-1])
        if kwargs.get('logits', False):
            return self.logits

        start_logits, end_logits = tf.split(self.logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)
        return start_logits, end_logits

    def infer(self, input_features):
        input_ids = [f.input_ids for f in input_features]
        segment_ids = [f.segment_ids for f in input_features]
        return self([input_ids, segment_ids])

    def infer_graph(self, config):
        input_ids_ph = tf.placeholder(shape=[None, config.max_seq_length],
                                      dtype=tf.int32, name='input/input_ids')

        segment_ids_ph = tf.placeholder(shape=[None, config.max_seq_length],
                                        dtype=tf.int32,
                                        name='input/segment_ids')

        def feed_fn(input_features=None):
            if input_features is None:
                return {'input_ids': input_ids_ph,
                        'segment_ids': segment_ids_ph}
            if isinstance(input_features, dict):
                return {input_ids_ph: input_features['input_ids'],
                        segment_ids_ph: input_features['segment_ids']}
            # named_tuple here
            input_ids = [f.input_ids for f in input_features]
            segment_ids = [f.segment_ids for f in input_features]
            return {input_ids_ph: input_ids, segment_ids_ph: segment_ids}

        return feed_fn, self([input_ids_ph, segment_ids_ph])

    def get_logits(self, features, **kwargs):
        return self([features["input_ids"], features["segment_ids"]], **kwargs)

    @staticmethod
    def get_context_start(example_item):
        return len(example_item['question_tokens']) + 2  # for cls and sep

    def warm_up(self):
        input_ids = segment_ids = tf.zeros([1, self.max_seq_length], tf.int32)
        self([input_ids, segment_ids])

    # def export_graph(self, config, **kwargs):
    #     import numpy as np
    #     if hasattr(config, 'batch_size'):
    #         batch_size = config.batch_size
    #     else:
    #         batch_size = None
    #     export = kwargs.get('export', False)
    #
    #     input_ids_ph = tf.placeholder(shape=[batch_size, self.max_seq_length],
    #                                   dtype=tf.int32, name='input/input_ids')
    #     if export:
    #         embeddings_ph = tf.placeholder(
    #             shape=[batch_size, config.max_seq_length, config.hidden_size],
    #             dtype=tf.float32, name='input/embeddings')
    #         logits = self(input_ids_ph, embeddings=embeddings_ph, **kwargs)
    #         inputs_dict = {'input_ids': input_ids_ph,
    #                        'embeddings': embeddings_ph}
    #     else:
    #         logits = self(input_ids_ph, **kwargs)
    #         inputs_dict = {
    #             input_ids_ph: np.zeros([batch_size, config.max_seq_length],
    #                                    dtype=np.int32), }
    #     return inputs_dict, logits
