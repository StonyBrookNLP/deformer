#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from common import layers
from common import tf
from common.tf_util import get_initializer
from features.feature_bert_classifier import BertClassifierDataBuilder
from models.bert import Bert
from tasks.classifier import ClassifierModel


class BertClassifier(ClassifierModel):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.bert_encoder = Bert(config)

        initializer = get_initializer(config)
        self.num_classes = config.num_classes
        self.max_seq_length = config.max_seq_length
        self.pooler = layers.Dense(config.hidden_size,
                                   kernel_initializer=initializer,
                                   name='bert/pooler/dense', activation='tanh')

        self.cls_dropout_layer = layers.Dropout(config.hidden_dropout_prob)
        self.num_choices = config.get('num_choices', 0)
        num_classes = 1 if self.num_choices else self.num_classes
        self.cls_layer = layers.Dense(num_classes,
                                      kernel_initializer=initializer,
                                      name='classifier/dense')

        self.pooled_output = None
        self.attentions = None
        self.encoded_output = None
        self.embeddings = None
        self.logits = None
        if config.use_replace_map:
            self.replace_map = {'LayerNorm': 'layer_norm',
                                'bert/': 'bert_' + config.task + '/bert/'}
        else:
            self.replace_map = {}
        self.data_builder = BertClassifierDataBuilder(config)

    def call(self, inputs, **kwargs):
        input_ids, token_type_ids = inputs
        if self.num_choices:
            input_ids = tf.reshape(input_ids, [-1, self.max_seq_length])
            token_type_ids = tf.reshape(token_type_ids,
                                        [-1, self.max_seq_length])
        encoder_inputs = [input_ids, token_type_ids]
        self.encoded_output = self.bert_encoder(encoder_inputs, **kwargs)
        self.embeddings = self.bert_encoder.embeddings
        self.attentions = self.bert_encoder.attentions
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

    def core_graph(self, config, **kwargs):
        import numpy as np
        if hasattr(config, 'batch_size'):
            batch_size = config.batch_size
            if self.num_choices:
                batch_size *= self.num_choices
        else:
            batch_size = None
        export = kwargs.get('export', False)

        input_ids_ph = tf.placeholder(shape=[batch_size, config.max_seq_length],
                                      dtype=tf.int32, name='input/input_ids')
        token_type_ids_ph = tf.placeholder(
            shape=[batch_size, config.max_seq_length],
            dtype=tf.int32, name='input/segment_ids')
        if export:
            embeddings_ph = tf.placeholder(
                shape=[batch_size, config.max_seq_length, config.hidden_size],
                dtype=tf.float32, name='input/embeddings')
            logits = self(input_ids_ph, embeddings=embeddings_ph, **kwargs)
            inputs_dict = {'input_ids': input_ids_ph,
                           'embeddings': embeddings_ph}
        else:
            logits = self(input_ids_ph, token_type_ids_ph, **kwargs)
            inputs_dict = {input_ids_ph: np.zeros([batch_size,
                                                   config.max_seq_length],
                                                  dtype=np.int32),
                           token_type_ids_ph: np.zeros([batch_size,
                                                        config.max_seq_length],
                                                       dtype=np.int32), }
        return inputs_dict, logits

    # @staticmethod
    # def get_inputs(inputs, config, return_name=False):
    #     text_a, text_b = inputs
    #     input_ids, segment_ids, _, _ = gen_one_example(text_a, text_b, config)
    #     if return_name:
    #         return (input_ids, segment_ids), ('input_ids', 'segment_ids')
    #     else:
    #         return input_ids, segment_ids

    def warm_up(self, config):
        input_ids = segment_ids = tf.zeros([1, config.max_seq_length], tf.int32)
        self([input_ids, segment_ids])

    def infer(self, input_features, **kwargs):
        input_ids = [f.input_ids for f in input_features]
        segment_ids = [f.segment_ids for f in input_features]
        logits = self([input_ids, segment_ids], **kwargs)
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        probabilities = tf.nn.softmax(logits, axis=-1)
        return predictions, probabilities

    def infer_graph(self, config):
        input_ids_ph = tf.placeholder(
            shape=[None, config.max_seq_length],
            dtype=tf.int32, name='input/input_ids')

        segment_ids_ph = tf.placeholder(
            shape=[None, config.max_seq_length],
            dtype=tf.int32, name='input/segment_ids')

        def feed_fn(inputs):
            input_ids, segment_ids = inputs
            return {input_ids_ph: input_ids, segment_ids_ph: segment_ids}

        return feed_fn, self.infer((input_ids_ph, segment_ids_ph))

    def get_logits(self, features, **kwargs):
        input_ids = features["input_ids"]
        segment_ids = features["segment_ids"]
        return self([input_ids, segment_ids], **kwargs)
