#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from common import layers
from common import tf
from common.tf_util import get_attention_mask
from .layers.transformer import BertEmbedding
from .layers.transformer import BertEncoder


class Ebert(layers.Layer):
    def __init__(self, config):
        super().__init__(name='ebert')
        self.embedder = BertEmbedding(config, name='embeddings')

        self.max_seq_length = config.max_seq_length
        self.max_first_length = config.max_first_length + 2
        self.max_second_length = config.max_seq_length - self.max_first_length
        self.sep_layers = config.sep_layers

        self.lower_range = (0, self.sep_layers)
        self.lower_encoder = BertEncoder(config, layer_range=self.lower_range,
                                         name='lower_encoder', )
        self.upper_range = (self.sep_layers, config.num_hidden_layers)
        self.upper_encoder = BertEncoder(config, layer_range=self.upper_range,
                                         name='upper_encoder')

        self.encoded_output = None
        self.first_encoded = None
        self.second_encoded = None
        self.first_embeddings = None
        self.second_embeddings = None

    def call(self, inputs, **kwargs):
        first_ids, second_ids = inputs

        # embeddings is optional, most of time being empty
        # one useful case is for freezing or separating embeddings from main
        # model architecture
        embeddings = kwargs.get('embeddings', None)
        if embeddings:
            assert_error = 'pass two element as first and second embeddings'
            assert len(embeddings) == 2, assert_error
            self.first_embeddings, self.second_embeddings = embeddings
        else:
            first_type_ids = tf.zeros_like(first_ids, dtype=tf.int32)
            first_inputs = [first_ids, first_type_ids]
            kwargs['max_seq_length'] = self.max_first_length

            self.first_embeddings = self.embedder(first_inputs, **kwargs)
            second_type_ids = tf.ones_like(second_ids, dtype=tf.int32)
            second_inputs = [second_ids, second_type_ids]

            kwargs['max_seq_length'] = self.max_second_length
            self.second_embeddings = self.embedder(second_inputs, **kwargs)
            kwargs.pop('max_seq_length')  # avoid any possible side effect

        with tf.variable_scope('attention_mask'):
            first_attn_mask = get_attention_mask(first_ids)
            second_attn_mask = get_attention_mask(second_ids)

        if self.sep_layers > 0:
            # fake_cake is useful for profiling model complexity
            if kwargs.get('fake_cache_first', False):
                # in practice, should lookup encoded top from some db
                first_encoded_top = self.first_embeddings
            else:
                lower_inputs = [self.first_embeddings, first_attn_mask]
                self.first_encoded = self.lower_encoder(lower_inputs, **kwargs)
                first_encoded_top = self.first_encoded[-1]

            if kwargs.get('fake_cache_second', False):
                second_encoded_top = self.second_embeddings
            else:
                lower_inputs = [self.second_embeddings, second_attn_mask]
                self.second_encoded = self.lower_encoder(lower_inputs, **kwargs)
                second_encoded_top = self.second_encoded[-1]
        else:
            first_encoded_top = self.first_embeddings
            second_encoded_top = self.second_embeddings
        # first dim is batch, second dim should be seq len
        both_encoded = tf.concat([first_encoded_top, second_encoded_top],
                                 axis=1)
        # print('both_encoded: {}'.format(both_encoded.get_shape().as_list()))
        both_attn_mask = tf.concat([first_attn_mask, second_attn_mask],
                                   axis=-1)  # last dim should be seq len
        upper_inputs = [both_encoded, both_attn_mask]
        upper_output = self.upper_encoder(upper_inputs, **kwargs)

        output_lower_encodings = kwargs.get('output_lower_encodings', False)
        if output_lower_encodings:
            lower_encodings = [tf.concat([f, s], axis=1) for f, s in
                               zip(self.first_encoded, self.second_encoded)]
            self.encoded_output = lower_encodings + upper_output
        else:
            self.encoded_output = upper_output

        return self.encoded_output
