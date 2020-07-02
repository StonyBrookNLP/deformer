#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from common import layers
from common import tf

from .layers.transformer import BertEmbedding
from .layers.transformer import BertEncoder


class Bert(layers.Layer):
    def __init__(self, config):
        super().__init__(name='bert')
        self.embedder = BertEmbedding(config, name='embeddings')
        self.encoder = BertEncoder(config, name='encoder')

        self.attentions = None
        self.encoded_output = None
        self.embeddings = None
        self.logits = None

        self.max_seq_length = config.max_seq_length
        self.use_tpu = config.use_tpu
        self.optimize_padding = config.optimize_padding
        self.ablation_args = {k: v for k, v in config.items()
                              if k.startswith('attn_')}

    def call(self, inputs, **kwargs):
        kwargs.update(self.ablation_args)

        input_ids, token_type_ids = inputs

        input_mask = tf.cast(tf.cast(input_ids, tf.bool), tf.float32)
        if not self.use_tpu and self.optimize_padding:
            # remove extra padding zeros for faster compute
            seq_len = tf.reduce_sum(input_mask, axis=1)
            self.max_seq_length = tf.cast(tf.reduce_max(seq_len), tf.int32)
            input_ids = input_ids[:, :self.max_seq_length]
            input_mask = input_mask[:, :self.max_seq_length]
            token_type_ids = token_type_ids[:, :self.max_seq_length]

        # embeddings is optional, most of time being empty
        # one useful case is for freezing or separating embeddings from main
        # model architecture
        embeddings = kwargs.get('embeddings', None)
        if embeddings:
            assert len(embeddings) == 1, 'pass one element as embeddings'
            self.embeddings = embeddings
        else:
            kwargs['max_seq_length'] = self.max_seq_length
            embedder_inputs = [input_ids, token_type_ids]
            self.embeddings = self.embedder(embedder_inputs, **kwargs)

        with tf.name_scope('attention_mask'):
            attn_mask = tf.expand_dims(input_mask, 1)

        encoder_inputs = [self.embeddings, attn_mask]
        self.encoded_output = self.encoder(encoder_inputs, **kwargs)
        self.attentions = self.encoder.attentions
        return self.encoded_output
