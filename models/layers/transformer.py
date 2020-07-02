#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
common model building layers should be put here
layers here must be shared by at least two models,
otherwise should be put into the model's own layers.py
"""
import math

from common import layers
from common import tf
from common.tf_util import get_activation
from common.tf_util import get_initializer


class BertEmbedding(layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = config.vocab_size
        self.embedding_size = config.hidden_size
        self.use_tpu = config.use_tpu
        self.type_vocab_size = config.type_vocab_size
        self.max_position_embeddings = config.max_position_embeddings
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.embedding_table = None
        self.token_type_table = None
        self.position_table = None
        self.layer_norm = layers.LayerNormalization(name="layer_norm",
                                                    axis=-1, epsilon=1e-12)
        self.dropout = layers.Dropout(self.hidden_dropout_prob)

    def call(self, inputs, **kwargs):
        input_ids, token_type_ids = inputs
        max_seq_length = kwargs.get('max_seq_length', 512)
        training = kwargs.get('training', False)
        position_ids = tf.range(max_seq_length)
        if self.use_tpu:
            # use one hot embeddings for faster look up
            flat_input_ids = tf.reshape(input_ids, [-1])
            one_hot_input_ids = tf.one_hot(
                flat_input_ids, depth=self.vocab_size)
            word_embeddings = tf.matmul(
                one_hot_input_ids, self.embedding_table)

            flat_position_ids = tf.reshape(position_ids, [-1])
            one_hot_position_ids = tf.one_hot(
                flat_position_ids, depth=self.max_position_embeddings)
            position_embeddings = tf.matmul(
                one_hot_position_ids, self.position_table)
        else:
            position_embeddings = tf.nn.embedding_lookup(
                self.position_table, position_ids)
            word_embeddings = tf.nn.embedding_lookup(
                self.embedding_table, input_ids)

        word_embeddings = tf.reshape(word_embeddings,
                                     [-1, max_seq_length, self.embedding_size])
        # expand first axis for batch_size broadcasting
        position_embeddings = tf.expand_dims(position_embeddings, axis=0)
        output = word_embeddings + position_embeddings
        if self.type_vocab_size > 1:
            # for a small vocabulary, it is always faster to do one-hot
            flat_token_type_ids = tf.reshape(token_type_ids, [-1])
            # token_type_embeddings = tf.gather(self.token_type_table,
            #                                   flat_token_type_ids)
            one_hot_ids = tf.one_hot(
                flat_token_type_ids, depth=self.type_vocab_size)
            token_type_embeddings = tf.matmul(
                one_hot_ids, self.token_type_table)
            token_type_embeddings = tf.reshape(
                token_type_embeddings,
                [-1, max_seq_length, self.embedding_size])
            output = output + token_type_embeddings

        output = self.layer_norm(output)
        output = self.dropout(output, training=training)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[1], self.embedding_size

    def build(self, input_shape):
        self.embedding_table = self.add_weight(
            "word_embeddings", trainable=self.trainable, dtype=tf.float32,
            shape=[self.vocab_size, self.embedding_size])
        if self.type_vocab_size > 1:
            self.token_type_table = self.add_weight(
                "token_type_embeddings", trainable=self.trainable,
                dtype=tf.float32,
                shape=[self.type_vocab_size, self.embedding_size])
        self.position_table = self.add_weight(
            "position_embeddings", trainable=self.trainable, dtype=tf.float32,
            shape=[self.max_position_embeddings, self.embedding_size])
        super().build(input_shape)


class MultiHeadAttention(layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.attn_head_size = config.attention_head_size
        qkv_size = self.attn_head_size * self.num_heads
        initializer = get_initializer(config)
        self.query_layer = layers.Dense(qkv_size, name="self/query",
                                        kernel_initializer=initializer)
        self.key_layer = layers.Dense(qkv_size, name="self/key",
                                      kernel_initializer=initializer)
        self.value_layer = layers.Dense(qkv_size, name="self/value",
                                        kernel_initializer=initializer)
        self.attn_dropout = layers.Dropout(config.attention_dropout_prob)
        self.attn_output_layer = layers.Dense(hidden_size, name='output/dense',
                                              kernel_initializer=initializer)
        self.attn_output_dropout = layers.Dropout(config.hidden_dropout_prob,
                                                  seed=config.random_seed)
        self.attn_norm_layer = layers.LayerNormalization(
            name="output/layer_norm", axis=-1, epsilon=1e-12)

        self.w_layer = layers.Dense(1, name="self/w")
        self.attention = None
        self.random_seed = config.random_seed
        self.debug = config.debug
        self.debug_save_dir = config.debug_save_dir if config.debug else None

    def call(self, inputs, **kwargs):
        training = kwargs.get('training', False)
        export = kwargs.get('export', False)
        # percentile = kwargs.get('attn_drop_percentile', None)
        # attn_profile = kwargs.get('attn_profile', False)
        # attn_drop_type = kwargs.get('attn_drop_type', None)
        # attn_col_type = kwargs.get('attn_col_type', 'min')
        # attn_drop_renormalize = kwargs.get('attn_drop_renormalize', False)
        from_tensor, to_tensor, attn_mask = inputs
        from_shape = tf.shape(from_tensor)
        to_shape = tf.shape(to_tensor)
        batch_size = from_shape[0]
        from_seq_len = from_shape[1]
        to_seq_len = to_shape[1]

        #   B = batch size (number of sequences)
        #   F = `from_tensor` sequence length
        #   T = `to_tensor` sequence length
        #   N = `self.num_heads`
        #   H = `self.attention_head_size`

        # `queries` = [B, F, N*H]
        queries = self.query_layer(from_tensor)

        # `keys` = [B, T, N*H]
        keys = self.key_layer(to_tensor)

        # `values` = [B, T, N*H]
        values = self.value_layer(to_tensor)
        # save_dir = self.debug_save_dir
        # if self.weights:
        #     import re
        #     layer_found = re.findall(r'layer_\d+', self.weights[0].name)
        #     layer_str = layer_found[0]
        # else:
        #     layer_str = ''
        #     save_dir = None
        # save_tensor(queries, "2queries", save_dir, layer_str)
        # save_tensor(keys, "3keys", save_dir, layer_str)
        # save_tensor(values, "4values", save_dir, layer_str)

        # if attn_profile and percentile and percentile < 1:
        #     attn_drop_type = 'profile'
        #     logger.info("{}, percentile={}".format(layer_str, percentile))
        #     predict_keys = tf.squeeze(self.w_layer(from_tensor), axis=[-1])
        #     top_k = tf.cast(tf.ceil((1.0 - percentile) * tf.cast(
        #         from_seq_len, tf.float32)), tf.int32)
        #     _, col_idx = tf.nn.top_k(predict_keys, k=top_k)
        #     if self.debug:
        #         logger.info("keys={}".format(keys.numpy().shape))
        #         logger.info("attn_mask={}".format(
        #             attn_mask.numpy().shape))
        #         logger.info("col_idx={}".format(col_idx.numpy().shape))
        #
        #     keys = tf.gather_nd(keys, tf.expand_dims(col_idx, -1),
        #     batch_dims=1)
        #     values = tf.gather_nd(values,
        #                           tf.expand_dims(col_idx, -1), batch_dims=1)
        #     logger.info("{}, keys_shape={}, values_shape={}".format(
        #         layer_str, keys.get_shape().as_list(),
        #         values.get_shape().as_list()))
        #     attn_mask = tf.gather_nd(
        #         attn_mask, tf.expand_dims(tf.expand_dims(col_idx, 1), -1),
        #         batch_dims=2)
        #     if self.debug:
        #         logger.info("keys={}".format(keys.numpy().shape))
        #         logger.info("values={}".format(values.numpy().shape))
        #         logger.info("attn_mask={}".format(
        #             attn_mask.numpy().shape))
        #
        #     to_seq_len = top_k

        # `queries` = [B, F, N, H]
        queries = tf.reshape(queries, [batch_size, from_seq_len,
                                       self.num_heads, self.attn_head_size])
        # `queries` = [B, N, F, H]
        queries = tf.transpose(queries, [0, 2, 1, 3])

        # `keys` = [B, T, N, H]
        keys = tf.reshape(keys, [batch_size, to_seq_len,
                                 self.num_heads, self.attn_head_size])
        # `keys` = [B, N, H, T]
        keys = tf.transpose(keys, [0, 2, 3, 1])

        if export:
            queries = tf.reshape(queries, [batch_size * self.num_heads,
                                           from_seq_len, self.attn_head_size])
            keys = tf.reshape(keys, [batch_size * self.num_heads,
                                     self.attn_head_size, to_seq_len])
        # `attn_scores` = [B, N, F, T]
        attn_scores = tf.matmul(queries, keys, name='AttnMatmul')
        attn_scores = tf.multiply(attn_scores,
                                  1.0 / math.sqrt(float(self.attn_head_size)))

        if attn_mask is not None:
            # `attn_mask` = [B, 1, 1, T]
            attn_mask = tf.expand_dims(attn_mask, axis=1)
            # save_tensor(attn_mask * tf.ones_like(attn_scores), 'attn_mask', 
            #             save_dir, layer_str)
            attn_scores += (1.0 - attn_mask) * -1e10

        # `attn_probabilities` = [B, N, F, T]
        attn_probabilities = tf.nn.softmax(attn_scores)
        attn_batch = self.attn_dropout(attn_probabilities, training=training)
        # save_tensor(attn_batch, "5attn", save_dir, layer_str)
        # `values` = [B, T, N, H]
        values = tf.reshape(values, [batch_size, to_seq_len,
                                     self.num_heads, self.attn_head_size])
        # `values` = [B, N, T, H]
        values = tf.transpose(values, [0, 2, 1, 3])
        self.attention = attn_batch
        # if percentile and percentile < 1:
        #     logger.info("{}, percentile={}".format(layer_str, percentile))
        # assert 0 <= percentile, 'attn_drop_percentile must be in [0, 1)'
        # if attn_drop_type == 'row':
        #     attn_mask_batch = tf.squeeze(attn_mask, [1, 2])
        #
        #     def drop_one(one_input):
        #         seq_len, attn = one_input
        #         top = tf.cast(tf.ceil((1.0 - percentile) * seq_len),
        #                       tf.int32)
        #         if self.debug:
        #             logger.info("{},len={},k={}".format(
        #                 layer_str, int(seq_len), top))
        #         val, idx = tf.nn.top_k(attn, k=top)
        #         dropped_attn = tf.where_v2(
        #             attn >= tf.expand_dims(val[:, :, -1], 2), attn, 0)
        #         return dropped_attn
        #
        #     seq_len_batch = tf.reduce_sum(attn_mask_batch, axis=1)
        #     top_attn = tf.map_fn(drop_one, [seq_len_batch, attn_batch],
        #                          dtype=tf.float32)
        #     if attn_drop_renormalize:
        #         top_attn = top_attn / tf.reduce_sum(top_attn, -1,
        #                                             keepdims=True)
        #         attn_save_name = 'renorm_dropped_attn_row'
        #     else:
        #         attn_save_name = 'dropped_attn_row'
        #
        #     self.attention = top_attn
        #     save_tensor(self.attention, attn_save_name, save_dir, layer_str)
        # elif attn_drop_type == 'col':
        #     top_k = tf.cast(tf.ceil((1.0 - percentile) * tf.cast(
        #         from_seq_len, tf.float32)), tf.int32)
        #     col_selected = None
        #     col_idx = None
        #     if self.debug:
        #         logger.info("top_k={}".format(top_k))
        #     if attn_col_type == 'min':
        #         col_selected = tf.reduce_min(attn_batch, axis=-2)
        #     elif attn_col_type == 'max':
        #         col_selected = tf.reduce_max(attn_batch, axis=-2)
        #     elif attn_col_type == 'mean':
        #         col_selected = tf.reduce_mean(attn_batch, axis=-2)
        #     elif attn_col_type == 'var':
        #         col_selected = tf.math.reduce_variance(attn_batch, axis=-2)
        #     elif attn_col_type == 'norm':
        #         col_selected = tf.math.reduce_euclidean_norm(
        #             attn_batch, axis=-2)
        #     elif attn_col_type == 'entropy':
        #         col_selected = tf.reduce_sum(-1 * attn_batch * tf.log(
        #             attn_batch + 1e-7), axis=-2)
        #     elif attn_col_type == 'random':
        #         col_idx = tf.random.uniform(
        #             [batch_size, self.num_heads, top_k], maxval=top_k,
        #             dtype=tf.int32, seed=self.random_seed)
        #     else:
        #         raise ValueError('attn_col_type: {} not supported!'.format(
        #             attn_col_type))
        #     if col_selected is not None:
        #         # logger.info("col_selected={}".format(
        #         # col_selected.numpy().shape))
        #         _, col_idx = tf.nn.top_k(col_selected, k=top_k)
        #     top_attn = tf.transpose(tf.gather_nd(
        #         tf.transpose(attn_batch, [0, 1, 3, 2]),
        #         tf.expand_dims(col_idx, -1), batch_dims=2),
        #         [0, 1, 3, 2])
        #     if self.debug:
        #         logger.info("col_idx={}".format(col_idx.numpy().shape))
        #         logger.info("top_attn={}".format(top_attn.numpy().shape))
        #
        #     # slice values
        #     values = tf.gather_nd(values, tf.expand_dims(col_idx, -1),
        #                           batch_dims=2)
        #     if self.debug:
        #         logger.info("values={}".format(values.numpy().shape))
        #         logger.info("{},len={},k={}, a:{}, v:{}".format(
        #             layer_str, int(from_seq_len), top_k,
        #             top_attn.numpy().shape, values.numpy().shape))
        #     if attn_drop_renormalize:
        #         top_attn = top_attn / tf.reduce_sum(top_attn, -1,
        #                                             keepdims=True)
        #         attn_save_name = 'renorm_dropped_attn_col_{}'.format(
        #             attn_col_type)
        #     else:
        #         attn_save_name = 'dropped_attn_col_{}'.format(
        #             attn_col_type)
        #     self.attention = top_attn
        #     save_tensor(self.attention, attn_save_name,
        #                 save_dir, layer_str)
        # elif attn_drop_type == 'profile':
        #     self.attention = attn_batch
        # else:
        #     raise ValueError('attn_drop_type: {} not supported'.format(
        #         attn_drop_type))

        if export:
            self.attention = tf.reshape(self.attention,
                                        [batch_size * self.num_heads,
                                         from_seq_len, to_seq_len])
            values = tf.reshape(values,
                                [batch_size * self.num_heads,
                                 to_seq_len, self.attn_head_size])

        # `context_layer` = [B, N, F, H]
        context_layer = tf.matmul(self.attention, values, name='ContextMatmul')

        if export:
            context_layer = tf.reshape(context_layer,
                                       [batch_size, self.num_heads,
                                        from_seq_len, self.attn_head_size])
        # `context_layer` = [B, F, N, H]
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

        # `context_layer` = [B, F, N*H]
        attn_output = tf.reshape(context_layer,
                                 [batch_size, from_seq_len,
                                  self.num_heads * self.attn_head_size])
        # save_tensor(attn_output, "6ctx_values", save_dir, layer_str)

        attn_output = self.attn_output_layer(attn_output)
        # save_tensor(attn_output, "7ctx_outputs", save_dir, layer_str)

        attn_output = self.attn_dropout(attn_output, training=training)
        attn_output = self.attn_norm_layer(attn_output + from_tensor)
        return attn_output


class TransformerLayer(layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        intermediate_act_fn = get_activation(config.intermediate_act_fn)

        kwargs['name'] = 'attention'
        self.attention_layer = MultiHeadAttention(config, **kwargs)

        if config.get('svd_units', 0) > 0:
            self.intermediate_layer0 = layers.Dense(
                config.svd_units, name='dense0')
        else:
            self.intermediate_layer0 = None
        self.intermediate_layer = layers.Dense(
            intermediate_size, name='dense', activation=intermediate_act_fn)
        if config.get('svd_units', 0) > 0:
            self.output_layer0 = layers.Dense(config.svd_units, name='dense0')
        else:
            self.output_layer0 = None

        self.output_layer = layers.Dense(self.hidden_size, name='dense')
        self.output_dropout = layers.Dropout(config.hidden_dropout_prob,
                                             seed=config.random_seed)
        self.output_norm_layer = layers.LayerNormalization(
            name="layer_norm", axis=-1, epsilon=1e-12)

        self.attention = None
        self.debug_save_dir = config.debug_save_dir if config.debug else None

    def call(self, inputs, **kwargs):
        layer_input, attn_mask = inputs

        # save_dir = self.debug_save_dir
        # if self.weights:
        #     import re
        #     layer_found = re.findall(r'layer_\d+', self.weights[0].name)
        #     layer_str = layer_found[0]
        # else:
        #     layer_str = ''
        #     save_dir = None
        # save_tensor(layer_input, "1inputs", save_dir, layer_str)

        attention_output = self.attention_layer(
            [layer_input, layer_input, attn_mask], **kwargs)
        self.attention = self.attention_layer.attention
        # save_tensor(attention_output, "8attn_outputs", save_dir, layer_str)

        with tf.name_scope("intermediate"):
            if self.intermediate_layer0 is not None:
                attention_output = self.intermediate_layer0(attention_output)
            intermediate_output = self.intermediate_layer(attention_output)
        # save_tensor(intermediate_output, "9intermediate_outputs", save_dir,
        #             layer_str)

        with tf.name_scope("output"):
            if self.output_layer0 is not None:
                intermediate_output = self.output_layer0(intermediate_output)
            layer_output = self.output_layer(intermediate_output)
            # save_tensor(layer_output, "10layer_outputs", save_dir, layer_str)
            layer_output = self.output_dropout(
                layer_output, training=kwargs.get('training', False))
            layer_output = self.output_norm_layer(
                layer_output + attention_output)
        # save_tensor(layer_output, "11outputs", save_dir, layer_str)
        return layer_output


class BertEncoder(layers.Layer):
    """ Bert Transformer encoding layer"""

    def __init__(self, config, **kwargs):
        if kwargs.get('layer_range', None):
            # decomposed bert
            self.start_idx, self.end_idx = kwargs['layer_range']
            kwargs.pop('layer_range')
        else:
            # original bert, use all layers
            self.start_idx, self.end_idx = 0, config.num_hidden_layers
        super().__init__(**kwargs)
        self.num_hidden_layers = config.num_hidden_layers
        self.transformer_layers = []
        self.attentions = []

        for layer_idx in range(self.start_idx, self.end_idx):
            kwargs['name'] = "layer_%d" % layer_idx
            self.transformer_layers.append(TransformerLayer(config, **kwargs))

    def call(self, inputs, **kwargs):
        embeddings, attention_mask = inputs
        layer_output = embeddings
        all_layer_outputs = []
        for layer_idx in range(len(self.transformer_layers)):
            layer_input = layer_output
            layer_output = self.transformer_layers[layer_idx](
                [layer_input, attention_mask], **kwargs)
            layer_attention = self.transformer_layers[layer_idx].attention
            self.attentions.append(layer_attention)
            all_layer_outputs.append(layer_output)
        return all_layer_outputs
