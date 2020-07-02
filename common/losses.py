#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
model losses

"""

from . import tf


def get_logit_loss(logits, positions, seq_length):
    one_hot_positions = tf.one_hot(positions, depth=seq_length, dtype=tf.float32)
    log_probabilities = tf.nn.log_softmax(logits, axis=-1)
    loss = -tf.reduce_mean(tf.reduce_sum(one_hot_positions * log_probabilities, axis=-1))
    return loss


def get_cross_entropy_loss(logits, positions, seq_length):
    onehot_positions = tf.one_hot(positions, depth=seq_length, dtype=tf.float32)
    return tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                      labels=tf.stop_gradient(onehot_positions))
