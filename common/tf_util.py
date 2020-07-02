#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

from common import tf


def flat_tensor(input_tensor):
    if isinstance(input_tensor, tf.Tensor):
        return input_tensor.numpy().flat
    elif isinstance(input_tensor, np.ndarray):
        return input_tensor.flat
    else:
        raise TypeError('unknown tensor type: {}'.format(type(input_tensor)))


def convert_to_tensor(*list_like_input):
    input_tensors = []
    for list_input in list_like_input:
        input_tensors.append(tf.convert_to_tensor(list_input))
    return input_tensors


def get_input_mask(input_ids):
    return tf.cast(tf.cast(input_ids, tf.bool), tf.float32)


def get_input_length(input_mask):
    return tf.reduce_sum(tf.cast(input_mask, tf.int32), axis=1)


def may_normalize(inputs, normalize=True):
    if normalize:
        return tf.nn.softmax(inputs)
    else:
        return tf.math.exp(inputs)


def get_attention_mask(input_ids, to_ids=None):
    input_mask = tf.cast(tf.cast(input_ids, tf.bool), tf.float32)
    if to_ids is None:
        attention_mask = tf.expand_dims(input_mask, axis=1)
        # `attention_mask` = [B, 1, F]
        return attention_mask
    else:
        # zeros = [B, T]
        zeros = tf.zeros_like(to_ids, dtype=tf.float32)

        # zeros_mask = [B, 1, T]
        zeros_mask = tf.expand_dims(zeros, axis=1)

        # from_mask = [B, F, 1]
        from_mask = tf.expand_dims(input_mask, axis=2)
        # `attention_mask` = [B, F, T]
        attention_mask = from_mask + zeros_mask
        return attention_mask


def gelu(x):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415

    Args:
      x: float Tensor to perform activation.

    Returns:
      `input_tensor` with the GELU activation applied.
    """
    # cdf = 0.5 * (1.0 + tf.math.erf(x / tf.math.sqrt(2.0)))
    # return x * cdf
    return 0.5 * x * (
        1 + tf.math.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))


def get_activation(activation_string):
    """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

    Args:
      activation_string: String name of the activation function.

    Returns:
      A Python function corresponding to the activation function. If
      `activation_string` is None, empty, or "linear", this will return None.
      If `activation_string` is not a string, it will return `activation_string`.

    Raises:
      ValueError: The `activation_string` does not correspond to a known
        activation.
    """

    # We assume that anything that"s not a string is already an activation
    # function, so we just return it.
    if not isinstance(activation_string, str):
        return activation_string

    if not activation_string:
        return None

    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return tf.nn.relu
    elif act == "gelu":
        return gelu
    elif act == "tanh":
        return tf.tanh
    else:
        raise ValueError("Unsupported activation: %s" % act)


def get_initializer(config):
    initializer_range = config.initializer_range or 0.02
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)


def get_tpu_run_config(config):
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    if config.use_tpu and config.tpu_name:
        save_summary_steps = None
        tpu_name = config.tpu_name
        tpu_zone = config.get('tpu_zone', None)
        gcp_project = config.get('gcp_project', None)
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            tpu_name, zone=tpu_zone, project=gcp_project)
        tf.contrib.distribute.initialize_tpu_system(tpu_cluster_resolver)
    else:
        tpu_cluster_resolver = None
        save_summary_steps = 10
    config_master = config.get('master', None)
    tpu_config = tf.contrib.tpu.TPUConfig(
        iterations_per_loop=config.iterations_per_loop,
        num_shards=config.num_tpu_cores,
        per_host_input_for_training=is_per_host)
    run_config = tf.contrib.tpu.RunConfig(
        save_summary_steps=save_summary_steps,
        cluster=tpu_cluster_resolver, master=config_master,
        model_dir=config.checkpoint_dir,
        save_checkpoints_steps=config.steps_per_checkpoint,
        keep_checkpoint_max=config.keep_checkpoint_max,
        tpu_config=tpu_config)
    return run_config
