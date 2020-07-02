#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
save, restore, manipulate model checkpoints

partial recover, scope mapping etc.

"""
import collections
import re

from common import tf


def list_variables(checkpoint_dir):
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
    variables = tf.train.list_variables(checkpoint_path)
    for var in variables:
        name, shape = var
        print(name, shape)


def init_from_checkpoint(init_checkpoint, to_be_initialized_variables=None, replace_map=None):
    """Compute the union of the current variables and checkpoint variables."""
    if to_be_initialized_variables is None:
        to_be_initialized_variables = tf.trainable_variables()
    initialized_variable_names = collections.OrderedDict()
    # logger.info('model_trainable_variables: \n{}'.format(trainable_variables))

    name_to_variable = collections.OrderedDict()
    for var in to_be_initialized_variables:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var
    # logger.info('name_to_variable: \n{}'.format(name_to_variable))

    init_vars = tf.train.list_variables(init_checkpoint)
    # logger.info('init_checkpoint_vars: \n{}'.format(init_vars))

    assignment_map = collections.OrderedDict()
    for init_var in init_vars:
        init_name, init_var = init_var[0], init_var[1]
        new_init_name = init_name
        for old_name in replace_map or {}:
            new_init_name = new_init_name.replace(old_name, replace_map[old_name])

        if new_init_name not in name_to_variable:
            continue

        assignment_map[init_name] = name_to_variable[new_init_name]
        initialized_variable_names[new_init_name] = 1
        initialized_variable_names[new_init_name + ":0"] = 1

    return assignment_map, initialized_variable_names
