#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tasks import task_map

MODEL_FEATURE_MAP = {
    'sbert': 'ebert'
}


def get_record_parser(model_name, task):
    """ implements the task specific models under ..features folder,
    following the convention: feature_model_task.py
    """
    builder_class = get_example_builder(model_name, task)
    return builder_class.record_parser


def get_example_builder(model_name, task):
    """ implements the task specific models under ..features folder,
    following the convention: feature_model_task.py
    """
    import importlib
    mapped_task = task_map.get(task, task)
    model_name = MODEL_FEATURE_MAP.get(model_name, model_name)
    module = importlib.import_module('features.feature_{}_{}'.format(
        model_name, mapped_task))
    builder_class = getattr(module, '{}{}DataBuilder'.format(
        model_name.capitalize(), mapped_task.capitalize()))
    return builder_class
