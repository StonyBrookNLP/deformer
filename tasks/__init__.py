#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import abstractmethod

task_map = {
    'mnli': 'classifier',
    'qqp': 'classifier',
    'boolq': 'classifier',
    'squad_v1.1': 'qa',
    'race': 'classifier',
    'squad_v2.0': 'qa',
    'hotpot': 'qa',
    'sts': 'regression',
}

max_seq_length_map = {
    'mnli': 120,
    'qqp': 100,
    'boolq': 320,
    'squad_v1.1': 320,
    'race': 512,
    'squad_v2.0': 320,
    'hotpot': 2048,
}

num_classes = {
    'mnli': 3,
    'qqp': 2,
    'boolq': 2,
    'squad_v1.1': 0,
    'race': 4,
    'squad_v2.0': 2,
    'hotpot': 3,
}

num_choices = {
    'race': 4,
}

max_first_length_map = {
    'mnli': 85,
    'qqp': 40,
    'boolq': 15,
    'squad_v1.1': 25,
    'race': 30,
    'squad_v2.0': 25,
    'hotpot': 40,
}


def get_task_model_class(model_name, task):
    """ implements the task specific models under ..tasks folder,
    following the convention:
    task_model.py, where the actual model class name is ModelTask
    """
    import importlib
    mapped_task = task_map.get(task, task)
    module = importlib.import_module('tasks.{}_{}'.format(
        mapped_task, model_name))
    task_model_class = getattr(module, '{}{}'.format(model_name.capitalize(),
                                                     mapped_task.capitalize()))
    return task_model_class


class TaskModel(object):

    @staticmethod
    @abstractmethod
    def get_inputs(inputs, config):
        pass

    @abstractmethod
    def get_logits(self, features, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def get_prediction(features, outputs):
        pass

    @staticmethod
    @abstractmethod
    def generate_predictions(logits_dict, examples, config):
        pass
