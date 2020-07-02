#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import os

from . import CONFIG_DIR
from . import logger
from . import tf
from .io import read_config
from tasks import max_first_length_map
from tasks import max_seq_length_map
from tasks import num_classes
from tasks import num_choices
from tasks import task_map


class Config(dict):

    def __init__(self, local_config):
        dict.__init__(self, dict(local_config.items('default')))
        self.__dict__ = self

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)
        self.__dict__ = self

    def __getattribute__(self, name):
        try:
            attr = object.__getattribute__(self, name)
        except AttributeError:
            if not name.startswith('_'):
                logger.warning('\033[1m\033[35mCAUTION: no {} attribute, '
                               'will default to None!\033[0m'.format(name))
            return None
        return attr

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def __repr__(self):
        return '\n'.join(['{}: {}'.format(i[0], i[1])
                          for i in sorted(self.__dict__.items())])


MODES = ('train', 'dev', 'infer', 'tune', 'analyze')


def get_config_from_file_or_string(config_file_or_string, mode='infer'):
    if mode not in MODES:
        raise ValueError('\033[1m\033[31mmode: {} not supported, '
                         'must be one of {}\033[0m'.format(mode, MODES))
    config = read_config(config_file_or_string)
    preprocess_config(config, mode)
    return Config(config)


def extract_example_size(filename):
    import re
    try:
        s = re.search(r'(?s:.*)\.(\d+)', filename)
        num = int(s.group(1))
    except ValueError as e:
        logger.error('\033[1m\033[31mcannot extract size info from'
                     ' {}\033[0m'.format(filename))
        raise e
    return num


def preprocess_config(config, mode):
    for k, v in config.items('default'):
        if k in os.environ:
            config.set('default', k, os.environ[k])
            logger.info('\033[1m\033[34m{} set to env {} instead of '
                        'provided {}\033[0m'.format(k, os.environ[k], v))
        v = config.get('default', k)
        if isinstance(v, str):
            # may expand user path
            config.set('default', k, os.path.expanduser(v))

    if mode == 'infer':
        config.dataset_file = None

    if config.dataset_file:
        dataset_files = tf.io.gfile.glob(config.dataset_file)
        if not dataset_files:
            raise ValueError('\033[1m\033[31mdataset file: {} not exist!'
                             '\033[0m'.format(config.dataset_file))
        else:
            config.dataset_file = dataset_files[0]
        # may check file existence
        logger.info('({}) dataset_file: {}'.format(mode, config.dataset_file))
        if not tf.io.gfile.exists(config.dataset_file):
            raise ValueError('\033[1m\033[31mfile:{} not exist!'
                             '\033[0m'.format(config.dataset_file))

        # may extract dataset_size size from file name
        if not config.dataset_size:
            dataset_size = extract_example_size(config.dataset_file)
            config.dataset_size = dataset_size

    if mode == 'dev':
        # TODO: only need the ground truth file for predictions
        ground_truth_files = tf.io.gfile.glob(config.ground_truth_file)
        if not ground_truth_files:
            raise ValueError('\033[1m\033[31mground_truthfile:{} not exist!'
                             '\033[0m'.format(config.ground_truth_file))
        else:
            config.ground_truth_file = ground_truth_files[0]


def get_config_from_args(args, mode):
    config_file = args.config_file
    task = args.task
    os.environ['task'] = task
    os.environ['mode'] = mode
    if not config_file:
        model = args.model
        if not model:
            raise ValueError('pass either config_file (--config_file, -c)'
                             ' or model (--model, -m) !')
        mapped_task = task_map.get(task, task)
        config_file = os.path.join(CONFIG_DIR,
                                   '{}_{}.ini'.format(model, mapped_task))
    config_file = os.path.expanduser(config_file)
    logger.info('config_file: {}'.format(config_file))
    config = get_config_from_file_or_string(config_file, mode)

    config.max_seq_length = config.max_seq_length or max_seq_length_map[task]
    mfl = config.get('max_first_length', 0)
    config.max_first_length = mfl or max_first_length_map[task]
    config.num_classes = config.get('num_classes', 0) or num_classes.get(task,
                                                                         0)
    config.num_choices = config.get('num_choices', 0) or num_choices.get(task,
                                                                         0)

    return config
