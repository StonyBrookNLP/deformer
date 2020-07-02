#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
loader for data pre-processing

data_loader loads the eqa format into required model input format,
the output will be processed examples (json) and TFRecords (binary)

"""
import argparse
import os

from common import DATA_DIR
from common import RES_DIR
from common.config import get_config_from_file_or_string
from features import get_example_builder
from models import vocab_map
from tasks import max_first_length_map
from tasks import max_seq_length_map
from tasks import task_map

default_config_string = """
[DEFAULT]
model =
vocab_file =
dataset_file =
output_file = 
# save_mode: tf, jsonl, both (default)
# tf means TFRecord format, jsonl means json line format, both means save both
# train dataset is saved to TFRecord binary file,
# dev dataset is saved to json line examples, (currently need both format)
save_mode =

# print raw feature for questions ranging from min to max
debug_min = 0
debug_max = 10

lower_case = yes
context_stride = 128
slide_window = yes

num_choices = 

# max number of input (text_a+text_b+flags) tokens
max_seq_length =

# for ebert
max_first_length =
"""


def main(args):
    config = get_config_from_file_or_string(default_config_string)

    config.model = args.model
    config.save_mode = args.save_mode

    task = args.task
    config.max_seq_length = args.max_seq_length or max_seq_length_map[task]
    max_first_length = max_first_length_map[task]
    config.max_first_length = args.max_first_length or max_first_length

    mapped_task = task_map.get(task, task)

    input_file = args.input_file
    vocab_file = args.vocab_file
    if not vocab_file:
        vocab_file = os.path.join(RES_DIR, vocab_map[args.model])
    config.vocab_file = vocab_file

    if not input_file:
        converted_data_dir = os.path.join(DATA_DIR, 'datasets', 'deformer')
        input_file = os.path.join(converted_data_dir,
                                  '{}-{}.jsonl'.format(task, args.split))
    config.dataset_file = input_file

    output_file = args.output_file
    if not output_file:
        converted_data_dir = os.path.join(DATA_DIR, 'datasets', 'converted',
                                          args.model)
        msl = args.max_seq_length
        output_file = os.path.join(converted_data_dir, str(msl) if msl else '',
                                   '{}-{}.out'.format(task, args.split))
    config.output_file = output_file
    print('config: \n{}'.format(config))
    builder = get_example_builder(config.model, task=mapped_task)(config)
    builder.build_examples()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', default=None,
                        help='input dataset file')
    parser.add_argument('-o', '--output_file', default=None,
                        help='output dataset file')
    parser.add_argument('-vf', '--vocab_file', default=None,
                        help='vocab file')
    parser.add_argument('-m', '--model', type=str, default='xlnet',
                        choices=('bert', 'ebert', 'common', 'xlnet',
                                 'exlnet'),
                        help='choose model to load default configuration')
    parser.add_argument('-t', '--task', type=str, default='squad_v1.1',
                        choices=('squad_v1.1', 'squad_v2.0', 'hotpot',
                                 'mnli', 'qqp', 'boolq', 'race'),
                        help='choose model to load default configuration')
    parser.add_argument('-s', '--split', default='debug',
                        choices=('train', 'dev', 'tune', 'debug'))
    parser.add_argument('-sm', '--save_mode', default='both',
                        choices=('tf', 'jsonl', 'both'))
    parser.add_argument('-msl', '--max_seq_length', type=int, default=0)
    parser.add_argument('-mfl', '--max_first_length', type=int, default=0)
    main(parser.parse_args())
