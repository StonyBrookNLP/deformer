#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import sys

import numpy as np
import tensorflow as tf


def main(args):
    checkpoint_path = args.checkpoint_path
    variable_names = args.variable_names
    if not variable_names:

        variables = tf.train.list_variables(checkpoint_path)
        var_dict = dict()
        for var in variables:
            name, shape = var
            if name.endswith('adam_m') or name.endswith('adam_v'):
                continue
            print(name, shape)
            var_dict[name] = tf.train.load_variable(checkpoint_path, name)
        save_path = args.save_path
        if save_path:
            np.savez_compressed(save_path, **var_dict)
        return

    if args.iterator:
        checkpoints_iterator = tf.train.checkpoints_iterator(
            checkpoint_path, timeout=args.time_out)
    else:
        checkpoints_iterator = [checkpoint_path]
    for checkpoint_path in checkpoints_iterator:
        # print(checkpoint_path)
        if tf.io.gfile.isdir(checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        step = re.findall(r'\d+$', checkpoint_path)[0]
        for variable_name in variable_names:
            print('step={},{}={}'.format(
                step, variable_name, tf.train.load_variable(checkpoint_path,
                                                            variable_name)))
            sys.stdout.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint_path', type=str)
    parser.add_argument('-s', '--save_path', type=str)
    parser.add_argument('-i', '--iterator', action='store_true')
    parser.add_argument('-to', '--time_out', type=int, default=600)
    parser.add_argument('-v', '--variable_names', type=str, nargs='+',
                        help='pass one or multiple variable names')
    main(parser.parse_args())
