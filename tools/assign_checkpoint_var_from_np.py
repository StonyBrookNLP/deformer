#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse

import numpy as np
import tensorflow as tf


def main(args):
    checkpoint_dir = args.checkpoint_dir
    numpy_file = args.numpy_file
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    out_file = args.out_file
    if not out_file:
        out_file = checkpoint.model_checkpoint_path
    print('loading weights from:', numpy_file)
    weights = np.load(numpy_file)
    with tf.Session() as sess:
        new_vars = []
        for var_name, var_shape in tf.train.list_variables(checkpoint_dir):
            var = tf.train.load_variable(checkpoint_dir, var_name)
            if var_name.endswith('adam_m') or var_name.endswith('adam_v'):
                continue
            if var_name in weights:
                print('{}: {} will be replaced'.format(var_name, var_shape))
                var = weights[var_name]
            if not args.dry_run:
                # Rename the variable
                new_var = tf.Variable(var, name=var_name)
                new_vars.append(new_var)
        print('saving weights to:', out_file)
        if not args.dry_run:
            # Save the variables
            saver = tf.train.Saver(new_vars)
            sess.run(tf.global_variables_initializer())
            saver.save(sess, out_file, write_meta_graph=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint_dir', type=str)
    parser.add_argument('-nf', '--numpy_file', type=str)
    parser.add_argument('-o', '--out_file', type=str)
    parser.add_argument("-dr", "--dry_run", action='store_true',
                        help="dry run renaming")

    main(parser.parse_args())
