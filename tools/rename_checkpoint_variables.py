#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse

import tensorflow as tf


def pairwise(iterable):
    """s -> (s0, s1), (s2, s3), (s4, s5), ..."""
    a = iter(iterable)
    return zip(a, a)


def main(args):
    checkpoint_dir = args.input
    patterns = args.patterns
    if len(patterns) % 2 != 0:
        raise ValueError('patterns must appear in pairs')
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)

    with tf.Session() as sess:
        new_vars = []
        for var_name, var_shape in tf.train.list_variables(checkpoint_dir):
            var = tf.train.load_variable(checkpoint_dir, var_name)
            # Set the new name
            new_name = var_name
            for p1, p2 in pairwise(patterns):
                # print(p1, p2)
                new_name = new_name.replace(p1, p2)

            if var_name == new_name:
                print('same var name, skip {}'.format(var_name))
            else:
                print('{} will be renamed to {}'.format(var_name, new_name))
            if not args.dry_run:
                # Rename the variable
                new_var = tf.Variable(var, name=new_name)
                new_vars.append(new_var)

        if not args.dry_run:
            # Save the variables
            saver = tf.train.Saver(new_vars)
            sess.run(tf.global_variables_initializer())
            saver.save(sess, checkpoint.model_checkpoint_path,
                       write_meta_graph=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='original checkpoint')
    parser.add_argument("-dr", "--dry_run", action='store_true',
                        help="dry run renaming")
    parser.add_argument('-p', '--patterns', type=str, nargs='+',
                        help='pass multiple replace patterns, '
                             'e.g. "bert/" "bert_qa/bert/"')
    main(parser.parse_args())
