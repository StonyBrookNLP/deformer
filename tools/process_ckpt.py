#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse

import tensorflow as tf


def main(args):
    checkpoint_dir = args.input
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)

    with tf.Session() as sess:
        new_vars = []
        for var_name, var_shape in tf.train.list_variables(checkpoint_dir):
            var = tf.train.load_variable(checkpoint_dir, var_name)
            if var_name.startswith('sbert_classifier/ebert_classifier/') or \
                var_name.endswith('adam_m') or var_name.endswith('adam_v'):
                continue

            # Set the new name
            new_name = var_name.replace('sbert_classifier/', 'ebert_classifier/')

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
    main(parser.parse_args())
