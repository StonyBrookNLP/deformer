#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bert_qa/bert/encoder/layer_0/intermediate/dense/kernel
bert_qa/bert/encoder/layer_0/output/dense/kernel
"""
import argparse
import os
import re
import sys

import numpy as np
import tensorflow as tf


def svd(weight, units, debug=True):
    u_mat, s_mat, v_mat = np.linalg.svd(weight, full_matrices=True)
    # cum_norm = np.cumsum(s_mat) / np.sum(s_mat)
    # Getting the sub-space dimension
    sub_dim = units  # np.sum(cum_norm <= ratio)
    u = u_mat[:, 0:sub_dim]
    v = np.mat(np.diag(s_mat[0:sub_dim])) * np.mat(v_mat[0:sub_dim, :])
    if debug:
        error = np.power(weight - np.matmul(u, v), 2)
        mean_error = np.mean(np.sqrt(np.sum(error, 1)))
        print('[performSVD] Mean Error = {:.4f}'.format(mean_error))
        sys.stdout.flush()

    return u, np.squeeze(np.asarray(v))


def main(args):
    checkpoint_path = args.checkpoint_path
    checkpoint = tf.train.get_checkpoint_state(checkpoint_path)
    checkpoint_dir, checkpoint_name = os.path.split(checkpoint.model_checkpoint_path)
    units = args.units
    new_path = os.path.join(checkpoint_dir, 'svd_{}'.format(units), checkpoint_name)
    print('will save new checkpoint to {}'.format(new_path))
    sys.stdout.flush()
    with tf.Session() as sess:
        save_vars = []
        for var_name, var_shape in tf.train.list_variables(checkpoint_path):
            if 'adam_' in var_name:
                # skip gradients
                continue
            var = tf.train.load_variable(checkpoint_path, var_name)
            if re.match(r"bert_qa/bert/encoder/layer_\d+/(intermediate|output)/dense/kernel", var_name):
                print('begin svd for {}...'.format(var_name))
                var0, var1 = svd(var, units)
                var0_name = var_name.replace('dense', 'dense0')
                save_vars.append(tf.Variable(var0, name=var0_name))
                save_vars.append(tf.Variable(var1, name=var_name))
            # elif re.match(r"bert_qa/bert/encoder/layer_\d+/output/dense/kernel", var_name):
            #     var0, var1 = svd(var, units)
            #     save_vars[var_name] = var0
            #     save_vars[var_name.replace('dense', 'dense1')] = var0
            else:
                save_vars.append(tf.Variable(var, name=var_name))

        # Save the variables
        saver = tf.train.Saver(save_vars)
        sess.run(tf.global_variables_initializer())
        saver.save(sess, new_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint_path', type=str)
    parser.add_argument('-u', '--units', type=int, default=512, help='pass hidden units')
    parser.add_argument('-l', '--layers', type=str, default='all',
                        help='which layer to apply svd, syntax: 0,4;8,10 '
                             'will be layer0 to layer4 and layer8 to layer10')
    main(parser.parse_args())
