#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import re

import tensorflow as tf


def gen_variable_from_checkpoint(checkpoint):
    for var_name, var_shape in tf.train.list_variables(checkpoint):
        var = tf.train.load_variable(checkpoint, var_name)
        if var_name.endswith('adam_m') or var_name.endswith('adam_v'):
            continue
        print('{}: {} in {}'.format(var_name, var_shape, checkpoint))
        yield tf.Variable(var, name=var_name)


def main(args):
    checkpoint_one = args.checkpoint_one
    checkpoint_two = args.checkpoint_two
    out_file = args.out_file
    with tf.Session() as sess:
        new_vars = []
        for new_var in gen_variable_from_checkpoint(checkpoint_one):
            new_vars.append(new_var)
        if args.from_one:
            upper_start = args.from_one_upper
            one_var_names_map = dict()
            one_var_names = []
            for var_name, _ in tf.train.list_variables(checkpoint_one):
                if var_name.endswith('adam_m') or var_name.endswith('adam_v'):
                    continue
                one_var_names.append(var_name)
                search_result = re.search(r'layer_.*', var_name)
                if search_result:
                    layer_suffix = search_result.group()
                    one_var_names_map[layer_suffix] = var_name
            if upper_start:
                # only upper from checkpoint one,
                # lower layers and other scope from checkpoint two
                for var_name, _ in tf.train.list_variables(checkpoint_two):
                    if var_name.endswith('adam_m') or var_name.endswith('adam_v'):
                        continue
                    search_result = re.search(r'layer_.*', var_name)
                    if search_result:
                        layer_suffix = search_result.group()
                        layer_suffix = layer_suffix.replace('LayerNorm',
                                                            'layer_norm')
                        layer_str = re.search(r'layer_(\d+).*',
                                              layer_suffix).group(1)
                        layer = int(layer_str)
                        if layer >= upper_start:
                            # use variable from checkpoint one
                            one_var_name = one_var_names_map[layer_suffix]
                            one_var = tf.train.load_variable(checkpoint_one,
                                                             one_var_name)
                            new_var_name = 'init_sbert/encoder/' + layer_suffix
                            print(new_var_name, 'using',
                                  one_var_name, 'from checkpoint one')
                            new_var = tf.Variable(one_var, name=new_var_name)
                            new_vars.append(new_var)
                        else:
                            two_var = tf.train.load_variable(checkpoint_two,
                                                             var_name)
                            new_var_name = 'init_sbert/encoder/' + layer_suffix
                            print(new_var_name, 'using',
                                  var_name, ' from checkpoint two')
                            new_var = tf.Variable(two_var, name=new_var_name)
                            new_vars.append(new_var)
                    else:
                        two_var = tf.train.load_variable(checkpoint_two,
                                                         var_name)
                        new_var_name = re.sub(r'.*bert/', 'init_sbert/',
                                              var_name)
                        new_var_name = new_var_name.replace('LayerNorm',
                                                            'layer_norm')
                        print(new_var_name, 'using',
                              var_name, 'from checkpoint two')
                        new_var = tf.Variable(two_var, name=new_var_name)
                        new_vars.append(new_var)
            else:
                # all from checkpoint one
                for one_var_name in one_var_names:
                    one_var = tf.train.load_variable(checkpoint_one,
                                                     one_var_name)
                    new_var_name = re.sub(r'.*encoder/', 'init_sbert/encoder/',
                                          one_var_name)
                    new_var_name = re.sub(r'.*bert/', 'init_sbert/',
                                          new_var_name)
                    new_var_name = new_var_name.replace('LayerNorm',
                                                        'layer_norm')
                    print(new_var_name, 'using',
                          one_var_name, 'from checkpoint one')
                    new_var = tf.Variable(one_var, name=new_var_name)
                    new_vars.append(new_var)

        else:
            for new_var in gen_variable_from_checkpoint(checkpoint_two):
                new_vars.append(new_var)

        print('saving weights to:', out_file)
        if not args.dry_run:
            # Save the variables
            saver = tf.train.Saver(new_vars)
            sess.run(tf.global_variables_initializer())
            saver.save(sess, out_file, write_meta_graph=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c1', '--checkpoint_one', type=str)
    parser.add_argument('-c2', '--checkpoint_two', type=str)
    parser.add_argument('-o', '--out_file', type=str)
    parser.add_argument("-dr", "--dry_run", action='store_true',
                        help="dry run renaming")
    parser.add_argument("-fo", "--from_one", action='store_true',
                        help="init from checkpoint one")
    parser.add_argument("-fou", "--from_one_upper", type=int, default=0,
                        help="init from checkpoint one upper layer")

    main(parser.parse_args())
