#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os

import numpy as np


def main(args):
    # calculate mean norm for attention weights
    bert_qa_weights = np.load(args.input_numpy_file)
    norm_ord = 2
    rows = 12
    weight_names = ['attention/self/query', 'attention/self/key', 'attention/self/value', 'attention/output/dense']
    layer_norms = []
    layer_weights = []
    for weight_name in weight_names:
        norm = []
        layer_weight = []
        if weight_name.endswith('dense'):
            axis = 1  # attention/output/dense
        else:
            axis = 0
        for i in range(rows):
            weight_key = 'bert_qa/bert/encoder/layer_{}/{}/kernel'.format(i, weight_name)
            weight = bert_qa_weights[weight_key]
            layer_weight.append(weight)
            print('calculating norm for {} layer {}'.format(weight_name, i))
            weight_norm = np.linalg.norm(weight, norm_ord, axis=axis)
            head_weight = weight_norm.reshape(12, 64)
            head_weight_mean = np.mean(head_weight, 1)
            norm.append(head_weight_mean)
        layer_weights.append(layer_weight)
        layer_norms.append(norm)
    print('sorting heads l2 norm')

    layer_norms_mean = np.mean(layer_norms, 0)
    layer_norms_mean_idx = np.argsort(layer_norms_mean, 1)

    pruned_layer_weights = [[np.delete(layer_norms_mean[l][i], layer_norms_mean_idx[i, :3], 0) for i in range(rows)]
                            for l in range(len(weight_names))]

    new_weights = dict()
    prune_num = args.prune_num
    for j, weight_name in enumerate(weight_names):
        for i in range(rows):
            layer_weight = layer_weights[j][i]
            weight_key = 'bert_qa/bert/encoder/layer_{}/{}/kernel'.format(i, weight_name)
            bias_key = 'bert_qa/bert/encoder/layer_{}/{}/bias'.format(i, weight_name)
            layer_bias = bert_qa_weights[bias_key]
            print('deleting {} heads for {} layer {}'.format(prune_num, weight_name, i))

            if weight_name.endswith('dense'):
                # attention/output/dense
                layer_head_weight = layer_weight.reshape(12, 64, 768)
                pruned_layer_head_weight = np.delete(layer_head_weight, layer_norms_mean_idx[i, :prune_num], 0)
                pruned_layer_weight = pruned_layer_head_weight.reshape(-1, 768)
                pruned_layer_bias = layer_bias
            else:
                # 'attention/self/query', 'attention/self/key', 'attention/self/value'
                layer_head_weight = layer_weight.reshape(768, 12, 64)
                pruned_layer_head_weight = np.delete(layer_head_weight, layer_norms_mean_idx[i, :prune_num], 1)
                pruned_layer_weight = pruned_layer_head_weight.reshape(768, -1)

                layer_head_bias = layer_bias.reshape(12, 64)
                pruned_layer_head_bias = np.delete(layer_head_bias, layer_norms_mean_idx[i, :prune_num], 0)
                pruned_layer_bias = pruned_layer_head_bias.flatten()

            new_weights[weight_key] = pruned_layer_weight
            new_weights[bias_key] = pruned_layer_bias

    output_numpy_file = args.output_numpy_file
    if not output_numpy_file:
        output_prefix, ext = os.path.splitext(args.input_numpy_file)
        output_numpy_file = output_prefix + '_p{}'.format(prune_num)
    print('saving pruned weights to:', output_numpy_file)
    np.savez_compressed(output_numpy_file, **new_weights)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_numpy_file', type=str, default='data/ckpt/bert_qa.npz')
    parser.add_argument('-o', '--output_numpy_file', type=str, default=None)
    parser.add_argument('-n', '--prune_num', type=int, default=6)

    main(parser.parse_args())
