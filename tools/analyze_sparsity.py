#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from collections import defaultdict

import numpy as np
from scipy.spatial import distance
from tqdm import tqdm

np.set_printoptions(threshold=np.inf, suppress=True)


def main(args):
    num_batches = args.num_batches
    bert_data = defaultdict(list)

    s_or_e_bert_data = defaultdict(list)
    print('loading data...')
    for para_idx in range(num_batches):
        bert_filename = os.path.join(args.in_dir, 'bert_b{}.npz'.format(para_idx + 1))
        bert_outputs = np.load(bert_filename)
        for k, v in bert_outputs.items():
            bert_data[k].append(v)

    print('stacking all examples of bert...')
    for k, v in bert_data.items():
        bert_data[k] = np.concatenate(v)  # stack along batch dim

    print('begin computing...')
    all_tokens_sparsity = [[] for _ in range(12)]
    all_attentions_sparsity = [[] for _ in range(12)]

    # 100 examples
    for para_idx in tqdm(range(100)):
        in_ids = bert_data['input_ids'][para_idx]
        sequence_length = np.sum(in_ids.astype(np.bool))
        for l in range(12):
            # seq, dim, e.g. [171, 768
            b_layer_vectors = bert_data['layer_{}'.format(l)][para_idx][:sequence_length]
            b_layer_vectors_sparsity = np.sum(np.isclose(np.zeros_like(b_layer_vectors),
                                                         b_layer_vectors, atol=args.threshold)) / b_layer_vectors.size
            # 12 heads, [h, seq, seq], e.g. [12, 171, 171]
            b_layer_attn = bert_data['attn_{}'.format(l)][para_idx][:, :sequence_length, :sequence_length]
            b_layer_attn_sparsity = np.sum(np.isclose(np.zeros_like(b_layer_attn),
                                                      b_layer_attn, atol=args.threshold)) / b_layer_attn.size

            all_tokens_sparsity[l].append(b_layer_vectors_sparsity)
            all_attentions_sparsity[l].append(b_layer_attn_sparsity)

    # all_para_variances has 12 list, each has 100 variances
    all_tokens_sparsity_mean = [np.mean(v) for v in all_tokens_sparsity]
    all_attentions_sparsity_mean = [np.mean(v) for v in all_attentions_sparsity]
    print(all_tokens_sparsity_mean)
    print(all_attentions_sparsity_mean)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_dir', type=str, default=None)
    parser.add_argument('-n', '--num_batches', type=int, default=4)
    parser.add_argument('-t', '--threshold', type=float, default=1e-5)
    main(parser.parse_args())
