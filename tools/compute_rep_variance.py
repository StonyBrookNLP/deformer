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

    sbert_data = defaultdict(list)
    print('loading data...')
    for para_idx in range(num_batches):
        bert_filename = os.path.join(args.in_dir, 'bert_b{}.npz'.format(para_idx + 1))
        bert_outputs = np.load(bert_filename)
        for k, v in bert_outputs.items():
            if k.startswith('attn_'):
                continue
            bert_data[k].append(v)
        print('loaded {}'.format(bert_filename))
        if args.compare:
            sbert_filename = os.path.join(args.in_dir, 'sbert_b{}.npz'.format(para_idx + 1))
            sbert_outputs = np.load(sbert_filename)
            for k, v in sbert_outputs.items():
                sbert_data[k].append(v)

    if args.compare:
        for k, v in sbert_data.items():
            sbert_data[k] = np.concatenate(v)  # stack along batch dim

    print('stacking all examples...')
    for k, v in bert_data.items():
        bert_data[k] = np.concatenate(v)  # stack along batch dim

    print('begin computing...')
    all_para_variances = [[] for _ in range(12)]
    all_para_means = [[] for _ in range(12)]
    # 100 paragraphs
    size = bert_data['input_ids'].shape[0]
    print('data size={}'.format(size))
    for para_idx in tqdm(range(0, size, 5)):
        same_para_in_ids = bert_data['input_ids'][para_idx:para_idx + 5]
        same_para_seg_ids = bert_data['segment_ids'][para_idx:para_idx + 5]
        same_para_feature_ids = bert_data['feature_id'][para_idx:para_idx + 5]
        # q_ids = features["question_ids"]
        # c_ids = features["context_ids"]
        # first_lengths = np.sum(q_ids.astype(np.bool), 1)
        # second_lengths = np.sum(c_ids.astype(np.bool), 1)
        sequence_lengths = np.sum(same_para_in_ids.astype(np.bool), 1)
        second_lengths = np.sum(same_para_seg_ids.astype(np.bool), 1)
        if not np.all(second_lengths == second_lengths[0]):
            # exceed 320, so passage got shifted due to different question lengths
            print('shifted paragraphs:', same_para_feature_ids, second_lengths)
            continue
        first_lengths = sequence_lengths - second_lengths
        # print(same_para_feature_ids, first_lengths, sequence_lengths)
        for l in range(12):
            layer_vectors = bert_data['layer_{}'.format(l)][para_idx:para_idx + 5]
            # print(layer_vectors[0].shape)
            # pvs is layer tokens vectors for the same paragraph
            pvs = [layer_vectors[i][f:s] for i, (f, s) in enumerate(zip(first_lengths, sequence_lengths))]
            # pvs_m is the centroid vector of those 5 paragraph vectors
            pvs_m = np.mean(pvs, axis=0)

            # calculate variance of distances of 5 paragraph vectors to the centroid
            p_dist = [np.mean([distance.cosine(pvst, pvs_mi)
                               for pvst, pvs_mi in zip(pvsi, pvs_m)])
                      for pvsi in pvs]
            p_variance = np.var(p_dist)
            p_mean = np.mean(p_dist)
            all_para_means[l].append(p_mean)
            all_para_variances[l].append(p_variance)
    # all_para_variances has 12 list, each has 100 variances
    all_para_mean_mean = [np.mean(v) for v in all_para_means]
    all_para_mean_variances = [np.var(v) for v in all_para_means]
    all_para_var_mean = [np.mean(v) for v in all_para_variances]
    all_para_var_variances = [np.var(v) for v in all_para_variances]
    print('mean mean={}'.format(all_para_mean_mean))
    print('mean var={}'.format(all_para_mean_variances))
    print('var mean={}'.format(all_para_var_mean))
    print('var var={}'.format(all_para_var_variances))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_dir', type=str, default=None)
    parser.add_argument('-n', '--num_batches', type=int, default=2)
    parser.add_argument('-c', '--compare', action='store_true')
    main(parser.parse_args())
