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

        sbert_filename = os.path.join(args.in_dir, '{}_b{}.npz'.format(args.model, para_idx + 1))
        sbert_outputs = np.load(sbert_filename)
        for k, v in sbert_outputs.items():
            s_or_e_bert_data[k].append(v)

    print('stacking all examples of both bert and {}...'.format(args.model))
    for k, v in s_or_e_bert_data.items():
        s_or_e_bert_data[k] = np.concatenate(v)  # stack along batch dim

    for k, v in bert_data.items():
        bert_data[k] = np.concatenate(v)  # stack along batch dim

    print('begin computing...')
    all_para_distances = [[] for _ in range(12)]
    all_q_distances = [[] for _ in range(12)]
    # 500 examples paragraphs
    for para_idx in tqdm(range(500)):
        in_ids = bert_data['input_ids'][para_idx]
        seg_ids = bert_data['segment_ids'][para_idx]
        feature_ids = bert_data['feature_id'][para_idx]
        q_ids = s_or_e_bert_data["question_ids"][para_idx]
        c_ids = s_or_e_bert_data["context_ids"][para_idx]
        q_length = np.sum(q_ids.astype(np.bool))
        c_length = np.sum(c_ids.astype(np.bool))
        sequence_length = np.sum(in_ids.astype(np.bool))
        second_length = np.sum(seg_ids.astype(np.bool))
        first_length = sequence_length - second_length
        if not (c_length == second_length):
            print('shifted paragraphs:', feature_ids, c_length, second_length)
            continue
        if not (q_length == first_length):
            print('shifted questions:', feature_ids, q_length, first_length)
            continue
        for l in range(12):
            b_layer_vectors = bert_data['layer{}'.format(l)][para_idx]
            s_layer_vectors = s_or_e_bert_data['layer{}'.format(l)][para_idx]

            # b_pvs is layer paragraph tokens vectors for bert
            b_pvs = b_layer_vectors[first_length:second_length]
            s_pvs = s_layer_vectors[len(q_ids):len(q_ids) + c_length]

            # calculate variance of distances of 5 paragraph vectors to the centroid
            p_dist = np.mean([distance.cosine(b_p, s_p) for b_p, s_p in zip(b_pvs, s_pvs)])
            all_para_distances[l].append(p_dist)

            # q_pvs is layer question tokens vectors for bert
            b_qvs = b_layer_vectors[:first_length]
            s_qvs = s_layer_vectors[:q_length]

            q_dist = np.mean([distance.cosine(b_q, s_q) for b_q, s_q in zip(b_qvs, s_qvs)])
            all_q_distances[l].append(q_dist)

    # all_para_variances has 12 list, each has 100 variances
    all_para_mean_variances = [np.mean(v) for v in all_para_distances]
    all_q_mean_variances = [np.mean(v) for v in all_q_distances]
    print(all_para_mean_variances)
    print(all_q_mean_variances)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_dir', type=str, default=None)
    parser.add_argument('-n', '--num_batches', type=int, default=20)
    parser.add_argument('-m', '--model', type=str, default='sbert', choices=('ebert', 'sbert'),
                        help='choose which model compare distance')
    main(parser.parse_args())
