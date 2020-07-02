#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
from collections import Counter, defaultdict

import json_lines
from tqdm import tqdm


def word_tokenize(text):
    """Split on whitespace and punctuation."""
    return re.findall(r'\w+|[^\w\s]', text, re.U)


def get_stats(hist):
    average = 0.0
    total_values = 0.0
    for val in hist:
        total_values += hist[val]
        average += float(hist[val]) * float(val)
    if total_values:
        return average / total_values, total_values
    else:
        return 0, 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', type=str, default=None)
    parser.add_argument('-s', '--show', action='store_true')

    # dev_v2.1 dev-v2.0
    args = parser.parse_args()
    print('begin processing...')

    data_file = args.data_file
    data_filename = os.path.basename(data_file)
    tokens_dict = dict()

    key_set = {'seq1', 'seq2'}
    with json_lines.open(data_file) as f:
        for item in tqdm(f):
            all_tokens_key = []
            all_tokens_sum = 0
            for k, v in item.items():
                # if k.endswith('_tokens'):
                if k in key_set:
                    if not isinstance(v, list):
                        v = [v]
                    for vi in v:
                        k_token_len = len(word_tokenize(vi))
                        counter_dict = tokens_dict.get(k, defaultdict(int))
                        counter_dict[k_token_len] += 1
                        tokens_dict[k] = counter_dict
                        all_tokens_key.append(k)
                        all_tokens_sum += k_token_len
                if k == 'label':
                    ans_val = v['ans']
                    if ans_val:
                        answer = ans_val[0][1]
                        answer_len = len(word_tokenize(answer))
                        counter_dict = tokens_dict.get(k, defaultdict(int))
                        counter_dict[answer_len] += 1
                        tokens_dict[k] = counter_dict
            total_tokens_str = '+'.join(all_tokens_key)
            counter_dict = tokens_dict.get(total_tokens_str, defaultdict(int))
            counter_dict[all_tokens_sum] += 1
            tokens_dict[total_tokens_str] = counter_dict

    print("###################################\n")
    for feature_name, length_dict in tokens_dict.items():
        feature_counter = Counter(length_dict)
        feature_length_avg, total_feature_lengths = get_stats(feature_counter)
        feature_length_max = max(list(feature_counter.keys()))
        feature_length_min = min(list(feature_counter.keys()))
        num_sum = 0
        for key in sorted(feature_counter.keys()):
            num = feature_counter[key]
            num_sum += num
            print("{}:len={}:num={}:num_sum:{:.4f}".format(feature_name, key, num, num_sum / total_feature_lengths))

        print("{} length avg={:.1f}, max={}, min={}".format(
            feature_name, feature_length_avg, feature_length_max, feature_length_min))
        print("###################################\n")

    if args.show:
        import matplotlib.pyplot as plt
        import matplotlib._color_data as mcd
        import matplotlib.ticker as ticker

        num_colors = len(mcd.XKCD_COLORS)
        color_names = list(mcd.XKCD_COLORS.keys())
        fig, axes = plt.subplots(len(tokens_dict))
        for n, ((feature_name, length_dict), axis) in enumerate(zip(tokens_dict.items(), axes)):
            data = []
            for k, v in sorted(length_dict.items()):
                data.extend([k] * v)

            cn = mcd.XKCD_COLORS[color_names[n % num_colors]]
            cn2 = mcd.XKCD_COLORS[color_names[(n + 1) % num_colors]]
            feature_counter = Counter(length_dict)
            axis.bar(list(feature_counter.keys()), feature_counter.values(), color=cn, label='histogram')
            axis.set_title(feature_name)
            axis.set_xlabel("tokens lengths")
            axis.set_ylabel("count of tokens lengths")
            axis.xaxis.grid()
            # axis.xaxis.set_major_locator(ticker.MultipleLocator(15))
            axis.legend(loc='upper left')

            ax1 = axis.twinx()
            ax1.plot(data, [i / len(data) for i in range(len(data))], color=cn2, label='cdf')
            ax1.set_ylabel("cdf of tokens lengths")

            ax1.yaxis.grid()
            # ax1.xaxis.set_major_locator(ticker.MultipleLocator(15))
            ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
            ax1.legend(loc='upper right')
        fig.suptitle(data_filename, fontsize=16)
        plt.tight_layout()
        plt.show()
    print('all done.')
