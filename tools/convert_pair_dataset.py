#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv

import ujson as json
from tqdm import tqdm


def main(args):
    id_map = {
        'mnli': 'pairID',
        'qqp': 'id',
        'boolq': 'idx',
        'sts': 'index',
    }
    text_a_map = {
        'mnli': 'sentence1',
        'qqp': 'question1',
        'boolq': 'question',
        'sts': 'sentence1',
    }
    text_b_map = {
        'mnli': 'sentence2',
        'qqp': 'question2',
        'boolq': 'passage',
        'sts': 'sentence2',
    }
    label_map = {
        'mnli': 'gold_label',
        'qqp': 'is_duplicate',
        'boolq': 'label',
        'sts': 'score',
    }
    number_label = {
        'mnli': lambda x: ["contradiction", "entailment", "neutral"].index(x),
        'qqp': lambda x: int(x),
        'boolq': lambda x: int(x),  # 0 for False, 1 for True
        'sts': lambda x: float(x)
    }
    task = args.task
    with open(args.output, 'w',
              encoding='utf-8') as of, open(args.input, "r",
                                            encoding="utf-8") as f:
        if task == 'boolq':
            reader = f
        else:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for i, line in tqdm(enumerate(reader)):
            if task == 'boolq':
                line = json.loads(line.strip())
            label_str = line.get(label_map[task], None)
            if label_str is None:
                print('\nbad example at:', i + 2, line)
                print()
                continue
            label = number_label[task](label_str)
            record = {
                'id': line[id_map[task]],
                'seq1': line[text_a_map[task]],
                'seq2': line[text_b_map[task]],
                "label": {
                    "cls": label,
                    "ans": []
                }
            }
            of.write(json.dumps(record))
            of.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='original pair dataset file')
    parser.add_argument('output', type=str)
    parser.add_argument('-t', '--task', type=str, default='qqp',
                        choices=('mnli', 'qqp', 'sts', 'boolq'),
                        help='choose task to run')
    main(parser.parse_args())
