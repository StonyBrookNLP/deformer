#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os

import ujson as json
from tqdm import tqdm


def rm_extra_space(text):
    return ' '.join(text.split())


def _read_txt(input_dir):
    files = glob.glob(input_dir + "/*/*txt")
    for file in tqdm(files, desc="read files"):
        with open(file, 'r', encoding='utf-8') as fin:
            data_raw = json.load(fin)
            yield data_raw


def main(args):
    out_file = args.output
    out_dir = os.path.dirname(out_file)
    os.makedirs(out_dir, exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as of:
        for line in tqdm(_read_txt(args.input)):
            label_str = line.get('answers', None)
            if label_str is None:
                print('\nbad example at:', line)
                print()
                continue
            article = line["article"]
            for i in range(len(line["answers"])):
                label = ord(line['answers'][i]) - ord('A')
                question = line['questions'][i]
                options = line['options'][i]

                if question.find("_") != -1:
                    question_list = [rm_extra_space(
                        question.replace('_', ' ' + options[j] + ' '))
                        for j in range(4)]
                else:
                    question_list = [rm_extra_space(
                        question + ' ' + options[j]) for j in range(4)]

                record = {
                    'id': '{}_{}'.format(line['id'], i),
                    'seq1': question_list,
                    'seq2': article,
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
    main(parser.parse_args())
