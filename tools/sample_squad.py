#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import uuid

import ujson as json
from tqdm import tqdm
import random


def main(args):
    reproducible_random = random.Random(args.random_seed)
    num_paragraphs = args.num_paragraphs
    num_questions_per_paragraph = args.num_questions_per_paragraph

    with open(args.input, encoding='utf-8') as f:
        dataset = json.load(f)
    all_paragraphs = []
    for article in tqdm(dataset['data']):
        for paragraph in article['paragraphs']:
            if len(paragraph['qas']) < num_questions_per_paragraph:
                continue
            all_paragraphs.append(paragraph)
    reproducible_random.shuffle(all_paragraphs)
    with open(args.output, 'w', encoding='utf-8') as f:
        for paragraph in all_paragraphs[:num_paragraphs]:
            context = paragraph['context']
            para_id = uuid.uuid5(uuid.NAMESPACE_DNS, context).hex
            all_qas = paragraph['qas']
            reproducible_random.shuffle(all_qas)
            for qa in all_qas[:num_questions_per_paragraph]:
                question = qa['question']
                answer = []
                for a in qa['answers']:
                    # a_info = {'start': a['answer_start'], 'text': a['text']}
                    a_info = (a['answer_start'], a['text'])
                    if a_info in answer:  # remove duplicates
                        continue
                    else:
                        answer.append(a_info)

                record = {
                    'id': '{}:{}'.format(para_id, qa['id']),
                    'question': question,
                    'context': context,
                    'answer': answer
                }
                f.write(json.dumps(record))
                f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='converted squad dataset deformer file')
    parser.add_argument('output', type=str)
    parser.add_argument('-rs', '--random_seed', type=int, default=0)
    parser.add_argument('-np', '--num_paragraphs', type=int, default=100)
    parser.add_argument('-nq', '--num_questions_per_paragraph', type=int, default=5)
    main(parser.parse_args())
