#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import uuid

import ujson as json
from tqdm import tqdm

"""
SQuAD format:

{
  "data": [
    {
      "title": "Super_Bowl_50",
      "paragraphs": [
        {
          "context": "Super Bowl 50 was an American football game...",
          "qas": [
            {
              "answers": [
                {
                  "answer_start": 177,
                  "text": "Denver Broncos"
                },
                ...
              ],
              "question": "Which NFL team...",
              "id": "56be4db0acb8001400a502ec"
            },
            ...,
            ]
        },
        ...,
        ]
    }
    ....,
  ]
}

"""


def main(args):
    with open(args.input, encoding='utf-8') as f:
        dataset = json.load(f)

    with open(args.output, 'w', encoding='utf-8') as f:
        for article in tqdm(dataset['data']):
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                para_id = uuid.uuid5(uuid.NAMESPACE_DNS, context).hex
                for qa in paragraph['qas']:
                    question = qa['question']
                    answer = []
                    for ans in qa['answers']:
                        # ans_info = {'start': a['answer_start'],
                        # 'text': a['text']}
                        ans_info = (ans['answer_start'], ans['text'])
                        if ans_info in answer:  # remove duplicates
                            continue
                        else:
                            answer.append(ans_info)
                    # possible: 0, impossible: 1, (hotpot qa) yes/no: 2
                    is_impossible = qa.get("is_impossible", False)
                    record = {
                        'id': '{}:{}'.format(para_id, qa['id']),
                        'seq1': question,
                        'seq2': context,
                        'label': {'cls': 1 if is_impossible else 0,
                                  'ans': answer},
                    }

                    f.write(json.dumps(record, ensure_ascii=False))
                    f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='original squad dataset file')
    parser.add_argument('output', type=str)
    main(parser.parse_args())
