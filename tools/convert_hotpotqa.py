#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import ujson as json
from tqdm import tqdm

"""
Hotpot QA format:
[
  {
    "_id": "5a8b57f25542995d1e6f1371",
    "answer": "yes",
    "question": "Were Scott Derrickson and Ed Wood...?",
    "supporting_facts": [
      [
        "Scott Derrickson",
        0
      ],
      ...,
    ],
    "context": [
      [
        "Ed Wood (film)",
        [
          "Ed Wood is a 1994....",
          " The film concerns ....",
          " Sarah Jessica Parker, ..."
        ]
      ],
      ...,
    ],
    "type": "comparison",
    "level": "hard"
  }
  ...,
]
"""


def rm_extra_space(text):
    return ' '.join(text.split())


def main(args):
    with open(args.input, encoding='utf-8') as f:
        dataset = json.load(f)

    with open(args.output, 'w', encoding='utf-8') as f:
        for example in tqdm(dataset):
            question = example['question']
            answer = example['answer']
            sentences = []
            for c in example['context']:
                if len(c) != 2:
                    print('bad list in context, skip!')
                    continue
                sentences.append(c[0])
                sentences.extend(c[1])
            context = rm_extra_space(' '.join(sentences))
            answer_lower = answer.lower()
            answers = []
            if answer_lower == 'yes':
                cls = 1
            elif answer_lower == 'no':
                cls = 2
            else:
                # span answer
                cls = 0
                answer_pattern = r'(?<!\w){}(?!\w)'.format(re.escape(answer))
                for match in re.finditer(answer_pattern, context):
                    answers.append((match.start(), answer))
                if not answers:
                    if answer in context:
                        idx = context.find(answer)
                        answers.append((idx, answer))
                        # getting more possible answers
                        while True:
                            idx = context.find(answer, idx + len(answer))
                            answer_info = (idx, answer)
                            if idx != -1:
                                answers.append(answer_info)
                            else:
                                break
                    else:
                        print('skip invalid answer, id={}, q={}, a={}'.format(
                            example['_id'], question, answer))
                        continue
            record = {
                'id': example['_id'],
                'seq1': question,
                'seq2': context,
                'label': {'cls': cls,
                          'ans': answers},
            }

            f.write(json.dumps(record, ensure_ascii=False))
            f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='original hotpot dataset file')
    parser.add_argument('output', type=str)
    main(parser.parse_args())
