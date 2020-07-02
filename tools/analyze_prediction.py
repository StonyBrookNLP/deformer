#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import re
import string


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_fn(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def is_correct(predicted_text, ground_truths):
    score = metric_max_over_ground_truths(exact_match_fn, predicted_text,
                                          ground_truths)
    return score


def main(args):
    base_dir = 'data/predictions'
    gt_file = args.gt_file or os.path.join(
        base_dir, 'squad_v1.1-dev.groudtruth.jsonl')
    ori_pred_file = args.ori_pred_file or os.path.join(
        base_dir, 'xlnet_base_squad_v1.1-dev.original.predictions.json')
    dec_pred_file = args.dec_pred_file or os.path.join(
        base_dir, 'xlnet_base_squad_v1.1-dev.s{}.predictions.json')
    dec_layers = args.dec_layers or (0, 1, 8, 9)
    out_file = args.out_file or os.path.join(
        base_dir, 'analysis-{}.json'.format(''.join(['s{}'.format(dl)
                                                     for dl in dec_layers])))

    print(dec_layers)
    gt = {}
    ctx = {}
    q = {}
    with open(gt_file) as f:
        for line in f:
            item = json.loads(line)
            qid = item['id'].split(':')[1]
            answers = [ans[1] for ans in item['label']['ans']]
            gt[qid] = answers
            ctx[qid] = item['seq2']
            q[qid] = item['seq1']

    with open(ori_pred_file) as f:
        ori_p = json.load(f)

    dec_p = {}
    for sep in dec_layers:
        with open(dec_pred_file.format(sep)) as f:
            dec_p[sep] = json.load(f)

    ana_data = {}
    for qid, ans in gt.items():
        dec_pred = {}
        for sep in dec_layers:
            if ori_p[qid] == dec_p[sep][qid]:
                continue

            # skip s8, s9 correct ones
            if is_correct(dec_p[sep][qid], ans):
                continue
            dec_pred['s{}'.format(sep)] = dec_p[sep][qid]
        if not dec_pred:
            continue
        if len(dec_pred) == 1 and 's8' in dec_pred:
            # skip s8 not correct ones
            continue

        # if len(dec_pred) == 2 and dec_pred['s8'] == dec_pred['s9']:
        #     continue
        ana_data[qid] = {'q': q[qid], 'ctx': ctx[qid], 'ans': ans,
                         'ori_pred': ori_p[qid], 'dec_pred': dec_pred}

    with open(out_file, "w") as f:
        f.write(json.dumps(ana_data, indent=2, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_file', type=str, default=None)
    parser.add_argument('--ori_pred_file', type=str, default=None)
    parser.add_argument('--dec_pred_file', type=str, default=None)
    parser.add_argument('-dl', '--dec_layers', type=str, nargs='+',
                        help='pass one or multiple layers')
    parser.add_argument('--out_file', type=str, default=None)
    main(parser.parse_args())
