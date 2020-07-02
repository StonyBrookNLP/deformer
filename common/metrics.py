#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import string
from collections import Counter
# from scipy.stats import pearsonr, spearmanr
# from sklearn.metrics import matthews_corrcoef, f1_score
import numpy as np

from . import tf


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


def qa_f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_fn(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def regex_match_fn(prediction, pattern):
    """Check if the prediction matches the given regular expression."""
    try:
        compiled = re.compile(pattern,
                              flags=re.IGNORECASE + re.UNICODE + re.MULTILINE)
    except BaseException as e:
        print('Regular expression failed to compile: %s' % pattern)
        print(e)
        return False
    return compiled.match(prediction) is not None


def get_context_start(config, item):
    model_name = config.model
    if model_name == 'ebert':
        max_q_len = config.max_first_length
        return max_q_len + 2
        # return len(item['question_tokens']) + 2
    elif model_name == 'bert':
        return len(item['question_tokens']) + 2
    else:
        return 0
