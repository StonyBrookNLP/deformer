#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import collections
import os
from configparser import ExtendedInterpolation
from functools import lru_cache

import ujson as json
from localconfig import LocalConfig

from common import tf
from . import RES_DIR


class EnvInterpolation(ExtendedInterpolation):
    """Interpolation which expands environment variables in values."""

    def before_get(self, parser, section, option, value, defaults):
        value = os.path.expandvars(value)
        return super().before_get(parser, section, option, value, defaults)


def read_config(config_file_or_string):
    config = LocalConfig(interpolation=EnvInterpolation())
    config.read(config_file_or_string)
    return config


def read_data(text_file):
    with open(text_file, encoding='utf-8') as f:
        content = f.read()
    return content


BERT_VOCAB_FILE = os.path.join(RES_DIR, 'bert.vocab')


@lru_cache(maxsize=16)
def load_vocab(vocab_file=BERT_VOCAB_FILE):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding='utf-8') as f:
        for line in f:
            token = line.strip()
            vocab[token] = index
            index += 1
    return vocab


SP_MODEL_FILE = os.path.join(RES_DIR, 'spiece.model')


def load_sp_model(model_file=SP_MODEL_FILE):
    import sentencepiece as spm
    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(model_file)
    return sp_model


def get_example_and_dev_ids(feature_id):
    # e.g. 6 for 1000_56be4db0acb8001400a502ec
    separator_idx = feature_id.find('_')
    para_sep_idx = feature_id.find(':')  # for squad compatibility
    if para_sep_idx == -1:
        para_sep_idx = separator_idx
    return int(feature_id[:separator_idx]), feature_id[para_sep_idx + 1:]


def load_examples(input_file):
    all_examples = dict()  # id to example map
    if not tf.io.gfile.exists(input_file):
        raise ValueError('{} does not exist!'.format(input_file))

    all_ground_truths = collections.OrderedDict()
    with tf.io.gfile.GFile(input_file, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            feature_id = item['feature_id']  # 1_56be4db0acb8001400a502ec
            pred_id, orig_id = get_example_and_dev_ids(feature_id)
            item['orig_id'] = orig_id
            all_examples[pred_id] = item

            if 'label' in item:
                answers = item['label']['ans']
                answer_cls = item['label']['cls']
                if answers:
                    answer_texts = [a[1] for a in answers]
                    all_ground_truths[orig_id] = answer_texts
                else:
                    all_ground_truths[orig_id] = answer_cls
                    # # for hotpot qa
                    # answer = 'yes' if answer_cls == 1 else 'no'
                    # all_ground_truths[orig_id] = [answer]
    return all_examples, all_ground_truths
