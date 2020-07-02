#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .feature import DataBuilder
from .feature_bert_classifier import BertClassifierDataBuilder


class EbertClassifierDataBuilder(BertClassifierDataBuilder):
    TASK_FEATURES = ('feature_id', 'seq1_ids', 'seq2_ids')

    def __init__(self, config):
        super().__init__(config)
        self.max_first_length = config.max_first_length + 2  # for [CLS], [SEP]
        self.max_second_length = self.max_seq_length - self.max_first_length

    def build_one_input_ids(self, seq1_codes, seq2_codes):
        # TODO: check max_first_length
        seq1_ids = seq1_codes.ids[:self.max_first_length - 2]
        seq2_ids = seq2_codes.ids[:self.max_second_length - 1]
        seq1_tokens = seq1_codes.tokens[:self.max_first_length - 2]
        seq2_tokens = seq2_codes.tokens[:self.max_second_length - 1]

        first_ids = [self.cls_id] + seq1_ids + [self.sep_id]
        first_len = len(first_ids)
        input_first_ids = first_ids + [0] * (self.max_first_length - first_len)

        second_ids = seq2_ids + [self.sep_id]
        sec_len = len(second_ids)
        input_second_ids = second_ids + [0] * (self.max_second_length - sec_len)
        return input_first_ids, input_second_ids, seq1_tokens, seq2_tokens

    def set_ids(self, feature_dict, one_output):
        seq1_ids, seq2_ids, seq1_tokens, seq2_tokens = one_output
        feature_dict['seq1_ids'] = seq1_ids
        feature_dict['seq2_ids'] = seq2_ids
        feature_dict['seq1_tokens'] = seq1_tokens
        feature_dict['seq2_tokens'] = seq2_tokens
        return feature_dict

    @staticmethod
    def record_parser(record, config):
        max_first_length = config.max_first_length + 2
        max_second_length = config.max_seq_length - max_first_length
        num_choices = config.get('num_choices', 0)
        if num_choices:
            max_first_length *= num_choices
            max_second_length *= num_choices
        from common import tf
        name_to_features = {
            "feature_id": tf.io.FixedLenFeature([], tf.int64),
            "seq1_ids": tf.io.FixedLenFeature([max_first_length], tf.int64),
            "seq2_ids": tf.io.FixedLenFeature([max_second_length], tf.int64),
        }
        if config.mode == 'train':
            name_to_features["cls"] = tf.io.FixedLenFeature([], tf.int64)

        example = DataBuilder.record_to_example(record, name_to_features)
        features = {
            'feature_id': example['feature_id'],
            'seq1_ids': example['seq1_ids'],
            'seq2_ids': example['seq2_ids'],
        }

        if config.mode == 'train':
            labels = {
                'cls': example['cls'],
            }
        else:
            labels = {}
        return features, labels

    @staticmethod
    def two_seq_str_fn(feat):
        seq1_str = ['|{:>5}|{:>15}|{:>10}'.format(
            'seq1_idx', 'token', 'seq2_id')]
        seq1_str.extend(['|{:>5}|{:>15}|{:>10}'.format(
            q_idx, q_token, feat.seq1_ids[q_idx])
            for q_idx, q_token in enumerate(feat.seq1_tokens)])

        seq2_str = ['|{:>5}|{:>15}|{:>10}'.format(
            'seq2_idx', 'token', 'seq2_id')]
        seq2_str.extend(['|{:>5}|{:>15}|{:>10}'.format(
            c_idx, c_token, feat.seq2_ids[c_idx])
            for c_idx, c_token in enumerate(feat.seq2_tokens)])

        return seq1_str, seq2_str

    @staticmethod
    def inputs_str_fn(feat):
        feature_strings = ['\tseq1_ids={}'.format(feat.seq1_ids),
                           '\tseq2_ids={}'.format(feat.seq2_ids)]
        return feature_strings
