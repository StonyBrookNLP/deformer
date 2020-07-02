#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .feature_bert_qa import BertQaDataBuilder


class EbertQaDataBuilder(BertQaDataBuilder):
    TASK_FEATURES = ('feature_id', 'question_ids', 'context_ids')

    def __init__(self, config):
        super().__init__(config)
        self.max_first_length = config.max_first_length + 2  # for [CLS], [SEP]
        self.max_second_length = self.max_seq_length - self.max_first_length

    def get_max_ctx_tokens(self, q_len=0):
        return self.max_second_length - 1  # for [SEP]

    def get_ctx_offset(self, q_len=0):
        return self.max_first_length  # length of q is fixed

    def build_ids(self, feature_dict, q_ids, win_ctx_ids):
        # for EBERT, first put cls, then put q, sep and ctx, sep
        q_ids = q_ids[:self.max_first_length - 2]
        first_part_ids = [self.cls_id] + q_ids + [self.sep_id]
        first_len = len(first_part_ids)
        first_ids = first_part_ids + [0] * (self.max_first_length - first_len)

        second_part_ids = win_ctx_ids + [self.sep_id]
        sec_len = len(second_part_ids)
        second_ids = second_part_ids + [0] * (self.max_second_length - sec_len)

        feature_dict['question_ids'] = first_ids
        feature_dict['context_ids'] = second_ids
        return feature_dict

    @staticmethod
    def record_parser(record, config):
        max_q_length = config.max_first_length + 2
        max_c_length = config.max_seq_length - max_q_length

        from common import tf
        name_to_features = {
            "feature_id": tf.io.FixedLenFeature([], tf.int64),
            "question_ids": tf.io.FixedLenFeature([max_q_length], tf.int64),
            "context_ids": tf.io.FixedLenFeature([max_c_length], tf.int64),
        }
        if config.mode == 'train':
            name_to_features["answer_start"] = tf.io.FixedLenFeature([],
                                                                     tf.int64)
            name_to_features["answer_end"] = tf.io.FixedLenFeature([], tf.int64)
            name_to_features["cls"] = tf.io.FixedLenFeature([], tf.int64)

        example = BertQaDataBuilder.record_to_example(record, name_to_features)
        features = {
            'feature_id': example['feature_id'],
            'question_ids': example['question_ids'],
            'context_ids': example['context_ids'],
        }

        if config.mode == 'train':
            labels = {
                'cls': example['cls'],
                'answer_start': example['answer_start'],
                'answer_end': example['answer_end'],
            }
        else:
            labels = {}
        return features, labels

    @staticmethod
    def two_seq_str_fn(feat):
        q_str = ['|{:>5}|{:>15}|{:>10}'.format(
            'q_idx', 'token', 'q_id')]
        q_str.extend(['|{:>5}|{:>15}|{:>10}'.format(
            q_idx, q_token, feat.question_ids[q_idx])
            for q_idx, q_token in enumerate(feat.question_tokens)])

        ctx_str = ['|{:>5}|{:>15}|{:>15}|{:>10}'.format(
            'c_idx', 'token', 'span', 'c_id')]
        ctx_str.extend(['|{:>5}|{:>15}|{:>15}|{:>10}'.format(
            c_idx, c_token, str(feat.context_spans[c_idx]),
            feat.context_ids[c_idx])
            for c_idx, c_token in enumerate(feat.context_tokens)])

        return q_str, ctx_str

    @staticmethod
    def inputs_str_fn(feat):
        feature_strings = ['\tquestion_ids={}'.format(feat.question_ids),
                           '\tcontext_ids={}'.format(feat.context_ids)]
        return feature_strings
