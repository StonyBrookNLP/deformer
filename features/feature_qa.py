#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import abstractmethod
from collections import namedtuple

from .feature_classifier import ClassifierDataBuilder


class QaDataBuilder(ClassifierDataBuilder):
    """
    QaDataBuilder class abstracts the QA task input features and labels

    bert model on span qa task:
    ebert model on span qa task:
    xlnet model on span qa task:
    """
    TASK_FEATURES = ('feature_id', 'input_ids', 'segment_ids')
    DETAIL_FEATURES = ('feature_id', 'question', 'question_tokens',
                       'context', 'context_tokens', 'context_spans',
                       'label',)
    TASK_LABELS = ('cls', 'answer_start', 'answer_end')

    @property
    def feature(self):
        all_features = set(self.TASK_FEATURES + self.DETAIL_FEATURES
                           + self.TASK_LABELS)
        QaFeatures = namedtuple('QaFeature', all_features)
        QaFeatures.__new__.__defaults__ = (None,) * len(all_features)
        return QaFeatures

    @staticmethod
    def record_parser(record, config):
        max_seq_length = config.max_seq_length
        from common import tf
        name_to_features = {
            "feature_id": tf.io.FixedLenFeature([], tf.int64),
            "input_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
            "segment_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        }
        if config.mode == 'train':
            name_to_features["answer_start"] = tf.io.FixedLenFeature([],
                                                                     tf.int64)
            name_to_features["answer_end"] = tf.io.FixedLenFeature([], tf.int64)
            name_to_features["cls"] = tf.io.FixedLenFeature([], tf.int64)

        example = ClassifierDataBuilder.record_to_example(record,
                                                          name_to_features)
        features = {
            'feature_id': example['feature_id'],
            'input_ids': example['input_ids'],
            'segment_ids': example['segment_ids'],
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
    @abstractmethod
    def two_seq_str_fn(feat):
        pass

    @staticmethod
    def get_feature_str(feat, q_ctx_str_fn, inputs_str_fn):
        string_list = ['feature_id', 'question', 'context', 'label', ]
        feature_str = ['{}={}'.format(s, getattr(feat, s))
                       for s in string_list]
        feature_strings = ['\n\n\t'.join(feature_str),
                           '\tanswer_start={}, answer_end={}'.format(
                               feat.answer_start, feat.answer_end)]

        q_str, ctx_str = q_ctx_str_fn(feat)
        feature_strings.append('\tq_info:\n\t\t{}'.format(
            '\n\t\t'.join(q_str)))
        feature_strings.append('\tctx_info:\n\t\t{}'.format(
            '\n\t\t'.join(ctx_str)))
        inputs_str = inputs_str_fn(feat)
        feature_strings.extend(inputs_str)
        return '\n\n'.join(feature_strings)

    @staticmethod
    def may_process_label(label, ctx_info):
        answer_cls, ans_start, ans_end = None, None, None
        if not label:
            return answer_cls, ans_start, ans_end

        # only look at answer for training
        answer_cls = label.get('cls', None)
        answer = label.get('ans', None)
        if not answer:
            return answer_cls, -1, -1

        # only use first answer
        ans_char_start = answer[0][0]
        answer_text = answer[0][1]
        ans_char_end = ans_char_start + len(answer_text)

        ans_char_span = (ans_char_start, ans_char_end)
        ans_start, ans_end = QaDataBuilder.get_answer_token_span(
            ctx_info, ans_char_span, answer_text)
        return answer_cls, ans_start, ans_end

    @staticmethod
    def adjust_label(feature_dict, offset, win_span):
        ans_cls = feature_dict['cls']
        ans_start = feature_dict['answer_start']
        ans_end = feature_dict['answer_end']
        win_start, win_end = win_span
        if ans_cls == 0:  # has answer info
            # adjust ans token offset
            if win_start <= ans_start <= ans_end <= win_end:
                ans_start = ans_start - win_start + offset
                ans_end = ans_end - win_start + offset
                return ans_cls, ans_start, ans_end
            else:
                return None, None, None
        else:  # ans_cls = 1, 2 or None
            return ans_cls, ans_start, ans_end

    @staticmethod
    def get_answer_token_span(ctx_info, answer_char_span, answer=None):
        context, ctx_codes = ctx_info
        answer_char_start, answer_char_end = answer_char_span
        answer_token_start, answer_token_end = -1, -1
        token_spans = ctx_codes.offsets
        tokens = ctx_codes.tokens
        for token_idx, token_span in enumerate(token_spans):
            token_char_start, token_char_end = token_span
            if token_char_start <= answer_char_start <= token_char_end:
                answer_token_start = token_idx
            if token_char_start <= answer_char_end <= token_char_end:
                answer_token_end = token_idx
            if answer_token_start >= 0 and answer_token_end >= 0:
                return answer_token_start, answer_token_end

        token_str = ['idx={}, token={}, span={}'.format(i, t, s)
                     for i, (t, s) in enumerate(zip(tokens, token_spans))]
        debug_str = [
            'answer: {}'.format(answer),
            'no answer_char_span: {},'.format(answer_char_span),
            'tokens: {},'.format('\n\t'.join(token_str)),
            'context={}'.format(context)]
        raise ValueError('\n'.join(debug_str))
