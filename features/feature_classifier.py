#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from abc import abstractmethod
from collections import namedtuple

from tqdm import tqdm

from common import logger
from .feature import DataBuilder


class ClassifierDataBuilder(DataBuilder):
    """
    ClassifierDataBuilder class abstracts the Classifier task
    input features and labels

    """
    TASK_FEATURES = ('feature_id', 'input_ids', 'segment_ids')
    DETAIL_FEATURES = ('feature_id', 'seq1', 'seq1_tokens',
                       'seq2', 'seq2_tokens', 'label',)
    TASK_LABELS = ('cls',)

    @property
    def feature(self):
        all_features = set(self.TASK_FEATURES + self.DETAIL_FEATURES
                           + self.TASK_LABELS)
        ClassifierFeatures = namedtuple('ClassifierFeatures', all_features)
        ClassifierFeatures.__new__.__defaults__ = (None,) * len(all_features)
        return ClassifierFeatures

    @abstractmethod
    def input_to_feature(self, one_input):
        raise NotImplementedError

    def example_generator(self):
        for line in tqdm(open(self.config.dataset_file)):
            for feature in self.input_to_feature(self.extract_line(line)):
                self.may_debug_feature(feature)
                yield feature

    @staticmethod
    def record_parser(record, config):
        max_seq_length = config.max_seq_length
        num_choices = config.get('num_choices', 0)
        if num_choices:
            max_seq_length *= num_choices
        from common import tf
        name_to_features = {
            "feature_id": tf.io.FixedLenFeature([], tf.int64),
            "input_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
            "segment_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        }
        if config.mode == 'train':
            name_to_features["cls"] = tf.io.FixedLenFeature([], tf.int64)

        example = DataBuilder.record_to_example(record, name_to_features)
        features = {
            'feature_id': example['feature_id'],
            'input_ids': example['input_ids'],
            'segment_ids': example['segment_ids'],
        }

        if config.mode == 'train':
            labels = {
                'cls': example['cls'],
            }
        else:
            labels = {}
        return features, labels

    @staticmethod
    def extract_line(line):
        line_item = json.loads(line)
        qid = line_item['id']
        seq1 = line_item['seq1']
        seq2 = line_item['seq2']
        label = line_item.get('label', None)
        return qid, seq1, seq2, label

    def may_debug_feature(self, feature):
        debug_min = self.config.debug_min or 0
        debug_max = self.config.debug_max or 0
        if debug_min <= self.num_examples <= debug_max:
            # debug_min and debug_max is useful for locating the question range
            logger.info('\n{}\n'.format(self.get_feature_str(
                feature, self.two_seq_str_fn, self.inputs_str_fn)))

    @staticmethod
    @abstractmethod
    def two_seq_str_fn(feat):
        pass

    @staticmethod
    def inputs_str_fn(feat):
        feature_strings = ['\tinput_ids={}'.format(feat.input_ids),
                           '\tsegment_ids={}'.format(feat.segment_ids)]
        return feature_strings

    @staticmethod
    def get_feature_str(feat, two_seq_str_fn, inputs_str_fn):
        string_list = ['feature_id', 'seq1', 'seq2', 'label', ]
        feature_str = ['{}={}'.format(s, getattr(feat, s))
                       for s in string_list]
        cls_str = getattr(feat, 'cls')
        feature_strings = ['\n\n\t'.join(feature_str),
                           '\tcls={}'.format(cls_str)]

        seq1_str, seq2_str = two_seq_str_fn(feat)
        feature_strings.append('\tseq1_info:\n\t\t{}'.format(
            '\n\t\t'.join(seq1_str)))
        feature_strings.append('\tseq2_info:\n\t\t{}'.format(
            '\n\t\t'.join(seq2_str)))
        inputs_str = inputs_str_fn(feat)
        feature_strings.extend(inputs_str)
        return '\n\n'.join(feature_strings)

    @staticmethod
    def may_process_label(label, seq_info):
        if label is None:
            return None
        return label.get('cls', None)
