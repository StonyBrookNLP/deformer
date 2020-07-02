#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from abc import ABC
from abc import abstractmethod
from datetime import datetime
from pathlib import Path

from common import logger
from common import tf


class DataBuilder(ABC):
    """
    Feature class abstracts the task+model specific input features and labels

    bert model on span qa task:

    bert model on classification task:

    bert model on regression task:

    """
    TASK_FEATURES = ()
    DETAIL_FEATURES = ()
    TASK_LABELS = ()

    def __init__(self, config):
        self.config = config
        self.preprocessor = None
        self.num_examples = 0

    @abstractmethod
    def example_generator(self):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def record_parser(record, config):
        raise NotImplementedError

    def build_examples(self):
        start_time = datetime.now()
        save_mode = self.config.save_mode
        output_file = self.config.output_file
        out_file = Path(output_file)
        tf.io.gfile.makedirs(out_file.parent.as_posix())
        out_tf_file = out_file.with_suffix('.tfrecord')
        out_jsonl_file = out_file.with_suffix('.jsonl')

        if save_mode == 'tf' or save_mode == 'both':
            tf_writer = tf.io.TFRecordWriter(out_tf_file.as_posix())
        else:
            tf_writer = None

        if save_mode == 'jsonl' or save_mode == 'both':
            jsonl_writer = open(out_jsonl_file, 'w', encoding='utf-8')
        else:
            jsonl_writer = None
        num_examples = 0
        model_features = self.TASK_FEATURES
        label_features = self.TASK_LABELS
        detail_features = self.DETAIL_FEATURES
        for raw_feature in self.example_generator():
            if tf_writer:
                feature_values = {k: getattr(raw_feature, k) for k in
                                  model_features + label_features}
                tf_example = create_tf_record(feature_values,
                                              num_examples + 1)
                tf_writer.write(tf_example.SerializeToString())
            if jsonl_writer:
                feature_values = ({k: getattr(raw_feature, k)
                                   for k in detail_features + model_features})
                jsonl_writer.write(json.dumps(feature_values,
                                              ensure_ascii=False) + '\n')
            num_examples += 1

        self.may_cleanup_writer(tf_writer, out_tf_file, num_examples,
                                'tfrecord', start_time)
        self.may_cleanup_writer(jsonl_writer, out_jsonl_file, num_examples,
                                'jsonl', start_time)

    @classmethod
    def record_to_example(cls, record, name_to_features):
        example = tf.io.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, tf.int32)  # for TPU compatibility
            example[name] = t
        return example

    @staticmethod
    def may_cleanup_writer(writer, out_file, num, suffix, start_time):
        if writer:
            writer.close()
            # append num_examples to file extension
            out_path = out_file.with_suffix('.{}.{}'.format(num, suffix))
            tf.io.gfile.rename(out_file.as_posix(),
                               out_path.as_posix(), overwrite=True)
            logger.info('\nconverted {} examples to {} in {} s\n'.format(
                num, out_path, datetime.now() - start_time))


def create_tf_record(feature_name_val_dict, num_examples):
    features_dict = dict()
    # update id to integer
    feature_name_val_dict['feature_id'] = num_examples
    for feature_name, feature_val in feature_name_val_dict.items():
        if isinstance(feature_val, float):
            feature = tf.train.Feature(
                float_list=tf.train.FloatList(value=[feature_val]))
        else:
            if isinstance(feature_val, int):
                int_list_value = [feature_val]
            elif isinstance(feature_val, list):
                int_list_value = feature_val
            else:
                raise TypeError('{}={} type={} not supported!'.format(
                    feature_name, feature_val, type(feature_val)))
            feature = tf.train.Feature(
                int64_list=tf.train.Int64List(value=int_list_value))
        features_dict[feature_name] = feature

    tf_example = tf.train.Example(
        features=tf.train.Features(feature=features_dict))
    return tf_example
