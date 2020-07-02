#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import abstractmethod

from common import tf

vocab_map = {
    'bert': 'bert.vocab',
    'ebert': 'bert.vocab',
    'xlnet': 'xlnet_large_cased.spiece.model',
    'exlnet': 'xlnet_large_cased.spiece.model',
}


class BaseModel(tf.keras.Model):

    @abstractmethod
    def get_logits(self, features, **kwargs):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_prediction(features, outputs):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def generate_predictions(logits_dict, examples, config,):
        raise NotImplementedError

    @abstractmethod
    def text_to_feature(self, inputs, config):
        raise NotImplementedError

    @abstractmethod
    def infer_graph(self, config):
        raise NotImplementedError

    def get_eval_metrics(self, features, labels, config):
        eval_fn = self.eval_fn_builder(self.get_logits, config)
        return eval_fn, [features, labels]

    @abstractmethod
    def eval_predictions(self, predictions, ground_truths):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def eval_fn_builder(get_logits_fn, config):
        raise NotImplementedError

    @abstractmethod
    def get_loss(self, features, labels, config):
        raise NotImplementedError

    @staticmethod
    def cross_entropy_loss(logits, positions, seq_length):
        one_hot_positions = tf.one_hot(positions, depth=seq_length,
                                       dtype=tf.float32)
        log_probabilities = tf.nn.log_softmax(logits, axis=-1)
        loss = -tf.reduce_mean(
            tf.reduce_sum(one_hot_positions * log_probabilities, axis=-1))
        return loss

    @staticmethod
    def kl_div_loss(student_logits, teacher_logits, temperature=1):
        """The Kullback–Leibler divergence from Q to P:
         D_kl (P||Q) = sum(P * log(P / Q))
        from student to teacher: sum(teacher * log(teacher / student))
        """
        teacher_softmax = tf.nn.softmax(teacher_logits / temperature)
        teacher_log_softmax = tf.nn.log_softmax(teacher_logits / temperature)
        student_log_softmax = tf.nn.log_softmax(student_logits / temperature)
        kl_dist = teacher_softmax * (teacher_log_softmax - student_log_softmax)
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_dist, -1))
        return kl_loss
