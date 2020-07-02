#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import OrderedDict

from common import logger
from common import tf
from models.ebert import Ebert
from models.layers.classification import LabelClassifier
from tasks.classifier_ebert import EbertClassifier


class SbertClassifier(EbertClassifier):
    def __init__(self, config, **kwargs):
        task = config.task
        kwargs['name'] = 'sbert_' + task
        super().__init__(**kwargs)
        self.num_classes = config.num_classes
        self.max_first_length = config.max_first_length + 2
        self.max_second_length = config.max_seq_length - self.max_first_length
        self.ebert_encoder = Ebert(config)

        self.num_choices = config.get('num_choices', 0)
        labels = 1 if self.num_choices else self.num_classes
        self.classifier = LabelClassifier(labels, config.hidden_dropout_prob)

        replace_map = OrderedDict(
            {
                'init_sbert/answer_classifier':
                    'sbert_{}/ebert/answer_classifier'.format(task),
                'init_sbert/embeddings':
                    'sbert_{}/ebert/embeddings'.format(task),
                'init_sbert/pooler': 'sbert_{}/ebert/pooler'.format(task),
            })
        # upper layers must be replaced first (i.e., longest match)
        layer_key = 'init_sbert/encoder/layer_{}'
        layer_val = 'sbert_{}/ebert/{}_encoder/layer_{}'
        for layer_idx in range(config.sep_layers, config.num_hidden_layers):
            replace_map[layer_key.format(layer_idx)] = layer_val.format(
                task, 'upper', layer_idx)
        for layer_idx in range(config.sep_layers):
            replace_map[layer_key.format(layer_idx)] = layer_val.format(
                task, 'lower', layer_idx)

        self.replace_map = replace_map if config.use_replace_map else {}
        self.encoded_output = None
        self.pooled_output = None
        self.embeddings = None
        self.logits = None
        self.first_embeddings = None
        self.second_embeddings = None

        if config.distill:
            sep_layers = config.sep_layers
            config.sep_layers = 0
            self.teacher = EbertClassifier(config)
            config.sep_layers = sep_layers
            replace_map['ebert_classifier/'] = ('sbert_classifier/'
                                                'ebert_classifier/')
        else:
            self.teacher = None

        self.teacher_logits = None
        self.teacher_output = None
        self.teacher_pooled_output = None

    def call(self, inputs, **kwargs):
        self.pooled_output = self.ebert_encoder(inputs, **kwargs)
        self.encoded_output = self.ebert_encoder.encoded_output
        self.first_embeddings = self.ebert_encoder.first_embeddings
        self.second_embeddings = self.ebert_encoder.second_embeddings
        self.logits = self.classifier(self.pooled_output)
        if self.num_choices:
            self.logits = tf.reshape(self.logits, [-1, self.num_choices])
        if self.teacher is not None:
            self.teacher_logits = self.teacher(inputs, **kwargs)
            self.teacher_pooled_output = self.teacher.pooled_output
            self.teacher_output = self.teacher.ebert_encoder.encoded_output

        return self.logits

    def get_loss(self, features, labels, config, training=True):
        logits = self.get_logits(features, training=training)
        label_values = labels["cls"]
        ce_loss = self.cross_entropy_loss(
            logits, label_values, self.num_classes)
        teacher_logits = self.teacher_logits
        temp = config.kd_temperature
        alpha = config.kd_alpha
        kd_kl_loss = self.kl_div_loss(logits, teacher_logits, temp) * (
            temp ** 2)

        upper_outputs = self.encoded_output
        num_upper_layers = len(upper_outputs)
        teacher_upper_outputs = self.teacher_output[-num_upper_layers:]
        mse_loss = 0
        for upper_output, teacher_output in zip(upper_outputs,
                                                teacher_upper_outputs):
            mse_loss += tf.losses.mean_squared_error(
                upper_output, teacher_output)
        mse_loss += tf.losses.mean_squared_error(
            self.pooled_output, self.teacher_pooled_output)
        mse_loss /= (num_upper_layers + 1)

        beta = config.kd_mse_beta
        gama = config.get('ce_gama', 0)
        if gama:
            logger.info("using customized combination of three losses: "
                        "ce_gama={}. kd_alpha={}, kd_mse_beta={}"
                        .format(gama, alpha, beta))
            total_loss = ce_loss * gama + kd_kl_loss * alpha + mse_loss * beta
        else:
            logger.info("using weighted average of three losses: "
                        "kd_alpha={}, kd_mse_beta={}".format(alpha, beta))
            total_loss = ce_loss * (1.0 - alpha - beta) + kd_kl_loss * alpha + (
                mse_loss * beta)
        print_tensor_dict = {
            'loss/total': total_loss,
            'loss/ce': ce_loss,
            'loss/kd_kl': kd_kl_loss,
            'loss/mse': mse_loss,
            'params/kd_alpha': alpha,
            'params/kd_mse_beta': beta,
        }
        if config.debug:
            print_tensor_dict.update({
                "feature_id": features["feature_id"],
                'logits': logits,
                'cls': label_values,
            })
        return total_loss, print_tensor_dict
