#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import OrderedDict

from common import tf, logger
from tasks.qa_ebert import EbertQa
from models.ebert import Ebert
from models.layers.classification import AnswerClassifier


class SbertQA(EbertQa):

    def __init__(self, config, **kwargs):
        task = config.task
        kwargs['name'] = 'sbert_' + task
        super().__init__(**kwargs)
        self.max_seq_length = config.max_seq_length
        self.max_first_length = config.max_first_length + 2
        self.max_c_length = config.max_seq_length - self.max_first_length
        self.ebert_encoding_layer = Ebert(config)
        self.answer_layer = AnswerClassifier(2)
        replace_map = OrderedDict(
            {'init_sbert/answer_classifier'
             : 'sbert_{}/ebert/answer_classifier'.format(task),
             'init_sbert/embeddings': 'sbert_{}/ebert/embeddings'.format(task)
             })
        # upper layers must be replaced first (i.e., longest match)
        layer_key = 'init_sbert/encoder/layer_{}'
        layer_val = 'sbert_{}/ebert/{}_encoder/layer_{}'
        for layer_idx in range(config.sep_layers, config.num_hidden_layers):
            k = layer_key.format(layer_idx)
            replace_map[k] = layer_val.format(task, 'upper', layer_idx)
        for layer_idx in range(config.sep_layers):
            k = layer_key.format(layer_idx)
            replace_map[k] = layer_val.format(task, 'lower', layer_idx)

        if config.distill:
            sep_layers = config.sep_layers
            config.sep_layers = 0
            self.teacher = EbertQa(config)
            config.sep_layers = sep_layers
            replace_map['ebert_qa/'] = 'sbert_qa/ebert_qa/'
        else:
            self.teacher = None

        self.replace_map = replace_map if config.use_replace_map else {}
        self.encoded_output = None
        self.q_embeddings = None
        self.c_embeddings = None
        self.logits = None
        self.teacher_logits = None
        self.teacher_encoded_output = None

    def call(self, question_ids, context_ids=None,
             q_embeddings=None, c_embeddings=None, **kwargs):

        encoded_output = self.ebert_encoding_layer(question_ids, context_ids,
                                                   q_embeddings, c_embeddings,
                                                   **kwargs)
        self.logits = self.answer_layer(encoded_output[-1])
        if kwargs.get('logits', False):
            return self.logits

        unstacked_logits = tf.unstack(self.logits, axis=0)
        start_logits, end_logits = unstacked_logits[0], unstacked_logits[1]

        if self.teacher is not None:
            self.teacher_logits = self.teacher(question_ids, context_ids,
                                               q_embeddings, c_embeddings,
                                               **kwargs)
            self.teacher_encoded_output = self.teacher.encoded_output
        self.encoded_output = self.ebert_encoding_layer.encoded_output
        return start_logits, end_logits

    def get_loss(self, features, labels, config, training=True):
        start_logits, end_logits = self.get_logits(features, training=training)
        answer_start = labels["answer_start"]
        answer_end = labels["answer_end"]
        max_seq_length = config.max_seq_length
        start_loss = self.cross_entropy_loss(start_logits, answer_start,
                                             max_seq_length)
        end_loss = self.cross_entropy_loss(end_logits, answer_end,
                                           max_seq_length)
        ce_loss = (start_loss + end_loss) / 2

        teacher_start_logits, teacher_end_logits = self.teacher_logits
        temp = config.kd_temperature
        alpha = config.kd_alpha
        kd_start_kl_loss = self.kl_div_loss(start_logits, teacher_start_logits,
                                            temp) * (temp ** 2)
        kd_end_kl_loss = self.kl_div_loss(end_logits, teacher_end_logits,
                                          temp) * (temp ** 2)
        kd_kl_loss = (kd_start_kl_loss + kd_end_kl_loss) / 2

        upper_outputs = self.encoded_output
        num_upper_layers = len(upper_outputs)
        logger.info("upper_outputs: {}".format(num_upper_layers))
        logger.info("teacher_upper_outputs before: {}".format(
            len(self.teacher_encoded_output)))
        teacher_upper_outputs = self.teacher_encoded_output[-num_upper_layers:]
        logger.info("teacher_upper_outputs: {}".format(
            len(teacher_upper_outputs)))
        mse_loss = 0
        for upper_output, teacher_upper_output in zip(upper_outputs,
                                                      teacher_upper_outputs):
            mse_loss += tf.losses.mean_squared_error(
                upper_output, teacher_upper_output)
        mse_loss /= num_upper_layers
        beta = config.kd_mse_beta
        gama = config.get('ce_gama', 0)
        if gama:
            logger.info("using customized combination of three losses: "
                        "ce_gama={}. kd_alpha={}, kd_mse_beta={}"
                        "".format(gama, alpha, beta))
            total_loss = ce_loss * gama + kd_kl_loss * alpha + mse_loss * beta
        else:
            logger.info("using weighted average of three losses: "
                        "kd_alpha={}, kd_mse_beta={}".format(alpha, beta))
            total_loss = ce_loss * (
                1.0 - alpha - beta) + kd_kl_loss * alpha + mse_loss * beta
        print_tensor_dict = {
            'loss/total': total_loss,
            'loss/ce': ce_loss,
            'loss/kd_kl': kd_kl_loss,
            'loss/mse': mse_loss,
            'params/kd_alpha': alpha,
            'params/kd_mse_beta': beta,
            'params/ce_gama': gama,
        }
        if config.debug:
            print_tensor_dict.update({
                "feature_id": features["feature_id"],
                'start_logit': start_logits,
                'end_logit': end_logits,
                'start_pos': answer_start,
                'end_pos': answer_end,
            })
        return total_loss, print_tensor_dict
