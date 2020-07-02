import collections
from abc import ABC

import numpy as np
from scipy.special import softmax

from common import tf
from common.io import get_example_and_dev_ids
from common.losses import get_cross_entropy_loss
from common.metrics import exact_match_fn
from common.metrics import metric_max_over_ground_truths
from common.metrics import qa_f1_score
from common.tf_util import flat_tensor
from models import BaseModel


class QaModel(BaseModel, ABC):

    @staticmethod
    def get_one_em_f1(predicted_text, ground_truths):
        em = metric_max_over_ground_truths(exact_match_fn, predicted_text,
                                           ground_truths)
        f1 = metric_max_over_ground_truths(qa_f1_score, predicted_text,
                                           ground_truths)
        return em, f1

    def text_to_feature(self, inputs, config):
        input_features = [f for i in inputs for f in
                          self.data_builder.input_to_feature(
                              (i['qid'], i['question'], i['context']))]
        return input_features

    def prepare_outputs(self, model_outputs, config,
                        input_features=None):
        examples = dict()
        ids = []
        for feat in input_features:
            item = feat._asdict()
            feature_id = item['feature_id']  # ex1_56be4db0acb8001400a502ec
            pred_id, orig_id = get_example_and_dev_ids(feature_id)
            item['orig_id'] = orig_id
            examples[pred_id] = item
            ids.append(pred_id)
        batch_result = {"feature_id": ids,
                        "start_logits": model_outputs[0],
                        "end_logits": model_outputs[1]}
        predictions, prediction_scores = self.generate_predictions(
            [batch_result], examples, config, return_scores=True)
        return [{'qid': qid, 'prediction': pred,
                 'score': prediction_scores[qid]}
                for qid, pred in predictions.items()]

    @staticmethod
    def get_em_f1(all_predictions, all_ground_truths):
        f1_score_cum = em_score_cum = 0
        total = len(all_predictions)
        for orig_id, predicted_text in all_predictions.items():
            ground_truths = all_ground_truths[orig_id]
            em, f1 = QaModel.get_one_em_f1(predicted_text, ground_truths)
            em_score_cum += em
            f1_score_cum += f1
        em_score_cum = 100.0 * em_score_cum / total
        f1_score_cum = 100.0 * f1_score_cum / total
        return em_score_cum, f1_score_cum

    @staticmethod
    def get_prediction(features, outputs):
        predictions = {
            "feature_id": features["feature_id"],
            "start_logits": outputs[0],
            "end_logits": outputs[1],
        }
        return predictions

    @staticmethod
    def get_context_start(example_item):
        return 0

    def generate_predictions(self, logits_dict, examples, config,
                             return_scores=False):
        final_cls = collections.OrderedDict()
        final_cls_prob = collections.OrderedDict()
        final_predictions = collections.OrderedDict()
        final_span_scores = collections.OrderedDict()
        max_answer_span = config.max_answer_span

        for batch_result in logits_dict:
            ids = batch_result["feature_id"]
            start_logits = batch_result["start_logits"]
            end_logits = batch_result["end_logits"]
            cls_logits = batch_result.get("cls_logits", None)
            for i, qid in enumerate(ids):
                s_logits = [float(x) for x in flat_tensor(start_logits[i])]
                e_logits = [float(x) for x in flat_tensor(end_logits[i])]

                item = examples[int(qid)]
                orig_id = item['orig_id']
                # or len(item['context_tokens'])
                context_len = len(item['context_spans'])
                context_start = self.get_context_start(item)
                context_end = context_start + context_len
                # only consider valid context logits
                x_s = softmax(s_logits[context_start:context_end])
                y_s = softmax(e_logits[context_start:context_end])
                z = np.outer(x_s, y_s)
                zn = np.tril(np.triu(z), max_answer_span)
                pred_start, pred_end = np.unravel_index(np.argmax(zn), zn.shape)
                pred_score = zn[pred_start, pred_end]

                if pred_score > final_span_scores.get(orig_id, 0):
                    start_span = item['context_spans'][pred_start]
                    predicted_char_start = start_span[0]
                    end_span = item['context_spans'][pred_end]
                    predicted_char_end = end_span[1]
                    predicted_text = item['context'][
                                     predicted_char_start:predicted_char_end]
                    final_predictions[orig_id] = predicted_text
                    final_span_scores[orig_id] = pred_score

                if cls_logits is not None:
                    c_logits = [float(x) for x in cls_logits[i].flat]
                    cls_prob = np.exp(c_logits)
                else:
                    cls_prob = np.zeros(1)  # placeholder
                cls_idx = np.argmax(cls_prob)
                pred_cls_prob = cls_prob[cls_idx]
                if pred_cls_prob >= final_cls_prob.get(orig_id, 0):
                    final_cls_prob[orig_id] = pred_cls_prob
                    final_cls[orig_id] = cls_idx

                if final_cls[orig_id] == 1:
                    # yes for hotpot, impossible for squad 2.0
                    if config.task == 'squad_v2.0':
                        final_predictions[orig_id] = ''
                    elif config.task == 'hotpot':
                        final_predictions[orig_id] = 'yes'
                elif final_cls[orig_id] == 2:  # cls == 2
                    # no
                    final_predictions[orig_id] = 'no'

        if return_scores:
            return final_predictions, final_span_scores
        else:
            return final_predictions

    def eval_predictions(self, predictions, ground_truths):
        ans_em_score, ans_f1_score = self.get_em_f1(predictions,
                                                    ground_truths)
        return {'em': ans_em_score, 'f1': ans_f1_score, 'metric': ans_f1_score}

    @staticmethod
    def eval_fn_builder(get_logits_fn, config):
        def eval_fn(features, labels):
            start_logits, end_logits = get_logits_fn(features, training=False)
            outer = tf.matmul(
                tf.expand_dims(tf.nn.softmax(start_logits), axis=2),
                tf.expand_dims(tf.nn.softmax(end_logits), axis=1))
            outer = tf.matrix_band_part(outer, 0, config.max_answer_span)
            predicted_start = tf.argmax(tf.reduce_max(outer, axis=2), axis=1,
                                        output_type=tf.int32)
            predicted_end = tf.argmax(tf.reduce_max(outer, axis=1), axis=1,
                                      output_type=tf.int32)
            answer_start = labels["answer_start"]
            answer_end = labels["answer_end"]
            answer_start = tf.expand_dims(answer_start, -1)
            answer_end = tf.expand_dims(answer_end, -1)
            start_correct = tf.reduce_any(
                tf.equal(predicted_start, answer_start), 1)
            end_correct = tf.reduce_any(tf.equal(predicted_end, answer_end), 1)
            correct = start_correct & end_correct
            eval_metric_ops = {
                'accuracy/start': tf.metrics.accuracy(answer_start,
                                                      predicted_start),
                'accuracy/end': tf.metrics.accuracy(answer_end, predicted_end),
                'accuracy/all': tf.metrics.mean(tf.cast(correct, 'float'))
            }
            return eval_metric_ops

        return eval_fn

    def get_loss(self, features, labels, config):
        training = config.mode == 'train'
        start_logits, end_logits = self.get_logits(features, training=training)
        answer_start = labels["answer_start"]
        answer_end = labels["answer_end"]
        start_loss = get_cross_entropy_loss(start_logits, answer_start,
                                            config.max_seq_length)
        end_loss = get_cross_entropy_loss(end_logits, answer_end,
                                          config.max_seq_length)
        total_loss = tf.reduce_mean(start_loss + end_loss)
        print_tensor_dict = {
            'total_loss': total_loss,
        }
        if config.debug:
            print_tensor_dict.update({
                "feature_id": features["feature_id"],
                'start_loss': start_loss,
                'end_loss': end_loss,
                'start_logit': start_logits,
                'end_logit': end_logits,
                'start_pos': answer_start,
                'end_pos': answer_end,
            })
        return total_loss, print_tensor_dict
