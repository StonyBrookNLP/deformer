import collections
from abc import ABC

import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from common import tf
from models import BaseModel


class ClassifierModel(BaseModel, ABC):

    @staticmethod
    def get_prediction(features, outputs):
        predictions = {
            "feature_id": features["feature_id"],
            "logits": outputs,
        }
        return predictions

    def text_to_feature(self, inputs, config):
        input_features = [f for i in inputs for f in
                          self.data_builder.input_to_feature(
                              (i['id'], i['seq1'], i['seq2']))]
        return input_features

    def eval_predictions(self, all_predictions, all_ground_truths):
        correct = 0
        total = len(all_predictions)
        labels = []
        predicted_labels = []
        for orig_id, predicted_label in all_predictions.items():
            ground_truth = all_ground_truths[orig_id]
            labels.append(ground_truth)
            predicted_labels.append(predicted_label)
            if predicted_label == ground_truth:
                correct += 1
        accuracy = correct * 100 / total
        eval_result = {'accuracy': accuracy, 'metric': accuracy}
        if len(set(labels)) == 2:
            f1 = f1_score(y_true=labels, y_pred=predicted_labels)
            eval_result['f1'] = f1
            eval_result['metric'] = f1
        return eval_result

    @staticmethod
    def generate_predictions(logits_dict, examples, config,
                             return_scores=False):
        final_cls_prob = collections.OrderedDict()
        final_predictions = collections.OrderedDict()
        for batch_result in logits_dict:
            batch_ids = batch_result["feature_id"]
            # tf.logging.info("batch_ids={}".format(batch_ids))
            batch_logits = batch_result["logits"]
            for pred_id, pred_logits in zip(batch_ids, batch_logits):
                logits = [float(x) for x in pred_logits.flat]
                scores = softmax(logits)
                item = examples[int(pred_id)]
                orig_id = item['orig_id']
                predicted_label = int(np.argmax(scores))
                final_predictions[orig_id] = predicted_label
                final_cls_prob[orig_id] = scores[predicted_label]

        if return_scores:
            return final_predictions, final_cls_prob
        else:
            return final_predictions

    @staticmethod
    def eval_fn_builder(get_logits_fn, config):
        def eval_fn(features, labels):
            logits = get_logits_fn(features, training=False)
            predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)

            label_values = labels["cls"]
            accuracy = tf.metrics.accuracy(labels=label_values,
                                           predictions=predictions)

            log_probs = tf.nn.log_softmax(logits, axis=-1)
            one_hot_labels = tf.one_hot(label_values, depth=config.num_classes,
                                        dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs,
                                              axis=-1)

            loss = tf.metrics.mean(values=per_example_loss)

            return {
                "eval_accuracy": accuracy,
                "eval_loss": loss,
            }

        return eval_fn

    def get_loss(self, features, labels, config, training=True):
        logits = self.get_logits(features, training=training)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        label_values = labels["cls"]
        one_hot_labels = tf.one_hot(label_values, depth=self.num_classes,
                                    dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        total_loss = tf.reduce_mean(per_example_loss)

        print_tensor_dict = {
            'total_loss': total_loss,
            # 'accuracy': accuracy,
        }
        if config.debug:
            print_tensor_dict.update({
                "feature_id": features["feature_id"],
                'logits': logits,
                'cls': label_values,
                'log_probs': log_probs,
                'one_hot_labels': one_hot_labels,
            })
        return total_loss, print_tensor_dict
