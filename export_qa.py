#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

from common import logger
from common import tf
from common.config import get_config_from_args
from tasks import get_task_model_class


def main(args):
    config = get_config_from_args(args, mode='infer')
    max_seq_length = args.max_seq_length or config.max_seq_length
    config.max_seq_length = max_seq_length
    logger.info("exporting {} model...".format(config.model))
    checkpoint_path = tf.train.latest_checkpoint(config.checkpoint_dir)

    with tf.Session() as sess:

        model = get_task_model_class(config.model, config.task)(config)
        input_nodes, logits_ph = model.export_graph(config, training=False,
                                                    logits=True)

        saver = tf.train.Saver(var_list=tf.global_variables())
        logger.info('begin restoring model from checkpoints...')
        saver.restore(sess, checkpoint_path)

        inference_graph_file = config.inference_graph

        saved_model_path = os.path.join(os.path.dirname(inference_graph_file), 'saved_model')
        if not os.path.exists(saved_model_path):
            logger.info("exporting saved_model...")
            tf.saved_model.simple_save(sess, saved_model_path,
                                       inputs=input_nodes,
                                       outputs={'logits': logits_ph})

        if args.quantize:
            save_name = "{}.quant.tflite".format(model.name)
        else:
            save_name = "{}.tflite".format(model.name)
        tflite_file = os.path.join(os.path.dirname(inference_graph_file), save_name)
        if not os.path.exists(tflite_file):
            logger.info("exporting tflite model...")
            converter = tf.lite.TFLiteConverter.from_session(sess, list(input_nodes.values()),
                                                             [logits_ph])
            if args.quantize:
                converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]

            converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
            tflite_model = converter.convert()
            with open(tflite_file, "wb") as f:
                f.write(tflite_model)

        """freeze_graph --input_saved_model_dir=data/ckpt/bert/saved_model \
                      --input_binary=true --output_graph=data/ckpt/bert/frozen_bert.pb \
                      --output_node_names=bert/answer_classifier/dense/logits
        """


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str)
    parser.add_argument('-m', '--model', type=str, default='bert', choices=('bert', 'ebert'),
                        help='choose model to load default configuration')
    parser.add_argument("-msl", "--max_seq_length", type=int, default=None, help="max_seq_length")
    parser.add_argument("-q", "--quantize", action='store_true', help="quantize the tflite model")
    main(parser.parse_args())
