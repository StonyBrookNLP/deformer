#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np

from common import logger
from common import tf
from common.builder import input_fn_builder
from common.config import get_config_from_args
from features import get_record_parser
from tasks import get_task_model_class

np.set_printoptions(threshold=np.inf, suppress=True)


def main(args):
    config = get_config_from_args(args, mode='dev')
    config.batch_size = args.batch_size
    logger.info('config: \n{}'.format('\n'.join(['{}: {}'.format(i[0], i[1]) for i in sorted(config.items())])))

    record_parser = get_record_parser(config.model, config.task)
    predict_input_fn = input_fn_builder(record_parser, config)(config)
    iterator = predict_input_fn.make_initializable_iterator()
    logger.info("running in batch mode...")
    checkpoint_path = tf.train.latest_checkpoint(config.checkpoint_dir)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    with tf.Session() as sess:
        features, labels = iterator.get_next()

        model = get_task_model_class(config.model, config.task)(config)
        feed_fn, output_tensors = model.infer_graph(config)

        saver = tf.train.Saver(var_list=tf.global_variables())
        logger.info("restoring model weights from: {}...".format(checkpoint_path))
        saver.restore(sess, checkpoint_path)
        batches = 1
        sess.run(iterator.initializer)
        while True:
            try:
                feature_values = sess.run(features)
                # logger.info('feature_values={}...'.format(feature_values))
                feed = feed_fn(feature_values)

                attentions, encoded_output = sess.run([model.attentions, model.encoded_output], feed_dict=feed)
                layers = len(encoded_output)
                for layer in range(layers):
                    feature_values['layer_{}'.format(layer)] = encoded_output[layer]
                    feature_values['attn_{}'.format(layer)] = attentions[layer]

                if batches < 3:
                    logger.info('num layers={}'.format(layers))
                    logger.info('\n'.join(['{}={}'.format(k, v.shape) for k, v in feature_values.items()]))

                output_path = os.path.join(out_dir, '{}_b{}'.format(args.model, batches))
                logger.info('saving outputs for b={}...'.format(batches))
                np.savez_compressed(output_path, **feature_values)
                logger.info('outputs saved to: {}'.format(output_path))

                batches += 1
            except tf.errors.OutOfRangeError:
                logger.info('all done')
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", "--batch_size", type=int, default=25, help="batch_size")
    parser.add_argument('-c', '--config_file', type=str)
    parser.add_argument('-o', '--out_dir', type=str, default=None)
    parser.add_argument('-m', '--model', type=str, default='bert', choices=('bert', 'ebert', 'sbert'),
                        help='choose model to load default configuration')
    parser.add_argument('-t', '--task', type=str, default='squad_v1.1',
                        choices=('squad_v1.1', 'mnli', 'qqp', 'boolq', 'race'),
                        help='choose task to run')
    main(parser.parse_args())
