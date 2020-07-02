#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import shutil
from datetime import datetime

import ujson as json

from common import logger
from common import tf
from common.builder import get_estimator
from common.builder import input_fn_builder
from common.config import get_config_from_args
from common.io import load_examples
from features import get_record_parser
from tasks import get_task_model_class


def evaluate(config):
    start = datetime.now()
    tf.set_random_seed(config.random_seed or 1)
    output_file = config.output_file
    output_dir = os.path.dirname(output_file)
    tf.io.gfile.makedirs(output_dir)

    logger.info('config: \n{}'.format(
        '\n'.join(['{}: {}'.format(i[0], i[1])
                   for i in sorted(config.items())])))
    model_class = get_task_model_class(config.model, config.task)
    estimator, model = get_estimator(config, model_class)
    record_parser = get_record_parser(config.model, config.task)
    predict_input_fn = input_fn_builder(record_parser, config)

    output_file = config.output_file
    output_dir = os.path.dirname(output_file)
    tf.io.gfile.makedirs(output_dir)

    example_file = config.ground_truth_file
    logger.info('loading examples from {}....'.format(example_file))
    eval_examples, eval_ground_truths = load_examples(example_file)
    best_metric_result, checkpoints_iterator = get_checkpoint_iterator(config)
    total_batches = int(config.dataset_size / config.dev_batch_size)
    metric_results = None
    for checkpoint_path in checkpoints_iterator:
        logger.info('begin evaluating {}...'.format(checkpoint_path))
        model_name = ''
        if checkpoint_path:
            model_name = os.path.split(checkpoint_path)[1]
        logits = eval_checkpoint(estimator, predict_input_fn,
                                 checkpoint_path, total_batches)
        final_predictions = model.generate_predictions(logits, eval_examples,
                                                       config)
        with tf.io.gfile.GFile(output_file, "w") as f:
            f.write(json.dumps(final_predictions, indent=2) + "\n")
        metric_results = model.eval_predictions(final_predictions,
                                                eval_ground_truths)
        metric_result = metric_results['metric']
        metric_str = ['{}={}'.format(k, v) for k, v in metric_results.items()]
        if metric_result > best_metric_result:
            best_metric_result = metric_result
            logger.info("best {}, {}".format(model_name, ', '.join(metric_str)))
            save_best_model(checkpoint_path)
        else:
            logger.info("{}, {}".format(model_name, ', '.join(metric_str)))

    end = datetime.now()
    logger.info('evaluation done, took {} s!'.format(end - start))
    logger.info('final_predictions saved to: {}'.format(output_file))
    return metric_results


def get_checkpoint_iterator(config):
    if config.iterate_checkpoints:
        checkpoints_iterator = tf.train.checkpoints_iterator(
            config.checkpoint_dir, timeout=config.iterate_timeout)
        best_metric_result = 0
    else:
        best_metric_result = float('inf')
        # evaluate once, ignore best comparison
        if config.checkpoint_path:
            checkpoint_path = config.checkpoint_path
        else:
            checkpoint_path = tf.train.latest_checkpoint(config.checkpoint_dir)
        checkpoints_iterator = [checkpoint_path]
    return best_metric_result, checkpoints_iterator


def save_best_model(model_checkpoint):
    old_path, name = os.path.split(model_checkpoint)
    best_dir = os.path.join(old_path, 'best')
    if os.path.exists(best_dir):
        shutil.rmtree(best_dir)
    os.makedirs(best_dir)
    for f in glob.glob("%s*" % model_checkpoint):
        file_name = os.path.split(f)[1]
        new_path = os.path.join(best_dir, file_name)
        shutil.copyfile(f, new_path)
    checkpoint_file = os.path.join(best_dir, 'checkpoint')
    with open(checkpoint_file, 'w') as f:
        f.write('model_checkpoint_path: "%s"\n' % name)


def eval_checkpoint(estimator, predict_input_fn,
                    checkpoint_path, total_batches):
    all_predictions = []
    total = 0
    model_name = ''
    if checkpoint_path:
        model_name = os.path.split(checkpoint_path)[1]
    for batch_result in estimator.predict(predict_input_fn,
                                          checkpoint_path=checkpoint_path,
                                          yield_single_examples=False):
        all_predictions.append(batch_result)
        total += 1
        if total % 10 == 0:
            logger.info("{}, predicted {}/({}) batches".format(
                model_name, total, total_batches))

    return all_predictions


def main(args):
    config = get_config_from_args(args, mode='dev')
    config.iterate_checkpoints = args.iterate_checkpoints
    config.checkpoint_path = args.checkpoint_path
    config.iterate_timeout = args.iterate_timeout
    evaluate(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str)
    parser.add_argument('-cp', '--checkpoint_path', type=str, default=None)
    parser.add_argument("-i", "--iterate_checkpoints",
                        action='store_true', help="iterate_checkpoints")
    parser.add_argument("-it", "--iterate_timeout", type=int,
                        default=3600, help="checkpoint iterator timeout")
    parser.add_argument('-m', '--model', type=str, default='bert',
                        choices=('bert', 'ebert', 'sbert'),
                        help='choose model to load default configuration')
    parser.add_argument('-t', '--task', type=str, default='squad_v1.1',
                        choices=('squad_v1.1', 'mnli', 'qqp', 'boolq', 'race'),
                        help='choose model to load default configuration')
    main(parser.parse_args())
