#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time

from common import logger
from common import tf
from common.config import get_config_from_args
from common.util import batch
from tasks import get_task_model_class


def main(args):
    config = get_config_from_args(args, mode='infer')

    max_seq_length = args.max_seq_length or config.max_seq_length
    config.max_seq_length = max_seq_length
    contexts = [
        "The American Football Conference (AFC) champion Denver Broncos "
        "defeated the National Football Conference (NFC) champion Carolina "
        "Panthers 24–10 to earn their third Super Bowl title.",

        "The game was played on February 7, 2016, at Levi's Stadium in the "
        "San Francisco Bay Area at Santa Clara, California.",

        "College sports are also popular in southern California. "
        "The UCLA Bruins and the USC Trojans both field teams in NCAA Division"
        " I in the Pac-12 Conference, and there is a longtime "
        "rivalry between the schools.",
    ]

    questions = ["What is the AFC short for?",
                 "What day was the game played on?",
                 "What other kind of sport is popular in southern California?",
                 ]
    max_answer_span = args.max_answer_span or config.max_answer_span
    config.max_answer_span = max_answer_span
    text_inputs = [{'qid': qid, 'question': q, 'context': ctx} for qid, (q, ctx)
                   in enumerate(zip(questions, contexts))]
    outputs = []
    if args.eager:

        logger.info("running in eager mode...")
        tf.enable_eager_execution()
        checkpoint_path = tf.train.latest_checkpoint(config.checkpoint_dir)

        logger.info("restoring weights from: {}...".format(checkpoint_path))
        # with tf.contrib.eager.restore_variables_on_create(None):
        with tf.contrib.eager.restore_variables_on_create(checkpoint_path):

            model = get_task_model_class(config.model, config.task)(config)
            logger.info("warming up model...")
            model.warm_up()

        # trainable_count = int(numpy.sum([tf.keras.backend.count_params(p)
        # for p in set(model.trainable_weights)]))
        # non_trainable_count = int(numpy.sum([tf.keras.backend.count_params(p)
        # for p in set(model.non_trainable_weights)]))
        # print('trainable_count', abbreviate(trainable_count))
        # print('non_trainable_count', abbreviate(non_trainable_count))
        # # #### testing TF 2.0 ####
        # logger.info("restoring model weights...")
        # model = get_model(config)
        # checkpoint = tf.train.Checkpoint(model=model)
        # checkpoint.restore(os.path.join(config.checkpoint_dir, 'ckpt-1'))
        # with tf.contrib.eager.restore_variables_on_create(checkpoint_path):
        #
        #     model = get_model(config)
        #     logger.info("warming up model...")
        #     model.warm_up(config)
        # checkpoint = tf.train.Checkpoint(model=model)
        # manager = tf.train.CheckpointManager(checkpoint,
        # os.path.join(config.checkpoint_dir, 'keras1.14'),  max_to_keep=1)
        # manager.save()

        text_features = model.text_to_feature(text_inputs, config)
        # inputs_tensor = [tf.convert_to_tensor(i) for i in inputs]
        logger.info("begin inferring...")
        start_time = time.time()
        model_outputs = model.infer(text_features)
        output = model.prepare_outputs(model_outputs, config,
                                       text_features)
        logger.info('output={}\n\n'.format(output))
        outputs.extend(output)
    else:
        logger.info("running in graph mode...")
        checkpoint_path = tf.train.latest_checkpoint(config.checkpoint_dir)
        with tf.Session() as sess:

            model = get_task_model_class(config.model, config.task)(config)
            feed_fn, output_tensors = model.infer_graph(config)

            # inference_graph_file = config.inference_graph
            # if not os.path.exists(inference_graph_file):
            #     logger.info("generating inference graph...")
            #     graph_def = sess.graph_def
            #     with tf.gfile.GFile(inference_graph_file, 'wb') as f:
            #         f.write(graph_def.SerializeToString())
            #     with tf.io.gfile.GFile(inference_graph_file + '.txt', 'w') as f:
            #         f.write(str(graph_def))
            #     logger.info("inference graph saved to: {}".format(
            #         inference_graph_file))

            saver = tf.train.Saver(var_list=tf.global_variables())
            logger.info('begin restoring model from checkpoints...')
            saver.restore(sess, checkpoint_path)

            logger.info('begin predicting...')
            text_features = model.text_to_feature(text_inputs, config)
            start_time = time.time()
            for text_features in batch(text_features, args.batch_size):
                feed = feed_fn(text_features)
                model_outputs = sess.run(output_tensors, feed)
                output = model.prepare_outputs(model_outputs, config,
                                               text_features)
                logger.info('output={}\n\n'.format(output))
                outputs.extend(output)
    end_time = time.time()
    logger.info('infer time: {:.4f} s'.format(end_time - start_time))
    for q, c, a in zip(questions, contexts, outputs):
        logger.info('q={}\na={}\n\tcontext={}\n\n'.format(q, a, c))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str)
    parser.add_argument('-m', '--model', type=str, default='bert',
                        choices=('bert', 'ebert'),
                        help='choose model to load default configuration')
    parser.add_argument("-bd", "--batch_size", type=int, default=2,
                        help="batch_size")
    parser.add_argument("-mas", "--max_answer_span", type=int, default=30,
                        help="max_answer_span")
    parser.add_argument("-msl", "--max_seq_length", type=int, default=None,
                        help="max_seq_length")
    parser.add_argument("-e", "--eager", action='store_true',
                        help="use graph session mode (default) or eager mode ")
    parser.add_argument('-t', '--task', type=str, default='squad_v1.1',
                        choices=('squad_v1.1', 'mnli', 'qqp', 'boolq', 'race'),
                        help='choose task to run')
    main(parser.parse_args())
