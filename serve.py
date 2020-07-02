# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import signal
import uuid
from queue import Queue
from threading import Thread

import bottle
from tqdm import tqdm

from common import logger
from common import tf
from common.config import get_config_from_args
from common.util import batch
from tasks import get_task_model_class

bottle.BaseRequest.MEMFILE_MAX = 10 * 1024 * 1024
app = bottle.Bottle()
request_queue = Queue()
response_queue = Queue()


def serve(args):
    config = get_config_from_args(args, mode='infer')
    # tf.enable_eager_execution()
    # tf.set_random_seed(config.random_seed)
    checkpoint_path = tf.train.latest_checkpoint(config.checkpoint_dir)

    # initialize model
    sess = tf.Session()
    model = get_task_model_class(config.model, config.task)(config)
    feed_fn, output_tensors = model.infer_graph(config)
    saver = tf.train.Saver(var_list=tf.global_variables())
    saver.restore(sess, checkpoint_path)

    logger.info("{} loaded, waiting for questions...".format(checkpoint_path))

    while True:
        msg = request_queue.get()
        if msg is None:
            break
        # call model to do prediction
        (request_id, model_id, inputs) = msg
        logger.info("begin preprocessing on request={}".format(request_id))
        outputs = []
        input_features = model.text_to_feature(inputs, config)
        logger.info("begin predicting on request={}".format(request_id))
        total_batches = len(input_features) // args.batch_size
        for batch_feature in tqdm(batch(input_features, args.batch_size),
                                  total=total_batches):
            feed = feed_fn(batch_feature)
            # logger.info("{}: batch {} started...".format(request_id, idx))

            model_outputs = sess.run(output_tensors, feed)
            output = model.prepare_outputs(model_outputs, config,
                                           batch_feature)
            # logger.info("{}: batch {} done...".format(request_id, idx))
            outputs.extend(output)
            # prediction_answers = decode_answer(
            #     contexts, context_spans, start_predictions, end_predictions,
            #     output_char_start)
            # all_answers.extend(prediction_answers)
            # all_probabilities.extend([round(float(s), 6)
            # for s in norm_scores])
        logger.info("prediction for {} finished".format(request_id))
        response_queue.put((request_id, model_id, outputs))


@app.post('/qa')
def add_message_to_queue():
    user_request = bottle.request.json
    user_request_id = user_request.get('request_id', uuid.uuid4().hex[:8])
    request_model = user_request.get('model_name', 'bert')
    user_input = user_request['input']
    bottle_env = bottle.request.environ
    client_ip = bottle_env.get('HTTP_X_FORWARDED_FOR') or bottle_env.get(
        'REMOTE_ADDR')

    logger.info("received request={}, model_name={}, from={}".format(
        user_request_id, request_model, client_ip))
    request_queue.put((user_request_id, request_model, user_input))

    (request_id, model_name, output) = response_queue.get()
    logger.info('sending results back to={} for request={}...'.format(
        client_ip, request_id))
    return {"request_id": request_id, "model_name": model_name,
            "output": output}


def main(args):
    prediction_worker = Thread(target=serve, args=(args,))
    prediction_worker.daemon = True
    prediction_worker.start()

    def signal_handler(_signal, _frame):
        print('You pressed Ctrl+C, exiting now...')
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    host = args.ip or 'localhost'
    bottle.run(app, host=host, port=args.port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', '--ip', type=str, default=None,
                        help='ip to serve.')
    parser.add_argument('-p', '--port', type=int, default=60005,
                        help='port to serve')
    parser.add_argument('-c', '--config_file', type=str, default=None,
                        help='Path to qa model config')
    parser.add_argument('-b', '--batch_size', type=int, default=48,)
    parser.add_argument('-m', '--model', type=str, default='bert',
                        choices=('bert', 'ebert'),
                        help='choose model to load default configuration')
    parser.add_argument('-t', '--task', type=str, default='squad_v1.1',
                        choices=('squad_v1.1', 'squad_v2.0', 'hotpot',
                                 'mnli', 'qqp', 'boolq', 'race'),
                        help='choose model to load default configuration')
    main(parser.parse_args())
