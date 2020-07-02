#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os

import numpy as np

from common import logger
from common import tf
from common.config import get_config_from_args
from common.text import decode_answer
from models import get_model


def main(args):
    config = get_config_from_args(args, mode='infer')
    max_seq_length = args.max_seq_length or config.max_seq_length
    config.max_seq_length = max_seq_length
    max_answer_span = args.max_answer_span or config.max_answer_span
    config.max_answer_span = max_answer_span

    model_file = args.model_file
    questions = [
        "What is the AFC short for?",
        # "What day was the game played on?",
    ]
    contexts = [
        "The American Football Conference (AFC) champion Denver Broncos defeated the National "
        "Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title.",
        # "The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area " \
        # "at Santa Clara, California.",
    ]

    logger.info("running in eager mode...")
    tf.enable_eager_execution()
    checkpoint_path = tf.train.latest_checkpoint(config.checkpoint_dir)

    logger.info("restoring model weights...")

    with tf.contrib.eager.restore_variables_on_create(checkpoint_path):

        model = get_model(config)
        logger.info("warming up model...")
        model.warm_up(config)

    context_spans, inputs = model.get_inputs(questions, contexts, config)
    inputs_tensor = [tf.convert_to_tensor(i, dtype=tf.int32) for i in inputs.values()]
    logger.info("begin inferring...")
    start_predictions, end_predictions, norm_scores = model.infer(inputs_tensor,
                                                                  max_answer_span=config.max_answer_span,
                                                                  export=True)

    prediction_answers = decode_answer(contexts, context_spans, start_predictions, end_predictions)
    for q, c, a, ns in zip(questions, contexts, prediction_answers, norm_scores):
        logger.info('q={}\na={}\n\tcontext={}\n\n'.format(q, (a, round(float(ns), 4)), c))

    print(model.embeddings.shape)
    print(model.logits.shape)
    input_ids = inputs_tensor[0]
    print(input_ids.shape)

    input_ids_file = os.path.join(os.path.dirname(model_file), 'input_ids')
    input_embeddings_file = os.path.join(os.path.dirname(model_file), 'input_embeddings')
    output_logits_file = os.path.join(os.path.dirname(model_file), 'output_logits')
    np.save(input_ids_file, input_ids)
    np.save(input_embeddings_file, model.embeddings)
    np.save(output_logits_file, model.logits)

    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    print(input_details)
    print(output_details)
    print(model.logits)
    interpreter.set_tensor(input_details[0]['index'], input_ids)
    interpreter.set_tensor(input_details[1]['index'], model.embeddings)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data.shape)
    print(output_data)
    print(np.allclose(output_data, model.logits, rtol=1e-4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mf', '--model_file', type=str, default=None, help='frozen TensorFlow model or TFLite model')
    parser.add_argument('-c', '--config_file', type=str)
    parser.add_argument('-m', '--model', type=str, default='bert', choices=('bert', 'qanet', 'rnet', 'rmr', 'drqa'),
                        help='choose model to load default configuration')
    parser.add_argument("-mas", "--max_answer_span", type=int, default=30, help="max_answer_span")
    parser.add_argument("-msl", "--max_seq_length", type=int, default=None, help="max_seq_length")
    parser.add_argument("-e", "--eager", action='store_true', help="use graph session mode (default) or eager mode ")
    main(parser.parse_args())
