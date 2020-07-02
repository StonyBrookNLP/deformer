#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from . import logger
from . import tf
from .checkpoint import init_from_checkpoint
from .optimizer import create_optimizer

from .tf_util import get_tpu_run_config


def input_fn_builder(record_parser, config):
    def input_fn(params):
        # batch_size itself is set by the Estimator
        batch_size = params['batch_size']

        # for TPU drop_remainder is always true, see
        # https://www.tensorflow.org/guide/using_tpu#static_shapes_and_batch_size
        if config.mode == 'train':
            drop_remainder = True
        elif config.mode == 'dev':
            if config.use_tpu:
                drop_remainder = True
            else:
                drop_remainder = False  # eval but not on tpu
        else:
            drop_remainder = False  # infer, tune, analyze

        dataset = tf.data.TFRecordDataset(config.dataset_file)
        if config.mode == 'train':
            dataset = dataset.cache()
            buffer_size = config.input_buffer_size
            dataset = dataset.shuffle(buffer_size, seed=config.random_seed,
                                      reshuffle_each_iteration=True)
            dataset = dataset.repeat()
        dataset = dataset.map(lambda record: record_parser(record, config),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size=batch_size,
                                drop_remainder=drop_remainder)
        return dataset

    return input_fn


def model_fn_builder(model_creator, config):
    model = model_creator(config)

    def model_fn(features, labels, mode, params):
        logger.info("*** Features ***")
        for name in sorted(features.keys()):
            logger.info("  name=%s, shape=%s" % (name, features[name].shape))

        if config.use_tpu and config.bfloat16:
            logger.info("using bfloat16 for tpu")
            with tf.tpu.bfloat16_scope():
                outputs = model.get_logits(features)
            if isinstance(outputs, list):
                outputs = [tf.cast(o, tf.float32) for o in outputs]
            else:
                outputs = tf.cast(outputs, tf.float32)
        else:
            outputs = model.get_logits(features)

        all_trainable_variables = tf.trainable_variables()
        # set fine tune scope
        if 'tune_scopes' in config:
            tune_scopes = config.tune_scopes.split(',')
        else:
            tune_scopes = None
        if isinstance(tune_scopes, list):
            scoped_variables = []
            for scope in tune_scopes:
                scoped_variables.extend(tf.trainable_variables(scope))
            trainable_variables = scoped_variables
        else:
            trainable_variables = all_trainable_variables

        if 'init_scopes' in config:
            init_scopes = config.init_scopes.split(',')
        else:
            init_scopes = None

        if isinstance(init_scopes, list):
            to_be_init_variables = []
            for scope in init_scopes:
                to_be_init_variables.extend(tf.trainable_variables(scope))
        else:
            to_be_init_variables = all_trainable_variables

        initialized_variable_names = {}
        scaffold_fn = None
        init_checkpoint = config.init_checkpoint
        if init_checkpoint:
            assign_map, initialized_variable_names = init_from_checkpoint(
                init_checkpoint, to_be_init_variables,
                replace_map=model.replace_map)
            # logger.info('assign_map: \n{}'.format(assign_map))
            # logger.info('initialized_variable_names: \n{}'.format(
            # initialized_variable_names))

            if config.use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assign_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assign_map)

        logger.info("**** Initialized Variables ****")
        for var in to_be_init_variables:
            init_str = ""
            if var.name in initialized_variable_names:
                init_str = ", *INIT_FROM_CKPT*"
            logger.info("  name=%s, shape=%s%s", var.name, var.shape, init_str)

        if mode == tf.estimator.ModeKeys.TRAIN:
            logger.info("**** Trainable Variables ****")
            for var in trainable_variables:
                init_str = ""
                if var.name in initialized_variable_names:
                    init_str = ", *INIT_FROM_CKPT*"
                logger.info("  name=%s, shape=%s%s", var.name, var.shape,
                            init_str)

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = model.get_prediction(features, outputs)
            # for var in model.trainable_weights:
            #     logger.info("var=%s, shape=%s, trainable=True" % (
            #         var.name, var.shape))
            # for var in model.non_trainable_weights:
            #     logger.info("var=%s, shape=%s, trainable=False" % (
            #         var.name, var.shape))
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
        else:
            # EVAL and TRAIN share the same code,
            # only dataset source is different
            total_loss, print_tensor_dict = model.get_loss(features, labels,
                                                           config)
            train_op, global_step = create_optimizer(total_loss, config,
                                                     trainable_variables)
            eval_metrics = model.get_eval_metrics(features, labels, config)

            if config.get('use_host_call', True):
                from tensorflow.contrib import summary
                # make sure global_step the first one
                tensors_to_print_names = ['global_step']

                # To log the loss, current learning rate, and epoch for
                # TensorBoard, the summary op needs to be run on the host CPU
                # via host_call. host_call expects [batch_size, ...] Tensors,
                # thus reshape to introduce a batch dimension.
                # These Tensors are implicitly concatenated to
                # [params['batch_size']].
                tensors_to_print = [tf.reshape(global_step, [1])]

                for tensor_name, tensor_to_print in print_tensor_dict.items():
                    tensors_to_print_names.append(tensor_name)
                    tensors_to_print.append(tf.reshape(tensor_to_print, [1]))

                def host_call_fn(*tensors):
                    """Training host call. Creates scalar summaries for
                    training metrics. This function is executed on the CPU and
                    should not directly reference any Tensors in the rest of
                    the `model_fn`. To pass Tensors from the  model to
                    the `metric_fn`, provide as part of the `host_call`. See
                    https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
                    for more information. Arguments should match the list of
                    `Tensor` objects passed as the second element in the tuple
                     passed to `host_call`.
                    """
                    gs = tensors[0][0]
                    # Host call fns are executed params['iterations_per_loop']
                    # times after one TPU loop is finished, setting max_queue
                    # value to the same as number of iterations will make the
                    # summary writer only flush the data to storage once per
                    # loop.
                    summary_writer = summary.create_file_writer(
                        config.checkpoint_dir,
                        max_queue=config.get('iterations_per_loop', 1000))
                    with summary_writer.as_default():
                        with summary.always_record_summaries():
                            for idx in range(len(tensors_to_print)):
                                summary.scalar(tensors_to_print_names[idx],
                                               tensors[idx][0], step=gs)
                            return summary.all_summary_ops()

                host_call = (host_call_fn, tensors_to_print)
            else:
                host_call = None

            if not config.use_tpu:
                print_steps = config.print_steps or 100
                print_tensor_dict['global_step'] = global_step
                hook = tf.train.LoggingTensorHook(print_tensor_dict,
                                                  every_n_iter=print_steps)
                eval_metric_ops = eval_metrics[0](*eval_metrics[1])
                for name, metric_op in eval_metric_ops.items():
                    tf.summary.scalar(name, metric_op[1])
                output_spec = tf.estimator.EstimatorSpec(
                    mode,
                    loss=total_loss,
                    train_op=train_op,
                    training_hooks=[hook],
                    eval_metric_ops=eval_metric_ops)
            else:
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op,
                    # training_hooks=[hook],
                    host_call=host_call,
                    eval_metrics=eval_metrics,
                    scaffold_fn=scaffold_fn)

        return output_spec

    return model_fn, model


def get_estimator(config, model_class):
    run_config = get_tpu_run_config(config)
    model_fn, model = model_fn_builder(model_class, config)
    # If TPU is not available, this will fall back to Estimator on CPU or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=config.use_tpu,
        model_fn=model_fn,
        config=run_config,
        model_dir=config.checkpoint_dir,
        train_batch_size=config.train_batch_size,
        eval_batch_size=config.dev_batch_size,
        predict_batch_size=config.dev_batch_size)
    return estimator, model
