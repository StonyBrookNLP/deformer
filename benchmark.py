#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import time

import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops

from common import logger
from common import tf
from common.config import get_config_from_args
from common.util import abbreviate
from tasks import get_task_model_class


# TODO: see https://github.com/tensorflow/tensorflow/pull/30575
@ops.RegisterStatistics("BatchMatMulV2", "flops")
def _calc_mat_mul_flops(graph, node):
    """Calculates the compute resources needed for MatMul."""
    transpose_a = node.attr["transpose_a"].b
    a_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[0])
    a_shape.assert_is_fully_defined()
    if transpose_a:
        k = int(a_shape[-2])
    else:
        k = int(a_shape[-1])
    output_shape = graph_util.tensor_shape_from_node_def_name(graph, node.name)
    output_shape.assert_is_fully_defined()
    output_count = np.prod(output_shape.as_list())
    return ops.OpStats("flops", (k * output_count * 2))


def main(args):
    config = get_config_from_args(args, mode='infer')
    model_name = config.model
    kwargs = dict(training=False, logits=True)
    if model_name == 'ebert':
        kwargs['fake_cache_first'] = args.cache_segment == 1
        kwargs['fake_cache_second'] = args.cache_segment == 2

    config.batch_size = args.batch_size
    config.max_seq_length = args.max_seq_length or config.max_seq_length
    logger.info("running in graph mode...")
    run_metadata = tf.RunMetadata()

    with tf.Session() as sess:
        model = get_task_model_class(config.model, task=args.task)(config)
        inputs_dict, logits_ph = model.export_graph(config, **kwargs)
        sess.run(tf.global_variables_initializer())
        # saver = tf.train.Saver()
        # saver.save(sess, 'data/sbert', write_meta_graph=False)
        opt_builder = tf.profiler.ProfileOptionBuilder
        if args.print_parameters:
            tf.profiler.profile(
                sess.graph, options=opt_builder.trainable_variables_parameter())

        if not args.not_profile_flops:
            prof_options = opt_builder.float_operation()
            prof_options['hide_name_regexes'] = ['.*/Initializer/.*']
            tfprof_node = tf.profiler.profile(sess.graph, options=prof_options)
            profile_metric(model_name, tfprof_node, metric='total_float_ops',
                           metric_name='flops')

        if args.profile_memory:
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = run_metadata
        else:
            options = None
            run_metadata = None
        _ = sess.run([logits_ph], feed_dict=inputs_dict,
                     options=options,
                     run_metadata=run_metadata)

        if args.profile_memory:
            opts = tf.profiler.ProfileOptionBuilder(
                tf.profiler.ProfileOptionBuilder.time_and_memory()).build()

            tfprof_node = tf.profiler.profile(
                tf.get_default_graph(),
                run_meta=run_metadata,
                cmd='scope',
                options=opts)

            profile_metric(model_name, tfprof_node,
                           metric='total_requested_bytes', metric_name='mem')

        if args.profile_time:
            # warm up two rounds
            logger.info("warm up for two rounds...")

            for _ in range(2):
                sess.run([logits_ph], feed_dict=inputs_dict, )

            logger.info("start running 10 rounds...")
            start_time = time.time()
            # bench 10 rounds, take avg
            for _ in range(10):
                sess.run([logits_ph], feed_dict=inputs_dict, )
            end_time = time.time()
            print('infer_time: {:.4f} s'.format((end_time - start_time) / 10))


def profile_metric(model_name, tfprof_node, metric='total_float_ops',
                   metric_name='flops'):
    metric_value = dict()
    # attn_dict = defaultdict(dict)
    attn_other = dict()
    attn_mm = dict()
    attn_attn = dict()
    attn_ctx = dict()
    attn_softmax = dict()
    ffn_metric_other = dict()
    ffn_metric_mm = dict()
    other_metric = dict()

    def traverse_node(node):
        if node.children:
            for child in node.children:
                traverse_node(child)
        else:
            ki = node.name
            vi = getattr(node, metric)
            metric_value[ki] = vi

            if model_name == 'bert':
                attn_pattern = r".*/bert/encoder/layer_\d+/attention.*"
                ffn_pattern = r".*/bert/encoder/layer_\d+/(output|intermediate)/dense.*"
            elif model_name in ['ebert', 'sbert']:
                attn_pattern = r".*/ebert/(upper|lower)_encoder(_1)?/layer_\d+/attention.*"
                ffn_pattern = r".*/ebert/(upper|lower)_encoder(_1)?/layer_\d+/(output|intermediate)/dense.*"
            else:
                # for others
                attn_pattern = ''
                ffn_pattern = ''

            if re.match(attn_pattern, ki):
                if ki.endswith('MatMul'):
                    attn_mm[ki] = vi
                elif ki.endswith('AttnMatmul'):
                    attn_attn[ki] = vi
                elif ki.endswith('ContextMatmul'):
                    attn_ctx[ki] = vi
                elif ki.endswith('Softmax'):
                    attn_softmax[ki] = vi
                else:
                    attn_other[ki] = vi
            elif re.match(ffn_pattern, ki):
                if ki.endswith('MatMul'):
                    ffn_metric_mm[ki] = vi
                else:
                    ffn_metric_other[ki] = vi
            else:
                other_metric[ki] = vi

    traverse_node(tfprof_node)
    total_metric_value = getattr(tfprof_node, metric)
    print()
    print('{}: {}, {}'.format(metric, total_metric_value,
                              abbreviate(total_metric_value)))
    # print('total_sum:', sum(flops.values()))
    attn_mm_sum = sum(attn_mm.values())
    attn_attn_sum = sum(attn_attn.values())
    attn_ctx_sum = sum(attn_ctx.values())
    attn_softmax_sum = sum(attn_softmax.values())
    attn_other_sum = sum(attn_other.values())
    attn_sum = attn_mm_sum + attn_attn_sum + attn_ctx_sum + attn_softmax_sum \
               + attn_other_sum
    print('attn_{}: {}, {}, ({:.2f}%)'.format(
        metric_name, attn_sum, abbreviate(attn_sum),
        attn_sum * 100 / total_metric_value))
    print('  {}_attn_tran: {}, {}, ({:.2f}%)'.format(
        metric_name, attn_mm_sum, abbreviate(attn_mm_sum),
        attn_mm_sum * 100 / total_metric_value))
    print('  {}_attn_matmul_attn: {}, {}, ({:.2f}%)'.format(
        metric_name, attn_attn_sum, abbreviate(attn_attn_sum),
        attn_attn_sum * 100 / total_metric_value))
    print('  {}_attn_matmul_ctx: {}, {}, ({:.2f}%)'.format(
        metric_name, attn_ctx_sum, abbreviate(attn_ctx_sum),
        attn_ctx_sum * 100 / total_metric_value))
    print('  {}_attn_softmax: {}, {}, ({:.2f}%)'.format(
        metric_name, attn_softmax_sum, abbreviate(attn_softmax_sum),
        attn_softmax_sum * 100 / total_metric_value))
    print('  {}_attn_other: {}, {}, ({:.2f}%)'.format(
        metric_name, attn_other_sum, abbreviate(attn_other_sum),
        attn_other_sum * 100 / total_metric_value))
    ffn_metric_mm_sum = sum(ffn_metric_mm.values())
    ffn_metric_other_sum = sum(ffn_metric_other.values())
    ffn_metric_sum = ffn_metric_mm_sum + ffn_metric_other_sum
    print('ffn_{}: {}, {}, ({:.2f}%)'.format(
        metric_name, ffn_metric_sum, abbreviate(ffn_metric_sum),
        ffn_metric_sum * 100 / total_metric_value))
    print('  {}_ffn_matmul: {}, {}, ({:.2f}%)'.format(
        metric_name, ffn_metric_mm_sum, abbreviate(ffn_metric_mm_sum),
        ffn_metric_mm_sum * 100 / total_metric_value))
    print('  {}_ffn_other: {}, {}, ({:.2f}%)'.format(
        metric_name, ffn_metric_other_sum, abbreviate(ffn_metric_other_sum),
        ffn_metric_other_sum * 100 / total_metric_value))
    print('other_{}: {}, {}, ({:.2f}%)'.format(
        metric_name, sum(other_metric.values()),
        abbreviate(sum(other_metric.values())),
        sum(other_metric.values()) * 100 / total_metric_value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str)
    parser.add_argument('-m', '--model', type=str, default='bert',
                        choices=('bert', 'ebert'),
                        help='choose model to load default configuration')
    parser.add_argument('-t', '--task', type=str, default='squad_v1.1',
                        choices=('squad_v1.1', 'mnli', 'qqp', 'boolq', 'race'),
                        help='choose task to run')
    parser.add_argument("-msl", "--max_seq_length", type=int, default=0,
                        help="max_seq_length")
    parser.add_argument("-bs", "--batch_size", type=int, default=1,
                        help="batch_size")
    parser.add_argument("-pp", "--print_parameters", action='store_true')
    parser.add_argument("-pm", "--profile_memory", action='store_true')
    parser.add_argument("-pt", "--profile_time", action='store_true')
    parser.add_argument("-npf", "--not_profile_flops", action='store_true')
    parser.add_argument("-cs", "--cache_segment", default=0, type=int,
                        choices=(0, 1, 2),
                        help='cache first or second segment lower encoding')
    main(parser.parse_args())
