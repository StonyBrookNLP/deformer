#! /usr/bin/env python3
import argparse
import os

import numpy as np
import tensorflow as tf
from IPython.display import HTML
from tensorflow.python.framework import tensor_util


def convert_weights(inf_graph):
    """Strip large constant values from graph_def."""
    how_many_converted = 0
    out_graph_def = tf.GraphDef()
    for input_node in inf_graph.node:
        output_node = out_graph_def.node.add()
        output_node.MergeFrom(input_node)
        if output_node.op == 'Const':
            tensor_proto = output_node.attr['value'].tensor
            if tensor_proto.tensor_content:
                np_array = tensor_util.MakeNdarray(tensor_proto)
                tensor_proto.tensor_content = str(np_array).encode()
                how_many_converted += 1

    print("converted %d weights to numpy string." % how_many_converted)
    return out_graph_def


def save_html(graph_def, html_file=None):
    """Visualize TensorFlow graph."""
    strip_def = convert_weights(graph_def)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:1200px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph' + str(np.random.rand()))

    iframe = """
        <iframe seamless style="overflow: hidden; height: 100%;
        width: 100%; position: absolute;" height="100%" width="100%" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    h = HTML(iframe)
    if html_file:
        with open("{}.html".format(html_file), 'w') as f:
            f.write(h.data)
        print("graph vis file saved to {}.html".format(os.path.abspath(html_file)))


if __name__ == '__main__':
    inference_graph_def = tf.GraphDef()
    parser = argparse.ArgumentParser()
    parser.add_argument("inference_graph", type=str,
                        help="a TensorFlow inference graph file, endswith .pb")
    args = parser.parse_args()

    model_name = os.path.splitext(args.inference_graph)[0]
    with tf.io.gfile.GFile(args.inference_graph, "rb") as f:
        inference_graph_def.ParseFromString(f.read())
    save_html(inference_graph_def, model_name)
