#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import tensorflow

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(SRC_DIR, "config")
DATA_DIR = os.environ.get("deformer_data", None) or os.path.join(SRC_DIR, "data")
RES_DIR = os.path.join(DATA_DIR, "res")

logger = logging.getLogger('deformer')

logger.setLevel(logging.INFO)
fmt_str = "%(levelname)s:%(asctime)s.%(msecs)03d:%(pathname)s:%(lineno)d: " \
          "%(message)s"
fmt = logging.Formatter(fmt_str, "%Y-%m-%d_%H:%M:%S")
handler = logging.StreamHandler()
handler.setFormatter(fmt)
logger.addHandler(handler)
logger.propagate = False

tf = tensorflow
layers = tf.keras.layers
