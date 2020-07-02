#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import itertools
import os

import numpy as np


def abbreviate(x):
    abbreviations = ["", "K", "M", "B", "T"]
    thing = "1"
    a = 0
    while len(thing) <= len(str(x)) - 3:
        thing += "000"
        a += 1
    b = int(thing)
    thing = round(x / b, 2)
    return str(thing) + " " + abbreviations[a]


def batch(iterable, batch_size=1):
    length = len(iterable)
    for ndx in range(0, length, batch_size):
        yield iterable[ndx:min(ndx + batch_size, length)]


def grouper(iterable, n, fillvalue=None):
    """Collect data into fixed-length chunks or blocks"""
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def repeat_iterable(iterable, num):
    return itertools.chain.from_iterable(itertools.repeat(iterable, num))


def batch_idx(length, batch_size):
    return ((pos, pos + batch_size) for pos in range(0, length, batch_size))


def save_tensor(tensor, tensor_name, save_dir, layer_str=None):
    if not save_dir:
        return
    os.makedirs(save_dir, exist_ok=True)
    if layer_str:
        tensor_filename = '{}_{}'.format(layer_str, tensor_name)
    else:
        tensor_filename = tensor_name

    np.save(os.path.join(save_dir, tensor_filename), tensor)
