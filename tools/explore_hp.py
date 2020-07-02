#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import sys

import redis
from bayes_opt import BayesianOptimization
from bayes_opt.event import Events
from bayes_opt.observer import JSONLogger


class RedisQueue(object):
    """Simple Queue with Redis Backend, modified from
    http://peter-hoffmann.com/2012/python-simple-queue-redis-queue.html
    """

    def __init__(self, name, namespace='queue', **redis_kwargs):
        """The default connection parameters are: host='localhost', port=6379, db=0"""
        self.__db = redis.Redis(**redis_kwargs)
        self.key = '%s:%s' % (namespace, name)

    def clear(self):
        return self.__db.delete(self.key)

    def list(self, start=0, end=-1):
        """
        LRANGE queue:params 0 -1
        """
        return self.__db.lrange(self.key, start, end)

    def put_before(self, ref_value, item):
        """
        LINSERT queue:params before ref_value "hello"
        """
        self.__db.linsert(self.key, 'before', ref_value, item)

    def put_after(self, ref_value, item):
        """
        LINSERT queue:params after ref_value "hello"
        """
        self.__db.linsert(self.key, 'after', ref_value, item)

    def remove(self, item, count=0):
        """
        LREM queue:params 0 "hello"
        """
        self.__db.lrem(self.key, count, item)

    def put(self, item):
        """Put item into the queue.
        RPUSH queue:params "hello"
        """
        self.__db.rpush(self.key, item)

    def get(self, block=True, timeout=None):
        """Remove and return an item from the queue.

        If optional args block is true and timeout is None (the default), block
        if necessary until an item is available."""
        if block:
            item = self.__db.blpop(self.key, timeout=timeout)
        else:
            item = self.__db.lpop(self.key)

        if item:
            item = item[1]
        return item


"""
explore logic:
1. start explore_hp: with initial params, and wait for eval results to come, 
until all params are processed

(duo to tpu compile bug?, train-and-evaluate has to be issued separately across multiple runs)
2. start train and eval shell script: 
use redis-cli to get params from params-queue 
(stop training if params are not available within specified timeout)
and put results to results-queue

"""


def main(args):
    start_params = args.start_params
    redis_port = args.redis_port

    params_queue = RedisQueue('params-' + args.task, port=redis_port)
    params_queue.clear()
    results_queue = RedisQueue('results-' + args.task, port=redis_port)
    results_queue.clear()
    hyper_parameters = args.hyper_parameters
    print('hyper_parameters:', hyper_parameters)
    hp_db = redis.Redis(port=redis_port)
    hp_db.set('hp-{}-{}'.format(args.task, args.size), hyper_parameters)
    print('start_params:', start_params)
    print('hyper_parameters key:', 'hp-{}-{}'.format(args.task, args.size))
    print('params queue:', 'params-' + args.task)
    print('results queue:', 'results-' + args.task)
    sys.stdout.flush()

    def train_and_eval(kd_alpha, kd_mse_beta, ce_gama):
        kd_alpha, kd_mse_beta, ce_gama = round(kd_alpha, 1), round(kd_mse_beta, 1), round(ce_gama, 1)
        params_queue.put('{},{},{}'.format(kd_alpha, kd_mse_beta, ce_gama))
        print("waiting results for a={},b={},c={}...".format(kd_alpha, kd_mse_beta, ce_gama))
        sys.stdout.flush()
        eval_result = results_queue.get(block=True)
        print("eval_result: {} for a={},b={},c={}".format(eval_result, kd_alpha, kd_mse_beta, ce_gama))
        sys.stdout.flush()
        return round(float(eval_result), 2)

    # Bounded region of parameter space
    param_space = {'kd_alpha': (0.1, 2.0), 'kd_mse_beta': (0.1, 2.0), 'ce_gama': (0.1, 2.0)}

    optimizer = BayesianOptimization(
        f=train_and_eval,
        pbounds=param_space,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )
    optimizer.probe(
        params={'kd_alpha': start_params[0], 'kd_mse_beta': start_params[1], 'ce_gama': start_params[2]},
        # params={'kd_alpha': 0.5, 'kd_mse_beta': 0.2, 'ce_gama': 0.3},  # s9
        # params={'kd_alpha': 0.3, 'kd_mse_beta': 0.2, 'ce_gama': 0.8}, #s10
        lazy=False,
    )
    logger = JSONLogger(path=args.progress_log)
    optimizer.subscribe(Events.OPTMIZATION_STEP, logger)
    optimizer.maximize(
        init_points=args.init_points,
        n_iter=args.n_iter,
    )
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))
    print(optimizer.max)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--progress_log', type=str, default='bayes-opt.json')
    parser.add_argument('-hp', '--hyper_parameters', type=str, default='5e-5,5,32',
                        help="learning_rate,epochs,train_batch_size")
    parser.add_argument('-t', '--task', type=str, default='squad_v1.1',
                        choices=('squad_v1.1', 'mnli', 'qqp', 'boolq', 'race'),
                        help='choose model to load default configuration')
    parser.add_argument('-s', '--size', type=str, default='base',
                        choices=('base', 'large'), help='choose model size')
    parser.add_argument('-i', '--init_points', type=int, default=2)
    parser.add_argument('-rp', '--redis_port', type=int, default=60001)
    parser.add_argument('-sp', '--start_params', type=float, nargs=3, default=(0.5, 0.2, 0.3),
                        help='initial start params, in the order of kd_alpha, kd_mse_beta, ce_gama')
    parser.add_argument('-n', '--n_iter', type=int, default=50)
    main(parser.parse_args())
