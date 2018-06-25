"""
Various utility functions used across several files.
"""

import os
import itertools
import collections
import hashlib
import random
import warnings

import numpy as np
import torch

from . import log


def _next_seeds(n):
    # deterministically generate seeds for envs
    # not perfect due to correlation between generators,
    # but we can't use urandom here to have replicable experiments
    # https://stats.stackexchange.com/questions/233061
    mt_state_size = 624
    seeds = []
    for _ in range(n):
        state = np.random.randint(2**32, size=mt_state_size)
        digest = hashlib.sha224(state.tobytes()).digest()
        seed = np.frombuffer(digest, dtype=np.uint32)[0]
        seeds.append(int(seed))
        if seeds[-1] is None:
            seeds[-1] = int(state.sum())
    return seeds


def seed_all(seed):
    """Seed all devices deterministically off of seed and somewhat
    independently."""
    log.debug('seeding with seed {}', seed)
    np.random.seed(seed)
    rand_seed, torch_cpu_seed, torch_gpu_seed = _next_seeds(3)
    random.seed(rand_seed)
    torch.manual_seed(torch_cpu_seed)
    torch.cuda.manual_seed_all(torch_gpu_seed)


class RollingAverageWindow:
    """Creates an automatically windowed rolling average."""

    def __init__(self, window_size):
        self._window_size = window_size
        self._items = collections.deque([], window_size)
        self._total = 0

    def update(self, value):
        """updates the rolling window"""
        if len(self._items) < self._window_size:
            self._total += value
            self._items.append(value)
        else:
            self._total -= self._items.popleft()
            self._total += value
            self._items.append(value)

    def value(self):
        """returns the current windowed avg"""
        if not self._items:
            return 0
        return self._total / len(self._items)


def import_matplotlib():
    """import and return the matplotlib module in a way that uses
    a display-independent backend (import when generating images on
    servers"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def gpus():
    """
    Lists all GPUs specified by CUDA_VISIBLE_DEVICES or all those available.
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        gpulist = list(range(torch.cuda.device_count()))
    else:
        gpulist = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        gpulist = list(map(int, filter(None, gpulist)))
    gpulist.sort()
    return gpulist


def get_device():
    """
    Retrieve gpu index from env var CUDA_VISIBLE_DEVICES or use the CPU if
    no GPUs are available.

    Verifies at most one GPU is being used.
    """
    gpulist = gpus()

    # multiple GPUs would require DistributedDataParallel (DataParallel
    # doesn't use NCCL and puts all outputs on gpu0). That complexity
    # isn't worth it right now.
    assert len(gpulist) <= 1, 'expecting at most one GPU, found {}'.format(
        len(gpulist))
    if len(gpulist) == 1:
        # no need to set_device b/c we rely on CUDA_VISIBLE_DEVICES
        return torch.device('cuda')
    return torch.device('cpu')


def intfmt(maxval, fill=' '):
    """
    returns the appropriate format string for integers that can go up to
    maximum value maxvalue, inclusive.
    """
    vallen = len(str(maxval))
    return '{:' + fill + str(vallen) + 'd}'


def disable_contiguous_rnn_warning():
    """
    disables a warning due to a pytorch bug
    https://discuss.pytorch.org/t/18969/1
    """
    msg = 'RNN module weights are not part of single contiguous chunk'
    warnings.filterwarnings("ignore", msg, UserWarning)


def disable_source_code_warning():
    """ignore warning about out-of-date models"""
    warnings.simplefilter('ignore', torch.serialization.SourceChangeWarning)


def chunkify(iterable, n):
    """
    Break up an iterable into chunks of size n, except for the last chunk,
    if the iterable does not divide evenly.
    """
    # https://stackoverflow.com/questions/8991506
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk
