"""
Trains a simple WikiSQL-specific architecture on the WikiSQL
dataset.

Assumes that the processed-toy(0|1).pth dataset exists in
{dataroot}/wikisql/ already.
"""

import os
import sys
import itertools

from absl import app
from absl import flags
from contextlib import closing
import torch
import torch.multiprocessing as mp
from torch import optim
import torch.utils
from tqdm import tqdm
import numpy as np

from asn4sql import log
from asn4sql import data
from asn4sql import wikisql_specific
from asn4sql.utils import (seed_all, gpus, get_device, RollingAverageWindow,
                           intfmt)

# dataset and initialization config
flags.DEFINE_integer('seed', 1, 'random seed')
flags.DEFINE_boolean('toy', False, 'use a toy dataset for debugging')

# training logistics
# TODO: batching for batch size > !
# TODO: checkpointing (and checkpoint restores)
# flags.DEFINE_string('restore_checkpoint', None, 'checkpoint to restore '
#                     'training from')
# flags.DEFINE_integer(
#     'persist_every', 25,
#     'period of mini-batches between checkpoint persists (0 to disable)')
flags.DEFINE_integer('evaluate_every', 1,
                     'period of epochs between evaluations (0 to disable)')
flags.DEFINE_integer('max_epochs', 10, 'maximum number of epochs for training')
flags.DEFINE_integer('workers', 4,
                     'number of CPU workers for parallelizing '
                     'training in a data-parallel manner (we only ever use '
                     'at most one GPU, but python-heavy processing can be '
                     'parallelized')

# optimizer
flags.DEFINE_integer('batch_size', 4, 'batch size')
flags.DEFINE_float('learning_rate', 0.1, 'learning rate')


def _main(argv):

    seed_all(flags.FLAGS.seed)
    log_subdir = log.flaghash_dirname([argv[0]], ['seed'])
    log.init(log_subdir)

    log.debug('found gpus {}', gpus())

    dataset_file = os.path.join(
        flags.FLAGS.dataroot, 'wikisql',
        'processed-toy{}.pth'.format(1 if flags.FLAGS.toy else 0))
    log.debug('loading data from {}', dataset_file)
    train, val, _ = torch.load(dataset_file)

    log.debug('building model')
    model = wikisql_specific.WikiSQLSpecificModel(train.fields)
    log.debug('built model:\n{}', model)
    num_parameters = int(sum(p.numel() for p in model.parameters()))
    log.debug('number of parameters in model {}', num_parameters)

    device = get_device()
    torch.save(
        model.to(torch.device('cpu')),
        os.path.join(log.logging_directory(), 'untrained_model.pth'))
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=flags.FLAGS.learning_rate)
    # need to init the buffers before sharing
    loss, _ = model(model.prepare_example(train[0]))
    loss.backward()
    optimizer.zero_grad()
    model = model.share_memory()


    with closing(_SharedTrainer(model)) as trainer:
        _do_training(model, train, val, trainer, optimizer)

# TODO shared memory, only transfer indices.
class _Worker:
    """
    A training process, over which batches are multiplexed.
    """
    def __init__(self, num_workers, worker_idx, model):
        self._worker_idx = worker_idx
        self._num_workers = num_workers
        fmt = '{: ' + str(len(str(num_workers))) + 'd}'
        self._id_str = ('worker ' + fmt + ' of ' + fmt).format(
            worker_idx, num_workers)

        ctx = mp.get_context('forkserver') # .get_context()#'spawn')
        self._conn, child_conn = ctx.Pipe()
        from asn4sql.parallel_train import _child_loop
        self._proc = ctx.Process(target=_child_loop, args=(
            self._conn, child_conn, self._id_str, model))
        self._proc.start()
        child_conn.close()

    def _lo(self, batch_size):
        return self._worker_idx * batch_size // self._num_workers

    def _hi(self, batch_size):
        return (self._worker_idx + 1) * batch_size // self._num_workers

    def _push(self, method_name, args, swallow_errors=False):
        try:
            self._conn.send((method_name, args))
        except IOError as e:
            if swallow_errors:
                msg = 'parent swallowing IOError {} from {}\n'.format(
                    str(e), self._id_str)
                print(msg, end='', file=sys.stderr)
            else:
                raise e

    def _pull(self, expected_name):
        method_name, result = self._conn.recv()
        assert method_name == expected_name, (method_name, expected_name)
        return result

    def train(self, examples):
        """remote batch sharding train step"""
        batch_size = len(examples)
        examples = examples[self._lo(batch_size):self._hi(batch_size)]
        self._push('train', (examples, batch_size))

    def train_finish(self):
        """
        wait until remote train completes; return loss, acc
        contribution of this worker's part of the batch.
        """
        return self._pull('train')

    def opt(self):
        self._push('opt', None)
    def opt_finish(self):
        self._pull('opt')

    def close(self):
        """initiate remote close"""
        # racy if, so we swallow errors.
        if self._proc.is_alive():
            self._push('close', tuple(), swallow_errors=True)

    def close_finish(self):
        """join remote worker process"""
        self._conn.close()
        self._proc.join()

class _SharedTrainer:
    def __init__(self, model):
        self.n = flags.FLAGS.workers
        self._workers = [_Worker(self.n, i, model) for i in range(self.n)]

    def train(self, examples):
        """
        shard and perform fwd/bwd pass on a batch of examples, returning
        mean loss and accuracy.
        """
        for worker in self._workers:
            worker.train(examples)
        loss, acc = 0, 0
        for worker in self._workers:
            worker_loss, worker_acc = worker.train_finish()
            loss += worker_loss
            acc += worker_acc
        return loss, acc

    def opt(self):
        """close workers"""
        for worker in self._workers:
            worker.opt()
        for worker in self._workers:
            worker.opt_finish()

    def close(self):
        """close workers"""
        for worker in self._workers:
            worker.close()
        for worker in self._workers:
            worker.close_finish()
        self._workers = []


def _do_training(model, train, val, trainer, optimizer):

    # training_state = _TrainingState()
    # if flags.FLAGS.restore_checkpoint:
    #     log.debug('restoring model from {}', flags.FLAGS.restore_checkpoint)
    #     _load_checkpoint(flags.FLAGS.restore_checkpoint, model, optimizer,
    #                      training_state)

    batch_size = flags.FLAGS.batch_size
    loss_window = RollingAverageWindow(len(train) // 10 // batch_size)
    acc_window = RollingAverageWindow(len(train) // 10 // batch_size)
    grad_window = RollingAverageWindow(len(train) // 10 // batch_size)

    model.train()

    perm = np.arange(len(train))
    for epoch in range(1, 1 + flags.FLAGS.max_epochs):
        epochfmt = intfmt(flags.FLAGS.max_epochs)
        log.debug('begin epoch ' + epochfmt, epoch)
        # one sample at a time greatly simplifies pytorch seq2seq!
        np.random.shuffle(perm)
        samples = (train[i] for i in perm)
        for exs in _chunkify(tqdm(samples, total=len(train)), batch_size):
            # TODO batch size > 1 by not zeroing, see lbs/training.py
            # optimizer.zero_grad()
            loss, acc = trainer.train(exs)
            # prepared_ex = model.prepare_example(ex)
            # loss, acc = model.forward(prepared_ex)
            loss_window.update(loss)
            acc_window.update(acc)
            # loss.backward()
            with torch.no_grad():
                grad = torch.cat(
                    tuple(p.grad.data.view(-1) for p in model.parameters()))
                gradnorm = torch.norm(grad)
            grad_window.update(gradnorm.detach().cpu().numpy())
            trainer.opt()
            # optimizer.step()

        log.debug('end epoch ' + epochfmt +
                  ' rolling loss {:8.4g}'
                  ' rolling acc {:8.4g}'
                  ' rolling grad norm {:8.4g}',
                  epoch,
                  loss_window.value(),
                  acc_window.value(),
                  grad_window.value())

        if _check_period(epoch, flags.FLAGS.evaluate_every):
            model.eval()
            val_diagnostics = _diagnose(val, model)
            train_diagnositcs = _diagnose(train, model, len(val))
            val_diagnostics = _str_diagnostics('val', val_diagnostics)
            train_diagnositcs = _str_diagnostics('(sampled) train',
                                                 train_diagnositcs)
            log.debug('epoch ' + epochfmt + ' of ' + epochfmt + '\n{}\n{}',
                      epoch, flags.FLAGS.max_epochs, val_diagnostics,
                      train_diagnositcs)
            model.train()

        # if _check_period(training_state.batch_idx, flags.FLAGS.persist_every):
        #     fmt = '{:0' + str(len(str(flags.FLAGS.max_batches))) + 'd}.pth'
        #     checkpoint_file = os.path.join(
        #         log.logging_directory(), 'checkpoints',
        #         fmt.format(training_state.batch_idx))
        #     log.debug('persisting model to {}', checkpoint_file)
        #     _save_checkpoint(checkpoint_file, model, optimizer, training_state)


def _diagnose(dataset, model, subsample=None):
    if subsample is None:
        subsample = range(len(dataset))
    else:
        subsample = np.random.choice(len(dataset), subsample, replace=False)
    sum_diagnostics = {}
    with torch.no_grad():
        for ex in (dataset[i] for i in subsample):
            prepared_ex = model.prepare_example(ex)
            diagnostics = model.diagnose(prepared_ex)
            for k, (value, fmt) in diagnostics.items():
                if k not in sum_diagnostics:
                    sum_diagnostics[k] = (value.detach().cpu().numpy(), fmt)
                else:
                    sum_value, sum_fmt = sum_diagnostics[k]
                    assert sum_fmt == fmt, (sum_fmt, fmt)
                    sum_value += value.detach().cpu().numpy()
                    sum_diagnostics[k] = (sum_value, fmt)
    avg_diagnostics = {
        k: (value / len(dataset), fmt)
        for k, (value, fmt) in sum_diagnostics.items()
    }
    return avg_diagnostics


def _str_diagnostics(diagnostics_name, diagnostics):
    preamble = '  ' + diagnostics_name
    if not diagnostics:
        return preamble
    newline_and_indent = '\n    '
    maxlen = max(map(len, diagnostics))
    namefmt = '{:<' + str(maxlen) + '}'
    values = [(k, ) + diagnostics[k] for k in sorted(diagnostics)]
    return preamble + newline_and_indent + newline_and_indent.join(
        (namefmt + ' ' + valuefmt).format(name, value)
        for name, value, valuefmt in values)

def _chunkify(iterable, n):
    # https://stackoverflow.com/questions/8991506
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

def _check_period(idx, period):
    if period == 0:
        return False
    return idx == 1 or idx == flags.FLAGS.max_epochs or idx % period == 0


# def _load_checkpoint(checkpoint_file, model, optimizer, training_state):
#     state_dict = torch.load(checkpoint_file)
#     model.load_state_dict(state_dict['model'])
#     optimizer.load_state_dict(state_dict['optimizer'])
#     training_state.load_state_dict(state_dict['training_state'])

# def _save_checkpoint(checkpoint_file, model, optimizer, training_state):
#     state_dict = {
#         'model': model.state_dict(),
#         'optimizer': optimizer.state_dict(),
#         'training_state': training_state.state_dict()
#     }

#     os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
#     with open(checkpoint_file, 'wb') as f:
#         torch.save(state_dict, f)

# class _TrainingState:
#     def __init__(self):
#         self.batch_idx = 0

#     def state_dict(self):
#         """return all training state for algorithm"""
#         return {'batch_idx': self.batch_idx}

#     def load_state_dict(self, d):
#         """re-load training state from dictionary"""
#         self.batch_idx = d['batch_idx']

if __name__ == '__main__':
    app.run(_main)
