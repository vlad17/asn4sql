"""
Trains a simple WikiSQL-specific architecture on the WikiSQL
dataset.

Assumes that the processed-toy(0|1).pth dataset exists in
{dataroot}/wikisql/ already.
"""

from contextlib import closing
import os
import shutil
import itertools

from absl import app
from absl import flags
import torch
import torch.utils
from tqdm import tqdm
import numpy as np

from asn4sql import log
from asn4sql.parallel_train import SyncTrainer
from asn4sql import data
from asn4sql import wikisql_specific
from asn4sql.utils import (seed_all, gpus, get_device, RollingAverageWindow,
                           intfmt)

# dataset and initialization config
flags.DEFINE_integer('seed', 1, 'random seed')
flags.DEFINE_boolean('toy', False, 'use a toy dataset for debugging')

# training logistics
flags.DEFINE_string(
    'restore_checkpoint', None,
    'restore training from this checkpoint; the current best '
    'from the restored run must be in the same directory ')
flags.DEFINE_integer(
    'persist_every', 10,
    'period of epochs between checkpoint persists (0 to disable)')
flags.DEFINE_integer('max_epochs', 100,
                     'maximum number of epochs for training')
flags.DEFINE_integer(
    'workers', 4, 'number of CPU workers for parallelizing '
    'training in a data-parallel manner (we only ever use '
    'at most one GPU, but python-heavy processing can be '
    'parallelized. Use a single process if set to 0.')

# optimizer
flags.DEFINE_integer('batch_size', 64, 'batch size')
flags.DEFINE_float('learning_rate', 0.1, 'initial learning rate')


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
    training_state = _TrainingState()
    if flags.FLAGS.restore_checkpoint:
        _copy_best_checkpoint(flags.FLAGS.restore_checkpoint)
        _load_checkpoint(flags.FLAGS.restore_checkpoint, model, training_state)
    model = model.share_memory()

    num_workers = flags.FLAGS.workers
    log.debug('initializing {} workers', num_workers)
    with closing(SyncTrainer(model, num_workers)) as trainer:
        trainer.zero_grad()
        log.debug('all {} remote workers initialized', num_workers)
        _do_training(model, train, val, trainer, training_state)


def _do_training(model, train, val, trainer, training_state):
    batch_size = flags.FLAGS.batch_size

    loss_window = RollingAverageWindow(len(train) // 10 // batch_size)
    acc_window = RollingAverageWindow(len(train) // 10 // batch_size)
    grad_window = RollingAverageWindow(len(train) // 10 // batch_size)

    def _tqdm_postfix():
        return {
            'loss': '{:06.3f}'.format(loss_window.value()),
            'acc': '{:05.1%}'.format(acc_window.value()),
            'gradnorm': '{:08.2e}'.format(grad_window.value())
        }

    model.train()
    trainer.lr(training_state.lr)
    perm = np.arange(len(train))

    for epoch in range(1 + training_state.epoch, 1 + flags.FLAGS.max_epochs):
        epochfmt = intfmt(flags.FLAGS.max_epochs)
        training_state.epoch = epoch
        log.debug('begin epoch ' + epochfmt, epoch)
        # one sample at a time greatly simplifies pytorch seq2seq!
        np.random.shuffle(perm)

        samples = (train[i] for i in perm)
        with tqdm(total=len(train), postfix=_tqdm_postfix()) as progbar:
            for exs in _chunkify(samples, batch_size):
                loss, acc, gradnorm = trainer.train(exs)
                loss_window.update(loss)
                acc_window.update(acc)
                grad_window.update(gradnorm)
                trainer.step()  # auto-zeros grad
                progbar.update(len(exs))
                progbar.set_postfix(**_tqdm_postfix())

        model.eval()
        val_diagnostics = _diagnose(val, model)
        train_diagnositcs = _diagnose(train, model, len(val))
        val_diagnostics_str = _str_diagnostics('val', val_diagnostics)
        train_diagnositcs_str = _str_diagnostics('(sampled) train',
                                                 train_diagnositcs)
        log.debug('epoch ' + epochfmt + ' of ' + epochfmt + '\n{}\n{}', epoch,
                  flags.FLAGS.max_epochs, val_diagnostics_str,
                  train_diagnositcs_str)
        model.train()

        cur_val_loss = val_diagnostics['loss (*total)'][0]
        if cur_val_loss < training_state.best_val_loss:
            training_state.best_val_loss = cur_val_loss
            best_file = _checkpoint_file('best.pth')
            log.debug('updating best model into file {}', best_file)
            _save_checkpoint(best_file, model, training_state)
        else:
            # TODO: early stopping with patience
            # TODO: decay learning rate by 1/10
            pass

        if _check_period(epoch, flags.FLAGS.persist_every):
            checkpoint_file = _checkpoint_file(epochfmt.format(epoch) + '.pth')
            log.debug('persisting model to {}', checkpoint_file)
            _save_checkpoint(checkpoint_file, model, training_state)


def _diagnose(dataset, model, subsample=None):
    if subsample is None:
        subsample = range(len(dataset))
        num_items = len(dataset)
    else:
        subsample = np.random.choice(len(dataset), subsample, replace=False)
        num_items = len(subsample)
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
        k: (value / num_items, fmt)
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


def _checkpoint_file(basename):
    checkpoint_file = os.path.join(log.logging_directory(), 'checkpoints',
                                   basename)
    return checkpoint_file


def _copy_best_checkpoint(checkpoint_file):
    bestfile = os.path.join(os.path.dirname(checkpoint_file), 'best.pth')
    if not os.path.isfile(bestfile):
        raise ValueError(
            'was expecting checkpoint file {} to have a sibling '
            'file best.pth for the running best model'.format(checkpoint_file))
    best_dst = _checkpoint_file('best.pth')
    log.debug('copying best running model file from {} to {}', bestfile,
              best_dst)
    os.makedirs(os.path.dirname(best_dst), exist_ok=True)
    shutil.copyfile(bestfile, best_dst)


def _load_checkpoint(checkpoint_file, model, training_state):
    log.debug('restoring model from {}', flags.FLAGS.restore_checkpoint)
    state_dict = torch.load(checkpoint_file)
    model.load_state_dict(state_dict['model'])
    training_state.load_state_dict(state_dict['training_state'])


def _save_checkpoint(checkpoint_file, model, training_state):
    state_dict = {
        'model': model.state_dict(),
        'training_state': training_state.state_dict()
    }

    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
    with open(checkpoint_file, 'wb') as f:
        torch.save(state_dict, f)


class _TrainingState:
    def __init__(self):
        self.epoch = 0
        self.lr = flags.FLAGS.learning_rate
        self.best_val_loss = np.inf

    def state_dict(self):
        """return all training state for algorithm"""
        return self.__dict__

    def load_state_dict(self, d):
        """re-load training state from dictionary"""
        self.__dict__.update(d)


if __name__ == '__main__':
    app.run(_main)