"""
Trains a simple WikiSQL-specific architecture on the WikiSQL
dataset.

Assumes that the processed-toy(0|1).pth dataset exists in
{dataroot}/wikisql/ already.
"""

from contextlib import closing
from collections import defaultdict
import tempfile
import os
import shutil

from absl import app
from absl import flags
import track
import torch
from torch import optim
import torch.utils
from tqdm import tqdm
import numpy as np

from asn4sql.shared_gpu import SharedGPU
from asn4sql import data
from asn4sql import wikisql_specific
from asn4sql.utils import (seed_all, gpus, get_device, RollingAverageWindow,
                           intfmt, chunkify)

# dataset and initialization config
flags.DEFINE_integer('seed', 1, 'random seed')
flags.DEFINE_boolean('toy', False, 'use a toy dataset for debugging')
flags.DEFINE_string('trial_prefix', '', 'track.Trial name prefix')

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
flags.DEFINE_integer(
    'patience', 3, 'number of consecutive epochs of '
    'a lack of improvement in validation loss to tolerate '
    'before reducing the LR (on next unimproving epoch)')
flags.DEFINE_float('learning_rate', 0.1, 'initial learning rate')
flags.DEFINE_float(
    'lr_decay_rate', 0.5, 'decay rate for learning rate, '
    'used when validation loss stops improving')
flags.DEFINE_float('min_lr', 1e-5, 'stop training when lr gets this low')
flags.DEFINE_enum('optimizer', 'sgd', ['sgd', 'adam'],
                  'optimization algorithm')


def _main(_):
    with track.trial(
            param_map=track.absl_flags(),
            trial_prefix=flags.FLAGS.trial_prefix):
        seed_all(flags.FLAGS.seed)
        track.debug('found gpus {}', gpus())

        dataset_file = os.path.join(
            flags.FLAGS.dataroot, 'wikisql',
            'processed-toy{}.pth'.format(1 if flags.FLAGS.toy else 0))
        track.debug('loading data from {}', dataset_file)
        train, val, _ = torch.load(dataset_file)

        track.debug('building model')
        model = wikisql_specific.WikiSQLSpecificModel(train.fields)
        track.debug('built model:\n{}', model)
        num_parameters = int(
            sum(p.numel() for p in model.parameters() if p.requires_grad))
        track.debug('number of parameters in model {}', num_parameters)

        device = get_device()
        torch.save(
            model.to(torch.device('cpu')),
            os.path.join(track.trial_dir(), 'untrained_model.pth'))
        model = model.to(device)
        training_state = _TrainingState()
        if flags.FLAGS.restore_checkpoint:
            _copy_best_checkpoint(flags.FLAGS.restore_checkpoint)
            _load_checkpoint(flags.FLAGS.restore_checkpoint, model,
                             training_state)
        params_to_optimize = [p for p in model.parameters() if p.requires_grad]
        if flags.FLAGS.optimizer == 'sgd':
            # lr required here but will be set in _do_training
            optimizer = optim.SGD(params_to_optimize, lr=1)
        elif flags.FLAGS.optimizer == 'adam':
            optimizer = optim.Adam(params_to_optimize)
        else:
            raise ValueError('unrecognized optimizer {}'.format(
                flags.FLAGS.optimizer))

        num_workers = flags.FLAGS.workers
        track.debug('initializing {} workers', num_workers)
        with closing(SharedGPU(optimizer, model, num_workers)) as shared:
            _do_training(train, val, shared, training_state)


def _do_training(train, val, shared, training_state):
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

    shared.set_mode(evaluation=False)
    shared.lr(training_state.lr)
    perm = np.arange(len(train))

    for epoch in range(1 + training_state.epoch, 1 + flags.FLAGS.max_epochs):
        epochfmt = intfmt(flags.FLAGS.max_epochs)
        training_state.epoch = epoch
        track.debug('begin epoch ' + epochfmt, epoch)
        # one sample at a time greatly simplifies pytorch seq2seq!
        np.random.shuffle(perm)

        samples = (train[i] for i in perm)
        with tqdm(total=len(train), postfix=_tqdm_postfix()) as progbar:
            for exs in chunkify(samples, batch_size):
                shared.zero_grad()
                loss, acc, gradnorm = shared.train(exs)
                loss_window.update(loss)
                acc_window.update(acc)
                grad_window.update(gradnorm)
                shared.step()
                progbar.update(len(exs))
                progbar.set_postfix(**_tqdm_postfix())

        shared.set_mode(evaluation=True)
        val_diagnostics = _diagnose(val, shared)
        train_diagnostics = _diagnose(train, shared, len(val))
        track.metric(iteration=epoch, lr=training_state.lr)
        track.metric(
            iteration=epoch,
            **{'val ' + k: v
               for k, v in val_diagnostics.items()})
        track.metric(
            iteration=epoch,
            **{'train ' + k: v
               for k, v in train_diagnostics.items()})
        shared.set_mode(evaluation=False)
        val_diagnostics_str = _str_diagnostics('val', val_diagnostics)
        train_diagnositcs_str = _str_diagnostics('(sampled) train',
                                                 train_diagnostics)
        track.debug('epoch ' + epochfmt + ' of ' + epochfmt + '\n{}\n{}',
                    epoch, flags.FLAGS.max_epochs, val_diagnostics_str,
                    train_diagnositcs_str)

        cur_val_loss = val_diagnostics['loss (*total)']
        if cur_val_loss < training_state.best_val_loss:
            training_state.patience = training_state.initial_patience
            training_state.best_val_loss = cur_val_loss
            best_file = _checkpoint_file('best.pth')
            track.debug('updating best model into file {}', best_file)
            _save_checkpoint(best_file, shared.model, training_state)
        else:
            training_state.patience -= 1
            track.debug('val loss not improving; dropping patience')
            shared.lr(training_state.lr)

        if training_state.patience == 0:
            track.debug('out of patience, dropping lr')
            training_state.lr *= flags.FLAGS.lr_decay_rate
            training_state.patience = training_state.initial_patience

        track.debug('lr {} patience {} best val loss so far {}',
                    training_state.lr, training_state.patience,
                    training_state.best_val_loss)

        early_stop = training_state.lr < flags.FLAGS.min_lr
        if early_stop:
            track.debug(
                'lr dropped to {} < min tolerable lr {}, early stopping',
                training_state.lr, flags.FLAGS.min_lr)

        if _check_period(epoch, flags.FLAGS.persist_every) or early_stop:
            epochfmt = intfmt(flags.FLAGS.max_epochs, fill='0')
            checkpoint_file = _checkpoint_file(epochfmt.format(epoch) + '.pth')
            track.debug('persisting model to {}', checkpoint_file)
            _save_checkpoint(checkpoint_file, shared.model, training_state)

        if early_stop:
            break


def _diagnose(dataset, shared, subsample=None):
    if subsample is None:
        subsample = range(len(dataset))
        num_items = len(dataset)
    else:
        subsample = np.random.choice(len(dataset), subsample, replace=False)
        num_items = len(subsample)
    sum_diagnostics = defaultdict(float)
    samples = (dataset[i] for i in subsample)
    # can afford larger batch size for eval
    batch_size = flags.FLAGS.batch_size * max(flags.FLAGS.workers, 1)
    for exs in chunkify(samples, batch_size):
        diagnostics = shared.diagnose(exs)
        for ex in diagnostics:
            del ex['prediction']
            for k, value in ex.items():
                sum_diagnostics[k] += value
    avg_diagnostics = {
        k: value / num_items
        for k, value in sum_diagnostics.items()
    }
    return avg_diagnostics


def _str_diagnostics(diagnostics_name, diagnostics):
    preamble = '  ' + diagnostics_name
    if not diagnostics:
        return preamble
    newline_and_indent = '\n    '
    maxlen = max(map(len, diagnostics))
    namefmt = '{:<' + str(maxlen) + '}'
    values = [(k, diagnostics[k]) for k in sorted(diagnostics)]
    lossfmt = '{:8.4g}'
    accfmt = '{:8.2%}'
    return preamble + newline_and_indent + newline_and_indent.join(
        (namefmt + ' ' +
         (lossfmt if 'loss' in name else accfmt)).format(name, value)
        for name, value in values)


def _check_period(idx, period):
    if period == 0:
        return False
    return idx == 1 or idx == flags.FLAGS.max_epochs or idx % period == 0


def _checkpoint_file(basename):
    checkpoint_file = os.path.join(track.trial_dir(), 'checkpoints', basename)
    return checkpoint_file


def _copy_best_checkpoint(checkpoint_file):
    bestfile = os.path.join(os.path.dirname(checkpoint_file), 'best.pth')
    if not os.path.isfile(bestfile):
        raise ValueError(
            'was expecting checkpoint file {} to have a sibling '
            'file best.pth for the running best model'.format(checkpoint_file))
    best_dst = _checkpoint_file('best.pth')
    track.debug('copying best running model file from {} to {}', bestfile,
                best_dst)
    os.makedirs(os.path.dirname(best_dst), exist_ok=True)
    shutil.copyfile(bestfile, best_dst)


def _load_checkpoint(checkpoint_file, model, training_state):
    track.debug('restoring model from {}', flags.FLAGS.restore_checkpoint)
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
        self.initial_patience = flags.FLAGS.patience
        self.patience = self.initial_patience

    def state_dict(self):
        """return all training state for algorithm"""
        return self.__dict__

    def load_state_dict(self, d):
        """re-load training state from dictionary"""
        self.__dict__.update(d)


if __name__ == '__main__':
    app.run(_main)
