"""
Trains a simple WikiSQL-specific architecture on the WikiSQL
dataset.

Assumes that the processed-toy(0|1).pth dataset exists in
{dataroot}/wikisql/ already.
"""

import os

from absl import app
from absl import flags
import torch
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

# optimizer
flags.DEFINE_float('learning_rate', 0.1, 'learning rate')


def _main(argv):
    log.debug('found gpus {}', gpus())

    seed_all(flags.FLAGS.seed)
    log_subdir = log.flaghash_dirname([argv[0]], ['seed'])
    log.init(log_subdir)

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

    _do_training(model, train, val)


def _do_training(model, train, val):
    device = get_device()
    torch.save(
        model.to(torch.device('cpu')),
        os.path.join(log.logging_directory(), 'untrained_model.pth'))
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=flags.FLAGS.learning_rate)
    # training_state = _TrainingState()
    # if flags.FLAGS.restore_checkpoint:
    #     log.debug('restoring model from {}', flags.FLAGS.restore_checkpoint)
    #     _load_checkpoint(flags.FLAGS.restore_checkpoint, model, optimizer,
    #                      training_state)
    # TODO(NOW) multiple losses?
    loss_window = RollingAverageWindow(len(train))

    model.train()

    perm = np.arange(len(train))
    for epoch in range(1, 1 + flags.FLAGS.max_epochs):
        epochfmt = intfmt(flags.FLAGS.max_epochs)
        log.debug('begin epoch ' + epochfmt, epoch)
        # one sample at a time greatly simplifies pytorch seq2seq!
        np.random.shuffle(perm)
        samples = (train[i] for i in perm)
        for ex in tqdm(samples, total=len(train)):
            # TODO batch size > 1 by not zeroing, see lbs/training.py
            optimizer.zero_grad()
            prepared_ex = model.prepare_example(ex)
            loss = model.forward(prepared_ex)
            loss_window.update(loss.detach().cpu().numpy())
            loss.backward()
            optimizer.step()

        log.debug('end epoch ' + epochfmt + ' rolling loss {:8.4g}', epoch,
                  loss_window.value())

        if _check_period(epoch, flags.FLAGS.evaluate_every):
            model.eval()
            val_diagnostics = _diagnose(val, model)
            train_diagnositcs = _diagnose(train, model)
            val_diagnostics = _str_diagnostics('val', val_diagnostics)
            train_diagnositcs = _str_diagnostics('train', train_diagnositcs)
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


def _diagnose(dataset, model):
    sum_diagnostics = {}
    with torch.no_grad():
        for ex in dataset:
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
    values = [(k,) + diagnostics[k] for k in sorted(diagnostics)]
    return preamble + newline_and_indent + newline_and_indent.join(
        (namefmt + ' ' + valuefmt).format(name, value)
        for name, value, valuefmt in values)


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
