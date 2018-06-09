"""
Runs coarse2fine on the WikiSQL dataset.

Assumes that the processed-toy(0|1).pth dataset exists in
{dataroot}/wikisql/ already.
"""

import os

from absl import app
from absl import flags
import torch

from asn4sql import log
from asn4sql import data
from asn4sql import coarse2fine
from asn4sql.utils import seed_all, gpus, get_device

flags.DEFINE_integer('seed', 1, 'random seed')
flags.DEFINE_boolean('toy', False, 'use a toy dataset for debugging')

# flags.DEFINE_string('restore_checkpoint', None, 'checkpoint to restore '
#      'training from')
# flags.DEFINE_integer(
#     'persist_every', 25,
#     'period of mini-batches between evaluations (0 to disable)')
# flags.DEFINE_integer('eval_batches', 100,
#                      'number of mini-batches to use to print diagnostic info')

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
    model = coarse2fine.build_model(train.fields)
    log.debug('built model:\n{}', model)
    model = model.to(get_device())
    # TODO use my setup for optimization / training instead with
    # lbs/training.py style checkpoint, eval, persist, eval_batches

    optim = coarse2fine.Optim()

    coarse2fine.do_training(model, optim, train, val)

if __name__ == '__main__':
    app.run(_main)
