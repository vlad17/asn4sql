"""
Training procedure for coarse2fine.
"""

from absl import flags
import torchtext.data

from .. import log
from ..utils import get_device
from . import Loss
from .Trainer import Trainer, Statistics

flags.DEFINE_integer('batch_size', 32, 'mini-batch size')
flags.DEFINE_integer(
    'evaluate_every', 50,
    'period of mini-batches between evaluations (0 to disable)')
flags.DEFINE_integer('epochs', 40, 'number of training epochs')

def report_func(epoch, batch, num_batches,
                start_time, lr, report_stats):
    """
    This is the user-defined batch-level traing progress
    report function.

    Args:
        epoch(int): current epoch count.
        batch(int): current batch count.
        num_batches(int): total number of batches.
        start_time(float): last report time.
        lr(float): current learning rate.
        report_stats(Statistics): old Statistics instance.
    Returns:
        report_stats(Statistics): updated Statistics instance.
    """
    if flags.FLAGS.evaluate_every and (
            batch % flags.FLAGS.evaluate_every ==
            -1 % flags.FLAGS.evaluate_every):
        report_stats.output(epoch, batch + 1, num_batches, start_time)
        report_stats = Statistics(0, {})

    return report_stats

def _sort_key(ex):
    s = len(ex.src) + len(ex.ent) + len(ex.tbl) + len(ex.cond_op)
    s += len(ex.cond_col) + len(ex.cond_span_l) + len(ex.cond_span_r)
    # TODO filter out original and after somehow from the examples
    s += len(ex.original) + len(ex.after)
    return s

def do_training(model, optim, train, val):
    """run the training procedure on the given model and optimizer"""
    train_iter = torchtext.data.Iterator(
        dataset=train, batch_size=flags.FLAGS.batch_size, device=get_device(),
        train=True, sort_within_batch=True, sort_key=_sort_key)
    valid_iter = torchtext.data.Iterator(
        dataset=val, batch_size=flags.FLAGS.batch_size, device=get_device(),
        train=False, sort_key=_sort_key)

    # apparently the aggregation learns quickly so coarse2fine reduces training
    # iterations there by half
    agg_sample_rate = 0.5
    smooth_eps = 0 # label smoothing
    train_loss = Loss.TableLossCompute(agg_sample_rate, smooth_eps).cuda()
    valid_loss = Loss.TableLossCompute(agg_sample_rate, smooth_eps).cuda()

    trainer = Trainer(model, train_iter, valid_iter,
                      train_loss, valid_loss, optim)

    word_vec_update_delay = 10 # epochs to wait before updating the embedding
    for epoch in range(1, flags.FLAGS.epochs + 1):
        log.debug('-------------------- EPOCH {:' +
                  str(len(str(flags.FLAGS.epochs))) +
                  '} --------------------', epoch)

        if epoch >= word_vec_update_delay:
            model.q_encoder.embeddings.set_update(True)

        # 1. Train for one epoch on the training set.
        train_stats = trainer.train(epoch, report_func)
        log.debug('    train accuracy {}', train_stats.accuracy(True))

        # 2. Validate on the validation set.
        valid_stats = trainer.validate()
        print('    val accuracy    {}' % valid_stats.accuracy(True))

        # 4. Update the learning rate
        trainer.epoch_step(None, epoch)

        # 5. Drop a checkpoint if needed.
        # TODO
