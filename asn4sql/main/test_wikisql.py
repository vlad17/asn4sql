"""
Tests a WikiSQL model.

Assumes that the processed-toy(0|1).pth dataset exists in
{dataroot}/wikisql/ already.
"""

from contextlib import closing
import os

from absl import app
from absl import flags
import torch
from tqdm import tqdm

from asn4sql import data
from asn4sql.shared_gpu import SharedGPU
from asn4sql.utils import (seed_all, get_device, gpus, chunkify,
                           disable_source_code_warning, OnlineSampler)

flags.DEFINE_integer('seed', 1, 'random seed')
flags.DEFINE_boolean('toy', False, 'use a toy dataset for debugging')

flags.DEFINE_string(
    'model', None, 'the checkpoint file which should contain '
    "a dictionary with a 'model' key. We assume that the "
    'parent directory of the model file contains a persisted '
    'pytorch module called untrained_model.pth which can be '
    'updated with the model parameters')
flags.mark_flag_as_required('model')
flags.DEFINE_integer(
    'batch_size', 512, 'batch size for evaluation. This is '
    'the batch size used on each of the individual workers.')
flags.DEFINE_integer(
    'workers', 4, 'number of CPU workers for parallelizing '
    'training in a data-parallel manner (we only ever use '
    'at most one GPU, but python-heavy processing can be '
    'parallelized. Use a single process if set to 0.')


def _main(_):
    seed_all(flags.FLAGS.seed)

    if flags.FLAGS.toy:
        print('using toy data subset')

    print('found gpus {}'.format(gpus()))
    dataset_file = os.path.join(
        flags.FLAGS.dataroot, 'wikisql',
        'processed-toy{}.pth'.format(1 if flags.FLAGS.toy else 0))
    print('loading data from {}'.format(dataset_file))
    train, val, test = torch.load(dataset_file)

    model_file = flags.FLAGS.model
    initial_model = os.path.join(
        os.path.dirname(os.path.dirname(model_file)), 'untrained_model.pth')

    print('loading initial model from {}'.format(initial_model))
    disable_source_code_warning()
    model = torch.load(initial_model)
    model = model.to(get_device())

    print('loading model parameters from {}'.format(model_file))
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict['model'])
    model = model.share_memory()

    num_workers = flags.FLAGS.workers
    print('initializing {} workers'.format(num_workers))
    with closing(SharedGPU(model, num_workers)) as shared:
        shared.set_mode(evaluation=True)
        print('all {} remote workers initialized'.format(num_workers))
        _do_evaluation('test', test, shared)


def _do_evaluation(dataset_name, dataset, shared):
    batch_size = flags.FLAGS.batch_size * max(flags.FLAGS.workers, 1)
    sum_diagnostics = {
        'agg match': 0,
        'sel match': 0,
        'cond exact match': 0,
        'cond logical match': 0,
        'exact match': 0,
        'logical match': 0,
        'execution match': 0,
    }

    mistakes = OnlineSampler(5)

    for exs in chunkify(tqdm(dataset), batch_size):
        diagnostics = shared.diagnose(exs)
        for true_ex, result_ex in zip(exs, diagnostics):
            agg_match = result_ex['acc (agg)'][0]
            sum_diagnostics['agg match'] += agg_match

            sel_match = result_ex['acc (sel)'][0]
            sum_diagnostics['sel match'] += sel_match

            prediction = result_ex['prediction']
            logical_cond_match = prediction.condition_logical_match(true_ex)
            sum_diagnostics['cond logical match'] += logical_cond_match

            logical_match = agg_match and sel_match and logical_cond_match
            sum_diagnostics['logical match'] += logical_match

            sum_diagnostics['cond exact match'] += result_ex['acc (cond)'][0]
            exact_match = result_ex['acc (*total)'][0]
            sum_diagnostics['exact match'] += exact_match

            pred_ex = prediction.as_query_example(true_ex)
            true_result = dataset.db_engine.execute_query(true_ex)
            pred_result = dataset.db_engine.execute_query(pred_ex)

            sum_diagnostics['execution match'] += true_result == pred_result

            if not exact_match:
                mistakes.update((true_ex, pred_ex))
    avg_diagnostics = {
        k: value / len(dataset)
        for k, value in sum_diagnostics.items()
    }

    print(dataset_name)
    maxlen = max(map(len, avg_diagnostics))
    fmt = '    {:<' + str(maxlen) + '} {:8.2%}'
    for k, v in sorted(avg_diagnostics.items()):
        print(fmt.format(k, v))

    print('    sampled mistakes')
    for true, pred in mistakes.sample:
        print(' ' * 8 + 'true ' + dataset.db_engine.interpolated_query(true))
        print(' ' * 8 + 'pred ' + dataset.db_engine.interpolated_query(pred))


if __name__ == '__main__':
    app.run(_main)
