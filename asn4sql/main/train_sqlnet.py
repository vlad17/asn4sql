"""
Trains my re-implementation of SQLNet.

Xiaojun Xu, Chang Liu, Dawn Song. 2017. SQLNet: Generating Structured Queries
from Natural Language Without Reinforcement Learning.

Original implementation available at
git@github.com:xiaojunxu/SQLNet.git

Logs are written to {logroot}/{flag hash}/{seed}, where
the flag hash determines a unique experiment setup up to the seed.
The data in that directory is stored as:

{logroot}/{flag hash}/{seed}:
  * githash.txt - repository git hash at time of invocation
  * invocation.txt - command line invocation to run same experiment
  * flags.json - json version of invocation flags
  * flags.flags - absl version of invocation flags
  * untrained_model.pth - raw version of the just the pre-training sqlnet model
  * log.txt - debug logs from training
"""

import os
import hashlib

from absl import app
from absl import flags
import torch

from asn4sql import log
from asn4sql import sqlnet
from asn4sql import data
from asn4sql.utils import seed_all, get_device, gpus

flags.DEFINE_integer('seed', 1, 'random seed')
flags.DEFINE_string(
    'treatment_name', 'sqlnet',
    'the name of the experimental treatment that this training'
    ' procedure will perform; this should be held the same '
    'across a set of seeds for a fixed set of parameter values'
    ' and may be used later by other python modules '
    'in creating plot legends')
flags.DEFINE_boolean(
    'validate_data', False, 'do not train anything; just do a couple data '
    'integrity checks')
flags.DEFINE_boolean('toy', False, 'use a toy dataset for debugging')


def _main(argv):
    seed_all(flags.FLAGS.seed)
    log_subdir = os.path.join(_flags_hashstr(argv[0]), str(flags.FLAGS.seed))
    log.init(log_subdir)

    log.debug('downloading and reading pre-annotated wikisql data')
    if flags.FLAGS.toy:
        log.debug('using toy data subset')
    all_data = data.wikisql(flags.FLAGS.toy)
    if flags.FLAGS.validate_data:
        _validate_data(all_data)
        return
    # from SQLNet, cache/dl data.tar.bz2 and corresponding rows

    log.debug('downloading and reading initial glove embeddings')
    # again per sqlnet

    log.debug('creating SQLNet model')
    model = sqlnet.SQLNet()

    log.debug('found gpus {}', gpus())

    savefile = os.path.join(log.logging_directory(), 'untrained_model.pth')
    log.debug('saving untrained model to {}', savefile)
    device = get_device()
    torch.save(model.to(torch.device('cpu')), savefile)
    model = model.to(device)


def _flags_hashstr(seed_module_name):
    # computes a hash from flags which determine a unique experiment
    # (which is all flags from this package minus the seed)
    flags_dict = flags.FLAGS.flags_by_module_dict().copy()
    flags_dict = {
        k: {v.name: v.value
            for v in vs}
        for k, vs in flags_dict.items()
        if k == seed_module_name or 'asn4sql' in k
    }
    del flags_dict[seed_module_name]['seed']
    flags_set = [(k, list(sorted(v.items())))
                 for k, v in sorted(flags_dict.items())]
    flags_str = str(flags_set).encode('utf-8')
    return hashlib.md5(flags_str).hexdigest()


def _validate_data(all_data):
    train_db, train_queries, val_db, val_queries, test_db, test_queries = (
        all_data)

    print('5 example training queries')
    indent = ' ' * 3
    for i in train_queries[-5:]:
        print()
        print(indent, data.Token.detokenize(i.question))
        print(indent, i.interpolated_query())
    print()

    trainqs = set(tq.table_id for tq in train_queries)
    valqs = set(tq.table_id for tq in val_queries)
    testqs = set(tq.table_id for tq in test_queries)
    print('distinct table counts across datasets')
    print(indent, 'num train tables  ', len(trainqs))
    print(indent, 'num val tables    ', len(valqs))
    print(indent, 'num test tables   ', len(testqs))
    print(indent, 'intersection in train/val', len(trainqs & valqs))
    print(indent, 'intersection in train/test', len(testqs & trainqs))
    print(indent, 'intersection in val/test', len(testqs & valqs))

    print('evaluating query ', train_queries[-1].interpolated_query())
    rows = train_db.execute_query(train_queries[-1])
    print('\n'.join(indent + ' ' + r for r in rows))

    word_to_idx, _embeding = {}, {} # data.load_glove()
    conds = [
        norm_str.token for q in train_queries for cond in q.conds
        for norm_str in cond.literal_toks(q.question)
    ]
    question = [
        norm_str.token for q in train_queries for norm_str in q.question
    ]
    columns = [
        norm_str.token for q in train_queries for c in q.column_descriptions
        for norm_str in c
    ]

    nc = sum(c in word_to_idx for c in conds)
    nq = sum(c in word_to_idx for c in question)
    nco = sum(c in word_to_idx for c in columns)
    print('words in glove of total words')
    print(indent, 'among conditionals {} of {}'.format(nc, len(conds)))
    print(indent, 'among questions    {} of {}'.format(nq, len(question)))
    print(indent, 'among columns      {} of {}'.format(nco, len(columns)))

    # TODO: unknown output/input strs, need LSTM, no?... why use embedding at
    # all? see coarse2fine oov

    # TODO: how to do batched rnn...? Maybe implement unbatched first.
    # TODO: don't use sqlnet. use COARSE TO FINE.... MAKE THE SWITCH.
    # https://github.com/donglixp/coarse2fine

    # todo: formally specify wikisql grammar (already done in validation...)
    # todo: TEST on reshuffles of wikisql data (i.e., coarse2fine assumes
    # sketches are from training set, can be missing some easily -- can even
    # check this manually here!)

    # 2-pass is not essential; classifier-based sketching is detrimental.
    # TODO: coarse2fine assumes literal *span* not just words suffices
    #       is this correct? Check manually.

    # TODO: if literals / string matches are not in the input, this can get
    # arbitrarily hard....

    # TODO: apply ASN to simplified wikisql grammar -- still should work! No
    # weaker ground, but no architecure specialization to the dataset either,
    # we are more moral. Don't need to use formal checker for this...

    # TODO: forward should compute loss directly (b/c of dynamicness)
    #       create separate method for query gen
    # TODO: print sample of stuff not in glove
    # TODO: wikisql probably sucks b/c glove is broken


if __name__ == '__main__':
    app.run(_main)
