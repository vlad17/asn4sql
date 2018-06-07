"""
Validates the data.

Logs are written to {logroot}/{flag hash}/{seed}, where
the flag hash determines a unique experiment setup up to the seed.
The data in that directory is stored as:

{logroot}/{flag hash}/{seed}:
  * githash.txt - repository git hash at time of invocation
  * invocation.txt - command line invocation to run same experiment
  * flags.json - json version of invocation flags
  * flags.flags - absl version of invocation flags
  * log.txt - debug logs from training
"""

import os
import hashlib

from absl import app
from absl import flags
import torchtext.vocab as vocab

from asn4sql import log
from asn4sql import data
from asn4sql.utils import seed_all, get_device, gpus

flags.DEFINE_integer('seed', 1, 'random seed')
flags.DEFINE_boolean('toy', False, 'use a toy dataset for debugging')


def _main(argv):
    seed_all(flags.FLAGS.seed)
    log_subdir = log.flaghash_dirname([argv[0]], ['seed'])
    log.init(log_subdir)

    log.debug('downloading and reading pre-annotated wikisql data')
    if flags.FLAGS.toy:
        log.debug('using toy data subset')
    all_data = data.wikisql(flags.FLAGS.toy)
    _validate_data(all_data)


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
        rows = train_db.execute_query(i)
        print(indent, 'result:', rows)

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

    dim = 50 if flags.FLAGS.toy else 300
    glove = vocab.GloVe(name='6B', dim=dim)
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

    nc = sum(c in glove.stoi for c in conds)
    nq = sum(c in glove.stoi for c in question)
    nco = sum(c in glove.stoi for c in columns)
    print('words in glove of total words')
    print(indent, 'among conditionals {} of {}'.format(nc, len(conds)))
    print(indent, 'among questions    {} of {}'.format(nq, len(question)))
    print(indent, 'among columns      {} of {}'.format(nco, len(columns)))


if __name__ == '__main__':
    app.run(_main)
