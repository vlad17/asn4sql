"""
Preprocess and validate pytorch data.

Writes out the processed data to dataroot/wikisql/parsed_data.pth.
"""
import os

from absl import app
from absl import flags

from asn4sql import log
from asn4sql import data
from asn4sql.utils import seed_all, get_device, gpus

flags.DEFINE_integer('seed', 1, 'random seed')
flags.DEFINE_boolean('toy', False, 'use a toy dataset for debugging')


def _main(argv):
    seed_all(flags.FLAGS.seed)
    log_subdir = log.flaghash_dirname([argv[0]], ['seed'])
    log.init(log_subdir)

    if flags.FLAGS.toy:
        log.debug('using toy data subset')
    train, val, test = data.cached_fetch(
        os.path.join(
            'wikisql',
            'processed-toy{}.pth'.format(1 if flags.FLAGS.toy else 0)),
        _gen_data)
    _validate_data(train, val, test)


def _gen_data():
    log.debug('downloading and reading pre-annotated wikisql data')
    train, val, test = data.wikisql.wikisql(flags.FLAGS.toy)

    log.debug('building vocab')
    train.build_vocab(None, flags.FLAGS.toy, [val, test])
    return train, val, test


def _validate_data(train, val, test):
    print('5 example training queries')
    indent = ' ' * 3
    for i in train.examples[-5:]:
        print()
        print(indent, train.question(i))
        print(indent, train.db_engine.interpolated_query(i))
        rows = train.db_engine.execute_query(i)
        print(indent, 'result:', rows)

    print()
    trainqs = set(tq.table_id for tq in train.examples)
    valqs = set(tq.table_id for tq in val.examples)
    testqs = set(tq.table_id for tq in test.examples)
    print('distinct table counts across datasets')
    print(indent, 'num train tables  ', len(trainqs))
    print(indent, 'num val tables    ', len(valqs))
    print(indent, 'num test tables   ', len(testqs))
    print(indent, 'intersection in train/val', len(trainqs & valqs))
    print(indent, 'intersection in train/test', len(testqs & trainqs))
    print(indent, 'intersection in val/test', len(testqs & valqs))

    print()
    question = set(s for q in train.examples for s in q.src)
    columns = set(s for q in train.examples for s in q.tbl
                  if s != data.wikisql.SPLIT_WORD)
    vocab = data.wikisql.pretrained_vocab(flags.FLAGS.toy).stoi
    nq = sum(c in vocab for c in question)
    nc = sum(c in vocab for c in columns)
    print('words in glove of total words in train')
    fmt = '{:' + str(len(str(max(len(question), len(columns))))) + 'd}'
    print(indent,
          ('among questions    ' + fmt + ' of ' + fmt + ' {:.1%}').format(
              nq, len(question), nq / len(question)))
    print(indent,
          ('among columns      ' + fmt + ' of ' + fmt + ' {:.1%}').format(
              nc, len(columns), nc / len(columns)))

    print()
    print('field vocab sizes')
    for name, field in train.fields.items():
        if hasattr(field, 'vocab'):
            print(indent, name, len(field.vocab))


if __name__ == '__main__':
    app.run(_main)