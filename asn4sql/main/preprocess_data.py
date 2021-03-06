"""
Preprocess and validate pytorch data.

Writes out the processed data to dataroot/wikisql/processed-toy(0|1).pth
"""
import os
import random

from absl import app
from absl import flags
import track

from asn4sql import data
from asn4sql.utils import seed_all

flags.DEFINE_integer('seed', 1, 'random seed')
flags.DEFINE_boolean('toy', False, 'use a toy dataset for debugging')


def _print(s, *args, **kwargs):
    print(s.format(*args, **kwargs))


track.debug = _print


def _main(_argv):
    seed_all(flags.FLAGS.seed)

    if flags.FLAGS.toy:
        print('using toy data subset')
    train, val, test = data.cached_fetch(
        os.path.join(
            'wikisql',
            'processed-toy{}.pth'.format(1 if flags.FLAGS.toy else 0)),
        _gen_data)
    _validate_data(train, val, test)


def _gen_data():
    print('downloading and reading pre-annotated wikisql data')
    train, val, test = data.wikisql.wikisql(flags.FLAGS.toy)

    print('building vocab')
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

    vocab = data.wikisql.pretrained_vocab(flags.FLAGS.toy).stoi
    all_words = set()
    for dataset, dataset_name in [(train, 'train'), (val, 'val'), (test,
                                                                   'test')]:
        print()
        question = set(s for q in dataset.examples for s in q.src)
        columns = set(s for q in dataset.examples for s in q.tbl
                      if s != data.wikisql.SPLIT_WORD)
        all_words.update(question, columns)
        nq = sum(c in vocab for c in question)
        nc = sum(c in vocab for c in columns)
        print('words in glove of total words in', dataset_name)
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

    print()
    print('10 longest unknown words not in train')
    all_words = set(s for q in train.examples for s in q.src)
    all_words |= set(s for q in train.examples for s in q.tbl
                     if s != data.wikisql.SPLIT_WORD)
    unknown = [w for w in all_words if w not in vocab]
    unknown.sort(key=len, reverse=True)
    for i in unknown[:10]:
        print(' ' * 4 + i)
    print()
    print('and now 10 random ones')
    for i in random.sample(unknown, 10):
        print(' ' * 4 + i)

    nw = len(all_words)
    included = sum(w in vocab for w in all_words)
    print()
    print('total words in glove {} of {}, {:.1%}'.format(
        included, nw, included / nw))


if __name__ == '__main__':
    app.run(_main)
