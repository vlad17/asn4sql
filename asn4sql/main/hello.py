"""
Prints "Hello, World!" to logs and stdout.

The spec can be supplied with command-line flags or a flagfile
(use --flagfile).

Logs are written to {logroot}/{flag hash}/{seed}, where
the flag hash determines a unique experiment setup up to the seed.
"""

import os
import pickle
import hashlib

from absl import app
from absl import flags

from asn4sql import log
from asn4sql.utils import seed_all

flags.DEFINE_integer('seed', 1, 'random seed')


def _main(argv):
    seed_all(flags.FLAGS.seed)
    log_subdir = os.path.join(_flags_hashstr(argv[0]), str(flags.FLAGS.seed))
    log.init(log_subdir)
    log.debug('Hello, World!')
    # per sqlnet perform a learned embedding
    # replicate sqlnet performance
    # lifting wikisql assumptions to solve general sql synthesis problem
    # ---> TODO need to create a dataset
    # TODO need consistent md5 hashing in dir as well


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


if __name__ == '__main__':
    app.run(_main)
