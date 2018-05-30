"""
Prints "Hello, World!" to logs and stdout.

The spec can be supplied with command-line flags or a flagfile
(use --flagfile).

Logs are written to {logroot}/{experiment_name}/{seed}, where
the experiment name is by default a flag hash, after the seed value
has been removed
"""

import os

from absl import app
from absl import flags

from asn4sql import log
from asn4sql.utils import seed_all

flags.DEFINE_integer('seed', 1, 'random seed')

flags.DEFINE_string(
    'experiment_name', None,
    'experiment name, determines logging directory name (by default, '
    'this is a hash of the flags after the seed is removed)')


def _main(argv):
    seed_all(flags.FLAGS.seed)
    exp_name = flags.FLAGS.experiment_name
    if not exp_name:
        flags_dict = flags.FLAGS.flags_by_module_dict().copy()
        flags_dict = {
            k: {v.name: v.value
                for v in vs}
            for k, vs in flags_dict.items()
        }
        del flags_dict[argv[0]]['seed']
        flags_set = frozenset(
            (k, frozenset(v.items())) for k, v in flags_dict.items())
        print(flags_set)
        exp_name = str(abs(hash(flags_set)))
    log.init(os.path.join(exp_name, str(flags.FLAGS.seed)))
    log.debug('Hello, World!')


if __name__ == '__main__':
    app.run(_main)
