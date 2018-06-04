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

from asn4sql import datasets
from asn4sql import log
from asn4sql import sqlnet
from asn4sql.utils import seed_all, get_device, gpus

flags.DEFINE_integer('seed', 1, 'random seed')
flags.DEFINE_string(
    'treatment_name', 'sqlnet',
    'the name of the experimental treatment that this training'
    ' procedure will perform; this should be held the same '
    'across a set of seeds for a fixed set of parameter values'
    ' and may be used later by other python modules '
    'in creating plot legends')


def _main(argv):
    seed_all(flags.FLAGS.seed)
    log_subdir = os.path.join(_flags_hashstr(argv[0]), str(flags.FLAGS.seed))
    log.init(log_subdir)

    log.debug('downloading and reading pre-annotated wikisql data')
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

    datasets.wikisql(True)


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
