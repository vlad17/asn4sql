"""
The pre-initialized glove English word embedding,
code originally from SQLNet.
"""

import os
import json

import numpy as np

from .fetch import check_or_fetch
from .. import log


def load_glove():
    """loads the glove embedding"""
    url = 'http://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip'
    glove_dir = check_or_fetch('glove', 'glove.zip', url)
    log.debug('loading word embedding data from {}', glove_dir)
    with open(os.path.join(glove_dir, 'word2idx.json'), 'r') as f:
        # if lookups get slow consider a TST
        w2i = json.load(f)
    word_emb_val = np.load(os.path.join(glove_dir, 'usedwordemb.npy'))
    return w2i, word_emb_val
