"""
A static natural language embedding.
"""

import torch
from torch import nn

from ..data import wikisql


class NLEmbedding(nn.Module):
    """
    The column embedding accepts a description of the table header field
    from the WikiSQL torchtext dataset, which is assumed to already have
    built out a vocabulary, with a pretrained base already loaded.

    After the learnable embedding is initialized, this module defines
    weights and networks for the following operations:

    * Upon receiving a wikisql.SPLIT_WORD-delimited sequence of indices
      for words describing the variable number of columns in a table,
      this module splits by the split token and then applies a
      bidirectional LSTM to each column description individually.
    * The per-column final bidirectional LSTM state is used as an
      embedding for that specific column, resulting in a new sequence
      equal in length to the number of columns for the current table.
    * Finally, a unidirectional LSTM computes a cross-column embedding
      for the whole table.

    The sequence embedding size is in the sequence_size member,
    whereas final_size describes the embedding size for the final
    whole-table summary.
    """

    def __init__(self, src_field):
        super().__init__()
        vocab = src_field.vocab
        vecs = vocab.vectors
        self.lo, self.hi = _get_specials_range(vocab)
        self.embedding = nn.Embedding.from_pretrained(vecs, freeze=True)
        self.embedding_dim = vecs.size()[1]
        self.specials_embedding = nn.Embedding(self.hi - self.lo,
                                               self.embedding_dim)

    def forward(self, x):
        # a little awkward because we need to propogate grads to
        with torch.no_grad():
            nonspecial = torch.nonzero(1 - (self.lo <= x) * (x < self.hi))
        xx = x.clone()
        xx[nonspecial] = 0
        xx -= self.lo
        embed = self.specials_embedding(xx)
        embed[nonspecial] = self.embedding(x[nonspecial])
        return embed


def _get_specials_range(vocab):
    idxs = [vocab.stoi[w] for w in wikisql.SPECIALS]
    lo, hi = min(idxs), max(idxs) + 1
    if hi - lo != len(idxs):
        raise ValueError('special tokens must occupy a contiguous range '
                         'in the vocabulary')
    return lo, hi
