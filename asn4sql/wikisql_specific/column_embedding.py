"""
Defines the column embedding for WikiSQL; see
WikiSQLSpecificModel for full docs.
"""

from absl import flags
import numpy as np
import torch
from torch import nn

from ..data import wikisql

flags.DEFINE_integer(
    'sequence_column_embedding_size', 64,
    'hidden state size for the per-column embedding; should '
    'be even')
flags.DEFINE_integer('column_embedding_size', 64,
                     'hidden state size for the whole-table column embedding')


class ColumnEmbedding(nn.Module):
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

    def __init__(self, tbl_field):
        super().__init__()
        self.split_idx = tbl_field.vocab.stoi[wikisql.SPLIT_WORD]
        vecs = tbl_field.vocab.vectors
        self.embedding = nn.Embedding(*vecs.size())
        self.embedding.weight.data.copy_(vecs)

        self.sequence_size = flags.FLAGS.sequence_column_embedding_size
        self.final_size = flags.FLAGS.column_embedding_size

        assert self.sequence_size % 2 == 0, self.sequence_size

        self.lstm = nn.LSTM(
            self.embedding.embedding_dim,
            self.sequence_size // 2,
            num_layers=1,
            bidirectional=True)
        self.summary_lstm = nn.LSTM(
            self.sequence_size,
            self.final_size,
            num_layers=1,
            bidirectional=False)

    def forward(self, seq_s):
        """
        seq_s should be the SPLIT_WORD numericalized LongTensor of the
        table header as prepared by torchtext.

        returns an embedding for each column and then a whole embedding over
        ever column
        """
        # s = flat sequence length with split index specifying columns
        # c = num columns
        # e = generic embedding index
        seq_se = self.embedding(seq_s)
        ends = torch.nonzero(seq_s == self.split_idx).detach().cpu().numpy()
        ends = ends.ravel()
        begins = np.roll(ends, 1)
        begins[0] = 0  # need at least 1 col
        cols_ce = []
        for begin, end in zip(begins, ends):
            # note the LSTM is reinitialized in terms of hidden state to 0
            # at the beginning of every column parse unlike the corresponding
            # embedding stage for coarse2fine. We do this because each column's
            # description is logically distinct.
            _, (final_hidden, _) = self.lstm(seq_se[begin:end].unsqueeze(1))
            cols_ce.append(final_hidden.view(-1))
        cols_ce = torch.stack(cols_ce)
        _, (final_hidden, _) = self.summary_lstm(cols_ce.unsqueeze(1))
        return cols_ce, final_hidden.view(-1)
