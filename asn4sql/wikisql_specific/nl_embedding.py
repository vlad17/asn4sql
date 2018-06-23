"""
A static natural language embedding.
"""

from torch import nn

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
        vecs = src_field.vocab.vectors
        self.embedding = nn.Embedding(*vecs.size())
        self.embedding.weight.data.copy_(vecs)
        self.embedding.weight.requires_grad = False

    def forward(self, x):
        return self.embedding(x)
