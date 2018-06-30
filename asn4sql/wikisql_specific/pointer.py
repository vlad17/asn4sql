"""
Defines a general pointer network that accepts some context
on which to condition the pointing mechanism, which
uses "general" attention per the pointer network paper.

In other words, this performs a bilinear inner product over
the context for each item in the sequence; this gives the Luong
attention values that are then directly used as logits
over the sequence itself.
"""

import math

from absl import flags
import torch
from torch import nn
import numpy as np

flags.DEFINE_boolean('normalized_attention', False,
                     'whether to normalize attention vectors')


class Pointer(nn.Module):
    """
    Compute attention logits over a sequence of states of size
    sequence_size according to a context of size context_size
    with a bilinear form, creating a pointer network.
    """

    def __init__(self, sequence_size, context_size):
        super().__init__()
        self.context_size = context_size
        if not context_size:
            self.weights = nn.parameter.Parameter(torch.Tensor(sequence_size))
            stdv = 1. / math.sqrt(sequence_size)
            self.weights.data.uniform_(-stdv, stdv)
        else:
            self.inner = nn.Linear(context_size, sequence_size, bias=False)
        self.normalized_attention = flags.FLAGS.normalized_attention

    def forward(self, seq_se, context_c=None):
        """
        s = sequence length
        e = sequence embedding size
        c = context size

        seq_se - sequence to point to
        context_c - context to use for pointing

        returns logits over seq_se as a vector of length s
        """
        if self.context_size:
            attn_weights_e = self.inner(context_c)
        else:
            attn_weights_e = self.weights
        # two random symmetric unit vectors will have dot product magnitude
        # 1/dim in expectation (compute w/ a trace trick). Therefore,
        # we normalize embeddings (similar to scaled dot-product attn in
        # attention is all you need)
        if self.normalized_attention:
            attn_weights_e = attn_weights_e / torch.norm(attn_weights_e)
            attn_weights_e = attn_weights_e * np.sqrt(len(attn_weights_e))
        return torch.mv(seq_se, attn_weights_e)
