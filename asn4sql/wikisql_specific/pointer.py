"""
Defines a general pointer network that accepts some context
on which to condition the pointing mechanism, which
uses "general" attention per the pointer network paper.

In other words, this performs a bilinear inner product over
the context for each item in the sequence; this gives the Luong
attention values that are then directly used as logits
over the sequence itself.
"""

import torch
from torch import nn

class Pointer(nn.Module):
    """
    Compute attention logits over a sequence of states of size
    sequence_size according to a context of size context_size
    with a bilinear form, creating a pointer network.
    """
    def __init__(self, sequence_size, context_size):
        super().__init__()
        self.inner = nn.Linear(context_size, sequence_size, bias=False)

    def forward(self, seq_se, context_c):
        """
        s = sequence length
        e = sequence embedding size
        c = context size

        seq_se - sequence to point to
        context_c - context to use for pointing

        returns logits over seq_se as a vector of length s
        """
        attn_weights_e = self.inner(context_c)
        return torch.mv(seq_se, attn_weights_e)
