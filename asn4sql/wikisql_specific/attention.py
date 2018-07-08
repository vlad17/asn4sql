"""
Attention over a single sequence given a context vector.
"""

from absl import flags
import torch
from torch import nn
from torch.nn import functional as F

from .pointer import Pointer

flags.DEFINE_boolean(
    'detach_attn_input', False, 'if activated, do not '
    'permit any gradients to flow through the attention '
    'mechanism into the decoder')


class Attention(nn.Module):
    """
    Attend to a part of a sequence with Bahdanau general attention.
    """

    def __init__(self, sequence_size, context_size):
        super().__init__()
        self.pointer = Pointer(sequence_size, context_size)
        self.detach_attn_input = flags.FLAGS.detach_attn_input

    def forward(self, seq_se, context_c=None):
        """
        s = sequence length
        e = sequence embedding size
        c = context size

        seq_se - sequence to point to
        context_c - context to use for pointing

        returns attended_e, a vector of the sequence embedding size
        dimension, averaged with attention weights.
        """
        if self.detach_attn_input:
            seq_se = seq_se.detach()
        attn_logits_s = self.pointer(seq_se, context_c)
        attn_distribution_s = F.softmax(attn_logits_s, dim=0)
        return torch.mv(seq_se.t(), attn_distribution_s)
