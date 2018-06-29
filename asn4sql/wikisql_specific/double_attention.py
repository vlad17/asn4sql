"""
Defines several modules for two-sequence attention. We are interested in
contextualizing both sequences with respect to one another.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .attention import Attention


class DoubleAttention(nn.Module):
    """
    Attend to the first sequence, then use the attention to inform attention
    to the second sequence.
    """

    def __init__(self, seq1_size, seq2_size, context_size):
        super().__init__()
        self.attn1 = Attention(seq1_size, context_size)
        self.attn2 = Attention(seq2_size, context_size + seq1_size)

    def forward(self, seq1_ae, seq2_be, context_c):
        """
        a = sequence 1 size
        b = sequence 2 size
        c = context size
        e = generic encoding dim

        returns tuple of attentions over both sequences
        """
        attended1_e = self.attn1(seq1_ae, context_c)
        attended2_e = self.attn2(seq2_be, torch.cat([context_c, attended1_e]))
        return attended1_e, attended2_e


class SymmetricDoubleAttention(nn.Module):
    """
    Apply double attention to both sequences
    """

    def __init__(self, seq1_size, seq2_size, context_size):
        super().__init__()
        self.attn1 = DoubleAttention(seq1_size, seq2_size, context_size)
        self.attn2 = DoubleAttention(seq2_size, seq1_size, context_size)

    def forward(self, seq1_ae, seq2_be, context_c):
        """
        a = sequence 1 size
        b = sequence 2 size
        c = context size
        e = generic encoding dim

        returns tuple of attentions over both sequences
        """
        _, attended2_e = self.attn1(seq1_ae, seq2_be, context_c)
        _, attended1_e = self.attn2(seq2_be, seq1_ae, context_c)

        return attended1_e, attended2_e


class OuterAttention(nn.Module):
    """
    Computes the "outer product" attention between two sequences, which
    computes a matrix of attention logits and "marginalizes" them.
    """

    def __init__(self, seq1_size, seq2_size, context_size):
        super().__init__()
        minsize = min(seq1_size, seq2_size)
        # can alternatively keep size and use inner attention when computing
        # matrix
        self.context1 = nn.Linear(context_size, minsize, bias=False)
        self.mlp1 = nn.Linear(seq1_size, minsize)
        self.context2 = nn.Linear(context_size, minsize, bias=False)
        self.mlp2 = nn.Linear(seq2_size, minsize)
        self.activation = F.elu

    def forward(self, seq1_ae, seq2_be, context_c):
        """
        a = sequence 1 size
        b = sequence 2 size
        c = context size
        m = min(a,b)
        e = generic encoding dim

        returns tuple of attentions over both sequences
        """
        context1_m = self.context1(context_c)
        mlp1_am = self.mlp1(seq1_ae)
        projected_seq1_am = self.activation(mlp1_am + context1_m.unsqueeze(0))
        context2_m = self.context2(context_c)
        mlp2_bm = self.mlp2(seq2_be)
        projected_seq2_bm = self.activation(mlp2_bm + context2_m.unsqueeze(0))

        # we can probably do this more efficiently
        matrix_ab = projected_seq1_am.matmul(projected_seq2_bm.t())
        attn_logits_a = matrix_ab.sum(1)
        attn_logits_b = matrix_ab.sum(0)

        att1 = seq1_ae.t().mv(F.softmax(attn_logits_a, dim=0))
        att2 = seq2_be.t().mv(F.softmax(attn_logits_b, dim=0))
        return att1, att2
