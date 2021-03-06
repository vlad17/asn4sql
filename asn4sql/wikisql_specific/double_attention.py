"""
Defines several modules for two-sequence attention. We are interested in
contextualizing both sequences with respect to one another.
"""

import math

from absl import flags
import torch
from torch import nn
from torch.nn import functional as F

from .attention import Attention
from .self_attention import DoubleSeqAttention

flags.DEFINE_enum(
    'multi_attn', 'indep',
    ['double', 'symm', 'outer1', 'indep', 'outer2', 'outer3', 'self'],
    'multi-sequence attention type')


def double_attention(*args, **kwargs):
    """Use the form of double attention selected by the flag"""
    if flags.FLAGS.multi_attn == 'double':
        double_attn = DoubleAttention
    elif flags.FLAGS.multi_attn == 'symm':
        double_attn = SymmetricDoubleAttention
    elif flags.FLAGS.multi_attn == 'indep':
        double_attn = IndependentAttention
    elif flags.FLAGS.multi_attn == 'outer1':
        double_attn = Outer1Attention
    elif flags.FLAGS.multi_attn == 'outer2':
        double_attn = Outer2Attention
    elif flags.FLAGS.multi_attn == 'outer3':
        double_attn = Outer3Attention
    elif flags.FLAGS.multi_attn == 'self':
        double_attn = DoubleSeqAttention
    else:
        raise ValueError('unsupported multi attention {}'
                         .format(flags.FLAGS.multi_attn))
    return double_attn(*args, **kwargs)


class DoubleAttention(nn.Module):
    """
    Attend to the first sequence, then use the attention to inform attention
    to the second sequence.
    """

    def __init__(self, seq1_size, seq2_size, context_size):
        super().__init__()
        self.attn1 = Attention(seq1_size, context_size)
        self.attn2 = Attention(seq2_size, context_size + seq1_size)

    def forward(self, seq1_ae, seq2_be, context_c=None):
        """
        a = sequence 1 size
        b = sequence 2 size
        c = context size
        e = generic encoding dim

        returns tuple of attentions over both sequences
        """
        attended1_e = self.attn1(seq1_ae, context_c)
        if context_c is None:
            context_c = attended1_e
        else:
            context_c = torch.cat([context_c, attended1_e])
        attended2_e = self.attn2(seq2_be, context_c)
        return attended1_e, attended2_e


class SymmetricDoubleAttention(nn.Module):
    """
    Apply double attention to both sequences
    """

    def __init__(self, seq1_size, seq2_size, context_size):
        super().__init__()
        self.attn1 = DoubleAttention(seq1_size, seq2_size, context_size)
        self.attn2 = DoubleAttention(seq2_size, seq1_size, context_size)

    def forward(self, seq1_ae, seq2_be, context_c=None):
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


class Outer1Attention(nn.Module):
    """
    Computes the "outer product" attention between two sequences, which
    computes a matrix of attention logits and "marginalizes" them.
    """

    def __init__(self, seq1_size, seq2_size, context_size):
        super().__init__()
        minsize = min(seq1_size, seq2_size)
        # can alternatively keep size and use inner attention when computing
        # matrix
        self.context_size = context_size
        if context_size:
            self.context1 = nn.Linear(context_size, minsize, bias=False)
        self.mlp1 = nn.Linear(seq1_size, minsize)
        if context_size:
            self.context2 = nn.Linear(context_size, minsize, bias=False)
        self.mlp2 = nn.Linear(seq2_size, minsize)
        self.activation = F.elu

    def forward(self, seq1_ae, seq2_be, context_c=None):
        """
        a = sequence 1 size
        b = sequence 2 size
        c = context size
        m = min(a,b)
        e = generic encoding dim

        returns tuple of attentions over both sequences
        """
        if self.context_size:
            context1_m = self.context1(context_c).unsqueeze(0)
        else:
            context1_m = 0
        mlp1_am = self.mlp1(seq1_ae)
        projected_seq1_am = self.activation(mlp1_am + context1_m)
        if self.context_size:
            context2_m = self.context2(context_c).unsqueeze(0)
        else:
            context2_m = 0
        mlp2_bm = self.mlp2(seq2_be)
        projected_seq2_bm = self.activation(mlp2_bm + context2_m)

        # we can probably do this more efficiently
        matrix_ab = projected_seq1_am.matmul(projected_seq2_bm.t())
        attn_logits_a = matrix_ab.sum(1)
        attn_logits_b = matrix_ab.sum(0)

        att1 = seq1_ae.t().mv(F.softmax(attn_logits_a, dim=0))
        att2 = seq2_be.t().mv(F.softmax(attn_logits_b, dim=0))
        return att1, att2


class IndependentAttention(nn.Module):
    """
    Attend to both sequences separately
    """

    def __init__(self, seq1_size, seq2_size, context_size):
        super().__init__()
        self.attn1 = Attention(seq1_size, context_size)
        self.attn2 = Attention(seq2_size, context_size)

    def forward(self, seq1_ae, seq2_be, context_c=None):
        """
        a = sequence 1 size
        b = sequence 2 size
        c = context size
        e = generic encoding dim

        returns tuple of attentions over both sequences
        """
        attended1_e = self.attn1(seq1_ae, context_c)
        attended2_e = self.attn2(seq2_be, context_c)
        return attended1_e, attended2_e


class Outer2Attention(nn.Module):
    """
    Computes the "outer product" attention between two sequences, which
    computes a matrix of attention logits and "marginalizes" them.
    But differently from the first outer.
    """

    def __init__(self, seq1_size, seq2_size, context_size):
        super().__init__()
        self.context_size = context_size
        if context_size:
            self.context1 = nn.Linear(context_size, seq1_size, bias=False)
        self.mlp1 = nn.Linear(seq1_size, seq1_size)
        if context_size:
            self.context2 = nn.Linear(context_size, seq2_size, bias=False)
        self.mlp2 = nn.Linear(seq2_size, seq2_size)
        self.activation = F.elu
        self.inner = nn.Linear(seq2_size, seq1_size, bias=False)

    def forward(self, seq1_ae, seq2_be, context_c=None):
        """
        a = sequence 1 size
        b = sequence 2 size
        c = context size
        e = generic encoding dim

        returns tuple of attentions over both sequences
        """
        if self.context_size:
            context1_e = self.context1(context_c).unsqueeze(0)
        else:
            context1_e = 0
        mlp1_ae = self.mlp1(seq1_ae)
        projected_seq1_ae = self.activation(mlp1_ae + context1_e)
        if self.context_size:
            context2_e = self.context2(context_c).unsqueeze(0)
        else:
            context2_e = 0
        mlp2_be = self.mlp2(seq2_be)
        projected_seq2_be = self.activation(mlp2_be + context2_e)

        # we can probably do this more efficiently (bilinear?)
        inner_be = self.inner(projected_seq2_be)
        matrix_ab = projected_seq1_ae.matmul(inner_be.t())
        attn_logits_a = matrix_ab.sum(1)
        attn_logits_b = matrix_ab.sum(0)

        att1 = seq1_ae.t().mv(F.softmax(attn_logits_a, dim=0))
        att2 = seq2_be.t().mv(F.softmax(attn_logits_b, dim=0))
        return att1, att2


class Outer3Attention(nn.Module):
    """
    Computes the "outer product" attention between two sequences, which
    computes a matrix of attention logits and "marginalizes" them.
    But differently from the outer1 and outer2: uses a bilinear product.
    """

    def __init__(self, seq1_size, seq2_size, context_size):
        super().__init__()
        self.seq1_size = seq1_size
        self.seq2_size = seq2_size
        self.context_size = max(context_size, 1)
        self.bilinear = nn.parameter.Parameter(
            torch.Tensor(self.context_size, seq1_size, seq2_size))
        self.reset_parameters()

    def reset_parameters(self):
        """initialize the weight with pytorch's usual rule"""
        stdv = 1. / math.sqrt(self.bilinear.size(1))
        self.bilinear.data.uniform_(-stdv, stdv)

    def forward(self, seq1_ax, seq2_by, context_c=None):
        """
        a = sequence 1 len (embedding size is x)
        b = sequence 2 len (embedding size is y)
        c = context size

        returns tuple of attentions over both sequences
        """
        bilinear_cxy = self.bilinear
        prod_cxb = bilinear_cxy.matmul(seq2_by.t())
        prod_cbx = prod_cxb.transpose(1, 2)
        outer_cba = prod_cbx.matmul(seq1_ax.t())
        outer_abc = outer_cba.transpose(0, 2)
        if context_c is None:
            assert self.context_size <= 1, self.context_size
            outer_ab = outer_abc.squeeze(2)
        else:
            outer_ab = outer_abc.matmul(context_c)
            assert len(outer_ab.size()) == 2, outer_ab.size()

        attn_logits_a = outer_ab.sum(1)
        attn_logits_b = outer_ab.sum(0)

        att1_x = seq1_ax.t().mv(F.softmax(attn_logits_a, dim=0))
        att2_y = seq2_by.t().mv(F.softmax(attn_logits_b, dim=0))
        return att1_x, att2_y
