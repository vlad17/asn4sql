"""
Defines the self-attention mechanism defined in the Attention Is All You Need
Paper.

I implemented this following along the annoted Transformer blog post
http://nlp.seas.harvard.edu/2018/04/03/attention.html

Note that since the input to the attention layers in our use case is already
defined by a bi-LSTM, we don't need to augment with any kind of positional
encoding, though that certainly is a possibility in the future (with
probably a different self-attention layer.
"""

from torch import nn
from torch.nn import functional as F

from .attention import Attention


class SeqAttention(nn.Module):
    """
    Compute the simultaneous attention for a set of queries
    with respect to a list of keys and average over the keys.
    """

    def __init__(self, key_dim, query_dim):
        super().__init__()
        self.query_dim = query_dim
        self.inner = nn.Linear(query_dim, key_dim, bias=False)

    def forward(self, keys_ke, queries_qd):
        """
        Compute self-attention wrt each key for every query

        q = num queries
        d = query dim
        k = num keys
        e = key dim

        returns a tensor of size qe, with attened-to keys for every
        query.
        """

        proj_qe = self.inner(queries_qd)
        attn_weights_qk = proj_qe.matmul(keys_ke.t())
        # TODO scale by /d_k or normalize
        attn_probs_qk = F.softmax(attn_weights_qk, dim=-1)
        vals_ke = attn_probs_qk.matmul(keys_ke)
        return vals_ke


class DoubleSeqAttention(nn.Module):
    """
    Use SeqAttention to attend to each sequence wrt the other, then attend to
    the resulting sequences.
    """

    def __init__(self, seq1_size, seq2_size, context_size):
        super().__init__()
        self.attn1 = SeqAttention(seq1_size, seq2_size)
        self.attn2 = SeqAttention(seq2_size, seq1_size)
        self.meta_attn1 = Attention(seq1_size, context_size)
        self.meta_attn2 = Attention(seq2_size, context_size)

    def forward(self, seq1_ae, seq2_be, context_c=None):
        """
        a = sequence 1 size
        b = sequence 2 size
        c = context size
        e = generic encoding dim

        returns tuple of attentions over both sequences
        """
        attended1_be = self.attn1(seq1_ae, seq2_be)
        attended2_ae = self.attn2(seq2_be, seq1_ae)

        attended1_e = self.meta_attn1(attended1_be, context_c)
        attended2_e = self.meta_attn2(attended2_ae, context_c)

        return attended1_e, attended2_e
