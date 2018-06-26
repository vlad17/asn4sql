"""
Defines the conditional decoder for WikiSQL; see
WikiSQLSpecificModel for full docs.
"""

import functools

from absl import flags
import torch
from torch import nn
from torch.nn import functional as F

from .mlp import MLP
from .attention import Attention
from .pointer import Pointer
from ..data import wikisql
from ..utils import get_device

flags.DEFINE_integer('decoder_size', 128, 'hidden state size for the decoder')
flags.DEFINE_integer('op_embedding_size', 16,
                     'embedding size for conditional operators')


class ConditionDecoder(nn.Module):
    """nt_
    The decoder produces seq2seq-style output for the conditional
    columns in WikiSQL; it has quite a few moving parts:

                            new output
                                ^
                                |
    (decoder hidden state) -> LSTM -> (new hidden state)
                                ^
                                |
    previous output   -> prepared input

    Where the input and output encapsulate the index of the
    conditional operator for the conditional statement,
    the index of the column within the current table,
    the span across the question for the literal of the
    corresponding conditional statement, and whether
    to stop the decoding (all the conditional columns
    have been generated).
    """

    def __init__(self, col_seq_size, src_seq_size):
        super().__init__()
        num_words = len(wikisql.CONDITIONAL)
        self.op_embedding = nn.Embedding(num_words,
                                         flags.FLAGS.op_embedding_size)

        self.src_attn = Attention(src_seq_size, flags.FLAGS.decoder_size)
        self.col_attn = Attention(col_seq_size,
                                  flags.FLAGS.decoder_size + src_seq_size)

        input_size = flags.FLAGS.decoder_size
        self.stop_logits = MLP(input_size, [], 2)
        self.op_logits = MLP(input_size, [], len(wikisql.CONDITIONAL))
        input_size += self.op_logits.output_size
        self.col_ptr_logits = Pointer(col_seq_size, input_size)
        input_size += col_seq_size  # add an attention context
        self.span_l_ptr_logits = Pointer(src_seq_size, input_size)
        self.span_r_ptr_logits = Pointer(src_seq_size, input_size)

        decoder_input_size = (self.op_embedding.embedding_dim + col_seq_size +
                              col_seq_size + src_seq_size)

        self.decoder_lstm = nn.LSTM(
            decoder_input_size,
            flags.FLAGS.decoder_size,
            num_layers=1,
            bidirectional=False)

    @staticmethod
    def dummy_input(stop=False):
        """
        stop, op, col, span_l, span_r
        """
        stop = 1 if stop else 0
        return (stop, -1, -1, -1, -1)

    @staticmethod
    def _fetch_or_zero(seq, idx):
        _, e = seq.size()
        if idx < 0:
            return torch.zeros((e, ), dtype=seq.dtype, device=get_device())
        return seq[idx]

    @staticmethod
    def create_initial_state(initial_state_e):
        """Given a vector of size flags.FLAGS.decoder_size,
        creates the corresponding initial RNN state"""
        # just pytorch API massaging (need cells to be
        # initialized as well; not just hidden state)
        initial_decoder_state_11e = initial_state_e.view(1, 1, -1)
        initial_decoder_state = (initial_decoder_state_11e,
                                 torch.zeros_like(initial_decoder_state_11e))
        return initial_decoder_state

    def forward(self, hidden_state, decoder_input, sqe_qe, sce_ce):
        """
        e = some embedding dimension, varies tensor to tensor
        i = decoder input dimension
        q = question length
        c = num columns
        o = cardinality of binary comparison operators

        sqe is the sequence of question word embeddings
        sce is the sequence of column description embeddings

        returns the decoder output and next hidden state, where
        the decoder output is a tuple containing

        * whether the decoder should terminate (2 logits, 0 is don't
          stop, 1 is to stop on the current input, i.e., don't produce
          the values below).
        * predicted logits for the next operator
        * logits for the next column being conditioned on
        * logits for the beginning of the span of the question
          defining the literal
        * logits for the end of the span

        """
        stop, op, col, _span_l, _span_r = decoder_input
        if stop:
            ncols = len(sce_ce)
            nwords = len(sqe_qe)
            return self._stop_logits(ncols, nwords), hidden_state

        op_idx = self._optensor(op)
        op_e = self.op_embedding(op_idx)
        col_e = self._fetch_or_zero(sce_ce, col)
        # TODO figure out a good way of feeding span_l and span_r as inputs
        # to the decoder.

        # TODO add attention based on decoder hidden state over both
        # sequences.
        # TODO create separate method for decoding and hidden
        # updates to avoid dummy input
        # Double Attention as done in RobustFill
        hidden_cells_11e, _ = hidden_state
        attn_context = hidden_cells_11e.view(-1)
        attended_src = self.src_attn(sqe_qe, attn_context)
        attn_context = torch.cat([attn_context, attended_src])
        attended_col = self.col_attn(sce_ce, attn_context)
        decoder_input_11i = torch.cat(
            [op_e, col_e, attended_src, attended_col]).view(1, 1, -1)

        decoder_output_11e, hidden_state = self.decoder_lstm(
            decoder_input_11i, hidden_state)

        context = decoder_output_11e.view(-1)
        stop_2 = self.stop_logits(context)

        op_logits_o = self.op_logits(context)

        context = torch.cat([context, op_logits_o.detach()])
        col_logits_c = self.col_ptr_logits(sce_ce, context)

        # Attend to column inputs as context for question pointing
        col_attn_e = sce_ce.t().mv(F.softmax(col_logits_c, dim=0))
        context = torch.cat([context, col_attn_e])
        span_l_logits_q = self.span_l_ptr_logits(sqe_qe, context)

        with torch.no_grad():
            # r = number entries after l
            l = span_l_logits_q.argmax().detach().cpu().numpy()
            left_pad = [self._neg100()] * (l + 1)
        sqe_re = sqe_qe[l:]
        span_r_logits_r = self.span_r_ptr_logits(sqe_re, context)
        span_r_logits_q1 = torch.cat(left_pad + [span_r_logits_r])

        return (stop_2, op_logits_o, col_logits_c, span_l_logits_q,
                span_r_logits_q1), hidden_state

    @staticmethod
    def _zeros(*shape):
        return torch.zeros(shape, device=get_device())

    @functools.lru_cache(maxsize=None)
    def _stop_logits(self, ncols, nwords):
        stop_2 = self._zeros(2)
        stop_2[1] = 1
        op_logits_o = self._zeros(len(wikisql.CONDITIONAL))
        col_logits_c = self._zeros(ncols)
        span_l_logits_q = self._zeros(nwords)
        span_r_logits_q = self._zeros(nwords + 1)
        return (stop_2, op_logits_o, col_logits_c, span_l_logits_q,
                span_r_logits_q)

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def _neg100():
        return -100 * torch.ones([1], dtype=torch.float32, device=get_device())

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _optensor(op):
        x = torch.ones([], dtype=torch.long, device=get_device())
        return x * max(op, 0)
