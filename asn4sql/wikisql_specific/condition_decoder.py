"""
Defines the conditional decoder for WikiSQL; see
WikiSQLSpecificModel for full docs.
"""

from absl import flags
import torch
from torch import nn
from torch.nn import functional as F

from .mlp import MLP
from ..data import wikisql

flags.DEFINE_integer('decoder_size', 512, 'hidden state size for the decoder')
flags.DEFINE_integer('op_embedding_size', 32,
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
    def __init__(self, col_seq_size, src_seq_size, op_field):
        num_words = op_field.vocab.vectors
        self.op_embedding = nn.Embedding(
            num_words, flags.FLAGS.op_embedding_size)

        decoder_input_size = (
            self.op_embedding.embedding_dim +
            col_seq_size +
            src_seq_size +
            src_seq_size)

        self.decoder_lstm = nn.LSTM(
            decoder_input_size, flags.FLAGS.decoder_size,
            num_layers=1, bidirectional=False)

        input_size = flags.FLAGS.decoder_size
        self.stop_logits = MLP(
            input_size, 2)
        input_size += self.stop_logits.output_size
        self.op_logits = MLP(
            input_size, len(wikisql.CONDITIONAL))
        input_size += self.op_logits.output_size
        self.col_ptr_logits = Pointer(col_seq_size, input_size)
        input_size += col_seq_size # add an attention context
        self.span_l_ptr_logits = Pointer(src_seq_size, input_size)
        input_size += src_seq_size # same as above
        self.span_r_ptr_logits = Pointer(src_seq_size, input_size)


    def dummy_input(stop=False):
        """
        stop, op, col, span_l, span_r
        """
        stop = 1 if stop else 0
        return (stop, -1, -1, -1, -1)

    def _fetch_or_zero(seq, idx):
        _, e = seq.size()
        if idx < 0:
            return torch.zeros((e,), dtype=seq.dtype)
        return seq[idx]

    def create_initial_state(self, initial_state_e):
        """Given a vector of size flags.FLAGS.decoder_size,
        creates the corresponding initial RNN state"""
        # just pytorch API massaging (need cells to be
        # initialized as well; not just hidden state)
        initial_decoder_state_11e.view(1, 1, -1)
        initial_decoder_state = (
            initial_decoder_state_11e,
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
        stop, op, col, span_l, span_r = decoder_input_i
        if stop:
            return self._stop_logits()

        op_e = self.op_embedding(op)
        col_e = self._fetch_or_zero(sce_ce, col)
        span_l_e = self._fetch_or_zero(sqe_qe, span_l)
        span_r_e = self._fetch_or_zero(sqe_qe, span_r)

        decoder_input_11i = torch.cat(
            [op_e, col_e, span_l_e, span_r_e)).view(1, 1, -1)

        decoder_output_11e, hidden_state = self.decoder_lstm(
            decoder_input_11i, hidden_state)

        context = decoder_output_11e.view(-1)
        stop_2 = self.stop_logits(context)

        context = torch.cat([context, stop_2])
        op_logits_o = self.op_logits(context)

        context = torch.cat([context, op_logits_o])
        col_logits_c = self.col_ptr_logits(sce_ce, context)

        col_attn_e = F.softmax(col_logits_c).dot(sce_ce)
        context = torch.cat([context, col_attn_e])
        span_l_logits_q = span_l_ptr_logits(sqe_qe, context)

        src_attn_e = F.softmax(src_logits_q).dot(sqe_qe)
        context = torch.cat([context, src_attn_e])
        span_r_logits_q = self.span_r_ptr_logits(sqe_qe, context)

        return (stop_2, op_logits_o, col_logits_c,
                span_l_logits_q, span_r_logits_q)
