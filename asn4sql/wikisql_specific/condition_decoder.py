"""
Defines the conditional decoder for WikiSQL; see
WikiSQLSpecificModel for full docs.
"""

import functools

from absl import flags
import torch
from torch import nn

from .mlp import MLP
from .wikisql_input_encoding import WikiSQLInputEncoding
from .pointer import Pointer
from .double_attention import double_attention, IndependentAttention
from ..data import wikisql
from ..utils import get_device

flags.DEFINE_integer('decoder_size', 64, 'hidden state size for the decoder')
flags.DEFINE_integer('op_embedding_size', 16,
                     'embedding size for conditional operators')
flags.DEFINE_boolean(
    'tie_encodings', False, 'whether to use the same sequence'
    ' encoder weights for all four decoding stages, the '
    'column prediction, operator prediction, literal '
    'prediction, and stop prediction')

# pytorch convenience helpers


def _double_attn(*args):
    return _Sequential(_Unwrap(double_attention(*args)), _Cat())


class _Unwrap(nn.Module):
    # nn.Sequential only allows one argument for submodules, this lets us
    # pass multiple args with tuples
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(*x)


class _Sequential(nn.Sequential):
    # nn.Sequential only allows one argument, this lets us
    # create nn.Sequential instances which accept multiple
    def forward(self, *args):
        return super().forward(args)


class _Cat(nn.Module):
    def forward(self, x):
        return torch.cat(x)


class _InitDecoderState(nn.Module):
    def forward(self, x):
        # just pytorch API massaging (need cells to be
        # initialized as well; not just hidden state)
        # given a vector of the size of the decoder states, creates an emptyu
        initial_decoder_state_11e = x.view(1, 1, -1)
        initial_decoder_state = (initial_decoder_state_11e,
                                 torch.zeros_like(initial_decoder_state_11e))
        return initial_decoder_state


class ConditionDecoder(nn.Module):  # pylint: disable=too-many-instance-attributes
    """
    The decoder produces seq2seq-style output for the conditional
    columns in WikiSQL.

    It process input in several stages.

    The decoder accepts the question and table sequence encodings.

    Then we perform the following in order:
    1. the decoder predicts the number of predicates with a feed-forward
       network attending to the input sequences.
    2. the decoder predicts which columns to predicate on with an RNN
       with state initialized from the joint summary (automatically
       stopped after the number of predictions specified by (1)). Input
       is augmented with attention over the input sequences.
    3. Same as above, but we predict the comparison operators.
    4. Same as above, but we predict the literals.

    Each successive step uses the previous one as input.
    """

    def __init__(self, fields, max_conds):
        super().__init__()

        if flags.FLAGS.tie_encodings:
            self.shared_encoder = WikiSQLInputEncoding(fields, attend=False)
            self.tied_encodings = True
        else:
            self.col_input_encoder = WikiSQLInputEncoding(fields, attend=False)
            self.op_input_encoder = WikiSQLInputEncoding(fields, attend=False)
            self.literal_input_encoder = WikiSQLInputEncoding(
                fields, attend=False)
            self.stop_input_encoder = WikiSQLInputEncoding(
                fields, attend=False)
            self.tied_encodings = True

        col_seq_size = self.col_input_encoder.column_seq_size
        src_seq_size = self.col_input_encoder.question_seq_size

        # predict stop
        joint_embedding_size = col_seq_size + src_seq_size
        self.stop_logits = _Sequential(
            _Unwrap(IndependentAttention(col_seq_size, src_seq_size, 0)),
            _Cat(), MLP(joint_embedding_size, [], max_conds))

        # predict cols
        decoder_size = flags.FLAGS.decoder_size
        self.col_init = _Sequential(
            _Unwrap(IndependentAttention(col_seq_size, src_seq_size, 0)),
            _Cat(), MLP(joint_embedding_size, [], decoder_size),
            _InitDecoderState())
        self.col_attn = _double_attn(col_seq_size, src_seq_size, decoder_size)
        input_size = joint_embedding_size  # attention over inputs
        input_size += col_seq_size  # feed own outputs
        self.col_rnn = nn.LSTM(
            input_size, decoder_size, num_layers=1, bidirectional=False)
        self.col_proj = Pointer(col_seq_size, decoder_size)

        # predict ops
        self.op_init = _Sequential(
            _Unwrap(IndependentAttention(col_seq_size, src_seq_size, 0)),
            _Cat(), MLP(joint_embedding_size, [], decoder_size),
            _InitDecoderState())
        self.op_attn = _double_attn(col_seq_size, src_seq_size, decoder_size)
        num_ops = len(wikisql.CONDITIONAL)
        self.op_embedding = nn.Embedding(num_ops,
                                         flags.FLAGS.op_embedding_size)
        input_size = joint_embedding_size  # attention over inputs
        input_size += col_seq_size  # feed predicted column
        input_size += self.op_embedding.embedding_dim  # feed predicted op
        self.op_rnn = nn.LSTM(
            input_size, decoder_size, num_layers=1, bidirectional=False)
        self.op_proj = MLP(decoder_size, [], num_ops)

        # predict spans
        self.span_init = _Sequential(
            _Unwrap(IndependentAttention(col_seq_size, src_seq_size, 0)),
            _Cat(), MLP(joint_embedding_size, [], decoder_size),
            _InitDecoderState())
        self.span_attn = _double_attn(col_seq_size, src_seq_size, decoder_size)
        input_size = joint_embedding_size  # attention over inputs
        input_size += col_seq_size  # feed predicted column
        input_size += self.op_embedding.embedding_dim  # feed predicted op
        # don't feed own inputs, not sure how to do this right yet
        # input_size += 2 * src_seq_size
        self.span_rnn = nn.LSTM(
            input_size, decoder_size, num_layers=1, bidirectional=False)
        self.span_l_ptr_logits = Pointer(src_seq_size, decoder_size)
        self.span_r_ptr_logits = Pointer(src_seq_size, decoder_size)

    def forward(self, prepared_ex):
        """
        e = some embedding dimension, varies tensor to tensor
        i = decoder input dimension
        q = question length
        q1 = question length + 1
        c = num columns
        o = cardinality of binary comparison operators

        sqe is the sequence of question word embeddings
        sce is the sequence of column description embeddings

        Returns a tuple with the following information (shape in parens)
        * (max_conds) logits for the predicted number of condition columns, p
        * (pc) logits over the column sequence for selected columns
        * (po) logits over operators
        * (pq) logits over left endpoint for span of literals
        * (pq1) logits over right endpoint for span of literals
        """
        # TODO separate sequence inputs for these three
        # TODO separate op_encoding for op/span predictions

        if self.tied_encodings:
            src_seq_for_stop_qe, tbl_seq_for_stop_ce = self.stop_input_encoder(
                prepared_ex)
            src_seq_for_col_qe, tbl_seq_for_col_ce = self.col_input_encoder(
                prepared_ex)
            src_seq_for_op_qe, tbl_seq_for_op_ce = self.op_input_encoder(
                prepared_ex)
            src_seq_for_literal_qe, tbl_seq_for_literal_ce = (
                self.literal_input_encoder(prepared_ex))
        else:
            src_seq_for_stop_qe, tbl_seq_for_stop_ce = self.shared_encoder(
                prepared_ex)
            src_seq_for_col_qe, tbl_seq_for_col_ce = self.shared_encoder(
                prepared_ex)
            src_seq_for_op_qe, tbl_seq_for_op_ce = self.shared_encoder(
                prepared_ex)
            src_seq_for_literal_qe, tbl_seq_for_literal_ce = (
                self.shared_encoder(prepared_ex))

        stop = self.stop_logits(tbl_seq_for_stop_ce, src_seq_for_stop_qe)

        if self.training:
            seq_len = len(prepared_ex['cond_op'])
        else:
            seq_len = stop.argmax().cpu().detach().numpy()

        col_pc = []
        h = self.col_init(tbl_seq_for_col_ce, src_seq_for_col_qe)
        prev_col = [torch.zeros_like(tbl_seq_for_col_ce[0])]
        for i in range(seq_len):
            rnn_input = torch.cat([
                self.col_attn(tbl_seq_for_col_ce, src_seq_for_col_qe,
                              h[0].view(-1)), prev_col[-1]
            ]).view(1, 1, -1)
            out, h = self.col_rnn(rnn_input, h)
            col_c = self.col_proj(tbl_seq_for_col_ce, out.view(-1))
            col_pc.append(col_c)
            col = col_c.argmax().cpu().detach().numpy()
            if self.training:
                true_col = prepared_ex['cond_col'][i]
                prev_col.append(tbl_seq_for_col_ce[true_col].detach())
            else:
                prev_col.append(tbl_seq_for_col_ce[col].detach())
        prev_col = prev_col[1:]

        op_po = []
        h = self.op_init(tbl_seq_for_op_ce, src_seq_for_op_qe)
        prev_op = [
            torch.zeros(self.op_embedding.embedding_dim, device=get_device())
        ]
        for i in range(seq_len):
            rnn_input = torch.cat([
                self.op_attn(tbl_seq_for_op_ce, src_seq_for_op_qe,
                             h[0].view(-1)), prev_col[i], prev_op[-1]
            ]).view(1, 1, -1)
            out, h = self.op_rnn(rnn_input, h)
            op_o = self.op_proj(out.view(-1))
            op_po.append(op_o)
            op = op_o.argmax().detach()
            if self.training:
                true_op = prepared_ex['cond_op'][i]
                prev_op.append(self.op_embedding(true_op))
            else:
                prev_op.append(self.op_embedding(op))
        prev_op = prev_op[1:]

        l_span_pq = []
        r_span_pq1 = []
        h = self.span_init(tbl_seq_for_literal_ce, src_seq_for_literal_qe)
        for _ in range(seq_len):
            rnn_input = torch.cat([
                self.span_attn(tbl_seq_for_literal_ce, src_seq_for_literal_qe,
                               h[0].view(-1)), prev_col[i], prev_op[i]
            ]).view(1, 1, -1)
            out, h = self.span_rnn(rnn_input, h)
            context = out.view(-1)
            l_span_q = self.span_l_ptr_logits(src_seq_for_literal_qe, context)
            l_span_pq.append(l_span_q)
            with torch.no_grad():
                # r = number entries after l
                l = l_span_q.argmax().detach().cpu().numpy()
                left_pad = [self._neg100()] * (l + 1)
            sqe_re = src_seq_for_literal_qe[l:]
            r_span_r = self.span_r_ptr_logits(sqe_re, context)
            r_span_q1 = torch.cat(left_pad + [r_span_r])
            r_span_pq1.append(r_span_q1)

        outs = (col_pc, op_po, l_span_pq, r_span_pq1)
        outs = tuple(map(_stack, outs))
        return (stop, ) + outs

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def _neg100():
        return -100 * torch.ones([1], dtype=torch.float32, device=get_device())


def _stack(x):
    if x:
        return torch.stack(x)
    return torch.Tensor([])
