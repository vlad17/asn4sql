"""
Defines the WikiSQLSpecificModel for the WikiSQL task of predicting the
SQL query from the natural language question and schema.

This model isn't simple in the sense that the model itself is that
small or easy to comprehend, but it is simple in that it
is specialized to the structure of the WikiSQL prediction
problem, which considers a small subset of the SQL language.

The model is very similar to Seq2SQL and its related spin-offs.

The src and ent embeddings are concatenated entrywise. These
comprise the "question".

The tbl embeddings comprise the "columns".

Both tbl and src embeddings are based on the glove natural language
embedding, which remains fixed throughout training.

We encode the question and column embeddings with untied weights
for the three separate wikisql prediction tasks:

* aggregation prediction
* selection column prediction
* conjunctive condition prediction

Currently, a question encoding transforms a sequence into an encoded sequence
through the hidden states of a bidirectional LSTM. A summary of the
sequence is also offered.

In the case of the columns, however, a bidirectional LSTM is also applied
to every column description (the same LSTM, applied separately to each column).
The final hidden state for each column is used to construct the
sequence column encoding.

The final question encoding conditions attention over the sequence column
encoding, producing a final column encoding.

The final column encoding and final question encodings are fed into an MLP
for aggregation classification.

We apply a pointer network conditioned on the final question encoding
over the sequence column encoding
to obtain a selection classifier.

The final column and question encodings are fed into a network which
initializes the decoder state.

Starting from initialization, the decoder produces hidden states
that are then used to:

(0) whether the decoder should terminate.
(1) condition a column pointer network to select a column over
    the SCE.
(2) conditioned on the selected column, select a conditional op
(3) condition a pointer network with (1) and (2) to select a
    left and then right endpoint for a literal over the SQE.

(0), (1), (2), and (3) are fed back as inputs into the decoder; this
is teacher-forced during training.

"""

import sys

from absl import flags
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .mlp import MLP
from .attention import Attention
from .nl_embedding import NLEmbedding
from .column_encoding import ColumnEncoding
from .question_encoding import QuestionEncoding
from .pointer import Pointer
from .condition_decoder import ConditionDecoder
from ..utils import get_device
from ..data import wikisql

_MAX_COND_LENGTH = 5

flags.DEFINE_integer(
    'aggregation_hidden', 64, 'aggregation 2-layer classifier has this many '
    'hidden layer neurons')


class WikiSQLSpecificModel(nn.Module):
    """
    Defines the WikiSQL-specific model, given the WikiSQL input fields
    as specified in data.wikisql (this assumes their vocabularies
    have already been built and pretrained).

    The forpward pass computes the training loss on the example directly
    (the loss is a combination of negative log likelihood terms).
    """

    def __init__(self, fields):
        super().__init__()
        self.fields = fields

        natural_language_embedding = NLEmbedding(fields['src'])
        ent_field = fields['ent']
        tbl_field = fields['tbl']
        self.question_encoding_for_agg = QuestionEncoding(
            natural_language_embedding, ent_field)
        self.column_encoding_for_agg = ColumnEncoding(
            natural_language_embedding, tbl_field)
        self.question_encoding_for_sel = QuestionEncoding(
            natural_language_embedding, ent_field)
        self.column_encoding_for_sel = ColumnEncoding(
            natural_language_embedding, tbl_field)
        self.question_encoding_for_cond = QuestionEncoding(
            natural_language_embedding, ent_field)
        self.column_encoding_for_cond = ColumnEncoding(
            natural_language_embedding, tbl_field)

        self.agg_attention = Attention(
            self.column_encoding_for_agg.sequence_size,
            self.question_encoding_for_agg.final_size)
        self.cond_attention = Attention(
            self.column_encoding_for_cond.sequence_size,
            self.question_encoding_for_cond.final_size)

        joint_embedding_size = (
            self.column_encoding_for_agg.sequence_size +
            self.question_encoding_for_agg.final_size)
        self.aggregation_mlp = MLP(joint_embedding_size,
                                   [flags.FLAGS.aggregation_hidden] * 2,
                                   len(wikisql.AGGREGATION))
        self.selection_ptrnet = Pointer(
            self.column_encoding_for_sel.sequence_size,
            self.question_encoding_for_sel.final_size)
        self.decoder_initialization_mlp = MLP(joint_embedding_size, [],
                                              flags.FLAGS.decoder_size)
        self.condition_decoder = ConditionDecoder(
            self.column_encoding_for_cond.sequence_size,
            self.question_encoding_for_cond.final_size)

    def prepare_example(self, ex):
        """
        extracts relevant input/output fields from the example and runs
        torchtext numericalization. returns a dictionary over batched
        """
        extracted_fields = [
            'src', 'ent', 'agg', 'sel', 'tbl', 'cond_op', 'cond_col',
            'cond_span_l', 'cond_span_r'
        ]
        prepared = {}
        with torch.no_grad():
            for field in extracted_fields:
                try:
                    attr = getattr(ex, field)
                    if self.fields[field].sequential and not attr:
                        prepared[field] = self.fields[field].tensor_type([])
                    else:
                        # torchtext, for some reason, has a different device
                        # API than the rest of pytorch
                        device = get_device()
                        device = device if device.type == 'cuda' else -1
                        prepared[field] = self.fields[field].process(
                            [attr], device=device, train=True)[0]
                except:
                    print(
                        'field {} being prepared from {}'.format(
                            field, getattr(ex, field)),
                        file=sys.stderr)
                    raise
                # we make sure we're not including lengths, since those
                # would be redundant for batch size 1
                assert not self.fields[field].include_lengths
        # print('preparing ex')
        # import json
        # print(json.dumps(
        #     {field: str(getattr(ex, field)) for field in extracted_fields},
        #     sort_keys=True, indent=4))
        # print(json.dumps({
        #     field: str(value.detach().cpu().numpy().tolist())
        #     for field, value in prepared.items()}, sort_keys=True, indent=4))
        return prepared

    def predict(self, prepared_ex):
        """
        returns the logits over the aggregation options,
        logits over column selection options, and
        finally predictions about conditionals,
        which is a tuple of vectors for operator, column,
        and span selection logits, of some predicted length.
        """

        # q = question token length
        # q1 = q + 1
        # c = number of columns
        # e = an encoding of some dimension, differs tensor to tensor
        # a = cardinality of aggregations
        # o = cardinality of binary comparison operators
        # w = _MAX_COND_LENGTH

        # aggregation prediction
        _, final_question_encoding_for_agg_e = self.question_encoding_for_agg(
            prepared_ex['src'], prepared_ex['ent'])
        sequence_column_encoding_for_agg_ce = self.column_encoding_for_agg(
            prepared_ex['tbl'])
        final_column_encoding_for_agg_e = self.agg_attention(
            sequence_column_encoding_for_agg_ce,
            final_question_encoding_for_agg_e)
        joint_encoding_for_agg_e = torch.cat([
            final_question_encoding_for_agg_e,
            final_column_encoding_for_agg_e], dim=0)
        aggregation_logits_a = self.aggregation_mlp(joint_encoding_for_agg_e)

        # selection prediction
        _, final_question_encoding_for_sel_e = self.question_encoding_for_sel(
            prepared_ex['src'], prepared_ex['ent'])
        sequence_column_encoding_for_sel_ce = self.column_encoding_for_sel(
            prepared_ex['tbl'])
        selection_logits_c = self.selection_ptrnet(
            sequence_column_encoding_for_sel_ce,
            final_question_encoding_for_sel_e)

        # condition (decoding) prediction
        (sequence_question_encoding_for_cond_qe,
         final_question_encoding_for_cond_e) = self.question_encoding_for_cond(
             prepared_ex['src'], prepared_ex['ent'])
        sequence_column_encoding_for_cond_ce = self.column_encoding_for_sel(
            prepared_ex['tbl'])
        final_column_encoding_for_cond_e = self.cond_attention(
            sequence_column_encoding_for_cond_ce,
            final_question_encoding_for_cond_e)
        joint_encoding_for_cond_e = torch.cat([
            final_question_encoding_for_cond_e,
            final_column_encoding_for_cond_e], dim=0)
        initial_decoder_state = self.decoder_initialization_mlp(
            joint_encoding_for_cond_e)
        initial_decoder_state = self.condition_decoder.create_initial_state(
            initial_decoder_state)
        conds = self._predicted_conds(
            initial_decoder_state,
            sequence_question_encoding_for_cond_qe,
            sequence_column_encoding_for_cond_ce,
            prepared_ex)

        return aggregation_logits_a, selection_logits_c, conds

    def _predicted_conds(self, hidden_state, question, columns, prepared_ex):
        stop_w2 = []
        op_logits_wo = []
        col_logits_wc = []
        span_l_logits_wq = []
        span_r_logits_wq1 = []

        true_cond_fields = [
            'cond_op', 'cond_col', 'cond_span_l', 'cond_span_r'
        ]
        num_cols = len(prepared_ex['cond_op'])
        decoder_input = self.condition_decoder.dummy_input(stop=False)
        for i in range(_MAX_COND_LENGTH):
            decoder_output, hidden_state = self.condition_decoder(
                hidden_state, decoder_input, question, columns)
            (pred_stop_2, pred_op_logits_o, pred_col_logits_c,
             pred_span_l_logits_q, pred_span_r_logits_q1) = decoder_output
            stop_w2.append(pred_stop_2)
            op_logits_wo.append(pred_op_logits_o)
            col_logits_wc.append(pred_col_logits_c)
            span_l_logits_wq.append(pred_span_l_logits_q)
            span_r_logits_wq1.append(pred_span_r_logits_q1)

            # prepare the next input: see ConditionDecoder.dummy_input
            # for the expected order of inputs
            if self.training:
                # teacher force
                if i < num_cols:
                    stop = 0
                    decoder_input = (stop, ) + tuple(prepared_ex[x][i]
                                                     for x in true_cond_fields)
                else:
                    decoder_input = self.condition_decoder.dummy_input(
                        stop=True)
            else:
                with torch.no_grad():
                    decoder_input = tuple(x.argmax() for x in decoder_output)
        return tuple(
            map(torch.stack, [
                op_logits_wo, col_logits_wc, span_l_logits_wq,
                span_r_logits_wq1, stop_w2
            ]))

    def forward(self, prepared_ex):
        """
        Compute the model loss on the prepared example as well as whether
        the example was perfectly guessed.
        """
        results = self.diagnose(prepared_ex, with_prediction=False)
        loss, _ = results['loss (*total)']
        acc, _ = results['acc (*total)']
        return loss, acc

    @staticmethod
    def _truncated_nll_acc(input_il, target_o):
        # i = input length
        # o = output length
        assert len(input_il) >= len(target_o), (len(input_il), len(target_o))
        # m = min(i, o)
        # l = logits size (so target is in the range [0, o))
        m = min(len(input_il), len(target_o))
        if not m:
            return (torch.zeros([], device=get_device()),
                    torch.ones([], device=get_device()))
        input_ml = input_il[:m]
        target_m = target_o[:m]
        nll = F.cross_entropy(input_ml, target_m)
        acc = torch.prod(input_ml.argmax(dim=1) == target_o).type(
            torch.float32)
        return nll, acc

    @staticmethod
    def _unsqueezed_cross_entropy(pred, targ):
        pred = torch.unsqueeze(pred, dim=0)
        targ = torch.unsqueeze(targ, dim=0)
        return F.cross_entropy(pred, targ)

    def diagnose(self, prepared_ex, with_prediction=True):
        """
        diagnostic info - returns a dictionary keyed by diagnostic name
        with the value being a tuple of the numerical value of the
        diagnostic and a string format.
        """
        (aggregation_logits_a, selection_logits_c,
         conds) = self.predict(prepared_ex)
        (op_logits_wo, col_logits_wc, span_l_logits_wq, span_r_logits_wq1,
         stop_w2) = conds
        ncols = len(prepared_ex['cond_op'])
        true_stop = torch.zeros(
            [_MAX_COND_LENGTH], dtype=torch.long).to(get_device())
        true_stop[ncols:] = 1

        op_loss, op_acc = self._truncated_nll_acc(op_logits_wo,
                                                  prepared_ex['cond_op'])
        col_loss, col_acc = self._truncated_nll_acc(col_logits_wc,
                                                    prepared_ex['cond_col'])
        span_l_loss, span_l_acc = self._truncated_nll_acc(
            span_l_logits_wq, prepared_ex['cond_span_l'])
        span_r_loss, span_r_acc = self._truncated_nll_acc(
            span_r_logits_wq1, prepared_ex['cond_span_r'])
        stop_loss, stop_acc = self._truncated_nll_acc(stop_w2, true_stop)

        cond_acc = torch.prod(
            torch.stack([op_acc, col_acc, span_l_acc, span_r_acc, stop_acc]))
        cond_loss = op_loss + col_loss + span_l_loss + span_r_loss + stop_loss
        agg_loss = self._unsqueezed_cross_entropy(aggregation_logits_a,
                                                  prepared_ex['agg'])
        agg_acc = aggregation_logits_a.argmax() == prepared_ex['agg']
        agg_acc = agg_acc.type(torch.float32)
        sel_loss = self._unsqueezed_cross_entropy(selection_logits_c,
                                                  prepared_ex['sel'])
        sel_acc = selection_logits_c.argmax() == prepared_ex['sel']
        sel_acc = sel_acc.type(torch.float32)

        if with_prediction:
            stop = stop_w2.argmax(1).detach().cpu().numpy()
            stop = stop.argmax() if np.any(stop) else None
            ops = op_logits_wo.argmax(1).detach().cpu().numpy()[:stop]
            cols = col_logits_wc.argmax(1).detach().cpu().numpy()[:stop]
            span_l = span_l_logits_wq.argmax(1).detach().cpu().numpy()[:stop]
            span_r = span_r_logits_wq1.argmax(1).detach().cpu().numpy()[:stop]
            agg = aggregation_logits_a.argmax().detach().cpu().numpy()
            sel = selection_logits_c.argmax().detach().cpu().numpy()
            prediction = wikisql.Prediction(ops, cols, span_l, span_r, agg,
                                            sel)
        else:
            prediction = None

        return {
            'loss (*total)': (cond_loss + agg_loss + sel_loss, '{:8.4g}'),
            'loss (cond)': (cond_loss, '{:8.4g}'),
            'loss (agg)': (agg_loss, '{:8.4g}'),
            'loss (sel)': (sel_loss, '{:8.4g}'),
            'acc (*total)': (cond_acc * agg_acc * sel_acc, '{:8.2%}'),
            'acc (cond)': (cond_acc, '{:8.2%}'),
            'acc (agg)': (agg_acc, '{:8.2%}'),
            'acc (sel)': (sel_acc, '{:8.2%}'),
            'prediction': prediction
        }
