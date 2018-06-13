"""
Defines the WikiSQLSpecificModel for the WikiSQL task of predicting the
SQL query from the natural language question and schema.

This model isn't simple in the sense that the model itself is that
small or easy to comprehend, but it is simple in that it
is specialized to the structure of the WikiSQL prediction
problem, which considers a small subset of the SQL language.

The model is itself pretty minimal and very similar to Seq2SQL.

The src and ent embeddings are concatenated entrywise. Next,
a bidirectional LSTM produces the concatenated forward and
backward hidden states for the src and ent concatenation.
The intermediate states comprise a sequence question embedding
(SQE), and a unidirectional LSTM the produces a single final
hidden state over the SCE that gives the question embedding
(QE).

A bidirectional LSTM is also applied to every column
description (the same LSTM, applied separately to each column).
The final hidden state for each column is used to
construct a sequence column embedding (SCE).
Finally, we apply a second (unidirectional) LSTM to the SCE
to obtain its final state, the column embedding (CE).

We map the QE and CE into an aggregation via MLP.

We apply a pointer network conditioned on the QE to the SCE
to obtain a selection classifier.

The CE and QE are fed into a network which initializes the decoder
state.

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
import torch
from torch import nn
from torch.nn import functional as F

from .mlp import MLP
from .column_embedding import ColumnEmbedding
from .question_embedding import QuestionEmbedding
from .pointer import Pointer
from .condition_decoder import ConditionDecoder
from ..utils import get_device
from ..data import wikisql

_MAX_COND_LENGTH = 5

flags.DEFINE_integer('aggregation_hidden', 256,
                     'aggregation 2-layer classifier has this many '
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

        self.column_embedding = ColumnEmbedding(fields['tbl'])
        self.question_embedding = QuestionEmbedding(
            self.column_embedding.embedding,
            fields['ent'])
        joint_embedding_size = (
            self.column_embedding.final_size +
            self.question_embedding.final_size)
        self.aggregation_mlp = MLP(
            joint_embedding_size,
            [flags.FLAGS.aggregation_hidden] * 2,
            len(wikisql.AGGREGATION))
        self.selection_ptrnet = Pointer(
            self.column_embedding.sequence_size,
            self.question_embedding.final_size)
        self.decoder_initialization_mlp = MLP(
            joint_embedding_size,
            [],
            flags.FLAGS.decoder_size)
        self.condition_decoder = ConditionDecoder(
            self.column_embedding.sequence_size,
            self.question_embedding.sequence_size)

    def prepare_example(self, ex):
        """
        extracts relevant input/output fields from the example and runs
        torchtext numericalization.
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
                        prepared[field] = self.fields[field].process(
                            [attr], device=get_device(), train=True)[0]
                except:
                    print('field {} being prepared from {}'.format(
                        field, getattr(ex, field)), file=sys.stderr)
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
        #     field: str(value.detach().cpu().numpy().tolist()) for field, value in
        #     prepared.items()}, sort_keys=True, indent=4))
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

        sqe_qe, qe_e = self.question_embedding(
            prepared_ex['src'], prepared_ex['ent'])
        sce_ce, ce_e = self.column_embedding(prepared_ex['tbl'])
        joint_embedding_e = torch.cat([qe_e, ce_e], dim=0)
        aggregation_logits_a = self.aggregation_mlp(joint_embedding_e)
        selection_logits_c = self.selection_ptrnet(sce_ce, qe_e)

        # pytorch-related initialization setup
        initial_decoder_state = self.decoder_initialization_mlp(
            joint_embedding_e)
        initial_decoder_state = self.condition_decoder.create_initial_state(
            initial_decoder_state)
        conds = self._predicted_conds(
            initial_decoder_state, sqe_qe, sce_ce, prepared_ex)

        return aggregation_logits_a, selection_logits_c, conds

    def _predicted_conds(self, hidden_state, sqe_qe, sce_ce, prepared_ex):
        stop_w2 = []
        op_logits_wo = []
        col_logits_wc = []
        span_l_logits_wq = []
        span_r_logits_wq1 = []

        true_cond_fields = [
            'cond_op', 'cond_col', 'cond_span_l', 'cond_span_r']
        num_cols = len(prepared_ex['cond_op'])
        decoder_input = self.condition_decoder.dummy_input(stop=False)
        for i in range(_MAX_COND_LENGTH):
            decoder_output, hidden_state = self.condition_decoder(
                hidden_state, decoder_input, sqe_qe, sce_ce)
            (pred_stop_2, pred_op_logits_o,
             pred_col_logits_c, pred_span_l_logits_q,
             pred_span_r_logits_q1) = decoder_output
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
                    decoder_input = (stop,) + tuple(
                        prepared_ex[x][i] for x in true_cond_fields)
                else:
                    decoder_input = self.condition_decoder.dummy_input(
                        stop=True)
            else:
                with torch.no_grad():
                    decoder_input = tuple(
                        x.argmax() for x in decoder_output)
        return tuple(map(torch.stack, [
            op_logits_wo,
            col_logits_wc,
            span_l_logits_wq,
            span_r_logits_wq1,
            stop_w2]))

    def forward(self, prepared_ex):
        """
        Compute the model loss on the prepared example.
        """
        value, _ = self.diagnose(prepared_ex)['total loss *']
        return value

    @staticmethod
    def _truncated_nll_acc(input_il, target_o):
        # i = input length
        # o = output length
        assert len(input_il) >= len(target_o), (len(input_il), len(target_o))
        # m = min(i, o)
        # l = logits size (so target is in the range [0, o))
        m = min(len(input_il), len(target_o))
        if not m:
            return (
                torch.zeros([], device=get_device()),
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


    def diagnose(self, prepared_ex):
        """
        diagnostic info - returns a dictionary keyed by diagnostic name
        with the value being a tuple of the numerical value of the
        diagnostic and a string format.
        """
        (aggregation_logits_a,
         selection_logits_c,
         conds) = self.predict(prepared_ex)
        (op_logits_wo, col_logits_wc,
         span_l_logits_wq, span_r_logits_wq1,
         stop_w2) = conds
        ncols = len(prepared_ex['cond_op'])
        true_stop = torch.zeros([_MAX_COND_LENGTH], dtype=torch.long).to(
            get_device())
        true_stop[ncols:] = 1

        op_loss, op_acc = self._truncated_nll_acc(
            op_logits_wo, prepared_ex['cond_op'])
        col_loss, col_acc = self._truncated_nll_acc(
            col_logits_wc, prepared_ex['cond_col'])
        span_l_loss, span_l_acc = self._truncated_nll_acc(
            span_l_logits_wq, prepared_ex['cond_span_l'])
        span_r_loss, span_r_acc = self._truncated_nll_acc(
            span_r_logits_wq1, prepared_ex['cond_span_r'])
        stop_loss, stop_acc = self._truncated_nll_acc(stop_w2, true_stop)

        cond_acc = torch.prod(
            torch.stack([
                op_acc, col_acc, span_l_acc, span_r_acc, stop_acc]))
        cond_loss = op_loss + col_loss + span_l_loss + span_r_loss + stop_loss
        agg_loss = self._unsqueezed_cross_entropy(
            aggregation_logits_a, prepared_ex['agg'])
        agg_acc = aggregation_logits_a.argmax() == prepared_ex['agg']
        agg_acc = agg_acc.type(torch.float32)
        sel_loss = self._unsqueezed_cross_entropy(
            selection_logits_c, prepared_ex['sel'])
        sel_acc = selection_logits_c.argmax() == prepared_ex['sel']
        sel_acc = sel_acc.type(torch.float32)

        return {
            'total loss *': (cond_loss + agg_loss + sel_loss, '{:8.4g}'),
            'cond loss': (cond_loss, '{:8.4g}'),
            'agg loss': (agg_loss, '{:8.4g}'),
            'sel loss': (sel_loss, '{:8.4g}'),
            'total acc *': (cond_acc * agg_acc * sel_acc, '{:5.1%}'),
            'cond acc': (cond_acc, '{:5.1%}'),
            'agg acc': (agg_acc, '{:5.1%}'),
            'sel acc': (sel_acc, '{:5.1%}'),
        }
