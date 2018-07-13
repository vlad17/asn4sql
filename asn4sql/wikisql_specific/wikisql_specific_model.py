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

TODO rest of arch.
TODO use more Sequential where possible.
"""

import sys

from absl import flags
import torch
from torch import nn
from torch.nn import functional as F

from .mlp import MLP
from .wikisql_input_encoding import WikiSQLInputEncoding
from .pointer import Pointer
from .condition_decoder import ConditionDecoder
from ..utils import get_device
from ..data import wikisql

_MAX_COND_LENGTH = 5

flags.DEFINE_integer('aggregation_hidden', 256, 'aggregation net width')
flags.DEFINE_integer('aggregation_depth', 2, 'aggregation net depth')
flags.DEFINE_integer('selection_hidden', 256, 'selection net width')
flags.DEFINE_integer('selection_depth', 0, 'selection net depth')


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

        self.process_sel_input = WikiSQLInputEncoding(fields)
        self.process_agg_input = WikiSQLInputEncoding(fields)

        joint_embedding_size = self.process_sel_input.attn_size
        agg_net = flags.FLAGS.aggregation_depth * [
            flags.FLAGS.aggregation_hidden
        ]
        self.aggregation_mlp = MLP(joint_embedding_size, agg_net,
                                   len(wikisql.AGGREGATION))
        sel_net = flags.FLAGS.selection_depth * [flags.FLAGS.selection_hidden]
        self.selection_mlp = MLP(joint_embedding_size, sel_net,
                                 flags.FLAGS.selection_hidden)
        self.selection_ptrnet = Pointer(self.process_sel_input.column_seq_size,
                                        flags.FLAGS.selection_hidden)
        self.condition_decoder = ConditionDecoder(fields, _MAX_COND_LENGTH)

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
        _, _, joint_encoding_for_agg_e = self.process_agg_input(prepared_ex)
        aggregation_logits_a = self.aggregation_mlp(joint_encoding_for_agg_e)

        # selection prediction
        _, sequence_column_encoding_for_sel_ce, joint_encoding_for_sel_e = (
            self.process_sel_input(prepared_ex))
        selection_logits_c = self.selection_ptrnet(
            sequence_column_encoding_for_sel_ce,
            self.selection_mlp(joint_encoding_for_sel_e))

        # condition (decoding) prediction
        conds = self.condition_decoder(prepared_ex)

        return aggregation_logits_a, selection_logits_c, conds

    def forward(self, prepared_ex):
        """
        Compute the model loss on the prepared example as well as whether
        the example was perfectly guessed.
        """
        results = self.diagnose(
            prepared_ex, with_prediction=False, detach=False)
        loss = results['loss (*total)']
        acc = results['acc (*total)']
        return loss, acc

    def _truncated_nll_acc(self, input_il, target_o):
        # m = min(i, o)
        # l = logits size (so target is in the range [0, o))
        m = min(len(input_il), len(target_o))
        if self.training:
            # should be teacher forced
            assert m == len(input_il), (m, len(input_il), target_o)
        if not m:
            return (torch.zeros([], device=get_device()),
                    torch.ones([], device=get_device()))
        input_ml = input_il[:m]
        target_m = target_o[:m]
        nll = F.cross_entropy(input_ml, target_m)
        acc = torch.prod(input_ml.argmax(dim=1) == target_m).type(
            torch.float32)
        return nll, acc

    @staticmethod
    def _unsqueezed_cross_entropy(pred, targ):
        pred = torch.unsqueeze(pred, dim=0)
        targ = torch.unsqueeze(targ, dim=0)
        return F.cross_entropy(pred, targ)

    def diagnose(self, prepared_ex, with_prediction=True, detach=True):
        """
        diagnostic info - returns a dictionary keyed by diagnostic name
        with the value being a tuple of the numerical value of the
        diagnostic and a string format.
        """
        (aggregation_logits_a, selection_logits_c,
         conds) = self.predict(prepared_ex)
        stop, col_pc, op_po, l_span_pq, r_span_pq1 = conds
        ncols = len(prepared_ex['cond_op'])

        op_loss, op_acc = self._truncated_nll_acc(op_po,
                                                  prepared_ex['cond_op'])
        col_loss, col_acc = self._truncated_nll_acc(col_pc,
                                                    prepared_ex['cond_col'])
        span_l_loss, span_l_acc = self._truncated_nll_acc(
            l_span_pq, prepared_ex['cond_span_l'])
        span_r_loss, span_r_acc = self._truncated_nll_acc(
            r_span_pq1, prepared_ex['cond_span_r'])
        stop_loss, stop_acc = self._truncated_nll_acc(
            stop.unsqueeze(0),
            torch.tensor([ncols], dtype=torch.long, device=get_device()))  # pylint: disable=not-callable

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
            stop = stop.argmax().cpu().detach().numpy()
            ops = _argmax1(op_po).detach().cpu().numpy()[:stop]
            cols = _argmax1(col_pc).detach().cpu().numpy()[:stop]
            span_l = _argmax1(l_span_pq).detach().cpu().numpy()[:stop]
            span_r = _argmax1(r_span_pq1).detach().cpu().numpy()[:stop]
            agg = aggregation_logits_a.argmax().detach().cpu().numpy()
            sel = selection_logits_c.argmax().detach().cpu().numpy()
            prediction = wikisql.Prediction(ops, cols, span_l, span_r, agg,
                                            sel)
        else:
            prediction = None

        out = {
            'loss (*total)': cond_loss + agg_loss + sel_loss,
            'acc (*total)': cond_acc * agg_acc * sel_acc,
            'acc (cond)': cond_acc,
            'acc (agg)': agg_acc,
            'acc (sel)': sel_acc,
            'acc (cond: op)': op_acc,
            'acc (cond: col)': col_acc,
            'acc (cond: span_l)': span_l_acc,
            'acc (cond: span_r)': span_r_acc,
            'acc (cond: stop)': stop_acc,
            'prediction': prediction
        }
        if detach:
            for k in out:
                if hasattr(out[k], 'detach'):
                    out[k] = out[k].detach().cpu().numpy()
        return out


def _argmax1(x):
    if x.numel():
        return x.argmax(1)
    return x
