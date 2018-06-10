"""
Defines the WikiSQLSpecificModel for the WikiSQL task of predicting the
SQL query from the natural language question and schema.

This model isn't simple in the sense that the model itself is that
small or easy to comprehend, but it is simple in that it
is specialized to the structure of the WikiSQL prediction
problem, which considers a small subset of the SQL language.
"""

import copy
import sys

import torch
from torch import nn

from ..utils import get_device
from ..data import wikisql

class WikiSQLSpecificModel(nn.Module):
    """
    Defines the WikiSQL-specific model, given the WikiSQL input fields
    as specified in data.wikisql (this assumes their vocabularies
    have already been built and pretrained).

    The forward pass computes the training loss on the example directly
    (the loss is a combination of negative log likelihood terms).
    """

    def __init__(self, train):
        super().__init__()
        self.fc2 = nn.Linear(50, 10)
        # TODO: really just need a pickleable set of fields,
        # right now that's implemented inside of a dataset
        self.training_dataset = copy.copy(train)
        self.training_dataset.examples = []


    def prepare_example(self, ex):
        """
        extracts relevant input/output fields from the example and runs
        torchtext numericalization.
        """
        fields = self.training_dataset.fields
        extracted_fields = [
            'src', 'ent', 'agg', 'sel', 'tbl',
            'cond_op', 'cond_col', 'cond_span_l', 'cond_span_r']
        prepared = {}
        with torch.no_grad():
            for field in extracted_fields:
                try:
                    attr = getattr(ex, field)
                    if fields[field].sequential and not attr:
                        prepared[field] = fields[field].tensor_type([])
                    else:
                        prepared[field] = fields[field].process(
                            [attr],
                            device=get_device(), train=True)[0]
                except:
                    print('field {} being prepared from {}'
                          .format(field, getattr(ex, field)))
                    raise
                # we make sure we're not including lengths, since those
                # would be redundant for batch size 1
                assert not fields[field].include_lengths
        return prepared

    def forward(self, prepared_ex):
        """
        Compute the model loss on the prepared example.
        """
        for i in ['src', 'tbl', 'ent']:
            print(i, [
                self.training_dataset.fields[i].vocab.itos[x] for x in prepared_ex[i]])
        print('agg', wikisql.AGGREGATION[prepared_ex['agg']])
        print('sel', prepared_ex['sel'])
        print('cond_op', [
            wikisql.CONDITIONAL[i] for i in prepared_ex['cond_op']])
        print('cond_col', [i for i in prepared_ex['cond_col']])
        print('cond_spans', list(zip(
            prepared_ex['cond_span_l'], prepared_ex['cond_span_r'])))

        return self.fc2(torch.zeros((1, 50,)).to(get_device())).sum()


    def diagnose(self, prepared_ex):
        """

        """
        return {}
        pass # should return a dict as used in simple.py
